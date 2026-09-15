"""Fused categorical convolution on CUDA; imported only when Triton is available."""

import torch
import triton
import triton.language as tl


@triton.jit
def _category_conv(bins, weight, bias, output, height, width: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_channel = tl.program_id(1)
    channel = batch_channel % 64
    batch = batch_channel // 64
    y, x = position // width, position % width
    valid = position < height * width
    value = tl.full((BLOCK,), 0, tl.float32) + tl.load(bias + channel).to(tl.float32)
    for row in tl.static_range(3):
        for col in tl.static_range(3):
            yy, xx = y + (row - 1) * 8, x + (col - 1) * 8
            inside = valid & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            category = tl.load(bins + batch * height * width + yy * width + xx,
                               mask=inside, other=0).to(tl.int32)
            offset = channel * 21 * 9 + category * 9 + row * 3 + col
            value += tl.load(weight + offset, mask=inside, other=0).to(tl.float32)
    tl.store(output + batch_channel * height * width + position, value, mask=valid)


# Runtime width variant for controlled comparisons on variable-size JPEGs.
@triton.jit
def _category_conv_dynamic_width(bins, weight, bias, output, height, width, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_channel = tl.program_id(1)
    channel = batch_channel % 64
    batch = batch_channel // 64
    y, x = position // width, position % width
    valid = position < height * width
    value = tl.full((BLOCK,), 0, tl.float32) + tl.load(bias + channel).to(tl.float32)
    for row in tl.static_range(3):
        for col in tl.static_range(3):
            yy, xx = y + (row - 1) * 8, x + (col - 1) * 8
            inside = valid & (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            category = tl.load(bins + batch * height * width + yy * width + xx,
                               mask=inside, other=0).to(tl.int32)
            offset = channel * 21 * 9 + category * 9 + row * 3 + col
            value += tl.load(weight + offset, mask=inside, other=0).to(tl.float32)
    tl.store(output + batch_channel * height * width + position, value, mask=valid)


class CUDACategoryConv(torch.autograd.Function):
    """Gather in forward, use the native convolution weight gradient in backward.

    Only bins and the small weight tensor are saved. Training reconstructs
    one-hot temporarily in backward; inference never materializes it.
    """

    @staticmethod
    def forward(ctx, bins, weight, bias, specialize_width=True):
        ctx.input_count = len(ctx.needs_input_grad)
        ctx.save_for_backward(bins, weight)
        batch, height, width = bins.shape
        output = torch.empty((batch, 64, height, width), device=bins.device, dtype=weight.dtype)
        with torch.cuda.device(bins.device):
            kernel = _category_conv if specialize_width else _category_conv_dynamic_width
            kernel[(triton.cdiv(height * width, 256), batch * 64)](
                bins, weight, bias, output, height, width, BLOCK=256)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        bins, weight = ctx.saved_tensors
        grad_weight = grad_bias = None
        with torch.autocast(device_type='cuda', enabled=False):
            grad_output = grad_output.to(weight.dtype).contiguous()
            if ctx.needs_input_grad[1]:
                categories = torch.arange(21, device=bins.device, dtype=torch.uint8)[None, :, None, None]
                volume = (bins[:, None] == categories).to(weight.dtype)
                grad_weight = torch.nn.grad.conv2d_weight(
                    volume, weight.shape, grad_output, padding=8, dilation=8)
            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))
        return (None, grad_weight, grad_bias, None)[:ctx.input_count]
