from torch import nn

from src.decoders.layers import make_norm


class DWSConv2d(nn.Module):
    """Depthwise-separable convolution layer
    Легковесная свертка, сильно снижает число параметров и FLOPs относительно обычной 3×3 свертки
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, norm='batch'):
        super().__init__()

        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False,
        )
        self.depthwise_norm = make_norm(norm, in_ch)

        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pointwise_norm = make_norm(norm, out_ch)

        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        return self.activation(self.pointwise_norm(self.pointwise(x)))
