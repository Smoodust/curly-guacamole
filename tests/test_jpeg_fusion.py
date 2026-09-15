import torch

from src.modules.gated_fuse import GatedFuse


def test_fusion_starts_as_identity_then_trains_both_inputs():
    block = GatedFuse(8, 4)
    rgb = torch.randn(2, 8, 5, 7, requires_grad=True)
    jpeg = torch.randn(2, 4, 3, 4, requires_grad=True)
    torch.testing.assert_close(block(rgb, jpeg), rgb, rtol=0, atol=0)
    with torch.no_grad():
        block.channel_gate.fill_(0.1)
    block(rgb, jpeg).square().mean().backward()
    for value in (rgb, jpeg):
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_local_fusion_keeps_checkpoint_keys():
    block = GatedFuse(8, 4)
    assert set(block.state_dict()) == {
        'channel_gate', 'residual.rgb.weight', 'residual.rgb.bias',
        'residual.jpeg.weight', 'residual.jpeg.bias', 'residual.context.0.weight',
        'residual.context.0.bias', 'residual.output.weight', 'residual.output.bias',
    }
