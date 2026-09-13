import pytest
import torch

from src.decoders.emcad import EMCADDecoder


@pytest.mark.parametrize('position,size,last_stage', [
    ('pre_refine_4', 16, 2), ('post_refine_8', 8, 2), ('post_refine_16', 4, 1),
])
def test_aux_position_gradient_boundary(position, size, last_stage):
    torch.set_num_threads(1)
    decoder = EMCADDecoder([16, 32, 64, 128], [4, 8, 16, 32],
                           use_aux=True, aux_position=position).train()
    features = [torch.randn(2, c, s, s, requires_grad=True)
                for c, s in zip([16, 32, 64, 128], [16, 8, 4, 2])]
    output, aux = decoder(features)
    assert output.shape == (2, 16, 16, 16)
    assert aux.shape == (2, 1, size, size)
    aux.square().mean().backward()
    for stage, block in enumerate(decoder.refinement):
        has_gradient = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in block.parameters())
        assert has_gradient == (stage <= last_stage)
    decoder.eval()
    with torch.no_grad():
        assert decoder(features)[1] is None


def test_aux_position_rejects_typo():
    with pytest.raises(ValueError, match='aux_position'):
        EMCADDecoder([16, 32, 64, 128], [4, 8, 16, 32], aux_position='typo')
