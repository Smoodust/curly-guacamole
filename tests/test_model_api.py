import pytest


def test_fusion_rejects_missing_stride_and_misaligned_metadata():
    from src.modules.forensic_fusion import ForensicFusion

    with pytest.raises(ValueError, match='missing fusion strides'):
        ForensicFusion([4, 8, 16], [4, 8, 16], (4, 8, 16))
    with pytest.raises(ValueError, match='same length'):
        ForensicFusion([4, 8, 16, 32], [4, 8, 16], (4, 8, 16))
    with pytest.raises(ValueError, match='unique'):
        ForensicFusion([8, 8, 16, 32], [4, 8, 16, 32], (4, 8, 16))


def test_segmenter_gate_stats_use_public_fusion_api():
    import torch

    from src.modules.forensic_fusion import ForensicFusion
    from src.modules.segmenter import Segmenter

    model = Segmenter.__new__(Segmenter)
    torch.nn.Module.__init__(model)
    model.forensic_fusion = ForensicFusion([8, 16, 32], [4, 8, 16], (4, 8, 16))
    with torch.no_grad():
        model.forensic_fusion.fusion_blocks['16'].channel_gate.fill_(-0.75)
    assert model.forensic_gate_stats()['max_abs'] == pytest.approx(0.75)


def test_default_segmenter_uses_emcad_and_preserves_state_names():
    import torch

    from src.decoders import EMCADDecoder
    from src.modules.segmenter import Segmenter

    model = Segmenter(pretrained=False).eval()
    assert isinstance(model.decoder, EMCADDecoder)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 64, 96), jpeg=[{'available': False}])
    assert output['logits'].shape == (1, 1, 64, 96)
    assert output['cls_logits'].shape == (1, 1)
    assert 'aux_logits' not in output
    keys = model.state_dict()
    for prefix in ('encoder.', 'decoder.', 'segmentation_head.', 'classification_head.'):
        assert any(key.startswith(prefix) for key in keys)


@pytest.mark.parametrize('option,value', [
    ('use_forensics', False), ('jpeg_variant', 'signed'), ('fusion_variant', 'film'),
    ('decoder_kwargs', {}), ('forensic_mode', 'maps'), ('decoder_name', 'unet'), ('decoder_channels', [16, 8]), ('decoder_embed_dim', 8),
])
def test_segmenter_rejects_obsolete_decoder_arguments(option, value):
    from src.modules.segmenter import Segmenter

    with pytest.raises(TypeError, match=option):
        Segmenter('pvt_v2_b2', pretrained=False, **{option: value})


def test_cleanup_matches_captured_baseline_initialization_and_outputs():
    """Local cleanup audit; the original binary capture is intentionally untracked."""
    import hashlib
    import json
    from pathlib import Path

    import torch

    from src.modules.segmenter import Segmenter
    from src.modules.sync_batchnorm import SynchronizedBatchNorm

    folder = Path(__file__).resolve().parents[1] / 'tmp/jpeg640_cleanup'
    if not (folder / 'reference.pt').exists():
        pytest.skip('Original baseline capture is available only in the cleanup workspace')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = SynchronizedBatchNorm.apply(Segmenter(pretrained=False))
    manifest = {name: {'shape': list(value.shape),
                       'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
                for name, value in model.state_dict().items()}
    assert manifest == json.loads((folder / 'model_manifest.json').read_text())
    reference = torch.load(folder / 'reference.pt', weights_only=True)
    # Reproduce the capture's draws so classification dropout sees the same RNG.
    torch.testing.assert_close(torch.randn(2, 3, 64, 64), reference['image'], rtol=0, atol=0)
    torch.testing.assert_close(torch.randint(0, 21, (64, 64), dtype=torch.uint8),
                               reference['jpeg'][0]['bins'], rtol=0, atol=0)
    for training, key in ((False, 'eval'), (True, 'train')):
        model.train(training)
        output = model(reference['image'], jpeg=reference['jpeg'])
        assert output.keys() == reference[key].keys()
        for name, value in output.items():
            torch.testing.assert_close(value, reference[key][name], rtol=0, atol=0)
