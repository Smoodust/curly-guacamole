import io
import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from src.forensic.jpeg_input import JPEGInput


def test_signed_coefficients_preserve_sign_and_large_values():
    pixels=np.repeat(np.repeat(np.array([[20,128],[180,240]],np.uint8),8,0),8,1)
    stream=io.BytesIO(); Image.fromarray(pixels).save(stream,format='JPEG',qtables=[[8]*64])
    plain=JPEGInput.read(stream.getvalue())
    sample=JPEGInput.read(stream.getvalue(),include_coefficients=True)
    assert plain.coefficients is None
    np.testing.assert_array_equal(sample.coefficients[::8,::8],pixels[::8,::8].astype(int)-128)
    np.testing.assert_array_equal(sample.bins,np.minimum(np.abs(sample.coefficients.astype(int)),20))
    assert sample.coefficients.dtype==np.int16


def test_subblock_dct_matches_independent_spatial_transform():
    from scipy.fft import dctn,idctn
    from src.modules.jpeg_experiments import SubblockDCT
    rng=np.random.default_rng(12)
    c=rng.normal(size=(16,24)).astype(np.float32)
    q=rng.integers(1,30,size=(8,8)).astype(np.float32)
    pixels=np.empty_like(c)
    for y in range(0,16,8):
        for x in range(0,24,8):pixels[y:y+8,x:x+8]=idctn(c[y:y+8,x:x+8]*q,norm='ortho')
    expected=np.empty((16,4,6),np.float32)
    for y in range(4):
        for x in range(6):expected[:,y,x]=dctn(pixels[y*4:y*4+4,x*4:x*4+4],norm='ortho').reshape(-1)
    actual=SubblockDCT()(torch.tensor(c)[None],torch.tensor(q)[None])
    torch.testing.assert_close(actual,torch.tensor(expected)[None],atol=3e-4,rtol=3e-5)


@pytest.mark.parametrize('variant',['signed','attention','subblock4'])
def test_experiment_gradients_geometry_and_checkpoint(variant):
    from src.modules.jpeg_branch import JPEGBranch
    torch.manual_seed(4)
    model=JPEGBranch((8,12,16),variant=variant).eval()
    sample=dict(bins=torch.randint(0,21,(64,96),dtype=torch.uint8),coefficients=torch.randint(-200,200,(64,96),dtype=torch.int16),qtable=torch.ones(8,8),geometry=(8,8,40,64,1,1,0))
    sizes={4:(16,24),8:(8,12),16:(4,6),32:(2,3)}
    out=model([sample],sizes)
    assert set(out)==({4,8,16,32} if variant=='subblock4' else {8,16,32})
    for k,v in out.items(): assert v.shape[-2:]==sizes[k] and torch.isfinite(v).all()
    sum(v.square().mean() for v in out.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.enhancement.parameters())
    copy=JPEGBranch((8,12,16),variant=variant).eval();copy.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        for k,v in copy([sample],sizes).items():torch.testing.assert_close(v,out[k])

@pytest.mark.parametrize('variant', ['signed', 'subblock4'])
def test_coefficients_survive_dataset_recompression_and_transfer(tmp_path, variant):
    import pandas as pd
    from src.data.data_workspace import DataWorkspace
    from src.data.dataset import AIIJCDataset
    from src.data.collation import ValidationCollator
    from src.data.data_sample import DataSample
    from src.data.augmentation.transforms.random_jpeg_recompression import RandomJPEGRecompression
    from src.training.transfer import BatchTransfer
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir()
    image = np.random.default_rng(7).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    # PNG deliberately named .jpg; recompression must activate signed input too.
    Image.fromarray(image).save(workspace.train_root / 'source.jpg', format='PNG')
    rows = pd.DataFrame([dict(chng_img_path='source.jpg', gt_path='', target_kind='original_zero')])
    ds = AIIJCDataset(workspace, rows, False, 64, 42, mode='val', forensic_mode='jpeg', jpeg_variant=variant)
    batch = BatchTransfer('cpu')(ValidationCollator()([ds[0]]))
    assert batch['jpeg'][0]['coefficients'].dtype == torch.int16
    assert not batch['jpeg'][0]['available']
    native = JPEGInput.read(workspace.train_root / 'source.jpg', include_coefficients=True)
    result = RandomJPEGRecompression((60, 61), 1).apply(DataSample(image=image, jpeg=native), np.random.default_rng(8))
    assert result.jpeg.available and result.jpeg.coefficients.shape == (64, 96)
    np.testing.assert_array_equal(result.jpeg.bins, np.minimum(np.abs(result.jpeg.coefficients.astype(int)), 20))
    moved = BatchTransfer.move_jpeg([result.jpeg.crop(8, 16, 32, 48).transform(1, True, False).tensors()], 'cpu')[0]
    assert moved['geometry'] == (8, 16, 32, 48, 1, 1, 0)
    np.testing.assert_array_equal(moved['coefficients'].numpy(), result.jpeg.coefficients)


@pytest.mark.parametrize('variant', ['signed', 'attention', 'subblock4'])
def test_recipe_roundtrip_and_native_flops(variant):
    from dataclasses import replace
    from src.config import load_experiment_config, ModelConfig
    from src.inference.submission import InferenceConfig
    from src.training.builders import build_model
    from src.budget import count_gflops
    base = load_experiment_config('configs/jpeg576_pretrained.yaml')
    config = load_experiment_config(f'configs/jpeg576_pretrained_{variant}.yaml')
    assert config.model == replace(base.model, jpeg_variant=variant)
    assert config.train == base.train and config.loss == base.loss
    assert config.augmentation == base.augmentation and config.dataset == base.dataset
    assert InferenceConfig.from_snapshot(config.to_flat_dict()).model.jpeg_variant == variant
    with pytest.raises(ValueError):
        ModelConfig(jpeg_variant=variant)
    with torch.device('meta'):
        model = build_model(config.model, pretrained=False)
        baseline_model = build_model(base.model, pretrained=False)
    baseline_flops = count_gflops(baseline_model, 576, native_size=(1024, 1024))
    assert baseline_flops < count_gflops(model, 576, native_size=(1024, 1024)) < 75
