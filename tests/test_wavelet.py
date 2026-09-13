from dataclasses import replace

import numpy as np
import pytest
import torch


def test_wavelet_jpeg_dataset_loaders_and_budget(tmp_path):
    import pandas as pd
    from PIL import Image
    from src.config import load_experiment_config
    from src.training.builders import build_datasets, build_loaders, build_model
    from src.data.data_workspace import DataWorkspace
    from src.data.collation import ValidationCollator
    from src.training.transfer import BatchTransfer
    from src.budget import count_gflops

    config = load_experiment_config('configs/jpeg576_wavelet.yaml')
    workspace = DataWorkspace(tmp_path)
    workspace.train_root.mkdir()
    rows = []
    for i, shape in enumerate([(65, 77), (96, 128)]):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (*shape, 3), dtype=np.uint8)).save(
            workspace.train_root / f'{i}.jpg')
        rows.append(dict(chng_img_path=f'{i}.jpg', gt_path='', target_kind='original_zero', is_negative=bool(i)))
    small = replace(config, dataset=replace(config.dataset, image_size=64),
                    model=replace(config.model, wavelet_image_size=64))
    train, val = build_datasets(small, workspace, pd.DataFrame(rows), pd.DataFrame(rows))
    loaders = build_loaders(replace(small.train, workers=2, batch_size=2), train, val)
    assert all(isinstance(loader.collate_fn, ValidationCollator) for loader in loaders)
    assert all(loader.prefetch_factor == 1 and loader.batch_size == 2 for loader in loaders)
    batch = BatchTransfer('cpu')(ValidationCollator()([val[0], val[1]]))
    assert [tuple(x.shape) for x in batch['native_rgb']] == [(3, 65, 77), (3, 96, 128)]
    assert all(x.dtype == torch.uint8 for x in batch['native_rgb'])
    assert 'local_input' not in batch
    model = build_model(small.model, pretrained=False).eval()
    with torch.no_grad():
        output = model(batch['image'], native_rgb=batch['native_rgb'], jpeg=batch['jpeg'])
    assert output['logits'].shape == (2, 1, 64, 64)
    assert torch.isfinite(output['logits']).all()
    with pytest.raises(ValueError, match='native_rgb'):
        model(batch['image'], jpeg=batch['jpeg'])
    with torch.device('meta'):
        full = build_model(config.model, pretrained=False)
        baseline = build_model(replace(config.model, wavelet_image_size=0), pretrained=False)
    cost = count_gflops(full, 576, native_size=(1024, 1024))
    base_cost = count_gflops(baseline, 576, native_size=(1024, 1024))
    assert base_cost < cost < 100


@pytest.mark.parametrize('nested', [False, True])
def test_old_resume_defaults_wavelet_to_disabled(tmp_path, monkeypatch, nested):
    from src.config import load_experiment_config
    from src.training.engine import ExperimentRunner, EvaluationProtocol
    from src.training.runs import Run

    base = load_experiment_config('configs/baseline.yaml')
    config = replace(base, paths=replace(base.paths, runs_path=tmp_path),
                     train=replace(base.train, resume=True))
    run = Run.create(tmp_path, config.paths.run_name, tensorboard=False)
    snapshot = config.to_dict() if nested else config.to_flat_dict()
    (snapshot['model'] if nested else snapshot).pop('wavelet_image_size')
    run.save_snapshot(snapshot)
    (run.dir / 'ckpt' / 'last.pt').touch()
    class Protocol:
        def verify_run(self, saved):
            pass
    monkeypatch.setattr(EvaluationProtocol, 'load', lambda path: Protocol())
    ExperimentRunner(config)._check_resume_protocol()
    with pytest.raises(ValueError, match='wavelet_image_size'):
        ExperimentRunner(replace(config, model=replace(config.model, wavelet_image_size=64)))._check_resume_protocol()


def test_wavelet_checkpoint_to_submission_png(tmp_path):
    import pandas as pd
    from PIL import Image
    from src.config import load_experiment_config
    from src.training.builders import build_model
    from src.training.runs import Run
    from src.inference.submission import create_submission

    base = load_experiment_config('configs/jpeg576_wavelet.yaml')
    config = replace(base, dataset=replace(base.dataset, image_size=64),
                     model=replace(base.model, wavelet_image_size=64),
                     train=replace(base.train, device='cpu', amp='off', workers=0, batch_size=1))
    run = Run.create(tmp_path / 'runs', 'wavelet', tensorboard=False)
    run.save_snapshot(config.to_dict())
    run.save_summary({'best': {'mask_threshold': .5, 'cls_threshold': 0., 'min_area': 0.}})
    model = build_model(config.model, pretrained=False)
    torch.save({'model': model.state_dict()}, run.dir / 'ckpt' / 'best.pt')
    data = tmp_path / 'data'
    test_root = data / 'test_stage1' / 'test_stage1'
    test_root.mkdir(parents=True)
    Image.fromarray(np.zeros((65, 77, 3), np.uint8)).save(test_root / 'a.jpg')
    template = pd.DataFrame({'img_path': ['a.jpg'], 'prediction_path': ['predictions/a.png']})
    template[['img_path']].to_csv(test_root / 'test.csv', index=False)
    template.to_csv(test_root / 'submission.csv', index=False)
    path = create_submission(run.dir, tmp_path / 'output', data_path=data)
    pd.testing.assert_frame_equal(pd.read_csv(path), template)
    with Image.open(path.parent / 'predictions/a.png') as mask:
        assert mask.mode == 'L' and mask.size == (77, 65)
        assert set(np.unique(mask)).issubset({0, 255})


def test_haar_native_energy_survives_reduction_and_constant_is_zero():
    from src.modules.wavelet_branch import WaveletPreprocessor

    prep = WaveletPreprocessor(32)
    constant = torch.full((3, 35, 47), 127, dtype=torch.uint8)
    assert prep([constant]).abs().max() < 1e-6
    # Alternating signs on consecutive Haar cells cancel after area reduction.
    stripe = torch.tensor([255, 0, 0, 255], dtype=torch.uint8).repeat(16)
    image = stripe.expand(3, 64, 64).clone()
    features = prep([image])
    assert features.shape == (1, 6, 16, 16)
    assert features[:, :3].abs().max() < 1e-6
    torch.testing.assert_close(features[:, 3], torch.ones(1, 16, 16))
    assert features[:, 4:].abs().max() < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_wavelet_cuda_amp_preserves_extraction_and_finite_backward():
    from src.modules.wavelet_branch import WaveletBranch

    branch = WaveletBranch(64, 64, use_aux=True).cuda().to(memory_format=torch.channels_last).train()
    # HWC-backed CHW view follows the zero-copy CPU dataset path after transfer.
    native = [torch.randint(256, (65, 77, 3), dtype=torch.uint8, device='cuda').permute(2, 0, 1)]
    expected = branch.preprocess(native)
    context = torch.randn(1, 64, 16, 16, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        torch.testing.assert_close(branch.preprocess(native), expected, rtol=0, atol=0)
        output, aux = branch(native, context)
        loss = output.square().mean() + aux.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert branch.stem[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in branch.parameters() if p.grad is not None)


def test_wavelet_identity_aux_gradients_and_context_gradients():
    from src.modules.wavelet_branch import WaveletBranch

    torch.set_num_threads(1)
    branch = WaveletBranch(64, 64, use_aux=True).train()
    native = [torch.randint(256, (3, 65, 77), dtype=torch.uint8)]
    context = torch.randn(1, 64, 16, 16, requires_grad=True)
    output, aux = branch(native, context)
    torch.testing.assert_close(output, context, rtol=0, atol=0)
    aux.square().mean().backward()
    assert context.grad is None
    assert branch.stem[0].weight.grad.abs().sum() > 0
    branch.zero_grad(set_to_none=True)
    with torch.no_grad():
        branch.gamma.fill_(.1)
    branch(native, context)[0].square().mean().backward()
    assert branch.stem[0].weight.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs', [dict(wavelet_image_size=True), dict(wavelet_image_size=31),
    dict(wavelet_image_size=-32), dict(wavelet_image_size=64, luma_image_size=64),
    dict(wavelet_image_size=64, local_image_size=64),
    dict(wavelet_image_size=64, decoder_kwargs={'output_refinement_channels': 24})])
def test_wavelet_rejects_incompatible_config(kwargs):
    from src.config import ModelConfig
    with pytest.raises(ValueError, match='wavelet'):
        ModelConfig(**kwargs)


def test_wavelet_pipeline_training_prediction_and_snapshot():
    from src.config import load_experiment_config, ExperimentConfig
    from src.training.builders import build_model, build_amp, build_optimizer, build_scheduler, build_ema
    from src.training.engine import train_one_epoch
    from src.training.validation import validate
    from src.inference.predict import Predictor, ThresholdConfig
    from src.training.metric import score_masks

    base = load_experiment_config('configs/baseline.yaml')
    config = replace(base, model=replace(base.model, wavelet_image_size=64),
                     train=replace(base.train, device='cpu', amp='off', workers=0, accum_steps=1),
                     eval=replace(base.eval, mask_thresholds=(.5,), cls_thresholds=(0.,), min_areas=(0.,)))
    assert ExperimentConfig.from_dict(config.to_dict()).model == config.model
    from src.inference.submission import InferenceConfig
    assert InferenceConfig.from_snapshot(config.to_flat_dict()).model == config.model
    model = build_model(config.model, pretrained=False)
    amp = build_amp(config.train)
    optimizer = build_optimizer(config.train, model)
    target = torch.zeros(2, 1, 64, 64)
    target[0, :, 20:28, 36:44] = 1
    batch = dict(image=torch.randn(2, 3, 64, 64), fmap=torch.randn(2, 12, 8, 8),
                 native_rgb=[torch.randint(256, (3, h, w), dtype=torch.uint8) for h,w in [(96,128),(81,111)]],
                 mask=target, label=torch.tensor([[1.], [0.]]),
                 original_mask=[target[0,0].bool(),target[1,0].bool()],
                 original_size=torch.tensor([[64,64],[64,64]]), image_path=['a.png','b.png'])
    trained = train_one_epoch(model=model, loader=[batch], optimizer=optimizer,
        scheduler=build_scheduler(config.train, optimizer, 1), scaler=amp.scaler(),
        ema=build_ema(config.train, model), amp=amp, config=config, device=amp.device)
    assert trained.seen == 2 and np.isfinite(trained.loss)
    assert trained.loss_components['aux_bce'] > 0
    validation = validate(model, [batch], amp, config, amp.device)
    predictions = list(Predictor(model, ThresholdConfig(), amp).predict([batch]))
    direct = score_masks([p.mask for p in predictions], [m.numpy() for m in batch['original_mask']])
    assert validation.tuned.dice_pos == pytest.approx(direct.dice_pos)
    assert validation.tuned.fpr_neg == direct.fpr_neg
    restored = build_model(config.model, pretrained=False).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        args = (batch['image'], batch['fmap'])
        torch.testing.assert_close(restored(*args, native_rgb=batch['native_rgb'])['logits'],
                                   model.eval()(*args, native_rgb=batch['native_rgb'])['logits'])
