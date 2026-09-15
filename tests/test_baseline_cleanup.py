from pathlib import Path

import pytest

from src.config import ExperimentConfig, ModelConfig, TrainConfig, _ConfigLoader


def test_two_self_contained_baseline_recipes():
    assert {p.name for p in Path('configs').glob('*.yaml')} == {'baseline.yaml', 'baseline_long.yaml'}
    short = ExperimentConfig.from_dict(_ConfigLoader().load(Path('configs/baseline.yaml')))
    long = ExperimentConfig.from_dict(_ConfigLoader().load(Path('configs/baseline_long.yaml')))
    assert short.model == long.model
    assert short.model.encoder == 'pvt_v2_b2'
    assert short.model.jpeg_channels == (64, 96, 128)
    assert short.model.jpeg_pretrained == 'DCT_djpeg.pth'
    assert short.train.samples_per_epoch == long.train.samples_per_epoch == 24000
    assert (short.train.epochs, short.train.full_pass_epochs, short.augmentation.final_full_frame_epochs) == (6, 0, 2)
    assert (long.train.epochs, long.train.full_pass_epochs, long.augmentation.final_full_frame_epochs) == (18, 3, 3)
    assert short.loss.aux_weight == .4
    assert short.eval.selection_small_mask_weight == 1.6
    assert ExperimentConfig.from_dict(short.to_dict()) == short


def test_removed_experiment_switches_fail_explicitly():
    with pytest.raises(ValueError, match='unknown'):
        ModelConfig.from_dict({'wavelet_image_size': 1024})
    with pytest.raises(ValueError, match='unknown'):
        TrainConfig.from_dict({'jpeg_pair_training': True})


def test_migrate_selected_legacy_snapshot_and_reject_other_architecture():
    from src.config import SnapshotAdapter
    old = {'pipeline_version': 'emcad_v1', 'run_name': 'selected', 'seed': 42,
           'encoder_name': 'pvt_v2_b2', 'forensic_mode': 'jpeg', 'fusion_variant': 'local',
           'jpeg_variant': 'baseline', 'norm': 'batch', 'sync_batchnorm': True,
           'image_size': 640, 'aux_weight': .4, 'small_mask_weight': 1.6,
           'epochs': 18, 'epoch_size': 24000, 'full_train_epochs': 3,
           'final_full_frame_epochs': 3, 'batch_size': 8, 'accum_steps': 1}
    config = ExperimentConfig.from_dict(SnapshotAdapter.normalize(old))
    assert config.run_name == 'selected'
    assert config.train.full_pass_epochs == 3
    assert config.loss.aux_weight == .4
    assert config.train.grad_accum_steps == 1
    with pytest.raises(ValueError, match='historical|Unsupported'):
        SnapshotAdapter.normalize({**old, 'wavelet_image_size': 1024})


def test_current_snapshot_rejects_unknown_architecture_keys():
    from src.config import SnapshotAdapter
    config = ExperimentConfig(run_name='current').to_dict()
    config['model']['fusion_variant'] = 'spatial'
    with pytest.raises(ValueError, match='unknown.*fusion_variant'):
        SnapshotAdapter.normalize(config)


@pytest.mark.parametrize('head_weight,loss_weight', [(.4, 0.), (0., .4)])
def test_legacy_aux_override_cannot_change_head_presence(head_weight, loss_weight):
    from src.config import SnapshotAdapter
    old = {'pipeline_version': 'emcad_v1', 'forensic_mode': 'jpeg',
           'fusion_variant': 'local', 'sync_batchnorm': True,
           'aux_weight': head_weight, 'aux_loss_weight': loss_weight}
    with pytest.raises(ValueError, match='auxiliary'):
        SnapshotAdapter.normalize(old)


def test_finetune_rejects_unsupported_source_before_loading_weights(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    import src.training.engine as engine
    from src.training.engine import ExperimentRunner
    config = ExperimentConfig(run_name='target')
    config = replace(config, paths=replace(config.paths, runs_path=tmp_path),
                     train=replace(config.train, device='cpu', finetune_from='source.pt'))
    torch.save({'cfg': {'pipeline_version': 'emcad_v1', 'forensic_mode': 'jpeg',
                       'fusion_variant': 'local', 'sync_batchnorm': False},
                'model': {'weight': torch.ones(1, 1), 'bias': torch.zeros(1)}}, tmp_path / 'source.pt')
    monkeypatch.setattr(engine.EvaluationProtocol, 'load',
                        lambda path: SimpleNamespace(verify_run=lambda snapshot: None))
    model = torch.nn.Linear(1, 1)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match='historical architecture'):
        ExperimentRunner(config)._load_finetune_weights(model)
    torch.testing.assert_close(model.weight, before)
