import pytest

import global_config
from src.config import ExperimentConfig, load_experiment_config


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(global_config, 'PROJECT_ROOT', tmp_path)
    for key in ('BATCH_SIZE', 'ACCUM_STEPS', 'AMP', 'DEVICE', 'WORKERS', 'DEVICES', 'DISTRIBUTED_BACKEND'):
        monkeypatch.delenv('AIIJC_' + key, raising=False)
    path = tmp_path / 'baseline.yaml'
    path.write_text('run_name: baseline\ntrain: {batch_size: 4}\n')
    return path


def test_runtime_precedence_and_snapshot_preservation(recipe, monkeypatch):
    recipe.with_name('.env').write_text(
        'AIIJC_BATCH_SIZE=8\nAIIJC_ACCUM_STEPS=2\nAIIJC_AMP=fp16\n'
        'AIIJC_DEVICE=cuda:1\nAIIJC_WORKERS=0\n')
    monkeypatch.setenv('AIIJC_BATCH_SIZE', '16')
    cfg = load_experiment_config(recipe)
    assert (cfg.train.batch_size, cfg.train.grad_accum_steps, cfg.train.amp,
            cfg.train.device, cfg.train.workers) == (16, 2, 'fp16', 'cuda:1', 0)
    snapshot = cfg.to_dict()
    assert snapshot['train']['batch_size'] == 16
    assert cfg.to_flat_dict()['amp'] == 'fp16'
    monkeypatch.setenv('AIIJC_BATCH_SIZE', '32')
    assert ExperimentConfig.from_dict(snapshot).train == cfg.train


def test_unset_runtime_preserves_recipe(recipe):
    assert load_experiment_config(recipe).train.batch_size == 4


def test_runtime_devices_precedence(recipe, monkeypatch):
    recipe.with_name('.env').write_text('AIIJC_DEVICES="[0, 1]"\nAIIJC_DISTRIBUTED_BACKEND=gloo\n')
    assert load_experiment_config(recipe).train.devices == (0, 1)
    monkeypatch.setenv('AIIJC_DEVICES', '[1, 2]')
    config = load_experiment_config(recipe)
    assert config.train.devices == (1, 2)
    assert config.train.distributed_backend == 'gloo'
    assert ExperimentConfig.from_dict(config.to_dict()).train.devices == (1, 2)


def test_malformed_devices_environment_is_value_error(recipe, monkeypatch):
    monkeypatch.setenv('AIIJC_DEVICES', '[0, 1')
    with pytest.raises(ValueError, match='AIIJC_DEVICES'):
        load_experiment_config(recipe)


@pytest.mark.parametrize('flat', [False, True])
def test_submission_data_path_uses_current_environment(recipe, monkeypatch, flat):
    from src.inference.submission import InferenceConfig

    monkeypatch.delenv('AIIJC_DATA_PATH', raising=False)
    cfg = load_experiment_config(recipe)
    snapshot = cfg.to_flat_dict() if flat else cfg.to_dict()
    paths = snapshot if flat else snapshot['paths']
    paths['data_path'] = '/old/server/data'
    recipe.with_name('.env').write_text('AIIJC_DATA_PATH=local_data\n')
    assert InferenceConfig.from_snapshot(snapshot).data_path == recipe.parent / 'local_data'
    monkeypatch.setenv('AIIJC_DATA_PATH', str(recipe.parent / 'override'))
    assert InferenceConfig.from_snapshot(snapshot).data_path == recipe.parent / 'override'
    del paths['data_path']
    assert InferenceConfig.from_snapshot(snapshot).data_path == recipe.parent / 'override'


@pytest.mark.parametrize('key,value', [('BATCH_SIZE', 'no'), ('ACCUM_STEPS', '0'),
                                     ('AMP', 'bad'), ('WORKERS', '-1')])
def test_invalid_runtime_is_rejected(recipe, monkeypatch, key, value):
    monkeypatch.setenv('AIIJC_' + key, value)
    with pytest.raises(ValueError):
        load_experiment_config(recipe)
