from dataclasses import replace

import pytest

from src.config import load_experiment_config
from src.training.engine import ExperimentRunner


@pytest.mark.parametrize('resume,exists,expected', [(True, False, True), (True, True, False),
                                                  (False, True, True), (False, False, True)])
def test_pretrained_skipped_only_when_checkpoint_is_available(tmp_path, monkeypatch, resume, exists, expected):
    import src.training.engine as engine

    cfg = load_experiment_config('configs/baseline.yaml')
    cfg = replace(cfg, paths=replace(cfg.paths, runs_path=tmp_path),
                  train=replace(cfg.train, resume=resume, device='cpu'))
    if exists:
        checkpoint = tmp_path / cfg.run_name / 'ckpt' / 'last.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    calls = []

    class Model:
        def to(self, device):
            return self

    def build(config, *, pretrained, aux_weight):
        calls.append(pretrained)
        return Model()

    monkeypatch.setattr(engine, 'build_model', build)
    monkeypatch.setattr(engine, 'configure_memory_format', lambda model: model)
    ExperimentRunner(cfg)._build_training_model()
    assert calls == [expected]
