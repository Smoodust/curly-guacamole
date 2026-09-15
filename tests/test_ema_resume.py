from dataclasses import replace

import numpy  # noqa: F401 - initialize NumPy before Torch in the Windows conda runtime.
import pytest
import torch

from src.config import load_experiment_config
from src.training.builders import build_amp, build_ema
from src.training.engine import ExperimentRunner
from src.training.runs import Run


@pytest.mark.parametrize('save_count', [False, True])
def test_resumed_ema_keeps_history_on_first_update(tmp_path, save_count):
    config = load_experiment_config('configs/baseline.yaml')
    config = replace(config, train=replace(config.train, device='cpu', amp='off', resume=True))
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = build_ema(config.train, model)
    ema.update_parameters(model)
    with torch.no_grad():
        model.weight.fill_(10)
    ema.update_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    scaler = build_amp(config.train).scaler()
    run = Run(tmp_path)
    (run.dir / 'ckpt').mkdir()
    checkpoint = dict(model=model.state_dict(), ema=ema.module.state_dict(),
                      optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=0)
    if save_count:
        checkpoint['ema_n_averaged'] = int(ema.n_averaged)
    run.save_state(checkpoint, 'last.pt')
    resumed = build_ema(config.train, model)
    ExperimentRunner(config)._resume_if_needed(
        run=run, model=model, ema=resumed, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    assert int(resumed.n_averaged) == (2 if save_count else 1)
    resumed.update_parameters(model)
    assert resumed.module.weight.item() == pytest.approx(.01999)
