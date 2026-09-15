import numpy as np  # noqa: F401 -- Initialize NumPy before torch on Windows.
import torch

from src.config import TrainConfig
from src.training.builders import build_ema


def test_grouped_ema_matches_previous_updates_and_resumes():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    cfg = TrainConfig()
    actual = build_ema(cfg, model)
    expected = torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda avg, cur, _: cfg.ema_decay * avg + (1 - cfg.ema_decay) * cur)
    assert actual.multi_avg_fn is not None
    for step in range(8):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * .2)
            model[1].running_mean.fill_(step)
            model[1].num_batches_tracked.fill_(step)
        actual.update_parameters(model)
        expected.update_parameters(model)
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key], rtol=1e-5, atol=1e-6)
        if step == 3:
            resumed = build_ema(cfg, model)
            resumed.load_state_dict(actual.state_dict())
            actual = resumed
