import random

import numpy as np
import torch


class TorchRNGState:
    """Checkpoint the CPU/CUDA streams used by contrastive position sampling."""

    @staticmethod
    def capture():
        return {'cpu': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    @staticmethod
    def restore(state):
        torch.set_rng_state(state['cpu'].cpu())
        if state['cuda'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
