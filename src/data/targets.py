import numpy as np
import torch


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Бинаризует mask и возвращает тензор [1, H, W]."""
    binary_mask = (mask >= 0.5).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)
