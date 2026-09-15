import torch
from torch.utils.flop_counter import FlopCounterMode


def count_gflops(model, size: int, *, channels: int = 3,
                 native_size: tuple[int, int] | None = None) -> float:
    """Count a complete eval forward; JPEG cost depends on native image dimensions.

    This is an operation count, not a latency measurement. Module devices and
    training modes are preserved. A native size must be specified explicitly.
    """
    if native_size is None:
        raise ValueError('native_size is required for JPEG FLOP counting')
    if len(native_size) != 2 or any(type(value) is not int or value <= 0 for value in native_size):
        raise ValueError('native_size must contain two positive integers')
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    counter = FlopCounterMode(display=False)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()  # Exclude training-only auxiliary heads and preserve BN buffers.
        with torch.no_grad(), counter:
            h, w = native_size
            jpeg = [{'bins': torch.zeros(((h+7)//8*8, (w+7)//8*8), dtype=torch.uint8, device=device),
                     'qtable': torch.ones(8, 8, device=device), 'geometry': (0, 0, h, w, 0, 0, 0)}]
            model(torch.zeros(1, channels, size, size, device=device), jpeg=jpeg)
    finally:
        for module, training in modes:
            module.training = training
    return counter.get_total_flops() / 1e9
