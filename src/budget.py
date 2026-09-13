import torch
from torch.utils.flop_counter import FlopCounterMode


def count_gflops(model, size: int, *, channels: int = 3, use_valid_mask: bool = False,
                 native_size: tuple[int, int] | None = None) -> float:
    """GFLOPs одного eval forward по FlopCounterMode.

    Счётчик не учитывает все операции, в частности RGB->Y и interpolation.
    native_size задаёт геометрию входа luma/wavelet/JPEG, но не оценивает runtime/память.
    Для JPEG размер обязателен: стоимость зависит от исходника, а не только RGB.

    Модель считается там, где лежит: перекладывать её здесь нельзя, иначе
    вызывающий получил бы обратно испорченный объект. На `meta`-устройстве
    работает и даёт то же число, но без арифметики и без памяти — доли секунды
    против нескольких секунд на 768px.
    """

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    counter = FlopCounterMode(display=False)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()  # Exclude training-only auxiliary heads and preserve BN buffers.
        with torch.no_grad(), counter:
            kwargs = ({"valid_mask": torch.ones(1, 1, size, size, device=device, dtype=torch.bool)}
                      if use_valid_mask else {})
            local_size = getattr(model, 'local_image_size', 0)
            if local_size:
                kwargs['local_input'] = torch.zeros(1, 15, local_size, local_size, device=device)
            if getattr(model, 'forensic_mode', 'maps') == 'jpeg':
                if native_size is None:
                    raise ValueError('native_size is required for JPEG FLOP counting')
                h, w = native_size
                kwargs['jpeg'] = [{'bins': torch.zeros(((h+7)//8*8, (w+7)//8*8), dtype=torch.uint8, device=device),
                                  'qtable': torch.ones(8, 8, device=device),
                                  'geometry': (0, 0, h, w, 0, 0, 0)}]
                if getattr(model, 'jpeg_variant', 'baseline') in {'signed', 'subblock4'}:
                    kwargs['jpeg'][0]['coefficients'] = torch.zeros_like(kwargs['jpeg'][0]['bins'], dtype=torch.int16)
            detail_size = getattr(model, 'luma_image_size', 0) or getattr(model, 'wavelet_image_size', 0)
            if detail_size:
                height, width = native_size or (detail_size, detail_size)
                kwargs['native_rgb'] = [torch.zeros(3, height, width, dtype=torch.uint8, device=device)]
            input_size = size * 2 if getattr(model, 'strided_resize', False) else size
            model(torch.zeros(1, channels, input_size, input_size, device=device), **kwargs)
    finally:
        for module, training in modes:
            module.training = training
    return counter.get_total_flops() / 1e9
