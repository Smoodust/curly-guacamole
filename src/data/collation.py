from torch.utils.data import default_collate


class ValidationCollator:
    """Stack fixed-size inputs; keep native RGB and original masks as lists."""

    def __call__(self, samples):
        batch = default_collate([
            {key: value for key, value in sample.items() if key not in ("original_mask", "native_rgb", "jpeg", "paired_jpeg")}
            for sample in samples
        ])
        if "original_mask" in samples[0]:
            batch["original_mask"] = [sample["original_mask"] for sample in samples]
        if "native_rgb" in samples[0]:
            batch["native_rgb"] = [sample["native_rgb"] for sample in samples]
        if 'jpeg' in samples[0]:
            batch['jpeg'] = [sample['jpeg'] for sample in samples]
        if 'paired_jpeg' in samples[0]:
            batch['paired_jpeg'] = [sample['paired_jpeg'] for sample in samples]
        return batch
