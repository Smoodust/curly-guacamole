"""Synchronize fixed-grid branches without changing native per-frame normalization."""

import torch
from torch import nn


class FusionSyncBatchNorm(nn.SyncBatchNorm):
    """Use matching FP32 collective payloads on empty and nonempty AMP ranks."""

    def forward(self, value):
        synchronize = (self.training and torch.distributed.is_initialized()
                       and torch.distributed.get_world_size() > 1)
        if synchronize and value.dtype in (torch.float16, torch.bfloat16):
            return super().forward(value.float()).to(value.dtype)
        return super().forward(value)

    @classmethod
    def convert(cls, module):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            result = cls(module.num_features, module.eps, module.momentum,
                         module.affine, module.track_running_stats)
            if module.affine:
                result.weight, result.bias = module.weight, module.bias
            result.running_mean, result.running_var = module.running_mean, module.running_var
            result.num_batches_tracked = module.num_batches_tracked
            result.training = module.training
            return result
        for name, child in module.named_children():
            setattr(module, name, cls.convert(child))
        return module


class SynchronizedBatchNorm:
    @staticmethod
    def apply(model):
        # Native/detail branches may execute a different number of forwards per
        # rank. Converting the entire model would introduce unmatched collectives.
        model.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(model.encoder)
        model.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(model.decoder)
        if model.forensic_fusion is not None:
            fusion = model.forensic_fusion
            fusion.fusion_blocks = FusionSyncBatchNorm.convert(fusion.fusion_blocks)
            fusion.sync_batchnorm = any(isinstance(layer, nn.SyncBatchNorm)
                                        for layer in fusion.fusion_blocks.modules())
        return model
