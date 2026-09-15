"""Synchronize fixed image grids while retaining native JPEG frame statistics."""

from torch import nn


class SynchronizedBatchNorm:
    @staticmethod
    def apply(model):
        # Native JPEG frames run separately and their count can differ by rank.
        # Local fusion contains no BatchNorm; only fixed image grids synchronize.
        model.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(model.encoder)
        model.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(model.decoder)
        return model
