import torch.nn.functional as F
from torch import nn

from src.decoders import EMCADDecoder
from src.modules.forensic_fusion import ForensicFusion
from src.modules.gate_head import GateHead
from src.modules.utils import build_timm_encoder


class Segmenter(nn.Module):
    """PVT image encoder with native JPEG fusion and an EMCAD mask decoder."""

    def __init__(self, encoder="pvt_v2_b2", jpeg_channels=(64, 96, 128),
                 aux_weight=0.4, *, pretrained=True):
        super().__init__()
        # Construction order is part of reproducible baseline initialization.
        self.encoder, self.strides, self.channels = build_timm_encoder(
            encoder, pretrained=pretrained)
        self.forensic_fusion = ForensicFusion(self.strides, self.channels, jpeg_channels)
        self.decoder = EMCADDecoder(self.channels, self.strides, use_aux=aux_weight > 0)
        self.segmentation_head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)
        self.classification_head = GateHead(self.channels[-1])
        self.aux_weight = aux_weight

    def forensic_gate_stats(self) -> dict[str, float]:
        """Detached channel-gate statistics for training logs."""
        return self.forensic_fusion.gate_stats()

    def forward(self, image, *, jpeg):
        if not isinstance(jpeg, (list, tuple)) or len(jpeg) != image.shape[0]:
            raise ValueError('jpeg inputs must contain one native frame per image')
        input_size = image.shape[-2:]
        encoder_features = self.forensic_fusion(list(self.encoder(image)), jpeg=jpeg)
        decoder_features, aux_logits = self.decoder(encoder_features)
        result = {
            "logits": self._resize(self.segmentation_head(decoder_features), input_size),
            "cls_logits": self.classification_head(encoder_features[-1]),
        }
        if self.training and aux_logits is not None:
            result["aux_logits"] = self._resize(aux_logits, input_size)
        return result

    @staticmethod
    def _resize(features, size):
        if features.shape[-2:] == size:
            return features
        return F.interpolate(features, size=size, mode="bilinear", align_corners=False)
