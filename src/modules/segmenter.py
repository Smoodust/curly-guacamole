import torch
import torch.nn.functional as F
from torch import nn

from src.decoders import EMCADDecoder
from src.forensic.dct.constants import CHANNEL_COUNT, STRIDE
from src.modules.forensic_fusion import ForensicFusion
from src.modules.gate_head import GateHead
from src.modules.local_branch import LocalBranch
from src.modules.luma_branch import LumaBranch
from src.modules.strided_resize import StridedResize
from src.modules.utils import build_timm_encoder
from src.modules.wavelet_branch import MultiScaleWaveletBranch, WaveletBranch


class Segmenter(nn.Module):
    """Сегментатор с pretrained encoder и forensic-веткой."""

    FORENSIC_INPUT_CHANNELS = CHANNEL_COUNT

    def __init__(
        self,
        encoder_name="pvt_v2_b2",
        forensic_channels=(64, 96, 128),
        norm="batch",
        aux_weight=0.0,
        *,
        pretrained=True,
        use_forensics=True,
        forensic_mode='maps',
        jpeg_variant='baseline',
        fusion_variant='baseline',
        dct_aux_weight=0.0,
        forensic_contrastive_dim=0,
        decoder_kwargs=None,
        local_image_size=0,
        luma_image_size=0,
        wavelet_image_size=0,
        wavelet_fusion='late',
        strided_resize=False,
        resize_variant='linear',
        noise_encoder_name=None,
        guided_radius=2,
        guided_epsilon=.01,
        guided_scale=.25,
        dual_fusion_width=16,
    ):
        super().__init__()
        if type(forensic_contrastive_dim) is not int or forensic_contrastive_dim < 0:
            raise ValueError('forensic_contrastive_dim must be an integer >= 0')
        if forensic_contrastive_dim and not use_forensics:
            raise ValueError('forensic_contrastive_dim requires use_forensics')
        if type(wavelet_image_size) is not int or wavelet_image_size < 0 or wavelet_image_size % 32:
            raise ValueError('wavelet_image_size must be 0 or a positive multiple of 32')
        if wavelet_image_size and (local_image_size or luma_image_size or strided_resize or
                any((decoder_kwargs or {}).get(k, 0) for k in
                    ('rgb_refinement_channels', 'output_refinement_channels'))):
            raise ValueError('wavelet_image_size requires EMCAD without other detail/resize branches')
        self.wavelet_image_size = wavelet_image_size
        if wavelet_fusion not in {'late', 'stride4', 'stride8', 'stride4_stride8_late'}:
            raise ValueError('wavelet_fusion must be late, stride4, stride8 or stride4_stride8_late')
        if wavelet_fusion != 'late' and not wavelet_image_size:
            raise ValueError('early wavelet_fusion requires wavelet_image_size')
        self.wavelet_fusion = wavelet_fusion
        if strided_resize and (use_forensics or local_image_size or luma_image_size):
            raise ValueError('strided_resize requires RGB-only without local/luma branches')
        self.strided_resize = strided_resize
        if resize_variant != 'linear' and not strided_resize:
            raise ValueError('resize_variant requires strided_resize')
        self.input_resize = StridedResize(resize_variant) if strided_resize else None
        self.forensic_mode = forensic_mode
        self.jpeg_variant = jpeg_variant
        if type(local_image_size) is not int or local_image_size < 0 or local_image_size % 32:
            raise ValueError('local_image_size must be 0 or a positive multiple of 32')
        if local_image_size and any(
                (decoder_kwargs or {}).get(key, 0)
                for key in ('rgb_refinement_channels', 'output_refinement_channels')):
            raise ValueError('local_image_size requires EMCAD without other refinement experiments')
        self.local_image_size = local_image_size
        if type(luma_image_size) is not int or luma_image_size < 0 or luma_image_size % 32:
            raise ValueError('luma_image_size must be 0 or a positive multiple of 32')
        if luma_image_size and (local_image_size or any((decoder_kwargs or {}).get(key, 0)
                for key in ('rgb_refinement_channels', 'output_refinement_channels'))):
            raise ValueError('luma_image_size requires EMCAD without other detail branches')
        self.luma_image_size = luma_image_size

        self.noise_encoder_name = noise_encoder_name
        if noise_encoder_name is not None:
            from src.modules.dual_encoder import DualEncoder
            if not wavelet_image_size or strided_resize:
                raise ValueError('dual encoder requires wavelet native input without strided resize')
            self.encoder = DualEncoder(encoder_name, noise_encoder_name, pretrained=pretrained,
                                       radius=guided_radius, epsilon=guided_epsilon,
                                       scale=guided_scale, hidden=dual_fusion_width)
            self.strides, self.channels = list(self.encoder.strides), list(self.encoder.channels)
        else:
            self.encoder, self.strides, self.channels = build_timm_encoder(
                encoder_name, pretrained=pretrained
            )

        if dct_aux_weight > 0 and not use_forensics:
            raise ValueError("DCT auxiliary head requires use_forensics")
        self.forensic_fusion = ForensicFusion(
            self.strides,
            self.channels,
            forensic_channels,
            use_aux=dct_aux_weight > 0,
            forensic_mode=forensic_mode,
            jpeg_variant=jpeg_variant,
            fusion_variant=fusion_variant,
            forensic_contrastive_dim=forensic_contrastive_dim,
        ) if use_forensics else None

        self.decoder = EMCADDecoder(
            encoder_channels=self.channels,
            encoder_strides=self.strides,
            norm=norm,
            use_aux=aux_weight > 0 and not (local_image_size or luma_image_size or wavelet_image_size),
            **(decoder_kwargs or {}),
        )
        self.local_branch = LocalBranch(self.decoder.out_channels, aux_weight > 0) if local_image_size else None
        self.luma_branch = LumaBranch(self.decoder.out_channels, luma_image_size, aux_weight > 0) if luma_image_size else None
        wavelet_stride = {'stride4': 4, 'stride8': 8}.get(wavelet_fusion)
        if wavelet_stride is not None and wavelet_stride not in self.strides:
            raise ValueError(f'wavelet_fusion={wavelet_fusion} requires encoder stride {wavelet_stride}')
        self.wavelet_feature_index = self.strides.index(wavelet_stride) if wavelet_stride is not None else None
        wavelet_channels = (self.channels[self.wavelet_feature_index] if wavelet_stride is not None
                            else self.decoder.out_channels)
        self.wavelet_feature_indices = {}
        if wavelet_fusion == 'stride4_stride8_late':
            if any(stride not in self.strides for stride in (4, 8)):
                raise ValueError('wavelet_fusion=stride4_stride8_late requires encoder strides 4 and 8')
            self.wavelet_feature_indices = {f'stride{stride}': self.strides.index(stride) for stride in (4, 8)}
            early_channels = {name: self.channels[index] for name, index in self.wavelet_feature_indices.items()}
            self.wavelet_branch = MultiScaleWaveletBranch(
                self.decoder.out_channels, early_channels, wavelet_image_size, aux_weight > 0)
        else:
            self.wavelet_branch = WaveletBranch(wavelet_channels, wavelet_image_size, aux_weight > 0) if wavelet_image_size else None

        head_kernel = self.decoder.head_kernel_size
        self.segmentation_head = nn.Conv2d(
            self.decoder.out_channels,
            1,
            kernel_size=head_kernel,
            padding=head_kernel // 2,
        )

        self.classification_head = GateHead(
            self.channels[-1]
        )

        self.aux_weight = aux_weight

    def forensic_gate_stats(self) -> dict[str, float]:
        """Detached gate statistics for experiment logging."""
        return self.forensic_fusion.gate_stats() if self.forensic_fusion is not None else {"max_abs": 0.0}

    def forward(self, image, forensic_map=None, valid_mask=None, *, local_input=None, native_rgb=None, jpeg=None):
        if self.input_resize is not None:
            if valid_mask is not None:
                raise ValueError('strided_resize requires stretch geometry')
            image = self.input_resize(image)
        input_size = image.shape[-2:]
        if self.forensic_mode == 'jpeg':
            if not isinstance(jpeg, (list, tuple)) or len(jpeg) != image.shape[0]:
                raise ValueError('jpeg inputs must contain one native frame per image')
            if valid_mask is not None:
                raise ValueError('jpeg inputs require stretch geometry')
        elif jpeg is not None:
            raise ValueError('jpeg inputs require forensic_mode=jpeg')
        if self.luma_branch is not None or self.wavelet_branch is not None:
            if not isinstance(native_rgb, (list, tuple)) or len(native_rgb) != image.shape[0]:
                raise ValueError('native_rgb must contain one native image per batch sample')
            if valid_mask is not None:
                raise ValueError('native_rgb currently requires stretch geometry')
        elif native_rgb is not None:
            raise ValueError('native_rgb supplied with native detail branches disabled')
        if self.local_branch is not None:
            expected = (image.shape[0], 15, self.local_image_size, self.local_image_size)
            if local_input is None or tuple(local_input.shape) != expected:
                raise ValueError(f'local_input must have shape {expected}')
            if valid_mask is not None:
                raise ValueError('local_input currently requires stretch geometry without valid_mask')
        elif local_input is not None:
            raise ValueError('local_input supplied to a model with the local branch disabled')

        encoder_features = list(
            self.encoder(image, native_rgb) if self.noise_encoder_name is not None else self.encoder(image)
        )

        dct_aux_logits = None
        forensic_result = None
        if self.forensic_fusion is not None:
            if forensic_map is None:
                forensic_map = self._empty_forensic_map(image)
            if self.training:
                forensic_result = self.forensic_fusion.forward_training(
                    encoder_features, forensic_map, jpeg=jpeg,
                )
                encoder_features, dct_aux_logits = forensic_result.features, forensic_result.aux_logits
            else:
                encoder_features = self.forensic_fusion(encoder_features, forensic_map, jpeg=jpeg)

        wavelet_aux = None
        wavelet_detail = None
        if self.wavelet_feature_indices:
            wavelet_detail, wavelet_aux = self.wavelet_branch.extract(native_rgb)
            for name, index in self.wavelet_feature_indices.items():
                encoder_features[index] = self.wavelet_branch.early_fusions[name](wavelet_detail, encoder_features[index])
        elif self.wavelet_branch is not None and self.wavelet_feature_index is not None:
            index = self.wavelet_feature_index
            encoder_features[index], wavelet_aux = self.wavelet_branch(native_rgb, encoder_features[index])

        decoder_features, aux_logits = self.decoder(
            encoder_features
        )
        if wavelet_aux is not None:
            aux_logits = wavelet_aux
        if self.local_branch is not None:
            decoder_features, aux_logits = self.local_branch(local_input, decoder_features)
        if self.luma_branch is not None:
            decoder_features, aux_logits = self.luma_branch(native_rgb, decoder_features)
        if self.wavelet_branch is not None and self.wavelet_fusion == 'late':
            decoder_features, aux_logits = self.wavelet_branch(native_rgb, decoder_features)
        elif wavelet_detail is not None:
            decoder_features = self.wavelet_branch.fuse(wavelet_detail, decoder_features)

        logits = self.segmentation_head(
            decoder_features
        )
        logits = self.decoder.refine_logits(image, decoder_features, logits)

        result = {
            "logits": self._resize(logits, input_size),
            "cls_logits": self.classification_head(
                encoder_features[-1], valid_mask=valid_mask
            ),
        }

        if self.training and aux_logits is not None:
            result["aux_logits"] = self._resize(
                aux_logits,
                input_size,
            )

        if dct_aux_logits is not None:
            result["dct_aux_logits"] = dct_aux_logits
        if forensic_result is not None and forensic_result.embeddings is not None:
            result['forensic_embeddings'] = forensic_result.embeddings
            result['forensic_available'] = forensic_result.available
        if self.training and self.forensic_mode == 'jpeg' and self.forensic_fusion.aux_head is not None:
            result['dct_aux_available'] = torch.tensor(
                [sample.get('available', True) for sample in jpeg], device=image.device, dtype=torch.bool)
        return result

    def _empty_forensic_map(self, image):
        height, width = image.shape[-2:]

        return image.new_zeros(
            image.shape[0],
            self.FORENSIC_INPUT_CHANNELS,
            height // STRIDE,
            width // STRIDE,
        )

    @staticmethod
    def _resize(features, size):
        if features.shape[-2:] == size:
            return features

        return F.interpolate(
            features,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
