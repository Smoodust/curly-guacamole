# EMCAD decoder

`Segmenter` defaults to PVTv2-B2 with `EMCADDecoder`, the existing forensic
native JPEG branch and LocalFusionResidual blocks. Construct EMCAD directly:

```python
from src.decoders import EMCADDecoder

decoder = EMCADDecoder(
    encoder_channels=[64, 128, 320, 512],
    encoder_strides=[4, 8, 16, 32],
    norm="batch",
    use_aux=True,
)
features, aux_logits = decoder(encoder_features)
```

The four input features run from fine to coarse. Channels must be positive
and even. Output features have stride 4 and the finest encoder width;
explicit skip sizes support odd and rectangular geometry. Auxiliary logits
are returned only during training when enabled.

This adaptation uses channel attention, shared spatial attention, parallel
additive depthwise convolutions, channel shuffle, residual connections,
efficient upsampling, and grouped skip attention. Auxiliary supervision
uses the merged final skip before refinement. The paper's multiple mask
heads and training scheme are not reproduced.

`Segmenter` uses the default decoder settings: batch normalization, parallel
kernels 1/3/5, expansion factor 2, grouped attention kernel 3 and ReLU.
Both supported recipes supervise the final merged skip with weight 0.4.

Segmenter owns the mask and classification heads and restores logits to
input size. Module names (`encoder`, `decoder`, `forensic_fusion`,
`segmentation_head`, `classification_head`) retain the
existing EMCAD checkpoint layout. Removed UNet/SegFormer snapshots are
not supported by this API.

Architecture reference: [EMCAD, CVPR 2024](https://arxiv.org/abs/2405.06880).
