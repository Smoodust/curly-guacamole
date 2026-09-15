import timm


def build_timm_encoder(name: str, *, pretrained: bool = True):
    encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=3)
    reductions = list(encoder.feature_info.reduction())
    channels = list(encoder.feature_info.channels())
    return encoder, reductions, channels
