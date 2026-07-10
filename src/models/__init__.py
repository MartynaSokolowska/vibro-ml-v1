"""Model backbones and model-building helpers for vibroacoustic experiments."""

from .audio_backbones import (
    AudioResNet18,
    CNN14Backbone,
    ModelSettings,
    SpectrogramTransformerBackbone,
    build_audio_model,
    canonical_backbone_name,
    resolve_model_settings,
)

__all__ = [
    "AudioResNet18",
    "CNN14Backbone",
    "ModelSettings",
    "SpectrogramTransformerBackbone",
    "build_audio_model",
    "canonical_backbone_name",
    "resolve_model_settings",
]
