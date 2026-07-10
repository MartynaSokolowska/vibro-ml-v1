from __future__ import annotations

import torch.nn as nn

from src.models.audio_backbones import build_audio_model, resolve_model_settings


class VibroNet(nn.Module):
    """Configurable audio classifier / regressor.
    Supports multiple audio backbones:
    - resnet18 (current baseline)
    - cnn14 / panns / panns_cnn14
    - ast
    - htsat / hts-at
    - beats
    """

    def __init__(
        self,
        mode: str = "classification",
        num_classes: int = 8,
        backbone: str | None = None,
        config: dict | None = None,
        pretrained: bool | None = None,
        input_channels: int | None = None,
    ):
        super().__init__()

        settings = resolve_model_settings(config=config, backbone=backbone, input_channels=input_channels)
        self.backbone_name = settings.backbone
        self.mode = mode
        self.num_classes = num_classes

        pretrained_weights = None
        if isinstance(config, dict):
            pretrained_weights = config.get('model', {}).get('pretrained_weights')

        self.model = build_audio_model(
            backbone=settings.backbone,
            mode=mode,
            num_classes=num_classes,
            pretrained=settings.pretrained if pretrained is None else pretrained,
            input_channels=settings.input_channels,
            pretrained_weights=pretrained_weights,
            config=config,
        )

    def forward(self, x):
        return self.model(x)
