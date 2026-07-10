from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.hub import load_state_dict_from_url

from .third_party.panns.models import Cnn14


@dataclass(frozen=True)
class ModelSettings:
    backbone: str
    input_channels: int
    spectrogram_size: Optional[Tuple[int, int]]
    pretrained: bool


_BACKBONE_ALIASES = {
    "resnet": "resnet18",
    "resnet18": "resnet18",
    "cnn14": "cnn14",
    "panns": "cnn14",
    "pannscnn14": "cnn14",
    "ast": "ast",
    "audiospectrogramtransformer": "ast",
    "htsat": "htsat",
    "beats": "beats",
}

_BACKBONE_PRESETS = {
    "resnet18": {"input_channels": 3, "spectrogram_size": None},
    "cnn14": {"input_channels": 1, "spectrogram_size": None},
    "ast": {"input_channels": 1, "spectrogram_size": (128, 224)},
    "htsat": {"input_channels": 1, "spectrogram_size": (128, 224)},
    "beats": {"input_channels": 1, "spectrogram_size": (128, 256)},
}

_TRANSFORMER_PRESETS = {
    "ast": {"patch_size": (16, 16), "embed_dim": 256, "depth": 6, "num_heads": 8, "dropout": 0.1},
    "htsat": {"patch_size": (8, 8), "embed_dim": 224, "depth": 8, "num_heads": 8, "dropout": 0.1},
    "beats": {"patch_size": (4, 4), "embed_dim": 320, "depth": 10, "num_heads": 8, "dropout": 0.15},
}


def canonical_backbone_name(name: Optional[str]) -> str:
    if not name:
        return "resnet18"
    normalized = str(name).strip().lower().replace("_", "").replace("-", "").replace("/", "")
    return _BACKBONE_ALIASES.get(normalized, normalized)


def _parse_spectrogram_size(value: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("spectrogram_size must contain exactly two values: [mel_bins, frames]")
    return int(value[0]), int(value[1])


def resolve_model_settings(
    config: Optional[dict] = None,
    backbone: Optional[str] = None,
    input_channels: Optional[int] = None,
) -> ModelSettings:
    model_config = (config or {}).get("model", {}) if isinstance(config, dict) else {}
    resolved_backbone = canonical_backbone_name(
        backbone or model_config.get("backbone") or model_config.get("architecture") or "resnet18"
    )

    if resolved_backbone not in _BACKBONE_PRESETS:
        raise ValueError(
            f"Unsupported backbone '{resolved_backbone}'. Available: {', '.join(sorted(_BACKBONE_PRESETS))}"
        )

    preset = _BACKBONE_PRESETS[resolved_backbone]
    resolved_input_channels = int(
        input_channels if input_channels is not None else model_config.get("input_channels", preset["input_channels"])
    )
    resolved_spectrogram_size = _parse_spectrogram_size(model_config.get("spectrogram_size") or preset["spectrogram_size"])
    pretrained = bool(model_config.get("pretrained", True))

    return ModelSettings(
        backbone=resolved_backbone,
        input_channels=resolved_input_channels,
        spectrogram_size=resolved_spectrogram_size,
        pretrained=pretrained,
    )


def _adapt_first_conv(conv: nn.Conv2d, input_channels: int) -> nn.Conv2d:
    kernel_size = cast(tuple[int, int], conv.kernel_size)
    stride = cast(tuple[int, int], conv.stride)
    padding = cast(tuple[int, int], conv.padding)
    dilation = cast(tuple[int, int], conv.dilation)

    new_conv = nn.Conv2d(
        in_channels=input_channels,
        out_channels=conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if input_channels == conv.in_channels:
            new_conv.weight.copy_(conv.weight)
        elif conv.in_channels == 3 and input_channels == 1:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        elif conv.in_channels == 1 and input_channels > 1:
            new_conv.weight.copy_(conv.weight.repeat(1, input_channels, 1, 1) / float(input_channels))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class _AudioHead(nn.Module):
    def __init__(self, in_features: int, mode: str, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        output_dim = 1 if mode == "regression" else num_classes
        self.fc = nn.Linear(in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class AudioResNet18(nn.Module):
    def __init__(self, mode: str, num_classes: int, input_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        if input_channels != backbone.conv1.in_channels:
            backbone.conv1 = _adapt_first_conv(backbone.conv1, input_channels)

        backbone.fc = _AudioHead(backbone.fc.in_features, mode=mode, num_classes=num_classes, dropout=0.0)
        self.backbone = backbone
        self.backbone_name = "resnet18"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _build_panns_cnn14_kwargs(
    num_classes: int,
    config: Optional[dict] = None,
) -> dict:
    preprocessing_config = (config or {}).get("preprocessing", {}) if isinstance(config, dict) else {}
    mel_config = preprocessing_config.get("mel_spectrogram", {}) if isinstance(preprocessing_config, dict) else {}

    mel_bins = int(mel_config.get("n_mels", 64))

    # Cnn14's bn0 is hardcoded as nn.BatchNorm2d(64) -- it is NOT parameterized by
    # mel_bins. Any value other than 64 will fail with a channel-mismatch error inside
    # bn0, before the input even reaches the conv blocks. 64 is also the value CNN14's
    # official pretrained checkpoints were trained with.
    _REQUIRED_MEL_BINS_FOR_CNN14 = 64
    if mel_bins != _REQUIRED_MEL_BINS_FOR_CNN14:
        raise ValueError(
            f"CNN14 requires mel_bins == {_REQUIRED_MEL_BINS_FOR_CNN14} (got {mel_bins} from "
            "config['preprocessing']['mel_spectrogram']['n_mels']). CNN14's bn0 layer is "
            "hardcoded to 64 channels, so any other value fails inside the model. Set "
            "n_mels to 64 to match CNN14's architecture and pretrained weights, or use a "
            "different backbone for other mel-bin resolutions."
        )

    # Cnn14's real constructor signature is (sample_rate, window_size, hop_size, mel_bins,
    # fmin, fmax, classes_num) -- no enable_specaug/enable_mixup/dropout params exist.
    # spec_augmenter is unconditionally built and only applied when self.training is
    # True; we disable it explicitly after construction below instead. Mixup is applied
    # via a mixup_lambda argument to forward(), which CNN14Backbone.forward() never
    # passes, so it's already inert by default.
    return {
        "sample_rate": int((config or {}).get("data", {}).get("sample_rate", 16000)) if isinstance(config, dict) else 16000,
        "window_size": int(mel_config.get("n_fft", 1024)),
        "hop_size": int(mel_config.get("hop_length", 320)),
        "mel_bins": mel_bins,
        "fmin": float(mel_config.get("f_min", 50.0)),
        "fmax": float(mel_config.get("f_max", 8000.0)),
        "classes_num": 527,
    }


class CNN14Backbone(nn.Module):
    def __init__(
        self,
        mode: str,
        num_classes: int,
        input_channels: int = 1,
        pretrained: bool = True,
        config: Optional[dict] = None,
    ):
        super().__init__()

        del pretrained  # The vendored PANNs model is initialized independently and loaded via checkpoints.

        self.num_classes = num_classes
        cnn14_kwargs = _build_panns_cnn14_kwargs(num_classes=num_classes, config=config)
        self.panns = Cnn14(**cnn14_kwargs)

        # CNN14 halves the time axis 5 times (conv_block1-5, each pool_size=(2, 2)), same
        # as it does the frequency axis. A clip that's too short in raw samples produces
        # too few spectrogram time-frames and collapses to zero mid-network, failing deep
        # inside torchlibrosa with a confusing shape error. Compute the minimum number of
        # raw samples needed and validate it explicitly in forward() instead.
        self._hop_size = cnn14_kwargs["hop_size"]
        self._min_time_frames = 32  # 2 ** 5
        self._min_input_samples = (self._min_time_frames - 1) * self._hop_size

        # Upstream Cnn14 always builds spec_augmenter and applies it whenever
        # self.training is True -- there's no constructor flag to disable it. Swap it
        # for a no-op when we don't want augmentation, regardless of train/eval mode.
        # (Mixup is applied via a mixup_lambda argument to forward(), which we never
        # pass, so it's already inert -- no equivalent guard needed for it.)
        model_config = (config or {}).get("model", {}) if isinstance(config, dict) else {}
        enable_specaug = bool(model_config.get("enable_specaug", False))
        if not enable_specaug and hasattr(self.panns, "spec_augmenter"):
            self.panns.spec_augmenter = nn.Identity()

        self.head = _AudioHead(self._infer_embedding_dim(), mode=mode, num_classes=num_classes, dropout=0.3)
        self.input_channels = input_channels
        self.backbone_name = "cnn14"

    def _infer_embedding_dim(self) -> int:
        for attr_name in ("fc1", "embedding", "fc", "classifier"):
            layer = getattr(self.panns, attr_name, None)
            if hasattr(layer, "in_features"):
                return int(layer.in_features)
            if hasattr(layer, "out_features"):
                return int(layer.out_features)
        return 2048

    def _extract_embedding(self, output: object) -> torch.Tensor:
        if isinstance(output, dict):
            for key in ("embedding", "clipwise_output", "logits", "logit"):
                if key in output and torch.is_tensor(output[key]):
                    tensor = cast(torch.Tensor, output[key])
                    if tensor.dim() >= 2 and tensor.shape[-1] in (1, self.num_classes):
                        return tensor
                    return tensor
            raise ValueError("PANNs CNN14 forward() did not return a usable tensor embedding")

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item) and item.dim() >= 2 and item.shape[-1] in (1, self.num_classes):
                    return cast(torch.Tensor, item)
            for item in output:
                if torch.is_tensor(item):
                    return cast(torch.Tensor, item)
            raise ValueError("PANNs CNN14 forward() returned no tensor outputs")

        if torch.is_tensor(output):
            return cast(torch.Tensor, output)

        raise TypeError(f"Unsupported PANNs CNN14 output type: {type(output)!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.mean(dim=1)
        elif x.dim() == 2:
            pass
        else:
            x = x.reshape(x.shape[0], -1)

        if x.dim() == 2 and x.shape[-1] < self._min_input_samples:
            raise ValueError(
                f"CNN14 input clip has {x.shape[-1]} raw samples, which is too short. "
                f"CNN14 halves the spectrogram time axis {int(math.log2(self._min_time_frames))} times, "
                f"so it needs at least {self._min_time_frames} time-frames "
                f"(>= {self._min_input_samples} raw samples at hop_size={self._hop_size}) per clip "
                "to avoid collapsing to zero mid-network. Increase your segment/window length "
                "(or pad/tile short vibration windows up to at least this many samples), or use "
                "a shallower backbone (e.g. resnet18) that's sized for short windows."
            )

        embedding = self._extract_embedding(self.panns(x))
        if embedding.dim() >= 2 and embedding.shape[-1] in (1, self.num_classes):
            return embedding
        if embedding.dim() > 2:
            embedding = embedding.flatten(1)
        return self.head(embedding)


class SpectrogramTransformerBackbone(nn.Module):
    def __init__(
        self,
        mode: str,
        num_classes: int,
        input_channels: int,
        variant: str,
        spectrogram_size: Tuple[int, int],
        pretrained: bool = True,
    ):
        super().__init__()
        del pretrained  # These are training-friendly transformer variants.

        if variant not in _TRANSFORMER_PRESETS:
            raise ValueError(f"Unsupported transformer variant '{variant}'")

        preset = _TRANSFORMER_PRESETS[variant]
        patch_size = preset["patch_size"]
        embed_dim = preset["embed_dim"]
        depth = preset["depth"]
        num_heads = preset["num_heads"]
        dropout = preset["dropout"]

        self.variant = variant
        self.backbone_name = variant
        self.spectrogram_size = spectrogram_size
        self.patch_size = patch_size
        self.grid_size = (
            max(1, math.ceil(spectrogram_size[0] / patch_size[0])),
            max(1, math.ceil(spectrogram_size[1] / patch_size[1])),
        )

        self.patch_embed = nn.Conv2d(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = _AudioHead(embed_dim, mode=mode, num_classes=num_classes, dropout=dropout)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _interpolate_pos_embed(self, target_grid: Tuple[int, int]) -> torch.Tensor:
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        if target_grid == self.grid_size:
            return self.pos_embed

        patch_pos = patch_pos.reshape(1, self.grid_size[0], self.grid_size[1], -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=target_grid, mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_grid[0] * target_grid[1], -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        grid_h, grid_w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        pos_embed = self._interpolate_pos_embed((grid_h, grid_w))
        x = x + pos_embed[:, 1:]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_audio_model(
    backbone: Optional[str] = None,
    mode: str = "classification",
    num_classes: int = 8,
    pretrained: bool = True,
    input_channels: Optional[int] = None,
    pretrained_weights: Optional[str] = None,
    config: Optional[dict] = None,
) -> nn.Module:
    settings = resolve_model_settings(config=config, backbone=backbone, input_channels=input_channels)

    model_config = (config or {}).get("model", {}) if isinstance(config, dict) else {}
    resolved_pretrained_weights = pretrained_weights or model_config.get("pretrained_weights")

    if settings.backbone == "resnet18":
        model = AudioResNet18(
            mode=mode,
            num_classes=num_classes,
            input_channels=settings.input_channels,
            pretrained=settings.pretrained if pretrained is None else pretrained,
        )
    elif settings.backbone == "cnn14":
        model = CNN14Backbone(
            mode=mode,
            num_classes=num_classes,
            input_channels=settings.input_channels,
            pretrained=settings.pretrained if pretrained is None else pretrained,
            config=config,
        )
    else:
        model = SpectrogramTransformerBackbone(
            mode=mode,
            num_classes=num_classes,
            input_channels=settings.input_channels,
            variant=settings.backbone,
            spectrogram_size=settings.spectrogram_size or _BACKBONE_PRESETS[settings.backbone]["spectrogram_size"],
            pretrained=settings.pretrained if pretrained is None else pretrained,
        )

    if resolved_pretrained_weights:
        target_model = getattr(model, "backbone", None) or getattr(model, "panns", None) or model
        _load_pretrained_weights_into_model(target_model, resolved_pretrained_weights)

    return model


def _is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def _unwrap_checkpoint(state: dict) -> dict:
    if not isinstance(state, dict):
        raise ValueError("Loaded state is not a dict")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            return state[key]

    return state


def _strip_prefix_from_keys(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    new = {}
    plen = len(prefix)
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new[k[plen:]] = v
        else:
            new[k] = v
    return new


def _load_pretrained_weights_into_model(model: nn.Module, path_or_url: str) -> None:
    """Load weights from a URL or local path into `model` permissively (non-strict).

    The loader will try to extract a state_dict from common checkpoint wrappers and will
    attempt to strip common prefixes (like 'module.' from DataParallel) and only loads
    parameters whose names and tensor shapes match the current model.
    """

    def _matching_state_dict(candidate_state: dict) -> dict:
        model_state = model.state_dict()
        matched = {}
        for key, value in candidate_state.items():
            if key not in model_state:
                continue
            if hasattr(value, "shape") and model_state[key].shape != value.shape:
                continue
            matched[key] = value
        return matched

    if _is_url(path_or_url):
        raw = load_state_dict_from_url(path_or_url, map_location='cpu')
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Pretrained weights file not found: {path_or_url}")
        try:
            raw = torch.load(path_or_url, map_location='cpu', weights_only=True)
        except TypeError:
            raw = torch.load(path_or_url, map_location='cpu')
        except Exception:
            raw = torch.load(path_or_url, map_location='cpu')

    state = _unwrap_checkpoint(raw)

    candidate_states = [state]

    if any(k.startswith('module.') for k in state.keys()):
        candidate_states.append(_strip_prefix_from_keys(state, 'module.'))

    if any(k.startswith('model.') for k in state.keys()):
        candidate_states.append(_strip_prefix_from_keys(state, 'model.'))

    last_error: Exception | None = None
    for candidate in candidate_states:
        matching_state = _matching_state_dict(candidate)
        if not matching_state:
            continue
        try:
            model.load_state_dict(matching_state, strict=False)
            print("Loaded pretrained weights from {}".format(path_or_url))
            return
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise ValueError("No matching parameters were found in the pretrained checkpoint")
