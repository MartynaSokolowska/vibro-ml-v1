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
from .third_party.ast.ast_models import ASTModel


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
    # (mel_bins, frames). NOTE: the real AST (used below) was pretrained on 1024-frame
    # (10.24s) AudioSet clips. It will still run at other lengths -- ASTModel cuts/
    # bilinear-interpolates its positional embedding to fit -- but 224 frames (~2.24s)
    # is a big departure from the pretraining length. If your clips are short, either
    # override spectrogram_size in config, or expect the positional embedding transfer
    # to carry less benefit than it did in the paper.
    "ast": {"input_channels": 1, "spectrogram_size": (128, 224)},
    "htsat": {"input_channels": 1, "spectrogram_size": (128, 224)},
    "beats": {"input_channels": 1, "spectrogram_size": (128, 256)},
}

_TRANSFORMER_PRESETS = {
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
    config: Optional[dict] = None,
) -> dict:
    preprocessing_config = (config or {}).get("preprocessing", {}) if isinstance(config, dict) else {}
    mel_config = preprocessing_config.get("mel_spectrogram", {}) if isinstance(preprocessing_config, dict) else {}

    mel_bins = int(mel_config.get("n_mels", 64))

    _REQUIRED_MEL_BINS_FOR_CNN14 = 64
    if mel_bins != _REQUIRED_MEL_BINS_FOR_CNN14:
        raise ValueError(
            f"CNN14 requires mel_bins == {_REQUIRED_MEL_BINS_FOR_CNN14} (got {mel_bins} from "
            "config['preprocessing']['mel_spectrogram']['n_mels']). CNN14's bn0 layer is "
            "hardcoded to 64 channels, so any other value fails inside the model. Set "
            "n_mels to 64 to match CNN14's architecture and pretrained weights, or use a "
            "different backbone for other mel-bin resolutions."
        )

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
        config: Optional[dict] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        cnn14_kwargs = _build_panns_cnn14_kwargs(config=config)
        self.panns = Cnn14(**cnn14_kwargs)

        # CNN14 halves the time axis 5 times (conv_block1-5, each pool_size=(2, 2)), same
        # as it does the frequency axis. A clip that's too short in raw samples produces
        # too few spectrogram time-frames and collapses to zero mid-network, failing deep
        # inside torchlibrosa with a confusing shape error. Compute the minimum number of
        # raw samples needed and validate it explicitly in forward() instead.
        self._hop_size = cnn14_kwargs["hop_size"]
        self._min_time_frames = 32  # 2 ** 5
        self._min_input_samples = (self._min_time_frames - 1) * self._hop_size

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


class ASTBackbone(nn.Module):
    """Wrapper around AST (`ast_models.ASTModel`).

    Two independent pretraining knobs, both handled inside the vendored ASTModel:
      - imagenet_pretrain: downloads/loads a DeiT ViT checkpoint via timm (cheap, always
        recommended per the paper).
      - audioset_pretrain: downloads/loads the official AudioSet-pretrained AST weights
        (0.459 mAP checkpoint). Requires model_size='base384' and imagenet_pretrain=True.
    These are separate from this project's generic `pretrained_weights` config option,
    which -- if set -- is loaded afterwards on top via `_load_pretrained_weights_into_model`
    (e.g. to resume from your own previously fine-tuned checkpoint).
    """

    _MIN_PATCH_KERNEL = 16  # AST always uses a 16x16 patch kernel regardless of stride.

    def __init__(
        self,
        mode: str,
        num_classes: int,
        input_channels: int = 1,
        pretrained: bool = True,
        spectrogram_size: Optional[Tuple[int, int]] = None,
        config: Optional[dict] = None,
    ):
        super().__init__()

        if input_channels != 1:
            raise ValueError(
                f"ASTBackbone requires input_channels=1 (got {input_channels}). AST's patch "
                "embedding conv is single-channel by construction (it was adapted from a "
                "3-channel ViT by averaging the 3 input channels down to 1, per the paper's "
                "ImageNet-transfer scheme) -- there's no multi-channel variant to fall back to."
            )

        model_config = (config or {}).get("model", {}) if isinstance(config, dict) else {}

        mel_bins, num_frames = spectrogram_size or _BACKBONE_PRESETS["ast"]["spectrogram_size"]
        if mel_bins < self._MIN_PATCH_KERNEL or num_frames < self._MIN_PATCH_KERNEL:
            raise ValueError(
                f"AST spectrogram_size {(mel_bins, num_frames)} is too small: both dimensions "
                f"must be at least {self._MIN_PATCH_KERNEL} (AST's patch kernel is fixed at "
                f"{self._MIN_PATCH_KERNEL}x{self._MIN_PATCH_KERNEL}). Increase your clip length "
                "or the mel_bins count."
            )

        fstride = int(model_config.get("ast_fstride", 10))  # 10 => overlap of 6, matches the paper.
        tstride = int(model_config.get("ast_tstride", 10))
        audioset_pretrain = bool(model_config.get("ast_audioset_pretrain", False))
        model_size = model_config.get("ast_model_size", "base384")

        if audioset_pretrain and (model_size != "base384" or not pretrained):
            raise ValueError(
                "ast_audioset_pretrain=True requires ast_model_size='base384' and pretrained=True "
                "(imagenet_pretrain must also be on) -- these are the only weights the official "
                "AST release provides."
            )

        label_dim = 1 if mode == "regression" else num_classes

        self.model = ASTModel(
            label_dim=label_dim,
            fstride=fstride,
            tstride=tstride,
            input_fdim=mel_bins,
            input_tdim=num_frames,
            imagenet_pretrain=pretrained,
            audioset_pretrain=audioset_pretrain,
            model_size=model_size,
            verbose=False,
        )
        self.backbone_name = "ast"
        self.mel_bins = mel_bins
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This project's spectrogram convention (matches AudioResNet18 / the generic
        # transformer backbone) is channels-first: (B, 1, mel_bins, time_frames), or
        # already-squeezed (B, mel_bins, time_frames). The real ASTModel instead expects
        # (B, time_frames, mel_bins) -- it does its own unsqueeze(1)+transpose(2,3)
        # internally. Convert here rather than pushing this quirk onto the dataloader.
        if x.dim() == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    f"ASTBackbone expects a single-channel spectrogram, got {x.shape[1]} channels."
                )
            x = x.squeeze(1)
        elif x.dim() != 3:
            raise ValueError(f"ASTBackbone expects a 3D or 4D spectrogram tensor, got shape {tuple(x.shape)}.")

        # x is now (B, mel_bins, time_frames) -> AST wants (B, time_frames, mel_bins)
        x = x.transpose(1, 2)
        return self.model(x)


class SpectrogramTransformerBackbone(nn.Module):
    def __init__(
        self,
        mode: str,
        num_classes: int,
        input_channels: int,
        variant: str,
        spectrogram_size: Tuple[int, int],
    ):
        super().__init__()

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
    elif settings.backbone == "ast":
        model = ASTBackbone(
            mode=mode,
            num_classes=num_classes,
            input_channels=settings.input_channels,
            pretrained=settings.pretrained if pretrained is None else pretrained,
            spectrogram_size=settings.spectrogram_size or _BACKBONE_PRESETS["ast"]["spectrogram_size"],
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
        target_model = (
            getattr(model, "backbone", None)
            or getattr(model, "panns", None)
            or getattr(model, "model", None)
            or model
        )
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
    """Load weights from a URL or local path into `model` permissively (non-strict)."""

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
