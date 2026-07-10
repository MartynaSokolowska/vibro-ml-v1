from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torchaudio.transforms as T


@dataclass(frozen=True)
class MelSpectrogramSettings:
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 8000.0
    top_db: int = 80


def resolve_mel_spectrogram_settings(config: Mapping[str, Any] | None = None) -> MelSpectrogramSettings:
    preprocessing_config: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        maybe_preprocessing = config.get("preprocessing", {})
        if isinstance(maybe_preprocessing, Mapping):
            preprocessing_config = maybe_preprocessing

    mel_config: Mapping[str, Any] = {}
    maybe_mel = preprocessing_config.get("mel_spectrogram", {})
    if isinstance(maybe_mel, Mapping):
        mel_config = maybe_mel

    defaults = MelSpectrogramSettings()
    return MelSpectrogramSettings(
        n_fft=int(mel_config.get("n_fft", defaults.n_fft)),
        hop_length=int(mel_config.get("hop_length", defaults.hop_length)),
        n_mels=int(mel_config.get("n_mels", defaults.n_mels)),
        f_min=float(mel_config.get("f_min", defaults.f_min)),
        f_max=float(mel_config.get("f_max", defaults.f_max)),
        top_db=int(mel_config.get("top_db", defaults.top_db)),
    )


def build_mel_spectrogram_transforms(
    sample_rate: int,
    config: Mapping[str, Any] | None = None,
) -> tuple[T.MelSpectrogram, T.AmplitudeToDB, MelSpectrogramSettings]:
    settings = resolve_mel_spectrogram_settings(config)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        n_mels=settings.n_mels,
        f_min=settings.f_min,
        f_max=settings.f_max,
    )
    db_transform = T.AmplitudeToDB(top_db=settings.top_db)
    return mel_spectrogram, db_transform, settings
