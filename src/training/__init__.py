"""Training module for audio temperature classification"""

from .VibroNet import VibroNet
from .trainer_classification import AudioClassificationTrainer
from .trainer_regression import AudioRegressionTrainer
from src.models.audio_backbones import build_audio_model, resolve_model_settings

__all__ = [
	'VibroNet',
	'AudioClassificationTrainer',
	'AudioRegressionTrainer',
	'build_audio_model',
	'resolve_model_settings',
]
