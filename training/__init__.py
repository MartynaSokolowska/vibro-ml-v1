"""Training module for audio temperature classification"""

from .VibroNetClassifier import VibroNetClassifier
from .trainer import AudioClassificationTrainer
from .VibroNetRegressor import VibroNetRegressor
from .trainerRegression import AudioRegressionTrainer

__all__ = ['VibroNet', 'AudioTrainer']