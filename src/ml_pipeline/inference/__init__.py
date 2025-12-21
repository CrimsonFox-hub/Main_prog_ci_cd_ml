"""
Пакет для инференса моделей.
"""

from .predictor import ModelPredictor
from .explainer import PredictionExplainer, get_explainer, init_explainer

__all__ = [
    "ModelPredictor",
    "PredictionExplainer",
    "get_explainer",
    "init_explainer"
]