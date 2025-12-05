"""Traffic accident severity prediction module"""
from .trainer import SeverityClassifier, compare_models
from .data_loader import load_and_prepare_data

__all__ = ['SeverityClassifier', 'compare_models', 'load_and_prepare_data']
