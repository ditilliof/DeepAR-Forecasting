"""
DeepAR Trade Forecast

Production-grade forecasting system for cryptocurrencies and ETFs using
DeepAR-style autoregressive RNN with Student's t likelihood.
"""

__version__ = "0.1.0"

from . import backtest, data, evaluation, features, models

__all__ = ["data", "features", "models", "evaluation", "backtest"]
