"""Base model interface for forecasting models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


class BaseModel(ABC):
    """
    Interface for forecasting models.
    Implement this to add new model architectures.
    """

    @abstractmethod
    def fit(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]],
        config: Dict[str, Any],
    ) -> str:
        """
        Train the model.

        Args:
            train_data: Training dataset dict with keys like 'past_target', 'future_target', etc.
            val_data: Validation dataset (optional)
            config: Training configuration (epochs, learning_rate, etc.)

        Returns:
            run_id: Unique identifier for this training run
        """
        pass

    @abstractmethod
    def predict(
        self,
        context: Dict[str, np.ndarray],
        horizon: int,
        n_samples: int = 100,
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """
        Generate probabilistic forecasts.

        Args:
            context: Context data dict with past_target, past_features, etc.
            horizon: Number of steps to forecast
            n_samples: Number of sample paths to generate

        Returns:
            Tuple of (samples, quantiles):
            - samples: (batch, n_samples, horizon)
            - quantiles: Dict mapping quantile_level -> (batch, horizon)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from file."""
        pass
