"""Training utilities for DeepAR model with Student's t likelihood."""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .deepar import DeepARStudentT, student_t_log_likelihood


class DeepARTrainer:
    """
    Trainer for DeepAR model with Student's t likelihood.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: DeepARStudentT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            model: DeepAR model instance
            device: 'cuda' or 'cpu'
            seed: Random seed for reproducibility
        """
        self.model = model.to(device)
        self.device = device
        self.seed = seed

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info(
            f"Initialized DeepARTrainer on device: {device}, "
            f"module_file={__file__}, torch={torch.__version__}"
        )

    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Optional[Dict[str, np.ndarray]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Train the model.

        [REF_DEEPAR_PAPER] Algorithm 1 - Training procedure with teacher forcing

        Args:
            train_data: Dict with 'past_target', 'past_features', 'future_target'
            val_data: Validation data (same format as train_data)
            config: Training config with keys:
                - epochs: Number of training epochs
                - batch_size: Batch size
                - learning_rate: Learning rate
                - weight_decay: L2 regularization
                - patience: Early stopping patience
                - checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dict with losses and metrics
        """
        # Extract config
        epochs = config.get("epochs", 50)
        batch_size = config.get("batch_size", 32)
        learning_rate = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 1e-5)
        patience = config.get("patience", 10)
        checkpoint_dir = config.get("checkpoint_dir", "./models")

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Prepare data loaders
        train_loader = self._create_dataloader(train_data, batch_size, shuffle=True)
        val_loader = None
        if val_data is not None:
            val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler — verbose kwarg removed (gone in torch ≥2.4)
        sched_kwargs = dict(mode="min", factor=0.5, patience=patience // 2)
        logger.info(
            f"Creating ReduceLROnPlateau: kwargs={sched_kwargs}, "
            f"torch={torch.__version__}, "
            f"trainer_file={__file__}"
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **sched_kwargs
        )

        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "epoch": [],
        }

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Starting training for {epochs} epochs")

        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer)
            history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                # Learning rate scheduling
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != old_lr:
                    logger.info(f"ReduceLROnPlateau: lr {old_lr:.2e} -> {new_lr:.2e}")

                # Early stopping and checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                    self.save_checkpoint(checkpoint_path, epoch, optimizer, val_loss)
                    logger.info(f"Saved best model (val_loss={val_loss:.6f})")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")

            history["epoch"].append(epoch + 1)

        logger.info("Training complete")

        return history

    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            past_target, past_features, future_target = batch
            past_target = past_target.to(self.device)
            past_features = past_features.to(self.device)
            future_target = future_target.to(self.device)

            # Add feature dimension if missing
            if past_target.dim() == 2:
                past_target = past_target.unsqueeze(-1)
            if future_target.dim() == 2:
                future_target = future_target.unsqueeze(-1)

            # Forward pass with teacher forcing
            mu, sigma, nu = self.model(
                past_target=past_target,
                past_features=past_features,
                future_target=future_target,
            )

            # Compute negative log-likelihood loss
            # Target: concatenate past and future
            full_target = torch.cat([past_target, future_target], dim=1)

            # Log-likelihood for each timestep
            log_likelihood = student_t_log_likelihood(full_target, mu, sigma, nu)

            # Negative log-likelihood (we want to minimize)
            loss = -log_likelihood.mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Run validation epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                past_target, past_features, future_target = batch
                past_target = past_target.to(self.device)
                past_features = past_features.to(self.device)
                future_target = future_target.to(self.device)

                if past_target.dim() == 2:
                    past_target = past_target.unsqueeze(-1)
                if future_target.dim() == 2:
                    future_target = future_target.unsqueeze(-1)

                # Forward pass
                mu, sigma, nu = self.model(
                    past_target=past_target,
                    past_features=past_features,
                    future_target=future_target,
                )

                # Compute loss
                full_target = torch.cat([past_target, future_target], dim=1)
                log_likelihood = student_t_log_likelihood(full_target, mu, sigma, nu)
                loss = -log_likelihood.mean()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def _create_dataloader(
        self,
        data: Dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays."""
        past_target = torch.FloatTensor(data["past_target"])
        past_features = torch.FloatTensor(data["past_features"])
        future_target = torch.FloatTensor(data["future_target"])

        dataset = TensorDataset(past_target, past_features, future_target)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return loader

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        val_loss: float,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint


def save_model_metadata(
    run_id: str,
    config: Dict[str, Any],
    history: Dict[str, Any],
    model_path: str,
    metadata_dir: str = "./models",
):
    """
    Save model metadata for experiment tracking.

    Args:
        run_id: Unique run identifier
        config: Training configuration
        history: Training history
        model_path: Path to saved model
        metadata_dir: Directory to save metadata JSON
    """
    os.makedirs(metadata_dir, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config,
        "history": history,
        "model_path": model_path,
    }

    metadata_path = os.path.join(metadata_dir, f"{run_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


def load_model_metadata(run_id: str, metadata_dir: str = "./models") -> Dict[str, Any]:
    """Load model metadata from JSON file."""
    metadata_path = os.path.join(metadata_dir, f"{run_id}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata
