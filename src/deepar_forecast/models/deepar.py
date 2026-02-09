"""
DeepAR model with Student's t likelihood for heavy-tailed financial returns.

[REF_DEEPAR_PAPER] Salinas et al. (2020) - See attached PDF 1704.04110v3.pdf
[REF_STUDENT_T_LIKELIHOOD_TS] Student's t distribution for fat-tailed time series
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy import stats


class StudentTOutput(nn.Module):
    """
    Output layer that predicts parameters of Student's t distribution.

    For each timestep t, predicts:
    - μ_t (location): mean of the distribution
    - σ_t (scale): standard deviation, must be > 0
    - ν_t (degrees of freedom): controls tail heaviness, must be > 2 for finite variance

    [REF_STUDENT_T_LIKELIHOOD_TS]
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.nu_layer = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict Student's t parameters from hidden state.

        Args:
            h: Hidden state tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Tuple of (mu, sigma, nu):
            - mu: Location parameter (batch, seq_len, 1)
            - sigma: Scale parameter > 0 (batch, seq_len, 1)
            - nu: Degrees of freedom > 2 (batch, seq_len, 1)
        """
        # μ_t = affine(h)
        mu = self.mu_layer(h)

        # σ_t = softplus(affine(h)) + 1e-6  (ensure positive)
        sigma = F.softplus(self.sigma_layer(h)) + 1e-6

        # ν_t = softplus(affine(h)) + 2.0  (ensure > 2 for finite variance)
        nu = F.softplus(self.nu_layer(h)) + 2.0

        return mu, sigma, nu


def student_t_log_likelihood(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log-likelihood of y under Student's t distribution.

    Log-likelihood formula:
    log p(y | μ, σ, ν) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(πν) - log(σ)
                          - ((ν+1)/2) * log(1 + (1/ν)*((y-μ)/σ)²)

    Args:
        y: Observed values (batch, seq_len, 1)
        mu: Location parameter (batch, seq_len, 1)
        sigma: Scale parameter (batch, seq_len, 1)
        nu: Degrees of freedom (batch, seq_len, 1)

    Returns:
        Log-likelihood tensor (batch, seq_len, 1)
    """
    # Standardized residuals
    z = (y - mu) / sigma

    # Log-likelihood components
    # Part 1: log Γ((ν+1)/2) - log Γ(ν/2)
    log_gamma_term = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)

    # Part 2: -0.5*log(πν)
    log_norm = -0.5 * torch.log(torch.tensor(np.pi, device=y.device) * nu)

    # Part 3: -log(σ)
    log_scale = -torch.log(sigma)

    # Part 4: -((ν+1)/2) * log(1 + z²/ν)
    log_kernel = -((nu + 1) / 2) * torch.log1p((z ** 2) / nu)

    # Total log-likelihood
    log_prob = log_gamma_term + log_norm + log_scale + log_kernel

    return log_prob


class DeepARStudentT(nn.Module):
    """
    DeepAR model with Student's t likelihood for financial time series.

    Architecture:
    - Embeddings for symbol, timeframe, asset_type (categorical features)
    - LSTM/GRU encoder for autoregressive modeling
    - Student's t output heads (μ, σ, ν) at each timestep

    Training:
    - Teacher forcing: use observed past target values
    - Maximize log-likelihood under Student's t distribution

    Inference:
    - Ancestral sampling: sample from predicted distributions step-by-step

    [REF_DEEPAR_PAPER] Section 3 for architecture details
    """

    def __init__(
        self,
        input_size: int,  # Number of exogenous features
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = "lstm",  # 'lstm' or 'gru'
        embedding_dim: int = 8,
        num_symbols: int = 10,
        num_timeframes: int = 5,
        num_asset_types: int = 3,
    ):
        """
        Initialize DeepAR model.

        Args:
            input_size: Number of exogenous features
            hidden_size: Size of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: 'lstm' or 'gru'
            embedding_dim: Dimension of categorical embeddings
            num_symbols: Number of unique symbols for embedding
            num_timeframes: Number of unique timeframes
            num_asset_types: Number of unique asset types
        """
        super().__init__()

        # GUARD: input_size must NEVER be 0 — clamp to 1 minimum
        if input_size < 1:
            logger.warning(
                f"DeepARStudentT.__init__ received input_size={input_size}, clamping to 1"
            )
            input_size = 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        # Embeddings for categorical features
        self.symbol_embedding = nn.Embedding(num_symbols, embedding_dim)
        self.timeframe_embedding = nn.Embedding(num_timeframes, embedding_dim)
        self.asset_type_embedding = nn.Embedding(num_asset_types, embedding_dim)

        # RNN encoder
        # Input: target + exogenous features + embeddings
        rnn_input_size = 1 + input_size + 3 * embedding_dim  # target + features + 3 embeddings

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        # Student's t output heads
        self.output_layer = StudentTOutput(hidden_size)

        logger.info(
            f"Initialized DeepAR model: {rnn_type.upper()} "
            f"hidden={hidden_size}, layers={num_layers}, input={input_size}"
        )

    def forward(
        self,
        past_target: torch.Tensor,  # (batch, context_length, 1)
        past_features: torch.Tensor,  # (batch, context_length, input_size)
        future_target: Optional[torch.Tensor] = None,  # (batch, horizon, 1) - for training
        symbol_ids: Optional[torch.Tensor] = None,  # (batch,)
        timeframe_ids: Optional[torch.Tensor] = None,  # (batch,)
        asset_type_ids: Optional[torch.Tensor] = None,  # (batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        During training (future_target provided):
        - Use teacher forcing with observed future targets
        - Return parameters for all timesteps (context + horizon)

        During inference (future_target None):
        - Use autoregressive sampling for future timesteps
        - Return parameters for horizon timesteps

        Args:
            past_target: Past log-returns
            past_features: Past exogenous features
            future_target: Future log-returns (training only)
            symbol_ids: Symbol IDs for embedding
            timeframe_ids: Timeframe IDs for embedding
            asset_type_ids: Asset type IDs for embedding

        Returns:
            Tuple of (mu, sigma, nu) for predicted timesteps
        """
        batch_size = past_target.size(0)
        context_length = past_target.size(1)
        device = past_target.device

        # Get embeddings (constant across time)
        if symbol_ids is None:
            symbol_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if timeframe_ids is None:
            timeframe_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        if asset_type_ids is None:
            asset_type_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        symbol_emb = self.symbol_embedding(symbol_ids)  # (batch, emb_dim)
        timeframe_emb = self.timeframe_embedding(timeframe_ids)
        asset_type_emb = self.asset_type_embedding(asset_type_ids)

        # Repeat embeddings for each timestep
        # (batch, 1, emb_dim) -> (batch, seq_len, emb_dim)
        symbol_emb = symbol_emb.unsqueeze(1).repeat(1, context_length, 1)
        timeframe_emb = timeframe_emb.unsqueeze(1).repeat(1, context_length, 1)
        asset_type_emb = asset_type_emb.unsqueeze(1).repeat(1, context_length, 1)

        # Concatenate: target + features + embeddings
        rnn_input = torch.cat(
            [past_target, past_features, symbol_emb, timeframe_emb, asset_type_emb],
            dim=-1,
        )

        # Encode context
        rnn_output, hidden = self.rnn(rnn_input)  # rnn_output: (batch, context_length, hidden)

        if future_target is not None:
            # TRAINING MODE: Teacher forcing
            horizon = future_target.size(1)

            # Prepare embeddings for future timesteps
            symbol_emb_fut = self.symbol_embedding(symbol_ids).unsqueeze(1).repeat(1, horizon, 1)
            timeframe_emb_fut = self.timeframe_embedding(timeframe_ids).unsqueeze(1).repeat(1, horizon, 1)
            asset_type_emb_fut = self.asset_type_embedding(asset_type_ids).unsqueeze(1).repeat(1, horizon, 1)

            # For future, we need to infer features - use zeros (or last observed value)
            # In practice, exogenous features for future are unknown
            future_features = torch.zeros(
                batch_size, horizon, self.input_size, device=device
            )

            # Concatenate future inputs
            future_input = torch.cat(
                [future_target, future_features, symbol_emb_fut, timeframe_emb_fut, asset_type_emb_fut],
                dim=-1,
            )

            # Continue RNN from context hidden state
            future_output, _ = self.rnn(future_input, hidden)

            # Combine context + future outputs
            full_output = torch.cat([rnn_output, future_output], dim=1)

            # Predict parameters
            mu, sigma, nu = self.output_layer(full_output)

            return mu, sigma, nu

        else:
            # INFERENCE MODE: Autoregressive sampling
            # We only predict parameters for context (for evaluation)
            # Actual forecasting is done via sample() method

            mu, sigma, nu = self.output_layer(rnn_output)

            return mu, sigma, nu

    def sample(
        self,
        past_target: torch.Tensor,
        past_features: torch.Tensor,
        horizon: int,
        n_samples: int = 100,
        symbol_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None,
        asset_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate forecast samples via ancestral sampling.

        [REF_DEEPAR_PAPER] Section 3.3 - Sampling-based forecasting

        Args:
            past_target: Past log-returns (batch, context_length, 1)
            past_features: Past exogenous features (batch, context_length, input_size)
            horizon: Number of steps to forecast
            n_samples: Number of sample paths
            symbol_ids, timeframe_ids, asset_type_ids: Categorical IDs

        Returns:
            Samples tensor of shape (batch, n_samples, horizon, 1)
        """
        self.eval()
        with torch.no_grad():
            # ── dtype safety: ensure float32 inputs ──
            past_target = past_target.to(dtype=torch.float32)
            past_features = past_features.to(dtype=torch.float32)

            batch_size = past_target.size(0)
            context_length = past_target.size(1)
            device = past_target.device

            # ── Defensive: feature dim must be >= 1 ──
            safe_input = max(self.input_size, 1)
            if past_features.size(-1) < 1:
                logger.warning(
                    f"sample() received past_features with dim={past_features.size(-1)}, "
                    f"shape={tuple(past_features.shape)}. "
                    f"Replacing with zeros of size (batch={batch_size}, "
                    f"context={context_length}, input={safe_input})."
                )
                past_features = torch.zeros(
                    batch_size, context_length, safe_input,
                    device=device, dtype=torch.float32,
                )
            elif past_features.size(-1) != safe_input:
                logger.warning(
                    f"sample() past_features dim={past_features.size(-1)} != "
                    f"model.input_size={safe_input}. Replacing with zeros."
                )
                past_features = torch.zeros(
                    batch_size, context_length, safe_input,
                    device=device, dtype=torch.float32,
                )

            # Replicate inputs for each sample
            # (batch, context, ...) -> (batch*n_samples, context, ...)
            past_target_rep = past_target.repeat(n_samples, 1, 1)
            past_features_rep = past_features.repeat(n_samples, 1, 1)

            if symbol_ids is not None:
                symbol_ids = symbol_ids.repeat(n_samples)
            if timeframe_ids is not None:
                timeframe_ids = timeframe_ids.repeat(n_samples)
            if asset_type_ids is not None:
                asset_type_ids = asset_type_ids.repeat(n_samples)

            # Encode context
            symbol_emb = self.symbol_embedding(symbol_ids if symbol_ids is not None else torch.zeros(batch_size * n_samples, dtype=torch.long, device=device))
            timeframe_emb = self.timeframe_embedding(timeframe_ids if timeframe_ids is not None else torch.zeros(batch_size * n_samples, dtype=torch.long, device=device))
            asset_type_emb = self.asset_type_embedding(asset_type_ids if asset_type_ids is not None else torch.zeros(batch_size * n_samples, dtype=torch.long, device=device))

            symbol_emb = symbol_emb.unsqueeze(1).repeat(1, context_length, 1)
            timeframe_emb = timeframe_emb.unsqueeze(1).repeat(1, context_length, 1)
            asset_type_emb = asset_type_emb.unsqueeze(1).repeat(1, context_length, 1)

            rnn_input = torch.cat(
                [past_target_rep, past_features_rep, symbol_emb, timeframe_emb, asset_type_emb],
                dim=-1,
            )

            _, hidden = self.rnn(rnn_input)

            # Autoregressive sampling for horizon steps
            samples = []
            current_target = past_target_rep[:, -1:, :]  # Last observed target

            for t in range(horizon):
                # Prepare input for time t
                symbol_emb_t = self.symbol_embedding(symbol_ids if symbol_ids is not None else torch.zeros(batch_size * n_samples, dtype=torch.long, device=device)).unsqueeze(1)
                timeframe_emb_t = self.timeframe_embedding(timeframe_ids if timeframe_ids is not None else torch.zeros(batch_size * n_samples, dtype=torch.long, device=device)).unsqueeze(1)
                asset_type_emb_t = self.asset_type_embedding(asset_type_ids if asset_type_ids is not None else torch.zeros(batch_size * n_samples, dtype=torch.long, device=device)).unsqueeze(1)

                # Future features unknown - use zeros (explicit float32)
                features_t = torch.zeros(batch_size * n_samples, 1, self.input_size, device=device, dtype=torch.float32)

                input_t = torch.cat(
                    [current_target, features_t, symbol_emb_t, timeframe_emb_t, asset_type_emb_t],
                    dim=-1,
                )

                # RNN step
                output_t, hidden = self.rnn(input_t, hidden)

                # Predict parameters
                mu_t, sigma_t, nu_t = self.output_layer(output_t)

                # Sample from Student's t
                # Use scipy.stats.t and convert to tensor
                mu_np = mu_t.squeeze(-1).cpu().numpy()
                sigma_np = sigma_t.squeeze(-1).cpu().numpy()
                nu_np = nu_t.squeeze(-1).cpu().numpy()

                # Sample from standard t, then scale and shift
                samples_t = []
                for i in range(mu_np.shape[0]):
                    s = stats.t.rvs(df=nu_np[i, 0], loc=mu_np[i, 0], scale=sigma_np[i, 0], size=1)
                    samples_t.append(s[0])

                sample_t = torch.tensor(samples_t, dtype=torch.float32, device=device).unsqueeze(-1).unsqueeze(-1)

                samples.append(sample_t)

                # Update current_target for next step
                current_target = sample_t

            # Stack samples: (batch*n_samples, horizon, 1)
            samples = torch.cat(samples, dim=1)

            # Reshape to (batch, n_samples, horizon, 1)
            samples = samples.view(batch_size, n_samples, horizon, 1)

            return samples

    def compute_quantiles(
        self,
        samples: torch.Tensor,
        quantiles: list = [0.1, 0.5, 0.9],
    ) -> Dict[float, torch.Tensor]:
        """
        Compute quantiles from forecast samples.

        Args:
            samples: Sample tensor (batch, n_samples, horizon, 1)
            quantiles: List of quantile levels

        Returns:
            Dict mapping quantile -> tensor of shape (batch, horizon, 1)
        """
        quantile_dict = {}
        for q in quantiles:
            quantile_dict[q] = torch.quantile(samples, q, dim=1)

        return quantile_dict
