"""
Feature engineering pipeline for time series forecasting.

CRITICAL: All features must use ONLY past data to prevent leakage.
[REF_LEAKAGE_PREVENTION]
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from ta.momentum import RSIIndicator
from ta.trend import MACD


def compute_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Compute log-returns: r_t = log(P_t / P_{t-1}).

    This is the PRIMARY forecast target for the DeepAR model.
    [REF_DEEPAR_PAPER] Section 3.1 - we model r_t with Student's t distribution.

    Args:
        df: DataFrame with price column
        price_col: Name of price column

    Returns:
        DataFrame with 'log_return' column added
    """
    df = df.copy()

    # CRITICAL: Use shift(1) to avoid leakage
    # r_t uses P_t and P_{t-1}, so it's available at time t
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

    # First row will be NaN (no previous price)
    df["log_return"] = df["log_return"].fillna(0.0)

    return df


def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int = 20,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """
    Compute rolling standard deviation of returns.

    LEAKAGE PREVENTION: Uses min_periods=1 and only past data.

    Args:
        df: DataFrame with returns
        window: Rolling window size
        return_col: Name of return column

    Returns:
        DataFrame with 'volatility_{window}' column added
    """
    df = df.copy()

    # rolling with closed='left' ensures we only use past data
    # However, pandas rolling doesn't support closed='left' directly
    # So we shift(1) AFTER rolling to ensure no leakage
    vol = df[return_col].rolling(window=window, min_periods=1).std()

    # CRITICAL: Shift by 1 so vol at time t uses data up to t-1
    df[f"volatility_{window}"] = vol.shift(1).fillna(0.0)

    return df


def compute_rolling_mean_return(
    df: pd.DataFrame,
    window: int = 20,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """
    Compute rolling mean of returns (momentum indicator).

    LEAKAGE PREVENTION: Shifted to use only past data.

    Args:
        df: DataFrame with returns
        window: Rolling window size
        return_col: Name of return column

    Returns:
        DataFrame with 'mean_return_{window}' column added
    """
    df = df.copy()

    mean_ret = df[return_col].rolling(window=window, min_periods=1).mean()

    # CRITICAL: Shift by 1
    df[f"mean_return_{window}"] = mean_ret.shift(1).fillna(0.0)

    return df


def compute_rsi(
    df: pd.DataFrame,
    window: int = 14,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute RSI (Relative Strength Index) using 'ta' library.

    LEAKAGE PREVENTION: RSI is computed from past prices, then shifted.

    Args:
        df: DataFrame with price column
        window: RSI window
        price_col: Name of price column

    Returns:
        DataFrame with 'rsi_{window}' column added
    """
    df = df.copy()

    rsi_indicator = RSIIndicator(close=df[price_col], window=window, fillna=True)
    rsi = rsi_indicator.rsi()

    # CRITICAL: Shift by 1 (RSI at time t should use data up to t-1)
    df[f"rsi_{window}"] = rsi.shift(1).fillna(50.0)  # Neutral RSI = 50

    return df


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).

    LEAKAGE PREVENTION: All components shifted to use past data.

    Args:
        df: DataFrame with price column
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        price_col: Name of price column

    Returns:
        DataFrame with 'macd', 'macd_signal', 'macd_diff' columns added
    """
    df = df.copy()

    macd_indicator = MACD(
        close=df[price_col],
        window_slow=slow,
        window_fast=fast,
        window_sign=signal,
        fillna=True,
    )

    macd = macd_indicator.macd()
    macd_signal = macd_indicator.macd_signal()
    macd_diff = macd_indicator.macd_diff()

    # CRITICAL: Shift all MACD components
    df["macd"] = macd.shift(1).fillna(0.0)
    df["macd_signal"] = macd_signal.shift(1).fillna(0.0)
    df["macd_diff"] = macd_diff.shift(1).fillna(0.0)

    return df


def compute_volume_zscore(
    df: pd.DataFrame,
    window: int = 20,
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Compute z-score of volume (standardized volume).

    LEAKAGE PREVENTION: Rolling stats shifted by 1.

    Args:
        df: DataFrame with volume column
        window: Rolling window size
        volume_col: Name of volume column

    Returns:
        DataFrame with 'volume_zscore_{window}' column added
    """
    df = df.copy()

    rolling_mean = df[volume_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[volume_col].rolling(window=window, min_periods=1).std()

    # Compute z-score
    zscore = (df[volume_col] - rolling_mean) / (rolling_std + 1e-8)

    # CRITICAL: Shift by 1
    df[f"volume_zscore_{window}"] = zscore.shift(1).fillna(0.0)

    return df


def engineer_features(
    df: pd.DataFrame,
    feature_config: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Computes log-returns (always) and optional exogenous features based on config.

    STRICT LEAKAGE PREVENTION: All features use only past data.
    [REF_LEAKAGE_PREVENTION]

    Args:
        df: Raw OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
        feature_config: Dict specifying which features to compute, e.g.:
            {
                'rolling_volatility': True,
                'volatility_window': 20,
                'rolling_mean': True,
                'mean_window': 20,
                'rsi': True,
                'rsi_window': 14,
                'macd': True,
                'volume_zscore': True,
                'volume_window': 20,
            }

    Returns:
        DataFrame with engineered features, sorted by timestamp
    """
    if df.empty:
        raise ValueError("Cannot engineer features on empty DataFrame")

    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    
    # CRITICAL: Force numeric types for OHLCV columns to prevent object dtype issues
    # This prevents "can't convert np.ndarray of type numpy.object_" errors during training
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing close prices (critical for returns)
    df = df.dropna(subset=["close"])
    
    # Fill remaining NaNs in volume with 0
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0.0)
    
    logger.info(f"Enforced numeric types. Shape after cleanup: {df.shape}, dtypes: {df.dtypes.to_dict()}")

    # Default config: all features enabled
    if feature_config is None:
        feature_config = {
            "rolling_volatility": True,
            "volatility_window": 20,
            "rolling_mean": True,
            "mean_window": 20,
            "rsi": True,
            "rsi_window": 14,
            "macd": True,
            "volume_zscore": True,
            "volume_window": 20,
        }

    logger.info(f"Engineering features with config: {feature_config}")

    # 1. ALWAYS compute log-returns (primary target)
    df = compute_log_returns(df, price_col="close")

    # 2. Optional exogenous features
    if feature_config.get("rolling_volatility", False):
        window = feature_config.get("volatility_window", 20)
        df = compute_rolling_volatility(df, window=window)

    if feature_config.get("rolling_mean", False):
        window = feature_config.get("mean_window", 20)
        df = compute_rolling_mean_return(df, window=window)

    if feature_config.get("rsi", False):
        window = feature_config.get("rsi_window", 14)
        df = compute_rsi(df, window=window)

    if feature_config.get("macd", False):
        df = compute_macd(df)

    if feature_config.get("volume_zscore", False):
        window = feature_config.get("volume_window", 20)
        df = compute_volume_zscore(df, window=window)

    # 3. Handle any remaining NaNs (should be minimal with fillna)
    # For first few rows, forward fill then backward fill
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")

    return df


def create_sequences(
    df: pd.DataFrame,
    context_length: int,
    horizon: int,
    target_col: str = "log_return",
    feature_cols: Optional[List[str]] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training sequences for DeepAR.

    [REF_DEEPAR_PAPER] Section 3 - uses context window to predict horizon steps.

    Args:
        df: DataFrame with features
        context_length: Number of past timesteps to use as input
        horizon: Number of future timesteps to predict
        target_col: Name of target column (log_return)
        feature_cols: List of exogenous feature columns (None = only use target)
        stride: Step size for sliding window (1 = no overlap reduction)

    Returns:
        Tuple of (past_target, past_features, future_target):
            - past_target: (N, context_length) - past log-returns
            - past_features: (N, context_length, n_features) - past exogenous features
            - future_target: (N, horizon) - future log-returns to predict

    LEAKAGE CHECK: At each time t, we use data from [t-context_length, t)
    to predict [t, t+horizon). No overlap between context and prediction.
    """
    if len(df) < context_length + horizon:
        raise ValueError(
            f"DataFrame too short. Need at least {context_length + horizon} rows, got {len(df)}"
        )

    # Ensure sorted by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Robust numeric enforcement ---
    # Target
    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).astype("float32").values

    # Features
    if feature_cols is None:
        feature_cols = []

    # Drop any non-numeric feature columns that slipped through
    if feature_cols:
        obj_cols = df[feature_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if obj_cols:
            logger.warning(f"create_sequences: dropping non-numeric feature cols: {obj_cols}")
            feature_cols = [c for c in feature_cols if c not in obj_cols]

    n_features = len(feature_cols)
    if n_features > 0:
        features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").values
    else:
        features = np.zeros((len(df), 1), dtype=np.float32)
        n_features = 1

    # Sanity check: log dtypes entering sequence creation
    logger.info(
        f"create_sequences: target_col={target_col} dtype=float32, "
        f"feature_cols({n_features})={feature_cols}, "
        f"object_cols_remaining={df.select_dtypes(include=['object']).columns.tolist()}"
    )

    # Create sequences with sliding window
    past_target_list = []
    past_features_list = []
    future_target_list = []

    for i in range(0, len(df) - context_length - horizon + 1, stride):
        # Context window: [i, i+context_length)
        past_t = target[i : i + context_length]
        past_f = features[i : i + context_length]

        # Prediction window: [i+context_length, i+context_length+horizon)
        future_t = target[i + context_length : i + context_length + horizon]

        past_target_list.append(past_t)
        past_features_list.append(past_f)
        future_target_list.append(future_t)

    # Convert lists to arrays and ensure float32 dtype
    past_target = np.array(past_target_list, dtype=np.float32)  # (N, context_length)
    past_features = np.array(past_features_list, dtype=np.float32)  # (N, context_length, n_features)
    future_target = np.array(future_target_list, dtype=np.float32)  # (N, horizon)

    logger.info(
        f"Created {len(past_target)} sequences: "
        f"context={context_length}, horizon={horizon}, stride={stride}, "
        f"dtypes: past_target={past_target.dtype}, past_features={past_features.dtype}, future_target={future_target.dtype}"
    )

    return past_target, past_features, future_target


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by time into train/val/test sets.

    STRICT TIME ORDERING: train < val < test, no shuffle.
    [REF_WALK_FORWARD_VALIDATION]

    Args:
        df: DataFrame with timestamp column
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test gets remainder)

    Returns:
        (train_df, val_df, test_df)
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Time split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
        f"(ratios: {train_ratio:.2f}/{val_ratio:.2f}/{1-train_ratio-val_ratio:.2f})"
    )

    # Verify no overlap
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()

    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, exclude_base: bool = True) -> List[str]:
    """
    Get list of feature columns (excluding base OHLCV, target, and non-numeric).

    Args:
        df: DataFrame with features
        exclude_base: If True, exclude timestamp, OHLCV, log_return, symbol

    Returns:
        List of numeric-only feature column names
    """
    base_cols = {
        "timestamp", "open", "high", "low", "close", "volume",
        "log_return", "symbol", "timeframe",
    }

    if exclude_base:
        candidates = [col for col in df.columns if col not in base_cols]
    else:
        candidates = list(df.columns)

    # CRITICAL: Only keep numeric columns to prevent string-to-float errors
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in candidates if col in numeric_cols]

    dropped = set(candidates) - set(feature_cols)
    if dropped:
        logger.warning(f"get_feature_columns: dropped non-numeric columns: {sorted(dropped)}")

    return feature_cols
