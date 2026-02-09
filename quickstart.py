"""
Quick start script for DeepAR Trade Forecast.

Demonstrates the complete workflow: ingest -> train -> forecast -> backtest
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from loguru import logger

from deepar_forecast.backtest import BacktestConfig, BacktestEngine, prepare_forecast_signals
from deepar_forecast.data import DataStorage, get_provider
from deepar_forecast.evaluation import compute_all_metrics
from deepar_forecast.features import create_sequences, engineer_features, split_by_time
from deepar_forecast.models import DeepARStudentT, DeepARTrainer

# Configuration
SYMBOL = "BTC/USDT"
ASSET_TYPE = "crypto"
TIMEFRAME = "1h"
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 1)
CONTEXT_LENGTH = 168  # 1 week
HORIZON = 24  # 1 day
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Starting quick demo with {DEVICE}")


def main():
    """Run complete demo workflow."""

    # Step 1: Ingest data
    logger.info("=" * 80)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 80)

    storage = DataStorage("sqlite:///data/demo.db")
    provider = get_provider(ASSET_TYPE)

    logger.info(f"Fetching {SYMBOL} data...")
    df = provider.fetch_ohlcv(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    logger.info(f"Fetched {len(df)} bars")

    storage.store_ohlcv(df, SYMBOL, ASSET_TYPE, TIMEFRAME)

    # Step 2: Feature engineering
    logger.info("=" * 80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 80)

    df = engineer_features(df)
    logger.info(f"Engineered features: {df.columns.tolist()}")

    # Step 3: Split data
    logger.info("=" * 80)
    logger.info("STEP 3: DATA SPLITTING")
    logger.info("=" * 80)

    train_df, val_df, test_df = split_by_time(df, train_ratio=0.7, val_ratio=0.15)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Step 4: Create sequences
    logger.info("=" * 80)
    logger.info("STEP 4: SEQUENCE CREATION")
    logger.info("=" * 80)

    from deepar_forecast.features import get_feature_columns

    feature_cols = get_feature_columns(df)

    train_past_target, train_past_features, train_future_target = create_sequences(
        train_df, CONTEXT_LENGTH, HORIZON, feature_cols=feature_cols
    )

    val_past_target, val_past_features, val_future_target = create_sequences(
        val_df, CONTEXT_LENGTH, HORIZON, feature_cols=feature_cols
    )

    logger.info(f"Created {len(train_past_target)} training sequences")

    # Step 5: Train model
    logger.info("=" * 80)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("=" * 80)

    model = DeepARStudentT(
        input_size=len(feature_cols),
        hidden_size=64,
        num_layers=2,
        rnn_type="lstm",
    )

    trainer = DeepARTrainer(model, device=DEVICE)

    train_data = {
        "past_target": train_past_target,
        "past_features": train_past_features,
        "future_target": train_future_target,
    }

    val_data = {
        "past_target": val_past_target,
        "past_features": val_past_features,
        "future_target": val_future_target,
    }

    config = {
        "epochs": 20,  # Quick demo
        "batch_size": 32,
        "learning_rate": 1e-3,
        "patience": 5,
        "checkpoint_dir": "./models",
    }

    history = trainer.train(train_data, val_data, config)
    logger.info(f"Training complete. Final val loss: {history['val_loss'][-1]:.6f}")

    # Step 6: Generate forecasts
    logger.info("=" * 80)
    logger.info("STEP 6: FORECASTING")
    logger.info("=" * 80)

    # Use last context_length points from validation
    context_df = val_df.iloc[-CONTEXT_LENGTH:]

    past_target = torch.FloatTensor(context_df["log_return"].values).unsqueeze(0).unsqueeze(-1)
    past_features = torch.FloatTensor(context_df[feature_cols].values).unsqueeze(0)

    samples = model.sample(
        past_target=past_target.to(DEVICE),
        past_features=past_features.to(DEVICE),
        horizon=HORIZON,
        n_samples=100,
    )

    quantiles = model.compute_quantiles(samples, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])

    logger.info(f"Generated {samples.shape[1]} sample paths")

    # Step 7: Evaluate
    logger.info("=" * 80)
    logger.info("STEP 7: EVALUATION")
    logger.info("=" * 80)

    # Get true future values
    future_true = val_df.iloc[-CONTEXT_LENGTH : -CONTEXT_LENGTH + HORIZON]["log_return"].values
    future_true = future_true.reshape(1, -1)

    samples_np = samples.cpu().numpy().squeeze(0).transpose(1, 0)  # (1, horizon)
    quantiles_np = {q: quantiles[q].cpu().numpy().squeeze() for q in quantiles}

    metrics = compute_all_metrics(
        y_true=future_true,
        samples=samples_np.reshape(1, 100, -1),
        quantiles={q: quantiles_np[q].reshape(1, -1) for q in quantiles_np},
    )

    logger.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.6f}")

    # Step 8: Backtest
    logger.info("=" * 80)
    logger.info("STEP 8: BACKTESTING")
    logger.info("=" * 80)

    # Prepare backtest data
    backtest_start = test_df["timestamp"].min()
    backtest_end = test_df["timestamp"].max()

    # Simple forecast signals (demo)
    forecast_signals = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"],
            "prob_up": np.random.uniform(0.4, 0.6, len(test_df)),
            "median_return": test_df["log_return"],
            "predicted_volatility": np.abs(test_df["log_return"]),
        }
    )

    config_backtest = BacktestConfig(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=backtest_start,
        end_date=backtest_end,
        initial_capital=10000.0,
        transaction_cost=0.001,
        slippage=0.0005,
        strategy="threshold",
        strategy_params={"threshold": 0.55},
    )

    engine = BacktestEngine(config_backtest)
    result = engine.run(test_df[["timestamp", "close"]], forecast_signals)

    logger.info("Backtest Results:")
    for key, value in result.metrics.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 80)
    logger.info("DEMO COMPLETE!")
    logger.info("=" * 80)

    logger.info(
        """
    Next steps:
    1. Explore the Streamlit UI: streamlit run src/deepar_forecast/app_ui/main.py
    2. Use the FastAPI: uvicorn deepar_forecast.app_api.main:app --reload
    3. Check out docs/REFERENCES.md for more information
    4. Run tests: pytest tests/ -v
    """
    )


if __name__ == "__main__":
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    main()
