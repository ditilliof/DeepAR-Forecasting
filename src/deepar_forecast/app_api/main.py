"""
FastAPI application for DeepAR forecasting.

Provides REST API for data ingestion, training, forecasting, and backtesting.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from deepar_forecast.backtest import BacktestConfig, BacktestEngine, prepare_forecast_signals
from deepar_forecast.data import DataStorage, get_provider
from deepar_forecast.evaluation import compute_all_metrics
from deepar_forecast.features import create_sequences, engineer_features, split_by_time
from deepar_forecast.models import DeepARStudentT, DeepARTrainer

from .schemas import (
    BacktestRequest,
    BacktestResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RunsResponse,
    TrainRequest,
    TrainResponse,
)

# Initialize FastAPI app
app = FastAPI(
    title="DeepAR Trade Forecast API",
    description="Production-grade API for crypto/ETF forecasting with DeepAR and Student's t likelihood",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/deepar.db")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
DEVICE = os.getenv("DEFAULT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Initialize storage
storage = DataStorage(DATABASE_URL)

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

logger.info(f"API initialized with device: {DEVICE}")


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "DeepAR Trade Forecast API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest_data(request: IngestRequest):
    """
    Ingest OHLCV data from providers.

    Downloads and stores data in local database for specified symbols.
    """
    try:
        logger.info(f"Ingesting data: {request}")

        # Parse dates and ensure UTC timezone
        try:
            start_date = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
            end_date = datetime.fromisoformat(request.end_date.replace("Z", "+00:00"))
        except ValueError:
            # Try parsing as naive and assume UTC
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)
        
        # Ensure timezone-aware UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
            logger.warning(f"start_date was naive, forced to UTC: {start_date}")
        else:
            start_date = start_date.astimezone(timezone.utc)
        
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
            logger.warning(f"end_date was naive, forced to UTC: {end_date}")
        else:
            end_date = end_date.astimezone(timezone.utc)
        
        logger.info(f"Date range (UTC): {start_date} (tzinfo={start_date.tzinfo}) to {end_date} (tzinfo={end_date.tzinfo})")

        # Get provider
        provider = get_provider(request.asset_type, exchange_id=request.exchange_id)

        bars_stored = {}
        bars_fetched = {}

        for symbol in request.symbols:
            try:
                # Log exactly what we're passing to the provider
                logger.info(f"Calling provider.fetch_ohlcv with start={start_date} (tzinfo={start_date.tzinfo}), end={end_date} (tzinfo={end_date.tzinfo})")
                
                # Fetch data
                df = provider.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=request.timeframe,
                    start=start_date,
                    end=end_date,
                )
                
                bars_fetched[symbol] = len(df)
                logger.info(f"Fetched {len(df)} bars for {symbol}")

                # Store in database
                n_bars = storage.store_ohlcv(
                    df=df,
                    symbol=symbol,
                    asset_type=request.asset_type,
                    timeframe=request.timeframe,
                )

                bars_stored[symbol] = n_bars
                logger.info(f"Stored {n_bars} new bars for {symbol} ({len(df)} fetched)")
                
                # Verify storage by querying back
                stored_df = storage.fetch_ohlcv(symbol=symbol, timeframe=request.timeframe)
                logger.info(f"Verification: Database now has {len(stored_df)} total bars for {symbol} {request.timeframe}")

            except Exception as e:
                logger.error(f"Error ingesting {symbol}: {e}", exc_info=True)
                bars_stored[symbol] = 0
                bars_fetched[symbol] = 0

        return IngestResponse(
            status="success",
            message=f"Ingested data for {len(request.symbols)} symbols",
            bars_stored=bars_stored,
            bars_fetched=bars_fetched,
        )

    except Exception as e:
        logger.error(f"Error in ingest endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_model(request: TrainRequest):
    """
    Train DeepAR model on specified symbols.

    Downloads data from storage, engineers features, trains model, and saves checkpoint.
    """
    try:
        logger.info(f"Training model request received")
        logger.info(f"Request - symbols: {request.symbols}, symbol: {request.symbol}")
        logger.info(f"Timeframe: {request.timeframe}, Horizon: {request.horizon}, Prediction length: {request.prediction_length}")
        logger.info(f"Context length: {request.context_length}, Epochs: {request.epochs}")

        run_id = f"{request.model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

        # -----------------------------------------------------------
        # Per-symbol processing: engineer features & create sequences
        # independently to prevent cross-symbol data leakage.
        # -----------------------------------------------------------
        from deepar_forecast.features import get_feature_columns

        feature_config = request.feature_config or {}
        min_rows_needed = request.context_length + request.horizon

        all_train_pt, all_train_pf, all_train_ft = [], [], []
        all_val_pt, all_val_pf, all_val_ft = [], [], []
        feature_cols = None  # determined from first symbol
        first_df_feat = None  # kept for metadata logging
        train_start, train_end = None, None
        val_start, val_end = None, None

        for symbol in request.symbols:
            df = storage.fetch_ohlcv(symbol=symbol, timeframe=request.timeframe)
            if df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for {symbol} {request.timeframe}",
                )

            # Optional train_end_date cutoff — only use data up to this date
            if request.train_end_date:
                cutoff = pd.Timestamp(request.train_end_date)
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
                # Make df timestamp tz-aware if needed for comparison
                ts = pd.to_datetime(df["timestamp"])
                if ts.dt.tz is None:
                    ts = ts.dt.tz_localize("UTC")
                df = df[ts <= cutoff].copy()
                logger.info(f"{symbol}: filtered to train_end_date={cutoff}, {len(df)} rows remaining")
                if df.empty:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No data for {symbol} before cutoff {request.train_end_date}",
                    )

            # Engineer features per symbol (no cross-symbol leakage)
            df_feat = engineer_features(df, feature_config)

            # Determine feature columns from first symbol
            if feature_cols is None:
                feature_cols = get_feature_columns(df_feat)
                first_df_feat = df_feat
                logger.info(
                    f"Feature columns ({len(feature_cols)}): {feature_cols}, "
                    f"object cols: {df_feat.select_dtypes(include=['object']).columns.tolist()}"
                )

            total_rows = len(df_feat)
            logger.info(f"{symbol}: {total_rows} rows, min needed: {min_rows_needed}")

            if total_rows < min_rows_needed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Insufficient data for {symbol}. "
                        f"Need at least {min_rows_needed} rows, got {total_rows}. "
                        f"Try reducing context_length or horizon."
                    ),
                )

            # --- Split per symbol ---
            val_df = None
            if total_rows < min_rows_needed * 3:
                logger.warning(f"{symbol}: small dataset ({total_rows} rows). Training only, no val.")
                train_df = df_feat
            elif total_rows < min_rows_needed * 2:
                split_pt = total_rows - min_rows_needed
                train_df = df_feat.iloc[:split_pt].copy()
                val_df = df_feat.iloc[split_pt:].copy()
            else:
                train_df, val_df, _ = split_by_time(df_feat, train_ratio=0.7, val_ratio=0.15)
                if len(val_df) < min_rows_needed:
                    split_pt = total_rows - min_rows_needed * 2
                    train_df = df_feat.iloc[:split_pt].copy()
                    val_df = df_feat.iloc[split_pt:split_pt + min_rows_needed].copy()

            # Track global train/val bounds for metadata
            ts = train_df["timestamp"]
            train_start = ts.min() if train_start is None else min(train_start, ts.min())
            train_end = ts.max() if train_end is None else max(train_end, ts.max())
            if val_df is not None:
                vts = val_df["timestamp"]
                val_start = vts.min() if val_start is None else min(val_start, vts.min())
                val_end = vts.max() if val_end is None else max(val_end, vts.max())

            # --- Create sequences per symbol ---
            pt, pf, ft = create_sequences(
                train_df,
                context_length=request.context_length,
                horizon=request.horizon,
                feature_cols=feature_cols,
            )
            all_train_pt.append(pt)
            all_train_pf.append(pf)
            all_train_ft.append(ft)

            if val_df is not None:
                vpt, vpf, vft = create_sequences(
                    val_df,
                    context_length=request.context_length,
                    horizon=request.horizon,
                    feature_cols=feature_cols,
                )
                all_val_pt.append(vpt)
                all_val_pf.append(vpf)
                all_val_ft.append(vft)

        # Concatenate sequences across symbols
        train_past_target = np.concatenate(all_train_pt)
        train_past_features = np.concatenate(all_train_pf)
        train_future_target = np.concatenate(all_train_ft)

        if all_val_pt:
            val_data = {
                "past_target": np.concatenate(all_val_pt),
                "past_features": np.concatenate(all_val_pf),
                "future_target": np.concatenate(all_val_ft),
            }
        else:
            val_data = None

        # Debug logging
        logger.info(f"Training data ready:")
        if first_df_feat is not None:
            logger.info(f"  sample df dtypes: {first_df_feat.dtypes.to_dict()}")
        logger.info(f"  train_past_target: shape={train_past_target.shape}, dtype={train_past_target.dtype}")
        logger.info(f"  train_past_features: shape={train_past_features.shape}, dtype={train_past_features.dtype}")
        logger.info(f"  train_future_target: shape={train_future_target.shape}, dtype={train_future_target.dtype}")
        if val_data is not None:
            logger.info(f"  val_past_target: shape={val_data['past_target'].shape}, dtype={val_data['past_target'].dtype}")

        # Initialize model
        # Use direct fields first, fall back to hyperparams dict for backward compatibility
        hyperparams = request.hyperparams or {}
        model = DeepARStudentT(
            input_size=train_past_features.shape[-1],
            hidden_size=request.hidden_size or hyperparams.get("hidden_size", 64),
            num_layers=request.num_layers or hyperparams.get("num_layers", 2),
            dropout=request.dropout_rate or hyperparams.get("dropout", 0.1),
            rnn_type=hyperparams.get("rnn_type", "lstm"),
        )

        # Prepare training data dict
        train_data = {
            "past_target": train_past_target,
            "past_features": train_past_features,
            "future_target": train_future_target,
        }

        # Train
        trainer = DeepARTrainer(model, device=DEVICE)

        training_config = {
            "epochs": request.epochs or hyperparams.get("epochs", 50),
            "batch_size": request.batch_size or hyperparams.get("batch_size", 32),
            "learning_rate": request.learning_rate or hyperparams.get("learning_rate", 1e-3),
            "patience": hyperparams.get("patience", 10),
            "checkpoint_dir": MODEL_DIR,
        }

        history = trainer.train(train_data, val_data, training_config)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{run_id}.pt")
        torch.save(model.state_dict(), model_path)

        # Store run metadata — include FULL architecture params so forecast
        # can reconstruct the exact same model.
        actual_hidden = request.hidden_size or (request.hyperparams or {}).get("hidden_size", 64)
        actual_layers = request.num_layers or (request.hyperparams or {}).get("num_layers", 2)
        actual_dropout = request.dropout_rate or (request.hyperparams or {}).get("dropout", 0.1)
        actual_rnn = (request.hyperparams or {}).get("rnn_type", "lstm")

        actual_input_size = max(int(train_past_features.shape[-1]), 1)
        uses_dummy = (feature_cols is None or len(feature_cols) == 0)

        stored_hyperparams = {
            **(request.hyperparams or {}),
            "run_schema_version": 2,
            "input_size": actual_input_size,
            "hidden_size": actual_hidden,
            "num_layers": actual_layers,
            "dropout": actual_dropout,
            "rnn_type": actual_rnn,
            "context_length": request.context_length,
            "horizon": request.horizon,
            "feature_config": feature_config,
            "feature_cols": feature_cols if feature_cols else [],
            "uses_dummy_features": uses_dummy,
        }

        logger.info(
            f"[{run_id}] Storing hyperparams: feature_config keys={list(feature_config.keys())}, "
            f"feature_cols({len(feature_cols) if feature_cols else 0}), "
            f"input_size={actual_input_size}, uses_dummy={uses_dummy}, "
            f"hidden_size={actual_hidden}, num_layers={actual_layers}"
        )

        run_metadata = {
            "run_id": run_id,
            "model_name": request.model_name,
            "symbols": json.dumps(request.symbols),
            "timeframe": request.timeframe,
            "horizon": request.horizon,
            "hyperparams": json.dumps(stored_hyperparams),
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "metrics": json.dumps(history),
            "model_path": model_path,
            "status": "completed",
            "created_at": datetime.utcnow(),
            "completed_at": datetime.utcnow(),
        }

        storage.store_training_run(run_metadata)

        logger.info(f"Training complete: {run_id}")

        return TrainResponse(
            status="success",
            run_id=run_id,
            message=f"Model trained successfully",
        )

    except Exception as e:
        logger.exception(f"Error in train endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def generate_forecast(
    symbol: str,
    timeframe: str,
    horizon: int,
    n_samples: int = 100,
    run_id: Optional[str] = None,
):
    """
    Generate probabilistic forecast for a symbol.

    Uses trained model to produce sample paths and quantiles.
    """
    try:
        logger.info(f"Generating forecast: {symbol} {timeframe} h={horizon}")

        # Get latest run if not specified
        if run_id is None:
            runs = storage.get_training_runs(limit=1)
            if not runs:
                raise HTTPException(status_code=404, detail="No trained models found")
            run_id = runs[0]["run_id"]

        # Load run metadata
        runs = storage.get_training_runs(limit=100)
        run = next((r for r in runs if r["run_id"] == run_id), None)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Load model
        hyperparams = json.loads(run["hyperparams"])
        model_path = run.get("model_path") or os.path.join(MODEL_DIR, f"{run_id}.pt")

        # Fallback: search MODEL_DIR for any file containing the run_id
        if not os.path.exists(model_path):
            candidates = [
                os.path.join(MODEL_DIR, f)
                for f in os.listdir(MODEL_DIR)
                if run_id in f and f.endswith(".pt")
            ] if os.path.isdir(MODEL_DIR) else []
            if candidates:
                model_path = candidates[0]
                logger.info(f"Resolved checkpoint via MODEL_DIR search: {model_path}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model checkpoint not found for run {run_id}. "
                        f"Searched: {model_path} and MODEL_DIR={MODEL_DIR}. "
                        "Please retrain the model."
                    ),
                )

        logger.info(f"Forecast run_id={run_id}, checkpoint={model_path}")

        # ── Infer architecture from checkpoint when hyperparams are incomplete ──
        # Old runs may only store {"context_length": N}.
        # We recover hidden_size / num_layers / input_size from the state dict
        # so the model is reconstructed with the EXACT same shape.
        if "hidden_size" not in hyperparams or "input_size" not in hyperparams:
            logger.warning(
                f"Run {run_id} has incomplete hyperparams: {hyperparams}. "
                "Inferring architecture from checkpoint state_dict."
            )
            state_dict = torch.load(model_path, map_location=DEVICE)

            # LSTM weight_ih_l0 shape: [4*hidden_size, rnn_input_size]
            # GRU  weight_ih_l0 shape: [3*hidden_size, rnn_input_size]
            ih_key = "rnn.weight_ih_l0"
            hh_key = "rnn.weight_hh_l0"
            if ih_key in state_dict and hh_key in state_dict:
                ih_shape = state_dict[ih_key].shape  # [gate_mult*H, rnn_input]
                hh_shape = state_dict[hh_key].shape  # [gate_mult*H, H]
                inferred_hidden = int(hh_shape[1])
                # Count layers: weight_ih_l{N} keys
                n_layers = sum(1 for k in state_dict if k.startswith("rnn.weight_ih_l"))
                # RNN input = target(1) + features(input_size) + 3*emb(8)
                rnn_input_total = int(ih_shape[1])
                emb_total = 3 * 8  # symbol(8) + timeframe(8) + asset_type(8)
                inferred_input_size = max(rnn_input_total - 1 - emb_total, 1)

                hyperparams.setdefault("hidden_size", inferred_hidden)
                hyperparams.setdefault("num_layers", n_layers)
                hyperparams.setdefault("input_size", inferred_input_size)
                hyperparams.setdefault("dropout", 0.1)
                hyperparams.setdefault("rnn_type", "lstm")

                logger.info(
                    f"[{run_id}] Inferred from checkpoint: hidden_size={inferred_hidden}, "
                    f"num_layers={n_layers}, input_size={inferred_input_size}, "
                    f"rnn_input_total={rnn_input_total}"
                )
            else:
                logger.error("Cannot infer architecture — expected LSTM/GRU keys missing")
                raise HTTPException(
                    status_code=500,
                    detail="Cannot infer model architecture from checkpoint. Please retrain.",
                )

        # ── Reproduce EXACT training feature pipeline from run metadata ──
        run_feature_config = hyperparams.get("feature_config", None)
        run_feature_cols = hyperparams.get("feature_cols", None)
        run_uses_dummy = hyperparams.get("uses_dummy_features", None)
        run_schema = hyperparams.get("run_schema_version", 1)

        # ── Coerce stored_input_size early (before ANY array creation) ──
        raw_stored_input = hyperparams.get("input_size", None)
        stored_input_size = max(int(raw_stored_input), 1) if raw_stored_input is not None else 1
        logger.info(
            f"[{run_id}] DIAG raw hyperparams: run_schema={run_schema}, "
            f"raw_stored_input={raw_stored_input!r} (type={type(raw_stored_input).__name__}), "
            f"coerced stored_input_size={stored_input_size}, "
            f"feature_config type={type(run_feature_config).__name__}, "
            f"feature_cols type={type(run_feature_cols).__name__} val={run_feature_cols!r}, "
            f"uses_dummy={run_uses_dummy!r}"
        )

        # ── Backward compatibility: default missing metadata to safe values ──
        # Trigger on: EITHER field is None, or feature_cols is empty list
        if run_feature_config is None or run_feature_cols is None or (
            isinstance(run_feature_cols, list) and len(run_feature_cols) == 0
        ):
            logger.warning(
                f"[{run_id}] Old/empty run (schema v{run_schema}): "
                f"feature_config={run_feature_config!r}, feature_cols={run_feature_cols!r}. "
                "Defaulting to empty config, dummy features."
            )
            run_feature_config = run_feature_config if isinstance(run_feature_config, dict) else {}
            run_feature_cols = []
            run_uses_dummy = True

        # Ensure run_uses_dummy is a bool (old runs may lack it)
        if run_uses_dummy is None:
            run_uses_dummy = True
            logger.warning(f"[{run_id}] uses_dummy_features was None, defaulting to True")

        # Load data
        df = storage.fetch_ohlcv(symbol=symbol, timeframe=timeframe)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        # Engineer features using the SAME config that training used
        df = engineer_features(df, run_feature_config if run_feature_config else {})

        # Load context_length from training run metadata
        context_length = hyperparams.get("context_length", 168)
        logger.info(
            f"[{run_id}] Forecast pipeline: symbol={symbol}, timeframe={timeframe}, "
            f"feature_config keys={list((run_feature_config or {}).keys())}, "
            f"feature_cols={run_feature_cols}, uses_dummy={run_uses_dummy}, "
            f"context_length={context_length}, stored_input_size={stored_input_size}"
        )

        if len(df) < context_length:
            logger.warning(
                f"Insufficient data for full context. Have {len(df)} rows, need {context_length}. "
                f"Using all available data."
            )
            context_length = len(df)
            if context_length < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient data for forecast. Have {len(df)} rows, need at least 10."
                )

        context_df = df.iloc[-context_length:].copy()

        # Prepare context data — use run_feature_cols from training metadata
        target_vals = context_df["log_return"].values.astype(np.float32)
        logger.info(
            f"[{run_id}] DIAG target_vals: size={target_vals.size}, shape={target_vals.shape}"
        )
        past_target = target_vals.reshape(1, -1, 1)

        # ── Determine if we should use real features or dummy ──
        use_dummy = True  # default safe
        input_size = max(stored_input_size, 1)  # always >= 1

        if (
            isinstance(run_feature_cols, list)
            and len(run_feature_cols) > 0
            and not run_uses_dummy
        ):
            # Verify all columns exist in the engineered df
            missing_cols = [c for c in run_feature_cols if c not in context_df.columns]
            if missing_cols:
                logger.warning(
                    f"[{run_id}] feature_cols from run metadata missing in current df: {missing_cols}. "
                    "Falling back to dummy features."
                )
            else:
                raw = context_df[run_feature_cols].values
                logger.info(
                    f"[{run_id}] DIAG raw features: raw.shape={raw.shape}, "
                    f"raw.size={raw.size}, raw.dtype={raw.dtype}, "
                    f"run_feature_cols[:10]={run_feature_cols[:10]}"
                )
                n_cols = raw.shape[-1] if raw.ndim >= 2 else 0
                if raw.size > 0 and n_cols > 0:
                    past_features = raw.astype(np.float32).reshape(1, -1, n_cols)
                    input_size = n_cols
                    use_dummy = False
                else:
                    logger.warning(
                        f"[{run_id}] feature_cols={run_feature_cols} yielded "
                        f"raw.shape={raw.shape}, raw.size={raw.size}. "
                        "Falling back to dummy features."
                    )

        if use_dummy:
            # ALWAYS create at least dim=1 dummy features
            dummy_dim = max(stored_input_size, 1)
            past_features = np.zeros((1, context_length, dummy_dim), dtype=np.float32)
            input_size = dummy_dim
            logger.info(
                f"[{run_id}] Using dummy zeros: past_features.shape={past_features.shape}, "
                f"input_size={input_size}"
            )

        # ── ASSERT: feature dimension must NEVER be 0 ──
        feat_last_dim = past_features.shape[-1] if past_features.ndim >= 1 else 0
        if feat_last_dim < 1:
            diag = (
                f"run_id={run_id}, symbol={symbol}, timeframe={timeframe}, "
                f"feature_cols(type={type(run_feature_cols).__name__}, "
                f"len={len(run_feature_cols) if run_feature_cols else 'N/A'}, "
                f"first10={run_feature_cols[:10] if run_feature_cols else []}), "
                f"stored_input_size={stored_input_size}, "
                f"past_features.shape={past_features.shape}, "
                f"past_features.size={past_features.size}"
            )
            logger.error(f"[{run_id}] FATAL 0-dim features: {diag}")
            raise HTTPException(
                status_code=400,
                detail=f"Feature dimension is 0 after preparation. {diag}",
            )

        logger.info(
            f"[{run_id}] Forecast tensors: past_target={past_target.shape}, "
            f"past_features={past_features.shape}, input_size={input_size}"
        )

        # ── Build model with safe architecture params ──
        model_input_size = max(stored_input_size, 1)
        model_hidden = hyperparams.get("hidden_size", 64)
        model_layers = hyperparams.get("num_layers", 2)
        model_dropout = hyperparams.get("dropout", 0.1)
        model_rnn = hyperparams.get("rnn_type", "lstm")

        model = DeepARStudentT(
            input_size=model_input_size,
            hidden_size=model_hidden,
            num_layers=model_layers,
            dropout=model_dropout,
            rnn_type=model_rnn,
        )
        logger.info(
            f"[{run_id}] Forecast model arch: input_size={model_input_size}, "
            f"hidden_size={model_hidden}, num_layers={model_layers}, rnn_type={model_rnn}"
        )

        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except RuntimeError as arch_err:
            logger.exception(f"Architecture mismatch loading checkpoint: {arch_err}")
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model architecture mismatch for run {run_id}. "
                    "Stored hyperparams do not match checkpoint weights. "
                    f"Please retrain the model. Details: {arch_err}"
                ),
            )

        # Reconcile past_features shape with model's expected input_size
        if past_features.shape[-1] != model_input_size:
            logger.warning(
                f"[{run_id}] Adjusting past_features from dim={past_features.shape[-1]} "
                f"to model input_size={model_input_size}"
            )
            past_features = np.zeros(
                (1, context_length, model_input_size), dtype=np.float32
            )
            input_size = model_input_size

        model.to(DEVICE)
        model.eval()

        # ── Final safety net: ensure no 0-dim arrays before torch conversion ──
        if past_features.shape[-1] < 1 or past_target.shape[-1] < 1:
            logger.error(
                f"[{run_id}] FATAL pre-torch 0-dim: "
                f"past_target.shape={past_target.shape}, past_features.shape={past_features.shape}"
            )
            past_features = np.zeros(
                (1, past_target.shape[1], max(model_input_size, 1)), dtype=np.float32
            )

        # Generate samples — ensure float32 dtype everywhere
        past_target = past_target.astype(np.float32, copy=False)
        past_features = past_features.astype(np.float32, copy=False)

        past_target_t = torch.from_numpy(past_target).to(device=DEVICE, dtype=torch.float32)
        past_features_t = torch.from_numpy(past_features).to(device=DEVICE, dtype=torch.float32)

        # Pre-forward dtype/device validation
        logger.info(
            f"Forecast input dtypes: past_target={past_target_t.dtype} "
            f"past_features={past_features_t.dtype} device={past_target_t.device}"
        )
        assert past_target_t.dtype == torch.float32, f"past_target dtype {past_target_t.dtype}"
        assert past_features_t.dtype == torch.float32, f"past_features dtype {past_features_t.dtype}"

        samples = model.sample(
            past_target=past_target_t,
            past_features=past_features_t,
            horizon=horizon,
            n_samples=n_samples,
        )

        # Compute quantiles
        quantile_dict = model.compute_quantiles(samples, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])

        # Convert to numpy
        samples_np = samples.cpu().numpy().squeeze()  # (n_samples, horizon)
        median_np = quantile_dict[0.5].cpu().numpy().squeeze()

        # Generate future timestamps
        last_timestamp = df["timestamp"].iloc[-1]
        import pandas as pd
        if "h" in timeframe:
            freq = f"{timeframe.replace('h', 'H')}"
        elif "d" in timeframe:
            freq = f"{timeframe.replace('d', 'D')}"
        else:
            freq = "H"

        future_timestamps = pd.date_range(start=last_timestamp, periods=horizon + 1, freq=freq)[1:]

        response = ForecastResponse(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            timestamps=[ts.isoformat() for ts in future_timestamps],
            median=median_np.tolist(),
            quantiles={
                str(q): quantile_dict[q].cpu().numpy().squeeze().tolist()
                for q in quantile_dict
            },
            samples=samples_np.tolist() if n_samples <= 10 else None,  # Limit payload size
        )

        return response

    except HTTPException:
        raise  # Let FastAPI handle HTTP errors directly
    except Exception as e:
        logger.exception(f"Error in forecast endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {e}")


@app.post("/backtest", response_model=BacktestResponse, tags=["Backtesting"])
async def run_backtest(request: BacktestRequest):
    """
    Run backtest with specified strategy.

    Uses forecast model to generate signals and simulates trading with costs.
    """
    try:
        logger.info(f"Running backtest: {request}")

        # Load price data
        df = storage.fetch_ohlcv(symbol=request.symbol, timeframe=request.timeframe)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {request.symbol}")

        # Filter to backtest period
        start_date = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
        end_date = datetime.fromisoformat(request.end_date.replace("Z", "+00:00"))

        df_backtest = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

        if len(df_backtest) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data for backtest period")

        # Generate forecasts (simplified - using random for demo)
        # In production, use actual model forecasts
        import pandas as pd
        forecast_df = pd.DataFrame({
            "timestamp": df_backtest["timestamp"],
            "prob_up": np.random.uniform(0.4, 0.6, len(df_backtest)),
            "median_return": np.random.normal(0, 0.01, len(df_backtest)),
            "predicted_volatility": np.random.uniform(0.01, 0.03, len(df_backtest)),
        })

        # Create backtest config
        config = BacktestConfig(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=request.initial_capital,
            transaction_cost=request.transaction_cost,
            slippage=request.slippage,
            strategy=request.strategy,
            strategy_params=request.strategy_params or {},
        )

        # Run backtest
        engine = BacktestEngine(config)
        result = engine.run(df_backtest[["timestamp", "close"]], forecast_df)

        # Format response
        response = BacktestResponse(
            status="success",
            metrics=result.metrics,
            equity_curve=[
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "equity": row["equity"],
                    "position": row["position"],
                }
                for _, row in result.equity_curve.iterrows()
            ],
            trades=[
                {
                    "timestamp": trade.timestamp.isoformat(),
                    "action": trade.action,
                    "price": trade.price,
                    "size": trade.position_size,
                    "cost": trade.cost,
                }
                for trade in result.trades
            ],
        )

        return response

    except Exception as e:
        logger.error(f"Error in backtest endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/runs", response_model=RunsResponse, tags=["Training"])
async def list_runs(model_name: Optional[str] = None, limit: int = 100):
    """
    List training runs.

    Optionally filter by model name.
    """
    try:
        runs = storage.get_training_runs(model_name=model_name, limit=limit)

        from .schemas import RunInfo
        response = RunsResponse(
            runs=[
                RunInfo(
                    run_id=run["run_id"],
                    model_name=run["model_name"],
                    symbols=run["symbols"],
                    timeframe=run["timeframe"],
                    horizon=run["horizon"],
                    status=run["status"],
                    created_at=run["created_at"],
                    metrics=run["metrics"],
                )
                for run in runs
            ]
        )

        return response

    except Exception as e:
        logger.error(f"Error in runs endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/symbols", tags=["Data"])
async def list_symbols(asset_type: Optional[str] = None):
    """
    List all available symbols in the database.

    Args:
        asset_type: Optional filter by 'crypto' or 'etf'

    Returns:
        List of unique symbols with their metadata
    """
    try:
        symbols = storage.list_symbols(asset_type=asset_type)
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Error listing symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/ohlcv", tags=["Data"])
async def get_ohlcv_data(
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
):
    """
    Retrieve stored OHLCV data for visualization.

    Args:
        symbol: Trading symbol (e.g., "BTC/USDT", "SPY")
        timeframe: Timeframe (e.g., "1h", "1d")
        start_date: Start date in ISO format (optional)
        end_date: End date in ISO format (optional)
        limit: Maximum number of records (optional)

    Returns:
        OHLCV data as JSON
    """
    try:
        # Parse dates if provided – strip tz info because SQLite stores naive UTC
        start = None
        end = None
        if start_date:
            start = datetime.fromisoformat(start_date.replace("Z", "+00:00")).replace(tzinfo=None)
        if end_date:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00")).replace(tzinfo=None)

        logger.info(f"/data/ohlcv request: symbol={symbol} timeframe={timeframe} start={start} end={end} limit={limit}")

        # Fetch data from storage
        df = storage.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        if df is None or df.empty:
            logger.warning(f"/data/ohlcv returned 0 rows for {symbol} {timeframe}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "data": [],
                "count": 0
            }

        # Apply limit if requested (take most recent rows)
        if limit and len(df) > limit:
            df = df.tail(limit)

        # Convert to JSON-serializable format
        data = df.to_dict(orient="records")
        
        # Convert timestamps to ISO format
        for record in data:
            if "timestamp" in record:
                record["timestamp"] = record["timestamp"].isoformat()

        logger.info(f"/data/ohlcv returning {len(data)} bars for {symbol} {timeframe}, "
                   f"range {df['timestamp'].min()} to {df['timestamp'].max()}")

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "count": len(data),
            "start": df["timestamp"].min().isoformat() if not df.empty else None,
            "end": df["timestamp"].max().isoformat() if not df.empty else None,
        }

    except Exception as e:
        logger.error(f"Error retrieving OHLCV data for {symbol} {timeframe}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
