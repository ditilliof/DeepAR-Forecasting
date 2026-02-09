"""
Backtest Mode — Streamlit UI for walk-forward backtesting with DeepAR.

Run with:
    streamlit run src/deepar_forecast/app_ui/main_backtest.py

This app lets the user:
  1. Select an asset, timeframe, and forecast horizon.
  2. Choose a cutoff date (train on data ≤ cutoff, forecast afterwards).
  3. Train a model using only pre-cutoff data.
  4. Compare forecast distribution to realised prices after the cutoff.
  5. View MAE / RMSE / directional accuracy metrics.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from loguru import logger

# ── Re-use helpers & constants from the main UI ────────────────────────────
from deepar_forecast.app_ui.main import (
    API_BASE_URL,
    CRYPTO_EXAMPLES,
    ETF_CATEGORIES,
    TIMEFRAME_INFO,
    init_session_state,
    interpret_horizon,
    normalize_crypto_symbol,
    normalize_etf_symbol,
)
from deepar_forecast.app_ui.plot_helpers import (
    create_backtest_plot as _shared_backtest_plot,
)

# ============================================================================
# BACKTEST HELPERS
# ============================================================================

def _price_from_log_returns(last_close: float, log_returns: np.ndarray) -> np.ndarray:
    """Reconstruct price path from log-returns (float32-safe)."""
    log_returns = np.asarray(log_returns, dtype=np.float32)
    return (last_close * np.exp(np.cumsum(log_returns))).astype(np.float32)


def run_backtest_pipeline(
    asset_type: str,
    symbol: str,
    timeframe: str,
    horizon: int,
    cutoff_date: str,
    lookback_days: int = 1825,
) -> Dict:
    """
    Execute the full backtest workflow:
    1. Ingest data (extended window).
    2. Train model using ONLY data ≤ cutoff_date.
    3. Forecast horizon steps from cutoff.
    4. Fetch actual realised bars after cutoff.
    5. Compare forecast vs actual.
    """
    result: Dict = {
        "success": False,
        "error": None,
        "forecast": None,
        "actual": None,
        "hist": None,
        "metrics": None,
        "training_info": None,
        "cutoff_date": cutoff_date,
    }

    try:
        # ── 1. Ingest extended window ──────────────────────────────────
        st.info("📥 Ingesting historical data (extended window)…")
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=lookback_days)

        ingest_payload = {
            "symbols": [symbol],
            "asset_type": asset_type,
            "timeframe": timeframe,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
        }
        if asset_type == "crypto":
            ingest_payload["exchange_id"] = "binance"

        resp = requests.post(f"{API_BASE_URL}/ingest", json=ingest_payload, timeout=180)
        if resp.status_code != 200:
            result["error"] = f"Ingest failed: {resp.text}"
            return result

        ingest_result = resp.json()
        bars_fetched = ingest_result.get("bars_fetched", {}).get(symbol, 0)
        bars_stored  = ingest_result.get("bars_stored",  {}).get(symbol, 0)
        if bars_fetched == 0:
            result["error"] = f"No data available for {symbol} ({timeframe})."
            return result
        if bars_stored == 0 and bars_fetched > 0:
            st.info(f"ℹ️ Data already cached ({bars_fetched} bars).")
        else:
            st.success(f"✅ Fetched {bars_fetched} bars ({bars_stored} new).")

        # ── 2. Train model with cutoff ─────────────────────────────────
        st.info(f"🧠 Training model on data ≤ **{cutoff_date}** …")

        train_payload = {
            "symbols": [symbol],
            "symbol": symbol,
            "asset_type": asset_type,
            "timeframe": timeframe,
            "context_length": min(30, horizon * 4),
            "horizon": horizon,
            "prediction_length": horizon,
            "num_layers": 2,
            "hidden_size": 40,
            "dropout_rate": 0.1,
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.001,
            "train_end_date": cutoff_date,   # ← restrict training data
        }

        resp = requests.post(f"{API_BASE_URL}/train", json=train_payload, timeout=600)
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            result["error"] = f"Training failed: {detail}"
            return result

        train_result = resp.json()
        run_id = train_result.get("run_id", "unknown")
        st.success(f"✅ Model trained (Run ID: {run_id})")
        result["training_info"] = train_result

        # ── 3. Generate forecast ───────────────────────────────────────
        st.info("🔮 Generating forecast…")
        resp = requests.get(
            f"{API_BASE_URL}/forecast",
            params={
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": horizon,
                "n_samples": 100,
                "run_id": run_id,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            result["error"] = f"Forecast failed: {resp.text}"
            return result

        fc = resp.json()
        quantiles = fc.get("quantiles", {})
        result["forecast"] = {
            "dates": [pd.Timestamp(t) for t in fc.get("timestamps", [])],
            "median": np.asarray(fc.get("median", []), dtype=np.float32),
            "lower_80": np.asarray(quantiles.get("0.1", quantiles.get("0.25", [])), dtype=np.float32),
            "upper_80": np.asarray(quantiles.get("0.9", quantiles.get("0.75", [])), dtype=np.float32),
            "lower_95": np.asarray(quantiles.get("0.1", []), dtype=np.float32),
            "upper_95": np.asarray(quantiles.get("0.9", []), dtype=np.float32),
            "samples": fc.get("samples"),
        }

        # ── 4. Fetch full history + actual post-cutoff bars ────────────
        resp = requests.get(
            f"{API_BASE_URL}/data/ohlcv",
            params={"symbol": symbol, "timeframe": timeframe},
            timeout=30,
        )
        if resp.status_code != 200:
            result["error"] = f"Could not fetch OHLCV data: {resp.text}"
            return result

        ohlcv = resp.json().get("data", [])
        if not ohlcv:
            result["error"] = "No OHLCV data returned."
            return result

        full_df = pd.DataFrame(ohlcv)
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)

        cutoff_ts = pd.Timestamp(cutoff_date)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        ts_col = full_df["timestamp"]
        if ts_col.dt.tz is None:
            ts_col = ts_col.dt.tz_localize("UTC")

        hist_mask = ts_col <= cutoff_ts
        result["hist"] = full_df[hist_mask].copy()

        # Actual future bars (up to horizon steps after cutoff)
        future_df = full_df[~hist_mask].head(horizon).copy()
        if future_df.empty:
            st.warning("⚠️ No realised data after cutoff — cannot compare forecast to actual.")
        result["actual"] = future_df

        # ── 5. Compute metrics ─────────────────────────────────────────
        if not future_df.empty and result["forecast"]["median"].size > 0:
            hist_df = result["hist"]
            last_close = float(hist_df["close"].iloc[-1])
            median_returns = result["forecast"]["median"]

            # Forecast price path
            fc_prices = _price_from_log_returns(last_close, median_returns)

            actual_prices = future_df["close"].values.astype(np.float32)
            n_compare = min(len(fc_prices), len(actual_prices))
            fc_prices = fc_prices[:n_compare]
            actual_prices = actual_prices[:n_compare]

            mae = float(np.mean(np.abs(fc_prices - actual_prices)))
            rmse = float(np.sqrt(np.mean((fc_prices - actual_prices) ** 2)))
            mape = float(np.mean(np.abs((fc_prices - actual_prices) / (actual_prices + 1e-8))) * 100)

            # Directional accuracy: did cumulative return sign match?
            fc_total = float(np.sum(median_returns[:n_compare]))
            actual_total = float(np.log(actual_prices[-1] / last_close)) if n_compare > 0 else 0.0
            direction_correct = (fc_total > 0) == (actual_total > 0)

            result["metrics"] = {
                "mae": mae,
                "rmse": rmse,
                "mape_pct": mape,
                "direction_correct": direction_correct,
                "fc_cumulative_return_pct": fc_total * 100,
                "actual_cumulative_return_pct": actual_total * 100,
                "bars_compared": n_compare,
            }

        result["success"] = True
        st.success("✅ Backtest complete!")

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out."
    except requests.exceptions.ConnectionError:
        result["error"] = "Cannot connect to API. Is the server running on port 8000?"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
        logger.exception(f"Backtest pipeline error: {e}")

    return result


def create_backtest_plot(
    hist_df: pd.DataFrame,
    forecast_data: Dict,
    actual_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cutoff_date: str,
    scale_mode: str = "robust",
) -> go.Figure:
    """Delegate to shared plotting module."""
    return _shared_backtest_plot(
        hist_df, forecast_data, actual_df, symbol, timeframe,
        cutoff_date, scale_mode=scale_mode,
    )


# ============================================================================
# MAIN BACKTEST APP
# ============================================================================

def main():
    st.set_page_config(page_title="DeepAR Backtest", page_icon="🔬", layout="wide")

    # ── Dark-dashboard CSS (same as main app) ──────────────────────────
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; font-size: .82rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e6edf3 !important; }
    details[data-testid="stExpander"] {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    }
    hr { border-color: #30363d !important; }
    h2, h3 { color: #e6edf3 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔬 DeepAR Backtest Mode")
    st.markdown(
        "Train a model using **only data before a cutoff date**, then compare its "
        "forecast to the **actual realised prices** after the cutoff."
    )
    st.divider()

    # ── Session state ──────────────────────────────────────────────────
    if "bt_results" not in st.session_state:
        st.session_state.bt_results = None

    # ==================================================================
    # STEP 1 — Asset selection (mirrors main UI)
    # ==================================================================
    st.header("Step 1: Select Asset")
    col1, col2 = st.columns([1, 2])

    with col1:
        asset_type = st.selectbox(
            "Asset Type",
            options=["crypto", "etf"],
            format_func=lambda x: "Cryptocurrency" if x == "crypto" else "ETF",
        )
    with col2:
        if asset_type == "crypto":
            crypto_input = st.selectbox(
                "Cryptocurrency",
                options=[""] + list(CRYPTO_EXAMPLES.keys()) + ["Custom…"],
            )
            if crypto_input == "Custom…":
                custom = st.text_input("Enter crypto symbol", placeholder="BTC")
                symbol = normalize_crypto_symbol(custom) if custom else None
            elif crypto_input:
                symbol = normalize_crypto_symbol(crypto_input)
            else:
                symbol = None
        else:
            etf_cat = st.selectbox("ETF Category", options=[""] + list(ETF_CATEGORIES.keys()) + ["Other…"])
            if etf_cat and etf_cat != "Other…":
                etf_ticker = st.selectbox("ETF Ticker", options=ETF_CATEGORIES[etf_cat])
                symbol = normalize_etf_symbol(etf_ticker)
            elif etf_cat == "Other…":
                custom = st.text_input("Enter ETF ticker", placeholder="SPY")
                symbol = normalize_etf_symbol(custom) if custom else None
            else:
                symbol = None

        if symbol:
            st.info(f"📊 Selected: **{symbol}**")

    st.divider()

    # ==================================================================
    # STEP 2 — Parameters
    # ==================================================================
    st.header("Step 2: Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        default_tf = "1h" if asset_type == "crypto" else "1d"
        timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(default_tf))
        st.caption(TIMEFRAME_INFO[timeframe])

    with col2:
        if timeframe == "1h":
            max_h, def_h = 168, 24
        elif timeframe == "4h":
            max_h, def_h = 42, 12
        else:
            max_h, def_h = 60, 7
        horizon = st.slider("Forecast Horizon", 1, max_h, def_h)
        st.caption(f"= {interpret_horizon(timeframe, horizon)}")

    with col3:
        cutoff_date = st.date_input(
            "Cutoff Date",
            value=datetime(2024, 12, 31),
            min_value=datetime(2015, 1, 1),
            max_value=datetime.now() - timedelta(days=1),
            help="Model will only see data up to this date.",
        )
        cutoff_iso = datetime.combine(cutoff_date, datetime.min.time()).replace(tzinfo=timezone.utc).isoformat()

    with st.expander("⚙️ Advanced"):
        lookback_days = st.slider("Lookback (days)", 90, 1825, 730, help="How far back to fetch data before cutoff.")

    st.divider()

    # ==================================================================
    # STEP 3 — Run
    # ==================================================================
    st.header("Step 3: Run Backtest")
    can_run = symbol is not None
    if not can_run:
        st.warning("⚠️ Please select an asset first.")

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("🚀 Run Backtest", disabled=not can_run, type="primary", width="stretch")

    if run_btn:
        with st.spinner("Running backtest pipeline…"):
            res = run_backtest_pipeline(
                asset_type=asset_type,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                cutoff_date=cutoff_iso,
                lookback_days=lookback_days,
            )
            if res["success"]:
                st.session_state.bt_results = {
                    "res": res,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "horizon": horizon,
                    "cutoff_date": cutoff_iso,
                }
            else:
                st.error(f"❌ {res['error']}")

    st.divider()

    # ==================================================================
    # STEP 4 — Results
    # ==================================================================
    if st.session_state.bt_results:
        bt = st.session_state.bt_results
        res = bt["res"]
        sym = bt["symbol"]
        tf  = bt["timeframe"]
        h   = bt["horizon"]
        cut = bt["cutoff_date"]

        st.header("📊 Backtest Results")

        # ── Metrics row ────────────────────────────────────────────────
        metrics = res.get("metrics")
        if metrics:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MAE (price)", f"${metrics['mae']:.2f}")
            c2.metric("RMSE (price)", f"${metrics['rmse']:.2f}")
            c3.metric("MAPE", f"{metrics['mape_pct']:.2f}%")
            c4.metric(
                "Direction",
                "✅ Correct" if metrics["direction_correct"] else "❌ Wrong",
            )
            c5.metric("Bars Compared", metrics["bars_compared"])

            st.markdown(
                f"**Forecast cum. return:** {metrics['fc_cumulative_return_pct']:.2f}%  •  "
                f"**Actual cum. return:** {metrics['actual_cumulative_return_pct']:.2f}%"
            )
        else:
            st.info("No realised data after cutoff — metrics unavailable.")

        st.divider()

        # ── Chart ──────────────────────────────────────────────────────
        st.subheader("Forecast vs Actual")

        scale_choice = st.radio(
            "Y-axis scaling",
            options=["Robust (recommended)", "Full range"],
            index=0, horizontal=True,
            help="Robust clips outliers via percentiles; Full range shows raw min/max.",
        )
        bt_scale = "robust" if "Robust" in scale_choice else "full"

        hist_df = res["hist"]
        actual_df = res["actual"]
        fc = res["forecast"]
        if hist_df is not None and not hist_df.empty and fc is not None:
            fig = create_backtest_plot(hist_df, fc, actual_df, sym, tf, cut, scale_mode=bt_scale)
            st.plotly_chart(fig, width="stretch")

            st.caption(
                "**Steel-blue** = historical  •  **Teal dashed** = forecast median  •  "
                "**Orange** = actual realised  •  **Shaded** = intervals  •  "
                "**Grey dotted** = cutoff"
            )

        st.divider()

        # ── Raw data ──────────────────────────────────────────────────
        with st.expander("🔬 Raw Forecast Data"):
            if fc is not None:
                fc_df = pd.DataFrame({
                    "Date": fc["dates"],
                    "Median (log-ret)": fc["median"],
                    "Lower 80%": fc["lower_80"],
                    "Upper 80%": fc["upper_80"],
                    "Lower 95%": fc["lower_95"],
                    "Upper 95%": fc["upper_95"],
                })
                st.dataframe(fc_df, width="stretch")
                csv = fc_df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"{sym}_{tf}_backtest.csv", "text/csv")

    # ── Footer ─────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**DeepAR Backtest** — Walk-forward evaluation. The model is trained on data "
        "strictly before the cutoff and evaluated on unseen future bars."
    )


if __name__ == "__main__":
    main()
