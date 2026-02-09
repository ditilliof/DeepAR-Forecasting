"""
Single-Page Streamlit App for DeepAR Forecasting
A guided wizard-like workflow for crypto and ETF forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import requests
from typing import Dict, List, Optional, Tuple
import hashlib
import json
from loguru import logger

from deepar_forecast.app_ui.plot_helpers import (
    create_forecast_plot as _shared_forecast_plot,
)

# ============================================================================
# CONFIGURATION & MAPPINGS
# ============================================================================

# Crypto examples (user-friendly name -> exchange symbol)
CRYPTO_EXAMPLES = {
    "Bitcoin": "BTC/USDT",
    "Ethereum": "ETH/USDT",
    "Chainlink": "LINK/USDT",
    "Binance Coin": "BNB/USDT",
    "Cardano": "ADA/USDT",
    "Solana": "SOL/USDT",
    "Ripple": "XRP/USDT",
    "Polkadot": "DOT/USDT",
    "Dogecoin": "DOGE/USDT",
    "Avalanche": "AVAX/USDT",
}

# ETF categories and tickers
ETF_CATEGORIES = {
    "S&P 500": ["SPY", "VOO", "IVV"],
    "Technology": ["QQQ", "VGT", "XLK"],
    "Cybersecurity": ["HACK", "CIBR", "BUG"],
    "Semiconductors": ["SMH", "SOXX", "PSI"],
    "Clean Energy": ["ICLN", "TAN", "QCLN"],
    "Global Economy": ["VT", "ACWI", "URTH"],
    "Netherlands": ["EWN"],
    "Europe": ["VGK", "IEV", "FEZ"],
    "Emerging Markets": ["EEM", "VWO", "IEMG"],
    "Japan": ["EWJ", "DXJ"],
    "China": ["FXI", "MCHI", "GXC"],
    "Healthcare": ["XLV", "VHT", "IYH"],
    "Finance": ["XLF", "VFH", "IYF"],
    "Energy": ["XLE", "VDE", "IYE"],
    "Real Estate": ["VNQ", "IYR", "XLRE"],
    "Utilities": ["XLU", "VPU", "IDU"],
}

# Timeframe explanations
TIMEFRAME_INFO = {
    "1h": "1 hour per data point - good for short-term trading (intraday)",
    "4h": "4 hours per data point - good for swing trading (multi-day)",
    "1d": "1 day per data point - good for position trading (weeks/months)",
}

# API endpoint
API_BASE_URL = "http://localhost:8000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'last_run_params' not in st.session_state:
        st.session_state.last_run_params = None
    if 'show_diagnostics' not in st.session_state:
        st.session_state.show_diagnostics = False

def normalize_crypto_symbol(user_input: str) -> str:
    """Convert user-friendly crypto name or symbol to exchange format."""
    # Check if it's a known friendly name
    if user_input in CRYPTO_EXAMPLES:
        return CRYPTO_EXAMPLES[user_input]
    
    # If user enters just "BTC", convert to "BTC/USDT"
    user_input = user_input.strip().upper()
    if "/" not in user_input:
        return f"{user_input}/USDT"
    
    return user_input

def normalize_etf_symbol(ticker: str) -> str:
    """Normalize ETF ticker."""
    return ticker.strip().upper()

def calculate_date_range(timeframe: str, lookback_days: int = 365) -> Tuple[str, str]:
    """Calculate start and end dates for data fetching with UTC timezone."""
    # CRITICAL: Use UTC-aware datetimes to prevent timezone comparison errors
    end_date = datetime.now(timezone.utc)
    
    # Adjust lookback based on timeframe to get enough data points
    if timeframe == "1h":
        # For hourly, limit to ~60 days (1440 hours) to avoid rate limits
        lookback_days = min(lookback_days, 60)
    elif timeframe == "4h":
        lookback_days = min(lookback_days, 180)
    elif timeframe == "1d":
        # Allow up to 5 years for daily data
        lookback_days = max(lookback_days, 365)
    
    start_date = end_date - timedelta(days=lookback_days)
    
    # Return ISO format with timezone (+00:00 suffix)
    return start_date.isoformat(), end_date.isoformat()

def create_cache_key(asset_type: str, symbol: str, timeframe: str, horizon: int, 
                     start_date: str, end_date: str) -> str:
    """Create a cache key for model/forecast results."""
    params = f"{asset_type}_{symbol}_{timeframe}_{horizon}_{start_date}_{end_date}"
    return hashlib.md5(params.encode()).hexdigest()

def fetch_and_train(asset_type: str, symbol: str, timeframe: str, 
                   horizon: int, force_retrain: bool = False, lookback_days: int = 180) -> Dict:
    """
    Fetch data, train model, and generate forecast.
    Returns a dictionary with results and diagnostics.
    """
    results = {
        "success": False,
        "error": None,
        "data": None,
        "forecast": None,
        "metrics": None,
        "training_info": None,
    }
    
    try:
        # Step 1: Fetch historical data
        st.info("📥 Fetching historical data...")
        start_date, end_date = calculate_date_range(timeframe, lookback_days)
        
        ingest_payload = {
            "symbols": [symbol],
            "asset_type": asset_type,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
        }
        
        if asset_type == "crypto":
            ingest_payload["exchange_id"] = "binance"
        
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json=ingest_payload,
            timeout=120
        )
        
        if response.status_code != 200:
            results["error"] = f"Data ingestion failed: {response.text}"
            return results
        
        ingest_result = response.json()
        bars_fetched = ingest_result.get('bars_fetched', {}).get(symbol, 0)
        bars_stored = ingest_result.get('bars_stored', {}).get(symbol, 0)
        
        # Check if provider actually returned data
        if bars_fetched == 0:
            results["error"] = f"No data available for {symbol} with timeframe {timeframe}. Check if the ticker is valid."
            st.error(f"❌ {results['error']}")
            return results
        elif bars_stored == 0 and bars_fetched > 0:
            # Data already exists in cache
            st.info(f"ℹ️ Data already cached ({bars_fetched} bars). Using existing data.")
        else:
            # New data stored
            st.success(f"✅ Fetched {bars_fetched} data points ({bars_stored} newly stored)")
        
        # Step 2: Train model
        st.info("🧠 Training forecasting model...")
        
        # Use reasonable defaults for training
        train_payload = {
            "symbols": [symbol],  # Backend expects list
            "symbol": symbol,  # Also send singular for compatibility
            "asset_type": asset_type,
            "timeframe": timeframe,
            "context_length": min(30, horizon * 4),  # At least 4x the horizon
            "horizon": horizon,  # Use horizon, not prediction_length
            "prediction_length": horizon,  # Also send for compatibility
            "num_layers": 2,
            "hidden_size": 40,
            "dropout_rate": 0.1,
            "epochs": 30 if force_retrain else 20,  # Faster if not forcing
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        
        response = requests.post(
            f"{API_BASE_URL}/train",
            json=train_payload,
            timeout=600
        )
        
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                if "detail" in error_json:
                    # Show user-friendly summary
                    if isinstance(error_json["detail"], list):
                        missing_fields = [item["loc"][-1] for item in error_json["detail"] if item["type"] == "missing"]
                        if missing_fields:
                            error_detail = f"Missing required fields: {', '.join(missing_fields)}"
                    else:
                        error_detail = error_json["detail"]
            except:
                pass
            results["error"] = f"Model training failed: {error_detail}"
            logger.error(f"Training failed with status {response.status_code}: {response.text}")
            return results
        
        train_result = response.json()
        st.success(f"✅ Model trained (Run ID: {train_result.get('run_id', 'unknown')})")
        
        results["training_info"] = train_result
        
        # Step 3: Generate forecast
        st.info("🔮 Generating probabilistic forecast...")
        
        # Forecast endpoint uses GET with query params
        response = requests.get(
            f"{API_BASE_URL}/forecast",
            params={
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": horizon,
                "n_samples": 100,
            },
            timeout=120
        )
        
        if response.status_code != 200:
            results["error"] = f"Forecast generation failed: {response.text}"
            return results
        
        forecast_result = response.json()
        
        # Transform API response to expected format
        # API returns: {timestamps, median, quantiles: {0.1, 0.25, 0.5, 0.75, 0.9}}
        # App expects: {forecast_dates, median, lower_80, upper_80, lower_95, upper_95}
        # 80% interval ≈ 0.1 to 0.9 (covers 80% in middle)
        # 95% interval ≈ use 0.1 to 0.9 as approximation (API doesn't return 0.025/0.975)
        quantiles = forecast_result.get("quantiles", {})
        transformed_forecast = {
            "forecast_dates": forecast_result.get("timestamps", []),
            "median": forecast_result.get("median", []),
            "lower_80": quantiles.get("0.1", quantiles.get("0.25", [])),  # 10th or 25th percentile
            "upper_80": quantiles.get("0.9", quantiles.get("0.75", [])),  # 90th or 75th percentile
            "lower_95": quantiles.get("0.1", []),  # Best approximation with available data
            "upper_95": quantiles.get("0.9", []),  # Best approximation with available data
            "samples": forecast_result.get("samples", []),
        }
        results["forecast"] = transformed_forecast
        
        # Step 4: Fetch ALL stored historical data for plotting (no date filter)
        response = requests.get(
            f"{API_BASE_URL}/data/ohlcv",
            params={
                "symbol": symbol,
                "timeframe": timeframe,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data_result = response.json()
            results["data"] = data_result.get("data", [])
            if not results["data"]:
                st.warning(f"⚠️ /data/ohlcv returned 0 rows for {symbol} {timeframe}")
        else:
            st.warning(f"⚠️ /data/ohlcv returned HTTP {response.status_code}: {response.text[:200]}")
        
        results["success"] = True
        st.success("✅ Forecast complete!")
        
    except requests.exceptions.Timeout:
        results["error"] = "Request timed out. Please try again or choose a shorter timeframe."
    except requests.exceptions.ConnectionError:
        results["error"] = "Cannot connect to API server. Make sure it's running on port 8000."
    except Exception as e:
        results["error"] = f"Unexpected error: {str(e)}"
    
    return results

def interpret_horizon(timeframe: str, steps: int) -> str:
    """Convert horizon steps to human-readable time."""
    if timeframe == "1h":
        if steps == 1:
            return "1 hour"
        elif steps < 24:
            return f"{steps} hours"
        else:
            days = steps / 24
            return f"{days:.1f} days ({steps} hours)"
    elif timeframe == "4h":
        if steps == 1:
            return "4 hours"
        elif steps < 6:
            return f"{steps * 4} hours"
        else:
            days = steps * 4 / 24
            return f"{days:.1f} days"
    elif timeframe == "1d":
        if steps == 1:
            return "1 day"
        elif steps < 7:
            return f"{steps} days"
        elif steps < 30:
            weeks = steps / 7
            return f"{weeks:.1f} weeks ({steps} days)"
        else:
            months = steps / 30
            return f"{months:.1f} months ({steps} days)"
    return f"{steps} steps"

def create_forecast_plot(historical_data: List[Dict], forecast_data: Dict,
                        symbol: str, timeframe: str,
                        scale_mode: str = "robust") -> go.Figure:
    """Delegate to shared plotting module (single source of truth)."""
    return _shared_forecast_plot(
        historical_data, forecast_data, symbol, timeframe,
        scale_mode=scale_mode,
    )

def calculate_forecast_summary(forecast_data: Dict) -> Dict:
    """Calculate summary statistics from forecast.

    Robust against missing / empty / malformed samples and quantile arrays.
    """
    median = np.array(forecast_data.get('median', []), dtype=np.float64)
    summary: Dict = {}

    if median.size == 0:
        return summary

    # --- Direction (from median, always available) ---
    if abs(median.mean()) < 0.5:  # Log-return space
        total_return = median.sum()
        summary['direction'] = 'UP' if total_return > 0 else 'DOWN'
        summary['direction_prob'] = None

        # Try to compute direction probability from samples
        raw_samples = forecast_data.get('samples')
        if raw_samples is not None and raw_samples != []:
            try:
                samples = np.asarray(raw_samples, dtype=np.float64)
                if samples.ndim == 2 and samples.shape[0] > 0 and samples.shape[1] == len(median):
                    total_returns = samples.sum(axis=1)
                    prob_up = float((total_returns > 0).mean())
                    summary['direction_prob'] = prob_up if total_return > 0 else (1 - prob_up)
            except (ValueError, TypeError):
                pass  # samples unconvertible — skip

    # --- Volatility ---
    summary['volatility'] = float(median.std())

    # --- Uncertainty (width of prediction intervals) ---
    upper_raw = forecast_data.get('upper_95', [])
    lower_raw = forecast_data.get('lower_95', [])
    try:
        upper_95 = np.array(upper_raw, dtype=np.float64)
        lower_95 = np.array(lower_raw, dtype=np.float64)
        if upper_95.size > 0 and lower_95.size > 0 and upper_95.shape == lower_95.shape:
            summary['uncertainty'] = float((upper_95 - lower_95).mean())
        else:
            summary['uncertainty'] = None
    except (ValueError, TypeError):
        summary['uncertainty'] = None

    return summary

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main single-page application."""
    
    # Initialize session state
    init_session_state()
    
    # Page config
    st.set_page_config(
        page_title="DeepAR Forecasting",
        page_icon="🔮",
        layout="wide",
    )

    # ── Dark-dashboard CSS ─────────────────────────────────────────────
    st.markdown(\"\"\"
    <style>
    /* Card-like sections */
    div[data-testid="stVerticalBlock"] > div {
        border-radius: 10px;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
    }
    /* Expander styling */
    details[data-testid="stExpander"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
    }
    /* Dividers */
    hr { border-color: #30363d !important; }
    /* Subheaders */
    h2, h3 { color: #e6edf3 !important; }
    </style>
    \"\"\", unsafe_allow_html=True)
    
    # Title and description
    st.title("🔮 DeepAR Price Forecasting")
    st.markdown("""
    **Probabilistic forecasting for cryptocurrencies and ETFs using deep learning.**
    
    Follow the guided workflow below to generate your forecast in 3 simple steps.
    """)
    
    st.divider()
    
    # ========================================================================
    # STEP 1: SELECT ASSET
    # ========================================================================
    
    st.header("Step 1: Select Your Asset")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        asset_type = st.selectbox(
            "Asset Type",
            options=["crypto", "etf"],
            format_func=lambda x: "Cryptocurrency" if x == "crypto" else "ETF (Exchange-Traded Fund)",
            help="Choose whether you want to forecast a cryptocurrency or an ETF."
        )
    
    with col2:
        if asset_type == "crypto":
            st.markdown("**Select Cryptocurrency**")
            
            # Searchable crypto input
            crypto_input = st.selectbox(
                "Choose a cryptocurrency",
                options=[""] + list(CRYPTO_EXAMPLES.keys()) + ["Custom..."],
                help="Select from popular cryptocurrencies or enter a custom symbol."
            )
            
            if crypto_input == "Custom...":
                custom_crypto = st.text_input(
                    "Enter crypto symbol",
                    placeholder="e.g., BTC, ETH, DOGE",
                    help="Enter the symbol (e.g., BTC for Bitcoin). Will be paired with USDT automatically."
                )
                if custom_crypto:
                    symbol = normalize_crypto_symbol(custom_crypto)
                else:
                    symbol = None
            elif crypto_input:
                symbol = normalize_crypto_symbol(crypto_input)
            else:
                symbol = None
            
            if symbol:
                st.info(f"📊 Selected: **{symbol}**")
                if "/USDT" in symbol:
                    st.caption("*USDT is a USD-pegged stablecoin used for trading pairs.*")
        
        else:  # ETF
            st.markdown("**Select ETF**")
            
            # Category selector
            etf_category = st.selectbox(
                "Choose ETF category",
                options=[""] + list(ETF_CATEGORIES.keys()) + ["Other..."],
                help="Select a category to see relevant ETFs."
            )
            
            if etf_category and etf_category != "Other...":
                tickers = ETF_CATEGORIES[etf_category]
                etf_ticker = st.selectbox(
                    "Select ETF ticker",
                    options=tickers,
                    help=f"Popular ETFs in {etf_category} category."
                )
                symbol = normalize_etf_symbol(etf_ticker)
            elif etf_category == "Other...":
                custom_etf = st.text_input(
                    "Enter ETF ticker",
                    placeholder="e.g., SPY, QQQ, ARKK",
                    help="Enter any valid ETF ticker symbol."
                )
                symbol = normalize_etf_symbol(custom_etf) if custom_etf else None
            else:
                symbol = None
            
            if symbol:
                st.info(f"📊 Selected: **{symbol}**")
    
    st.divider()
    
    # ========================================================================
    # STEP 2: SELECT TIMEFRAME & HORIZON
    # ========================================================================
    
    st.header("Step 2: Configure Forecast Parameters")
    
    st.markdown("""
    **Timeframe** is the frequency of data points used for training:
    - **1h** = one data point per hour (good for short-term, intraday forecasts)
    - **4h** = one data point per 4 hours (good for swing trading, multi-day forecasts)
    - **1d** = one data point per day (good for position trading, weeks/months forecasts)
    
    **Forecast Horizon** is how many steps into the future to predict, measured in the chosen timeframe.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Default timeframe based on asset type
        default_tf = "1h" if asset_type == "crypto" else "1d"
        
        timeframe = st.selectbox(
            "Data Timeframe",
            options=["1h", "4h", "1d"],
            index=["1h", "4h", "1d"].index(default_tf),
            help="Frequency of historical data points."
        )
        
        st.caption(f"ℹ️ {TIMEFRAME_INFO[timeframe]}")
    
    with col2:
        # Horizon slider with dynamic range
        if timeframe == "1h":
            max_horizon = 168  # 1 week
            default_horizon = 24  # 1 day
        elif timeframe == "4h":
            max_horizon = 42  # 1 week
            default_horizon = 12  # 2 days
        else:  # 1d
            max_horizon = 30  # 1 month
            default_horizon = 7  # 1 week
        
        horizon = st.slider(
            "Forecast Horizon (steps ahead)",
            min_value=1,
            max_value=max_horizon,
            value=default_horizon,
            help="Number of time steps to forecast into the future."
        )
        
        interpreted_time = interpret_horizon(timeframe, horizon)
        st.caption(f"📅 **{horizon} steps** = **{interpreted_time}**")
    
    # Advanced options (collapsed)
    with st.expander("⚙️ Advanced Options"):
        force_retrain = st.checkbox(
            "Force model retraining",
            value=False,
            help="If unchecked, the app will reuse previously trained models when possible to save time."
        )
        
        lookback_days = st.slider(
            "Historical data lookback (days)",
            min_value=30,
            max_value=1825,
            value=180,
            help="How many days of historical data to fetch for training. Up to 5 years (1825 days) for daily timeframe."
        )
    
    st.divider()
    
    # ========================================================================
    # STEP 3: RUN FORECAST
    # ========================================================================
    
    st.header("Step 3: Generate Forecast")
    
    # Check if all required inputs are provided
    can_run = symbol is not None and timeframe is not None and horizon is not None
    
    if not can_run:
        st.warning("⚠️ Please complete Steps 1 and 2 before running the forecast.")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_button = st.button(
            "🚀 Run Forecast",
            disabled=not can_run,
            type="primary",
            use_container_width=True,
        )
    
    with col2:
        if st.session_state.forecast_results:
            clear_button = st.button(
                "🗑️ Clear Results",
                use_container_width=True,
            )
            if clear_button:
                st.session_state.forecast_results = None
                st.session_state.last_run_params = None
                st.rerun()
    
    # Run forecast
    if run_button:
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            with st.spinner("Running forecast pipeline..."):
                results = fetch_and_train(
                    asset_type=asset_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon=horizon,
                    force_retrain=force_retrain
                )
                
                if results["success"]:
                    st.session_state.forecast_results = results
                    st.session_state.last_run_params = {
                        "asset_type": asset_type,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "horizon": horizon,
                    }
                else:
                    st.error(f"❌ Forecast failed: {results['error']}")
    
    st.divider()
    
    # ========================================================================
    # STEP 4: RESULTS
    # ========================================================================
    
    if st.session_state.forecast_results:
        results = st.session_state.forecast_results
        params = st.session_state.last_run_params
        
        st.header("📊 Forecast Results")
        
        # Summary metrics
        st.subheader("Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        forecast_summary = calculate_forecast_summary(results["forecast"])
        
        with col1:
            st.metric(
                "Asset",
                params["symbol"],
            )
        
        with col2:
            st.metric(
                "Timeframe",
                params["timeframe"],
            )
        
        with col3:
            st.metric(
                "Horizon",
                f"{params['horizon']} steps",
                delta=interpret_horizon(params["timeframe"], params["horizon"]),
            )
        
        with col4:
            if forecast_summary.get('direction'):
                direction = forecast_summary['direction']
                prob = forecast_summary.get('direction_prob')
                if prob:
                    st.metric(
                        "Direction",
                        direction,
                        delta=f"{prob*100:.1f}% confidence",
                    )
                else:
                    st.metric("Direction", direction)
        
        st.divider()
        
        # Forecast plot
        st.subheader("Price Forecast with Prediction Intervals")
        
        if results["data"] and results["forecast"]:
            # Y-axis autoscale toggle
            scale_choice = st.radio(
                "Y-axis scaling",
                options=["Robust (recommended)", "Full range"],
                index=0,
                horizontal=True,
                help="Robust clips outliers via percentiles; Full range shows raw min/max.",
            )
            scale_mode = "robust" if "Robust" in scale_choice else "full"

            fig = create_forecast_plot(
                historical_data=results["data"],
                forecast_data=results["forecast"],
                symbol=params["symbol"],
                timeframe=params["timeframe"],
                scale_mode=scale_mode,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            **How to read this chart:**
            - **Steel-blue line**: Historical prices
            - **Teal dashed line**: Median forecast (most likely outcome)
            - **Darker shaded area**: 80 % prediction interval
            - **Lighter shaded area**: 95 % prediction interval
            """)
        elif not results["data"]:
            st.error("⚠️ No historical OHLCV data returned from /data/ohlcv. "
                     "Make sure the API server is running and data has been ingested for this symbol/timeframe.")
        elif not results["forecast"]:
            st.warning("Forecast data is missing — the chart cannot be rendered.")
        
        st.divider()
        
        # What this means
        st.subheader("💡 What This Means")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Forecast Statistics**")
            
            median_forecast = results["forecast"]["median"]
            
            # Calculate cumulative return
            if abs(np.mean(median_forecast)) < 0.5:  # Log-returns
                cumulative_return = np.sum(median_forecast)
                st.metric(
                    "Expected Return",
                    f"{cumulative_return*100:.2f}%",
                    delta="Cumulative over forecast horizon",
                )
            
            # Volatility
            st.metric(
                "Forecast Volatility",
                f"{forecast_summary['volatility']*100:.2f}%",
                help="Standard deviation of forecast returns. Higher = more uncertain.",
            )
            
            # Uncertainty
            unc = forecast_summary.get('uncertainty')
            if unc is not None:
                st.metric(
                    "Uncertainty Range",
                    f"{unc*100:.2f}%",
                    help="Width of 95% prediction interval. Wider = less confident forecast.",
                )
            else:
                st.metric("Uncertainty Range", "N/A", help="Prediction intervals not available.")
        
        with col2:
            st.markdown("**Model Information**")
            
            if results.get("training_info"):
                train_info = results["training_info"]
                
                st.metric(
                    "Training Loss (NLL)",
                    f"{train_info.get('final_train_nll', 0):.4f}",
                    help="Negative log-likelihood. Lower is better.",
                )
                
                st.metric(
                    "Validation Loss (NLL)",
                    f"{train_info.get('final_val_nll', 0):.4f}",
                    help="Model performance on held-out data.",
                )
                
                st.caption(f"Run ID: `{train_info.get('run_id', 'unknown')}`")
        
        # Advanced diagnostics (collapsible)
        with st.expander("🔬 Advanced Diagnostics", expanded=st.session_state.show_diagnostics):
            st.markdown("**Model Training Details**")
            
            if results.get("training_info"):
                train_info = results["training_info"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Context Length", train_info.get("context_length", "N/A"))
                    st.metric("Hidden Size", train_info.get("hidden_size", "N/A"))
                    st.metric("Num Layers", train_info.get("num_layers", "N/A"))
                
                with col2:
                    st.metric("Epochs Trained", train_info.get("epochs", "N/A"))
                    st.metric("Batch Size", train_info.get("batch_size", "N/A"))
                    st.metric("Learning Rate", train_info.get("learning_rate", "N/A"))
                
                with col3:
                    st.metric("Dropout Rate", train_info.get("dropout_rate", "N/A"))
                    if train_info.get("training_time"):
                        st.metric("Training Time", f"{train_info['training_time']:.1f}s")
            
            st.markdown("**Forecast Distribution**")
            
            # Plot distribution of forecast samples if available
            if results["forecast"].get("samples"):
                samples = np.array(results["forecast"]["samples"])
                
                # Plot final step distribution
                final_step_samples = samples[:, -1]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=final_step_samples,
                    nbinsx=50,
                    name="Sample Distribution",
                ))
                
                fig.update_layout(
                    title=f"Distribution of Final Step Forecast",
                    xaxis_title="Return",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=300,
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Raw Forecast Data**")
            
            # Show forecast dataframe
            forecast_df = pd.DataFrame({
                "Date": pd.to_datetime(results["forecast"]["forecast_dates"]),
                "Median": results["forecast"]["median"],
                "Lower 80%": results["forecast"]["lower_80"],
                "Upper 80%": results["forecast"]["upper_80"],
                "Lower 95%": results["forecast"]["lower_95"],
                "Upper 95%": results["forecast"]["upper_95"],
            })
            
            st.dataframe(forecast_df, use_container_width=True)
            
            # Download button
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv,
                file_name=f"{params['symbol']}_{params['timeframe']}_forecast.csv",
                mime="text/csv",
            )
    
    # Footer
    st.divider()
    st.caption("""
    **DeepAR Forecasting** - Probabilistic time series forecasting using deep autoregressive models with Student's t likelihood.
    The model learns to predict distributions (not just point estimates) to quantify forecast uncertainty.
    """)

if __name__ == "__main__":
    main()
