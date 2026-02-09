# DeepAR Trade Forecast

Production-grade forecasting system for cryptocurrencies and ETFs using DeepAR-style autoregressive RNN with Student's t likelihood for heavy-tailed financial returns.

## Architecture

```
src/deepar_forecast/
├── data/           # Data providers (CCXT, yfinance) + SQLite storage
├── features/       # Feature engineering (log-returns, indicators, no leakage)
├── models/         # DeepAR with Student's t likelihood, training loop
├── evaluation/     # Metrics (NLL, CRPS, coverage, calibration)
├── backtest/       # Walk-forward backtesting with transaction costs
├── app_api/        # FastAPI REST endpoints (/ingest, /train, /forecast, /backtest)
└── app_ui/         # Streamlit dashboards (main forecast + backtest mode)
```

## Quick Start

### 1. Install

```bash
poetry install
cp .env.example .env          # optional, defaults work out of the box
```

### 2. Start the API server

```bash
poetry run uvicorn deepar_forecast.app_api.main:app --host 0.0.0.0 --port 8000
```

API docs: <http://localhost:8000/docs>

### 3. Start the main Streamlit UI

```bash
poetry run streamlit run src/deepar_forecast/app_ui/main.py --server.port 8501
```

Open <http://localhost:8501> — follow the 3-step wizard (select asset → configure → run).

### 4. Start the backtest UI (optional)

```bash
poetry run streamlit run src/deepar_forecast/app_ui/main_backtest.py --server.port 8502
```

Open <http://localhost:8502> — train with a cutoff date and compare forecast vs actual.
