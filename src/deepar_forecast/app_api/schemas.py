"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class IngestRequest(BaseModel):
    """Request schema for data ingestion."""

    symbols: List[str] = Field(..., description="List of trading symbols")
    asset_type: str = Field(..., description="Asset type: 'crypto' or 'etf'")
    timeframe: str = Field(..., description="Timeframe: '1h', '1d', etc.")
    start_date: str = Field(..., description="Start date (ISO format)")
    end_date: str = Field(..., description="End date (ISO format)")
    exchange_id: Optional[str] = Field("binance", description="Exchange ID for crypto")


class IngestResponse(BaseModel):
    """Response schema for data ingestion."""

    status: str
    message: str
    bars_stored: Dict[str, int]
    bars_fetched: Dict[str, int] = Field(default_factory=dict, description="Total bars fetched from provider")


class TrainRequest(BaseModel):
    """Request schema for model training."""

    model_name: str = Field("deepar", description="Model name")
    symbols: Optional[List[str]] = Field(None, description="Symbols to train on")
    symbol: Optional[str] = Field(None, description="Single symbol (alternative to symbols)")
    asset_type: Optional[str] = Field(None, description="Asset type: 'crypto' or 'etf'")
    timeframe: str = Field(..., description="Timeframe")
    horizon: Optional[int] = Field(None, description="Forecast horizon")
    prediction_length: Optional[int] = Field(None, description="Prediction length (alias for horizon)")
    context_length: int = Field(168, description="Context window length")
    num_layers: int = Field(2, description="Number of RNN layers")
    hidden_size: int = Field(40, description="Hidden layer size")
    dropout_rate: float = Field(0.1, description="Dropout rate")
    epochs: int = Field(30, description="Training epochs")
    batch_size: int = Field(32, description="Batch size")
    learning_rate: float = Field(0.001, description="Learning rate")
    feature_config: Optional[Dict] = Field(None, description="Feature engineering config")
    hyperparams: Optional[Dict] = Field(None, description="Model hyperparameters (deprecated, use direct fields)")
    train_end_date: Optional[str] = Field(None, description="Optional cutoff date (ISO) — only train on data <= this date")
    
    @model_validator(mode='before')
    @classmethod
    def normalize_fields(cls, data):
        """Normalize symbol/symbols and horizon/prediction_length before validation."""
        if isinstance(data, dict):
            # Handle symbol vs symbols
            if 'symbol' in data and 'symbols' not in data:
                data['symbols'] = [data['symbol']]
            elif 'symbols' in data and 'symbol' not in data and data['symbols']:
                data['symbol'] = data['symbols'][0]
            
            # Handle horizon vs prediction_length
            if 'prediction_length' in data and 'horizon' not in data:
                data['horizon'] = data['prediction_length']
            elif 'horizon' in data and 'prediction_length' not in data:
                data['prediction_length'] = data['horizon']
        
        return data


class TrainResponse(BaseModel):
    """Response schema for model training."""

    status: str
    run_id: str
    message: str


class ForecastRequest(BaseModel):
    """Request schema for generating forecasts."""

    symbol: str
    timeframe: str
    horizon: int
    n_samples: int = Field(100, description="Number of sample paths")
    run_id: Optional[str] = Field(None, description="Specific run to use (optional)")


class ForecastResponse(BaseModel):
    """Response schema for forecasts."""

    symbol: str
    timeframe: str
    horizon: int
    timestamps: List[str]
    median: List[float]
    quantiles: Dict[str, List[float]]
    samples: Optional[List[List[float]]] = None


class BacktestRequest(BaseModel):
    """Request schema for backtesting."""

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    strategy: str = Field("threshold", description="Strategy name")
    strategy_params: Optional[Dict] = Field(None, description="Strategy parameters")
    transaction_cost: float = Field(0.001, description="Transaction cost rate")
    slippage: float = Field(0.0005, description="Slippage rate")
    initial_capital: float = Field(10000.0, description="Initial capital")
    run_id: Optional[str] = Field(None, description="Model run to use")


class BacktestResponse(BaseModel):
    """Response schema for backtest results."""

    status: str
    metrics: Dict[str, float]
    equity_curve: List[Dict]
    trades: List[Dict]


class RunInfo(BaseModel):
    """Training run information."""

    run_id: str
    model_name: str
    symbols: str
    timeframe: str
    horizon: int
    status: str
    created_at: Optional[str]
    metrics: Optional[str]


class RunsResponse(BaseModel):
    """Response schema for listing runs."""

    runs: List[RunInfo]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
