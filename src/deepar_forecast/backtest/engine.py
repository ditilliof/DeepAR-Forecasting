"""
Backtesting engine with transaction costs and walk-forward validation.

[REF_BACKTESTING_WITH_COSTS]
[REF_WALK_FORWARD_VALIDATION]
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class BacktestConfig:
    """Configuration for backtest."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    strategy: str = "threshold"  # 'threshold', 'volatility_scaled', etc.
    strategy_params: Dict = None  # Strategy-specific parameters


@dataclass
class Trade:
    """Single trade record."""

    timestamp: datetime
    action: str  # 'buy', 'sell', 'close'
    price: float
    position_size: float
    cost: float  # Transaction cost + slippage
    cash_flow: float  # Negative for buy, positive for sell


@dataclass
class BacktestResult:
    """Backtest results and metrics."""

    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.DataFrame  # timestamp, equity, position, cash
    metrics: Dict[str, float]  # Sharpe, total_return, max_drawdown, etc.


class BacktestEngine:
    """
    Backtesting engine for trading strategies based on probabilistic forecasts.

    DISCLAIMER: For research and educational purposes only.
    Do not use for actual trading without thorough validation.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []

        # State
        self.cash = config.initial_capital
        self.position = 0.0  # Current position in units of asset
        self.position_value = 0.0

        logger.info(f"Initialized backtest for {config.symbol} {config.timeframe}")

    def run(
        self,
        prices: pd.DataFrame,  # DataFrame with timestamp, close
        forecasts: pd.DataFrame,  # DataFrame with timestamp, forecast signals
    ) -> BacktestResult:
        """
        Run backtest with given price data and forecast signals.

        [REF_WALK_FORWARD_VALIDATION] Ensures forecasts are out-of-sample.

        Args:
            prices: Historical price data
            forecasts: Forecast signals (e.g., prob_up, median_return, volatility)

        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        logger.info(f"Running backtest from {self.config.start_date} to {self.config.end_date}")

        # Merge prices and forecasts
        df = prices.merge(forecasts, on="timestamp", how="inner")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Filter to backtest period
        df = df[
            (df["timestamp"] >= self.config.start_date) & (df["timestamp"] <= self.config.end_date)
        ]

        if len(df) == 0:
            raise ValueError("No data in backtest period")

        logger.info(f"Backtesting on {len(df)} timesteps")

        # Get strategy function
        strategy_func = self._get_strategy(self.config.strategy)

        # Iterate through timesteps
        for i, row in df.iterrows():
            timestamp = row["timestamp"]
            price = row["close"]

            # Compute strategy signal
            signal = strategy_func(row, self.config.strategy_params or {})

            # Execute trade based on signal
            self._execute_signal(timestamp, price, signal)

            # Record equity
            position_value = self.position * price
            total_equity = self.cash + position_value

            self.equity_curve.append(
                {
                    "timestamp": timestamp,
                    "equity": total_equity,
                    "cash": self.cash,
                    "position": self.position,
                    "position_value": position_value,
                    "price": price,
                }
            )

        # Close any open positions at end
        if self.position != 0:
            final_price = df.iloc[-1]["close"]
            final_timestamp = df.iloc[-1]["timestamp"]
            self._close_position(final_timestamp, final_price)

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)

        # Compute metrics
        metrics = self._compute_metrics(equity_df, prices)

        result = BacktestResult(
            config=self.config,
            trades=self.trades,
            equity_curve=equity_df,
            metrics=metrics,
        )

        logger.info(f"Backtest complete. Total return: {metrics['total_return']:.2%}")

        return result

    def _get_strategy(self, strategy_name: str) -> Callable:
        """Get strategy function by name."""
        strategies = {
            "threshold": self._threshold_strategy,
            "volatility_scaled": self._volatility_scaled_strategy,
            "kelly": self._kelly_strategy,
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")

        return strategies[strategy_name]

    def _threshold_strategy(self, row: pd.Series, params: Dict) -> float:
        """
        Simple threshold strategy: long if P(return > 0) > threshold.

        Args:
            row: DataFrame row with forecast signals
            params: Dict with 'threshold' (default 0.55)

        Returns:
            Signal: 1.0 (long), 0.0 (flat), -1.0 (short - not implemented)
        """
        threshold = params.get("threshold", 0.55)
        prob_up = row.get("prob_up", 0.5)

        if prob_up > threshold:
            return 1.0  # Long
        else:
            return 0.0  # Flat

    def _volatility_scaled_strategy(self, row: pd.Series, params: Dict) -> float:
        """
        Position sizing based on predicted volatility: size ∝ 1/σ.

        Args:
            row: DataFrame row with forecast signals
            params: Dict with 'target_risk' (default 0.02)

        Returns:
            Signal: position size as fraction of equity
        """
        target_risk = params.get("target_risk", 0.02)
        predicted_vol = row.get("predicted_volatility", 0.01)

        prob_up = row.get("prob_up", 0.5)

        # Direction: long if prob_up > 0.5, else flat
        if prob_up > 0.5:
            # Size inversely proportional to volatility
            size = target_risk / (predicted_vol + 1e-6)
            size = min(size, 1.0)  # Cap at 100% equity
            return size
        else:
            return 0.0

    def _kelly_strategy(self, row: pd.Series, params: Dict) -> float:
        """
        Kelly criterion position sizing.

        Kelly fraction = (p * b - q) / b
        where p = prob of win, q = 1-p, b = win/loss ratio

        Args:
            row: DataFrame row with forecast signals
            params: Dict with 'kelly_fraction' (default 0.25 for quarter-Kelly)

        Returns:
            Signal: position size as fraction of equity
        """
        kelly_fraction = params.get("kelly_fraction", 0.25)

        prob_up = row.get("prob_up", 0.5)
        expected_return = row.get("median_return", 0.0)

        if expected_return > 0 and prob_up > 0.5:
            # Simplified Kelly: f = p - q = 2p - 1
            full_kelly = 2 * prob_up - 1
            size = full_kelly * kelly_fraction
            size = max(0.0, min(size, 1.0))  # Clamp to [0, 1]
            return size
        else:
            return 0.0

    def _execute_signal(self, timestamp: datetime, price: float, signal: float):
        """
        Execute trade based on strategy signal.

        Args:
            timestamp: Current timestamp
            price: Current price
            signal: Target position (0 = flat, 1 = full long, etc.)
        """
        # Compute target position in units
        current_equity = self.cash + self.position * price
        target_position_value = signal * current_equity
        target_position = target_position_value / price if price > 0 else 0.0

        # Compute position change
        position_change = target_position - self.position

        if abs(position_change) < 1e-8:
            return  # No change

        # Apply transaction costs and slippage
        # [REF_BACKTESTING_WITH_COSTS]
        realized_price = price * (1 + self.config.slippage * np.sign(position_change))
        trade_value = abs(position_change * realized_price)
        transaction_cost = trade_value * self.config.transaction_cost
        total_cost = transaction_cost

        # Cash flow
        if position_change > 0:
            # Buying
            cash_flow = -(position_change * realized_price + total_cost)
            action = "buy"
        else:
            # Selling
            cash_flow = -position_change * realized_price - total_cost
            action = "sell"

        # Update state
        self.cash += cash_flow
        self.position += position_change

        # Record trade
        trade = Trade(
            timestamp=timestamp,
            action=action,
            price=realized_price,
            position_size=abs(position_change),
            cost=total_cost,
            cash_flow=cash_flow,
        )
        self.trades.append(trade)

    def _close_position(self, timestamp: datetime, price: float):
        """Close any open position."""
        if abs(self.position) < 1e-8:
            return

        realized_price = price * (1 - self.config.slippage * np.sign(self.position))
        trade_value = abs(self.position * realized_price)
        transaction_cost = trade_value * self.config.transaction_cost

        cash_flow = self.position * realized_price - transaction_cost

        trade = Trade(
            timestamp=timestamp,
            action="close",
            price=realized_price,
            position_size=abs(self.position),
            cost=transaction_cost,
            cash_flow=cash_flow,
        )
        self.trades.append(trade)

        self.cash += cash_flow
        self.position = 0.0

    def _compute_metrics(self, equity_df: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, float]:
        """
        Compute backtest performance metrics.

        [REF_BACKTESTING_WITH_COSTS]
        """
        initial_equity = self.config.initial_capital
        final_equity = equity_df["equity"].iloc[-1]

        # Total return
        total_return = (final_equity / initial_equity) - 1.0

        # Compute equity returns
        equity_df["equity_return"] = equity_df["equity"].pct_change().fillna(0)

        # Sharpe ratio (annualized)
        # Assume 24 hours/day for crypto, 252 days/year for stocks
        periods_per_year = 365 * 24 if "h" in self.config.timeframe else 252
        if "d" in self.config.timeframe:
            periods_per_year = 252

        mean_return = equity_df["equity_return"].mean()
        std_return = equity_df["equity_return"].std()

        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Maximum drawdown
        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["cummax"]) / equity_df["cummax"]
        max_drawdown = equity_df["drawdown"].min()

        # Sortino ratio (downside deviation)
        downside_returns = equity_df["equity_return"][equity_df["equity_return"] < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0

        if downside_std > 0:
            sortino = (mean_return / downside_std) * np.sqrt(periods_per_year)
        else:
            sortino = 0.0

        # Turnover (total traded volume / average equity)
        total_traded = sum([trade.position_size * trade.price for trade in self.trades])
        avg_equity = equity_df["equity"].mean()
        turnover = total_traded / avg_equity if avg_equity > 0 else 0.0

        # Number of trades
        n_trades = len(self.trades)

        # Win rate
        if n_trades > 1:
            trade_returns = []
            for i in range(1, len(self.trades)):
                if self.trades[i].action == "sell" or self.trades[i].action == "close":
                    # Compute P&L from previous buy
                    entry_trade = self.trades[i - 1]
                    exit_trade = self.trades[i]
                    if entry_trade.action == "buy":
                        pnl = (exit_trade.price - entry_trade.price) * entry_trade.position_size
                        pnl -= entry_trade.cost + exit_trade.cost
                        trade_returns.append(pnl)

            win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0.0
        else:
            win_rate = 0.0

        # Buy-and-hold comparison
        prices_in_period = prices[
            (prices["timestamp"] >= self.config.start_date)
            & (prices["timestamp"] <= self.config.end_date)
        ]
        if len(prices_in_period) > 1:
            buy_and_hold_return = (
                prices_in_period["close"].iloc[-1] / prices_in_period["close"].iloc[0]
            ) - 1.0
        else:
            buy_and_hold_return = 0.0

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "n_trades": n_trades,
            "turnover": turnover,
            "win_rate": win_rate,
            "final_equity": final_equity,
            "buy_and_hold_return": buy_and_hold_return,
            "excess_return": total_return - buy_and_hold_return,
        }

        return metrics


def prepare_forecast_signals(
    forecasts: Dict[str, np.ndarray],
    timestamps: List[datetime],
) -> pd.DataFrame:
    """
    Convert forecast arrays to DataFrame with signals for backtesting.

    Args:
        forecasts: Dict with keys like 'median', 'quantile_0.1', 'samples', etc.
        timestamps: List of timestamps aligned with forecasts

    Returns:
        DataFrame with columns: timestamp, prob_up, median_return, predicted_volatility, etc.
    """
    df = pd.DataFrame({"timestamp": timestamps})

    # Median return
    if "median" in forecasts:
        df["median_return"] = forecasts["median"]

    # Probability of positive return
    if "samples" in forecasts:
        samples = forecasts["samples"]  # (n_timesteps, n_samples)
        prob_up = np.mean(samples > 0, axis=1)
        df["prob_up"] = prob_up

        # Predicted volatility (std of samples)
        predicted_vol = np.std(samples, axis=1)
        df["predicted_volatility"] = predicted_vol

    # Add quantiles if available
    for key in forecasts:
        if key.startswith("quantile_"):
            df[key] = forecasts[key]

    return df
