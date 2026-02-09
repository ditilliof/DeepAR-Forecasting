"""Data provider interfaces and implementations for crypto and ETF data ingestion."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import ccxt
import pandas as pd
import yfinance as yf
from loguru import logger


class IDataProvider(ABC):
    """Interface for data providers. Implement this to add new data sources."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'SPY')
            timeframe: Candle interval (e.g., '1h', '1d')
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            timestamp is timezone-aware UTC
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid for this provider."""
        pass

    @abstractmethod
    def list_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass


class CCXTProvider(IDataProvider):
    """
    Cryptocurrency data provider using CCXT.
    Supports 100+ exchanges. Default: Binance for liquidity.
    """

    def __init__(self, exchange_id: str = "binance", rate_limit: bool = True):
        """
        Initialize CCXT provider.

        Args:
            exchange_id: CCXT exchange ID (e.g., 'binance', 'coinbase', 'kraken')
            rate_limit: Enable rate limiting to avoid API bans
        """
        self.exchange_id = exchange_id
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({"enableRateLimit": rate_limit})
            logger.info(f"Initialized CCXT provider: {exchange_id}")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_id}' not supported by CCXT")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch crypto OHLCV data.

        [REF_DEEPAR_PAPER] Raw price data is converted to log-returns in feature pipeline.
        """
        if not self.validate_symbol(symbol):
            raise ValueError(f"Symbol '{symbol}' not available on {self.exchange_id}")

        # CRITICAL: Defensively ensure start/end are UTC-aware to prevent timezone comparison errors
        from datetime import timezone as tz
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz.utc)
            logger.warning(f"start datetime was naive, forcing UTC: {start}")
        else:
            start = start.astimezone(tz.utc)
        
        if end.tzinfo is None:
            end = end.replace(tzinfo=tz.utc)
            logger.warning(f"end datetime was naive, forcing UTC: {end}")
        else:
            end = end.astimezone(tz.utc)
        
        logger.info(f"Normalized start={start} (tzinfo={start.tzinfo}), end={end} (tzinfo={end.tzinfo})")

        # Convert datetime to milliseconds
        since = int(start.timestamp() * 1000)
        until = int(end.timestamp() * 1000)

        all_candles = []
        current_since = since

        logger.info(f"Fetching {symbol} {timeframe} from {self.exchange_id}: {start} to {end}")

        while current_since < until:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000,  # Max per request for most exchanges
                )

                if not candles:
                    break

                all_candles.extend(candles)
                current_since = candles[-1][0] + 1  # Move to next millisecond

                # Stop if we've reached the end
                if candles[-1][0] >= until:
                    break

            except ccxt.NetworkError as e:
                logger.warning(f"Network error fetching {symbol}: {e}. Retrying...")
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching {symbol}: {e}")
                raise

        if not all_candles:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp to datetime (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Filter to exact date range
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(df)} candles for {symbol}")

        return df

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists on exchange."""
        try:
            markets = self.exchange.load_markets()
            return symbol in markets
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False

    def list_symbols(self) -> List[str]:
        """Get all available trading pairs."""
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Error listing symbols: {e}")
            return []


class YFinanceProvider(IDataProvider):
    """
    Stock/ETF data provider using yfinance (Yahoo Finance).
    Free, no API key required, good for US equities and major ETFs.
    """

    def __init__(self):
        """Initialize yfinance provider."""
        logger.info("Initialized yfinance provider")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch stock/ETF OHLCV data.

        Note: yfinance uses different timeframe notation:
        - '1h' -> '1h'
        - '1d' -> '1d'
        - '1m' -> '1m' (only last 7 days available)
        """
        logger.info(f"Fetching {symbol} {timeframe} from yfinance: {start} to {end}")

        try:
            ticker = yf.Ticker(symbol)

            # Map timeframe to yfinance interval
            interval = self._map_timeframe(timeframe)

            # Fetch data
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,  # Adjust for splits/dividends
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol} with timeframe {timeframe} from {start} to {end}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            logger.info(f"Raw yfinance columns: {df.columns.tolist()}")
            logger.info(f"Raw yfinance shape: {df.shape}")
            logger.info(f"Raw yfinance index type: {type(df.index)}, name: {df.index.name}")

            # Reset index first to get Date/Datetime as column
            df = df.reset_index()
            logger.info(f"After reset_index, columns: {df.columns.tolist()}")
            
            # Rename columns to match our schema
            # CRITICAL: yfinance uses 'Date' for daily, 'Datetime' for intraday (1h, etc.)
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ["open", "high", "low", "close", "volume"]:
                    column_mapping[col] = col_lower
                elif col_lower in ["date", "datetime"]:  # Handle both Date and Datetime (case-insensitive)
                    column_mapping[col] = "timestamp"
                    logger.info(f"Mapped time column '{col}' -> 'timestamp'")
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
                logger.info(f"After column mapping: {df.columns.tolist()}")
            
            # Verify we have all required columns
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
                raise ValueError(f"Missing columns: {missing_cols}")

            # Ensure timezone-aware UTC
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            # Select only required columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            logger.info(f"Fetched {len(df)} candles for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} from yfinance: {e}")
            raise

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is valid by attempting to fetch info.
        Note: This makes an API call, so use sparingly.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return "symbol" in info or "shortName" in info
        except Exception:
            return False

    def list_symbols(self) -> List[str]:
        """
        yfinance doesn't provide a symbol list endpoint.
        Return empty list; users must provide their own symbols.
        """
        logger.warning("yfinance does not support listing symbols")
        return []

    def _map_timeframe(self, timeframe: str) -> str:
        """Map our timeframe notation to yfinance intervals."""
        # yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
        }
        if timeframe in mapping:
            return mapping[timeframe]
        else:
            logger.warning(f"Timeframe '{timeframe}' not in standard mapping, using as-is")
            return timeframe


def get_provider(asset_type: str, **kwargs) -> IDataProvider:
    """
    Factory function to get appropriate data provider.

    Args:
        asset_type: 'crypto' or 'etf'
        **kwargs: Provider-specific arguments

    Returns:
        Configured IDataProvider instance
    """
    if asset_type.lower() == "crypto":
        exchange_id = kwargs.get("exchange_id", "binance")
        return CCXTProvider(exchange_id=exchange_id)
    elif asset_type.lower() in ("etf", "stock", "equity"):
        return YFinanceProvider()
    else:
        raise ValueError(f"Unknown asset_type: {asset_type}. Use 'crypto' or 'etf'")
