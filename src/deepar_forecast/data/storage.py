"""Storage layer for OHLCV data using SQLAlchemy (SQLite or Postgres)."""

import os
from datetime import datetime
from typing import List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    desc,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


class OHLCVBar(Base):
    """
    OHLCV bar data model.
    Stores raw price/volume data with metadata for tracking.
    """

    __tablename__ = "ohlcv_bars"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    asset_type = Column(String, nullable=False, index=True)  # 'crypto', 'etf', etc.
    timeframe = Column(String, nullable=False, index=True)  # '1h', '1d', etc.
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    ingested_at = Column(DateTime(timezone=True), nullable=False)  # When data was stored

    # Ensure no duplicate bars
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_symbol_timeframe_timestamp"),
    )

    def __repr__(self):
        return (
            f"<OHLCVBar({self.symbol} {self.timeframe} "
            f"{self.timestamp} close={self.close:.2f})>"
        )


class TrainingRun(Base):
    """
    Training run metadata for experiment tracking.
    Stores configuration, metrics, and paths for each model training run.
    """

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False, index=True)
    model_name = Column(String, nullable=False, index=True)
    symbols = Column(String, nullable=False)  # JSON string of list
    timeframe = Column(String, nullable=False)
    horizon = Column(Integer, nullable=False)
    hyperparams = Column(String, nullable=False)  # JSON string
    train_start = Column(DateTime(timezone=True))
    train_end = Column(DateTime(timezone=True))
    val_start = Column(DateTime(timezone=True))
    val_end = Column(DateTime(timezone=True))
    metrics = Column(String)  # JSON string of metrics dict
    model_path = Column(String)  # Path to saved model file
    status = Column(String, default="running")  # running, completed, failed
    created_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True))

    def __repr__(self):
        return f"<TrainingRun({self.run_id} {self.model_name} {self.status})>"


class DataStorage:
    """
    Storage manager for OHLCV data and training metadata.
    Supports SQLite (default) and PostgreSQL via DATABASE_URL.
    """

    def __init__(self, database_url: str = "sqlite:///data/deepar.db"):
        """
        Initialize storage with database connection.

        Args:
            database_url: SQLAlchemy database URL
                - SQLite: sqlite:///path/to/db.db
                - Postgres: postgresql://user:pass@host:port/dbname
        """
        self.database_url = database_url
        
        # Create directory for SQLite database if needed
        if database_url.startswith("sqlite:///"):
            db_path = database_url.replace("sqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        logger.info(f"Initialized storage: {database_url}")

    def store_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_type: str,
        timeframe: str,
    ) -> int:
        """
        Store OHLCV data from DataFrame.

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Trading symbol
            asset_type: 'crypto', 'etf', etc.
            timeframe: Candle interval

        Returns:
            Number of bars stored (excluding duplicates)
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, nothing to store")
            return 0

        session = self.SessionLocal()
        try:
            ingested_at = datetime.utcnow()
            
            # Normalize timestamps to UTC and convert to naive for consistent SQLite storage
            df = df.copy()
            if df["timestamp"].dt.tz is None:
                # Assume UTC if naive
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize('UTC').dt.tz_localize(None)
            else:
                # Convert to UTC, then make naive
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert('UTC').dt.tz_localize(None)
            
            bars = []
            for _, row in df.iterrows():
                bar = OHLCVBar(
                    symbol=symbol,
                    asset_type=asset_type,
                    timeframe=timeframe,
                    timestamp=row["timestamp"].to_pydatetime(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    ingested_at=ingested_at,
                )
                bars.append(bar)

            # Use SAVEPOINT (nested transaction) to handle duplicates without rolling back entire batch
            # This ensures that duplicate errors only rollback the individual row, not all previous inserts
            stored_count = 0
            duplicate_count = 0
            first_non_duplicate_error = None
            
            for bar in bars:
                try:
                    # Begin nested transaction (SAVEPOINT)
                    with session.begin_nested():
                        session.add(bar)
                    stored_count += 1
                except Exception as e:
                    # Nested transaction automatically rolls back only this row
                    # Check if it's a duplicate constraint violation
                    error_msg = str(e).lower()
                    if 'unique constraint' in error_msg or 'duplicate' in error_msg or 'integrity' in error_msg:
                        duplicate_count += 1
                    else:
                        # Real error, not a duplicate
                        if first_non_duplicate_error is None:
                            first_non_duplicate_error = e
                            logger.error(f"Non-duplicate error storing bar for {symbol} at {bar.timestamp}: {e}", exc_info=True)

            # Commit the outer transaction with all successful insertions
            try:
                session.commit()
                if stored_count > 0:
                    logger.info(f"Stored {stored_count}/{len(bars)} bars for {symbol} {timeframe} (skipped {duplicate_count} duplicates)")
                else:
                    logger.info(f"No new bars stored for {symbol} {timeframe} (all {duplicate_count} were duplicates)")
            except Exception as e:
                logger.error(f"Commit failed: {e}", exc_info=True)
                session.rollback()
                raise
            
            return stored_count

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing OHLCV data: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data from storage.

        Args:
            symbol: Trading symbol
            timeframe: Candle interval
            start: Optional start datetime
            end: Optional end datetime

        Returns:
            DataFrame with OHLCV data, sorted by timestamp
        """
        session = self.SessionLocal()
        try:
            query = session.query(OHLCVBar).filter(
                OHLCVBar.symbol == symbol,
                OHLCVBar.timeframe == timeframe,
            )

            if start:
                query = query.filter(OHLCVBar.timestamp >= start)
            if end:
                query = query.filter(OHLCVBar.timestamp <= end)

            query = query.order_by(OHLCVBar.timestamp)

            bars = query.all()

            if not bars:
                logger.warning(f"No data found for {symbol} {timeframe}")
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Convert to DataFrame
            df = pd.DataFrame(
                [
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                    for bar in bars
                ]
            )

            logger.info(f"Retrieved {len(df)} bars for {symbol} {timeframe}")
            return df

        finally:
            session.close()

    def list_symbols(self, asset_type: Optional[str] = None) -> List[str]:
        """Get list of symbols in storage, optionally filtered by asset type."""
        session = self.SessionLocal()
        try:
            query = session.query(OHLCVBar.symbol).distinct()
            if asset_type:
                query = query.filter(OHLCVBar.asset_type == asset_type)

            symbols = [row[0] for row in query.all()]
            return sorted(symbols)

        finally:
            session.close()

    def store_training_run(self, run_metadata: dict) -> str:
        """
        Store training run metadata.

        Args:
            run_metadata: Dict with run information (see TrainingRun model)

        Returns:
            run_id
        """
        session = self.SessionLocal()
        try:
            run = TrainingRun(**run_metadata)
            session.add(run)
            session.commit()
            logger.info(f"Stored training run: {run.run_id}")
            return run.run_id

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing training run: {e}")
            raise
        finally:
            session.close()

    def update_training_run(self, run_id: str, updates: dict):
        """Update training run with new information (e.g., metrics, status)."""
        session = self.SessionLocal()
        try:
            run = session.query(TrainingRun).filter(TrainingRun.run_id == run_id).first()
            if not run:
                raise ValueError(f"Training run {run_id} not found")

            for key, value in updates.items():
                setattr(run, key, value)

            session.commit()
            logger.info(f"Updated training run: {run_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating training run: {e}")
            raise
        finally:
            session.close()

    def get_training_runs(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """
        Retrieve training runs, optionally filtered by model.

        Returns:
            List of run metadata dicts
        """
        session = self.SessionLocal()
        try:
            query = session.query(TrainingRun)
            if model_name:
                query = query.filter(TrainingRun.model_name == model_name)

            query = query.order_by(desc(TrainingRun.created_at)).limit(limit)

            runs = query.all()

            return [
                {
                    "run_id": run.run_id,
                    "model_name": run.model_name,
                    "symbols": run.symbols,
                    "timeframe": run.timeframe,
                    "horizon": run.horizon,
                    "hyperparams": run.hyperparams,
                    "metrics": run.metrics,
                    "status": run.status,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                }
                for run in runs
            ]

        finally:
            session.close()

    def get_date_range(self, symbol: str, timeframe: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get earliest and latest timestamps for a symbol/timeframe."""
        session = self.SessionLocal()
        try:
            query = session.query(OHLCVBar).filter(
                OHLCVBar.symbol == symbol,
                OHLCVBar.timeframe == timeframe,
            )

            first = query.order_by(OHLCVBar.timestamp).first()
            last = query.order_by(desc(OHLCVBar.timestamp)).first()

            if first and last:
                return first.timestamp, last.timestamp
            else:
                return None, None

        finally:
            session.close()
