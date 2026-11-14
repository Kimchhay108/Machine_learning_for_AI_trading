"""Live market data feed service.

Provides real-time candle polling from free financial endpoints (Dukascopy Jetta API)
with automatic schema normalization and fallback handling for market closures.
"""

from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.data.dukascopy_client import DukascopyClient
from src.data.resampler import CandlestickResampler
from src.data.storage import ParquetStorage
from src.data.validator import DataValidator

logger = get_logger("live.feed")


class LiveMarketFeed:
    """Manages real-time candle polling and ingestion for live paper trading."""

    def __init__(
        self,
        client: Optional[DukascopyClient] = None,
        storage: Optional[ParquetStorage] = None,
        validator: Optional[DataValidator] = None,
    ):
        self.client = client or DukascopyClient()
        self.storage = storage or ParquetStorage()
        self.validator = validator or DataValidator()
        self.resampler = CandlestickResampler()

    def fetch_latest_candles(
        self,
        instrument: str = DEFAULT_INSTRUMENT,
        timeframe: Timeframe = Timeframe.M5,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch the most recent verified candles.

        Attempts to query current UTC day from live endpoint; if market is closed (e.g. weekend)
        or connection fails, smoothly falls back to the most recent stored Parquet records.
        """
        now = datetime.now(timezone.utc)
        live_df = pd.DataFrame()

        try:
            # Attempt to pull current UTC day's raw M1 candles
            m1_df = self.client.fetch_day_candles(
                instrument=instrument,
                day=now,
                timeframe=Timeframe.M1,
                price_type="BID",
            )
            if not m1_df.empty:
                # Resample into desired timeframe
                resampled = self.resampler.resample(m1_df, timeframe)
                clean_df, report = self.validator.validate(resampled, instrument=instrument, timeframe=timeframe)
                if not clean_df.empty:
                    live_df = clean_df
                    logger.debug(f"Fetched {len(live_df)} live {timeframe.value} candles from Dukascopy.")
        except Exception as e:
            logger.warning(f"Live candle fetch notice: {e}; falling back to stored dataset.")

        # Fallback to stored processed candles if live candles empty (e.g. weekend or pre-market)
        if live_df.empty:
            stored_df = self.storage.load_candles(
                timeframe=timeframe,
                instrument=instrument,
                is_processed=True,
            )
            if not stored_df.empty:
                return stored_df.tail(limit).reset_index(drop=True)
            return pd.DataFrame()

        return live_df.tail(limit).reset_index(drop=True)
