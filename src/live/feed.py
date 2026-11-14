"""Live market data feed and market session status monitor.

Tracks official global Gold (XAUUSD) trading sessions, detects market closures
(weekend and daily roll maintenance), and maps live Dukascopy feeds to the canonical schema.
"""

from datetime import datetime, time, timezone
from typing import Any, Dict, Optional, Tuple
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
    """Manages real-time candle polling, market session detection, and schema validation."""

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

    @staticmethod
    def is_market_open(dt: Optional[datetime] = None) -> Tuple[bool, str, str]:
        """Determine whether the global Gold (XAUUSD) market is open or closed.

        Gold market standard schedule (UTC):
        - Opens: Sunday 22:00 UTC (Tokyo/Sydney pre-market)
        - Closes: Friday 21:00 UTC (New York close)
        - Daily Maintenance Break: Monday-Thursday 21:00 to 22:00 UTC
        - Weekend Closed: Friday 21:00 UTC through Sunday 22:00 UTC

        Returns:
            Tuple of (is_open: bool, status_code: str, human_reason: str)
        """
        now = dt or datetime.now(timezone.utc)
        weekday = now.weekday()  # Monday = 0, Friday = 4, Saturday = 5, Sunday = 6
        hour = now.hour

        # 1. Saturday: Entire day closed
        if weekday == 5:
            return False, "CLOSED_WEEKEND", "Market is CLOSED for the weekend. Reopens Sunday at 22:00 UTC."

        # 2. Friday: Closes at 21:00 UTC
        if weekday == 4 and hour >= 21:
            return False, "CLOSED_WEEKEND", "Market is CLOSED for the weekend (closed Friday 21:00 UTC). Reopens Sunday at 22:00 UTC."

        # 3. Sunday: Closed before 22:00 UTC
        if weekday == 6 and hour < 22:
            return False, "CLOSED_WEEKEND", "Market is CLOSED for the weekend. Reopens Sunday at 22:00 UTC."

        # 4. Monday - Thursday: Daily 21:00 - 22:00 UTC settlement roll break
        if weekday in [0, 1, 2, 3] and hour == 21:
            return False, "CLOSED_DAILY_ROLL", "Market is in the 1-hour daily maintenance roll break (21:00-22:00 UTC)."

        # Determine active trading session
        session_name = "Asian / Sydney"
        if 7 <= hour < 16:
            session_name = "London European"
        elif 12 <= hour < 21:
            session_name = "New York American"

        return True, "OPEN", f"Market is OPEN (Active: {session_name} Session)"

    def get_market_status(self, instrument: str = DEFAULT_INSTRUMENT) -> Dict[str, Any]:
        """Generate structured market status metadata for API and frontend display."""
        now = datetime.now(timezone.utc)
        is_open, status_code, reason = self.is_market_open(now)

        # Get latest stored candle
        stored_df = self.storage.load_candles(
            timeframe=Timeframe.M5,
            instrument=instrument,
            is_processed=True,
        )

        latest_time_str = "No stored data"
        latest_price = 0.0
        data_count = 0
        if not stored_df.empty:
            data_count = len(stored_df)
            last_row = stored_df.iloc[-1]
            latest_price = float(last_row["close"])
            raw_ts = int(last_row["timestamp"])
            ts_sec = raw_ts // 1000 if raw_ts > 10_000_000_000 else raw_ts
            latest_time_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return {
            "instrument": instrument.upper(),
            "is_market_open": is_open,
            "status_code": status_code,
            "status_message": reason,
            "current_utc_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "data_source_mode": "STORED_DATASET" if not is_open else "LIVE_FEED_READY",
            "latest_candle_time_utc": latest_time_str,
            "latest_available_price": latest_price,
            "total_stored_m5_candles": data_count,
        }

    def fetch_latest_candles(
        self,
        instrument: str = DEFAULT_INSTRUMENT,
        timeframe: Timeframe = Timeframe.M5,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch the most recent verified candles.

        If market is closed (e.g. weekend), seamlessly serves the latest stored historical candles
        without fabricating missing bars.
        """
        now = datetime.now(timezone.utc)
        is_open, status_code, _ = self.is_market_open(now)

        # Only query live network feed if market is actually open
        if is_open:
            try:
                raw_m1 = self.client.fetch_day_minute_candles(
                    instrument=instrument,
                    year=now.year,
                    month=now.month,
                    day=now.day,
                    price_type="BID",
                )
                if raw_m1:
                    m1_df = pd.DataFrame(raw_m1)
                    resampled = self.resampler.resample(m1_df, timeframe)
                    clean_df, _ = self.validator.validate(resampled, instrument=instrument, timeframe=timeframe)
                    if not clean_df.empty:
                        logger.debug(f"Fetched {len(clean_df)} live {timeframe.value} candles.")
                        return clean_df.tail(limit).reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Live fetch error during open hours: {e}; falling back to storage.")

        # Fall back to latest stored processed candles
        stored_df = self.storage.load_candles(
            timeframe=timeframe,
            instrument=instrument,
            is_processed=True,
        )
        if not stored_df.empty:
            return stored_df.tail(limit).reset_index(drop=True)

        return pd.DataFrame()
