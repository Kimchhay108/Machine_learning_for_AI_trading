"""Resampling engine for converting base candlestick bars into higher timeframes.

Strictly preserves UTC timestamps and applies exact financial aggregation rules:
- Open: first open in bucket
- High: max high in bucket
- Low: min low in bucket
- Close: last close in bucket
- Volume: sum of volumes in bucket
- Timestamp: bucket interval floor (UTC epoch ms)
"""

from typing import List, Union
import pandas as pd
import numpy as np

from src.core.constants import (
    CANONICAL_COLUMNS,
    DataSource,
    DEFAULT_INSTRUMENT,
    Timeframe,
    TIMEFRAME_MS,
)
from src.core.logger import get_logger

logger = get_logger("resampler")


class ResamplingError(Exception):
    """Raised when resampling input data is invalid or cannot be aggregated."""
    pass


class CandlestickResampler:
    """Resamples minute-level candles into M5, M15, and H1 timeframes."""

    @staticmethod
    def resample(
        df_or_records: Union[pd.DataFrame, List[dict]],
        target_timeframe: Union[Timeframe, str],
        instrument: str = DEFAULT_INSTRUMENT,
        data_source: str = DataSource.DUKASCOPY.value,
    ) -> pd.DataFrame:
        """Resample input M1 candle records into target timeframe.

        Args:
            df_or_records: DataFrame or list of dicts with [timestamp, open, high, low, close, volume]
            target_timeframe: Target timeframe enum or string ('m5', 'm15', 'h1')
            instrument: Symbol identifier, e.g. "XAUUSD"
            data_source: Source tag, e.g. "dukascopy"

        Returns:
            Standardized DataFrame sorted by timestamp ascending with canonical columns.
        """
        tf_str = target_timeframe.value if isinstance(target_timeframe, Timeframe) else target_timeframe.lower()

        if isinstance(df_or_records, list):
            if not df_or_records:
                return pd.DataFrame(columns=CANONICAL_COLUMNS)
            df = pd.DataFrame(df_or_records)
        else:
            df = df_or_records.copy()

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        required_raw_cols = ["timestamp", "open", "high", "low", "close"]
        for col in required_raw_cols:
            if col not in df.columns:
                raise ResamplingError(f"Missing required candle column '{col}' for resampling.")

        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Ensure correct numeric types
        df["timestamp"] = df["timestamp"].astype("int64")
        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].astype("float64")

        # Sort chronologically and drop duplicate timestamps before bucketing
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        # If already at target timeframe, normalize and return
        interval_ms = TIMEFRAME_MS.get(tf_str)
        if interval_ms is None:
            raise ResamplingError(f"Unsupported target timeframe: {tf_str}. Must be in {list(TIMEFRAME_MS.keys())}")

        if tf_str == Timeframe.M1.value:
            df["instrument"] = instrument
            df["timeframe"] = tf_str
            df["data_source"] = data_source
            return df[CANONICAL_COLUMNS].reset_index(drop=True)

        # Floor timestamp to bucket start
        # e.g., for M5 (300,000 ms), timestamp // 300000 * 300000
        df["bucket"] = (df["timestamp"] // interval_ms) * interval_ms

        # Group by bucket and compute exact OHLCV aggregations
        aggregated = df.groupby("bucket", as_index=False).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).rename(columns={"bucket": "timestamp"})

        # Sanity check: High >= Low, High >= Open, High >= Close
        aggregated["high"] = aggregated[["open", "high", "close"]].max(axis=1)
        aggregated["low"] = aggregated[["open", "low", "close"]].min(axis=1)
        aggregated["volume"] = aggregated["volume"].round(6)

        # Append metadata columns
        aggregated["instrument"] = instrument
        aggregated["timeframe"] = tf_str
        aggregated["data_source"] = data_source

        # Sort ascending and enforce canonical column order
        aggregated = aggregated.sort_values("timestamp").reset_index(drop=True)
        return aggregated[CANONICAL_COLUMNS]
