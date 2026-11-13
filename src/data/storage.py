"""Parquet storage manager for raw and processed market data.

Enforces strict PyArrow schemas, supports partition-by-month storage for scalability,
and implements atomic incremental merges so repeated ingestions never create duplicates.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.config import settings
from src.core.constants import CANONICAL_COLUMNS, DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger

logger = get_logger("storage")

# Canonical PyArrow schema for all candlestick Parquet files
PYARROW_CANDLE_SCHEMA = pa.schema([
    ("timestamp", pa.int64()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("instrument", pa.string()),
    ("timeframe", pa.string()),
    ("data_source", pa.string()),
])


class ParquetStorage:
    """Manages reading, writing, and merging Parquet candlestick partitions."""

    def __init__(
        self,
        raw_base_dir: Optional[Path] = None,
        processed_base_dir: Optional[Path] = None,
    ):
        self.raw_base_dir = raw_base_dir or settings.RAW_DATA_DIR
        self.processed_base_dir = processed_base_dir or settings.PROCESSED_DATA_DIR

    def _get_target_dir(
        self,
        is_processed: bool,
        instrument: str,
        timeframe: str,
    ) -> Path:
        base = self.processed_base_dir if is_processed else self.raw_base_dir
        target_dir = base / instrument.lower() / timeframe.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _get_partition_filename(self, timestamp_ms: int) -> str:
        """Partition by year and month (e.g. 2024_05.parquet)."""
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        return f"{dt.year:04d}_{dt.month:02d}.parquet"

    def save_candles(
        self,
        df: pd.DataFrame,
        timeframe: Union[Timeframe, str],
        instrument: str = DEFAULT_INSTRUMENT,
        is_processed: bool = False,
        compression: str = "snappy",
    ) -> List[Path]:
        """Save candles into month-partitioned Parquet files, merging incrementally with existing data.

        Args:
            df: Candlestick DataFrame with CANONICAL_COLUMNS
            timeframe: e.g. "m5", "m15", "h1"
            instrument: Symbol, default "XAUUSD"
            is_processed: False for data/raw, True for data/processed
            compression: Parquet compression ('snappy' or 'zstd')

        Returns:
            List of Path objects for all updated or created Parquet files.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to save_candles. No files written.")
            return []

        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else timeframe.lower()
        target_dir = self._get_target_dir(is_processed, instrument, tf_str)

        clean_df = df[CANONICAL_COLUMNS].copy()
        clean_df["timestamp"] = clean_df["timestamp"].astype("int64")

        # Create year-month partition column
        dt_series = pd.to_datetime(clean_df["timestamp"], unit="ms", utc=True)
        clean_df["partition_key"] = dt_series.dt.strftime("%Y_%m")

        saved_files: List[Path] = []

        for part_key, group in clean_df.groupby("partition_key"):
            file_path = target_dir / f"{part_key}.parquet"
            part_data = group.drop(columns=["partition_key"])

            # Merge with existing partition if already present
            if file_path.exists():
                try:
                    existing_table = pq.read_table(file_path)
                    existing_df = existing_table.to_pandas()
                    combined = pd.concat([existing_df, part_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
                    combined = combined.sort_values("timestamp").reset_index(drop=True)
                    part_data = combined
                except Exception as exc:
                    logger.error(f"Error merging with existing file {file_path}: {exc}. Overwriting.")

            # Convert to PyArrow Table with strict schema
            table = pa.Table.from_pandas(part_data[CANONICAL_COLUMNS], schema=PYARROW_CANDLE_SCHEMA)
            pq.write_table(table, file_path, compression=compression)
            saved_files.append(file_path)
            logger.info(f"Saved {len(part_data)} candles to {file_path}")

        return saved_files

    def load_candles(
        self,
        timeframe: Union[Timeframe, str],
        instrument: str = DEFAULT_INSTRUMENT,
        is_processed: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load candles across partitions for a date range into a single DataFrame.

        Args:
            timeframe: e.g. "m5", "m15", "h1"
            instrument: Symbol, default "XAUUSD"
            is_processed: Whether to read from raw or processed directory
            start_date: Optional UTC start datetime filter
            end_date: Optional UTC end datetime filter

        Returns:
            Concatenated, sorted DataFrame with CANONICAL_COLUMNS.
        """
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else timeframe.lower()
        target_dir = self._get_target_dir(is_processed, instrument, tf_str)

        parquet_files = sorted(list(target_dir.glob("*.parquet")))
        if not parquet_files:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        tables = []
        for file_path in parquet_files:
            try:
                tbl = pq.read_table(file_path)
                tables.append(tbl)
            except Exception as exc:
                logger.error(f"Failed to read Parquet file {file_path}: {exc}")

        if not tables:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        full_table = pa.concat_tables(tables)
        df = full_table.to_pandas()

        # Date filtering
        if start_date is not None:
            start_ts = int(start_date.replace(tzinfo=timezone.utc).timestamp() * 1000)
            df = df[df["timestamp"] >= start_ts]

        if end_date is not None:
            end_ts = int(end_date.replace(tzinfo=timezone.utc).timestamp() * 1000)
            df = df[df["timestamp"] <= end_ts]

        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[CANONICAL_COLUMNS]

    def get_stored_range(
        self,
        timeframe: Union[Timeframe, str],
        instrument: str = DEFAULT_INSTRUMENT,
        is_processed: bool = False,
    ) -> Optional[Tuple[int, int, int]]:
        """Return (min_timestamp, max_timestamp, total_records) for stored dataset, or None if empty."""
        tf_str = timeframe.value if isinstance(timeframe, Timeframe) else timeframe.lower()
        target_dir = self._get_target_dir(is_processed, instrument, tf_str)
        parquet_files = sorted(list(target_dir.glob("*.parquet")))

        if not parquet_files:
            return None

        total_rows = 0
        min_ts = float("inf")
        max_ts = float("-inf")

        for file_path in parquet_files:
            meta = pq.read_metadata(file_path)
            total_rows += meta.num_rows

            # Read first and last row timestamp quickly
            t = pq.read_table(file_path, columns=["timestamp"])
            ts_arr = t["timestamp"].to_numpy()
            if len(ts_arr) > 0:
                min_ts = min(min_ts, int(ts_arr.min()))
                max_ts = max(max_ts, int(ts_arr.max()))

        if total_rows == 0:
            return None

        return int(min_ts), int(max_ts), total_rows
