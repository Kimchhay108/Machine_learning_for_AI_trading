"""Unit tests for Parquet storage, schema enforcement, and incremental merge."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.core.constants import CANONICAL_COLUMNS, Timeframe
from src.data.storage import ParquetStorage


class TestParquetStorage:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.processed_dir = Path(self.temp_dir.name) / "processed"
        self.storage = ParquetStorage(raw_base_dir=self.raw_dir, processed_base_dir=self.processed_dir)
        self.base_ts = 1714521600000  # 2024-05-01 00:00:00 UTC

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_save_and_load_candles(self):
        candles = []
        for i in range(12):  # 1 hour of M5 candles
            candles.append({
                "timestamp": self.base_ts + i * 300_000,
                "open": 2300.0 + i,
                "high": 2305.0 + i,
                "low": 2295.0 + i,
                "close": 2302.0 + i,
                "volume": 1.5,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            })
        df = pd.DataFrame(candles)

        saved_files = self.storage.save_candles(df, timeframe=Timeframe.M5, is_processed=False)
        assert len(saved_files) == 1
        assert saved_files[0].name == "2024_05.parquet"

        # Load back
        loaded_df = self.storage.load_candles(timeframe=Timeframe.M5, is_processed=False)
        assert len(loaded_df) == 12
        assert list(loaded_df.columns) == CANONICAL_COLUMNS
        assert loaded_df["timestamp"].iloc[0] == self.base_ts

    def test_incremental_merge_without_duplicates(self):
        # First batch: 5 candles
        batch1 = [
            {
                "timestamp": self.base_ts + i * 300_000,
                "open": 2300.0,
                "high": 2305.0,
                "low": 2295.0,
                "close": 2302.0,
                "volume": 1.0,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            }
            for i in range(5)
        ]
        self.storage.save_candles(pd.DataFrame(batch1), timeframe=Timeframe.M5, is_processed=True)

        # Second batch: 3 overlapping candles + 3 new candles (total 8 unique candles)
        batch2 = [
            {
                "timestamp": self.base_ts + i * 300_000,
                "open": 2300.0,
                "high": 2305.0,
                "low": 2295.0,
                "close": 2302.0,
                "volume": 2.0,  # updated volume
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            }
            for i in range(2, 8)
        ]
        self.storage.save_candles(pd.DataFrame(batch2), timeframe=Timeframe.M5, is_processed=True)

        loaded_df = self.storage.load_candles(timeframe=Timeframe.M5, is_processed=True)
        assert len(loaded_df) == 8  # Exactly 8 unique timestamps
        # Check that the overlapping record was updated with the latest batch value
        overlap_row = loaded_df[loaded_df["timestamp"] == (self.base_ts + 2 * 300_000)].iloc[0]
        assert overlap_row["volume"] == 2.0

    def test_get_stored_range(self):
        candles = [
            {
                "timestamp": self.base_ts,
                "open": 2300.0,
                "high": 2305.0,
                "low": 2295.0,
                "close": 2302.0,
                "volume": 1.0,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            },
            {
                "timestamp": self.base_ts + 300_000,
                "open": 2302.0,
                "high": 2307.0,
                "low": 2298.0,
                "close": 2304.0,
                "volume": 1.0,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            },
        ]
        self.storage.save_candles(pd.DataFrame(candles), timeframe=Timeframe.M5, is_processed=False)
        res = self.storage.get_stored_range(timeframe=Timeframe.M5, is_processed=False)
        assert res is not None
        min_ts, max_ts, count = res
        assert min_ts == self.base_ts
        assert max_ts == self.base_ts + 300_000
        assert count == 2
