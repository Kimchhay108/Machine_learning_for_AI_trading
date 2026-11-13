"""Unit tests for DuckDB analytical manager and view queries."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.core.constants import Timeframe
from src.data.duckdb_manager import DuckDBManager
from src.data.storage import ParquetStorage


class TestDuckDBManager:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.proc_dir = Path(self.temp_dir.name) / "processed"
        self.db_file = Path(self.temp_dir.name) / "test.duckdb"

        self.storage = ParquetStorage(raw_base_dir=self.raw_dir, processed_base_dir=self.proc_dir)
        self.manager = DuckDBManager(db_path=self.db_file, raw_dir=self.raw_dir, processed_dir=self.proc_dir)

        # Seed test candles
        base_ts = 1714521600000
        candles = []
        for i in range(20):
            candles.append({
                "timestamp": base_ts + i * 300_000,
                "open": 2300.0 + i * 0.5,
                "high": 2305.0 + i * 0.5,
                "low": 2295.0 + i * 0.5,
                "close": 2302.0 + i * 0.5,
                "volume": 5.0,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            })
        self.storage.save_candles(pd.DataFrame(candles), timeframe=Timeframe.M5, is_processed=True)

    def teardown_method(self):
        self.manager.close()
        self.temp_dir.cleanup()

    def test_create_views_and_query(self):
        views = self.manager.create_views(instrument="XAUUSD")
        assert "v_xauusd_m5_processed" in views

        df = self.manager.query("SELECT COUNT(*) as cnt FROM v_xauusd_m5_processed")
        assert df["cnt"].iloc[0] == 20

    def test_candle_summary(self):
        summary = self.manager.get_candle_summary(instrument="XAUUSD", timeframe="m5", is_processed=True)
        assert summary["total_candles"] == 20
        assert summary["lowest_price"] == 2295.0
        assert summary["highest_price"] == 2305.0 + 19 * 0.5
        assert summary["total_volume"] == 100.0

    def test_get_recent_candles(self):
        recent = self.manager.get_recent_candles(instrument="XAUUSD", timeframe="m5", limit=5)
        assert len(recent) == 5
        # Must be sorted ascending
        assert recent["timestamp"].is_monotonic_increasing
