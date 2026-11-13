"""Unit tests for the HistoricalDownloader orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.core.constants import Timeframe
from src.data.downloader import HistoricalDownloader
from src.data.storage import ParquetStorage


class TestHistoricalDownloader:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.temp_dir.name) / "raw"
        self.proc_dir = Path(self.temp_dir.name) / "processed"
        self.storage = ParquetStorage(raw_base_dir=self.raw_dir, processed_base_dir=self.proc_dir)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_downloader_orchestration(self):
        # Mock DukascopyClient
        mock_client = MagicMock()
        base_ts = 1714521600000  # 2024-05-01 (Wednesday)

        # Generate 60 M1 candles for Wednesday
        wednesday_candles = [
            {
                "timestamp": base_ts + i * 60_000,
                "open": 2300.0 + i * 0.1,
                "high": 2301.0 + i * 0.1,
                "low": 2299.0 + i * 0.1,
                "close": 2300.5 + i * 0.1,
                "volume": 0.5,
            }
            for i in range(60)
        ]

        mock_client.fetch_day_minute_candles.return_value = wednesday_candles

        downloader = HistoricalDownloader(client=mock_client, storage=self.storage)
        result = downloader.download_range(
            start_date="2024-05-01",
            end_date="2024-05-01",
            instrument="XAUUSD",
            progress_bar=False,
        )

        assert result["total_days"] == 1
        assert result["days_with_data"] == 1
        assert result["total_m1_candles"] == 60
        assert result["m5_candles"] == 12   # 60 / 5 = 12 M5 candles
        assert result["m15_candles"] == 4   # 60 / 15 = 4 M15 candles
        assert result["h1_candles"] == 1    # 60 / 60 = 1 H1 candle
        assert len(result["files_written"]) > 0

        # Verify Parquet files exist in storage
        m5_proc = self.storage.load_candles(timeframe=Timeframe.M5, is_processed=True)
        assert len(m5_proc) == 12

        h1_proc = self.storage.load_candles(timeframe=Timeframe.H1, is_processed=True)
        assert len(h1_proc) == 1
