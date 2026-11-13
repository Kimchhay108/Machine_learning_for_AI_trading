"""Unit tests for the data quality reporter."""

import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data.quality_reporter import QualityReporter
from src.data.validator import ValidationResult


class TestQualityReporter:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reports_dir = Path(self.temp_dir.name)
        self.reporter = QualityReporter(reports_dir=self.reports_dir)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_generate_report_files(self):
        base_ts = 1714521600000
        candles = [
            {
                "timestamp": base_ts + i * 300_000,
                "open": 2300.0,
                "high": 2305.0,
                "low": 2295.0,
                "close": 2302.0,
                "volume": 10.0,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            }
            for i in range(10)
        ]
        df = pd.DataFrame(candles)

        val_result = ValidationResult(
            is_valid=True,
            total_candles=10,
            valid_candles=10,
            duplicate_count=0,
            ohlc_violations=0,
            price_range_violations=0,
            anomaly_spikes=0,
            volume_anomalies=0,
            missing_gaps_count=0,
            weekend_gaps_count=0,
            intraweek_gaps_count=0,
        )

        rep = self.reporter.generate_report(df, val_result, instrument="XAUUSD", timeframe="m5")

        json_path = Path(rep["json_path"])
        md_path = Path(rep["md_path"])

        assert json_path.exists()
        assert md_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["metadata"]["instrument"] == "XAUUSD"
            assert data["validation_summary"]["is_passed"] is True
            assert data["dataset_overview"]["valid_candles_retained"] == 10

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
            assert "XAUUSD" in md_text
            assert "PASSED" in md_text
