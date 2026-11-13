"""Unit tests for the data validation and anomaly detection engine."""

import pandas as pd
import pytest

from src.core.constants import Timeframe
from src.data.validator import DataValidator


class TestDataValidator:
    def setup_method(self):
        self.validator = DataValidator(min_price=1000.0, max_price=3500.0)
        self.base_ts = 1714521600000  # 2024-05-01 00:00:00 UTC (Wednesday)

    def test_valid_candles_pass(self):
        # 10 normal M5 candles
        candles = []
        for i in range(10):
            candles.append({
                "timestamp": self.base_ts + i * 300_000,
                "open": 2300.0 + i,
                "high": 2305.0 + i,
                "low": 2295.0 + i,
                "close": 2302.0 + i,
                "volume": 10.0,
                "instrument": "XAUUSD",
                "timeframe": "m5",
                "data_source": "dukascopy",
            })
        df = pd.DataFrame(candles)
        res, clean_df = self.validator.validate(df, timeframe="m5")

        assert res.is_valid is True
        assert res.total_candles == 10
        assert res.valid_candles == 10
        assert res.ohlc_violations == 0
        assert res.duplicate_count == 0
        assert res.price_range_violations == 0
        assert res.missing_gaps_count == 0
        assert len(clean_df) == 10

    def test_ohlc_geometric_violations_detected(self):
        candles = [
            # High is lower than Open (Violates High >= Open)
            {"timestamp": self.base_ts, "open": 2300.0, "high": 2290.0, "low": 2280.0, "close": 2285.0, "volume": 1.0},
            # Low is higher than Close (Violates Low <= Close)
            {"timestamp": self.base_ts + 300_000, "open": 2300.0, "high": 2310.0, "low": 2305.0, "close": 2302.0, "volume": 1.0},
            # Negative price
            {"timestamp": self.base_ts + 600_000, "open": -2300.0, "high": 2310.0, "low": -2310.0, "close": 2302.0, "volume": 1.0},
        ]
        df = pd.DataFrame(candles)
        res, clean_df = self.validator.validate(df, timeframe="m5")

        assert res.is_valid is False
        assert res.ohlc_violations == 3
        assert len(clean_df) == 0

    def test_price_range_violations_detected(self):
        candles = [
            # Too cheap ($100 when minimum is $1000)
            {"timestamp": self.base_ts, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1.0},
            # Too expensive ($50,000 when maximum is $3500)
            {"timestamp": self.base_ts + 300_000, "open": 50000.0, "high": 50050.0, "low": 49950.0, "close": 50010.0, "volume": 1.0},
        ]
        df = pd.DataFrame(candles)
        res, clean_df = self.validator.validate(df, timeframe="m5")

        assert res.is_valid is False
        assert res.price_range_violations == 2
        assert len(clean_df) == 0

    def test_duplicate_timestamps_handled(self):
        candles = [
            {"timestamp": self.base_ts, "open": 2300.0, "high": 2305.0, "low": 2295.0, "close": 2302.0, "volume": 1.0},
            {"timestamp": self.base_ts, "open": 2300.0, "high": 2305.0, "low": 2295.0, "close": 2303.0, "volume": 2.0},  # Duplicate
            {"timestamp": self.base_ts + 300_000, "open": 2303.0, "high": 2306.0, "low": 2300.0, "close": 2304.0, "volume": 1.0},
        ]
        df = pd.DataFrame(candles)
        res, clean_df = self.validator.validate(df, timeframe="m5")

        assert res.duplicate_count == 1
        assert len(clean_df) == 2
        assert clean_df.iloc[0]["close"] == 2303.0  # Kept latest

    def test_missing_candle_gaps_detected(self):
        # Candle at 00:00, then jump 30 minutes to 00:30 (missing 5 bars on M5) on Wednesday
        candles = [
            {"timestamp": self.base_ts, "open": 2300.0, "high": 2305.0, "low": 2295.0, "close": 2302.0, "volume": 1.0},
            {"timestamp": self.base_ts + 1_800_000, "open": 2302.0, "high": 2306.0, "low": 2300.0, "close": 2305.0, "volume": 1.0},
        ]
        df = pd.DataFrame(candles)
        res, clean_df = self.validator.validate(df, timeframe="m5")

        assert res.missing_gaps_count == 1
        assert res.intraweek_gaps_count == 1
        assert res.gaps[0].expected_missing_candles == 5
        assert res.gaps[0].is_weekend_gap is False

    def test_single_candle_spike_anomaly_detected(self):
        # 25% single candle price range
        candles = [
            {"timestamp": self.base_ts, "open": 2000.0, "high": 2600.0, "low": 2000.0, "close": 2550.0, "volume": 50.0},
        ]
        df = pd.DataFrame(candles)
        res, clean_df = self.validator.validate(df, timeframe="m5")

        assert res.anomaly_spikes == 1
        assert any("extreme price span" in w for w in res.warnings)
