"""Unit tests for candlestick resampling engine."""

import pandas as pd
import pytest

from src.core.constants import CANONICAL_COLUMNS, Timeframe
from src.data.resampler import CandlestickResampler, ResamplingError


class TestCandlestickResampler:
    def test_resample_m1_to_m5(self):
        # Create 5 contiguous 1-minute bars
        # Base: 1714521600000 (00:00 UTC)
        base_ts = 1714521600000
        m1_bars = [
            {"timestamp": base_ts + 0 * 60000, "open": 2000.0, "high": 2002.0, "low": 1999.0, "close": 2001.0, "volume": 1.0},
            {"timestamp": base_ts + 1 * 60000, "open": 2001.0, "high": 2005.0, "low": 2000.5, "close": 2004.0, "volume": 2.0},
            {"timestamp": base_ts + 2 * 60000, "open": 2004.0, "high": 2004.5, "low": 1998.0, "close": 1998.5, "volume": 1.5},
            {"timestamp": base_ts + 3 * 60000, "open": 1998.5, "high": 2001.0, "low": 1998.0, "close": 2000.0, "volume": 0.5},
            {"timestamp": base_ts + 4 * 60000, "open": 2000.0, "high": 2003.0, "low": 1999.5, "close": 2002.5, "volume": 1.0},
        ]

        df_m5 = CandlestickResampler.resample(m1_bars, target_timeframe=Timeframe.M5, instrument="XAUUSD")

        assert len(df_m5) == 1
        m5_bar = df_m5.iloc[0]

        # Timestamp must be bucket start (00:00 UTC)
        assert m5_bar["timestamp"] == base_ts
        # Open: first bar's open
        assert m5_bar["open"] == 2000.0
        # High: max high across all 5 bars (2005.0)
        assert m5_bar["high"] == 2005.0
        # Low: min low across all 5 bars (1998.0)
        assert m5_bar["low"] == 1998.0
        # Close: last bar's close (2002.5)
        assert m5_bar["close"] == 2002.5
        # Volume: sum of volumes (1.0 + 2.0 + 1.5 + 0.5 + 1.0 = 6.0)
        assert m5_bar["volume"] == 6.0
        assert m5_bar["instrument"] == "XAUUSD"
        assert m5_bar["timeframe"] == "m5"
        assert list(df_m5.columns) == CANONICAL_COLUMNS

    def test_resample_m15_and_h1(self):
        base_ts = 1714521600000
        # Create 60 1-minute bars (1 hour total)
        m1_bars = []
        for i in range(60):
            m1_bars.append({
                "timestamp": base_ts + i * 60000,
                "open": 2000.0 + i * 0.1,
                "high": 2001.0 + i * 0.1,
                "low": 1999.0 + i * 0.1,
                "close": 2000.5 + i * 0.1,
                "volume": 0.1,
            })

        df_m15 = CandlestickResampler.resample(m1_bars, target_timeframe=Timeframe.M15)
        assert len(df_m15) == 4  # 60 min / 15 min = 4 candles

        df_h1 = CandlestickResampler.resample(m1_bars, target_timeframe=Timeframe.H1)
        assert len(df_h1) == 1   # 60 min / 60 min = 1 candle
        assert df_h1.iloc[0]["open"] == m1_bars[0]["open"]
        assert df_h1.iloc[0]["close"] == m1_bars[-1]["close"]
        assert round(df_h1.iloc[0]["volume"], 4) == round(60 * 0.1, 4)

    def test_empty_resample(self):
        df_empty = CandlestickResampler.resample([], target_timeframe=Timeframe.M5)
        assert df_empty.empty
        assert list(df_empty.columns) == CANONICAL_COLUMNS

    def test_unsupported_timeframe(self):
        m1_bars = [{"timestamp": 1000, "open": 2000, "high": 2001, "low": 1999, "close": 2000, "volume": 1}]
        with pytest.raises(ResamplingError):
            CandlestickResampler.resample(m1_bars, target_timeframe="d100")
