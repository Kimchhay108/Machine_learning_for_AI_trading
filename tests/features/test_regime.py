"""Unit tests for market regime classification and volatility states."""

import numpy as np
import pandas as pd
import pytest

from src.features.regime import MarketRegimeFeatures
from src.features.technical import TechnicalFeatures


class TestMarketRegimeFeatures:
    def setup_method(self):
        # 100 bars with known upward trend followed by choppy range
        n = 100
        timestamps = 1000 + np.arange(n) * 300_000
        # Uptrend for first 50 bars, range for next 50 bars
        uptrend = 2000.0 + np.arange(50) * 2.0
        ranging = 2100.0 + np.sin(np.arange(50)) * 2.0
        close = np.concatenate([uptrend, ranging])
        high = close + 1.0
        low = close - 1.0
        open_p = close - 0.5

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 10.0),
        })
        self.df = TechnicalFeatures.add_atr(df, period=14)
        self.df = TechnicalFeatures.add_bollinger_bands(self.df, period=20)

    def test_efficiency_ratio_trending_vs_ranging(self):
        res = MarketRegimeFeatures.add_trend_efficiency(self.df, period=20)
        assert "efficiency_ratio_20" in res.columns
        assert "trend_slope_20" in res.columns
        # During pure linear uptrend (bars 25-45), efficiency ratio should be close to 1.0
        assert res["efficiency_ratio_20"].iloc[35] > 0.80
        # During sine wave range (bars 75-95), efficiency ratio should drop significantly
        assert res["efficiency_ratio_20"].iloc[85] < 0.50

    def test_volatility_regime(self):
        res = MarketRegimeFeatures.add_volatility_regime(self.df)
        assert "is_high_volatility" in res.columns
        assert "is_low_volatility" in res.columns

    def test_composite_regime(self):
        res = MarketRegimeFeatures.compute_all(self.df)
        assert "market_regime" in res.columns
        assert "regime_label" in res.columns
        assert set(res["market_regime"].unique()).issubset({0, 1, 2, 3})
