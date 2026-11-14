"""Unit tests for technical indicator feature calculations."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalFeatures


class TestTechnicalFeatures:
    def setup_method(self):
        # Generate 100 synthetic M5 candles
        np.random.seed(42)
        base_price = 2300.0
        n = 100
        timestamps = 1714521600000 + np.arange(n) * 300_000

        # Random walk prices
        returns = np.random.normal(0.0001, 0.001, n)
        close = base_price * np.exp(np.cumsum(returns))
        high = close + np.random.uniform(0.5, 2.0, n)
        low = close - np.random.uniform(0.5, 2.0, n)
        open_p = low + np.random.uniform(0.1, 1.0, n) * (high - low)
        volume = np.random.uniform(10.0, 100.0, n)

        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "instrument": "XAUUSD",
            "timeframe": "m5",
            "data_source": "dukascopy",
        })

    def test_emas_and_alignment(self):
        df_ema = TechnicalFeatures.add_emas(self.df, periods=[9, 20, 50, 100, 200])
        for p in [9, 20, 50, 100, 200]:
            assert f"ema_{p}" in df_ema.columns
            assert f"ema_{p}_dist_pct" in df_ema.columns
            assert f"ema_{p}_slope" in df_ema.columns
        assert "ema_alignment" in df_ema.columns
        assert set(df_ema["ema_alignment"].unique()).issubset({-1, 0, 1})

    def test_atr(self):
        df_atr = TechnicalFeatures.add_atr(self.df, period=14)
        assert "true_range" in df_atr.columns
        assert "atr_14" in df_atr.columns
        assert "atr_14_pct" in df_atr.columns
        # ATR must always be strictly positive
        assert (df_atr["atr_14"].dropna() > 0).all()

    def test_rsi(self):
        df_rsi = TechnicalFeatures.add_rsi(self.df, period=14)
        assert "rsi_14" in df_rsi.columns
        assert "rsi_14_norm" in df_rsi.columns
        assert "rsi_14_overbought" in df_rsi.columns
        assert "rsi_14_oversold" in df_rsi.columns
        # RSI must be strictly bounded in [0, 100]
        assert (df_rsi["rsi_14"] >= 0).all() and (df_rsi["rsi_14"] <= 100).all()

    def test_macd(self):
        df_macd = TechnicalFeatures.add_macd(self.df)
        assert "macd_line" in df_macd.columns
        assert "macd_signal" in df_macd.columns
        assert "macd_hist" in df_macd.columns
        # Histogram must equal line - signal
        diff = (df_macd["macd_line"] - df_macd["macd_signal"]) - df_macd["macd_hist"]
        assert np.allclose(diff.dropna(), 0.0)

    def test_bollinger_bands(self):
        df_bb = TechnicalFeatures.add_bollinger_bands(self.df, period=20, num_std=2.0)
        assert "bb_upper" in df_bb.columns
        assert "bb_middle" in df_bb.columns
        assert "bb_lower" in df_bb.columns
        assert "bb_pct_b" in df_bb.columns
        assert "bb_bandwidth" in df_bb.columns
        # Upper must be >= Middle >= Lower
        valid = df_bb.dropna(subset=["bb_upper"])
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_adx(self):
        df_adx = TechnicalFeatures.add_adx(self.df, period=14)
        assert "plus_di_14" in df_adx.columns
        assert "minus_di_14" in df_adx.columns
        assert "adx_14" in df_adx.columns
        assert (df_adx["adx_14"] >= 0).all()

    def test_stochastic_rsi(self):
        df_stoch = TechnicalFeatures.add_stochastic_rsi(self.df)
        assert "stoch_rsi_k" in df_stoch.columns
        assert "stoch_rsi_d" in df_stoch.columns
        assert (df_stoch["stoch_rsi_k"] >= 0).all() and (df_stoch["stoch_rsi_k"] <= 100).all()

    def test_momentum_and_volume(self):
        df_mom = TechnicalFeatures.add_momentum_returns(self.df)
        assert "log_ret_1" in df_mom.columns
        assert "roc_5" in df_mom.columns

        df_vol = TechnicalFeatures.add_volume_features(self.df, period=20)
        assert "vol_sma_20" in df_vol.columns
        assert "vol_ratio_20" in df_vol.columns
        assert "vwap_dist_pct" in df_vol.columns

    def test_compute_all(self):
        res = TechnicalFeatures.compute_all(self.df)
        assert len(res) == len(self.df)
        # Should have added at least 30 feature columns
        assert len(res.columns) > 35
