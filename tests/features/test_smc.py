"""Unit tests for Smart Money Concepts (SMC) algorithmic features."""

import numpy as np
import pandas as pd
import pytest

from src.features.smc import SmartMoneyConcepts


class TestSmartMoneyConcepts:
    def setup_method(self):
        # Construct controlled price swings to verify BOS, CHoCH, Sweeps
        base_ts = 1714521600000
        # Swing pattern:
        # 0..6: Low around 2000, then rises to Swing High at bar 3 (2050), then drops to 2010
        # 7..12: Rises to 2060 (BOS Bullish break of 2050), then drops to 2020
        # 13..18: Drops to 1990 (CHoCH Bearish break of prior swing low)
        n = 30
        timestamps = base_ts + np.arange(n) * 300_000
        prices = [
            2000, 2010, 2030, 2050, 2030, 2020, 2010,  # 0..6 (SH at 3 = 2050)
            2025, 2040, 2055, 2065, 2045, 2030,        # 7..12 (breaks 2050)
            2025, 2015, 2005, 1990, 1995, 2000,        # 13..18 (breaks 2010)
            2005, 2010, 2015, 2020, 2025, 2030,        # 19..24
            2035, 2040, 2045, 2050, 2055,               # 25..29
        ]
        close = np.array(prices, dtype=float)
        high = close + 2.0
        low = close - 2.0
        open_p = close - 0.5
        volume = np.full(n, 10.0)

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

    def test_identify_swings_no_lookahead(self):
        sh_flag, sl_flag, act_sh, act_sl = SmartMoneyConcepts.identify_swings(self.df, lookback=2)
        # Verify that swing highs are confirmed only with lookback lag
        # At bar 3 price was peak (2050), so with lookback 2 it should be confirmed at bar 3+2 = 5
        assert sh_flag.iloc[5] == 1
        # Before confirmation at bar 5, active swing high must not be 2052
        assert act_sh.iloc[4] != 2052.0 or np.isnan(act_sh.iloc[4])
        # At bar 5, active swing high must be confirmed as 2052.0 (high of bar 3)
        assert act_sh.iloc[5] == 2052.0

    def test_market_structure_bos_choch(self):
        res = SmartMoneyConcepts.add_market_structure(self.df, lookback=2)
        assert "bos_bullish" in res.columns
        assert "bos_bearish" in res.columns
        assert "choch_bullish" in res.columns
        assert "choch_bearish" in res.columns
        assert "market_structure_trend" in res.columns

    def test_fvg_detection(self):
        # Create an artificial Fair Value Gap:
        # Bar 0: High = 2000
        # Bar 1: Strong bullish candle (Low 2001, High 2020)
        # Bar 2: Low = 2010, High = 2025 (Low 2010 > Bar 0 High 2000 -> 10 pt gap!)
        fvg_df = pd.DataFrame({
            "timestamp": [1000, 2000, 3000, 4000],
            "open": [1990, 2001, 2012, 2015],
            "high": [2000, 2020, 2025, 2020],
            "low": [1985, 2001, 2010, 2011],
            "close": [1998, 2019, 2022, 2018],
            "volume": [10, 50, 20, 10],
            "instrument": ["XAUUSD"] * 4,
            "timeframe": ["m5"] * 4,
            "data_source": ["dukascopy"] * 4,
        })
        res = SmartMoneyConcepts.add_fvg(fvg_df)
        assert res["fvg_bullish_new"].iloc[2] == 1
        assert res["fvg_bullish_unmitigated_count"].iloc[2] >= 1

    def test_premium_discount(self):
        res = SmartMoneyConcepts.add_premium_discount(self.df)
        assert "dealing_range_pos" in res.columns
        assert "is_discount_zone" in res.columns
        assert "is_premium_zone" in res.columns
        assert (res["dealing_range_pos"] >= 0.0).all() and (res["dealing_range_pos"] <= 1.0).all()

    def test_sessions_and_pdh_pdl(self):
        res = SmartMoneyConcepts.add_session_and_pdh_pdl(self.df)
        assert "session_asian" in res.columns
        assert "session_london" in res.columns
        assert "session_ny" in res.columns
        assert "session_london_ny_overlap" in res.columns
        assert "pdh" in res.columns
        assert "pdl" in res.columns
        assert "swept_pdh" in res.columns
        assert "swept_pdl" in res.columns

    def test_compute_all_smc(self):
        res = SmartMoneyConcepts.compute_all(self.df)
        assert len(res) == len(self.df)
        assert "market_structure_trend" in res.columns
        assert "order_block_bullish" in res.columns
        assert "ict_2022_bullish" in res.columns
        assert "ict_2022_bearish" in res.columns
        assert "ict_silver_bullet_window" in res.columns
        assert "ict_silver_bullet_bullish" in res.columns
        assert "ict_silver_bullet_bearish" in res.columns
        assert "ict_judas_swing_bullish" in res.columns
        assert "ict_judas_swing_bearish" in res.columns
        assert "ict_turtle_soup_bullish" in res.columns
        assert "ict_turtle_soup_bearish" in res.columns

    def test_ict_patterns_explicit(self):
        res = SmartMoneyConcepts.compute_all(self.df)
        # Verify columns exist and are binary 0/1 series
        for col in [
            "ict_2022_bullish", "ict_2022_bearish",
            "ict_silver_bullet_window", "ict_silver_bullet_bullish", "ict_silver_bullet_bearish",
            "ict_judas_swing_bullish", "ict_judas_swing_bearish",
            "ict_turtle_soup_bullish", "ict_turtle_soup_bearish",
        ]:
            assert col in res.columns
            assert set(res[col].unique()).issubset({0, 1})
