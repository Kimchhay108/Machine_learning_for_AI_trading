"""Unit tests for price action and candlestick morphology features."""

import numpy as np
import pandas as pd
import pytest

from src.features.price_action import PriceActionFeatures


class TestPriceActionFeatures:
    def test_pin_bar_detection(self):
        # Bullish Pin bar: long lower wick, small body at the top
        # Open 2000, Close 2002, High 2003, Low 1990 (wick = 10, range = 13, wick ratio > 0.7)
        pin_df = pd.DataFrame({
            "timestamp": [1000],
            "open": [2000.0],
            "high": [2003.0],
            "low": [1990.0],
            "close": [2002.0],
            "volume": [10.0],
        })
        res = PriceActionFeatures.compute_all(pin_df)
        assert res["pin_bar_bullish"].iloc[0] == 1
        assert res["pin_bar_bearish"].iloc[0] == 0
        assert res["lower_wick_ratio"].iloc[0] > 0.60

    def test_engulfing_patterns(self):
        # Bar 0: Bearish (Open 2010, Close 2000)
        # Bar 1: Bullish Engulfing (Open 1998, Close 2012)
        engulf_df = pd.DataFrame({
            "timestamp": [1000, 2000],
            "open": [2010.0, 1998.0],
            "high": [2012.0, 2015.0],
            "low": [1998.0, 1995.0],
            "close": [2000.0, 2012.0],
            "volume": [10.0, 20.0],
        })
        res = PriceActionFeatures.compute_all(engulf_df)
        assert res["engulfing_bullish"].iloc[1] == 1
        assert res["engulfing_bearish"].iloc[1] == 0

    def test_inside_bar(self):
        inside_df = pd.DataFrame({
            "timestamp": [1000, 2000],
            "open": [2000.0, 2005.0],
            "high": [2020.0, 2015.0],  # Bar 1 high < Bar 0 high
            "low": [1980.0, 1990.0],   # Bar 1 low > Bar 0 low
            "close": [2010.0, 2008.0],
            "volume": [10.0, 10.0],
        })
        res = PriceActionFeatures.compute_all(inside_df)
        assert res["inside_bar"].iloc[1] == 1

    def test_streaks(self):
        # 3 consecutive bullish candles followed by 1 bearish
        streak_df = pd.DataFrame({
            "timestamp": [1000, 2000, 3000, 4000],
            "open": [2000.0, 2005.0, 2010.0, 2015.0],
            "high": [2006.0, 2011.0, 2016.0, 2017.0],
            "low": [1999.0, 2004.0, 2009.0, 2008.0],
            "close": [2005.0, 2010.0, 2015.0, 2010.0],  # Green, Green, Green, Red
            "volume": [10.0, 10.0, 10.0, 10.0],
        })
        res = PriceActionFeatures.compute_all(streak_df)
        assert res["streak_bullish"].tolist() == [1, 2, 3, 0]
        assert res["streak_bearish"].tolist() == [0, 0, 0, 1]
