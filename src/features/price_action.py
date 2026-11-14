"""Price action feature extraction for candlestick patterns.

Computes mathematically precise single-candle and multi-candle price action dynamics:
- Bullish and Bearish Engulfing
- Pin Bars / Hammer / Shooting Star rejection candles
- Upper and Lower Rejection Wick Ratios
- Inside Bars and Outside Bars
- Rolling 20-bar Breakouts and Retests
- Candle Body-to-Range Ratios
- Consecutive Bullish and Bearish Candle Streaks
"""

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("features.price_action")


class PriceActionFeatures:
    """Computes price action and candlestick morphology indicators."""

    @staticmethod
    def add_candle_morphology(df: pd.DataFrame) -> pd.DataFrame:
        """Compute body size, wick sizes, and body-to-wick ratios."""
        res = df.copy()

        open_p = res["open"]
        high_p = res["high"]
        low_p = res["low"]
        close_p = res["close"]

        candle_range = (high_p - low_p).replace(0, np.nan)
        body = (close_p - open_p).abs()
        upper_wick = high_p - np.maximum(open_p, close_p)
        lower_wick = np.minimum(open_p, close_p) - low_p

        res["candle_range"] = candle_range.fillna(0.0)
        res["candle_body"] = body
        res["upper_wick"] = upper_wick
        res["lower_wick"] = lower_wick

        # Normalized ratios (0.0 to 1.0)
        res["body_to_range_ratio"] = (body / candle_range).fillna(0.0)
        res["upper_wick_ratio"] = (upper_wick / candle_range).fillna(0.0)
        res["lower_wick_ratio"] = (lower_wick / candle_range).fillna(0.0)

        # Bullish / Bearish candle sign
        res["is_bullish_candle"] = (close_p > open_p).astype(int)
        res["is_bearish_candle"] = (close_p < open_p).astype(int)

        return res

    @staticmethod
    def add_rejection_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect Pin Bars, Hammer, Shooting Star, and Engulfing patterns."""
        res = df.copy()
        if "upper_wick_ratio" not in res.columns:
            res = PriceActionFeatures.add_candle_morphology(res)

        # Pin Bar / Rejection Wick: Wick is >= 60% of total candle range and body is small (<= 30%)
        res["pin_bar_bullish"] = (
            (res["lower_wick_ratio"] >= 0.60) & (res["body_to_range_ratio"] <= 0.30)
        ).astype(int)
        res["pin_bar_bearish"] = (
            (res["upper_wick_ratio"] >= 0.60) & (res["body_to_range_ratio"] <= 0.30)
        ).astype(int)

        # Engulfing Patterns
        # Bullish Engulfing: previous candle was bearish, current candle open <= prev close, close > prev open
        prev_open = res["open"].shift(1)
        prev_close = res["close"].shift(1)

        bull_engulf = (
            (prev_close < prev_open) &
            (res["close"] > res["open"]) &
            (res["close"] >= prev_open) &
            (res["open"] <= prev_close)
        )

        bear_engulf = (
            (prev_close > prev_open) &
            (res["close"] < res["open"]) &
            (res["close"] <= prev_open) &
            (res["open"] >= prev_close)
        )

        res["engulfing_bullish"] = bull_engulf.astype(int)
        res["engulfing_bearish"] = bear_engulf.astype(int)

        # Inside Bar: High <= Prev High and Low >= Prev Low
        res["inside_bar"] = (
            (res["high"] <= res["high"].shift(1)) &
            (res["low"] >= res["low"].shift(1))
        ).astype(int)

        return res

    @staticmethod
    def add_breakouts_and_streaks(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detect multi-bar breakouts, retests, and consecutive directional streaks."""
        res = df.copy()
        close = res["close"]

        rolling_high = res["high"].shift(1).rolling(window=window).max()
        rolling_low = res["low"].shift(1).rolling(window=window).min()

        # Breakout of rolling 20-bar channel
        res[f"breakout_high_{window}"] = (close > rolling_high).astype(int)
        res[f"breakout_low_{window}"] = (close < rolling_low).astype(int)

        # Retest of broken high or low
        res[f"retest_high_{window}"] = (
            (res["low"] <= rolling_high) & (res["close"] >= rolling_high)
        ).astype(int)
        res[f"retest_low_{window}"] = (
            (res["high"] >= rolling_low) & (res["close"] <= rolling_low)
        ).astype(int)

        # Consecutive Bullish / Bearish Streaks
        is_bull = (res["close"] > res["open"]).astype(int).to_numpy()
        is_bear = (res["close"] < res["open"]).astype(int).to_numpy()

        n = len(res)
        bull_streak = np.zeros(n, dtype=int)
        bear_streak = np.zeros(n, dtype=int)

        curr_bull = 0
        curr_bear = 0

        for i in range(n):
            if is_bull[i]:
                curr_bull += 1
                curr_bear = 0
            elif is_bear[i]:
                curr_bear += 1
                curr_bull = 0
            else:
                curr_bull = 0
                curr_bear = 0
            bull_streak[i] = curr_bull
            bear_streak[i] = curr_bear

        res["streak_bullish"] = bull_streak
        res["streak_bearish"] = bear_streak
        return res

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all price action features in order."""
        res = cls.add_candle_morphology(df)
        res = cls.add_rejection_patterns(res)
        res = cls.add_breakouts_and_streaks(res)
        return res
