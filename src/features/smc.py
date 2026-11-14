"""Smart Money Concepts (SMC) algorithmic detection engine.

Implements rigorous, mathematically objective detection of institutional market structure:
- Swing Highs / Lows (with explicit lag to eliminate lookahead bias)
- Market Structure Shifts: Break of Structure (BOS) and Change of Character (CHoCH)
- Buy-Side & Sell-Side Liquidity Sweeps
- Equal Highs (EQH) & Equal Lows (EQL) liquidity pools
- Fair Value Gaps (FVG) with dynamic mitigation tracking
- Order Blocks (OB) with displacement criteria
- Premium, Discount, and Optimal Trade Entry (OTE) dealing range zones
- Previous Day High/Low (PDH/PDL) and Previous Session Levels
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("features.smc")


class SmartMoneyConcepts:
    """Algorithmic Smart Money Concepts detector for XAUUSD."""

    @staticmethod
    def identify_swings(
        df: pd.DataFrame,
        lookback: int = 3,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Identify swing highs and swing lows without lookahead bias.

        A swing high at candle i requires i to be the local maximum across [i - lookback, i + lookback].
        Crucially, this swing is confirmed and available ONLY at candle i + lookback.
        At candle t, the most recently confirmed swing high is the one confirmed at or before t.

        Returns:
            Tuple of (swing_high_flag, swing_low_flag, active_swing_high_level, active_swing_low_level)
        """
        n = len(df)
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()

        swing_high_flag = np.zeros(n, dtype=int)
        swing_low_flag = np.zeros(n, dtype=int)
        active_sh = np.full(n, np.nan, dtype=float)
        active_sl = np.full(n, np.nan, dtype=float)

        curr_sh = np.nan
        curr_sl = np.nan

        # Confirmation happens at index t = i + lookback
        for t in range(2 * lookback, n):
            i = t - lookback
            # Check if index i was a swing high
            window_highs = highs[i - lookback : i + lookback + 1]
            if highs[i] == np.max(window_highs) and np.sum(window_highs == highs[i]) == 1:
                swing_high_flag[t] = 1
                curr_sh = highs[i]

            # Check if index i was a swing low
            window_lows = lows[i - lookback : i + lookback + 1]
            if lows[i] == np.min(window_lows) and np.sum(window_lows == lows[i]) == 1:
                swing_low_flag[t] = 1
                curr_sl = lows[i]

            active_sh[t] = curr_sh
            active_sl[t] = curr_sl

        # Forward fill active levels
        active_sh_series = pd.Series(active_sh, index=df.index).ffill()
        active_sl_series = pd.Series(active_sl, index=df.index).ffill()

        return (
            pd.Series(swing_high_flag, index=df.index),
            pd.Series(swing_low_flag, index=df.index),
            active_sh_series,
            active_sl_series,
        )

    @classmethod
    def add_market_structure(cls, df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
        """Calculate Swing Points, BOS (Break of Structure), and CHoCH (Change of Character)."""
        res = df.copy()
        sh_flag, sl_flag, active_sh, active_sl = cls.identify_swings(res, lookback=lookback)

        res["swing_high_confirmed"] = sh_flag
        res["swing_low_confirmed"] = sl_flag
        res["active_swing_high"] = active_sh
        res["active_swing_low"] = active_sl

        close = res["close"]
        high = res["high"]
        low = res["low"]

        # Trend Tracking state machine: +1 (bullish), -1 (bearish)
        n = len(res)
        trend = np.zeros(n, dtype=int)
        bos_bull = np.zeros(n, dtype=int)
        bos_bear = np.zeros(n, dtype=int)
        choch_bull = np.zeros(n, dtype=int)
        choch_bear = np.zeros(n, dtype=int)

        current_trend = 1  # default initial trend

        for t in range(1, n):
            c_val = close.iloc[t]
            c_prev = close.iloc[t - 1]
            sh_val = active_sh.iloc[t - 1]  # using prior candle's active level to avoid current bar leak
            sl_val = active_sl.iloc[t - 1]

            if not np.isnan(sh_val) and c_val > sh_val and c_prev <= sh_val:
                if current_trend == 1:
                    bos_bull[t] = 1
                else:
                    choch_bull[t] = 1
                    current_trend = 1

            elif not np.isnan(sl_val) and c_val < sl_val and c_prev >= sl_val:
                if current_trend == -1:
                    bos_bear[t] = 1
                else:
                    choch_bear[t] = 1
                    current_trend = -1

            trend[t] = current_trend

        res["market_structure_trend"] = trend
        res["bos_bullish"] = bos_bull
        res["bos_bearish"] = bos_bear
        res["choch_bullish"] = choch_bull
        res["choch_bearish"] = choch_bear
        return res

    @classmethod
    def add_liquidity_sweeps(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Detect institutional liquidity sweeps on active swing levels.

        Bullish Sweep: Low < Active Swing Low, but Close >= Active Swing Low (Stop hunt below support).
        Bearish Sweep: High > Active Swing High, but Close <= Active Swing High (Stop hunt above resistance).
        """
        res = df.copy()
        if "active_swing_high" not in res.columns:
            res = cls.add_market_structure(res)

        sh = res["active_swing_high"].shift(1)
        sl = res["active_swing_low"].shift(1)

        high = res["high"]
        low = res["low"]
        close = res["close"]

        # Sweeps
        bull_sweep = (low < sl) & (close >= sl)
        bear_sweep = (high > sh) & (close <= sh)

        res["liquidity_sweep_bullish"] = bull_sweep.astype(int)
        res["liquidity_sweep_bearish"] = bear_sweep.astype(int)

        # Equal Highs (EQH) and Equal Lows (EQL)
        # Distance between current active swing and prior swing <= 0.05%
        res["equal_highs"] = (
            (res["swing_high_confirmed"] == 1) &
            ((sh - res["high"].shift(1)).abs() / sh < 0.0005)
        ).astype(int)
        res["equal_lows"] = (
            (res["swing_low_confirmed"] == 1) &
            ((sl - res["low"].shift(1)).abs() / sl < 0.0005)
        ).astype(int)

        return res

    @staticmethod
    def add_fvg(df: pd.DataFrame) -> pd.DataFrame:
        """Identify Fair Value Gaps (FVG) and track real-time mitigation status."""
        res = df.copy()
        n = len(res)

        high = res["high"].to_numpy()
        low = res["low"].to_numpy()
        close = res["close"].to_numpy()

        fvg_bull_active = np.zeros(n, dtype=int)
        fvg_bear_active = np.zeros(n, dtype=int)
        fvg_bull_size = np.zeros(n, dtype=float)
        fvg_bear_size = np.zeros(n, dtype=float)

        # Maintain list of active unmitigated gaps: (top, bottom, is_bullish)
        active_bull_gaps: List[Tuple[float, float]] = []  # (top=low[i], bottom=high[i-2])
        active_bear_gaps: List[Tuple[float, float]] = []  # (top=low[i-2], bottom=high[i])

        for i in range(2, n):
            c_high = high[i]
            c_low = low[i]
            prev2_high = high[i - 2]
            prev2_low = low[i - 2]

            # 1. Check if new Bullish FVG formed at bar i (C1 High < C3 Low)
            if c_low > prev2_high:
                gap = c_low - prev2_high
                active_bull_gaps.append((c_low, prev2_high))
                fvg_bull_size[i] = gap

            # 2. Check if new Bearish FVG formed at bar i (C1 Low > C3 High)
            if c_high < prev2_low:
                gap = prev2_low - c_high
                active_bear_gaps.append((prev2_low, c_high))
                fvg_bear_size[i] = gap

            # 3. Update mitigation status for active gaps
            # Bullish gap is mitigated if subsequent low <= bottom of gap
            active_bull_gaps = [(top, bot) for (top, bot) in active_bull_gaps if c_low > bot]
            # Bearish gap is mitigated if subsequent high >= top of gap
            active_bear_gaps = [(top, bot) for (top, bot) in active_bear_gaps if c_high < top]

            fvg_bull_active[i] = len(active_bull_gaps)
            fvg_bear_active[i] = len(active_bear_gaps)

        res["fvg_bullish_new"] = (fvg_bull_size > 0).astype(int)
        res["fvg_bearish_new"] = (fvg_bear_size > 0).astype(int)
        res["fvg_bullish_unmitigated_count"] = fvg_bull_active
        res["fvg_bearish_unmitigated_count"] = fvg_bear_active
        return res

    @staticmethod
    def add_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
        """Detect Order Blocks (OB) based on directional displacement criteria."""
        res = df.copy()
        n = len(res)
        open_p = res["open"].to_numpy()
        close_p = res["close"].to_numpy()
        high_p = res["high"].to_numpy()
        low_p = res["low"].to_numpy()

        # ATR proxy or body size
        body = np.abs(close_p - open_p)
        avg_body = pd.Series(body).rolling(14, min_periods=1).mean().to_numpy()

        ob_bull = np.zeros(n, dtype=int)
        ob_bear = np.zeros(n, dtype=int)

        for i in range(1, n):
            # Bullish OB: Previous candle was bearish, current candle is strong bullish displacement
            if (close_p[i - 1] < open_p[i - 1]) and (close_p[i] > open_p[i]) and (body[i] > 1.3 * avg_body[i]):
                ob_bull[i] = 1

            # Bearish OB: Previous candle was bullish, current candle is strong bearish displacement
            if (close_p[i - 1] > open_p[i - 1]) and (close_p[i] < open_p[i]) and (body[i] > 1.3 * avg_body[i]):
                ob_bear[i] = 1

        res["order_block_bullish"] = ob_bull
        res["order_block_bearish"] = ob_bear
        return res

    @staticmethod
    def add_premium_discount(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Dealing Range equilibrium (50%) and Premium/Discount/OTE zones."""
        res = df.copy()
        if "active_swing_high" not in res.columns:
            res = SmartMoneyConcepts.add_market_structure(res)

        sh = res["active_swing_high"]
        sl = res["active_swing_low"]
        close = res["close"]

        dealing_range = (sh - sl).replace(0, np.nan)
        # Position in range: 0.0 at Swing Low, 1.0 at Swing High
        range_pos = (close - sl) / dealing_range

        res["dealing_range_pos"] = range_pos.clip(0.0, 1.0).fillna(0.5)
        res["is_discount_zone"] = (res["dealing_range_pos"] < 0.50).astype(int)
        res["is_premium_zone"] = (res["dealing_range_pos"] > 0.50).astype(int)
        # Optimal Trade Entry (OTE) between 61.8% and 78.6%
        res["is_ote_long_zone"] = (
            (res["dealing_range_pos"] >= 0.214) & (res["dealing_range_pos"] <= 0.382)
        ).astype(int)
        res["is_ote_short_zone"] = (
            (res["dealing_range_pos"] >= 0.618) & (res["dealing_range_pos"] <= 0.786)
        ).astype(int)
        return res

    @staticmethod
    def add_session_and_pdh_pdl(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Previous Day High (PDH), Previous Day Low (PDL), and Trading Session features."""
        res = df.copy()
        dt_utc = pd.to_datetime(res["timestamp"], unit="ms", utc=True)

        res["utc_hour"] = dt_utc.dt.hour
        res["utc_day"] = dt_utc.dt.date

        # Session Flags (UTC)
        # Asian Session: 00:00 - 09:00 UTC
        res["session_asian"] = ((res["utc_hour"] >= 0) & (res["utc_hour"] < 9)).astype(int)
        # London Session: 07:00 - 16:00 UTC
        res["session_london"] = ((res["utc_hour"] >= 7) & (res["utc_hour"] < 16)).astype(int)
        # New York Session: 12:00 - 21:00 UTC
        res["session_ny"] = ((res["utc_hour"] >= 12) & (res["utc_hour"] < 21)).astype(int)
        # London / NY Liquidity Overlap: 12:00 - 16:00 UTC
        res["session_london_ny_overlap"] = ((res["utc_hour"] >= 12) & (res["utc_hour"] < 16)).astype(int)

        # Previous Day High / Low
        # Group by date to get daily High / Low
        daily_agg = res.groupby("utc_day").agg(day_high=("high", "max"), day_low=("low", "min")).reset_index()
        daily_agg["pdh"] = daily_agg["day_high"].shift(1)
        daily_agg["pdl"] = daily_agg["day_low"].shift(1)

        res = res.merge(daily_agg[["utc_day", "pdh", "pdl"]], on="utc_day", how="left")
        res["pdh"] = res["pdh"].ffill()
        res["pdl"] = res["pdl"].ffill()

        # Swept PDH or PDL
        res["swept_pdh"] = ((res["high"] > res["pdh"]) & (res["close"] <= res["pdh"])).astype(int)
        res["swept_pdl"] = ((res["low"] < res["pdl"]) & (res["close"] >= res["pdl"])).astype(int)

        res = res.drop(columns=["utc_day"])
        return res

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all Smart Money Concepts features sequentially."""
        res = cls.add_market_structure(df)
        res = cls.add_liquidity_sweeps(res)
        res = cls.add_fvg(res)
        res = cls.add_order_blocks(res)
        res = cls.add_premium_discount(res)
        res = cls.add_session_and_pdh_pdl(res)
        return res
