"""Market regime identification and volatility state classification.

Categorizes market conditions into distinct quantitative regimes:
- Trend State: Strong Trend vs Ranging/Mean-Reverting (Efficiency Ratio & ADX)
- Volatility State: Low Volatility (Squeeze) vs Normal vs High Volatility (Expansion)
- Composite Regime Encoding (0 = Quiet Range, 1 = Bull Trend, 2 = Bear Trend, 3 = High Volatility Spike)
"""

import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("features.regime")


class MarketRegimeFeatures:
    """Computes regime features and statistical market condition classifications."""

    @staticmethod
    def add_trend_efficiency(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Kaufman's Efficiency Ratio (ER) and normalized linear regression slope."""
        res = df.copy()
        close = res["close"]

        # Net price change over period
        net_change = (close - close.shift(period)).abs()
        # Sum of absolute 1-bar price changes over period
        path_length = close.diff().abs().rolling(window=period).sum().replace(0, np.nan)

        # Efficiency Ratio: 1.0 = perfect direct line, 0.0 = pure noise/sideways
        efficiency_ratio = (net_change / path_length).fillna(0.0)
        res[f"efficiency_ratio_{period}"] = efficiency_ratio

        # Normalized 20-bar price slope: (Close - Close[t-20]) / (20 * ATR)
        atr_col = "atr_14" if "atr_14" in res.columns else None
        if atr_col:
            res[f"trend_slope_{period}"] = (close - close.shift(period)) / (period * res[atr_col].replace(0, np.nan))
        else:
            res[f"trend_slope_{period}"] = (close - close.shift(period)) / close.shift(period).replace(0, np.nan)

        return res

    @staticmethod
    def add_volatility_regime(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """Compute rolling ATR and Bollinger Bandwidth percentiles to establish volatility regimes."""
        res = df.copy()

        # ATR Percentile
        if "atr_14" in res.columns:
            atr_series = res["atr_14"]
            # Rolling percentile rank: 0.0 to 1.0
            res["atr_percentile"] = (
                atr_series.rolling(window=window, min_periods=20).rank(pct=True).fillna(0.5)
            )
            res["is_high_volatility"] = (res["atr_percentile"] >= 0.70).astype(int)
            res["is_low_volatility"] = (res["atr_percentile"] <= 0.30).astype(int)

        # Bollinger Band Squeeze (Contraction) vs Expansion
        if "bb_bandwidth" in res.columns:
            bbw = res["bb_bandwidth"]
            res["bbw_percentile"] = bbw.rolling(window=window, min_periods=20).rank(pct=True).fillna(0.5)
            # Squeeze: Bandwidth in bottom 15%
            res["volatility_squeeze"] = (res["bbw_percentile"] <= 0.15).astype(int)
            # Expansion: Bandwidth increasing and above median
            res["volatility_expansion"] = (
                (bbw > bbw.shift(3)) & (res["bbw_percentile"] >= 0.60)
            ).astype(int)

        return res

    @staticmethod
    def add_composite_regime(df: pd.DataFrame) -> pd.DataFrame:
        """Classify each candle into one of 4 discrete market regimes:

        0: Ranging / Consolidation
        1: Bullish Trend
        2: Bearish Trend
        3: Volatile Expansion / Spike
        """
        res = df.copy()

        er = res.get("efficiency_ratio_20", pd.Series(0.5, index=res.index))
        slope = res.get("trend_slope_20", pd.Series(0.0, index=res.index))
        is_high_vol = res.get("is_high_volatility", pd.Series(0, index=res.index))
        is_squeeze = res.get("volatility_squeeze", pd.Series(0, index=res.index))

        # Default: 0 = Ranging
        regime = np.zeros(len(res), dtype=int)

        # High Volatility Expansion
        high_vol_mask = (is_high_vol == 1) & (er < 0.35)
        # Bullish Trend: High efficiency and positive slope
        bull_trend_mask = (er >= 0.35) & (slope > 0.05)
        # Bearish Trend: High efficiency and negative slope
        bear_trend_mask = (er >= 0.35) & (slope < -0.05)

        regime[bull_trend_mask] = 1
        regime[bear_trend_mask] = 2
        regime[high_vol_mask] = 3

        res["market_regime"] = regime
        res["regime_label"] = "Ranging"
        res.loc[res["market_regime"] == 1, "regime_label"] = "Bullish_Trend"
        res.loc[res["market_regime"] == 2, "regime_label"] = "Bearish_Trend"
        res.loc[res["market_regime"] == 3, "regime_label"] = "High_Volatility_Expansion"

        return res

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute complete market regime features."""
        res = cls.add_trend_efficiency(df)
        res = cls.add_volatility_regime(res)
        res = cls.add_composite_regime(res)
        return res
