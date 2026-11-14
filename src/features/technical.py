"""Technical indicator feature engineering for financial candlesticks.

All indicators are calculated sequentially with strictly backward-looking windows,
ensuring zero future information leakage or lookahead bias.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("features.technical")


class TechnicalFeatures:
    """Computes technical indicators and normalized momentum features on OHLCV DataFrames."""

    @staticmethod
    def add_emas(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Calculate Exponential Moving Averages and normalized price distances."""
        if periods is None:
            periods = [9, 20, 50, 100, 200]

        res = df.copy()
        close = res["close"]

        for p in periods:
            ema_series = close.ewm(span=p, adjust=False).mean()
            res[f"ema_{p}"] = ema_series
            # Relative percentage distance: (Close - EMA) / EMA
            res[f"ema_{p}_dist_pct"] = (close - ema_series) / ema_series
            # EMA slope over 3 bars
            res[f"ema_{p}_slope"] = (ema_series - ema_series.shift(3)) / ema_series.shift(3)

        # EMA Stack Alignment: Bullish if 9 > 20 > 50 > 200, Bearish if 9 < 20 < 50 < 200
        if all(p in periods for p in [9, 20, 50, 200]):
            bull_stack = (res["ema_9"] > res["ema_20"]) & (res["ema_20"] > res["ema_50"]) & (res["ema_50"] > res["ema_200"])
            bear_stack = (res["ema_9"] < res["ema_20"]) & (res["ema_20"] < res["ema_50"]) & (res["ema_50"] < res["ema_200"])
            res["ema_alignment"] = 0
            res.loc[bull_stack, "ema_alignment"] = 1
            res.loc[bear_stack, "ema_alignment"] = -1

        return res

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (Wilder's smoothing) and normalized ATR."""
        res = df.copy()
        high = res["high"]
        low = res["low"]
        close_prev = res["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res["true_range"] = true_range

        # Wilder's Smoothing: ewm with alpha=1/period
        atr = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
        res[f"atr_{period}"] = atr
        # ATR as a percentage of close price
        res[f"atr_{period}_pct"] = atr / res["close"]
        return res

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
        res = df.copy()
        delta = res["close"].diff()

        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder's smoothing
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        res[f"rsi_{period}"] = rsi.fillna(50.0)

        # Normalized RSI centered around 0 (-1.0 to +1.0)
        res[f"rsi_{period}_norm"] = (res[f"rsi_{period}"] - 50.0) / 50.0
        # Overbought (>70) and Oversold (<30) flags
        res[f"rsi_{period}_overbought"] = (res[f"rsi_{period}"] > 70).astype(int)
        res[f"rsi_{period}_oversold"] = (res[f"rsi_{period}"] < 30).astype(int)
        return res

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """Calculate Moving Average Convergence Divergence (MACD)."""
        res = df.copy()
        close = res["close"]

        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        res["macd_line"] = macd_line
        res["macd_signal"] = signal_line
        res["macd_hist"] = histogram
        # Normalized histogram by price
        res["macd_hist_pct"] = histogram / close
        # MACD histogram slope (momentum)
        res["macd_hist_slope"] = histogram.diff(2)
        return res

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands, %B, and Bandwidth."""
        res = df.copy()
        close = res["close"]

        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std(ddof=0)

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        res["bb_upper"] = upper
        res["bb_middle"] = middle
        res["bb_lower"] = lower

        # %B: Position within bands (0 = lower band, 0.5 = middle, 1 = upper band)
        band_span = (upper - lower).replace(0, np.nan)
        res["bb_pct_b"] = (close - lower) / band_span
        # Bandwidth: relative width normalized by middle
        res["bb_bandwidth"] = (upper - lower) / middle
        return res

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX) and +DI / -DI."""
        res = df.copy()
        high = res["high"]
        low = res["low"]
        close_prev = res["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        alpha = 1.0 / period
        atr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=res.index).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=res.index).ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100.0 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
        minus_di = 100.0 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        res[f"plus_di_{period}"] = plus_di.fillna(0.0)
        res[f"minus_di_{period}"] = minus_di.fillna(0.0)
        res[f"adx_{period}"] = adx.fillna(0.0)
        res[f"adx_{period}_trending"] = (res[f"adx_{period}"] > 25.0).astype(int)
        return res

    @staticmethod
    def add_stochastic_rsi(
        df: pd.DataFrame,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Calculate Stochastic RSI (%K and %D lines)."""
        res = df.copy()
        if f"rsi_{rsi_period}" not in res.columns:
            res = TechnicalFeatures.add_rsi(res, period=rsi_period)

        rsi_series = res[f"rsi_{rsi_period}"]
        rsi_min = rsi_series.rolling(window=stoch_period).min()
        rsi_max = rsi_series.rolling(window=stoch_period).max()

        stoch = (rsi_series - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
        k_line = stoch.rolling(window=k_period).mean() * 100.0
        d_line = k_line.rolling(window=d_period).mean()

        res["stoch_rsi_k"] = k_line.fillna(50.0)
        res["stoch_rsi_d"] = d_line.fillna(50.0)
        res["stoch_rsi_diff"] = res["stoch_rsi_k"] - res["stoch_rsi_d"]
        return res

    @staticmethod
    def add_momentum_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Rate of Change (ROC) and log returns across multiple lookbacks."""
        res = df.copy()
        close = res["close"]

        # Log returns
        res["log_ret_1"] = np.log(close / close.shift(1)).fillna(0.0)
        res["log_ret_3"] = np.log(close / close.shift(3)).fillna(0.0)
        res["log_ret_6"] = np.log(close / close.shift(6)).fillna(0.0)
        res["log_ret_12"] = np.log(close / close.shift(12)).fillna(0.0)

        # Rate of Change (ROC in %)
        for n in [5, 10, 20]:
            res[f"roc_{n}"] = ((close - close.shift(n)) / close.shift(n).replace(0, np.nan)) * 100.0
            res[f"roc_{n}"] = res[f"roc_{n}"].fillna(0.0)

        return res

    @staticmethod
    def add_volume_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Compute volume moving averages, ratios, and on-balance volume proxy."""
        res = df.copy()
        if "volume" not in res.columns:
            res["volume"] = 0.0

        vol = res["volume"]
        vol_sma = vol.rolling(window=period).mean()
        res[f"vol_sma_{period}"] = vol_sma
        res[f"vol_ratio_{period}"] = (vol / vol_sma.replace(0, np.nan)).fillna(1.0)

        # Price-Volume proxy: VWAP (rolling 50-bar)
        price_vol = (res["close"] * vol).rolling(window=50).sum()
        cum_vol = vol.rolling(window=50).sum()
        vwap = (price_vol / cum_vol.replace(0, np.nan)).fillna(res["close"])
        res["vwap_dist_pct"] = (res["close"] - vwap) / vwap

        return res

    @classmethod
    def compute_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the complete suite of technical indicators in a single forward pipeline."""
        res = cls.add_emas(df)
        res = cls.add_atr(res)
        res = cls.add_rsi(res)
        res = cls.add_macd(res)
        res = cls.add_bollinger_bands(res)
        res = cls.add_adx(res)
        res = cls.add_stochastic_rsi(res)
        res = cls.add_momentum_returns(res)
        res = cls.add_volume_features(res)
        return res
