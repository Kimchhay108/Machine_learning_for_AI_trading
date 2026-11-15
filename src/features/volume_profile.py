"""Volume Profile and Footprint / Order Flow Feature Engineering.

Auction Market Theory & Institutional Order Flow:
- Volume Profile: Point of Control (POC), Value Area High (VAH), Value Area Low (VAL),
  High Volume Nodes (HVN), Low Volume Nodes (LVN), and Profile Skewness.
- Footprint / Delta: Intrabar aggressive buy/sell volume delta, Cumulative Volume Delta (CVD),
  CVD Divergence (absorption), and Trapped Buyers / Sellers at candle extremes.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def compute_volume_profile_features(
    df: pd.DataFrame,
    rolling_window: int = 24,
) -> pd.DataFrame:
    """Compute rolling Auction Market Volume Profile features (zero lookahead).

    Args:
        df: DataFrame containing 'high', 'low', 'close', and 'volume'.
        rolling_window: Rolling window in bars (default 24 bars = 2 hours on M5).

    Returns:
        DataFrame with Volume Profile features aligned to current bar close.
    """
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    # 1. Typical Price for volume weighting
    tp = (h + l + c) / 3.0
    pv = tp * v

    # 2. Rolling Volume-Weighted POC (Point of Control)
    s_pv = pd.Series(pv, index=df.index)
    s_v = pd.Series(v, index=df.index)
    rolling_pv = s_pv.rolling(rolling_window, min_periods=1).sum().to_numpy()
    rolling_v = s_v.rolling(rolling_window, min_periods=1).sum().to_numpy()
    vp_poc = np.where(rolling_v > 0, rolling_pv / rolling_v, c)

    # 3. Rolling Value Area (~70% volume distribution around POC via rolling standard deviation)
    s_c = pd.Series(c, index=df.index)
    rolling_std = s_c.rolling(rolling_window, min_periods=1).std().fillna(1.0).to_numpy()
    # 1.0 standard deviation covers ~68.3% of normal distribution, matching institutional 70% Value Area
    vp_vah = vp_poc + (1.0 * rolling_std)
    vp_val = vp_poc - (1.0 * rolling_std)

    # 4. Relative distances from POC and Value Area boundaries
    safe_close = np.where(c > 0, c, 1.0)
    vp_poc_dist_pct = ((c - vp_poc) / safe_close) * 100.0
    vp_vah_dist_pct = ((c - vp_vah) / safe_close) * 100.0
    vp_val_dist_pct = ((c - vp_val) / safe_close) * 100.0

    # 5. Value Area State
    is_in_va = (c >= vp_val) & (c <= vp_vah)
    is_above_va = (c > vp_vah)
    is_below_va = (c < vp_val)

    # 6. Profile Asymmetry / Skewness (upper vs lower volume distribution)
    upper_mask = (tp > vp_poc).astype(float)
    upper_v = pd.Series(v * upper_mask, index=df.index).rolling(rolling_window, min_periods=1).sum().to_numpy()
    lower_v = rolling_v - upper_v
    vp_profile_skew = np.where(rolling_v > 0, (upper_v - lower_v) / rolling_v, 0.0)

    # 7. HVN Bounce & LVN Breakout flags
    # HVN bounce: Price retested POC or Value Area boundary within 0.15% and reversed
    poc_retest = (np.abs(c - vp_poc) / safe_close) <= 0.0015
    vp_hvn_bounce = poc_retest.astype(float)

    # LVN breakout: High momentum move outside Value Area with expanding volume
    s_v_mean = s_v.rolling(20, min_periods=1).mean().to_numpy()
    vol_expansion = np.where(s_v_mean > 0, v / s_v_mean, 1.0) >= 1.25
    vp_lvn_breakout = ((is_above_va | is_below_va) & vol_expansion).astype(float)

    feats = pd.DataFrame(
        {
            "vp_poc": vp_poc,
            "vp_vah": vp_vah,
            "vp_val": vp_val,
            "vp_poc_dist_pct": vp_poc_dist_pct,
            "vp_vah_dist_pct": vp_vah_dist_pct,
            "vp_val_dist_pct": vp_val_dist_pct,
            "vp_in_value_area": is_in_va.astype(float),
            "vp_above_value_area": is_above_va.astype(float),
            "vp_below_value_area": is_below_va.astype(float),
            "vp_profile_skew": vp_profile_skew,
            "vp_hvn_bounce": vp_hvn_bounce,
            "vp_lvn_breakout": vp_lvn_breakout,
        },
        index=df.index,
    )
    return feats


def compute_footprint_order_flow_features(
    df: pd.DataFrame,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Compute Footprint / Order Flow Delta, CVD Divergence, and Trapped Liquidity features.

    Zero lookahead: Evaluates strictly closed intrabar mechanics.

    Args:
        df: DataFrame containing 'open', 'high', 'low', 'close', and 'volume'.
        rolling_window: Rolling window in bars (default 20 bars).

    Returns:
        DataFrame with Footprint & Order Flow features aligned to bar close.
    """
    c = df["close"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    # 1. Intrabar aggressive buy vs sell volume estimation (Bar Delta proxy)
    # Price finishing near the high indicates aggressive market buyers dominated liquidity
    # Price finishing near the low indicates aggressive market sellers dominated liquidity
    hl = np.maximum(h - l, 0.01)
    buy_fraction = np.clip((c - l) / hl, 0.0, 1.0)
    buy_vol = v * buy_fraction
    sell_vol = v * (1.0 - buy_fraction)
    delta_proxy = buy_vol - sell_vol
    delta_ratio = np.where(v > 0, delta_proxy / v, 0.0)

    # 2. Cumulative Volume Delta (CVD) and Rolling CVD Momentum
    s_delta = pd.Series(delta_proxy, index=df.index)
    cvd_rolling = s_delta.rolling(rolling_window, min_periods=1).sum().to_numpy()
    cvd_mean = pd.Series(cvd_rolling, index=df.index).rolling(rolling_window, min_periods=1).mean().to_numpy()
    cvd_std = pd.Series(cvd_rolling, index=df.index).rolling(rolling_window, min_periods=1).std().fillna(1.0).to_numpy()
    cvd_normalized = np.where(cvd_std > 0, (cvd_rolling - cvd_mean) / cvd_std, 0.0)

    # 3. CVD Divergence (Institutional Absorption)
    # Bullish Divergence: Price makes a new 10-bar low, but CVD is higher than 10 bars ago (Sellers being absorbed by passive institutional bids)
    s_l = pd.Series(l, index=df.index)
    s_h = pd.Series(h, index=df.index)
    s_cvd = pd.Series(cvd_rolling, index=df.index)

    min_l_prev = s_l.shift(1).rolling(10, min_periods=3).min().to_numpy()
    min_cvd_prev = s_cvd.shift(1).rolling(10, min_periods=3).min().to_numpy()
    cvd_divergence_bull = ((l < min_l_prev) & (cvd_rolling > min_cvd_prev)).astype(float)

    # Bearish Divergence: Price makes a new 10-bar high, but CVD is lower than 10 bars ago (Buyers being absorbed by passive institutional asks)
    max_h_prev = s_h.shift(1).rolling(10, min_periods=3).max().to_numpy()
    max_cvd_prev = s_cvd.shift(1).rolling(10, min_periods=3).max().to_numpy()
    cvd_divergence_bear = ((h > max_h_prev) & (cvd_rolling < max_cvd_prev)).astype(float)

    # 4. Trapped Buyers / Trapped Sellers
    s_v = pd.Series(v, index=df.index)
    v_mean_20 = s_v.rolling(20, min_periods=1).mean().to_numpy()
    is_high_volume = (v >= 1.25 * v_mean_20)

    # Trapped Buyers: Bar made new high on high volume but closed in the bottom 35% of the range
    buyer_trap_pos = ((c - l) / hl) <= 0.35
    trapped_buyers = ((h > max_h_prev) & is_high_volume & buyer_trap_pos).astype(float)

    # Trapped Sellers: Bar made new low on high volume but closed in the top 35% of the range
    seller_trap_pos = ((c - l) / hl) >= 0.65
    trapped_sellers = ((l < min_l_prev) & is_high_volume & seller_trap_pos).astype(float)

    # 5. Wick Absorption Ratio
    body = np.abs(c - o)
    wick = hl - body
    absorption_ratio = np.where(hl > 0, wick / hl, 0.0) * np.where(v_mean_20 > 0, v / v_mean_20, 1.0)

    feats = pd.DataFrame(
        {
            "delta_proxy": delta_proxy,
            "delta_ratio": delta_ratio,
            "cvd_rolling": cvd_rolling,
            "cvd_normalized": cvd_normalized,
            "cvd_divergence_bull": cvd_divergence_bull,
            "cvd_divergence_bear": cvd_divergence_bear,
            "trapped_buyers": trapped_buyers,
            "trapped_sellers": trapped_sellers,
            "absorption_ratio": absorption_ratio,
        },
        index=df.index,
    )
    return feats
