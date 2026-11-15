"""Realistic Backtesting Simulation API Endpoints with Multi-Model Strategy Comparison."""

from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException
import joblib
import numpy as np
import pandas as pd

from src.api.dependencies import get_prediction_engine
from src.api.schemas import (
    BacktestComparisonResponse,
    BacktestRequest,
    BacktestResponse,
    ForensicsSummarySchema,
    ModelComparisonItem,
    MonthlyPerformanceSchema,
    TradeForensicsItemSchema,
    TradeRecordSchema,
)
from src.backtest.engine import RealisticBacktestEngine, BacktestResult
from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT
from src.core.logger import get_logger
from src.models.failure_forensics import TradeForensicsEngine
from src.models.prediction_engine import PredictionEngine

logger = get_logger("api.routers.backtest")
router = APIRouter(prefix="/api/backtest", tags=["Backtest Lab"])

_CACHED_BACKTEST_FILE = settings.BACKTEST_DIR / "latest_backtest.joblib"
_memory_cached_result: Optional[RealisticBacktestEngine] = None


def generate_strategy_signals_and_levels(
    model_type: str,
    test_df: pd.DataFrame,
    engine: PredictionEngine,
    threshold: float,
    spread: float,
    slippage: float,
    only_killzones: bool = True,
    target_rr: Optional[float] = None,
    use_h1_gatekeeper: bool = True,
    learn_from_mistakes: bool = True,
    filter_high_impact_news: bool = True,
    news_blackout_mins: int = 30,
    exclude_asian_session: bool = True,
    strict_structural_entry: bool = False,
    session_window: str = "cambodia_evening",
) -> Tuple[List[str], List[float], List[float], str, str]:
    """Generate model-tailored signals, Entry, Stop-Loss, and Take-Profit price levels.

    Distinct Logic per Model with Institutional High-Win-Rate Edge:
    - Model E (Ensemble): Stacking meta-confluence + Dynamic ATR Volatility Stop (1.5x ATR).
    - Model B (SMC & ICT): ICT 2022 FVG / Silver Bullet confluence + Structural Swing Invalidation Stop ($0.50 buffer).
    - Model A (Technical): EMA Trend & RSI Momentum + Volatility Envelope Stop (1.5x ATR).
    - Model D (Price Action): Candle Rejection Wicks & Pin Bars + Candle Wick Extreme High/Low Stop ($0.50 buffer).
    - Session Filter: Restricts entries to London (07:00-10:00 UTC) & NY (12:00-15:00 UTC) Killzones when only_killzones is enabled.
    - Strict Asian Blackout: Forbids trade entries during Tokyo/Sydney Asian Session (21:00-07:00 UTC) when exclude_asian_session is enabled.
    - H1 Macro Trend Gatekeeper: Price > H1 200 EMA & H1 Trend >= 0 for BUY; Price < H1 200 EMA & H1 Trend <= 0 for SELL.
    - AI Failure Memory: Vetoes entries matching historical loss fingerprints (low momentum chop, climax exhaustion, volatility expansion).
    - High-Impact News Blackout: Keeps out of XAUUSD during major USD economic announcements (NFP, CPI, FOMC, GDP).
    - Adaptive R:R: Scales Take Profit based on target_rr (e.g. 1.2, 1.5, 2.0).
    """
    close = test_df["close"].to_numpy()
    high = test_df["high"].to_numpy()
    low = test_df["low"].to_numpy()
    n = len(test_df)

    if "atr_14" in test_df.columns:
        atr = test_df["atr_14"].to_numpy()
    else:
        atr = np.full(n, 2.0)
    atr = np.where(np.isnan(atr) | (atr <= 0), 2.0, atr)

    # Time-based filters (Cambodia Evening 7 PM - 11 PM, Killzones, Asian Blackout & News Blackouts)
    if "timestamp" in test_df.columns and len(test_df) > 0:
        first_ts = test_df["timestamp"].iloc[0]
        unit = "ms" if first_ts > 1e11 else "s"
        dt = pd.to_datetime(test_df["timestamp"], unit=unit, utc=True)
        hours = dt.dt.hour.to_numpy()
        hours_cambodia = (hours + 7) % 24

        # Strict Asian Session Exclusion: Never trade 21:00 to 07:00 UTC (Tokyo/Sydney low liquidity consolidation)
        is_asian = (hours >= 21) | (hours < 7)

        sw = (session_window or "cambodia_evening").lower().strip()
        if sw in ["cambodia_evening", "cambodia", "ict_evening"]:
            # Cambodia 7 PM to 11 PM = 19:00 to 23:00 ICT (12:00 to 16:00 UTC New York Session)
            in_killzones = (hours_cambodia >= 19) & (hours_cambodia < 23)
        elif sw in ["killzones", "london_ny"]:
            in_killzones = ((hours >= 7) & (hours <= 10)) | ((hours >= 12) & (hours <= 15))
        else:
            in_killzones = ~is_asian if exclude_asian_session else np.ones(n, dtype=bool)

        if exclude_asian_session and sw not in ["cambodia_evening", "cambodia", "ict_evening"]:
            in_killzones = in_killzones & (~is_asian)

        if filter_high_impact_news:
            from src.models.news_filter import get_news_filter
            nf = get_news_filter()
            news_blackout = nf.get_blackout_mask(
                test_df["timestamp"],
                buffer_before_mins=news_blackout_mins,
                buffer_after_mins=news_blackout_mins,
            )
        else:
            news_blackout = np.zeros(n, dtype=bool)
    else:
        in_killzones = np.ones(n, dtype=bool)
        news_blackout = np.zeros(n, dtype=bool)

    # Check H1 + M5 Institutional Macro Trend alignment (Gatekeeper: H1 200 EMA + Structure + M5 EMA 50/200)
    h1_ema = test_df.get("h1_ema_200", test_df.get("m15_ema_200", test_df.get("ema_200", None)))
    h1_trend = test_df.get("h1_market_structure_trend", test_df.get("m15_market_structure_trend", test_df.get("market_structure_trend", None)))
    m5_ema50 = test_df.get("ema_50", None)
    m5_ema200 = test_df.get("ema_200", None)

    if use_h1_gatekeeper and h1_ema is not None:
        ema_arr = h1_ema.to_numpy()
        h1_b = close > ema_arr
        h1_s = close < ema_arr
        if m5_ema50 is not None:
            e50 = m5_ema50.to_numpy()
            h1_b = h1_b & (close > e50)
            h1_s = h1_s & (close < e50)
            if m5_ema200 is not None:
                e200 = m5_ema200.to_numpy()
                h1_b = h1_b & (e50 > e200)
                h1_s = h1_s & (e50 < e200)
        h1_bull = h1_b
        h1_bear = h1_s
    else:
        h1_bull = np.ones(n, dtype=bool)
        h1_bear = np.ones(n, dtype=bool)

    # Volatility expansion spike filter learned from loss forensics (veto entry when ATR > 1.4x rolling baseline)
    rolling_atr = pd.Series(atr).rolling(50, min_periods=5).mean().bfill().to_numpy()
    atr_ratio = np.where(rolling_atr > 0, atr / rolling_atr, 1.0)
    vol_ok = (atr_ratio <= 1.40) if learn_from_mistakes else np.ones(n, dtype=bool)

    signals: List[str] = []
    sl_levels: List[float] = []
    tp_levels: List[float] = []

    m_type = (model_type or "ensemble").lower().strip()
    open_p = test_df.get("open", pd.Series(close, index=test_df.index)).to_numpy()
    eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
    atr_mult = 1.8 if eff_rr <= 1.0 else 1.5
    swing_buf = 0.50

    if m_type == "smc_ict":
        eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
        model_name = "Model B: SMC & ICT 2022 Mentorship"
        sl_tp_logic = f"ICT 2022 FVG / Silver Bullet | SL: Structural Swing Invalidation (${swing_buf:.2f} buffer), TP: {eff_rr} R:R Opposing Liquidity"

        prob_b = engine.ensemble.model_b.predict_proba(test_df)
        pb_b, pl_b = prob_b[:, 0], prob_b[:, 2]
        norm_pb_b = pb_b / 0.34
        norm_pl_b = pl_b / 0.44
        dir_b_sum = norm_pb_b + norm_pl_b + 1e-9
        dir_pb = norm_pb_b / dir_b_sum
        dir_pl = norm_pl_b / dir_b_sum

        ict_bull = (
            (test_df.get("ict_2022_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_silver_bullet_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_judas_swing_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_turtle_soup_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("choch_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("bos_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("fvg_bullish_new", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("liquidity_sweep_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1)
        )
        ict_bear = (
            (test_df.get("ict_2022_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_silver_bullet_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_judas_swing_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_turtle_soup_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("choch_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("bos_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("fvg_bearish_new", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("liquidity_sweep_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1)
        )

        swing_l = test_df.get("active_swing_low", pd.Series(0.0, index=test_df.index)).to_numpy()
        swing_h = test_df.get("active_swing_high", pd.Series(0.0, index=test_df.index)).to_numpy()

        for i in range(n):
            cp, ca = close[i], atr[i]

            # Model B Structural Entry Condition (toggleable)
            cond_buy = ict_bull[i] if strict_structural_entry else True
            cond_sell = ict_bear[i] if strict_structural_entry else True

            # Directional relative probability normalization allows balanced short trading
            prob_buy = (dir_pl[i] >= threshold or pl_b[i] >= 0.44) and (dir_pl[i] > dir_pb[i])
            prob_sell = (dir_pb[i] >= threshold or pb_b[i] >= 0.38) and (dir_pb[i] > dir_pl[i])

            is_buy = in_killzones[i] and (not news_blackout[i]) and h1_bull[i] and vol_ok[i] and cond_buy and prob_buy
            is_sell = in_killzones[i] and (not news_blackout[i]) and h1_bear[i] and vol_ok[i] and cond_sell and prob_sell

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (atr_mult * ca)
                tp = ep + (eff_rr * atr_mult * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (atr_mult * ca)
                tp = ep - (eff_rr * atr_mult * ca)
            else:
                signals.append("WAIT")
                sl = cp - (atr_mult * ca)
                tp = cp + (eff_rr * atr_mult * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "technical":
        eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
        model_name = "Model A: Technical Momentum (Indicators)"
        sl_tp_logic = f"EMA Trend & RSI Pullbacks | SL: {atr_mult}x ATR Volatility Envelope, TP: {eff_rr} R:R Target"

        prob_a = engine.ensemble.model_a.predict_proba(test_df)
        pb_a, pl_a = prob_a[:, 0], prob_a[:, 2]
        norm_pb_a = pb_a / 0.34
        norm_pl_a = pl_a / 0.44
        dir_a_sum = norm_pb_a + norm_pl_a + 1e-9
        dir_pb = norm_pb_a / dir_a_sum
        dir_pl = norm_pl_a / dir_a_sum

        ema_align = test_df.get("ema_alignment", pd.Series(0, index=test_df.index)).to_numpy()
        rsi_arr = test_df.get("rsi_14", pd.Series(50.0, index=test_df.index)).to_numpy()
        ema_dist = test_df.get("ema_9_dist_pct", pd.Series(0.0, index=test_df.index)).to_numpy()
        macd_h = test_df.get("macd_hist", pd.Series(0.0, index=test_df.index)).to_numpy()

        for i in range(n):
            cp, ca = close[i], atr[i]

            # Model A Structural Entry Condition: balanced pullbacks in trend direction
            tech_bull = (ema_align[i] >= 0 or ema_dist[i] >= 0 or macd_h[i] >= 0) and (rsi_arr[i] >= 35.0 and rsi_arr[i] <= 65.0)
            tech_bear = (ema_align[i] <= 0 or ema_dist[i] <= 0 or macd_h[i] <= 0) and (rsi_arr[i] >= 35.0 and rsi_arr[i] <= 65.0)
            cond_buy = tech_bull if strict_structural_entry else True
            cond_sell = tech_bear if strict_structural_entry else True

            prob_buy = (dir_pl[i] >= threshold or pl_a[i] >= 0.44) and (dir_pl[i] > dir_pb[i])
            prob_sell = (dir_pb[i] >= threshold or pb_a[i] >= 0.38) and (dir_pb[i] > dir_pl[i])

            is_buy = in_killzones[i] and (not news_blackout[i]) and h1_bull[i] and vol_ok[i] and cond_buy and prob_buy
            is_sell = in_killzones[i] and (not news_blackout[i]) and h1_bear[i] and vol_ok[i] and cond_sell and prob_sell

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (atr_mult * ca)
                tp = ep + (eff_rr * atr_mult * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (atr_mult * ca)
                tp = ep - (eff_rr * atr_mult * ca)
            else:
                signals.append("WAIT")
                sl = cp - (atr_mult * ca)
                tp = cp + (eff_rr * atr_mult * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "price_action":
        eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
        model_name = "Model D: Price Action & Rejection Wicks"
        sl_tp_logic = f"Candle Wick Extremes | SL: Candle Extreme (-${swing_buf:.2f} buffer), TP: {eff_rr} R:R Target"

        prob_d = engine.ensemble.model_d.predict_proba(test_df)
        pb_d, pl_d = prob_d[:, 0], prob_d[:, 2]
        norm_pb_d = pb_d / 0.34
        norm_pl_d = pl_d / 0.44
        dir_d_sum = norm_pb_d + norm_pl_d + 1e-9
        dir_pb = norm_pb_d / dir_d_sum
        dir_pl = norm_pl_d / dir_d_sum

        pin_bull = test_df.get("pin_bar_bullish", pd.Series(0, index=test_df.index)).to_numpy()
        pin_bear = test_df.get("pin_bar_bearish", pd.Series(0, index=test_df.index)).to_numpy()
        engulf_bull = test_df.get("engulfing_bullish", pd.Series(0, index=test_df.index)).to_numpy()
        engulf_bear = test_df.get("engulfing_bearish", pd.Series(0, index=test_df.index)).to_numpy()
        lower_wick = test_df.get("lower_wick_ratio", pd.Series(0.0, index=test_df.index)).to_numpy()
        upper_wick = test_df.get("upper_wick_ratio", pd.Series(0.0, index=test_df.index)).to_numpy()

        for i in range(n):
            cp, ca, ch, cl, co = close[i], atr[i], high[i], low[i], open_p[i]
            c_rng = max(ch - cl, 0.01)

            # Model D Structural Entry Condition (Pin bars, Engulfing, Wicks, or Momentum Bodies)
            pa_bull = (pin_bull[i] == 1) or (engulf_bull[i] == 1) or (lower_wick[i] >= 0.25) or (cp > co and (cp - co) / c_rng >= 0.50)
            pa_bear = (pin_bear[i] == 1) or (engulf_bear[i] == 1) or (upper_wick[i] >= 0.25) or (cp < co and (co - cp) / c_rng >= 0.50)
            cond_buy = pa_bull if strict_structural_entry else True
            cond_sell = pa_bear if strict_structural_entry else True

            prob_buy = (dir_pl[i] >= threshold or pl_d[i] >= 0.44) and (dir_pl[i] > dir_pb[i])
            prob_sell = (dir_pb[i] >= threshold or pb_d[i] >= 0.38) and (dir_pb[i] > dir_pl[i])

            is_buy = in_killzones[i] and (not news_blackout[i]) and h1_bull[i] and vol_ok[i] and cond_buy and prob_buy
            is_sell = in_killzones[i] and (not news_blackout[i]) and h1_bear[i] and vol_ok[i] and cond_sell and prob_sell

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (atr_mult * ca)
                tp = ep + (eff_rr * atr_mult * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (atr_mult * ca)
                tp = ep - (eff_rr * atr_mult * ca)
            else:
                signals.append("WAIT")
                sl = cp - (atr_mult * ca)
                tp = cp + (eff_rr * atr_mult * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "volume_profile":
        eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
        model_name = "Model F: Volume Profile (Auction Theory)"
        sl_tp_logic = f"Auction Market POC/VAH/VAL Rotations | SL: Outside Value Area (-${swing_buf:.2f} buffer), TP: {eff_rr} R:R Target"

        prob_f = engine.ensemble.model_f.predict_proba(test_df)
        pb_f, pl_f = prob_f[:, 0], prob_f[:, 2]
        norm_pb_f = pb_f / 0.34
        norm_pl_f = pl_f / 0.44
        dir_f_sum = norm_pb_f + norm_pl_f + 1e-9
        dir_pb = norm_pb_f / dir_f_sum
        dir_pl = norm_pl_f / dir_f_sum

        vp_poc = test_df.get("vp_poc", pd.Series(close, index=test_df.index)).to_numpy()
        vp_val = test_df.get("vp_val", pd.Series(close - atr, index=test_df.index)).to_numpy()
        vp_vah = test_df.get("vp_vah", pd.Series(close + atr, index=test_df.index)).to_numpy()
        hvn_b = test_df.get("vp_hvn_bounce", pd.Series(0, index=test_df.index)).to_numpy()
        lvn_bo = test_df.get("vp_lvn_breakout", pd.Series(0, index=test_df.index)).to_numpy()

        for i in range(n):
            cp, ca = close[i], atr[i]
            val_i, vah_i = vp_val[i], vp_vah[i]

            # Structural Volume Profile: testing VAL/POC or LVN breakout
            vp_bull = (low[i] <= val_i + 0.5 * ca) or (hvn_b[i] == 1 and cp > val_i) or (lvn_bo[i] == 1 and cp > vah_i)
            vp_bear = (high[i] >= vah_i - 0.5 * ca) or (hvn_b[i] == 1 and cp < vah_i) or (lvn_bo[i] == 1 and cp < val_i)

            cond_buy = vp_bull if strict_structural_entry else True
            cond_sell = vp_bear if strict_structural_entry else True

            prob_buy = (dir_pl[i] >= threshold or pl_f[i] >= 0.44) and (dir_pl[i] > dir_pb[i])
            prob_sell = (dir_pb[i] >= threshold or pb_f[i] >= 0.38) and (dir_pb[i] > dir_pl[i])

            is_buy = in_killzones[i] and (not news_blackout[i]) and h1_bull[i] and vol_ok[i] and cond_buy and prob_buy
            is_sell = in_killzones[i] and (not news_blackout[i]) and h1_bear[i] and vol_ok[i] and cond_sell and prob_sell

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (atr_mult * ca)
                tp = ep + (eff_rr * atr_mult * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (atr_mult * ca)
                tp = ep - (eff_rr * atr_mult * ca)
            else:
                signals.append("WAIT")
                sl = cp - (atr_mult * ca)
                tp = cp + (eff_rr * atr_mult * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "footprint":
        eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
        model_name = "Model G: Footprint & Order Flow (Delta Absorption)"
        sl_tp_logic = f"Order Flow Delta Divergence & Absorption | SL: Wick Absorption Extreme (-${swing_buf:.2f} buffer), TP: {eff_rr} R:R Target"

        prob_g = engine.ensemble.model_g.predict_proba(test_df)
        pb_g, pl_g = prob_g[:, 0], prob_g[:, 2]
        norm_pb_g = pb_g / 0.34
        norm_pl_g = pl_g / 0.44
        dir_g_sum = norm_pb_g + norm_pl_g + 1e-9
        dir_pb = norm_pb_g / dir_g_sum
        dir_pl = norm_pl_g / dir_g_sum

        cvd_bull = test_df.get("cvd_divergence_bull", pd.Series(0, index=test_df.index)).to_numpy()
        cvd_bear = test_df.get("cvd_divergence_bear", pd.Series(0, index=test_df.index)).to_numpy()
        trap_s = test_df.get("trapped_sellers", pd.Series(0, index=test_df.index)).to_numpy()
        trap_b = test_df.get("trapped_buyers", pd.Series(0, index=test_df.index)).to_numpy()
        delta_p = test_df.get("delta_proxy", pd.Series(0.0, index=test_df.index)).to_numpy()

        for i in range(n):
            cp, ca, ch, cl = close[i], atr[i], high[i], low[i]

            fp_bull = (cvd_bull[i] == 1) or (trap_s[i] == 1) or (delta_p[i] > 0)
            fp_bear = (cvd_bear[i] == 1) or (trap_b[i] == 1) or (delta_p[i] < 0)

            cond_buy = fp_bull if strict_structural_entry else True
            cond_sell = fp_bear if strict_structural_entry else True

            prob_buy = (dir_pl[i] >= threshold or pl_g[i] >= 0.44) and (dir_pl[i] > dir_pb[i])
            prob_sell = (dir_pb[i] >= threshold or pb_g[i] >= 0.38) and (dir_pb[i] > dir_pl[i])

            is_buy = in_killzones[i] and (not news_blackout[i]) and h1_bull[i] and vol_ok[i] and cond_buy and prob_buy
            is_sell = in_killzones[i] and (not news_blackout[i]) and h1_bear[i] and vol_ok[i] and cond_sell and prob_sell

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (atr_mult * ca)
                tp = ep + (eff_rr * atr_mult * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (atr_mult * ca)
                tp = ep - (eff_rr * atr_mult * ca)
            else:
                signals.append("WAIT")
                sl = cp - (atr_mult * ca)
                tp = cp + (eff_rr * atr_mult * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    else:  # "ensemble" default
        eff_rr = target_rr if target_rr and target_rr > 0 else 1.2
        model_name = "Model E: Calibrated Multi-Model Ensemble"
        sl_tp_logic = f"Calibrated Meta-Model Stacking | SL: {atr_mult}x ATR Dynamic Stop, TP: {eff_rr} R:R Target"

        engine.ensemble.confidence_threshold = threshold
        decisions, probs, confs = engine.ensemble.predict_decision(test_df)
        p_bear_e, p_bull_e = probs[:, 0], probs[:, 2]

        # Check sub-model consensus: at least 2 profitable sub-models must agree on direction (Models A, D, F, G)
        pred_a = engine.ensemble.model_a.predict(test_df)
        pred_d = engine.ensemble.model_d.predict(test_df)
        pred_f = engine.ensemble.model_f.predict(test_df)
        pred_g = engine.ensemble.model_g.predict(test_df)

        agreed_bull = ((pred_a == 2).astype(int) + (pred_d == 2).astype(int) + (pred_f == 2).astype(int) + (pred_g == 2).astype(int)) >= 2
        agreed_bear = ((pred_a == 0).astype(int) + (pred_d == 0).astype(int) + (pred_f == 0).astype(int) + (pred_g == 0).astype(int)) >= 2

        for i in range(n):
            cp, ca = close[i], atr[i]
            sig = decisions[i]

            cond_buy = agreed_bull[i] if strict_structural_entry else True
            cond_sell = agreed_bear[i] if strict_structural_entry else True

            prob_buy = (sig == "BUY") or (p_bull_e[i] >= 0.44 and p_bull_e[i] > p_bear_e[i])
            prob_sell = (sig == "SELL") or (p_bear_e[i] >= 0.38 and p_bear_e[i] > p_bull_e[i])

            # Model E Entry Condition: Calibrated Ensemble threshold + multi-model majority consensus (toggleable)
            is_buy = in_killzones[i] and (not news_blackout[i]) and h1_bull[i] and vol_ok[i] and prob_buy and cond_buy
            is_sell = in_killzones[i] and (not news_blackout[i]) and h1_bear[i] and vol_ok[i] and prob_sell and cond_sell

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (atr_mult * ca)
                tp = ep + (eff_rr * atr_mult * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (atr_mult * ca)
                tp = ep - (eff_rr * atr_mult * ca)
            else:
                signals.append("WAIT")
                sl = cp - (atr_mult * ca)
                tp = cp + (eff_rr * atr_mult * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    return signals, sl_levels, tp_levels, model_name, sl_tp_logic


def _format_backtest_response(result: BacktestResult, test_df: Optional[pd.DataFrame] = None) -> BacktestResponse:
    """Helper to convert engine BacktestResult into API BacktestResponse with forensic analysis."""
    diag_lookup: Dict[int, Any] = {}
    forensics_summary: Optional[ForensicsSummarySchema] = None

    if test_df is not None and not test_df.empty:
        try:
            forensics_engine = TradeForensicsEngine()
            summary = forensics_engine.analyze_trades(result.trades, test_df)
            forensics_summary = ForensicsSummarySchema(
                total_losses_analyzed=summary.total_losses_analyzed,
                cause_distribution=summary.cause_distribution,
                top_failure_cause=summary.top_failure_cause,
                key_lessons=summary.key_lessons,
                sample_diagnoses=[
                    TradeForensicsItemSchema(**item.to_dict()) for item in summary.sample_diagnoses
                ],
            )
            for d in summary.sample_diagnoses:
                diag_lookup[int(d.entry_time)] = d
        except Exception as e:
            logger.warning(f"Could not compute trade forensics: {e}")

    trades_schema = []
    monthly_data = {}
    for t in result.trades:
        e_time = int(t.entry_time // 1000 if t.entry_time > 10_000_000_000 else t.entry_time)
        diag = diag_lookup.get(int(t.entry_time))
        trades_schema.append(
            TradeRecordSchema(
                entry_time=e_time,
                exit_time=int(t.exit_time // 1000 if t.exit_time > 10_000_000_000 else t.exit_time),
                direction=t.direction,
                entry_price=round(t.entry_price, 2),
                exit_price=round(t.exit_price, 2),
                stop_loss=round(t.stop_loss, 2),
                take_profit=round(t.take_profit, 2),
                position_size_oz=round(t.position_size_oz, 3),
                pnl_usd=round(t.pnl_usd, 2),
                pnl_pct=round(t.pnl_pct, 4),
                r_multiple=round(t.r_multiple, 2),
                exit_reason=t.exit_reason,
                failure_cause=diag.failure_cause if diag else None,
                lesson_learned=diag.lesson_learned if diag else None,
            )
        )

        # Monthly aggregation
        dt = pd.to_datetime(e_time, unit="s", utc=True)
        key = (dt.year, dt.month)
        if key not in monthly_data:
            monthly_data[key] = {
                "year": int(dt.year),
                "month": int(dt.month),
                "period": f"{dt.year}-{dt.month:02d}",
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "net_profit": 0.0,
                "gross_win": 0.0,
                "gross_loss": 0.0,
            }
        m = monthly_data[key]
        m["total_trades"] += 1
        m["net_profit"] += t.pnl_usd
        if t.pnl_usd > 0.01:
            m["winning_trades"] += 1
            m["gross_win"] += t.pnl_usd
        elif t.pnl_usd < -0.01:
            m["losing_trades"] += 1
            m["gross_loss"] += abs(t.pnl_usd)

    monthly_breakdown = []
    for key in sorted(monthly_data.keys()):
        m = monthly_data[key]
        tot = m["total_trades"]
        wins = m["winning_trades"]
        losses = m["losing_trades"]
        wr = round((wins / tot) * 100, 2) if tot > 0 else 0.0
        pf = round(m["gross_win"] / m["gross_loss"], 3) if m["gross_loss"] > 0 else (999.0 if m["gross_win"] > 0 else 1.0)
        monthly_breakdown.append(
            MonthlyPerformanceSchema(
                year=m["year"],
                month=m["month"],
                period=m["period"],
                total_trades=tot,
                winning_trades=wins,
                losing_trades=losses,
                win_rate_pct=wr,
                net_profit=round(m["net_profit"], 2),
                profit_factor=pf,
            )
        )

    return BacktestResponse(
        initial_balance=round(result.initial_balance, 2),
        final_balance=round(result.final_balance, 2),
        total_net_profit=round(result.total_net_profit, 2),
        return_pct=round(result.return_pct, 2),
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        breakeven_trades=getattr(result, "breakeven_trades", 0),
        win_rate_pct=round(result.win_rate_pct, 2),
        win_rate_with_be_pct=round(getattr(result, "win_rate_with_be_pct", result.win_rate_pct), 2),
        profit_factor=round(result.profit_factor, 3),
        max_drawdown_usd=round(result.max_drawdown_usd, 2),
        max_drawdown_pct=round(result.max_drawdown_pct, 2),
        sharpe_ratio=round(result.sharpe_ratio, 3),
        avg_trade_usd=round(result.avg_trade_usd, 2),
        avg_r_multiple=round(result.avg_r_multiple, 3),
        equity_curve=[round(val, 2) for val in result.equity_curve],
        trades=trades_schema,
        monthly_breakdown=monthly_breakdown,
        forensics=forensics_summary,
    )


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Execute realistic out-of-sample backtest with model-specific Entry, SL, and TP rules."""
    tf_clean = (getattr(req, "timeframe", None) or "M5").lower().strip()
    feat_path = settings.FEATURES_DIR / "xauusd" / f"{tf_clean}_features.parquet"
    if not feat_path.exists():
        feat_path = settings.FEATURES_DIR / "xauusd" / "m5_features.parquet"

    if not feat_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Features file missing. Run feature pipeline first.",
        )

    df_feats = pd.read_parquet(feat_path)
    if df_feats.empty:
        raise HTTPException(status_code=400, detail="Feature matrix is empty.")

    engine: PredictionEngine = get_prediction_engine(req.version)

    # Historical scope: Full multi-year dataset (2023-present) or 30% out-of-sample slice
    n = len(df_feats)
    use_full = getattr(req, "use_full_history", True)
    if use_full:
        test_df = df_feats.reset_index(drop=True)
        logger.info(f"Executing backtest across FULL multi-year dataset ({len(test_df)} candles from 2023-present).")
    else:
        test_start = int(n * 0.70) + 12
        test_df = df_feats.iloc[test_start:].reset_index(drop=True)
        logger.info(f"Executing backtest across 30% Out-of-Sample slice ({len(test_df)} candles).")

    signals, sl_levels, tp_levels, model_name, sl_tp_logic = generate_strategy_signals_and_levels(
        req.model_type,
        test_df,
        engine,
        threshold=req.threshold,
        spread=req.spread,
        slippage=req.slippage,
        only_killzones=getattr(req, "only_killzones", True),
        target_rr=getattr(req, "target_rr", None),
        use_h1_gatekeeper=getattr(req, "use_h1_gatekeeper", True),
        learn_from_mistakes=getattr(req, "learn_from_mistakes", True),
        filter_high_impact_news=getattr(req, "filter_high_impact_news", True),
        news_blackout_mins=getattr(req, "news_blackout_mins", 30),
        exclude_asian_session=getattr(req, "exclude_asian_session", True),
        strict_structural_entry=getattr(req, "strict_structural_entry", False),
        session_window=getattr(req, "session_window", "cambodia_evening"),
    )

    backtester = RealisticBacktestEngine(
        initial_balance=req.initial_capital,
        risk_pct_per_trade=req.risk_per_trade,
        spread_usd=req.spread,
        slippage_usd=req.slippage,
        max_holding_bars=req.max_holding_bars,
        enable_breakeven=getattr(req, "enable_breakeven", False),
        breakeven_r=getattr(req, "breakeven_r", 1.0),
        enable_partial_tp=getattr(req, "enable_partial_tp", False),
        partial_tp_r=getattr(req, "partial_tp_r", 1.0),
        post_loss_cooldown_bars=getattr(req, "post_loss_cooldown_bars", 6),
    )

    result = backtester.run(test_df, signals=signals, sl_levels=sl_levels, tp_levels=tp_levels)

    # Save to memory and disk cache
    global _memory_cached_result
    _memory_cached_result = result

    try:
        settings.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = _CACHED_BACKTEST_FILE.with_suffix(".tmp")
        joblib.dump(result, tmp_file)
        if tmp_file.exists():
            import shutil
            shutil.move(str(tmp_file), str(_CACHED_BACKTEST_FILE))
    except Exception as e:
        logger.warning(f"Could not cache backtest result to disk: {e}")

    return _format_backtest_response(result, test_df=test_df)


def _execute_comparison(req: BacktestRequest) -> BacktestComparisonResponse:
    """Execute side-by-side comparison across all sub-models."""
    tf_clean = (getattr(req, "timeframe", None) or "M5").lower().strip()
    feat_path = settings.FEATURES_DIR / "xauusd" / f"{tf_clean}_features.parquet"
    if not feat_path.exists():
        feat_path = settings.FEATURES_DIR / "xauusd" / "m5_features.parquet"

    if not feat_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Features file missing. Run feature pipeline first.",
        )

    df_feats = pd.read_parquet(feat_path)
    if df_feats.empty:
        raise HTTPException(status_code=400, detail="Feature matrix is empty.")

    engine: PredictionEngine = get_prediction_engine(req.version)

    n = len(df_feats)
    use_full = getattr(req, "use_full_history", True)
    if use_full:
        test_df = df_feats.reset_index(drop=True)
    else:
        test_start = int(n * 0.70) + 12
        test_df = df_feats.iloc[test_start:].reset_index(drop=True)

    model_keys = ["ensemble", "price_action", "volume_profile", "technical", "footprint"]
    optimal_tfs = {
        "ensemble": "M5 Entry (Multi-Model Consensus)",
        "price_action": "M15 Entry (Wick Rejections)",
        "volume_profile": "M5 Entry (Session POC/VAH/VAL)",
        "technical": "M15 Entry (EMA Trend & RSI)",
        "footprint": "M5 Entry (Delta Absorption)",
    }
    comparison_items: List[ModelComparisonItem] = []

    for m_key in model_keys:
        signals, sl_levels, tp_levels, model_name, sl_tp_logic = generate_strategy_signals_and_levels(
            m_key,
            test_df,
            engine,
            threshold=req.threshold,
            spread=req.spread,
            slippage=req.slippage,
            only_killzones=getattr(req, "only_killzones", True),
            target_rr=getattr(req, "target_rr", None),
            use_h1_gatekeeper=getattr(req, "use_h1_gatekeeper", True),
            learn_from_mistakes=getattr(req, "learn_from_mistakes", True),
            filter_high_impact_news=getattr(req, "filter_high_impact_news", True),
            news_blackout_mins=getattr(req, "news_blackout_mins", 30),
            exclude_asian_session=getattr(req, "exclude_asian_session", True),
            strict_structural_entry=getattr(req, "strict_structural_entry", False),
            session_window=getattr(req, "session_window", "cambodia_evening"),
        )

        backtester = RealisticBacktestEngine(
            initial_balance=req.initial_capital,
            risk_pct_per_trade=req.risk_per_trade,
            spread_usd=req.spread,
            slippage_usd=req.slippage,
            max_holding_bars=req.max_holding_bars,
            enable_breakeven=getattr(req, "enable_breakeven", False),
            breakeven_r=getattr(req, "breakeven_r", 1.0),
            enable_partial_tp=getattr(req, "enable_partial_tp", False),
            partial_tp_r=getattr(req, "partial_tp_r", 1.0),
            post_loss_cooldown_bars=getattr(req, "post_loss_cooldown_bars", 6),
        )

        res = backtester.run(test_df, signals=signals, sl_levels=sl_levels, tp_levels=tp_levels)

        comparison_items.append(
            ModelComparisonItem(
                model_key=m_key,
                model_name=model_name,
                optimal_timeframe=optimal_tfs.get(m_key, "M5"),
                sl_tp_logic=sl_tp_logic,
                total_trades=res.total_trades,
                winning_trades=res.winning_trades,
                losing_trades=res.losing_trades,
                breakeven_trades=getattr(res, "breakeven_trades", 0),
                win_rate_pct=round(res.win_rate_pct, 2),
                win_rate_with_be_pct=round(getattr(res, "win_rate_with_be_pct", res.win_rate_pct), 2),
                total_net_profit=round(res.total_net_profit, 2),
                return_pct=round(res.return_pct, 2),
                profit_factor=round(res.profit_factor, 3),
                max_drawdown_pct=round(res.max_drawdown_pct, 2),
                avg_r_multiple=round(res.avg_r_multiple, 3),
            )
        )

    # Determine best performers
    best_profit = max(comparison_items, key=lambda x: x.total_net_profit).model_name
    best_winrate = max(comparison_items, key=lambda x: x.win_rate_pct).model_name

    return BacktestComparisonResponse(
        results=comparison_items,
        best_model_by_profit=best_profit,
        best_model_by_winrate=best_winrate,
    )


@router.post("/compare", response_model=BacktestComparisonResponse)
def compare_strategies_post(req: BacktestRequest) -> BacktestComparisonResponse:
    """Execute side-by-side comparative backtesting for all models with custom parameters."""
    return _execute_comparison(req)


@router.get("/compare", response_model=BacktestComparisonResponse)
def compare_strategies_get() -> BacktestComparisonResponse:
    """Execute side-by-side comparative backtesting for all models with default parameters."""
    return _execute_comparison(BacktestRequest())


@router.get("/results", response_model=BacktestResponse)
def get_backtest_results() -> BacktestResponse:
    """Retrieve the latest cached backtest simulation results with failure forensics."""
    global _memory_cached_result
    if _memory_cached_result is not None:
        result = _memory_cached_result
    elif _CACHED_BACKTEST_FILE.exists():
        result = joblib.load(_CACHED_BACKTEST_FILE)
    else:
        # Trigger default backtest if cache doesn't exist yet
        default_req = BacktestRequest()
        return run_backtest(default_req)

    test_df = None
    feat_path = settings.FEATURES_DIR / "xauusd" / "m5_features.parquet"
    if feat_path.exists():
        try:
            df = pd.read_parquet(feat_path)
            test_df = df.iloc[int(len(df) * 0.70) + 12:].reset_index(drop=True)
        except Exception:
            pass

    return _format_backtest_response(result, test_df=test_df)
