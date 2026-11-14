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
) -> Tuple[List[str], List[float], List[float], str, str]:
    """Generate model-tailored signals, Entry, Stop-Loss, and Take-Profit price levels.

    Distinct Logic per Model with Institutional High-Win-Rate Edge:
    - Model E (Ensemble): Stacking meta-confluence + Dynamic ATR Volatility Stop (1.5x ATR).
    - Model B (SMC & ICT): ICT 2022 FVG / Silver Bullet confluence + Structural Swing Invalidation Stop ($0.50 buffer).
    - Model A (Technical): EMA Trend & RSI Momentum + Volatility Envelope Stop (1.5x ATR).
    - Model D (Price Action): Candle Rejection Wicks & Pin Bars + Candle Wick Extreme High/Low Stop ($0.50 buffer).
    - Session Filter: Restricts entries to London (07:00-10:00 UTC) & NY (12:00-15:00 UTC) Killzones when only_killzones is enabled.
    - H1 Macro Trend Gatekeeper: Price > H1 200 EMA & H1 Trend >= 0 for BUY; Price < H1 200 EMA & H1 Trend <= 0 for SELL.
    - AI Failure Memory: Vetoes entries matching historical loss fingerprints (low momentum chop, climax exhaustion, volatility expansion).
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

    # Check Killzone filter (London Open: 07:00-10:00 UTC, NY AM: 12:00-15:00 UTC)
    if "timestamp" in test_df.columns and only_killzones:
        dt = pd.to_datetime(test_df["timestamp"], unit="s")
        hours = dt.dt.hour.to_numpy()
        in_killzones = ((hours >= 7) & (hours <= 10)) | ((hours >= 12) & (hours <= 15))
    else:
        in_killzones = np.ones(n, dtype=bool)

    # Check H1 Macro Trend alignment (Gatekeeper: 200 EMA + Structure)
    h1_ema = test_df.get("h1_ema_200", test_df.get("m15_ema_200", test_df.get("ema_200", None)))
    h1_trend = test_df.get("h1_market_structure_trend", test_df.get("m15_market_structure_trend", test_df.get("market_structure_trend", None)))
    if use_h1_gatekeeper and h1_ema is not None and h1_trend is not None:
        ema_arr = h1_ema.to_numpy()
        trend_arr = h1_trend.to_numpy()
        h1_bull = (close > ema_arr) & (trend_arr >= 0)
        h1_bear = (close < ema_arr) & (trend_arr <= 0)
    else:
        h1_bull = np.ones(n, dtype=bool)
        h1_bear = np.ones(n, dtype=bool)

    # Check AI Failure Memory / Learned Mistake Filters
    if learn_from_mistakes:
        adx_arr = test_df.get("adx_14", pd.Series(25.0, index=test_df.index)).to_numpy()
        rsi_arr = test_df.get("rsi_14", pd.Series(50.0, index=test_df.index)).to_numpy()
        rolling_atr = pd.Series(atr).rolling(50, min_periods=5).mean().bfill().to_numpy()
        atr_ratio = np.where(rolling_atr > 0, atr / rolling_atr, 1.0)

        # Self-correction rules learned from historical loss autopsies:
        # 1. Low momentum chop: ADX < 18
        # 2. Climax exhaustion: RSI > 68 (BUY), RSI < 32 (SELL)
        # 3. Volatility expansion spike: ATR ratio > 1.40
        mistake_ok_buy = (adx_arr >= 18.0) & (rsi_arr <= 68.0) & (atr_ratio <= 1.40)
        mistake_ok_sell = (adx_arr >= 18.0) & (rsi_arr >= 32.0) & (atr_ratio <= 1.40)
    else:
        mistake_ok_buy = np.ones(n, dtype=bool)
        mistake_ok_sell = np.ones(n, dtype=bool)

    signals: List[str] = []
    sl_levels: List[float] = []
    tp_levels: List[float] = []

    m_type = (model_type or "ensemble").lower().strip()

    if m_type == "smc_ict":
        eff_rr = target_rr if target_rr and target_rr > 0 else 2.5
        model_name = "Model B: SMC & ICT 2022 Mentorship"
        sl_tp_logic = f"ICT 2022 FVG / Silver Bullet | SL: Structural Swing Invalidation ($0.50 buffer), TP: {eff_rr} R:R Opposing Liquidity"

        prob_b = engine.ensemble.model_b.predict_proba(test_df)
        ict_bull = (
            (test_df.get("ict_2022_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_silver_bullet_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_judas_swing_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_turtle_soup_bullish", pd.Series(0, index=test_df.index)).to_numpy() == 1)
        )
        ict_bear = (
            (test_df.get("ict_2022_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_silver_bullet_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_judas_swing_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1) |
            (test_df.get("ict_turtle_soup_bearish", pd.Series(0, index=test_df.index)).to_numpy() == 1)
        )

        swing_l = test_df.get("active_swing_low", pd.Series(0.0, index=test_df.index)).to_numpy()
        swing_h = test_df.get("active_swing_high", pd.Series(0.0, index=test_df.index)).to_numpy()

        for i in range(n):
            pb, pl = prob_b[i, 0], prob_b[i, 2]
            cp, ca = close[i], atr[i]

            is_buy = in_killzones[i] and h1_bull[i] and mistake_ok_buy[i] and ((ict_bull[i] and pl > 0.40) or (pl >= threshold and pl > pb))
            is_sell = in_killzones[i] and h1_bear[i] and mistake_ok_sell[i] and ((ict_bear[i] and pb > 0.40) or (pb >= threshold and pb > pl))

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                s_low = swing_l[i]
                if s_low > 0 and 0.50 <= (ep - s_low) <= 4.0 * ca:
                    sl = s_low - 0.50
                else:
                    sl = ep - (1.5 * ca)
                risk = max(ep - sl, 0.50)
                tp = ep + (eff_rr * risk)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                s_high = swing_h[i]
                if s_high > 0 and 0.50 <= (s_high - ep) <= 4.0 * ca:
                    sl = s_high + 0.50
                else:
                    sl = ep + (1.5 * ca)
                risk = max(sl - ep, 0.50)
                tp = ep - (eff_rr * risk)
            else:
                signals.append("WAIT")
                sl = cp - (1.5 * ca)
                tp = cp + (eff_rr * 1.5 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "technical":
        eff_rr = target_rr if target_rr and target_rr > 0 else 2.0
        model_name = "Model A: Technical Momentum (Indicators)"
        sl_tp_logic = f"EMA Trend & RSI Pullbacks | SL: 1.5x ATR Volatility Envelope, TP: {eff_rr} R:R Target"

        prob_a = engine.ensemble.model_a.predict_proba(test_df)
        ema_align = test_df.get("ema_alignment", pd.Series(0, index=test_df.index)).to_numpy()
        rsi_os = test_df.get("rsi_14_oversold", pd.Series(0, index=test_df.index)).to_numpy()
        rsi_ob = test_df.get("rsi_14_overbought", pd.Series(0, index=test_df.index)).to_numpy()

        for i in range(n):
            pb, pl = prob_a[i, 0], prob_a[i, 2]
            cp, ca = close[i], atr[i]

            is_buy = in_killzones[i] and h1_bull[i] and mistake_ok_buy[i] and ((pl > 0.40 and pl > pb and (ema_align[i] >= 1 or rsi_os[i] == 1)) or (pl >= threshold and pl > pb))
            is_sell = in_killzones[i] and h1_bear[i] and mistake_ok_sell[i] and ((pb > 0.40 and pb > pl and (ema_align[i] <= -1 or rsi_ob[i] == 1)) or (pb >= threshold and pb > pl))

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (1.5 * ca)
                tp = ep + (eff_rr * 1.5 * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (1.5 * ca)
                tp = ep - (eff_rr * 1.5 * ca)
            else:
                signals.append("WAIT")
                sl = cp - (1.5 * ca)
                tp = cp + (eff_rr * 1.5 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "price_action":
        eff_rr = target_rr if target_rr and target_rr > 0 else 2.0
        model_name = "Model D: Price Action & Rejection Wicks"
        sl_tp_logic = f"Candle Wick Extremes | SL: Candle High/Low Extreme (-$0.50 buffer), TP: {eff_rr} R:R Target"

        prob_d = engine.ensemble.model_d.predict_proba(test_df)
        pin_bull = test_df.get("pin_bar_bullish", pd.Series(0, index=test_df.index)).to_numpy()
        pin_bear = test_df.get("pin_bar_bearish", pd.Series(0, index=test_df.index)).to_numpy()
        engulf_bull = test_df.get("engulfing_bullish", pd.Series(0, index=test_df.index)).to_numpy()
        engulf_bear = test_df.get("engulfing_bearish", pd.Series(0, index=test_df.index)).to_numpy()
        lower_wick = test_df.get("lower_wick_ratio", pd.Series(0.0, index=test_df.index)).to_numpy()
        upper_wick = test_df.get("upper_wick_ratio", pd.Series(0.0, index=test_df.index)).to_numpy()

        for i in range(n):
            pb, pl = prob_d[i, 0], prob_d[i, 2]
            cp, ca, ch, cl = close[i], atr[i], high[i], low[i]

            is_buy = in_killzones[i] and h1_bull[i] and mistake_ok_buy[i] and ((pl > 0.40 and pl > pb and (pin_bull[i] == 1 or engulf_bull[i] == 1 or lower_wick[i] >= 0.40)) or (pl >= threshold and pl > pb))
            is_sell = in_killzones[i] and h1_bear[i] and mistake_ok_sell[i] and ((pb > 0.40 and pb > pl and (pin_bear[i] == 1 or engulf_bear[i] == 1 or upper_wick[i] >= 0.40)) or (pb >= threshold and pb > pl))

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                if 0.40 <= (ep - cl) <= 3.0 * ca:
                    sl = cl - 0.50
                else:
                    sl = ep - (1.2 * ca)
                risk = max(ep - sl, 0.40)
                tp = ep + (eff_rr * risk)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                if 0.40 <= (ch - ep) <= 3.0 * ca:
                    sl = ch + 0.50
                else:
                    sl = ep + (1.2 * ca)
                risk = max(sl - ep, 0.40)
                tp = ep - (eff_rr * risk)
            else:
                signals.append("WAIT")
                sl = cp - (1.2 * ca)
                tp = cp + (eff_rr * 1.2 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    else:  # "ensemble" default
        eff_rr = target_rr if target_rr and target_rr > 0 else 2.0
        model_name = "Model E: Calibrated Multi-Model Ensemble"
        sl_tp_logic = f"Calibrated Meta-Model Stacking | SL: 1.5x ATR Dynamic Stop, TP: {eff_rr} R:R Target"

        engine.ensemble.confidence_threshold = threshold
        decisions, probs, confs = engine.ensemble.predict_decision(test_df)

        for i in range(n):
            cp, ca = close[i], atr[i]
            sig = decisions[i]

            is_buy = in_killzones[i] and h1_bull[i] and mistake_ok_buy[i] and (sig == "BUY")
            is_sell = in_killzones[i] and h1_bear[i] and mistake_ok_sell[i] and (sig == "SELL")

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (1.5 * ca)
                tp = ep + (eff_rr * 1.5 * ca)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (1.5 * ca)
                tp = ep - (eff_rr * 1.5 * ca)
            else:
                signals.append("WAIT")
                sl = cp - (1.5 * ca)
                tp = cp + (eff_rr * 1.5 * ca)

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

    # Split: test period is out-of-sample last 30%
    n = len(df_feats)
    test_start = int(n * 0.70) + 12
    test_df = df_feats.iloc[test_start:].reset_index(drop=True)

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
    )

    backtester = RealisticBacktestEngine(
        initial_balance=req.initial_capital,
        risk_pct_per_trade=req.risk_per_trade,
        spread_usd=req.spread,
        slippage_usd=req.slippage,
        max_holding_bars=req.max_holding_bars,
        enable_breakeven=getattr(req, "enable_breakeven", True),
        breakeven_r=getattr(req, "breakeven_r", 1.0),
        enable_partial_tp=getattr(req, "enable_partial_tp", True),
        partial_tp_r=getattr(req, "partial_tp_r", 1.0),
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
    test_start = int(n * 0.70) + 12
    test_df = df_feats.iloc[test_start:].reset_index(drop=True)

    model_keys = ["ensemble", "smc_ict", "technical", "price_action"]
    optimal_tfs = {
        "ensemble": "M5 Entry (H1/M15 Bias)",
        "smc_ict": "M5 Entry (H1 Liquidity)",
        "technical": "M15 Entry (H1 Trend)",
        "price_action": "M15 Entry (Key Levels)",
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
        )

        backtester = RealisticBacktestEngine(
            initial_balance=req.initial_capital,
            risk_pct_per_trade=req.risk_per_trade,
            spread_usd=req.spread,
            slippage_usd=req.slippage,
            max_holding_bars=req.max_holding_bars,
            enable_breakeven=getattr(req, "enable_breakeven", True),
            breakeven_r=getattr(req, "breakeven_r", 1.0),
            enable_partial_tp=getattr(req, "enable_partial_tp", True),
            partial_tp_r=getattr(req, "partial_tp_r", 1.0),
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
