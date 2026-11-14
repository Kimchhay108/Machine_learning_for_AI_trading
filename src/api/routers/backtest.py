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
    ModelComparisonItem,
    TradeRecordSchema,
)
from src.backtest.engine import RealisticBacktestEngine, BacktestResult
from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT
from src.core.logger import get_logger
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
) -> Tuple[List[str], List[float], List[float], str, str]:
    """Generate model-tailored signals, Entry, Stop-Loss, and Take-Profit price levels.

    Distinct Logic per Model:
    - Model E (Ensemble): Stacking meta-confluence + Dynamic ATR Volatility Stop (1.5x ATR) / 3.0x ATR TP (2.0 R:R).
    - Model B (SMC & ICT): ICT 2022 FVG / Silver Bullet confluence + Structural Swing Invalidation Stop ($0.50 buffer) & 2.5 R:R Opposing Liquidity TP.
    - Model A (Technical): EMA Trend & RSI Momentum + Volatility Envelope Stop (1.5x ATR) & 3.0x ATR TP (2.0 R:R).
    - Model D (Price Action): Candle Rejection Wicks & Pin Bars + Candle Wick Extreme High/Low Stop ($0.50 buffer) & 2.0 R:R Target.
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

    signals: List[str] = []
    sl_levels: List[float] = []
    tp_levels: List[float] = []

    m_type = (model_type or "ensemble").lower().strip()

    if m_type == "smc_ict":
        model_name = "Model B: SMC & ICT 2022 Mentorship"
        sl_tp_logic = "ICT 2022 FVG / Silver Bullet | SL: Structural Swing Invalidation ($0.50 buffer), TP: 2.5 R:R Opposing Liquidity"

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

            is_buy = (ict_bull[i] and pl > 0.35) or (pl >= threshold and pl > pb)
            is_sell = (ict_bear[i] and pb > 0.35) or (pb >= threshold and pb > pl)

            if is_buy and not is_sell:
                signals.append("BUY")
                ep = cp + spread + slippage
                s_low = swing_l[i]
                if s_low > 0 and 0.50 <= (ep - s_low) <= 4.0 * ca:
                    sl = s_low - 0.50
                else:
                    sl = ep - (1.5 * ca)
                risk = max(ep - sl, 0.50)
                tp = ep + (2.5 * risk)
            elif is_sell and not is_buy:
                signals.append("SELL")
                ep = cp - slippage
                s_high = swing_h[i]
                if s_high > 0 and 0.50 <= (s_high - ep) <= 4.0 * ca:
                    sl = s_high + 0.50
                else:
                    sl = ep + (1.5 * ca)
                risk = max(sl - ep, 0.50)
                tp = ep - (2.5 * risk)
            else:
                signals.append("WAIT")
                sl = cp - (1.5 * ca)
                tp = cp + (3.75 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "technical":
        model_name = "Model A: Technical Momentum (Indicators)"
        sl_tp_logic = "EMA Trend & RSI Pullbacks | SL: 1.5x ATR Volatility Envelope, TP: 3.0x ATR (2.0 R:R)"

        prob_a = engine.ensemble.model_a.predict_proba(test_df)
        for i in range(n):
            pb, pl = prob_a[i, 0], prob_a[i, 2]
            cp, ca = close[i], atr[i]

            if pl >= threshold and pl > pb:
                signals.append("BUY")
                ep = cp + spread + slippage
                sl = ep - (1.5 * ca)
                tp = ep + (3.0 * ca)
            elif pb >= threshold and pb > pl:
                signals.append("SELL")
                ep = cp - slippage
                sl = ep + (1.5 * ca)
                tp = ep - (3.0 * ca)
            else:
                signals.append("WAIT")
                sl = cp - (1.5 * ca)
                tp = cp + (3.0 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    elif m_type == "price_action":
        model_name = "Model D: Price Action & Rejection Wicks"
        sl_tp_logic = "Candle Wick Extremes | SL: Candle High/Low Extreme (-$0.50 buffer), TP: 2.0 R:R Target"

        prob_d = engine.ensemble.model_d.predict_proba(test_df)
        for i in range(n):
            pb, pl = prob_d[i, 0], prob_d[i, 2]
            cp, ca, ch, cl = close[i], atr[i], high[i], low[i]

            if pl >= threshold and pl > pb:
                signals.append("BUY")
                ep = cp + spread + slippage
                if 0.40 <= (ep - cl) <= 3.0 * ca:
                    sl = cl - 0.50
                else:
                    sl = ep - (1.2 * ca)
                risk = max(ep - sl, 0.40)
                tp = ep + (2.0 * risk)
            elif pb >= threshold and pb > pl:
                signals.append("SELL")
                ep = cp - slippage
                if 0.40 <= (ch - ep) <= 3.0 * ca:
                    sl = ch + 0.50
                else:
                    sl = ep + (1.2 * ca)
                risk = max(sl - ep, 0.40)
                tp = ep - (2.0 * risk)
            else:
                signals.append("WAIT")
                sl = cp - (1.2 * ca)
                tp = cp + (2.4 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    else:  # "ensemble" default
        model_name = "Model E: Calibrated Multi-Model Ensemble"
        sl_tp_logic = "Calibrated Meta-Model Stacking | SL: 1.5x ATR Dynamic Stop, TP: 3.0x ATR (2.0 R:R)"

        engine.ensemble.confidence_threshold = threshold
        decisions, probs, confs = engine.ensemble.predict_decision(test_df)
        signals = decisions

        for i in range(n):
            cp, ca = close[i], atr[i]
            sig = signals[i]

            if sig == "BUY":
                ep = cp + spread + slippage
                sl = ep - (1.5 * ca)
                tp = ep + (3.0 * ca)
            elif sig == "SELL":
                ep = cp - slippage
                sl = ep + (1.5 * ca)
                tp = ep - (3.0 * ca)
            else:
                sl = cp - (1.5 * ca)
                tp = cp + (3.0 * ca)

            sl_levels.append(sl)
            tp_levels.append(tp)

    return signals, sl_levels, tp_levels, model_name, sl_tp_logic


def _format_backtest_response(result: BacktestResult) -> BacktestResponse:
    """Helper to convert engine BacktestResult into API BacktestResponse."""
    trades_schema = [
        TradeRecordSchema(
            entry_time=int(t.entry_time // 1000 if t.entry_time > 10_000_000_000 else t.entry_time),
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
        )
        for t in result.trades
    ]

    return BacktestResponse(
        initial_balance=round(result.initial_balance, 2),
        final_balance=round(result.final_balance, 2),
        total_net_profit=round(result.total_net_profit, 2),
        return_pct=round(result.return_pct, 2),
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate_pct=round(result.win_rate_pct, 2),
        profit_factor=round(result.profit_factor, 3),
        max_drawdown_usd=round(result.max_drawdown_usd, 2),
        max_drawdown_pct=round(result.max_drawdown_pct, 2),
        sharpe_ratio=round(result.sharpe_ratio, 3),
        avg_trade_usd=round(result.avg_trade_usd, 2),
        avg_r_multiple=round(result.avg_r_multiple, 3),
        equity_curve=[round(val, 2) for val in result.equity_curve],
        trades=trades_schema,
    )


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Execute realistic out-of-sample backtest with model-specific Entry, SL, and TP rules."""
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
        req.threshold,
        req.spread,
        req.slippage,
    )

    backtester = RealisticBacktestEngine(
        initial_balance=req.initial_capital,
        risk_pct_per_trade=req.risk_per_trade,
        spread_usd=req.spread,
        slippage_usd=req.slippage,
        max_holding_bars=req.max_holding_bars,
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

    return _format_backtest_response(result)


def _execute_comparison(req: BacktestRequest) -> BacktestComparisonResponse:
    """Execute side-by-side comparison across all sub-models."""
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
    comparison_items: List[ModelComparisonItem] = []

    for m_key in model_keys:
        signals, sl_levels, tp_levels, model_name, sl_tp_logic = generate_strategy_signals_and_levels(
            m_key,
            test_df,
            engine,
            req.threshold,
            req.spread,
            req.slippage,
        )

        backtester = RealisticBacktestEngine(
            initial_balance=req.initial_capital,
            risk_pct_per_trade=req.risk_per_trade,
            spread_usd=req.spread,
            slippage_usd=req.slippage,
            max_holding_bars=req.max_holding_bars,
        )

        res = backtester.run(test_df, signals=signals, sl_levels=sl_levels, tp_levels=tp_levels)

        comparison_items.append(
            ModelComparisonItem(
                model_key=m_key,
                model_name=model_name,
                sl_tp_logic=sl_tp_logic,
                total_trades=res.total_trades,
                win_rate_pct=round(res.win_rate_pct, 2),
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
    """Retrieve the latest cached backtest simulation results."""
    global _memory_cached_result
    if _memory_cached_result is not None:
        result = _memory_cached_result
    elif _CACHED_BACKTEST_FILE.exists():
        result = joblib.load(_CACHED_BACKTEST_FILE)
    else:
        # Trigger default backtest if cache doesn't exist yet
        default_req = BacktestRequest()
        return run_backtest(default_req)

    return _format_backtest_response(result)
