"""Realistic Backtesting Simulation API Endpoints."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
import joblib
import pandas as pd

from src.api.dependencies import get_prediction_engine
from src.api.schemas import BacktestRequest, BacktestResponse, TradeRecordSchema
from src.backtest.engine import RealisticBacktestEngine
from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT
from src.core.logger import get_logger
from src.models.prediction_engine import PredictionEngine

logger = get_logger("api.routers.backtest")
router = APIRouter(prefix="/api/backtest", tags=["Backtest Lab"])

_CACHED_BACKTEST_FILE = settings.BACKTEST_DIR / "latest_backtest.joblib"


@router.post("/run", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Execute realistic out-of-sample backtest with conservative conflict resolution and transaction costs."""
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
    engine.ensemble.confidence_threshold = req.threshold

    # Split: test period is out-of-sample last 30%
    n = len(df_feats)
    test_start = int(n * 0.70) + 12
    test_df = df_feats.iloc[test_start:].reset_index(drop=True)

    decisions, probs, confs = engine.ensemble.predict_decision(test_df)

    backtester = RealisticBacktestEngine(
        initial_balance=req.initial_capital,
        risk_pct_per_trade=req.risk_per_trade,
        spread_usd=req.spread,
        slippage_usd=req.slippage,
        max_holding_bars=req.max_holding_bars,
    )

    result = backtester.run(test_df, signals=decisions)

    # Save to cache
    settings.BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, _CACHED_BACKTEST_FILE)

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


@router.get("/results", response_model=BacktestResponse)
def get_backtest_results() -> BacktestResponse:
    """Retrieve the latest cached backtest simulation results."""
    if not _CACHED_BACKTEST_FILE.exists():
        # Trigger default backtest if cache doesn't exist yet
        default_req = BacktestRequest()
        return run_backtest(default_req)

    result = joblib.load(_CACHED_BACKTEST_FILE)

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
