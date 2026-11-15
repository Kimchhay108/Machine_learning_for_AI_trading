"""FastAPI Router for Real-time Paper Trading and Virtual Portfolio Management."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_prediction_engine
from src.live.daemon import PaperTradingDaemon
from src.live.paper_trader import PaperTradingEngine
from src.models.prediction_engine import PredictionEngine

router = APIRouter(prefix="/api/paper", tags=["Paper Trading"])

# Singleton paper engine and daemon instance
_paper_engine: Optional[PaperTradingEngine] = None
_paper_daemon: Optional[PaperTradingDaemon] = None


def get_paper_engine() -> PaperTradingEngine:
    global _paper_engine
    if _paper_engine is None:
        _paper_engine = PaperTradingEngine(
            only_killzones=True,
            filter_high_impact_news=True,
            exclude_asian_session=True,
        )
    return _paper_engine


def get_paper_daemon() -> PaperTradingDaemon:
    global _paper_daemon
    if _paper_daemon is None:
        engine = get_paper_engine()
        pred_engine = get_prediction_engine()
        _paper_daemon = PaperTradingDaemon(paper_engine=engine, prediction_engine=pred_engine)
    return _paper_daemon


@router.get("/account")
def get_paper_account(engine: PaperTradingEngine = Depends(get_paper_engine)) -> Dict[str, Any]:
    """Retrieve real-time paper trading account status, portfolio equity, and open positions."""
    return engine.get_account_summary()


@router.get("/trades")
def get_paper_trades(engine: PaperTradingEngine = Depends(get_paper_engine)) -> List[Dict[str, Any]]:
    """Retrieve history of completed simulated paper trades."""
    return [t.to_dict() for t in engine.trade_history]


@router.post("/start")
def start_paper_trading(daemon: PaperTradingDaemon = Depends(get_paper_daemon)) -> Dict[str, Any]:
    """Start automated background paper trading polling and execution."""
    daemon.start()
    return {"status": "RUNNING", "message": "Paper trading daemon started successfully."}


@router.post("/stop")
def stop_paper_trading(daemon: PaperTradingDaemon = Depends(get_paper_daemon)) -> Dict[str, Any]:
    """Pause automated background paper trading daemon."""
    daemon.stop()
    return {"status": "PAUSED", "message": "Paper trading daemon stopped."}


@router.post("/reset")
def reset_paper_account(
    initial_balance: float = 10000.0,
    engine: PaperTradingEngine = Depends(get_paper_engine),
) -> Dict[str, Any]:
    """Reset virtual account balance and clear trade history."""
    engine.reset_account(initial_balance)
    return {"status": "RESET", "balance": engine.balance, "equity": engine.equity}


@router.post("/tick")
def trigger_paper_tick(daemon: PaperTradingDaemon = Depends(get_paper_daemon)) -> Dict[str, Any]:
    """Trigger a single real-time candle poll and execution cycle on demand."""
    event = daemon.step()
    return {
        "status": "TICK_COMPLETED",
        "event": event or {"action": "NONE", "message": "No new candle or waiting for data"},
        "account": daemon.paper_engine.get_account_summary(),
    }
