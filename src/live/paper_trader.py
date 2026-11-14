"""Real-time Paper Trading and Virtual Execution Engine.

Simulates live order placement, slippage, spread, intrabar TP/SL execution,
and persistent portfolio account tracking without risking real capital.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT
from src.core.logger import get_logger
from src.models.prediction_engine import TradePrediction

logger = get_logger("live.paper_trader")


@dataclass
class PaperPosition:
    position_id: str
    instrument: str
    direction: str                     # "BUY" or "SELL"
    entry_time: int
    entry_price: float
    size_oz: float
    stop_loss: float
    take_profit: float
    risk_usd: float
    current_price: float
    unrealized_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PaperTradeRecord:
    position_id: str
    instrument: str
    direction: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    size_oz: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str                   # "TAKE_PROFIT", "STOP_LOSS", "MANUAL", "TIME_LIMIT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PaperTradingEngine:
    """Virtual broker simulating execution, margin, and order management in real time."""

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        spread_usd: float = 0.20,
        slippage_usd: float = 0.05,
        risk_pct_per_trade: float = 0.01,
        state_file: Optional[Path] = None,
    ):
        self.initial_balance = initial_balance
        self.spread_usd = spread_usd
        self.slippage_usd = slippage_usd
        self.risk_pct_per_trade = risk_pct_per_trade
        self.state_file = state_file or (settings.PAPER_DIR / "account_state.json")

        self.balance: float = initial_balance
        self.equity: float = initial_balance
        self.open_position: Optional[PaperPosition] = None
        self.trade_history: List[PaperTradeRecord] = []
        self.is_running: bool = False
        self.last_candle_timestamp: Optional[int] = None

        # Load existing state if available
        self.load_state()

    def process_candle(
        self,
        candle: pd.Series,
        prediction: Optional[TradePrediction] = None,
    ) -> Dict[str, Any]:
        """Process incoming closed candle, check intrabar exit triggers, or execute new orders."""
        ts = int(candle["timestamp"])
        c_open = float(candle["open"])
        c_high = float(candle["high"])
        c_low = float(candle["low"])
        c_close = float(candle["close"])

        event_log = {"action": "NONE", "message": "Bar processed"}

        # 1. Check open position exit rules against intrabar High/Low
        if self.open_position is not None:
            pos = self.open_position
            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None

            if pos.direction == "BUY":
                # Conservative conflict check: Stop-Loss priority
                if c_low <= pos.stop_loss:
                    exit_price = pos.stop_loss - self.slippage_usd
                    exit_reason = "STOP_LOSS"
                elif c_high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "TAKE_PROFIT"
                else:
                    # Update mark-to-market unrealized PnL
                    pos.current_price = c_close
                    pos.unrealized_pnl = round((c_close - pos.entry_price) * pos.size_oz, 2)
                    self.equity = round(self.balance + pos.unrealized_pnl, 2)

            elif pos.direction == "SELL":
                # Conservative conflict check: Stop-Loss priority
                if c_high >= pos.stop_loss:
                    exit_price = pos.stop_loss + self.slippage_usd
                    exit_reason = "STOP_LOSS"
                elif c_low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "TAKE_PROFIT"
                else:
                    pos.current_price = c_close
                    pos.unrealized_pnl = round((pos.entry_price - c_close) * pos.size_oz, 2)
                    self.equity = round(self.balance + pos.unrealized_pnl, 2)

            # Close position if TP or SL triggered
            if exit_price is not None and exit_reason is not None:
                if pos.direction == "BUY":
                    pnl = (exit_price - pos.entry_price) * pos.size_oz
                else:
                    pnl = (pos.entry_price - exit_price) * pos.size_oz

                self.balance = round(self.balance + pnl, 2)
                self.equity = self.balance
                r_mult = round(pnl / pos.risk_usd, 2) if pos.risk_usd > 0 else 0.0

                closed_trade = PaperTradeRecord(
                    position_id=pos.position_id,
                    instrument=pos.instrument,
                    direction=pos.direction,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size_oz=pos.size_oz,
                    pnl_usd=round(pnl, 2),
                    pnl_pct=round((pnl / self.balance) * 100.0, 4),
                    r_multiple=r_mult,
                    exit_reason=exit_reason,
                )
                self.trade_history.append(closed_trade)
                self.open_position = None
                self.save_state()

                event_log = {
                    "action": "CLOSED",
                    "reason": exit_reason,
                    "pnl": round(pnl, 2),
                    "r_multiple": r_mult,
                }
                logger.info(f"Paper Trade Closed: {pos.direction} | PnL: ${pnl:.2f} ({r_mult}R) | Reason: {exit_reason}")
                return event_log

        # 2. Check for new order entry if flat and prediction is actionable
        if self.open_position is None and prediction is not None and prediction.direction in ["BUY", "SELL"]:
            risk_usd = self.balance * self.risk_pct_per_trade

            if prediction.direction == "BUY":
                entry_price = c_close + self.spread_usd + self.slippage_usd
                sl = prediction.stop_loss_reference
                tp = prediction.take_profit_reference
                risk_per_oz = entry_price - sl
            else:  # SELL
                entry_price = c_close - self.slippage_usd
                sl = prediction.stop_loss_reference
                tp = prediction.take_profit_reference
                risk_per_oz = sl - entry_price

            if risk_per_oz > 0:
                size_oz = round(risk_usd / risk_per_oz, 3)
                pos_id = f"POS_{ts}_{prediction.direction}"

                self.open_position = PaperPosition(
                    position_id=pos_id,
                    instrument=prediction.instrument,
                    direction=prediction.direction,
                    entry_time=ts,
                    entry_price=round(entry_price, 2),
                    size_oz=size_oz,
                    stop_loss=round(sl, 2),
                    take_profit=round(tp, 2),
                    risk_usd=round(risk_usd, 2),
                    current_price=round(entry_price, 2),
                    unrealized_pnl=0.0,
                )
                self.save_state()

                event_log = {
                    "action": "OPENED",
                    "direction": prediction.direction,
                    "entry_price": round(entry_price, 2),
                    "size_oz": size_oz,
                    "stop_loss": round(sl, 2),
                    "take_profit": round(tp, 2),
                }
                logger.info(f"Paper Trade Opened: {prediction.direction} @ ${entry_price:.2f} (Size: {size_oz} oz)")

        self.last_candle_timestamp = ts
        return event_log

    def reset_account(self, initial_balance: Optional[float] = None) -> None:
        """Reset virtual paper account balance to initial baseline."""
        self.balance = initial_balance or self.initial_balance
        self.equity = self.balance
        self.open_position = None
        self.trade_history = []
        self.save_state()
        logger.info(f"Paper trading account reset to ${self.balance:.2f}")

    def get_account_summary(self) -> Dict[str, Any]:
        """Retrieve live paper trading account status and metrics."""
        wins = [t for t in self.trade_history if t.pnl_usd > 0]
        losses = [t for t in self.trade_history if t.pnl_usd < 0]
        total = len(self.trade_history)

        win_rate = (len(wins) / total * 100.0) if total > 0 else 0.0
        gross_win = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (1.0 if not losses else 0.0)
        net_profit = round(self.equity - self.initial_balance, 2)
        return_pct = round((net_profit / self.initial_balance) * 100.0, 2)

        return {
            "initial_balance": self.initial_balance,
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "net_profit": net_profit,
            "return_pct": return_pct,
            "total_trades": total,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "open_position": self.open_position.to_dict() if self.open_position else None,
            "is_running": self.is_running,
        }

    def save_state(self) -> None:
        """Serialize state to disk."""
        data = {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "equity": self.equity,
            "open_position": self.open_position.to_dict() if self.open_position else None,
            "trade_history": [t.to_dict() for t in self.trade_history],
            "is_running": self.is_running,
            "last_candle_timestamp": self.last_candle_timestamp,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self) -> None:
        """Load state from disk if exists."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self.initial_balance = float(data.get("initial_balance", 10_000.0))
            self.balance = float(data.get("balance", 10_000.0))
            self.equity = float(data.get("equity", self.balance))
            self.is_running = bool(data.get("is_running", False))
            self.last_candle_timestamp = data.get("last_candle_timestamp")

            pos_data = data.get("open_position")
            if pos_data:
                self.open_position = PaperPosition(**pos_data)

            trades_data = data.get("trade_history", [])
            self.trade_history = [PaperTradeRecord(**t) for t in trades_data]
        except Exception as e:
            logger.warning(f"Could not load paper state: {e}; initializing new state.")
