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
    orig_size_oz: float = 0.0
    initial_risk_usd: float = 0.0
    breakeven_active: bool = False
    partial_tp_taken: bool = False
    realized_partial_pnl: float = 0.0

    def __post_init__(self):
        if self.orig_size_oz <= 0:
            self.orig_size_oz = self.size_oz
        if self.initial_risk_usd <= 0:
            self.initial_risk_usd = self.risk_usd

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
    exit_reason: str                   # "TAKE_PROFIT", "STOP_LOSS", "BREAKEVEN", "PARTIAL_TP_BE", "MANUAL", "TIME_LIMIT"

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
        enable_breakeven: bool = False,
        breakeven_r: float = 1.0,
        enable_partial_tp: bool = False,
        partial_tp_r: float = 1.0,
        partial_tp_ratio: float = 0.50,
        only_killzones: bool = False,
        filter_high_impact_news: bool = True,
        news_blackout_mins: int = 30,
        exclude_asian_session: bool = False,
        post_loss_cooldown_bars: int = 3,
    ):
        self.initial_balance = initial_balance
        self.spread_usd = spread_usd
        self.slippage_usd = slippage_usd
        self.risk_pct_per_trade = risk_pct_per_trade
        self.state_file = state_file or (settings.PAPER_DIR / "account_state.json")
        self.enable_breakeven = enable_breakeven
        self.breakeven_r = breakeven_r
        self.enable_partial_tp = enable_partial_tp
        self.partial_tp_r = partial_tp_r
        self.partial_tp_ratio = partial_tp_ratio
        self.only_killzones = only_killzones
        self.filter_high_impact_news = filter_high_impact_news
        self.news_blackout_mins = news_blackout_mins
        self.exclude_asian_session = exclude_asian_session
        self.post_loss_cooldown_bars = post_loss_cooldown_bars

        self.balance: float = initial_balance
        self.equity: float = initial_balance
        self.open_position: Optional[PaperPosition] = None
        self.trade_history: List[PaperTradeRecord] = []
        self.is_running: bool = False
        self.last_candle_timestamp: Optional[int] = None
        self.last_loss_exit_time: Optional[int] = None

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

            orig_risk_per_oz = (pos.initial_risk_usd / pos.orig_size_oz) if pos.orig_size_oz > 0 else max(abs(pos.entry_price - pos.stop_loss), 0.50)

            if pos.direction == "BUY":
                orig_sl = pos.entry_price - orig_risk_per_oz
                hit_orig_sl = c_low <= orig_sl
                hit_tp = c_high >= pos.take_profit

                if hit_orig_sl and hit_tp:
                    # Conservative conflict check: Stop-Loss priority
                    exit_price = orig_sl - self.slippage_usd
                    exit_reason = "STOP_LOSS"
                else:
                    # Trailing 50% Partial Take-Profit scale-out at +1.0R
                    if self.enable_partial_tp and not pos.partial_tp_taken and c_high >= pos.entry_price + (self.partial_tp_r * orig_risk_per_oz):
                        p_size = pos.orig_size_oz * self.partial_tp_ratio
                        ptp_price = (pos.entry_price + self.partial_tp_r * orig_risk_per_oz) - self.slippage_usd
                        realized = (ptp_price - pos.entry_price) * p_size
                        pos.realized_partial_pnl += realized
                        pos.size_oz = max(pos.size_oz - p_size, 0.0)
                        pos.partial_tp_taken = True
                        pos.stop_loss = pos.entry_price + self.spread_usd
                        pos.breakeven_active = True
                        logger.info(f"Paper BUY 50% Partial TP Banked: +${realized:.2f} | Remaining: {pos.size_oz:.3f} oz | SL moved to BE: ${pos.stop_loss:.2f}")

                    elif self.enable_breakeven and not pos.breakeven_active and c_high >= pos.entry_price + (self.breakeven_r * orig_risk_per_oz):
                        pos.stop_loss = pos.entry_price + self.spread_usd
                        pos.breakeven_active = True
                        logger.info(f"Paper BUY Breakeven Protected: SL moved to ${pos.stop_loss:.2f}")

                    if c_high >= pos.take_profit:
                        exit_price = pos.take_profit - self.slippage_usd
                        exit_reason = "TAKE_PROFIT"
                    elif c_low <= pos.stop_loss:
                        exit_price = pos.stop_loss - self.slippage_usd
                        exit_reason = "PARTIAL_TP_BE" if pos.partial_tp_taken else ("BREAKEVEN" if pos.breakeven_active else "STOP_LOSS")
                    else:
                        pos.current_price = c_close
                        rem_unrealized = (c_close - pos.entry_price) * pos.size_oz
                        pos.unrealized_pnl = round(pos.realized_partial_pnl + rem_unrealized, 2)
                        self.equity = round(self.balance + pos.unrealized_pnl, 2)

            elif pos.direction == "SELL":
                orig_sl = pos.entry_price + orig_risk_per_oz
                hit_orig_sl = c_high >= orig_sl
                hit_tp = c_low <= pos.take_profit

                if hit_orig_sl and hit_tp:
                    # Conservative conflict check: Stop-Loss priority
                    exit_price = orig_sl + self.slippage_usd
                    exit_reason = "STOP_LOSS"
                else:
                    # Trailing 50% Partial Take-Profit scale-out at +1.0R
                    if self.enable_partial_tp and not pos.partial_tp_taken and c_low <= pos.entry_price - (self.partial_tp_r * orig_risk_per_oz):
                        p_size = pos.orig_size_oz * self.partial_tp_ratio
                        ptp_price = (pos.entry_price - self.partial_tp_r * orig_risk_per_oz) + self.slippage_usd
                        realized = (pos.entry_price - ptp_price) * p_size
                        pos.realized_partial_pnl += realized
                        pos.size_oz = max(pos.size_oz - p_size, 0.0)
                        pos.partial_tp_taken = True
                        pos.stop_loss = pos.entry_price - self.spread_usd
                        pos.breakeven_active = True
                        logger.info(f"Paper SELL 50% Partial TP Banked: +${realized:.2f} | Remaining: {pos.size_oz:.3f} oz | SL moved to BE: ${pos.stop_loss:.2f}")

                    elif self.enable_breakeven and not pos.breakeven_active and c_low <= pos.entry_price - (self.breakeven_r * orig_risk_per_oz):
                        pos.stop_loss = pos.entry_price - self.spread_usd
                        pos.breakeven_active = True
                        logger.info(f"Paper SELL Breakeven Protected: SL moved to ${pos.stop_loss:.2f}")

                    if c_low <= pos.take_profit:
                        exit_price = pos.take_profit + self.slippage_usd
                        exit_reason = "TAKE_PROFIT"
                    elif c_high >= pos.stop_loss:
                        exit_price = pos.stop_loss + self.slippage_usd
                        exit_reason = "PARTIAL_TP_BE" if pos.partial_tp_taken else ("BREAKEVEN" if pos.breakeven_active else "STOP_LOSS")
                    else:
                        pos.current_price = c_close
                        rem_unrealized = (pos.entry_price - c_close) * pos.size_oz
                        pos.unrealized_pnl = round(pos.realized_partial_pnl + rem_unrealized, 2)
                        self.equity = round(self.balance + pos.unrealized_pnl, 2)

            # Close position if TP, SL, or BE triggered
            if exit_price is not None and exit_reason is not None:
                if pos.direction == "BUY":
                    rem_pnl = (exit_price - pos.entry_price) * pos.size_oz
                else:
                    rem_pnl = (pos.entry_price - exit_price) * pos.size_oz

                pnl = pos.realized_partial_pnl + rem_pnl
                self.balance = round(self.balance + pnl, 2)
                self.equity = self.balance
                r_mult = round(pnl / pos.initial_risk_usd, 2) if pos.initial_risk_usd > 0 else 0.0

                closed_trade = PaperTradeRecord(
                    position_id=pos.position_id,
                    instrument=pos.instrument,
                    direction=pos.direction,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    entry_price=pos.entry_price,
                    exit_price=round(exit_price, 2),
                    size_oz=pos.orig_size_oz,
                    pnl_usd=round(pnl, 2),
                    pnl_pct=round((pnl / self.balance) * 100.0, 4),
                    r_multiple=r_mult,
                    exit_reason=exit_reason,
                )
                self.trade_history.append(closed_trade)
                self.open_position = None
                if exit_reason == "STOP_LOSS":
                    self.last_loss_exit_time = ts
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
            # AI Mistake Learning Cooldown: Abstain after a Stop Loss hit to avoid revenge chop
            if self.last_loss_exit_time is not None:
                scale = 1000.0 if ts > 10_000_000_000 else 1.0
                delta_sec = (ts - self.last_loss_exit_time) / scale
                cooldown_sec = self.post_loss_cooldown_bars * 300.0  # 5-min bars
                if delta_sec < cooldown_sec:
                    self.last_candle_timestamp = ts
                    return {"action": "WAIT", "message": f"Filtered out: AI Mistake Quarantine active ({int(cooldown_sec - delta_sec)}s remaining)"}

            # Strict Asian session filter (Never trade 21:00 - 07:00 UTC)
            if self.exclude_asian_session:
                dt_utc = pd.to_datetime(ts // 1000 if ts > 10_000_000_000 else ts, unit="s", utc=True)
                hr = dt_utc.hour
                if hr >= 21 or hr < 7:
                    self.last_candle_timestamp = ts
                    return {"action": "WAIT", "message": "Filtered out: Asian Session Blackout (Do NOT trade 21:00 - 07:00 UTC)"}

            # Optional Killzone session filter
            if self.only_killzones:
                dt = pd.to_datetime(ts // 1000 if ts > 10_000_000_000 else ts, unit="s", utc=True)
                hr = dt.hour
                in_kz = ((hr >= 7) and (hr <= 10)) or ((hr >= 12) and (hr <= 15))
                if not in_kz:
                    self.last_candle_timestamp = ts
                    return {"action": "WAIT", "message": "Filtered out: Outside London / NY Killzones"}

            # High-Impact News Blackout filter
            if self.filter_high_impact_news:
                try:
                    from src.models.news_filter import get_news_filter
                    nf = get_news_filter()
                    dt_utc = pd.to_datetime(ts // 1000 if ts > 10_000_000_000 else ts, unit="s", utc=True).to_pydatetime()
                    is_blackout, active_evt, delta_mins = nf.is_blackout_active(
                        dt_utc,
                        buffer_before_mins=self.news_blackout_mins,
                        buffer_after_mins=self.news_blackout_mins,
                    )
                    if is_blackout and active_evt:
                        self.last_candle_timestamp = ts
                        msg = f"News Blackout Active: {active_evt.title} ({delta_mins:+.0f}m from release). Trading halted."
                        logger.warning(msg)
                        return {"action": "WAIT", "message": msg}
                except Exception as e:
                    logger.warning(f"Error checking news blackout in paper trader: {e}")

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
                    orig_size_oz=size_oz,
                    stop_loss=round(sl, 2),
                    take_profit=round(tp, 2),
                    risk_usd=round(risk_usd, 2),
                    initial_risk_usd=round(risk_usd, 2),
                    current_price=round(entry_price, 2),
                    unrealized_pnl=0.0,
                    breakeven_active=False,
                    partial_tp_taken=False,
                    realized_partial_pnl=0.0,
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
        losses = [t for t in self.trade_history if t.pnl_usd < 0 and "BREAKEVEN" not in t.exit_reason and "PARTIAL" not in t.exit_reason]
        breakevens = [t for t in self.trade_history if t not in wins and t not in losses]
        total = len(self.trade_history)

        win_rate = (len(wins) / total * 100.0) if total > 0 else 0.0
        win_rate_with_be = ((len(wins) + len(breakevens)) / total * 100.0) if total > 0 else 0.0
        gross_win = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
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
            "breakeven_trades": len(breakevens),
            "win_rate_pct": round(win_rate, 2),
            "win_rate_with_be_pct": round(win_rate_with_be, 2),
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
