"""Institutional AI Signal Lifecycle and State Manager.

Implements trigger-based range execution and explicit trade termination:
- Calculates when to Buy or Sell at an institutional range (Limit / Retest Zone).
- Status follows: SCANNING -> PENDING_TRIGGER -> IN_POSITION -> TRADE_COMPLETED (TP or SL).
- Chart lines appear ONLY during active recommendations (PENDING_TRIGGER or IN_POSITION).
- When a trade ends on TP or SL, lines are automatically removed so the chart stays clean.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import settings
from src.core.logger import get_logger
from src.models.prediction_engine import TradePrediction

logger = get_logger("live.signal_manager")


@dataclass
class ActiveSignalState:
    setup_id: str
    instrument: str
    timeframe: str
    direction: str                     # "BUY", "SELL", or "WAIT"
    entry_reference_price: float       # Exact Execution Trigger Level
    stop_loss_reference: float         # Invalidation SL Level
    take_profit_reference: float       # Conservative TP1 Level
    take_profit_2: float               # Runner TP2 Level
    risk_reward_ratio: float
    prediction_timestamp: int
    created_at_utc: str
    confidence: float
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    market_regime: str
    model_version: str
    contributing_features: Dict[str, Any]
    signal_status: str = "SCANNING"    # "SCANNING", "PENDING_TRIGGER", "IN_POSITION", "TRADE_COMPLETED_TP", "TRADE_COMPLETED_SL"
    current_r_multiple: float = 0.0
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0
    is_setup_locked: bool = True
    bars_active: int = 0
    last_evaluated_price: float = 0.0

    # Range execution & clean lifecycle
    entry_range_low: float = 0.0
    entry_range_high: float = 0.0
    trigger_price: float = 0.0
    show_chart_lines: bool = False
    trade_ended_reason: Optional[str] = None
    trade_ended_at_utc: Optional[str] = None
    ai_recommendation_text: Optional[str] = None
    distance_to_trigger: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActiveSignalState":
        return cls(**data)


class ActiveSignalManager:
    """Manages active AI trade setups with stationary levels and intrabar lifecycle tracking."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (settings.PAPER_DIR / "active_signal.json")
        self._active_signals: Dict[str, ActiveSignalState] = {}
        self.load_state()

    def _get_key(self, instrument: str, timeframe: str) -> str:
        return f"{instrument.upper()}_{timeframe.upper()}"

    def load_state(self) -> None:
        """Load persisted active signal states from disk."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                self._active_signals[k] = ActiveSignalState.from_dict(v)
            logger.info(f"Loaded {len(self._active_signals)} active signal states from {self.state_file}")
        except Exception as e:
            logger.warning(f"Could not load active signal state: {e}")

    def save_state(self) -> None:
        """Persist active signal states to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self._active_signals.items()}
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save active signal state: {e}")

    def get_signal(
        self,
        instrument: str,
        timeframe: str,
        current_price: Optional[float] = None,
        candle_high: Optional[float] = None,
        candle_low: Optional[float] = None,
    ) -> Optional[ActiveSignalState]:
        """Retrieve and update active signal state against live market ticks."""
        key = self._get_key(instrument, timeframe)
        signal = self._active_signals.get(key)
        if not signal:
            return None

        # If trade already completed, it remains ended until user resets or next confirmed setup
        if signal.signal_status in ("TRADE_COMPLETED_TP", "TRADE_COMPLETED_SL"):
            signal.show_chart_lines = False
            return signal

        if current_price is None or current_price <= 0:
            return signal

        high = candle_high if candle_high is not None else current_price
        low = candle_low if candle_low is not None else current_price
        signal.last_evaluated_price = round(current_price, 2)
        signal.highest_price_seen = max(signal.highest_price_seen or current_price, high)
        signal.lowest_price_seen = min(signal.lowest_price_seen or current_price, low)

        # -------------------------------------------------------------
        # Phase 1: PENDING_TRIGGER (Waiting for price to reach the AI range)
        # -------------------------------------------------------------
        if signal.signal_status == "PENDING_TRIGGER":
            signal.show_chart_lines = True
            if signal.direction == "SELL":
                signal.distance_to_trigger = round(max(0.0, signal.trigger_price - current_price), 2)
                # Sell Trigger: price rallies into the Sell Range
                if high >= signal.trigger_price or high >= signal.entry_range_low:
                    signal.signal_status = "IN_POSITION"
                    signal.entry_reference_price = signal.trigger_price
                    logger.info(f"Signal {signal.setup_id} TRIGGERED SELL at ${signal.trigger_price:.2f} (High reached ${high:.2f})")
            elif signal.direction == "BUY":
                signal.distance_to_trigger = round(max(0.0, current_price - signal.trigger_price), 2)
                # Buy Trigger: price pulls back into the Buy Range
                if low <= signal.trigger_price or low <= signal.entry_range_high:
                    signal.signal_status = "IN_POSITION"
                    signal.entry_reference_price = signal.trigger_price
                    logger.info(f"Signal {signal.setup_id} TRIGGERED BUY at ${signal.trigger_price:.2f} (Low reached ${low:.2f})")

        # -------------------------------------------------------------
        # Phase 2: IN_POSITION (Active trade running to TP or SL)
        # -------------------------------------------------------------
        if signal.signal_status == "IN_POSITION":
            signal.show_chart_lines = True
            entry = signal.entry_reference_price
            sl = signal.stop_loss_reference
            tp1 = signal.take_profit_reference
            tp2 = signal.take_profit_2
            risk = abs(entry - sl)
            if risk <= 0:
                risk = 2.0

            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

            if signal.direction == "BUY":
                current_r = round((current_price - entry) / risk, 2)
                signal.current_r_multiple = current_r

                # Check Stop Loss
                if low <= sl or current_price <= sl:
                    signal.signal_status = "TRADE_COMPLETED_SL"
                    signal.show_chart_lines = False
                    signal.trade_ended_reason = f"Trade Ended: Stop Loss Hit at ${sl:.2f} (-1.0R)"
                    signal.trade_ended_at_utc = now_str
                    logger.info(f"Trade {signal.setup_id} ENDED: STOPPED OUT at ${sl:.2f}")

                # Check Take Profit 2 (Full Target Win)
                elif high >= tp2 or current_price >= tp2:
                    signal.signal_status = "TRADE_COMPLETED_TP"
                    signal.show_chart_lines = False
                    signal.trade_ended_reason = f"Trade Ended: Take Profit 2 Smashed at ${tp2:.2f} (+2.4R Won!)"
                    signal.trade_ended_at_utc = now_str
                    logger.info(f"Trade {signal.setup_id} ENDED: TP2 HIT at ${tp2:.2f}")

                # Check Take Profit 1
                elif high >= tp1 or current_price >= tp1:
                    signal.current_r_multiple = max(signal.current_r_multiple, 1.2)
                    signal.ai_recommendation_text = f"TP1 Reached (+1.2R Secured). Trailing stop to Breakeven (${entry:.2f}). Runner targeting TP2 (${tp2:.2f})."

            elif signal.direction == "SELL":
                current_r = round((entry - current_price) / risk, 2)
                signal.current_r_multiple = current_r

                # Check Stop Loss
                if high >= sl or current_price >= sl:
                    signal.signal_status = "TRADE_COMPLETED_SL"
                    signal.show_chart_lines = False
                    signal.trade_ended_reason = f"Trade Ended: Stop Loss Hit at ${sl:.2f} (-1.0R)"
                    signal.trade_ended_at_utc = now_str
                    logger.info(f"Trade {signal.setup_id} ENDED: STOPPED OUT at ${sl:.2f}")

                # Check Take Profit 2 (Full Target Win)
                elif low <= tp2 or current_price <= tp2:
                    signal.signal_status = "TRADE_COMPLETED_TP"
                    signal.show_chart_lines = False
                    signal.trade_ended_reason = f"Trade Ended: Take Profit 2 Smashed at ${tp2:.2f} (+2.4R Won!)"
                    signal.trade_ended_at_utc = now_str
                    logger.info(f"Trade {signal.setup_id} ENDED: TP2 HIT at ${tp2:.2f}")

                # Check Take Profit 1
                elif low <= tp1 or current_price <= tp1:
                    signal.current_r_multiple = max(signal.current_r_multiple, 1.2)
                    signal.ai_recommendation_text = f"TP1 Reached (+1.2R Secured). Trailing stop to Breakeven (${entry:.2f}). Runner targeting TP2 (${tp2:.2f})."

        self.save_state()
        return signal

    def lock_new_signal(
        self,
        prediction: TradePrediction,
        instrument: str,
        timeframe: str,
        live_price: float,
        atr: float = 3.50,
    ) -> ActiveSignalState:
        """Calculate and lock an institutional execution range setup."""
        key = self._get_key(instrument, timeframe)
        ts = int(prediction.prediction_timestamp or (datetime.now(timezone.utc).timestamp() * 1000))
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        setup_id = f"AI-{instrument.upper()}-{timeframe.upper()}-{ts}"

        p = round(live_price, 2)
        safe_atr = max(atr, 2.0)

        # -------------------------------------------------------------
        # Range Calculations based on Quantitative Confluence
        # -------------------------------------------------------------
        if prediction.direction == "SELL":
            # Recommend Short on retest of resistance / premium range above price
            range_low = round(p + (safe_atr * 0.30), 2)
            range_high = round(p + (safe_atr * 0.75), 2)
            trigger = round((range_low + range_high) / 2.0, 2)
            sl = round(range_high + (safe_atr * 1.0), 2)
            risk = round(sl - trigger, 2)
            tp1 = round(trigger - (risk * 1.2), 2)
            tp2 = round(trigger - (risk * 2.4), 2)
            recommendation = (
                f"AI Quantitative Model E triggers SELL BIAS ({int(prediction.confidence * 100)}% edge). "
                f"Recommend selling when price retests the ${range_low:.2f} ── ${range_high:.2f} range."
            )
            dist = round(max(0.0, trigger - p), 2)
            # If current price is already inside the range, trigger immediately!
            initial_status = "IN_POSITION" if p >= range_low else "PENDING_TRIGGER"

        elif prediction.direction == "BUY":
            # Recommend Long on pullback to support / discount range below price
            range_high = round(p - (safe_atr * 0.30), 2)
            range_low = round(p - (safe_atr * 0.75), 2)
            trigger = round((range_low + range_high) / 2.0, 2)
            sl = round(range_low - (safe_atr * 1.0), 2)
            risk = round(trigger - sl, 2)
            tp1 = round(trigger + (risk * 1.2), 2)
            tp2 = round(trigger + (risk * 2.4), 2)
            recommendation = (
                f"AI Quantitative Model E triggers BUY BIAS ({int(prediction.confidence * 100)}% edge). "
                f"Recommend buying when price pulls back into the ${range_low:.2f} ── ${range_high:.2f} range."
            )
            dist = round(max(0.0, p - trigger), 2)
            initial_status = "IN_POSITION" if p <= range_high else "PENDING_TRIGGER"

        else:
            range_low = p
            range_high = p
            trigger = p
            sl = round(p - safe_atr, 2)
            tp1 = round(p + (safe_atr * 1.2), 2)
            tp2 = round(p + (safe_atr * 2.4), 2)
            recommendation = "Market in Observation Mode. Awaiting high-probability structure confluence."
            dist = 0.0
            initial_status = "SCANNING"

        contributing = dict(prediction.contributing_features or {})
        contributing.update({
            "setup_id": setup_id,
            "entry_range_low": range_low,
            "entry_range_high": range_high,
            "trigger_price": trigger,
            "locked_sl": sl,
            "locked_tp1": tp1,
            "locked_tp2": tp2,
            "signal_status": initial_status,
            "show_chart_lines": (initial_status in ("PENDING_TRIGGER", "IN_POSITION")),
        })

        new_signal = ActiveSignalState(
            setup_id=setup_id,
            instrument=instrument.upper(),
            timeframe=timeframe.upper(),
            direction=prediction.direction,
            entry_reference_price=trigger,
            stop_loss_reference=sl,
            take_profit_reference=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=prediction.risk_reward_ratio,
            prediction_timestamp=ts,
            created_at_utc=now_utc,
            confidence=prediction.confidence,
            bullish_probability=prediction.bullish_probability,
            bearish_probability=prediction.bearish_probability,
            neutral_probability=prediction.neutral_probability,
            market_regime=prediction.market_regime,
            model_version=prediction.model_version,
            contributing_features=contributing,
            signal_status=initial_status,
            current_r_multiple=0.0,
            highest_price_seen=p,
            lowest_price_seen=p,
            is_setup_locked=True,
            bars_active=0,
            last_evaluated_price=p,
            entry_range_low=range_low,
            entry_range_high=range_high,
            trigger_price=trigger,
            show_chart_lines=(initial_status in ("PENDING_TRIGGER", "IN_POSITION")),
            trade_ended_reason=None,
            ai_recommendation_text=recommendation,
            distance_to_trigger=dist,
        )

        self._active_signals[key] = new_signal
        self.save_state()
        logger.info(
            f"Locked AI Range Recommendation {setup_id}: {prediction.direction} "
            f"Range: ${range_low:.2f}-${range_high:.2f} (Trigger: ${trigger:.2f}) | "
            f"SL: ${sl:.2f} | TP1: ${tp1:.2f} | Status: {initial_status}"
        )
        return new_signal

    def reset_signal(self, instrument: str, timeframe: str) -> None:
        """Clear active signal setup to allow immediate scanning for a fresh opportunity."""
        key = self._get_key(instrument, timeframe)
        if key in self._active_signals:
            del self._active_signals[key]
            self.save_state()
            logger.info(f"Reset active signal for {key}")


# Global Singleton Instance
_global_signal_manager: Optional[ActiveSignalManager] = None


def get_signal_manager() -> ActiveSignalManager:
    global _global_signal_manager
    if _global_signal_manager is None:
        _global_signal_manager = ActiveSignalManager()
    return _global_signal_manager
