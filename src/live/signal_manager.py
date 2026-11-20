"""Institutional AI Signal Lifecycle and State Manager.

Locks AI trade setups (Entry, SL, TP1, TP2) so that price lines remain
100% stationary and never drift/adjust with tick noise or new bars while a trade is active.
Tracks trade progress in real-time (Floating R, TP1 Hit, TP2 Hit, Stopped Out).
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
    direction: str                     # "BUY" or "SELL"
    entry_reference_price: float       # 100% LOCKED
    stop_loss_reference: float         # 100% LOCKED
    take_profit_reference: float       # 100% LOCKED
    take_profit_2: float               # 100% LOCKED
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
    signal_status: str = "ACTIVE"      # "ACTIVE", "TP1_HIT", "TP2_HIT", "STOPPED_OUT", "EXPIRED"
    current_r_multiple: float = 0.0
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0
    is_setup_locked: bool = True
    bars_active: int = 0
    last_evaluated_price: float = 0.0

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
        """Retrieve the current active signal state and evaluate it against live market price."""
        key = self._get_key(instrument, timeframe)
        signal = self._active_signals.get(key)
        if not signal:
            return None

        # If trade is already closed / terminal, do not evaluate further
        if signal.signal_status in ("TP2_HIT", "STOPPED_OUT", "EXPIRED"):
            return signal

        if current_price is None or current_price <= 0:
            return signal

        # Track intrabar extremes
        high = candle_high if candle_high is not None else current_price
        low = candle_low if candle_low is not None else current_price
        signal.last_evaluated_price = round(current_price, 2)
        signal.highest_price_seen = max(signal.highest_price_seen or current_price, high)
        signal.lowest_price_seen = min(signal.lowest_price_seen or current_price, low)

        entry = signal.entry_reference_price
        sl = signal.stop_loss_reference
        tp1 = signal.take_profit_reference
        tp2 = signal.take_profit_2
        risk = abs(entry - sl)
        if risk <= 0:
            risk = 2.0

        if signal.direction == "BUY":
            current_r = round((current_price - entry) / risk, 2)
            signal.current_r_multiple = current_r

            # Check Stop Loss Invalidation
            if low <= sl or current_price <= sl:
                signal.signal_status = "STOPPED_OUT"
                logger.info(f"Signal {signal.setup_id} STOPPED OUT at ${sl:.2f} (-1.0R)")
            # Check Take Profit 2 Full Target
            elif high >= tp2 or current_price >= tp2:
                signal.signal_status = "TP2_HIT"
                logger.info(f"Signal {signal.setup_id} TP2 REACHED at ${tp2:.2f} (+2.4R)")
            # Check Take Profit 1 Target
            elif high >= tp1 or current_price >= tp1:
                if signal.signal_status == "ACTIVE":
                    signal.signal_status = "TP1_HIT"
                    logger.info(f"Signal {signal.setup_id} TP1 REACHED at ${tp1:.2f} (+1.2R)")

        elif signal.direction == "SELL":
            current_r = round((entry - current_price) / risk, 2)
            signal.current_r_multiple = current_r

            # Check Stop Loss Invalidation
            if high >= sl or current_price >= sl:
                signal.signal_status = "STOPPED_OUT"
                logger.info(f"Signal {signal.setup_id} STOPPED OUT at ${sl:.2f} (-1.0R)")
            # Check Take Profit 2 Full Target
            elif low <= tp2 or current_price <= tp2:
                signal.signal_status = "TP2_HIT"
                logger.info(f"Signal {signal.setup_id} TP2 REACHED at ${tp2:.2f} (+2.4R)")
            # Check Take Profit 1 Target
            elif low <= tp1 or current_price <= tp1:
                if signal.signal_status == "ACTIVE":
                    signal.signal_status = "TP1_HIT"
                    logger.info(f"Signal {signal.setup_id} TP1 REACHED at ${tp1:.2f} (+1.2R)")

        self.save_state()
        return signal

    def lock_new_signal(
        self,
        prediction: TradePrediction,
        instrument: str,
        timeframe: str,
        live_price: float,
    ) -> ActiveSignalState:
        """Lock in a brand new AI trade setup. Levels are 100% frozen from this moment onwards."""
        key = self._get_key(instrument, timeframe)
        ts = int(prediction.prediction_timestamp or (datetime.now(timezone.utc).timestamp() * 1000))
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        setup_id = f"AI-{instrument.upper()}-{timeframe.upper()}-{ts}"

        entry = round(live_price, 2)
        risk = abs(prediction.entry_reference_price - prediction.stop_loss_reference)
        if risk <= 0:
            risk = 3.50

        # Maintain exact R:R distance anchored to initial entry
        if prediction.direction == "BUY":
            sl = round(entry - risk, 2)
            tp1 = round(entry + (risk * prediction.risk_reward_ratio), 2)
            tp2 = round(entry + (risk * prediction.risk_reward_ratio * 2.0), 2)
        elif prediction.direction == "SELL":
            sl = round(entry + risk, 2)
            tp1 = round(entry - (risk * prediction.risk_reward_ratio), 2)
            tp2 = round(entry - (risk * prediction.risk_reward_ratio * 2.0), 2)
        else:
            sl = prediction.stop_loss_reference
            tp1 = prediction.take_profit_reference
            tp2 = prediction.take_profit_2 or round(entry + (risk * 2.4), 2)

        contributing = dict(prediction.contributing_features or {})
        contributing.update({
            "setup_id": setup_id,
            "locked_entry": entry,
            "locked_sl": sl,
            "locked_tp1": tp1,
            "locked_tp2": tp2,
            "signal_status": "ACTIVE",
            "is_setup_locked": True,
        })

        new_signal = ActiveSignalState(
            setup_id=setup_id,
            instrument=instrument.upper(),
            timeframe=timeframe.upper(),
            direction=prediction.direction,
            entry_reference_price=entry,
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
            signal_status="ACTIVE",
            current_r_multiple=0.0,
            highest_price_seen=entry,
            lowest_price_seen=entry,
            is_setup_locked=True,
            bars_active=0,
            last_evaluated_price=entry,
        )

        self._active_signals[key] = new_signal
        self.save_state()
        logger.info(f"Locked new stationary AI setup {setup_id}: {prediction.direction} @ ${entry:.2f} | SL: ${sl:.2f} | TP1: ${tp1:.2f} | TP2: ${tp2:.2f}")
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
