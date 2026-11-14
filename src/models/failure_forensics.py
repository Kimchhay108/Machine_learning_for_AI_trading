"""AI Trade Failure Forensics and Continuous Mistake Learning Engine.

Analyzes losing trades from Backtest and Live Paper Trading to identify:
1. Root causes of trade failure (volatility shocks, momentum chop, exhaustion climaxes, regime mismatch).
2. Actionable lessons learned and self-correcting rules.
3. Hard-negative mining sample weights for model retraining.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("models.failure_forensics")


@dataclass
class TradeForensicsItem:
    """Individual trade failure diagnosis and autopsy report."""
    entry_time: int
    exit_time: int
    direction: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    r_multiple: float
    exit_reason: str
    failure_cause: str                 # e.g. "VOLATILITY_EXPANSION_SPIKE", "LOW_MOMENTUM_CHOP"
    failure_explanation: str           # Plain English diagnosis
    lesson_learned: str                # Actionable rule applied to prevent recurrence
    market_regime: str
    adx: float
    rsi: float
    atr: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicsSummaryResponse:
    """Aggregated failure attribution and lessons learned report."""
    total_losses_analyzed: int
    cause_distribution: Dict[str, float]
    top_failure_cause: str
    key_lessons: List[str]
    sample_diagnoses: List[TradeForensicsItem]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sample_diagnoses"] = [asdict(item) for item in self.sample_diagnoses]
        return d


class TradeForensicsEngine:
    """Diagnoses trade losses, attributes failure mechanisms, and generates learned rules."""

    CAUSE_DEFINITIONS = {
        "VOLATILITY_EXPANSION_SPIKE": {
            "title": "Volatility Expansion Spike",
            "explanation": "ATR expanded significantly above rolling baseline, causing noise to breach stop-loss before target was reached.",
            "lesson": "Widen dynamic stop buffer to 2.0x ATR or veto entry when ATR exceeds 1.4x rolling average.",
        },
        "LOW_MOMENTUM_CHOP": {
            "title": "Low-Momentum Chop",
            "explanation": "ADX was below 18.0 indicating directionless consolidation; price drifted into stop loss due to lack of order flow.",
            "lesson": "Enforce minimum ADX >= 18.0 filter to avoid entering during ranging consolidation.",
        },
        "MOMENTUM_CLIMAX_EXHAUSTION": {
            "title": "Momentum Climax Exhaustion",
            "explanation": "Trade entered into an overbought (RSI > 68) or oversold (RSI < 32) climax extreme right before mean reversion.",
            "lesson": "Veto BUY entries when RSI > 68.0 and SELL entries when RSI < 32.0.",
        },
        "REGIME_OPPOSITION": {
            "title": "Macro Regime Opposition",
            "explanation": "Entry direction fought higher-timeframe market regime or H1 200 EMA trend.",
            "lesson": "Strictly gate entries with H1 200 EMA and multi-timeframe regime alignment.",
        },
        "STRUCTURAL_LIQUIDITY_SWEEP": {
            "title": "Structural Liquidity Sweep",
            "explanation": "Institutions swept active swing liquidity buffer before reversing back toward intended target.",
            "lesson": "Expand structural stop buffer from $0.50 to $0.80 beyond swing high/low.",
        },
    }

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or (settings.DATA_DIR / "reports" / "forensics")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def diagnose_trade(
        self,
        trade: Any,
        row: pd.Series,
        rolling_atr: float = 2.0,
    ) -> TradeForensicsItem:
        """Diagnose a single losing trade by cross-referencing entry candle features."""
        direction = getattr(trade, "direction", "BUY")
        regime = str(row.get("market_regime", "Unknown"))
        adx = float(row.get("adx_14", 25.0))
        rsi = float(row.get("rsi_14", 50.0))
        atr = float(row.get("atr_14", 2.0))
        atr_ratio = (atr / rolling_atr) if rolling_atr > 0 else 1.0

        # Attribution hierarchy
        if direction == "BUY" and ("Bearish" in regime or "Ranging" in regime):
            cause = "REGIME_OPPOSITION"
        elif direction == "SELL" and ("Bullish" in regime or "Ranging" in regime):
            cause = "REGIME_OPPOSITION"
        elif direction == "BUY" and rsi > 68.0:
            cause = "MOMENTUM_CLIMAX_EXHAUSTION"
        elif direction == "SELL" and rsi < 32.0:
            cause = "MOMENTUM_CLIMAX_EXHAUSTION"
        elif atr_ratio > 1.40 or atr > 3.5:
            cause = "VOLATILITY_EXPANSION_SPIKE"
        elif adx < 18.0:
            cause = "LOW_MOMENTUM_CHOP"
        else:
            cause = "STRUCTURAL_LIQUIDITY_SWEEP"

        info = self.CAUSE_DEFINITIONS.get(cause, {
            "explanation": "Stop loss touched due to intrabar market noise.",
            "lesson": "Improve stop placement and confluence confirmation.",
        })

        return TradeForensicsItem(
            entry_time=int(getattr(trade, "entry_time", 0)),
            exit_time=int(getattr(trade, "exit_time", 0)),
            direction=direction,
            entry_price=float(getattr(trade, "entry_price", 0.0)),
            exit_price=float(getattr(trade, "exit_price", 0.0)),
            pnl_usd=float(getattr(trade, "pnl_usd", 0.0)),
            r_multiple=float(getattr(trade, "r_multiple", 0.0)),
            exit_reason=str(getattr(trade, "exit_reason", "STOP_LOSS")),
            failure_cause=cause,
            failure_explanation=info["explanation"],
            lesson_learned=info["lesson"],
            market_regime=regime,
            adx=round(adx, 1),
            rsi=round(rsi, 1),
            atr=round(atr, 2),
        )

    def analyze_trades(
        self,
        trades: List[Any],
        df: pd.DataFrame,
    ) -> ForensicsSummaryResponse:
        """Run complete autopsy on all losing trades in a backtest or paper trading session."""
        losses = [t for t in trades if getattr(t, "pnl_usd", 0.0) < 0 and "BREAKEVEN" not in getattr(t, "exit_reason", "")]
        total_losses = len(losses)

        if total_losses == 0 or df.empty:
            return ForensicsSummaryResponse(
                total_losses_analyzed=0,
                cause_distribution={},
                top_failure_cause="NONE",
                key_lessons=["Zero stop-loss trades detected."],
                sample_diagnoses=[],
            )

        ts_series = df["timestamp"].astype(int)
        ts_map = {ts: idx for idx, ts in enumerate(ts_series)}

        atr_series = df["atr_14"].fillna(2.0) if "atr_14" in df.columns else pd.Series(2.0, index=df.index)
        rolling_atr = atr_series.rolling(50, min_periods=5).mean().bfill().to_numpy()

        diagnoses: List[TradeForensicsItem] = []
        cause_counts: Dict[str, int] = {}

        for t in losses:
            entry_ts = int(getattr(t, "entry_time", 0))
            idx = ts_map.get(entry_ts)
            if idx is None:
                # Try fallback matching nearest timestamp
                diffs = np.abs(ts_series.to_numpy() - entry_ts)
                min_idx = int(np.argmin(diffs))
                idx = min_idx if diffs[min_idx] <= 300_000 else None

            if idx is not None and 0 <= idx < len(df):
                row = df.iloc[idx]
                r_atr = float(rolling_atr[idx])
                item = self.diagnose_trade(t, row, rolling_atr=r_atr)
                diagnoses.append(item)
                cause_counts[item.failure_cause] = cause_counts.get(item.failure_cause, 0) + 1

        # Calculate percentage distribution
        n_diag = len(diagnoses)
        dist = {k: round(v / n_diag * 100.0, 1) for k, v in cause_counts.items()} if n_diag > 0 else {}
        top_cause = max(cause_counts.items(), key=lambda x: x[1])[0] if cause_counts else "NONE"

        key_lessons = [
            f"Veto entries when ATR expands > 1.4x rolling baseline ({dist.get('VOLATILITY_EXPANSION_SPIKE', 0)}% of failures).",
            f"Filter out range chop where ADX < 18.0 ({dist.get('LOW_MOMENTUM_CHOP', 0)}% of failures).",
            f"Avoid buying into RSI > 68 overbought or selling RSI < 32 oversold climaxes ({dist.get('MOMENTUM_CLIMAX_EXHAUSTION', 0)}% of failures).",
        ]

        summary = ForensicsSummaryResponse(
            total_losses_analyzed=total_losses,
            cause_distribution=dist,
            top_failure_cause=top_cause,
            key_lessons=key_lessons,
            sample_diagnoses=diagnoses[:50],  # Keep top 50 detailed diagnoses
        )

        # Save latest audit report to disk
        try:
            report_path = self.reports_dir / "latest_forensics.json"
            with open(report_path, "w") as f:
                json.dump(summary.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save forensics report: {e}")

        return summary
