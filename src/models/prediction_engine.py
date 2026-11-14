"""End-to-end Inference and Trade Recommendation Engine.

Produces validated, calibrated trading signals adhering strictly to the contract in Section 8:
- BUY, SELL, or WAIT decisions
- Directional calibrated probabilities
- Dynamic ATR-based Entry, Stop-Loss, and Take-Profit calculations
- Explicit Risk/Reward ratio
- Model agreement and top contributing feature explanations
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.models.ensemble import EnsemblePredictionModel

logger = get_logger("models.prediction_engine")


@dataclass
class TradePrediction:
    """Structured contract for an individual trading signal."""
    instrument: str
    timeframe: str
    prediction_timestamp: int
    direction: str                     # "BUY", "SELL", or "WAIT"
    bullish_probability: float         # P(Bullish)
    bearish_probability: float         # P(Bearish)
    neutral_probability: float         # P(Neutral)
    confidence: float                  # max(P_bull, P_bear)
    market_regime: str                 # "Ranging", "Bullish_Trend", "Bearish_Trend", etc.
    entry_reference_price: float       # Reference current market price
    stop_loss_reference: float         # Structural / Volatility Stop-Loss level
    take_profit_reference: float       # Take-Profit level
    risk_reward_ratio: float           # Target Risk/Reward multiple (e.g. 2.0)
    model_version: str                 # Version string
    contributing_features: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PredictionEngine:
    """Inference engine executing real-time predictions and generating risk management parameters."""

    def __init__(
        self,
        ensemble: EnsemblePredictionModel,
        model_version: str = "v1.0.0",
        default_rr_ratio: float = 2.0,
        sl_atr_multiplier: float = 1.5,
    ):
        self.ensemble = ensemble
        self.model_version = model_version
        self.default_rr_ratio = default_rr_ratio
        self.sl_atr_multiplier = sl_atr_multiplier

    def predict_candle(
        self,
        candle_features: pd.DataFrame,
        instrument: str = DEFAULT_INSTRUMENT,
        timeframe: str = Timeframe.M5.value,
    ) -> TradePrediction:
        """Generate prediction and trading levels for the latest candle."""
        if candle_features.empty:
            raise ValueError("candle_features DataFrame cannot be empty.")

        latest_row = candle_features.iloc[[-1]]
        decisions, probs, confs = self.ensemble.predict_decision(latest_row)

        decision = decisions[0]
        prob = probs[0]
        conf = float(confs[0])

        p_bear = float(prob[0])
        p_neut = float(prob[1])
        p_bull = float(prob[2])

        ts = int(latest_row["timestamp"].iloc[0])
        entry_price = float(latest_row["close"].iloc[0])

        # ATR for dynamic risk levels
        if "atr_14" in latest_row.columns:
            atr = float(latest_row["atr_14"].iloc[0])
        else:
            atr = float(latest_row.get("true_range", pd.Series([2.0])).iloc[0])
        if np.isnan(atr) or atr <= 0:
            atr = 2.0  # safe default for gold ($2.00)

        sl_distance = self.sl_atr_multiplier * atr
        tp_distance = sl_distance * self.default_rr_ratio

        if decision == "BUY":
            stop_loss = round(entry_price - sl_distance, 3)
            take_profit = round(entry_price + tp_distance, 3)
        elif decision == "SELL":
            stop_loss = round(entry_price + sl_distance, 3)
            take_profit = round(entry_price - tp_distance, 3)
        else:  # WAIT
            stop_loss = round(entry_price - sl_distance, 3)
            take_profit = round(entry_price + tp_distance, 3)

        regime_label = str(latest_row.get("regime_label", pd.Series(["Neutral"])).iloc[0])

        # Top Contributing Explanations (from sub-models)
        tech_imp = self.ensemble.model_a.get_feature_importances()
        top_tech = list(tech_imp.keys())[:3] if tech_imp else []

        smc_imp = self.ensemble.model_b.get_feature_importances()
        top_smc = list(smc_imp.keys())[:3] if smc_imp else []

        contributing = {
            "top_technical_drivers": top_tech,
            "top_smc_drivers": top_smc,
            "submodel_agreement": {
                "tech_prediction": int(self.ensemble.model_a.predict(latest_row)[0]),
                "smc_prediction": int(self.ensemble.model_b.predict(latest_row)[0]),
                "price_action_prediction": int(self.ensemble.model_d.predict(latest_row)[0]),
            }
        }

        return TradePrediction(
            instrument=instrument,
            timeframe=timeframe,
            prediction_timestamp=ts,
            direction=decision,
            bullish_probability=round(p_bull, 4),
            bearish_probability=round(p_bear, 4),
            neutral_probability=round(p_neut, 4),
            confidence=round(conf, 4),
            market_regime=regime_label,
            entry_reference_price=round(entry_price, 3),
            stop_loss_reference=stop_loss,
            take_profit_reference=take_profit,
            risk_reward_ratio=self.default_rr_ratio,
            model_version=self.model_version,
            contributing_features=contributing,
        )
