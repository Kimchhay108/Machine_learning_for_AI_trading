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

        # Active Institutional ICT Models detected on latest candle
        ict_setups = []
        if latest_row.get("ict_2022_bullish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT 2022 Mentorship (Bullish FVG + Discount)")
        if latest_row.get("ict_2022_bearish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT 2022 Mentorship (Bearish FVG + Premium)")
        if latest_row.get("ict_silver_bullet_bullish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT Silver Bullet Macro (Bullish)")
        if latest_row.get("ict_silver_bullet_bearish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT Silver Bullet Macro (Bearish)")
        if latest_row.get("ict_judas_swing_bullish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT Judas Swing (London Open Bullish Reversal)")
        if latest_row.get("ict_judas_swing_bearish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT Judas Swing (London Open Bearish Reversal)")
        if latest_row.get("ict_turtle_soup_bullish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT Turtle Soup (Liquidity Run Rejection Long)")
        if latest_row.get("ict_turtle_soup_bearish", pd.Series([0])).iloc[0] == 1:
            ict_setups.append("ICT Turtle Soup (Liquidity Run Rejection Short)")

        # Individual Sub-Model Predictions & Probabilities
        sub_tech = int(self.ensemble.model_a.predict(latest_row)[0])
        sub_smc = int(self.ensemble.model_b.predict(latest_row)[0])
        sub_pa = int(self.ensemble.model_d.predict(latest_row)[0])

        tech_prob = self.ensemble.model_a.predict_proba(latest_row)[0]
        smc_prob = self.ensemble.model_b.predict_proba(latest_row)[0]
        pa_prob = self.ensemble.model_d.predict_proba(latest_row)[0]

        # -------------------------------------------------------------
        # Generate Next Entry, SL, TP, and 'Why' Explanations for ALL Models
        # -------------------------------------------------------------
        c_high = float(latest_row.get("high", pd.Series([entry_price])).iloc[0])
        c_low = float(latest_row.get("low", pd.Series([entry_price])).iloc[0])
        swing_l = float(latest_row.get("active_swing_low", pd.Series([0.0])).iloc[0])
        swing_h = float(latest_row.get("active_swing_high", pd.Series([0.0])).iloc[0])
        dealing_pos = str(latest_row.get("dealing_range_pos", pd.Series(["Neutral"])).iloc[0])
        rsi_val = round(float(latest_row.get("rsi_14", pd.Series([50.0])).iloc[0]), 1)
        macd_h = round(float(latest_row.get("macd_hist", pd.Series([0.0])).iloc[0]), 2)
        ema_align = int(latest_row.get("ema_alignment", pd.Series([0])).iloc[0])

        # 1. Model E (Ensemble) Setup
        agreed_bull = sum(1 for v in [sub_tech, sub_smc, sub_pa] if v == 2)
        agreed_bear = sum(1 for v in [sub_tech, sub_smc, sub_pa] if v == 0)

        # Strict Model E Condition: Requires Calibrated threshold AND multi-model agreement (>=2 sub-models)
        if decision == "BUY" and agreed_bull < 2:
            decision = "WAIT"
        elif decision == "SELL" and agreed_bear < 2:
            decision = "WAIT"

        if decision == "BUY":
            why_e = (
                f"Stacked meta-calibrator triggers BUY with {round(conf * 100, 1)}% confidence, "
                f"clearing the {round(self.ensemble.confidence_threshold * 100, 1)}% execution filter. "
                f"{agreed_bull}/3 sub-models agree on Bullish upside. Dynamic ATR Stop Loss at ${stop_loss} with 2.0 R:R Target at ${take_profit}."
            )
        elif decision == "SELL":
            why_e = (
                f"Stacked meta-calibrator triggers SELL with {round(conf * 100, 1)}% confidence, "
                f"clearing the {round(self.ensemble.confidence_threshold * 100, 1)}% execution filter. "
                f"{agreed_bear}/3 sub-models agree on Bearish downside. Dynamic ATR Stop Loss at ${stop_loss} with 2.0 R:R Target at ${take_profit}."
            )
        else:
            why_e = (
                f"Ensemble advises WAIT: directional confidence ({round(conf * 100, 1)}%) or multi-model agreement ({agreed_bull if decision=='BUY' else agreed_bear}/3) "
                f"did not meet strict entry confluence (Tech: {'BULL' if sub_tech==2 else ('BEAR' if sub_tech==0 else 'NEUT')}, "
                f"SMC: {'BULL' if sub_smc==2 else ('BEAR' if sub_smc==0 else 'NEUT')}, "
                f"PA: {'BULL' if sub_pa==2 else ('BEAR' if sub_pa==0 else 'NEUT')}). Preserving capital until all conditions align."
            )

        setup_ensemble = {
            "model_key": "ensemble",
            "model_name": "Model E: Calibrated Multi-Model Ensemble",
            "direction": decision,
            "entry_price": round(entry_price, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": self.default_rr_ratio,
            "risk_distance_usd": round(abs(entry_price - stop_loss), 2),
            "confidence_pct": round(conf * 100, 1),
            "why_explanation": why_e,
            "key_drivers": ["Meta-Model Stacking", f"Confidence {round(conf * 100, 1)}%", f"Regime: {regime_label}"],
        }

        # 2. Model B (SMC & ICT 2022 Mentorship) Setup
        has_ict_bull = (
            latest_row.get("ict_2022_bullish", pd.Series([0])).iloc[0] == 1 or
            latest_row.get("ict_silver_bullet_bullish", pd.Series([0])).iloc[0] == 1 or
            latest_row.get("ict_turtle_soup_bullish", pd.Series([0])).iloc[0] == 1
        )
        has_ict_bear = (
            latest_row.get("ict_2022_bearish", pd.Series([0])).iloc[0] == 1 or
            latest_row.get("ict_silver_bullet_bearish", pd.Series([0])).iloc[0] == 1 or
            latest_row.get("ict_turtle_soup_bearish", pd.Series([0])).iloc[0] == 1
        )
        p_b_bear, p_b_bull = float(smc_prob[0]), float(smc_prob[2])
        if has_ict_bull and (p_b_bull >= 0.40 and p_b_bull > p_b_bear):
            dir_b = "BUY"
            sl_b = round((swing_l - 0.50) if (swing_l > 0 and 0.50 <= (entry_price - swing_l) <= 4.0 * atr) else (entry_price - 1.5 * atr), 2)
            risk_b = max(entry_price - sl_b, 0.50)
            tp_b = round(entry_price + 2.5 * risk_b, 2)
            conf_b = round(p_b_bull * 100, 1)
            setups_str = ", ".join(ict_setups) if ict_setups else "Smart Money Order Flow"
            why_b = (
                f"Model B triggers BUY ({conf_b}% prob) on {setups_str}. "
                f"Price in {dealing_pos} zone. Structural Swing Invalidation SL set at ${sl_b} (-$0.50 buffer under ${swing_l:.2f}), "
                f"targeting opposing buy-side liquidity pool at ${tp_b} (2.5 R:R)."
            )
        elif has_ict_bear and (p_b_bear >= 0.40 and p_b_bear > p_b_bull):
            dir_b = "SELL"
            sl_b = round((swing_h + 0.50) if (swing_h > 0 and 0.50 <= (swing_h - entry_price) <= 4.0 * atr) else (entry_price + 1.5 * atr), 2)
            risk_b = max(sl_b - entry_price, 0.50)
            tp_b = round(entry_price - 2.5 * risk_b, 2)
            conf_b = round(p_b_bear * 100, 1)
            setups_str = ", ".join(ict_setups) if ict_setups else "Smart Money Institutional Distribution"
            why_b = (
                f"Model B triggers SELL ({conf_b}% prob) on {setups_str}. "
                f"Price in {dealing_pos} zone. Structural Swing Invalidation SL set at ${sl_b} (+$0.50 buffer above ${swing_h:.2f}), "
                f"targeting opposing sell-side liquidity pool at ${tp_b} (2.5 R:R)."
            )
        else:
            dir_b = "WAIT"
            sl_b = round(entry_price - 1.5 * atr, 2)
            tp_b = round(entry_price + 3.75 * atr, 2)
            conf_b = round(max(p_b_bull, p_b_bear) * 100, 1)
            why_b = (
                f"Model B is in WAIT: No active ICT 2022 Mentorship FVG or Silver Bullet macro window currently triggered. "
                f"Active swing range: Low ${swing_l:.2f} ── High ${swing_h:.2f} ({dealing_pos} dealing range). Awaiting liquidity sweep."
            )

        setup_smc = {
            "model_key": "smc_ict",
            "model_name": "Model B: SMC & ICT 2022 Mentorship",
            "direction": dir_b,
            "entry_price": round(entry_price, 2),
            "stop_loss": sl_b,
            "take_profit": tp_b,
            "risk_reward_ratio": 2.5,
            "risk_distance_usd": round(abs(entry_price - sl_b), 2),
            "confidence_pct": conf_b,
            "why_explanation": why_b,
            "key_drivers": ict_setups if ict_setups else [f"Dealing Range: {dealing_pos}", f"Swing Low: ${swing_l:.2f}"],
        }

        # 3. Model A (Technical Momentum - Indicators) Setup
        p_a_bear, p_a_bull = float(tech_prob[0]), float(tech_prob[2])
        has_tech_bull = (ema_align >= 1) and (latest_row.get("rsi_14_oversold", pd.Series([0])).iloc[0] == 1 or (rsi_val >= 35.0 and rsi_val <= 62.0))
        has_tech_bear = (ema_align <= -1) and (latest_row.get("rsi_14_overbought", pd.Series([0])).iloc[0] == 1 or (rsi_val >= 38.0 and rsi_val <= 65.0))
        if has_tech_bull and p_a_bull >= 0.40 and p_a_bull > p_a_bear:
            dir_a = "BUY"
            sl_a = round(entry_price - 1.5 * atr, 2)
            tp_a = round(entry_price + 3.0 * atr, 2)
            conf_a = round(p_a_bull * 100, 1)
            why_a = (
                f"Model A triggers BUY ({conf_a}% prob): Bullish EMA alignment (code: {ema_align}), "
                f"RSI(14) at {rsi_val} in healthy pullback expansion, MACD hist at {macd_h}. "
                f"Volatility envelope SL at ${sl_a} (1.5x ATR buffer: ${round(1.5*atr, 2)}), TP at ${tp_a} (2.0 R:R)."
            )
        elif has_tech_bear and p_a_bear >= 0.40 and p_a_bear > p_a_bull:
            dir_a = "SELL"
            sl_a = round(entry_price + 1.5 * atr, 2)
            tp_a = round(entry_price - 3.0 * atr, 2)
            conf_a = round(p_a_bear * 100, 1)
            why_a = (
                f"Model A triggers SELL ({conf_a}% prob): Bearish EMA alignment (code: {ema_align}), "
                f"RSI(14) at {rsi_val} showing downward expansion, MACD hist at {macd_h}. "
                f"Volatility envelope SL at ${sl_a} (1.5x ATR buffer: ${round(1.5*atr, 2)}), TP at ${tp_a} (2.0 R:R)."
            )
        else:
            dir_a = "WAIT"
            sl_a = round(entry_price - 1.5 * atr, 2)
            tp_a = round(entry_price + 3.0 * atr, 2)
            conf_a = round(max(p_a_bull, p_a_bear) * 100, 1)
            why_a = (
                f"Model A is in WAIT: Technical indicator conditions not met (RSI: {rsi_val}, MACD hist: {macd_h}, EMA alignment: {ema_align}). "
                f"Awaiting clear pullback or momentum expansion."
            )

        setup_tech = {
            "model_key": "technical",
            "model_name": "Model A: Technical Momentum (Indicators)",
            "direction": dir_a,
            "entry_price": round(entry_price, 2),
            "stop_loss": sl_a,
            "take_profit": tp_a,
            "risk_reward_ratio": 2.0,
            "risk_distance_usd": round(abs(entry_price - sl_a), 2),
            "confidence_pct": conf_a,
            "why_explanation": why_a,
            "key_drivers": [f"RSI(14): {rsi_val}", f"MACD Hist: {macd_h}", f"EMA Align: {ema_align}"],
        }

        # 4. Model D (Price Action & Rejections) Setup
        p_d_bear, p_d_bull = float(pa_prob[0]), float(pa_prob[2])
        upper_wick = c_high - max(entry_price, float(latest_row.get("open", pd.Series([entry_price])).iloc[0]))
        lower_wick = min(entry_price, float(latest_row.get("open", pd.Series([entry_price])).iloc[0])) - c_low
        has_pa_bull = (latest_row.get("pin_bar_bullish", pd.Series([0])).iloc[0] == 1 or latest_row.get("engulfing_bullish", pd.Series([0])).iloc[0] == 1 or float(latest_row.get("lower_wick_ratio", pd.Series([0.0])).iloc[0]) >= 0.35)
        has_pa_bear = (latest_row.get("pin_bar_bearish", pd.Series([0])).iloc[0] == 1 or latest_row.get("engulfing_bearish", pd.Series([0])).iloc[0] == 1 or float(latest_row.get("upper_wick_ratio", pd.Series([0.0])).iloc[0]) >= 0.35)

        if has_pa_bull and p_d_bull >= 0.40 and p_d_bull > p_d_bear:
            dir_d = "BUY"
            sl_d = round((c_low - 0.50) if 0.40 <= (entry_price - c_low) <= 3.0 * atr else (entry_price - 1.2 * atr), 2)
            risk_d = max(entry_price - sl_d, 0.40)
            tp_d = round(entry_price + 2.0 * risk_d, 2)
            conf_d = round(p_d_bull * 100, 1)
            why_d = (
                f"Model D triggers BUY ({conf_d}% prob): Candlestick displays lower wick rejection (${lower_wick:.2f}) absorbing sellers off low ${c_low:.2f}. "
                f"Tight wick SL at ${sl_d} (-$0.50 buffer), TP at ${tp_d} (2.0 R:R)."
            )
        elif has_pa_bear and p_d_bear >= 0.40 and p_d_bear > p_d_bull:
            dir_d = "SELL"
            sl_d = round((c_high + 0.50) if 0.40 <= (c_high - entry_price) <= 3.0 * atr else (entry_price + 1.2 * atr), 2)
            risk_d = max(sl_d - entry_price, 0.40)
            tp_d = round(entry_price - 2.0 * risk_d, 2)
            conf_d = round(p_d_bear * 100, 1)
            why_d = (
                f"Model D triggers SELL ({conf_d}% prob): Candlestick displays upper wick rejection (${upper_wick:.2f}) rejecting high ${c_high:.2f}. "
                f"Tight wick SL at ${sl_d} (+$0.50 buffer), TP at ${tp_d} (2.0 R:R)."
            )
        else:
            dir_d = "WAIT"
            sl_d = round(entry_price - 1.2 * atr, 2)
            tp_d = round(entry_price + 2.4 * atr, 2)
            conf_d = round(max(p_d_bull, p_d_bear) * 100, 1)
            why_d = (
                f"Model D is in WAIT: Candlestick morphology (High ${c_high:.2f} / Low ${c_low:.2f}) lacks decisive wick rejection or pin bar formation. "
                f"Waiting for extreme wick test."
            )

        setup_pa = {
            "model_key": "price_action",
            "model_name": "Model D: Price Action & Rejections",
            "direction": dir_d,
            "entry_price": round(entry_price, 2),
            "stop_loss": sl_d,
            "take_profit": tp_d,
            "risk_reward_ratio": 2.0,
            "risk_distance_usd": round(abs(entry_price - sl_d), 2),
            "confidence_pct": conf_d,
            "why_explanation": why_d,
            "key_drivers": [f"High: ${c_high:.2f}", f"Low: ${c_low:.2f}", f"ATR: ${atr:.2f}"],
        }

        all_setups = [setup_ensemble, setup_smc, setup_tech, setup_pa]

        contributing = {
            "top_technical_drivers": top_tech,
            "top_smc_drivers": top_smc,
            "active_ict_setups": ict_setups,
            "all_model_setups": all_setups,
            "submodel_agreement": {
                "tech_prediction": sub_tech,
                "smc_prediction": sub_smc,
                "price_action_prediction": sub_pa,
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
