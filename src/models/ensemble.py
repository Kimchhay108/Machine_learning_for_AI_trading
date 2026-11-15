"""Model E: Calibrated Multi-Model Ensemble and Selective Abstention.

Fuses probability distributions from:
- Model A: Technical Analysis (XGBoost)
- Model B: Smart Money Concepts (LightGBM)
- Model C: Market Regime (Random Forest)
- Model D: Price Action (XGBoost)

Applies regime-conditioned probability weighting, confidence calibration,
and a selective prediction threshold that allows the AI to abstain (WAIT) during noise.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.core.logger import get_logger
from src.models.evaluation import compute_metrics
from src.models.price_action_model import PriceActionModel
from src.models.regime_model import MarketRegimeModel
from src.models.smc_model import SmartMoneyConceptsModel
from src.models.technical_model import TechnicalAnalysisModel
from src.models.volume_profile_model import VolumeProfileModel
from src.models.footprint_model import FootprintModel

logger = get_logger("models.ensemble")


class EnsemblePredictionModel:
    """Ensemble prediction model fusing specialized technical, SMC, regime, and price action models."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        model_a: Optional[TechnicalAnalysisModel] = None,
        model_b: Optional[SmartMoneyConceptsModel] = None,
        model_c: Optional[MarketRegimeModel] = None,
        model_d: Optional[PriceActionModel] = None,
        model_f: Optional[VolumeProfileModel] = None,
        model_g: Optional[FootprintModel] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.model_a = model_a or TechnicalAnalysisModel()
        self.model_b = model_b or SmartMoneyConceptsModel()
        self.model_c = model_c or MarketRegimeModel()
        self.model_d = model_d or PriceActionModel()
        self.model_f = model_f or VolumeProfileModel()
        self.model_g = model_g or FootprintModel()

        self.calibrator: Optional[LogisticRegression] = None
        self.is_fitted = False

        # Baseline weights across models (equal weight prior)
        self.weights = {"tech": 0.25, "smc": 0.25, "pa": 0.20, "vp": 0.15, "footprint": 0.15}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        regime_labels: Optional[pd.Series] = None,
    ) -> "EnsemblePredictionModel":
        """Train all individual models and fit ensemble meta-calibrator."""
        logger.info(f"Training complete model suite on {len(X_train)} samples...")

        # 1. Fit Individual Sub-Models (with sample_weight for hard-negative mistake learning)
        self.model_a.fit(X_train, y_train, sample_weight=sample_weight)
        self.model_b.fit(X_train, y_train, sample_weight=sample_weight)
        self.model_d.fit(X_train, y_train, sample_weight=sample_weight)
        self.model_f.fit(X_train, y_train, sample_weight=sample_weight)
        self.model_g.fit(X_train, y_train, sample_weight=sample_weight)

        # Fit Regime model if regime labels provided
        if regime_labels is not None:
            self.model_c.fit(X_train, regime_labels)
        elif "market_regime" in X_train.columns:
            self.model_c.fit(X_train, X_train["market_regime"])

        # 2. Extract probability features for meta-learner
        prob_a = self.model_a.predict_proba(X_train)
        prob_b = self.model_b.predict_proba(X_train)
        prob_d = self.model_d.predict_proba(X_train)

        meta_X = np.hstack([prob_a, prob_b, prob_d])

        # 3. Fit Logistic Regression Calibrator (Platt Scaling / Stacking)
        self.calibrator = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=42,
        )
        self.calibrator.fit(meta_X, y_train, sample_weight=sample_weight)
        self.is_fitted = True
        logger.info("Ensemble model suite and probability calibrator successfully trained.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Produce calibrated probabilities [P(Bearish), P(Neutral), P(Bullish)]."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted.")

        prob_a = self.model_a.predict_proba(X)
        prob_b = self.model_b.predict_proba(X)
        prob_d = self.model_d.predict_proba(X)

        if self.calibrator is not None:
            n_in = getattr(self.calibrator, "n_features_in_", 9)
            if n_in == 15:
                prob_f = self.model_f.predict_proba(X)
                prob_g = self.model_g.predict_proba(X)
                meta_X = np.hstack([prob_a, prob_b, prob_d, prob_f, prob_g])
            else:
                meta_X = np.hstack([prob_a, prob_b, prob_d])
            calibrated_probs = self.calibrator.predict_proba(meta_X)
            return calibrated_probs

        # Fallback to weighted average
        weighted = (
            self.weights["tech"] * prob_a +
            self.weights["smc"] * prob_b +
            self.weights["pa"] * prob_d
        )
        return weighted / np.sum(weighted, axis=1, keepdims=True)

    def predict_decision(
        self,
        X: pd.DataFrame,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Generate BUY, SELL, or WAIT decisions based on selective confidence threshold.

        Returns:
            Tuple of:
            - decisions: List of "BUY", "SELL", or "WAIT" strings
            - calibrated_probs: np.ndarray shape (N, 3) for [Bearish, Neutral, Bullish]
            - confidence_scores: np.ndarray shape (N,) containing max directional confidence
        """
        probs = self.predict_proba(X)
        p_bear = probs[:, 0]
        p_neut = probs[:, 1]
        p_bull = probs[:, 2]

        # Directional confidence is max(p_bull, p_bear)
        dir_confidence = np.maximum(p_bull, p_bear)

        decisions = []
        for i in range(len(X)):
            conf = dir_confidence[i]
            if conf < self.confidence_threshold:
                decisions.append("WAIT")
            elif p_bull[i] > p_bear[i]:
                decisions.append("BUY")
            else:
                decisions.append("SELL")

        return decisions, probs, dir_confidence

    def compare_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Compare ensemble performance against individual sub-models on identical out-of-sample data."""
        results = {}

        # Model A
        pred_a = self.model_a.predict(X_test)
        prob_a = self.model_a.predict_proba(X_test)
        results["Model_A_Technical"] = compute_metrics(y_test, pred_a, prob_a, self.confidence_threshold)

        # Model B
        pred_b = self.model_b.predict(X_test)
        prob_b = self.model_b.predict_proba(X_test)
        results["Model_B_SMC"] = compute_metrics(y_test, pred_b, prob_b, self.confidence_threshold)

        # Model D
        pred_d = self.model_d.predict(X_test)
        prob_d = self.model_d.predict_proba(X_test)
        results["Model_D_PriceAction"] = compute_metrics(y_test, pred_d, prob_d, self.confidence_threshold)

        # Model E Ensemble
        prob_e = self.predict_proba(X_test)
        pred_e = np.argmax(prob_e, axis=1)
        results["Model_E_Ensemble"] = compute_metrics(y_test, pred_e, prob_e, self.confidence_threshold)

        return results

    def save_suite(self, registry_dir: Path, version: str = "v1.0.0") -> Path:
        """Save all model artifacts in the versioned registry directory."""
        v_dir = registry_dir / version
        v_dir.mkdir(parents=True, exist_ok=True)

        self.model_a.save(v_dir / "model_a_technical.joblib")
        self.model_b.save(v_dir / "model_b_smc.joblib")
        self.model_c.save(v_dir / "model_c_regime.joblib")
        self.model_d.save(v_dir / "model_d_price_action.joblib")
        self.model_f.save(v_dir / "model_f_volume_profile.joblib")
        self.model_g.save(v_dir / "model_g_footprint.joblib")

        if self.calibrator is not None:
            joblib.dump(self.calibrator, v_dir / "ensemble_calibrator.joblib")

        meta = {
            "version": version,
            "confidence_threshold": self.confidence_threshold,
            "weights": self.weights,
        }
        joblib.dump(meta, v_dir / "metadata.joblib")
        logger.info(f"Model Suite {version} saved to {v_dir}")
        return v_dir

    @classmethod
    def load_suite(cls, registry_dir: Path, version: str = "v1.0.0") -> "EnsemblePredictionModel":
        v_dir = registry_dir / version
        meta = joblib.load(v_dir / "metadata.joblib")
        instance = cls(confidence_threshold=meta.get("confidence_threshold", 0.55))
        instance.model_a = TechnicalAnalysisModel.load(v_dir / "model_a_technical.joblib")
        instance.model_b = SmartMoneyConceptsModel.load(v_dir / "model_b_smc.joblib")
        instance.model_c = MarketRegimeModel.load(v_dir / "model_c_regime.joblib")
        instance.model_d = PriceActionModel.load(v_dir / "model_d_price_action.joblib")

        vp_path = v_dir / "model_f_volume_profile.joblib"
        if vp_path.exists():
            instance.model_f = VolumeProfileModel.load(vp_path)
        else:
            instance.model_f = VolumeProfileModel()

        fp_path = v_dir / "model_g_footprint.joblib"
        if fp_path.exists():
            instance.model_g = FootprintModel.load(fp_path)
        else:
            instance.model_g = FootprintModel()

        cal_path = v_dir / "ensemble_calibrator.joblib"
        if cal_path.exists():
            instance.calibrator = joblib.load(cal_path)

        instance.weights = meta.get("weights", instance.weights)
        instance.is_fitted = True
        return instance
