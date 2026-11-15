"""Model G: Footprint & Order Flow Model using XGBoost.

Trains on institutional order flow delta mechanics:
- Aggressive Buy vs Sell Volume Delta proxy
- Cumulative Volume Delta (CVD) momentum and normalization
- CVD Divergences (Passive limit order absorption at extremes)
- Trapped Buyers at highs and Trapped Sellers at lows
- Wick Volume Absorption ratios
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.logger import get_logger

logger = get_logger("models.footprint")

FOOTPRINT_FEATURE_COLUMNS = [
    "delta_proxy", "delta_ratio", "cvd_rolling", "cvd_normalized",
    "cvd_divergence_bull", "cvd_divergence_bear",
    "trapped_buyers", "trapped_sellers", "absorption_ratio",
    "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "vol_ratio_20", "atr_14_pct", "rsi_14"
]


class FootprintModel:
    """XGBoost classifier focused on Footprint order flow, delta divergences, and trapped traders."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.feature_names = FOOTPRINT_FEATURE_COLUMNS
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="mlogloss",
            tree_method="hist",
        )
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and fill available footprint feature columns."""
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df[available_cols].copy()
        X = X.ffill().bfill().fillna(0.0)
        return X

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "FootprintModel":
        """Fit the Footprint XGBoost classifier."""
        X_clean = self._prepare_features(X_train)
        logger.info(f"Fitting FootprintModel on {len(X_clean)} samples with {X_clean.shape[1]} features...")
        self.model.fit(X_clean, y_train, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Output class probabilities [P(Neutral), P(Bullish), P(Bearish)]."""
        if not self.is_fitted:
            n = len(df)
            probs = np.zeros((n, 3))
            probs[:, 0] = 1.0
            return probs

        X_clean = self._prepare_features(df)
        raw_probs = self.model.predict_proba(X_clean)
        # Ensure 3-class shape
        if raw_probs.shape[1] == 3:
            return raw_probs
        elif raw_probs.shape[1] == 2:
            out = np.zeros((len(df), 3))
            out[:, 1:] = raw_probs
            return out
        else:
            out = np.zeros((len(df), 3))
            out[:, 0] = 1.0
            return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return discrete class prediction: 0=Bearish, 1=Neutral, 2=Bullish."""
        return np.argmax(self.predict_proba(df), axis=1)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importance scores mapped to feature names."""
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, [float(x) for x in importances]))

    def save(self, file_path: Path) -> Path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        logger.info(f"Saved FootprintModel to {file_path}")
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "FootprintModel":
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        instance = joblib.load(file_path)
        logger.info(f"Loaded FootprintModel from {file_path}")
        return instance
