"""Model C: Market Regime Model using Scikit-Learn Random Forest.

Classifies trend efficiency, volatility compression (squeeze), and volatility expansion.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.core.logger import get_logger

logger = get_logger("models.regime")

REGIME_FEATURE_COLUMNS = [
    "efficiency_ratio_20", "trend_slope_20", "atr_percentile",
    "is_high_volatility", "is_low_volatility", "bbw_percentile",
    "volatility_squeeze", "volatility_expansion",
    "session_asian", "session_london", "session_ny", "session_london_ny_overlap"
]


class MarketRegimeModel:
    """Scikit-Learn Random Forest classifier for market regime identification."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        self.feature_names = REGIME_FEATURE_COLUMNS
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df[available_cols].copy()
        X = X.ffill().bfill().fillna(0.0)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MarketRegimeModel":
        """Fit Random Forest model on regime features.

        y should be: 0 (Ranging), 1 (Bull Trend), 2 (Bear Trend), 3 (High Vol Expansion).
        """
        X_clean = self._prepare_features(X)
        logger.info(f"Training Model C (Regime) on {len(X_clean)} samples with {len(X_clean.columns)} features...")
        self.model.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities: [P(Ranging), P(Bull), P(Bear), P(Expansion)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict discrete regime class (0, 1, 2, 3)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict(X_clean)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        cols = [c for c in self.feature_names if c in self.model.feature_names_in_]
        res = {k: float(v) for k, v in zip(cols, importances)}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    def save(self, file_path: Path) -> Path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, file_path)
        logger.info(f"Model C saved to {file_path}")
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "MarketRegimeModel":
        data = joblib.load(file_path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["features"]
        instance.is_fitted = True
        return instance
