"""Model D: Price Action Model using XGBoost.

Trains on candlestick morphology, wick rejection ratios, engulfing patterns, inside bars, and streaks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.logger import get_logger

logger = get_logger("models.price_action")

PRICE_ACTION_FEATURE_COLUMNS = [
    "body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "is_bullish_candle", "is_bearish_candle",
    "pin_bar_bullish", "pin_bar_bearish",
    "engulfing_bullish", "engulfing_bearish",
    "inside_bar", "breakout_high_20", "breakout_low_20",
    "retest_high_20", "retest_low_20",
    "streak_bullish", "streak_bearish"
]


class PriceActionModel:
    """XGBoost classifier focused on short-term price action dynamics."""

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.feature_names = PRICE_ACTION_FEATURE_COLUMNS
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
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df[available_cols].copy()
        X = X.ffill().bfill().fillna(0.0)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriceActionModel":
        X_clean = self._prepare_features(X)
        logger.info(f"Training Model D (Price Action) on {len(X_clean)} samples with {len(X_clean.columns)} features...")
        self.model.fit(X_clean, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
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
        logger.info(f"Model D saved to {file_path}")
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "PriceActionModel":
        data = joblib.load(file_path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["features"]
        instance.is_fitted = True
        return instance
