"""Model A: Technical Analysis Model using XGBoost.

Trains strictly on mathematical indicators, momentum oscillators, moving averages, and volume dynamics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.core.logger import get_logger

logger = get_logger("models.technical")

TECHNICAL_FEATURE_COLUMNS = [
    "ema_9_dist_pct", "ema_20_dist_pct", "ema_50_dist_pct", "ema_100_dist_pct", "ema_200_dist_pct",
    "ema_9_slope", "ema_20_slope", "ema_50_slope", "ema_alignment",
    "atr_14_pct", "rsi_14", "rsi_14_norm", "rsi_14_overbought", "rsi_14_oversold",
    "macd_line", "macd_signal", "macd_hist", "macd_hist_pct", "macd_hist_slope",
    "bb_pct_b", "bb_bandwidth",
    "plus_di_14", "minus_di_14", "adx_14", "adx_14_trending",
    "stoch_rsi_k", "stoch_rsi_d", "stoch_rsi_diff",
    "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "roc_5", "roc_10", "roc_20",
    "vol_ratio_20", "vwap_dist_pct"
]


class TechnicalAnalysisModel:
    """XGBoost model focused on classical technical indicators and momentum."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.feature_names = TECHNICAL_FEATURE_COLUMNS
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
        """Filter and fill available technical feature columns."""
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df[available_cols].copy()
        # Impute NaNs with median/0
        X = X.ffill().bfill().fillna(0.0)
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "TechnicalAnalysisModel":
        """Fit XGBoost model on technical features.

        y should be integer-encoded: 0 (Bearish), 1 (Neutral), 2 (Bullish).
        """
        X_clean = self._prepare_features(X)
        logger.info(f"Training Model A (Technical Analysis) on {len(X_clean)} samples with {len(X_clean.columns)} features...")

        prepared_eval_set = None
        if eval_set:
            prepared_eval_set = [(self._prepare_features(ex), ey) for (ex, ey) in eval_set]

        self.model.fit(X_clean, y, sample_weight=sample_weight, eval_set=prepared_eval_set, verbose=False)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities: [P(Bearish), P(Neutral), P(Bullish)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict discrete class index: 0 (Bearish), 1 (Neutral), 2 (Bullish)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict(X_clean)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dict of feature importances sorted descending."""
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        cols = [c for c in self.feature_names if c in self.model.feature_names_in_]
        res = {k: float(v) for k, v in zip(cols, importances)}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    def save(self, file_path: Path) -> Path:
        """Serialize model to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, file_path)
        logger.info(f"Model A saved to {file_path}")
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "TechnicalAnalysisModel":
        """Deserialize model from file."""
        data = joblib.load(file_path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["features"]
        instance.is_fitted = True
        return instance
