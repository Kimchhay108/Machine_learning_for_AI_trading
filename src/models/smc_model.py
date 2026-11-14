"""Model B: Smart Money Concepts (SMC) Model using LightGBM.

Trains on institutional order flow footprints:
- Market Structure Shifts (BOS / CHoCH)
- Liquidity Sweeps and Equal Highs / Lows
- Fair Value Gaps and Order Blocks
- Dealing Range equilibrium, Premium / Discount / OTE Zones
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("models.smc")

SMC_FEATURE_COLUMNS = [
    "market_structure_trend", "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
    "liquidity_sweep_bullish", "liquidity_sweep_bearish", "equal_highs", "equal_lows",
    "fvg_bullish_new", "fvg_bearish_new", "fvg_bullish_unmitigated_count", "fvg_bearish_unmitigated_count",
    "order_block_bullish", "order_block_bearish",
    "dealing_range_pos", "is_discount_zone", "is_premium_zone", "is_ote_long_zone", "is_ote_short_zone",
    "session_asian", "session_london", "session_ny", "session_london_ny_overlap",
    "swept_pdh", "swept_pdl"
]


class SmartMoneyConceptsModel:
    """LightGBM classifier focused on institutional market structure and SMC patterns."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        num_leaves: int = 31,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.feature_names = SMC_FEATURE_COLUMNS
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="multiclass",
            verbosity=-1,
        )
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available_cols = [c for c in self.feature_names if c in df.columns]
        X = df[available_cols].copy()
        X = X.ffill().bfill().fillna(0.0)
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "SmartMoneyConceptsModel":
        """Fit LightGBM model on SMC features.

        y should be integer-encoded: 0 (Bearish), 1 (Neutral), 2 (Bullish).
        """
        X_clean = self._prepare_features(X)
        logger.info(f"Training Model B (SMC) on {len(X_clean)} samples with {len(X_clean.columns)} features...")

        prepared_eval_set = None
        if eval_set:
            prepared_eval_set = [(self._prepare_features(ex), ey) for (ex, ey) in eval_set]

        self.model.fit(X_clean, y, eval_set=prepared_eval_set)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities: [P(Bearish), P(Neutral), P(Bullish)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict_proba(X_clean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict discrete class index: 0, 1, 2."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_clean = self._prepare_features(X)
        return self.model.predict(X_clean)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dict of feature importances sorted descending."""
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        cols = [c for c in self.feature_names if c in self.model.feature_name_]
        res = {k: float(v) for k, v in zip(cols, importances)}
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    def save(self, file_path: Path) -> Path:
        """Serialize model to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, file_path)
        logger.info(f"Model B saved to {file_path}")
        return file_path

    @classmethod
    def load(cls, file_path: Path) -> "SmartMoneyConceptsModel":
        """Deserialize model from file."""
        data = joblib.load(file_path)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data["features"]
        instance.is_fitted = True
        return instance
