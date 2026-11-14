"""Unit tests for ContinuousRetrainingOrchestrator and champion-challenger promotion."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.core.config import settings
from src.models.ensemble import EnsemblePredictionModel
from src.retrain.orchestrator import ContinuousRetrainingOrchestrator


class TestContinuousRetraining:
    """Test challenger vs champion validation logic."""

    @pytest.fixture
    def real_data_slice(self):
        feat_path = settings.FEATURES_DIR / "xauusd" / "m5_features.parquet"
        lbl_path = settings.LABELS_DIR / "xauusd" / "m5_labels.parquet"

        if feat_path.exists() and lbl_path.exists():
            df_feats = pd.read_parquet(feat_path).tail(300)
            df_lbls = pd.read_parquet(lbl_path).tail(300)
            y = (df_lbls["label_dir_ternary_15m"] + 1).astype(int)
            drop_cols = ["timestamp", "instrument", "timeframe", "data_source"]
            X = df_feats[[c for c in df_feats.columns if c not in drop_cols]]
            return X, y
        else:
            pytest.skip("Historical features not generated yet.")

    def test_promotion_evaluation_rejection_low_coverage(self, tmp_path, real_data_slice):
        X, y = real_data_slice
        reg_dir = tmp_path / "registry"
        rep_dir = tmp_path / "reports"

        orchestrator = ContinuousRetrainingOrchestrator(
            min_coverage_pct=95.0,  # Unreasonably high coverage requirement -> should reject
            registry_dir=reg_dir,
            reports_dir=rep_dir,
        )

        challenger = EnsemblePredictionModel(confidence_threshold=0.80)
        challenger.fit(X.iloc[:200], y.iloc[:200])

        eval_res = orchestrator.evaluate_and_promote(
            challenger=challenger,
            challenger_version="v2.0.0-test",
            champion_version="v1.0.0-nonexistent",
            X_test=X.iloc[200:],
            y_test=y.iloc[200:],
        )

        assert not eval_res.promoted
        assert any("coverage" in r.lower() for r in eval_res.rejection_reasons)
        assert len(list(rep_dir.glob("*.json"))) == 1

    def test_promotion_evaluation_successful_promotion(self, tmp_path, real_data_slice):
        X, y = real_data_slice
        reg_dir = tmp_path / "registry"
        rep_dir = tmp_path / "reports"

        orchestrator = ContinuousRetrainingOrchestrator(
            min_selective_accuracy=0.20,
            min_coverage_pct=5.0,
            max_ece=0.80,
            registry_dir=reg_dir,
            reports_dir=rep_dir,
        )

        challenger = EnsemblePredictionModel(confidence_threshold=0.35)
        challenger.fit(X.iloc[:200], y.iloc[:200])

        eval_res = orchestrator.evaluate_and_promote(
            challenger=challenger,
            challenger_version="v2.0.0-promoted",
            champion_version="v1.0.0-nonexistent",
            X_test=X.iloc[200:],
            y_test=y.iloc[200:],
        )

        assert eval_res.promoted
        assert len(eval_res.rejection_reasons) == 0
        assert (reg_dir / "v2.0.0-promoted").exists()
        assert len(list(rep_dir.glob("*.json"))) == 1
