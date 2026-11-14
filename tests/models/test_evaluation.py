"""Unit tests for chronological cross-validation and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import PurgedTimeSeriesSplit, compute_metrics, expected_calibration_error


class TestEvaluation:
    def test_purged_time_series_split(self):
        n = 100
        df = pd.DataFrame({"feat": np.arange(n)})
        cv = PurgedTimeSeriesSplit(n_splits=3, purge_window=5)

        splits = list(cv.split(df))
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            # Chronological check: max train index must be strictly less than min test index
            assert np.max(train_idx) < np.min(test_idx)
            # Purge gap check: test_start - train_end must equal purge_window
            assert (np.min(test_idx) - np.max(train_idx)) == (5 + 1)

    def test_expected_calibration_error(self):
        # Perfect calibration
        y_true = np.array([1, 1, 0, 0])
        # Probabilities
        y_prob = np.array([
            [0.1, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.1],
        ])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert isinstance(ece, float)
        assert ece >= 0.0

    def test_compute_metrics_with_selective_threshold(self):
        y_true = np.array([2, 2, 0, 0, 1])
        y_pred = np.array([2, 2, 0, 1, 1])
        # 3-class probabilities
        y_prob = np.array([
            [0.1, 0.1, 0.8],  # High conf Bullish (2)
            [0.1, 0.1, 0.8],  # High conf Bullish (2)
            [0.8, 0.1, 0.1],  # High conf Bearish (0)
            [0.4, 0.3, 0.3],  # Low conf
            [0.2, 0.6, 0.2],  # Moderate conf Neutral (1)
        ])

        metrics = compute_metrics(y_true, y_pred, y_prob, selective_threshold=0.70)
        assert "accuracy" in metrics
        assert "selective_accuracy" in metrics
        assert "coverage_pct" in metrics
        assert metrics["selective_samples"] == 3  # exactly 3 samples with max_prob >= 0.70
        assert metrics["selective_accuracy"] == 1.0  # all 3 high-conf predictions were correct!
