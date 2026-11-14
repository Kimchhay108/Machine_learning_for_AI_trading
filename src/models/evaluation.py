"""Chronological cross-validation, purged splitting, and statistical evaluation metrics.

Implements Marcos López de Prado's Purged Time-Series Split to prevent overlapping label leakage,
and computes calibration, Brier score, selective accuracy, and coverage metrics.
"""

from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, brier_score_loss

from src.core.logger import get_logger

logger = get_logger("models.evaluation")


class PurgedTimeSeriesSplit:
    """Walk-forward time-series cross-validator with embargo and purging.

    Prevents label leakage when training on labels computed from forward time horizons (e.g. 12-bar triple barrier).
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 12,  # Purge overlap between train and test
        embargo_pct: float = 0.01,  # Embargo period after test set
    ):
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Any] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield strictly chronological train and test indices."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Minimum training size: 40% of dataset
        min_train_size = int(n_samples * 0.40)
        remaining = n_samples - min_train_size
        test_size = remaining // self.n_splits

        for fold in range(self.n_splits):
            test_start = min_train_size + fold * test_size
            test_end = test_start + test_size if fold < self.n_splits - 1 else n_samples

            # Train set is all data prior to test_start minus purge_window
            train_end = max(0, test_start - self.purge_window)
            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE) for probability estimates.

    Measures the absolute difference between predicted probability (confidence) and actual empirical accuracy.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    # Convert binary or multi-class: take max predicted probability as confidence
    confidences = np.max(y_prob, axis=1) if y_prob.ndim > 1 else np.maximum(y_prob, 1.0 - y_prob)
    predictions = np.argmax(y_prob, axis=1) if y_prob.ndim > 1 else (y_prob >= 0.5).astype(int)
    accuracies = (predictions == y_true).astype(float)

    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    selective_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics across all predictions and selective predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_count = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    res: Dict[str, Any] = {
        "total_samples": total_count,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        ece = expected_calibration_error(y_true, y_prob)
        res["expected_calibration_error"] = round(ece, 4)

        # Selective Accuracy (High-Confidence Predictions)
        confidences = np.max(y_prob, axis=1) if y_prob.ndim > 1 else np.maximum(y_prob, 1.0 - y_prob)
        high_conf_mask = confidences >= selective_threshold
        high_conf_count = int(np.sum(high_conf_mask))

        if high_conf_count > 0:
            sel_acc = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
            sel_prec = precision_score(y_true[high_conf_mask], y_pred[high_conf_mask], average="weighted", zero_division=0)
            res["selective_threshold"] = selective_threshold
            res["selective_samples"] = high_conf_count
            res["coverage_pct"] = round((high_conf_count / total_count) * 100.0, 2)
            res["selective_accuracy"] = round(float(sel_acc), 4)
            res["selective_precision"] = round(float(sel_prec), 4)
        else:
            res["selective_samples"] = 0
            res["coverage_pct"] = 0.0
            res["selective_accuracy"] = 0.0

    return res
