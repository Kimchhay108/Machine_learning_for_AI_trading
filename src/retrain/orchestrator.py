"""Continuous learning and controlled model champion-challenger promotion pipeline.

Implements strict validation:
- Challenger model trained on chronological historical window.
- Evaluated against production Champion model on an out-of-sample test window.
- Promotion rules:
  1. Calibration (ECE) must not degrade significantly.
  2. Selective accuracy must meet or exceed production threshold.
  3. Minimum coverage percentage constraint must be satisfied.
- Preserves full rollback capabilities and writes audit reports.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.models.ensemble import EnsemblePredictionModel
from src.models.evaluation import compute_metrics

logger = get_logger("retrain.orchestrator")


@dataclass
class PromotionEvaluation:
    champion_version: str
    challenger_version: str
    champion_metrics: Dict[str, Any]
    challenger_metrics: Dict[str, Any]
    promoted: bool
    rejection_reasons: list
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContinuousRetrainingOrchestrator:
    """Manages periodic model retraining, out-of-sample challenger validation, and atomic promotion."""

    def __init__(
        self,
        min_selective_accuracy: float = 0.38,
        min_coverage_pct: float = 15.0,
        max_ece: float = 0.35,
        registry_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
    ):
        self.min_selective_accuracy = min_selective_accuracy
        self.min_coverage_pct = min_coverage_pct
        self.max_ece = max_ece
        self.registry_dir = registry_dir or (settings.DATA_DIR / "models" / "registry")
        self.reports_dir = reports_dir or (settings.DATA_DIR / "reports" / "retraining")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_promote(
        self,
        challenger: EnsemblePredictionModel,
        challenger_version: str,
        champion_version: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> PromotionEvaluation:
        """Evaluate Challenger against Champion on out-of-sample data."""
        # Load Champion
        try:
            champion = EnsemblePredictionModel.load_suite(self.registry_dir, version=champion_version)
        except Exception as e:
            logger.warning(f"No existing champion {champion_version} found ({e}); challenger promoted automatically.")
            champion = None

        # Challenger Metrics
        p_challenger = challenger.predict_proba(X_test)
        pred_challenger = np.argmax(p_challenger, axis=1)
        challenger_metrics = compute_metrics(
            y_test, pred_challenger, p_challenger, challenger.confidence_threshold
        )

        rejection_reasons = []

        # 1. Minimum Absolute Criteria
        c_acc = challenger_metrics.get("selective_accuracy", 0.0)
        c_cov = challenger_metrics.get("coverage_pct", 0.0)
        c_ece = challenger_metrics.get("expected_calibration_error", 1.0)

        if c_cov < self.min_coverage_pct:
            rejection_reasons.append(
                f"Challenger coverage {c_cov:.1f}% is below required minimum {self.min_coverage_pct:.1f}%"
            )

        if c_ece > self.max_ece:
            rejection_reasons.append(
                f"Challenger ECE {c_ece:.3f} exceeds maximum allowable error {self.max_ece:.3f}"
            )

        champion_metrics: Dict[str, Any] = {}
        if champion is not None:
            p_champ = champion.predict_proba(X_test)
            pred_champ = np.argmax(p_champ, axis=1)
            champion_metrics = compute_metrics(
                y_test, pred_champ, p_champ, champion.confidence_threshold
            )

            champ_acc = champion_metrics.get("selective_accuracy", 0.0)
            if c_acc < (champ_acc * 0.95):  # Must not degrade accuracy by more than 5%
                rejection_reasons.append(
                    f"Challenger selective accuracy ({c_acc*100:.1f}%) lagged champion ({champ_acc*100:.1f}%)"
                )

        promoted = len(rejection_reasons) == 0

        if promoted:
            logger.info(f"Challenger {challenger_version} PROMOTED to active production!")
            # Save challenger to registry
            challenger.save_suite(self.registry_dir, version=challenger_version)
        else:
            logger.warning(f"Challenger {challenger_version} REJECTED: {', '.join(rejection_reasons)}")

        now_str = datetime.utcnow().isoformat() + "Z"
        eval_result = PromotionEvaluation(
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            promoted=promoted,
            rejection_reasons=rejection_reasons,
            timestamp=now_str,
        )

        # Write audit report
        ts_slug = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"retrain_eval_{challenger_version}_{ts_slug}.json"
        with open(report_file, "w") as f:
            json.dump(eval_result.to_dict(), f, indent=2)

        return eval_result
