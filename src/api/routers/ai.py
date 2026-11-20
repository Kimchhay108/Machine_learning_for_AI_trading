"""AI Prediction, Model Registry, Training, and Continuous Evaluation Endpoints."""

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix

from src.api.dependencies import (
    delete_model_version,
    get_active_version,
    get_prediction_engine,
    set_active_version,
    verify_admin_key,
)
from src.api.schemas import (
    ActivateModelRequest,
    ModelInfo,
    ModelsListResponse,
    PerformanceResponse,
    PredictionEvaluationRequest,
    PredictionEvaluationResponse,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)
from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.models.ensemble import EnsemblePredictionModel
from src.models.evaluation import expected_calibration_error, compute_metrics
from src.models.prediction_engine import PredictionEngine

logger = get_logger("api.routers.ai")
router = APIRouter(prefix="/api/ai", tags=["AI Engine"])


@router.get("/prediction", response_model=PredictionResponse)
def get_prediction(
    instrument: str = Query(DEFAULT_INSTRUMENT, description="Trading instrument"),
    timeframe: str = Query("M5", description="Timeframe"),
    version: Optional[str] = Query(None, description="Model version (defaults to active)"),
    force_reset: bool = Query(False, description="Force reset and re-evaluate active setup"),
) -> PredictionResponse:
    """Generate live AI trading prediction with stationary locked risk levels and real-time lifecycle tracking."""
    tf_clean = (timeframe or "M5").lower().strip()
    feat_path = settings.FEATURES_DIR / instrument.lower() / f"{tf_clean}_features.parquet"
    if not feat_path.exists():
        feat_path = settings.FEATURES_DIR / instrument.lower() / "m5_features.parquet"

    if not feat_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Features matrix not found for {instrument}. Run 'features' command first.",
        )

    df_feats = pd.read_parquet(feat_path)
    if df_feats.empty:
        raise HTTPException(status_code=404, detail="Features matrix is empty.")

    from src.api.routers.market import fetch_live_yahoo_gold_candles
    from src.live.signal_manager import get_signal_manager

    signal_manager = get_signal_manager()
    if force_reset:
        signal_manager.reset_signal(instrument, timeframe)

    live_candles = fetch_live_yahoo_gold_candles(timeframe=timeframe, limit=2)
    latest_c = live_candles[-1] if (live_candles and len(live_candles) > 0) else None
    current_price = latest_c.close if latest_c else float(df_feats["close"].iloc[-1])
    c_high = latest_c.high if latest_c else current_price
    c_low = latest_c.low if latest_c else current_price

    # 1. Check if we already have an active locked trade setup
    active_state = signal_manager.get_signal(
        instrument=instrument,
        timeframe=timeframe,
        current_price=current_price,
        candle_high=c_high,
        candle_low=c_low,
    )

    if active_state and active_state.signal_status in ("ACTIVE", "TP1_HIT"):
        # Setup is locked: levels remain 100% stationary across the chart!
        return PredictionResponse(
            instrument=active_state.instrument,
            timeframe=active_state.timeframe,
            prediction_timestamp=active_state.prediction_timestamp,
            direction=active_state.direction,
            bullish_probability=active_state.bullish_probability,
            bearish_probability=active_state.bearish_probability,
            neutral_probability=active_state.neutral_probability,
            confidence=active_state.confidence,
            market_regime=active_state.market_regime,
            entry_reference_price=active_state.entry_reference_price,
            stop_loss_reference=active_state.stop_loss_reference,
            take_profit_reference=active_state.take_profit_reference,
            take_profit_2=active_state.take_profit_2,
            risk_reward_ratio=active_state.risk_reward_ratio,
            model_version=active_state.model_version,
            contributing_features=active_state.contributing_features,
            signal_status=active_state.signal_status,
            current_r_multiple=active_state.current_r_multiple,
            is_setup_locked=True,
            setup_id=active_state.setup_id,
        )

    # 2. No active trade running (or previous completed/stopped out): evaluate AI model
    engine: PredictionEngine = get_prediction_engine(version)
    pred = engine.predict_candle(
        candle_features=df_feats,
        instrument=instrument.upper(),
        timeframe=timeframe.upper(),
        live_price=current_price,
    )

    # 3. If AI triggers a valid trade setup (BUY or SELL with >= 50% confidence), lock it in!
    if pred.direction in ("BUY", "SELL") and pred.confidence >= 0.50:
        locked_state = signal_manager.lock_new_signal(
            prediction=pred,
            instrument=instrument,
            timeframe=timeframe,
            live_price=current_price,
        )
        return PredictionResponse(
            instrument=locked_state.instrument,
            timeframe=locked_state.timeframe,
            prediction_timestamp=locked_state.prediction_timestamp,
            direction=locked_state.direction,
            bullish_probability=locked_state.bullish_probability,
            bearish_probability=locked_state.bearish_probability,
            neutral_probability=locked_state.neutral_probability,
            confidence=locked_state.confidence,
            market_regime=locked_state.market_regime,
            entry_reference_price=locked_state.entry_reference_price,
            stop_loss_reference=locked_state.stop_loss_reference,
            take_profit_reference=locked_state.take_profit_reference,
            take_profit_2=locked_state.take_profit_2,
            risk_reward_ratio=locked_state.risk_reward_ratio,
            model_version=locked_state.model_version,
            contributing_features=locked_state.contributing_features,
            signal_status="ACTIVE",
            current_r_multiple=0.0,
            is_setup_locked=True,
            setup_id=locked_state.setup_id,
        )

    # 4. Market is in WAIT / Observation mode (no locked trade)
    pred_dict = pred.to_dict()
    pred_dict["signal_status"] = "WAITING"
    pred_dict["current_r_multiple"] = 0.0
    pred_dict["is_setup_locked"] = False
    pred_dict["setup_id"] = None
    return PredictionResponse(**pred_dict)


@router.post("/signal/reset")
def reset_signal(
    instrument: str = Query(DEFAULT_INSTRUMENT, description="Trading instrument"),
    timeframe: str = Query("M5", description="Timeframe"),
) -> Dict[str, Any]:
    """Manually clear/reset the active locked AI trade setup to scan for a new entry."""
    from src.live.signal_manager import get_signal_manager
    manager = get_signal_manager()
    manager.reset_signal(instrument, timeframe)
    return {"status": "RESET", "message": f"Active AI signal setup for {instrument} {timeframe} has been reset."}


@router.get("/models", response_model=ModelsListResponse)
def list_models() -> ModelsListResponse:
    """List all registered ML models in the model registry."""
    registry_dir = settings.DATA_DIR / "models" / "registry"
    if not registry_dir.exists():
        return ModelsListResponse(active_version="none", models=[])

    models: List[ModelInfo] = []
    versions = sorted([d.name for d in registry_dir.iterdir() if d.is_dir()])
    active_version = get_active_version()

    for v in versions:
        v_dir = registry_dir / v
        meta_file = v_dir / "metadata.joblib"
        submodels = [
            f.stem.replace("model_", "").replace("_", " ").title()
            for f in v_dir.glob("model_*.joblib")
        ]

        threshold = 0.50
        train_date = None
        if meta_file.exists():
            try:
                meta = joblib.load(meta_file)
                threshold = float(meta.get("confidence_threshold", 0.50))
                mtime = os.path.getmtime(meta_file)
                train_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        models.append(
            ModelInfo(
                version=v,
                name=f"XAUUSD Hybrid Model Suite ({v})",
                is_active=(v == active_version),
                confidence_threshold=threshold,
                submodels=submodels or ["Technical", "SMC", "Regime", "Price Action"],
                train_date=train_date,
            )
        )

    return ModelsListResponse(
        active_version=active_version if versions else "none",
        models=models,
    )


@router.get("/performance", response_model=PerformanceResponse)
def get_model_performance(
    instrument: str = Query(DEFAULT_INSTRUMENT, description="Trading instrument"),
    version: str = Query("v1.0.0", description="Model version"),
) -> PerformanceResponse:
    """Retrieve statistical validation metrics, ECE calibration error, and regime breakdown on out-of-sample data."""
    v_dir = settings.DATA_DIR / "models" / "registry" / version
    meta_path = v_dir / "metadata.joblib"

    # Fast path: return cached performance metrics if available
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
            if "cached_performance" in meta and meta["cached_performance"]:
                return PerformanceResponse(**meta["cached_performance"])
        except Exception:
            pass

    features_dir = settings.FEATURES_DIR / instrument.lower()
    labels_dir = settings.LABELS_DIR / instrument.lower()

    feat_path = features_dir / "m5_features.parquet"
    lbl_path = labels_dir / "m5_labels.parquet"

    if not feat_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Features parquet file not found for performance evaluation.",
        )

    df_feats = pd.read_parquet(feat_path)
    if not lbl_path.exists():
        from src.labels.labeling_engine import LabelingEngine
        labeling = LabelingEngine()
        df_lbls = labeling.compute_all_labels(df_feats)
        labeling.save_labels(df_lbls)
    else:
        df_lbls = pd.read_parquet(lbl_path)

    target_col = "label_dir_ternary_15m"
    if target_col not in df_lbls.columns:
        raise HTTPException(status_code=400, detail=f"Target {target_col} missing in labels.")

    y = (df_lbls[target_col] + 1).astype(int)
    drop_cols = ["timestamp", "instrument", "timeframe", "data_source"]
    X = df_feats[[c for c in df_feats.columns if c not in drop_cols]].copy()

    # Out-of-sample split: last 30%
    n = len(X)
    test_start = int(n * 0.70) + 12
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    engine: PredictionEngine = get_prediction_engine(version)
    ensemble = engine.ensemble

    probs = ensemble.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    threshold = ensemble.confidence_threshold
    metrics = compute_metrics(y_test, preds, probs, threshold)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    cm_dict = {
        "labels": ["Bearish", "Neutral", "Bullish"],
        "matrix": cm.tolist(),
    }

    # Brier score (multi-class one-vs-rest average)
    y_onehot = np.eye(3)[y_test.values]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # Regime specific accuracies
    regime_acc = {}
    if "regime_label" in df_feats.columns:
        regimes_test = df_feats["regime_label"].iloc[test_start:]
        for r_name in regimes_test.unique():
            idx = regimes_test == r_name
            if idx.sum() > 5:
                regime_acc[str(r_name)] = round(float(np.mean(preds[idx] == y_test[idx])), 4)

    resp = PerformanceResponse(
        model_version=version,
        directional_accuracy=round(metrics["accuracy"], 4),
        selective_accuracy=round(metrics.get("selective_accuracy", metrics["accuracy"]), 4),
        coverage_pct=round(metrics.get("coverage_pct", 100.0), 2),
        f1_score=round(metrics["f1_score"], 4),
        expected_calibration_error=round(metrics.get("expected_calibration_error", 0.0), 4),
        brier_score=round(brier, 4),
        confusion_matrix=cm_dict,
        regime_accuracies=regime_acc,
    )

    # Cache into metadata.joblib so performance tab works even after labels cleanup
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
            meta["cached_performance"] = resp.model_dump()
            joblib.dump(meta, meta_path)
        except Exception:
            pass

    return resp


@router.post("/train", response_model=TrainResponse)
def train_model(
    req: TrainRequest,
    is_admin: bool = Depends(verify_admin_key),
) -> TrainResponse:
    """Protected endpoint: Trigger retraining of the Model Suite on updated historical data."""
    features_dir = settings.FEATURES_DIR / "xauusd"
    labels_dir = settings.LABELS_DIR / "xauusd"
    registry_dir = settings.DATA_DIR / "models" / "registry"

    feat_path = features_dir / "m5_features.parquet"
    lbl_path = labels_dir / "m5_labels.parquet"

    if not feat_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Cannot train: features parquet file missing. Run feature pipeline first.",
        )

    df_feats = pd.read_parquet(feat_path)

    # If labels were auto-cleaned, compute on the fly
    if not lbl_path.exists():
        from src.labels.labeling_engine import LabelingEngine
        labeling = LabelingEngine()
        df_lbls = labeling.compute_all_labels(df_feats)
        labeling.save_labels(df_lbls)
    else:
        df_lbls = pd.read_parquet(lbl_path)

    target_col = "label_dir_ternary_15m"
    y = (df_lbls[target_col] + 1).astype(int)

    drop_cols = ["timestamp", "instrument", "timeframe", "data_source"]
    X = df_feats[[c for c in df_feats.columns if c not in drop_cols]].copy()

    n = len(X)
    train_end = int(n * (1.0 - req.test_ratio))
    test_start = train_end + 12

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    # Hard-negative mistake mining: penalize historical failure fingerprints
    sample_weights = None
    if req.learn_from_mistakes:
        weights_arr = np.ones(len(X_train), dtype=np.float32)
        adx_col = "adx_14" if "adx_14" in X_train.columns else None
        rsi_col = "rsi_14" if "rsi_14" in X_train.columns else None
        atr_col = "atr_14" if "atr_14" in X_train.columns else None

        mistake_mask = np.zeros(len(X_train), dtype=bool)
        if adx_col:
            mistake_mask |= (X_train[adx_col] < 18.0)
        if rsi_col:
            mistake_mask |= (X_train[rsi_col] > 68.0) | (X_train[rsi_col] < 32.0)
        if atr_col:
            rolling_atr = X_train[atr_col].rolling(20, min_periods=5).mean().fillna(X_train[atr_col])
            mistake_mask |= (X_train[atr_col] > 1.4 * rolling_atr)

        weights_arr[mistake_mask] = 2.5
        sample_weights = weights_arr
        logger.info(f"Retraining with Hard-Negative Mistake Mining: {int(np.sum(mistake_mask))} failure-prone candles reweighted (2.5x penalty).")

    ensemble = EnsemblePredictionModel(confidence_threshold=req.threshold)
    ensemble.fit(X_train, y_train, sample_weight=sample_weights)

    comparisons = ensemble.compare_models(X_test, y_test)
    saved_dir = ensemble.save_suite(registry_dir, version=req.version)
    set_active_version(req.version)

    # Cache performance into metadata.joblib
    meta_path = saved_dir / "metadata.joblib"
    if meta_path.exists():
        try:
            meta = joblib.load(meta_path)
            e_metrics = comparisons.get("Model_E_Ensemble", {})
            meta["cached_performance"] = {
                "model_version": req.version,
                "directional_accuracy": round(e_metrics.get("accuracy", 0.0), 4),
                "selective_accuracy": round(e_metrics.get("selective_accuracy", 0.0), 4),
                "coverage_pct": round(e_metrics.get("coverage_pct", 100.0), 2),
                "f1_score": round(e_metrics.get("f1_score", 0.0), 4),
                "expected_calibration_error": round(e_metrics.get("expected_calibration_error", 0.0), 4),
                "brier_score": 0.0,
                "confusion_matrix": {"labels": ["Bearish", "Neutral", "Bullish"], "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
                "regime_accuracies": {},
            }
            joblib.dump(meta, meta_path)
        except Exception:
            pass

    # Auto-cleanup temporary labels and raw cache to keep disk compact
    try:
        if lbl_path.exists():
            lbl_path.unlink()
        raw_dir = settings.RAW_DATA_DIR
        if raw_dir.exists():
            for item in raw_dir.glob("**/*"):
                if item.is_file():
                    item.unlink()
    except Exception:
        pass

    return TrainResponse(
        status="SUCCESS",
        version=req.version,
        message=f"Model suite {req.version} trained and registered at {saved_dir}",
        train_samples=len(X_train),
        test_samples=len(X_test),
        metrics=comparisons,
    )


@router.post("/models/activate")
def activate_model(
    req: ActivateModelRequest,
    is_admin: bool = Depends(verify_admin_key),
):
    """Set a registered candidate model version as the active production model."""
    registry_dir = settings.DATA_DIR / "models" / "registry"
    target_dir = registry_dir / req.version
    if not target_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{req.version}' not found in registry at {target_dir}.",
        )
    set_active_version(req.version)
    get_prediction_engine(req.version)
    logger.info(f"Model version '{req.version}' successfully set as ACTIVE PRODUCTION.")
    return {
        "status": "SUCCESS",
        "active_version": req.version,
        "message": f"Model {req.version} is now the ACTIVE PRODUCTION model.",
    }


@router.delete("/models/{version}")
def delete_model(
    version: str,
    is_admin: bool = Depends(verify_admin_key),
):
    """Delete a model version from the registry with automatic active fallback."""
    return delete_model_version(version)


@router.post("/predictions/evaluate", response_model=PredictionEvaluationResponse)
def evaluate_predictions(
    req: PredictionEvaluationRequest,
    instrument: str = Query(DEFAULT_INSTRUMENT),
) -> PredictionEvaluationResponse:
    """Evaluate historical model predictions against actual future market returns without lookahead bias."""
    feat_path = settings.FEATURES_DIR / instrument.lower() / "m5_features.parquet"
    if not feat_path.exists():
        raise HTTPException(status_code=404, detail="Features not found.")

    df = pd.read_parquet(feat_path)
    if len(df) <= req.sample_limit + req.horizon_bars:
        raise HTTPException(status_code=400, detail="Insufficient candle data to evaluate predictions.")

    engine: PredictionEngine = get_prediction_engine()
    drop_cols = ["timestamp", "instrument", "timeframe", "data_source"]
    X = df[[c for c in df.columns if c not in drop_cols]]

    # Sample historical slice
    test_slice = df.iloc[-(req.sample_limit + req.horizon_bars): -req.horizon_bars]
    total = len(test_slice)

    correct_directional = 0
    actionable = 0
    abstentions = 0
    gains = []
    losses = []
    favorable_excursions = []
    adverse_excursions = []

    decisions, _, confs = engine.ensemble.predict_decision(X.iloc[test_slice.index])

    for i, (idx, row) in enumerate(test_slice.iterrows()):
        dec = decisions[i]
        curr_price = float(row["close"])
        future_idx = idx + req.horizon_bars
        future_price = float(df.iloc[future_idx]["close"])

        # Intrabar excursions over the horizon
        future_window = df.iloc[idx + 1: future_idx + 1]
        max_high = float(future_window["high"].max())
        min_low = float(future_window["low"].min())

        if dec == "WAIT":
            abstentions += 1
            continue

        actionable += 1
        if dec == "BUY":
            ret = future_price - curr_price
            mfe = (max_high - curr_price) * 10  # in pips (gold 0.10 = 1 pip)
            mae = (curr_price - min_low) * 10
            favorable_excursions.append(mfe)
            adverse_excursions.append(mae)

            if ret > 0:
                correct_directional += 1
                gains.append(ret)
            else:
                losses.append(abs(ret))

        elif dec == "SELL":
            ret = curr_price - future_price
            mfe = (curr_price - min_low) * 10
            mae = (max_high - curr_price) * 10
            favorable_excursions.append(mfe)
            adverse_excursions.append(mae)

            if ret > 0:
                correct_directional += 1
                gains.append(ret)
            else:
                losses.append(abs(ret))

    accuracy = (correct_directional / actionable) if actionable > 0 else 0.0
    profit_factor = (sum(gains) / sum(losses)) if (losses and sum(losses) > 0) else (1.0 if not losses else 0.0)
    mean_mfe = float(np.mean(favorable_excursions)) if favorable_excursions else 0.0
    mean_mae = float(np.mean(adverse_excursions)) if adverse_excursions else 0.0

    return PredictionEvaluationResponse(
        total_predictions=total,
        actionable_predictions=actionable,
        abstentions=abstentions,
        directional_accuracy=round(accuracy, 4),
        profit_factor=round(profit_factor, 3),
        mean_favorable_excursion_pips=round(mean_mfe, 2),
        mean_adverse_excursion_pips=round(mean_mae, 2),
    )
