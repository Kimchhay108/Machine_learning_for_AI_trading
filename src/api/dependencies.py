"""API Dependencies, shared services, and security verifiers."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import Header, HTTPException, status
import joblib

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.data.duckdb_manager import DuckDBManager
from src.data.storage import ParquetStorage
from src.models.ensemble import EnsemblePredictionModel
from src.models.prediction_engine import PredictionEngine

logger = get_logger("api.dependencies")

# Shared singletons
_duckdb_mgr: Optional[DuckDBManager] = None
_parquet_storage: Optional[ParquetStorage] = None
_active_ensemble: Optional[EnsemblePredictionModel] = None
_active_prediction_engine: Optional[PredictionEngine] = None
_active_version: str = "v1.0.0"


def get_active_version() -> str:
    """Retrieve the currently active production model version, checking persistent disk state first."""
    global _active_version
    active_file = settings.DATA_DIR / "models" / "active_version.txt"
    if active_file.exists():
        try:
            persisted = active_file.read_text(encoding="utf-8").strip()
            if persisted:
                _active_version = persisted
        except Exception:
            pass
    return _active_version


def get_parquet_storage() -> ParquetStorage:
    global _parquet_storage
    if _parquet_storage is None:
        _parquet_storage = ParquetStorage()
    return _parquet_storage


def get_duckdb_manager() -> DuckDBManager:
    global _duckdb_mgr
    if _duckdb_mgr is None:
        _duckdb_mgr = DuckDBManager()
        _duckdb_mgr.create_views()
    return _duckdb_mgr


def get_prediction_engine(version: Optional[str] = None) -> PredictionEngine:
    global _active_ensemble, _active_prediction_engine, _active_version
    current_active = get_active_version()
    target_version = version or current_active
    registry_dir = settings.DATA_DIR / "models" / "registry"

    # Intelligent fallback if target_version does not exist
    model_dir = registry_dir / target_version
    if not model_dir.exists():
        if current_active and (registry_dir / current_active).exists():
            logger.warning(
                f"Requested model '{target_version}' not found in registry. "
                f"Falling back to active production version '{current_active}'."
            )
            target_version = current_active
            model_dir = registry_dir / target_version
        else:
            existing_dirs = sorted([d.name for d in registry_dir.iterdir() if d.is_dir()])
            if existing_dirs:
                fallback_ver = existing_dirs[-1]
                logger.warning(
                    f"Neither '{target_version}' nor active '{current_active}' found. "
                    f"Falling back to latest registry version '{fallback_ver}'."
                )
                target_version = fallback_ver
                model_dir = registry_dir / target_version
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No models found in registry at {registry_dir}.",
                )

    if _active_prediction_engine is not None and _active_version == target_version:
        return _active_prediction_engine

    try:
        ensemble = EnsemblePredictionModel.load_suite(registry_dir, version=target_version)
        engine = PredictionEngine(ensemble=ensemble, model_version=target_version)
        _active_ensemble = ensemble
        _active_prediction_engine = engine
        _active_version = target_version
        logger.info(f"Loaded active model suite {target_version} successfully.")
        return engine
    except Exception as e:
        logger.error(f"Failed loading model suite {target_version}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed loading model suite '{target_version}': {str(e)}",
        )


def set_active_version(version: str) -> None:
    """Set the active production model version in memory and persist to disk."""
    global _active_version, _active_prediction_engine
    _active_version = version
    _active_prediction_engine = None  # Force reload on next request
    try:
        active_file = settings.DATA_DIR / "models" / "active_version.txt"
        active_file.parent.mkdir(parents=True, exist_ok=True)
        active_file.write_text(version, encoding="utf-8")
        logger.info(f"Persisted active model version '{version}' to {active_file}")
    except Exception as e:
        logger.warning(f"Could not persist active version to disk: {e}")


def verify_admin_key(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> bool:
    """Security dependency to protect administrative model training operations."""
    if not x_admin_key or x_admin_key != settings.API_ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Key header.",
        )
    return True


def delete_model_version(version: str) -> Dict[str, Any]:
    """Delete a model version from the registry with safe active fallback."""
    global _active_version, _active_prediction_engine, _active_ensemble
    registry_dir = settings.DATA_DIR / "models" / "registry"
    v_dir = registry_dir / version
    if not v_dir.exists() or not v_dir.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version}' not found in registry at {v_dir}.",
        )

    current_active = get_active_version()
    remaining = sorted([d.name for d in registry_dir.iterdir() if d.is_dir() and d.name != version])

    # If deleting the currently active version, ensure at least one fallback remains
    fallback_active = current_active
    if version == current_active:
        if not remaining:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the sole model version in registry. Retrain or register another model first.",
            )
        # Prefer v1.0.0 or the newest available version
        fallback_active = "v1.0.0" if "v1.0.0" in remaining else remaining[-1]
        set_active_version(fallback_active)

    # Invalidate cache if active engine was loaded from deleted version
    if _active_version == version:
        _active_prediction_engine = None
        _active_ensemble = None

    shutil.rmtree(v_dir, ignore_errors=True)
    logger.info(f"Model version '{version}' deleted successfully. Active model is now '{fallback_active}'.")

    return {
        "status": "SUCCESS",
        "deleted_version": version,
        "active_version": get_active_version(),
        "remaining_versions": remaining,
        "message": f"Version '{version}' deleted successfully. Active production model is now '{get_active_version()}'.",
    }
