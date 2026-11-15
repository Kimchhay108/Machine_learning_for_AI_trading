"""API Dependencies, shared services, and security verifiers."""

from pathlib import Path
from typing import Optional
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

    if _active_prediction_engine is not None and _active_version == target_version:
        return _active_prediction_engine

    model_dir = registry_dir / target_version
    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{target_version}' not found in registry at {registry_dir}.",
        )

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
