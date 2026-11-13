"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base workspace directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Data directory structure
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    FEATURES_DIR: Path = BASE_DIR / "data" / "features"
    LABELS_DIR: Path = BASE_DIR / "data" / "labels"
    REPORTS_DIR: Path = BASE_DIR / "data" / "reports"
    QUALITY_REPORTS_DIR: Path = BASE_DIR / "data" / "reports" / "quality"

    # DuckDB Database File Path
    DUCKDB_PATH: Path = BASE_DIR / "data" / "market_data.duckdb"

    # Dukascopy API Settings
    DUKASCOPY_BASE_URL: str = "https://jetta.dukascopy.com/v1"
    DUKASCOPY_REFERER: str = "https://widgets.dukascopy.com/en/historical-data-export"
    DUKASCOPY_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    DUKASCOPY_REQUEST_TIMEOUT: float = 15.0
    DUKASCOPY_MAX_RETRIES: int = 4
    DUKASCOPY_RETRY_DELAY: float = 2.0
    DUKASCOPY_RATE_LIMIT_DELAY: float = 0.25  # Polite delay between batch requests

    # App environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    def ensure_directories(self) -> None:
        """Create all required data and report directories if they do not exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR / "xauusd" / "m5",
            self.RAW_DATA_DIR / "xauusd" / "m15",
            self.RAW_DATA_DIR / "xauusd" / "h1",
            self.PROCESSED_DATA_DIR / "xauusd" / "m5",
            self.PROCESSED_DATA_DIR / "xauusd" / "m15",
            self.PROCESSED_DATA_DIR / "xauusd" / "h1",
            self.FEATURES_DIR / "xauusd",
            self.LABELS_DIR / "xauusd",
            self.REPORTS_DIR,
            self.QUALITY_REPORTS_DIR,
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
