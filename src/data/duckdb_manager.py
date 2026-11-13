"""DuckDB analytical engine and SQL view manager.

Creates high-speed analytical views directly over Parquet files without duplicating disk storage,
providing instant SQL queries, summary aggregations, and data quality inspection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import pandas as pd

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger

logger = get_logger("duckdb_manager")


class DuckDBManager:
    """Manages DuckDB analytical database connection and Parquet views."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.db_path = db_path or settings.DUCKDB_PATH
        self.raw_dir = raw_dir or settings.RAW_DATA_DIR
        self.processed_dir = processed_dir or settings.PROCESSED_DATA_DIR
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or initialize DuckDB connection."""
        if self.conn is None:
            self.conn = duckdb.connect(str(self.db_path))
        return self.conn

    def close(self) -> None:
        """Close connection cleanly."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "DuckDBManager":
        self.get_connection()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def create_views(self, instrument: str = DEFAULT_INSTRUMENT) -> List[str]:
        """Create or replace DuckDB views pointing to Parquet files for all timeframes.

        Views created:
        - v_{instrument}_m5_raw
        - v_{instrument}_m5_processed
        - v_{instrument}_m15_processed
        - v_{instrument}_h1_processed
        """
        conn = self.get_connection()
        inst_lower = instrument.lower()
        created_views: List[str] = []

        timeframes = [Timeframe.M5.value, Timeframe.M15.value, Timeframe.H1.value]

        for tf in timeframes:
            # Processed View
            proc_pattern = (self.processed_dir / inst_lower / tf / "*.parquet").as_posix()
            proc_view = f"v_{inst_lower}_{tf}_processed"
            # Check if any parquet files exist before creating view
            files = list((self.processed_dir / inst_lower / tf).glob("*.parquet"))
            if files:
                sql = f"CREATE OR REPLACE VIEW {proc_view} AS SELECT * FROM read_parquet('{proc_pattern}') ORDER BY timestamp ASC;"
                conn.execute(sql)
                created_views.append(proc_view)
                logger.info(f"Registered DuckDB view: {proc_view}")

            # Raw View
            raw_pattern = (self.raw_dir / inst_lower / tf / "*.parquet").as_posix()
            raw_view = f"v_{inst_lower}_{tf}_raw"
            raw_files = list((self.raw_dir / inst_lower / tf).glob("*.parquet"))
            if raw_files:
                sql = f"CREATE OR REPLACE VIEW {raw_view} AS SELECT * FROM read_parquet('{raw_pattern}') ORDER BY timestamp ASC;"
                conn.execute(sql)
                created_views.append(raw_view)
                logger.info(f"Registered DuckDB view: {raw_view}")

        return created_views

    def query(self, sql: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        """Execute arbitrary SQL query and return result as pandas DataFrame."""
        conn = self.get_connection()
        logger.debug(f"Executing DuckDB SQL: {sql}")
        if params:
            return conn.execute(sql, params).df()
        return conn.execute(sql).df()

    def get_candle_summary(
        self,
        instrument: str = DEFAULT_INSTRUMENT,
        timeframe: str = "m5",
        is_processed: bool = True,
    ) -> Dict[str, Any]:
        """Compute fast analytical summary of candles via DuckDB."""
        inst_lower = instrument.lower()
        tf_lower = timeframe.lower()
        view_suffix = "processed" if is_processed else "raw"
        view_name = f"v_{inst_lower}_{tf_lower}_{view_suffix}"

        # Ensure view exists
        self.create_views(instrument)

        # Check if view is registered
        conn = self.get_connection()
        view_exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [view_name]
        ).fetchone()[0] > 0

        if not view_exists:
            return {
                "instrument": instrument,
                "timeframe": timeframe,
                "view": view_name,
                "total_candles": 0,
                "message": "No parquet files or view available.",
            }

        query = f"""
            SELECT 
                COUNT(*) as total_candles,
                MIN(timestamp) as min_timestamp,
                MAX(timestamp) as max_timestamp,
                to_timestamp(MIN(timestamp) / 1000) as first_candle_utc,
                to_timestamp(MAX(timestamp) / 1000) as last_candle_utc,
                MIN(low) as lowest_price,
                MAX(high) as highest_price,
                AVG(close) as avg_price,
                SUM(volume) as total_volume,
                AVG(high - low) as avg_range
            FROM {view_name}
        """
        df = conn.execute(query).df()
        if df.empty or df["total_candles"].iloc[0] == 0:
            return {"total_candles": 0, "message": "Dataset is empty"}

        row = df.iloc[0]
        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "view": view_name,
            "total_candles": int(row["total_candles"]),
            "first_candle_utc": str(row["first_candle_utc"]),
            "last_candle_utc": str(row["last_candle_utc"]),
            "min_timestamp": int(row["min_timestamp"]),
            "max_timestamp": int(row["max_timestamp"]),
            "lowest_price": round(float(row["lowest_price"]), 3),
            "highest_price": round(float(row["highest_price"]), 3),
            "avg_price": round(float(row["avg_price"]), 3),
            "total_volume": round(float(row["total_volume"]), 4),
            "avg_candle_range": round(float(row["avg_range"]), 3),
        }

    def get_recent_candles(
        self,
        instrument: str = DEFAULT_INSTRUMENT,
        timeframe: str = "m5",
        limit: int = 100,
        is_processed: bool = True,
    ) -> pd.DataFrame:
        """Fetch the most recent N candles."""
        view_suffix = "processed" if is_processed else "raw"
        view_name = f"v_{instrument.lower()}_{timeframe.lower()}_{view_suffix}"
        self.create_views(instrument)

        query = f"""
            SELECT * FROM {view_name}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        df = self.query(query)
        return df.sort_values("timestamp").reset_index(drop=True)
