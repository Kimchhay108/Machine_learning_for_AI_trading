"""Multi-timeframe feature union and end-to-end transformation pipeline.

Fuses primary M5 features with higher-timeframe M15 and H1 context.
Uses backward-looking as-of merges on closed HTF candles to eliminate lookahead bias.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.features.price_action import PriceActionFeatures
from src.features.regime import MarketRegimeFeatures
from src.features.smc import SmartMoneyConcepts
from src.features.technical import TechnicalFeatures

logger = get_logger("features.pipeline")


class FeaturePipeline:
    """Orchestrates multi-timeframe feature extraction and backward as-of alignment."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.FEATURES_DIR / "xauusd"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_single_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract all technical, SMC, price action, and regime features for a single timeframe."""
        logger.debug(f"Extracting features for timeframe with {len(df)} candles...")
        res = df.copy()
        res = TechnicalFeatures.compute_all(res)
        res = SmartMoneyConcepts.compute_all(res)
        res = PriceActionFeatures.compute_all(res)
        res = MarketRegimeFeatures.compute_all(res)
        return res

    def build_multi_timeframe_feature_matrix(
        self,
        df_m5: pd.DataFrame,
        df_m15: Optional[pd.DataFrame] = None,
        df_h1: Optional[pd.DataFrame] = None,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> pd.DataFrame:
        """Extract primary M5 features and merge backward-aligned M15 and H1 context.

        Args:
            df_m5: Primary M5 candlestick DataFrame
            df_m15: Optional M15 candlestick DataFrame for intermediate context
            df_h1: Optional H1 candlestick DataFrame for macro context
            instrument: Symbol, default "XAUUSD"

        Returns:
            Unified feature DataFrame with strict chronological ordering and zero lookahead bias.
        """
        logger.info(f"Building feature matrix for {instrument} (M5 rows: {len(df_m5)})")

        # 1. Primary M5 Feature Extraction
        m5_feats = self.extract_single_timeframe_features(df_m5)

        # 2. Higher Timeframe M15 Context (if provided)
        if df_m15 is not None and not df_m15.empty:
            logger.info("Extracting and aligning M15 contextual features...")
            m15_raw_feats = self.extract_single_timeframe_features(df_m15)

            # Select key contextual features to avoid explosion
            m15_select_cols = [
                "timestamp", "ema_50", "ema_200", "rsi_14", "macd_hist", "adx_14",
                "market_structure_trend", "dealing_range_pos", "market_regime"
            ]
            m15_subset = m15_raw_feats[[c for c in m15_select_cols if c in m15_raw_feats.columns]].copy()
            # Shift by 1 period so only the COMPLETED M15 candle is referenced at any M5 timestamp
            feature_cols = [c for c in m15_subset.columns if c != "timestamp"]
            m15_subset[feature_cols] = m15_subset[feature_cols].shift(1)
            # Prefix columns
            m15_subset = m15_subset.rename(columns={c: f"m15_{c}" for c in feature_cols})
            m15_subset = m15_subset.dropna(subset=["timestamp"]).sort_values("timestamp")

            # Merge as-of backwards
            m5_feats = pd.merge_asof(
                m5_feats.sort_values("timestamp"),
                m15_subset,
                on="timestamp",
                direction="backward",
            )

        # 3. Higher Timeframe H1 Context (if provided)
        if df_h1 is not None and not df_h1.empty:
            logger.info("Extracting and aligning H1 contextual features...")
            h1_raw_feats = self.extract_single_timeframe_features(df_h1)

            h1_select_cols = [
                "timestamp", "ema_50", "ema_200", "rsi_14", "macd_hist", "adx_14",
                "market_structure_trend", "dealing_range_pos", "market_regime"
            ]
            h1_subset = h1_raw_feats[[c for c in h1_select_cols if c in h1_raw_feats.columns]].copy()
            # Shift by 1 period so only the COMPLETED H1 candle is referenced at any M5 timestamp
            h1_feature_cols = [c for c in h1_subset.columns if c != "timestamp"]
            h1_subset[h1_feature_cols] = h1_subset[h1_feature_cols].shift(1)
            # Prefix columns
            h1_subset = h1_subset.rename(columns={c: f"h1_{c}" for c in h1_feature_cols})
            h1_subset = h1_subset.dropna(subset=["timestamp"]).sort_values("timestamp")

            # Merge as-of backwards
            m5_feats = pd.merge_asof(
                m5_feats.sort_values("timestamp"),
                h1_subset,
                on="timestamp",
                direction="backward",
            )

        m5_feats = m5_feats.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Feature matrix complete. Total features generated: {len(m5_feats.columns)}")
        return m5_feats

    def save_feature_matrix(self, df: pd.DataFrame, filename: str = "m5_features.parquet") -> Path:
        """Persist feature matrix to Parquet format."""
        file_path = self.output_dir / filename
        df.to_parquet(file_path, index=False, compression="snappy")
        logger.info(f"Saved feature matrix ({len(df)} rows, {len(df.columns)} cols) to {file_path}")
        return file_path
