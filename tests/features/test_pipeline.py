"""Unit tests for the multi-timeframe feature pipeline and backward as-of merges."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline


class TestFeaturePipeline:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pipeline = FeaturePipeline(output_dir=Path(self.temp_dir.name))

        # 60 M5 candles (5 hours)
        base_ts = 1714521600000
        n_m5 = 60
        ts_m5 = base_ts + np.arange(n_m5) * 300_000
        self.df_m5 = pd.DataFrame({
            "timestamp": ts_m5,
            "open": 2300.0 + np.arange(n_m5) * 0.2,
            "high": 2302.0 + np.arange(n_m5) * 0.2,
            "low": 2299.0 + np.arange(n_m5) * 0.2,
            "close": 2301.0 + np.arange(n_m5) * 0.2,
            "volume": np.full(n_m5, 10.0),
            "instrument": "XAUUSD",
            "timeframe": "m5",
            "data_source": "dukascopy",
        })

        # 20 M15 candles (5 hours)
        n_m15 = 20
        ts_m15 = base_ts + np.arange(n_m15) * 900_000
        self.df_m15 = pd.DataFrame({
            "timestamp": ts_m15,
            "open": 2300.0 + np.arange(n_m15) * 0.6,
            "high": 2304.0 + np.arange(n_m15) * 0.6,
            "low": 2298.0 + np.arange(n_m15) * 0.6,
            "close": 2303.0 + np.arange(n_m15) * 0.6,
            "volume": np.full(n_m15, 30.0),
            "instrument": "XAUUSD",
            "timeframe": "m15",
            "data_source": "dukascopy",
        })

        # 5 H1 candles (5 hours)
        n_h1 = 5
        ts_h1 = base_ts + np.arange(n_h1) * 3600_000
        self.df_h1 = pd.DataFrame({
            "timestamp": ts_h1,
            "open": 2300.0 + np.arange(n_h1) * 2.4,
            "high": 2308.0 + np.arange(n_h1) * 2.4,
            "low": 2296.0 + np.arange(n_h1) * 2.4,
            "close": 2306.0 + np.arange(n_h1) * 2.4,
            "volume": np.full(n_h1, 120.0),
            "instrument": "XAUUSD",
            "timeframe": "h1",
            "data_source": "dukascopy",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_multi_timeframe_alignment_no_lookahead(self):
        matrix = self.pipeline.build_multi_timeframe_feature_matrix(
            df_m5=self.df_m5,
            df_m15=self.df_m15,
            df_h1=self.df_h1,
        )

        assert len(matrix) == len(self.df_m5)
        # Check higher timeframe prefixed columns are present
        assert any(c.startswith("m15_") for c in matrix.columns)
        assert any(c.startswith("h1_") for c in matrix.columns)

        # Verify that at timestamp 0 (00:00:00), M15 and H1 features from subsequent bars are NOT leaked
        # Due to shift(1), the very first M15/H1 feature should be NaN because no prior M15/H1 candle had completed yet!
        assert np.isnan(matrix["m15_ema_50"].iloc[0])
        assert np.isnan(matrix["h1_ema_50"].iloc[0])

    def test_save_feature_matrix(self):
        matrix = self.pipeline.build_multi_timeframe_feature_matrix(self.df_m5)
        path = self.pipeline.save_feature_matrix(matrix, filename="test_feats.parquet")
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(matrix)
