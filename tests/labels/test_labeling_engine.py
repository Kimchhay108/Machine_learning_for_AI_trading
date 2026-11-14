"""Unit tests for statistical labeling engine and triple-barrier execution."""

import numpy as np
import pandas as pd
import pytest

from src.labels.labeling_engine import LabelingEngine


class TestLabelingEngine:
    def setup_method(self):
        n = 50
        base_ts = 1714521600000
        timestamps = base_ts + np.arange(n) * 300_000

        # Predictable price series:
        # Bars 0..9: Flat around 2000
        # Bars 10..19: Strong rally to 2050 (+50)
        # Bars 20..29: Strong drop to 1950 (-100)
        prices = np.concatenate([
            np.full(10, 2000.0),
            2000.0 + np.arange(1, 11) * 5.0,
            2050.0 - np.arange(1, 11) * 10.0,
            np.full(20, 1950.0),
        ])
        close = prices
        high = close + 2.0
        low = close - 2.0
        open_p = close - 0.5

        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "atr_14": np.full(n, 5.0),
            "volume": np.full(n, 10.0),
        })

    def test_directional_returns(self):
        res = LabelingEngine.add_directional_returns(self.df)
        assert "fwd_ret_15m" in res.columns
        assert "fwd_ret_30m" in res.columns
        assert "fwd_ret_60m" in res.columns
        assert "label_dir_binary_15m" in res.columns
        assert "label_dir_ternary_15m" in res.columns

        # At bar 10, strong rally follows -> fwd_ret should be positive
        assert res["label_dir_binary_15m"].iloc[10] == 1
        assert res["label_dir_ternary_15m"].iloc[10] == 1

        # At bar 20, strong drop follows -> fwd_ret should be negative
        assert res["label_dir_binary_15m"].iloc[20] == 0
        assert res["label_dir_ternary_15m"].iloc[20] == -1

    def test_triple_barrier_labeling(self):
        res = LabelingEngine.add_triple_barrier_labels(
            self.df,
            tp_mult=2.0,  # 2 * 5.0 = 10.0 TP
            sl_mult=1.0,  # 1 * 5.0 = 5.0 SL
            horizon_bars=6,
        )
        assert "label_tb_long" in res.columns
        assert "label_tb_short" in res.columns
        assert "mfe_long_atr" in res.columns
        assert "mae_long_atr" in res.columns

        # At bar 10, rally of +50 occurs -> Long TP must hit (+1)
        assert res["label_tb_long"].iloc[10] == 1
        # Short SL must hit (-1)
        assert res["label_tb_short"].iloc[10] == -1

        # At bar 20, drop of -100 occurs -> Long SL hits (-1), Short TP hits (+1)
        assert res["label_tb_long"].iloc[20] == -1
        assert res["label_tb_short"].iloc[20] == 1

    def test_compute_all_labels(self):
        res = LabelingEngine.compute_all_labels(self.df)
        assert len(res) == len(self.df)
        assert "label_tb_long" in res.columns
        assert "label_dir_ternary_60m" in res.columns
