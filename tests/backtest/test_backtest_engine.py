"""Unit tests for the realistic backtesting execution engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import RealisticBacktestEngine


class TestRealisticBacktestEngine:
    def setup_method(self):
        n = 30
        timestamps = 1714521600000 + np.arange(n) * 300_000

        # Bar 0..4: 2000 flat
        # Bar 5: BUY signal at 2000
        # Bar 6..8: Rallies to 2015 (hits TP at 2006.0)
        # Bar 15: SELL signal at 2010
        # Bar 16..18: Drops to 1995 (hits TP at 2004.0)
        close = np.full(n, 2000.0)
        close[6:10] = 2015.0
        close[10:15] = 2010.0
        close[16:20] = 1995.0

        high = close + 1.0
        low = close - 1.0
        open_p = close

        self.df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "atr_14": np.full(n, 2.0),
        })

    def test_backtest_execution_and_metrics(self):
        signals = ["WAIT"] * len(self.df)
        signals[5] = "BUY"
        signals[15] = "SELL"

        engine = RealisticBacktestEngine(initial_balance=10_000.0, risk_pct_per_trade=0.01)
        res = engine.run(self.df, signals=signals)

        assert res.total_trades >= 1
        assert res.win_rate_pct >= 0.0
        assert len(res.trades) == res.total_trades
        assert len(res.equity_curve) > 0
        assert res.max_drawdown_pct >= 0.0

    def test_conservative_conflict_rule(self):
        # Create a single candle where BOTH high exceeds TP AND low breaches SL
        # Entry BUY at 2000 (TP = 2006, SL = 1997)
        # In candle 1: High = 2010 (touches TP) and Low = 1990 (touches SL)
        # Conservative execution rule MUST trigger Stop-Loss!
        df_conflict = pd.DataFrame({
            "timestamp": [1000, 2000],
            "open": [2000.0, 2000.0],
            "high": [2001.0, 2010.0],  # touches TP (2006)
            "low": [1999.0, 1990.0],   # touches SL (1997)
            "close": [2000.0, 2005.0],
            "atr_14": [2.0, 2.0],
        })
        signals = ["BUY", "WAIT"]
        engine = RealisticBacktestEngine(initial_balance=10_000.0)
        res = engine.run(df_conflict, signals=signals)

        assert res.total_trades == 1
        assert res.trades[0].exit_reason == "STOP_LOSS"  # Conservative stop out!
        assert res.trades[0].pnl_usd < 0
