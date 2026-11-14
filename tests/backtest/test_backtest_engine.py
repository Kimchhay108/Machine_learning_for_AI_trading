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

    def test_partial_tp_and_breakeven_lock(self):
        # Entry BUY at 2000.25 (spread+slippage). SL = 1998.25 (risk = 2.0). TP = 2006.25 (+3.0R).
        # Candle 1: High reaches 2003.0 (+1.37R > 1.0R) -> triggers 50% Partial TP, SL moved to BE!
        # Candle 2: Pullback drops to 2000.0 -> hits BE SL.
        # Should record exit_reason = PARTIAL_TP_BE and overall profitable trade!
        df_partial = pd.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "open": [2000.0, 2001.0, 2001.0],
            "high": [2000.5, 2003.5, 2001.5],  # reaches +1.0R
            "low": [1999.5, 2000.5, 1999.5],   # drops back in candle 2
            "close": [2000.0, 2002.0, 2000.0],
            "atr_14": [2.0, 2.0, 2.0],
        })
        signals = ["BUY", "WAIT", "WAIT"]
        engine = RealisticBacktestEngine(
            initial_balance=10_000.0,
            enable_partial_tp=True,
            partial_tp_r=1.0,
            partial_tp_ratio=0.50,
            enable_breakeven=True,
        )
        res = engine.run(df_partial, signals=signals)

        assert res.total_trades == 1
        assert res.trades[0].exit_reason == "PARTIAL_TP_BE"
        assert res.trades[0].pnl_usd > 0
        assert res.breakeven_trades == 1
        assert res.win_rate_with_be_pct == 100.0
