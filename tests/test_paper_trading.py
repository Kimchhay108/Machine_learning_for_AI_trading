"""Unit and integration tests for live feed, paper trading engine, and daemon."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from src.api.main import app
from src.live.paper_trader import PaperTradingEngine
from src.models.prediction_engine import TradePrediction


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


class TestPaperTrading:
    """Test virtual order execution, intrabar exit rules, and state management."""

    def test_paper_buy_execution_and_tp_exit(self, tmp_path):
        state_file = tmp_path / "paper_state.json"
        engine = PaperTradingEngine(
            initial_balance=10_000.0,
            spread_usd=0.20,
            slippage_usd=0.05,
            risk_pct_per_trade=0.01,
            state_file=state_file,
        )

        candle_1 = pd.Series({
            "timestamp": 1714521600000,
            "open": 2320.0,
            "high": 2322.0,
            "low": 2319.0,
            "close": 2321.0,
        })

        pred_buy = TradePrediction(
            instrument="XAUUSD",
            timeframe="M5",
            prediction_timestamp=1714521600000,
            direction="BUY",
            bullish_probability=0.75,
            bearish_probability=0.10,
            neutral_probability=0.15,
            confidence=0.75,
            market_regime="Bullish_Trend",
            entry_reference_price=2321.0,
            stop_loss_reference=2316.0,  # $5 risk
            take_profit_reference=2331.0, # $10 target (1:2 R:R)
            risk_reward_ratio=2.0,
            model_version="v1.0.0",
            contributing_features={},
        )

        # 1. Process entry
        res_open = engine.process_candle(candle_1, prediction=pred_buy)
        assert res_open["action"] == "OPENED"
        assert engine.open_position is not None
        assert engine.open_position.direction == "BUY"
        assert engine.open_position.entry_price == 2321.25  # close + spread(0.20) + slippage(0.05)

        # 2. Next candle surges and hits TP (high reaches 2332.0 > 2331.0)
        candle_surge = pd.Series({
            "timestamp": 1714521900000,
            "open": 2322.0,
            "high": 2333.0,
            "low": 2321.0,
            "close": 2332.0,
        })
        res_close = engine.process_candle(candle_surge)
        assert res_close["action"] == "CLOSED"
        assert res_close["reason"] == "TAKE_PROFIT"
        assert res_close["pnl"] > 0
        assert engine.open_position is None
        assert engine.balance > 10_000.0
        assert len(engine.trade_history) == 1

    def test_conservative_sl_priority_same_bar(self, tmp_path):
        state_file = tmp_path / "paper_state.json"
        engine = PaperTradingEngine(
            initial_balance=10_000.0,
            state_file=state_file,
        )

        candle_1 = pd.Series({
            "timestamp": 1714521600000,
            "open": 2320.0,
            "high": 2322.0,
            "low": 2319.0,
            "close": 2321.0,
        })

        pred_buy = TradePrediction(
            instrument="XAUUSD",
            timeframe="M5",
            prediction_timestamp=1714521600000,
            direction="BUY",
            bullish_probability=0.70,
            bearish_probability=0.15,
            neutral_probability=0.15,
            confidence=0.70,
            market_regime="Bullish_Trend",
            entry_reference_price=2321.0,
            stop_loss_reference=2316.0,
            take_profit_reference=2331.0,
            risk_reward_ratio=2.0,
            model_version="v1.0.0",
            contributing_features={},
        )

        engine.process_candle(candle_1, prediction=pred_buy)
        assert engine.open_position is not None

        # Wild flash bar touching both SL (2316) and TP (2331) in same bar
        flash_bar = pd.Series({
            "timestamp": 1714521900000,
            "open": 2322.0,
            "high": 2340.0,  # Exceeds TP
            "low": 2310.0,   # Exceeds SL
            "close": 2335.0,
        })
        res_close = engine.process_candle(flash_bar)
        # Conservative rule: SL hit first
        assert res_close["action"] == "CLOSED"
        assert res_close["reason"] == "STOP_LOSS"
        assert res_close["pnl"] < 0
        assert engine.balance < 10_000.0


class TestPaperRouterAPI:
    """Test FastAPI paper trading endpoints."""

    def test_paper_account_and_controls(self, test_client: TestClient):
        # 1. Get account
        res = test_client.get("/api/paper/account")
        assert res.status_code == 200
        data = res.json()
        assert "balance" in data
        assert "equity" in data
        assert "is_running" in data

        # 2. Start daemon
        res_start = test_client.post("/api/paper/start")
        assert res_start.status_code == 200
        assert res_start.json()["status"] == "RUNNING"

        # 3. Stop daemon
        res_stop = test_client.post("/api/paper/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["status"] == "PAUSED"

        # 4. Reset account
        res_reset = test_client.post("/api/paper/reset?initial_balance=10000")
        assert res_reset.status_code == 200
        assert res_reset.json()["balance"] == 10000.0

        # 5. Get trades list
        res_trades = test_client.get("/api/paper/trades")
        assert res_trades.status_code == 200
        assert isinstance(res_trades.json(), list)
