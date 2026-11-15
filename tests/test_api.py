"""Unit and integration tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Test suite covering health, market data, AI predictions, and backtesting endpoints."""

    def test_root_and_health(self, client: TestClient):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ONLINE"
        assert data["instrument"] == "XAUUSD"

        res_health = client.get("/api/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "HEALTHY"

    def test_get_candles_m5(self, client: TestClient):
        res = client.get("/api/market/candles?instrument=XAUUSD&timeframe=M5&limit=50")
        assert res.status_code == 200
        data = res.json()
        assert data["instrument"] == "XAUUSD"
        assert data["timeframe"] == "M5"
        assert data["count"] > 0
        assert len(data["candles"]) == data["count"]

        # Validate candle structure
        c = data["candles"][-1]
        assert "open" in c and "high" in c and "low" in c and "close" in c and "timestamp" in c
        assert c["high"] >= c["low"]
        assert c["high"] >= c["open"]
        assert c["high"] >= c["close"]

    def test_get_candles_invalid_timeframe(self, client: TestClient):
        res = client.get("/api/market/candles?timeframe=D1")
        assert res.status_code == 400
        assert "Unsupported timeframe" in res.json()["detail"]

    def test_get_market_features(self, client: TestClient):
        res = client.get("/api/market/features?instrument=XAUUSD&limit=50")
        assert res.status_code == 200
        data = res.json()
        assert data["instrument"] == "XAUUSD"
        assert "indicators" in data
        assert "regime" in data
        assert "smc" in data

        indicators = data["indicators"]
        assert "rsi_14" in indicators
        assert "ema_20" in indicators
        assert "macd_hist" in indicators

        regime = data["regime"]
        assert "regime_label" in regime
        assert "efficiency_ratio" in regime

        smc = data["smc"]
        assert "swing_highs" in smc
        assert "swing_lows" in smc

    def test_get_ai_prediction(self, client: TestClient):
        res = client.get("/api/ai/prediction?instrument=XAUUSD&timeframe=M5")
        assert res.status_code == 200
        data = res.json()

        assert data["direction"] in ["BUY", "SELL", "WAIT"]
        assert "bullish_probability" in data
        assert "bearish_probability" in data
        assert "neutral_probability" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["entry_reference_price"] > 0
        assert data["stop_loss_reference"] > 0
        assert data["take_profit_reference"] > 0
        assert data["risk_reward_ratio"] == 2.0
        assert "contributing_features" in data

    def test_list_models(self, client: TestClient):
        res = client.get("/api/ai/models")
        assert res.status_code == 200
        data = res.json()
        assert "active_version" in data
        assert len(data["models"]) >= 1
        versions = [m["version"] for m in data["models"]]
        assert "v1.0.0" in versions

    def test_get_performance(self, client: TestClient):
        res = client.get("/api/ai/performance?instrument=XAUUSD&version=v1.0.0")
        assert res.status_code == 200
        data = res.json()
        assert data["model_version"] == "v1.0.0"
        assert 0.0 <= data["directional_accuracy"] <= 1.0
        assert 0.0 <= data["selective_accuracy"] <= 1.0
        assert 0.0 <= data["coverage_pct"] <= 100.0
        assert "expected_calibration_error" in data
        assert "confusion_matrix" in data
        assert "regime_accuracies" in data

    def test_admin_retrain_protection(self, client: TestClient):
        # Call without admin header -> should fail with 401
        res = client.post("/api/ai/train", json={"version": "v9.9.9", "threshold": 0.50})
        assert res.status_code == 401

        # Call with incorrect key -> 401
        res2 = client.post(
            "/api/ai/train",
            json={"version": "v9.9.9", "threshold": 0.50},
            headers={"X-Admin-Key": "wrong-key"},
        )
        assert res2.status_code == 401

    def test_admin_retrain_authorized(self, client: TestClient):
        # Call with valid admin key -> 200
        res = client.post(
            "/api/ai/train",
            json={"version": "v1.0.1", "threshold": 0.52, "test_ratio": 0.25},
            headers={"X-Admin-Key": settings.API_ADMIN_KEY},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert data["version"] == "v1.0.1"
        assert data["train_samples"] > 0

    def test_evaluate_predictions(self, client: TestClient):
        res = client.post(
            "/api/ai/predictions/evaluate?instrument=XAUUSD",
            json={"horizon_bars": 3, "sample_limit": 50},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_predictions"] == 50
        assert data["actionable_predictions"] + data["abstentions"] == 50
        assert 0.0 <= data["directional_accuracy"] <= 1.0
        assert "mean_favorable_excursion_pips" in data
        assert "mean_adverse_excursion_pips" in data

    def test_backtest_run_and_get(self, client: TestClient):
        req = {
            "version": "v1.0.0",
            "initial_capital": 10000.0,
            "spread": 0.20,
            "slippage": 0.05,
            "risk_per_trade": 0.01,
            "threshold": 0.45,
            "max_holding_bars": 24,
        }
        res = client.post("/api/backtest/run", json=req)
        assert res.status_code == 200
        data = res.json()
        assert data["initial_balance"] == 10000.0
        assert data["total_trades"] > 0
        assert len(data["equity_curve"]) > 0
        assert len(data["trades"]) == data["total_trades"]

        # Validate cached results retrieval
        res_cache = client.get("/api/backtest/results")
        assert res_cache.status_code == 200
        data_cache = res_cache.json()
        assert data_cache["total_trades"] == data["total_trades"]

    def test_backtest_model_types(self, client: TestClient):
        """Verify backtest runs with specific sub-models."""
        for m_type in ["smc_ict", "technical", "price_action"]:
            req = {
                "version": "v1.0.0",
                "model_type": m_type,
                "initial_capital": 10000.0,
                "threshold": 0.50,
            }
            res = client.post("/api/backtest/run", json=req)
            assert res.status_code == 200
            data = res.json()
            assert data["initial_balance"] == 10000.0
            assert "total_trades" in data
            assert len(data["equity_curve"]) > 0

    def test_backtest_compare(self, client: TestClient):
        """Verify comparative backtesting endpoint runs all models side-by-side."""
        res = client.post("/api/backtest/compare", json={"version": "v1.0.0", "threshold": 0.50})
        assert res.status_code == 200
        data = res.json()
        assert "results" in data
        assert len(data["results"]) == 6
        keys = [item["model_key"] for item in data["results"]]
        assert "ensemble" in keys
        assert "smc_ict" in keys
        assert "technical" in keys
        assert "price_action" in keys
        assert "volume_profile" in keys
        assert "footprint" in keys
        assert "best_model_by_profit" in data
        assert "best_model_by_winrate" in data
        for item in data["results"]:
            assert item["total_trades"] >= 0
            assert "sl_tp_logic" in item

    def test_backtest_monthly_breakdown(self, client: TestClient):
        """Verify backtest results return monthly performance breakdown."""
        res = client.get("/api/backtest/results")
        assert res.status_code == 200
        data = res.json()
        assert "monthly_breakdown" in data
        assert len(data["monthly_breakdown"]) > 0
        first_m = data["monthly_breakdown"][0]
        assert "year" in first_m
        assert "month" in first_m
        assert "period" in first_m
        assert "total_trades" in first_m
        assert "winning_trades" in first_m
        assert "losing_trades" in first_m
        assert "win_rate_pct" in first_m
        assert "net_profit" in first_m
        assert "profit_factor" in first_m

    def test_delete_model_version(self, client: TestClient):
        """Verify model version deletion with admin authentication and error handling."""
        import shutil
        registry_dir = settings.DATA_DIR / "models" / "registry"
        dummy_dir = registry_dir / "v_test_delete_candidate"
        dummy_dir.mkdir(parents=True, exist_ok=True)
        (dummy_dir / "dummy.txt").write_text("dummy model content", encoding="utf-8")

        # 1. Unauthorized without X-Admin-Key
        res_unauth = client.delete("/api/ai/models/v_test_delete_candidate")
        assert res_unauth.status_code == 401

        # 2. Successfully delete candidate
        headers = {"X-Admin-Key": settings.API_ADMIN_KEY}
        res_del = client.delete("/api/ai/models/v_test_delete_candidate", headers=headers)
        assert res_del.status_code == 200
        data = res_del.json()
        assert data["status"] == "SUCCESS"
        assert data["deleted_version"] == "v_test_delete_candidate"
        assert not dummy_dir.exists()

        # 3. 404 on deleting non-existent model
        res_404 = client.delete("/api/ai/models/v_non_existent_model_999", headers=headers)
        assert res_404.status_code == 404

