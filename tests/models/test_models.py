"""Unit tests for ML models, ensemble stacking, and prediction engine."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import EnsemblePredictionModel
from src.models.prediction_engine import PredictionEngine
from src.models.price_action_model import PriceActionModel
from src.models.regime_model import MarketRegimeModel
from src.models.smc_model import SmartMoneyConceptsModel
from src.models.technical_model import TechnicalAnalysisModel


class TestMachineLearningSuite:
    def setup_method(self):
        np.random.seed(42)
        n = 120
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = Path(self.temp_dir.name)

        # Synthetic feature dataframe containing all expected feature names
        data = {
            "timestamp": 1714521600000 + np.arange(n) * 300_000,
            "open": 2300.0 + np.random.randn(n),
            "high": 2305.0 + np.random.randn(n),
            "low": 2295.0 + np.random.randn(n),
            "close": 2302.0 + np.random.randn(n),
            "volume": np.random.uniform(10, 50, n),
            "atr_14": np.full(n, 2.5),
            "regime_label": ["Bullish_Trend" if i % 2 == 0 else "Ranging" for i in range(n)],
        }

        # Add technical features
        for f in ["ema_9_dist_pct", "ema_20_dist_pct", "ema_50_dist_pct", "ema_100_dist_pct", "ema_200_dist_pct",
                  "ema_9_slope", "ema_20_slope", "ema_50_slope", "ema_alignment", "atr_14_pct",
                  "rsi_14", "rsi_14_norm", "rsi_14_overbought", "rsi_14_oversold", "macd_line", "macd_signal",
                  "macd_hist", "macd_hist_pct", "macd_hist_slope", "bb_pct_b", "bb_bandwidth", "plus_di_14",
                  "minus_di_14", "adx_14", "adx_14_trending", "stoch_rsi_k", "stoch_rsi_d", "stoch_rsi_diff",
                  "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "roc_5", "roc_10", "roc_20", "vol_ratio_20", "vwap_dist_pct"]:
            data[f] = np.random.randn(n)

        # Add SMC features
        for f in ["market_structure_trend", "bos_bullish", "bos_bearish", "choch_bullish", "choch_bearish",
                  "liquidity_sweep_bullish", "liquidity_sweep_bearish", "equal_highs", "equal_lows",
                  "fvg_bullish_new", "fvg_bearish_new", "fvg_bullish_unmitigated_count", "fvg_bearish_unmitigated_count",
                  "order_block_bullish", "order_block_bearish", "dealing_range_pos", "is_discount_zone",
                  "is_premium_zone", "is_ote_long_zone", "is_ote_short_zone", "session_asian", "session_london",
                  "session_ny", "session_london_ny_overlap", "swept_pdh", "swept_pdl"]:
            data[f] = np.random.randint(0, 2, n)

        # Add Regime features
        for f in ["efficiency_ratio_20", "trend_slope_20", "atr_percentile", "is_high_volatility",
                  "is_low_volatility", "bbw_percentile", "volatility_squeeze", "volatility_expansion"]:
            data[f] = np.random.uniform(0, 1, n)

        # Add Price Action features
        for f in ["body_to_range_ratio", "upper_wick_ratio", "lower_wick_ratio", "is_bullish_candle",
                  "is_bearish_candle", "pin_bar_bullish", "pin_bar_bearish", "engulfing_bullish",
                  "engulfing_bearish", "inside_bar", "breakout_high_20", "breakout_low_20",
                  "retest_high_20", "retest_low_20", "streak_bullish", "streak_bearish"]:
            data[f] = np.random.uniform(0, 1, n)

        self.df = pd.DataFrame(data)
        # 3 classes: 0 = Bearish, 1 = Neutral, 2 = Bullish
        self.y = pd.Series(np.random.randint(0, 3, n))
        # 4 regime classes: 0, 1, 2, 3
        self.y_regime = pd.Series(np.random.randint(0, 4, n))

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_model_a_technical(self):
        m = TechnicalAnalysisModel(n_estimators=10)
        m.fit(self.df, self.y)
        probs = m.predict_proba(self.df)
        preds = m.predict(self.df)
        assert probs.shape == (len(self.df), 3)
        assert len(preds) == len(self.df)
        assert len(m.get_feature_importances()) > 0

    def test_model_b_smc(self):
        m = SmartMoneyConceptsModel(n_estimators=10)
        m.fit(self.df, self.y)
        probs = m.predict_proba(self.df)
        assert probs.shape == (len(self.df), 3)
        assert len(m.get_feature_importances()) > 0

    def test_model_c_regime(self):
        m = MarketRegimeModel(n_estimators=10)
        m.fit(self.df, self.y_regime)
        preds = m.predict(self.df)
        assert len(preds) == len(self.df)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_model_d_price_action(self):
        m = PriceActionModel(n_estimators=10)
        m.fit(self.df, self.y)
        probs = m.predict_proba(self.df)
        assert probs.shape == (len(self.df), 3)

    def test_ensemble_and_prediction_engine(self):
        ensemble = EnsemblePredictionModel(confidence_threshold=0.50)
        ensemble.fit(self.df, self.y, regime_labels=self.y_regime)

        decisions, probs, confs = ensemble.predict_decision(self.df)
        assert len(decisions) == len(self.df)
        assert set(decisions).issubset({"BUY", "SELL", "WAIT"})

        # Save and reload ensemble
        save_dir = ensemble.save_suite(self.registry, version="test_v1")
        reloaded = EnsemblePredictionModel.load_suite(self.registry, version="test_v1")
        assert reloaded.is_fitted is True

        # Prediction Engine
        engine = PredictionEngine(ensemble=reloaded, model_version="test_v1")
        pred = engine.predict_candle(self.df.iloc[[-1]])

        assert pred.direction in ["BUY", "SELL", "WAIT"]
        assert pred.risk_reward_ratio == 2.0
        assert pred.entry_reference_price > 0
        assert pred.stop_loss_reference != pred.take_profit_reference
        assert "submodel_agreement" in pred.contributing_features
