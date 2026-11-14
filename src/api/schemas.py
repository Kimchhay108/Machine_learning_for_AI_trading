"""Pydantic schemas for the XAUUSD AI Trading API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CandleItem(BaseModel):
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    instrument: str = "XAUUSD"
    timeframe: str = "M5"


class CandlesResponse(BaseModel):
    instrument: str
    timeframe: str
    count: int
    candles: List[CandleItem]


class SMCStructure(BaseModel):
    swing_highs: List[Dict[str, Any]] = []
    swing_lows: List[Dict[str, Any]] = []
    active_fvgs: List[Dict[str, Any]] = []
    recent_bos: List[Dict[str, Any]] = []
    recent_choch: List[Dict[str, Any]] = []
    order_blocks: List[Dict[str, Any]] = []
    liquidity_sweeps: List[Dict[str, Any]] = []


class MarketFeaturesResponse(BaseModel):
    instrument: str
    timeframe: str
    timestamp: int
    current_price: float
    regime: Dict[str, Any]
    indicators: Dict[str, Any]
    smc: SMCStructure


class PredictionResponse(BaseModel):
    instrument: str
    timeframe: str
    prediction_timestamp: int
    direction: str                     # "BUY", "SELL", or "WAIT"
    bullish_probability: float
    bearish_probability: float
    neutral_probability: float
    confidence: float
    market_regime: str
    entry_reference_price: float
    stop_loss_reference: float
    take_profit_reference: float
    risk_reward_ratio: float
    model_version: str
    contributing_features: Dict[str, Any]


class ModelInfo(BaseModel):
    version: str
    name: str
    is_active: bool
    confidence_threshold: float
    submodels: List[str]
    train_date: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ModelsListResponse(BaseModel):
    active_version: str
    models: List[ModelInfo]


class PerformanceResponse(BaseModel):
    model_version: str
    directional_accuracy: float
    selective_accuracy: float
    coverage_pct: float
    f1_score: float
    expected_calibration_error: float
    brier_score: float
    confusion_matrix: Dict[str, Any]
    regime_accuracies: Dict[str, float]


class TrainRequest(BaseModel):
    version: str = Field(default="v1.0.1", description="Model version identifier")
    threshold: float = Field(default=0.50, ge=0.33, le=0.99, description="Confidence threshold for selective abstention")
    test_ratio: float = Field(default=0.30, ge=0.10, le=0.50, description="Out-of-sample test fraction")


class TrainResponse(BaseModel):
    status: str
    version: str
    message: str
    train_samples: int
    test_samples: int
    metrics: Dict[str, Any]


class BacktestRequest(BaseModel):
    version: str = Field(default="v1.0.0", description="Model version to use")
    initial_capital: float = Field(default=10000.0, ge=100.0)
    spread: float = Field(default=0.20, ge=0.0, description="Bid-ask spread in USD")
    slippage: float = Field(default=0.05, ge=0.0, description="Slippage in USD")
    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.10, description="Risk fraction per trade (e.g. 0.01 = 1%)")
    threshold: float = Field(default=0.45, ge=0.33, le=0.95)
    max_holding_bars: int = Field(default=24, ge=1, le=200)


class TradeRecordSchema(BaseModel):
    entry_time: int
    exit_time: int
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size_oz: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str


class BacktestResponse(BaseModel):
    initial_balance: float
    final_balance: float
    total_net_profit: float
    return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_usd: float
    avg_r_multiple: float
    equity_curve: List[float]
    trades: List[TradeRecordSchema]


class PredictionEvaluationRequest(BaseModel):
    horizon_bars: int = Field(default=3, description="Evaluation horizon in candles (e.g. 3 M5 candles = 15m)")
    sample_limit: int = Field(default=200, ge=10, le=2000)


class PredictionEvaluationResponse(BaseModel):
    total_predictions: int
    actionable_predictions: int
    abstentions: int
    directional_accuracy: float
    profit_factor: float
    mean_favorable_excursion_pips: float
    mean_adverse_excursion_pips: float
