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
    ict_2022_setups: List[Dict[str, Any]] = []
    silver_bullet_setups: List[Dict[str, Any]] = []


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
    learn_from_mistakes: bool = Field(
        default=True,
        description="Hard-negative mining: penalize historical trade mistake patterns (chop, spikes, exhaustion)",
    )


class TrainResponse(BaseModel):
    status: str
    version: str
    message: str
    train_samples: int
    test_samples: int
    metrics: Dict[str, Any]


class ActivateModelRequest(BaseModel):
    version: str = Field(description="Model version identifier to activate for production")


class BacktestRequest(BaseModel):
    version: str = Field(default="v1.0.0", description="Model version to use")
    model_type: str = Field(default="ensemble", description="Strategy to backtest: 'ensemble', 'smc_ict', 'technical', 'price_action', 'volume_profile', 'footprint'")
    timeframe: str = Field(default="M5", description="Candle timeframe to backtest: 'M5', 'M15', 'H1'")
    initial_capital: float = Field(default=10000.0, ge=100.0)
    spread: float = Field(default=0.20, ge=0.0, description="Bid-ask spread in USD")
    slippage: float = Field(default=0.05, ge=0.0, description="Slippage in USD")
    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.10, description="Risk fraction per trade (e.g. 0.01 = 1%)")
    threshold: float = Field(default=0.60, ge=0.33, le=0.95)
    max_holding_bars: int = Field(default=24, ge=1, le=200)
    enable_breakeven: bool = Field(default=False, description="Move Stop Loss to entry once price reaches +1.0R in profit (default False for higher winrate)")
    breakeven_r: float = Field(default=1.0, description="R:Multiple trigger to activate Breakeven")
    enable_partial_tp: bool = Field(default=False, description="Lock in 50% profit at +1.0R and trail stop-loss to breakeven (default False to let winners run)")
    partial_tp_r: float = Field(default=1.0, description="R:Multiple trigger to take 50% partial profit")
    only_killzones: bool = Field(default=True, description="Execute trades strictly during London (07:00-10:00) & NY (12:00-15:00) Killzones")
    use_h1_gatekeeper: bool = Field(default=False, description="Enforce H1 Macro Trend alignment: Price > H1 200 EMA for BUY, Price < H1 200 EMA for SELL (default False)")
    learn_from_mistakes: bool = Field(default=True, description="AI Failure Memory: Vetoes setups matching historical loss fingerprints (low momentum chop, climax exhaustion, volatility spikes)")
    filter_high_impact_news: bool = Field(default=True, description="News Blackout: Abstain from trading ±30m around high-impact USD events (NFP, CPI, FOMC, GDP, etc.)")
    news_blackout_mins: int = Field(default=30, description="Minutes before and after high-impact news release to block trade entries")
    exclude_asian_session: bool = Field(default=True, description="Strict Asian Session Blackout: Do NOT trade between 21:00 and 07:00 UTC")
    use_full_history: bool = Field(default=True, description="Backtest the full multi-year 2023-present dataset (all 274k+ bars) instead of only the 30% out-of-sample slice")
    strict_structural_entry: bool = Field(default=True, description="Strict Structural Pattern Confluence (Model B: ICT FVG; Model A: EMA+RSI; Model D: Pin/Engulfing; Model F: Value Area/POC; Model G: Delta Divergence/Absorption). Uncheck for pure raw ML probability entries")
    target_rr: Optional[float] = Field(default=None, description="Optional target Risk:Reward ratio (e.g. 1.2, 1.5, 2.0)")


class NewsEventItemSchema(BaseModel):
    id: str
    title: str
    currency: str
    impact: str
    event_time_utc: str
    category: str
    description: str
    blackout_before_mins: int
    blackout_after_mins: int
    minutes_until_event: Optional[int] = None
    is_blackout_now: bool = False


class NewsCalendarStatusResponse(BaseModel):
    is_blackout_active: bool
    active_event: Optional[str] = None
    minutes_delta_from_event: Optional[float] = None
    minutes_until_next_event: Optional[int] = None
    next_event_title: Optional[str] = None
    next_event_time_utc: Optional[str] = None
    upcoming_events: List[NewsEventItemSchema]


class ModelComparisonItem(BaseModel):
    model_key: str
    model_name: str
    optimal_timeframe: str = Field(default="M5", description="Recommended timeframe for this model")
    sl_tp_logic: str
    total_trades: int
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float
    win_rate_with_be_pct: float = 0.0
    total_net_profit: float
    return_pct: float
    profit_factor: float
    max_drawdown_pct: float
    avg_r_multiple: float


class BacktestComparisonResponse(BaseModel):
    results: List[ModelComparisonItem]
    best_model_by_profit: str
    best_model_by_winrate: str


class TradeForensicsItemSchema(BaseModel):
    entry_time: int
    exit_time: int
    direction: str
    entry_price: float
    exit_price: float
    pnl_usd: float
    r_multiple: float
    exit_reason: str
    failure_cause: str
    failure_explanation: str
    lesson_learned: str
    market_regime: str
    adx: float
    rsi: float
    atr: float


class ForensicsSummarySchema(BaseModel):
    total_losses_analyzed: int
    cause_distribution: Dict[str, float]
    top_failure_cause: str
    key_lessons: List[str]
    sample_diagnoses: List[TradeForensicsItemSchema]


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
    failure_cause: Optional[str] = None
    lesson_learned: Optional[str] = None


class BacktestResponse(BaseModel):
    initial_balance: float
    final_balance: float
    total_net_profit: float
    return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int = 0
    win_rate_pct: float
    win_rate_with_be_pct: float = 0.0
    profit_factor: float
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_usd: float
    avg_r_multiple: float
    equity_curve: List[float]
    trades: List[TradeRecordSchema]
    forensics: Optional[ForensicsSummarySchema] = None


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
