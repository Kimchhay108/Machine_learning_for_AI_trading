/**
 * API client for the XAUUSD Hybrid AI Trading Platform.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  instrument: string;
  timeframe: string;
}

export interface CandlesResponse {
  instrument: string;
  timeframe: string;
  count: number;
  candles: Candle[];
}

export interface SMCStructure {
  swing_highs: { timestamp: number; price: number }[];
  swing_lows: { timestamp: number; price: number }[];
  active_fvgs: { timestamp: number; type: 'BULLISH' | 'BEARISH'; top: number; bottom: number }[];
  recent_bos: { timestamp: number; direction: 'BULLISH' | 'BEARISH'; price: number }[];
  recent_choch: { timestamp: number; direction: 'BULLISH' | 'BEARISH'; price: number }[];
  order_blocks: { timestamp: number; direction: 'BULLISH' | 'BEARISH'; price: number }[];
  liquidity_sweeps: { timestamp: number; direction: 'BULLISH' | 'BEARISH'; price: number }[];
  ict_2022_setups?: { timestamp: number; type: 'BULLISH' | 'BEARISH'; price: number }[];
  silver_bullet_setups?: { timestamp: number; type: 'BULLISH' | 'BEARISH'; price: number }[];
}

export interface MarketFeaturesResponse {
  instrument: string;
  timeframe: string;
  timestamp: number;
  current_price: number;
  regime: {
    regime_label: string;
    regime_code: number;
    efficiency_ratio: number;
    trend_slope: number;
    is_trending: boolean;
    is_ranging: boolean;
    is_expansion: boolean;
    session: string;
  };
  indicators: {
    ema_9: number;
    ema_20: number;
    ema_50: number;
    ema_100: number;
    ema_200: number;
    ema_alignment: number;
    rsi_14: number;
    macd_line: number;
    macd_signal: number;
    macd_hist: number;
    atr_14: number;
    bb_upper: number;
    bb_middle: number;
    bb_lower: number;
    bb_bandwidth: number;
    stoch_rsi_k: number;
    stoch_rsi_d: number;
  };
  smc: SMCStructure;
}

export interface ModelSetupItem {
  model_key: string;
  model_name: string;
  direction: 'BUY' | 'SELL' | 'WAIT';
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
  risk_distance_usd: number;
  confidence_pct: number;
  why_explanation: string;
  key_drivers: string[];
}

export interface PredictionResponse {
  instrument: string;
  timeframe: string;
  prediction_timestamp: number;
  direction: 'BUY' | 'SELL' | 'WAIT';
  bullish_probability: number;
  bearish_probability: number;
  neutral_probability: number;
  confidence: number;
  market_regime: string;
  entry_reference_price: number;
  stop_loss_reference: number;
  take_profit_reference: number;
  risk_reward_ratio: number;
  model_version: string;
  contributing_features: {
    top_technical_drivers?: string[];
    top_smc_drivers?: string[];
    active_ict_setups?: string[];
    all_model_setups?: ModelSetupItem[];
    submodel_agreement?: {
      tech_prediction?: number;
      smc_prediction?: number;
      price_action_prediction?: number;
    };
  };
}

export interface ModelInfo {
  version: string;
  name: string;
  is_active: boolean;
  confidence_threshold: number;
  submodels: string[];
  train_date?: string;
}

export interface PerformanceResponse {
  model_version: string;
  directional_accuracy: number;
  selective_accuracy: number;
  coverage_pct: number;
  f1_score: number;
  expected_calibration_error: number;
  brier_score: number;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  regime_accuracies: Record<string, number>;
}

export interface TradeRecord {
  entry_time: number;
  exit_time: number;
  direction: 'BUY' | 'SELL';
  entry_price: number;
  exit_price: number;
  stop_loss: number;
  take_profit: number;
  position_size_oz: number;
  pnl_usd: number;
  pnl_pct: number;
  r_multiple: number;
  exit_reason: string;
}

export interface BacktestResponse {
  initial_balance: number;
  final_balance: number;
  total_net_profit: number;
  return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate_pct: number;
  profit_factor: number;
  max_drawdown_usd: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  avg_trade_usd: number;
  avg_r_multiple: number;
  equity_curve: number[];
  trades: TradeRecord[];
}

export interface ModelComparisonItem {
  model_key: string;
  model_name: string;
  optimal_timeframe?: string;
  sl_tp_logic: string;
  total_trades: number;
  win_rate_pct: number;
  total_net_profit: number;
  return_pct: number;
  profit_factor: number;
  max_drawdown_pct: number;
  avg_r_multiple: number;
}

export interface BacktestComparisonResponse {
  results: ModelComparisonItem[];
  best_model_by_profit: string;
  best_model_by_winrate: string;
}

export interface MarketStatus {
  instrument: string;
  is_market_open: boolean;
  status_code: string;
  status_message: string;
  current_utc_time: string;
  data_source_mode: string;
  latest_candle_time_utc: string;
  latest_available_price: number;
  total_stored_m5_candles: number;
}

export async function fetchMarketStatus(instrument: string = 'XAUUSD'): Promise<MarketStatus> {
  const res = await fetch(`${API_BASE}/api/market/status?instrument=${instrument}`);
  if (!res.ok) throw new Error(`Market status fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchCandles(
  instrument: string = 'XAUUSD',
  timeframe: string = 'M5',
  limit: number = 300
): Promise<CandlesResponse> {
  const res = await fetch(`${API_BASE}/api/market/candles?instrument=${instrument}&timeframe=${timeframe}&limit=${limit}`);
  if (!res.ok) throw new Error(`Candle fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchMarketFeatures(instrument: string = 'XAUUSD', timeframe: string = 'M5', limit: number = 100): Promise<MarketFeaturesResponse> {
  const res = await fetch(`${API_BASE}/api/market/features?instrument=${instrument}&timeframe=${timeframe}&limit=${limit}`);
  if (!res.ok) throw new Error(`Features fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchAIPrediction(instrument: string = 'XAUUSD', timeframe: string = 'M5'): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/api/ai/prediction?instrument=${instrument}&timeframe=${timeframe}`);
  if (!res.ok) throw new Error(`Prediction fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelPerformance(instrument: string = 'XAUUSD', version: string = 'v1.0.0'): Promise<PerformanceResponse> {
  const res = await fetch(`${API_BASE}/api/ai/performance?instrument=${instrument}&version=${version}`);
  if (!res.ok) throw new Error(`Performance fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<{ active_version: string; models: ModelInfo[] }> {
  const res = await fetch(`${API_BASE}/api/ai/models`);
  if (!res.ok) throw new Error(`Models fetch failed: ${res.statusText}`);
  return res.json();
}

export async function triggerRetrain(
  version: string,
  threshold: number,
  testRatio: number = 0.3,
  adminKey: string = 'xauusd-admin-secret-key'
): Promise<any> {
  const res = await fetch(`${API_BASE}/api/ai/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Admin-Key': adminKey,
    },
    body: JSON.stringify({ version, threshold, test_ratio: testRatio }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Retrain request failed');
  }
  return res.json();
}

export async function runBacktest(params: {
  version?: string;
  model_type?: string;
  timeframe?: string;
  initial_capital?: number;
  spread?: number;
  slippage?: number;
  risk_per_trade?: number;
  threshold?: number;
  max_holding_bars?: number;
}): Promise<BacktestResponse> {
  const res = await fetch(`${API_BASE}/api/backtest/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Backtest failed: ${res.statusText}`);
  return res.json();
}

export async function runBacktestComparison(params?: {
  version?: string;
  timeframe?: string;
  initial_capital?: number;
  spread?: number;
  slippage?: number;
  risk_per_trade?: number;
  threshold?: number;
  max_holding_bars?: number;
}): Promise<BacktestComparisonResponse> {
  const res = await fetch(`${API_BASE}/api/backtest/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params || {}),
  });
  if (!res.ok) throw new Error(`Model comparison failed: ${res.statusText}`);
  return res.json();
}

export async function fetchBacktestResults(): Promise<BacktestResponse> {
  const res = await fetch(`${API_BASE}/api/backtest/results`);
  if (!res.ok) throw new Error(`Backtest results fetch failed: ${res.statusText}`);
  return res.json();
}

export async function evaluatePredictions(horizonBars: number = 3, sampleLimit: number = 100): Promise<any> {
  const res = await fetch(`${API_BASE}/api/ai/predictions/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ horizon_bars: horizonBars, sample_limit: sampleLimit }),
  });
  if (!res.ok) throw new Error(`Evaluation failed: ${res.statusText}`);
  return res.json();
}

export async function fetchHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export interface PaperPosition {
  position_id: string;
  instrument: string;
  direction: 'BUY' | 'SELL';
  entry_time: number;
  entry_price: number;
  size_oz: number;
  stop_loss: number;
  take_profit: number;
  risk_usd: number;
  current_price: number;
  unrealized_pnl: number;
}

export interface PaperAccountSummary {
  initial_balance: number;
  balance: number;
  equity: number;
  net_profit: number;
  return_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate_pct: number;
  profit_factor: number;
  open_position: PaperPosition | null;
  is_running: boolean;
}

export async function fetchPaperAccount(): Promise<PaperAccountSummary> {
  const res = await fetch(`${API_BASE}/api/paper/account`);
  if (!res.ok) throw new Error('Paper account fetch failed');
  return res.json();
}

export async function fetchPaperTrades(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/api/paper/trades`);
  if (!res.ok) throw new Error('Paper trades fetch failed');
  return res.json();
}

export async function startPaperTrading(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/paper/start`, { method: 'POST' });
  if (!res.ok) throw new Error('Start paper trading failed');
  return res.json();
}

export async function stopPaperTrading(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/paper/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Stop paper trading failed');
  return res.json();
}

export async function resetPaperAccount(balance: number = 10000): Promise<any> {
  const res = await fetch(`${API_BASE}/api/paper/reset?initial_balance=${balance}`, { method: 'POST' });
  if (!res.ok) throw new Error('Reset paper account failed');
  return res.json();
}

export async function triggerPaperTick(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/paper/tick`, { method: 'POST' });
  if (!res.ok) throw new Error('Paper tick failed');
  return res.json();
}

