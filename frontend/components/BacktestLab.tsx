'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  BacktestResponse,
  BacktestComparisonResponse,
  ModelComparisonItem,
  runBacktest,
  fetchBacktestResults,
  runBacktestComparison,
  fetchModels,
} from '../lib/api';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  Time,
} from 'lightweight-charts';
import {
  Play,
  RotateCcw,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Shield,
  Layers,
  Award,
  Zap,
  Target,
  BarChart3,
  Sliders,
  Clock,
  Brain,
  Calendar,
  Cpu,
} from 'lucide-react';

export default function BacktestLab() {
  const [loading, setLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [results, setResults] = useState<BacktestResponse | null>(null);
  const [comparison, setComparison] = useState<BacktestComparisonResponse | null>(null);

  // Model & Parameter states
  const [availableVersions, setAvailableVersions] = useState<string[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<string>('');
  const [modelType, setModelType] = useState<string>('ensemble');
  const [timeframe, setTimeframe] = useState<string>('M5');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [spread, setSpread] = useState(0.20);
  const [slippage, setSlippage] = useState(0.05);
  const [riskPct, setRiskPct] = useState(0.01);
  const [threshold, setThreshold] = useState(0.60);
  const [maxBars, setMaxBars] = useState(36);
  const [targetRR, setTargetRR] = useState<number | undefined>(1.2);
  const [sessionWindow, setSessionWindow] = useState<string>('cambodia_evening');
  const [timezoneDisplay, setTimezoneDisplay] = useState<'ICT' | 'UTC'>('ICT');

  // Chart ref
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const areaSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  // Load cached backtest results and model versions on mount
  useEffect(() => {
    fetchBacktestResults()
      .then((res) => setResults(res))
      .catch((err) => console.log('No cached backtest:', err));

    fetchModels()
      .then((res) => {
        const vList = (res.models || []).map((m) => m.version);
        setAvailableVersions(vList);
        setSelectedVersion(res.active_version || (vList.length > 0 ? vList[0] : ''));
      })
      .catch((err) => console.log('Failed to fetch model registry:', err));
  }, []);

  // Equity curve chart setup
  useEffect(() => {
    if (!chartContainerRef.current) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0d1117' },
        textColor: '#8b949e',
      },
      grid: {
        vertLines: { color: '#161b22' },
        horzLines: { color: '#161b22' },
      },
      rightPriceScale: {
        borderColor: '#21262d',
      },
      timeScale: {
        borderColor: '#21262d',
      },
    });

    const areaSeries = chart.addAreaSeries({
      topColor: 'rgba(16, 185, 129, 0.4)',
      bottomColor: 'rgba(16, 185, 129, 0.0)',
      lineColor: '#10b981',
      lineWidth: 2,
    });

    chartRef.current = chart;
    areaSeriesRef.current = areaSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update equity curve data
  useEffect(() => {
    if (!areaSeriesRef.current || !results || !results.equity_curve) return;

    const data = results.equity_curve.map((val, idx) => ({
      time: (idx + 1) as unknown as Time,
      value: val,
    }));
    areaSeriesRef.current.setData(data);
  }, [results]);

  const handleRunSimulation = async (selectedModel: string = modelType) => {
    setLoading(true);
    try {
      const res = await runBacktest({
        version: selectedVersion || undefined,
        model_type: selectedModel,
        timeframe: timeframe,
        session_window: sessionWindow,
        initial_capital: Number(initialCapital),
        spread: Number(spread),
        slippage: Number(slippage),
        risk_per_trade: Number(riskPct),
        threshold: Number(threshold),
        max_holding_bars: Number(maxBars),
        enable_breakeven: false,
        breakeven_r: 1.0,
        enable_partial_tp: false,
        partial_tp_r: 1.0,
        only_killzones: true,
        use_h1_gatekeeper: true,
        learn_from_mistakes: true,
        filter_high_impact_news: true,
        news_blackout_mins: 30,
        exclude_asian_session: true,
        use_full_history: true,
        strict_structural_entry: true,
        target_rr: targetRR || 1.2,
        post_loss_cooldown_bars: 6,
      });
      setResults(res);
    } catch (err: any) {
      alert(`Simulation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRunComparison = async () => {
    setComparisonLoading(true);
    try {
      const compRes = await runBacktestComparison({
        version: selectedVersion || undefined,
        timeframe: timeframe,
        session_window: sessionWindow,
        initial_capital: Number(initialCapital),
        spread: Number(spread),
        slippage: Number(slippage),
        risk_per_trade: Number(riskPct),
        threshold: Number(threshold),
        max_holding_bars: Number(maxBars),
        enable_breakeven: false,
        breakeven_r: 1.0,
        enable_partial_tp: false,
        partial_tp_r: 1.0,
        only_killzones: true,
        use_h1_gatekeeper: true,
        learn_from_mistakes: true,
        filter_high_impact_news: true,
        news_blackout_mins: 30,
        exclude_asian_session: true,
        use_full_history: true,
        strict_structural_entry: true,
        target_rr: targetRR || 1.2,
        post_loss_cooldown_bars: 6,
      });
      setComparison(compRes);
    } catch (err: any) {
      alert(`Comparison failed: ${err.message}`);
    } finally {
      setComparisonLoading(false);
    }
  };

  // Strategy logic description dictionary
  const strategyDetails: Record<string, { title: string; entry: string; sl: string; tp: string; badge: string }> = {
    ensemble: {
      title: 'Model E: Calibrated Multi-Model Ensemble',
      entry: 'Meta-calibrator stacked consensus of Technical, SMC, and Price Action submodels (P ≥ 50%)',
      sl: 'Dynamic Volatility Stop: 1.5x ATR buffer from execution fill',
      tp: `Volatility Target: 1.8x ATR buffer (${targetRR || 1.2} Risk:Reward ratio)`,
      badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
    },
    smc_ict: {
      title: 'Model B: Smart Money Concepts & ICT 2022 Mentorship',
      entry: 'ICT 2022 Mentorship (FVG + Discount/Premium retest), Silver Bullet window, Judas Swing, or Model B agreement',
      sl: 'Structural Invalidation: Active Swing Low -$0.50 (BUY) / Active Swing High +$0.50 (SELL)',
      tp: 'Targeting Opposing Liquidity Pool: 2.5x Risk multiple (2.5 Risk:Reward ratio)',
      badge: 'bg-gold-500/20 text-gold-300 border-gold-500/30',
    },
    technical: {
      title: 'Model A: Technical Momentum (Indicators)',
      entry: 'Trend EMA 9/20/50/200 alignment + RSI pullbacks (42-58) + MACD histogram directional expansion',
      sl: 'Volatility Envelope Stop: 1.5x ATR buffer from execution fill',
      tp: 'Trend Expansion Target: 3.0x ATR buffer (2.0 Risk:Reward ratio)',
      badge: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
    },
    price_action: {
      title: 'Model D: Price Action & Candlestick Morphology',
      entry: 'Pin bar rejection wicks, bullish/bearish engulfing candles evaluated by Model D classifier',
      sl: 'Candle Wick Extreme: Prior Candle Low -$0.50 (BUY) / Candle High +$0.50 (SELL)',
      tp: 'Key Level Reversal Target: 2.0x Risk multiple (2.0 Risk:Reward ratio)',
      badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    },
    volume_profile: {
      title: 'Model F: Volume Profile (Auction Market Theory)',
      entry: 'Tests of Value Area Low (VAL), Point of Control (POC) mean reversion, and Low Volume Node (LVN) breakouts',
      sl: 'Outside Value Area: VAL -$0.50 (BUY) / VAH +$0.50 (SELL)',
      tp: 'Opposing Value Area Boundary or 2.5x Risk multiple (2.5 Risk:Reward ratio)',
      badge: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
    },
    footprint: {
      title: 'Model G: Footprint & Order Flow (Delta Absorption)',
      entry: 'Cumulative Volume Delta (CVD) divergences, trapped buyers at highs, and trapped sellers at candle extremes',
      sl: 'Order Flow Invalidation: Candle Low -$0.50 (BUY) / Candle High +$0.50 (SELL)',
      tp: 'Order Flow Expansion Target: 2.0x Risk multiple (2.0 Risk:Reward ratio)',
      badge: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    },
  };

  const currentStrategy = strategyDetails[modelType] || strategyDetails['ensemble'];

  return (
    <div className="space-y-6">
      {/* Simulation Parameter Controls Bar */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-4 gap-2">
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200 flex items-center space-x-2">
            <Shield className="w-4 h-4 text-gold-500" />
            <span>Realistic Backtesting & Model Performance Lab</span>
          </h2>
          <span className="text-xs text-gray-400">
            Intrabar SL First Conflict Resolution | Spread & Slippage Applied | Zero Optimistic Bias
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-10 gap-3 text-xs">
          {/* Model Version Selector */}
          <div className="col-span-2 md:col-span-2">
            <label className="text-gray-400 block mb-1 font-semibold flex items-center space-x-1">
              <Cpu className="w-3.5 h-3.5 text-gold-400" />
              <span>Model Version</span>
            </label>
            <select
              value={selectedVersion}
              onChange={(e) => setSelectedVersion(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white font-medium focus:border-gold-500 outline-none"
            >
              {availableVersions.length === 0 ? (
                <option value="">Active Prod</option>
              ) : (
                availableVersions.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Model Selector */}
          <div className="col-span-2 md:col-span-2">
            <label className="text-gray-400 block mb-1 font-semibold flex items-center space-x-1">
              <Layers className="w-3.5 h-3.5 text-gold-400" />
              <span>Strategy Model</span>
            </label>
            <select
              value={modelType}
              onChange={(e) => {
                setModelType(e.target.value);
              }}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white font-medium focus:border-gold-500 outline-none"
            >
              <option value="ensemble">Model E: Calibrated Ensemble (Multi-Model)</option>
              <option value="smc_ict">Model B: SMC & ICT 2022 Mentorship</option>
              <option value="technical">Model A: Technical Momentum (Indicators)</option>
              <option value="price_action">Model D: Price Action (Wick Extremes)</option>
              <option value="volume_profile">Model F: Volume Profile (Auction Theory)</option>
              <option value="footprint">Model G: Footprint & Order Flow (Delta)</option>
            </select>
          </div>

          <div>
            <label className="text-gray-400 block mb-1 font-semibold flex items-center space-x-1">
              <Clock className="w-3.5 h-3.5 text-gold-400" />
              <span>Timeframe</span>
            </label>
            <select
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white font-semibold focus:border-gold-500 outline-none"
            >
              <option value="M5">M5 (ICT / Ens Entry)</option>
              <option value="M15">M15 (Tech / PA Entry)</option>
              <option value="H1">H1 (Macro Trend Bias)</option>
            </select>
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Initial Capital ($)</label>
            <input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Spread ($ USD)</label>
            <input
              type="number"
              step="0.05"
              value={spread}
              onChange={(e) => setSpread(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Slippage ($ USD)</label>
            <input
              type="number"
              step="0.01"
              value={slippage}
              onChange={(e) => setSlippage(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Risk Per Trade</label>
            <select
              value={riskPct}
              onChange={(e) => setRiskPct(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            >
              <option value={0.005}>0.5% Account</option>
              <option value={0.01}>1.0% Account</option>
              <option value={0.015}>1.5% Account</option>
              <option value={0.02}>2.0% Account</option>
            </select>
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Conf Threshold</label>
            <input
              type="number"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white font-mono"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Target R:R Mode</label>
            <select
              value={targetRR === undefined ? 1.2 : targetRR}
              onChange={(e) => setTargetRR(e.target.value ? Number(e.target.value) : 1.2)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white font-medium"
            >
              <option value={1.2}>1:1.2 (High Win-Rate Target - Recommended)</option>
              <option value={1.5}>1:1.5 (Standard Target)</option>
              <option value={2.0}>1:2.0 (Swing Runner)</option>
            </select>
          </div>

          {/* Session Window Selector */}
          <div className="col-span-2 md:col-span-2">
            <label className="text-gray-400 block mb-1 font-semibold flex items-center space-x-1">
              <Clock className="w-3.5 h-3.5 text-gold-400" />
              <span>Trading Session Window</span>
            </label>
            <select
              value={sessionWindow}
              onChange={(e) => setSessionWindow(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white font-medium focus:border-gold-500 outline-none"
            >
              <option value="cambodia_evening">🇰🇭 Cambodia Prime (7 PM - 11 PM ICT)</option>
              <option value="killzones">🌍 London & NY Killzones (14:00 - 22:00 ICT)</option>
              <option value="all_sessions">🌐 Full Multi-Session (Excl. Asian)</option>
            </select>
          </div>
        </div>

        {/* Built-In Institutional Safeguards Banner (Active by Default) */}
        <div className="flex flex-wrap items-center gap-2.5 mt-3 pt-3 border-t border-dark-800/80 text-[11px] text-gray-300">
          <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-dark-700/60 rounded px-2.5 py-1 text-emerald-400">
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-semibold">Triple-Trend Alignment Active</span>
            <span className="text-gray-400">(H1 200 EMA + M5 EMA 50/200)</span>
          </div>

          {sessionWindow === 'cambodia_evening' ? (
            <div className="flex items-center space-x-1.5 bg-dark-800/80 border border-gold-500/50 rounded px-2.5 py-1 text-gold-400 shadow-sm">
              <Clock className="w-3.5 h-3.5 text-gold-400" />
              <span className="font-bold">🇰🇭 Cambodia Prime (7 PM - 11 PM ICT)</span>
              <span className="text-gray-300 font-mono">(12:00 - 16:00 UTC)</span>
            </div>
          ) : sessionWindow === 'killzones' ? (
            <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-dark-700/60 rounded px-2.5 py-1 text-gold-400">
              <Zap className="w-3.5 h-3.5 text-gold-400" />
              <span className="font-semibold">Killzones Only</span>
              <span className="text-gray-400">(London & NY)</span>
            </div>
          ) : (
            <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-dark-700/60 rounded px-2.5 py-1 text-gold-400">
              <Clock className="w-3.5 h-3.5 text-gold-400" />
              <span className="font-semibold">Multi-Session Active</span>
            </div>
          )}

          <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-cyan-500/40 rounded px-2.5 py-1 text-cyan-400">
            <Shield className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-semibold">Vol Spike Veto</span>
            <span className="text-gray-400">(ATR ≤ 1.4x rolling)</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-indigo-500/40 rounded px-2.5 py-1 text-indigo-400">
            <span className="font-semibold">Anti-Cluster Cooldown</span>
            <span className="text-gray-400">(30m spacing)</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-dark-700/60 rounded px-2.5 py-1 text-rose-400">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
            <span className="font-semibold">News Blackout</span>
            <span className="text-gray-400">(±30m NFP/CPI/FOMC)</span>
          </div>

          <div className="flex items-center space-x-1.5 bg-dark-800/60 border border-dark-700/60 rounded px-2.5 py-1 text-amber-400">
            <Brain className="w-3.5 h-3.5 text-amber-400" />
            <span className="font-semibold">Failure Memory</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3 mt-3 pt-3 border-t border-dark-800">
          <button
            onClick={() => handleRunSimulation(modelType)}
            disabled={loading}
            className="bg-gold-500 hover:bg-gold-600 disabled:opacity-50 text-dark-950 font-bold py-2 px-4 rounded text-xs flex items-center space-x-1.5 transition shadow-sm"
          >
            {loading ? (
              <span>Simulating...</span>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Run {currentStrategy.title.split(':')[0]} Backtest</span>
              </>
            )}
          </button>

          <button
            onClick={handleRunComparison}
            disabled={comparisonLoading}
            className="bg-dark-800 hover:bg-dark-750 border border-gold-500/40 text-gold-400 hover:text-gold-300 font-bold py-2 px-4 rounded text-xs flex items-center space-x-1.5 transition"
          >
            {comparisonLoading ? (
              <span>Comparing 6 Models...</span>
            ) : (
              <>
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Compare All 6 Models Side-by-Side</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Model Strategy Specifications Banner */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <span className={`text-[11px] font-bold px-2 py-0.5 rounded border ${currentStrategy.badge}`}>
              {currentStrategy.title}
            </span>
            <span className="text-xs text-gray-400 font-medium">Configured Execution Rules</span>
          </div>
          <span className="text-[11px] text-gray-500 font-mono">Out-of-sample Gold M5</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs mt-3">
          <div className="bg-dark-850 p-2.5 rounded border border-dark-800">
            <span className="text-gray-400 block font-semibold mb-1 flex items-center space-x-1">
              <Zap className="w-3 h-3 text-gold-400" />
              <span>Entry Logic</span>
            </span>
            <span className="text-gray-200">{currentStrategy.entry}</span>
          </div>

          <div className="bg-dark-850 p-2.5 rounded border border-dark-800">
            <span className="text-gray-400 block font-semibold mb-1 flex items-center space-x-1">
              <Shield className="w-3 h-3 text-bear-400" />
              <span>Stop-Loss (SL) Rule</span>
            </span>
            <span className="text-bear-300">{currentStrategy.sl}</span>
          </div>

          <div className="bg-dark-850 p-2.5 rounded border border-dark-800">
            <span className="text-gray-400 block font-semibold mb-1 flex items-center space-x-1">
              <Target className="w-3 h-3 text-bull-400" />
              <span>Take-Profit (TP) Rule</span>
            </span>
            <span className="text-bull-300">{currentStrategy.tp}</span>
          </div>
        </div>
      </div>

      {/* Side-by-Side Comparative Performance Matrix */}
      {comparison && (
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5 space-y-4">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-2">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-white flex items-center space-x-2">
                <Award className="w-4 h-4 text-gold-400" />
                <span>Side-by-Side Strategy Comparison</span>
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                Evaluated under identical execution frictions ($0.20 spread, $0.05 slippage, same out-of-sample data).
              </p>
            </div>
            <div className="flex items-center space-x-2 text-xs">
              <span className="bg-bull-500/10 border border-bull-500/30 text-bull-400 px-2 py-0.5 rounded">
                🏆 Best Profit: <strong>{comparison.best_model_by_profit}</strong>
              </span>
              <span className="bg-gold-500/10 border border-gold-500/30 text-gold-400 px-2 py-0.5 rounded">
                🎯 Best Win Rate: <strong>{comparison.best_model_by_winrate}</strong>
              </span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-dark-850 text-gray-400 border-b border-dark-700 font-semibold">
                <tr>
                  <th className="py-2.5 px-3">Strategy Model</th>
                  <th className="py-2.5 px-3">Optimal Timeframe</th>
                  <th className="py-2.5 px-3">SL & TP Execution Rules</th>
                  <th className="py-2.5 px-3 text-right">Trades</th>
                  <th className="py-2.5 px-3 text-right">Win Rate</th>
                  <th className="py-2.5 px-3 text-right">Net Profit</th>
                  <th className="py-2.5 px-3 text-right">Return %</th>
                  <th className="py-2.5 px-3 text-right">Profit Factor</th>
                  <th className="py-2.5 px-3 text-right">Max DD</th>
                  <th className="py-2.5 px-3 text-right">Avg R</th>
                  <th className="py-2.5 px-3 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800 text-gray-300">
                {comparison.results.map((item) => {
                  const isSelected = modelType === item.model_key;
                  return (
                    <tr
                      key={item.model_key}
                      className={`hover:bg-dark-850/60 transition ${
                        isSelected ? 'bg-gold-500/5 border-l-2 border-gold-500' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 font-semibold text-white">
                        <div className="flex items-center space-x-1.5">
                          {isSelected && <span className="text-gold-400">▶</span>}
                          <span>{item.model_name}</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-gold-300 font-mono text-[11px] whitespace-nowrap">
                        {item.optimal_timeframe || 'M5 Entry'}
                      </td>
                      <td className="py-2.5 px-3 text-gray-400 text-[11px] max-w-xs truncate" title={item.sl_tp_logic}>
                        {item.sl_tp_logic}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono">{item.total_trades.toLocaleString()}</td>
                      <td className="py-2.5 px-3 text-right font-mono">
                        <div className="font-bold text-white">{item.win_rate_pct.toFixed(1)}%</div>
                        {item.win_rate_with_be_pct && item.win_rate_with_be_pct > item.win_rate_pct ? (
                          <div className="text-[10px] text-emerald-400 font-sans font-semibold">
                            {item.win_rate_with_be_pct.toFixed(1)}% w/ BE
                          </div>
                        ) : null}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right font-mono font-bold ${
                          item.total_net_profit >= 0 ? 'text-bull-400' : 'text-bear-400'
                        }`}
                      >
                        {item.total_net_profit >= 0 ? '+' : ''}${item.total_net_profit.toLocaleString()}
                      </td>
                      <td
                        className={`py-2.5 px-3 text-right font-mono ${
                          item.return_pct >= 0 ? 'text-bull-400' : 'text-bear-400'
                        }`}
                      >
                        {item.return_pct >= 0 ? '+' : ''}{item.return_pct.toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-gold-400">
                        {item.profit_factor.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-bear-400">
                        -{item.max_drawdown_pct.toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono">
                        {item.avg_r_multiple > 0 ? '+' : ''}{item.avg_r_multiple.toFixed(2)}R
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <button
                          onClick={() => {
                            setModelType(item.model_key);
                            handleRunSimulation(item.model_key);
                          }}
                          className={`px-2 py-1 rounded text-[11px] font-semibold transition ${
                            isSelected
                              ? 'bg-gold-500 text-dark-950'
                              : 'bg-dark-800 hover:bg-dark-750 text-gray-300 border border-dark-700'
                          }`}
                        >
                          {isSelected ? 'Active' : 'Load Model'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* KPI Cards */}
      {results && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Net Profit</span>
            <span className={`text-lg font-bold ${results.total_net_profit >= 0 ? 'text-bull-400' : 'text-bear-400'}`}>
              ${results.total_net_profit.toLocaleString()}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              {results.return_pct >= 0 ? '+' : ''}{results.return_pct}% return
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Win Rate</span>
            <span className="text-lg font-bold text-white">
              {results.win_rate_pct}%
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              {results.winning_trades}W / {results.losing_trades}L
              {results.breakeven_trades ? ` / ${results.breakeven_trades} BE` : ''} ({results.total_trades} trades)
            </span>
            {results.win_rate_with_be_pct && results.win_rate_with_be_pct > results.win_rate_pct ? (
              <span className="text-[10px] text-emerald-400 block font-semibold mt-0.5">
                🛡️ Win + BE Protection: {results.win_rate_with_be_pct}%
              </span>
            ) : null}
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Profit Factor</span>
            <span className="text-lg font-bold text-gold-400">
              {results.profit_factor.toFixed(2)}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">Gross Win / Gross Loss</span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Max Drawdown</span>
            <span className="text-lg font-bold text-bear-400">
              -{results.max_drawdown_pct}%
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              -${results.max_drawdown_usd.toFixed(2)}
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Sharpe Ratio</span>
            <span className="text-lg font-bold text-white">
              {results.sharpe_ratio.toFixed(2)}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">Risk-adjusted return</span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Avg R-Multiple</span>
            <span className="text-lg font-bold text-white">
              {results.avg_r_multiple > 0 ? '+' : ''}{results.avg_r_multiple.toFixed(2)}R
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Avg Trade: ${results.avg_trade_usd.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {/* Equity Curve Canvas */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
            Simulated Equity Curve (Out-of-Sample Period)
          </h3>
          <span className="text-xs text-gray-400 font-medium">
            Active: <strong className="text-gold-400">{currentStrategy.title}</strong>
          </span>
        </div>
        <div ref={chartContainerRef} className="w-full h-72" />
      </div>

      {/* AI Failure Forensics & Continuous Learning Panel */}
      {results && results.forensics && results.forensics.total_losses_analyzed > 0 && (
        <div className="bg-dark-900 border border-amber-500/30 rounded-lg p-5 space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-dark-800 pb-3">
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-amber-400" />
              <div>
                <h3 className="text-xs font-bold uppercase tracking-wider text-gray-100 flex items-center space-x-2">
                  <span>AI Trade Failure Forensics & Self-Correcting Engine</span>
                  <span className="bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2 py-0.5 rounded text-[10px]">
                    {results.forensics.total_losses_analyzed} Loss Autopsies Analyzed
                  </span>
                </h3>
                <p className="text-[11px] text-gray-400 mt-0.5">
                  Post-trade autopsy diagnoses why stop losses were triggered and prevents repeating known failure fingerprints.
                </p>
              </div>
            </div>

            <div className="text-right">
              <span className="text-[10px] text-gray-400 block">Primary Loss Trigger:</span>
              <span className="text-xs font-bold text-amber-300 font-mono">
                {results.forensics.top_failure_cause.replace(/_/g, ' ')}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Failure Distribution Breakdown */}
            <div className="bg-dark-850 border border-dark-750 rounded-lg p-3.5 space-y-2.5">
              <h4 className="text-[11px] font-bold uppercase tracking-wider text-gray-300 flex items-center space-x-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                <span>Loss Root-Cause Fingerprint Distribution</span>
              </h4>
              <div className="space-y-2 text-xs">
                {Object.entries(results.forensics.cause_distribution).map(([cause, pct]) => {
                  const label = cause.replace(/_/g, ' ');
                  return (
                    <div key={cause} className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="text-gray-300">{label}</span>
                        <span className="font-mono font-bold text-amber-400">{pct}%</span>
                      </div>
                      <div className="w-full bg-dark-800 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-amber-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${Math.min(100, pct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Continuous Self-Correction Rules Active */}
            <div className="bg-dark-850 border border-dark-750 rounded-lg p-3.5 space-y-2.5">
              <h4 className="text-[11px] font-bold uppercase tracking-wider text-gray-300 flex items-center space-x-1.5">
                <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                <span>Learned Self-Correction Rules (Veto Filters Active)</span>
              </h4>
              <div className="space-y-2 text-[11px]">
                <div className="p-2 rounded bg-dark-800/80 border border-dark-700/60 flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold">✓</span>
                  <div>
                    <span className="font-semibold text-gray-200">Low-Momentum Chop Filter:</span>
                    <p className="text-gray-400 text-[10px]">
                      Vetoes entries when ADX &lt; 18.0 to eliminate sideways consolidation stop-loss whipsaws.
                    </p>
                  </div>
                </div>

                <div className="p-2 rounded bg-dark-800/80 border border-dark-700/60 flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold">✓</span>
                  <div>
                    <span className="font-semibold text-gray-200">Continuous Mistake Quarantine:</span>
                    <p className="text-gray-400 text-[10px]">
                      Abstains from re-entering for 3 bars (15 mins) after an SL hit to avoid revenge chop and recurring traps.
                    </p>
                  </div>
                </div>

                <div className="p-2 rounded bg-dark-800/80 border border-dark-700/60 flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold">✓</span>
                  <div>
                    <span className="font-semibold text-gray-200">Momentum Climax Exhaustion Filter:</span>
                    <p className="text-gray-400 text-[10px]">
                      Vetoes BUY when RSI &gt; 68.0 and SELL when RSI &lt; 32.0 to avoid buying tops or selling bottoms.
                    </p>
                  </div>
                </div>

                <div className="p-2 rounded bg-dark-800/80 border border-dark-700/60 flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold">✓</span>
                  <div>
                    <span className="font-semibold text-gray-200">Volatility Expansion Spike Filter:</span>
                    <p className="text-gray-400 text-[10px]">
                      Abstains when ATR surges &gt; 1.4x rolling 20-period baseline to avoid high-spread slippage traps.
                    </p>
                  </div>
                </div>

                <div className="p-2 rounded bg-dark-800/80 border border-dark-700/60 flex items-start space-x-2">
                  <span className="text-emerald-400 font-bold">✓</span>
                  <div>
                    <span className="font-semibold text-gray-200">Tier-1 High-Impact News Blackout Window:</span>
                    <p className="text-gray-400 text-[10px]">
                      Abstains from trading ±30 minutes around US NFP, CPI inflation, and FOMC rate decisions to avoid extreme slippage.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Sample Trade Autopsy Cards */}
          {results.forensics.sample_diagnoses && results.forensics.sample_diagnoses.length > 0 && (
            <div className="border-t border-dark-800 pt-3">
              <h4 className="text-[11px] font-bold uppercase tracking-wider text-gray-300 mb-2.5">
                Recent Failed Trade Autopsies & Improvement Lessons
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                {results.forensics.sample_diagnoses.slice(0, 3).map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-dark-850/80 border border-dark-700 rounded-lg text-xs space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          item.direction === 'BUY'
                            ? 'bg-bull-500/20 text-bull-400'
                            : 'bg-bear-500/20 text-bear-400'
                        }`}
                      >
                        {item.direction}
                      </span>
                      <span className="text-bear-400 font-mono font-bold text-[11px]">
                        ${item.pnl_usd.toFixed(2)} ({item.r_multiple.toFixed(2)}R)
                      </span>
                    </div>

                    <div className="text-[10px] bg-amber-500/10 text-amber-300 border border-amber-500/20 px-2 py-0.5 rounded font-mono">
                      ⚠️ {item.failure_cause.replace(/_/g, ' ')}
                    </div>

                    <p className="text-[11px] text-gray-300 leading-tight">
                      {item.failure_explanation}
                    </p>

                    <div className="pt-1.5 border-t border-dark-750 text-[10px] text-emerald-400">
                      💡 <strong>Lesson:</strong> {item.lesson_learned}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Monthly Performance Breakdown (SL, TP & Profit by Year/Month) */}
      {results && results.monthly_breakdown && results.monthly_breakdown.length > 0 && (
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gold-500" />
              <span>Monthly Performance Breakdown (Year & Month SL & Profit Report)</span>
            </h3>
            <span className="text-xs text-gray-400">
              {results.monthly_breakdown.length} Months Tracked across Historical Dataset
            </span>
          </div>

          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-dark-850 text-gray-400 sticky top-0 border-dark-700 border-b">
                <tr>
                  <th className="py-2 px-3">Period</th>
                  <th className="py-2 px-3">Total Trades</th>
                  <th className="py-2 px-3 text-bull-400">Winning Trades (TP)</th>
                  <th className="py-2 px-3 text-bear-400">Losing Trades (SL Hits)</th>
                  <th className="py-2 px-3">Win Rate %</th>
                  <th className="py-2 px-3 text-right">Net Profit ($)</th>
                  <th className="py-2 px-3 text-right">Profit Factor</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800 text-gray-300 font-mono">
                {results.monthly_breakdown.map((m) => {
                  const isProfitable = m.net_profit >= 0;
                  return (
                    <tr key={m.period} className="hover:bg-dark-850/50">
                      <td className="py-2 px-3 font-bold text-white">{m.period}</td>
                      <td className="py-2 px-3 text-gray-300">{m.total_trades}</td>
                      <td className="py-2 px-3 text-bull-400 font-bold">{m.winning_trades}</td>
                      <td className="py-2 px-3 text-bear-400 font-bold">{m.losing_trades}</td>
                      <td className="py-2 px-3">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                            m.win_rate_pct >= 40
                              ? 'bg-bull-500/15 text-bull-400'
                              : 'bg-dark-750 text-gray-400'
                          }`}
                        >
                          {m.win_rate_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td
                        className={`py-2 px-3 text-right font-bold ${
                          isProfitable ? 'text-bull-400' : 'text-bear-400'
                        }`}
                      >
                        {isProfitable ? '+' : ''}${m.net_profit.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </td>
                      <td
                        className={`py-2 px-3 text-right font-bold ${
                          m.profit_factor >= 1.0 ? 'text-bull-400' : 'text-bear-400'
                        }`}
                      >
                        {m.profit_factor.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Simulated Trades Log Table */}
      {results && results.trades && results.trades.length > 0 && (
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3 gap-2">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
                Execution Log ({results.trades.length} Completed Positions)
              </h3>
              <span className="text-[11px] text-gray-400">
                Stop Loss & Take Profit Levels Set by Model Rules | Trade Autopsy Attached to Losses
              </span>
            </div>
            {/* Timezone Toggle */}
            <div className="flex items-center space-x-1.5 bg-dark-800 p-1 rounded border border-dark-750 text-[11px]">
              <span className="text-gray-400 px-1 font-medium">Timezone:</span>
              <button
                type="button"
                onClick={() => setTimezoneDisplay('ICT')}
                className={`px-2 py-0.5 rounded font-semibold transition ${
                  timezoneDisplay === 'ICT'
                    ? 'bg-gold-500 text-dark-950 shadow-sm'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                🇰🇭 Cambodia (ICT / UTC+7)
              </button>
              <button
                type="button"
                onClick={() => setTimezoneDisplay('UTC')}
                className={`px-2 py-0.5 rounded font-semibold transition ${
                  timezoneDisplay === 'UTC'
                    ? 'bg-gold-500 text-dark-950 shadow-sm'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                UTC
              </button>
            </div>
          </div>
          <div className="overflow-x-auto max-h-80 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-dark-850 text-gray-400 sticky top-0 border-b border-dark-700">
                <tr>
                  <th className="py-2 px-3">Date & Time ({timezoneDisplay === 'ICT' ? 'ICT' : 'UTC'})</th>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Entry Price</th>
                  <th className="py-2 px-3">Stop Loss</th>
                  <th className="py-2 px-3">Take Profit</th>
                  <th className="py-2 px-3">Exit Price</th>
                  <th className="py-2 px-3">PnL ($)</th>
                  <th className="py-2 px-3">R-Multiple</th>
                  <th className="py-2 px-3">Exit Reason</th>
                  <th className="py-2 px-3">Forensics & Lesson Learned</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800 text-gray-300 font-mono">
                {results.trades.map((t, idx) => {
                  const ms = t.entry_time > 1e11 ? t.entry_time : t.entry_time * 1000;
                  const d = timezoneDisplay === 'ICT' ? new Date(ms + 7 * 3600 * 1000) : new Date(ms);
                  const dateStr = d.toISOString().slice(0, 10);
                  const timeStr = d.toISOString().slice(11, 16);
                  const hourNum = parseInt(timeStr.slice(0, 2), 10);
                  const minStr = timeStr.slice(3, 5);
                  const ampm = hourNum >= 12 ? 'PM' : 'AM';
                  const hour12 = hourNum % 12 || 12;
                  const time12Str = `${hour12}:${minStr} ${ampm}`;

                  const exitMs = t.exit_time > 1e11 ? t.exit_time : t.exit_time * 1000;
                  const exitD = timezoneDisplay === 'ICT' ? new Date(exitMs + 7 * 3600 * 1000) : new Date(exitMs);
                  const exitDateStr = exitD.toISOString().slice(0, 10);
                  const exitTimeStr = exitD.toISOString().slice(11, 16);

                  return (
                    <tr key={idx} className="hover:bg-dark-850/50">
                      <td className="py-2 px-3 whitespace-nowrap">
                        <div className="flex flex-col" title={`Entered: ${dateStr} ${timeStr} ${timezoneDisplay}\nExited: ${exitDateStr} ${exitTimeStr} ${timezoneDisplay}`}>
                          <span className="text-white font-medium text-[11px]">{dateStr}</span>
                          <span className="text-[10px] text-gold-400 font-semibold">
                            {timeStr} {timezoneDisplay} {timezoneDisplay === 'ICT' && `(${time12Str})`}
                          </span>
                        </div>
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                            t.direction === 'BUY' ? 'bg-bull-500/20 text-bull-400' : 'bg-bear-500/20 text-bear-400'
                          }`}
                        >
                          {t.direction}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-semibold text-white">${t.entry_price.toFixed(2)}</td>
                      <td className="py-2 px-3 text-bear-400">${t.stop_loss.toFixed(2)}</td>
                      <td className="py-2 px-3 text-bull-400">${t.take_profit.toFixed(2)}</td>
                      <td className="py-2 px-3">${t.exit_price.toFixed(2)}</td>
                      <td className={`py-2 px-3 font-semibold ${t.pnl_usd >= 0 ? 'text-bull-400' : 'text-bear-400'}`}>
                        {t.pnl_usd >= 0 ? '+' : ''}${t.pnl_usd.toFixed(2)}
                      </td>
                      <td className="py-2 px-3">{t.r_multiple > 0 ? '+' : ''}{t.r_multiple.toFixed(2)}R</td>
                      <td className="py-2 px-3">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[10px] font-sans font-bold ${
                          t.exit_reason === 'TAKE_PROFIT'
                            ? 'bg-bull-500/20 text-bull-300'
                            : t.exit_reason === 'STOP_LOSS'
                            ? 'bg-bear-500/20 text-bear-300'
                            : 'bg-dark-700 text-gray-300'
                        }`}
                      >
                        {t.exit_reason}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-sans">
                      {t.failure_cause ? (
                        <div className="flex flex-col space-y-0.5">
                          <span className="text-[10px] font-semibold text-amber-400 flex items-center space-x-1">
                            <span>⚠️</span>
                            <span>{t.failure_cause.replace(/_/g, ' ')}</span>
                          </span>
                          {t.lesson_learned && (
                            <span className="text-[9px] text-gray-400 truncate max-w-xs" title={t.lesson_learned}>
                              💡 {t.lesson_learned}
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-[10px] text-emerald-400/80">✓ Validated Trade</span>
                      )}
                    </td>
                  </tr>
                );
              })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
