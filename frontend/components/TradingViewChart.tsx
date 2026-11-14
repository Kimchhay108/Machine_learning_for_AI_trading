'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  ColorType,
  Time,
  SeriesMarker,
} from 'lightweight-charts';
import { Candle, MarketFeaturesResponse, PredictionResponse } from '../lib/api';
import {
  Maximize2,
  Minimize2,
  Layers,
  Sparkles,
  Zap,
  Clock,
  Compass,
  Filter,
  Eye,
  Info,
} from 'lucide-react';

interface TradingViewChartProps {
  candles: Candle[];
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
  features?: MarketFeaturesResponse | null;
  prediction?: PredictionResponse | null;
}

type MarkerFilterMode = 'triggers' | 'structure' | 'all' | 'clean';

export default function TradingViewChart({
  candles,
  timeframe,
  onTimeframeChange,
  features,
  prediction,
}: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const ema20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ema50SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ema200SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Overlay visibility toggles & anti-stacking filters
  const [showEMAs, setShowEMAs] = useState(true);
  const [markerFilter, setMarkerFilter] = useState<MarkerFilterMode>('triggers');
  const [showTfGuide, setShowTfGuide] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    }, 50);
    return () => clearTimeout(timer);
  }, [isFullscreen]);

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
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#21262d',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: '#21262d',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    // EMA overlays
    const ema20Series = chart.addLineSeries({
      color: '#38bdf8',
      lineWidth: 1,
      title: 'EMA 20',
    });

    const ema50Series = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 1,
      title: 'EMA 50',
    });

    const ema200Series = chart.addLineSeries({
      color: '#a855f7',
      lineWidth: 2,
      title: 'EMA 200',
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    ema20SeriesRef.current = ema20Series;
    ema50SeriesRef.current = ema50Series;
    ema200SeriesRef.current = ema200Series;

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

  // Update chart data & apply anti-collision marker consolidation
  useEffect(() => {
    if (!candleSeriesRef.current || candles.length === 0) return;

    // Convert candle data
    const chartData: CandlestickData<Time>[] = candles.map((c) => ({
      time: c.timestamp as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    // Ensure sorted and unique timestamps
    const uniqueMap = new Map<number, CandlestickData<Time>>();
    chartData.forEach((d) => uniqueMap.set(d.time as number, d));
    const sortedData = Array.from(uniqueMap.values()).sort((a, b) => (a.time as number) - (b.time as number));

    candleSeriesRef.current.setData(sortedData);

    // Calculate lightweight EMA lines if enabled
    if (showEMAs && ema20SeriesRef.current && ema50SeriesRef.current && ema200SeriesRef.current) {
      const calcEMA = (period: number) => {
        const k = 2 / (period + 1);
        let ema = sortedData[0]?.close || 0;
        return sortedData.map((d, i) => {
          if (i === 0) return { time: d.time, value: ema };
          ema = d.close * k + ema * (1 - k);
          return { time: d.time, value: ema };
        });
      };

      ema20SeriesRef.current.setData(calcEMA(20));
      ema50SeriesRef.current.setData(calcEMA(50));
      ema200SeriesRef.current.setData(calcEMA(200));
    } else if (ema20SeriesRef.current && ema50SeriesRef.current && ema200SeriesRef.current) {
      ema20SeriesRef.current.setData([]);
      ema50SeriesRef.current.setData([]);
      ema200SeriesRef.current.setData([]);
    }

    // =========================================================================
    // ANTI-STACKING & CLEAN CONSOLIDATION ENGINE
    // Groups events per bar timestamp so markers never overlap or stack vertically
    // =========================================================================
    if (markerFilter === 'clean') {
      candleSeriesRef.current.setMarkers([]);
      return;
    }

    interface BarSignalBundle {
      timestamp: number;
      aiDirection?: 'BUY' | 'SELL';
      aiConfidence?: number;
      ictSetup?: 'BULLISH' | 'BEARISH';
      silverBullet?: 'BULLISH' | 'BEARISH';
      choch?: 'BULLISH' | 'BEARISH';
      bos?: 'BULLISH' | 'BEARISH';
      swing?: { type: 'SH' | 'SL'; price: number };
    }

    const barMap = new Map<number, BarSignalBundle>();

    const getBundle = (ts: number): BarSignalBundle => {
      let b = barMap.get(ts);
      if (!b) {
        b = { timestamp: ts };
        barMap.set(ts, b);
      }
      return b;
    };

    if (features?.smc) {
      // Collect Swings
      features.smc.swing_highs?.forEach((sh) => {
        if (uniqueMap.has(sh.timestamp)) {
          getBundle(sh.timestamp).swing = { type: 'SH', price: sh.price };
        }
      });
      features.smc.swing_lows?.forEach((sl) => {
        if (uniqueMap.has(sl.timestamp)) {
          getBundle(sl.timestamp).swing = { type: 'SL', price: sl.price };
        }
      });

      // Collect Structure Breaks
      features.smc.recent_bos?.forEach((bos) => {
        if (uniqueMap.has(bos.timestamp)) {
          getBundle(bos.timestamp).bos = bos.direction as 'BULLISH' | 'BEARISH';
        }
      });

      // Collect Character Shifts
      features.smc.recent_choch?.forEach((ch) => {
        if (uniqueMap.has(ch.timestamp)) {
          getBundle(ch.timestamp).choch = ch.direction as 'BULLISH' | 'BEARISH';
        }
      });

      // Collect ICT Setups
      features.smc.ict_2022_setups?.forEach((ict) => {
        if (uniqueMap.has(ict.timestamp)) {
          getBundle(ict.timestamp).ictSetup = ict.type as 'BULLISH' | 'BEARISH';
        }
      });

      // Collect Silver Bullet
      features.smc.silver_bullet_setups?.forEach((sb) => {
        if (uniqueMap.has(sb.timestamp)) {
          getBundle(sb.timestamp).silverBullet = sb.type as 'BULLISH' | 'BEARISH';
        }
      });
    }

    // Collect Latest AI Prediction
    if (prediction && prediction.direction !== 'WAIT') {
      const latestTime = sortedData[sortedData.length - 1]?.time as number;
      if (latestTime) {
        const b = getBundle(latestTime);
        b.aiDirection = prediction.direction as 'BUY' | 'SELL';
        b.aiConfidence = prediction.confidence;
      }
    }

    // Resolve exactly ONE clean consolidated marker per bar
    const finalMarkers: SeriesMarker<Time>[] = [];

    barMap.forEach((bundle, ts) => {
      const isBullish =
        bundle.aiDirection === 'BUY' ||
        bundle.ictSetup === 'BULLISH' ||
        bundle.silverBullet === 'BULLISH' ||
        bundle.choch === 'BULLISH' ||
        bundle.bos === 'BULLISH' ||
        bundle.swing?.type === 'SL';

      // 1. FILTER: Triggers Only (Actionable ICT & AI setups)
      if (markerFilter === 'triggers') {
        if (bundle.aiDirection) {
          finalMarkers.push({
            time: ts as Time,
            position: bundle.aiDirection === 'BUY' ? 'belowBar' : 'aboveBar',
            color: bundle.aiDirection === 'BUY' ? '#10b981' : '#ef4444',
            shape: bundle.aiDirection === 'BUY' ? 'arrowUp' : 'arrowDown',
            text: `AI ${bundle.aiDirection} (${((bundle.aiConfidence || 0) * 100).toFixed(0)}%)`,
          });
        } else if (bundle.ictSetup) {
          const isBuy = bundle.ictSetup === 'BULLISH';
          finalMarkers.push({
            time: ts as Time,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: '#eab308',
            shape: isBuy ? 'arrowUp' : 'arrowDown',
            text: `ICT 2022 ${isBuy ? 'BUY' : 'SELL'}`,
          });
        } else if (bundle.silverBullet) {
          const isBuy = bundle.silverBullet === 'BULLISH';
          finalMarkers.push({
            time: ts as Time,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: '#c084fc',
            shape: 'circle',
            text: `Silver Bullet ${isBuy ? 'BUY' : 'SELL'}`,
          });
        }
        return;
      }

      // 2. FILTER: Structure Only (CHoCH, BOS, Key Swings)
      if (markerFilter === 'structure') {
        if (bundle.choch) {
          const isBuy = bundle.choch === 'BULLISH';
          finalMarkers.push({
            time: ts as Time,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: '#f59e0b',
            shape: 'square',
            text: `CHoCH ${isBuy ? '▲' : '▼'}`,
          });
        } else if (bundle.bos) {
          const isBuy = bundle.bos === 'BULLISH';
          finalMarkers.push({
            time: ts as Time,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: '#38bdf8',
            shape: 'circle',
            text: `BOS ${isBuy ? '▲' : '▼'}`,
          });
        } else if (bundle.swing) {
          const isLow = bundle.swing.type === 'SL';
          finalMarkers.push({
            time: ts as Time,
            position: isLow ? 'belowBar' : 'aboveBar',
            color: isLow ? '#10b981' : '#ef4444',
            shape: isLow ? 'arrowUp' : 'arrowDown',
            text: `${bundle.swing.type} $${bundle.swing.price.toFixed(0)}`,
          });
        }
        return;
      }

      // 3. FILTER: All Signals (Merged cleanly with secondary tags to prevent overlap)
      if (markerFilter === 'all') {
        let label = '';
        let shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square' = isBullish ? 'arrowUp' : 'arrowDown';
        let color = isBullish ? '#10b981' : '#ef4444';
        let position: 'belowBar' | 'aboveBar' = isBullish ? 'belowBar' : 'aboveBar';

        if (bundle.aiDirection) {
          color = bundle.aiDirection === 'BUY' ? '#10b981' : '#ef4444';
          label = `AI ${bundle.aiDirection}`;
          if (bundle.ictSetup) label += ' + ICT';
        } else if (bundle.ictSetup) {
          color = '#eab308';
          label = `ICT ${bundle.ictSetup === 'BULLISH' ? 'BUY' : 'SELL'}`;
          if (bundle.choch) label += ' (CHoCH)';
          else if (bundle.bos) label += ' (BOS)';
        } else if (bundle.silverBullet) {
          color = '#c084fc';
          shape = 'circle';
          label = `Silver Bullet ${bundle.silverBullet === 'BULLISH' ? 'BUY' : 'SELL'}`;
        } else if (bundle.choch) {
          color = '#f59e0b';
          shape = 'square';
          label = `CHoCH ${bundle.choch === 'BULLISH' ? '▲' : '▼'}`;
        } else if (bundle.bos) {
          color = '#38bdf8';
          shape = 'circle';
          label = `BOS ${bundle.bos === 'BULLISH' ? '▲' : '▼'}`;
        } else if (bundle.swing) {
          label = `${bundle.swing.type} $${bundle.swing.price.toFixed(0)}`;
        }

        if (label) {
          finalMarkers.push({
            time: ts as Time,
            position,
            color,
            shape,
            text: label,
          });
        }
      }
    });

    finalMarkers.sort((a, b) => (a.time as number) - (b.time as number));
    candleSeriesRef.current.setMarkers(finalMarkers);
  }, [candles, showEMAs, markerFilter, features, prediction]);

  const lastCandleDate =
    candles.length > 0
      ? new Date(candles[candles.length - 1].timestamp * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' UTC'
      : '';

  return (
    <div
      className={`flex flex-col bg-dark-900 border border-dark-700 rounded-lg overflow-hidden transition-all ${
        isFullscreen ? 'fixed inset-0 z-[100] w-screen h-screen rounded-none' : 'w-full h-full'
      }`}
    >
      {/* Chart Toolbar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2 border-b border-dark-700 bg-dark-850 gap-2">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-gold-400 text-sm tracking-wide">XAUUSD</span>
          <span className="text-xs text-gray-400">Gold / USD</span>
          {lastCandleDate && (
            <span className="text-[10px] bg-dark-800 text-gold-400 px-2 py-0.5 rounded border border-dark-700 font-mono">
              Last Bar: {lastCandleDate}
            </span>
          )}
          <div className="h-4 w-px bg-dark-700 mx-1" />

          {/* Timeframe Selector Buttons */}
          <div className="flex items-center space-x-1 bg-dark-900 p-0.5 rounded border border-dark-750">
            {(['M5', 'M15', 'H1'] as const).map((tf) => (
              <button
                key={tf}
                onClick={() => onTimeframeChange(tf)}
                className={`px-2.5 py-1 text-xs font-bold rounded transition ${
                  timeframe === tf
                    ? 'bg-gold-500 text-dark-950 shadow-sm'
                    : 'text-gray-400 hover:text-white hover:bg-dark-800'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Model Optimal Timeframe Presets */}
          <div className="hidden lg:flex items-center space-x-1 pl-1">
            <button
              onClick={() => onTimeframeChange('M5')}
              className={`text-[10px] px-2 py-0.5 rounded border transition font-medium ${
                timeframe === 'M5'
                  ? 'bg-gold-500/20 border-gold-500/50 text-gold-300'
                  : 'bg-dark-800 border-dark-700 text-gray-400 hover:text-gray-200'
              }`}
              title="Optimal execution for Model B (ICT 2022 FVG & Silver Bullet)"
            >
              ICT 2022 (M5 Entry)
            </button>

            <button
              onClick={() => onTimeframeChange('M15')}
              className={`text-[10px] px-2 py-0.5 rounded border transition font-medium ${
                timeframe === 'M15'
                  ? 'bg-blue-500/20 border-blue-500/50 text-blue-300'
                  : 'bg-dark-800 border-dark-700 text-gray-400 hover:text-gray-200'
              }`}
              title="Optimal execution for Model A (Technical Momentum) & Model D (Price Action)"
            >
              Tech / PA (M15 Entry)
            </button>

            <button
              onClick={() => onTimeframeChange('H1')}
              className={`text-[10px] px-2 py-0.5 rounded border transition font-medium ${
                timeframe === 'H1'
                  ? 'bg-purple-500/20 border-purple-500/50 text-purple-300'
                  : 'bg-dark-800 border-dark-700 text-gray-400 hover:text-gray-200'
              }`}
              title="Higher Timeframe Narrative, Draw on Liquidity & Macro Trend Confirmation"
            >
              HTF Bias (H1 Confirmation)
            </button>
          </div>
        </div>

        {/* Clean Display Filters & Tooling */}
        <div className="flex items-center space-x-3 text-xs">
          {/* Anti-Stacking Marker Filter Selector */}
          <div className="flex items-center space-x-1.5 bg-dark-900 border border-dark-700 rounded px-2 py-1">
            <Filter className="w-3 h-3 text-gold-400" />
            <span className="text-[11px] text-gray-400 font-medium">Markers:</span>
            <select
              value={markerFilter}
              onChange={(e) => setMarkerFilter(e.target.value as MarkerFilterMode)}
              className="bg-transparent text-white text-[11px] font-semibold outline-none cursor-pointer"
            >
              <option value="triggers" className="bg-dark-900 text-white">
                Triggers Only (ICT / AI)
              </option>
              <option value="structure" className="bg-dark-900 text-white">
                Structure Only (BOS / CHoCH)
              </option>
              <option value="all" className="bg-dark-900 text-white">
                All (Consolidated 1/Bar)
              </option>
              <option value="clean" className="bg-dark-900 text-white">
                Clean (Naked Candles)
              </option>
            </select>
          </div>

          {/* EMA Overlay Toggle */}
          <label className="flex items-center space-x-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={showEMAs}
              onChange={(e) => setShowEMAs(e.target.checked)}
              className="rounded bg-dark-700 border-dark-600 text-gold-500 focus:ring-0"
            />
            <span className="text-[11px]">EMAs</span>
          </label>

          {/* Timeframe Guide Toggle */}
          <button
            onClick={() => setShowTfGuide(!showTfGuide)}
            className={`p-1 px-2 rounded text-[11px] font-medium border transition flex items-center space-x-1 ${
              showTfGuide
                ? 'bg-gold-500/20 border-gold-500 text-gold-300'
                : 'bg-dark-800 hover:bg-dark-700 border-dark-700 text-gray-400 hover:text-white'
            }`}
            title="View Multi-Timeframe Entry & Confirmation Guide"
          >
            <Compass className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">TF Guide</span>
          </button>

          <div className="h-4 w-px bg-dark-700 mx-1" />

          {/* Full Screen Toggle Button */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 px-2.5 rounded bg-dark-700 hover:bg-dark-600 text-gray-200 hover:text-white border border-dark-600 transition flex items-center space-x-1"
            title={isFullscreen ? 'Exit Full Screen' : 'Expand to Full Screen'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Multi-Timeframe Effectiveness Strategy Guide Panel */}
      {showTfGuide && (
        <div className="bg-dark-850 border-b border-dark-700 px-4 py-3 text-xs transition-all">
          <div className="flex items-center justify-between mb-2">
            <span className="font-bold uppercase tracking-wider text-gold-400 flex items-center space-x-1.5 text-[11px]">
              <Compass className="w-3.5 h-3.5" />
              <span>Multi-Timeframe Strategy Matrix — Optimal Entry & Confirmation</span>
            </span>
            <button
              onClick={() => setShowTfGuide(false)}
              className="text-gray-400 hover:text-white text-xs"
            >
              ✕ Close
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
            {/* Model B */}
            <div className="bg-dark-900 p-2.5 rounded border border-gold-500/30">
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-gold-400">Model B: SMC & ICT 2022</span>
                <span className="text-[10px] bg-gold-500/20 text-gold-300 px-1.5 py-0.5 rounded font-mono">
                  M5 Entry
                </span>
              </div>
              <p className="text-gray-300 text-[11px] leading-relaxed">
                <strong>HTF Confirmation:</strong> H1 for Draw on Liquidity & Old High/Low liquidity pool run.<br />
                <strong>LTF Entry:</strong> M5 displacement candle + Fair Value Gap (FVG) discount retest during London/NY macro windows.<br />
                <em>Prevents false FVG traps on low timeframes.</em>
              </p>
              <button
                onClick={() => onTimeframeChange('M5')}
                className="mt-2 text-[10px] font-bold text-dark-950 bg-gold-500 hover:bg-gold-400 px-2 py-0.5 rounded transition w-full"
              >
                Set Chart to M5 Entry
              </button>
            </div>

            {/* Model A */}
            <div className="bg-dark-900 p-2.5 rounded border border-blue-500/30">
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-blue-400">Model A: Technical Momentum</span>
                <span className="text-[10px] bg-blue-500/20 text-blue-300 px-1.5 py-0.5 rounded font-mono">
                  M15 Entry
                </span>
              </div>
              <p className="text-gray-300 text-[11px] leading-relaxed">
                <strong>HTF Confirmation:</strong> H1 EMA 50/200 macro trend filter.<br />
                <strong>LTF Entry:</strong> M15 RSI pullbacks (42–58) and MACD histogram expansion.<br />
                <em>M15 removes M5 noise & whipsaws while keeping tight risk:reward.</em>
              </p>
              <button
                onClick={() => onTimeframeChange('M15')}
                className="mt-2 text-[10px] font-bold text-white bg-blue-600 hover:bg-blue-500 px-2 py-0.5 rounded transition w-full"
              >
                Set Chart to M15 Entry
              </button>
            </div>

            {/* Model D */}
            <div className="bg-dark-900 p-2.5 rounded border border-emerald-500/30">
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-emerald-400">Model D: Price Action</span>
                <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-mono">
                  M15 Entry
                </span>
              </div>
              <p className="text-gray-300 text-[11px] leading-relaxed">
                <strong>HTF Confirmation:</strong> H1 major Support/Resistance & Session High/Low.<br />
                <strong>LTF Entry:</strong> M15 clean pin bar rejection wicks and engulfing absorption.<br />
                <em>Avoids false 1-tick spread wick artifacts common on M5.</em>
              </p>
              <button
                onClick={() => onTimeframeChange('M15')}
                className="mt-2 text-[10px] font-bold text-white bg-emerald-600 hover:bg-emerald-500 px-2 py-0.5 rounded transition w-full"
              >
                Set Chart to M15 Entry
              </button>
            </div>

            {/* Model E */}
            <div className="bg-dark-900 p-2.5 rounded border border-purple-500/30">
              <div className="flex items-center justify-between mb-1">
                <span className="font-bold text-purple-400">Model E: Calibrated Ensemble</span>
                <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded font-mono">
                  M5 Precision
                </span>
              </div>
              <p className="text-gray-300 text-[11px] leading-relaxed">
                <strong>HTF Confirmation:</strong> H1 & M15 stacked probability agreement.<br />
                <strong>LTF Entry:</strong> M5 calibrated execution when multi-model confidence exceeds threshold.<br />
                <em>Balances macro stability with precise risk sizing.</em>
              </p>
              <button
                onClick={() => onTimeframeChange('M5')}
                className="mt-2 text-[10px] font-bold text-white bg-purple-600 hover:bg-purple-500 px-2 py-0.5 rounded transition w-full"
              >
                Set Chart to M5 Ensemble
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Candlestick Chart Viewport */}
      <div ref={chartContainerRef} className="flex-1 w-full min-h-[420px]" />
    </div>
  );
}
