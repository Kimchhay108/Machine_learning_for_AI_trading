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
import { Maximize2, Minimize2 } from 'lucide-react';

interface TradingViewChartProps {
  candles: Candle[];
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
  features?: MarketFeaturesResponse | null;
  prediction?: PredictionResponse | null;
}

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

  // Overlay visibility toggles
  const [showEMAs, setShowEMAs] = useState(true);
  const [showSMC, setShowSMC] = useState(true);
  const [showAILevels, setShowAILevels] = useState(true);
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

    // Clean up prior chart
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

  // Update chart data
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
    }

    // Add SMC Markers (Swings, BOS, CHoCH, Sweeps)
    const markers: SeriesMarker<Time>[] = [];

    if (showSMC && features?.smc) {
      // Swing Highs
      features.smc.swing_highs.forEach((sh) => {
        if (uniqueMap.has(sh.timestamp)) {
          markers.push({
            time: sh.timestamp as Time,
            position: 'aboveBar',
            color: '#ef4444',
            shape: 'arrowDown',
            text: `SH ${sh.price}`,
          });
        }
      });

      // Swing Lows
      features.smc.swing_lows.forEach((sl) => {
        if (uniqueMap.has(sl.timestamp)) {
          markers.push({
            time: sl.timestamp as Time,
            position: 'belowBar',
            color: '#10b981',
            shape: 'arrowUp',
            text: `SL ${sl.price}`,
          });
        }
      });

      // BOS markers
      features.smc.recent_bos.forEach((bos) => {
        if (uniqueMap.has(bos.timestamp)) {
          markers.push({
            time: bos.timestamp as Time,
            position: bos.direction === 'BULLISH' ? 'belowBar' : 'aboveBar',
            color: '#38bdf8',
            shape: 'circle',
            text: `BOS ${bos.direction[0]}`,
          });
        }
      });

      // CHoCH markers
      features.smc.recent_choch.forEach((ch) => {
        if (uniqueMap.has(ch.timestamp)) {
          markers.push({
            time: ch.timestamp as Time,
            position: ch.direction === 'BULLISH' ? 'belowBar' : 'aboveBar',
            color: '#f59e0b',
            shape: 'square',
            text: `CHoCH ${ch.direction[0]}`,
          });
        }
      });

      // ICT 2022 Mentorship Model Markers
      features.smc.ict_2022_setups?.forEach((ict) => {
        if (uniqueMap.has(ict.timestamp)) {
          markers.push({
            time: ict.timestamp as Time,
            position: ict.type === 'BULLISH' ? 'belowBar' : 'aboveBar',
            color: '#eab308',
            shape: ict.type === 'BULLISH' ? 'arrowUp' : 'arrowDown',
            text: `ICT 2022 ${ict.type === 'BULLISH' ? 'BUY' : 'SELL'}`,
          });
        }
      });

      // ICT Silver Bullet Markers
      features.smc.silver_bullet_setups?.forEach((sb) => {
        if (uniqueMap.has(sb.timestamp)) {
          markers.push({
            time: sb.timestamp as Time,
            position: sb.type === 'BULLISH' ? 'belowBar' : 'aboveBar',
            color: '#c084fc',
            shape: 'circle',
            text: `Silver Bullet ${sb.type[0]}`,
          });
        }
      });
    }

    // Add AI Signal Marker on the latest candle
    if (showAILevels && prediction && prediction.direction !== 'WAIT') {
      const latestTime = sortedData[sortedData.length - 1]?.time;
      if (latestTime) {
        markers.push({
          time: latestTime,
          position: prediction.direction === 'BUY' ? 'belowBar' : 'aboveBar',
          color: prediction.direction === 'BUY' ? '#10b981' : '#ef4444',
          shape: prediction.direction === 'BUY' ? 'arrowUp' : 'arrowDown',
          text: `AI ${prediction.direction} (${(prediction.confidence * 100).toFixed(0)}%)`,
        });
      }
    }

    // Set markers sorted by timestamp
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    candleSeriesRef.current.setMarkers(markers);
  }, [candles, showEMAs, showSMC, showAILevels, features, prediction]);

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
      <div className="flex items-center justify-between px-4 py-2 border-b border-dark-700 bg-dark-850">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-gold-400 text-sm tracking-wide">XAUUSD</span>
          <span className="text-xs text-gray-400">Gold / USD</span>
          {lastCandleDate && (
            <span className="text-[10px] bg-dark-800 text-gold-400 px-2 py-0.5 rounded border border-dark-700 font-mono">
              Last Bar: {lastCandleDate}
            </span>
          )}
          <div className="h-4 w-px bg-dark-700 mx-1" />

          {/* Timeframe Buttons */}
          {(['M5', 'M15', 'H1'] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeChange(tf)}
              className={`px-2.5 py-1 text-xs font-semibold rounded transition ${
                timeframe === tf
                  ? 'bg-gold-500 text-dark-950 shadow-sm'
                  : 'bg-dark-700 text-gray-300 hover:bg-dark-600'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Overlay Layer Toggles & Fullscreen */}
        <div className="flex items-center space-x-3 text-xs">
          <label className="flex items-center space-x-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={showEMAs}
              onChange={(e) => setShowEMAs(e.target.checked)}
              className="rounded bg-dark-700 border-dark-600 text-gold-500 focus:ring-0"
            />
            <span>EMAs</span>
          </label>

          <label className="flex items-center space-x-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={showSMC}
              onChange={(e) => setShowSMC(e.target.checked)}
              className="rounded bg-dark-700 border-dark-600 text-gold-500 focus:ring-0"
            />
            <span>SMC Levels</span>
          </label>

          <label className="flex items-center space-x-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={showAILevels}
              onChange={(e) => setShowAILevels(e.target.checked)}
              className="rounded bg-dark-700 border-dark-600 text-gold-500 focus:ring-0"
            />
            <span>AI Signals</span>
          </label>

          <div className="h-4 w-px bg-dark-700 mx-1" />

          {/* Full Screen Toggle Button */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 px-2.5 rounded bg-dark-700 hover:bg-dark-600 text-gray-200 hover:text-white border border-dark-600 transition flex items-center space-x-1"
            title={isFullscreen ? 'Exit Full Screen' : 'Expand to Full Screen'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            <span className="text-[11px] font-medium">{isFullscreen ? 'Exit' : 'Full Screen'}</span>
          </button>
        </div>
      </div>

      {/* Trading Levels Banner (if active signal) */}
      {prediction && prediction.direction !== 'WAIT' && showAILevels && (
        <div className="flex items-center justify-between px-4 py-1.5 bg-dark-800 border-b border-dark-700 text-xs">
          <div className="flex items-center space-x-4">
            <span className="font-semibold text-gray-400">Target Setup:</span>
            <span className="text-gray-300">
              Entry: <strong className="text-white">${prediction.entry_reference_price.toFixed(2)}</strong>
            </span>
            <span className="text-bear-400">
              Stop Loss: <strong>${prediction.stop_loss_reference.toFixed(2)}</strong>
            </span>
            <span className="text-bull-400">
              Take Profit: <strong>${prediction.take_profit_reference.toFixed(2)}</strong>
            </span>
            <span className="text-gold-400">
              R:R: <strong>1:{prediction.risk_reward_ratio}</strong>
            </span>
          </div>
          <span className="text-gray-400">
            Confidence: <strong className="text-white">{(prediction.confidence * 100).toFixed(1)}%</strong>
          </span>
        </div>
      )}

      {/* Chart Canvas Area */}
      <div ref={chartContainerRef} className="flex-1 w-full min-h-[580px]" />
    </div>
  );
}
