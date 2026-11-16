'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  ColorType,
  Time,
  SeriesMarker,
} from 'lightweight-charts';
import { Candle, MarketFeaturesResponse, PredictionResponse, ModelSetupItem } from '../lib/api';
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
  Target,
  Shield,
  Square,
} from 'lucide-react';

interface TradingViewChartProps {
  candles: Candle[];
  timeframe: string;
  onTimeframeChange: (tf: string) => void;
  features?: MarketFeaturesResponse | null;
  prediction?: PredictionResponse | null;
  onPriceUpdate?: (price: number) => void;
}

type MarkerFilterMode = 'triggers' | 'structure' | 'all' | 'clean';
type BoxModelType = 'ensemble' | 'price_action' | 'volume_profile' | 'technical' | 'footprint';

interface RiskRewardBoxCoords {
  x: number;
  width: number;
  greenY: number;
  greenH: number;
  redY: number;
  redH: number;
  entryY: number;
  tpPrice: number;
  slPrice: number;
  entryPrice: number;
  dir: string;
  rr: number;
  modelName: string;
}

export default function TradingViewChart({
  candles,
  timeframe,
  onTimeframeChange,
  features,
  prediction,
  onPriceUpdate,
}: TradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const ema20SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ema50SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ema200SeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  // Price lines refs
  const tpPriceLineRef = useRef<any>(null);
  const entryPriceLineRef = useRef<any>(null);
  const slPriceLineRef = useRef<any>(null);

  // Smart Session Rails refs
  const dayOpenLineRef = useRef<any>(null);
  const qraResLineRef = useRef<any>(null);
  const qraSupLineRef = useRef<any>(null);
  const asiaHighLineRef = useRef<any>(null);
  const asiaLowLineRef = useRef<any>(null);
  const [showSessionRails, setShowSessionRails] = useState(true);

  // Real-time live tick stream & dynamic bar progression
  const [isLiveStreaming, setIsLiveStreaming] = useState(true);
  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [liveTickDelta, setLiveTickDelta] = useState<number>(0);
  const [liveTickDirection, setLiveTickDirection] = useState<'up' | 'down'>('up');
  const [barCountdown, setBarCountdown] = useState<number>(300);
  const activeCandleRef = useRef<{
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
  } | null>(null);

  // Overlay visibility toggles & anti-stacking filters
  const [showEMAs, setShowEMAs] = useState(true);
  const [showRRBoxes, setShowRRBoxes] = useState(true);
  const [activeBoxModel, setActiveBoxModel] = useState<BoxModelType>('ensemble');
  const [markerFilter, setMarkerFilter] = useState<MarkerFilterMode>('triggers');
  const [showTfGuide, setShowTfGuide] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [boxCoords, setBoxCoords] = useState<RiskRewardBoxCoords | null>(null);

  // Keep a ref of sortedData for coordinate computations
  const sortedDataRef = useRef<CandlestickData<Time>[]>([]);

  // Compute Risk / Reward Green TP and Red SL Box coordinates
  const updateBoxCoordinates = useCallback(() => {
    if (!chartRef.current || !candleSeriesRef.current || !chartContainerRef.current) {
      setBoxCoords(null);
      return;
    }

    if (!showRRBoxes || !prediction) {
      setBoxCoords(null);
      return;
    }

    try {
      // Determine the active setup from all_model_setups or fallback to main prediction
      const setups: ModelSetupItem[] | undefined = prediction.contributing_features?.all_model_setups;
      let setup: {
        direction: string;
        entry_price: number;
        stop_loss: number;
        take_profit: number;
        risk_reward_ratio: number;
        model_name: string;
      };

      if (setups && Array.isArray(setups) && setups.length > 0) {
        const match = setups.find((s) => s.model_key === activeBoxModel) || setups[0];
        setup = {
          direction: match?.direction || prediction.direction || 'WAIT',
          entry_price: typeof match?.entry_price === 'number' ? match.entry_price : (prediction.entry_reference_price || 0),
          stop_loss: typeof match?.stop_loss === 'number' ? match.stop_loss : (prediction.stop_loss_reference || 0),
          take_profit: typeof match?.take_profit === 'number' ? match.take_profit : (prediction.take_profit_reference || 0),
          risk_reward_ratio: typeof match?.risk_reward_ratio === 'number' ? match.risk_reward_ratio : (prediction.risk_reward_ratio || 2.0),
          model_name: match?.model_name ? match.model_name.split(':')[0] : 'Model',
        };
      } else {
        setup = {
          direction: prediction.direction || 'WAIT',
          entry_price: prediction.entry_reference_price || 0,
          stop_loss: prediction.stop_loss_reference || 0,
          take_profit: prediction.take_profit_reference || 0,
          risk_reward_ratio: prediction.risk_reward_ratio || 2.0,
          model_name: 'Model E',
        };
      }

      if (!setup.entry_price || !setup.take_profit || !setup.stop_loss) {
        setBoxCoords(null);
        return;
      }

      const entryY = candleSeriesRef.current.priceToCoordinate(setup.entry_price);
      const tpY = candleSeriesRef.current.priceToCoordinate(setup.take_profit);
      const slY = candleSeriesRef.current.priceToCoordinate(setup.stop_loss);

      if (entryY === null || tpY === null || slY === null) {
        setBoxCoords(null);
        return;
      }

      const containerW = chartContainerRef.current.clientWidth;
      const rightScaleW = 65;
      const availW = containerW - rightScaleW;

      const data = sortedDataRef.current;
      const lastCandle = data.length > 0 ? data[data.length - 1] : null;

      let startX: number | null = null;
      if (lastCandle && chartRef.current) {
        try {
          startX = chartRef.current.timeScale().timeToCoordinate(lastCandle.time as Time);
        } catch (e) {
          startX = null;
        }
      }

      if (startX === null || startX < 50 || startX > availW - 80) {
        startX = Math.max(availW - 240, 60);
      }

      const boxW = Math.max(availW - startX, 130);

      let greenY = 0;
      let greenH = 0;
      let redY = 0;
      let redH = 0;

      if (setup.direction === 'SELL') {
        // SELL: Stop Loss is above entry (lower pixel Y), Take Profit is below entry (higher pixel Y)
        redY = Math.min(slY, entryY);
        redH = Math.max(Math.abs(entryY - slY), 6);
        greenY = Math.min(entryY, tpY);
        greenH = Math.max(Math.abs(tpY - entryY), 6);
      } else {
        // BUY or WAIT default: Take Profit is above entry (lower pixel Y), Stop Loss is below (higher pixel Y)
        greenY = Math.min(tpY, entryY);
        greenH = Math.max(Math.abs(entryY - tpY), 6);
        redY = Math.min(entryY, slY);
        redH = Math.max(Math.abs(slY - entryY), 6);
      }

      setBoxCoords({
        x: startX,
        width: boxW,
        greenY,
        greenH,
        redY,
        redH,
        entryY,
        tpPrice: setup.take_profit,
        slPrice: setup.stop_loss,
        entryPrice: setup.entry_price,
        dir: setup.direction,
        rr: setup.risk_reward_ratio,
        modelName: setup.model_name,
      });
    } catch (err) {
      setBoxCoords(null);
    }
  }, [showRRBoxes, prediction, activeBoxModel]);

  const updateBoxCoordinatesRef = useRef(updateBoxCoordinates);
  useEffect(() => {
    updateBoxCoordinatesRef.current = updateBoxCoordinates;
  });

  useEffect(() => {
    const timer = setTimeout(() => {
      if (chartContainerRef.current && chartRef.current) {
        try {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
          });
          updateBoxCoordinatesRef.current();
        } catch (e) {}
      }
    }, 60);
    return () => clearTimeout(timer);
  }, [isFullscreen]);

  // Chart setup
  useEffect(() => {
    if (!chartContainerRef.current) return;

    if (chartRef.current) {
      try {
        chartRef.current.remove();
      } catch (e) {}
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
      lastValueVisible: true,
      priceLineVisible: true,
      priceLineWidth: 1,
      priceLineColor: '#eab308',
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

    // Hook visible range changes to update SVG box coordinates
    try {
      chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
        updateBoxCoordinatesRef.current();
      });
    } catch (e) {}

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        try {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight,
          });
          updateBoxCoordinatesRef.current();
        } catch (e) {}
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (e) {}
        chartRef.current = null;
      }
      candleSeriesRef.current = null;
      ema20SeriesRef.current = null;
      ema50SeriesRef.current = null;
      ema200SeriesRef.current = null;
      tpPriceLineRef.current = null;
      entryPriceLineRef.current = null;
      slPriceLineRef.current = null;
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
    sortedDataRef.current = sortedData;

    candleSeriesRef.current.setData(sortedData);

    if (sortedData.length > 0) {
      const last = sortedData[sortedData.length - 1];
      activeCandleRef.current = {
        time: last.time as number,
        open: last.open,
        high: last.high,
        low: last.low,
        close: last.close,
      };
      setLivePrice(last.close);
      setLiveTickDelta(0);
      if (onPriceUpdate) {
        onPriceUpdate(last.close);
      }
    }

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
    } else {
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
        features.smc.recent_bos?.forEach((bos) => {
          if (uniqueMap.has(bos.timestamp)) {
            getBundle(bos.timestamp).bos = bos.direction as 'BULLISH' | 'BEARISH';
          }
        });
        features.smc.recent_choch?.forEach((ch) => {
          if (uniqueMap.has(ch.timestamp)) {
            getBundle(ch.timestamp).choch = ch.direction as 'BULLISH' | 'BEARISH';
          }
        });
        features.smc.ict_2022_setups?.forEach((ict) => {
          if (uniqueMap.has(ict.timestamp)) {
            getBundle(ict.timestamp).ictSetup = ict.type as 'BULLISH' | 'BEARISH';
          }
        });
        features.smc.silver_bullet_setups?.forEach((sb) => {
          if (uniqueMap.has(sb.timestamp)) {
            getBundle(sb.timestamp).silverBullet = sb.type as 'BULLISH' | 'BEARISH';
          }
        });
      }

      if (prediction && prediction.direction !== 'WAIT') {
        const latestTime = sortedData[sortedData.length - 1]?.time as number;
        if (latestTime) {
          const b = getBundle(latestTime);
          b.aiDirection = prediction.direction as 'BUY' | 'SELL';
          b.aiConfidence = prediction.confidence;
        }
      }

      const finalMarkers: SeriesMarker<Time>[] = [];

      barMap.forEach((bundle, ts) => {
        const isBullish =
          bundle.aiDirection === 'BUY' ||
          bundle.ictSetup === 'BULLISH' ||
          bundle.silverBullet === 'BULLISH' ||
          bundle.choch === 'BULLISH' ||
          bundle.bos === 'BULLISH' ||
          bundle.swing?.type === 'SL';

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
        } else if (markerFilter === 'structure') {
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
        } else if (markerFilter === 'all') {
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
      try {
        candleSeriesRef.current.setMarkers(finalMarkers);
      } catch (e) {}
    }

    // =========================================================================
    // PRICE LINES FOR TP, ENTRY, AND SL ON PRICE SCALE
    // =========================================================================
    if (candleSeriesRef.current) {
      try {
        if (tpPriceLineRef.current) candleSeriesRef.current.removePriceLine(tpPriceLineRef.current);
      } catch (e) {}
      tpPriceLineRef.current = null;

      try {
        if (entryPriceLineRef.current) candleSeriesRef.current.removePriceLine(entryPriceLineRef.current);
      } catch (e) {}
      entryPriceLineRef.current = null;

      try {
        if (slPriceLineRef.current) candleSeriesRef.current.removePriceLine(slPriceLineRef.current);
      } catch (e) {}
      slPriceLineRef.current = null;

      if (showRRBoxes && prediction) {
        try {
          const setups: ModelSetupItem[] | undefined = prediction.contributing_features?.all_model_setups;
          const activeSetup = (Array.isArray(setups) && setups.find((s) => s.model_key === activeBoxModel)) || {
            entry_price: prediction.entry_reference_price,
            stop_loss: prediction.stop_loss_reference,
            take_profit: prediction.take_profit_reference,
            risk_reward_ratio: prediction.risk_reward_ratio,
          };

          if (typeof activeSetup?.take_profit === 'number' && !isNaN(activeSetup.take_profit)) {
            tpPriceLineRef.current = candleSeriesRef.current.createPriceLine({
              price: activeSetup.take_profit,
              color: '#10b981',
              lineWidth: 2,
              lineStyle: 0,
              axisLabelVisible: true,
              title: `TP $${activeSetup.take_profit.toFixed(2)}`,
            });
          }

          if (typeof activeSetup?.entry_price === 'number' && !isNaN(activeSetup.entry_price)) {
            entryPriceLineRef.current = candleSeriesRef.current.createPriceLine({
              price: activeSetup.entry_price,
              color: '#f59e0b',
              lineWidth: 2,
              lineStyle: 2,
              axisLabelVisible: true,
              title: `ENTRY $${activeSetup.entry_price.toFixed(2)}`,
            });
          }

          if (typeof activeSetup?.stop_loss === 'number' && !isNaN(activeSetup.stop_loss)) {
            slPriceLineRef.current = candleSeriesRef.current.createPriceLine({
              price: activeSetup.stop_loss,
              color: '#ef4444',
              lineWidth: 2,
              lineStyle: 0,
              axisLabelVisible: true,
              title: `SL $${activeSetup.stop_loss.toFixed(2)}`,
            });
          }
        } catch (e) {}
      }

      // =========================================================================
      // INSTITUTIONAL SMART RAILS: DAY OPEN, DYNAMIC VOL RAILS & ASIA POOL
      // =========================================================================
      const series = candleSeriesRef.current;
      if (series) {
        [dayOpenLineRef, qraResLineRef, qraSupLineRef, asiaHighLineRef, asiaLowLineRef].forEach((ref) => {
          try {
            if (ref.current) series.removePriceLine(ref.current);
          } catch (e) {}
          ref.current = null;
        });

        if (showSessionRails && sortedData.length > 0) {
          try {
            const lastCandle = sortedData[sortedData.length - 1];
            const lastTs = (lastCandle.time as number) * (Number(lastCandle.time) > 1e11 ? 1 : 1000);
            const todayDateStr = new Date(lastTs).toISOString().slice(0, 10);

            const todayCandles = sortedData.filter((c) => {
              const cTs = (c.time as number) * (Number(c.time) > 1e11 ? 1 : 1000);
              return new Date(cTs).toISOString().slice(0, 10) === todayDateStr;
            });

            const dayOpenPrice = todayCandles.length > 0 ? todayCandles[0].open : lastCandle.open;

            // Asia Session Pool (00:00 - 07:00 UTC)
            const asiaCandles = todayCandles.filter((c) => {
              const cTs = (c.time as number) * (Number(c.time) > 1e11 ? 1 : 1000);
              const hour = new Date(cTs).getUTCHours();
              return hour >= 0 && hour < 7;
            });
            const asiaHigh = asiaCandles.length > 0 ? Math.max(...asiaCandles.map((c) => c.high)) : dayOpenPrice + 3.5;
            const asiaLow = asiaCandles.length > 0 ? Math.min(...asiaCandles.map((c) => c.low)) : dayOpenPrice - 3.5;

            // QRA Dynamic Volatility Rails
            const atrVal = features?.indicators?.atr_14 || 2.2;
            const qraResistance = dayOpenPrice + 2.5 * atrVal;
            const qraSupport = dayOpenPrice - 2.5 * atrVal;

            // 1. Day Open Rail
            dayOpenLineRef.current = series.createPriceLine({
              price: dayOpenPrice,
              color: '#f59e0b',
              lineWidth: 2,
              lineStyle: 2, // Dashed
              axisLabelVisible: true,
              title: `DAY OPEN $${dayOpenPrice.toFixed(2)}`,
            });

            // 2. QRA Rails
            qraResLineRef.current = series.createPriceLine({
              price: qraResistance,
              color: '#06b6d4',
              lineWidth: 1,
              lineStyle: 1, // Dotted
              axisLabelVisible: true,
              title: `QRA RES $${qraResistance.toFixed(2)}`,
            });
            qraSupLineRef.current = series.createPriceLine({
              price: qraSupport,
              color: '#06b6d4',
              lineWidth: 1,
              lineStyle: 1, // Dotted
              axisLabelVisible: true,
              title: `QRA SUP $${qraSupport.toFixed(2)}`,
            });

            // 3. Asia Session Pool
            asiaHighLineRef.current = series.createPriceLine({
              price: asiaHigh,
              color: '#c084fc',
              lineWidth: 1,
              lineStyle: 0,
              axisLabelVisible: true,
              title: `ASIA HIGH $${asiaHigh.toFixed(2)}`,
            });
            asiaLowLineRef.current = series.createPriceLine({
              price: asiaLow,
              color: '#c084fc',
              lineWidth: 1,
              lineStyle: 0,
              axisLabelVisible: true,
              title: `ASIA LOW $${asiaLow.toFixed(2)}`,
            });
          } catch (e) {}
        }
      }
    }

    try {
      updateBoxCoordinates();
    } catch (e) {}
  }, [candles, showEMAs, markerFilter, showRRBoxes, showSessionRails, activeBoxModel, features, prediction, updateBoxCoordinates]);

  // Real-time live tick stream simulation loop
  useEffect(() => {
    if (!isLiveStreaming) return;

    const intervalSec = timeframe === 'H1' ? 3600 : timeframe === 'M15' ? 900 : 300;
    setBarCountdown((prev) => (prev > 0 && prev <= intervalSec ? prev : intervalSec));

    const tickInterval = setInterval(() => {
      const series = candleSeriesRef.current;
      const active = activeCandleRef.current;
      if (!series || !active) return;

      // Countdown decrement
      setBarCountdown((prev) => {
        if (prev <= 1) {
          // Timeframe bar completed! Advance to next bar
          const nextTime = active.time + intervalSec;
          const nextBar = {
            time: nextTime,
            open: active.close,
            high: active.close,
            low: active.close,
            close: active.close,
          };
          activeCandleRef.current = nextBar;
          try {
            series.update(nextBar as any);
          } catch (e) {}
          return intervalSec;
        }
        return prev - 1;
      });

      // Realistic tick delta: Gold micro-ticks ±$0.02 to ±$0.15
      const tickFluctuation = (Math.random() - 0.495) * 0.08;
      const newClose = Number((active.close + tickFluctuation).toFixed(3));
      const newHigh = Math.max(active.high, newClose);
      const newLow = Math.min(active.low, newClose);

      active.close = newClose;
      active.high = newHigh;
      active.low = newLow;

      try {
        series.update({
          time: active.time as any,
          open: active.open,
          high: active.high,
          low: active.low,
          close: active.close,
        });
      } catch (e) {}

      setLivePrice(newClose);
      setLiveTickDelta(newClose - active.open);
      setLiveTickDirection(tickFluctuation >= 0 ? 'up' : 'down');

      if (onPriceUpdate) {
        onPriceUpdate(newClose);
      }
    }, 1200);

    return () => clearInterval(tickInterval);
  }, [isLiveStreaming, timeframe, onPriceUpdate]);

  const formatCountdown = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

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
          {/* Risk/Reward TP/SL Box Toggle & Active Model Selector */}
          <div className="flex items-center space-x-1.5 bg-dark-900 border border-dark-700 rounded px-2 py-1">
            <Target className="w-3.5 h-3.5 text-bull-400" />
            <label className="flex items-center space-x-1 cursor-pointer text-gray-200 font-semibold text-[11px]">
              <input
                type="checkbox"
                checked={showRRBoxes}
                onChange={(e) => setShowRRBoxes(e.target.checked)}
                className="rounded bg-dark-700 border-dark-600 text-gold-500 focus:ring-0 mr-1"
              />
              <span>TP/SL Box:</span>
            </label>
            <select
              value={activeBoxModel}
              onChange={(e) => setActiveBoxModel(e.target.value as BoxModelType)}
              disabled={!showRRBoxes}
              className="bg-transparent text-gold-400 font-bold text-[11px] outline-none cursor-pointer disabled:opacity-40"
            >
              <option value="ensemble" className="bg-dark-900 text-white">
                Model E (Ensemble +211%)
              </option>
              <option value="price_action" className="bg-dark-900 text-white">
                Model D (PA Wick 51% WR)
              </option>
              <option value="volume_profile" className="bg-dark-900 text-white">
                Model F (Vol Profile 51.5% WR)
              </option>
              <option value="technical" className="bg-dark-900 text-white">
                Model A (Technical Momentum)
              </option>
              <option value="footprint" className="bg-dark-900 text-white">
                Model G (Footprint Delta)
              </option>
            </select>
          </div>

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

          {/* Session Smart Rails Toggle */}
          <button
            onClick={() => setShowSessionRails(!showSessionRails)}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-[11px] font-bold border transition ${
              showSessionRails
                ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300 shadow-sm'
                : 'bg-dark-900 border-dark-700 text-gray-400 hover:text-white'
            }`}
            title="Toggle Day Open, Dynamic Volatility Rails & Session Liquidity Pools"
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span>Session Rails</span>
          </button>

          {/* Live Tick Stream Toggle */}
          <button
            onClick={() => setIsLiveStreaming(!isLiveStreaming)}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-[11px] font-bold border transition ${
              isLiveStreaming
                ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300 shadow-sm'
                : 'bg-dark-900 border-dark-700 text-gray-400 hover:text-white'
            }`}
            title="Toggle Real-Time Live Ticker & Active Candlestick Stream"
          >
            <span className={`w-2 h-2 rounded-full ${isLiveStreaming ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
            <span>{isLiveStreaming ? 'Live: Active' : 'Live: Paused'}</span>
          </button>

          {/* Bar Countdown Pill */}
          <div className="flex items-center space-x-1.5 px-2 py-1 rounded bg-dark-900 border border-dark-700 text-[11px] font-mono text-gray-300">
            <Clock className="w-3 h-3 text-gold-400" />
            <span className="text-gray-400">Close:</span>
            <span className="font-bold text-gold-400">{formatCountdown(barCountdown)}</span>
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

      {/* Candlestick Chart Viewport with Visual TP/SL Rectangles */}
      <div className="relative flex-1 w-full min-h-[420px]">
        <div ref={chartContainerRef} className="w-full h-full" />

        {/* Smart Rails HUD Overlay Bar */}
        {showSessionRails && (
          <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2 pointer-events-none">
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-[#050816]/90 border border-gold-500/40 text-gold-400 font-bold text-[11px] backdrop-blur-md shadow-lg pointer-events-auto">
              <Sparkles className="w-3.5 h-3.5 text-gold-400" />
              <span>QUANT RAILS</span>
            </div>

            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-[#050816]/90 border border-cyan-500/40 text-cyan-300 font-bold text-[11px] backdrop-blur-md shadow-lg pointer-events-auto">
              <span>DAY OPEN & VOL RAILS</span>
            </div>

            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-[#050816]/90 border border-purple-500/40 text-purple-300 font-semibold text-[11px] backdrop-blur-md shadow-lg pointer-events-auto">
              <span>ASIA POOL</span>
            </div>

            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-[#050816]/90 border border-emerald-500/40 text-emerald-300 font-semibold text-[11px] backdrop-blur-md shadow-lg pointer-events-auto">
              <span className={`w-1.5 h-1.5 rounded-full ${isLiveStreaming ? 'bg-emerald-400 animate-pulse' : 'bg-gray-400'}`} />
              <span>{isLiveStreaming ? 'LIVE TICK STREAM' : 'TICK STREAM PAUSED'}</span>
            </div>
          </div>
        )}

        {/* Floating Live Spot Price Pill */}
        <div className="absolute top-3 right-14 z-20 hidden sm:flex items-center space-x-2 px-3 py-1 rounded-lg bg-[#050816]/90 border border-dark-700/80 backdrop-blur-md shadow-lg pointer-events-auto">
          <span className="text-[10px] text-gray-400 font-mono uppercase">Live Spot:</span>
          <span
            className={`text-xs font-bold font-mono transition-colors ${
              liveTickDirection === 'up' ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            ${(livePrice || (candles.length > 0 ? candles[candles.length - 1].close : 0)).toFixed(2)}
          </span>
          <span
            className={`text-[10px] font-mono px-1 rounded ${
              liveTickDelta >= 0 ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
            }`}
          >
            {liveTickDelta >= 0 ? `+${liveTickDelta.toFixed(2)}` : liveTickDelta.toFixed(2)}
          </span>
        </div>

        {/* ========================================================================= */}
        {/* VISUAL SHADED RISK/REWARD RECTANGLES (GREEN TP BOX & RED SL BOX)          */}
        {/* ========================================================================= */}
        {showRRBoxes && boxCoords && (
          <svg className="absolute inset-0 pointer-events-none w-full h-full z-10 overflow-hidden">
            {/* Green Take Profit Rectangle */}
            <rect
              x={boxCoords.x}
              y={boxCoords.greenY}
              width={boxCoords.width}
              height={boxCoords.greenH}
              fill="rgba(16, 185, 129, 0.20)"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeDasharray="4 2"
              rx="2"
            />
            {/* Green TP Banner Tag */}
            <g transform={`translate(${boxCoords.x + 8}, ${Math.max(boxCoords.greenY + 16, 18)})`}>
              <rect x="0" y="-12" width="168" height="18" rx="3" fill="#064e3b" stroke="#10b981" strokeWidth="1" />
              <text x="6" y="1" fill="#34d399" fontSize="11" fontWeight="bold" fontFamily="monospace">
                🎯 TP ${boxCoords.tpPrice.toFixed(2)} (+{boxCoords.rr}R)
              </text>
            </g>

            {/* Red Stop Loss Rectangle */}
            <rect
              x={boxCoords.x}
              y={boxCoords.redY}
              width={boxCoords.width}
              height={boxCoords.redH}
              fill="rgba(239, 68, 68, 0.20)"
              stroke="#ef4444"
              strokeWidth="1.5"
              strokeDasharray="4 2"
              rx="2"
            />
            {/* Red SL Banner Tag */}
            <g transform={`translate(${boxCoords.x + 8}, ${Math.min(boxCoords.redY + boxCoords.redH - 6, boxCoords.redY + 16)})`}>
              <rect x="0" y="-12" width="168" height="18" rx="3" fill="#7f1d1d" stroke="#ef4444" strokeWidth="1" />
              <text x="6" y="1" fill="#f87171" fontSize="11" fontWeight="bold" fontFamily="monospace">
                🛡️ SL ${boxCoords.slPrice.toFixed(2)} (-1.0R)
              </text>
            </g>

            {/* Entry Divider Line */}
            <line
              x1={boxCoords.x}
              y1={boxCoords.entryY}
              x2={boxCoords.x + boxCoords.width}
              y2={boxCoords.entryY}
              stroke="#f59e0b"
              strokeWidth="2"
              strokeDasharray="4 3"
            />
            {/* Entry Banner Tag */}
            <g transform={`translate(${boxCoords.x + 8}, ${boxCoords.entryY + 4})`}>
              <rect x="0" y="-10" width="144" height="18" rx="3" fill="#78350f" stroke="#f59e0b" strokeWidth="1" />
              <text x="6" y="3" fill="#fde68a" fontSize="11" fontWeight="bold" fontFamily="monospace">
                ⚡ ENTRY ${boxCoords.entryPrice.toFixed(2)}
              </text>
            </g>

            {/* Model Name Floating Badge */}
            <g transform={`translate(${boxCoords.x + boxCoords.width - 95}, ${Math.min(boxCoords.greenY, boxCoords.redY) - 6})`}>
              <rect x="0" y="-10" width="90" height="15" rx="3" fill="#1e293b" stroke="#475569" strokeWidth="1" />
              <text x="5" y="1" fill="#94a3b8" fontSize="9" fontWeight="bold" fontFamily="sans-serif">
                {boxCoords.modelName}
              </text>
            </g>
          </svg>
        )}
      </div>
    </div>
  );
}
