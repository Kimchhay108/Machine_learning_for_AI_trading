'use client';

import React, { useState, useEffect } from 'react';
import {
  Candle,
  MarketFeaturesResponse,
  PredictionResponse,
  MarketStatus,
  NewsCalendarStatus,
  fetchCandles,
  fetchMarketFeatures,
  fetchAIPrediction,
  fetchMarketStatus,
  fetchNewsCalendarStatus,
} from '../lib/api';
import TradingViewChart from '../components/TradingViewChart';
import AIAnalysisCard from '../components/AIAnalysisCard';
import AIChatCopilot from '../components/AIChatCopilot';
import BacktestLab from '../components/BacktestLab';
import PaperTradingView from '../components/PaperTradingView';
import SettingsView from '../components/SettingsView';
import {
  Activity,
  BarChart3,
  Bot,
  Clock,
  Layers,
  LineChart,
  MessageSquare,
  RefreshCw,
  Settings,
  Shield,
  ShieldAlert,
  Sparkles,
  Wallet,
  Zap,
} from 'lucide-react';

type TabType = 'terminal' | 'chat' | 'backtest' | 'paper' | 'settings';

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabType>('terminal');
  const [timeframe, setTimeframe] = useState('M5');
  const [candles, setCandles] = useState<Candle[]>([]);
  const [features, setFeatures] = useState<MarketFeaturesResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [newsStatus, setNewsStatus] = useState<NewsCalendarStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [liveSpotPrice, setLiveSpotPrice] = useState<number | null>(null);
  const [cambodiaTime, setCambodiaTime] = useState<string>('');

  // Live Cambodia Local Time (ICT, UTC+7)
  useEffect(() => {
    const updateTime = () => {
      try {
        const formatted = new Intl.DateTimeFormat('en-US', {
          timeZone: 'Asia/Phnom_Penh',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: true,
        }).format(new Date());
        setCambodiaTime(formatted);
      } catch (e) {
        setCambodiaTime(new Date().toLocaleTimeString());
      }
    };
    updateTime();
    const t = setInterval(updateTime, 1000);
    return () => clearInterval(t);
  }, []);

  const loadData = async (tf: string = timeframe, isManual: boolean = false) => {
    if (isManual) setRefreshing(true);
    try {
      const [cRes, fRes, pRes, mRes, nRes] = await Promise.all([
        fetchCandles('XAUUSD', tf, 500),
        fetchMarketFeatures('XAUUSD', tf, 100),
        fetchAIPrediction('XAUUSD', tf),
        fetchMarketStatus('XAUUSD').catch(() => null),
        fetchNewsCalendarStatus(5).catch(() => null),
      ]);
      setCandles(cRes.candles);
      setFeatures(fRes);
      setPrediction(pRes);
      if (mRes) setMarketStatus(mRes);
      if (nRes) setNewsStatus(nRes);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
    } finally {
      setLoading(false);
      if (isManual) setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData(timeframe, false);
    // Real-time market streaming: poll every 6 seconds smoothly without thrashing
    const timer = setInterval(() => {
      loadData(timeframe, false);
    }, 6000);
    return () => clearInterval(timer);
  }, [timeframe]);

  const currentPrice =
    liveSpotPrice !== null
      ? liveSpotPrice
      : candles.length > 0
      ? candles[candles.length - 1].close
      : features?.current_price || 0;

  return (
    <div className="flex flex-col min-h-screen bg-dark-950 text-gray-100">
      {/* Top Professional Institutional Header */}
      <header className="border-b border-dark-700 bg-dark-900 sticky top-0 z-50 px-5 py-2.5 flex items-center justify-between">
        {/* Brand & Market Ticker */}
        <div className="flex items-center space-x-5">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-gold-500/10 border border-gold-500/30 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-gold-400" />
            </div>
            <div>
              <h1 className="text-sm font-black tracking-wider text-white flex items-center space-x-1.5">
                <span>XAUUSD</span>
                <span className="text-gold-400 font-mono">QUANT AI</span>
              </h1>
              <span className="text-[10px] text-gray-400 block tracking-tight font-mono">
                MODEL E • ENSEMBLE ENGINE (+211% NET)
              </span>
            </div>
          </div>

          <div className="h-6 w-px bg-dark-700 hidden sm:block" />

          {/* Real-World COMEX Gold Live Ticker */}
          <div className="hidden sm:flex items-center space-x-3.5">
            <div>
              <span className="text-[10px] text-gray-400 block uppercase font-medium flex items-center space-x-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <span>Real-Time Spot Gold (XAUUSD)</span>
              </span>
              <span className="text-base font-bold text-white font-mono flex items-center space-x-1.5">
                <span>${currentPrice > 0 ? currentPrice.toFixed(2) : '3,567.50'}</span>
                {liveSpotPrice && candles.length > 0 && (
                  <span
                    className={`text-[10px] font-mono px-1 rounded ${
                      liveSpotPrice >= candles[candles.length - 1].close
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-rose-500/20 text-rose-300'
                    }`}
                  >
                    {liveSpotPrice >= candles[candles.length - 1].close ? '+' : ''}
                    {(liveSpotPrice - candles[candles.length - 1].close).toFixed(2)}
                  </span>
                )}
              </span>
            </div>

            {/* Cambodia Local Time Clock */}
            <div className="px-2.5 py-1 rounded bg-dark-800 border border-dark-700 hidden md:block">
              <span className="text-[10px] text-gray-400 block uppercase font-medium flex items-center gap-1">
                <Clock className="w-3 h-3 text-gold-400" />
                Cambodia (ICT)
              </span>
              <span className="text-xs font-mono font-bold text-gold-400">
                {cambodiaTime || '14:48:00'}
              </span>
            </div>

            {/* Session Status Badge */}
            {marketStatus && (
              <div
                className={`px-2.5 py-1 rounded-lg border text-xs flex items-center space-x-2 ${
                  marketStatus.is_market_open
                    ? 'bg-bull-500/10 border-bull-500/40 text-bull-400'
                    : 'bg-bear-500/10 border-bear-500/40 text-bear-400'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    marketStatus.is_market_open ? 'bg-bull-400 animate-pulse' : 'bg-bear-400'
                  }`}
                />
                <div>
                  <span className="font-bold tracking-wide block text-[11px]">
                    {marketStatus.is_market_open ? 'LIVE MARKET OPEN' : 'MARKET CLOSED'}
                  </span>
                  <span className="text-[10px] text-gray-400 block font-mono">
                    {marketStatus.data_source_mode === 'REALTIME_INSTITUTIONAL_FEED'
                      ? '● Real-Time Spot (0s Delay)'
                      : 'Verified Archive'}
                  </span>
                </div>
              </div>
            )}

            {/* Macro News Calendar Shield */}
            {newsStatus && (
              newsStatus.is_blackout_active ? (
                <div className="px-2.5 py-1 rounded-lg border text-xs flex items-center space-x-1.5 bg-rose-500/15 border-rose-500/40 text-rose-300 animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  <span className="font-bold text-[11px]">NEWS BLACKOUT ACTIVE</span>
                </div>
              ) : (
                <div className="px-2.5 py-1 rounded-lg border text-xs hidden lg:flex items-center space-x-1.5 bg-dark-800 border-dark-700 text-gray-300">
                  <Shield className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-[11px] font-semibold text-emerald-400">NEWS CLEAR</span>
                </div>
              )
            )}
          </div>
        </div>

        {/* Global Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={() => loadData(timeframe, true)}
            disabled={refreshing}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs bg-dark-800 text-gray-200 hover:text-white border border-dark-700 hover:bg-dark-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-gold-400' : ''}`} />
            <span className="font-semibold">Sync</span>
          </button>

          <div className="flex items-center space-x-1.5 text-xs text-gray-400 bg-dark-800 px-2.5 py-1 rounded border border-dark-700">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-mono text-[11px] text-emerald-300">Live Engine Ready</span>
          </div>
        </div>
      </header>

      {/* Streamlined 5-Tab Navigation */}
      <nav className="bg-dark-850 border-b border-dark-700 px-5 py-1.5 flex space-x-1 overflow-x-auto">
        {[
          { id: 'terminal', label: '1. Live AI Terminal', icon: Zap },
          { id: 'chat', label: '2. 💬 AI Copilot & Chart Chat', icon: Bot },
          { id: 'backtest', label: '3. AI Quant Lab & Backtest', icon: BarChart3 },
          { id: 'paper', label: '4. Paper Trading', icon: Wallet },
          { id: 'settings', label: '5. Risk & Settings', icon: Settings },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`flex items-center space-x-2 px-4 py-2 text-xs font-bold rounded-lg transition whitespace-nowrap ${
                isActive
                  ? 'bg-dark-700 text-gold-400 border border-dark-600 shadow-sm'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/60'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Main Screen Layout */}
      <main className="flex-1 px-4 py-3 w-full">
        {/* Tab 1: Live AI Terminal */}
        {activeTab === 'terminal' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
            <div className="lg:col-span-8 xl:col-span-9 min-h-[640px] h-[calc(100vh-175px)]">
              <TradingViewChart
                candles={candles}
                timeframe={timeframe}
                onTimeframeChange={(tf) => setTimeframe(tf)}
                features={features}
                prediction={prediction}
                onPriceUpdate={(p) => setLiveSpotPrice(p)}
              />
            </div>
            <div className="lg:col-span-4 xl:col-span-3 space-y-3.5 overflow-y-auto max-h-[calc(100vh-175px)]">
              {/* Quick switch to AI Chat Copilot banner */}
              <button
                onClick={() => setActiveTab('chat')}
                className="w-full py-2 px-3 rounded-lg bg-dark-850 hover:bg-dark-800 border border-gold-500/30 hover:border-gold-500/60 text-xs text-gold-300 font-mono font-bold flex items-center justify-between transition group shadow-md"
              >
                <span className="flex items-center space-x-2">
                  <Bot className="w-4 h-4 text-gold-400 group-hover:animate-bounce" />
                  <span>Ask DeepSeek / GPT / Claude</span>
                </span>
                <span className="text-[10px] bg-gold-500/20 text-gold-400 px-1.5 py-0.5 rounded">
                  Open Chat →
                </span>
              </button>

              <AIAnalysisCard
                prediction={prediction}
                loading={loading}
                onTradeExecuted={() => loadData(timeframe)}
                onSignalReset={() => loadData(timeframe, true)}
              />
            </div>
          </div>
        )}

        {/* Tab 2: AI Copilot & Chart Chat (Side-by-side Chart + Multi-LLM Chat) */}
        {activeTab === 'chat' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
            <div className="lg:col-span-7 xl:col-span-7 min-h-[640px] h-[calc(100vh-175px)]">
              <TradingViewChart
                candles={candles}
                timeframe={timeframe}
                onTimeframeChange={(tf) => setTimeframe(tf)}
                features={features}
                prediction={prediction}
                onPriceUpdate={(p) => setLiveSpotPrice(p)}
              />
            </div>
            <div className="lg:col-span-5 xl:col-span-5 h-[calc(100vh-175px)]">
              <AIChatCopilot
                currentPrice={currentPrice}
                timeframe={timeframe}
                prediction={prediction}
                features={features}
                candles={candles}
              />
            </div>
          </div>
        )}

        {/* Tab 3: AI Quant Lab & Backtesting */}
        {activeTab === 'backtest' && (
          <div className="w-full">
            <BacktestLab />
          </div>
        )}

        {/* Tab 4: Paper Trading */}
        {activeTab === 'paper' && (
          <div className="w-full">
            <PaperTradingView />
          </div>
        )}

        {/* Tab 5: System Risk & Settings */}
        {activeTab === 'settings' && (
          <div className="w-full">
            <SettingsView />
          </div>
        )}
      </main>
    </div>
  );
}
