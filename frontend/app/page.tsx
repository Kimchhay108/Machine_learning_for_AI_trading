'use client';

import React, { useState, useEffect } from 'react';
import {
  Candle,
  MarketFeaturesResponse,
  PredictionResponse,
  fetchCandles,
  fetchMarketFeatures,
  fetchAIPrediction,
} from '../lib/api';
import TradingViewChart from '../components/TradingViewChart';
import AIAnalysisCard from '../components/AIAnalysisCard';
import BacktestLab from '../components/BacktestLab';
import ModelPerformanceView from '../components/ModelPerformanceView';
import TrainingCenterView from '../components/TrainingCenterView';
import PredictionHistoryView from '../components/PredictionHistoryView';
import SettingsView from '../components/SettingsView';
import PaperTradingView from '../components/PaperTradingView';
import {
  Activity,
  BarChart3,
  Cpu,
  History,
  Layers,
  LineChart,
  RefreshCw,
  Settings,
  ShieldAlert,
  Sparkles,
  Wallet,
} from 'lucide-react';

type TabType =
  | 'overview'
  | 'chart'
  | 'analysis'
  | 'history'
  | 'performance'
  | 'training'
  | 'backtest'
  | 'paper'
  | 'settings';

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [timeframe, setTimeframe] = useState('M5');
  const [candles, setCandles] = useState<Candle[]>([]);
  const [features, setFeatures] = useState<MarketFeaturesResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = async (tf: string = timeframe) => {
    setRefreshing(true);
    try {
      const [cRes, fRes, pRes] = await Promise.all([
        fetchCandles('XAUUSD', tf, 500),
        fetchMarketFeatures('XAUUSD', 100),
        fetchAIPrediction('XAUUSD', tf),
      ]);
      setCandles(cRes.candles);
      setFeatures(fRes);
      setPrediction(pRes);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData(timeframe);
  }, [timeframe]);

  const currentPrice =
    candles.length > 0
      ? candles[candles.length - 1].close
      : features?.current_price || 0;

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top Navbar Header */}
      <header className="border-b border-dark-700 bg-dark-900 sticky top-0 z-50 px-6 py-3 flex items-center justify-between">
        {/* Brand & Market Ticker */}
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gold-500/10 border border-gold-500/30 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-gold-500" />
            </div>
            <div>
              <h1 className="text-sm font-black tracking-wider text-white">
                XAUUSD <span className="text-gold-400">AI QUANT</span>
              </h1>
              <span className="text-[10px] text-gray-400 block tracking-tight font-mono">
                HYBRID SMC & MACHINE LEARNING
              </span>
            </div>
          </div>

          <div className="h-6 w-px bg-dark-700 hidden sm:block" />

          {/* Current Live Ticker */}
          <div className="hidden sm:flex items-center space-x-4">
            <div>
              <span className="text-[10px] text-gray-400 block uppercase font-medium">Spot Gold</span>
              <span className="text-base font-bold text-white font-mono">
                ${currentPrice.toFixed(2)}
              </span>
            </div>

            {features?.regime && (
              <div className="px-2.5 py-1 rounded bg-dark-800 border border-dark-700 text-xs">
                <span className="text-[10px] text-gray-400 block uppercase">Regime</span>
                <span className="font-semibold text-gold-400">
                  {features.regime.regime_label} ({features.regime.session})
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Global Controls & Status */}
        <div className="flex items-center space-x-4">
          <button
            onClick={() => loadData(timeframe)}
            disabled={refreshing}
            className="flex items-center space-x-1 px-3 py-1.5 rounded text-xs bg-dark-800 text-gray-300 hover:text-white border border-dark-700 hover:bg-dark-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            <span>Sync</span>
          </button>

          <div className="flex items-center space-x-1.5 text-xs text-gray-400">
            <span className="w-2 h-2 rounded-full bg-bull-400 animate-pulse" />
            <span className="hidden md:inline font-mono">DuckDB / Engine Online</span>
          </div>
        </div>
      </header>

      {/* Navigation Sub-Tabs */}
      <nav className="bg-dark-850 border-b border-dark-700 px-6 py-1 flex space-x-1 overflow-x-auto">
        {[
          { id: 'overview', label: '1. Overview', icon: LineChart },
          { id: 'chart', label: '2. Live Chart', icon: BarChart3 },
          { id: 'analysis', label: '3. AI Analysis', icon: Activity },
          { id: 'history', label: '4. Prediction History', icon: History },
          { id: 'performance', label: '5. Model Performance', icon: Layers },
          { id: 'training', label: '6. Training Center', icon: Cpu },
          { id: 'backtest', label: '7. Backtesting Lab', icon: Sparkles },
          { id: 'paper', label: '8. Paper Trading', icon: Wallet },
          { id: 'settings', label: '9. Settings', icon: Settings },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`flex items-center space-x-1.5 px-3 py-2 text-xs font-semibold rounded-md transition whitespace-nowrap ${
                isActive
                  ? 'bg-dark-700 text-gold-400 border border-dark-600 shadow-sm'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-dark-800/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 p-6 max-w-7xl w-full mx-auto">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <TradingViewChart
                candles={candles}
                timeframe={timeframe}
                onTimeframeChange={(tf) => setTimeframe(tf)}
                features={features}
                prediction={prediction}
              />
            </div>
            <div className="space-y-4">
              <AIAnalysisCard prediction={prediction} loading={loading} />
            </div>
          </div>
        )}

        {activeTab === 'chart' && (
          <div className="h-[750px]">
            <TradingViewChart
              candles={candles}
              timeframe={timeframe}
              onTimeframeChange={(tf) => setTimeframe(tf)}
              features={features}
              prediction={prediction}
            />
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="max-w-2xl mx-auto">
            <AIAnalysisCard prediction={prediction} loading={loading} />
          </div>
        )}

        {activeTab === 'history' && <PredictionHistoryView />}

        {activeTab === 'performance' && <ModelPerformanceView />}

        {activeTab === 'training' && <TrainingCenterView />}

        {activeTab === 'backtest' && <BacktestLab />}

        {activeTab === 'paper' && <PaperTradingView />}

        {activeTab === 'settings' && <SettingsView />}
      </main>
    </div>
  );
}
