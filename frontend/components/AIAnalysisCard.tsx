'use client';

import React from 'react';
import { PredictionResponse } from '../lib/api';
import {
  TrendingUp,
  TrendingDown,
  PauseCircle,
  Activity,
  Layers,
  ShieldAlert,
  Target,
} from 'lucide-react';

interface AIAnalysisCardProps {
  prediction: PredictionResponse | null;
  loading?: boolean;
}

export default function AIAnalysisCard({ prediction, loading }: AIAnalysisCardProps) {
  if (loading || !prediction) {
    return (
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5 flex flex-col items-center justify-center min-h-[300px]">
        <Activity className="w-8 h-8 text-gold-500 animate-spin mb-3" />
        <p className="text-gray-400 text-sm">Evaluating market structure and ensemble models...</p>
      </div>
    );
  }

  const {
    direction,
    bullish_probability,
    bearish_probability,
    neutral_probability,
    confidence,
    market_regime,
    entry_reference_price,
    stop_loss_reference,
    take_profit_reference,
    risk_reward_ratio,
    contributing_features,
    model_version,
  } = prediction;

  const isBuy = direction === 'BUY';
  const isSell = direction === 'SELL';
  const isWait = direction === 'WAIT';

  const badgeBg = isBuy
    ? 'bg-bull-600/20 text-bull-400 border-bull-500/40 glow-bull'
    : isSell
    ? 'bg-bear-600/20 text-bear-400 border-bear-500/40 glow-bear'
    : 'bg-gray-800/60 text-gray-400 border-gray-600';

  const formatDecision = (val: number | undefined) => {
    if (val === 2) return <span className="text-bull-400 font-semibold">BULL</span>;
    if (val === 0) return <span className="text-bear-400 font-semibold">BEAR</span>;
    return <span className="text-gray-400 font-semibold">NEUT</span>;
  };

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-lg p-5 space-y-5">
      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-dark-700 pb-3">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-gold-500" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
            AI Ensemble Signal
          </h2>
        </div>
        <span className="text-xs bg-dark-800 text-gray-400 px-2 py-0.5 rounded border border-dark-700 font-mono">
          {model_version}
        </span>
      </div>

      {/* Main Signal Display */}
      <div className={`p-4 rounded-xl border flex items-center justify-between ${badgeBg}`}>
        <div className="flex items-center space-x-3">
          {isBuy && <TrendingUp className="w-9 h-9 text-bull-400" />}
          {isSell && <TrendingDown className="w-9 h-9 text-bear-400" />}
          {isWait && <PauseCircle className="w-9 h-9 text-gray-400" />}
          <div>
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-widest block">
              Recommendation
            </span>
            <span className="text-2xl font-black tracking-wider">
              {direction}
            </span>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-semibold text-gray-400 block uppercase">Confidence</span>
          <span className="text-xl font-bold">{(confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Calibrated Probabilities Breakdown */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-400 font-medium">
          <span>Calibrated Distribution</span>
          <span>Selective Threshold: 50%</span>
        </div>

        {/* Bullish Bar */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-bull-400 font-semibold">Bullish</span>
            <span className="text-gray-300">{(bullish_probability * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-2 overflow-hidden border border-dark-700">
            <div
              className="bg-bull-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${bullish_probability * 100}%` }}
            />
          </div>
        </div>

        {/* Bearish Bar */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-bear-400 font-semibold">Bearish</span>
            <span className="text-gray-300">{(bearish_probability * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-2 overflow-hidden border border-dark-700">
            <div
              className="bg-bear-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${bearish_probability * 100}%` }}
            />
          </div>
        </div>

        {/* Neutral Bar */}
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-gray-400 font-semibold">Neutral / Noise</span>
            <span className="text-gray-300">{(neutral_probability * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-dark-800 rounded-full h-2 overflow-hidden border border-dark-700">
            <div
              className="bg-gray-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${neutral_probability * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Market Regime & Submodel Agreement */}
      <div className="grid grid-cols-2 gap-3 pt-2 border-t border-dark-700 text-xs">
        <div className="bg-dark-850 p-2.5 rounded border border-dark-700">
          <span className="text-gray-400 block mb-1">Market Regime</span>
          <span className="font-semibold text-gold-400">{market_regime}</span>
        </div>

        <div className="bg-dark-850 p-2.5 rounded border border-dark-700">
          <span className="text-gray-400 block mb-1">Risk / Reward</span>
          <span className="font-semibold text-white">1 : {risk_reward_ratio}</span>
        </div>
      </div>

      {/* Submodel Agreement Grid */}
      {contributing_features?.submodel_agreement && (
        <div className="bg-dark-850 p-3 rounded border border-dark-700 space-y-2">
          <span className="text-xs font-semibold text-gray-300 flex items-center space-x-1">
            <Layers className="w-3.5 h-3.5 text-gold-500" />
            <span>Sub-Model Agreement</span>
          </span>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model A (Tech)</span>
              {formatDecision(contributing_features.submodel_agreement.tech_prediction)}
            </div>
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model B (SMC)</span>
              {formatDecision(contributing_features.submodel_agreement.smc_prediction)}
            </div>
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model D (PA)</span>
              {formatDecision(contributing_features.submodel_agreement.price_action_prediction)}
            </div>
          </div>
        </div>
      )}

      {/* Trade Execution References */}
      <div className="p-3 bg-dark-850 rounded border border-dark-700 space-y-1.5 text-xs">
        <span className="text-gray-300 font-semibold block mb-1 flex items-center space-x-1">
          <Target className="w-3.5 h-3.5 text-gold-500" />
          <span>Execution Reference Levels</span>
        </span>
        <div className="flex justify-between text-gray-400">
          <span>Reference Price:</span>
          <span className="text-white font-mono font-medium">${entry_reference_price.toFixed(2)}</span>
        </div>
        <div className="flex justify-between text-gray-400">
          <span>Stop Loss (1.5x ATR):</span>
          <span className="text-bear-400 font-mono font-medium">${stop_loss_reference.toFixed(2)}</span>
        </div>
        <div className="flex justify-between text-gray-400">
          <span>Take Profit (3.0x ATR):</span>
          <span className="text-bull-400 font-mono font-medium">${take_profit_reference.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
