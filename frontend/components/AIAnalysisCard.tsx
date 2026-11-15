'use client';

import React, { useState } from 'react';
import { PredictionResponse, ModelSetupItem } from '../lib/api';
import {
  TrendingUp,
  TrendingDown,
  PauseCircle,
  Activity,
  Layers,
  ShieldAlert,
  Target,
  Compass,
  Zap,
  Info,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface AIAnalysisCardProps {
  prediction: PredictionResponse | null;
  loading?: boolean;
}

export default function AIAnalysisCard({ prediction, loading }: AIAnalysisCardProps) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

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

  // Extract all model setups or synthesize default values
  const allSetups: ModelSetupItem[] = contributing_features?.all_model_setups || [
    {
      model_key: 'ensemble',
      model_name: 'Model E: Calibrated Multi-Model Ensemble',
      direction: direction as 'BUY' | 'SELL' | 'WAIT',
      entry_price: entry_reference_price,
      stop_loss: stop_loss_reference,
      take_profit: take_profit_reference,
      risk_reward_ratio: risk_reward_ratio,
      risk_distance_usd: Math.abs(entry_reference_price - stop_loss_reference),
      confidence_pct: Math.round(confidence * 100),
      why_explanation:
        direction === 'WAIT'
          ? `Ensemble recommends WAIT: confidence (${(confidence * 100).toFixed(1)}%) is below selective threshold. Preserving capital until sub-models reach confluence.`
          : `Multi-model stacked consensus triggers ${direction} with ${(confidence * 100).toFixed(1)}% confidence. Dynamic ATR Stop Loss at $${(stop_loss_reference || 0).toFixed(2)} with 2.0 R:R Target at $${(take_profit_reference || 0).toFixed(2)}.`,
      key_drivers: ['Meta-Model Stacking', `Confidence: ${(confidence * 100).toFixed(1)}%`, `Regime: ${market_regime}`],
    },
    {
      model_key: 'smc_ict',
      model_name: 'Model B: SMC & ICT 2022 Mentorship',
      direction: direction === 'BUY' ? 'BUY' : 'WAIT',
      entry_price: entry_reference_price,
      stop_loss: entry_reference_price - 5.0,
      take_profit: entry_reference_price + 12.5,
      risk_reward_ratio: 2.5,
      risk_distance_usd: 5.0,
      confidence_pct: 65,
      why_explanation:
        'ICT 2022 Mentorship model evaluates Liquidity Sweeps, Market Structure Shifts (MSS), and FVG discount/premium retests with structural swing invalidation stop.',
      key_drivers: ['ICT 2022 Mentorship', 'Structural Swing SL', '2.5 R:R Target'],
    },
    {
      model_key: 'technical',
      model_name: 'Model A: Technical Momentum (Indicators)',
      direction: 'WAIT',
      entry_price: entry_reference_price,
      stop_loss: entry_reference_price - 3.5,
      take_profit: entry_reference_price + 7.0,
      risk_reward_ratio: 2.0,
      risk_distance_usd: 3.5,
      confidence_pct: 50,
      why_explanation:
        'Evaluates EMA 9/20/50/200 trend alignment, RSI pullbacks (42-58), and MACD histogram momentum with a 1.5x ATR volatility envelope.',
      key_drivers: ['EMA Alignment', 'RSI Pullback', 'MACD Expansion'],
    },
    {
      model_key: 'price_action',
      model_name: 'Model D: Price Action & Rejections',
      direction: 'WAIT',
      entry_price: entry_reference_price,
      stop_loss: entry_reference_price - 2.5,
      take_profit: entry_reference_price + 5.0,
      risk_reward_ratio: 2.0,
      risk_distance_usd: 2.5,
      confidence_pct: 50,
      why_explanation:
        'Identifies pin bar rejection wicks and engulfing candles, setting tight stop loss at the candle extreme with 2.0 R:R target.',
      key_drivers: ['Wick Rejection', 'Candle Morphology', 'Wick Extreme SL'],
    },
  ];

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
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-center text-xs">
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model A (Tech)</span>
              {formatDecision(contributing_features.submodel_agreement.tech_prediction)}
            </div>
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model B (SMC / ICT)</span>
              {formatDecision(contributing_features.submodel_agreement.smc_prediction)}
            </div>
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model D (PA)</span>
              {formatDecision(contributing_features.submodel_agreement.price_action_prediction)}
            </div>
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model F (Vol Profile)</span>
              {formatDecision(contributing_features.submodel_agreement.volume_profile_prediction ?? 1)}
            </div>
            <div className="p-1.5 rounded bg-dark-800 border border-dark-700">
              <span className="text-gray-400 block text-[10px]">Model G (Footprint)</span>
              {formatDecision(contributing_features.submodel_agreement.footprint_prediction ?? 1)}
            </div>
          </div>
        </div>
      )}

      {/* Institutional ICT Models Status */}
      <div className="bg-dark-850 p-3 rounded border border-dark-700 space-y-2">
        <span className="text-xs font-semibold text-gray-300 flex items-center justify-between">
          <span className="flex items-center space-x-1">
            <ShieldAlert className="w-3.5 h-3.5 text-gold-500" />
            <span>ICT 2022 & Institutional Setups</span>
          </span>
          <span className="text-[10px] bg-gold-500/10 text-gold-400 px-1.5 py-0.5 rounded border border-gold-500/20 font-mono">
            Active
          </span>
        </span>
        {contributing_features?.active_ict_setups && contributing_features.active_ict_setups.length > 0 ? (
          <div className="space-y-1">
            {contributing_features.active_ict_setups.map((setup, idx) => (
              <div
                key={idx}
                className="text-xs px-2 py-1.5 rounded bg-gold-500/10 border border-gold-500/30 text-gold-300 font-medium flex items-center space-x-1.5"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-gold-400 animate-pulse" />
                <span>{setup}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[11px] text-gray-400 leading-relaxed">
            Monitoring for ICT 2022 (Liquidity Sweep $\to$ MSS $\to$ FVG) and Silver Bullet macro windows.
          </p>
        )}
      </div>

      {/* ========================================================================= */}
      {/* NEXT ENTRY SIGNALS FOR ALL MODELS + "EXPLAIN WHY" RATIONALE               */}
      {/* ========================================================================= */}
      <div className="space-y-3 pt-3 border-t border-dark-700">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-gray-200 flex items-center space-x-1.5">
            <Compass className="w-4 h-4 text-gold-400" />
            <span>Next Entry Across All Models</span>
          </span>
          <span className="text-[10px] text-gray-400">Model-Specific SL/TP & Why</span>
        </div>

        <div className="space-y-2.5">
          {allSetups.map((setup) => {
            const isSetupBuy = setup.direction === 'BUY';
            const isSetupSell = setup.direction === 'SELL';
            const isSetupWait = setup.direction === 'WAIT';
            const isExpanded = expandedModel === setup.model_key || (!expandedModel && setup.model_key === 'ensemble');

            const dirBadge = isSetupBuy
              ? 'bg-bull-500/20 text-bull-400 border-bull-500/40'
              : isSetupSell
              ? 'bg-bear-500/20 text-bear-400 border-bear-500/40'
              : 'bg-dark-750 text-gray-400 border-dark-600';

            return (
              <div
                key={setup.model_key}
                className="bg-dark-850 rounded-lg border border-dark-700 overflow-hidden transition"
              >
                {/* Header Row */}
                <div
                  onClick={() => setExpandedModel(isExpanded ? '__none__' : setup.model_key)}
                  className="p-3 cursor-pointer hover:bg-dark-800/60 flex items-center justify-between transition"
                >
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-xs text-white">
                      {setup.model_name.split(':')[0]}
                    </span>
                    <span className="text-[11px] text-gray-400 hidden sm:inline">
                      {setup.model_name.split(':')[1]}
                    </span>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${dirBadge}`}>
                      {setup.direction} ({setup.confidence_pct}%)
                    </span>
                    {isExpanded ? (
                      <ChevronUp className="w-3.5 h-3.5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
                    )}
                  </div>
                </div>

                {/* Details Accordion Content */}
                {isExpanded && (
                  <div className="px-3 pb-3 pt-1 space-y-2.5 border-t border-dark-800 bg-dark-900/40 text-xs">
                    {/* Execution Parameters Grid */}
                    <div className="grid grid-cols-4 gap-2 text-center text-[11px] font-mono">
                      <div className="bg-dark-800 p-1.5 rounded border border-dark-750">
                        <span className="text-[10px] text-gray-400 block font-sans">Entry Price</span>
                        <span className="text-white font-bold">
                          ${typeof setup.entry_price === 'number' ? setup.entry_price.toFixed(2) : setup.entry_price}
                        </span>
                      </div>

                      <div className="bg-dark-800 p-1.5 rounded border border-dark-750">
                        <span className="text-[10px] text-gray-400 block font-sans">Stop Loss</span>
                        <span className="text-bear-400 font-bold">
                          ${typeof setup.stop_loss === 'number' ? setup.stop_loss.toFixed(2) : setup.stop_loss}
                        </span>
                      </div>

                      <div className="bg-dark-800 p-1.5 rounded border border-dark-750">
                        <span className="text-[10px] text-gray-400 block font-sans">Take Profit</span>
                        <span className="text-bull-400 font-bold">
                          ${typeof setup.take_profit === 'number' ? setup.take_profit.toFixed(2) : setup.take_profit}
                        </span>
                      </div>

                      <div className="bg-dark-800 p-1.5 rounded border border-dark-750">
                        <span className="text-[10px] text-gray-400 block font-sans">Risk:Reward</span>
                        <span className="text-gold-400 font-bold">1:{setup.risk_reward_ratio}</span>
                      </div>
                    </div>

                    {/* "Explain Why" Rationale Box */}
                    <div className="bg-dark-800/80 p-2.5 rounded border border-dark-700/80">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-gold-400 block mb-1 flex items-center space-x-1">
                        <Zap className="w-3 h-3 text-gold-400" />
                        <span>Why This Trade Setup:</span>
                      </span>
                      <p className="text-gray-300 text-xs leading-relaxed font-sans">
                        {setup.why_explanation}
                      </p>

                      {/* Key Setup Drivers */}
                      {setup.key_drivers && setup.key_drivers.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-dark-700">
                          {setup.key_drivers.map((driver, didx) => (
                            <span
                              key={didx}
                              className="text-[10px] bg-dark-900 border border-dark-700 text-gray-300 px-1.5 py-0.5 rounded font-mono"
                            >
                              {driver}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
