'use client';

import React, { useState, useEffect } from 'react';
import { evaluatePredictions } from '../lib/api';
import { History, TrendingUp, TrendingDown, Target, RefreshCw } from 'lucide-react';

export default function PredictionHistoryView() {
  const [horizon, setHorizon] = useState(3); // 3 candles = 15m on M5
  const [sampleLimit, setSampleLimit] = useState(100);
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const runEvaluation = () => {
    setLoading(true);
    evaluatePredictions(horizon, sampleLimit)
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed evaluating predictions:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    runEvaluation();
  }, [horizon, sampleLimit]);

  return (
    <div className="space-y-6">
      {/* Control Bar */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <History className="w-5 h-5 text-gold-500" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
              Forward Outcome Verification & Excursion Audit
            </h2>
          </div>
          <button
            onClick={runEvaluation}
            disabled={loading}
            className="text-xs bg-dark-800 text-gray-300 hover:text-white px-3 py-1.5 rounded border border-dark-700 flex items-center space-x-1"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Re-evaluate</span>
          </button>
        </div>

        <div className="flex items-center space-x-6 text-xs text-gray-400">
          <div className="flex items-center space-x-2">
            <span>Evaluation Forward Horizon:</span>
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              className="bg-dark-800 border border-dark-700 text-white rounded px-2.5 py-1"
            >
              <option value={3}>15 Minutes (3 bars)</option>
              <option value={6}>30 Minutes (6 bars)</option>
              <option value={12}>60 Minutes (12 bars)</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <span>Sampled Predictions:</span>
            <select
              value={sampleLimit}
              onChange={(e) => setSampleLimit(Number(e.target.value))}
              className="bg-dark-800 border border-dark-700 text-white rounded px-2.5 py-1"
            >
              <option value={50}>Recent 50 Candles</option>
              <option value={100}>Recent 100 Candles</option>
              <option value={200}>Recent 200 Candles</option>
            </select>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      {data && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Directional Accuracy</span>
            <span className="text-2xl font-bold text-bull-400">
              {(data.directional_accuracy * 100).toFixed(1)}%
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Across {data.actionable_predictions} actionable signals
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Profit Factor</span>
            <span className="text-2xl font-bold text-gold-400">
              {data.profit_factor.toFixed(2)}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Gross Favorable / Gross Adverse
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Mean Favorable Excursion (MFE)</span>
            <span className="text-2xl font-bold text-bull-400">
              +{data.mean_favorable_excursion_pips.toFixed(1)} pips
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Avg peak profit potential
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Mean Adverse Excursion (MAE)</span>
            <span className="text-2xl font-bold text-bear-400">
              -{data.mean_adverse_excursion_pips.toFixed(1)} pips
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Avg drawdown exposure
            </span>
          </div>
        </div>
      )}

      {/* Signal Distribution Breakdown */}
      {data && (
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-3">
            Signal Selectivity Breakdown ({data.total_predictions} Total Predictions Tested)
          </h3>

          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-dark-850 rounded border border-dark-700">
              <span className="text-xs text-gray-400 block mb-1">Actionable Predictions (BUY / SELL)</span>
              <span className="text-xl font-bold text-white">{data.actionable_predictions}</span>
              <span className="text-xs text-bull-400 block mt-1">
                {((data.actionable_predictions / data.total_predictions) * 100).toFixed(1)}% Active Rate
              </span>
            </div>

            <div className="p-4 bg-dark-850 rounded border border-dark-700">
              <span className="text-xs text-gray-400 block mb-1">Selective Abstentions (WAIT)</span>
              <span className="text-xl font-bold text-white">{data.abstentions}</span>
              <span className="text-xs text-gray-400 block mt-1">
                {((data.abstentions / data.total_predictions) * 100).toFixed(1)}% Noise Rejection Rate
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
