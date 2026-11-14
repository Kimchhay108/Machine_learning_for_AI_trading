'use client';

import React, { useEffect, useState } from 'react';
import { PerformanceResponse, fetchModelPerformance } from '../lib/api';
import { Award, CheckCircle2, Crosshair, HelpCircle, ShieldCheck } from 'lucide-react';

export default function ModelPerformanceView() {
  const [data, setData] = useState<PerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelPerformance('XAUUSD', 'v1.0.0')
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed fetching performance:', err);
        setLoading(false);
      });
  }, []);

  if (loading || !data) {
    return (
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-8 text-center text-gray-400">
        Computing calibration metrics and out-of-sample confusion matrix...
      </div>
    );
  }

  const {
    model_version,
    directional_accuracy,
    selective_accuracy,
    coverage_pct,
    f1_score,
    expected_calibration_error,
    brier_score,
    confusion_matrix,
    regime_accuracies,
  } = data;

  return (
    <div className="space-y-6">
      {/* Top Banner: Statistical Validity Notice */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center space-x-3 mb-2">
          <ShieldCheck className="w-5 h-5 text-gold-500" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
            Out-of-Sample Statistical Validation ({model_version})
          </h2>
        </div>
        <p className="text-xs text-gray-400 leading-relaxed">
          Evaluated strictly on chronological unseen test data separated by a 12-bar purge window.
          Zero lookahead leakage. High-confidence predictions apply selective abstention thresholding.
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Selective Accuracy</span>
          <span className="text-xl font-bold text-bull-400">
            {(selective_accuracy * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">
            Coverage: {coverage_pct}% of bars
          </span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">All Accuracy</span>
          <span className="text-xl font-bold text-white">
            {(directional_accuracy * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">F1 Score: {f1_score.toFixed(3)}</span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Calibration Error (ECE)</span>
          <span className="text-xl font-bold text-gold-400">
            {(expected_calibration_error * 100).toFixed(1)}%
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">
            Expected Calibration Error
          </span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Brier Score</span>
          <span className="text-xl font-bold text-white">
            {brier_score.toFixed(4)}
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">Mean squared prob error</span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Selective Coverage</span>
          <span className="text-xl font-bold text-white">
            {coverage_pct}%
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">High-confidence triggers</span>
        </div>
      </div>

      {/* Confusion Matrix and Regime Performance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-4 flex items-center space-x-2">
            <Crosshair className="w-4 h-4 text-gold-500" />
            <span>3-Class Confusion Matrix (Actual vs Predicted)</span>
          </h3>

          <div className="overflow-x-auto">
            <table className="w-full text-center text-xs">
              <thead>
                <tr className="text-gray-400 border-b border-dark-700">
                  <th className="py-2 text-left">Actual \ Pred</th>
                  {confusion_matrix.labels.map((lbl) => (
                    <th key={lbl} className="py-2">{lbl}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800 font-mono">
                {confusion_matrix.matrix.map((row, rIdx) => (
                  <tr key={rIdx}>
                    <td className="py-3 font-semibold text-left text-gray-300 font-sans">
                      {confusion_matrix.labels[rIdx]}
                    </td>
                    {row.map((val, cIdx) => (
                      <td
                        key={cIdx}
                        className={`py-3 font-bold ${
                          rIdx === cIdx ? 'bg-gold-500/10 text-gold-400' : 'text-gray-400'
                        }`}
                      >
                        {val}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Regime-Specific Performance */}
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-4 flex items-center space-x-2">
            <Award className="w-4 h-4 text-gold-500" />
            <span>Performance Breakdown by Market Regime</span>
          </h3>

          <div className="space-y-3">
            {Object.entries(regime_accuracies).map(([regimeName, acc]) => (
              <div key={regimeName} className="p-3 bg-dark-850 rounded border border-dark-700">
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-semibold text-gray-200">{regimeName}</span>
                  <span className="font-bold text-bull-400 font-mono">{(acc * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-dark-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className="bg-gold-500 h-1.5 rounded-full"
                    style={{ width: `${acc * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
