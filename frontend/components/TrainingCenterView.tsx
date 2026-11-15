'use client';

import React, { useState, useEffect } from 'react';
import { ModelInfo, fetchModels, triggerRetrain, activateModel } from '../lib/api';
import { Cpu, RefreshCw, CheckCircle, Clock, ShieldAlert, Zap } from 'lucide-react';

export default function TrainingCenterView() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeVersion, setActiveVersion] = useState<string>('v1.0.0');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [activatingVersion, setActivatingVersion] = useState<string | null>(null);

  // Form states
  const [newVersion, setNewVersion] = useState('v1.0.4');
  const [threshold, setThreshold] = useState(0.50);
  const [testRatio, setTestRatio] = useState(0.30);
  const [adminKey, setAdminKey] = useState('xauusd-admin-secret-key');
  const [learnFromMistakes, setLearnFromMistakes] = useState(true);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const loadModels = () => {
    setRefreshing(true);
    fetchModels()
      .then((res) => {
        setModels(res.models);
        setActiveVersion(res.active_version);
        setRefreshing(false);
      })
      .catch((err) => {
        console.error('Error fetching models:', err);
        setRefreshing(false);
      });
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleRetrain = async () => {
    setLoading(true);
    setStatusMsg(null);
    try {
      const res = await triggerRetrain(
        newVersion,
        Number(threshold),
        Number(testRatio),
        adminKey,
        learnFromMistakes
      );
      setStatusMsg(`Success! ${res.message}`);
      loadModels();
    } catch (err: any) {
      setStatusMsg(`Retraining failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleActivate = async (version: string) => {
    setActivatingVersion(version);
    setStatusMsg(null);
    try {
      const res = await activateModel(version, adminKey);
      setStatusMsg(`Success! ${res.message}`);
      loadModels();
    } catch (err: any) {
      setStatusMsg(`Activation failed: ${err.message}`);
    } finally {
      setActivatingVersion(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Retrain Trigger Form */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200 flex items-center space-x-2">
            <Cpu className="w-4 h-4 text-gold-500" />
            <span>Controlled Continuous Retraining Pipeline</span>
          </h2>
          <span className="text-xs bg-dark-800 text-gold-400 px-2 py-1 rounded border border-dark-700 font-mono">
            Active: {activeVersion}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs mb-4">
          <div>
            <label className="text-gray-400 block mb-1">New Version Identifier</label>
            <input
              type="text"
              value={newVersion}
              onChange={(e) => setNewVersion(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-3 py-2 text-white font-mono"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Confidence Threshold</label>
            <input
              type="number"
              step="0.02"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-3 py-2 text-white font-mono"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Test Set Ratio (OOS)</label>
            <select
              value={testRatio}
              onChange={(e) => setTestRatio(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-3 py-2 text-white"
            >
              <option value={0.20}>20% Out-of-sample</option>
              <option value={0.30}>30% Out-of-sample</option>
              <option value={0.35}>35% Out-of-sample</option>
            </select>
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Admin Secret Key</label>
            <input
              type="password"
              value={adminKey}
              onChange={(e) => setAdminKey(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-3 py-2 text-white font-mono"
            />
          </div>
        </div>

        {/* AI Continuous Learning & Hard-Negative Mining Option */}
        <div className="mb-4 p-3 bg-dark-850 border border-dark-750 rounded-lg">
          <label className="flex items-center space-x-2 text-gray-200 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={learnFromMistakes}
              onChange={(e) => setLearnFromMistakes(e.target.checked)}
              className="rounded bg-dark-800 border-dark-600 text-gold-500 focus:ring-0 w-4 h-4"
            />
            <span className="font-semibold text-gold-400 text-xs">
              🧠 AI Hard-Negative Mistake Mining & Continuous Learning
            </span>
          </label>
          <p className="text-[11px] text-gray-400 mt-1 pl-6">
            When enabled, historical failure trap candles (Volatility Spikes, Low-Momentum Chop, Climax Exhaustion) are re-weighted with 2.5x sample penalties, forcing the gradient boosted decision trees to actively avoid repeating those false-breakout patterns.
          </p>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-dark-800">
          <p className="text-xs text-gray-400">
            Retraining fits Model A (XGBoost), B (LightGBM), C (Regime), D (Price Action), and calibrates Model E.
          </p>

          <button
            onClick={handleRetrain}
            disabled={loading}
            className="bg-gold-500 hover:bg-gold-600 disabled:opacity-50 text-dark-950 font-bold px-4 py-2 rounded text-xs flex items-center space-x-1.5 transition"
          >
            {loading ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Retraining Suite...</span>
              </>
            ) : (
              <>
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Train New Candidate</span>
              </>
            )}
          </button>
        </div>

        {statusMsg && (
          <div className="mt-3 p-3 rounded bg-dark-850 border border-dark-700 text-xs text-gold-400 font-mono">
            {statusMsg}
          </div>
        )}
      </div>

      {/* Version Registry Table */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300">
            Model Version Registry
          </h3>
          <button
            onClick={loadModels}
            className="text-xs text-gray-400 hover:text-white flex items-center space-x-1"
          >
            <RefreshCw className={`w-3 h-3 ${refreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-dark-850 text-gray-400 border-b border-dark-700">
              <tr>
                <th className="py-2.5 px-3">Status</th>
                <th className="py-2.5 px-3">Version</th>
                <th className="py-2.5 px-3">Suite Name</th>
                <th className="py-2.5 px-3">Sub-Models</th>
                <th className="py-2.5 px-3">Selective Threshold</th>
                <th className="py-2.5 px-3">Registered At</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-800 text-gray-300">
              {models.map((m) => (
                <tr key={m.version} className="hover:bg-dark-850/50">
                  <td className="py-2.5 px-3">
                    {m.is_active ? (
                      <span className="inline-flex items-center space-x-1.5 bg-bull-500/20 border border-bull-500/40 text-bull-400 px-2.5 py-1 rounded text-[10px] font-bold">
                        <span className="w-1.5 h-1.5 rounded-full bg-bull-400 animate-pulse" />
                        <span>ACTIVE PRODUCTION</span>
                      </span>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <span className="bg-dark-750 text-gray-400 px-2 py-0.5 rounded text-[10px] font-mono">
                          CANDIDATE
                        </span>
                        <button
                          onClick={() => handleActivate(m.version)}
                          disabled={activatingVersion === m.version}
                          className="bg-gold-500/15 hover:bg-gold-500/30 text-gold-400 hover:text-gold-300 border border-gold-500/40 px-2.5 py-0.5 rounded text-[10px] font-bold transition flex items-center space-x-1 disabled:opacity-50"
                          title={`Switch production engine to ${m.version}`}
                        >
                          {activatingVersion === m.version ? (
                            <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                          ) : (
                            <CheckCircle className="w-2.5 h-2.5" />
                          )}
                          <span>Set Active</span>
                        </button>
                      </div>
                    )}
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-white">{m.version}</td>
                  <td className="py-2.5 px-3">{m.name}</td>
                  <td className="py-2.5 px-3">
                    <div className="flex space-x-1">
                      {m.submodels.map((sm) => (
                        <span key={sm} className="bg-dark-800 px-1.5 py-0.5 rounded text-[10px] text-gray-400">
                          {sm}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-2.5 px-3 font-mono">{(m.confidence_threshold * 100).toFixed(0)}%</td>
                  <td className="py-2.5 px-3 text-gray-400 font-mono">{m.train_date || 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
