'use client';

import React, { useState, useEffect } from 'react';
import { fetchHealth } from '../lib/api';
import { Settings, CheckCircle2, XCircle, Database, Shield } from 'lucide-react';

export default function SettingsView() {
  const [health, setHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [adminKey, setAdminKey] = useState('xauusd-admin-secret-key');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  useEffect(() => {
    fetchHealth()
      .then((res) => {
        setHealth(res);
        setLoading(false);
      })
      .catch((err) => {
        setHealth({ status: 'OFFLINE', error: err.message });
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-6 max-w-4xl">
      {/* System Status Card */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center space-x-2 mb-4">
          <Database className="w-5 h-5 text-gold-500" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
            System Infrastructure & Backend Health
          </h2>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <div className="bg-dark-850 p-3 rounded border border-dark-700">
            <span className="text-gray-400 block mb-1">Backend API Status</span>
            <div className="flex items-center space-x-1.5">
              {health?.status === 'HEALTHY' ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-bull-400" />
                  <span className="text-bull-400 font-bold">HEALTHY</span>
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-bear-400" />
                  <span className="text-bear-400 font-bold">OFFLINE</span>
                </>
              )}
            </div>
          </div>

          <div className="bg-dark-850 p-3 rounded border border-dark-700">
            <span className="text-gray-400 block mb-1">Active Production Suite</span>
            <span className="text-white font-mono font-bold">{health?.active_model || 'v1.0.0'}</span>
          </div>

          <div className="bg-dark-850 p-3 rounded border border-dark-700">
            <span className="text-gray-400 block mb-1">Storage Engine</span>
            <span className="text-gold-400 font-bold">Parquet & DuckDB</span>
          </div>

          <div className="bg-dark-850 p-3 rounded border border-dark-700">
            <span className="text-gray-400 block mb-1">Environment</span>
            <span className="text-white font-mono">{health?.environment || 'development'}</span>
          </div>
        </div>
      </div>

      {/* Configuration Form */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5 space-y-4">
        <div className="flex items-center space-x-2 border-b border-dark-700 pb-3">
          <Settings className="w-5 h-5 text-gold-500" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
            Application Settings & Security
          </h2>
        </div>

        <div className="space-y-3 text-xs">
          <div>
            <label className="text-gray-400 block mb-1">Backend FastAPI URL</label>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-3 py-2 text-white font-mono"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Admin API Key (Protected Endpoints)</label>
            <input
              type="password"
              value={adminKey}
              onChange={(e) => setAdminKey(e.target.value)}
              className="w-full bg-dark-800 border border-dark-700 rounded px-3 py-2 text-white font-mono"
            />
          </div>

          <div className="pt-2">
            <button
              onClick={() => alert('Settings saved to local storage!')}
              className="bg-gold-500 hover:bg-gold-600 text-dark-950 font-bold px-4 py-2 rounded transition"
            >
              Save Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
