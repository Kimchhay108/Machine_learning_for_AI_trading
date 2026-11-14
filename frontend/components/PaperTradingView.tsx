'use client';

import React, { useState, useEffect } from 'react';
import {
  PaperAccountSummary,
  fetchPaperAccount,
  fetchPaperTrades,
  startPaperTrading,
  stopPaperTrading,
  resetPaperAccount,
  triggerPaperTick,
} from '../lib/api';
import {
  Play,
  Pause,
  RotateCcw,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Target,
  Shield,
  Wallet,
  Activity,
} from 'lucide-react';

export default function PaperTradingView() {
  const [account, setAccount] = useState<PaperAccountSummary | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [tickResult, setTickResult] = useState<string | null>(null);

  const loadData = () => {
    Promise.all([fetchPaperAccount(), fetchPaperTrades()])
      .then(([acc, trs]) => {
        setAccount(acc);
        setTrades(trs);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching paper data:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Polling every 5s
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setActionLoading(true);
    await startPaperTrading();
    loadData();
    setActionLoading(false);
  };

  const handleStop = async () => {
    setActionLoading(true);
    await stopPaperTrading();
    loadData();
    setActionLoading(false);
  };

  const handleReset = async () => {
    if (confirm('Are you sure you want to reset paper account balance to $10,000.00?')) {
      setActionLoading(true);
      await resetPaperAccount(10000);
      loadData();
      setActionLoading(false);
    }
  };

  const handleTick = async () => {
    setActionLoading(true);
    setTickResult(null);
    try {
      const res = await triggerPaperTick();
      setTickResult(
        `Tick executed: ${res.event.action} - ${res.event.message || JSON.stringify(res.event)}`
      );
      loadData();
    } catch (err: any) {
      setTickResult(`Tick failed: ${err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading || !account) {
    return (
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-8 text-center text-gray-400">
        Connecting to paper trading engine and virtual ledger...
      </div>
    );
  }

  const {
    balance,
    equity,
    net_profit,
    return_pct,
    total_trades,
    winning_trades,
    losing_trades,
    win_rate_pct,
    profit_factor,
    open_position,
    is_running,
  } = account;

  return (
    <div className="space-y-6">
      {/* Top Banner & Engine Daemon Controls */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5 flex flex-col md:flex-row items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2 mb-1">
            <Wallet className="w-5 h-5 text-gold-500" />
            <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200">
              Live Paper Trading & Execution Daemon (Phase 10)
            </h2>
          </div>
          <p className="text-xs text-gray-400">
            Real-time simulated order fills, 1.0% fractional risk sizing, dynamic ATR TP/SL, and zero capital risk.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {is_running ? (
            <button
              onClick={handleStop}
              disabled={actionLoading}
              className="bg-bear-500 hover:bg-bear-600 text-white font-bold px-3 py-1.5 rounded text-xs flex items-center space-x-1 transition"
            >
              <Pause className="w-3.5 h-3.5 fill-current" />
              <span>Pause Daemon</span>
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={actionLoading}
              className="bg-bull-500 hover:bg-bull-600 text-dark-950 font-bold px-3 py-1.5 rounded text-xs flex items-center space-x-1 transition"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Start Daemon</span>
            </button>
          )}

          <button
            onClick={handleTick}
            disabled={actionLoading}
            className="bg-dark-800 hover:bg-dark-700 text-gray-200 border border-dark-700 px-3 py-1.5 rounded text-xs flex items-center space-x-1 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${actionLoading ? 'animate-spin' : ''}`} />
            <span>Poll Now</span>
          </button>

          <button
            onClick={handleReset}
            disabled={actionLoading}
            className="bg-dark-800 hover:bg-dark-700 text-gray-400 hover:text-white border border-dark-700 px-3 py-1.5 rounded text-xs flex items-center space-x-1 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset ($10k)</span>
          </button>
        </div>
      </div>

      {tickResult && (
        <div className="p-3 bg-dark-850 border border-dark-700 rounded text-xs font-mono text-gold-400">
          {tickResult}
        </div>
      )}

      {/* Account KPI Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Portfolio Equity</span>
          <span className="text-xl font-bold text-white font-mono">
            ${equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">
            Balance: ${balance.toFixed(2)}
          </span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Net Realized PnL</span>
          <span className={`text-xl font-bold font-mono ${net_profit >= 0 ? 'text-bull-400' : 'text-bear-400'}`}>
            {net_profit >= 0 ? '+' : ''}${net_profit.toFixed(2)}
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">
            {return_pct >= 0 ? '+' : ''}{return_pct}% return
          </span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Win Rate</span>
          <span className="text-xl font-bold text-white">
            {win_rate_pct}%
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">
            {winning_trades}W / {losing_trades}L ({total_trades} trades)
          </span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Profit Factor</span>
          <span className="text-xl font-bold text-gold-400 font-mono">
            {profit_factor.toFixed(2)}
          </span>
          <span className="text-[10px] text-gray-400 block mt-0.5">Gross Win / Gross Loss</span>
        </div>

        <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
          <span className="text-xs text-gray-400 block mb-1">Daemon Status</span>
          <div className="flex items-center space-x-1.5 mt-1">
            <span className={`w-2.5 h-2.5 rounded-full ${is_running ? 'bg-bull-400 animate-pulse' : 'bg-gray-500'}`} />
            <span className={`font-bold text-xs ${is_running ? 'text-bull-400' : 'text-gray-400'}`}>
              {is_running ? 'ACTIVE POLLING' : 'IDLE / PAUSED'}
            </span>
          </div>
          <span className="text-[10px] text-gray-400 block mt-1">Updates every 15s</span>
        </div>
      </div>

      {/* Open Position Card */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-3 flex items-center space-x-2">
          <Activity className="w-4 h-4 text-gold-500" />
          <span>Active Position Tracker</span>
        </h3>

        {open_position ? (
          <div className="p-4 bg-dark-850 rounded-lg border border-dark-700 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span
                  className={`px-2.5 py-1 rounded font-bold text-xs ${
                    open_position.direction === 'BUY'
                      ? 'bg-bull-500/20 text-bull-400 border border-bull-500/40'
                      : 'bg-bear-500/20 text-bear-400 border border-bear-500/40'
                  }`}
                >
                  {open_position.direction}
                </span>
                <span className="text-white font-mono font-bold text-sm">
                  {open_position.instrument} ({open_position.size_oz} oz)
                </span>
              </div>

              <div className="text-right">
                <span className="text-xs text-gray-400 block">Unrealized PnL</span>
                <span
                  className={`text-lg font-bold font-mono ${
                    open_position.unrealized_pnl >= 0 ? 'text-bull-400' : 'text-bear-400'
                  }`}
                >
                  {open_position.unrealized_pnl >= 0 ? '+' : ''}${open_position.unrealized_pnl.toFixed(2)}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <span className="text-gray-400 block text-[10px]">Entry Price</span>
                <span className="text-white">${open_position.entry_price.toFixed(2)}</span>
              </div>
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <span className="text-gray-400 block text-[10px]">Current Price</span>
                <span className="text-white">${open_position.current_price.toFixed(2)}</span>
              </div>
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <span className="text-bear-400 block text-[10px]">Stop Loss</span>
                <span className="text-bear-400">${open_position.stop_loss.toFixed(2)}</span>
              </div>
              <div className="p-2 bg-dark-800 rounded border border-dark-700">
                <span className="text-bull-400 block text-[10px]">Take Profit</span>
                <span className="text-bull-400">${open_position.take_profit.toFixed(2)}</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-6 bg-dark-850 rounded border border-dark-700 text-center text-gray-400 text-xs">
            No active positions open. The AI is observing the market in selective abstention (`WAIT`) mode.
          </div>
        )}
      </div>

      {/* Trade Execution History */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-3">
          Completed Paper Trades ({trades.length} Closed)
        </h3>

        {trades.length > 0 ? (
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-dark-850 text-gray-400 sticky top-0 border-b border-dark-700">
                <tr>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Entry Price</th>
                  <th className="py-2 px-3">Exit Price</th>
                  <th className="py-2 px-3">Size (oz)</th>
                  <th className="py-2 px-3">PnL ($)</th>
                  <th className="py-2 px-3">R-Multiple</th>
                  <th className="py-2 px-3">Exit Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800 text-gray-300 font-mono">
                {trades.map((t, idx) => (
                  <tr key={idx} className="hover:bg-dark-850/50">
                    <td className="py-2 px-3">
                      <span
                        className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                          t.direction === 'BUY' ? 'bg-bull-500/20 text-bull-400' : 'bg-bear-500/20 text-bear-400'
                        }`}
                      >
                        {t.direction}
                      </span>
                    </td>
                    <td className="py-2 px-3">${t.entry_price.toFixed(2)}</td>
                    <td className="py-2 px-3">${t.exit_price.toFixed(2)}</td>
                    <td className="py-2 px-3">{t.size_oz}</td>
                    <td className={`py-2 px-3 font-semibold ${t.pnl_usd >= 0 ? 'text-bull-400' : 'text-bear-400'}`}>
                      {t.pnl_usd >= 0 ? '+' : ''}${t.pnl_usd.toFixed(2)}
                    </td>
                    <td className="py-2 px-3">{t.r_multiple > 0 ? '+' : ''}{t.r_multiple.toFixed(2)}R</td>
                    <td className="py-2 px-3">
                      <span className="text-gray-400 text-[11px] font-sans">{t.exit_reason}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-gray-400">No completed trades recorded yet.</p>
        )}
      </div>
    </div>
  );
}
