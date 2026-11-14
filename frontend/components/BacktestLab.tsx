'use client';

import React, { useState, useEffect, useRef } from 'react';
import { BacktestResponse, runBacktest, fetchBacktestResults } from '../lib/api';
import {
  createChart,
  IChartApi,
  ISeriesApi,
  ColorType,
  Time,
} from 'lightweight-charts';
import { Play, RotateCcw, TrendingUp, AlertTriangle, CheckCircle, Shield } from 'lucide-react';

export default function BacktestLab() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestResponse | null>(null);

  // Parameter states
  const [initialCapital, setInitialCapital] = useState(10000);
  const [spread, setSpread] = useState(0.20);
  const [slippage, setSlippage] = useState(0.05);
  const [riskPct, setRiskPct] = useState(0.01);
  const [threshold, setThreshold] = useState(0.45);
  const [maxBars, setMaxBars] = useState(24);

  // Chart ref
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const areaSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  // Load cached backtest results on mount
  useEffect(() => {
    fetchBacktestResults()
      .then((res) => setResults(res))
      .catch((err) => console.log('No cached backtest:', err));
  }, []);

  // Equity curve chart setup
  useEffect(() => {
    if (!chartContainerRef.current) return;

    if (chartRef.current) {
      chartRef.current.remove();
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
      rightPriceScale: {
        borderColor: '#21262d',
      },
      timeScale: {
        borderColor: '#21262d',
      },
    });

    const areaSeries = chart.addAreaSeries({
      topColor: 'rgba(16, 185, 129, 0.4)',
      bottomColor: 'rgba(16, 185, 129, 0.0)',
      lineColor: '#10b981',
      lineWidth: 2,
    });

    chartRef.current = chart;
    areaSeriesRef.current = areaSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update equity curve data
  useEffect(() => {
    if (!areaSeriesRef.current || !results || !results.equity_curve) return;

    // Map equity curve values
    const data = results.equity_curve.map((val, idx) => ({
      time: (idx + 1) as unknown as Time,
      value: val,
    }));
    areaSeriesRef.current.setData(data);
  }, [results]);

  const handleRunSimulation = async () => {
    setLoading(true);
    try {
      const res = await runBacktest({
        initial_capital: Number(initialCapital),
        spread: Number(spread),
        slippage: Number(slippage),
        risk_per_trade: Number(riskPct),
        threshold: Number(threshold),
        max_holding_bars: Number(maxBars),
      });
      setResults(res);
    } catch (err: any) {
      alert(`Simulation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Simulation Parameter Controls Bar */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-200 flex items-center space-x-2">
            <Shield className="w-4 h-4 text-gold-500" />
            <span>Realistic Execution Assumptions</span>
          </h2>
          <span className="text-xs text-gray-400">
            Intrabar SL First Conflict Resolution | No Optimistic Assumptions
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-xs">
          <div>
            <label className="text-gray-400 block mb-1">Initial Capital ($)</label>
            <input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Spread ($ USD)</label>
            <input
              type="number"
              step="0.05"
              value={spread}
              onChange={(e) => setSpread(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Slippage ($ USD)</label>
            <input
              type="number"
              step="0.01"
              value={slippage}
              onChange={(e) => setSlippage(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Risk Per Trade</label>
            <select
              value={riskPct}
              onChange={(e) => setRiskPct(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            >
              <option value={0.005}>0.5% Account</option>
              <option value={0.01}>1.0% Account</option>
              <option value={0.015}>1.5% Account</option>
              <option value={0.02}>2.0% Account</option>
            </select>
          </div>

          <div>
            <label className="text-gray-400 block mb-1">Confidence Threshold</label>
            <input
              type="number"
              step="0.02"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full bg-dark-800 border border-dark-700 rounded px-2.5 py-1.5 text-white"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={handleRunSimulation}
              disabled={loading}
              className="w-full bg-gold-500 hover:bg-gold-600 disabled:opacity-50 text-dark-950 font-bold py-1.5 px-3 rounded flex items-center justify-center space-x-1.5 transition"
            >
              {loading ? (
                <span>Simulating...</span>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Run Backtest</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      {results && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Net Profit</span>
            <span className={`text-lg font-bold ${results.total_net_profit >= 0 ? 'text-bull-400' : 'text-bear-400'}`}>
              ${results.total_net_profit.toLocaleString()}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              {results.return_pct >= 0 ? '+' : ''}{results.return_pct}% return
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Win Rate</span>
            <span className="text-lg font-bold text-white">
              {results.win_rate_pct}%
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              {results.winning_trades}W / {results.losing_trades}L ({results.total_trades} trades)
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Profit Factor</span>
            <span className="text-lg font-bold text-gold-400">
              {results.profit_factor.toFixed(2)}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">Gross Win / Gross Loss</span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Max Drawdown</span>
            <span className="text-lg font-bold text-bear-400">
              -{results.max_drawdown_pct}%
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              -${results.max_drawdown_usd.toFixed(2)}
            </span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Sharpe Ratio</span>
            <span className="text-lg font-bold text-white">
              {results.sharpe_ratio.toFixed(2)}
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">Risk-adjusted return</span>
          </div>

          <div className="bg-dark-900 border border-dark-700 rounded-lg p-4">
            <span className="text-xs text-gray-400 block mb-1">Avg R-Multiple</span>
            <span className="text-lg font-bold text-white">
              {results.avg_r_multiple > 0 ? '+' : ''}{results.avg_r_multiple.toFixed(2)}R
            </span>
            <span className="text-[10px] text-gray-400 block mt-0.5">
              Avg Trade: ${results.avg_trade_usd.toFixed(2)}
            </span>
          </div>
        </div>
      )}

      {/* Equity Curve Canvas */}
      <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
        <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-3">
          Simulated Equity Curve (Out-of-Sample Period)
        </h3>
        <div ref={chartContainerRef} className="w-full h-72" />
      </div>

      {/* Simulated Trades Log Table */}
      {results && results.trades && results.trades.length > 0 && (
        <div className="bg-dark-900 border border-dark-700 rounded-lg p-5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-gray-300 mb-3">
            Execution Log ({results.trades.length} Completed Positions)
          </h3>
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-dark-850 text-gray-400 sticky top-0 border-b border-dark-700">
                <tr>
                  <th className="py-2 px-3">Type</th>
                  <th className="py-2 px-3">Entry Price</th>
                  <th className="py-2 px-3">Exit Price</th>
                  <th className="py-2 px-3">PnL ($)</th>
                  <th className="py-2 px-3">R-Multiple</th>
                  <th className="py-2 px-3">Exit Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-800 text-gray-300 font-mono">
                {results.trades.map((t, idx) => (
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
        </div>
      )}
    </div>
  );
}
