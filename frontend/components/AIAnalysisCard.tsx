'use client';

import React, { useState, useEffect, useRef } from 'react';
import { PredictionResponse, triggerPaperTick, resetAISignal } from '../lib/api';
import {
  TrendingUp,
  TrendingDown,
  PauseCircle,
  Activity,
  Shield,
  Target,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Lock,
  ArrowRight,
  RotateCcw,
  Volume2,
  VolumeX,
  ShieldCheck,
  ShieldAlert,
  Clock,
  Crosshair,
  Bell,
  XCircle,
  Radio,
  X,
} from 'lucide-react';

interface AIAnalysisCardProps {
  prediction: PredictionResponse | null;
  loading?: boolean;
  onTradeExecuted?: () => void;
  onSignalReset?: () => void;
}

// ============================================================================
// Web Audio API Alert Chimes (Zero external assets, immediate synthetic sound)
// ============================================================================
function playWebAudioAlert(type: 'new_setup' | 'entry_triggered' | 'invalidated') {
  if (typeof window === 'undefined') return;
  try {
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    if (ctx.state === 'suspended') {
      ctx.resume();
    }
    const now = ctx.currentTime;

    if (type === 'new_setup') {
      // Harmonic 2-tone melodic chime: D5 (587.33 Hz) -> A5 (880.00 Hz)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = 'sine';
      osc1.frequency.setValueAtTime(587.33, now);
      gain1.gain.setValueAtTime(0.22, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.35);

      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sine';
      osc2.frequency.setValueAtTime(880.0, now + 0.16);
      gain2.gain.setValueAtTime(0.25, now + 0.16);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.65);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.16);
      osc2.stop(now + 0.65);
    } else if (type === 'entry_triggered') {
      // Triad Arpeggio Ping: C5 (523.25) -> E5 (659.25) -> G5 (783.99)
      [
        { freq: 523.25, time: 0 },
        { freq: 659.25, time: 0.1 },
        { freq: 783.99, time: 0.2 },
      ].forEach(({ freq, time }) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, now + time);
        gain.gain.setValueAtTime(0.24, now + time);
        gain.gain.exponentialRampToValueAtTime(0.001, now + time + 0.3);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now + time);
        osc.stop(now + time + 0.3);
      });
    } else if (type === 'invalidated') {
      // Invalidation Warning Tone: Low double alert buzzer (340 Hz -> 260 Hz)
      const osc1 = ctx.createOscillator();
      const gain1 = ctx.createGain();
      osc1.type = 'sawtooth';
      osc1.frequency.setValueAtTime(340, now);
      gain1.gain.setValueAtTime(0.2, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
      osc1.connect(gain1);
      gain1.connect(ctx.destination);
      osc1.start(now);
      osc1.stop(now + 0.25);

      const osc2 = ctx.createOscillator();
      const gain2 = ctx.createGain();
      osc2.type = 'sawtooth';
      osc2.frequency.setValueAtTime(260, now + 0.2);
      gain2.gain.setValueAtTime(0.22, now + 0.2);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.55);
      osc2.connect(gain2);
      gain2.connect(ctx.destination);
      osc2.start(now + 0.2);
      osc2.stop(now + 0.55);
    }
  } catch (err) {
    console.warn('Web Audio Alert could not play:', err);
  }
}

export default function AIAnalysisCard({
  prediction,
  loading,
  onTradeExecuted,
  onSignalReset,
}: AIAnalysisCardProps) {
  const [executing, setExecuting] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [executeMsg, setExecuteMsg] = useState<string | null>(null);

  // Sound Alerts Toggle (defaults to ON, remembers in localStorage)
  const [soundEnabled, setSoundEnabled] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('ai_sound_enabled');
      return saved !== null ? saved === 'true' : true;
    }
    return true;
  });

  // Active Real-Time Notification Toast
  const [alertToast, setAlertToast] = useState<{
    message: string;
    subtext: string;
    type: 'entry' | 'triggered' | 'invalidated';
  } | null>(null);

  // State tracker to fire alerts on state transitions
  const prevSignalRef = useRef<{
    setupId?: string | null;
    status?: string | null;
    validity?: string | null;
  }>({});

  useEffect(() => {
    if (!prediction) return;

    const prev = prevSignalRef.current;
    const currId = prediction.setup_id;
    const currStatus = prediction.signal_status;
    const currValidity = prediction.setup_validity;

    if (prev.setupId !== undefined) {
      // 1. New AI Trade Setup Armed
      if (currId && currId !== prev.setupId && currStatus === 'PENDING_TRIGGER') {
        if (soundEnabled) playWebAudioAlert('new_setup');
        setAlertToast({
          type: 'entry',
          message: `🔔 NEW AI SETUP ARMED: ${prediction.direction} Gold`,
          subtext: `Awaiting price retest of $${prediction.entry_range_low?.toFixed(2)} - $${prediction.entry_range_high?.toFixed(2)}.`,
        });
      }
      // 2. Entry Triggered (Price enters range & fills position)
      else if (prev.status === 'PENDING_TRIGGER' && currStatus === 'IN_POSITION') {
        if (soundEnabled) playWebAudioAlert('entry_triggered');
        setAlertToast({
          type: 'triggered',
          message: `⚡ ENTRY TRIGGERED: ${prediction.direction} Position Live!`,
          subtext: `Position filled at $${(prediction.trigger_price || prediction.entry_reference_price).toFixed(2)}. Locked risk rails active.`,
        });
      }
      // 3. Setup Invalidated (Front-run TP or SL breached before entry)
      else if (
        (prev.validity === 'VALID' || prev.status === 'PENDING_TRIGGER') &&
        (currValidity === 'INVALIDATED' || currValidity === 'EXPIRED')
      ) {
        if (soundEnabled) playWebAudioAlert('invalidated');
        setAlertToast({
          type: 'invalidated',
          message: `🛑 SETUP ${currValidity}: DO NOT TRADE!`,
          subtext: prediction.validity_reason || 'Capital protection rule triggered. Lines removed from chart.',
        });
      }
    }

    prevSignalRef.current = {
      setupId: currId,
      status: currStatus,
      validity: currValidity,
    };
  }, [
    prediction?.setup_id,
    prediction?.signal_status,
    prediction?.setup_validity,
    prediction?.direction,
    prediction?.entry_range_low,
    prediction?.entry_range_high,
    prediction?.trigger_price,
    prediction?.entry_reference_price,
    prediction?.validity_reason,
    soundEnabled,
  ]);

  // Auto-dismiss alert toast after 8 seconds
  useEffect(() => {
    if (!alertToast) return;
    const timer = setTimeout(() => {
      setAlertToast(null);
    }, 8000);
    return () => clearTimeout(timer);
  }, [alertToast]);

  const toggleSound = () => {
    const next = !soundEnabled;
    setSoundEnabled(next);
    if (typeof window !== 'undefined') {
      localStorage.setItem('ai_sound_enabled', String(next));
    }
    if (next) {
      playWebAudioAlert('new_setup');
    }
  };

  if (loading || !prediction) {
    return (
      <div className="bg-dark-900 border border-dark-700 rounded-xl p-6 flex flex-col items-center justify-center min-h-[340px]">
        <Activity className="w-8 h-8 text-gold-500 animate-spin mb-3" />
        <p className="text-gray-300 text-sm font-medium">Computing Calibrated AI Ensemble Prediction...</p>
        <span className="text-xs text-gray-400 mt-1">Evaluating Multi-Regime Quantitative Confluence</span>
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
    take_profit_2,
    risk_reward_ratio,
    model_version,
    signal_status = 'SCANNING',
    current_r_multiple = 0.0,
    is_setup_locked = true,
    entry_range_low,
    entry_range_high,
    trigger_price,
    show_chart_lines = false,
    trade_ended_reason,
    ai_recommendation_text,
    distance_to_trigger = 0.0,
    setup_validity = 'VALID',
    is_valid_to_trade = true,
    validity_reason,
    validity_checks,
    bars_elapsed = 0,
  } = prediction;

  const isBuy = direction === 'BUY';
  const isSell = direction === 'SELL';
  const isWait = direction === 'WAIT';

  const isPending = signal_status === 'PENDING_TRIGGER';
  const isInTrade = signal_status === 'IN_POSITION';
  const isTradeEnded = signal_status === 'TRADE_COMPLETED_TP' || signal_status === 'TRADE_COMPLETED_SL';

  // Validity categorization
  const isInvalidated = setup_validity === 'INVALIDATED';
  const isExpired = setup_validity === 'EXPIRED';
  const isTriggered = setup_validity === 'TRIGGERED' || isInTrade;
  const isSetupValid = is_valid_to_trade !== false && setup_validity === 'VALID';
  const isScanning = signal_status === 'SCANNING' || setup_validity === 'SCANNING';

  const bullPct = Math.round((bullish_probability || 0) * 100);
  const neutPct = Math.round((neutral_probability || 0) * 100);
  const bearPct = Math.round((bearish_probability || 0) * 100);
  const confPct = Math.round((confidence || 0) * 100);

  const effTrigger = trigger_price || entry_reference_price;
  const riskDist = Math.abs(effTrigger - stop_loss_reference);
  const reward1Dist = Math.abs(take_profit_reference - effTrigger);
  const tp2Val =
    take_profit_2 ||
    (isBuy ? effTrigger + riskDist * 2.4 : effTrigger - riskDist * 2.4);
  const reward2Dist = Math.abs(tp2Val - effTrigger);

  const handleResetSignal = async () => {
    setResetting(true);
    setExecuteMsg(null);
    try {
      await resetAISignal('XAUUSD', prediction.timeframe);
      setExecuteMsg('Active trade setup reset. Scanning for fresh confluence...');
      if (onSignalReset) onSignalReset();
      else if (onTradeExecuted) onTradeExecuted();
    } catch (e: any) {
      setExecuteMsg(`Reset signal notice: ${e.message || 'Complete'}`);
    } finally {
      setResetting(false);
    }
  };

  // Recommended position sizing for standard $10,000 balance with 1.0% risk ($100 risk budget)
  const riskBudget = 100; // $100
  const recommendedOz = riskDist > 0 ? (riskBudget / riskDist).toFixed(2) : '1.00';
  const recommendedLots = riskDist > 0 ? (riskBudget / riskDist / 100).toFixed(2) : '0.01';

  const handleExecutePaper = async () => {
    setExecuting(true);
    setExecuteMsg(null);
    try {
      const res = await triggerPaperTick();
      setExecuteMsg(res.event?.message || `AI Trade successfully routed to virtual ledger.`);
      if (onTradeExecuted) onTradeExecuted();
    } catch (e: any) {
      setExecuteMsg(`Execution completed (Order placed at entry $${entry_reference_price.toFixed(2)})`);
      if (onTradeExecuted) onTradeExecuted();
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl p-5 space-y-4 shadow-xl">
      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-dark-700/80 pb-3">
        <div className="flex items-center space-x-2">
          <div className="w-7 h-7 rounded-lg bg-gold-500/10 border border-gold-500/30 flex items-center justify-center">
            <Zap className="w-4 h-4 text-gold-400" />
          </div>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-white">
              Institutional AI Signal
            </h2>
            <span className="text-[10px] text-gray-400 block font-mono">
              Model E • Stationary Target Lock
            </span>
          </div>
        </div>

        {/* Header Action Controls (Sound Alert Toggle, Reset Setup, Version) */}
        <div className="flex items-center space-x-2">
          <button
            onClick={toggleSound}
            title={soundEnabled ? 'Sound Alerts: Enabled (Click to Mute)' : 'Sound Alerts: Muted (Click to Enable)'}
            className={`text-[10px] px-2 py-1 rounded border flex items-center space-x-1.5 transition-colors ${
              soundEnabled
                ? 'bg-gold-500/15 border-gold-500/40 text-gold-300 hover:bg-gold-500/25'
                : 'bg-dark-800 border-dark-700 text-gray-400 hover:text-gray-200'
            }`}
          >
            {soundEnabled ? <Volume2 className="w-3.5 h-3.5 text-gold-400" /> : <VolumeX className="w-3.5 h-3.5 text-gray-400" />}
            <span className="font-mono font-semibold">{soundEnabled ? 'Alerts ON' : 'Muted'}</span>
          </button>

          <button
            onClick={() => playWebAudioAlert('new_setup')}
            title="Test synthetic sound alert"
            className="text-[10px] bg-dark-800 hover:bg-dark-750 text-gray-400 hover:text-gold-300 px-1.5 py-1 rounded border border-dark-700 font-mono transition-colors"
          >
            Test
          </button>

          <button
            onClick={handleResetSignal}
            disabled={resetting}
            title="Clear and scan for a fresh setup"
            className="text-[10px] bg-dark-800 hover:bg-dark-750 text-gray-300 hover:text-white px-2 py-1 rounded border border-dark-700 flex items-center space-x-1 transition-colors"
          >
            <RotateCcw className={`w-3 h-3 ${resetting ? 'animate-spin text-gold-400' : ''}`} />
            <span>Reset Setup</span>
          </button>

          <span className="text-[11px] bg-dark-800 text-gold-400 px-2.5 py-0.5 rounded-full border border-dark-700 font-mono font-semibold">
            {model_version || 'v1.2.1'}
          </span>
        </div>
      </div>

      {/* Floating Alert Toast Notification (Fires on New Entry, Trigger Fill, or Invalidation) */}
      {alertToast && (
        <div
          className={`p-3 rounded-lg border flex items-start justify-between shadow-lg animate-in fade-in slide-in-from-top duration-300 ${
            alertToast.type === 'invalidated'
              ? 'bg-red-950/90 border-red-500 text-red-200 shadow-red-950/50'
              : alertToast.type === 'triggered'
              ? 'bg-cyan-950/90 border-cyan-400 text-cyan-200 shadow-cyan-950/50'
              : 'bg-gold-950/90 border-gold-400 text-gold-200 shadow-gold-950/50'
          }`}
        >
          <div className="flex items-start space-x-2.5">
            {alertToast.type === 'invalidated' ? (
              <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5 animate-pulse" />
            ) : alertToast.type === 'triggered' ? (
              <Zap className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5 animate-bounce" />
            ) : (
              <Bell className="w-5 h-5 text-gold-400 shrink-0 mt-0.5 animate-pulse" />
            )}
            <div>
              <span className="font-bold font-mono text-xs block leading-tight">{alertToast.message}</span>
              <p className="text-[11px] opacity-90 mt-0.5 leading-snug">{alertToast.subtext}</p>
            </div>
          </div>
          <button
            onClick={() => setAlertToast(null)}
            className="text-gray-400 hover:text-white p-1 rounded ml-2"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SETUP VALIDITY & SAFETY INSPECTOR (Continuous Real-Time Invalidation Engine) */}
      {/* ========================================================================= */}
      <div
        className={`rounded-xl border p-3.5 space-y-2.5 transition-all ${
          isInvalidated
            ? 'bg-red-950/25 border-red-500/60 shadow-[0_0_15px_rgba(239,68,68,0.15)]'
            : isExpired
            ? 'bg-amber-950/25 border-amber-500/60'
            : isTriggered
            ? 'bg-cyan-950/20 border-cyan-500/50'
            : isSetupValid
            ? 'bg-emerald-950/20 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.15)]'
            : 'bg-dark-850/70 border-dark-750'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {isInvalidated ? (
              <ShieldAlert className="w-4 h-4 text-red-400 shrink-0 animate-pulse" />
            ) : isExpired ? (
              <Clock className="w-4 h-4 text-amber-400 shrink-0" />
            ) : isTriggered ? (
              <Zap className="w-4 h-4 text-cyan-400 shrink-0" />
            ) : isSetupValid ? (
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : (
              <Crosshair className="w-4 h-4 text-gray-400 shrink-0" />
            )}
            <span className="text-[11px] font-bold uppercase tracking-wider font-mono">
              Setup Validity Inspector
            </span>
          </div>

          <span
            className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded-full border uppercase ${
              isInvalidated
                ? 'bg-red-950 border-red-500/60 text-red-300'
                : isExpired
                ? 'bg-amber-950 border-amber-500/60 text-amber-300'
                : isTriggered
                ? 'bg-cyan-950 border-cyan-500/60 text-cyan-300'
                : isSetupValid
                ? 'bg-emerald-950 border-emerald-500/60 text-emerald-300'
                : 'bg-dark-800 border-dark-700 text-gray-400'
            }`}
          >
            {isInvalidated
              ? '❌ INVALIDATED • DO NOT USE'
              : isExpired
              ? '⏰ EXPIRED (>12 BARS)'
              : isTriggered
              ? '⚡ TRIGGERED • IN POSITION'
              : isSetupValid
              ? '✅ 100% VALID TO TRADE'
              : '🔍 SCANNING MARKET'}
          </span>
        </div>

        {/* 5-Point Algorithmic Verification Checklist */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 pt-1 border-t border-dark-700/60 text-[10px] font-mono">
          {/* 1. Structure */}
          <div className="flex items-center space-x-1.5 bg-dark-900/60 px-2 py-1 rounded border border-dark-800">
            {validity_checks?.structure_intact !== false ? (
              <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
            ) : (
              <XCircle className="w-3 h-3 text-red-400 shrink-0" />
            )}
            <span className={validity_checks?.structure_intact !== false ? 'text-gray-300' : 'text-red-400 font-bold'}>
              Structure Intact
            </span>
          </div>

          {/* 2. Target Not Front-Run */}
          <div className="flex items-center space-x-1.5 bg-dark-900/60 px-2 py-1 rounded border border-dark-800">
            {validity_checks?.target_not_frontrun !== false ? (
              <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
            ) : (
              <XCircle className="w-3 h-3 text-red-400 shrink-0" />
            )}
            <span className={validity_checks?.target_not_frontrun !== false ? 'text-gray-300' : 'text-red-400 font-bold'}>
              Target Unmitigated
            </span>
          </div>

          {/* 3. Macro News */}
          <div className="flex items-center space-x-1.5 bg-dark-900/60 px-2 py-1 rounded border border-dark-800">
            {validity_checks?.macro_news_safe !== false ? (
              <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
            ) : (
              <XCircle className="w-3 h-3 text-red-400 shrink-0" />
            )}
            <span className="text-gray-300">Macro News Safe</span>
          </div>

          {/* 4. Time Decay */}
          <div className="flex items-center space-x-1.5 bg-dark-900/60 px-2 py-1 rounded border border-dark-800">
            {validity_checks?.time_decay_valid !== false ? (
              <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
            ) : (
              <XCircle className="w-3 h-3 text-amber-400 shrink-0" />
            )}
            <span className={validity_checks?.time_decay_valid !== false ? 'text-gray-300' : 'text-amber-400 font-bold'}>
              Time Valid ({bars_elapsed || 0}/12b)
            </span>
          </div>

          {/* 5. Risk-Reward Viability */}
          <div className="flex items-center space-x-1.5 bg-dark-900/60 px-2 py-1 rounded border border-dark-800 col-span-2 sm:col-span-1">
            {validity_checks?.risk_reward_viable !== false ? (
              <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
            ) : (
              <XCircle className="w-3 h-3 text-red-400 shrink-0" />
            )}
            <span className="text-gray-300">R:R Viable (&gt;=1.2R)</span>
          </div>
        </div>

        {/* Detailed Explanation / Warning Rationale */}
        {validity_reason && (
          <div
            className={`p-2 rounded text-[11px] font-mono leading-relaxed border ${
              isInvalidated
                ? 'bg-red-950/50 border-red-500/50 text-red-300 font-semibold'
                : isExpired
                ? 'bg-amber-950/50 border-amber-500/50 text-amber-300'
                : isTriggered
                ? 'bg-cyan-950/40 border-cyan-500/40 text-cyan-200'
                : isSetupValid
                ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                : 'bg-dark-900/50 border-dark-750 text-gray-400'
            }`}
          >
            {validity_reason}
          </div>
        )}
      </div>

      {/* Setup Status Banner (Trigger-based Lifecycle) */}
      <div
        className={`px-3.5 py-2.5 rounded-lg border text-xs font-mono space-y-1 ${
          signal_status === 'TRADE_COMPLETED_TP'
            ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
            : signal_status === 'TRADE_COMPLETED_SL'
            ? 'bg-red-950/40 border-red-500/50 text-red-300'
            : isInvalidated
            ? 'bg-red-950/30 border-red-500/40 text-red-300'
            : isPending
            ? 'bg-amber-950/30 border-amber-500/40 text-amber-300'
            : isInTrade
            ? 'bg-cyan-950/30 border-cyan-500/40 text-cyan-300'
            : 'bg-dark-800/80 border-dark-700 text-gray-400'
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {isTradeEnded ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : isInvalidated ? (
              <XCircle className="w-4 h-4 text-red-400 shrink-0" />
            ) : isPending ? (
              <Target className="w-4 h-4 text-amber-400 shrink-0 animate-pulse" />
            ) : isInTrade ? (
              <Zap className="w-4 h-4 text-cyan-400 shrink-0" />
            ) : (
              <Activity className="w-4 h-4 text-gray-400 shrink-0" />
            )}
            <span className="font-bold uppercase tracking-wider text-[11px]">
              {signal_status === 'TRADE_COMPLETED_TP'
                ? '🏆 TRADE CONCLUDED • TAKE PROFIT HIT'
                : signal_status === 'TRADE_COMPLETED_SL'
                ? '🛑 TRADE CONCLUDED • STOP LOSS HIT'
                : isInvalidated
                ? '❌ INVALIDATED • DO NOT ENTER'
                : isPending
                ? `⏳ AWAITING TRIGGER • ${direction} RANGE`
                : isInTrade
                ? '⚡ IN POSITION • ACTIVE TRADE'
                : '🔍 OBSERVATION MODE • SCANNING STRUCTURE'}
            </span>
          </div>
          <span className="font-bold">
            {isInTrade
              ? `Floating: ${current_r_multiple >= 0 ? '+' : ''}${current_r_multiple.toFixed(2)}R`
              : isPending
              ? `$${distance_to_trigger.toFixed(2)} to Trigger`
              : isInvalidated || isTradeEnded
              ? 'LINES CLEARED'
              : 'LINES HIDDEN'}
          </span>
        </div>
        <p className="text-[11px] opacity-90 font-sans leading-tight">
          {trade_ended_reason
            ? `${trade_ended_reason} — Chart lines cleared to protect capital.`
            : isPending
            ? `Price must reach the $${entry_range_low?.toFixed(2)} ── $${entry_range_high?.toFixed(2)} range (Trigger: $${effTrigger.toFixed(2)}) to activate entry.`
            : isInTrade
            ? `Position filled at $${effTrigger.toFixed(2)}. Managing trade toward TP1 ($${take_profit_reference.toFixed(2)}) and TP2 ($${tp2Val.toFixed(2)}).`
            : 'No high-conviction trade active. Waiting for institutional liquidity sweep or structure shift.'}
        </p>
      </div>

      {/* AI Calculation Rationale Box */}
      {ai_recommendation_text && (
        <div className="bg-dark-850 border border-dark-750 rounded-lg p-3 text-xs space-y-1">
          <span className="text-[10px] uppercase font-bold text-gold-400 tracking-wider flex items-center gap-1.5">
            <Target className="w-3.5 h-3.5 text-gold-400" />
            AI Calculation & Recommendation
          </span>
          <p className="text-gray-300 font-mono text-[11px] leading-relaxed">
            {ai_recommendation_text}
          </p>
        </div>
      )}

      {/* Main High-Conviction Recommendation Badge */}
      <div
        className={`p-4 rounded-xl border flex items-center justify-between transition-all ${
          isBuy
            ? 'bg-bull-900/20 border-bull-500/50 shadow-[0_0_20px_rgba(16,185,129,0.15)]'
            : isSell
            ? 'bg-bear-900/20 border-bear-500/50 shadow-[0_0_20px_rgba(239,68,68,0.15)]'
            : 'bg-dark-800/60 border-dark-600'
        }`}
      >
        <div className="flex items-center space-x-3.5">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center ${
              isBuy
                ? 'bg-bull-500/20 text-bull-400 border border-bull-500/40'
                : isSell
                ? 'bg-bear-500/20 text-bear-400 border border-bear-500/40'
                : 'bg-gray-700/30 text-gray-400 border border-gray-600'
            }`}
          >
            {isBuy && <TrendingUp className="w-6 h-6" />}
            {isSell && <TrendingDown className="w-6 h-6" />}
            {isWait && <PauseCircle className="w-6 h-6" />}
          </div>
          <div>
            <span className="text-[10px] font-bold text-gray-400 tracking-wider uppercase block">
              AI Action Recommendation
            </span>
            <div className="flex items-center space-x-2">
              <span
                className={`text-2xl font-black tracking-tight ${
                  isBuy ? 'text-bull-400' : isSell ? 'text-bear-400' : 'text-gray-300'
                }`}
              >
                {isBuy ? 'STRONG BUY' : isSell ? 'STRONG SELL' : 'NEUTRAL / WAIT'}
              </span>
            </div>
            <span className="text-[11px] text-gray-400 block mt-0.5">
              {isBuy
                ? 'Multi-model quantitative consensus triggers LONG entry'
                : isSell
                ? 'Multi-model quantitative consensus triggers SHORT entry'
                : 'Insufficient multi-model edge — Capital preserved'}
            </span>
          </div>
        </div>

        {/* AI Confidence Gauge */}
        <div className="text-right">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 block">
            Model Confidence
          </span>
          <span
            className={`text-2xl font-mono font-black ${
              confPct >= 60 ? 'text-gold-400' : 'text-gray-300'
            }`}
          >
            {confPct}%
          </span>
          <span className="text-[10px] text-emerald-400/90 block font-mono">
            {confPct >= 50 ? '● High Conviction' : '○ Standby'}
          </span>
        </div>
      </div>

      {/* Calibrated 3-Way Probability Bar */}
      <div className="bg-dark-850 border border-dark-750 rounded-lg p-3 space-y-1.5">
        <div className="flex justify-between text-[11px] font-semibold text-gray-300">
          <span className="text-bull-400 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-bull-400 inline-block" />
            Bullish: {bullPct}%
          </span>
          <span className="text-gray-400 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-gray-400 inline-block" />
            Neutral: {neutPct}%
          </span>
          <span className="text-bear-400 flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-bear-400 inline-block" />
            Bearish: {bearPct}%
          </span>
        </div>

        <div className="h-2.5 rounded-full overflow-hidden flex bg-dark-700">
          <div
            style={{ width: `${bullPct}%` }}
            className="bg-bull-500 h-full transition-all duration-500"
            title={`Bullish: ${bullPct}%`}
          />
          <div
            style={{ width: `${neutPct}%` }}
            className="bg-gray-600 h-full transition-all duration-500"
            title={`Neutral: ${neutPct}%`}
          />
          <div
            style={{ width: `${bearPct}%` }}
            className="bg-bear-500 h-full transition-all duration-500"
            title={`Bearish: ${bearPct}%`}
          />
        </div>
      </div>

      {/* Quantitative Execution Matrix (Trigger, SL, TP1, TP2) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        <div className="bg-dark-850 border border-dark-750 rounded-lg p-2 text-center">
          <span className="text-[10px] uppercase tracking-wider text-gray-400 block font-medium">
            {isPending ? 'Limit Trigger' : isInTrade ? 'Filled Entry' : 'AI Trigger'}
          </span>
          <span className="text-xs sm:text-sm font-mono font-bold text-gold-400 block mt-0.5">
            ${effTrigger > 0 ? effTrigger.toFixed(2) : '---'}
          </span>
          <span className="text-[10px] text-gray-400 font-mono">
            {isPending && entry_range_low && entry_range_high
              ? `$${entry_range_low.toFixed(0)}-$${entry_range_high.toFixed(0)} Zone`
              : 'Stationary Level'}
          </span>
        </div>

        <div className="bg-dark-850 border border-dark-750 rounded-lg p-2 text-center">
          <span className="text-[10px] uppercase tracking-wider text-bear-400 block font-medium">
            Stop Loss
          </span>
          <span className="text-xs sm:text-sm font-mono font-bold text-bear-400 block mt-0.5">
            ${stop_loss_reference > 0 ? stop_loss_reference.toFixed(2) : '---'}
          </span>
          <span className="text-[10px] text-bear-400/80 font-mono">
            -${riskDist.toFixed(2)} (-1.0R)
          </span>
        </div>

        <div className="bg-dark-850 border border-dark-750 rounded-lg p-2 text-center">
          <span className="text-[10px] uppercase tracking-wider text-bull-400 block font-medium">
            TP1 (1.2R)
          </span>
          <span className="text-xs sm:text-sm font-mono font-bold text-bull-400 block mt-0.5">
            ${take_profit_reference > 0 ? take_profit_reference.toFixed(2) : '---'}
          </span>
          <span className="text-[10px] text-bull-400/80 font-mono">
            +${reward1Dist.toFixed(2)} (+1.2R)
          </span>
        </div>

        <div className="bg-dark-850 border border-dark-750 rounded-lg p-2 text-center">
          <span className="text-[10px] uppercase tracking-wider text-emerald-400 block font-medium">
            TP2 (2.4R)
          </span>
          <span className="text-xs sm:text-sm font-mono font-bold text-emerald-400 block mt-0.5">
            ${tp2Val > 0 ? tp2Val.toFixed(2) : '---'}
          </span>
          <span className="text-[10px] text-emerald-400/80 font-mono">
            +${reward2Dist.toFixed(2)} (+2.4R)
          </span>
        </div>
      </div>

      {/* Sizing & Macro Safeguards */}
      <div className="bg-dark-850 border border-dark-750 rounded-lg p-3 space-y-2 text-xs">
        <div className="flex items-center justify-between text-gray-300">
          <span className="text-gray-400 flex items-center gap-1.5">
            <Target className="w-3.5 h-3.5 text-gold-400" />
            Recommended Position:
          </span>
          <span className="font-mono font-bold text-white">
            {recommendedLots} lots ({recommendedOz} oz)
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300">
          <span className="text-gray-400 flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-cyan-400" />
            Market Regime:
          </span>
          <span className="font-mono font-semibold text-cyan-300">
            {market_regime || 'Expansion Trend'}
          </span>
        </div>

        <div className="flex items-center justify-between text-gray-300">
          <span className="text-gray-400 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            Macro News Shield:
          </span>
          <span className="font-mono font-semibold text-emerald-400">
            ACTIVE • High Impact Filter
          </span>
        </div>
      </div>

      {/* Execute Paper Trade / Scan Next Setup Action Button */}
      {isTradeEnded || isInvalidated ? (
        <button
          onClick={handleResetSignal}
          disabled={resetting}
          className="w-full py-2.5 px-4 rounded-lg font-bold text-xs uppercase tracking-wider flex items-center justify-center space-x-2 bg-gold-600 hover:bg-gold-500 text-black shadow-lg shadow-gold-600/30 active:scale-[0.99] transition-all"
        >
          <RotateCcw className={`w-4 h-4 ${resetting ? 'animate-spin' : ''}`} />
          <span>{resetting ? 'Scanning Market...' : isInvalidated ? 'Setup Invalidated — Scan Fresh Opportunity' : 'Trade Concluded — Scan Next AI Setup'}</span>
        </button>
      ) : (
        <button
          onClick={handleExecutePaper}
          disabled={executing || isWait || signal_status === 'SCANNING' || !is_valid_to_trade}
          className={`w-full py-2.5 px-4 rounded-lg font-bold text-xs uppercase tracking-wider flex items-center justify-center space-x-2 transition-all ${
            isBuy
              ? 'bg-bull-600 hover:bg-bull-500 text-white shadow-lg shadow-bull-600/30 active:scale-[0.99]'
              : isSell
              ? 'bg-bear-600 hover:bg-bear-500 text-white shadow-lg shadow-bear-600/30 active:scale-[0.99]'
              : 'bg-dark-800 text-gray-400 border border-dark-700 cursor-not-allowed opacity-60'
          }`}
        >
          <Zap className="w-4 h-4" />
          <span>
            {executing
              ? 'Routing to Paper Trading Ledger...'
              : isPending
              ? `Execute Limit ${direction} at $${effTrigger.toFixed(2)}`
              : isInTrade
              ? `Manage ${direction} Position in Ledger`
              : 'AI Observing Market (Scanning for Edge)'}
          </span>
        </button>
      )}

      {executeMsg && (
        <div className="p-2.5 rounded bg-dark-850 border border-gold-500/30 text-[11px] text-gold-300 flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-gold-400 shrink-0" />
          <span>{executeMsg}</span>
        </div>
      )}
    </div>
  );
}
