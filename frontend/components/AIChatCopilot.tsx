'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  LLMModelItem,
  ChatMessageItem,
  ChartContextPayload,
  fetchChatModels,
  fetchProviderStatus,
  sendChatMessage,
  PredictionResponse,
  MarketFeaturesResponse,
  Candle,
} from '../lib/api';
import {
  Bot,
  Brain,
  Check,
  ChevronDown,
  ChevronRight,
  Copy,
  Cpu,
  Key,
  Layers,
  Maximize2,
  Minimize2,
  RefreshCw,
  Send,
  Shield,
  Sparkles,
  Target,
  Trash2,
  TrendingDown,
  TrendingUp,
  User,
  Zap,
  Info,
  X,
} from 'lucide-react';

interface AIChatCopilotProps {
  currentPrice?: number;
  timeframe?: string;
  prediction?: PredictionResponse | null;
  features?: MarketFeaturesResponse | null;
  candles?: Candle[];
  compact?: boolean;
}

const QUICK_PROMPTS = [
  { label: '📊 Analyze Current Chart', prompt: 'Analyze current Gold (XAUUSD) market structure, liquidity sweeps, order blocks, and provide an institutional execution plan with entry range, SL, and TP targets.' },
  { label: '🎯 Validate Current AI Setup', prompt: 'Is the current AI trade setup valid or invalidated? Verify front-run targets, swing structure integrity, and risk-reward payoff.' },
  { label: '🛡️ Calculate Lot Size (1% Risk)', prompt: 'Calculate exact lot size and dollar risk for a $10,000 account risking 1.0% ($100 budget) based on current stop loss distance.' },
  { label: '🔍 Multi-Timeframe Confluence', prompt: 'Assess multi-timeframe alignment across M5, M15, and H1 for Gold. Are higher timeframe institutional levels aligning with current intraday momentum?' },
];

export default function AIChatCopilot({
  currentPrice = 3565.4,
  timeframe = 'M5',
  prediction,
  features,
  candles = [],
  compact = false,
}: AIChatCopilotProps) {
  const [models, setModels] = useState<LLMModelItem[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('deepseek-reasoner');
  const [selectedProvider, setSelectedProvider] = useState<string>('deepseek');
  const [includeChartContext, setIncludeChartContext] = useState<boolean>(true);

  const [messages, setMessages] = useState<ChatMessageItem[]>([
    {
      role: 'assistant',
      content:
        '👋 Welcome to the **Institutional AI Trading Copilot**.\n\n' +
        'I can analyze live **Gold (XAUUSD)** price action, evaluate Smart Money Concepts (SMC), and formulate precise execution plans using **DeepSeek (R1/V3)**, **ChatGPT (GPT-4o)**, **Claude (3.5 Sonnet)**, or **Google Gemini**.\n\n' +
        '👉 Click **"📊 Analyze Current Chart"** below or ask me any question regarding your trading setup!',
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);

  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [expandedReasoning, setExpandedReasoning] = useState<Record<number, boolean>>({});

  // API Keys state
  const [apiKeys, setApiKeys] = useState<{
    deepseek: string;
    openai: string;
    anthropic: string;
    gemini: string;
  }>({
    deepseek: '',
    openai: '',
    anthropic: '',
    gemini: '',
  });

  const [serverKeyStatus, setServerKeyStatus] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load API keys from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setApiKeys({
        deepseek: localStorage.getItem('llm_key_deepseek') || '',
        openai: localStorage.getItem('llm_key_openai') || '',
        anthropic: localStorage.getItem('llm_key_anthropic') || '',
        gemini: localStorage.getItem('llm_key_gemini') || '',
      });
    }
  }, []);

  // Fetch model catalog & server status
  useEffect(() => {
    const init = async () => {
      try {
        const [mList, sStatus] = await Promise.all([
          fetchChatModels().catch(() => []),
          fetchProviderStatus().catch(() => ({})),
        ]);
        if (mList.length > 0) {
          setModels(mList);
          const def = mList.find((m) => m.is_default) || mList[0];
          setSelectedModelId(def.id);
          setSelectedProvider(def.provider);
        }
        setServerKeyStatus(sStatus);
      } catch (err) {
        console.error('Error fetching chat models:', err);
      }
    };
    init();
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleModelChange = (modelId: string) => {
    setSelectedModelId(modelId);
    const m = models.find((x) => x.id === modelId);
    if (m) {
      setSelectedProvider(m.provider);
    }
  };

  const handleSaveKey = (provider: 'deepseek' | 'openai' | 'anthropic' | 'gemini', val: string) => {
    const updated = { ...apiKeys, [provider]: val.trim() };
    setApiKeys(updated);
    if (typeof window !== 'undefined') {
      localStorage.setItem(`llm_key_${provider}`, val.trim());
    }
  };

  const currentSelectedModel = models.find((m) => m.id === selectedModelId);
  const activeKey = apiKeys[selectedProvider as keyof typeof apiKeys] || '';
  const hasServerKey = serverKeyStatus[selectedProvider] || selectedProvider === 'local';

  // Build live chart context payload
  const buildChartContext = (): ChartContextPayload => {
    const effPrice = currentPrice || prediction?.trigger_price || 3565.0;
    const latestC = candles.length > 0 ? candles[candles.length - 1] : null;

    return {
      instrument: 'XAUUSD',
      timeframe: timeframe,
      current_price: effPrice,
      candle_high: latestC?.high || effPrice,
      candle_low: latestC?.low || effPrice,
      atr_14: 3.85,
      rsi_14: 46.5,
      market_regime: prediction?.market_regime || 'Ranging',
      signal_status: prediction?.signal_status || 'PENDING_TRIGGER',
      direction: prediction?.direction || 'SELL',
      confidence: prediction?.confidence || 0.62,
      entry_range_low: prediction?.entry_range_low,
      entry_range_high: prediction?.entry_range_high,
      trigger_price: prediction?.trigger_price || prediction?.entry_reference_price,
      stop_loss: prediction?.stop_loss_reference,
      take_profit_1: prediction?.take_profit_reference,
      take_profit_2: prediction?.take_profit_2,
      setup_validity: prediction?.setup_validity || 'VALID',
      validity_reason: prediction?.validity_reason || 'Structure intact, targets unmitigated.',
      recent_candles: candles.slice(-5),
      macro_news_safe: true,
    };
  };

  const handleSend = async (textToSend?: string) => {
    const q = (textToSend || inputValue).trim();
    if (!q || loading) return;

    const userMsg: ChatMessageItem = {
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString(),
    };

    const newThread = [...messages, userMsg];
    setMessages(newThread);
    setInputValue('');
    setLoading(true);

    try {
      const chartCtx = includeChartContext ? buildChartContext() : undefined;

      const res = await sendChatMessage({
        messages: newThread.map((m) => ({ role: m.role, content: m.content })),
        provider: selectedProvider,
        model: selectedModelId,
        api_key: activeKey || undefined,
        chart_context: chartCtx,
        temperature: 0.4,
      });

      const assistantMsg: ChatMessageItem = {
        role: 'assistant',
        content: res.content,
        reasoning_content: res.reasoning_content || undefined,
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages((prev) => [...prev, assistantMsg]);

      // If response has reasoning, open it by default
      if (res.reasoning_content) {
        setExpandedReasoning((prev) => ({ ...prev, [newThread.length]: true }));
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ **Error communicating with ${selectedProvider.toUpperCase()}:** ${
            err.message || 'Unknown network error'
          }\n\nPlease verify your API key in **API Key Settings** or switch to the **Local Model E** engine.`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: '💬 Chat history cleared. Ready for your next query or chart analysis!',
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
  };

  return (
    <div
      className={`flex flex-col bg-dark-900 border border-dark-700 rounded-xl shadow-2xl overflow-hidden ${
        compact ? 'h-[640px]' : 'h-[calc(100vh-175px)] min-h-[640px]'
      }`}
    >
      {/* ========================================================================= */}
      {/* Header Bar */}
      {/* ========================================================================= */}
      <div className="bg-dark-850 border-b border-dark-700 px-4 py-2.5 flex items-center justify-between shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-gold-500/10 border border-gold-500/30 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-gold-400" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-xs font-bold uppercase tracking-wider text-white">
                AI Market Copilot
              </h2>
              <span className="text-[10px] bg-gold-500/15 border border-gold-500/30 text-gold-400 font-mono px-1.5 py-0.2 rounded font-semibold">
                Multi-LLM
              </span>
            </div>
            <span className="text-[10px] text-gray-400 block font-mono">
              DeepSeek • ChatGPT • Claude • Gemini • Model E
            </span>
          </div>
        </div>

        {/* Model Selector & API Key Settings */}
        <div className="flex items-center space-x-2">
          {/* Model Dropdown */}
          <div className="relative">
            <select
              value={selectedModelId}
              onChange={(e) => handleModelChange(e.target.value)}
              className="bg-dark-800 border border-dark-700 text-gold-300 text-xs rounded-lg px-2.5 py-1.5 font-mono font-semibold focus:outline-none focus:border-gold-500 cursor-pointer pr-7"
            >
              <optgroup label="🐋 DeepSeek (Flagship Reasoning & Chat)">
                <option value="deepseek-reasoner">DeepSeek R1 (Reasoner Chain-of-Thought)</option>
                <option value="deepseek-chat">DeepSeek V3 (Chat Fast)</option>
              </optgroup>
              <optgroup label="🟢 OpenAI (ChatGPT)">
                <option value="gpt-4o">ChatGPT (GPT-4o Multimodal)</option>
                <option value="gpt-4o-mini">ChatGPT (GPT-4o Mini)</option>
                <option value="o3-mini">OpenAI o3-mini (STEM Reasoning)</option>
              </optgroup>
              <optgroup label="🟣 Anthropic (Claude)">
                <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet (Institutional Precision)</option>
                <option value="claude-3-5-haiku-20241022">Claude 3.5 Haiku (Fast)</option>
              </optgroup>
              <optgroup label="🔵 Google Gemini">
                <option value="gemini-1.5-flash">Gemini 1.5 Flash (Fast Free Tier)</option>
                <option value="gemini-1.5-pro">Gemini 1.5 Pro (2M Context)</option>
                <option value="gemini-2.0-flash">Gemini 2.0 Flash (Next-Gen)</option>
              </optgroup>
              <optgroup label="⚡ Built-in Local Engine (Zero Key Needed)">
                <option value="local-model-e">Local Institutional Model E (Free / Built-in)</option>
              </optgroup>
            </select>
          </div>

          {/* API Key Modal Button */}
          <button
            onClick={() => setShowKeyModal(true)}
            title="Configure API Keys for DeepSeek, OpenAI, Claude, Gemini"
            className={`flex items-center space-x-1.5 px-2.5 py-1.5 text-xs rounded-lg border transition font-mono ${
              activeKey || hasServerKey
                ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300 hover:bg-emerald-900/50'
                : 'bg-amber-950/40 border-amber-500/40 text-amber-300 hover:bg-amber-900/50 animate-pulse'
            }`}
          >
            <Key className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">
              {activeKey || hasServerKey ? 'Key Connected' : 'Add API Key'}
            </span>
          </button>

          {/* Clear Chat */}
          <button
            onClick={clearChat}
            title="Clear Chat History"
            className="p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-dark-750 transition"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* Live Chart Status & One-Click Analysis Banner */}
      {/* ========================================================================= */}
      <div className="bg-dark-850/60 border-b border-dark-750 px-4 py-2 flex flex-wrap items-center justify-between gap-2 shrink-0">
        <div className="flex items-center space-x-2 text-xs font-mono">
          <span className="flex items-center space-x-1.5 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-bold">LIVE SNAPSHOT:</span>
          </span>
          <span className="text-white font-bold">${currentPrice > 0 ? currentPrice.toFixed(2) : '3,565.40'}</span>
          <span className="text-gray-400">({timeframe})</span>
          {prediction && (
            <span
              className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${
                prediction.direction === 'BUY'
                  ? 'bg-bull-500/20 text-bull-400'
                  : prediction.direction === 'SELL'
                  ? 'bg-bear-500/20 text-bear-400'
                  : 'bg-gray-700/30 text-gray-400'
              }`}
            >
              {prediction.direction} ({Math.round((prediction.confidence || 0) * 100)}%)
            </span>
          )}
          {prediction?.setup_validity && (
            <span
              className={`text-[10px] px-1.5 py-0.2 rounded font-bold ${
                prediction.setup_validity === 'VALID'
                  ? 'bg-emerald-950 border border-emerald-500/40 text-emerald-300'
                  : prediction.setup_validity === 'TRIGGERED'
                  ? 'bg-cyan-950 border border-cyan-500/40 text-cyan-300'
                  : 'bg-red-950 border border-red-500/40 text-red-300'
              }`}
            >
              {prediction.setup_validity}
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <label className="flex items-center space-x-1.5 text-[11px] text-gray-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={includeChartContext}
              onChange={(e) => setIncludeChartContext(e.target.checked)}
              className="rounded bg-dark-750 border-dark-600 text-gold-500 focus:ring-0"
            />
            <span>Feed Live Chart Context</span>
          </label>

          {/* Prominent Action Button: Analyze Chart Now */}
          <button
            onClick={() => handleSend('Analyze current Gold (XAUUSD) market structure, order blocks, fair value gaps, and recommend an execution plan with entry, invalidation, and profit targets.')}
            disabled={loading}
            className="flex items-center space-x-1.5 bg-gradient-to-r from-gold-600 to-gold-500 hover:from-gold-500 hover:to-gold-400 text-black px-3 py-1 rounded-lg text-xs font-bold font-mono shadow-md shadow-gold-500/20 active:scale-95 transition"
          >
            <Zap className="w-3.5 h-3.5 fill-black" />
            <span>📊 Analyze Current Chart</span>
          </button>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* Quick Prompt Chips */}
      {/* ========================================================================= */}
      <div className="bg-dark-900/90 border-b border-dark-800 px-4 py-2 flex items-center space-x-2 overflow-x-auto shrink-0">
        <span className="text-[10px] uppercase font-bold text-gray-400 font-mono tracking-wider shrink-0 flex items-center gap-1">
          <Target className="w-3 h-3 text-gold-400" />
          Quick Actions:
        </span>
        {QUICK_PROMPTS.map((qp, i) => (
          <button
            key={i}
            onClick={() => handleSend(qp.prompt)}
            disabled={loading}
            className="text-[11px] bg-dark-800 hover:bg-dark-750 text-gray-300 hover:text-white px-2.5 py-1 rounded-full border border-dark-700 whitespace-nowrap transition active:scale-95 shrink-0"
          >
            {qp.label}
          </button>
        ))}
      </div>

      {/* ========================================================================= */}
      {/* Chat Thread Messages */}
      {/* ========================================================================= */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => {
          const isUser = msg.role === 'user';
          return (
            <div
              key={idx}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} space-y-1.5`}
            >
              <div className="flex items-center space-x-2 text-[10px] text-gray-400 px-1 font-mono">
                {isUser ? (
                  <>
                    <span>You</span>
                    <span>•</span>
                    <span>{msg.timestamp}</span>
                    <div className="w-5 h-5 rounded-full bg-gold-500/20 text-gold-400 flex items-center justify-center">
                      <User className="w-3 h-3" />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
                      <Bot className="w-3 h-3" />
                    </div>
                    <span className="font-bold text-gray-300">
                      {currentSelectedModel?.name || 'AI Copilot'}
                    </span>
                    <span>•</span>
                    <span>{msg.timestamp}</span>
                  </>
                )}
              </div>

              {/* Message Content Box */}
              <div
                className={`max-w-[90%] sm:max-w-[85%] rounded-2xl p-4 text-xs leading-relaxed border shadow-md relative group ${
                  isUser
                    ? 'bg-gold-600/10 border-gold-500/30 text-gold-100 rounded-tr-sm'
                    : 'bg-dark-850 border-dark-700 text-gray-200 rounded-tl-sm'
                }`}
              >
                {/* DeepSeek R1 Step-by-Step Reasoning (Chain-of-Thought) Accordion */}
                {!isUser && msg.reasoning_content && (
                  <div className="mb-3 rounded-lg border border-indigo-500/40 bg-indigo-950/20 overflow-hidden">
                    <button
                      onClick={() =>
                        setExpandedReasoning((prev) => ({
                          ...prev,
                          [idx]: !prev[idx],
                        }))
                      }
                      className="w-full px-3 py-2 flex items-center justify-between text-left text-[11px] font-mono font-bold text-indigo-300 hover:bg-indigo-900/30 transition"
                    >
                      <span className="flex items-center space-x-2">
                        <Brain className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
                        <span>
                          🧠 {currentSelectedModel?.has_reasoning ? `${currentSelectedModel.name} Thinking Process` : 'Model E Quantitative Reasoning Trace'} (Chain-of-Thought)
                        </span>
                      </span>
                      {expandedReasoning[idx] ? (
                        <ChevronDown className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5" />
                      )}
                    </button>

                    {expandedReasoning[idx] && (
                      <div className="p-3 bg-dark-900/80 border-t border-indigo-500/30 text-[11px] text-gray-300 font-mono whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
                        {msg.reasoning_content}
                      </div>
                    )}
                  </div>
                )}

                {/* Markdown text body */}
                <div className="whitespace-pre-wrap font-sans space-y-2">
                  {msg.content}
                </div>

                {/* Copy button */}
                <button
                  onClick={() => handleCopy(msg.content, idx)}
                  title="Copy to clipboard"
                  className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-white rounded bg-dark-800 border border-dark-700 transition"
                >
                  {copiedIndex === idx ? (
                    <Check className="w-3 h-3 text-emerald-400" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                </button>
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="flex items-start space-x-2">
            <div className="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0">
              <Bot className="w-3.5 h-3.5 animate-spin" />
            </div>
            <div className="bg-dark-850 border border-dark-700 rounded-2xl rounded-tl-sm p-4 text-xs text-gray-400 flex items-center space-x-2">
              <RefreshCw className="w-4 h-4 animate-spin text-gold-400" />
              <span className="font-mono">
                {selectedModelId.includes('reasoner')
                  ? 'DeepSeek R1 is computing step-by-step reasoning & SMC math...'
                  : `${currentSelectedModel?.name || 'AI'} is formulating institutional market analysis...`}
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ========================================================================= */}
      {/* Input Bar */}
      {/* ========================================================================= */}
      <div className="bg-dark-850 border-t border-dark-700 p-3 shrink-0">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex items-center space-x-2"
        >
          <div className="relative flex-1">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={`Ask ${currentSelectedModel?.name || 'AI Copilot'} about Gold structure, levels, or risk...`}
              disabled={loading}
              className="w-full bg-dark-900 border border-dark-700 rounded-xl px-4 py-2.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-gold-500 font-sans transition"
            />
          </div>

          <button
            type="submit"
            disabled={loading || !inputValue.trim()}
            className={`p-2.5 rounded-xl font-bold transition flex items-center justify-center ${
              inputValue.trim() && !loading
                ? 'bg-gold-500 hover:bg-gold-400 text-black shadow-lg shadow-gold-500/20 active:scale-95'
                : 'bg-dark-800 text-gray-500 border border-dark-700 cursor-not-allowed'
            }`}
          >
            <Send className="w-4 h-4" />
          </button>
        </form>

        <div className="flex items-center justify-between mt-2 px-1 text-[10px] text-gray-400 font-mono">
          <span>Active Provider: <strong className="text-gold-300">{selectedProvider.toUpperCase()}</strong> ({selectedModelId})</span>
          <span>Tip: Shift+Enter or click Send to submit</span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* API Key Settings Modal */}
      {/* ========================================================================= */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-dark-900 border border-dark-700 rounded-2xl w-full max-w-lg p-5 space-y-4 shadow-2xl relative">
            <div className="flex items-center justify-between border-b border-dark-700 pb-3">
              <div className="flex items-center space-x-2">
                <Key className="w-5 h-5 text-gold-400" />
                <h3 className="text-sm font-bold text-white uppercase tracking-wider font-mono">
                  LLM API Key Settings
                </h3>
              </div>
              <button
                onClick={() => setShowKeyModal(false)}
                className="text-gray-400 hover:text-white p-1 rounded-lg"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-gray-300 leading-relaxed">
              Enter your API keys below to enable direct streaming from DeepSeek, OpenAI, Claude, or Google Gemini. Keys are stored locally in your browser.
            </p>

            <div className="space-y-3">
              {/* DeepSeek */}
              <div>
                <label className="text-xs font-mono font-bold text-gray-300 flex items-center justify-between mb-1">
                  <span>🐋 DeepSeek API Key (R1 & V3)</span>
                  <a
                    href="https://platform.deepseek.com"
                    target="_blank"
                    rel="noreferrer"
                    className="text-[10px] text-gold-400 hover:underline"
                  >
                    platform.deepseek.com
                  </a>
                </label>
                <input
                  type="password"
                  value={apiKeys.deepseek}
                  onChange={(e) => handleSaveKey('deepseek', e.target.value)}
                  placeholder="sk-..."
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg px-3 py-2 text-xs font-mono text-white placeholder-gray-500 focus:outline-none focus:border-gold-500"
                />
              </div>

              {/* OpenAI */}
              <div>
                <label className="text-xs font-mono font-bold text-gray-300 flex items-center justify-between mb-1">
                  <span>🟢 OpenAI API Key (ChatGPT / GPT-4o)</span>
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noreferrer"
                    className="text-[10px] text-gold-400 hover:underline"
                  >
                    platform.openai.com
                  </a>
                </label>
                <input
                  type="password"
                  value={apiKeys.openai}
                  onChange={(e) => handleSaveKey('openai', e.target.value)}
                  placeholder="sk-proj-..."
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg px-3 py-2 text-xs font-mono text-white placeholder-gray-500 focus:outline-none focus:border-gold-500"
                />
              </div>

              {/* Anthropic Claude */}
              <div>
                <label className="text-xs font-mono font-bold text-gray-300 flex items-center justify-between mb-1">
                  <span>🟣 Anthropic API Key (Claude 3.5 Sonnet)</span>
                  <a
                    href="https://console.anthropic.com"
                    target="_blank"
                    rel="noreferrer"
                    className="text-[10px] text-gold-400 hover:underline"
                  >
                    console.anthropic.com
                  </a>
                </label>
                <input
                  type="password"
                  value={apiKeys.anthropic}
                  onChange={(e) => handleSaveKey('anthropic', e.target.value)}
                  placeholder="sk-ant-..."
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg px-3 py-2 text-xs font-mono text-white placeholder-gray-500 focus:outline-none focus:border-gold-500"
                />
              </div>

              {/* Google Gemini */}
              <div>
                <label className="text-xs font-mono font-bold text-gray-300 flex items-center justify-between mb-1">
                  <span>🔵 Google Gemini API Key (Flash / Pro)</span>
                  <a
                    href="https://aistudio.google.com"
                    target="_blank"
                    rel="noreferrer"
                    className="text-[10px] text-gold-400 hover:underline"
                  >
                    aistudio.google.com (Free tier available)
                  </a>
                </label>
                <input
                  type="password"
                  value={apiKeys.gemini}
                  onChange={(e) => handleSaveKey('gemini', e.target.value)}
                  placeholder="AIzaSy..."
                  className="w-full bg-dark-800 border border-dark-700 rounded-lg px-3 py-2 text-xs font-mono text-white placeholder-gray-500 focus:outline-none focus:border-gold-500"
                />
              </div>
            </div>

            <div className="pt-2 border-t border-dark-700 flex items-center justify-between">
              <span className="text-[11px] text-gray-400 flex items-center gap-1">
                <Info className="w-3.5 h-3.5 text-gold-400" />
                No key? Select <strong>Local Model E</strong> for free offline analysis.
              </span>
              <button
                onClick={() => setShowKeyModal(false)}
                className="bg-gold-500 hover:bg-gold-400 text-black px-4 py-1.5 rounded-lg text-xs font-bold font-mono transition"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
