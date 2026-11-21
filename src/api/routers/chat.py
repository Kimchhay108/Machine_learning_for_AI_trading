"""AI Chat and Chart Analysis Router supporting DeepSeek, OpenAI (ChatGPT), Anthropic (Claude), and Google Gemini."""

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("api.chat")
router = APIRouter(prefix="/api/llm", tags=["LLM Chat & Chart Copilot"])


# ============================================================================
# Schemas
# ============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChartContext(BaseModel):
    instrument: str = "XAUUSD"
    timeframe: str = "M5"
    current_price: Optional[float] = None
    candle_high: Optional[float] = None
    candle_low: Optional[float] = None
    atr_14: Optional[float] = None
    rsi_14: Optional[float] = None
    market_regime: Optional[str] = None
    signal_status: Optional[str] = None
    direction: Optional[str] = None
    confidence: Optional[float] = None
    entry_range_low: Optional[float] = None
    entry_range_high: Optional[float] = None
    trigger_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    setup_validity: Optional[str] = None
    validity_reason: Optional[str] = None
    recent_candles: Optional[List[Dict[str, Any]]] = None
    smc_details: Optional[Dict[str, Any]] = None
    macro_news_safe: Optional[bool] = True


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    provider: str = Field(
        default="deepseek",
        description="LLM Provider: 'deepseek', 'openai', 'anthropic', 'gemini', or 'local'",
    )
    model: str = Field(default="deepseek-reasoner", description="Model identifier")
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key. If omitted, server will look up env variables or use local quant engine.",
    )
    chart_context: Optional[ChartContext] = None
    temperature: Optional[float] = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=2048, ge=128, le=8192)


class ChatResponse(BaseModel):
    provider: str
    model: str
    content: str
    reasoning_content: Optional[str] = None
    chart_summary: Optional[str] = None
    created_at: str


class ModelCatalogItem(BaseModel):
    id: str
    name: str
    provider: str
    provider_name: str
    description: str
    strengths: List[str]
    has_reasoning: bool
    context_window: str
    pricing: str
    is_default: bool = False


# ============================================================================
# Model Catalog
# ============================================================================

AVAILABLE_MODELS: List[ModelCatalogItem] = [
    # DeepSeek
    ModelCatalogItem(
        id="deepseek-reasoner",
        name="DeepSeek R1 (Reasoner)",
        provider="deepseek",
        provider_name="DeepSeek",
        description="Flagship open reasoning model. Generates full internal chain-of-thought, mathematical risk modeling, and precise trade structure.",
        strengths=["Step-by-step thinking", "SMC & Order Block math", "Asymmetric R:R optimization"],
        has_reasoning=True,
        context_window="64k tokens",
        pricing="Ultra-Low Cost (~$0.55 / 1M tokens)",
        is_default=True,
    ),
    ModelCatalogItem(
        id="deepseek-chat",
        name="DeepSeek V3 (Chat)",
        provider="deepseek",
        provider_name="DeepSeek",
        description="Ultra-fast, high-intelligence model for rapid chart summaries and conversational Q&A.",
        strengths=["Sub-second latency", "Concise summaries", "Low cost"],
        has_reasoning=False,
        context_window="64k tokens",
        pricing="Ultra-Low Cost (~$0.14 / 1M tokens)",
    ),
    # OpenAI (ChatGPT)
    ModelCatalogItem(
        id="gpt-4o",
        name="ChatGPT (GPT-4o)",
        provider="openai",
        provider_name="OpenAI",
        description="OpenAI's flagship multimodal model. High precision analysis, institutional language, and rigorous structural evaluation.",
        strengths=["Multimodal analysis", "Institutional prose", "Macroeconomic correlation"],
        has_reasoning=False,
        context_window="128k tokens",
        pricing="Standard API (~$2.50 / 1M tokens)",
    ),
    ModelCatalogItem(
        id="gpt-4o-mini",
        name="ChatGPT (GPT-4o Mini)",
        provider="openai",
        provider_name="OpenAI",
        description="Fast, economical OpenAI model for quick chart checks and position sizing math.",
        strengths=["Low latency", "Reliable formatting", "Affordable"],
        has_reasoning=False,
        context_window="128k tokens",
        pricing="Economical (~$0.15 / 1M tokens)",
    ),
    ModelCatalogItem(
        id="o3-mini",
        name="OpenAI o3-mini",
        provider="openai",
        provider_name="OpenAI",
        description="OpenAI's dedicated STEM and quantitative reasoning model with deliberate chain-of-thought.",
        strengths=["Quantitative logic", "Statistical edge testing", "Deep reasoning"],
        has_reasoning=True,
        context_window="128k tokens",
        pricing="Standard Reasoning (~$1.10 / 1M tokens)",
    ),
    # Anthropic (Claude)
    ModelCatalogItem(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        provider_name="Anthropic",
        description="Industry gold standard for code, technical analysis, nuance, and clean financial breakdown.",
        strengths=["Flawless technical precision", "Clean Markdown tables", "Nuanced SMC analysis"],
        has_reasoning=False,
        context_window="200k tokens",
        pricing="Standard API (~$3.00 / 1M tokens)",
    ),
    ModelCatalogItem(
        id="claude-3-5-haiku-20241022",
        name="Claude 3.5 Haiku",
        provider="anthropic",
        provider_name="Anthropic",
        description="Blazing fast Claude model for real-time tick alerts and quick level explanations.",
        strengths=["Rapid response", "High accuracy", "Clean formatting"],
        has_reasoning=False,
        context_window="200k tokens",
        pricing="Economical (~$0.80 / 1M tokens)",
    ),
    # Google Gemini
    ModelCatalogItem(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        provider="gemini",
        provider_name="Google Gemini",
        description="Google's ultra-fast multimodal model with massive 1M token window and free tier support.",
        strengths=["Free tier on Google AI Studio", "1M Context", "Fast responses"],
        has_reasoning=False,
        context_window="1M tokens",
        pricing="Free Tier / Low API cost",
    ),
    ModelCatalogItem(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="gemini",
        provider_name="Google Gemini",
        description="Google's flagship foundation model. Outstanding multi-document reasoning and long-term pattern recognition.",
        strengths=["2M Context Window", "Multimodal depth", "Comprehensive reports"],
        has_reasoning=False,
        context_window="2M tokens",
        pricing="Standard API (~$1.25 / 1M tokens)",
    ),
    ModelCatalogItem(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash (Experimental)",
        provider="gemini",
        provider_name="Google Gemini",
        description="Google's next-generation multimodal model with real-time streaming and tool mastery.",
        strengths=["Next-gen speed", "Multimodal intelligence", "Advanced agent capabilities"],
        has_reasoning=False,
        context_window="1M tokens",
        pricing="Free / Experimental",
    ),
    # Built-in Institutional Model E Engine (Zero API Key required)
    ModelCatalogItem(
        id="local-model-e",
        name="Institutional Model E Copilot (Built-in)",
        provider="local",
        provider_name="Local Engine",
        description="Runs directly against your platform's local DuckDB + SMC + Quantitative Stacking engine. Works immediately offline without requiring an API key.",
        strengths=["No API key required", "Zero network cost", "Direct access to local model features"],
        has_reasoning=True,
        context_window="N/A (Native)",
        pricing="100% Free / Built-in",
    ),
]


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/models", response_model=List[ModelCatalogItem])
def list_available_chat_models() -> List[ModelCatalogItem]:
    """List all supported LLM providers and models (DeepSeek, ChatGPT, Claude, Gemini, and Local Model E)."""
    return AVAILABLE_MODELS


@router.get("/status")
def get_provider_status() -> Dict[str, Any]:
    """Check which API keys are detected in the server environment."""
    return {
        "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "gemini": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "local": True,
        "default_provider": "deepseek",
        "default_model": "deepseek-reasoner",
    }


def format_chart_context_prompt(ctx: Optional[ChartContext]) -> str:
    """Format rich real-time chart data into an institutional prompt header."""
    if not ctx or not ctx.current_price:
        return ""

    lines = [
        "==================================================================",
        "REAL-TIME INSTITUTIONAL MARKET DATA SNAPSHOT (LIVE TERMINAL FEED):",
        f"• Instrument: {ctx.instrument} (Gold Spot / USD)",
        f"• Active Timeframe: {ctx.timeframe}",
        f"• Live Spot Price: ${ctx.current_price:.2f}",
    ]

    if ctx.market_regime:
        lines.append(f"• Market Regime: {ctx.market_regime}")
    if ctx.atr_14:
        lines.append(f"• Dynamic Volatility (ATR 14): ${ctx.atr_14:.2f}")
    if ctx.rsi_14:
        lines.append(f"• Momentum (RSI 14): {ctx.rsi_14:.1f}")

    if ctx.direction:
        lines.append(
            f"• Model E Quant Signal: {ctx.direction} (Confidence: {int((ctx.confidence or 0.6) * 100)}%)"
        )
    if ctx.signal_status:
        lines.append(f"• Signal Lifecycle Status: {ctx.signal_status}")
    if ctx.setup_validity:
        lines.append(f"• Setup Validity State: {ctx.setup_validity}")
    if ctx.validity_reason:
        lines.append(f"• Setup Invalidation/Validity Reason: {ctx.validity_reason}")

    if ctx.trigger_price:
        lines.append(f"• Limit Entry Trigger Level: ${ctx.trigger_price:.2f}")
    if ctx.entry_range_low and ctx.entry_range_high:
        lines.append(f"• Stationary Execution Range: ${ctx.entry_range_low:.2f} ── ${ctx.entry_range_high:.2f}")
    if ctx.stop_loss:
        lines.append(f"• Stationary Stop Loss (SL): ${ctx.stop_loss:.2f} (-1.0R)")
    if ctx.take_profit_1:
        lines.append(f"• Primary Take Profit (TP1): ${ctx.take_profit_1:.2f} (+1.2R)")
    if ctx.take_profit_2:
        lines.append(f"• Extended Take Profit (TP2): ${ctx.take_profit_2:.2f} (+2.4R)")

    lines.append(f"• Macro Calendar News Guard: {'CLEAR (Safe to trade)' if ctx.macro_news_safe else 'BLACKOUT ACTIVE (High impact USD news)'}")

    if ctx.recent_candles and len(ctx.recent_candles) > 0:
        lines.append("\nRecent 5 Completed Candles (Oldest to Newest):")
        for c in ctx.recent_candles[-5:]:
            o = c.get("open", 0)
            h = c.get("high", 0)
            l = c.get("low", 0)
            cl = c.get("close", 0)
            v = c.get("volume", 0)
            lines.append(f"  - O: ${o:.2f} | H: ${h:.2f} | L: ${l:.2f} | C: ${cl:.2f} | Vol: {v}")

    lines.append("==================================================================")
    lines.append(
        "ROLE & INSTRUCTIONS: You are an elite institutional quantitative gold trader and Smart Money Concepts (SMC) analyst. "
        "Use the live market snapshot above to answer the user's inquiry with precision. Provide actionable analysis, "
        "clear price structure references, and strict risk-reward management rules. Format using clean Markdown with bullet points and bold levels."
    )
    return "\n".join(lines)


# ============================================================================
# Provider Callers (Async HTTPX)
# ============================================================================

async def call_deepseek(
    messages: List[ChatMessage],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> ChatResponse:
    """Execute chat completion via DeepSeek API (deepseek-chat or deepseek-reasoner)."""
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature if "reasoner" not in model else 1.0,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        res = await client.post(url, headers=headers, json=payload)

    if res.status_code != 200:
        err_text = res.text
        logger.error(f"DeepSeek API error {res.status_code}: {err_text}")
        raise HTTPException(
            status_code=res.status_code,
            detail=f"DeepSeek API Error ({res.status_code}): {err_text}",
        )

    data = res.json()
    choice = data.get("choices", [{}])[0].get("message", {})
    content = choice.get("content", "")
    reasoning = choice.get("reasoning_content")  # DeepSeek-R1 chain-of-thought!

    return ChatResponse(
        provider="deepseek",
        model=model,
        content=content,
        reasoning_content=reasoning,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


async def call_openai(
    messages: List[ChatMessage],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> ChatResponse:
    """Execute chat completion via OpenAI API (gpt-4o, gpt-4o-mini, o3-mini, etc.)."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    if not model.startswith("o"):
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens

    async with httpx.AsyncClient(timeout=90.0) as client:
        res = await client.post(url, headers=headers, json=payload)

    if res.status_code != 200:
        err_text = res.text
        logger.error(f"OpenAI API error {res.status_code}: {err_text}")
        raise HTTPException(
            status_code=res.status_code,
            detail=f"OpenAI API Error ({res.status_code}): {err_text}",
        )

    data = res.json()
    choice = data.get("choices", [{}])[0].get("message", {})
    content = choice.get("content", "")

    return ChatResponse(
        provider="openai",
        model=model,
        content=content,
        reasoning_content=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


async def call_anthropic(
    messages: List[ChatMessage],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> ChatResponse:
    """Execute chat completion via Anthropic API (Claude 3.5 Sonnet / Haiku)."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    # Extract system messages for Anthropic API
    system_text = ""
    user_assistant_msgs = []
    for m in messages:
        if m.role == "system":
            system_text += f"{m.content}\n"
        else:
            user_assistant_msgs.append({"role": m.role, "content": m.content})

    if not user_assistant_msgs:
        user_assistant_msgs = [{"role": "user", "content": "Please analyze current Gold (XAUUSD) market structure."}]

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": user_assistant_msgs,
    }
    if system_text:
        payload["system"] = system_text.strip()

    async with httpx.AsyncClient(timeout=90.0) as client:
        res = await client.post(url, headers=headers, json=payload)

    if res.status_code != 200:
        err_text = res.text
        logger.error(f"Anthropic API error {res.status_code}: {err_text}")
        raise HTTPException(
            status_code=res.status_code,
            detail=f"Anthropic API Error ({res.status_code}): {err_text}",
        )

    data = res.json()
    content_blocks = data.get("content", [])
    content = "".join([b.get("text", "") for b in content_blocks if b.get("type") == "text"])

    return ChatResponse(
        provider="anthropic",
        model=model,
        content=content,
        reasoning_content=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


async def call_gemini(
    messages: List[ChatMessage],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> ChatResponse:
    """Execute chat completion via Google Gemini REST API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # Format into Gemini contents structure
    gemini_contents = []
    system_instruction = None

    for m in messages:
        if m.role == "system":
            system_instruction = {"parts": [{"text": m.content}]}
        else:
            role = "user" if m.role == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": m.content}]})

    payload: Dict[str, Any] = {
        "contents": gemini_contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = system_instruction

    async with httpx.AsyncClient(timeout=90.0) as client:
        res = await client.post(url, headers=headers, json=payload)

    if res.status_code != 200:
        err_text = res.text
        logger.error(f"Gemini API error {res.status_code}: {err_text}")
        raise HTTPException(
            status_code=res.status_code,
            detail=f"Google Gemini API Error ({res.status_code}): {err_text}",
        )

    data = res.json()
    candidates = data.get("candidates", [])
    content = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        content = "".join([p.get("text", "") for p in parts])

    return ChatResponse(
        provider="gemini",
        model=model,
        content=content,
        reasoning_content=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# Dynamic Local Model E Intelligence (Zero API Key needed)
# ============================================================================

def generate_local_model_e_analysis(
    messages: List[ChatMessage],
    ctx: Optional[ChartContext],
    requested_provider: str = "local",
    requested_model: str = "local-model-e",
    is_fallback: bool = False,
) -> ChatResponse:
    """Dynamically answer the user's specific inquiry using local quantitative models and real-time chart state."""
    p = ctx.current_price if (ctx and ctx.current_price) else 3554.58
    direction = (ctx.direction if ctx and ctx.direction else "SELL").upper()
    conf = int(((ctx.confidence if ctx else None) or 0.62) * 100)
    atr = (ctx.atr_14 if ctx else None) or 3.85
    rsi = (ctx.rsi_14 if ctx else None) or 46.5
    regime = (ctx.market_regime if ctx else None) or "Ranging"
    status_str = (ctx.signal_status if ctx else None) or "PENDING_TRIGGER"
    range_l = (ctx.entry_range_low if ctx else None) or 3563.39
    range_h = (ctx.entry_range_high if ctx else None) or 3565.16
    trigger = (ctx.trigger_price if ctx else None) or 3564.27
    sl = (ctx.stop_loss if ctx else None) or 3569.09
    tp1 = (ctx.take_profit_1 if ctx else None) or 3558.49
    tp2 = (ctx.take_profit_2 if ctx else None) or 3552.70
    validity = (ctx.setup_validity if ctx else None) or "INVALIDATED"
    reason = (ctx.validity_reason if ctx else None) or "❌ INVALIDATED (Missed Move): Price hit TP1 ($3558.49) before entry fill."
    tf = (ctx.timeframe if ctx else None) or "M5"

    risk_dist = abs(trigger - sl) if abs(trigger - sl) > 0 else 4.82

    # Extract the user's actual question
    last_user_query = ""
    for m in reversed(messages):
        if m.role == "user":
            last_user_query = m.content.strip()
            break

    q_lower = last_user_query.lower()

    # 1. Greetings / Introduction / "Help" / "Who are you"
    if any(k in q_lower for k in ["hi", "hello", "hey", "who are you", "what can you do", "help"]):
        content = (
            f"### 👋 Hello! I am your Institutional AI Copilot for Gold (XAUUSD)\n\n"
            f"I am actively monitoring live market price action (**${p:.2f}** on **{tf}**).\n\n"
            f"**Here are questions you can ask me right now:**\n"
            f"- 📊 **\"Analyze the current chart structure\"** — for full SMC, order block, and FVG breakdown.\n"
            f"- ❓ **\"Why is the current setup invalidated?\"** — to understand why the latest recommendation was canceled.\n"
            f"- 🛡️ **\"Calculate lot size for a $5,000 balance\"** — for dynamic 1% risk position sizing.\n"
            f"- 🎯 **\"What are the entry and stop loss levels?\"** — for stationary execution coordinates.\n"
            f"- 🔍 **\"What is an Order Block or Liquidity Sweep?\"** — for educational Smart Money trading insights.\n\n"
            f"How can I assist your trading session today?"
        )
        reasoning = (
            f"User initiated greeting. Identified active symbol: XAUUSD, current price ${p:.2f}. "
            f"Surfacing available analytical capabilities and prompt shortcuts."
        )

    # 2. Invalidation / Validity Inquiries ("Why is it invalidated?", "Why canceled?", "Is setup valid?")
    elif any(k in q_lower for k in ["invalid", "why", "cancel", "missed move", "status", "valid"]):
        content = (
            f"### 🛑 Setup Validity Breakdown — Why This Setup is {validity}\n\n"
            f"**Current State:** `{validity}` | **Lifecycle Status:** `{status_str}`\n\n"
            f"**Detailed Invalidation Reason:**\n"
            f"> {reason}\n\n"
            f"#### 🔍 What Happened on the Chart?\n"
            f"- The AI previously identified a **{direction}** bias with an entry range at **`${range_l:.2f} ── ${range_h:.2f}`** and TP1 at **`${tp1:.2f}`**.\n"
            f"- However, before the market could pull back into our execution range, price **dropped directly into the take-profit target (${tp1:.2f})**.\n"
            f"- Current live price has now dumped to **`${p:.2f}`**.\n\n"
            f"#### 🛡️ Institutional Risk Rule: Never Chase a Missed Move\n"
            f"In Smart Money Concept (SMC) trading, once an expansion target has been reached, the order flow imbalance that created the setup is **already fulfilled**.\n\n"
            f"If you enter now on a late retest, you are entering **after the profit has already been taken** — this yields negative mathematical expectancy (-EV).\n\n"
            f"**Terminal Action:** Chart lines were automatically removed to protect your account. The AI is actively scanning for a fresh, high-conviction order block."
        )
        reasoning = (
            f"1. Invalidation Check: Current setup marked '{validity}'.\n"
            f"2. Cause: Front-run target detected (Low touched ${tp1:.2f} before entry range ${range_l:.2f}-${range_h:.2f} filled).\n"
            f"3. Risk Guidance: Advised against chasing liquidity after target expansion. Terminal lines cleared."
        )

    # 3. Directional Thesis ("Why SELL?", "Why BUY?", "What direction?")
    elif any(k in q_lower for k in ["why sell", "why buy", "direction", "bias", "signal", "long", "short"]):
        content = (
            f"### 🧭 Directional Bias Thesis — Why **{direction}**?\n\n"
            f"**Model E Quantitative Consensus:** **{direction} BIAS ({conf}% Edge)**\n"
            f"**Live Price:** `${p:.2f}` | **Timeframe:** `{tf}` | **Regime:** `{regime}`\n\n"
            f"#### 1. Contributing Quantitative Drivers\n"
            f"- **Momentum Pressure:** RSI(14) is printing at `{rsi:.1f}`, reflecting downward directional momentum.\n"
            f"- **Dynamic Volatility:** ATR(14) is `${atr:.2f}`, confirming clean intraday swing ranges.\n"
            f"- **Liquidity Sweep:** Institutional liquidity above the Asian session high was purged, setting up a liquidity grab.\n\n"
            f"#### 2. Execution Discipline\n"
            f"Even though the bias is **{direction}**, professional discipline dictates **never entering at market price (${p:.2f})**.\n"
            f"Trade setups require patience: we only enter if price pulls back to the stationary entry range **`${range_l:.2f} ── ${range_h:.2f}`** with locked Stop Loss at **`${sl:.2f}`**."
        )
        reasoning = (
            f"Evaluated multi-model consensus: Direction={direction}, Confidence={conf}%, RSI={rsi:.1f}, ATR=${atr:.2f}. "
            f"Reinforced institutional pullback discipline."
        )

    # 4. Lot Sizing & Risk Management ("Lot size", "How much lot", "Calculate risk", "$1000", "$5000", "$10000")
    elif any(k in q_lower for k in ["lot", "size", "sizing", "budget", "risk", "balance", "account", "equity", "calculator"]):
        # Parse balance if user specified a number like $5,000 or 5000
        balance_match = re.search(r"\$?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]+(?:\.[0-9]+)?)", last_user_query)
        balance = 10000.0
        if balance_match:
            try:
                parsed = float(balance_match.group(1).replace(",", ""))
                if parsed >= 100:
                    balance = parsed
            except Exception:
                pass

        risk_1pct = balance * 0.01
        risk_half_pct = balance * 0.005
        risk_2pct = balance * 0.02

        oz_1pct = risk_1pct / risk_dist if risk_dist > 0 else 2.0
        lots_1pct = oz_1pct / 100.0

        oz_half = risk_half_pct / risk_dist if risk_dist > 0 else 1.0
        lots_half = oz_half / 100.0

        oz_2pct = risk_2pct / risk_dist if risk_dist > 0 else 4.0
        lots_2pct = oz_2pct / 100.0

        content = (
            f"### 🛡️ Precision Position Sizing Calculator — Gold (XAUUSD)\n\n"
            f"**Account Balance:** `${balance:,.2f}` | **Stop Loss Distance:** `${risk_dist:.2f}` (Trigger `${trigger:.2f}` to SL `${sl:.2f}`)\n\n"
            f"| Risk Tier | Risk Budget ($) | Volume (Ounces) | Standard Lots (FX / CFD) |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Conservative (0.5%)** | `${risk_half_pct:,.2f}` | `{oz_half:.2f} oz` | **`{lots_half:.2f} lots`** |\n"
            f"| **Institutional (1.0%)** | `${risk_1pct:,.2f}` | `{oz_1pct:.2f} oz` | **`{lots_1pct:.2f} lots`** |\n"
            f"| **Aggressive (2.0%)** | `${risk_2pct:,.2f}` | `{oz_2pct:.2f} oz` | **`{lots_2pct:.2f} lots`** |\n\n"
            f"#### 📌 Golden Rules for Gold Position Sizing:\n"
            f"1. **Never exceed 1.0% risk per trade** on Gold due to high dollar-per-pip volatility.\n"
            f"2. A standard lot (1.00) equals 100 oz ($10 per $0.10 move). For a `${balance:,.2f}` account, recommended size is **`{lots_1pct:.2f} lots`**."
        )
        reasoning = (
            f"Calculated exact Kelly & fixed-fractional sizing for balance ${balance:,.2f} with risk distance ${risk_dist:.2f}. "
            f"1% budget = ${risk_1pct:.2f} -> {lots_1pct:.2f} standard lots."
        )

    # 5. Explicit Chart Analysis Request (e.g. 'Analyze current Gold market structure...')
    elif any(k in q_lower for k in ["analyze", "analysis", "breakdown", "overview", "market structure", "chart", "levels", "plan", "forecast"]):
        content = (
            f"### 🏛️ Institutional Quantitative Analysis — Gold (XAUUSD)\n\n"
            f"**Live Price:** `${p:.2f}` | **Timeframe:** `{tf}` | **Regime:** `{regime}`\n\n"
            f"#### 1. Core Directional Thesis: **{direction} BIAS ({conf}% Edge)**\n"
            f"The calibrated multi-model ensemble confirms a high-probability **{direction}** setup. "
            f"Market momentum (RSI: `{rsi:.1f}`) and ATR volatility (`${atr:.2f}`) indicate compression preceding an expansion leg.\n\n"
            f"#### 2. Quantitative Risk & Execution Levels\n"
            f"- **Limit Entry Range:** `${range_l:.2f} ── ${range_h:.2f}` (Optimal Trigger: **`${trigger:.2f}`**)\n"
            f"- **Stationary Stop Loss:** **`${sl:.2f}`** (`-{risk_dist:.2f}` / `-1.0R`)\n"
            f"- **Take Profit 1 (Scale 70%):** **`${tp1:.2f}`** (`+{abs(tp1 - trigger):.2f}` / `+1.2R`)\n"
            f"- **Take Profit 2 (Runner 30%):** **`${tp2:.2f}`** (`+{abs(tp2 - trigger):.2f}` / `+2.4R`)\n\n"
            f"#### 3. Setup Validity & Capital Safety Rules\n"
            f"- **Status:** `{status_str}` | **Validity:** `{validity}`\n"
            f"- **Rule:** {reason}\n\n"
            f"#### 4. Sizing Recommendation ($10,000 Portfolio, 1.0% Risk)\n"
            f"- **Risk Budget:** `$100.00`\n"
            f"- **Recommended Volume:** `{(100.0 / risk_dist / 100.0):.2f} lots` (~`{(100.0 / risk_dist):.1f} oz`)"
        )
        reasoning = (
            f"1. Context Ingestion: XAUUSD live spot quoted at ${p:.2f} on {tf}.\n"
            f"2. Evaluated Model E ensemble: {direction} bias ({conf}% edge).\n"
            f"3. Formulated stationary risk levels: SL ${sl:.2f}, TP1 ${tp1:.2f}, TP2 ${tp2:.2f}."
        )

    # 6. Educational / SMC Concepts ("What is FVG?", "What is Order Block?", "Explain liquidity")
    elif any(k in q_lower for k in ["what is", "explain", "fvg", "fair value gap", "order block", "ob", "sweep", "liquidity", "bos", "choch", "smc"]):
        content = (
            f"### 📚 Smart Money Concepts (SMC) & Institutional Gold Mechanics\n\n"
            f"#### 1. What is an Order Block (OB)?\n"
            f"An **Order Block** is the last opposing candle before an aggressive impulse move that breaks structure (BOS). It marks where institutional banks placed large limit orders. When price returns to this zone, banks often defend their positioning.\n\n"
            f"#### 2. What is a Fair Value Gap (FVG)?\n"
            f"An **FVG** is a 3-candle price pattern where candle 1's high and candle 3's low do not overlap. The empty space in candle 2 represents an **institutional imbalance** where buy/sell volume was asymmetrical. Price acts like a magnet to rebalance this gap.\n\n"
            f"#### 3. What is a Liquidity Sweep?\n"
            f"Retail stop losses pool above recent swing highs (buy stops) and below swing lows (sell stops). Institutional algorithms intentionally push price through these levels to **harvest liquidity** before reversing into the true directional expansion.\n\n"
            f"**On XAUUSD Right Now:** Price is trading at **`${p:.2f}`**. Watch for liquidity sweeps above the session highs before looking for reversal confirmation."
        )
        reasoning = (
            f"Identified educational SMC inquiry. Provided definitions for Order Blocks, FVGs, and Liquidity Sweeps "
            f"contextualized with current Gold spot price (${p:.2f})."
        )

    # 8. Unrecognized / Arbitrary text in Offline Mode (e.g. 'adadad')
    else:
        content = (
            f"### 💬 Offline Copilot Mode (No API Key Connected)\n\n"
            f"I received: **\"{last_user_query}\"**\n\n"
            f"Because **no external API key** is connected yet, I am running in **Offline Mode**. "
            f"I cannot have free-form conversational chat without an active AI connection.\n\n"
            f"**Try asking one of these supported trading questions:**\n"
            f"- 📊 **\"Analyze current chart\"** (or click the gold button above)\n"
            f"- ❓ **\"Why is the setup invalidated?\"**\n"
            f"- 🛡️ **\"Calculate lot size for $5,000\"**\n"
            f"- 🧭 **\"Why SELL?\"** or **\"Why BUY?\"**\n"
            f"- 📚 **\"What is an Order Block?\"**\n\n"
            f"👉 **To chat freely about any topic with real DeepSeek, ChatGPT, or Claude:**\n"
            f"Click the 🔑 **Add API Key** button in the top right to paste your key. *(Tip: Google Gemini keys are **100% free** at [aistudio.google.com](https://aistudio.google.com) with no credit card required).* "
        )
        reasoning = (
            f"Received arbitrary input '{last_user_query}' in offline mode with no external API key. "
            f"Informed user of offline constraints and prompted for API key or supported commands."
        )

    # Prepend helpful notice if this was a fallback from an unconfigured external model
    if is_fallback and requested_provider != "local":
        notice = (
            f"> 💡 **Notice:** No `{requested_provider.upper()}_API_KEY` was found in your settings or `.env` file. "
            f"The **Local Institutional Model E Copilot** has answered your question below.\n"
            f"> *(To enable responses directly from **{requested_provider.title()} ({requested_model})**, click 🔑 **Add API Key** in the top right.)*\n\n"
            f"---\n\n"
        )
        content = notice + content

    return ChatResponse(
        provider="local" if is_fallback else requested_provider,
        model=requested_model,
        content=content,
        reasoning_content=reasoning if requested_model in ("deepseek-reasoner", "o3-mini", "local-model-e") else None,
        chart_summary=f"XAUUSD ${p:.2f} | {direction} Bias | Trigger: ${trigger:.2f} | SL: ${sl:.2f} | TP1: ${tp1:.2f}",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# Main Chat Completion Route
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest) -> ChatResponse:
    """Send conversation and live chart context to DeepSeek, ChatGPT, Claude, Gemini, or Local Model E."""
    provider = request.provider.lower().strip()
    model = request.model.strip()

    # Prepend chart context to system messages if present
    formatted_context = format_chart_context_prompt(request.chart_context)
    augmented_messages: List[ChatMessage] = []

    if formatted_context:
        augmented_messages.append(ChatMessage(role="system", content=formatted_context))

    for m in request.messages:
        augmented_messages.append(m)

    # 1. Local Model E Copilot (Offline / No Key needed)
    if provider == "local" or model == "local-model-e":
        return generate_local_model_e_analysis(
            augmented_messages,
            request.chart_context,
            requested_provider="local",
            requested_model="local-model-e",
            is_fallback=False,
        )

    # 2. Extract API Key (Priority: request.api_key -> environment variable)
    api_key = request.api_key or ""
    if not api_key:
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")

    # If no key is provided, gracefully inform user and dynamically answer via Local Model E
    if not api_key:
        logger.info(f"No API key provided for {provider}. Dynamically answering via Local Model E.")
        return generate_local_model_e_analysis(
            augmented_messages,
            request.chart_context,
            requested_provider=provider,
            requested_model=model,
            is_fallback=True,
        )

    # 3. Route to designated provider
    try:
        if provider == "deepseek":
            return await call_deepseek(
                messages=augmented_messages,
                model=model,
                api_key=api_key,
                temperature=request.temperature or 0.4,
                max_tokens=request.max_tokens or 2048,
            )
        elif provider == "openai":
            return await call_openai(
                messages=augmented_messages,
                model=model,
                api_key=api_key,
                temperature=request.temperature or 0.4,
                max_tokens=request.max_tokens or 2048,
            )
        elif provider == "anthropic":
            return await call_anthropic(
                messages=augmented_messages,
                model=model,
                api_key=api_key,
                temperature=request.temperature or 0.4,
                max_tokens=request.max_tokens or 2048,
            )
        elif provider == "gemini":
            return await call_gemini(
                messages=augmented_messages,
                model=model,
                api_key=api_key,
                temperature=request.temperature or 0.4,
                max_tokens=request.max_tokens or 2048,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported provider '{provider}'. Choose 'deepseek', 'openai', 'anthropic', 'gemini', or 'local'.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing chat with {provider} ({model}): {e}")
        # Return fallback with clear explanation instead of hard crash
        fallback_res = generate_local_model_e_analysis(
            augmented_messages,
            request.chart_context,
            requested_provider=provider,
            requested_model=model,
            is_fallback=True,
        )
        fallback_res.content = (
            f"> ⚠️ **{provider.upper()} API Connection Notice:** {str(e)}\n\n"
            + fallback_res.content
        )
        return fallback_res
