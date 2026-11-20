"""Market data and feature extraction endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
import numpy as np
import pandas as pd

from src.api.dependencies import get_duckdb_manager, get_parquet_storage
from src.api.schemas import (
    CandleItem,
    CandlesResponse,
    MarketFeaturesResponse,
    SMCStructure,
)
from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.data.duckdb_manager import DuckDBManager
from src.data.storage import ParquetStorage

router = APIRouter(prefix="/api/market", tags=["Market Data"])

_LIVE_CACHE: Dict[str, Any] = {}
_LIVE_CACHE_TTL = 4  # 4 seconds in-memory cache


def fetch_live_realtime_gold_candles(
    timeframe: str = "M5", limit: int = 500
) -> Optional[List[CandleItem]]:
    """Fetch zero-delay institutional real-time Gold spot candles.

    Uses Tier 1 Kraken Institutional Spot Feed (0s delay, matching TradingView to the second),
    with Tier 2 Bitfinex and Tier 3 Yahoo Finance as resilient multi-tier backups.
    """
    import json
    import time
    import urllib.request

    now = time.time()
    tf_upper = timeframe.upper()
    cache_key = f"{tf_upper}_{limit}"
    if cache_key in _LIVE_CACHE:
        cached_time, cached_data = _LIVE_CACHE[cache_key]
        if now - cached_time < _LIVE_CACHE_TTL:
            return cached_data

    # -----------------------------------------------------------------
    # Tier 1: Kraken Institutional Spot Gold (PAXGUSD) - 0 Delay
    # -----------------------------------------------------------------
    try:
        interval_map = {"M1": 1, "M5": 5, "M15": 15, "H1": 60}
        inv = interval_map.get(tf_upper, 5)
        url = f"https://api.kraken.com/0/public/OHLC?pair=PAXGUSD&interval={inv}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        if "result" in data:
            pair_key = [x for x in data["result"].keys() if x != "last"][0]
            raw_candles = data["result"][pair_key]
            candles: List[CandleItem] = []
            for c in raw_candles[-limit:]:
                # Kraken format: [time_sec, open, high, low, close, vwap, vol, count]
                candles.append(
                    CandleItem(
                        timestamp=int(c[0]),
                        open=round(float(c[1]), 2),
                        high=round(float(c[2]), 2),
                        low=round(float(c[3]), 2),
                        close=round(float(c[4]), 2),
                        volume=round(float(c[6]), 4),
                        instrument="XAUUSD",
                        timeframe=tf_upper,
                    )
                )
            if len(candles) > 0:
                _LIVE_CACHE[cache_key] = (now, candles)
                return candles
    except Exception as e:
        logger.warning(f"Tier 1 Kraken real-time gold feed notice: {e}")

    # -----------------------------------------------------------------
    # Tier 2: Bitfinex Institutional Spot Gold (tXAUT:USD) - 0 Delay
    # -----------------------------------------------------------------
    try:
        tf_map_bf = {"M1": "1m", "M5": "5m", "M15": "15m", "H1": "1h"}
        intv_bf = tf_map_bf.get(tf_upper, "5m")
        url_bf = f"https://api-pub.bitfinex.com/v2/candles/trade:{intv_bf}:tXAUT:USD/hist?limit={limit}"
        req_bf = urllib.request.Request(url_bf, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req_bf, timeout=4) as resp:
            data_bf = json.loads(resp.read().decode())
        if isinstance(data_bf, list) and len(data_bf) > 0:
            data_bf.reverse()  # oldest to newest
            candles_bf: List[CandleItem] = []
            for c in data_bf:
                # Bitfinex format: [mts, open, close, high, low, volume]
                candles_bf.append(
                    CandleItem(
                        timestamp=int(c[0] // 1000),
                        open=round(float(c[1]), 2),
                        high=round(float(c[3]), 2),
                        low=round(float(c[4]), 2),
                        close=round(float(c[2]), 2),
                        volume=round(float(c[5]), 4),
                        instrument="XAUUSD",
                        timeframe=tf_upper,
                    )
                )
            if len(candles_bf) > 0:
                _LIVE_CACHE[cache_key] = (now, candles_bf)
                return candles_bf
    except Exception as e:
        logger.warning(f"Tier 2 Bitfinex real-time gold feed notice: {e}")

    # -----------------------------------------------------------------
    # Tier 3: Yahoo Finance (GC=F) - Fallback
    # -----------------------------------------------------------------
    tf_map = {
        "M1": ("1m", "1d"),
        "M5": ("5m", "5d"),
        "M15": ("15m", "10d"),
        "H1": ("1h", "1mo"),
    }
    interval, rng = tf_map.get(tf_upper, ("5m", "5d"))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={rng}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
    )

    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
        res = data["chart"]["result"][0]
        ts_list = res.get("timestamp", [])
        q = res["indicators"]["quote"][0]
        opens = q.get("open", [])
        highs = q.get("high", [])
        lows = q.get("low", [])
        closes = q.get("close", [])
        vols = q.get("volume", [])

        candles_yh: List[CandleItem] = []
        for i in range(len(ts_list)):
            t = ts_list[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]
            v = vols[i] if vols and i < len(vols) else 0.0
            if o is None or h is None or l is None or c is None:
                continue
            candles_yh.append(
                CandleItem(
                    timestamp=int(t),
                    open=round(float(o), 2),
                    high=round(float(h), 2),
                    low=round(float(l), 2),
                    close=round(float(c), 2),
                    volume=float(v or 0.0),
                    instrument="XAUUSD",
                    timeframe=tf_upper,
                )
            )

        if candles_yh:
            result = candles_yh[-limit:]
            _LIVE_CACHE[cache_key] = (now, result)
            return result
    except Exception:
        pass
    return None


# Backward-compatible alias
fetch_live_yahoo_gold_candles = fetch_live_realtime_gold_candles


@router.get("/status")
def get_market_status(
    instrument: str = Query(DEFAULT_INSTRUMENT, description="Trading instrument e.g. XAUUSD"),
) -> Dict[str, Any]:
    """Retrieve official global market session status (Open/Closed), UTC schedule, and data origin."""
    from src.live.feed import LiveMarketFeed
    feed = LiveMarketFeed()
    status = feed.get_market_status(instrument)

    # Enrich with live COMEX Gold market price if available
    live_candles = fetch_live_yahoo_gold_candles(timeframe="M5", limit=5)
    if live_candles and len(live_candles) > 0:
        latest_c = live_candles[-1]
        status["latest_available_price"] = latest_c.close
        status["latest_candle_time_utc"] = pd.to_datetime(
            latest_c.timestamp, unit="s", utc=True
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        status["data_source_mode"] = "REALTIME_INSTITUTIONAL_FEED"

    return status


@router.get("/candles", response_model=CandlesResponse)
def get_candles(
    instrument: str = Query(DEFAULT_INSTRUMENT, description="Trading instrument e.g. XAUUSD"),
    timeframe: str = Query("M5", description="Timeframe: M5, M15, or H1"),
    start: Optional[int] = Query(None, description="Start timestamp in ms or s"),
    end: Optional[int] = Query(None, description="End timestamp in ms or s"),
    limit: int = Query(500, ge=1, le=2000, description="Maximum candles to return"),
    storage: ParquetStorage = Depends(get_parquet_storage),
) -> CandlesResponse:
    """Fetch verified historical OHLCV candles for charting and display."""
    try:
        tf_enum = Timeframe(timeframe.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe '{timeframe}'. Use M5, M15, or H1.")

    # 1. First attempt to serve real-time live COMEX Gold candles for active live charting
    if start is None and end is None and instrument.upper() in ["XAUUSD", "GOLD", "GC=F"]:
        live_candles = fetch_live_yahoo_gold_candles(timeframe=tf_enum.value.upper(), limit=limit)
        if live_candles and len(live_candles) > 0:
            return CandlesResponse(
                instrument=instrument.upper(),
                timeframe=tf_enum.value.upper(),
                count=len(live_candles),
                candles=live_candles,
            )

    # 2. Historical or offline fallback
    df = storage.load_candles(timeframe=tf_enum, instrument=instrument.upper(), is_processed=True)
    if df.empty:
        # Fall back to raw if processed not yet created for this timeframe
        df = storage.load_candles(timeframe=tf_enum, instrument=instrument.upper(), is_processed=False)

    if df.empty:
        return CandlesResponse(
            instrument=instrument.upper(),
            timeframe=tf_enum.value.upper(),
            count=0,
            candles=[],
        )

    # Standardize timestamp to integer seconds
    # If in milliseconds (> 10^11), divide by 1000 for standard chart unix time
    if df["timestamp"].iloc[0] > 10_000_000_000:
        df = df.copy()
        df["timestamp"] = (df["timestamp"] // 1000).astype(int)

    if start is not None:
        if start > 10_000_000_000:
            start = start // 1000
        df = df[df["timestamp"] >= start]

    if end is not None:
        if end > 10_000_000_000:
            end = end // 1000
        df = df[df["timestamp"] <= end]

    # Return the most recent 'limit' records sorted chronologically
    df_slice = df.tail(limit)

    candles = [
        CandleItem(
            timestamp=int(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
            instrument=str(row.get("instrument", instrument.upper())),
            timeframe=tf_enum.value.upper(),
        )
        for _, row in df_slice.iterrows()
    ]

    return CandlesResponse(
        instrument=instrument.upper(),
        timeframe=tf_enum.value.upper(),
        count=len(candles),
        candles=candles,
    )


@router.get("/features", response_model=MarketFeaturesResponse)
def get_market_features(
    instrument: str = Query(DEFAULT_INSTRUMENT, description="Trading instrument"),
    timeframe: str = Query("M5", description="Timeframe: M5, M15, or H1"),
    limit: int = Query(100, ge=10, le=500, description="Lookback window for SMC patterns"),
) -> MarketFeaturesResponse:
    """Fetch current technical indicators, market regime, and detected SMC price structures."""
    tf_clean = (timeframe or "M5").lower().strip()
    feat_file = f"{tf_clean}_features.parquet"
    feat_path = settings.FEATURES_DIR / instrument.lower() / feat_file
    if not feat_path.exists():
        feat_path = settings.FEATURES_DIR / instrument.lower() / "m5_features.parquet"

    if not feat_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Features matrix not found for {instrument}. Run 'features' pipeline first.",
        )

    df = pd.read_parquet(feat_path)
    if df.empty:
        raise HTTPException(status_code=404, detail="Features matrix is empty.")

    latest = df.iloc[-1]
    lookback = df.tail(limit)

    # Standardize timestamp
    raw_ts = int(latest["timestamp"])
    ts_sec = raw_ts // 1000 if raw_ts > 10_000_000_000 else raw_ts

    # Indicators dictionary
    indicators = {
        "ema_9": round(float(latest.get("ema_9", 0.0)), 2),
        "ema_20": round(float(latest.get("ema_20", 0.0)), 2),
        "ema_50": round(float(latest.get("ema_50", 0.0)), 2),
        "ema_100": round(float(latest.get("ema_100", 0.0)), 2),
        "ema_200": round(float(latest.get("ema_200", 0.0)), 2),
        "ema_alignment": int(latest.get("ema_alignment", 0)),
        "rsi_14": round(float(latest.get("rsi_14", 50.0)), 2),
        "macd_line": round(float(latest.get("macd_line", 0.0)), 3),
        "macd_signal": round(float(latest.get("macd_signal", 0.0)), 3),
        "macd_hist": round(float(latest.get("macd_hist", 0.0)), 3),
        "atr_14": round(float(latest.get("atr_14", 2.0)), 3),
        "bb_upper": round(float(latest.get("bb_upper", 0.0)), 2),
        "bb_middle": round(float(latest.get("bb_middle", 0.0)), 2),
        "bb_lower": round(float(latest.get("bb_lower", 0.0)), 2),
        "bb_bandwidth": round(float(latest.get("bb_bandwidth", 0.0)), 4),
        "stoch_rsi_k": round(float(latest.get("stoch_rsi_k", 50.0)), 2),
        "stoch_rsi_d": round(float(latest.get("stoch_rsi_d", 50.0)), 2),
    }

    # Regime dictionary
    regime = {
        "regime_label": str(latest.get("regime_label", "Neutral")),
        "regime_code": int(latest.get("market_regime", 0)),
        "efficiency_ratio": round(float(latest.get("efficiency_ratio", 0.0)), 3),
        "trend_slope": round(float(latest.get("trend_slope", 0.0)), 4),
        "is_trending": bool(latest.get("regime_trending", False)),
        "is_ranging": bool(latest.get("regime_ranging", False)),
        "is_expansion": bool(latest.get("regime_expansion", False)),
        "session": "London" if latest.get("session_london", 0) else ("NewYork" if latest.get("session_newyork", 0) else "Asian"),
    }

    # Extract detected SMC structures within lookback
    swing_highs = []
    swing_lows = []
    active_fvgs = []
    recent_bos = []
    recent_choch = []
    order_blocks = []
    liquidity_sweeps = []
    ict_2022_setups = []
    silver_bullet_setups = []

    for _, row in lookback.iterrows():
        t = int(row["timestamp"])
        t = t // 1000 if t > 10_000_000_000 else t

        if row.get("swing_high_confirmed", 0) == 1:
            swing_highs.append({"timestamp": t, "price": round(float(row["high"]), 2)})
        if row.get("swing_low_confirmed", 0) == 1:
            swing_lows.append({"timestamp": t, "price": round(float(row["low"]), 2)})

        if row.get("fvg_bullish_new", 0) == 1:
            active_fvgs.append({
                "timestamp": t,
                "type": "BULLISH",
                "top": round(float(row["low"]), 2),
                "bottom": round(float(row["high"]), 2),
            })
        elif row.get("fvg_bearish_new", 0) == 1:
            active_fvgs.append({
                "timestamp": t,
                "type": "BEARISH",
                "top": round(float(row["low"]), 2),
                "bottom": round(float(row["high"]), 2),
            })

        if row.get("bos_bullish", 0) == 1:
            recent_bos.append({"timestamp": t, "direction": "BULLISH", "price": round(float(row["close"]), 2)})
        elif row.get("bos_bearish", 0) == 1:
            recent_bos.append({"timestamp": t, "direction": "BEARISH", "price": round(float(row["close"]), 2)})

        if row.get("choch_bullish", 0) == 1:
            recent_choch.append({"timestamp": t, "direction": "BULLISH", "price": round(float(row["close"]), 2)})
        elif row.get("choch_bearish", 0) == 1:
            recent_choch.append({"timestamp": t, "direction": "BEARISH", "price": round(float(row["close"]), 2)})

        if row.get("order_block_bullish", 0) == 1:
            order_blocks.append({"timestamp": t, "direction": "BULLISH", "price": round(float(row["close"]), 2)})
        elif row.get("order_block_bearish", 0) == 1:
            order_blocks.append({"timestamp": t, "direction": "BEARISH", "price": round(float(row["close"]), 2)})

        if row.get("liquidity_sweep_bullish", 0) == 1:
            liquidity_sweeps.append({"timestamp": t, "direction": "BULLISH", "price": round(float(row["low"]), 2)})
        elif row.get("liquidity_sweep_bearish", 0) == 1:
            liquidity_sweeps.append({"timestamp": t, "direction": "BEARISH", "price": round(float(row["high"]), 2)})

        if row.get("ict_2022_bullish", 0) == 1:
            ict_2022_setups.append({"timestamp": t, "type": "BULLISH", "price": round(float(row["close"]), 2)})
        elif row.get("ict_2022_bearish", 0) == 1:
            ict_2022_setups.append({"timestamp": t, "type": "BEARISH", "price": round(float(row["close"]), 2)})

        if row.get("ict_silver_bullet_bullish", 0) == 1:
            silver_bullet_setups.append({"timestamp": t, "type": "BULLISH", "price": round(float(row["close"]), 2)})
        elif row.get("ict_silver_bullet_bearish", 0) == 1:
            silver_bullet_setups.append({"timestamp": t, "type": "BEARISH", "price": round(float(row["close"]), 2)})

    smc = SMCStructure(
        swing_highs=swing_highs[-10:],
        swing_lows=swing_lows[-10:],
        active_fvgs=active_fvgs[-10:],
        recent_bos=recent_bos[-5:],
        recent_choch=recent_choch[-5:],
        order_blocks=order_blocks[-5:],
        liquidity_sweeps=liquidity_sweeps[-5:],
        ict_2022_setups=ict_2022_setups[-5:],
        silver_bullet_setups=silver_bullet_setups[-5:],
    )

    return MarketFeaturesResponse(
        instrument=instrument.upper(),
        timeframe=tf_clean.upper(),
        timestamp=ts_sec,
        current_price=round(float(latest["close"]), 2),
        regime=regime,
        indicators=indicators,
        smc=smc,
    )


@router.get("/news-calendar")
def get_economic_news_calendar(
    limit: int = Query(10, ge=1, le=50, description="Maximum upcoming events to return"),
    buffer_mins: int = Query(30, ge=5, le=120, description="Blackout buffer in minutes before/after event"),
):
    """Retrieve live High-Impact USD economic news calendar, blackout status, and countdown to next release."""
    from datetime import datetime, timezone
    from src.models.news_filter import get_news_filter

    nf = get_news_filter()
    now_utc = datetime.now(timezone.utc)
    is_blackout, active_evt, delta_mins = nf.is_blackout_active(
        now_utc, buffer_before_mins=buffer_mins, buffer_after_mins=buffer_mins
    )

    upcoming_raw = nf.get_upcoming_events(from_time=now_utc, limit=limit)

    minutes_until_next = None
    next_title = None
    next_time = None
    if upcoming_raw:
        first_upcoming = upcoming_raw[0]
        minutes_until_next = first_upcoming.get("minutes_until_event")
        next_title = first_upcoming.get("title")
        next_time = first_upcoming.get("event_time_utc")

    return {
        "is_blackout_active": is_blackout,
        "active_event": active_evt.title if active_evt else None,
        "minutes_delta_from_event": delta_mins,
        "minutes_until_next_event": minutes_until_next,
        "next_event_title": next_title,
        "next_event_time_utc": next_time,
        "upcoming_events": upcoming_raw,
    }
