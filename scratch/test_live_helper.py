import json
import time
import urllib.request
from typing import List, Dict, Any, Optional

_CACHE: Dict[str, Any] = {}
_CACHE_TTL = 4  # seconds

def fetch_live_gold_candles(timeframe: str = "M5", limit: int = 500) -> Optional[List[Dict[str, Any]]]:
    now = time.time()
    cache_key = f"{timeframe.upper()}_{limit}"
    if cache_key in _CACHE:
        cached_time, cached_data = _CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_data

    tf_map = {
        "M1": ("1m", "1d"),
        "M5": ("5m", "5d"),
        "M15": ("15m", "10d"),
        "H1": ("1h", "1mo"),
    }
    interval, rng = tf_map.get(timeframe.upper(), ("5m", "5d"))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval={interval}&range={rng}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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

        candles = []
        for i in range(len(ts_list)):
            t = ts_list[i]
            o = opens[i]
            h = highs[i]
            l = lows[i]
            c = closes[i]
            v = vols[i] if vols and i < len(vols) else 0
            if o is None or h is None or l is None or c is None:
                continue
            candles.append({
                "timestamp": int(t),
                "open": round(float(o), 2),
                "high": round(float(h), 2),
                "low": round(float(l), 2),
                "close": round(float(c), 2),
                "volume": float(v or 0),
                "instrument": "XAUUSD",
                "timeframe": timeframe.upper(),
            })

        result = candles[-limit:]
        _CACHE[cache_key] = (now, result)
        return result
    except Exception as e:
        print("Live gold fetch error:", e)
        return None

if __name__ == "__main__":
    t0 = time.time()
    c1 = fetch_live_gold_candles("M5", 100)
    print(f"Fetch took {time.time() - t0:.2f}s, got {len(c1) if c1 else 0} candles.")
    t0 = time.time()
    c2 = fetch_live_gold_candles("M5", 100)
    print(f"Cached fetch took {time.time() - t0:.4f}s, got {len(c2) if c2 else 0} candles.")
