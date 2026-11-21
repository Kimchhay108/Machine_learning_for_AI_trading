import json
import urllib.request
import math

url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=5m&range=5d"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=5) as resp:
    data = json.loads(resp.read().decode())
res = data["chart"]["result"][0]
ts = res.get("timestamp", [])
q = res["indicators"]["quote"][0]
opens = q.get("open", [])
highs = q.get("high", [])
lows = q.get("low", [])
closes = q.get("close", [])
vols = q.get("volume", [])

candles = []
for i in range(len(ts)):
    t = ts[i]
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
    })

print(f"Valid parsed candles: {len(candles)}")
print(f"First: {candles[0]}")
print(f"Last: {candles[-1]}")
