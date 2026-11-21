import pandas as pd
import numpy as np
from src.core.config import settings
from src.api.dependencies import get_prediction_engine
from src.api.routers.backtest import generate_strategy_signals_and_levels, RealisticBacktestEngine

engine = get_prediction_engine("v1.2.1")
df_feats = pd.read_parquet("data/features/xauusd/m5_features.parquet").sort_values("timestamp").reset_index(drop=True)

raw_ts = df_feats["timestamp"].to_numpy()
if raw_ts[0] > 1e11:
    sec_ts = raw_ts // 1000
else:
    sec_ts = raw_ts

ts = pd.to_datetime(sec_ts, unit="s", utc=True)
hours_utc = ts.hour
hours_cambodia = (hours_utc + 7) % 24

# Generate unconstrained ensemble signals across all bars
signals, sl_levels, tp_levels, model_name, _ = generate_strategy_signals_and_levels(
    "ensemble",
    df_feats,
    engine,
    threshold=0.50,
    spread=0.25,
    slippage=0.15,
    only_killzones=False,
    target_rr=1.2,
    use_h1_gatekeeper=True,
    learn_from_mistakes=True,
    filter_high_impact_news=True,
    news_blackout_mins=30,
    exclude_asian_session=False,
    session_window="all_sessions",
)

bt = RealisticBacktestEngine(
    initial_balance=10000.0,
    risk_pct_per_trade=0.01,
    spread_usd=0.25,
    slippage_usd=0.15,
    max_holding_bars=36,
)
res = bt.run(df_feats, signals=signals, sl_levels=sl_levels, tp_levels=tp_levels)

print(f"Total trades unfiltered: {res.total_trades}")

# Break down each trade by Cambodia hour of entry
trades_by_hour = {h: [] for h in range(24)}
for t in res.trades:
    entry_ts = pd.to_datetime(t.entry_time, unit="s", utc=True)
    h_cambodia = (entry_ts.hour + 7) % 24
    trades_by_hour[h_cambodia].append(t)

print("\n=== HOURLY PROFITABILITY BREAKDOWN (CAMBODIA TIME UTC+7) ===")
print(f"{'Hour (Cambodia)':<18} | {'Session Description':<25} | {'Trades':<7} | {'Win Rate':<9} | {'Net PnL ($)':<12} | {'Verdict'}")
print("-" * 85)

session_names = {
    0: "Late NY (Choppy)",
    1: "Market Quiet / Roll",
    2: "Sydney Open",
    3: "Sydney Early",
    4: "Tokyo Pre-Market",
    5: "Tokyo Open (Tight)",
    6: "Tokyo Morning",
    7: "Asian Morning (7 AM)",
    8: "Asian Morning (8 AM)",
    9: "Asian Mid-day (9 AM)",
    10: "Asian Mid-day (10 AM)",
    11: "Asian Lunch (11 AM)",
    12: "Asian Afternoon (12 PM)",
    13: "Asian Late (1 PM)",
    14: "Frankfurt / London Pre (2 PM)",
    15: "London Open (3 PM)",
    16: "London Mid (4 PM)",
    17: "London Trend (5 PM)",
    18: "London / NY Pre (6 PM)",
    19: "NY Open Overlap (7 PM)",
    20: "NY Peak Volume (8 PM)",
    21: "NY Trend Expansion (9 PM)",
    22: "NY Session (10 PM)",
    23: "NY Afternoon (11 PM)",
}

awake_hours = list(range(7, 24))  # 7 AM to 11 PM

total_awake_trades = 0
total_awake_wins = 0
total_awake_pnl = 0.0

for h in range(24):
    trs = trades_by_hour[h]
    n_tr = len(trs)
    if n_tr == 0:
        continue
    wins = sum(1 for t in trs if t.pnl_usd > 0)
    wr = wins / n_tr * 100
    pnl = sum(t.pnl_usd for t in trs)
    
    if h in awake_hours:
        total_awake_trades += n_tr
        total_awake_wins += wins
        total_awake_pnl += pnl
        is_awake = "👁️ AWAKE"
    else:
        is_awake = "💤 ASLEEP"
        
    verdict = "✅ PROFITABLE" if pnl > 0 else "❌ UNPROFITABLE"
    h_str = f"{h:02d}:00 ({'AM' if h < 12 else 'PM'})"
    print(f"{h_str:<18} | {session_names.get(h, ''):<25} | {n_tr:<7} | {wr:6.2f}%   | ${pnl:+10.2f}  | {verdict} {is_awake}")

print("-" * 85)
wr_awake = (total_awake_wins / total_awake_trades * 100) if total_awake_trades else 0
print(f"TOTAL AWAKE HOURS (7 AM - 11 PM): {total_awake_trades} trades | Win Rate: {wr_awake:.2f}% | Net PnL: ${total_awake_pnl:+,.2f}")
