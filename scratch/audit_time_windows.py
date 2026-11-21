import pandas as pd
import numpy as np
from src.core.config import settings
from src.api.dependencies import get_prediction_engine
from src.api.routers.backtest import generate_strategy_signals_and_levels, RealisticBacktestEngine

engine = get_prediction_engine("v1.2.1")
df_feats = pd.read_parquet("data/features/xauusd/m5_features.parquet").sort_values("timestamp").reset_index(drop=True)

# Correct timestamp conversion using ms
ts = pd.to_datetime(df_feats["timestamp"], unit="ms", utc=True)
hours_cambodia = (ts.dt.hour + 7) % 24
year = ts.dt.year

print("Timestamp range:", ts.min(), "to", ts.max())
print("Years distribution:", year.value_counts().to_dict())

# Let's test different time sub-windows within Cambodia waking hours (7 AM - 11 PM ICT / 07:00 - 23:00):
# Window 1: Full waking day (7 AM - 11 PM ICT / 07:00 - 23:00)
# Window 2: London + NY (2 PM - 11 PM ICT / 14:00 - 23:00)
# Window 3: London Open + NY Open Killzones (2 PM - 5 PM & 7 PM - 11 PM ICT / 14:00-17:00 & 19:00-23:00)
# Window 4: Asian Morning Only (7 AM - 2 PM ICT / 07:00 - 14:00)
# Window 5: Evening Only (7 PM - 11 PM ICT / 19:00 - 23:00)

windows = {
    "Full Awake Day (7 AM - 11 PM ICT)": (hours_cambodia >= 7) & (hours_cambodia < 23),
    "London + NY Active (2 PM - 11 PM ICT)": (hours_cambodia >= 14) & (hours_cambodia < 23),
    "Killzones in Awake Hours (2-5 PM & 7-10 PM ICT)": ((hours_cambodia >= 14) & (hours_cambodia <= 17)) | ((hours_cambodia >= 19) & (hours_cambodia <= 22)),
    "Asian Morning (7 AM - 2 PM ICT)": (hours_cambodia >= 7) & (hours_cambodia < 14),
    "Evening Prime (7 PM - 11 PM ICT)": (hours_cambodia >= 19) & (hours_cambodia < 23),
}

# Generate raw ensemble signals
signals, sl_levels, tp_levels, model_name, sl_tp_logic = generate_strategy_signals_and_levels(
    "ensemble",
    df_feats,
    engine,
    threshold=0.50,
    spread=0.25,
    slippage=0.15,
    only_killzones=False,  # Let our window filters handle it
    target_rr=1.2,
    use_h1_gatekeeper=True,
    learn_from_mistakes=True,
    filter_high_impact_news=True,
    news_blackout_mins=30,
    exclude_asian_session=False,
    session_window="custom",
)

print("\n================ EVALUATION OF CAMBODIA TIME WINDOWS (MODEL E ENSEMBLE) ================")
for w_name, w_mask in windows.items():
    sig_w = [s if w_mask.iloc[i] else "WAIT" for i, s in enumerate(signals)]
    bt = RealisticBacktestEngine(
        initial_balance=10000.0,
        risk_pct_per_trade=0.01,
        spread_usd=0.25,
        slippage_usd=0.15,
        max_holding_bars=36,
    )
    res = bt.run(df_feats, signals=sig_w, sl_levels=sl_levels, tp_levels=tp_levels)
    
    trades_2026 = [t for t in res.trades if pd.to_datetime(t.entry_time).year == 2026]
    w_2026 = [t for t in trades_2026 if t.pnl_usd > 0]
    pnl_2026 = sum(t.pnl_usd for t in trades_2026)
    wr_2026 = (len(w_2026) / len(trades_2026) * 100) if trades_2026 else 0.0

    print(f"\nWindow: {w_name}")
    print(f"  Multi-Year: {res.total_trades} trades | Win Rate: {res.win_rate_pct:.2f}% | Return: ${res.total_net_profit:+,.2f} ({res.return_pct:+.1f}%) | Max DD: {res.max_drawdown_pct:.1f}% | PF: {res.profit_factor:.3f}")
    print(f"  2026 YTD:   {len(trades_2026)} trades | Win Rate: {wr_2026:.2f}% | Return: ${pnl_2026:+,.2f}")
