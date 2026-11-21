import pandas as pd
import numpy as np
from src.core.config import settings
from src.api.dependencies import get_prediction_engine
from src.api.routers.backtest import generate_strategy_signals_and_levels, RealisticBacktestEngine

engine = get_prediction_engine("v1.2.1")
df_feats = pd.read_parquet("data/features/xauusd/m5_features.parquet").sort_values("timestamp").reset_index(drop=True)

# Generate with FULL INSTITUTIONAL FILTERS:
# 1. H1 trend gatekeeper = True
# 2. Learn from mistakes = True
# 3. High impact news filter = True
# 4. Strict sub-model consensus = True
# 5. Volatility filter = True
# 6. Cambodia waking window: 7 AM - 11 PM (hours_cambodia in [7..22])

raw_ts = df_feats["timestamp"].to_numpy()
sec_ts = raw_ts // 1000 if raw_ts[0] > 1e11 else raw_ts
ts = pd.to_datetime(sec_ts, unit="s", utc=True)
hours_cambodia = (ts.hour + 7) % 24
idx_cambodia_day = (hours_cambodia >= 7) & (hours_cambodia < 23)

models = ["ensemble", "volume_profile", "price_action", "technical", "footprint"]

print("\n=== TESTING FULL INSTITUTIONAL FILTERS ON CAMBODIA 7 AM - 11 PM ===")
for m_key in models:
    signals, sl_levels, tp_levels, model_name, sl_tp_logic = generate_strategy_signals_and_levels(
        m_key,
        df_feats,
        engine,
        threshold=0.50,
        spread=0.25,
        slippage=0.15,
        only_killzones=False,  # We apply the 7 AM - 11 PM window filter directly
        target_rr=1.2,
        use_h1_gatekeeper=True,
        learn_from_mistakes=True,
        filter_high_impact_news=True,
        news_blackout_mins=30,
        exclude_asian_session=False,
        strict_structural_entry=True,  # Test strict consensus / structural entry
    )
    
    # Filter strictly to Cambodia 7 AM - 11 PM (07:00 - 23:00 ICT)
    sig_day = [s if idx_cambodia_day[i] else "WAIT" for i, s in enumerate(signals)]
    
    bt = RealisticBacktestEngine(
        initial_balance=10000.0,
        risk_pct_per_trade=0.01,
        spread_usd=0.25,
        slippage_usd=0.15,
        max_holding_bars=36,
    )
    res = bt.run(df_feats, signals=sig_day, sl_levels=sl_levels, tp_levels=tp_levels)
    
    # 2026 trades
    trades_2026 = [t for t in res.trades if pd.to_datetime(t.entry_time, unit="s", utc=True).year == 2026]
    w_2026 = [t for t in trades_2026 if t.pnl_usd > 0]
    pnl_2026 = sum(t.pnl_usd for t in trades_2026)
    wr_2026 = (len(w_2026) / len(trades_2026) * 100) if trades_2026 else 0.0

    print(f"\n[{m_key.upper()}] (Strict Structural / Multi-Model Consensus)")
    print(f"  Multi-Year: {res.total_trades} trades | Win Rate: {res.win_rate_pct:.2f}% | Net Profit: ${res.total_net_profit:+,.2f} ({res.return_pct:+.1f}%) | Max DD: {res.max_drawdown_pct:.1f}% | PF: {res.profit_factor:.3f}")
    print(f"  2026 YTD:   {len(trades_2026)} trades | Win Rate: {wr_2026:.2f}% | Net Profit: ${pnl_2026:+,.2f}")
