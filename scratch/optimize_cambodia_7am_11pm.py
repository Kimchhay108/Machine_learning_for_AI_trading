import pandas as pd
import numpy as np
from src.core.config import settings
from src.api.dependencies import get_prediction_engine
from src.api.routers.backtest import generate_strategy_signals_and_levels, RealisticBacktestEngine

engine = get_prediction_engine("v1.2.1")
df_feats = pd.read_parquet("data/features/xauusd/m5_features.parquet").sort_values("timestamp").reset_index(drop=True)

# Timestamps in ms -> convert to seconds for datetime
ts = pd.to_datetime(df_feats["timestamp"] // 1000 if df_feats["timestamp"].iloc[0] > 1e11 else df_feats["timestamp"], unit="s", utc=True)
hours_cambodia = (ts.dt.hour + 7) % 24
idx_cambodia_7am_11pm = (hours_cambodia >= 7) & (hours_cambodia < 23)

print("Total bars in Cambodia 7 AM - 11 PM:", idx_cambodia_7am_11pm.sum())

# Let's test various configurations across Cambodia 7 AM - 11 PM
# 1. Models: ensemble, price_action, volume_profile, technical, footprint
# 2. Risk:Reward ratios: 1.0, 1.2, 1.5, 2.0
# 3. Confidence thresholds: 0.50, 0.55, 0.60
# 4. Strict structural entry / Volatility filter (min ATR)

for m_key in ["ensemble", "volume_profile", "price_action", "technical", "footprint"]:
    for rr in [1.0, 1.2, 1.5]:
        for thresh in [0.50, 0.55]:
            signals, sl_levels, tp_levels, model_name, _ = generate_strategy_signals_and_levels(
                m_key,
                df_feats,
                engine,
                threshold=thresh,
                spread=0.25,
                slippage=0.15,
                only_killzones=False,
                target_rr=rr,
                use_h1_gatekeeper=True,
                learn_from_mistakes=True,
                filter_high_impact_news=True,
                news_blackout_mins=30,
                exclude_asian_session=False,
                strict_structural_entry=False,
                session_window="cambodia_day",
            )
            
            # Apply 7 AM - 11 PM filter
            sig_filtered = [s if idx_cambodia_7am_11pm.iloc[i] else "WAIT" for i, s in enumerate(signals)]
            
            bt = RealisticBacktestEngine(
                initial_balance=10000.0,
                risk_pct_per_trade=0.01,
                spread_usd=0.25,
                slippage_usd=0.15,
                max_holding_bars=36,
            )
            res = bt.run(df_feats, signals=sig_filtered, sl_levels=sl_levels, tp_levels=tp_levels)
            
            # Check 2026 performance
            trades_2026 = [t for t in res.trades if pd.to_datetime(t.entry_time, unit="s", utc=True).year == 2026]
            w_2026 = [t for t in trades_2026 if t.pnl_usd > 0]
            pnl_2026 = sum(t.pnl_usd for t in trades_2026)
            wr_2026 = (len(w_2026) / len(trades_2026) * 100) if trades_2026 else 0.0
            
            if res.total_net_profit > 0 or pnl_2026 > 0 or res.win_rate_pct >= 50:
                print(f"[{m_key.upper()} | RR={rr} | Thresh={thresh}]")
                print(f"  Multi-Year: {res.total_trades} trades | WR: {res.win_rate_pct:.2f}% | Profit: ${res.total_net_profit:+,.2f} ({res.return_pct:+.1f}%) | PF: {res.profit_factor:.3f} | Max DD: {res.max_drawdown_pct:.1f}%")
                print(f"  2026:       {len(trades_2026)} trades | WR: {wr_2026:.2f}% | Profit: ${pnl_2026:+,.2f}")
