import duckdb
import pandas as pd
import numpy as np
from src.core.config import settings
from src.api.dependencies import get_prediction_engine
from src.api.routers.backtest import generate_strategy_signals_and_levels, RealisticBacktestEngine

engine = get_prediction_engine("v1.2.1")
df_feats = pd.read_parquet("data/features/xauusd/m5_features.parquet").sort_values("timestamp").reset_index(drop=True)

timestamps = pd.to_datetime(df_feats["timestamp"], unit="s", utc=True)
hours_cambodia = (timestamps.dt.hour + 7) % 24
idx_cambodia_day = (hours_cambodia >= 7) & (hours_cambodia < 23)
print(f"Total candles: {len(df_feats):,}")
print(f"Candles in Cambodia 7 AM - 11 PM: {idx_cambodia_day.sum():,} ({idx_cambodia_day.mean()*100:.1f}%)")

# Let's inspect 2026 performance specifically
idx_2026 = timestamps.dt.year == 2026
print(f"Candles in 2026: {idx_2026.sum():,}")

models = ["ensemble", "price_action", "volume_profile", "technical", "footprint"]

for m_key in models:
    # Test with Cambodia 7am - 11pm session
    signals, sl_levels, tp_levels, model_name, sl_tp_logic = generate_strategy_signals_and_levels(
        m_key,
        df_feats,
        engine,
        threshold=0.50,
        spread=0.25,
        slippage=0.15,
        only_killzones=True,
        target_rr=1.2,
        use_h1_gatekeeper=True,
        learn_from_mistakes=True,
        filter_high_impact_news=True,
        news_blackout_mins=30,
        exclude_asian_session=False,  # 7am Cambodia is Asian session start, so we shouldn't exclude Asian if user wants 7am
        session_window="cambodia_day",  # 7 AM - 11 PM
    )
    
    # We will modify backtest.py to support cambodia_day, but first let's see how signals match
    # For now, manually filter signals outside 7 AM - 11 PM Cambodia
    signals_filtered = []
    for i, sig in enumerate(signals):
        if idx_cambodia_day.iloc[i]:
            signals_filtered.append(sig)
        else:
            signals_filtered.append("WAIT")
            
    bt = RealisticBacktestEngine(
        initial_balance=10000.0,
        risk_pct_per_trade=0.01,
        spread_usd=0.25,
        slippage_usd=0.15,
        max_holding_bars=36,
    )
    res = bt.run(df_feats, signals=signals_filtered, sl_levels=sl_levels, tp_levels=tp_levels)
    
    # Also evaluate 2026 subset
    res_2026_trades = [t for t in res.trades if pd.to_datetime(t.entry_time).year == 2026]
    w_2026 = [t for t in res_2026_trades if t.pnl_usd > 0]
    pnl_2026 = sum(t.pnl_usd for t in res_2026_trades)
    wr_2026 = (len(w_2026) / len(res_2026_trades) * 100) if res_2026_trades else 0.0
    
    print(f"\n--- {m_key.upper()} ({model_name}) ---")
    print(f"Multi-Year: {res.total_trades} trades | WR: {res.win_rate_pct:.2f}% | Profit: ${res.total_net_profit:+,.2f} ({res.return_pct:+.1f}%) | Max DD: {res.max_drawdown_pct:.1f}% | PF: {res.profit_factor:.3f}")
    print(f"2026 YTD:   {len(res_2026_trades)} trades | WR: {wr_2026:.2f}% | Profit: ${pnl_2026:+,.2f}")
