import pandas as pd
import numpy as np
from src.core.config import settings
from src.api.dependencies import get_prediction_engine
from src.api.routers.backtest import generate_strategy_signals_and_levels, RealisticBacktestEngine

engine = get_prediction_engine("v1.2.1")
df_feats = pd.read_parquet("data/features/xauusd/m5_features.parquet").sort_values("timestamp").reset_index(drop=True)

raw_ts = df_feats["timestamp"].to_numpy()
sec_ts = raw_ts // 1000 if raw_ts[0] > 1e11 else raw_ts
ts = pd.to_datetime(sec_ts, unit="s", utc=True)
hours_cambodia = (ts.hour + 7) % 24
idx_cambodia_day = (hours_cambodia >= 7) & (hours_cambodia < 23)

atr = df_feats["atr_14"].to_numpy() if "atr_14" in df_feats else df_feats["atr"].to_numpy()

# Test different Asian session filters for 7 AM - 2 PM Cambodia (hours 7..13)
# versus European/NY session (hours 14..22)

for min_asian_atr in [0.0, 1.0, 1.2, 1.5]:
    for asian_thresh in [0.50, 0.55, 0.60]:
        signals, sl_levels, tp_levels, _, _ = generate_strategy_signals_and_levels(
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
        )
        
        # Apply adaptive filter for Cambodia day:
        # In hours 7..13 (Asian morning), require higher ATR and higher threshold
        # In hours 14..22 (London & NY), standard institutional entry
        sig_adapted = []
        for i, s in enumerate(signals):
            if not idx_cambodia_day[i]:
                sig_adapted.append("WAIT")
                continue
            h = hours_cambodia[i]
            if 7 <= h < 14:
                # Asian morning
                if min_asian_atr > 0 and atr[i] < min_asian_atr:
                    sig_adapted.append("WAIT")
                    continue
                # If higher confidence required
                if asian_thresh > 0.50:
                    # In higher confidence, check if signal is very strong
                    pass
            sig_adapted.append(s)
            
        bt = RealisticBacktestEngine(
            initial_balance=10000.0,
            risk_pct_per_trade=0.01,
            spread_usd=0.25,
            slippage_usd=0.15,
            max_holding_bars=36,
        )
        res = bt.run(df_feats, signals=sig_adapted, sl_levels=sl_levels, tp_levels=tp_levels)
        if res.total_net_profit > 1000 or res.win_rate_pct >= 48:
            print(f"Asian Min ATR: ${min_asian_atr:.2f} | Asian Thresh: {asian_thresh} -> Trades: {res.total_trades} | WR: {res.win_rate_pct:.2f}% | Profit: ${res.total_net_profit:+,.2f} ({res.return_pct:+.1f}%) | Max DD: {res.max_drawdown_pct:.1f}%")
        break
