import numpy as np
import pandas as pd
from src.api.dependencies import get_prediction_engine
from src.backtest.engine import RealisticBacktestEngine
from src.api.routers.backtest import generate_strategy_signals_and_levels
from src.api.schemas import BacktestRequest

engine = get_prediction_engine()
feat_df = pd.read_parquet('data/features/xauusd/m5_features.parquet')

# Baseline Ensemble with current logic
signals, sl_levels, tp_levels, _, _ = generate_strategy_signals_and_levels(
    model_type="ensemble", test_df=feat_df, engine=engine, threshold=0.50, spread=0.20, slippage=0.05, target_rr=1.2, session_window="cambodia_evening"
)
bt = RealisticBacktestEngine(initial_balance=10000, spread_usd=0.20, slippage_usd=0.05, post_loss_cooldown_bars=6)
res_base = bt.run(feat_df, signals=signals, sl_levels=sl_levels, tp_levels=tp_levels)
print(f"Base Ensemble: Trades={res_base.total_trades}, WR={res_base.win_rate_pct:.2f}%, Profit=${res_base.total_net_profit:.2f}, Return={res_base.return_pct:.2f}%, PF={res_base.profit_factor:.2f}")

# Now let us test Ensemble consensus where we replace Model B with Model F and Model G!
pred_a = engine.ensemble.model_a.predict(feat_df)
pred_d = engine.ensemble.model_d.predict(feat_df)
pred_f = engine.ensemble.model_f.predict(feat_df)
pred_g = engine.ensemble.model_g.predict(feat_df)

# Pure 4-pillar profitable agreement (A, D, F, G)
agreed_bull_4 = ((pred_a == 2).astype(int) + (pred_d == 2).astype(int) + (pred_f == 2).astype(int) + (pred_g == 2).astype(int)) >= 2
agreed_bear_4 = ((pred_a == 0).astype(int) + (pred_d == 0).astype(int) + (pred_f == 0).astype(int) + (pred_g == 0).astype(int)) >= 2

# Test with 3-agreement or 2-agreement
for min_agree in [2, 3]:
    ag_b = ((pred_a == 2).astype(int) + (pred_d == 2).astype(int) + (pred_f == 2).astype(int) + (pred_g == 2).astype(int)) >= min_agree
    ag_s = ((pred_a == 0).astype(int) + (pred_d == 0).astype(int) + (pred_f == 0).astype(int) + (pred_g == 0).astype(int)) >= min_agree

    # Let us run backtest with this refined consensus
    # We will modify signals where agreement matches
    sigs_refined = []
    sls = []
    tps = []
    n = len(feat_df)
    close = feat_df['close'].to_numpy()
    atr = feat_df.get('atr_14', pd.Series(2.0, index=feat_df.index)).to_numpy()
    atr = np.where(np.isnan(atr) | (atr <= 0), 2.0, atr)

    # Time filters: Cambodia evening
    first_ts = feat_df['timestamp'].iloc[0]
    unit = 'ms' if first_ts > 1e11 else 's'
    dt = pd.to_datetime(feat_df['timestamp'], unit=unit, utc=True)
    hours = dt.dt.hour.to_numpy()
    hours_cambodia = (hours + 7) % 24
    in_window = (hours_cambodia >= 19) & (hours_cambodia < 23)

    # Volatility veto
    rolling_atr = pd.Series(atr).rolling(50, min_periods=5).mean().bfill().to_numpy()
    atr_ratio = np.where(rolling_atr > 0, atr / rolling_atr, 1.0)
    vol_ok = (atr_ratio <= 1.40)

    # H1 trend
    h1_close = feat_df.get('close_h1', pd.Series(close, index=feat_df.index)).to_numpy()
    h1_ema200 = feat_df.get('ema_200_h1', pd.Series(close, index=feat_df.index)).to_numpy()
    h1_bull = h1_close >= h1_ema200
    h1_bear = h1_close <= h1_ema200

    decisions, probs, _ = engine.ensemble.predict_decision(feat_df)
    p_bear_e, p_bull_e = probs[:, 0], probs[:, 2]

    for i in range(n):
        cp, ca = close[i], atr[i]
        sig = decisions[i]
        prob_buy = (sig == "BUY") or (p_bull_e[i] >= 0.44 and p_bull_e[i] > p_bear_e[i])
        prob_sell = (sig == "SELL") or (p_bear_e[i] >= 0.38 and p_bear_e[i] > p_bull_e[i])

        is_buy = in_window[i] and h1_bull[i] and vol_ok[i] and ag_b[i] and prob_buy
        is_sell = in_window[i] and h1_bear[i] and vol_ok[i] and ag_s[i] and prob_sell

        if is_buy and not is_sell:
            sigs_refined.append("BUY")
            ep = cp + 0.20 + 0.05
            sls.append(ep - 1.5 * ca)
            tps.append(ep + 1.2 * 1.5 * ca)
        elif is_sell and not is_buy:
            sigs_refined.append("SELL")
            ep = cp - 0.05
            sls.append(ep + 1.5 * ca)
            tps.append(ep - 1.2 * 1.5 * ca)
        else:
            sigs_refined.append("WAIT")
            sls.append(cp - 1.5 * ca)
            tps.append(cp + 1.2 * 1.5 * ca)

    res_ref = bt.run(feat_df, signals=sigs_refined, sl_levels=sls, tp_levels=tps)
    print(f"Refined Ensemble (No Model B, Min Agree={min_agree}/4): Trades={res_ref.total_trades}, WR={res_ref.win_rate_pct:.2f}%, Profit=${res_ref.total_net_profit:.2f}, Return={res_ref.return_pct:.2f}%, PF={res_ref.profit_factor:.2f}, MaxDD={res_ref.max_drawdown_pct:.2f}%")
