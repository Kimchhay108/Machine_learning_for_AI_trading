from src.api.routers.backtest import run_backtest
from src.api.schemas import BacktestRequest

models = ['ensemble', 'technical', 'price_action', 'volume_profile', 'footprint', 'smc_ict']
results = {}

for m in models:
    req = BacktestRequest(session_window='cambodia_evening', model_type=m, target_rr=1.2)
    res = run_backtest(req)
    recent_2026 = [x for x in (res.monthly_breakdown or []) if x.year == 2026]
    pnl_2026 = sum(x.net_profit for x in recent_2026)
    trades_2026 = sum(x.total_trades for x in recent_2026)
    wins_2026 = sum(x.winning_trades for x in recent_2026)
    wr_2026 = (wins_2026 / trades_2026 * 100) if trades_2026 > 0 else 0
    results[m] = {
        'all_trades': res.total_trades,
        'all_wr': res.win_rate_pct,
        'all_return': res.return_pct,
        'all_profit': res.total_net_profit,
        'pf': res.profit_factor,
        'max_dd': res.max_drawdown_pct,
        'trades_2026': trades_2026,
        'wr_2026': wr_2026,
        'pnl_2026': pnl_2026,
    }

print("=== MODEL PERFORMANCE AUDIT (Cambodia Evening 7-11 PM, 1:1.2 RR) ===")
for m, r in sorted(results.items(), key=lambda x: x[1]['all_profit'], reverse=True):
    print(f"{m:15}: MultiYear: {r['all_trades']} tr, WR {r['all_wr']:.2f}%, Profit +${r['all_profit']:.2f} ({r['all_return']:.1f}%), PF {r['pf']:.2f}, MaxDD {r['max_dd']:.1f}% | 2026: {r['trades_2026']} tr, WR {r['wr_2026']:.1f}%, PnL +${r['pnl_2026']:.2f}")
