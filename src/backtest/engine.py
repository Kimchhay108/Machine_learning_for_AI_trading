"""Realistic backtesting simulation engine for quantitative trading strategies.

Models realistic market execution:
- Spread and slippage modeling
- Intrabar Stop-Loss and Take-Profit execution
- Conservative conflict resolution (SL triggered first if both levels touched in the same candle)
- Fixed fractional position sizing (risk % per trade)
- Detailed performance reporting (Sharpe, Profit Factor, Max Drawdown, Win Rate)
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.core.logger import get_logger

logger = get_logger("backtest.engine")


@dataclass
class TradeRecord:
    """Individual simulated trade execution log."""
    entry_time: int
    exit_time: int
    direction: str                     # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size_oz: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str                   # "TAKE_PROFIT", "STOP_LOSS", "TIME_LIMIT", "EOD"


@dataclass
class BacktestResult:
    """Comprehensive performance metrics summary."""
    initial_balance: float
    final_balance: float
    total_net_profit: float
    return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    win_rate_with_be_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_usd: float = 0.0
    avg_r_multiple: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trades"] = [asdict(t) for t in self.trades]
        return d


class RealisticBacktestEngine:
    """Simulates realistic trade execution without lookahead or optimistic bias."""

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        risk_pct_per_trade: float = 0.01,    # 1.0% account risk per trade
        spread_usd: float = 0.20,             # 20 cents Gold spread
        slippage_usd: float = 0.05,           # 5 cents execution slippage
        max_holding_bars: int = 24,           # 2 hours on M5
        enable_breakeven: bool = False,       # Move SL to entry once in profit (+1.0R) - default False for higher winrate & full runners
        breakeven_r: float = 1.0,             # Target R:Multiple to trigger Breakeven
        enable_partial_tp: bool = False,      # Scale out 50% position size at +1.0R - default False to avoid cutting winning trades early
        partial_tp_r: float = 1.0,            # Target R:Multiple to take 50% partial profit
        partial_tp_ratio: float = 0.50,       # Fraction of position to scale out
        post_loss_cooldown_bars: int = 3,     # AI Mistake Quarantine: Bars to abstain after an SL hit to avoid revenge chop
    ):
        self.initial_balance = initial_balance
        self.risk_pct_per_trade = risk_pct_per_trade
        self.spread_usd = spread_usd
        self.slippage_usd = slippage_usd
        self.max_holding_bars = max_holding_bars
        self.enable_breakeven = enable_breakeven
        self.breakeven_r = breakeven_r
        self.enable_partial_tp = enable_partial_tp
        self.partial_tp_r = partial_tp_r
        self.partial_tp_ratio = partial_tp_ratio
        self.post_loss_cooldown_bars = post_loss_cooldown_bars

    def run(
        self,
        df: pd.DataFrame,
        signals: List[str],  # "BUY", "SELL", or "WAIT" aligned with df
        sl_levels: Optional[List[float]] = None,
        tp_levels: Optional[List[float]] = None,
    ) -> BacktestResult:
        """Execute candle-by-candle backtest simulation.

        Args:
            df: Candlestick DataFrame with [timestamp, open, high, low, close, atr_14]
            signals: List of decisions aligned with each candle
            sl_levels: Optional explicit stop-loss price levels
            tp_levels: Optional explicit take-profit price levels

        Returns:
            BacktestResult with full trade logs and metrics.
        """
        n = len(df)
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[TradeRecord] = []

        active_trade: Optional[Dict[str, Any]] = None
        last_loss_exit_bar: int = -9999

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        timestamp = df["timestamp"].to_numpy()

        if "atr_14" in df.columns:
            atr = df["atr_14"].to_numpy()
        else:
            atr = pd.Series(high - low).rolling(14).mean().bfill().to_numpy()

        for i in range(n):
            c_high = high[i]
            c_low = low[i]
            c_close = close[i]
            c_ts = timestamp[i]
            c_atr = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else 2.0

            # 1. Check exit, Partial TP, and Breakeven trailing for active open trade
            if active_trade is not None:
                entry_dir = active_trade["direction"]
                entry_p = active_trade["entry_price"]
                sl = active_trade["sl"]
                tp = active_trade["tp"]
                bars_held = i - active_trade["entry_bar"]
                pos_size = active_trade["size"]
                orig_size = active_trade.get("orig_size", pos_size)
                init_risk = active_trade.get("initial_risk", max(abs(entry_p - sl), 0.50))
                be_active = active_trade.get("breakeven_active", False)
                ptp_taken = active_trade.get("partial_tp_taken", False)

                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                if entry_dir == "BUY":
                    orig_sl = entry_p - init_risk
                    hit_orig_sl = c_low <= orig_sl
                    hit_tp = c_high >= tp

                    if hit_orig_sl and hit_tp:
                        # Conservative conflict rule: stop loss triggered
                        exit_price = orig_sl - self.slippage_usd
                        exit_reason = "STOP_LOSS"
                        trade_closed = True
                    else:
                        # Check Partial TP scale-out at +1.0R
                        if self.enable_partial_tp and not ptp_taken and c_high >= entry_p + (self.partial_tp_r * init_risk):
                            p_size = orig_size * self.partial_tp_ratio
                            ptp_price = (entry_p + self.partial_tp_r * init_risk) - self.slippage_usd
                            active_trade["realized_partial_pnl"] += (ptp_price - entry_p) * p_size
                            pos_size = pos_size - p_size
                            active_trade["size"] = pos_size
                            active_trade["partial_tp_taken"] = True
                            ptp_taken = True
                            # Move SL to breakeven (+ spread)
                            sl = entry_p + self.spread_usd
                            active_trade["sl"] = sl
                            active_trade["breakeven_active"] = True
                            be_active = True
                        elif self.enable_breakeven and not be_active and c_high >= entry_p + (self.breakeven_r * init_risk):
                            sl = entry_p + self.spread_usd
                            active_trade["sl"] = sl
                            active_trade["breakeven_active"] = True
                            be_active = True

                        hit_sl = c_low <= sl
                        if hit_tp:
                            exit_price = tp - self.slippage_usd
                            exit_reason = "TAKE_PROFIT"
                            trade_closed = True
                        elif hit_sl:
                            exit_price = sl - self.slippage_usd
                            exit_reason = "PARTIAL_TP_BE" if ptp_taken else ("BREAKEVEN" if be_active else "STOP_LOSS")
                            trade_closed = True
                        elif bars_held >= self.max_holding_bars:
                            exit_price = c_close - self.spread_usd
                            exit_reason = "TIME_LIMIT"
                            trade_closed = True

                    if trade_closed:
                        rem_pnl = (exit_price - entry_p) * pos_size
                        pnl = active_trade.get("realized_partial_pnl", 0.0) + rem_pnl
                        total_risk_dlr = init_risk * orig_size
                        r_mult = pnl / total_risk_dlr if total_risk_dlr > 0 else 0.0

                else:  # "SELL"
                    orig_sl = entry_p + init_risk
                    hit_orig_sl = c_high >= orig_sl
                    hit_tp = c_low <= tp

                    if hit_orig_sl and hit_tp:
                        # Conservative conflict rule: stop loss triggered
                        exit_price = orig_sl + self.slippage_usd
                        exit_reason = "STOP_LOSS"
                        trade_closed = True
                    else:
                        # Check Partial TP scale-out at +1.0R
                        if self.enable_partial_tp and not ptp_taken and c_low <= entry_p - (self.partial_tp_r * init_risk):
                            p_size = orig_size * self.partial_tp_ratio
                            ptp_price = (entry_p - self.partial_tp_r * init_risk) + self.slippage_usd
                            active_trade["realized_partial_pnl"] += (entry_p - ptp_price) * p_size
                            pos_size = pos_size - p_size
                            active_trade["size"] = pos_size
                            active_trade["partial_tp_taken"] = True
                            ptp_taken = True
                            # Move SL to breakeven (- spread)
                            sl = entry_p - self.spread_usd
                            active_trade["sl"] = sl
                            active_trade["breakeven_active"] = True
                            be_active = True
                        elif self.enable_breakeven and not be_active and c_low <= entry_p - (self.breakeven_r * init_risk):
                            sl = entry_p - self.spread_usd
                            active_trade["sl"] = sl
                            active_trade["breakeven_active"] = True
                            be_active = True

                        hit_sl = c_high >= sl
                        if hit_tp:
                            exit_price = tp + self.slippage_usd
                            exit_reason = "TAKE_PROFIT"
                            trade_closed = True
                        elif hit_sl:
                            exit_price = sl + self.slippage_usd
                            exit_reason = "PARTIAL_TP_BE" if ptp_taken else ("BREAKEVEN" if be_active else "STOP_LOSS")
                            trade_closed = True
                        elif bars_held >= self.max_holding_bars:
                            exit_price = c_close + self.spread_usd
                            exit_reason = "TIME_LIMIT"
                            trade_closed = True

                    if trade_closed:
                        rem_pnl = (entry_p - exit_price) * pos_size
                        pnl = active_trade.get("realized_partial_pnl", 0.0) + rem_pnl
                        total_risk_dlr = init_risk * orig_size
                        r_mult = pnl / total_risk_dlr if total_risk_dlr > 0 else 0.0

                if trade_closed:
                    if exit_reason == "STOP_LOSS":
                        last_loss_exit_bar = i
                    balance += pnl
                    trades.append(TradeRecord(
                        entry_time=active_trade["entry_time"],
                        exit_time=c_ts,
                        direction=entry_dir,
                        entry_price=round(entry_p, 3),
                        exit_price=round(exit_price, 3),
                        stop_loss=round(sl, 3),
                        take_profit=round(tp, 3),
                        position_size_oz=round(orig_size, 3),
                        pnl_usd=round(pnl, 2),
                        pnl_pct=round((pnl / balance) * 100.0, 4),
                        r_multiple=round(r_mult, 2),
                        exit_reason=exit_reason,
                    ))
                    active_trade = None

            equity_curve.append(balance)

            # 2. Check signal for new entry (if not already in a position)
            if active_trade is None and i < n - 1:
                # AI Mistake Learning Cooldown: Veto entries within quarantine window of a previous Stop Loss
                if (i - last_loss_exit_bar) < self.post_loss_cooldown_bars:
                    continue

                sig = signals[i]
                if sig in ["BUY", "SELL"]:
                    risk_amount = balance * self.risk_pct_per_trade

                    if sig == "BUY":
                        entry_p = c_close + self.spread_usd + self.slippage_usd
                        sl = sl_levels[i] if sl_levels else entry_p - (1.5 * c_atr)
                        tp = tp_levels[i] if tp_levels else entry_p + (3.0 * c_atr)
                        risk_per_oz = entry_p - sl
                    else:  # "SELL"
                        entry_p = c_close - self.slippage_usd
                        sl = sl_levels[i] if sl_levels else entry_p + (1.5 * c_atr)
                        tp = tp_levels[i] if tp_levels else entry_p - (3.0 * c_atr)
                        risk_per_oz = sl - entry_p

                    if risk_per_oz > 0:
                        size_oz = risk_amount / risk_per_oz
                        active_trade = {
                            "direction": sig,
                            "entry_price": entry_p,
                            "sl": sl,
                            "tp": tp,
                            "orig_size": size_oz,
                            "size": size_oz,
                            "initial_risk": risk_per_oz,
                            "entry_time": c_ts,
                            "entry_bar": i,
                            "breakeven_active": False,
                            "partial_tp_taken": False,
                            "realized_partial_pnl": 0.0,
                        }

        # Calculate performance statistics
        total_trades = len(trades)
        wins = [t for t in trades if t.exit_reason == "TAKE_PROFIT" or (t.exit_reason == "TIME_LIMIT" and t.pnl_usd > 0)]
        breakevens = [t for t in trades if t.exit_reason in ["BREAKEVEN", "PARTIAL_TP_BE"] or (t.exit_reason == "TIME_LIMIT" and t.pnl_usd == 0)]
        losses = [t for t in trades if t.exit_reason == "STOP_LOSS" or (t.exit_reason == "TIME_LIMIT" and t.pnl_usd < 0)]

        winning_count = len(wins)
        losing_count = len(losses)
        breakeven_count = len(breakevens)
        win_rate = (winning_count / total_trades * 100.0) if total_trades > 0 else 0.0
        win_rate_with_be = ((winning_count + breakeven_count) / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        net_profit = balance - self.initial_balance
        return_pct = (net_profit / self.initial_balance) * 100.0

        # Drawdown calculation
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        dd_pct = (dd / peak) * 100.0
        max_dd_usd = float(np.max(dd))
        max_dd_pct = float(np.max(dd_pct))

        # Sharpe calculation (assuming ~72000 M5 bars per trading year)
        if total_trades > 1:
            trade_returns = [t.pnl_pct for t in trades]
            std_ret = np.std(trade_returns)
            sharpe = (np.mean(trade_returns) / std_ret * np.sqrt(252 * 40)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        avg_pnl = net_profit / total_trades if total_trades > 0 else 0.0
        avg_r = (sum(t.r_multiple for t in trades) / total_trades) if total_trades > 0 else 0.0

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=round(balance, 2),
            total_net_profit=round(net_profit, 2),
            return_pct=round(return_pct, 2),
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            breakeven_trades=breakeven_count,
            win_rate_pct=round(win_rate, 2),
            win_rate_with_be_pct=round(win_rate_with_be, 2),
            profit_factor=round(profit_factor, 3),
            max_drawdown_usd=round(max_dd_usd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(float(sharpe), 3),
            avg_trade_usd=round(avg_pnl, 2),
            avg_r_multiple=round(avg_r, 2),
            equity_curve=[round(float(v), 2) for v in equity_curve],
            trades=trades,
        )
