"""Statistical labeling engine for quantitative trading machine learning.

Implements multiple prediction target formulations:
1. Multi-horizon directional returns (15m, 30m, 60m) with volatility deadbands
2. Marcos López de Prado's Triple-Barrier Method with transaction cost modeling
3. Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
4. Forward realized volatility and regime shifts
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT
from src.core.logger import get_logger

logger = get_logger("labels.engine")


class LabelingEngine:
    """Computes rigorous machine learning prediction targets from historical candles."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.LABELS_DIR / "xauusd"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def add_directional_returns(
        df: pd.DataFrame,
        horizons: Optional[Dict[str, int]] = None,
        deadband_atr_mult: float = 0.25,
    ) -> pd.DataFrame:
        """Compute forward percentage returns, binary direction, and volatility-deadband ternary labels.

        Args:
            df: DataFrame containing close prices and 'atr_14'
            horizons: Dict of label suffix to forward bars (default: 15m=3 bars, 30m=6 bars, 60m=12 bars on M5)
            deadband_atr_mult: Multiplier of ATR defining the neutral noise deadband

        Returns:
            DataFrame with forward returns and directional classification labels.
        """
        if horizons is None:
            horizons = {"15m": 3, "30m": 6, "60m": 12}

        res = df.copy()
        close = res["close"]

        # ATR proxy for deadband
        if "atr_14" in res.columns:
            atr = res["atr_14"]
        else:
            atr = (res["high"] - res["low"]).rolling(14).mean().bfill()

        deadband_pct = (deadband_atr_mult * atr / close).fillna(0.0005)

        for name, bars in horizons.items():
            fwd_close = close.shift(-bars)
            fwd_ret = (fwd_close - close) / close

            res[f"fwd_ret_{name}"] = fwd_ret
            # Binary label: 1 if positive return, 0 if negative
            res[f"label_dir_binary_{name}"] = (fwd_ret > 0).astype(int)

            # Ternary label: +1 (Bullish), -1 (Bearish), 0 (Neutral within noise deadband)
            ternary = pd.Series(0, index=res.index, dtype=int)
            ternary[fwd_ret > deadband_pct] = 1
            ternary[fwd_ret < -deadband_pct] = -1
            res[f"label_dir_ternary_{name}"] = ternary

        return res

    @staticmethod
    def add_triple_barrier_labels(
        df: pd.DataFrame,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        horizon_bars: int = 12,  # 60 minutes on M5
        trading_cost_usd: float = 0.20,  # 20 cents spread/slippage on Gold
    ) -> pd.DataFrame:
        """Compute Marcos López de Prado Triple-Barrier Labels for Long and Short trades.

        Barriers for long trade at index t:
        - Upper Barrier (TP): Close[t] + tp_mult * ATR[t]
        - Lower Barrier (SL): Close[t] - sl_mult * ATR[t]
        - Vertical Barrier: t + horizon_bars

        Labels:
        +1: Hit Upper Barrier (Take Profit)
        -1: Hit Lower Barrier (Stop Loss)
         0: Vertical Barrier expired before touching either level

        Conservative rule: If both barriers are breached within the same bar, trigger Stop Loss (-1).
        """
        res = df.copy()
        n = len(res)

        close = res["close"].to_numpy()
        high = res["high"].to_numpy()
        low = res["low"].to_numpy()

        if "atr_14" in res.columns:
            atr = res["atr_14"].to_numpy()
        else:
            atr = pd.Series(high - low).rolling(14).mean().bfill().to_numpy()

        tb_long = np.zeros(n, dtype=int)
        tb_short = np.zeros(n, dtype=int)
        mfe_long = np.zeros(n, dtype=float)
        mae_long = np.zeros(n, dtype=float)

        for t in range(n - horizon_bars):
            c_entry = close[t]
            c_atr = atr[t]
            if np.isnan(c_atr) or c_atr <= 0:
                continue

            # Long trade barriers (adjust for spread cost)
            tp_long = c_entry + (tp_mult * c_atr) + trading_cost_usd
            sl_long = c_entry - (sl_mult * c_atr) - trading_cost_usd

            # Short trade barriers
            tp_short = c_entry - (tp_mult * c_atr) - trading_cost_usd
            sl_short = c_entry + (sl_mult * c_atr) + trading_cost_usd

            long_event = 0
            short_event = 0

            max_fav_long = 0.0
            max_adv_long = 0.0

            # Scan forward window [t+1, t+horizon_bars]
            for step in range(1, horizon_bars + 1):
                fwd_high = high[t + step]
                fwd_low = low[t + step]

                # Update MFE / MAE
                max_fav_long = max(max_fav_long, fwd_high - c_entry)
                max_adv_long = min(max_adv_long, fwd_low - c_entry)

                # Long evaluation
                if long_event == 0:
                    tp_hit_long = fwd_high >= tp_long
                    sl_hit_long = fwd_low <= sl_long

                    if tp_hit_long and sl_hit_long:
                        long_event = -1  # Conservative stop out
                    elif tp_hit_long:
                        long_event = 1
                    elif sl_hit_long:
                        long_event = -1

                # Short evaluation
                if short_event == 0:
                    tp_hit_short = fwd_low <= tp_short
                    sl_hit_short = fwd_high >= sl_short

                    if tp_hit_short and sl_hit_short:
                        short_event = -1  # Conservative stop out
                    elif tp_hit_short:
                        short_event = 1
                    elif sl_hit_short:
                        short_event = -1

                if long_event != 0 and short_event != 0:
                    break

            tb_long[t] = long_event
            tb_short[t] = short_event
            mfe_long[t] = max_fav_long / c_atr
            mae_long[t] = max_adv_long / c_atr

        res["label_tb_long"] = tb_long
        res["label_tb_short"] = tb_short
        res["mfe_long_atr"] = mfe_long
        res["mae_long_atr"] = mae_long
        return res

    @classmethod
    def compute_all_labels(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the complete set of statistical training labels."""
        res = cls.add_directional_returns(df)
        res = cls.add_triple_barrier_labels(res)
        return res

    def save_labels(self, df: pd.DataFrame, filename: str = "m5_labels.parquet") -> Path:
        """Save computed label matrix to Parquet format."""
        file_path = self.output_dir / filename
        df.to_parquet(file_path, index=False, compression="snappy")
        logger.info(f"Saved labels matrix ({len(df)} rows, {len(df.columns)} cols) to {file_path}")
        return file_path
