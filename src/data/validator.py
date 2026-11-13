"""Validation engine for financial candlestick time-series data.

Performs exhaustive mathematical, temporal, and economic integrity checks:
1. Strict schema compliance and column types
2. Monotonic chronological ordering (UTC timestamps)
3. Duplicate timestamp identification
4. OHLC geometric relationship consistency:
   - High >= Open, Close, Low
   - Low <= Open, Close, High
   - All prices strictly positive
5. Economic feasibility / outlier detection (bounded within physical market extremes)
6. Single-candle spike anomaly detection (>15% jump in 5 minutes)
7. Missing candle detection without fabrication (detecting weekend breaks vs intra-week dropouts)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.core.constants import (
    CANONICAL_COLUMNS,
    DEFAULT_INSTRUMENT,
    TIMEFRAME_MS,
    XAUUSD_MAX_ALLOWED_CANDLE_PERCENT_CHANGE,
    XAUUSD_MAX_PRICE,
    XAUUSD_MIN_PRICE,
)
from src.core.logger import get_logger

logger = get_logger("validator")


@dataclass
class CandleGap:
    """Represents an identified temporal gap in candle records."""
    start_timestamp: int
    end_timestamp: int
    start_utc: str
    end_utc: str
    duration_minutes: float
    expected_missing_candles: int
    is_weekend_gap: bool


@dataclass
class ValidationResult:
    """Structured report of candle validation run."""
    is_valid: bool
    total_candles: int
    valid_candles: int
    duplicate_count: int
    ohlc_violations: int
    price_range_violations: int
    anomaly_spikes: int
    volume_anomalies: int
    missing_gaps_count: int
    weekend_gaps_count: int
    intraweek_gaps_count: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    gaps: List[CandleGap] = field(default_factory=list)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to serializable dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_candles": self.total_candles,
            "valid_candles": self.valid_candles,
            "duplicate_count": self.duplicate_count,
            "ohlc_violations": self.ohlc_violations,
            "price_range_violations": self.price_range_violations,
            "anomaly_spikes": self.anomaly_spikes,
            "volume_anomalies": self.volume_anomalies,
            "missing_gaps_count": self.missing_gaps_count,
            "weekend_gaps_count": self.weekend_gaps_count,
            "intraweek_gaps_count": self.intraweek_gaps_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "gaps": [
                {
                    "start_utc": g.start_utc,
                    "end_utc": g.end_utc,
                    "duration_minutes": g.duration_minutes,
                    "expected_missing_candles": g.expected_missing_candles,
                    "is_weekend_gap": g.is_weekend_gap,
                }
                for g in self.gaps
            ],
            "summary_metrics": self.summary_metrics,
        }


class DataValidator:
    """Performs rigorous quality validation on candlestick DataFrames."""

    def __init__(
        self,
        min_price: float = XAUUSD_MIN_PRICE,
        max_price: float = XAUUSD_MAX_PRICE,
        max_percent_change: float = XAUUSD_MAX_ALLOWED_CANDLE_PERCENT_CHANGE,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.max_percent_change = max_percent_change

    @staticmethod
    def _is_weekend(ts_ms: int) -> bool:
        """Check if timestamp falls into standard FX/Commodity weekend closure (Fri 22:00 to Sun 22:00 UTC)."""
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        # Saturday is weekday 5
        if dt.weekday() == 5:
            return True
        # Friday after 22:00 UTC
        if dt.weekday() == 4 and dt.hour >= 22:
            return True
        # Sunday before 21:00 UTC
        if dt.weekday() == 6 and dt.hour < 21:
            return True
        return False

    def validate(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str = DEFAULT_INSTRUMENT,
        data_source: str = "dukascopy",
        strict: bool = False,
    ) -> Tuple[ValidationResult, pd.DataFrame]:
        """Validate candle DataFrame and return detailed report plus cleaned dataset.

        Args:
            df: Candlestick DataFrame
            timeframe: e.g. "m5", "m15", "h1"
            instrument: Symbol, default "XAUUSD"
            strict: If True, any warning or missing candle fails is_valid.

        Returns:
            Tuple of (ValidationResult, cleaned_and_sorted_DataFrame)
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or df.empty:
            res = ValidationResult(
                is_valid=False,
                total_candles=0,
                valid_candles=0,
                duplicate_count=0,
                ohlc_violations=0,
                price_range_violations=0,
                anomaly_spikes=0,
                volume_anomalies=0,
                missing_gaps_count=0,
                weekend_gaps_count=0,
                intraweek_gaps_count=0,
                errors=["Dataset is empty or None."],
            )
            return res, pd.DataFrame(columns=CANONICAL_COLUMNS)

        working_df = df.copy()

        # Check required columns
        for col in ["timestamp", "open", "high", "low", "close"]:
            if col not in working_df.columns:
                errors.append(f"Required column '{col}' is missing.")
        if errors:
            return ValidationResult(
                is_valid=False,
                total_candles=len(working_df),
                valid_candles=0,
                duplicate_count=0,
                ohlc_violations=0,
                price_range_violations=0,
                anomaly_spikes=0,
                volume_anomalies=0,
                missing_gaps_count=0,
                weekend_gaps_count=0,
                intraweek_gaps_count=0,
                errors=errors,
            ), working_df

        if "volume" not in working_df.columns:
            working_df["volume"] = 0.0
        if "instrument" not in working_df.columns:
            working_df["instrument"] = instrument
        if "timeframe" not in working_df.columns:
            working_df["timeframe"] = timeframe
        if "data_source" not in working_df.columns:
            working_df["data_source"] = data_source

        total_records = len(working_df)

        # 1. Duplicate Timestamps
        duplicate_mask = working_df.duplicated(subset=["timestamp"], keep="last")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate timestamp records. Kept latest entries.")
            working_df = working_df.drop_duplicates(subset=["timestamp"], keep="last")

        # 2. Chronological Ordering
        working_df = working_df.sort_values("timestamp").reset_index(drop=True)

        # 3. OHLC Mathematical Consistency
        # High must be >= Open, High >= Close, High >= Low
        # Low must be <= Open, Low <= Close, Low <= High
        ohlc_mask = (
            (working_df["high"] >= working_df["open"]) &
            (working_df["high"] >= working_df["close"]) &
            (working_df["high"] >= working_df["low"]) &
            (working_df["low"] <= working_df["open"]) &
            (working_df["low"] <= working_df["close"]) &
            (working_df["low"] > 0)
        )
        ohlc_violations = int((~ohlc_mask).sum())
        if ohlc_violations > 0:
            errors.append(f"Found {ohlc_violations} candles violating OHLC geometric relationships.")

        # 4. Economic Price Range Bounds
        price_bound_mask = (
            (working_df["open"] >= self.min_price) & (working_df["open"] <= self.max_price) &
            (working_df["high"] >= self.min_price) & (working_df["high"] <= self.max_price) &
            (working_df["low"] >= self.min_price) & (working_df["low"] <= self.max_price) &
            (working_df["close"] >= self.min_price) & (working_df["close"] <= self.max_price)
        )
        price_range_violations = int((~price_bound_mask).sum())
        if price_range_violations > 0:
            errors.append(
                f"Found {price_range_violations} candles with prices outside economic range "
                f"[{self.min_price}, {self.max_price}]."
            )

        # 5. Volume Anomalies
        volume_mask = (working_df["volume"] >= 0) & (~working_df["volume"].isna())
        volume_anomalies = int((~volume_mask).sum())
        if volume_anomalies > 0:
            warnings.append(f"Found {volume_anomalies} negative or NaN volume records.")
            working_df.loc[~volume_mask, "volume"] = 0.0

        # 6. Single-Candle Anomaly Spikes
        pct_change = (working_df["high"] - working_df["low"]) / working_df["open"].replace(0, np.nan)
        spike_mask = pct_change > self.max_percent_change
        anomaly_spikes = int(spike_mask.sum())
        if anomaly_spikes > 0:
            warnings.append(
                f"Found {anomaly_spikes} candles with extreme price span >{self.max_percent_change*100:.0f}%."
            )

        # 7. Missing Candle Gaps Detection
        expected_interval_ms = TIMEFRAME_MS.get(timeframe.lower(), 300_000)
        time_diffs = working_df["timestamp"].diff().dropna()

        gaps: List[CandleGap] = []
        weekend_gaps = 0
        intraweek_gaps = 0

        gap_indices = working_df.index[1:][time_diffs > (expected_interval_ms * 1.5)]

        for idx in gap_indices:
            start_ts = int(working_df.loc[idx - 1, "timestamp"])
            end_ts = int(working_df.loc[idx, "timestamp"])
            diff_ms = end_ts - start_ts
            duration_min = diff_ms / 60000.0
            expected_missing = int(round(diff_ms / expected_interval_ms)) - 1

            start_utc = datetime.fromtimestamp(start_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            end_utc = datetime.fromtimestamp(end_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            is_weekend = self._is_weekend(start_ts) or self._is_weekend(end_ts)
            if is_weekend:
                weekend_gaps += 1
            else:
                intraweek_gaps += 1

            gaps.append(CandleGap(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                start_utc=start_utc,
                end_utc=end_utc,
                duration_minutes=duration_min,
                expected_missing_candles=expected_missing,
                is_weekend_gap=is_weekend,
            ))

        if intraweek_gaps > 0:
            warnings.append(
                f"Identified {intraweek_gaps} intra-week missing candle gaps (non-weekend market pauses/dropouts)."
            )

        # Filter strictly valid rows for the cleaned dataset
        valid_rows_mask = ohlc_mask & price_bound_mask
        cleaned_df = working_df[valid_rows_mask].copy()

        # Build summary metrics
        summary_metrics = {
            "first_timestamp_utc": datetime.fromtimestamp(
                cleaned_df["timestamp"].iloc[0] / 1000.0, tz=timezone.utc
            ).isoformat() if not cleaned_df.empty else None,
            "last_timestamp_utc": datetime.fromtimestamp(
                cleaned_df["timestamp"].iloc[-1] / 1000.0, tz=timezone.utc
            ).isoformat() if not cleaned_df.empty else None,
            "min_low": float(cleaned_df["low"].min()) if not cleaned_df.empty else 0.0,
            "max_high": float(cleaned_df["high"].max()) if not cleaned_df.empty else 0.0,
            "total_volume": float(cleaned_df["volume"].sum()) if not cleaned_df.empty else 0.0,
            "mean_atr_proxy": float((cleaned_df["high"] - cleaned_df["low"]).mean()) if not cleaned_df.empty else 0.0,
        }

        # Overall validation status
        is_valid = len(errors) == 0
        if strict and (intraweek_gaps > 0 or len(warnings) > 0):
            is_valid = False

        result = ValidationResult(
            is_valid=is_valid,
            total_candles=total_records,
            valid_candles=len(cleaned_df),
            duplicate_count=duplicate_count,
            ohlc_violations=ohlc_violations,
            price_range_violations=price_range_violations,
            anomaly_spikes=anomaly_spikes,
            volume_anomalies=volume_anomalies,
            missing_gaps_count=len(gaps),
            weekend_gaps_count=weekend_gaps,
            intraweek_gaps_count=intraweek_gaps,
            errors=errors,
            warnings=warnings,
            gaps=gaps,
            summary_metrics=summary_metrics,
        )

        return result, cleaned_df[CANONICAL_COLUMNS].reset_index(drop=True)
