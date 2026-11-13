"""Quality report generator for historical candlestick datasets.

Produces detailed analytical reports in both JSON and GitHub Flavored Markdown formats,
providing auditable verification of OHLC integrity, temporal continuity, and economic bounds.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from tabulate import tabulate

from src.core.config import settings
from src.core.constants import DEFAULT_INSTRUMENT
from src.core.logger import get_logger
from src.data.validator import ValidationResult

logger = get_logger("quality_reporter")


class QualityReporter:
    """Generates auditable data quality reports for market datasets."""

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or settings.QUALITY_REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        df: pd.DataFrame,
        validation_result: ValidationResult,
        instrument: str = DEFAULT_INSTRUMENT,
        timeframe: str = "m5",
        data_source: str = "dukascopy",
    ) -> Dict[str, Any]:
        """Generate comprehensive quality metrics and export Markdown & JSON reports.

        Args:
            df: Validated or cleaned DataFrame
            validation_result: ValidationResult object from DataValidator
            instrument: Symbol, default "XAUUSD"
            timeframe: e.g. "m5"
            data_source: e.g. "dukascopy"

        Returns:
            Dictionary containing the full report structure.
        """
        now_utc = datetime.now(timezone.utc)
        report_timestamp = now_utc.strftime("%Y%m%d_%H%M%S")

        total_rows = len(df)
        if total_rows > 0:
            first_ts = int(df["timestamp"].iloc[0])
            last_ts = int(df["timestamp"].iloc[-1])
            start_utc = datetime.fromtimestamp(first_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            end_utc = datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            min_price = float(df["low"].min())
            max_price = float(df["high"].max())
            mean_price = float(df["close"].mean())
            total_volume = float(df["volume"].sum())
            zero_vol_pct = float((df["volume"] == 0).mean() * 100.0)
            avg_spread_range = float((df["high"] - df["low"]).mean())
        else:
            start_utc = end_utc = "N/A"
            min_price = max_price = mean_price = total_volume = zero_vol_pct = avg_spread_range = 0.0

        report_data: Dict[str, Any] = {
            "metadata": {
                "instrument": instrument,
                "timeframe": timeframe,
                "data_source": data_source,
                "generated_at_utc": now_utc.isoformat(),
                "report_id": f"{instrument}_{timeframe}_{report_timestamp}",
            },
            "dataset_overview": {
                "total_candles_received": validation_result.total_candles,
                "valid_candles_retained": validation_result.valid_candles,
                "start_time_utc": start_utc,
                "end_time_utc": end_utc,
                "price_low_extreme": min_price,
                "price_high_extreme": max_price,
                "mean_close_price": round(mean_price, 3),
                "total_volume": round(total_volume, 4),
                "zero_volume_percent": round(zero_vol_pct, 2),
                "average_candle_range": round(avg_spread_range, 3),
            },
            "validation_summary": {
                "is_passed": validation_result.is_valid,
                "duplicate_timestamps": validation_result.duplicate_count,
                "ohlc_violations": validation_result.ohlc_violations,
                "price_range_violations": validation_result.price_range_violations,
                "extreme_spikes": validation_result.anomaly_spikes,
                "volume_anomalies": validation_result.volume_anomalies,
                "total_gaps": validation_result.missing_gaps_count,
                "weekend_closure_gaps": validation_result.weekend_gaps_count,
                "intraweek_anomalous_gaps": validation_result.intraweek_gaps_count,
            },
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "gap_details": [
                {
                    "start": g.start_utc,
                    "end": g.end_utc,
                    "duration_min": round(g.duration_minutes, 1),
                    "missing_bars": g.expected_missing_candles,
                    "type": "Weekend Closure" if g.is_weekend_gap else "Intra-week Gap",
                }
                for g in validation_result.gaps[:20]  # First 20 gaps
            ],
        }

        # 1. Export JSON report
        json_path = self.reports_dir / f"{instrument.lower()}_{timeframe.lower()}_{report_timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # 2. Export Markdown report
        md_path = self.reports_dir / f"{instrument.lower()}_{timeframe.lower()}_{report_timestamp}.md"
        md_content = self._render_markdown_report(report_data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Generated Quality Reports: {json_path.name} & {md_path.name}")
        report_data["json_path"] = str(json_path)
        report_data["md_path"] = str(md_path)
        return report_data

    def _render_markdown_report(self, r: Dict[str, Any]) -> str:
        meta = r["metadata"]
        overview = r["dataset_overview"]
        val = r["validation_summary"]

        status_badge = "✅ PASSED" if val["is_passed"] else "❌ FAILED"

        overview_table = [
            ["Instrument", meta["instrument"]],
            ["Timeframe", meta["timeframe"]],
            ["Data Source", meta["data_source"]],
            ["Date Range (UTC)", f"{overview['start_time_utc']} to {overview['end_time_utc']}"],
            ["Total Valid Candles", f"{overview['valid_candles_retained']:,}"],
            ["Price Range (USD)", f"${overview['price_low_extreme']:.3f} - ${overview['price_high_extreme']:.3f}"],
            ["Mean Close Price", f"${overview['mean_close_price']:.3f}"],
            ["Average Candle Range", f"${overview['average_candle_range']:.3f}"],
            ["Total Volume", f"{overview['total_volume']:.4f}"],
            ["Zero-Volume Candles", f"{overview['zero_volume_percent']}%"],
        ]

        val_table = [
            ["OHLC Geometric Violations", val["ohlc_violations"]],
            ["Duplicate Timestamps", val["duplicate_timestamps"]],
            ["Out of Range Prices", val["price_range_violations"]],
            ["Spike Anomalies (>15%)", val["extreme_spikes"]],
            ["Volume Anomalies", val["volume_anomalies"]],
            ["Weekend Closure Gaps", val["weekend_closure_gaps"]],
            ["Intra-week Dropouts", val["intraweek_anomalous_gaps"]],
        ]

        md = f"""# Market Data Quality & Integrity Report: {meta['instrument']} ({meta['timeframe'].upper()})

- **Generated UTC**: {meta['generated_at_utc']}
- **Report Status**: **{status_badge}**
- **Data Source**: {meta['data_source']}

## 1. Dataset Overview

{tabulate(overview_table, headers=["Metric", "Value"], tablefmt="github")}

## 2. Validation & Anomaly Checks

{tabulate(val_table, headers=["Integrity Check", "Count"], tablefmt="github")}

## 3. Findings & Notes
"""
        if r["errors"]:
            md += "\n### ❌ Critical Errors:\n"
            for err in r["errors"]:
                md += f"- {err}\n"
        else:
            md += "\n- **Zero OHLC geometric violations detected** (High >= Open/Close/Low, Low <= Open/Close/High, prices > 0).\n"

        if r["warnings"]:
            md += "\n### ⚠️ Warnings & Anomalies:\n"
            for warn in r["warnings"]:
                md += f"- {warn}\n"
        else:
            md += "- **Zero warnings detected.** Time series continuity satisfies all strict rules.\n"

        if r["gap_details"]:
            md += "\n## 4. Identified Gaps Breakdown (Sample)\n\n"
            gaps_table = [
                [g["start"], g["end"], g["duration_min"], g["missing_bars"], g["type"]]
                for g in r["gap_details"][:10]
            ]
            md += tabulate(gaps_table, headers=["Gap Start UTC", "Gap End UTC", "Duration (min)", "Missing Bars", "Type"], tablefmt="github")
            md += "\n"

        md += "\n---\n*Report generated automatically by XAUUSD Quantitative Research Pipeline.*\n"
        return md
