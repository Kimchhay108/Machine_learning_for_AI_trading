"""Command-line interface for the XAUUSD Historical Data Pipeline.

Provides subcommands for downloading, resampling, validating, reporting, and DuckDB analytical queries.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from tabulate import tabulate

from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.data.downloader import HistoricalDownloader
from src.data.duckdb_manager import DuckDBManager
from src.data.quality_reporter import QualityReporter
from src.data.storage import ParquetStorage
from src.data.validator import DataValidator

logger = get_logger("pipeline_cli")


def cmd_download(args: argparse.Namespace) -> int:
    """Execute download and ingestion pipeline."""
    downloader = HistoricalDownloader()
    reporter = QualityReporter()
    storage = ParquetStorage()
    duckdb_mgr = DuckDBManager()

    print("\n" + "=" * 65)
    print(f"  XAUUSD HISTORICAL DATA PIPELINE: INGESTION")
    print(f"  Instrument: {args.instrument} | Timeframe: {args.timeframe}")
    print(f"  Range:      {args.start} to {args.end}")
    print("=" * 65 + "\n")

    result = downloader.download_range(
        start_date=args.start,
        end_date=args.end,
        instrument=args.instrument,
        price_type=args.price_type,
        force_redownload=args.force,
    )

    print("\n" + "-" * 40)
    print(" Download Pipeline Summary:")
    print("-" * 40)
    summary_table = [
        ["Total Days Scanned", result["total_days"]],
        ["Active Trading Days", result["days_with_data"]],
        ["Market Closed Days", result["market_closed_days"]],
        ["Raw M1 Candles Fetched", f"{result['total_m1_candles']:,}"],
        ["M5 Candles Saved", f"{result['m5_candles']:,}"],
        ["M15 Candles Saved", f"{result['m15_candles']:,}"],
        ["H1 Candles Saved", f"{result['h1_candles']:,}"],
        ["Parquet Files Created/Updated", len(result["files_written"])],
    ]
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Register DuckDB views
    views = duckdb_mgr.create_views(instrument=args.instrument)
    print(f"\nRegistered {len(views)} DuckDB analytical views: {', '.join(views)}")

    # Generate Quality Report
    if result["m5_candles"] > 0:
        clean_df = storage.load_candles(timeframe=Timeframe.M5, instrument=args.instrument, is_processed=True)
        val = result.get("validation")
        if val:
            from src.data.validator import ValidationResult, CandleGap
            val_res = ValidationResult(
                is_valid=val["is_valid"],
                total_candles=val["total_candles"],
                valid_candles=val["valid_candles"],
                duplicate_count=val["duplicate_count"],
                ohlc_violations=val["ohlc_violations"],
                price_range_violations=val["price_range_violations"],
                anomaly_spikes=val["anomaly_spikes"],
                volume_anomalies=val["volume_anomalies"],
                missing_gaps_count=val["missing_gaps_count"],
                weekend_gaps_count=val["weekend_gaps_count"],
                intraweek_gaps_count=val["intraweek_gaps_count"],
                errors=val.get("errors", []),
                warnings=val.get("warnings", []),
                summary_metrics=val.get("summary_metrics", {}),
            )
            q_rep = reporter.generate_report(
                df=clean_df,
                validation_result=val_res,
                instrument=args.instrument,
                timeframe=Timeframe.M5.value,
            )
            print(f"\nData Quality Report written to:\n - {q_rep['md_path']}\n - {q_rep['json_path']}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run data validation and produce quality report on existing stored data."""
    storage = ParquetStorage()
    validator = DataValidator()
    reporter = QualityReporter()

    df = storage.load_candles(
        timeframe=args.timeframe,
        instrument=args.instrument,
        is_processed=not args.raw,
    )

    if df.empty:
        print(f"No stored candles found for {args.instrument} ({args.timeframe}).")
        return 1

    val_res, cleaned_df = validator.validate(
        df=df,
        timeframe=args.timeframe,
        instrument=args.instrument,
        strict=args.strict,
    )

    q_rep = reporter.generate_report(
        df=cleaned_df,
        validation_result=val_res,
        instrument=args.instrument,
        timeframe=args.timeframe,
    )

    print("\n" + "=" * 60)
    print(f"  DATA VALIDATION REPORT: {args.instrument} {args.timeframe.upper()}")
    print(f"  Status: {'PASSED' if val_res.is_valid else 'FAILED'}")
    print("=" * 60)
    print(f"Total Candles:       {val_res.total_candles:,}")
    print(f"Valid Candles:       {val_res.valid_candles:,}")
    print(f"OHLC Violations:     {val_res.ohlc_violations}")
    print(f"Duplicate Count:     {val_res.duplicate_count}")
    print(f"Price Out of Bounds: {val_res.price_range_violations}")
    print(f"Spike Anomalies:     {val_res.anomaly_spikes}")
    print(f"Weekend Gaps:        {val_res.weekend_gaps_count}")
    print(f"Intra-week Gaps:     {val_res.intraweek_gaps_count}")
    print(f"\nDetailed Markdown:   {q_rep['md_path']}")
    print(f"Detailed JSON:       {q_rep['json_path']}\n")

    return 0 if val_res.is_valid else 1


def cmd_duckdb_summary(args: argparse.Namespace) -> int:
    """Print analytical summary using DuckDB."""
    manager = DuckDBManager()
    summary = manager.get_candle_summary(
        instrument=args.instrument,
        timeframe=args.timeframe,
        is_processed=not args.raw,
    )

    print("\n" + "=" * 60)
    print(f"  DUCKDB ANALYTICAL SUMMARY: {args.instrument} {args.timeframe.upper()}")
    print("=" * 60)
    table = [[k, v] for k, v in summary.items()]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Execute arbitrary SQL query in DuckDB."""
    manager = DuckDBManager()
    manager.create_views(instrument=args.instrument)
    df = manager.query(args.sql)
    print(f"\nQuery: {args.sql}\n")
    print(tabulate(df.head(args.limit), headers="keys", tablefmt="psql", showindex=False))
    print(f"\nRows returned: {len(df)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XAUUSD AI Trading Platform - Data Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: download
    p_dl = subparsers.add_parser("download", help="Download historical candles from Dukascopy")
    p_dl.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p_dl.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    p_dl.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_dl.add_argument("--timeframe", default=Timeframe.M5.value, choices=["m5", "m15", "h1"], help="Timeframe")
    p_dl.add_argument("--price-type", default="BID", choices=["BID", "ASK"], help="Price type")
    p_dl.add_argument("--force", action="store_true", help="Force redownload even if existing")
    p_dl.set_defaults(func=cmd_download)

    # Subcommand: validate
    p_val = subparsers.add_parser("validate", help="Validate stored data and generate quality report")
    p_val.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_val.add_argument("--timeframe", default=Timeframe.M5.value, choices=["m5", "m15", "h1"], help="Timeframe")
    p_val.add_argument("--raw", action="store_true", help="Validate raw data instead of processed")
    p_val.add_argument("--strict", action="store_true", help="Strict validation mode")
    p_val.set_defaults(func=cmd_validate)

    # Subcommand: duckdb-summary
    p_ddb = subparsers.add_parser("duckdb-summary", help="Analytical summary using DuckDB")
    p_ddb.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_ddb.add_argument("--timeframe", default=Timeframe.M5.value, choices=["m5", "m15", "h1"], help="Timeframe")
    p_ddb.add_argument("--raw", action="store_true", help="Inspect raw instead of processed")
    p_ddb.set_defaults(func=cmd_duckdb_summary)

    # Subcommand: query
    p_q = subparsers.add_parser("query", help="Execute SQL query via DuckDB")
    p_q.add_argument("--sql", required=True, help="SQL query to execute")
    p_q.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker to create views for")
    p_q.add_argument("--limit", type=int, default=20, help="Max rows to display")
    p_q.set_defaults(func=cmd_query)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
