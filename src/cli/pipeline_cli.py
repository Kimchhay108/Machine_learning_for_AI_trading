"""Command-line interface for the XAUUSD Historical Data Pipeline.

Provides subcommands for downloading, resampling, validating, reporting, and DuckDB analytical queries.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
from tabulate import tabulate

from src.core.config import settings

from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.data.downloader import HistoricalDownloader
from src.data.duckdb_manager import DuckDBManager
from src.data.quality_reporter import QualityReporter
from src.data.storage import ParquetStorage
from src.data.validator import DataValidator
from src.features.pipeline import FeaturePipeline
from src.labels.labeling_engine import LabelingEngine
from src.models.ensemble import EnsemblePredictionModel
from src.models.prediction_engine import PredictionEngine
from src.backtest.engine import RealisticBacktestEngine

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


def cmd_features(args: argparse.Namespace) -> int:
    """Generate multi-timeframe feature matrix from processed Parquet datasets."""
    storage = ParquetStorage()
    pipeline = FeaturePipeline()

    print("\n" + "=" * 65)
    print(f"  FEATURE ENGINEERING PIPELINE: {args.instrument}")
    print("=" * 65)

    df_m5 = storage.load_candles(timeframe=Timeframe.M5, instrument=args.instrument, is_processed=True)
    df_m15 = storage.load_candles(timeframe=Timeframe.M15, instrument=args.instrument, is_processed=True)
    df_h1 = storage.load_candles(timeframe=Timeframe.H1, instrument=args.instrument, is_processed=True)

    if df_m5.empty:
        print(f"Error: No processed M5 candles found for {args.instrument}. Run download first.")
        return 1

    feature_matrix = pipeline.build_multi_timeframe_feature_matrix(
        df_m5=df_m5,
        df_m15=df_m15 if not df_m15.empty else None,
        df_h1=df_h1 if not df_h1.empty else None,
        instrument=args.instrument,
    )

    out_path = pipeline.save_feature_matrix(feature_matrix)
    print(f"\nFeature matrix generated successfully: {out_path}")
    print(f"Total Rows:     {len(feature_matrix):,}")
    print(f"Total Features: {len(feature_matrix.columns):,}")

    # Display preview
    cols_preview = [c for c in feature_matrix.columns if c not in ["timestamp", "instrument", "timeframe", "data_source"]][:10]
    print(f"\nSample Feature Columns: {', '.join(cols_preview)} ...")
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    """Generate forward returns and triple-barrier training labels."""
    storage = ParquetStorage()
    labeling = LabelingEngine()
    features_dir = settings.FEATURES_DIR / args.instrument.lower()
    feat_file = features_dir / "m5_features.parquet"

    print("\n" + "=" * 65)
    print(f"  STATISTICAL TARGET LABELING PIPELINE: {args.instrument}")
    print("=" * 65)

    if feat_file.exists():
        df = pd.read_parquet(feat_file)
    else:
        df = storage.load_candles(timeframe=Timeframe.M5, instrument=args.instrument, is_processed=True)

    if df.empty:
        print(f"Error: No data available for {args.instrument}.")
        return 1

    labeled_df = labeling.compute_all_labels(df)
    out_path = labeling.save_labels(labeled_df)

    print(f"\nLabels generated successfully: {out_path}")
    print(f"Total Rows:     {len(labeled_df):,}")

    # Summary of label distributions
    tb_long_counts = labeled_df["label_tb_long"].value_counts().to_dict()
    dir_15m_counts = labeled_df["label_dir_ternary_15m"].value_counts().to_dict()

    summary_table = [
        ["Triple Barrier Long (+1 TP, -1 SL, 0 Time)", str(tb_long_counts)],
        ["15-Min Directional Ternary (+1, -1, 0)", str(dir_15m_counts)],
        ["Mean Forward Return (15m)", f"{labeled_df['fwd_ret_15m'].mean():.6f}"],
        ["Mean Forward Return (60m)", f"{labeled_df['fwd_ret_60m'].mean():.6f}"],
    ]
    print("\n" + tabulate(summary_table, headers=["Target Formulation", "Distribution / Metric"], tablefmt="fancy_grid"))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train Model A, B, C, D, and E (Ensemble) with walk-forward validation and comparison."""
    features_dir = settings.FEATURES_DIR / args.instrument.lower()
    labels_dir = settings.LABELS_DIR / args.instrument.lower()
    registry_dir = settings.DATA_DIR / "models" / "registry"

    feat_path = features_dir / "m5_features.parquet"
    lbl_path = labels_dir / "m5_labels.parquet"

    if not feat_path.exists() or not lbl_path.exists():
        print("Error: Feature or Label Parquet files missing. Run 'features' and 'labels' first.")
        return 1

    df_feats = pd.read_parquet(feat_path)
    df_lbls = pd.read_parquet(lbl_path)

    # Use 15m ternary label mapped to 0 (Bearish), 1 (Neutral), 2 (Bullish)
    # Original: -1, 0, 1 -> +1 mapping
    target_col = "label_dir_ternary_15m"
    if target_col not in df_lbls.columns:
        print(f"Error: Target column {target_col} missing in labels.")
        return 1

    y = (df_lbls[target_col] + 1).astype(int)  # 0, 1, 2

    # Drop non-feature and forward-looking columns
    drop_cols = ["timestamp", "instrument", "timeframe", "data_source"]
    X = df_feats[[c for c in df_feats.columns if c not in drop_cols]].copy()

    # Chronological Split: 70% Train, 30% Out-Of-Sample Test (Purged by 12 bars)
    n = len(X)
    train_end = int(n * 0.70)
    test_start = train_end + 12  # 12-bar purge window

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    print("\n" + "=" * 65)
    print(f"  MACHINE LEARNING SUITE TRAINING: {args.instrument}")
    print(f"  Train Samples: {len(X_train):,} | Test Samples (OOS): {len(X_test):,}")
    print(f"  Confidence Threshold for Selective Abstention: {args.threshold}")
    print("=" * 65)

    ensemble = EnsemblePredictionModel(confidence_threshold=args.threshold)
    ensemble.fit(X_train, y_train)

    # Compare models on unseen test split
    print("\nEvaluating Model Suite on Chronological Out-Of-Sample Test Data...")
    comparisons = ensemble.compare_models(X_test, y_test)

    comp_table = []
    for model_name, metrics in comparisons.items():
        comp_table.append([
            model_name,
            f"{metrics['accuracy']*100:.2f}%",
            f"{metrics.get('selective_accuracy', 0.0)*100:.2f}%",
            f"{metrics.get('coverage_pct', 0.0):.1f}%",
            f"{metrics['f1_score']:.3f}",
            f"{metrics.get('expected_calibration_error', 0.0):.3f}",
        ])

    print("\n" + tabulate(
        comp_table,
        headers=["Model", "All Accuracy", f"Selective Acc (>={args.threshold})", "Coverage %", "F1 Score", "ECE"],
        tablefmt="fancy_grid"
    ))

    # Save trained suite
    saved_dir = ensemble.save_suite(registry_dir, version=args.version)
    print(f"\nTrained Model Suite registered at: {saved_dir}\n")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Execute realistic out-of-sample backtest with transaction costs and conservative execution."""
    features_dir = settings.FEATURES_DIR / args.instrument.lower()
    registry_dir = settings.DATA_DIR / "models" / "registry"

    feat_path = features_dir / "m5_features.parquet"
    if not feat_path.exists():
        print("Error: Feature Parquet missing. Run 'features' first.")
        return 1

    df_feats = pd.read_parquet(feat_path)
    model_dir = registry_dir / args.version
    if not model_dir.exists():
        print(f"Error: Model version {args.version} not found in registry. Run 'train' first.")
        return 1

    ensemble = EnsemblePredictionModel.load_suite(registry_dir, version=args.version)
    ensemble.confidence_threshold = args.threshold

    # Test on the last 30% of data (Out-of-sample)
    n = len(df_feats)
    test_start = int(n * 0.70) + 12
    oos_df = df_feats.iloc[test_start:].reset_index(drop=True)

    print("\n" + "=" * 65)
    print(f"  REALISTIC MARKET BACKTEST: {args.instrument}")
    print(f"  Model Version: {args.version} | Confidence Threshold: {args.threshold}")
    print(f"  Out-of-Sample Period: {len(oos_df):,} candles")
    print(f"  Execution Assumptions: Spread $0.20 | Slippage $0.05 | Risk 1.0%")
    print("=" * 65)

    # Generate signals
    decisions, probs, confs = ensemble.predict_decision(oos_df)

    backtester = RealisticBacktestEngine(
        initial_balance=10_000.0,
        risk_pct_per_trade=0.01,
        spread_usd=0.20,
        slippage_usd=0.05,
    )
    result = backtester.run(oos_df, signals=decisions)

    res_table = [
        ["Initial Balance", f"${result.initial_balance:,.2f}"],
        ["Final Balance", f"${result.final_balance:,.2f}"],
        ["Net Profit ($)", f"${result.total_net_profit:,.2f}"],
        ["Return on Account", f"{result.return_pct:.2f}%"],
        ["Total Trades", result.total_trades],
        ["Winning Trades", result.winning_trades],
        ["Losing Trades", result.losing_trades],
        ["Win Rate", f"{result.win_rate_pct:.2f}%"],
        ["Profit Factor", result.profit_factor],
        ["Max Drawdown ($)", f"${result.max_drawdown_usd:,.2f}"],
        ["Max Drawdown (%)", f"{result.max_drawdown_pct:.2f}%"],
        ["Sharpe Ratio", result.sharpe_ratio],
        ["Avg Trade PnL", f"${result.avg_trade_usd:.2f}"],
        ["Avg R-Multiple", f"{result.avg_r_multiple:.2f}R"],
    ]
    print("\n" + tabulate(res_table, headers=["Performance Metric", "Result"], tablefmt="fancy_grid"))

    if result.trades:
        print(f"\nRecent 5 Completed Trades:")
        trades_preview = [
            [t.direction, t.entry_price, t.exit_price, t.pnl_usd, f"{t.r_multiple}R", t.exit_reason]
            for t in result.trades[-5:]
        ]
        print(tabulate(trades_preview, headers=["Dir", "Entry", "Exit", "PnL ($)", "R-Mult", "Exit Reason"], tablefmt="psql"))

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

    # Subcommand: features
    p_feat = subparsers.add_parser("features", help="Extract multi-timeframe technical and SMC features")
    p_feat.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_feat.set_defaults(func=cmd_features)

    # Subcommand: labels
    p_lbl = subparsers.add_parser("labels", help="Generate directional and triple-barrier training labels")
    p_lbl.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_lbl.set_defaults(func=cmd_labels)

    # Subcommand: train
    p_tr = subparsers.add_parser("train", help="Train ML models and evaluate out-of-sample comparison")
    p_tr.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_tr.add_argument("--version", default="v1.0.0", help="Model registry version tag")
    p_tr.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold for selective prediction")
    p_tr.set_defaults(func=cmd_train)

    # Subcommand: backtest
    p_bt = subparsers.add_parser("backtest", help="Run realistic backtest on out-of-sample historical data")
    p_bt.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument ticker")
    p_bt.add_argument("--version", default="v1.0.0", help="Model registry version tag")
    p_bt.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
