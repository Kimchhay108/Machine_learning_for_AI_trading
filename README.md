# XAUUSD Hybrid AI Trading Research Platform

A production-grade, statistically honest quantitative trading and machine learning research platform focused exclusively on **XAUUSD (Gold vs. US Dollar)**.

---

## 🏛️ Platform Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │    Dukascopy Historical Jetta API     │
                      └──────────────────┬────────────────────┘
                                         │  (M1 Raw Candles)
                                         ▼
                      ┌───────────────────────────────────────┐
                      │        DukascopyClient (Python)       │
                      │  - Exponential backoff (HTTP 429)     │
                      │  - Pure Python delta decoder          │
                      └──────────────────┬────────────────────┘
                                         │
                                         ▼
                      ┌───────────────────────────────────────┐
                      │         CandlestickResampler          │
                      │   - Exact OHLCV bucket aggregation    │
                      │   - M5 (Primary) | M15 & H1 (Context) │
                      └──────────┬───────────────────┬────────┘
                                 │                   │
                     (Raw M5)    ▼                   ▼   (Raw M5, M15, H1)
            ┌───────────────────────────┐     ┌───────────────────────────┐
            │   Parquet Raw Storage     │     │       DataValidator       │
            │   data/raw/xauusd/m5/     │     │  - Strict OHLC geometry   │
            └───────────────────────────┘     │  - Economic bounds check  │
                                              │  - Anomaly / Spike filter │
                                              │  - Missing gap audit      │
                                              └─────────────┬─────────────┘
                                                            │
                                             (Clean Data)   ▼
            ┌───────────────────────────┐     ┌───────────────────────────┐
            │      QualityReporter      │◄────┤  Parquet Processed Store  │
            │  - JSON audit logs        │     │  data/processed/xauusd/   │
            │  - GitHub Markdown audit  │     └─────────────┬─────────────┘
            └───────────────────────────┘                   │
                                                            ▼
                                              ┌───────────────────────────┐
                                              │       DuckDB Engine       │
                                              │  - Zero-copy Parquet views│
                                              │  - Sub-millisecond SQL    │
                                              └───────────────────────────┘
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python**: 3.11+
- **Package Manager**: `uv` (recommended for ultra-fast installs) or standard `pip`

### 2. Environment Setup

Using `uv` (fastest):
```bash
# Clone / navigate to project root
cd "d:\ai trading  vioew"

# Create virtual environment with Python 3.11
uv venv .venv --python 3.11

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

---

## 📊 Ingestion & Validation Pipeline CLI

The platform provides a unified CLI via `src.cli.pipeline_cli`.

### 1. Download Historical XAUUSD Candles
Downloads M1 bars directly from Dukascopy, resamples them into M5 (primary), M15, and H1, validates OHLCV relationships, detects gaps, and stores month-partitioned Parquet datasets:

```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli download --start 2024-05-01 --end 2024-05-31 --instrument XAUUSD --timeframe m5
```

### 2. Validate Existing Stored Datasets
Validates dataset integrity and writes Markdown and JSON quality reports:

```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli validate --instrument XAUUSD --timeframe m5
```

### 3. DuckDB Analytical Summary
Inspect aggregated dataset statistics instantly via DuckDB without loading entire datasets into memory:

```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli duckdb-summary --instrument XAUUSD --timeframe m5
```

### 4. Direct Analytical SQL Query
Run real SQL queries against DuckDB zero-copy Parquet views:

```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli query --sql "SELECT to_timestamp(timestamp/1000) as time_utc, open, high, low, close, volume FROM v_xauusd_m5_processed ORDER BY timestamp DESC LIMIT 10"
```

---

## 📁 Storage Directory Structure

```
data/
├── raw/
│   └── xauusd/
│       ├── m5/          <- Raw downloaded M5 Parquet partitions (YYYY_MM.parquet)
│       ├── m15/         <- Raw M15 Parquet partitions
│       └── h1/          <- Raw H1 Parquet partitions
├── processed/
│   └── xauusd/
│       ├── m5/          <- Validated, anomaly-filtered M5 Parquet files
│       ├── m15/         <- Validated M15 Parquet files
│       └── h1/          <- Validated H1 Parquet files
├── features/            <- Calculated technical & SMC feature matrices (Phase 3 & 4)
├── labels/              <- Triple-barrier & forward return labels (Phase 4)
├── reports/
│   └── quality/         <- Automated audit Markdown and JSON reports
└── market_data.duckdb   <- DuckDB analytical database
```

---

## 🧪 Automated Testing

Run the full pytest test suite:

```powershell
.venv\Scripts\pytest.exe
```

Run tests with test coverage:

```powershell
.venv\Scripts\pytest.exe --cov=src --cov-report=term-missing
```

---

## 🛡️ Data Quality & Statistical Integrity Rules

1. **Strict Monotonic Ordering**: Preserves exact UTC epoch timestamps.
2. **Deterministic Resampling**: Open = first, High = max, Low = min, Close = last, Volume = sum.
3. **No Synthetic Fabrication**: Missing candles (such as weekend closures and daily 21:00 UTC maintenance roll breaks) are transparently audited and logged in quality reports—never fabricated.
4. **Economic Feasibility**: Every price point is bounded by gold physical boundaries ($300 to $10,000) and verified for geometric consistency (`High >= max(Open, Close, Low)` and `Low <= min(Open, Close, High)`).
