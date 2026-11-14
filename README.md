# XAUUSD Hybrid AI Trading Research Platform

A production-grade, statistically honest quantitative trading and machine learning research platform focused exclusively on **XAUUSD (Gold vs. US Dollar)**.

Built entirely with free, open-source technologies: **Python 3.11, FastAPI, DuckDB, PyArrow, XGBoost, LightGBM, Scikit-Learn, Next.js 14, TypeScript, Tailwind CSS, and TradingView Lightweight Charts**.

---

## 🏛️ System Architecture

```
                                  [ Dukascopy Jetta API ]
                                             │ (M1 Raw Feed)
                                             ▼
                                  [ Candlestick Resampler ]
                                   M5 (Primary) | M15 & H1
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            [ Parquet Raw Store ]                        [ Data Validator ]
             data/raw/xauusd/                             - Strict OHLC Geometry
                                                          - Economic Bounds ($300-$10,000)
                                                          - Spike & Anomaly Filters
                                                                   │
                                                                   ▼
                                                       [ Parquet Processed Store ]
                                                       data/processed/xauusd/
                                                                   │
                                                                   ▼
                                                         [ DuckDB Zero-Copy SQL ]
                                                                   │
                                ┌──────────────────────────────────┴──────────────────────────────────┐
                                ▼                                                                     ▼
                    [ Feature Engineering ]                                                [ Labeling Engine ]
          - Technical: EMA, RSI, MACD, ATR, BB, ADX                              - 15m / 30m / 60m Directional Ternary
          - SMC: BOS, CHoCH, FVG, Swings, Order Blocks                           - Triple Barrier (TP / SL / Time Limit)
          - Price Action: Pin Bars, Engulfing, Inside Bars                       - MFE / MAE Realized Excursion Audits
          - Regimes: Trend, Range, Volatility Expansion                                               │
                                └──────────────────────────────────┬──────────────────────────────────┘
                                                                   │ (Zero Lookahead As-Of Fusion)
                                                                   ▼
                                                    [ Machine Learning Model Suite ]
                                           - Model A: Technical Analysis (XGBoost)
                                           - Model B: Smart Money Concepts (LightGBM)
                                           - Model C: Market Regime (Random Forest)
                                           - Model D: Price Action (XGBoost)
                                           - Model E: Calibrated Multi-Model Ensemble
                                                                   │
                                                                   ▼
                                                       [ Prediction Engine ]
                                           - Calibrated Probabilities (Bull/Bear/Neut)
                                           - Dynamic ATR Entry, SL, TP (1:2 R:R)
                                           - Selective Abstention Threshold (WAIT)
                                                                   │
                                ┌──────────────────────────────────┴──────────────────────────────────┐
                                ▼                                                                     ▼
                  [ Realistic Backtest Lab ]                                           [ Continuous Retraining ]
          - Spread ($0.20) & Slippage ($0.05)                                    - Champion vs Challenger Out-of-Sample
          - Conservative Same-Bar SL Priority                                    - Expected Calibration Error (ECE) Guard
          - Fractional Position Risk (1.0%)                                      - Atomic Promotion & Rollback Registry
                                │                                                                     │
                                └──────────────────────────────────┬──────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                      [ FastAPI REST Backend ]
                                                 GET /api/market/candles
                                                 GET /api/market/features
                                                 GET /api/ai/prediction
                                                 GET /api/ai/performance
                                                 POST /api/backtest/run
                                                 POST /api/ai/train
                                                                   │
                                                                   ▼
                                                    [ Next.js 14 Dashboard ]
                                          - TradingView Lightweight Canvas Charts
                                          - Live Dynamic AI Probability Gauges
                                          - Interactive Backtesting Lab & Equity Curve
                                          - Model Performance & ECE Calibration Views
```

---

## ⚡ Key Capabilities & Statistical Safeguards

1. **Zero Lookahead Bias**: Swings and market structures (BOS/CHoCH) are algorithmically confirmed only using confirmed lagged pivots. Higher-timeframe context (M15, H1) is shifted by 1 bar before backward as-of merging.
2. **Purged Time-Series Splits**: Cross-validation enforces a 12-bar purge window and embargo to eliminate label overlap contamination.
3. **Calibrated Selective Abstention**: Uses logistic meta-calibration and allows the AI to abstain (`WAIT`) when market conditions are noisy or submodels conflict.
4. **Conservative Execution Modeling**: If both Stop-Loss and Take-Profit price levels are touched within the same candle, the simulation always triggers the Stop-Loss first.
5. **Completely Free Stack**: Zero reliance on Supabase, paid OpenAI APIs, paid market data feeds, or paid hosting.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 2. Backend Environment Setup

```powershell
# Create Python 3.11 virtual environment
uv venv .venv --python 3.11

# Activate virtual environment
.venv\Scripts\activate.ps1

# Install Python dependencies
uv pip install -e .
```

### 3. Frontend Environment Setup

```powershell
cd frontend
npm install
npm run build
```

---

## 💻 CLI Commands (`pipeline_cli.py`)

The platform includes a unified CLI:

### Download Historical Candles
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli download --start 2024-05-01 --end 2024-05-31 --instrument XAUUSD --timeframe m5
```

### Validate Data Integrity & Generate Quality Report
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli validate --instrument XAUUSD --timeframe m5
```

### Generate Feature Matrix (Technical + SMC + Price Action + Regimes)
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli features --instrument XAUUSD
```

### Generate Target Labels (Directional Returns + Triple Barrier)
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli labels --instrument XAUUSD
```

### Train Model Suite (Models A, B, C, D, and E Ensemble)
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli train --threshold 0.50 --version v1.0.0
```

### Run Realistic Backtest Simulation
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli backtest --version v1.0.0 --threshold 0.45
```

### Continuous Retraining (Champion vs Challenger Promotion)
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli retrain --champion-version v1.0.0 --challenger-version v1.1.0 --threshold 0.50
```

### Start FastAPI Backend Server
```powershell
.venv\Scripts\python.exe -m src.cli.pipeline_cli api --port 8000 --reload
```
- Interactive Swagger API Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

---

## 🌐 Launching the Next.js Frontend Dashboard

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

### Dashboard Views
1. **Overview**: High-level system stats, Spot Gold ticker, Live Chart, and AI Analysis card.
2. **Live XAUUSD Chart**: TradingView Lightweight Charts with M5/M15/H1 timeframes, EMA overlays (20/50/200), SMC markers (FVG, BOS, CHoCH, Swings), and AI signal markers.
3. **AI Analysis**: Calibrated Bullish/Bearish/Neutral probabilities, confidence gauge, market regime, submodel agreement (Technical, SMC, Price Action), and dynamic 1:2 R:R levels.
4. **Prediction History**: Forward outcome evaluation over 15m/30m/60m horizons, MFE (peak profit in pips), and MAE (drawdown in pips).
5. **Model Performance**: Expected Calibration Error (ECE), Brier score, 3-class confusion matrix, and regime breakdown.
6. **Training Center**: Model registry versions, test ratio controls, and admin-protected retraining launcher.
7. **Backtesting Lab**: Interactive execution simulator with spread/slippage sliders, equity curve chart, and complete trade log table.
8. **Settings**: System health status, backend API URL, and security key settings.

---

## 🧪 Automated Testing Suite

Run all 82 automated unit and integration tests:

```powershell
.venv\Scripts\pytest.exe
```

Test coverage includes:
- Dukascopy pure-Python client and delta decoder
- Candlestick resampler (M1 -> M5, M15, H1)
- Data validator and anomaly detection
- Parquet storage and DuckDB analytical views
- Technical indicators and SMC algorithms (BOS, CHoCH, FVG, Swings, Order Blocks)
- Labeling engine (Directional ternary, Triple Barrier, MFE/MAE)
- Model A, B, C, D, and E (Ensemble) training and inference
- Purged time-series splitting and Expected Calibration Error (ECE)
- Realistic backtesting simulation engine and conservative conflict rules
- Continuous retraining champion-challenger promotion orchestrator
- All FastAPI backend endpoints and administrative authentication
- Next.js production build and TypeScript compilation
