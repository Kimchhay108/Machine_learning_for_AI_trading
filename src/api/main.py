"""FastAPI Main Application for XAUUSD Hybrid AI Trading Platform."""

from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import get_duckdb_manager, get_prediction_engine, get_active_version
from src.api.routers import ai, backtest, market, paper
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan event."""
    logger.info("Initializing XAUUSD AI Trading API backend...")
    settings.ensure_directories()
    try:
        # Pre-initialize DuckDB views
        get_duckdb_manager()
        # Preload active prediction engine
        get_prediction_engine()
        logger.info("Market data views and Model Suite successfully loaded at startup.")
    except Exception as e:
        logger.warning(f"Startup warm-up notice: {e}")

    yield
    logger.info("Shutting down XAUUSD AI Trading API backend...")


app = FastAPI(
    title="XAUUSD Hybrid AI Trading Research API",
    description=(
        "Production-grade Quantitative AI Trading Platform for Gold (XAUUSD). "
        "Fuses Technical Analysis, Smart Money Concepts (SMC), Price Action, "
        "and Market Regime Classification with Calibrated Selective Ensemble Prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Next.js and frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(market.router)
app.include_router(ai.router)
app.include_router(backtest.router)
app.include_router(paper.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "name": "XAUUSD Hybrid AI Trading API",
        "version": "1.0.0",
        "status": "ONLINE",
        "instrument": "XAUUSD",
        "primary_timeframe": "M5",
        "docs_url": "/docs",
    }


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "HEALTHY",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": settings.ENVIRONMENT,
        "database": "DuckDB & Parquet Active",
        "active_model": get_active_version(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
