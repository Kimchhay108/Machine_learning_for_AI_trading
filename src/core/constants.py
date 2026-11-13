"""Domain constants and canonical schemas for XAUUSD AI platform."""

from enum import Enum
from typing import Dict, List


class Timeframe(str, Enum):
    M1 = "m1"
    M5 = "m5"
    M15 = "m15"
    H1 = "h1"


class DataSource(str, Enum):
    DUKASCOPY = "dukascopy"
    SYNTHETIC = "synthetic"
    BROKER = "broker"


DEFAULT_INSTRUMENT: str = "XAUUSD"

# Dukascopy API symbol mapping
DUKASCOPY_INSTRUMENT_MAP: Dict[str, str] = {
    "XAUUSD": "XAU-USD",
    "EURUSD": "EUR-USD",
    "GBPUSD": "GBP-USD",
    "USDJPY": "USD-JPY",
    "BTCUSD": "BTC-USD",
}

# Millisecond intervals per timeframe
TIMEFRAME_MS: Dict[str, int] = {
    Timeframe.M1.value: 60 * 1000,
    Timeframe.M5.value: 5 * 60 * 1000,
    Timeframe.M15.value: 15 * 60 * 1000,
    Timeframe.H1.value: 60 * 60 * 1000,
}

# Seconds per timeframe
TIMEFRAME_SECONDS: Dict[str, int] = {
    Timeframe.M1.value: 60,
    Timeframe.M5.value: 300,
    Timeframe.M15.value: 900,
    Timeframe.H1.value: 3600,
}

# Canonical columns required for every candle record
CANONICAL_COLUMNS: List[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "instrument",
    "timeframe",
    "data_source",
]

# Physical and plausible economic boundaries for XAUUSD (Gold in USD/oz)
XAUUSD_MIN_PRICE: float = 300.0
XAUUSD_MAX_PRICE: float = 10000.0
XAUUSD_MAX_ALLOWED_CANDLE_PERCENT_CHANGE: float = 0.15  # 15% single candle max before anomaly flag
XAUUSD_MAX_ALLOWED_GAP_PERCENT: float = 0.10            # 10% inter-candle gap threshold
