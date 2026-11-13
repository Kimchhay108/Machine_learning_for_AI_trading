"""High-performance, pure-Python client for Dukascopy historical market data.

Connects to the Dukascopy Jetta API endpoint, decodes delta-compressed OHLCV series,
handles rate limits with exponential backoff, and properly accommodates market closures
(weekends/holidays) without crashing.
"""

from datetime import datetime, timezone
import math
import time
from typing import Any, Dict, List, Optional
import httpx

from src.core.config import settings
from src.core.constants import DUKASCOPY_INSTRUMENT_MAP
from src.core.logger import get_logger

logger = get_logger("dukascopy_client")


class DukascopyClientError(Exception):
    """Base exception for Dukascopy client operations."""
    pass


class DukascopyClient:
    """Client for retrieving and decoding historical candle data from Dukascopy."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        rate_limit_delay: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.DUKASCOPY_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.DUKASCOPY_REQUEST_TIMEOUT
        self.max_retries = max_retries or settings.DUKASCOPY_MAX_RETRIES
        self.rate_limit_delay = rate_limit_delay or settings.DUKASCOPY_RATE_LIMIT_DELAY
        self.headers = {
            "User-Agent": settings.DUKASCOPY_USER_AGENT,
            "Referer": settings.DUKASCOPY_REFERER,
            "Accept": "application/json, text/plain, */*",
        }
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout),
                headers=self.headers,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close HTTP connection pool."""
        if self._client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> "DukascopyClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_instrument_code(self, symbol: str) -> str:
        """Map standard symbol (e.g. XAUUSD) to Dukascopy code (e.g. XAU-USD)."""
        clean_symbol = symbol.upper().replace("/", "").replace("_", "")
        return DUKASCOPY_INSTRUMENT_MAP.get(clean_symbol, clean_symbol)

    def _request_with_retry(self, url: str) -> Optional[Dict[str, Any]]:
        """Execute HTTP GET with exponential backoff on 429 and network errors."""
        client = self._get_client()
        delay = settings.DUKASCOPY_RETRY_DELAY

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.get(url)

                # 404 or empty content means market was closed / no candles
                if response.status_code == 404:
                    logger.debug(f"No data available (404) at {url} (likely market closed)")
                    return None

                # 429 Too Many Requests -> backoff
                if response.status_code == 429:
                    retry_after_hdr = response.headers.get("Retry-After")
                    wait_time = float(retry_after_hdr) if retry_after_hdr else delay
                    logger.warning(
                        f"Dukascopy 429 Rate Limit hit. Attempt {attempt}/{self.max_retries}. "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    time.sleep(wait_time)
                    delay *= 2
                    continue

                # Server error 5xx -> retry
                if response.status_code >= 500:
                    logger.warning(
                        f"Dukascopy Server error {response.status_code} at {url}. "
                        f"Attempt {attempt}/{self.max_retries}. Waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                response.raise_for_status()

                # Some responses are empty JSON or 0 length
                text = response.text.strip()
                if not text or text == "{}":
                    return None

                return response.json()

            except (httpx.RequestError, httpx.TimeoutException) as exc:
                logger.warning(
                    f"Network error ({type(exc).__name__}) connecting to {url}. "
                    f"Attempt {attempt}/{self.max_retries}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay *= 2

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts.")
        return None

    @staticmethod
    def decode_candles(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decode Dukascopy delta-encoded JSON candle response into standard OHLCV records.

        Dukascopy format:
        - timestamp: int (epoch ms of period start)
        - multiplier: float (e.g. 0.001 for gold)
        - shift: int (time interval in ms, e.g. 60000 for M1, 3600000 for H1)
        - open, high, low, close: base price levels
        - times: array of delta time steps
        - opens, highs, lows, closes: arrays of delta price increments
        - volumes: array of volume floats
        """
        if not data or "times" not in data or not data.get("times"):
            return []

        multiplier = float(data.get("multiplier", 0.001))
        scale = max(0, -math.floor(math.log10(multiplier))) if multiplier > 0 else 3
        shift = int(data.get("shift", 60000))
        timestamp = int(data["timestamp"])

        open_units = round(float(data["open"]) / multiplier)
        high_units = round(float(data["high"]) / multiplier)
        low_units = round(float(data["low"]) / multiplier)
        close_units = round(float(data["close"]) / multiplier)

        times = data["times"]
        opens = data["opens"]
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        volumes = data.get("volumes", [0.0] * len(times))

        num_records = len(times)
        candles: List[Dict[str, Any]] = []

        for i in range(num_records):
            timestamp += times[i] * shift
            open_units += opens[i]
            high_units += highs[i]
            low_units += lows[i]
            close_units += closes[i]

            vol = float(volumes[i]) if i < len(volumes) else 0.0

            # Convert units back to real price with strict precision
            candle_open = round(open_units * multiplier, scale)
            candle_high = round(high_units * multiplier, scale)
            candle_low = round(low_units * multiplier, scale)
            candle_close = round(close_units * multiplier, scale)

            # Defensive fix: ensure High is top and Low is bottom
            real_high = max(candle_open, candle_close, candle_high)
            real_low = min(candle_open, candle_close, candle_low)

            candles.append({
                "timestamp": timestamp,
                "open": candle_open,
                "high": real_high,
                "low": real_low,
                "close": candle_close,
                "volume": round(vol, 6),
            })

        return candles

    def fetch_day_minute_candles(
        self,
        instrument: str,
        year: int,
        month: int,
        day: int,
        price_type: str = "BID",
    ) -> List[Dict[str, Any]]:
        """Fetch all 1-minute candles for a specific day UTC.

        Args:
            instrument: Standard symbol, e.g. "XAUUSD"
            year: 4-digit year, e.g. 2024
            month: 1-indexed month (1 = January, 12 = December)
            day: Day of month (1-31)
            price_type: "BID" or "ASK"

        Returns:
            List of decoded candle dictionaries.
        """
        code = self.get_instrument_code(instrument)
        # URL structure: /candles/minute/{code}/{PRICE_TYPE}/{year}/{month}/{day}
        url = f"{self.base_url}/candles/minute/{code}/{price_type.upper()}/{year}/{month}/{day}"
        logger.debug(f"Fetching M1 candles from: {url}")

        raw_data = self._request_with_retry(url)
        if raw_data is None:
            return []

        candles = self.decode_candles(raw_data)
        logger.debug(f"Decoded {len(candles)} M1 candles for {instrument} on {year:04d}-{month:02d}-{day:02d}")

        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

        return candles

    def fetch_month_hourly_candles(
        self,
        instrument: str,
        year: int,
        month: int,
        price_type: str = "BID",
    ) -> List[Dict[str, Any]]:
        """Fetch all 1-hour candles for an entire month UTC.

        Args:
            instrument: Standard symbol, e.g. "XAUUSD"
            year: 4-digit year, e.g. 2024
            month: 1-indexed month (1 = January, 12 = December)
            price_type: "BID" or "ASK"

        Returns:
            List of decoded candle dictionaries.
        """
        code = self.get_instrument_code(instrument)
        # URL structure: /candles/hour/{code}/{PRICE_TYPE}/{year}/{month}
        url = f"{self.base_url}/candles/hour/{code}/{price_type.upper()}/{year}/{month}"
        logger.debug(f"Fetching H1 candles from: {url}")

        raw_data = self._request_with_retry(url)
        if raw_data is None:
            return []

        candles = self.decode_candles(raw_data)
        logger.debug(f"Decoded {len(candles)} H1 candles for {instrument} on {year:04d}-{month:02d}")

        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

        return candles
