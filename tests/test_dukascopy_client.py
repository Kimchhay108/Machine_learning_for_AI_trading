"""Unit tests for Dukascopy client and delta decoder."""

import pytest
from unittest.mock import MagicMock, patch

from src.data.dukascopy_client import DukascopyClient


class TestDukascopyClient:
    def test_instrument_code_mapping(self):
        client = DukascopyClient()
        assert client.get_instrument_code("XAUUSD") == "XAU-USD"
        assert client.get_instrument_code("xauusd") == "XAU-USD"
        assert client.get_instrument_code("XAU/USD") == "XAU-USD"
        assert client.get_instrument_code("EURUSD") == "EUR-USD"
        assert client.get_instrument_code("UNKNOWN") == "UNKNOWN"

    def test_decode_candles_empty(self):
        candles = DukascopyClient.decode_candles({})
        assert candles == []
        candles2 = DukascopyClient.decode_candles({"times": []})
        assert candles2 == []

    def test_decode_candles_synthetic(self):
        # Base timestamp: 1714521600000 (2024-05-01 00:00:00 UTC)
        # Multiplier: 0.001
        # Shift: 60000 ms (1 minute)
        mock_data = {
            "timestamp": 1714521600000,
            "multiplier": 0.001,
            "shift": 60000,
            "open": 2000.0,
            "high": 2000.0,
            "low": 2000.0,
            "close": 2000.0,
            "times": [0, 1, 1],
            "opens": [0, 500, -200],   # open: 2000.0, 2000.5, 2000.3
            "highs": [0, 800, 100],    # high: 2000.0, 2000.8, 2000.9
            "lows": [0, -100, -300],   # low: 2000.0, 1999.9, 1999.6
            "closes": [0, 200, -50],   # close: 2000.0, 2000.2, 2000.15
            "volumes": [0.05, 0.10, 0.15],
        }

        candles = DukascopyClient.decode_candles(mock_data)
        assert len(candles) == 3

        # First candle (at index 0, time_delta = 0)
        c0 = candles[0]
        assert c0["timestamp"] == 1714521600000
        assert c0["open"] == 2000.0
        assert c0["close"] == 2000.0
        assert c0["volume"] == 0.05

        # Second candle (time_delta = 1 -> +60,000ms)
        c1 = candles[1]
        assert c1["timestamp"] == 1714521660000
        assert c1["open"] == 2000.5
        assert c1["volume"] == 0.10
        assert c1["high"] >= c1["open"]
        assert c1["low"] <= c1["open"]

        # Third candle (time_delta = 1 -> +60,000ms)
        c2 = candles[2]
        assert c2["timestamp"] == 1714521720000
        assert c2["volume"] == 0.15

    @patch("src.data.dukascopy_client.httpx.Client.get")
    def test_request_retry_on_429(self, mock_get):
        # Simulate one 429 response followed by 200 OK
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.01"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = '{"times": [0], "opens": [0], "highs": [0], "lows": [0], "closes": [0], "timestamp": 1000, "multiplier": 0.001, "open": 2000, "high": 2000, "low": 2000, "close": 2000, "shift": 60000}'
        resp_200.json.return_value = {
            "times": [0], "opens": [0], "highs": [0], "lows": [0], "closes": [0],
            "timestamp": 1000, "multiplier": 0.001, "open": 2000, "high": 2000, "low": 2000, "close": 2000, "shift": 60000
        }

        mock_get.side_effect = [resp_429, resp_200]

        client = DukascopyClient(timeout=1.0, max_retries=2, rate_limit_delay=0.0)
        res = client._request_with_retry("https://example.com/test")
        assert res is not None
        assert "times" in res
