"""Background daemon service orchestrating live candle polling and automated paper execution."""

import threading
import time
from typing import Optional
import pandas as pd

from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.features.pipeline import FeaturePipeline
from src.live.feed import LiveMarketFeed
from src.live.paper_trader import PaperTradingEngine
from src.models.prediction_engine import PredictionEngine

logger = get_logger("live.daemon")


class PaperTradingDaemon:
    """Asynchronous background worker polling live candles and triggering paper execution."""

    def __init__(
        self,
        paper_engine: PaperTradingEngine,
        prediction_engine: PredictionEngine,
        feed: Optional[LiveMarketFeed] = None,
        poll_interval_sec: int = 15,
    ):
        self.paper_engine = paper_engine
        self.prediction_engine = prediction_engine
        self.feed = feed or LiveMarketFeed()
        self.feature_pipeline = FeaturePipeline()
        self.poll_interval_sec = poll_interval_sec

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.info("Paper trading daemon is already running.")
            return

        self._stop_event.clear()
        self.paper_engine.is_running = True
        self.paper_engine.save_state()

        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="PaperDaemon")
        self._thread.start()
        logger.info("Paper trading background daemon started.")

    def stop(self) -> None:
        """Stop daemon thread gracefully."""
        self._stop_event.set()
        self.paper_engine.is_running = False
        self.paper_engine.save_state()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Paper trading background daemon stopped.")

    def step(self) -> Optional[dict]:
        """Execute a single polling and execution cycle (useful for tests or timer ticks)."""
        candles_df = self.feed.fetch_latest_candles(
            instrument=DEFAULT_INSTRUMENT,
            timeframe=Timeframe.M5,
            limit=120,
        )
        if candles_df.empty or len(candles_df) < 20:
            return None

        latest_candle = candles_df.iloc[-1]
        ts = int(latest_candle["timestamp"])

        # Process features and prediction
        feat_df = self.feature_pipeline.compute_m5_features(candles_df)
        pred = self.prediction_engine.predict_candle(
            candle_features=feat_df,
            instrument=DEFAULT_INSTRUMENT,
            timeframe=Timeframe.M5.value,
        )

        event = self.paper_engine.process_candle(latest_candle, prediction=pred)
        return event

    def _run_loop(self) -> None:
        logger.info(f"Paper trading loop active (polling every {self.poll_interval_sec}s)...")
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception as e:
                logger.error(f"Error in paper trading daemon loop: {e}")

            # Sleep in short increments to allow prompt termination
            for _ in range(self.poll_interval_sec * 2):
                if self._stop_event.is_set():
                    break
                time.sleep(0.5)
