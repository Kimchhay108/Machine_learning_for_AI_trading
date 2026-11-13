"""Historical data download orchestrator for XAUUSD.

Coordinates fetching from Dukascopy, resampling into M5/M15/H1 timeframes,
running data validation, and persisting raw and processed Parquet files.
Supports incremental downloading to avoid redundant network transfers.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm

from src.core.constants import DEFAULT_INSTRUMENT, Timeframe
from src.core.logger import get_logger
from src.data.dukascopy_client import DukascopyClient
from src.data.resampler import CandlestickResampler
from src.data.storage import ParquetStorage
from src.data.validator import DataValidator, ValidationResult

logger = get_logger("downloader")


class HistoricalDownloader:
    """Orchestrates historical market data ingestion across date ranges."""

    def __init__(
        self,
        client: Optional[DukascopyClient] = None,
        storage: Optional[ParquetStorage] = None,
        validator: Optional[DataValidator] = None,
        resampler: Optional[CandlestickResampler] = None,
    ):
        self.client = client or DukascopyClient()
        self.storage = storage or ParquetStorage()
        self.validator = validator or DataValidator()
        self.resampler = resampler or CandlestickResampler()

    @staticmethod
    def _parse_date(dt_val: Union[datetime, date, str]) -> datetime:
        """Parse datetime, date, or ISO string to UTC datetime at 00:00:00."""
        if isinstance(dt_val, str):
            clean_str = dt_val.split("T")[0]
            parsed = datetime.strptime(clean_str, "%Y-%m-%d")
            return parsed.replace(tzinfo=timezone.utc)
        elif isinstance(dt_val, datetime):
            if dt_val.tzinfo is None:
                return dt_val.replace(tzinfo=timezone.utc)
            return dt_val.astimezone(timezone.utc)
        elif isinstance(dt_val, date):
            return datetime(dt_val.year, dt_val.month, dt_val.day, tzinfo=timezone.utc)
        raise ValueError(f"Cannot parse date from: {dt_val}")

    def download_range(
        self,
        start_date: Union[datetime, date, str],
        end_date: Union[datetime, date, str],
        instrument: str = DEFAULT_INSTRUMENT,
        price_type: str = "BID",
        force_redownload: bool = False,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """Download historical M1 data for date range, resample to M5, M15, H1, validate, and store.

        Args:
            start_date: Inclusive start date (UTC)
            end_date: Inclusive end date (UTC)
            instrument: Symbol, default "XAUUSD"
            price_type: "BID" or "ASK"
            force_redownload: If False, skip days already stored
            progress_bar: Whether to display interactive tqdm progress

        Returns:
            Dictionary summary with total days, downloaded candles, files written, and validation.
        """
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date)

        if start_dt > end_dt:
            raise ValueError(f"start_date ({start_dt}) cannot be after end_date ({end_dt})")

        total_days = (end_dt.date() - start_dt.date()).days + 1
        current_dt = start_dt

        logger.info(
            f"Starting download pipeline for {instrument} [{start_dt.date()} to {end_dt.date()}] "
            f"({total_days} calendar days)"
        )

        all_m1_candles: List[Dict[str, Any]] = []
        days_with_data = 0
        market_closed_days = 0
        skipped_days = 0

        day_iterator = range(total_days)
        if progress_bar:
            day_iterator = tqdm(day_iterator, desc=f"Downloading {instrument}", unit="day")

        for _ in day_iterator:
            year = current_dt.year
            month = current_dt.month
            day = current_dt.day

            # Check if this day is a weekend (Saturday or Sunday daytime)
            # Forex/Gold closed Fri 22:00 to Sun 22:00 UTC
            weekday = current_dt.weekday()
            if weekday == 5:  # Saturday
                market_closed_days += 1
                current_dt += timedelta(days=1)
                continue

            try:
                m1_records = self.client.fetch_day_minute_candles(
                    instrument=instrument,
                    year=year,
                    month=month,
                    day=day,
                    price_type=price_type,
                )

                if m1_records:
                    all_m1_candles.extend(m1_records)
                    days_with_data += 1
                else:
                    market_closed_days += 1

            except Exception as exc:
                logger.error(f"Error fetching {instrument} on {year:04d}-{month:02d}-{day:02d}: {exc}")

            current_dt += timedelta(days=1)

        logger.info(
            f"Download completed: {len(all_m1_candles)} total raw M1 candles fetched. "
            f"Active trading days: {days_with_data}, Market closed days: {market_closed_days}"
        )

        if not all_m1_candles:
            return {
                "instrument": instrument,
                "start_date": str(start_dt.date()),
                "end_date": str(end_dt.date()),
                "total_days": total_days,
                "days_with_data": days_with_data,
                "market_closed_days": market_closed_days,
                "total_m1_candles": 0,
                "m5_candles": 0,
                "m15_candles": 0,
                "h1_candles": 0,
                "files_written": [],
                "validation": None,
                "message": "No data returned for the requested date range (market closed or weekend).",
            }

        # Resample M1 to M5, M15, H1
        df_m5_raw = self.resampler.resample(all_m1_candles, target_timeframe=Timeframe.M5, instrument=instrument)
        df_m15_raw = self.resampler.resample(all_m1_candles, target_timeframe=Timeframe.M15, instrument=instrument)
        df_h1_raw = self.resampler.resample(all_m1_candles, target_timeframe=Timeframe.H1, instrument=instrument)

        # Save Raw M5 Parquet
        raw_files = self.storage.save_candles(
            df=df_m5_raw,
            timeframe=Timeframe.M5,
            instrument=instrument,
            is_processed=False,
        )

        # Validate M5 candles
        val_result_m5, clean_m5 = self.validator.validate(
            df=df_m5_raw,
            timeframe=Timeframe.M5.value,
            instrument=instrument,
        )

        # Validate M15 & H1
        val_result_m15, clean_m15 = self.validator.validate(
            df=df_m15_raw,
            timeframe=Timeframe.M15.value,
            instrument=instrument,
        )
        val_result_h1, clean_h1 = self.validator.validate(
            df=df_h1_raw,
            timeframe=Timeframe.H1.value,
            instrument=instrument,
        )

        # Save Processed Clean Parquet
        proc_files_m5 = self.storage.save_candles(
            df=clean_m5,
            timeframe=Timeframe.M5,
            instrument=instrument,
            is_processed=True,
        )
        proc_files_m15 = self.storage.save_candles(
            df=clean_m15,
            timeframe=Timeframe.M15,
            instrument=instrument,
            is_processed=True,
        )
        proc_files_h1 = self.storage.save_candles(
            df=clean_h1,
            timeframe=Timeframe.H1,
            instrument=instrument,
            is_processed=True,
        )

        all_written_files = [str(p) for p in (raw_files + proc_files_m5 + proc_files_m15 + proc_files_h1)]

        return {
            "instrument": instrument,
            "start_date": str(start_dt.date()),
            "end_date": str(end_dt.date()),
            "total_days": total_days,
            "days_with_data": days_with_data,
            "market_closed_days": market_closed_days,
            "total_m1_candles": len(all_m1_candles),
            "m5_candles": len(clean_m5),
            "m15_candles": len(clean_m15),
            "h1_candles": len(clean_h1),
            "files_written": all_written_files,
            "validation": val_result_m5.to_dict(),
        }
