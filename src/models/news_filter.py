"""High-Impact Economic News Calendar & Blackout Engine for XAUUSD.

Protects trading models from extreme spread spikes, slippage, and whipsaws
caused by Tier-1 USD macroeconomic announcements:
1. US Non-Farm Payrolls (NFP) & Unemployment Rate (1st Friday, 12:30/13:30 UTC)
2. FOMC Interest Rate Decision & Fed Chair Press Conference (8 Wednesdays/year, 18:00 & 18:30 UTC)
3. US Consumer Price Index (CPI) Inflation (Monthly, 12:30/13:30 UTC)
4. US Producer Price Index (PPI) (Monthly, 12:30/13:30 UTC)
5. US Advance / Preliminary GDP (Quarterly, 12:30/13:30 UTC)
6. ISM Manufacturing & Services PMI (1st and 3rd business days, 14:00/15:00 UTC)
7. US Retail Sales (Monthly, 12:30/13:30 UTC)
8. US Initial Jobless Claims (Weekly Thursdays, 12:30/13:30 UTC)
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("models.news_filter")


@dataclass
class NewsEvent:
    """Represents a high-impact macroeconomic event."""
    id: str
    title: str
    currency: str
    impact: str                  # "HIGH" or "EXTREME"
    event_time_utc: str          # ISO string: "2024-05-03T12:30:00Z"
    category: str                # "EMPLOYMENT", "CENTRAL_BANK", "INFLATION", "GROWTH", "PMI"
    description: str
    blackout_before_mins: int = 30
    blackout_after_mins: int = 30

    @property
    def event_dt(self) -> datetime:
        dt = datetime.fromisoformat(self.event_time_utc.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @property
    def blackout_start(self) -> datetime:
        return self.event_dt - timedelta(minutes=self.blackout_before_mins)

    @property
    def blackout_end(self) -> datetime:
        return self.event_dt + timedelta(minutes=self.blackout_after_mins)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["blackout_start_utc"] = self.blackout_start.isoformat()
        d["blackout_end_utc"] = self.blackout_end.isoformat()
        return d


class EconomicNewsFilter:
    """Deterministic institutional economic calendar and news blackout filter for XAUUSD."""

    DEFAULT_BLACKOUT_BEFORE_MINS = 30
    DEFAULT_BLACKOUT_AFTER_MINS = 30

    # Scheduled FOMC Rate Decisions & Press Conferences (18:00 UTC statement, 18:30 UTC presser)
    FOMC_DATES = [
        # 2024
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        # 2025
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        # 2026
        "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
        # 2027
        "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16",
    ]

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_dir = settings.DATA_DIR / "news"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_file or (self.cache_dir / "economic_calendar.json")
        self.events: List[NewsEvent] = []
        self._load_or_generate_events()

    def _load_or_generate_events(self):
        """Load calendar from disk or generate institutional event schedule."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self.events = [
                        NewsEvent(
                            id=item["id"],
                            title=item["title"],
                            currency=item["currency"],
                            impact=item["impact"],
                            event_time_utc=item["event_time_utc"],
                            category=item["category"],
                            description=item.get("description", ""),
                            blackout_before_mins=item.get("blackout_before_mins", 30),
                            blackout_after_mins=item.get("blackout_after_mins", 30),
                        )
                        for item in raw_data
                    ]
                logger.info(f"Loaded {len(self.events)} high-impact economic news events from cache.")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached economic calendar: {e}. Regenerating...")

        self.events = self._generate_full_calendar(start_year=2024, end_year=2027)
        self._save_to_cache()
        logger.info(f"Generated {len(self.events)} deterministic high-impact USD economic events.")

    def _save_to_cache(self):
        """Save calendar events to JSON cache."""
        try:
            payload = [evt.to_dict() for evt in self.events]
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist news calendar cache: {e}")

    @staticmethod
    def _is_dst(dt: date) -> bool:
        """Approximate US Daylight Saving Time (Second Sunday in March to first Sunday in November)."""
        year = dt.year
        # 2nd Sunday in March
        mar1 = date(year, 3, 1)
        first_sun_mar = mar1 + timedelta(days=(6 - mar1.weekday()) % 7)
        second_sun_mar = first_sun_mar + timedelta(days=7)

        # 1st Sunday in November
        nov1 = date(year, 11, 1)
        first_sun_nov = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)

        return second_sun_mar <= dt < first_sun_nov

    def _generate_full_calendar(self, start_year: int = 2024, end_year: int = 2027) -> List[NewsEvent]:
        """Generate deterministic institutional dates & times for high-impact USD news."""
        events: List[NewsEvent] = []

        # 1. FOMC Rate Decisions & Press Conferences
        for f_date_str in self.FOMC_DATES:
            dt_base = datetime.fromisoformat(f_date_str)
            if start_year <= dt_base.year <= end_year:
                # 18:00 UTC statement
                t_stmt = datetime(dt_base.year, dt_base.month, dt_base.day, 18, 0, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"fomc_rate_{f_date_str}",
                        title="FOMC Interest Rate Decision & Monetary Policy Statement",
                        currency="USD",
                        impact="EXTREME",
                        event_time_utc=t_stmt.isoformat(),
                        category="CENTRAL_BANK",
                        description="Federal Reserve interest rate announcement and economic projections.",
                        blackout_before_mins=45,
                        blackout_after_mins=30,
                    )
                )
                # 18:30 UTC press conference
                t_press = datetime(dt_base.year, dt_base.month, dt_base.day, 18, 30, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"fomc_press_{f_date_str}",
                        title="FOMC Fed Chair Press Conference",
                        currency="USD",
                        impact="EXTREME",
                        event_time_utc=t_press.isoformat(),
                        category="CENTRAL_BANK",
                        description="Fed Chair Jerome Powell answers live questions regarding rates and balance sheet.",
                        blackout_before_mins=15,
                        blackout_after_mins=45,
                    )
                )

        # Iterate by month for monthly releases
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                # Standard release hour in UTC: 12:30 UTC during US DST, 13:30 UTC during standard time
                mid_month = date(y, m, 15)
                rel_hour = 12 if self._is_dst(mid_month) else 13
                pm_hour = 14 if self._is_dst(mid_month) else 15

                # 2. Non-Farm Payrolls (NFP) & Unemployment Rate: First Friday of month
                # Find 1st day of month
                day1 = date(y, m, 1)
                # Friday is weekday 4
                days_until_fri = (4 - day1.weekday()) % 7
                first_fri = day1 + timedelta(days=days_until_fri)
                t_nfp = datetime(first_fri.year, first_fri.month, first_fri.day, rel_hour, 30, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"nfp_{first_fri.isoformat()}",
                        title="US Non-Farm Payrolls (NFP) & Unemployment Rate",
                        currency="USD",
                        impact="EXTREME",
                        event_time_utc=t_nfp.isoformat(),
                        category="EMPLOYMENT",
                        description="Headline employment report, wage growth, and national unemployment rate.",
                        blackout_before_mins=30,
                        blackout_after_mins=45,
                    )
                )

                # 3. US Consumer Price Index (CPI): Second Wednesday or Thursday (~10-14th)
                # Let's find second Wednesday
                days_until_wed = (2 - day1.weekday()) % 7
                second_wed = day1 + timedelta(days=days_until_wed + 7)
                t_cpi = datetime(second_wed.year, second_wed.month, second_wed.day, rel_hour, 30, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"cpi_{second_wed.isoformat()}",
                        title="US Consumer Price Index (CPI MoM & YoY)",
                        currency="USD",
                        impact="EXTREME",
                        event_time_utc=t_cpi.isoformat(),
                        category="INFLATION",
                        description="Headline & Core inflation metric dictating real interest rate trajectory.",
                        blackout_before_mins=30,
                        blackout_after_mins=45,
                    )
                )

                # 4. US Producer Price Index (PPI): Day following CPI
                ppi_day = second_wed + timedelta(days=1)
                t_ppi = datetime(ppi_day.year, ppi_day.month, ppi_day.day, rel_hour, 30, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"ppi_{ppi_day.isoformat()}",
                        title="US Producer Price Index (PPI MoM & YoY)",
                        currency="USD",
                        impact="HIGH",
                        event_time_utc=t_ppi.isoformat(),
                        category="INFLATION",
                        description="Wholesale inflation indicator measuring input price pressures.",
                        blackout_before_mins=20,
                        blackout_after_mins=25,
                    )
                )

                # 5. US Retail Sales: Usually ~15th-16th of the month
                rs_day = date(y, m, 15)
                if rs_day.weekday() >= 5:  # Weekend adjustment
                    rs_day = rs_day + timedelta(days=(7 - rs_day.weekday()))
                t_rs = datetime(rs_day.year, rs_day.month, rs_day.day, rel_hour, 30, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"retail_sales_{rs_day.isoformat()}",
                        title="US Retail Sales (Core MoM)",
                        currency="USD",
                        impact="HIGH",
                        event_time_utc=t_rs.isoformat(),
                        category="GROWTH",
                        description="Measures consumer spending representing 70% of US GDP.",
                        blackout_before_mins=20,
                        blackout_after_mins=25,
                    )
                )

                # 6. ISM Manufacturing PMI: First business day of the month
                bday1 = day1
                if bday1.weekday() >= 5:
                    bday1 = bday1 + timedelta(days=(7 - bday1.weekday()))
                t_ism_m = datetime(bday1.year, bday1.month, bday1.day, pm_hour, 0, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"ism_m_{bday1.isoformat()}",
                        title="ISM Manufacturing PMI & Prices Paid",
                        currency="USD",
                        impact="HIGH",
                        event_time_utc=t_ism_m.isoformat(),
                        category="PMI",
                        description="Purchasing managers sentiment and inflation prices paid in manufacturing.",
                        blackout_before_mins=20,
                        blackout_after_mins=25,
                    )
                )

                # 7. ISM Services PMI: Third business day of the month
                bday3 = bday1 + timedelta(days=2)
                if bday3.weekday() >= 5:
                    bday3 = bday3 + timedelta(days=(7 - bday3.weekday()))
                t_ism_s = datetime(bday3.year, bday3.month, bday3.day, pm_hour, 0, 0, tzinfo=timezone.utc)
                events.append(
                    NewsEvent(
                        id=f"ism_s_{bday3.isoformat()}",
                        title="ISM Services / Non-Manufacturing PMI",
                        currency="USD",
                        impact="HIGH",
                        event_time_utc=t_ism_s.isoformat(),
                        category="PMI",
                        description="Services sector economic expansion and employment indicator.",
                        blackout_before_mins=20,
                        blackout_after_mins=25,
                    )
                )

                # 8. Advance US GDP (Quarterly cycle in Jan, Apr, Jul, Oct)
                if m in (1, 4, 7, 10):
                    # Last Thursday of month
                    last_day = date(y, m, 28)
                    days_until_thu = (3 - last_day.weekday()) % 7
                    gdp_day = last_day + timedelta(days=days_until_thu)
                    t_gdp = datetime(gdp_day.year, gdp_day.month, gdp_day.day, rel_hour, 30, 0, tzinfo=timezone.utc)
                    events.append(
                        NewsEvent(
                            id=f"gdp_{gdp_day.isoformat()}",
                            title="US Advance GDP (Annualized QoQ)",
                            currency="USD",
                            impact="HIGH",
                            event_time_utc=t_gdp.isoformat(),
                            category="GROWTH",
                            description="First annualized estimate of total US economic output and growth.",
                            blackout_before_mins=30,
                            blackout_after_mins=30,
                        )
                    )

        # Sort chronologically
        events.sort(key=lambda e: e.event_dt)
        return events

    def is_blackout_active(
        self,
        query_time: Optional[datetime] = None,
        buffer_before_mins: int = DEFAULT_BLACKOUT_BEFORE_MINS,
        buffer_after_mins: int = DEFAULT_BLACKOUT_AFTER_MINS,
    ) -> Tuple[bool, Optional[NewsEvent], Optional[float]]:
        """Check if query_time falls inside any high-impact news blackout window.

        Returns:
            Tuple of:
            - is_blackout: bool
            - active_event: Optional[NewsEvent]
            - minutes_delta: float (e.g. -15.0 if 15 mins before release, +10.0 if 10 mins after)
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        elif query_time.tzinfo is None:
            query_time = query_time.replace(tzinfo=timezone.utc)
        else:
            query_time = query_time.astimezone(timezone.utc)

        for event in self.events:
            event_dt = event.event_dt
            b_start = event_dt - timedelta(minutes=buffer_before_mins or event.blackout_before_mins)
            b_end = event_dt + timedelta(minutes=buffer_after_mins or event.blackout_after_mins)

            if b_start <= query_time <= b_end:
                delta_mins = (query_time - event_dt).total_seconds() / 60.0
                return True, event, round(delta_mins, 1)

        return False, None, None

    def get_upcoming_events(
        self,
        from_time: Optional[datetime] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Retrieve upcoming high-impact USD economic events starting from from_time."""
        if from_time is None:
            from_time = datetime.now(timezone.utc)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=timezone.utc)

        upcoming = []
        for event in self.events:
            if event.event_dt >= from_time - timedelta(minutes=event.blackout_after_mins):
                delta_mins = int((event.event_dt - from_time).total_seconds() / 60)
                d = event.to_dict()
                d["minutes_until_event"] = delta_mins
                d["is_blackout_now"] = (
                    event.blackout_start <= from_time <= event.blackout_end
                )
                upcoming.append(d)
                if len(upcoming) >= limit:
                    break
        return upcoming

    def get_blackout_mask(
        self,
        timestamps: Union[pd.Series, np.ndarray],
        buffer_before_mins: int = DEFAULT_BLACKOUT_BEFORE_MINS,
        buffer_after_mins: int = DEFAULT_BLACKOUT_AFTER_MINS,
    ) -> np.ndarray:
        """Vectorized blackout mask calculation for backtest arrays.

        Parameters:
            timestamps: pd.Series or np.ndarray of epoch timestamps (seconds or ms).
            buffer_before_mins: minutes to blackout prior to release.
            buffer_after_mins: minutes to blackout following release.

        Returns:
            np.ndarray of bool shape (N,) where True indicates candle falls inside a news blackout window.
        """
        if len(timestamps) == 0:
            return np.array([], dtype=bool)

        # Convert to epoch seconds array
        first_val = float(timestamps.iloc[0] if isinstance(timestamps, pd.Series) else timestamps[0])
        is_ms = first_val > 1e11
        scale = 1000.0 if is_ms else 1.0

        ts_sec = (np.array(timestamps, dtype=np.float64) / scale)
        min_ts = ts_sec[0]
        max_ts = ts_sec[-1]

        # Extract interval seconds for events in scope
        buf_before_sec = buffer_before_mins * 60.0
        buf_after_sec = buffer_after_mins * 60.0

        in_scope_intervals = []
        for evt in self.events:
            evt_sec = evt.event_dt.timestamp()
            start_sec = evt_sec - buf_before_sec
            end_sec = evt_sec + buf_after_sec

            if end_sec >= min_ts and start_sec <= max_ts:
                in_scope_intervals.append((start_sec, end_sec))

        mask = np.zeros(len(ts_sec), dtype=bool)
        if not in_scope_intervals:
            return mask

        # Check intervals efficiently
        for start_sec, end_sec in in_scope_intervals:
            mask |= (ts_sec >= start_sec) & (ts_sec <= end_sec)

        return mask


# Singleton instance
_global_news_filter: Optional[EconomicNewsFilter] = None


def get_news_filter() -> EconomicNewsFilter:
    """Get or create singleton EconomicNewsFilter instance."""
    global _global_news_filter
    if _global_news_filter is None:
        _global_news_filter = EconomicNewsFilter()
    return _global_news_filter
