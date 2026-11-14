"""Unit tests for the Economic News Calendar & Blackout Engine."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from src.models.news_filter import EconomicNewsFilter, NewsEvent, get_news_filter


class TestEconomicNewsFilter:
    def setup_method(self):
        self.nf = EconomicNewsFilter()

    def test_calendar_has_events(self):
        """Verify calendar generates multi-year high impact USD events."""
        assert len(self.nf.events) >= 100
        # Check all events have valid times and impact
        for evt in self.nf.events:
            assert evt.currency == "USD"
            assert evt.impact in ["HIGH", "EXTREME"]
            assert evt.event_dt.tzinfo is not None

    def test_nfp_scheduled_first_friday(self):
        """Verify NFP events land on Friday."""
        nfp_events = [e for e in self.nf.events if "Non-Farm Payrolls" in e.title]
        assert len(nfp_events) > 0
        for e in nfp_events:
            assert e.event_dt.weekday() == 4  # Friday

    def test_fomc_scheduled_wednesdays(self):
        """Verify FOMC decisions land on scheduled Wednesdays."""
        fomc_events = [e for e in self.nf.events if "FOMC Interest Rate" in e.title]
        assert len(fomc_events) >= 16
        for e in fomc_events:
            assert e.event_dt.weekday() in [2, 3]  # Wednesday (or Thursday Nov 2024 exception)

    def test_blackout_active_during_event(self):
        """Verify blackout is detected 15m prior and 15m post event."""
        test_event = self.nf.events[0]
        event_dt = test_event.event_dt

        # Exactly at release time
        is_bo, evt, delta = self.nf.is_blackout_active(event_dt, buffer_before_mins=30, buffer_after_mins=30)
        assert is_bo is True
        assert evt is not None
        assert abs(delta) < 0.1

        # 10 minutes before release
        t_before = event_dt - pd.Timedelta(minutes=10)
        is_bo, evt, delta = self.nf.is_blackout_active(t_before, buffer_before_mins=30, buffer_after_mins=30)
        assert is_bo is True
        assert delta == -10.0

        # 2 hours after release (outside blackout window)
        t_after = event_dt + pd.Timedelta(hours=2)
        is_bo, evt, delta = self.nf.is_blackout_active(t_after, buffer_before_mins=30, buffer_after_mins=30)
        assert is_bo is False
        assert evt is None

    def test_vectorized_blackout_mask(self):
        """Verify vectorized mask identifies exact blackout candles."""
        test_event = self.nf.events[5]
        evt_dt = test_event.event_dt

        # Create timestamps: 10 candles (50 minutes) before and after
        ts_range = pd.date_range(evt_dt - pd.Timedelta(minutes=50), evt_dt + pd.Timedelta(minutes=50), freq="5min", tz="UTC")
        ts_ms = ts_range.astype("int64") // 10**6  # ms

        mask = self.nf.get_blackout_mask(ts_ms, buffer_before_mins=30, buffer_after_mins=30)
        assert len(mask) == len(ts_ms)
        assert mask.sum() > 0  # Blackout identified

        # Verify candles outside +-30m are False
        outside_before = ts_range < (evt_dt - pd.Timedelta(minutes=30))
        outside_after = ts_range > (evt_dt + pd.Timedelta(minutes=30))
        assert not mask[outside_before].any()
        assert not mask[outside_after].any()
