"""NYSE session logic. The single source of truth for 'when could we trade?'.

Rule (plan, alignment rule 2): the event minute tau is the first full minute
after the article's `created_at`. If tau is not strictly inside a regular
session, the event becomes 'overnight' and tau moves to the next session open.
Early closes, holidays and DST are handled by exchange_calendars (XNYS).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=1)
def _cal() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS", start="2005-01-01")


@dataclass(frozen=True)
class EventTime:
    tau: pd.Timestamp  # UTC
    session: pd.Timestamp  # naive session date
    overnight: bool


def session_open(session: pd.Timestamp) -> pd.Timestamp:
    return _cal().session_open(session)


def session_close(session: pd.Timestamp) -> pd.Timestamp:
    return _cal().session_close(session)


def next_session(ts_or_session: pd.Timestamp) -> pd.Timestamp:
    """First session whose open is at or after `ts` (UTC timestamp)."""
    cal = _cal()
    if ts_or_session.tzinfo is None:
        return cal.next_session(ts_or_session)
    return cal.minute_to_future_session(ts_or_session)


def add_sessions(session: pd.Timestamp, n: int) -> pd.Timestamp:
    cal = _cal()
    return cal.session_offset(session, n)


def sessions_in_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    return _cal().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))


def event_time(created_at: pd.Timestamp) -> EventTime:
    """Map an article timestamp (UTC) to its event minute tau."""
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware UTC")
    created_at = created_at.tz_convert("UTC")
    tau = created_at.floor("min") + pd.Timedelta(minutes=1)
    cal = _cal()
    # Which session (if any) is open at tau?
    if cal.is_trading_minute(tau):
        session = cal.minute_to_session(tau)
        close = cal.session_close(session)
        if tau < close:
            return EventTime(tau=tau, session=session, overnight=False)
    session = cal.minute_to_future_session(tau)
    return EventTime(tau=cal.session_open(session), session=session, overnight=True)


def horizon_exit(session: pd.Timestamp, horizon: str) -> pd.Timestamp:
    """Exit timestamp (UTC) for a holding horizon measured from session t0."""
    offsets = {"close_t0": 0, "t1": 1, "t3": 3, "t5": 5}
    if horizon not in offsets:
        raise ValueError(f"unknown horizon {horizon}")
    return session_close(add_sessions(session, offsets[horizon]))
