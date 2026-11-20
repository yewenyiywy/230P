"""Event-minute rules (plan, alignment rule 2). Times below are UTC."""

import pandas as pd
import pytest

from newsalpha.calendar import event_time, horizon_exit


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize(
    "created, tau, overnight",
    [
        # regular session: next full minute
        ("2025-11-20 15:00:00", "2025-11-20 15:01:00", False),
        ("2025-11-20 15:00:59", "2025-11-20 15:01:00", False),
        # exactly at the open (09:30 ET = 14:30 UTC) -> 14:31
        ("2025-11-20 14:30:00", "2025-11-20 14:31:00", False),
        # 15:59:30 ET -> tau would be the 16:00 close -> next session open
        ("2025-11-20 20:59:30", "2025-11-21 14:30:00", True),
        # after hours and pre-market
        ("2025-11-20 22:00:00", "2025-11-21 14:30:00", True),
        ("2025-11-21 12:00:00", "2025-11-21 14:30:00", True),
        # Saturday -> Monday
        ("2025-11-22 15:00:00", "2025-11-24 14:30:00", True),
        # Christmas Day (holiday) -> next session
        ("2025-12-25 15:00:00", "2025-12-26 14:30:00", True),
        # Day after Thanksgiving closes 13:00 ET (18:00 UTC)
        ("2025-11-28 17:58:10", "2025-11-28 17:59:00", False),
        ("2025-11-28 17:59:10", "2025-12-01 14:30:00", True),
        # first Monday after the DST switch: 10:00 EDT = 14:00 UTC is in session
        ("2025-03-10 14:00:00", "2025-03-10 14:01:00", False),
    ],
)
def test_event_minute(created, tau, overnight):
    et = event_time(ts(created))
    assert et.tau == ts(tau)
    assert et.overnight is overnight


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        event_time(pd.Timestamp("2025-11-20 15:00"))


def test_horizon_exit_respects_early_close():
    # t1 from Wednesday 2025-11-26 is Friday 11-28 (Thursday is Thanksgiving), early close 18:00 UTC
    assert horizon_exit(pd.Timestamp("2025-11-26"), "t1") == ts("2025-11-28 18:00:00")
