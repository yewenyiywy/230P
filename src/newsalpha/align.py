"""Stage 2, align: articles -> events, plus a point-in-time quote lookup.

Two things live here because every later stage depends on them being right:

1. `build_events` assigns each article its event minute tau (calendar rules).
2. `QuoteBook` answers "what was the NBBO at time T?" using only quotes stamped
   at or before T. Every price used anywhere in the pipeline goes through it,
   which is what makes the no-look-ahead tests meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.calendar import event_time
from newsalpha.schemas import ARTICLES, EVENTS, validate


def build_events(articles: pd.DataFrame) -> pd.DataFrame:
    validate(articles, ARTICLES, "articles")
    times = [event_time(t) for t in articles["created_at"]]
    events = pd.DataFrame(
        {
            "event_id": articles["article_id"].astype(str) + ":" + articles["symbol"].astype(str),
            "article_id": articles["article_id"].values,
            "symbol": articles["symbol"].values,
            "created_at": pd.DatetimeIndex(articles["created_at"]),
            "tau": pd.to_datetime([t.tau for t in times], utc=True),
            "session_date": pd.to_datetime([t.session for t in times]),
            "overnight": [t.overnight for t in times],
            "body_usable": articles.get(
                "body_usable", pd.Series(True, index=articles.index)
            ).values,
        }
    )
    events["created_at"] = pd.to_datetime(events["created_at"], utc=True)
    return validate(events, EVENTS, "events")


class QuoteBook:
    """Point-in-time NBBO lookup: `at(symbol, times)` returns the last snapshot with ts <= time."""

    def __init__(self, quotes: pd.DataFrame):
        self._by_symbol: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
        for sym, g in quotes.sort_values("ts").groupby("symbol"):
            g = g.reset_index(drop=True)
            self._by_symbol[sym] = (
                g["ts"].dt.tz_convert(None).to_numpy().astype("datetime64[ns]"),
                g,
            )

    def symbols(self) -> list[str]:
        return list(self._by_symbol)

    def at(self, symbol: str, times: pd.Series | pd.DatetimeIndex | list) -> pd.DataFrame:
        cols = ["bid", "ask", "bid_size", "ask_size"]
        t = pd.to_datetime(pd.Series(times), utc=True)
        if symbol not in self._by_symbol:
            return pd.DataFrame(np.nan, index=range(len(t)), columns=cols + ["mid", "quote_ts"])
        ts, g = self._by_symbol[symbol]
        idx = (
            np.searchsorted(
                ts, t.dt.tz_convert(None).to_numpy().astype("datetime64[ns]"), side="right"
            )
            - 1
        )
        out = pd.DataFrame(np.nan, index=range(len(t)), columns=cols)
        ok = idx >= 0
        out.loc[ok, cols] = g.loc[idx[ok], cols].values
        out["mid"] = (out["bid"] + out["ask"]) / 2
        qts = np.full(len(t), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        qts[ok] = ts[idx[ok]]
        out["quote_ts"] = pd.to_datetime(qts).tz_localize("UTC")
        return out

    def mid(self, symbol: str, times) -> np.ndarray:
        return self.at(symbol, times)["mid"].to_numpy(dtype=float)
