"""The interface every data source implements.

A source returns three tables (contracts in newsalpha.schemas): ARTICLES,
MINUTE_BARS and QUOTES. Quotes are minute snapshots of the NBBO, stamped at the
END of the minute they describe, so `asof(ts)` never sees the future.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketData(Protocol):
    def articles(self) -> pd.DataFrame: ...

    def minute_bars(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...

    def quotes(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame: ...


class InMemoryData:
    """Holds already-loaded tables; used by the synthetic world and by cached runs."""

    def __init__(self, articles: pd.DataFrame, bars: pd.DataFrame, quotes: pd.DataFrame):
        self._articles = articles
        self._bars = bars
        self._quotes = quotes

    def articles(self) -> pd.DataFrame:
        return self._articles

    def minute_bars(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        b = self._bars
        return b[b.symbol.isin(symbols) & (b.ts >= start) & (b.ts < end)]

    def quotes(self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        q = self._quotes
        return q[q.symbol.isin(symbols) & (q.ts >= start) & (q.ts <= end)]

    @property
    def all_bars(self) -> pd.DataFrame:
        return self._bars

    @property
    def all_quotes(self) -> pd.DataFrame:
        return self._quotes
