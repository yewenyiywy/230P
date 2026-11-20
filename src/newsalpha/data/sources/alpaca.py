"""Alpaca adapter: Benzinga news, SIP minute bars and SIP NBBO quotes.

Free Basic plan: full SIP history except the most recent 15 minutes, 200 API
calls per minute. Keys come from the environment (never from the repo):

    ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY

Endpoints (data.alpaca.markets):
    GET /v1beta1/news        news articles, paginated by next_page_token
    GET /v2/stocks/bars      1Min / 1Day bars, feed=sip
    GET /v2/stocks/quotes    tick NBBO quotes, feed=sip (resampled to minute snapshots here)

NOTE: written against Alpaca's public API docs; run `newsalpha fetch --days 5`
once against a real key before pulling years of data.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import pandas as pd
import requests

from newsalpha.config import DataConfig

BASE = "https://data.alpaca.markets"
MIN_SECONDS_BETWEEN_CALLS = 60 / 190  # stay under 200 calls/min


class AlpacaSource:
    def __init__(self, key_id: str, secret: str, cfg: DataConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})
        self._last_call = 0.0

    @classmethod
    def from_env(cls, cfg: DataConfig) -> AlpacaSource:
        key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        return cls(key, secret, cfg)

    # ------------------------------------------------------------------ http
    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(5):
            r = self.session.get(BASE + path, params=params, timeout=30)
            self._last_call = time.monotonic()
            if r.status_code == 429:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"rate-limited on {path}")

    def _paginate(self, path: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        params = dict(params)
        while True:
            page = self._get(path, params)
            yield page
            token = page.get("next_page_token")
            if not token:
                return
            params["page_token"] = token

    # ------------------------------------------------------------------ news
    def articles(self) -> pd.DataFrame:
        params: dict[str, Any] = {
            "start": self.cfg.start,
            "end": self.cfg.end,
            "limit": 50,
            "sort": "asc",
            "include_content": "true",
        }
        if self.cfg.symbols:
            params["symbols"] = ",".join(self.cfg.symbols)
        rows = []
        for page in self._paginate("/v1beta1/news", params):
            for a in page.get("news", []):
                for sym in a.get("symbols") or []:
                    rows.append(
                        {
                            "article_id": str(a["id"]),
                            "symbol": sym,
                            "created_at": a["created_at"],
                            "updated_at": a.get("updated_at") or a["created_at"],
                            "headline": a.get("headline") or "",
                            "summary": a.get("summary") or "",
                            "body": a.get("content") or "",
                            "source": a.get("source") or "benzinga",
                        }
                    )
        df = pd.DataFrame(rows)
        for c in ("created_at", "updated_at"):
            df[c] = pd.to_datetime(df[c], utc=True)
        return df

    # ------------------------------------------------------------------ bars
    def minute_bars(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        return self._bars(symbols, start, end, timeframe="1Min")

    def daily_bars(
        self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        return self._bars(symbols, start, end, timeframe="1Day", adjustment="all")

    def _bars(
        self,
        symbols: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        timeframe: str,
        adjustment: str = "raw",
    ) -> pd.DataFrame:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "adjustment": adjustment,
            "feed": "sip",
        }
        rows = []
        for page in self._paginate("/v2/stocks/bars", params):
            for sym, bars in (page.get("bars") or {}).items():
                for b in bars:
                    rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
        df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    # ---------------------------------------------------------------- quotes
    def quotes(self, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """NBBO ticks resampled to the last quote of each minute, stamped at minute end."""
        params = {
            "symbols": ",".join(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 10000,
            "feed": "sip",
        }
        rows = []
        for page in self._paginate("/v2/stocks/quotes", params):
            for sym, qs in (page.get("quotes") or {}).items():
                for q in qs:
                    rows.append((sym, q["t"], q["bp"], q["ap"], q["bs"], q["as"]))
        ticks = pd.DataFrame(rows, columns=["symbol", "ts", "bid", "ask", "bid_size", "ask_size"])
        if ticks.empty:
            return ticks
        ticks["ts"] = pd.to_datetime(ticks["ts"], utc=True)
        ticks = ticks[(ticks.bid > 0) & (ticks.ask >= ticks.bid)]
        ticks["minute_end"] = ticks["ts"].dt.floor("min") + pd.Timedelta(minutes=1)
        snap = ticks.sort_values("ts").groupby(["symbol", "minute_end"]).last().reset_index()
        return snap.drop(columns="ts").rename(columns={"minute_end": "ts"})
