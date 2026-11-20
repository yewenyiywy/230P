"""Databento adapter: L2 depth samples around events and official daily closes.

Every pull is priced first with `estimate_cost`; `fetch` refuses to run above
`max_usd` so a typo cannot burn the $125 credit. Key from DATABENTO_API_KEY.

    XNAS.ITCH  schema mbp-10   10-level depth, Nasdaq-listed, from 2018-05
    EQUS.SUMMARY ohlcv-1d      consolidated official daily bars, from 2024-07-01
"""

from __future__ import annotations

import os

import pandas as pd


class DatabentoSource:
    def __init__(self, key: str | None = None):
        import databento as db  # optional dependency: pip install '.[databento]'

        self._db = db
        self.client = db.Historical(key or os.environ["DATABENTO_API_KEY"])

    def estimate_cost(
        self, dataset: str, schema: str, symbols: list[str], start: str, end: str
    ) -> float:
        return float(
            self.client.metadata.get_cost(
                dataset=dataset, schema=schema, symbols=symbols, start=start, end=end
            )
        )

    def fetch(
        self,
        dataset: str,
        schema: str,
        symbols: list[str],
        start: str,
        end: str,
        max_usd: float = 5.0,
    ) -> pd.DataFrame:
        cost = self.estimate_cost(dataset, schema, symbols, start, end)
        if cost > max_usd:
            raise RuntimeError(f"Databento request would cost ${cost:.2f} > max_usd={max_usd}")
        data = self.client.timeseries.get_range(
            dataset=dataset, schema=schema, symbols=symbols, start=start, end=end
        )
        return data.to_df()

    def depth_around(self, symbol: str, tau: pd.Timestamp, minutes: int = 30, max_usd: float = 1.0):
        start = (tau - pd.Timedelta(minutes=minutes)).isoformat()
        end = (tau + pd.Timedelta(minutes=minutes)).isoformat()
        return self.fetch("XNAS.ITCH", "mbp-10", [symbol], start, end, max_usd=max_usd)
