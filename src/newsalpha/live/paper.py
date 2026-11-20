"""Paper-trading loop skeleton: news -> extraction -> decision -> paper order, with timings.

What is implemented: the latency log and the order call to Alpaca's PAPER
endpoint. What is not yet: the news WebSocket client and live feature
assembly (both TODO, week 2). The loop's contract is fixed here so those can
be filled in without touching the rest.

Free-plan caveat: real-time quotes are IEX-only, so simulated fills are
approximate (see plan).
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

PAPER_BASE = "https://paper-api.alpaca.markets"
NEWS_STREAM = "wss://stream.data.alpaca.markets/v1beta1/news"


@dataclass
class LatencyRecord:
    article_id: str
    symbol: str
    news_created_at: str
    received_at: float = field(default_factory=time.time)
    llm_start: float | None = None
    llm_end: float | None = None
    signal_at: float | None = None
    order_sent_at: float | None = None
    bid: float | None = None
    ask: float | None = None
    side: int = 0
    edge: float | None = None
    order_id: str | None = None


class LatencyLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: LatencyRecord) -> None:
        row = asdict(rec)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


class PaperBroker:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
                "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"],
            }
        )

    def market_order(self, symbol: str, qty: int, side: str) -> str:
        r = self.session.post(
            f"{PAPER_BASE}/v2/orders",
            json={
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "market",
                "time_in_force": "day",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["id"]


def run_paper_loop() -> None:  # pragma: no cover - needs live credentials
    raise NotImplementedError(
        "TODO week 2: subscribe to NEWS_STREAM, extract with the T6 model, build live features "
        "from the latest quote, call trading.rules.decide, send via PaperBroker, log LatencyRecord."
    )
