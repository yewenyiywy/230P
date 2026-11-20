"""A synthetic market with a KNOWN, planted news underreaction.

Used by CI and the smoke test, so the full pipeline runs without API keys and
we can check that it recovers an effect we put there on purpose.

Generative story per news event with true log impact mu:
    - a fraction PHI of mu is priced within 5 minutes of tau (the initial reaction)
    - the remaining (1 - PHI) * mu drifts in linearly over DRIFT_MINUTES (the underreaction)
    - for DRIFT_MINUTES / 6 after tau, NBBO sizes lean toward the drift direction
Headlines state the numbers (EPS vs. estimate, rating and target changes, guidance
ranges) the way Benzinga does, so the rules extractor can read the surprise back.
About 25% of articles are noise ("shares are trading higher") with mu = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from newsalpha.calendar import event_time, session_close, session_open, sessions_in_range
from newsalpha.config import DataConfig
from newsalpha.data.sources.base import InMemoryData

PHI = 0.5
DRIFT_MINUTES = 2 * 390
BROKERS = ["Morgan Stanley", "Goldman Sachs", "JP Morgan", "Citi", "UBS", "Barclays"]
RATINGS = ["Sell", "Underweight", "Neutral", "Overweight", "Buy"]


@dataclass
class _Event:
    symbol: str
    created_at: pd.Timestamp
    headline: str
    mu: float


def _trading_minutes(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    parts = [
        pd.date_range(session_open(s), session_close(s), freq="min", inclusive="left")
        for s in sessions
    ]
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def _make_headline(rng: np.random.Generator, sym: str, kind: str) -> tuple[str, float]:
    if kind == "earnings":
        expected = rng.uniform(0.5, 3.0)
        s = rng.normal(0, 0.12)
        actual = expected * (1 + s)
        rev_e = rng.uniform(200, 5000)
        rev_s = 0.5 * s + rng.normal(0, 0.03)
        verb = "Beats" if s > 0 else "Misses"
        h = (
            f"{sym} Q3 EPS ${actual:.2f} {verb} ${expected:.2f} Estimate, "
            f"Sales ${rev_e * (1 + rev_s):.0f}M vs ${rev_e:.0f}M Est"
        )
        return h, 0.30 * s + 0.15 * rev_s
    if kind == "analyst":
        step = int(rng.choice([-1, 1]))
        old = int(rng.integers(1, 4))
        new = old + step
        tc = rng.normal(0.12 * step, 0.05)
        old_t = rng.uniform(40, 300)
        verb, tverb = ("Upgrades", "Raises") if step > 0 else ("Downgrades", "Lowers")
        h = (
            f"{rng.choice(BROKERS)} {verb} {sym} to {RATINGS[new]} from {RATINGS[old]}, "
            f"{tverb} Price Target to ${old_t * (1 + tc):.0f} from ${old_t:.0f}"
        )
        return h, 0.012 * step + 0.12 * tc
    if kind == "guidance":
        g = rng.normal(0, 0.06)
        lo = rng.uniform(1, 20)
        verb = "Raises" if g > 0 else "Lowers"
        h = (
            f"{sym} {verb} FY26 Revenue Guidance to ${lo * (1 + g):.2f}B-${lo * 1.05 * (1 + g):.2f}B "
            f"from ${lo:.2f}B-${lo * 1.05:.2f}B"
        )
        return h, 0.35 * g
    direction = rng.choice(["higher", "lower"])
    return f"{sym} Shares Are Trading {direction}: What's Going On?", 0.0


class SyntheticWorld(InMemoryData):
    @classmethod
    def generate(cls, cfg: DataConfig) -> SyntheticWorld:
        rng = np.random.default_rng(cfg.synthetic_seed)
        all_sessions = sessions_in_range("2024-01-02", "2026-12-31")
        sessions = all_sessions[: cfg.synthetic_sessions]
        minutes = _trading_minutes(sessions)
        n_min = len(minutes)
        symbols = [f"SYN{i:02d}" for i in range(cfg.synthetic_symbols)]

        mkt = rng.normal(0, 0.0003, n_min)
        bars, quotes, events = [], [], []
        kinds = ["earnings", "analyst", "guidance", "noise"]
        kind_p = [0.3, 0.35, 0.1, 0.25]

        # benchmark first
        bars.append(cls._bars_for(cfg.benchmark, minutes, np.cumsum(mkt), 400.0, rng, []))
        quotes.append(cls._quotes_for(cfg.benchmark, minutes, np.cumsum(mkt), 400.0, 0.5, rng, []))

        for sym in symbols:
            base_px = float(rng.uniform(15, 200))
            idio = rng.normal(0, 0.0007, n_min)
            logp = np.cumsum(mkt * rng.uniform(0.7, 1.3) + idio)
            impacts: list[tuple[int, float]] = []
            n_events = rng.poisson(len(sessions) / 3)
            for _ in range(n_events):
                s = sessions[rng.integers(0, len(sessions) - 3)]
                if rng.random() < 0.2:  # overnight / pre-market
                    created = session_open(s) - pd.Timedelta(minutes=float(rng.uniform(20, 600)))
                else:
                    span = (session_close(s) - session_open(s)).total_seconds()
                    created = session_open(s) + pd.Timedelta(
                        seconds=float(rng.uniform(0, span - 120))
                    )
                kind = rng.choice(kinds, p=kind_p)
                headline, mu = _make_headline(rng, sym, kind)
                mu += rng.normal(0, 0.004) if kind != "noise" else 0.0
                tau = event_time(created).tau
                k = int(minutes.searchsorted(tau))
                if k >= n_min:
                    continue
                impacts.append((k, mu))
                events.append(_Event(sym, created, headline, mu))
            profile = np.zeros(n_min)
            j = np.arange(n_min)
            for k, mu in impacts:
                after = j >= k
                profile[after] += PHI * mu * np.minimum(1, (j[after] - k + 1) / 5)
                profile[after] += (1 - PHI) * mu * np.minimum(1, (j[after] - k) / DRIFT_MINUTES)
            path = logp + profile
            spread_bps = float(rng.uniform(2, 12))
            bars.append(cls._bars_for(sym, minutes, path, base_px, rng, impacts))
            quotes.append(cls._quotes_for(sym, minutes, path, base_px, spread_bps, rng, impacts))

        articles = pd.DataFrame(
            {
                "article_id": [f"a{i:05d}" for i in range(len(events))],
                "symbol": [e.symbol for e in events],
                "created_at": pd.to_datetime([e.created_at for e in events], utc=True),
                "headline": [e.headline for e in events],
                "summary": "",
                "body": "",
                "source": "synthetic",
                "true_mu": [e.mu for e in events],  # ground truth, never used as a feature
            }
        )
        articles["updated_at"] = articles["created_at"]
        articles = articles.sort_values("created_at").reset_index(drop=True)
        return cls(
            articles, pd.concat(bars, ignore_index=True), pd.concat(quotes, ignore_index=True)
        )

    @staticmethod
    def _bars_for(sym, minutes, path, base_px, rng, impacts) -> pd.DataFrame:
        close = base_px * np.exp(path)
        open_ = np.r_[close[0], close[:-1]]
        wiggle = np.abs(rng.normal(0, 0.0004, len(close)))
        vol = rng.lognormal(7, 0.6, len(close))
        j = np.arange(len(close))
        for k, mu in impacts:
            after = j >= k
            vol[after] *= 1 + 8 * min(abs(mu) / 0.02, 3) * np.exp(-(j[after] - k) / 30)
        return pd.DataFrame(
            {
                "symbol": sym,
                "ts": minutes,
                "open": open_,
                "high": np.maximum(open_, close) * (1 + wiggle),
                "low": np.minimum(open_, close) * (1 - wiggle),
                "close": close,
                "volume": vol,
            }
        )

    @staticmethod
    def _quotes_for(sym, minutes, path, base_px, spread_bps, rng, impacts) -> pd.DataFrame:
        mid = base_px * np.exp(path)
        half = mid * spread_bps * (1 + rng.uniform(0, 0.3, len(mid))) / 2e4
        bid_size = rng.uniform(300, 700, len(mid))
        ask_size = rng.uniform(300, 700, len(mid))
        j = np.arange(len(mid))
        for k, mu in impacts:
            win = (j >= k) & (j < k + DRIFT_MINUTES // 6)
            tilt = np.sign(mu) * min(abs(mu) / 0.02, 1) * 0.6
            bid_size[win] *= 1 + max(tilt, 0)
            ask_size[win] *= 1 + max(-tilt, 0)
        return pd.DataFrame(
            {
                "symbol": sym,
                "ts": minutes + pd.Timedelta(minutes=1),  # snapshot at minute END
                "bid": mid - half,
                "ask": mid + half,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        )
