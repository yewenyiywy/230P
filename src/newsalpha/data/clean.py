"""Article cleaning (plan, alignment rules 7-8).

- Revised articles: the API returns the latest body, which may postdate the
  event. Bodies revised more than `max_revision_minutes` after creation are
  marked unusable; headlines are always kept.
- Duplicates: same symbol, near-identical headline within `hours` -> keep the
  earliest. Token Jaccard similarity is the default; swap in embedding cosine
  later without changing the interface.
"""

from __future__ import annotations

import re

import pandas as pd

_TOKEN = re.compile(r"[a-z0-9$.%]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mark_revisions(articles: pd.DataFrame, max_revision_minutes: float) -> pd.DataFrame:
    out = articles.copy()
    lag = (out["updated_at"] - out["created_at"]).dt.total_seconds() / 60
    out["body_usable"] = lag <= max_revision_minutes
    return out


def dedupe(articles: pd.DataFrame, hours: float, similarity: float) -> pd.DataFrame:
    keep_idx = []
    window = pd.Timedelta(hours=hours)
    for _, g in articles.sort_values("created_at").groupby("symbol", sort=False):
        kept: list[tuple[pd.Timestamp, frozenset[str]]] = []
        for idx, row in g.iterrows():
            toks = _tokens(row["headline"])
            dup = any(
                row["created_at"] - t <= window and jaccard(toks, k) >= similarity for t, k in kept
            )
            if not dup:
                kept.append((row["created_at"], toks))
                keep_idx.append(idx)
    return articles.loc[sorted(keep_idx)].reset_index(drop=True)


def clean_articles(articles: pd.DataFrame, hours: float, similarity: float, max_rev_min: float):
    return dedupe(mark_revisions(articles, max_rev_min), hours, similarity)
