"""Extractor interface and the extraction cache.

LLM calls are the expensive step, so results are cached on disk keyed by
(article text hash, extractor name, extractor version). Changing a prompt means
bumping `version`, which re-extracts only what the new version has not seen.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import pandas as pd

from newsalpha.data.storage import DataStore
from newsalpha.extraction.schema import RawExtraction, finalize
from newsalpha.schemas import EXTRACTIONS, validate


class Extractor(Protocol):
    name: str
    version: str

    def extract_one(self, headline: str, summary: str, body: str) -> RawExtraction: ...


def article_hash(row: pd.Series) -> str:
    text = f"{row['headline']}\n{row.get('summary', '')}\n{row.get('body', '')}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def run_extractor(
    extractor: Extractor, articles: pd.DataFrame, store: DataStore | None = None
) -> pd.DataFrame:
    """Extract every unique article once (cached), return the EXTRACTIONS table."""
    uniq = articles.drop_duplicates("article_id").copy()
    uniq["text_hash"] = uniq.apply(article_hash, axis=1)
    cache_name = f"extractions_{extractor.name}_{extractor.version}"
    cached = (
        store.read("features", cache_name)
        if store is not None and store.exists("features", cache_name)
        else pd.DataFrame(columns=["text_hash"])
    )
    todo = uniq[~uniq["text_hash"].isin(cached["text_hash"])].drop_duplicates("text_hash")
    new_rows = []
    for _, row in todo.iterrows():
        body = row.get("body", "") if row.get("body_usable", True) else ""
        raw = extractor.extract_one(row["headline"], row.get("summary", ""), body)
        new_rows.append({"text_hash": row["text_hash"], **raw.model_dump(mode="json")})
    if new_rows:
        cached = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True)
        if store is not None:
            store.write(cached, "features", cache_name)
    raw_df = uniq[["article_id", "text_hash"]].merge(cached, on="text_hash", how="left")
    return validate(
        finalize(raw_df, f"{extractor.name}:{extractor.version}"), EXTRACTIONS, "extractions"
    )
