"""Extractor interface and the extraction cache.

LLM calls are the expensive step, so results are cached on disk keyed by
(article text + context hash, extractor name, extractor version). Changing a
prompt means bumping `version`, which re-extracts only what it has not seen.

An extractor implements `extract_one`; it MAY also implement `extract_batch`
(FinBERT does), which `run_extractor` then prefers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from newsalpha.data.storage import DataStore
from newsalpha.extraction.schema import RawExtraction, finalize
from newsalpha.schemas import EXTRACTIONS, validate


@dataclass(frozen=True)
class ArticleText:
    headline: str
    summary: str = ""
    body: str = ""
    context: str = ""  # RAG: the firm's prior news, built time-safely by signals.novelty


class Extractor(Protocol):
    name: str
    version: str

    def extract_one(self, item: ArticleText) -> RawExtraction: ...


def text_hash(item: ArticleText) -> str:
    text = f"{item.headline}\n{item.summary}\n{item.body}\n--context--\n{item.context}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _items(articles: pd.DataFrame, contexts: dict[str, str] | None) -> list[ArticleText]:
    items = []
    for row in articles.itertuples(index=False):
        body = getattr(row, "body", "") or ""
        if not getattr(row, "body_usable", True):
            body = ""
        items.append(
            ArticleText(
                headline=row.headline,
                summary=getattr(row, "summary", "") or "",
                body=body,
                context=(contexts or {}).get(row.article_id, ""),
            )
        )
    return items


def run_extractor(
    extractor: Extractor,
    articles: pd.DataFrame,
    store: DataStore | None = None,
    contexts: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Extract every unique article once (cached), return the EXTRACTIONS table."""
    uniq = articles.drop_duplicates("article_id").reset_index(drop=True)
    items = _items(uniq, contexts)
    uniq["text_hash"] = [text_hash(i) for i in items]
    cache_name = f"extractions_{extractor.name}_{extractor.version}"
    cached = (
        store.read("features", cache_name)
        if store is not None and store.exists("features", cache_name)
        else pd.DataFrame(columns=["text_hash"])
    )
    missing = uniq[~uniq["text_hash"].isin(cached["text_hash"])]
    todo_idx = missing.drop_duplicates("text_hash").index.tolist()
    todo = [items[i] for i in todo_idx]
    if todo:
        batch = getattr(extractor, "extract_batch", None)
        results = batch(todo) if batch else [extractor.extract_one(i) for i in todo]
        new = pd.DataFrame(
            [
                {"text_hash": uniq.loc[i, "text_hash"], **r.model_dump(mode="json")}
                for i, r in zip(todo_idx, results, strict=True)
            ]
        )
        cached = pd.concat([cached, new], ignore_index=True)
        if store is not None:
            store.write(cached, "features", cache_name)
    raw_df = uniq[["article_id", "text_hash"]].merge(cached, on="text_hash", how="left")
    return validate(
        finalize(raw_df, f"{extractor.name}:{extractor.version}"), EXTRACTIONS, "extractions"
    )
