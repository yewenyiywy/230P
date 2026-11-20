"""Teacher dataset for distillation.

Leakage rule: the sample is drawn ONLY from articles before `before` (the test
start), so the student never learns from test-period text.

Sampling is stratified by calendar month (equal draws per month, capped by
availability) so the student sees every regime, not just busy periods.
Train/validation split is by time: the latest `val_frac` of the sample is
validation, so validation measures how the student generalises forward.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from newsalpha.config import ExtractionConfig
from newsalpha.extraction.base import ArticleText, Extractor, _items
from newsalpha.extraction.llm import format_user_message, system_prompt


def sample_for_teacher(articles: pd.DataFrame, n: int, before: str, seed: int = 0) -> pd.DataFrame:
    a = articles.drop_duplicates("article_id")
    a = a[a["created_at"] < pd.Timestamp(before, tz="UTC")]
    if a.empty:
        return a
    month = a["created_at"].dt.strftime("%Y-%m")
    per_month = max(1, n // month.nunique())
    parts = [g.sample(min(len(g), per_month), random_state=seed) for _, g in a.groupby(month)]
    return pd.concat(parts).sort_values("created_at").head(n).reset_index(drop=True)


def teacher_records(
    sample: pd.DataFrame,
    teacher: Extractor,
    prompt_cfg: ExtractionConfig,
    contexts: dict[str, str] | None = None,
) -> list[dict]:
    """One chat example per article: system prompt, article, teacher's JSON answer."""
    sys = system_prompt(prompt_cfg)
    items: list[ArticleText] = _items(sample, contexts)
    batch = getattr(teacher, "extract_batch", None)
    answers = batch(items) if batch else [teacher.extract_one(i) for i in items]
    records = []
    for row, item, ans in zip(sample.itertuples(index=False), items, answers, strict=True):
        if ans.confidence == 0.0 and ans.event_type == "other":  # failed teacher call: skip
            continue
        records.append(
            {
                "article_id": row.article_id,
                "created_at": row.created_at.isoformat(),
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": format_user_message(item)},
                    {"role": "assistant", "content": ans.model_dump_json(exclude_none=False)},
                ],
            }
        )
    return records


def write_splits(records: list[dict], out_dir: str | Path, val_frac: float = 0.15) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda r: r["created_at"])
    n_val = max(1, int(round(len(records) * val_frac))) if len(records) > 1 else 0
    splits = {"train": records[: len(records) - n_val], "val": records[len(records) - n_val :]}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    meta = {k: len(v) for k, v in splits.items()}
    if splits["train"]:
        meta["train_last"] = splits["train"][-1]["created_at"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
