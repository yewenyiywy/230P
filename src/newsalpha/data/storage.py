"""On-disk layout. Mirrors the Google Drive `230P/` folder (synced with rclone).

    <root>/raw/<source>/...         immutable vendor responses
    <root>/interim/<name>.parquet   cleaned tables
    <root>/features/<key>.parquet   cached LLM extractions (key = article hash + model + prompt)
    <root>/processed/<name>.parquet panels ready for modelling
    <root>/runs/<run_id>/           predictions, trades, report.json, report.md

Every write appends to <root>/manifest.jsonl (path, sha256, rows, time) so a
result can always be traced back to exact input files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LAYERS = ("raw", "interim", "features", "processed", "runs")


class DataStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for layer in LAYERS:
            (self.root / layer).mkdir(parents=True, exist_ok=True)

    def path(self, layer: str, name: str) -> Path:
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}")
        p = self.root / layer / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, df: pd.DataFrame, layer: str, name: str) -> Path:
        p = self.path(layer, name if name.endswith(".parquet") else f"{name}.parquet")
        df.to_parquet(p, index=False)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        entry = {
            "path": str(p.relative_to(self.root)),
            "sha256": digest,
            "rows": len(df),
            "written_at": datetime.now(UTC).isoformat(),
        }
        with open(self.root / "manifest.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return p

    def read(self, layer: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(
            self.path(layer, name if name.endswith(".parquet") else f"{name}.parquet")
        )

    def exists(self, layer: str, name: str) -> bool:
        return self.path(layer, name if name.endswith(".parquet") else f"{name}.parquet").exists()
