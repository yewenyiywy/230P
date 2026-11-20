"""Teacher vs. student agreement, field by field, on the time-split validation set.

For categorical fields: share of articles where both agree.
For surprise fields: among articles where the teacher has a value, the share
where the student has one too (coverage) and is within `tol` of it (accuracy).
The student is good enough to replace the teacher when agreement is high AND
ablation A5 shows no loss of net profit; this table answers only the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["event_type", "direction"]
NUMERIC = ["eps_surprise", "rev_surprise", "rating_change", "target_change", "guidance_change"]


def agreement(teacher: pd.DataFrame, student: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    m = teacher.merge(student, on="article_id", suffixes=("_t", "_s"))
    rows = []
    for f in CATEGORICAL:
        rows.append(
            {"field": f, "n": len(m), "agreement": float((m[f"{f}_t"] == m[f"{f}_s"]).mean())}
        )
    for f in NUMERIC:
        has_t = m[f"{f}_t"].notna()
        n = int(has_t.sum())
        if n == 0:
            rows.append({"field": f, "n": 0, "coverage": np.nan, "agreement": np.nan})
            continue
        t, s = m.loc[has_t, f"{f}_t"], m.loc[has_t, f"{f}_s"]
        rows.append(
            {
                "field": f,
                "n": n,
                "coverage": float(s.notna().mean()),
                "agreement": float(((s - t).abs() <= tol).mean()),
            }
        )
    return pd.DataFrame(rows)
