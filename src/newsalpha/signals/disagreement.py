"""Cross-model disagreement (T4, after Cheng, Hu & Li 2026).

Each method m gives article i a signed score s_mi = direction x confidence in
[-1, 1]. Across the M methods that scored the article:

    models_mean_score   mean of s_mi
    models_dispersion   standard deviation of s_mi (0 when M = 1)
    models_agreement    share of methods whose non-zero direction matches the majority sign
    models_conflict     1 if at least one method says up and another says down

Inputs are EXTRACTIONS tables from different methods; output is keyed by
article_id and broadcast to events in the pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def disagreement_features(extractions: list[pd.DataFrame]) -> pd.DataFrame:
    if not extractions:
        return pd.DataFrame(columns=["article_id"])
    long = pd.concat(
        [e[["article_id", "method", "direction", "confidence"]] for e in extractions],
        ignore_index=True,
    )
    long["score"] = long["direction"].fillna(0) * long["confidence"].fillna(0)

    def per_article(g: pd.DataFrame) -> pd.Series:
        s, d = g["score"].to_numpy(), np.sign(g["direction"].fillna(0).to_numpy())
        majority = np.sign(d.sum())
        nonzero = d[d != 0]
        return pd.Series(
            {
                "models_n": len(g),
                "models_mean_score": s.mean(),
                "models_dispersion": s.std(ddof=0),
                "models_agreement": (nonzero == majority).mean()
                if len(nonzero) and majority
                else 0.0,
                "models_conflict": float((d > 0).any() and (d < 0).any()),
            }
        )

    return long.groupby("article_id").apply(per_article, include_groups=False).reset_index()
