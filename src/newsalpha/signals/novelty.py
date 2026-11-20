"""Time-safe retrieval of a firm's prior news (T5).

For each event (article x symbol) the candidate set is that symbol's articles with

    created_at in [created_at - days, created_at)      strictly earlier, never the same instant

From it we compute, per event:
    novelty          1 - max cosine similarity to any prior article (1.0 if none)
    max_prior_sim    max cosine similarity
    n_prior          number of prior articles in the window (news intensity)
    hours_since_prior  hours since the most recent prior article (NaN if none)
and, for the RAG prompt, the text of the k most similar prior headlines in time order.

Why it matters: only the unpredictable part of news is mispriced (Didisheim,
Kelly et al. 2026), and "is this new?" can only be judged against what was
already public. tests/test_signals.py checks that no later article is ever used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.signals.embed import Embedder


def _texts(articles: pd.DataFrame) -> list[str]:
    summary = articles.get("summary", pd.Series("", index=articles.index)).fillna("")
    return (articles["headline"].fillna("") + ". " + summary).str.strip(". ").tolist()


def novelty_features(
    articles: pd.DataFrame, embedder: Embedder, days: int = 90, k_context: int = 5
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Returns (features keyed by event_id, RAG context keyed by article_id)."""
    a = articles.sort_values("created_at").reset_index(drop=True)
    vecs = embedder.embed(_texts(a))
    window = pd.Timedelta(days=days)

    rows: list[dict] = []
    contexts: dict[str, str] = {}
    for _sym, idx in a.groupby("symbol").groups.items():
        idx = np.asarray(sorted(idx))
        times = a.loc[idx, "created_at"].to_numpy()
        for pos, i in enumerate(idx):
            t = times[pos]
            lo = np.searchsorted(times, t - window.to_timedelta64(), side="left")
            hi = np.searchsorted(times, t, side="left")  # strictly before t
            prior = idx[lo:hi]
            row = {
                "event_id": f"{a.at[i, 'article_id']}:{a.at[i, 'symbol']}",
                "n_prior": len(prior),
            }
            if len(prior):
                sims = vecs[prior] @ vecs[i]
                row["max_prior_sim"] = float(sims.max())
                row["novelty"] = 1.0 - float(sims.max())
                row["hours_since_prior"] = (t - times[hi - 1]) / np.timedelta64(1, "h")
                top = prior[np.argsort(-sims)[:k_context]]
                top = top[np.argsort(a.loc[top, "created_at"].to_numpy())]
                ctx = "\n".join(
                    f"- {a.at[j, 'created_at']:%Y-%m-%d %H:%M} UTC: {a.at[j, 'headline']}"
                    for j in top
                )
            else:
                row.update(max_prior_sim=0.0, novelty=1.0, hours_since_prior=np.nan)
                ctx = ""
            rows.append(row)
            contexts.setdefault(a.at[i, "article_id"], ctx)
    return pd.DataFrame(rows), contexts
