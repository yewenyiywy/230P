"""The extraction schema (plan: Signal -> event-specific extraction).

The LLM fills `RawExtraction`: numbers exactly as stated in the text, empty
when absent. `finalize` turns them into the EXTRACTIONS contract by computing
surprises in code.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class EventType(StrEnum):
    EARNINGS = "earnings"
    ANALYST = "analyst"
    GUIDANCE = "guidance"
    OTHER = "other"


# Analyst ratings mapped to a common -2..+2 scale.
RATING_SCALE = {
    "strong sell": -2, "sell": -2, "underperform": -1, "underweight": -1, "reduce": -1,
    "neutral": 0, "hold": 0, "equal-weight": 0, "equal weight": 0, "market perform": 0,
    "sector perform": 0, "in-line": 0, "peer perform": 0,
    "outperform": 1, "overweight": 1, "accumulate": 1, "sector outperform": 1,
    "buy": 2, "strong buy": 2, "conviction buy": 2,
}  # fmt: skip


def rating_value(r: object) -> float:
    if not isinstance(r, str) or not r:
        return np.nan
    return float(RATING_SCALE.get(r.strip().lower(), np.nan))


class RawExtraction(BaseModel):
    event_type: EventType = EventType.OTHER
    direction: int = Field(default=0, ge=-1, le=1)
    magnitude: int = Field(default=1, ge=1, le=5)
    confidence: float = Field(default=0.5, ge=0, le=1)
    eps_actual: float | None = None
    eps_expected: float | None = None
    revenue_actual: float | None = None
    revenue_expected: float | None = None
    rating_old: str | None = None
    rating_new: str | None = None
    target_old: float | None = None
    target_new: float | None = None
    guidance_old_mid: float | None = None
    guidance_new_mid: float | None = None
    evidence: str = ""  # quoted text supporting the fields, for audits


def _pct(new: pd.Series, old: pd.Series) -> pd.Series:
    new, old = pd.to_numeric(new, errors="coerce"), pd.to_numeric(old, errors="coerce")
    out = (new - old) / old.abs()
    return out.where(old.abs() > 1e-9)


def finalize(raw: pd.DataFrame, method: str) -> pd.DataFrame:
    """RawExtraction rows (+ article_id) -> EXTRACTIONS contract."""
    out = pd.DataFrame({"article_id": raw["article_id"].values, "method": method})
    out["event_type"] = raw["event_type"].astype(str).values
    out["eps_surprise"] = _pct(raw["eps_actual"], raw["eps_expected"]).clip(-1, 1).values
    out["rev_surprise"] = _pct(raw["revenue_actual"], raw["revenue_expected"]).clip(-1, 1).values
    out["rating_change"] = (
        raw["rating_new"].map(rating_value) - raw["rating_old"].map(rating_value)
    ).values
    out["target_change"] = _pct(raw["target_new"], raw["target_old"]).clip(-1, 1).values
    out["guidance_change"] = (
        _pct(raw["guidance_new_mid"], raw["guidance_old_mid"]).clip(-1, 1).values
    )
    # Direction from numbers when we have them; otherwise the extractor's own call.
    numeric = (
        out[["eps_surprise", "rating_change", "target_change", "guidance_change"]]
        .fillna(0)
        .sum(axis=1)
    )
    out["direction"] = np.where(numeric != 0, np.sign(numeric), raw["direction"].astype(float))
    out["magnitude"] = raw["magnitude"].astype(float).values
    out["confidence"] = raw["confidence"].astype(float).values
    return out
