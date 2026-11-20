"""Loughran-McDonald tone baseline (T0).

Ships with a tiny built-in word list so tests run offline. For real runs point
NEWSALPHA_LM_DICT at the official master dictionary CSV (columns Word,
Positive, Negative), downloaded from the Notre Dame SRAF site.
"""

from __future__ import annotations

import os
import re

import pandas as pd

from newsalpha.extraction.base import ArticleText
from newsalpha.extraction.schema import RawExtraction

_BUILTIN_POS = {
    "beat",
    "beats",
    "gain",
    "gains",
    "improve",
    "improved",
    "strong",
    "record",
    "raise",
    "raises",
    "upgrade",
    "upgrades",
    "surge",
    "exceed",
    "exceeds",
    "outperform",
}
_BUILTIN_NEG = {"miss", "misses", "loss", "losses", "decline", "weak", "lower", "lowers", "cut",
                "cuts", "downgrade", "downgrades", "probe", "lawsuit", "recall", "impairment"}  # fmt: skip
_WORD = re.compile(r"[a-z]+")


def _load() -> tuple[set[str], set[str]]:
    path = os.environ.get("NEWSALPHA_LM_DICT")
    if not path:
        return _BUILTIN_POS, _BUILTIN_NEG
    d = pd.read_csv(path)
    pos = set(d.loc[d["Positive"] > 0, "Word"].str.lower())
    neg = set(d.loc[d["Negative"] > 0, "Word"].str.lower())
    return pos, neg


class DictionaryExtractor:
    name = "lm_dictionary"
    version = "v1"

    def __init__(self) -> None:
        self.pos, self.neg = _load()

    def extract_one(self, item: ArticleText) -> RawExtraction:
        words = _WORD.findall(f"{item.headline} {item.summary} {item.body}".lower())
        p = sum(w in self.pos for w in words)
        n = sum(w in self.neg for w in words)
        tone = (p - n) / (p + n + 1)
        direction = int((tone > 0) - (tone < 0))
        return RawExtraction(
            direction=direction,
            magnitude=1 + min(4, abs(p - n)),
            confidence=min(1.0, abs(tone)),
        )
