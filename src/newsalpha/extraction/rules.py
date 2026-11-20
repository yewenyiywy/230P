"""Regex extractor for common Benzinga headline patterns.

Two jobs: (1) a free, deterministic extractor for CI and the synthetic world;
(2) the "simple event rule" baseline that the LLM must beat after costs.
"""

from __future__ import annotations

import re

from newsalpha.extraction.schema import EventType, RawExtraction

NUM = r"\$?(-?[\d,]*\.?\d+)([MB]\b)?"
EPS = re.compile(rf"EPS\s+{NUM}[^,]*?(?:Beats?|Miss(?:es)?|In-Line With|vs\.?)\s+{NUM}\s+Est", re.I)
SALES = re.compile(
    rf"(?:Sales|Revenue)\s+{NUM}[^,]*?(?:vs\.?|Beats?|Miss(?:es)?)\s+{NUM}\s+Est", re.I
)
RATING = re.compile(
    r"(Upgrades|Downgrades)\s+\S+\s+to\s+([A-Za-z\- ]+?)\s+from\s+([A-Za-z\- ]+?)(?:,|$)", re.I
)
TARGET = re.compile(
    rf"(?:Raises|Lowers|Maintains|Cuts|Boosts)\s+Price Target\s+to\s+{NUM}(?:\s+from\s+{NUM})?",
    re.I,
)
GUIDANCE = re.compile(
    rf"(Raises|Lowers|Cuts|Reaffirms|Sees).*?Guidance\s+to\s+{NUM}(?:\s*-\s*{NUM})?\s+from\s+{NUM}(?:\s*-\s*{NUM})?",
    re.I,
)
POS = re.compile(r"\b(beats?|raises?|upgrades?|higher|surges?|jumps?|record)\b", re.I)
NEG = re.compile(r"\b(miss(es)?|lowers?|cuts?|downgrades?|lower|plunges?|falls?|probe)\b", re.I)


def _f(x: str | None) -> float | None:
    return None if x is None else float(x.replace(",", ""))


def _mid(lo: str | None, hi: str | None) -> float | None:
    a, b = _f(lo), _f(hi)
    if a is None:
        return None
    return a if b is None else (a + b) / 2


class RulesExtractor:
    name = "rules"
    version = "v1"

    def extract_one(self, headline: str, summary: str = "", body: str = "") -> RawExtraction:
        text = headline
        if m := EPS.search(text):
            s = SALES.search(text)
            return RawExtraction(
                event_type=EventType.EARNINGS,
                eps_actual=_f(m.group(1)),
                eps_expected=_f(m.group(3)),
                revenue_actual=_f(s.group(1)) if s else None,
                revenue_expected=_f(s.group(3)) if s else None,
                magnitude=3,
                confidence=1.0,
                evidence=m.group(0),
            )
        r, t = RATING.search(text), TARGET.search(text)
        if r or t:
            return RawExtraction(
                event_type=EventType.ANALYST,
                rating_new=r.group(2).strip() if r else None,
                rating_old=r.group(3).strip() if r else None,
                target_new=_f(t.group(1)) if t else None,
                target_old=_f(t.group(3)) if t and t.group(3) else None,
                magnitude=2,
                confidence=1.0,
                evidence=(r.group(0) if r else "") + (t.group(0) if t else ""),
            )
        if g := GUIDANCE.search(text):
            return RawExtraction(
                event_type=EventType.GUIDANCE,
                guidance_new_mid=_mid(g.group(2), g.group(4)),
                guidance_old_mid=_mid(g.group(6), g.group(8)),
                magnitude=3,
                confidence=1.0,
                evidence=g.group(0),
            )
        pos, neg = len(POS.findall(text)), len(NEG.findall(text))
        direction = (pos > neg) - (pos < neg)
        return RawExtraction(direction=direction, magnitude=1, confidence=0.3 if direction else 0.1)
