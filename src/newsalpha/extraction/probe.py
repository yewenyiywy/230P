"""Token-probability confidence probe (T4, after Chen, Didisheim & Somoza 2026).

Asks for ONE letter (A = good for the stock, B = bad, C = neither) and reads the
model's next-token probabilities instead of trusting a stated confidence:

    p_up, p_down   normalised probabilities of 'A' and 'B' among A/B/C
    direction      argmax over A/B/C (C -> 0)
    confidence     inner confidence = 2 * |p_up / (p_up + p_down) - 0.5|   in [0, 1]

Needs an endpoint that returns `logprobs` with `top_logprobs` (vLLM does; most
closed APIs return only the top few, which is still enough for one letter).
"""

from __future__ import annotations

import logging
import math

import requests

from newsalpha.config import ExtractionConfig
from newsalpha.extraction.base import ArticleText
from newsalpha.extraction.llm import OpenAICompatClient, format_user_message
from newsalpha.extraction.schema import RawExtraction

log = logging.getLogger(__name__)

PROBE_PROMPT = (
    "You read one financial news article about one company. Using only the article, "
    "is it good or bad news for the company's stock price over the next few days?\n"
    "Answer with exactly one letter: A = good, B = bad, C = neither or unclear."
)


def letter_probs(top_logprobs: list[dict]) -> dict[str, float]:
    """Sum probabilities of tokens that are A/B/C (ignoring case and spaces), renormalised."""
    mass = {"A": 0.0, "B": 0.0, "C": 0.0}
    for t in top_logprobs:
        tok = t["token"].strip().upper()
        if tok in mass:
            mass[tok] += math.exp(t["logprob"])
    total = sum(mass.values())
    return {k: v / total for k, v in mass.items()} if total > 0 else {}


def probs_to_raw(p: dict[str, float]) -> RawExtraction:
    if not p:
        return RawExtraction(confidence=0.0)
    up, down = p["A"], p["B"]
    top = max(p, key=lambda k: p[k])
    direction = {"A": 1, "B": -1, "C": 0}[top]
    inner = 2 * abs(up / (up + down) - 0.5) if up + down > 0 else 0.0
    return RawExtraction(
        direction=direction, magnitude=1 + round(4 * abs(up - down)), confidence=inner
    )


class LogprobProbe:
    name = "probe"

    def __init__(self, cfg: ExtractionConfig):
        if not cfg.model:
            raise ValueError("extraction.model must name the served model")
        self.cfg = cfg
        self.version = f"{cfg.model.replace('/', '_')}_probe_v1"
        self.client = OpenAICompatClient(cfg.base_url)

    def extract_one(self, item: ArticleText) -> RawExtraction:
        payload = {
            "model": self.cfg.model,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 10,
            "messages": [
                {"role": "system", "content": PROBE_PROMPT},
                {"role": "user", "content": format_user_message(item)},
            ],
        }
        try:
            resp = self.client.chat(payload)
            top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            return probs_to_raw(letter_probs(top))
        except (requests.RequestException, KeyError, IndexError, TypeError) as e:
            log.warning("probe failed for %r: %s", item.headline[:80], e)
            return RawExtraction(confidence=0.0)
