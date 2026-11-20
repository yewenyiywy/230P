"""FinBERT baseline (T1): ProsusAI/finbert, a BERT classifier fine-tuned on financial text.

Reads headline + summary (FinBERT was trained on sentences, and long bodies are
truncated at 512 tokens anyway). Output:
    direction   sign(p_positive - p_negative) unless 'neutral' wins
    magnitude   1 + round(4 * |p_positive - p_negative|)
    confidence  the winning class probability

Runs on CPU (about 110M parameters) or GPU. Needs the `ml` extra. Tests inject
`classifier` (any callable with the transformers pipeline's output shape), so
they run without downloading weights.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from newsalpha.extraction.base import ArticleText
from newsalpha.extraction.schema import RawExtraction


class FinBERTExtractor:
    name = "finbert"

    def __init__(
        self,
        model: str = "ProsusAI/finbert",
        batch_size: int = 32,
        device: int | None = None,
        classifier: Callable[..., list[list[dict]]] | None = None,
    ):
        self.version = model.replace("/", "_")
        self.batch_size = batch_size
        self.clf: Any
        if classifier is not None:
            self.clf = classifier
            return
        import torch
        from transformers import pipeline

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.clf = pipeline(
            "text-classification", model=model, top_k=None, truncation=True, device=device
        )

    @staticmethod
    def _text(item: ArticleText) -> str:
        return f"{item.headline}. {item.summary}".strip(". ") if item.summary else item.headline

    @staticmethod
    def _to_raw(scores: list[dict]) -> RawExtraction:
        p = {s["label"].lower(): float(s["score"]) for s in scores}
        pos, neg, neu = p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)
        top = max(p, key=lambda k: p[k])
        diff = pos - neg
        direction = 0 if top == "neutral" else (1 if diff > 0 else -1)
        return RawExtraction(
            direction=direction,
            magnitude=1 + round(4 * abs(diff)),
            confidence=max(pos, neg, neu),
        )

    def extract_one(self, item: ArticleText) -> RawExtraction:
        return self.extract_batch([item])[0]

    def extract_batch(self, items: list[ArticleText]) -> list[RawExtraction]:
        out = self.clf([self._text(i) for i in items], batch_size=self.batch_size)
        return [self._to_raw(scores) for scores in out]
