"""Stage 3, extraction: article text -> structured event fields.

Every extractor returns the same raw fields (`schema.RawExtraction`); the
surprise numbers are then computed in code by `schema.finalize`, never by the
LLM, so all methods are comparable and auditable.

    rules       regex over Benzinga phrasing; CI default and the "simple rule" baseline
    dictionary  Loughran-McDonald tone (T0)
    finbert     ProsusAI/finbert sentence classifier, batched, CPU or GPU (T1)
    llm         any OpenAI-compatible endpoint: vLLM on the GPU server, or an API teacher (T3/T6);
                with cfg.rag the prompt also gets the firm's prior news (T5)
    probe       one-token good/bad/neutral answer read from token probabilities (T4 confidence)
"""

from __future__ import annotations

from newsalpha.config import ExtractionConfig
from newsalpha.extraction.base import Extractor


def get_extractor(cfg: ExtractionConfig) -> Extractor:
    if cfg.method == "rules":
        from newsalpha.extraction.rules import RulesExtractor

        return RulesExtractor()
    if cfg.method == "dictionary":
        from newsalpha.extraction.dictionary import DictionaryExtractor

        return DictionaryExtractor()
    if cfg.method == "finbert":
        from newsalpha.extraction.finbert import FinBERTExtractor

        return FinBERTExtractor(cfg.model or "ProsusAI/finbert", batch_size=cfg.batch_size)
    if cfg.method == "llm":
        from newsalpha.extraction.llm import LLMExtractor

        return LLMExtractor(cfg)
    if cfg.method == "probe":
        from newsalpha.extraction.probe import LogprobProbe

        return LogprobProbe(cfg)
    raise ValueError(f"unknown extraction method {cfg.method}")
