"""Stage 3, extraction: article text -> structured event fields.

Every extractor returns the same raw fields (`schema.RawExtraction`); the
surprise numbers are then computed in code by `schema.finalize`, never by the
LLM, so all methods are comparable and auditable.

    rules       regex over Benzinga phrasing; CI default and the "simple rule" baseline
    dictionary  Loughran-McDonald tone (T0)
    llm         any OpenAI-compatible endpoint: vLLM on the GPU server, or an API teacher (T3/T6)
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
    if cfg.method == "llm":
        from newsalpha.extraction.llm import LLMExtractor

        return LLMExtractor(cfg)
    raise ValueError(f"unknown extraction method {cfg.method}")
