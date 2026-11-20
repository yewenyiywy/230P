"""LLM extractor over any OpenAI-compatible chat endpoint.

On the GPU server:   vllm serve <model> --port 8000   (base_url http://localhost:8000/v1)
As an API teacher:   point base_url at the provider and set NEWSALPHA_LLM_API_KEY.

The prompt lives in extraction/prompts/<prompt_version>.md. Output must be a
JSON object matching RawExtraction; anything else becomes an 'other' event
with confidence 0 (logged, never guessed).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests
from pydantic import ValidationError

from newsalpha.config import ExtractionConfig
from newsalpha.extraction.schema import RawExtraction

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"


class LLMExtractor:
    name = "llm"

    def __init__(self, cfg: ExtractionConfig):
        if not cfg.model:
            raise ValueError("extraction.model must name the served model")
        self.cfg = cfg
        self.version = f"{cfg.model.replace('/', '_')}_{cfg.prompt_version}"
        self.system_prompt = (PROMPTS / f"{cfg.prompt_version}.md").read_text()
        self.session = requests.Session()
        key = os.environ.get("NEWSALPHA_LLM_API_KEY", "EMPTY")
        self.session.headers.update({"Authorization": f"Bearer {key}"})

    def extract_one(self, headline: str, summary: str = "", body: str = "") -> RawExtraction:
        user = f"HEADLINE: {headline}\nSUMMARY: {summary}\nBODY: {body[:6000]}"
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = self.session.post(
                f"{self.cfg.base_url}/chat/completions", json=payload, timeout=120
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return RawExtraction.model_validate(json.loads(content))
        except (requests.RequestException, KeyError, json.JSONDecodeError, ValidationError) as e:
            log.warning("extraction failed for %r: %s", headline[:80], e)
            return RawExtraction(confidence=0.0)
