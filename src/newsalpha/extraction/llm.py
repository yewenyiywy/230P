"""LLM extractor over any OpenAI-compatible chat endpoint.

On the GPU server:   vllm serve <model> --port 8000   (base_url http://localhost:8000/v1)
As an API teacher:   point base_url at the provider and set NEWSALPHA_LLM_API_KEY.
A LoRA student:      vllm serve <base> --enable-lora --lora-modules student=<adapter dir>,
                     then model: student

The prompt lives in extraction/prompts/<prompt_version>.md. With `rag: true`
the `_rag` variant is used and the user message carries the firm's PRIOR news
(built by signals.novelty, strictly before the article). Output must be a JSON
object matching RawExtraction; anything else becomes an 'other' event with
confidence 0 (logged, never guessed).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests
from pydantic import ValidationError

from newsalpha.config import ExtractionConfig
from newsalpha.extraction.base import ArticleText
from newsalpha.extraction.schema import RawExtraction

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"
MAX_BODY_CHARS = 6000


def prompt_name(cfg: ExtractionConfig) -> str:
    return f"{cfg.prompt_version}_rag" if cfg.rag else cfg.prompt_version


def system_prompt(cfg: ExtractionConfig) -> str:
    return (PROMPTS / f"{prompt_name(cfg)}.md").read_text()


def format_user_message(item: ArticleText) -> str:
    """The one place an article becomes prompt text (shared with distill.dataset)."""
    msg = f"HEADLINE: {item.headline}\nSUMMARY: {item.summary}\nBODY: {item.body[:MAX_BODY_CHARS]}"
    if item.context:
        msg += f"\n\nPRIOR NEWS ABOUT THIS COMPANY (oldest first, all published earlier):\n{item.context}"
    return msg


class OpenAICompatClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        key = os.environ.get("NEWSALPHA_LLM_API_KEY", "EMPTY")
        self.session.headers.update({"Authorization": f"Bearer {key}"})

    def post(self, path: str, payload: dict) -> dict:
        r = self.session.post(f"{self.base_url}/{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def chat(self, payload: dict) -> dict:
        return self.post("chat/completions", payload)


class LLMExtractor:
    name = "llm"

    def __init__(self, cfg: ExtractionConfig):
        if not cfg.model:
            raise ValueError("extraction.model must name the served model")
        self.cfg = cfg
        self.version = f"{cfg.model.replace('/', '_')}_{prompt_name(cfg)}"
        self.system_prompt = system_prompt(cfg)
        self.client = OpenAICompatClient(cfg.base_url)

    def extract_one(self, item: ArticleText) -> RawExtraction:
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": format_user_message(item)},
            ],
        }
        try:
            content = self.client.chat(payload)["choices"][0]["message"]["content"]
            return RawExtraction.model_validate(json.loads(content))
        except (requests.RequestException, KeyError, json.JSONDecodeError, ValidationError) as e:
            log.warning("extraction failed for %r: %s", item.headline[:80], e)
            return RawExtraction(confidence=0.0)
