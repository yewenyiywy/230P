"""Stage 4b and the new extractors: time safety first, then arithmetic."""

import json

import numpy as np
import pandas as pd
import pytest

from newsalpha.config import ExtractionConfig, RunConfig
from newsalpha.extraction.base import ArticleText, run_extractor
from newsalpha.extraction.finbert import FinBERTExtractor
from newsalpha.extraction.llm import LLMExtractor, format_user_message
from newsalpha.extraction.probe import LogprobProbe, letter_probs, probs_to_raw
from newsalpha.extraction.schema import RawExtraction
from newsalpha.features.panel import build_panel
from newsalpha.pipeline import run_extraction
from newsalpha.signals.disagreement import disagreement_features
from newsalpha.signals.embed import HashingEmbedder
from newsalpha.signals.novelty import novelty_features


def _articles(rows):
    df = pd.DataFrame(rows, columns=["article_id", "symbol", "created_at", "headline"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df.assign(summary="", body="")


ROWS = [
    ("a1", "XYZ", "2025-01-02 15:00", "XYZ wins large defense contract"),
    ("a2", "XYZ", "2025-01-03 15:00", "XYZ wins large defense contract"),  # repeat
    (
        "a3",
        "XYZ",
        "2025-01-03 15:00",
        "XYZ names new chief financial officer",
    ),  # same instant as a2
    ("a4", "XYZ", "2025-06-01 15:00", "XYZ wins large defense contract"),  # > 90 days later
    ("b1", "ABC", "2025-01-02 14:00", "XYZ wins large defense contract"),  # other symbol
]


@pytest.fixture
def nov():
    feats, ctx = novelty_features(_articles(ROWS), HashingEmbedder(), days=90)
    return feats.set_index("event_id"), ctx


# ---------------------------------------------------------------- T5 novelty
def test_first_article_is_fully_novel(nov):
    feats, ctx = nov
    assert feats.loc["a1:XYZ", "novelty"] == 1.0 and feats.loc["a1:XYZ", "n_prior"] == 0
    assert ctx["a1"] == ""


def test_repeat_is_not_novel_and_context_holds_only_the_past(nov):
    feats, ctx = nov
    assert feats.loc["a2:XYZ", "novelty"] < 0.01
    assert "defense contract" in ctx["a2"] and "chief financial officer" not in ctx["a2"]


def test_same_instant_and_other_symbols_and_old_news_are_excluded(nov):
    feats, _ = nov
    assert feats.loc["a3:XYZ", "n_prior"] == 1  # a1 only: a2 has the same timestamp
    assert feats.loc["a4:XYZ", "n_prior"] == 0  # everything is > 90 days old
    assert feats.loc["b1:ABC", "n_prior"] == 0  # XYZ news never counts for ABC


def test_adding_future_news_never_changes_past_features(nov):
    feats, ctx = nov
    later = ROWS + [("a9", "XYZ", "2025-01-02 16:00", "XYZ names new chief financial officer")]
    f2, c2 = novelty_features(_articles(later), HashingEmbedder(), days=90)
    f2 = f2.set_index("event_id")
    pd.testing.assert_series_equal(feats.loc["a1:XYZ"], f2.loc["a1:XYZ"])
    assert c2["a1"] == ctx["a1"]


# ---------------------------------------------------------------- T4 disagreement
def test_disagreement_arithmetic():
    a = pd.DataFrame(
        {"article_id": ["x"], "method": ["m1"], "direction": [1.0], "confidence": [1.0]}
    )
    b = pd.DataFrame(
        {"article_id": ["x"], "method": ["m2"], "direction": [-1.0], "confidence": [0.5]}
    )
    d = disagreement_features([a, b]).set_index("article_id").loc["x"]
    assert d["models_n"] == 2 and np.isclose(d["models_mean_score"], 0.25)
    assert np.isclose(d["models_dispersion"], 0.75) and d["models_conflict"] == 1.0
    assert d["models_agreement"] == 0.0  # a 1-1 split has no majority


# ---------------------------------------------------------------- T1 FinBERT
def _fake_finbert(scores):
    return lambda texts, batch_size=32: [
        [{"label": k, "score": v} for k, v in scores.items()] for _ in texts
    ]


def test_finbert_maps_probabilities():
    pos = FinBERTExtractor(
        classifier=_fake_finbert({"positive": 0.7, "negative": 0.2, "neutral": 0.1})
    )
    r = pos.extract_one(ArticleText("x"))
    assert (r.direction, r.magnitude, r.confidence) == (1, 3, 0.7)
    neu = FinBERTExtractor(
        classifier=_fake_finbert({"positive": 0.3, "negative": 0.1, "neutral": 0.6})
    )
    assert neu.extract_one(ArticleText("x")).direction == 0


# ---------------------------------------------------------------- T4 probe
def test_letter_probs_merges_case_and_spaces():
    p = letter_probs(
        [
            {"token": " A", "logprob": np.log(0.5)},
            {"token": "a", "logprob": np.log(0.1)},
            {"token": "B", "logprob": np.log(0.2)},
            {"token": "The", "logprob": np.log(0.2)},
        ]
    )
    assert np.isclose(p["A"], 0.75) and np.isclose(p["B"], 0.25) and p["C"] == 0


def test_inner_confidence():
    r = probs_to_raw({"A": 0.9, "B": 0.1, "C": 0.0})
    assert r.direction == 1 and np.isclose(r.confidence, 0.8)
    assert probs_to_raw({"A": 0.5, "B": 0.5, "C": 0.0}).confidence == 0.0


def test_probe_reads_logprobs_from_endpoint(monkeypatch):
    probe = LogprobProbe(ExtractionConfig(method="probe", model="m"))
    resp = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": "B", "logprob": np.log(0.8)}, {"token": "A", "logprob": np.log(0.2)}]}]}}]}  # fmt: skip
    monkeypatch.setattr(probe.client, "chat", lambda payload: resp)
    r = probe.extract_one(ArticleText("XYZ recalls flagship product"))
    assert r.direction == -1 and np.isclose(r.confidence, 0.6)


# ---------------------------------------------------------------- RAG prompting and cache
def test_rag_prompt_and_novelty_field(monkeypatch):
    item = ArticleText("XYZ wins contract", context="- 2025-01-01: XYZ bids for contract")
    assert "PRIOR NEWS" in format_user_message(item)
    assert "PRIOR NEWS" not in format_user_message(ArticleText("XYZ wins contract"))
    llm = LLMExtractor(ExtractionConfig(method="llm", model="m", rag=True))
    assert llm.version.endswith("event_v1_rag")
    answer = {
        "event_type": "other",
        "direction": 1,
        "magnitude": 2,
        "confidence": 0.9,
        "novelty_llm": 0.2,
    }
    resp = {"choices": [{"message": {"content": json.dumps(answer)}}]}
    monkeypatch.setattr(llm.client, "chat", lambda payload: resp)
    assert llm.extract_one(item).novelty_llm == 0.2


class _Counting:
    name, version = "count", "v1"

    def __init__(self):
        self.calls = 0

    def extract_one(self, item):
        self.calls += 1
        return RawExtraction(direction=1 if item.context else -1, confidence=1.0)


def test_context_is_part_of_the_cache_key():
    arts = _articles(ROWS[:2])
    ex = _Counting()
    plain = run_extractor(ex, arts)
    with_ctx = run_extractor(ex, arts, contexts={"a1": "earlier news", "a2": "earlier news"})
    assert (plain["direction"] == -1).all() and (with_ctx["direction"] == 1).all()


# ---------------------------------------------------------------- integration
def test_signals_reach_the_panel(cfg, world):
    c = cfg.model_copy(deep=True)
    c.signals.compare = [ExtractionConfig(method="dictionary")]
    _, _, events, extractions, signals = run_extraction(c)
    assert {"novelty", "n_prior", "models_dispersion", "models_conflict"} <= set(signals.columns)
    assert signals["event_id"].is_unique and len(signals) == len(events)
    panel = build_panel(events, extractions, world.all_bars, world.all_quotes, c, signals=signals)
    assert {"novelty", "models_dispersion"} <= set(panel.columns)


def test_default_config_has_signals():
    assert RunConfig().signals.novelty
