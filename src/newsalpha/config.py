"""Typed experiment configuration, loaded from YAML.

One YAML file fully describes a run. Its hash (see `config_hash`) keys every
cached artefact and every logged result, so two runs with the same config are
guaranteed to be comparable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

Horizon = Literal["close_t0", "t1", "t3", "t5"]


def _default_horizons() -> list[Horizon]:
    return ["close_t0", "t1"]


class DataConfig(BaseModel):
    source: Literal["synthetic", "alpaca"] = "synthetic"
    root: str = "data"  # local mirror of the Google Drive 230P folder
    symbols: list[str] = Field(default_factory=list)  # empty = all symbols the source returns
    start: str = "2016-01-01"
    end: str = "2026-11-01"
    benchmark: str = "SPY"
    sector_etf: dict[str, str] = Field(
        default_factory=dict
    )  # symbol -> hedge ETF; default benchmark
    min_price: float = 5.0
    # synthetic-only knobs
    synthetic_symbols: int = 20
    synthetic_sessions: int = 60
    synthetic_seed: int = 7


class AlignConfig(BaseModel):
    delays_min: list[int] = Field(default_factory=lambda: [1, 5, 15, 30, 60])
    horizons: list[Horizon] = Field(default_factory=_default_horizons)
    pre_window_min: int = 30
    dedupe_hours: float = 24.0
    dedupe_similarity: float = 0.9
    max_revision_minutes: float = 60.0


class ExtractionConfig(BaseModel):
    method: Literal["rules", "dictionary", "llm"] = "rules"
    model: str = ""  # e.g. a vLLM-served model name
    base_url: str = "http://localhost:8000/v1"  # OpenAI-compatible endpoint
    prompt_version: str = "event_v1"
    temperature: float = 0.0
    model_cutoff: str | None = None  # LLM knowledge cutoff, ISO date; required for real runs


class ModelConfig(BaseModel):
    kind: Literal["ridge", "lgbm"] = "lgbm"
    params: dict = Field(default_factory=dict)
    features: list[str] | None = None  # None = all feature columns


class TradingConfig(BaseModel):
    variant: Literal["long_short_hedged", "long_only_hedged", "long_only"] = "long_short_hedged"
    kappa: float = 0.5  # trade only if edge > kappa * forecast sigma
    impact_k: float = 1.0
    borrow_bps_per_year: float = 100.0
    no_short_below_price: float = 10.0
    etf_half_spread_bps: float = 0.5
    cap_adv_frac: float = 0.01
    cap_capital_frac: float = 0.05
    risk_scale: float = 0.002  # notional = capital * risk_scale * edge / sigma_daily^2, then capped
    capital: float = 1_000_000.0


class EvalConfig(BaseModel):
    train_start: str = "2016-01-01"
    test_start: str = "2024-07-01"  # must be after the LLM's knowledge cutoff for headline results
    test_end: str | None = None
    retrain_every_months: int = 1
    min_train_events: int = 200
    min_tradable_delay_min: int = 5  # selection ignores delays faster than our measured latency
    n_trials_logged: int = 1  # updated from the experiment log; feeds the deflated Sharpe


class RunConfig(BaseModel):
    name: str = "run"
    data: DataConfig = Field(default_factory=DataConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return RunConfig.model_validate(raw)


def config_hash(cfg: RunConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
