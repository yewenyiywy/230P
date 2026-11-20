from __future__ import annotations

import pytest

from newsalpha.align import build_events
from newsalpha.config import RunConfig
from newsalpha.data.clean import clean_articles
from newsalpha.data.sources.synthetic import SyntheticWorld
from newsalpha.extraction.base import run_extractor
from newsalpha.extraction.rules import RulesExtractor
from newsalpha.features.panel import build_panel


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> RunConfig:
    c = RunConfig(name="test")
    c.data.root = str(tmp_path_factory.mktemp("data"))
    c.data.synthetic_symbols = 8
    c.data.synthetic_sessions = 40
    c.align.delays_min = [5, 30]
    c.align.horizons = ["close_t0", "t1"]
    c.eval.train_start = "2024-01-01"
    c.eval.test_start = "2024-02-15"
    c.eval.min_train_events = 100
    return c


@pytest.fixture(scope="session")
def world(cfg):
    return SyntheticWorld.generate(cfg.data)


@pytest.fixture(scope="session")
def articles(world, cfg):
    a = cfg.align
    return clean_articles(
        world.articles(), a.dedupe_hours, a.dedupe_similarity, a.max_revision_minutes
    )


@pytest.fixture(scope="session")
def events(articles):
    return build_events(articles)


@pytest.fixture(scope="session")
def extractions(articles):
    return run_extractor(RulesExtractor(), articles)


@pytest.fixture(scope="session")
def panel(events, extractions, world, cfg):
    return build_panel(events, extractions, world.all_bars, world.all_quotes, cfg)
