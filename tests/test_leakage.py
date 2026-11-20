"""The tests that protect the paper: nothing known after entry may reach a feature.

Strategy: cut time at T. Corrupt every quote and bar stamped after T. Rebuild the
panel. Every row that entered at or before T must have IDENTICAL features; rows
whose exit is after T must have DIFFERENT labels (proving the corruption bit).
"""

import numpy as np
import pandas as pd

from newsalpha.align import QuoteBook
from newsalpha.evaluation.splits import walk_forward
from newsalpha.features.panel import build_panel
from newsalpha.schemas import feature_columns


def _corrupt_after(df: pd.DataFrame, cut: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    """Multiply every value stamped after `cut` by an independent random factor in [1.1, 1.5]."""
    out = df.copy()
    late = (out["ts"] > cut).to_numpy()
    factor = np.random.default_rng(42).uniform(1.1, 1.5, size=(late.sum(), 1))
    out.loc[late, cols] = out.loc[late, cols].to_numpy() * factor
    return out


def test_features_ignore_the_future(events, extractions, world, cfg, panel):
    cut = panel["entry_ts"].quantile(0.5)
    quotes = _corrupt_after(world.all_quotes, cut, ["bid", "ask", "bid_size", "ask_size"])
    bars = _corrupt_after(world.all_bars, cut, ["open", "high", "low", "close", "volume"])
    corrupted = build_panel(events, extractions, bars, quotes, cfg)

    key = ["event_id", "delay_min", "horizon"]
    a = panel.set_index(key).sort_index()
    b = corrupted.set_index(key).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    early = a["entry_ts"] <= cut
    assert early.sum() > 50
    feats = [c for c in feature_columns(panel) if c in a.columns]
    pd.testing.assert_frame_equal(
        a.loc[early, feats], b.loc[early, feats], check_exact=False, rtol=1e-12
    )

    late_exit = early & (a["exit_ts"] > cut)
    assert late_exit.sum() > 10
    assert not np.allclose(a.loc[late_exit, "target"], b.loc[late_exit, "target"])


def test_quotebook_never_returns_future_quote(world):
    book = QuoteBook(world.all_quotes)
    rng = np.random.default_rng(0)
    q = world.all_quotes
    times = q["ts"].sample(200, random_state=1) + pd.to_timedelta(
        rng.integers(0, 120, 200), unit="s"
    )
    got = book.at("SYN00", times.reset_index(drop=True))
    assert (got["quote_ts"] <= times.reset_index(drop=True)).all()


def test_entry_strictly_before_exit_and_after_news(panel, events):
    assert (panel["entry_ts"] < panel["exit_ts"]).all()
    m = panel.merge(events[["event_id", "created_at"]], on="event_id")
    assert (m["entry_ts"] > m["created_at"]).all()


def test_walk_forward_purges_overlapping_labels(panel):
    folds = list(walk_forward(panel, first_test="2024-01-01", min_train=50))
    assert folds
    for f in folds:
        assert (panel.loc[f.train, "exit_ts"] < f.start).all()
        assert (panel.loc[f.test, "entry_ts"] >= f.start).all()
        assert not (f.train & f.test).any()
