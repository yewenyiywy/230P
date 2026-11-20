"""End to end on the synthetic world: the pipeline runs and finds the planted effect."""

from newsalpha.config import RunConfig
from newsalpha.pipeline import run


def test_full_pipeline_recovers_planted_signal(tmp_path):
    cfg = RunConfig(name="smoke")
    cfg.data.root = str(tmp_path / "data")
    cfg.data.synthetic_symbols = 12
    cfg.data.synthetic_sessions = 70
    cfg.align.delays_min = [5, 30]
    cfg.align.horizons = ["close_t0", "t1"]
    cfg.model.kind = "ridge"
    cfg.trading.kappa = 0.1
    cfg.eval.train_start = "2024-01-01"
    cfg.eval.test_start = "2024-03-01"
    cfg.eval.min_train_events = 150

    report = run(cfg, tmp_path / "runs")

    assert report["selected"]["chosen_on"] == "validation"
    assert set(report["headline_test"]) == {"long_short_hedged", "long_only_hedged", "long_only"}
    assert [r["delay_min"] for r in report["decay_test"]] == [5, 30]
    # the synthetic world plants a drift after the initial reaction
    assert report["decay_test"][0]["ic"]["mean"] > 0
    assert (tmp_path / "runs").exists()
