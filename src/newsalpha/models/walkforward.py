"""Out-of-sample predictions for every panel row in the evaluation period.

Per fold: fit the gap model on train rows -> add `gap` to train and test ->
fit the return model on train -> predict test. sigma_hat is the training
standard deviation of the target per (delay, horizon).
"""

from __future__ import annotations

import logging

import pandas as pd

from newsalpha.config import RunConfig
from newsalpha.evaluation.splits import walk_forward
from newsalpha.features.gap import GapModel
from newsalpha.models.base import make_model
from newsalpha.schemas import feature_columns

log = logging.getLogger(__name__)


def predict_oos(panel: pd.DataFrame, cfg: RunConfig, first_test: str | None = None) -> pd.DataFrame:
    """Walk-forward from `first_test` (default: eval.train_start + warm-up) to the end of the panel."""
    longest = panel["horizon_days"].max()
    embargo = pd.Timedelta(days=0 if pd.isna(longest) else int(longest))
    first = first_test or panel["entry_ts"].min().normalize().tz_localize(None)
    out = []
    for fold in walk_forward(
        panel,
        first_test=first,
        months=cfg.eval.retrain_every_months,
        min_train=cfg.eval.min_train_events,
        embargo=embargo,
        last_test=cfg.eval.test_end,
    ):
        train, test = panel[fold.train], panel[fold.test]
        gap = GapModel().fit(train)
        train, test = gap.transform(train), gap.transform(test)
        cols = cfg.model.features or feature_columns(train)
        model = make_model(cfg.model).fit(train[cols], train["target"])
        sigma = train.groupby(["delay_min", "horizon"])["target"].std().rename("sigma_hat")
        pred = test.copy()
        pred["pred"] = model.predict(test[cols])
        pred = pred.merge(sigma, on=["delay_min", "horizon"], how="left")
        pred["fold_start"] = fold.start
        out.append(pred)
        log.info("fold %s: train=%d test=%d", fold.start.date(), fold.train.sum(), fold.test.sum())
    if not out:
        raise RuntimeError("no walk-forward fold had enough training data; lower min_train_events")
    preds = pd.concat(out, ignore_index=True)
    test_start = pd.Timestamp(cfg.eval.test_start, tz="UTC")
    preds["split"] = (preds["entry_ts"] >= test_start).map({True: "test", False: "validation"})
    return preds
