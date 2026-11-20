"""The per-delay reaction gap (plan: Signal -> per-delay reaction gap).

    G_{i,d} = E_hat[ AR_{i,[tau, tau+d]} | text_i ] - AR_{i,[tau, tau+d]}

E_hat is a ridge regression of the realised reaction on TEXT features only,
fitted separately for each delay d, on the TRAINING fold only. The realised
reaction `ar_react` stays in the panel as its own feature, so the model can
tell a genuine underreaction from plain short-term reversal (ablation A3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from newsalpha.features.panel import TEXT_FEATURES


class GapModel:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: dict[int, Ridge] = {}

    def fit(self, train: pd.DataFrame) -> GapModel:
        for d, g in train.groupby("delay_min"):
            g = g.dropna(subset=["ar_react"])
            if len(g) < 20:
                continue
            self.models[int(d)] = Ridge(alpha=self.alpha).fit(
                g[TEXT_FEATURES].fillna(0).to_numpy(), g["ar_react"].to_numpy()
            )
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = panel.copy()
        expected = np.full(len(out), np.nan)
        for d, model in self.models.items():
            m = (out["delay_min"] == d).to_numpy()
            if m.any():
                expected[m] = model.predict(out.loc[m, TEXT_FEATURES].fillna(0).to_numpy())
        out["expected_react"] = expected
        out["gap"] = out["expected_react"] - out["ar_react"]
        return out
