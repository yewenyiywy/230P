"""The report: choose on validation, score once on test, profit metrics first.

1. Selection: every (delay, horizon) strategy is backtested on the VALIDATION
   split; the one with the best net Sharpe is chosen. All of them count as
   trials for the deflated Sharpe.
2. Headline: the chosen strategy on the TEST split, for all three variants.
3. Decay curve, capacity sweep and diagnostics, also on TEST.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from newsalpha.config import RunConfig
from newsalpha.evaluation.metrics import rank_ic, strategy_metrics
from newsalpha.trading.backtest import backtest

VARIANTS = ("long_short_hedged", "long_only_hedged", "long_only")


def _bt(preds: pd.DataFrame, cfg: RunConfig, d: int, h: str, **overrides):
    tcfg = cfg.trading.model_copy(update=overrides)
    return backtest(preds, tcfg, d, h)


def build_report(preds: pd.DataFrame, cfg: RunConfig, config_hash: str) -> dict:
    val, test = preds[preds["split"] == "validation"], preds[preds["split"] == "test"]
    tradable = [d for d in cfg.align.delays_min if d >= cfg.eval.min_tradable_delay_min]
    combos = [(d, h) for d in (tradable or cfg.align.delays_min) for h in cfg.align.horizons]
    n_trials = max(cfg.eval.n_trials_logged, len(combos) * len(VARIANTS))

    # 1. selection on validation
    scores = []
    for d, h in combos:
        trades, daily = _bt(val, cfg, d, h)
        m = (
            strategy_metrics(trades, daily, n_trials)
            if len(daily)
            else {"net_sharpe": float("nan")}
        )
        scores.append({"delay_min": d, "horizon": h, **m})
    sel = pd.DataFrame(scores)
    best = sel.sort_values("net_sharpe", ascending=False, na_position="last").iloc[0]
    d_star, h_star = int(best["delay_min"]), str(best["horizon"])

    warnings = []
    if test.empty:
        warnings.append("Test split is empty: headline numbers below are from VALIDATION.")
        test = val
    cutoff = cfg.extraction.model_cutoff
    leaky = bool(cutoff) and pd.Timestamp(cfg.eval.test_start) <= pd.Timestamp(cutoff)
    if cfg.extraction.method == "llm" and leaky:
        warnings.append(
            "test_start is not after the LLM knowledge cutoff: results may be memorised."
        )

    # 2. headline, all variants
    headline = {}
    for v in VARIANTS:
        trades, daily = _bt(test, cfg, d_star, h_star, variant=v)
        headline[v] = strategy_metrics(trades, daily, n_trials)

    # 3. decay curve, capacity, diagnostics
    decay = []
    for d in cfg.align.delays_min:
        trades, daily = _bt(test, cfg, d, h_star)
        sub = test[(test["delay_min"] == d) & (test["horizon"] == h_star)]
        decay.append(
            {"delay_min": d, **strategy_metrics(trades, daily, n_trials), "ic": rank_ic(sub)}
        )
    capacity = []
    for mult in (1, 10, 100):
        capital = cfg.trading.capital * mult
        trades, daily = _bt(test, cfg, d_star, h_star, capital=capital)
        m = strategy_metrics(trades, daily, n_trials)
        capacity.append(
            {
                "capital": capital,
                "net_pnl": m.get("net_pnl", 0.0),
                "net_bps_per_trade": m.get("net_bps_per_trade", float("nan")),
                "net_sharpe": m["net_sharpe"],
            }
        )
    main_trades, _ = _bt(test, cfg, d_star, h_star)
    by_type = (
        main_trades.groupby("event_type")["net_ret"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_trades", "mean": "net_ret"})
        if len(main_trades)
        else pd.DataFrame()
    )
    if not by_type.empty:
        by_type["net_bps"] = by_type.pop("net_ret") * 1e4

    return {
        "name": cfg.name,
        "config_hash": config_hash,
        "warnings": warnings,
        "n_trials": n_trials,
        "min_tradable_delay_min": cfg.eval.min_tradable_delay_min,
        "selected": {"delay_min": d_star, "horizon": h_star, "chosen_on": "validation"},
        "headline_test": headline,
        "decay_test": decay,
        "capacity_test": capacity,
        "by_event_type_test": by_type.reset_index().to_dict(orient="records")
        if len(by_type)
        else [],
        "validation_grid": sel.to_dict(orient="records"),
    }


def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:,.3f}"
    return str(x)


def write_report(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    lines = [f"# {report['name']} ({report['config_hash']})", ""]
    lines += [f"> WARNING: {w}" for w in report["warnings"]]
    s = report["selected"]
    lines += [
        f"Selected on validation among delays >= {report['min_tradable_delay_min']} min: "
        f"entry delay {s['delay_min']} min, horizon {s['horizon']}. "
        f"Trials counted for the deflated Sharpe: {report['n_trials']}.",
        "",
        "## Headline (test, after costs)",
        "",
    ]
    keys = ["net_pnl", "net_sharpe", "deflated_sharpe", "max_drawdown", "net_bps_per_trade",
            "est_cost_bps_per_trade", "n_trades", "hit_rate", "payoff_ratio"]  # fmt: skip
    lines.append("| variant | " + " | ".join(keys) + " |")
    lines.append("|---" * (len(keys) + 1) + "|")
    for v, m in report["headline_test"].items():
        lines.append(f"| {v} | " + " | ".join(_fmt(m.get(k, float("nan"))) for k in keys) + " |")
    lines += ["", "## Decay curve (test)", "", "| delay_min | net_bps_per_trade | net_sharpe | n_trades | rank IC |",
              "|---|---|---|---|---|"]  # fmt: skip
    for r in report["decay_test"]:
        lines.append(
            f"| {r['delay_min']} | {_fmt(r.get('net_bps_per_trade', float('nan')))} | "
            f"{_fmt(r.get('net_sharpe', float('nan')))} | {r.get('n_trades', 0)} | {_fmt(r['ic']['mean'])} |"
        )
    lines += ["", "## Capacity (test): same strategy at larger capital", "",
              "| capital | net P&L | net bps/trade | net Sharpe |", "|---|---|---|---|"]  # fmt: skip
    lines += [
        f"| {c['capital']:,.0f} | {c['net_pnl']:,.0f} | {_fmt(c['net_bps_per_trade'])} | {_fmt(c['net_sharpe'])} |"
        for c in report["capacity_test"]
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
