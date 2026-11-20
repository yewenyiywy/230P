"""The whole study as seven explicit stages. Read this file top to bottom.

    1 load      source -> ARTICLES, MINUTE_BARS, QUOTES          (data.sources)
    2 clean     drop duplicates, flag revised bodies              (data.clean)
    3 align     ARTICLES -> EVENTS with event minute tau          (align)
    4 extract   ARTICLES -> EXTRACTIONS (cached)                  (extraction)
    4b signals  novelty vs prior news, cross-model disagreement   (signals)
    5 panel     EVENTS x delays x horizons -> PANEL               (features.panel)
    6 predict   walk-forward -> PREDICTIONS                       (models.walkforward)
    7 evaluate  choose on validation, score test -> report        (trading + evaluation)

Every stage writes its output to the DataStore, so any stage can be rerun or
inspected on its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from newsalpha.align import build_events
from newsalpha.config import RunConfig, config_hash
from newsalpha.data.clean import clean_articles
from newsalpha.data.sources import get_source
from newsalpha.data.sources.base import InMemoryData
from newsalpha.data.storage import DataStore
from newsalpha.evaluation.report import build_report, write_report
from newsalpha.extraction import get_extractor
from newsalpha.extraction.base import run_extractor
from newsalpha.features.panel import build_panel
from newsalpha.models.walkforward import predict_oos
from newsalpha.signals.disagreement import disagreement_features
from newsalpha.signals.embed import get_embedder
from newsalpha.signals.novelty import novelty_features

log = logging.getLogger(__name__)


def market_windows(events: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Time windows of quotes/bars a real-data run must fetch (events + exits + benchmark)."""
    from newsalpha.calendar import horizon_exit

    pre = pd.Timedelta(minutes=cfg.align.pre_window_min + 2)
    post = pd.Timedelta(minutes=max(cfg.align.delays_min) + 1)
    w = [events[["symbol"]].assign(start=events["tau"] - pre, end=events["tau"] + post)]
    for h in cfg.align.horizons:
        exits = pd.to_datetime([horizon_exit(s, h) for s in events["session_date"]], utc=True)
        w.append(events[["symbol"]].assign(start=exits - pd.Timedelta(minutes=5), end=exits))
    windows = pd.concat(w, ignore_index=True)
    bench = windows.assign(symbol=cfg.data.benchmark)
    return pd.concat([windows, bench], ignore_index=True).drop_duplicates()


def load_market(src, events: pd.DataFrame, cfg: RunConfig):
    if isinstance(src, InMemoryData):
        return src.all_bars, src.all_quotes, None
    bars, quotes = [], []
    for row in market_windows(events, cfg).itertuples():
        bars.append(src.minute_bars([row.symbol], row.start, row.end))
        quotes.append(src.quotes([row.symbol], row.start, row.end))
    symbols = sorted(set(events["symbol"]) | {cfg.data.benchmark})
    daily = src.daily_bars(
        symbols,
        events["tau"].min() - pd.Timedelta(days=45),
        events["tau"].max() + pd.Timedelta(days=10),
    )
    return pd.concat(bars), pd.concat(quotes).drop_duplicates(["symbol", "ts"]), daily


def load_articles(cfg: RunConfig, store: DataStore):
    """Stages 1-3: load, clean, align."""
    h = config_hash(cfg)
    src = get_source(cfg.data)
    articles = clean_articles(
        src.articles(),
        hours=cfg.align.dedupe_hours,
        similarity=cfg.align.dedupe_similarity,
        max_rev_min=cfg.align.max_revision_minutes,
    )
    store.write(articles.drop(columns=["true_mu"], errors="ignore"), "interim", f"articles_{h}")
    events = build_events(articles)
    store.write(events, "interim", f"events_{h}")
    return src, articles, events


def novelty_and_context(articles: pd.DataFrame, cfg: RunConfig):
    """T5 retrieval. Needed before extraction when the LLM prompt uses RAG context."""
    if not (cfg.signals.novelty or cfg.extraction.rag):
        return None, None
    return novelty_features(
        articles, get_embedder(cfg.signals), cfg.signals.novelty_days, cfg.extraction.rag_k
    )


def run_extraction(cfg: RunConfig):
    """Stages 1-4b. On the GPU server this warms every extraction cache."""
    store = DataStore(cfg.data.root)

    # 1-3 load, clean, align
    src, articles, events = load_articles(cfg, store)

    # 4 extract (with the firm's prior news in the prompt if rag is on)
    novelty, contexts = novelty_and_context(articles, cfg)
    extractions = run_extractor(
        get_extractor(cfg.extraction), articles, store, contexts if cfg.extraction.rag else None
    )

    # 4b signals: novelty (T5) and disagreement across methods (T4)
    signals = events[["event_id", "article_id"]].copy()
    if cfg.signals.novelty and novelty is not None:
        signals = signals.merge(novelty, on="event_id", how="left")
    if cfg.signals.compare:
        others = [run_extractor(get_extractor(c), articles, store) for c in cfg.signals.compare]
        signals = signals.merge(
            disagreement_features([extractions, *others]), on="article_id", how="left"
        )
    signals = signals.drop(columns="article_id")
    return src, store, events, extractions, signals


def run(cfg: RunConfig, out_root: str | Path | None = None) -> dict:
    h = config_hash(cfg)
    src, store, events, extractions, signals = run_extraction(cfg)

    # 5 panel
    bars, quotes, daily = load_market(src, events, cfg)
    panel = build_panel(events, extractions, bars, quotes, cfg, daily_bars=daily, signals=signals)
    store.write(panel, "processed", f"panel_{h}")
    log.info("panel: %d rows, %d events", len(panel), panel["event_id"].nunique())

    # 6 predict
    preds = predict_oos(panel, cfg, first_test=cfg.eval.train_start)

    # 7 evaluate
    report = build_report(preds, cfg, h)
    run_dir = Path(out_root or Path(cfg.data.root) / "runs") / f"{cfg.name}_{h}"
    run_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(run_dir / "predictions.parquet", index=False)
    write_report(report, run_dir)
    report["run_dir"] = str(run_dir)
    return report


# ---------------------------------------------------------------- T6 distillation
def distill_data(cfg: RunConfig) -> dict:
    """Teacher-label a stratified sample of TRAINING-period articles; write train/val JSONL."""
    from newsalpha.distill.dataset import sample_for_teacher, teacher_records, write_splits

    store = DataStore(cfg.data.root)
    _, articles, _ = load_articles(cfg, store)
    teacher_cfg = cfg.distill.teacher
    contexts = None
    if teacher_cfg.rag:
        _, contexts = novelty_and_context(
            articles, cfg.model_copy(update={"extraction": teacher_cfg})
        )
    sample = sample_for_teacher(
        articles, cfg.distill.n_samples, before=cfg.eval.test_start, seed=cfg.distill.seed
    )
    records = teacher_records(sample, get_extractor(teacher_cfg), teacher_cfg, contexts)
    return write_splits(records, cfg.distill.out_dir, cfg.distill.val_frac)


def distill_eval(cfg: RunConfig) -> pd.DataFrame:
    """Field agreement of the student (cfg.extraction) with the teacher on the validation split."""
    from newsalpha.distill.dataset import read_jsonl
    from newsalpha.distill.evaluate import agreement

    store = DataStore(cfg.data.root)
    _, articles, _ = load_articles(cfg, store)
    val_ids = {r["article_id"] for r in read_jsonl(Path(cfg.distill.out_dir) / "val.jsonl")}
    val = articles[articles["article_id"].isin(val_ids)]
    teacher = run_extractor(get_extractor(cfg.distill.teacher), val, store)
    student = run_extractor(get_extractor(cfg.extraction), val, store)
    table = agreement(teacher, student)
    table.to_csv(Path(cfg.distill.out_dir) / "agreement.csv", index=False)
    return table
