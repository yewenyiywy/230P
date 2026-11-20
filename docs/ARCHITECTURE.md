# Architecture

Three rules shape everything:

1. **One direction.** Stages only consume tables produced by earlier stages.
   No stage reaches back into a vendor API or into a later stage.
2. **Typed tables between stages.** The contracts in `schemas.py` are the only
   interface. Swapping an extractor, model or data vendor means producing the
   same table, nothing else.
3. **Time is checked, not trusted.** Every price used anywhere goes through
   `align.QuoteBook.at`, which only returns quotes stamped at or before the
   requested time. `tests/test_leakage.py` corrupts all data after a cut-off
   and proves no earlier feature moves.

## Data contracts

| Table | Key | Produced by | Notes |
|---|---|---|---|
| ARTICLES | (article_id, symbol) | `data.sources.*` | one row per tagged symbol; UTC `created_at`, `updated_at` |
| MINUTE_BARS | (symbol, ts) | `data.sources.*` | `ts` = minute start |
| QUOTES | (symbol, ts) | `data.sources.*` | last NBBO of each minute, `ts` = minute **end** |
| EVENTS | event_id | `align.build_events` | `tau`, `session_date`, `overnight`, `body_usable` |
| EXTRACTIONS | (article_id, method) | `extraction.base.run_extractor` | event type, direction, surprises computed in code |
| SIGNALS | event_id | `pipeline.run_extraction` (stage 4b) | novelty, prior-news counts, cross-model disagreement |
| PANEL | (event_id, delay_min, horizon) | `features.panel.build_panel` | ids + labels + features; see below |
| PREDICTIONS | same as PANEL | `models.walkforward.predict_oos` | adds `pred`, `sigma_hat`, `gap`, `split` |
| TRADES | same as PANEL | `trading.backtest.backtest` | side, notional, gross / net return, P&L |

PANEL column groups (`schemas.py`): `PANEL_ID_COLS`, `PANEL_LABEL_COLS`,
`PANEL_META_COLS`; everything else is a feature (`feature_columns`).

## Timing rules (plan, "Point-in-time alignment rules")

```
created_at ──► tau = floor(created_at) + 1 min      (outside the session: next open)
p0 = tau − 1 min        last quote before the news: reaction baseline
entry = tau + d         d ∈ delays_min; features use quotes ≤ entry
exit  = close of t0 + n sessions (close_t0, t1, t3, t5)
target = log mid(exit)/mid(entry) − same for the hedge ETF
```

Daily statistics (volatility, ADV, momentum, spreads) are rolling windows
shifted by one session, so they only use completed prior sessions.

News retrieval (`signals/novelty.py`) is time-safe by the same rule: for an
article at time t it searches the same symbol's articles in [t − 90 days, t),
strictly before t. The RAG prompt only ever sees that set.
`tests/test_signals.py` adds a later article and checks no earlier feature or
context changes.

## Stage 4b: signals and the RAG loop

```
articles ──► novelty (embed, retrieve prior news) ──► novelty features ─┐
                    │                                                   ├─► SIGNALS ─► PANEL
                    └── RAG context ──► main extractor ──► EXTRACTIONS ─┤
                                       compare extractors ──────────────┘ (disagreement)
```

Every extraction method writes the same schema, so disagreement is just the
dispersion of `direction × confidence` across methods. The `probe` method
turns token probabilities into an inner confidence, which is more
informative than a confidence the model states in words.

## The reaction gap

`features/gap.py`: for each delay d, a ridge regression of the realised
reaction `ar_react` over [tau, tau+d] on text features only, fitted inside
each walk-forward fold on training rows. `gap = expected − realised`. The
realised reaction stays a feature, so the model can only use `gap` if it adds
information beyond short-term reversal (ablation A3).

## Trading

`edge = side × pred − cost_est`, trade if `edge > κ·σ̂`. Costs: entry
half-spread (observed), exit half-spread (20-day median), square-root impact,
borrow for shorts, ETF hedge round trip. Fills: long buys the ask and sells the
bid, short the reverse. Variants: long-short hedged, long-only hedged, long-only.

## Evaluation protocol

1. Walk-forward monthly; train rows must exit before the fold starts.
2. Every (delay, horizon) strategy at or above `min_tradable_delay_min` is
   scored on validation; the best is chosen; all count as trials.
3. The chosen strategy is scored once on test. Headline = net P&L, net Sharpe,
   deflated Sharpe, drawdown, net bps per trade vs. cost, for all variants.
4. Decay curve over every delay, capacity at 1×/10×/100× capital, and
   profit by event type.

## Extension points

| To add | Implement | Register in |
|---|---|---|
| a data vendor | `articles()`, `minute_bars()`, `quotes()` returning the contracts | `data/sources/__init__.py` |
| a text method | `extract_one(ArticleText) -> RawExtraction` (optionally `extract_batch`) with `name`, `version` | `extraction/__init__.py` |
| an embedder | `embed(texts) -> unit vectors` | `signals/embed.py:get_embedder` |
| a signal feature | a column keyed by `event_id` from stage 4b | `pipeline.run_extraction` |
| a return model | `fit(X, y)`, `predict(X)` | `models/base.py:make_model` |
| a feature | a column in `build_panel` computed from data ≤ entry | covered automatically by the leakage test |
| a cost component | a function in `trading/costs.py` | `total_cost` and `realise` |
