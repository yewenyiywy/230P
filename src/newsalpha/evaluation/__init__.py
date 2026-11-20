"""Stage 7, evaluation: profit metrics first, diagnostics second.

splits   walk-forward folds with purging
metrics  Sharpe, drawdown, deflated Sharpe, per-trade stats, rank IC
report   chooses the strategy on validation, scores it once on test, writes report.json/.md
"""
