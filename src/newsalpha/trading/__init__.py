"""Stage 6, trading: PREDICTIONS -> TRADES and a daily P&L series.

costs     ex-ante cost estimate per trade (spread, impact, borrow, hedge)
rules     net edge = side * forecast - cost; trade only if edge > kappa * sigma_hat
sizing    notional proportional to edge / sigma^2, capped by capital and liquidity
backtest  realises each trade at bid/ask, hedges with the sector ETF, books P&L on exit
"""
