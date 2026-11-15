# Stock Radar

Personal screener for S&P 500: daily pullback (mean-reversion) signals with exit tracking and a forward-test log.

- **Web app:** GitHub Pages from `/docs` (add to iPhone Home Screen)
- **Scan:** GitHub Actions every weekday after US close (05:00 Bangkok) → `docs/data/latest.json`
- **Alerts (optional):** set repo secret `NTFY_TOPIC` and subscribe to the same topic in the ntfy app

## Strategy (pullback / mean-reversion)
- Universe: S&P 500, price > $5, 20-day $ volume > $20M, close > SMA200
- Signal: close < Bollinger(20,2) lower band **or** RSI(2) < 5
- Entry: next session open · Exit: next open after a close > SMA5, or close of day 10 · no stop
- Backtest 2016–2026 (Bollinger): 19,853 trades, win 64%, +0.40%/trade after 0.1% costs vs +0.04% random entry
- Every signal is logged in `docs/data/history.json` as a forward (paper) test

## Research
`scanner/research*.py` + `research/` hold the studies: Daily/Weekly/Monthly MACD, breakouts and 12-1 momentum showed **no edge** vs random / SPY once tested point-in-time.

> Data: Yahoo Finance via `yfinance` (unofficial, EOD). Backtests use current S&P members (survivorship bias). Not investment advice.
