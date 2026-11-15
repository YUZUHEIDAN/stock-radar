# Stock Radar

Personal screener for S&P 500 + Nasdaq-100: finds stocks whose **Daily MACD just crossed up**, filters out risky setups, scores the rest 0–100, and attaches a trade plan + backtest stats.

- **Web app:** GitHub Pages from `/docs` (add to iPhone Home Screen)
- **Scan:** GitHub Actions every weekday after US close (05:00 Bangkok) → `docs/data/latest.json`
- **Alerts (optional):** set repo secret `NTFY_TOPIC` and subscribe to the same topic in the ntfy app

## Logic
**Gates (must pass):** Daily MACD crossed signal ≤2 days · price > $5 & 20-day $ volume > $20M · R:R ≥ 2 (stop 1.5×ATR, target = next swing-high resistance or 2.5R) · no earnings within 5 days · alert threshold 70 (80 when SPY < EMA200)

**Score (100):** Monthly MACD up 25 · price>EMA200 & EMA50>EMA200 20 · ADX>20 & +DI>−DI 15 · volume ≥1.5× 15 · 3-mo relative strength vs SPY 10 · RSI 40–65 10 (>70 −10) · cross below zero line 5

Tune in `scanner/engine.py` (`CFG` + `score()`).

> Data: Yahoo Finance via `yfinance` (unofficial, EOD). Backtest has survivorship bias and excludes costs. Not investment advice.
