"""Daily scan: S&P 500 + Nasdaq-100 → docs/data/latest.json (+ optional push alert).

Run:  python scanner/run.py
Env:  NTFY_TOPIC (optional) — ntfy.sh topic to receive alerts on your phone.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from engine import CFG, analyze  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
UNIVERSE_CACHE = ROOT / "scanner" / "universe.csv"
UA = {"User-Agent": "Mozilla/5.0 (stock-radar personal screener)"}


# ---------------- universe ----------------
def fetch_universe() -> pd.DataFrame:
    """S&P 500 (GitHub datasets CSV) + Nasdaq-100 (Wikipedia). Falls back to scanner/universe.csv."""
    parts = []
    try:
        sp = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
                         storage_options=UA)
        parts.append(sp.rename(columns={"Symbol": "tk", "Security": "name", "GICS Sector": "sector"})[["tk", "name", "sector"]])
        print("S&P 500:", len(parts[-1]))
    except Exception as e:
        print("S&P 500 fetch failed:", repr(e))
    try:
        html = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=UA, timeout=30).text
        for t in pd.read_html(StringIO(html)):
            cols = {str(c).strip(): c for c in t.columns}
            tcol = next((cols[c] for c in ("Ticker", "Symbol") if c in cols), None)
            if tcol is None or not (90 <= len(t) <= 110):
                continue
            ncol = next((cols[c] for c in ("Company", "Security", "Name") if c in cols), tcol)
            scol = next((v for k, v in cols.items() if "Sector" in k), None)
            nd = pd.DataFrame({"tk": t[tcol], "name": t[ncol], "sector": t[scol] if scol is not None else ""})
            parts.append(nd)
            print("Nasdaq-100:", len(nd))
            break
        else:
            print("Nasdaq-100 table not found on Wikipedia")
    except Exception as e:
        print("Nasdaq-100 fetch failed:", repr(e))

    cache = pd.read_csv(UNIVERSE_CACHE) if UNIVERSE_CACHE.exists() else pd.DataFrame(columns=["tk", "name", "sector"])
    if not parts:
        print("using cached universe")
        return cache
    u = pd.concat(parts + [cache]).astype({"tk": str})
    u["tk"] = u["tk"].str.strip().str.replace(".", "-", regex=False)
    u = u.drop_duplicates("tk").reset_index(drop=True)
    if len(u) >= 450:
        u.to_csv(UNIVERSE_CACHE, index=False)
    return u


# ---------------- data ----------------
def download(tickers: list[str], period: str = "6y") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i : i + 100]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True, group_by="ticker",
                                  threads=True, progress=False)
                break
            except Exception as e:
                print("download retry", attempt, e)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        for tk in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    lv0 = raw.columns.get_level_values(0)
                    df = raw[tk] if tk in lv0 else raw.xs(tk, axis=1, level=1)
                else:
                    df = raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) > 300:
                    out[tk] = df
            except KeyError:
                pass
    return out


def days_to_earnings(tk: str, today: dt.date) -> int | None:
    try:
        import yfinance as yf
        cal = yf.Ticker(tk).calendar
        ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
        ds = [d for d in (ds or []) if d >= today]
        return (min(ds) - today).days if ds else None
    except Exception:
        return None


# ---------------- pullback (mean-reversion) mode ----------------
from engine import rsi, sma  # noqa: E402
from exits import simulate  # noqa: E402

MR_SPEC = dict(stop_atr=None, sma5_exit=True, max_hold=10)
WINDOW = 30          # bars re-simulated each run
SAFE = 12            # first SAFE bars of the window only warm up the 'already in a trade' state
# Backtest reference (research/mr_stops.csv, S&P 500 2016–2026, next-open entry, 0.1% cost)
BT = {"bb": {"n": 19853, "win": 0.642, "avg": 0.00396, "pf": 1.36, "p1": -0.125, "worst": -0.551},
      "rsi2": {"n": 27860, "win": 0.637, "avg": 0.00281, "pf": 1.25, "p1": -0.127, "worst": -0.527},
      "random_avg": 0.0004, "days": 4.9, "from": "2016", "to": "2026"}


def mr_indicators(I):
    c = I.c
    I.sma5 = sma(c, 5)
    I.sma200 = sma(c, 200)
    I.rsi2 = rsi(c, 2)
    mid = sma(c, 20)
    sd = pd.Series(c).rolling(20).std().to_numpy()
    I.bb_mid, I.bb_lo, I.bb_hi = mid, mid - 2 * sd, mid + 2 * sd
    return I


def mr_tags(I, t):
    if not (I.c[t] > CFG["min_price"] and I.dv20[t] > CFG["min_dollar_vol"] and I.c[t] > I.sma200[t]):
        return []
    tags = []
    if I.c[t] < I.bb_lo[t]:
        tags.append("bb")
    if I.rsi2[t] < 5:
        tags.append("rsi2")
    return tags


def action_for(o, I, last):
    if o["status"] == "pending":
        return "buy_open", "ซื้อที่ราคาเปิดรอบถัดไป"
    if o["status"] == "open":
        if o.get("exit_next_open"):
            return "sell_open", "ขายที่ราคาเปิดรอบถัดไป (ปิดเหนือ SMA5 แล้ว)"
        if o["day"] >= MR_SPEC["max_hold"] - 1:
            return "sell_close", "ขายตอนปิดตลาดรอบถัดไป (ครบ 10 วัน)"
        return "hold", f"ถือต่อ · ขายเมื่อราคาปิด > SMA5 (~${I.sma5[last]:.2f})"
    return "closed", ""


def chart(I, df, n0):
    sl = slice(n0, None)
    return {"d": [d.strftime("%Y-%m-%d") for d in df.index[sl]],
            **{k: np.round(getattr(I, k)[sl], 3).tolist() for k in ("o", "h", "l", "c", "sma5", "sma200", "bb_lo", "bb_hi", "bb_mid")}}


ETF_NAMES = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000", "DIA": "Dow Jones 30", "MDY": "S&P MidCap 400",
             "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care", "XLY": "Consumer Discretionary",
             "XLP": "Consumer Staples", "XLI": "Industrials", "XLU": "Utilities", "XLB": "Materials", "TLT": "US Treasury 20Y+",
             "GLD": "Gold", "EFA": "Developed ex-US", "EEM": "Emerging Markets"}
ETF_CFG = {"vix_min": 15, "lim_atr": 0.5, "thru": 0.002, "rsi_exit": 70, "max_hold": 10}
# Backtest reference: research/etf_trades.csv.gz rule A with VIX>15 (1996–2026, strict limit fills, 0.1% cost)
ETF_BT = {"n": 747, "win": 0.764, "avg": 0.01115, "pf": 2.91, "worst": -0.1835, "p5": -0.0393, "days": 5.3,
          "from": "1996", "to": "2026", "perYr": 24, "fillRate": 0.34, "random_win": 0.63, "random_avg": 0.001}


def etf_sim(I, t, last):
    """Signal at close t → buy limit for session t+1 → exit next open after RSI(2) close > 70, or close of day 10."""
    c, o, h, l = I.c, I.o, I.h, I.l
    L = c[t] - ETF_CFG["lim_atr"] * I.atr[t]
    base = {"limit": L}
    if t + 1 > last:
        return {**base, "status": "pending"}
    s = t + 1
    if o[s] <= L:
        e = o[s]
    elif l[s] <= L * (1 - ETF_CFG["thru"]):
        e = L
    else:
        return {**base, "status": "nofill", "days": 1}
    for k in range(ETF_CFG["max_hold"]):
        i = s + k
        if i > last:
            return {**base, "status": "open", "entry": e, "day": k, "ret": c[last] / e - 1}
        if k > 0 and I.rsi2[i - 1] > ETF_CFG["rsi_exit"]:
            return {**base, "status": "closed", "entry": e, "exit": o[i], "ret": o[i] / e - 1 - 0.001, "days": k, "why": "rsi2_70", "exit_ix": i}
        if k == ETF_CFG["max_hold"] - 1:
            return {**base, "status": "closed", "entry": e, "exit": c[i], "ret": c[i] / e - 1 - 0.001, "days": k + 1, "why": "time", "exit_ix": i}
    return {**base, "status": "error"}


def etf_action(o, I, last):
    st = o["status"]
    if st == "pending":
        return "buy_limit", f"ตั้ง Buy limit ${o['limit']:.2f} สำหรับรอบตลาดถัดไปเท่านั้น (ไม่ได้ของ = ยกเลิก)"
    if st == "open":
        if I.rsi2[last] > ETF_CFG["rsi_exit"]:
            return "sell_open", f"ขายที่ราคาเปิดรอบถัดไป (RSI(2) ปิด {I.rsi2[last]:.0f} > 70)"
        if o["day"] >= ETF_CFG["max_hold"] - 1:
            return "sell_close", "ขายตอนปิดตลาดรอบถัดไป (ครบ 10 วัน)"
        return "hold", f"ถือต่อ · ขายเมื่อ RSI(2) ปิด > 70 (ตอนนี้ {I.rsi2[last]:.0f})"
    return st, ""


def clean(o):
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def fwd_stats(book):
    closed = [t for t in book["trades"].values() if t["status"] == "closed" and t["sigDate"] >= book["started"]]
    rets = np.array([t["ret"] for t in closed]) if closed else np.array([])
    f = {"started": book["started"], "n": len(closed),
         "nofill": sum(1 for t in book["trades"].values() if t["status"] == "nofill" and t["sigDate"] >= book["started"])}
    if len(rets):
        w, l = rets[rets > 0], rets[rets <= 0]
        f.update(win=float((rets > 0).mean()), avg=float(rets.mean()), pf=float(w.sum() / -l.sum()) if len(l) else 99.0,
                 best=float(rets.max()), worst=float(rets.min()))
    return f, sorted(closed, key=lambda t: t["exitDate"], reverse=True)[:40]


def merge(book, trades):
    known = set(book["trades"])
    for tr in trades:
        book["trades"][tr["key"]] = {k: v for k, v in tr.items() if k != "chart"}
    return known


def run_etf(as_of, last_date, book):
    data = download(list(ETF_NAMES) + ["^VIX"], period="2y")
    vix = data.pop("^VIX", None)
    vix_now = float(vix["Close"].iloc[-1]) if vix is not None else None
    trades = []
    for tk, df in data.items():
        if df.index[-1] != last_date:
            continue
        I = mr_indicators(analyze(df))
        vx = vix["Close"].reindex(df.index).ffill().to_numpy() if vix is not None else np.full(len(df), 99.0)
        n, last = len(I.c), len(I.c) - 1
        busy = -1
        for t in range(n - WINDOW, n):
            sig = I.c[t] > I.sma200[t] and I.c[t] < I.bb_lo[t] and vx[t] > ETF_CFG["vix_min"]
            if not sig or t <= busy:
                continue
            o = etf_sim(I, t, last)
            busy = o.get("exit_ix", last if o["status"] in ("open", "pending") else t + 1)
            if t < n - WINDOW + SAFE:
                continue
            act, txt = etf_action(o, I, last)
            tr = {"key": f"{tk}|{df.index[t].date()}", "tk": tk, "name": ETF_NAMES[tk], "sector": "ETF",
                  "sigDate": str(df.index[t].date()), "tags": ["bb", "vix"], "status": o["status"], "action": act, "actionText": txt,
                  "sigClose": I.c[t], "limit": o["limit"], "rsi2": I.rsi2[last], "bbDepth": I.c[t] / I.bb_lo[t] - 1,
                  "pctSma200": I.c[t] / I.sma200[t] - 1, "vix": vx[t], "price": I.c[last], "chg": I.c[last] / I.c[last - 1] - 1,
                  "atr": I.atr[t]}
            if o["status"] in ("open", "closed"):
                tr.update(entry=o["entry"], entryDate=str(df.index[t + 1].date()))
            if o["status"] == "open":
                tr.update(day=o["day"], ret=o["ret"])
            if o["status"] == "closed":
                tr.update(exit=o["exit"], exitDate=str(df.index[o["exit_ix"]].date()), ret=o["ret"], days=o["days"], why=o["why"])
            if o["status"] in ("pending", "open"):
                tr["chart"] = chart(I, df, max(0, n - 90))
            trades.append(tr)
    known = merge(book, trades)
    fwd, closed = fwd_stats(book)
    today = [t for t in trades if t["status"] == "pending"]
    opened = [t for t in trades if t["status"] == "open"]
    watch = []
    for tk, df in data.items():  # ETFs closest to a signal, for context
        I = mr_indicators(analyze(df))
        watch.append({"tk": tk, "name": ETF_NAMES[tk], "dist": I.c[-1] / I.bb_lo[-1] - 1, "up": bool(I.c[-1] > I.sma200[-1]),
                      "price": I.c[-1], "chg": I.c[-1] / I.c[-2] - 1})
    watch.sort(key=lambda w: (not w["up"], w["dist"]))
    sec = {"bt": ETF_BT, "cfg": ETF_CFG, "fwd": fwd, "today": today, "open": opened, "recentClosed": closed,
           "vix": vix_now, "watch": watch, "scanned": len(data)}
    new = [t for t in today if t["key"] not in known]
    return sec, new, [t for t in opened if t["action"].startswith("sell")]


def run_stocks(as_of, last_date, book, meta, data):
    trades = []
    for tk, df in data.items():
        df = df.reindex(data_index[0]).dropna() if data_index else df
        if len(df) < 260 or df.index[-1] != last_date:
            continue
        I = mr_indicators(analyze(df))
        n, last = len(I.c), len(I.c) - 1
        busy_until = -1
        for t in range(n - WINDOW, n):
            tags = mr_tags(I, t)
            if not tags or t <= busy_until:
                continue
            o = simulate(I, t, MR_SPEC)
            exit_ix = t + o["days"] if o["status"] == "closed" else None
            busy_until = exit_ix if exit_ix is not None else n
            if t < n - WINDOW + SAFE:
                continue
            act, txt = action_for(o, I, last)
            name, sector = meta.get(tk, (tk, ""))
            tr = {"key": f"{tk}|{df.index[t].date()}", "tk": tk, "name": name, "sector": sector,
                  "sigDate": str(df.index[t].date()), "tags": tags, "status": o["status"], "action": act, "actionText": txt,
                  "sigClose": I.c[t], "rsi2": I.rsi2[t], "bbDepth": I.c[t] / I.bb_lo[t] - 1,
                  "pctSma200": I.c[t] / I.sma200[t] - 1, "price": I.c[last], "chg": I.c[last] / I.c[last - 1] - 1,
                  "sma5": I.sma5[last], "dv": I.dv20[t]}
            if o["status"] in ("open", "closed"):
                tr.update(entry=o["entry"], entryDate=str(df.index[t + 1].date()))
            if o["status"] == "open":
                tr.update(day=o["day"], ret=o["ret"])
            if o["status"] == "closed":
                tr.update(exit=o["exit"], exitDate=str(df.index[exit_ix].date()), ret=o["ret"], days=o["days"], why=o["why"])
            if o["status"] != "closed":
                tr["chart"] = chart(I, df, max(0, n - 90))
            trades.append(tr)
    merge(book, trades)
    fwd, closed = fwd_stats(book)
    return {"bt": BT, "fwd": fwd, "today": sorted([t for t in trades if t["status"] == "pending"], key=lambda t: t["bbDepth"]),
            "open": sorted([t for t in trades if t["status"] == "open"], key=lambda t: t["sigDate"]),
            "recentClosed": closed, "scanned": len(data)}


data_index = []


def main():
    uni = fetch_universe()
    meta = {r.tk: (r.name, r.sector if isinstance(r.sector, str) else "") for r in uni.itertuples()}
    data = download(["SPY"] + list(meta), period="2y")
    spy_df = data.pop("SPY")
    data_index.append(spy_df.index)
    spy_c = spy_df["Close"].to_numpy()
    spy_ok = bool(spy_c[-1] > pd.Series(spy_c).rolling(200).mean().iloc[-1])
    last_date = spy_df.index[-1]
    as_of = last_date.date()

    DATA.mkdir(parents=True, exist_ok=True)
    hist_path = DATA / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    if "trades" in hist:  # migrate old single-book format
        hist = {"stk": hist}
    for k in ("etf", "stk"):
        hist.setdefault(k, {"started": str(as_of), "trades": {}})

    etf, new_buys, sells = run_etf(as_of, last_date, hist["etf"])
    stk = run_stocks(as_of, last_date, hist["stk"], meta, data)
    out = {"mode": "v3", "asOf": str(as_of), "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
           "spy": {"aboveSma200": spy_ok, "price": float(spy_c[-1])}, "etf": etf, "stk": stk}
    (DATA / "latest.json").write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")))
    hist_path.write_text(json.dumps(clean(hist), ensure_ascii=False, separators=(",", ":")))
    print(f"as of {as_of} | ETF today {len(etf['today'])} open {len(etf['open'])} vix {etf['vix']} | "
          f"stocks today {len(stk['today'])} open {len(stk['open'])}")

    topic = os.environ.get("NTFY_TOPIC")
    if topic and (new_buys or sells):
        lines = [f"BUY LIMIT {t['tk']} @ ${t['limit']:.2f} (รอบถัดไปเท่านั้น)" for t in new_buys]
        lines += [f"SELL {t['tk']} {t['ret']*100:+.1f}% · {t['actionText']}" for t in sells]
        site = os.environ.get("SITE_URL", "")
        requests.post(f"https://ntfy.sh/{topic}", data="\n".join(lines).encode(), timeout=20,
                      headers={"Title": f"ETF Radar: {len(new_buys)} buy, {len(sells)} sell", **({"Click": site} if site else {})})
        print("alert sent")


if __name__ == "__main__":
    main()
