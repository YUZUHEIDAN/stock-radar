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
                df = raw[tk] if len(chunk) > 1 else raw
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


def main():
    uni = fetch_universe()
    meta = {r.tk: (r.name, r.sector if isinstance(r.sector, str) else "") for r in uni.itertuples()}
    data = download(["SPY"] + list(meta), period="2y")
    spy_df = data.pop("SPY")
    spy_c = spy_df["Close"].to_numpy()
    spy_ok = bool(spy_c[-1] > pd.Series(spy_c).rolling(200).mean().iloc[-1])
    last_date = spy_df.index[-1]
    as_of = last_date.date()
    print(f"universe {len(meta)}, data {len(data)}, as of {as_of}")

    trades = []
    for tk, df in data.items():
        df = df.reindex(spy_df.index).dropna()
        if len(df) < 260 or df.index[-1] != last_date:
            continue
        I = mr_indicators(analyze(df))
        n = len(I.c)
        last = n - 1
        busy_until = -1
        for t in range(n - WINDOW, n):
            tags = mr_tags(I, t)
            if not tags or t <= busy_until:
                continue
            warm = t < n - WINDOW + SAFE
            o = simulate(I, t, MR_SPEC)
            exit_ix = t + o["days"] if o["status"] == "closed" else None
            busy_until = exit_ix if exit_ix is not None else n
            if warm:
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

    # ---- history (forward test log): merge window results into persisted log
    DATA.mkdir(parents=True, exist_ok=True)
    hist_path = DATA / "history.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else {"started": str(as_of), "trades": {}}
    known_before = set(hist["trades"])
    for tr in trades:
        h = {k: tr[k] for k in tr if k != "chart"}
        hist["trades"][tr["key"]] = h
    closed = [t for t in hist["trades"].values() if t["status"] == "closed" and t["sigDate"] >= hist["started"]]
    rets = np.array([t["ret"] for t in closed]) if closed else np.array([])
    fwd = {"started": hist["started"], "n": len(closed)}
    if len(rets):
        w, l = rets[rets > 0], rets[rets <= 0]
        fwd.update(win=float((rets > 0).mean()), avg=float(rets.mean()), pf=float(w.sum() / -l.sum()) if len(l) else 99.0,
                   best=float(rets.max()), worst=float(rets.min()))

    today = [t for t in trades if t["status"] == "pending"]
    opened = [t for t in trades if t["status"] == "open"]
    out = {
        "mode": "pullback", "asOf": str(as_of), "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scanned": len(data), "spy": {"aboveSma200": spy_ok, "price": float(spy_c[-1])}, "bt": BT, "fwd": fwd,
        "today": sorted(today, key=lambda t: t["bbDepth"]), "open": sorted(opened, key=lambda t: t["sigDate"]),
        "recentClosed": sorted([t for t in closed], key=lambda t: t["exitDate"], reverse=True)[:40],
    }

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

    (DATA / "latest.json").write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")))
    hist_path.write_text(json.dumps(clean(hist), ensure_ascii=False, separators=(",", ":")))
    print(f"today {len(today)} | open {len(opened)} | forward closed {len(closed)}")

    # ---- alerts (ntfy): new signals + exit actions
    topic = os.environ.get("NTFY_TOPIC")
    new_sig = [t for t in today if t["key"] not in known_before]
    sells = [t for t in opened if t["action"] in ("sell_open", "sell_close")]
    if topic and (new_sig or sells):
        lines = [f"BUY {t['tk']} (ปิด ${t['sigClose']:.2f}, {'+'.join(t['tags'])})" for t in new_sig[:10]]
        lines += [f"SELL {t['tk']} {t['ret']*100:+.1f}% · {t['actionText']}" for t in sells]
        site = os.environ.get("SITE_URL", "")
        requests.post(f"https://ntfy.sh/{topic}", data="\n".join(lines).encode(), timeout=20,
                      headers={"Title": f"Stock Radar: {len(new_sig)} buy, {len(sells)} sell", **({"Click": site} if site else {})})
        print("alert sent")


if __name__ == "__main__":
    main()
