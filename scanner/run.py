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
from engine import CFG, analyze, fresh_cross, outcome, plan, score, stats  # noqa: E402

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
def download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i : i + 100]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, period="6y", interval="1d", auto_adjust=True, group_by="ticker",
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


# ---------------- main ----------------
def main():
    uni = fetch_universe()
    meta = {r.tk: (r.name, r.sector if isinstance(r.sector, str) else "") for r in uni.itertuples()}
    tickers = list(meta)
    print(f"universe: {len(tickers)}")

    data = download(["SPY"] + tickers)
    spy_df = data.pop("SPY")
    spy = analyze(spy_df)
    T = len(spy.c) - 1
    as_of = spy_df.index[-1].date()
    bull = bool(spy.c[T] > spy.e200[T])
    thr = CFG["alert_bull"] if bull else CFG["alert_bear"]
    print(f"data: {len(data)} tickers, as of {as_of}, SPY bull={bull}")

    base, gated, filt, rows = [], [], [], []
    for tk, df in data.items():
        df = df.reindex(spy_df.index).dropna()
        if len(df) < CFG["bt_start_bar"] + 60:
            continue
        I = analyze(df)
        spy_ix = spy_df.index.get_indexer(df.index)
        n = len(I.c)

        # ---- backtest
        mine = []
        for t in range(CFG["bt_start_bar"], n - CFG["hold_days"] - 1):
            if not fresh_cross(I, t) or not np.isfinite(I.atr[t]):
                continue
            p = plan(I, t)
            o = outcome(I, t, p, CFG["hold_days"])
            base.append(o)
            liq = I.c[t] > CFG["min_price"] and I.dv20[t] > CFG["min_dollar_vol"]
            if liq and p["rr"] >= CFG["min_rr"]:
                st = spy_ix[t]
                sc = score(I, t, spy, st)["total"]
                rec = {**o, "score": sc}
                gated.append(rec)
                t_thr = CFG["alert_bull"] if spy.c[st] > spy.e200[st] else CFG["alert_bear"]
                if sc >= t_thr:
                    filt.append(rec)
                    mine.append(rec)

        # ---- today
        t = n - 1
        if df.index[-1] != spy_df.index[-1]:
            continue
        cross_ago = 0 if fresh_cross(I, t) else 1 if fresh_cross(I, t - 1) else None
        if cross_ago is None:
            continue
        p = plan(I, t)
        sc = score(I, t, spy, T)
        n0 = max(0, t - 130)
        name, sector = meta.get(tk, (tk, ""))
        rows.append({
            "tk": tk, "name": name, "sector": sector, "price": I.c[t], "chg": I.c[t] / I.c[t - 1] - 1,
            "crossAgo": cross_ago, "dv": I.dv20[t], "plan": p, "score": sc, "bt": stats(mine),
            "mon": I.mon.at(t),
            "gates": {"cross": True, "liq": bool(I.c[t] > CFG["min_price"] and I.dv20[t] > CFG["min_dollar_vol"]),
                      "rr": bool(p["rr"] >= CFG["min_rr"]), "earn": True},
            "chart": {"d": [d.strftime("%Y-%m-%d") for d in df.index[n0:]],
                      **{k: np.round(getattr(I, k)[n0:], 4).tolist() for k in ("o", "h", "l", "c", "e50", "e200", "m", "s")}},
        })

    # earnings check only for candidates (slow per-ticker call)
    for r in rows:
        if r["gates"]["liq"] and r["gates"]["rr"]:
            d = days_to_earnings(r["tk"], as_of)
            r["earnIn"] = d
            r["gates"]["earn"] = d is None or d > CFG["earn_block_days"]
        else:
            r["earnIn"] = None

    buckets = [{"a": a, "b": b, **stats([x for x in gated if a <= x["score"] <= b])}
               for a, b in [(0, 49), (50, 69), (70, 84), (85, 100)]]
    out = {
        "asOf": str(as_of), "generated": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scanned": len(data), "from": str(spy_df.index[CFG["bt_start_bar"]].date()),
        "spy": {"price": spy.c[T], "e200": spy.e200[T], "bull": bull}, "threshold": thr,
        "bt": {"base": stats(base), "gated": stats(gated), "filt": stats(filt), "buckets": buckets},
        "stocks": rows,
    }

    DATA.mkdir(parents=True, exist_ok=True)
    prev_path = DATA / "latest.json"
    prev_alerts = set()
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text())
            if prev.get("asOf") != out["asOf"]:
                prev_alerts = {s["tk"] for s in prev["stocks"] if all(s["gates"].values()) and s["score"]["total"] >= prev["threshold"]}
            else:
                prev_alerts = None  # same day rerun → don't re-alert
        except Exception:
            pass

    def clean(o):
        if isinstance(o, float) or isinstance(o, np.floating):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o

    prev_path.write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")))
    alerts = sorted([s for s in rows if all(s["gates"].values()) and s["score"]["total"] >= thr],
                    key=lambda s: -s["score"]["total"])
    print(f"crossed {len(rows)}, alerts {len(alerts)}: {[a['tk'] for a in alerts]}")

    topic = os.environ.get("NTFY_TOPIC")
    if topic and alerts and prev_alerts is not None:
        new = [a for a in alerts if a["tk"] not in prev_alerts]
        if new:
            body = "\n".join(f"{a['tk']} {a['score']['total']} · Buy ${a['plan']['entry']:.2f} SL ${a['plan']['stop']:.2f} TP ${a['plan']['target']:.2f}" for a in new[:8])
            site = os.environ.get("SITE_URL", "")
            requests.post(f"https://ntfy.sh/{topic}", data=body.encode(), timeout=20,
                          headers={"Title": f"Stock Radar: {len(new)} new signal(s)", "Tags": "chart_with_upwards_trend",
                                   **({"Click": site} if site else {})})
            print("alert sent:", [a["tk"] for a in new])


if __name__ == "__main__":
    main()
