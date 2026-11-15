"""Momentum robustness: point-in-time S&P membership (by 'Date added'), buffer rule, costs, market filter.
Also: pullback (mean-reversion) exits with/without ATR stop.
Outputs research/momentum2.csv, research/mr_stops.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402
from engine import analyze, sma  # noqa: E402
from exits import simulate  # noqa: E402
from research import WeeklyMacd  # noqa: E402
from research_signals import signals  # noqa: E402

SRC = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
COST = 0.001  # per side


def main():
    sp = pd.read_csv(SRC, storage_options=run.UA)
    sp["tk"] = sp["Symbol"].str.replace(".", "-", regex=False)
    added = pd.to_datetime(sp["Date added"].astype(str).str[:10], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    added = dict(zip(sp.tk, added))
    data = run.download(["SPY"] + list(sp.tk), period="max")
    spy_df = data.pop("SPY")
    px = pd.DataFrame({k: v["Close"] for k, v in data.items()})
    px = px[px.index >= "2004-01-01"]
    mpx = px.resample("ME").last()
    spy_m = spy_df["Close"].resample("ME").last()
    mom = mpx.shift(1) / mpx.shift(12) - 1
    fwd = mpx.shift(-1) / mpx - 1
    spy_f = spy_m.shift(-1) / spy_m - 1
    spy_trend = spy_m > spy_m.rolling(10).mean()

    held = {"buf_pit": set(), "buf_all": set()}
    prev = {}
    recs = []
    for d in mpx.index[13:-1]:
        m_all = mom.loc[d].dropna()
        pit = [t for t in m_all.index if added.get(t, pd.Timestamp("1900-01-01")) <= d]
        r = {"date": d.date(), "spy": spy_f.loc[d], "spy_trend": bool(spy_trend.loc[d]), "n_all": len(m_all), "n_pit": len(pit)}
        f = fwd.loc[d]
        for lab, m in (("all", m_all), ("pit", m_all[pit])):
            if len(m) < 50:
                continue
            q = m.rank(pct=True)
            r[f"ew_{lab}"] = f[m.index].mean()
            for name, sel in ((f"dec_{lab}", q[q > 0.9].index), (f"top20_{lab}", m.nlargest(20).index),
                              (f"top50_{lab}", m.nlargest(50).index)):
                w = set(sel)
                to = len(w ^ prev.get(name, set())) / max(len(w), 1)
                prev[name] = w
                r[name] = f[list(w)].mean() - to * COST
                r[name + "_to"] = to
            # buffer: hold 25; keep while in top 20%; refill from top
            key = f"buf_{lab}"
            h = {t for t in held[key] if t in q.index and q[t] >= 0.8}
            for t in m.sort_values(ascending=False).index:
                if len(h) >= 25:
                    break
                h.add(t)
            to = len(h ^ held[key]) / 25
            held[key] = h
            r[key] = f[list(h)].mean() - to * COST
            r[key + "_to"] = to
            r[key + "_filt"] = r[key] if spy_trend.loc[d] else 0.0
        recs.append(r)
    out = Path(__file__).resolve().parent.parent / "research"
    pd.DataFrame(recs).to_csv(out / "momentum2.csv", index=False, float_format="%.5g")
    print("months", len(recs))

    # ---- MR exits with stops (10y window)
    specs = {"mr_nostop": dict(stop_atr=None, sma5_exit=True, max_hold=10),
             "mr_stop3": dict(stop_atr=3.0, sma5_exit=True, max_hold=10),
             "mr_stop2": dict(stop_atr=2.0, sma5_exit=True, max_hold=10)}
    rows = []
    cut = spy_df.index[-2520] if len(spy_df) > 2520 else spy_df.index[0]
    for tk, df in data.items():
        df = df[df.index >= cut].reindex(spy_df.index[spy_df.index >= cut]).dropna()
        if len(df) < 700:
            continue
        I = analyze(df)
        I.sma5 = sma(I.c, 5)
        S, up = signals(I, WeeklyMacd(df.index, I.c))
        n = len(I.c)
        for t in range(300, n - 12):
            if not (I.c[t] > 5 and I.dv20[t] > 20e6 and np.isfinite(I.atr[t])):
                continue
            for sname in ("MR_bb_lower_up", "MR_rsi2<5_up"):
                if S[sname][t]:
                    rr = {"sig": sname, "tk": tk, "date": df.index[t].date()}
                    for k, sp_ in specs.items():
                        o = simulate(I, t, sp_)
                        rr[k] = o.get("ret"); rr[k + "_w"] = o.get("why")
                    rows.append(rr)
    pd.DataFrame(rows).to_csv(out / "mr_stops.csv", index=False, float_format="%.5g")
    print("mr rows", len(rows))


if __name__ == "__main__":
    main()
