"""Indicator bake-off: many entry signals × exit styles vs matched random entries (10y, S&P 500).

Outputs research/signals.csv.gz (one row per entry) and research/momentum.csv (monthly portfolio returns).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402
from engine import analyze, ema, rsi, sma  # noqa: E402
from exits import simulate  # noqa: E402
from research import WeeklyMacd  # noqa: E402

EXITS = {
    "mr": dict(stop_atr=None, sma5_exit=True, max_hold=10),          # mean-reversion exit
    "t20": dict(stop_atr=2.0, max_hold=20),                          # fixed 20d with stop
    "tr": dict(stop_atr=2.5, trail_atr=3.0, max_hold=60),            # trend-following trail
}


def first_in(sig, gap=20):
    """Keep a True only if there was no True in the previous `gap` bars."""
    out = np.zeros_like(sig)
    last = -10**9
    for i, v in enumerate(sig):
        if v and i - last > gap:
            out[i] = True
        if v:
            last = i
    return out


def signals(I, wk):
    c, o, h, l, v = I.c, I.o, I.h, I.l, I.v
    n = len(c)
    s200 = sma(c, 200)
    up = c > s200
    r2 = rsi(c, 2)
    rng = np.where(h - l > 0, h - l, np.nan)
    ibs = (c - l) / rng
    sd20 = pd.Series(c).rolling(20).std().to_numpy()
    bb_lo = sma(c, 20) - 2 * sd20
    d1 = np.diff(c, prepend=c[0]) < 0
    down3 = d1 & np.roll(d1, 1) & np.roll(d1, 2)
    hi252 = pd.Series(c).shift(1).rolling(252).max().to_numpy()
    dh55 = pd.Series(h).shift(1).rolling(55).max().to_numpy()
    e50, e200 = I.e50, I.e200
    gc = (e50 > e200) & (np.roll(e50, 1) <= np.roll(e200, 1))
    mcross = (I.m > I.s) & (np.roll(I.m, 1) <= np.roll(I.s, 1))
    # weekly / monthly MACD fresh cross evaluated on the last bar of each period
    wk_x = np.zeros(n, bool); mo_x = np.zeros(n, bool)
    iso = I.dates.isocalendar()
    wkey = iso.year.to_numpy() * 100 + iso.week.to_numpy()
    mkey = I.dates.year * 12 + I.dates.month
    prev_w = prev_m = None
    for t in range(1, n):
        if t == n - 1 or wkey[t + 1] != wkey[t]:
            a = wk.at(t)
            if a is not None:
                if prev_w is not None and a["up"] and not prev_w:
                    wk_x[t] = True
                prev_w = a["up"]
        if t == n - 1 or mkey[t + 1] != mkey[t]:
            a = I.mon.at(t)
            if a is not None:
                if prev_m is not None and a["up"] and not prev_m:
                    mo_x[t] = True
                prev_m = a["up"]
    volx = (v > 3 * I.v20) & (c > o) & (ibs > 0.75) & up
    S = {
        "MR_rsi2<10_up": (r2 < 10) & up,
        "MR_rsi2<5_up": (r2 < 5) & up,
        "MR_down3_up": down3 & up,
        "MR_ibs<0.15_up": (ibs < 0.15) & up,
        "MR_bb_lower_up": (c < bb_lo) & up,
        "MR_rsi2<5_down": (r2 < 5) & ~up,
        "TR_52wh_new": first_in(c > hi252),
        "TR_donchian55": first_in(c > dh55),
        "TR_golden_cross": gc,
        "TR_weekly_macd": wk_x,
        "TR_monthly_macd": mo_x,
        "TR_vol_spike_up": first_in(volx, 10),
        "REF_daily_macd": mcross,
    }
    return S, up


def main():
    uni = run.fetch_universe()
    data = run.download(["SPY"] + list(uni.tk), period="10y")
    spy_df = data.pop("SPY")
    rows, closes = [], {}
    for tk, df in data.items():
        df = df.reindex(spy_df.index).dropna()
        if len(df) < 700:
            continue
        closes[tk] = df["Close"]
        I = analyze(df)
        I.sma5 = sma(I.c, 5)
        wk = WeeklyMacd(df.index, I.c)
        S, up = signals(I, wk)
        n = len(I.c)
        liq = (I.c > 5) & (I.dv20 > 20e6)
        for t in range(300, n - 62):
            if not liq[t] or not np.isfinite(I.atr[t]):
                continue
            fired = [k for k, a in S.items() if a[t]]
            rand = t % 10 == 0
            if not fired and not rand:
                continue
            res = {ex: simulate(I, t, spec) for ex, spec in EXITS.items()}
            base = {"tk": tk, "date": df.index[t].date(), "up": bool(up[t])}
            for ex, o in res.items():
                base[ex] = o.get("ret"); base[ex + "_d"] = o.get("days")
            for k in fired:
                rows.append({**base, "sig": k})
            if rand:
                rows.append({**base, "sig": "RANDOM"})
    out = Path(__file__).resolve().parent.parent / "research"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "signals.csv.gz", index=False, float_format="%.5g")
    print("signal rows", len(rows))

    # ---- cross-sectional momentum (monthly, 12-1)
    px = pd.DataFrame(closes)
    mpx = px.resample("ME").last()
    mom = mpx.shift(1) / mpx.shift(12) - 1
    fwd = mpx.shift(-1) / mpx - 1
    spy_m = spy_df["Close"].resample("ME").last()
    spy_f = spy_m.shift(-1) / spy_m - 1
    recs = []
    for d in mpx.index[13:-1]:
        m, f = mom.loc[d].dropna(), fwd.loc[d]
        if len(m) < 100:
            continue
        q = m.rank(pct=True)
        recs.append({"date": d.date(), "top20": f[m.nlargest(20).index].mean(), "top_decile": f[q > 0.9].mean(),
                     "bottom_decile": f[q <= 0.1].mean(), "ew_all": f[m.index].mean(), "spy": spy_f.loc[d]})
    pd.DataFrame(recs).to_csv(out / "momentum.csv", index=False, float_format="%.5g")
    print("momentum months", len(recs))


if __name__ == "__main__":
    main()
