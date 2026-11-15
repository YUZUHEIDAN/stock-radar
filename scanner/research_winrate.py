"""Win-rate search across ETFs (survivorship-free) and S&P 500 stocks.

Grid: entry signal × entry type × exit rule × regime filter, aggregated per (universe, combo, year).
Execution: signals at close t; entry next session (market-on-open or a limit order valid for that day only);
close-based exits executed at the following open; time exit at the close of day `MAXH`. Cost 0.1% round trip.
Also: turn-of-month on index ETFs. Output: research/winrate.csv
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402
from engine import atr_adx, rsi, sma  # noqa: E402

COST = 0.001
MAXH = 10
ETFS = ["SPY", "QQQ", "IWM", "DIA", "MDY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB",
        "TLT", "GLD", "EFA", "EEM"]
EQUITY_ETFS = set(ETFS) - {"TLT", "GLD"}


def streak(c):
    s = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            s[i] = s[i - 1] + 1 if s[i - 1] > 0 else 1
        elif c[i] < c[i - 1]:
            s[i] = s[i - 1] - 1 if s[i - 1] < 0 else -1
    return s


def feats(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    F = {"o": o, "h": h, "l": l, "c": c}
    F["sma5"], F["sma200"] = sma(c, 5), sma(c, 200)
    F["rsi2"], F["rsi3"], F["rsi4"] = rsi(c, 2), rsi(c, 3), rsi(c, 4)
    F["atr"] = atr_adx(h, l, c)[0]
    st = streak(c)
    F["crsi"] = (F["rsi3"] + _rsi_series(st, 2) +
                 pd.Series(np.diff(c, prepend=c[0]) / np.r_[c[0], c[:-1]]).rolling(100).rank(pct=True).to_numpy() * 100) / 3
    mid = sma(c, 20)
    sd = pd.Series(c).rolling(20).std().to_numpy()
    F["bb_lo"] = mid - 2 * sd
    rng = np.where(h - l > 0, h - l, np.nan)
    F["ibs"] = (c - l) / rng
    F["low5"] = pd.Series(c).shift(1).rolling(5).min().to_numpy()
    d = np.diff(c, prepend=c[0]) < 0
    F["down3"] = d & np.roll(d, 1) & np.roll(d, 2)
    F["down4"] = F["down3"] & np.roll(d, 3)
    return F


def _rsi_series(x, n):
    """RSI computed on an arbitrary series (used for streak RSI in ConnorsRSI)."""
    out = np.full(len(x), 50.0)
    dx = np.diff(x, prepend=x[0])
    g = np.clip(dx, 0, None)[1:n + 1].mean()
    l_ = np.clip(-dx, 0, None)[1:n + 1].mean()
    for i in range(n, len(x)):
        if i > n:
            g = (g * (n - 1) + max(dx[i], 0)) / n
            l_ = (l_ * (n - 1) + max(-dx[i], 0)) / n
        out[i] = 100.0 if l_ == 0 else 100 - 100 / (1 + g / l_)
    return out


def signals(F):
    up = F["c"] > F["sma200"]
    r2 = F["rsi2"]
    cum2 = r2 + np.roll(r2, 1)
    return {
        "rsi2<5": up & (r2 < 5), "rsi2<10": up & (r2 < 10), "rsi2<2": up & (r2 < 2),
        "cumrsi2<10": up & (cum2 < 10), "cumrsi2<35": up & (cum2 < 35),
        "rsi4<25": up & (F["rsi4"] < 25), "rsi4<20": up & (F["rsi4"] < 20),
        "crsi<10": up & (F["crsi"] < 10), "crsi<15": up & (F["crsi"] < 15),
        "ibs<0.2": up & (F["ibs"] < 0.2), "ibs<0.1": up & (F["ibs"] < 0.1),
        "down3": up & F["down3"], "down4": up & F["down4"],
        "new5dlow": up & (F["c"] < F["low5"]), "bb_lower": up & (F["c"] < F["bb_lo"]),
        "rsi2<10&ibs<0.25": up & (r2 < 10) & (F["ibs"] < 0.25),
        "down3&ibs<0.25": up & F["down3"] & (F["ibs"] < 0.25),
        "rsi2<5_noTrend": r2 < 5,
    }


EXIT = {
    "sma5": lambda F, i: F["c"][i] > F["sma5"][i],
    "upclose": lambda F, i: F["c"][i] > F["c"][i - 1],
    "phigh": lambda F, i: F["c"][i] > F["h"][i - 1],
    "rsi2>70": lambda F, i: F["rsi2"][i] > 70,
    "rsi4>55": lambda F, i: F["rsi4"][i] > 55,
}
ENTRY = ("moo", "lim0.5atr")


def trade(F, t, entry, ex):
    o, l, c = F["o"], F["l"], F["c"]
    n = len(c)
    s = t + 1
    if s + MAXH >= n:
        return None
    if entry == "moo":
        e = o[s]
    else:
        L = c[t] - 0.5 * F["atr"][t]
        if o[s] <= L:
            e = o[s]
        elif l[s] <= L:
            e = L
        else:
            return None
    f = EXIT[ex]
    for k in range(MAXH):
        i = s + k
        if f(F, i):
            return o[i + 1] / e - 1 - COST, k + 1
        if k == MAXH - 1:
            return c[i] / e - 1 - COST, k + 1


def main():
    vix = run.download(["^VIX"], period="max").get("^VIX")
    vix_c = vix["Close"] if vix is not None else None
    agg = defaultdict(lambda: np.zeros(7))  # n, sum, wins, gp, gl, days, min

    def add(key, r, d):
        a = agg[key]
        if a[0] == 0:
            a[6] = r
        a[0] += 1; a[1] += r; a[2] += r > 0; a[3] += max(r, 0); a[4] += min(r, 0); a[5] += d; a[6] = min(a[6], r)

    def scan(univ, data, start):
        for tk, df in data.items():
            df = df[df.index >= start]
            if len(df) < 400:
                continue
            F = feats(df)
            S = signals(F)
            vx = vix_c.reindex(df.index).ffill().to_numpy() if vix_c is not None else np.full(len(df), np.nan)
            years = df.index.year.to_numpy()
            liq_ok = np.ones(len(df), bool) if univ == "etf" else ((F["c"] > 5) & (sma(F["c"] * df["Volume"].to_numpy(float), 20) > 20e6))
            for t in range(220, len(df) - MAXH - 2):
                if not liq_ok[t] or not np.isfinite(F["atr"][t]):
                    continue
                fired = [k for k, a in S.items() if a[t]]
                rnd = t % (3 if univ == "etf" else 7) == 0
                if not fired and not rnd:
                    continue
                up = F["c"][t] > F["sma200"][t]
                hv = vx[t] > 20
                for en in ENTRY:
                    for ex in EXIT:
                        res = trade(F, t, en, ex)
                        if res is None:
                            continue
                        r, d = res
                        for sg in fired:
                            add((univ, sg, en, ex, "all", years[t]), r, d)
                            add((univ, sg, en, ex, "vix>20" if hv else "vix<=20", years[t]), r, d)
                        if rnd and up and en == "moo":
                            add((univ, "RANDOM_up", en, ex, "all", years[t]), r, d)
            print(univ, tk, "done", flush=True) if univ == "etf" else None

    etf = run.download(ETFS, period="max")
    scan("etf", etf, "1993-01-01")
    # turn of month on equity ETFs: buy open of last trading day, sell open of 4th trading day of next month
    for tk in EQUITY_ETFS:
        df = etf.get(tk)
        if df is None:
            continue
        o = df["Open"].to_numpy(float)
        m = df.index.to_period("M")
        last_ix = np.where(m[1:] != m[:-1])[0]  # last trading day of each month
        for i in last_ix:
            j = i + 1 + 3
            if j < len(o):
                add(("etf", "TOM", "moo", "4th_day_open", "all", df.index[i].year), o[j] / o[i] - 1 - COST, 4)

    uni = run.fetch_universe()
    stocks = run.download(list(uni.tk), period="max")
    scan("stk", stocks, "2005-01-01")

    rows = [dict(zip(("univ", "sig", "entry", "exit", "regime", "year"), k), n=v[0], sum=v[1], wins=v[2], gp=v[3], gl=v[4],
                 days=v[5], worst=v[6]) for k, v in agg.items()]
    out = Path(__file__).resolve().parent.parent / "research"
    pd.DataFrame(rows).to_csv(out / "winrate.csv", index=False, float_format="%.6g")
    print("rows", len(rows))


if __name__ == "__main__":
    main()
