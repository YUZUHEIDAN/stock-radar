"""Build a factor dataset of every Daily-MACD bull cross (10y) for offline analysis.

Output: research/crosses.csv.gz — one row per cross with features at signal time and forward returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402
from engine import MonthlyMacd, analyze, fresh_cross, outcome, plan, score  # noqa: E402


class WeeklyMacd(MonthlyMacd):
    def __init__(self, dates, c):
        iso = dates.isocalendar()
        super().__init__(pd.DatetimeIndex(dates), c)
        keys = (iso.year.to_numpy() * 100 + iso.week.to_numpy())
        self.mi = np.zeros(len(c), dtype=int)
        closes, cur = [], None
        for i, k in enumerate(keys):
            if k != cur:
                cur = k; closes.append(c[i])
            else:
                closes[-1] = c[i]
            self.mi[i] = len(closes) - 1
        self.S = []
        e12 = e26 = closes[0]; sg = 0.0
        for j, x in enumerate(closes):
            if j > 0:
                e12 = x * self.k12 + e12 * (1 - self.k12); e26 = x * self.k26 + e26 * (1 - self.k26)
            m = e12 - e26
            sg = m if j == 0 else m * self.k9 + sg * (1 - self.k9)
            self.S.append((e12, e26, sg, m - sg))


H = (20, 60, 120, 250)


def main():
    uni = run.fetch_universe()
    data = run.download(["SPY"] + list(uni.tk), period="10y")
    spy_df = data.pop("SPY")
    spy = analyze(spy_df)
    rows = []
    for tk, df in data.items():
        df = df.reindex(spy_df.index).dropna()
        if len(df) < 700:
            continue
        I = analyze(df)
        wk = WeeklyMacd(df.index, I.c)
        six = spy_df.index.get_indexer(df.index)
        n = len(I.c)
        for t in range(600, n - 21):
            is_x = fresh_cross(I, t)
            if not np.isfinite(I.atr[t]) or not (is_x or t % 10 == 0):
                continue
            if not (I.c[t] > 5 and I.dv20[t] > 20e6):
                continue
            st = six[t]
            p = plan(I, t) if is_x else {"rr": np.nan, "noRes": None}
            mo, wo = I.mon.at(t), wk.at(t)
            hi252 = I.h[max(0, t - 252): t + 1].max()
            r = {
                "tk": tk, "date": df.index[t].date(), "is_cross": is_x, "score": score(I, t, spy, st)["total"] if is_x else np.nan,
                "mon_up": mo["up"] if mo else None, "mon_rising": mo["rising"] if mo else None,
                "mon_pos": (mo["m"] > 0) if mo else None,
                "wk_up": wo["up"] if wo else None, "wk_pos": (wo["m"] > 0) if wo else None,
                "px_e200": I.c[t] / I.e200[t] - 1, "e50_gt_e200": I.e50[t] > I.e200[t],
                "e200_slope": I.e200[t] / I.e200[t - 20] - 1,
                "adx": I.adx[t], "di_pos": I.pdi[t] > I.mdi[t],
                "vol_ratio": max(I.v[t], I.v[t - 1]) / I.v20[t],
                "rs63": (I.c[t] / I.c[t - 63]) / (spy.c[st] / spy.c[st - 63]) - 1,
                "rs126": (I.c[t] / I.c[t - 126]) / (spy.c[st] / spy.c[st - 126]) - 1,
                "rs252": (I.c[t] / I.c[t - 252]) / (spy.c[st] / spy.c[st - 252]) - 1,
                "rsi": I.rsi[t], "macd_below0": I.m[t] < 0, "macd_norm": I.m[t] / I.c[t],
                "dist_52wh": I.c[t] / hi252 - 1, "atr_pct": I.atr[t] / I.c[t],
                "rr": p["rr"], "no_res": p["noRes"], "spy_bull": spy.c[st] > spy.e200[st],
                "spy_e200": spy.c[st] / spy.e200[st] - 1, "ldv": np.log10(I.dv20[t]),
                "plan_ret": outcome(I, t, p, 20)["ret"] if is_x else np.nan, "macd_gt_sig": I.m[t] > I.s[t],
            }
            for h in H:
                if t + h < n:
                    r[f"r{h}"] = I.c[t + h] / I.c[t] - 1
                    r[f"x{h}"] = (1 + r[f"r{h}"]) / (spy.c[st + h] / spy.c[st]) - 1
                    r[f"mdd{h}"] = I.l[t + 1: t + h + 1].min() / I.c[t] - 1
            rows.append(r)
    out = Path(__file__).resolve().parent.parent / "research"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "crosses.csv.gz", index=False, float_format="%.5g")
    print("rows", len(rows))


if __name__ == "__main__":
    main()
