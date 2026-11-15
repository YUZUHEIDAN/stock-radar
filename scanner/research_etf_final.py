"""Robustness for the best ETF pullback rules: strict limit fills, per-ETF/per-year, trade log for portfolio sim.
Output research/etf_trades.csv.gz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402
from research_winrate import COST, ETFS, EXIT, MAXH, feats, signals  # noqa: E402

THRU = 0.002  # price must trade 0.2% through the limit to count as filled

RULES = {
    "A_bb_lim0.5_rsi2x70": ("bb_lower", 0.5, "rsi2>70"),
    "A_bb_lim0.25_rsi2x70": ("bb_lower", 0.25, "rsi2>70"),
    "A_bb_lim1.0_rsi2x70": ("bb_lower", 1.0, "rsi2>70"),
    "A_bb_moo_rsi2x70": ("bb_lower", None, "rsi2>70"),
    "B_rsi2ibs_lim0.5_rsi2x70": ("rsi2<10&ibs<0.25", 0.5, "rsi2>70"),
    "B_rsi2ibs_moo_rsi2x70": ("rsi2<10&ibs<0.25", None, "rsi2>70"),
    "C_crsi15_lim0.5_rsi2x70": ("crsi<15", 0.5, "rsi2>70"),
    "D_connors_rsi4_moo_rsi4x55": ("rsi4<25", None, "rsi4>55"),
    "E_rsi2<5any_lim0.5_sma5": ("rsi2<5_noTrend", 0.5, "sma5"),
    "R_random_up_moo_rsi2x70": ("RANDOM", None, "rsi2>70"),
}


def trade(F, t, lim, ex):
    o, l, c = F["o"], F["l"], F["c"]
    s = t + 1
    if s + MAXH >= len(c):
        return None
    if lim is None:
        e = o[s]
    else:
        L = c[t] - lim * F["atr"][t]
        if o[s] <= L:
            e = o[s]
        elif l[s] <= L * (1 - THRU):
            e = L
        else:
            return None
    f = EXIT[ex]
    for k in range(MAXH):
        i = s + k
        if f(F, i):
            return o[i + 1] / e - 1 - COST, k + 1, i + 1
        if k == MAXH - 1:
            return c[i] / e - 1 - COST, k + 1, i


def main():
    data = run.download(ETFS + ["^VIX"], period="max")
    vix = data.pop("^VIX", None)
    print("vix rows", 0 if vix is None else len(vix))
    rows = []
    for tk, df in data.items():
        F = feats(df)
        S = signals(F)
        up = F["c"] > F["sma200"]
        vx = vix["Close"].reindex(df.index).ffill().to_numpy() if vix is not None else np.full(len(df), np.nan)
        for t in range(220, len(df) - MAXH - 2):
            if not np.isfinite(F["atr"][t]):
                continue
            for name, (sg, lim, ex) in RULES.items():
                fire = (up[t] and t % 3 == 0) if sg == "RANDOM" else S[sg][t]
                if not fire:
                    continue
                r = trade(F, t, lim, ex)
                if r:
                    rows.append({"rule": name, "tk": tk, "date": df.index[t].date(), "exit_date": df.index[r[2]].date(),
                                 "ret": r[0], "days": r[1], "vix": vx[t]})
    out = Path(__file__).resolve().parent.parent / "research"
    pd.DataFrame(rows).to_csv(out / "etf_trades.csv.gz", index=False, float_format="%.6g")
    spy = data["SPY"]["Close"]
    spy.to_frame("spy").to_csv(out / "spy_close.csv.gz", float_format="%.6g")
    print("rows", len(rows))


if __name__ == "__main__":
    main()
