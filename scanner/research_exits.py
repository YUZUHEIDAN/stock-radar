"""Simulate every exit strategy on every liquid Daily-MACD cross (10y). → research/exits.csv.gz"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402
from engine import analyze, fresh_cross, plan  # noqa: E402
from exits import STRATS, simulate  # noqa: E402


def res_fn(I, t):
    p = plan(I, t)
    return None if p["noRes"] else p["target"]


def main():
    uni = run.fetch_universe()
    data = run.download(["SPY"] + list(uni.tk), period="10y")
    spy_df = data.pop("SPY")
    rows = []
    for tk, df in data.items():
        df = df.reindex(spy_df.index).dropna()
        if len(df) < 700:
            continue
        I = analyze(df)
        n = len(I.c)
        for t in range(600, n - 63):
            is_x = fresh_cross(I, t)
            if not (is_x or t % 10 == 0) or not np.isfinite(I.atr[t]) or not (I.c[t] > 5 and I.dv20[t] > 20e6):
                continue
            r = {"tk": tk, "date": df.index[t].date(), "is_cross": is_x, "gap": I.o[t + 1] / I.c[t] - 1}
            for name, spec in STRATS.items():
                if not is_x and name not in ("S0_fixed_res", "S1_stop2_time20", "S8_trail3_60"):
                    continue
                o = simulate(I, t, spec, res_fn)
                r[name] = o.get("ret"); r[name + "_d"] = o.get("days"); r[name + "_w"] = o.get("why")
            rows.append(r)
    out = Path(__file__).resolve().parent.parent / "research"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "exits.csv.gz", index=False, float_format="%.5g")
    print("rows", len(rows))


if __name__ == "__main__":
    main()
