"""Exit-strategy simulator shared by research (backtest) and the live tracker.

Execution model (deliberately conservative):
- Entry at the NEXT day's open after the signal close.
- Stop: if a day opens below the stop → exit at the open (gap); else if low touches it → exit at the stop.
- Target: same logic upward; if stop and target both touch on one day, the stop wins.
- Close-based exits (MACD cross-down, close < EMA20) are executed at the next open.
- Time exit at the close of the last holding day.
- Round-trip cost COST is deducted.
"""
from __future__ import annotations

import math

COST = 0.001

STRATS = {
    "S0_fixed_res": dict(stop_atr=1.5, target="res", max_hold=20),
    "S1_stop2_time20": dict(stop_atr=2.0, max_hold=20),
    "S2_trail3_40": dict(stop_atr=2.0, trail_atr=3.0, max_hold=40),
    "S3_trail2.5_20": dict(stop_atr=2.0, trail_atr=2.5, max_hold=20),
    "S4_macd_down_40": dict(stop_atr=2.0, macd_exit=True, max_hold=40),
    "S5_ema20_40": dict(stop_atr=2.0, ema20_exit=True, max_hold=40),
    "S6_fixed3R_20": dict(stop_atr=1.5, target=3.0, max_hold=20),
    "S7_be_trail_30": dict(stop_atr=1.5, be_at_r=1.0, trail_after_r=2.0, trail_atr=2.0, max_hold=30),
    "S8_trail3_60": dict(stop_atr=2.5, trail_atr=3.0, max_hold=60),
}


def _target(I, t, entry, risk, spec, res_fn):
    tg = spec.get("target")
    if tg is None:
        return None
    if tg == "res":
        r = res_fn(I, t)
        return r if r and r > entry else entry + 2.5 * risk
    return entry + tg * risk


def simulate(I, t, spec, res_fn=None, until=None):
    """Simulate a trade from signal bar t. Returns dict with status 'closed' or 'open' (if `until` reached first)."""
    n = len(I.c)
    last = n - 1 if until is None else min(until, n - 1)
    if t + 1 > last:
        return {"status": "pending"}
    entry = I.o[t + 1]
    a = I.atr[t]
    risk = spec["stop_atr"] * a
    stop = entry - risk
    target = _target(I, t, entry, risk, spec, res_fn)
    hc = entry
    pending_exit = None
    trail_on = "trail_after_r" not in spec and "trail_atr" in spec
    for k in range(1, spec["max_hold"] + 1):
        i = t + k
        if i > last:
            return {"status": "open", "entry": entry, "stop": stop, "target": target, "day": k - 1,
                    "max_hold": spec["max_hold"], "ret": I.c[last] / entry - 1, "exit_next_open": pending_exit}
        if pending_exit:
            return _close(I.o[i], entry, k, pending_exit)
        if I.o[i] <= stop:
            return _close(I.o[i], entry, k, "stop")
        if I.l[i] <= stop:
            return _close(stop, entry, k, "stop")
        if target is not None:
            if I.o[i] >= target:
                return _close(I.o[i], entry, k, "target")
            if I.h[i] >= target:
                return _close(target, entry, k, "target")
        # end-of-day updates
        hc = max(hc, I.c[i])
        r_mult = (I.c[i] - entry) / risk if risk > 0 else 0
        if spec.get("be_at_r") and r_mult >= spec["be_at_r"]:
            stop = max(stop, entry)
        if spec.get("trail_after_r") and r_mult >= spec["trail_after_r"]:
            trail_on = True
        if trail_on and math.isfinite(I.atr[i]):
            stop = max(stop, hc - spec["trail_atr"] * I.atr[i])
        if spec.get("macd_exit") and I.m[i] < I.s[i] and I.m[i - 1] >= I.s[i - 1]:
            pending_exit = "macd_down"
        if spec.get("ema20_exit") and I.c[i] < I.e20[i]:
            pending_exit = "below_ema20"
        if k == spec["max_hold"]:
            return _close(I.c[i], entry, k, "time")
    return {"status": "error"}


def _close(px, entry, days, why):
    return {"status": "closed", "exit": px, "ret": px / entry - 1 - COST, "days": days, "why": why}
