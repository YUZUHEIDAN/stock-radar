"""Stock Radar engine: indicators, gates, scoring, trade plan, backtest.

Mirrors the logic in the web prototype. All functions work on numpy arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------- config ----------------
CFG = {
    "min_price": 5.0,
    "min_dollar_vol": 20e6,
    "min_rr": 2.0,
    "earn_block_days": 5,
    "stop_atr": 1.5,
    "no_res_r": 2.5,
    "hold_days": 20,
    "alert_bull": 70,
    "alert_bear": 80,
    "bt_start_bar": 600,
}


# ---------------- indicators ----------------
def ema(a: np.ndarray, n: int) -> np.ndarray:
    k = 2 / (n + 1)
    out = np.empty_like(a, dtype=float)
    p = a[0]
    out[0] = p
    for i in range(1, len(a)):
        p = a[i] * k + p * (1 - k)
        out[i] = p
    return out


def sma(a: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(a).rolling(n).mean().to_numpy()


def macd(c):
    m = ema(c, 12) - ema(c, 26)
    s = ema(m, 9)
    return m, s


def rsi(c, n=14):
    out = np.full(len(c), 50.0)
    d = np.diff(c, prepend=c[0])
    g = np.clip(d, 0, None)[1 : n + 1].mean()
    l = np.clip(-d, 0, None)[1 : n + 1].mean()
    for i in range(n, len(c)):
        if i > n:
            g = (g * (n - 1) + max(d[i], 0)) / n
            l = (l * (n - 1) + max(-d[i], 0)) / n
        out[i] = 100.0 if l == 0 else 100 - 100 / (1 + g / l)
    return out


def atr_adx(h, l, c, n=14):
    N = len(c)
    atr = np.full(N, np.nan)
    pdi = np.zeros(N)
    mdi = np.zeros(N)
    adx = np.zeros(N)
    sT = sP = sM = a = ad = 0.0
    for i in range(1, N):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pd_ = up if (up > dn and up > 0) else 0.0
        md = dn if (dn > up and dn > 0) else 0.0
        if i <= n:
            sT += tr; sP += pd_; sM += md; a += tr
            if i == n:
                a /= n
        else:
            sT = sT - sT / n + tr
            sP = sP - sP / n + pd_
            sM = sM - sM / n + md
            a = (a * (n - 1) + tr) / n
        if i >= n and sT > 0:
            atr[i] = a
            p, m = 100 * sP / sT, 100 * sM / sT
            pdi[i], mdi[i] = p, m
            dx = 100 * abs(p - m) / (p + m) if p + m else 0.0
            ad = dx if i == n else (ad * (n - 1) + dx) / n
            adx[i] = ad
    return atr, pdi, mdi, adx


class MonthlyMacd:
    """Monthly MACD evaluated at any daily bar t; current month uses close[t] as a partial bar."""

    def __init__(self, dates: pd.DatetimeIndex, c: np.ndarray):
        keys = dates.year * 12 + dates.month
        self.mi = np.zeros(len(c), dtype=int)
        closes = []
        cur = None
        for i, k in enumerate(keys):
            if k != cur:
                cur = k
                closes.append(c[i])
            else:
                closes[-1] = c[i]
            self.mi[i] = len(closes) - 1
        self.k12, self.k26, self.k9 = 2 / 13, 2 / 27, 2 / 10
        self.S = []
        e12 = e26 = closes[0]
        sg = 0.0
        for j, x in enumerate(closes):
            if j > 0:
                e12 = x * self.k12 + e12 * (1 - self.k12)
                e26 = x * self.k26 + e26 * (1 - self.k26)
            m = e12 - e26
            sg = m if j == 0 else m * self.k9 + sg * (1 - self.k9)
            self.S.append((e12, e26, sg, m - sg))
        self.c = c

    def at(self, t: int):
        j = self.mi[t]
        if j < 27:
            return None
        e12p, e26p, sgp, hp = self.S[j - 1]
        x = self.c[t]
        a = x * self.k12 + e12p * (1 - self.k12)
        b = x * self.k26 + e26p * (1 - self.k26)
        m = a - b
        s = m * self.k9 + sgp * (1 - self.k9)
        h = m - s
        return {"up": bool(m > s), "rising": bool(h > hp), "m": float(m), "s": float(s)}


@dataclass
class Ind:
    dates: pd.DatetimeIndex
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    m: np.ndarray
    s: np.ndarray
    e20: np.ndarray
    e50: np.ndarray
    e200: np.ndarray
    rsi: np.ndarray
    atr: np.ndarray
    pdi: np.ndarray
    mdi: np.ndarray
    adx: np.ndarray
    v20: np.ndarray
    dv20: np.ndarray
    mon: MonthlyMacd


def analyze(df: pd.DataFrame) -> Ind:
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    m, s = macd(c)
    atr, pdi, mdi, adx = atr_adx(h, l, c)
    return Ind(df.index, o, h, l, c, v, m, s, ema(c, 20), ema(c, 50), ema(c, 200), rsi(c),
               atr, pdi, mdi, adx, sma(v, 20), sma(c * v, 20), MonthlyMacd(df.index, c))


def fresh_cross(I: Ind, t: int) -> bool:
    return I.m[t - 1] <= I.s[t - 1] and I.m[t] > I.s[t]


# ---------------- plan / score ----------------
def plan(I: Ind, t: int) -> dict:
    entry, a = I.c[t], I.atr[t]
    stop = entry - CFG["stop_atr"] * a
    risk = entry - stop
    res = math.inf
    for i in range(max(5, t - 120), t - 2):
        hi = I.h[i]
        if hi <= entry * 1.01 or hi >= res:
            continue
        lo_j, hi_j = i - 5, min(t, i + 5)
        if I.h[lo_j : hi_j + 1].max() <= hi:
            res = hi
    no_res = not math.isfinite(res)
    target = entry + CFG["no_res_r"] * risk if no_res else res
    return {"entry": entry, "zoneLo": max(entry - 0.5 * a, I.e20[t]), "stop": stop, "target": target,
            "risk": risk, "rr": (target - entry) / risk if risk > 0 else 0, "noRes": no_res, "atr": a}


def score(I: Ind, t: int, spy: Ind, spy_t: int) -> dict:
    parts = []

    def add(key, label, pts, mx, ok, detail):
        parts.append({"key": key, "label": label, "pts": pts, "max": mx, "ok": ok, "detail": detail})

    mo = I.mon.at(t)
    if mo and mo["up"]:
        add("mon", "Monthly MACD ขาขึ้น", 25, 25, True, "MACD > Signal")
    elif mo and mo["rising"]:
        add("mon", "Monthly MACD ขาขึ้น", 12, 25, "part", "Histogram เริ่มยก")
    else:
        add("mon", "Monthly MACD ขาขึ้น", 0, 25, False, "ยังเป็นขาลง" if mo else "ข้อมูลไม่พอ")

    a, b = I.c[t] > I.e200[t], I.e50[t] > I.e200[t]
    add("ema", "ราคา > EMA200 / EMA50 > EMA200", 10 * a + 10 * b, 20, True if a and b else ("part" if a or b else False),
        f"{'✓' if a else '✗'} ราคา/EMA200 · {'✓' if b else '✗'} EMA50/200")

    adx, di = I.adx[t], I.pdi[t] > I.mdi[t]
    ap = 15 if di and adx > 20 else 7 if di and adx > 15 else 0
    add("adx", "ADX > 20 และ +DI > −DI", ap, 15, True if ap == 15 else ("part" if ap else False),
        f"ADX {adx:.1f} · +DI {I.pdi[t]:.0f} / −DI {I.mdi[t]:.0f}")

    vr = max(I.v[t], I.v[t - 1]) / I.v20[t] if I.v20[t] else 0
    vp = 15 if vr >= 1.5 else 8 if vr >= 1.2 else 0
    add("vol", "Volume ≥ 1.5× เฉลี่ย 20 วัน", vp, 15, True if vp == 15 else ("part" if vp else False), f"{vr:.2f}× เฉลี่ย")

    rs = (I.c[t] / I.c[t - 63] - 1) - (spy.c[spy_t] / spy.c[spy_t - 63] - 1)
    rp = 10 if rs > 0 else 4 if rs > -0.03 else 0
    add("rs", "แข็งกว่า SPY (3 เดือน)", rp, 10, True if rp == 10 else ("part" if rp else False),
        f"{'+' if rs >= 0 else ''}{rs*100:.1f}% เทียบ SPY")

    r = I.rsi[t]
    rsp = 10 if 40 <= r <= 65 else 5 if 65 < r <= 70 else -10 if r > 70 else 3
    add("rsi", "RSI 40–65 (ยังไม่ไล่ราคา)", rsp, 10, True if rsp == 10 else ("part" if rsp > 0 else False),
        f"RSI {r:.0f}" + (" · overbought หัก 10" if r > 70 else ""))

    z = bool(I.m[t] < 0)
    add("zero", "ตัดขึ้นใต้เส้น 0 (ต้นรอบ)", 5 if z else 0, 5, z, "MACD < 0" if z else "MACD > 0")

    total = int(max(0, min(100, sum(p["pts"] for p in parts))))
    return {"total": total, "parts": parts}


def outcome(I: Ind, t: int, p: dict, hold: int):
    for k in range(1, hold + 1):
        i = t + k
        if I.l[i] <= p["stop"]:
            return {"ret": p["stop"] / p["entry"] - 1, "exit": "stop", "days": k}
        if I.h[i] >= p["target"]:
            return {"ret": p["target"] / p["entry"] - 1, "exit": "target", "days": k}
    return {"ret": I.c[t + hold] / p["entry"] - 1, "exit": "time", "days": hold}


def stats(trs: list[dict]) -> dict:
    n = len(trs)
    if not n:
        return {"n": 0, "win": 0, "avg": 0, "avgW": 0, "avgL": 0, "pf": 0, "days": 0}
    rets = np.array([x["ret"] for x in trs])
    w, lo = rets[rets > 0], rets[rets <= 0]
    gl = -lo.sum()
    return {"n": n, "win": float((rets > 0).mean()), "avg": float(rets.mean()),
            "avgW": float(w.mean()) if len(w) else 0.0, "avgL": float(lo.mean()) if len(lo) else 0.0,
            "pf": float(w.sum() / gl) if gl else 99.0, "days": float(np.mean([x["days"] for x in trs]))}
