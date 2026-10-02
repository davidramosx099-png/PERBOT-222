import csv
import numpy as np
from pathlib import Path

LB = (20, 50, 100, 250, 500, 1000, 2000)
H = (10_000, 30_000, 60_000)
MIN_ROWS = 100_000
COOLDOWN = 60_000

def load(p):
    with Path(p).open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [(int(x["timestamp"]), float(x["bidPrice"]), float(x["askPrice"])) for x in r]
    if not rows:
        return None
    a = np.asarray(rows, float)
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2]

def features(ts, bid, ask, lb):
    mid = (bid + ask) / 2.0
    dt = np.diff(ts).astype(float)
    d = np.diff(mid)
    sec = np.maximum(np.convolve(dt, np.ones(lb), mode="valid"), 1.0)
    rate = 1000.0 * lb / sec

    move = np.convolve(d, np.ones(lb), mode="valid")
    abs_move = np.convolve(np.abs(d), np.ones(lb), mode="valid")
    pressure = move / np.maximum(abs_move, 1e-12)

    short = max(5, lb // 5)
    rs = 1000.0 * short / np.maximum(
        np.convolve(dt, np.ones(short), mode="valid"), 1.0
    )
    accel = rs[lb-short:] - rate
    return mid, rate, pressure, accel

def thresholds(fs, oi, lb):
    vals = []
    for item in fs[oi-3:oi]:
        ts, bid, ask = item[1:]
        _, rate, pressure, accel = features(ts, bid, ask, lb)
        vals.append((rate, pressure, accel))
    rate = np.concatenate([v[0] for v in vals])
    pressure = np.concatenate([v[1] for v in vals])
    accel = np.concatenate([v[2] for v in vals])
    return (
        float(np.quantile(rate, .20)), float(np.quantile(rate, .80)),
        float(np.quantile(pressure, .20)), float(np.quantile(pressure, .80)),
        float(np.quantile(accel, .20)), float(np.quantile(accel, .80)),
    )

def run(ts, bid, ask, mid, rate, pressure, accel, h, cut,
        rate_state, direction_state, direction_mode, rlo, rhi, plo, phi):
    gross = []
    quote = []
    last = None
    for k in range(len(rate)):
        i = k + cut
        if rate_state == "high" and rate[k] < rhi:
            continue
        if rate_state == "low" and rate[k] > rlo:
            continue

        if direction_mode == "momentum":
            if direction_state == "high" and pressure[k] < phi:
                continue
            if direction_state == "low" and pressure[k] > plo:
                continue
        else:
            if direction_state == "high" and pressure[k] > phi:
                continue
            if direction_state == "low" and pressure[k] < plo:
                continue

        cur = int(ts[i])
        if last is not None and cur - last < COOLDOWN:
            continue
        j = np.searchsorted(ts, cur + h)
        if j >= len(ts):
            continue

        gross.append((mid[j] / mid[i] - 1.0) * 1e4)
        quote.append((bid[j] - ask[i]) / ask[i] * 1e4)
        last = cur
    return np.asarray(gross), np.asarray(quote)

def main():
    fs = []
    for f in sorted(Path("data/multi").glob("*.csv")):
        x = load(f)
        if x is not None and len(x[0]) >= MIN_ROWS:
            fs.append((f, *x))
    print("VALID_DAYS", len(fs))
    print("PROTOCOL prior-3-days thresholds | quote-aware BUY | cooldown=60s")

    for lb in LB:
        for h in H:
            for rate_state in ("low", "high"):
                for direction_mode in ("contrarian", "momentum"):
                    for direction_state in ("low", "high"):
                        g_all, q_all = [], []
                        for oi in range(3, len(fs)):
                            rlo, rhi, plo, phi, _, _ = thresholds(fs, oi, lb)
                            ts, bid, ask = fs[oi][1:]
                            mid, rate, pressure, accel = features(ts, bid, ask, lb)
                            g, q = run(
                                ts, bid, ask, mid, rate, pressure, accel, h, lb,
                                rate_state, direction_state, direction_mode,
                                rlo, rhi, plo, phi
                            )
                            if len(g):
                                g_all.extend(g); q_all.extend(q)

                        if g_all:
                            g = np.asarray(g_all); q = np.asarray(q_all)
                            print(
                                "COMBO",
                                "lb", lb, "h", h // 1000,
                                "rate", rate_state,
                                "dir", direction_mode, direction_state,
                                "N", len(q),
                                "gross", round(float(g.mean()), 4),
                                "quote", round(float(q.mean()), 4),
                                "win", round(float(np.mean(q > 0)), 4),
                            )

if __name__ == "__main__":
    main()
