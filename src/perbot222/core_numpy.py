from __future__ import annotations

import numpy as np
from scipy.stats import linregress


def mid_prices(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    if bid.shape != ask.shape or bid.ndim != 1:
        raise ValueError("bid and ask must be 1-D arrays with the same shape")
    if np.any(ask < bid):
        raise ValueError("Found ask < bid.")
    return (bid + ask) / 2.0


def microstructure_snapshot(
    bid: np.ndarray,
    ask: np.ndarray,
    lookback: int = 100,
) -> dict[str, float]:
    """Causal snapshot using only the supplied historical tick window."""
    if lookback < 5:
        raise ValueError("lookback must be >= 5")
    mid = mid_prices(bid, ask)
    if len(mid) < lookback + 1:
        raise ValueError("not enough ticks for lookback")
    x = mid[-lookback:]
    d = np.diff(x)
    ret = x[-1] / x[0] - 1.0
    vol = float(np.std(np.diff(np.log(x)), ddof=1)) if len(x) > 2 else 0.0
    up = float(np.mean(d > 0))
    down = float(np.mean(d < 0))
    zero = float(np.mean(d == 0))
    pressure = up - down
    slope = float(linregress(np.arange(len(x), dtype=float), np.log(x)).slope)
    spread = np.asarray(ask, dtype=float)[-lookback:] - np.asarray(bid, dtype=float)[-lookback:]
    return {
        "mid": float(x[-1]),
        "spread": float(spread[-1]),
        "spread_mean": float(np.mean(spread)),
        "return": float(ret),
        "volatility": vol,
        "up_frac": up,
        "down_frac": down,
        "zero_frac": zero,
        "signed_pressure": float(pressure),
        "log_slope": slope,
        "tick_count": float(lookback),
    }


def forward_max_return(mid: np.ndarray, horizon_ticks: int) -> np.ndarray:
    """Leakage-safe forward label helper. Tail without full horizon is NaN."""
    values = np.asarray(mid, dtype=float)
    if values.ndim != 1 or horizon_ticks < 1:
        raise ValueError("invalid input")
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(len(values) - horizon_ticks):
        out[i] = np.max(values[i + 1:i + horizon_ticks + 1]) / values[i] - 1.0
    return out


def make_up_label(mid: np.ndarray, horizon_ticks: int, target_return: float) -> np.ndarray:
    if target_return <= 0:
        raise ValueError("target_return must be positive")
    fwd = forward_max_return(mid, horizon_ticks)
    y = np.full(fwd.shape, np.nan, dtype=float)
    valid = ~np.isnan(fwd)
    y[valid] = (fwd[valid] >= target_return).astype(float)
    return y
