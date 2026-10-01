from __future__ import annotations

import numpy as np
import pandas as pd


def forward_max_return(
    mid: pd.Series,
    timestamps: pd.Series,
    horizon_seconds: float,
) -> pd.Series:
    """Maximum future mid-price return inside the horizon."""
    if horizon_seconds <= 0:
        raise ValueError("horizon_seconds must be positive")

    ts = pd.to_datetime(timestamps, utc=True)
    values = mid.to_numpy(dtype=float)
    t_ns = ts.astype("int64").to_numpy()
    horizon_ns = int(horizon_seconds * 1_000_000_000)

    result = np.full(len(values), np.nan, dtype=float)
    j = 1
    for i in range(len(values)):
        j = max(j, i + 1)
        while j < len(values) and t_ns[j] <= t_ns[i] + horizon_ns:
            j += 1
        if i + 1 < j:
            result[i] = np.max(values[i + 1:j]) / values[i] - 1.0

    return pd.Series(result, index=mid.index, name="future_max_return")


def make_up_label(
    mid: pd.Series,
    timestamps: pd.Series,
    horizon_seconds: float,
    target_return: float,
) -> pd.Series:
    """Label only observations with a complete future horizon.

    Rows without a complete horizon remain missing and must be excluded from
    training/evaluation rather than being treated as failures.
    """
    if target_return <= 0:
        raise ValueError("target_return must be positive")

    fwd = forward_max_return(mid, timestamps, horizon_seconds)
    y = pd.Series(pd.NA, index=mid.index, dtype="Int64", name="y_up")
    valid = fwd.notna()
    y.loc[valid] = (fwd.loc[valid] >= target_return).astype("int64")
    return y
