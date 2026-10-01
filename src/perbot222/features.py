from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "bid", "ask"}

def validate_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(ticks.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = ticks.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    for col in ("bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df["ask"] < df["bid"]).any():
        raise ValueError("Found ask < bid.")
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp", kind="stable")
    return df.reset_index(drop=True)

def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    idx = np.arange(window, dtype=float)
    idx -= idx.mean()
    denom = float(np.dot(idx, idx))
    def slope(values: np.ndarray) -> float:
        y = values - values.mean()
        return float(np.dot(idx, y) / denom) if denom else 0.0
    return x.rolling(window, min_periods=window).apply(slope, raw=True)

def build_features(ticks: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Build strictly backward-looking features; no future tick is referenced."""
    if lookback < 5:
        raise ValueError("lookback must be >= 5")
    df = validate_ticks(ticks)
    mid = (df["bid"] + df["ask"]) / 2.0
    spread = df["ask"] - df["bid"]
    log_mid = np.log(mid)
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]
    out["mid"] = mid
    out["spread"] = spread
    out["spread_bps"] = spread / mid * 10_000
    out["ret_1"] = mid.pct_change()
    out["ret_lb"] = mid.pct_change(lookback)
    out["vol_lb"] = out["ret_1"].rolling(lookback, min_periods=lookback).std()
    out["slope_lb"] = _rolling_slope(log_mid, lookback)
    diff = mid.diff()
    out["up_frac"] = diff.gt(0).rolling(lookback, min_periods=lookback).mean()
    out["down_frac"] = diff.lt(0).rolling(lookback, min_periods=lookback).mean()
    out["zero_frac"] = diff.eq(0).rolling(lookback, min_periods=lookback).mean()
    abs_diff = diff.abs()
    out["mean_abs_tick"] = abs_diff.rolling(lookback, min_periods=lookback).mean()
    out["tick_range"] = mid.rolling(lookback, min_periods=lookback).max() - mid.rolling(lookback, min_periods=lookback).min()
    out["position_in_range"] = ((mid - mid.rolling(lookback, min_periods=lookback).min()) / out["tick_range"].replace(0, np.nan))
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        out["volume"] = vol
        out["volume_ratio"] = vol / vol.rolling(lookback, min_periods=lookback).mean()
    out["signed_tick_pressure"] = np.sign(diff).rolling(lookback, min_periods=lookback).mean()
    return out
