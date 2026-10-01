import numpy as np
import pandas as pd

from perbot222.grid import run_grid


def test_grid_runs_on_synthetic_ticks():
    n = 1200
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    price = 2000 + np.linspace(0, 8, n) + 0.3 * np.sin(np.arange(n) / 9)
    ticks = pd.DataFrame({
        "timestamp": ts,
        "bid": price - 0.02,
        "ask": price + 0.02,
    })

    result = run_grid(
        ticks,
        lookbacks=(20, 50),
        horizons=(1.0, 3.0),
        target_returns=(0.0001,),
        threshold=0.55,
    )

    assert not result.empty
    assert {"lookback", "horizon_seconds", "brier", "logloss"} <= set(result.columns)
    assert result["brier"].notna().all()
