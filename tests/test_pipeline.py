import numpy as np
import pandas as pd

from perbot222.features import build_features
from perbot222.labels import make_up_label


def synthetic_ticks(n=300):
    ts = pd.date_range("2026-01-01", periods=n, freq="100ms", tz="UTC")
    price = 2000.0 + np.linspace(0, 3, n) + 0.2 * np.sin(np.arange(n) / 7)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "bid": price - 0.02,
            "ask": price + 0.02,
        }
    )


def test_features_are_available_only_after_lookback():
    ticks = synthetic_ticks()
    f = build_features(ticks, lookback=50)
    assert f.loc[:48, "ret_lb"].isna().all()
    assert f.loc[49:, "ret_lb"].notna().all()


def test_labels_keep_incomplete_tail_missing():
    ticks = synthetic_ticks()
    mid = (ticks["bid"] + ticks["ask"]) / 2
    y = make_up_label(mid, ticks["timestamp"], horizon_seconds=5, target_return=0.0001)
    assert y.iloc[-1] is pd.NA or pd.isna(y.iloc[-1])
    assert y.iloc[:-60].notna().all()


def test_future_label_changes_do_not_enter_feature_frame():
    ticks = synthetic_ticks()
    features_a = build_features(ticks, lookback=50)
    ticks_b = ticks.copy()
    ticks_b.loc[ticks_b.index[-20:], "bid"] += 100
    ticks_b.loc[ticks_b.index[-20:], "ask"] += 100
    features_b = build_features(ticks_b, lookback=50)
    # A future shock must not alter features sufficiently far before it.
    pd.testing.assert_frame_equal(
        features_a.loc[:250].reset_index(drop=True),
        features_b.loc[:250].reset_index(drop=True),
    )
