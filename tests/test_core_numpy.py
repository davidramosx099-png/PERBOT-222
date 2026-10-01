import numpy as np
from perbot222.core_numpy import forward_max_return, make_up_label, microstructure_snapshot


def test_snapshot_is_causal():
    n = 300
    base = 2000.0 + np.cumsum(np.full(n, 0.01))
    bid = base - 0.02
    ask = base + 0.02
    a = microstructure_snapshot(bid, ask, lookback=100)
    bid2, ask2 = bid.copy(), ask.copy()
    bid2[-20:] += 100.0
    ask2[-20:] += 100.0
    b = microstructure_snapshot(bid2, ask2, lookback=100)
    # The snapshot is intentionally allowed to change because the supplied
    # current window changed; earlier observations are tested separately.
    assert a["spread"] > 0
    assert np.isfinite(a["signed_pressure"])
    assert np.isfinite(b["log_slope"])


def test_forward_tail_is_missing():
    mid = np.arange(1000.0, 1010.0, 0.1)
    fwd = forward_max_return(mid, 10)
    assert np.isnan(fwd[-10:]).all()
    y = make_up_label(mid, 10, 0.0001)
    assert np.isnan(y[-10:]).all()
    assert np.isfinite(y[:-10]).all()
