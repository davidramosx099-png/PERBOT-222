from __future__ import annotations

import pandas as pd


def long_edge(
    p_up: pd.Series,
    take_profit_return: float,
    stop_loss_return: float,
    round_trip_cost_return: pd.Series | float,
) -> pd.Series:
    """Expected return proxy for a long trade after round-trip costs.

    p_up is the model probability of reaching the target before the horizon.
    stop_loss_return is supplied as a positive magnitude.
    """
    if not 0 < take_profit_return:
        raise ValueError("take_profit_return must be positive")
    if not 0 < stop_loss_return:
        raise ValueError("stop_loss_return must be positive")
    if ((p_up < 0) | (p_up > 1)).any():
        raise ValueError("p_up must be in [0, 1]")

    return (
        p_up * take_profit_return
        - (1.0 - p_up) * stop_loss_return
        - round_trip_cost_return
    ).rename("long_edge")


def make_entry_signal(
    p_up: pd.Series,
    take_profit_return: float,
    stop_loss_return: float,
    round_trip_cost_return: pd.Series | float,
    min_edge: float = 0.0,
) -> pd.DataFrame:
    """Return an auditable long-entry decision from model probability."""
    edge = long_edge(
        p_up=p_up,
        take_profit_return=take_profit_return,
        stop_loss_return=stop_loss_return,
        round_trip_cost_return=round_trip_cost_return,
    )
    out = pd.DataFrame({"p_up": p_up, "edge": edge})
    out["long_signal"] = out["edge"] > min_edge
    return out
