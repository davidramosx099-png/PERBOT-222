from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, precision_score


@dataclass(frozen=True)
class ClassificationMetrics:
    logloss: float
    brier: float
    precision_at_threshold: float
    signal_rate: float


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split strictly by row order; never shuffle time-series observations."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    cut = int(len(frame) * train_fraction)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def classification_metrics(
    y_true: pd.Series,
    p_up: pd.Series,
    threshold: float = 0.60,
) -> ClassificationMetrics:
    valid = y_true.notna() & p_up.notna()
    y = y_true.loc[valid].astype(int)
    p = p_up.loc[valid].clip(1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    return ClassificationMetrics(
        logloss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        precision_at_threshold=float(
            precision_score(y, pred, zero_division=0)
        ),
        signal_rate=float(pred.mean()),
    )


def probability_threshold_for_positive_edge(
    take_profit_return: float,
    stop_loss_return: float,
    round_trip_cost_return: float,
    minimum_edge: float = 0.0,
) -> float:
    """Solve the minimum p needed for a positive long expected-return proxy."""
    denominator = take_profit_return + stop_loss_return
    if denominator <= 0:
        raise ValueError("TP and SL must be positive.")
    return float(
        (stop_loss_return + round_trip_cost_return + minimum_edge)
        / denominator
    )
