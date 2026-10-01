from __future__ import annotations

import pandas as pd

from .baseline import fit_logistic_baseline, predict_probability
from .experiment import classification_metrics, chronological_split
from .features import build_features
from .labels import make_up_label


def run_grid(
    ticks: pd.DataFrame,
    lookbacks: tuple[int, ...],
    horizons: tuple[float, ...],
    target_returns: tuple[float, ...],
    train_fraction: float = 0.70,
    threshold: float = 0.60,
) -> pd.DataFrame:
    """Run a leakage-safe chronological grid over prediction hypotheses."""
    rows: list[dict] = []

    for lookback in lookbacks:
        features = build_features(ticks, lookback=lookback)
        mid = features["mid"]

        for horizon in horizons:
            for target_return in target_returns:
                labels = make_up_label(
                    mid=mid,
                    timestamps=features["timestamp"],
                    horizon_seconds=horizon,
                    target_return=target_return,
                )

                frame = features.copy()
                frame["y_up"] = labels
                frame = frame.loc[frame["y_up"].notna()].reset_index(drop=True)

                train, test = chronological_split(
                    frame, train_fraction=train_fraction
                )
                if train["y_up"].nunique() < 2 or test["y_up"].nunique() < 2:
                    continue

                fitted = fit_logistic_baseline(train)
                p_test = predict_probability(fitted, test)
                metrics = classification_metrics(
                    test["y_up"], p_test, threshold=threshold
                )

                rows.append({
                    "lookback": lookback,
                    "horizon_seconds": horizon,
                    "target_return": target_return,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "logloss": metrics.logloss,
                    "brier": metrics.brier,
                    "precision": metrics.precision_at_threshold,
                    "signal_rate": metrics.signal_rate,
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "lookback", "horizon_seconds", "target_return",
                "train_rows", "test_rows", "logloss", "brier",
                "precision", "signal_rate",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["brier", "logloss"], ascending=[True, True]
    ).reset_index(drop=True)
