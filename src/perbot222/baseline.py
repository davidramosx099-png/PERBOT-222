from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class BaselineResult:
    model: Pipeline
    feature_columns: list[str]

def fit_logistic_baseline(train: pd.DataFrame, target: str = "y_up") -> BaselineResult:
    excluded = {target, "timestamp", "mid", "future_max_return"}
    feature_columns = [c for c in train.columns if c not in excluded and pd.api.types.is_numeric_dtype(train[c])]
    if not feature_columns:
        raise ValueError("No numeric features available.")
    x = train[feature_columns]
    y = train[target].astype(int)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=222)),
    ])
    model.fit(x, y)
    return BaselineResult(model=model, feature_columns=feature_columns)

def predict_probability(fitted: BaselineResult, frame: pd.DataFrame) -> pd.Series:
    p = fitted.model.predict_proba(frame[fitted.feature_columns])[:, 1]
    return pd.Series(p, index=frame.index, name="p_up")
