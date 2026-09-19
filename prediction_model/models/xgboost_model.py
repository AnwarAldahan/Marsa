"""XGBoost wrapper that fails loudly when the dependency is unavailable."""

from __future__ import annotations


def require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "XGBoost experiment cannot run because the `xgboost` package is not importable in this environment."
        ) from exc
    return xgb


def train_xgboost(*args, **kwargs):
    xgb = require_xgboost()
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=kwargs.get("random_seed", 42),
    )
