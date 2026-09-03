"""Leakage-safe nested hyperparameter tuning for the tabular baselines.

The ordinary baselines intentionally use fixed parameters.  This module keeps
them as reproducible reference points and adds a separate protocol:

1. make the project's channel-grouped 60/20/20 outer split;
2. tune only on channel-grouped folds inside the outer training rows;
3. choose the F1 threshold from out-of-fold predictions on those training rows;
4. refit on all outer-training rows and report the untouched outer validation.

The outer test fold is deliberately not exposed by this module.  It remains
sealed until the complete modelling policy has been frozen.
"""
from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .baselines import MIN_LABELED_ROWS, _preprocessor
from .features import select_columns
from .split import split_indices

DEFAULT_LOGISTIC_GRID: tuple[dict[str, Any], ...] = tuple(
    {"C": c, "class_weight": class_weight}
    for c, class_weight in itertools.product(
        (0.001, 0.01, 0.1, 1.0, 10.0, 100.0), (None, "balanced")
    )
)

XGB_PARAMETER_SPACE: Mapping[str, Sequence[Any]] = {
    "n_estimators": (150, 300, 600),
    "max_depth": (2, 3, 4, 6),
    "learning_rate": (0.01, 0.03, 0.05, 0.1),
    "min_child_weight": (1, 3, 5, 10),
    "subsample": (0.6, 0.8, 1.0),
    "colsample_bytree": (0.6, 0.8, 1.0),
    "reg_alpha": (0.0, 0.1, 1.0),
    "reg_lambda": (0.5, 1.0, 5.0, 10.0),
    "gamma": (0.0, 0.1, 0.5),
    "scale_pos_weight": (1.0, 2.0, 3.0),
}

# Always evaluate the old baseline configuration.  This makes the tuned result
# interpretable even when a small random search happens not to sample near it.
UNTUNED_XGB_PARAMETERS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "scale_pos_weight": 1.0,
}


def inner_group_folds(
    df: pd.DataFrame,
    outer_train: np.ndarray,
    seed: int,
    n_splits: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Global row indices for stratified, channel-disjoint inner folds."""
    subset = df.iloc[outer_train]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_local, val_local in splitter.split(
        np.zeros(len(subset)), subset.label.astype(int), subset.channel_id
    ):
        train_idx, val_idx = outer_train[train_local], outer_train[val_local]
        train_channels = set(df.channel_id.iloc[train_idx])
        val_channels = set(df.channel_id.iloc[val_idx])
        assert not train_channels & val_channels, "channel leaked across inner folds"
        folds.append((train_idx, val_idx))
    covered = np.sort(np.concatenate([val for _, val in folds]))
    assert np.array_equal(covered, np.sort(outer_train)), "inner validation folds do not partition training"
    return folds


def _threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda value: f1_score(y, probabilities >= value)))


def _metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    specificity = recall_score(y, prediction, pos_label=0, zero_division=0)
    return {
        "auc_roc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def _jsonable_params(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in parameters.items()
    }


def _tune_one(
    X: pd.DataFrame,
    y: np.ndarray,
    df: pd.DataFrame,
    outer_train: np.ndarray,
    outer_val: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    candidates: Sequence[Mapping[str, Any]],
    factory: Callable[[Mapping[str, Any]], BaseEstimator],
) -> dict[str, Any]:
    """Select by mean inner AUC, then score once on outer validation."""
    trials: list[dict[str, Any]] = []
    for parameters in candidates:
        fold_auc = []
        for train_idx, val_idx in folds:
            model = factory(parameters)
            model.fit(X.iloc[train_idx], y[train_idx])
            probability = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_auc.append(float(roc_auc_score(y[val_idx], probability)))
        trials.append({
            "parameters": _jsonable_params(parameters),
            "mean_inner_auc": float(np.mean(fold_auc)),
            "std_inner_auc": float(np.std(fold_auc)),
            "fold_auc": fold_auc,
        })
    best_index = max(
        range(len(trials)),
        key=lambda index: (trials[index]["mean_inner_auc"], -trials[index]["std_inner_auc"]),
    )
    best_parameters = candidates[best_index]

    # Training-fold OOF probabilities are the only data used to select the
    # operating threshold. Outer validation remains a genuine evaluation set.
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, val_idx in folds:
        model = factory(best_parameters)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    if np.isnan(oof[outer_train]).any():
        raise AssertionError("missing inner out-of-fold probabilities")
    threshold = _threshold(y[outer_train], oof[outer_train])

    final_model = factory(best_parameters)
    final_model.fit(X.iloc[outer_train], y[outer_train])
    val_probability = final_model.predict_proba(X.iloc[outer_val])[:, 1]
    categories = df.meta__category.iloc[outer_val].to_numpy()
    by_category = {}
    for category in sorted(set(categories)):
        mask = categories == category
        by_category[category] = (
            float(roc_auc_score(y[outer_val][mask], val_probability[mask]))
            if len(set(y[outer_val][mask])) == 2 else None
        )
    return {
        "selection_metric": "mean inner-fold AUC-ROC",
        "best_parameters": _jsonable_params(best_parameters),
        "best_inner_auc": trials[best_index]["mean_inner_auc"],
        "threshold": threshold,
        "threshold_source": "inner out-of-fold predictions on outer training rows",
        "val": _metrics(y[outer_val], val_probability, threshold),
        "val_auc_by_category": by_category,
        "trials": trials,
    }


def _xgb_candidates(n_trials: int, search_seed: int) -> list[dict[str, Any]]:
    sampled = [dict(item) for item in ParameterSampler(
        XGB_PARAMETER_SPACE, n_iter=n_trials, random_state=search_seed
    )]
    candidates = [dict(UNTUNED_XGB_PARAMETERS)]
    candidates.extend(item for item in sampled if item != UNTUNED_XGB_PARAMETERS)
    return candidates


def run_tuned_baselines(
    df: pd.DataFrame,
    groups: Sequence[str],
    seed: int = 0,
    inner_splits: int = 4,
    xgb_trials: int = 16,
    search_seed: int = 2026,
    models: Sequence[str] = ("logistic_regression", "xgboost"),
    logistic_grid: Sequence[Mapping[str, Any]] = DEFAULT_LOGISTIC_GRID,
) -> dict[str, Any]:
    """Tune tabular baselines without consulting outer validation or test."""
    total = len(df)
    df = df[df.label.notna()].reset_index(drop=True)
    if len(df) < MIN_LABELED_ROWS:
        raise ValueError(f"only {len(df)} of {total} rows have labels")
    unknown = set(models) - {"logistic_regression", "xgboost"}
    if unknown:
        raise ValueError(f"unknown tuned model(s): {sorted(unknown)}")

    columns = select_columns(df, groups)
    all_nan = [column for column in columns if df[column].isna().all()]
    columns = [column for column in columns if column not in all_nan]
    if not columns:
        raise ValueError(f"every column for groups {groups} is empty")

    outer = split_indices(df, seed=seed)
    folds = inner_group_folds(df, outer["train"], seed=seed, n_splits=inner_splits)
    X = df[columns]
    y = df.label.astype(int).to_numpy()
    result: dict[str, Any] = {
        "groups": list(groups),
        "n_features": len(columns),
        "dropped_all_nan": all_nan,
        "seed": seed,
        "split_sizes": {name: int(len(index)) for name, index in outer.items()},
        "inner_splits": inner_splits,
        "test_evaluated": False,
        "models": {},
    }

    if "logistic_regression" in models:
        def logistic_factory(parameters: Mapping[str, Any]) -> Pipeline:
            return Pipeline([
                ("prep", _preprocessor(columns, scale=True)),
                ("clf", LogisticRegression(
                    max_iter=3000,
                    C=float(parameters["C"]),
                    class_weight=parameters["class_weight"],
                    random_state=seed,
                )),
            ])

        result["models"]["logistic_regression"] = _tune_one(
            X, y, df, outer["train"], outer["val"], folds,
            list(logistic_grid), logistic_factory,
        )

    if "xgboost" in models:
        try:
            from xgboost import XGBClassifier
        except Exception as error:
            raise RuntimeError(
                "tuned XGBoost requires the xgboost dependency; unlike the fixed baseline, "
                "silently tuning a different estimator would make comparisons invalid"
            ) from error

        def xgb_factory(parameters: Mapping[str, Any]) -> Pipeline:
            return Pipeline([
                ("prep", _preprocessor(columns, scale=False)),
                ("clf", XGBClassifier(
                    **parameters,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=4,
                )),
            ])

        candidates = _xgb_candidates(xgb_trials, search_seed=search_seed)
        result["models"]["xgboost"] = _tune_one(
            X, y, df, outer["train"], outer["val"], folds,
            candidates, xgb_factory,
        )
    return result
