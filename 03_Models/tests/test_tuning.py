"""Focused checks for the nested tabular tuning protocol."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402
from ytdiag.synthetic import generate  # noqa: E402
from ytdiag.tuning import inner_group_folds, run_tuned_baselines  # noqa: E402


def _data(n: int = 400):
    return load_retrospective(generate(tempfile.mkdtemp(prefix="ytdiag_tune_"), n=n, seed=7))


def test_inner_folds_cover_outer_train_without_channel_leakage():
    df = _data()
    outer = split_indices(df, seed=2)
    folds = inner_group_folds(df, outer["train"], seed=2, n_splits=4)
    covered = np.sort(np.concatenate([val for _, val in folds]))
    assert np.array_equal(covered, outer["train"])
    for train, val in folds:
        assert not set(df.channel_id.iloc[train]) & set(df.channel_id.iloc[val])
        assert not set(val) & set(outer["val"])
        assert not set(val) & set(outer["test"])


def test_logistic_tuning_is_reproducible_and_scores_validation_only():
    df = _data()
    grid = (
        {"C": 0.01, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    )
    first = run_tuned_baselines(
        df, ("meta",), seed=3, inner_splits=3,
        models=("logistic_regression",), logistic_grid=grid,
    )
    second = run_tuned_baselines(
        df, ("meta",), seed=3, inner_splits=3,
        models=("logistic_regression",), logistic_grid=grid,
    )
    a = first["models"]["logistic_regression"]
    b = second["models"]["logistic_regression"]
    assert first["test_evaluated"] is False and "test" not in a
    assert a["threshold_source"].startswith("inner out-of-fold")
    assert len(a["trials"]) == len(grid)
    assert a["best_parameters"] == b["best_parameters"]
    assert a["threshold"] == b["threshold"]
    assert a["val"]["auc_roc"] == b["val"]["auc_roc"]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"{name} OK")
    print("ALL TUNING TESTS PASSED")
