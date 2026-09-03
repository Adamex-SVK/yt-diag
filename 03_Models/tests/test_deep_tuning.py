"""Offline checks for leakage-safe deep-head tuning."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.deep_tuning import DEFAULT_MLP_PARAMETERS, mlp_candidates, run_tuned_deep_seed  # noqa: E402
from ytdiag.synthetic import generate  # noqa: E402


def _data(n: int = 240):
    df = load_retrospective(generate(tempfile.mkdtemp(prefix="deep_tune_"), n=n, seed=11))
    rng = np.random.default_rng(11)
    return df, {
        "thumbnail": rng.normal(size=(n, 12)).astype(np.float32),
        "text": rng.normal(size=(n, 16)).astype(np.float32),
    }


def test_candidate_search_is_frozen_and_contains_old_head():
    first = mlp_candidates(3, search_seed=9)
    second = mlp_candidates(3, search_seed=9)
    assert first == second
    assert first[0] == DEFAULT_MLP_PARAMETERS and len(first) == 4


def test_nested_deep_tuning_runs_and_scores_validation_only():
    df, blocks = _data()
    result = run_tuned_deep_seed(
        df, blocks, seed=1,
        ablation_map={"both": ("thumbnail", "text")}, ablations=("both",),
        device="cpu", inner_splits=3, mlp_random_trials=1,
        max_epochs=4, patience=2,
        linear_grid=(
            {"C": 0.1, "class_weight": None},
            {"C": 1.0, "class_weight": "balanced"},
        ),
    )
    assert result["test_evaluated"] is False
    tuned = result["ablations"]["both"]
    assert "test" not in tuned["linear_probe"] and "test" not in tuned["late_fusion_mlp"]
    assert tuned["linear_probe"]["threshold_source"].startswith("inner out-of-fold")
    assert tuned["late_fusion_mlp"]["epoch_policy"].startswith("median")
    assert len(tuned["late_fusion_mlp"]["trials"]) == 2


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"{name} OK")
    print("ALL DEEP TUNING TESTS PASSED")
