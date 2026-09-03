from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ytdiag.attribution import fit_attributable_linear  # noqa: E402


def test_block_contributions_reconstruct_logit_and_threshold_is_inner_only():
    rng = np.random.default_rng(7)
    n = 120
    y = np.tile([0, 1], n // 2)
    signal = y[:, None] + rng.normal(0, 0.7, size=(n, 2))
    noise = rng.normal(size=(n, 3))
    result = fit_attributable_linear(
        {"signal": signal, "noise": noise}, y,
        train_idx=np.arange(80), eval_idx=np.arange(100, 120),
        inner_train_idx=np.arange(60), inner_val_idx=np.arange(60, 80),
    )
    reconstructed = result["intercept"] + sum(result["contributions"].values())
    np.testing.assert_allclose(reconstructed, result["logits"], atol=1e-6)
    assert result["threshold_source"] == "inner validation"
    assert 0.0 <= result["metrics"]["auc_roc"] <= 1.0
