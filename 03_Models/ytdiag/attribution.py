"""Exact modality-block attribution for a linear late-fusion classifier.

For standardized blocks x_j and logistic-regression weights w_j, the logit is
z = b + sum_j w_j^T x_j. Each dot product is therefore an exact contribution
to the model prediction, rather than an approximation or a causal claim.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .fusion import _best_threshold


def fit_attributable_linear(
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    candidates: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
) -> dict[str, Any]:
    """Fit a nested-selected linear fusion and exactly decompose eval logits."""
    names = list(blocks)
    matrix = np.column_stack([blocks[name] for name in names])
    inner_scores: dict[float, float] = {}
    inner_probabilities: dict[float, np.ndarray] = {}
    for candidate in candidates:
        trial = LogisticRegression(max_iter=3000, C=candidate, random_state=0)
        trial.fit(matrix[inner_train_idx], y[inner_train_idx])
        probability = trial.predict_proba(matrix[inner_val_idx])[:, 1]
        inner_probabilities[candidate] = probability
        inner_scores[candidate] = float(roc_auc_score(y[inner_val_idx], probability))

    selected = max(candidates, key=lambda value: (inner_scores[value], -value))
    threshold = _best_threshold(y[inner_val_idx], inner_probabilities[selected])
    model = LogisticRegression(max_iter=3000, C=selected, random_state=0)
    model.fit(matrix[train_idx], y[train_idx])

    logits = model.decision_function(matrix[eval_idx])
    probabilities = model.predict_proba(matrix[eval_idx])[:, 1]
    contributions: dict[str, np.ndarray] = {}
    start = 0
    for name in names:
        stop = start + blocks[name].shape[1]
        contributions[name] = blocks[name][eval_idx] @ model.coef_[0, start:stop]
        start = stop
    reconstructed = float(model.intercept_[0]) + sum(contributions.values())
    if not np.allclose(logits, reconstructed, atol=1e-5):
        raise AssertionError("block contributions do not reconstruct the model logit")

    truth = y[eval_idx]
    prediction = probabilities >= threshold
    return {
        "probabilities": probabilities,
        "logits": logits,
        "contributions": contributions,
        "intercept": float(model.intercept_[0]),
        "selected_C": float(selected),
        "inner_auc_by_C": {str(key): value for key, value in inner_scores.items()},
        "threshold": float(threshold),
        "threshold_source": "inner validation",
        "metrics": {
            "auc_roc": float(roc_auc_score(truth, probabilities)),
            "pr_auc": float(average_precision_score(truth, probabilities)),
            "f1": float(f1_score(truth, prediction)),
            "n": int(len(eval_idx)),
        },
    }
