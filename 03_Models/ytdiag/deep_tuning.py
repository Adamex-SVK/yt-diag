"""Nested hyperparameter search over frozen-embedding fusion classifiers.

Encoders remain frozen. Candidate linear probes and MLP heads are selected on
channel-grouped folds wholly inside each outer training set. Every inner fold
fits its own preprocessing, and MLP epoch selection uses a further grouped
monitor split inside that fold. Outer validation is evaluation-only; no test
evaluation exists in this module.
"""
from __future__ import annotations

import gc
import itertools
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold

from .embed import choose_device
from .fusion import _by_category, prepare_named_blocks
from .split import split_indices
from .tuning import _metrics, _threshold, inner_group_folds


DEFAULT_MLP_PARAMETERS: dict[str, Any] = {
    "projection_dim": 32,
    "hidden_dim": 32,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "batch_size": 64,
    "positive_weight": True,
}

MLP_PARAMETER_SPACE: Mapping[str, Sequence[Any]] = {
    "projection_dim": (16, 32, 64),
    "hidden_dim": (16, 32, 64),
    "dropout": (0.2, 0.35, 0.5, 0.65),
    "learning_rate": (3e-4, 1e-3, 3e-3),
    "weight_decay": (1e-4, 1e-3, 1e-2),
    "batch_size": (32, 64, 128),
    "positive_weight": (False, True),
}

DEFAULT_LINEAR_GRID: tuple[dict[str, Any], ...] = tuple(
    {"C": c, "class_weight": weight}
    for c, weight in itertools.product(
        (0.001, 0.01, 0.1, 1.0, 10.0, 100.0), (None, "balanced")
    )
)


def mlp_candidates(n_random: int, search_seed: int = 2026) -> list[dict[str, Any]]:
    """Frozen, reproducible random search with the old head always included."""
    sampled = [dict(value) for value in ParameterSampler(
        MLP_PARAMETER_SPACE, n_iter=n_random, random_state=search_seed
    )]
    candidates = [dict(DEFAULT_MLP_PARAMETERS)]
    candidates.extend(value for value in sampled if value != DEFAULT_MLP_PARAMETERS)
    return candidates


def _jsonable(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in parameters.items()
    }


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release(device: str) -> None:
    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _monitor_split(
    df: pd.DataFrame, rows: np.ndarray, seed: int, n_splits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Grouped train/monitor split inside one inner-training fold."""
    subset = df.iloc[rows]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_local, monitor_local = next(splitter.split(
        np.zeros(len(subset)), subset.label.astype(int), subset.channel_id
    ))
    train, monitor = rows[train_local], rows[monitor_local]
    assert not set(df.channel_id.iloc[train]) & set(df.channel_id.iloc[monitor])
    return train, monitor


def _fit_mlp(
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    parameters: Mapping[str, Any],
    seed: int,
    device: str,
    epochs: int,
    monitor_idx: np.ndarray | None = None,
    patience: int = 8,
) -> tuple[Any, int, float, float | None]:
    """Fit one head; optionally select its epoch on a disjoint monitor set."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(seed)
    names = list(blocks)
    projection_dim = int(parameters["projection_dim"])
    hidden_dim = int(parameters["hidden_dim"])
    dropout = float(parameters["dropout"])

    class LateFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.branches = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(blocks[name].shape[1], projection_dim),
                    nn.LayerNorm(projection_dim), nn.GELU(), nn.Dropout(dropout),
                ) for name in names
            })
            self.head = nn.Sequential(
                nn.Linear(projection_dim * len(names), hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )

        def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
            projected = [self.branches[name](value) for name, value in zip(names, inputs)]
            return self.head(torch.cat(projected, dim=1)).squeeze(1)

    tensors = [torch.from_numpy(blocks[name]) for name in names]
    labels = torch.from_numpy(y.astype(np.float32))
    model = LateFusion().to(device)
    dataset = TensorDataset(*[value[train_idx] for value in tensors], labels[train_idx])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(parameters["batch_size"]), shuffle=True, generator=generator
    )
    if bool(parameters["positive_weight"]):
        positives = float(y[train_idx].sum())
        pos_weight = torch.tensor(
            (len(train_idx) - positives) / positives, dtype=torch.float32, device=device
        )
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    best_auc = -float("inf")
    best_epoch = epochs
    stale = 0
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            inputs = [value.to(device) for value in batch[:-1]]
            target = batch[-1].to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach().cpu()) * len(target)
            seen += len(target)
        final_loss = total_loss / seen
        if monitor_idx is not None:
            probabilities = _predict_mlp(model, tensors, monitor_idx, device)
            auc = float(roc_auc_score(y[monitor_idx], probabilities))
            if auc > best_auc + 1e-4:
                best_auc, best_epoch, stale = auc, epoch, 0
            else:
                stale += 1
            if epoch >= 10 and stale >= patience:
                break
    return model, int(best_epoch), float(final_loss), (
        float(best_auc) if monitor_idx is not None else None
    )


def _predict_mlp(model: Any, tensors: Sequence[Any], rows: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        logits = model([value[rows].to(device) for value in tensors])
        return torch.sigmoid(logits).float().cpu().numpy()


def _fit_fixed_epoch_predict(
    blocks: Mapping[str, np.ndarray], y: np.ndarray, train_idx: np.ndarray,
    eval_idx: np.ndarray, parameters: Mapping[str, Any], seed: int,
    device: str, epochs: int,
) -> tuple[np.ndarray, int, float]:
    import torch

    model, _, loss, _ = _fit_mlp(
        blocks, y, train_idx, parameters, seed, device, epochs, monitor_idx=None
    )
    tensors = [torch.from_numpy(blocks[name]) for name in blocks]
    probabilities = _predict_mlp(model, tensors, eval_idx, device)
    n_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    del model
    _release(device)
    return probabilities, n_parameters, loss


def _select_epoch_and_score(
    df: pd.DataFrame,
    raw_blocks: Mapping[str, np.ndarray],
    block_names: Sequence[str],
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    parameters: Mapping[str, Any],
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, int, int]:
    """Select epoch inside train_idx, refit train_idx, then score eval_idx."""
    import torch

    pilot_train, monitor = _monitor_split(df, train_idx, seed=seed)
    pilot_blocks, _ = prepare_named_blocks(df, raw_blocks, pilot_train, block_names)
    pilot, selected_epoch, _, _ = _fit_mlp(
        pilot_blocks, y, pilot_train, parameters, seed, device,
        max_epochs, monitor_idx=monitor, patience=patience,
    )
    del pilot
    _release(device)

    fold_blocks, _ = prepare_named_blocks(df, raw_blocks, train_idx, block_names)
    probabilities, n_parameters, _ = _fit_fixed_epoch_predict(
        fold_blocks, y, train_idx, eval_idx, parameters, seed, device, selected_epoch
    )
    return probabilities, selected_epoch, n_parameters


def _tune_linear(
    df: pd.DataFrame,
    raw_blocks: Mapping[str, np.ndarray],
    block_names: Sequence[str],
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_val: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    candidates: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    trials = []
    oof_by_trial = []
    for parameters in candidates:
        oof = np.full(len(y), np.nan)
        aucs = []
        for train_idx, eval_idx in folds:
            blocks, _ = prepare_named_blocks(df, raw_blocks, train_idx, block_names)
            matrix = np.column_stack(list(blocks.values()))
            model = LogisticRegression(
                max_iter=3000, C=float(parameters["C"]),
                class_weight=parameters["class_weight"], random_state=seed,
            )
            model.fit(matrix[train_idx], y[train_idx])
            oof[eval_idx] = model.predict_proba(matrix[eval_idx])[:, 1]
            aucs.append(float(roc_auc_score(y[eval_idx], oof[eval_idx])))
        oof_by_trial.append(oof)
        trials.append({
            "parameters": _jsonable(parameters), "fold_auc": aucs,
            "mean_inner_auc": float(np.mean(aucs)), "std_inner_auc": float(np.std(aucs)),
        })
    best = max(range(len(trials)), key=lambda i: (trials[i]["mean_inner_auc"], -trials[i]["std_inner_auc"]))
    threshold = _threshold(y[outer_train], oof_by_trial[best][outer_train])
    blocks, preprocessing = prepare_named_blocks(df, raw_blocks, outer_train, block_names)
    matrix = np.column_stack(list(blocks.values()))
    parameters = candidates[best]
    model = LogisticRegression(
        max_iter=3000, C=float(parameters["C"]),
        class_weight=parameters["class_weight"], random_state=seed,
    )
    model.fit(matrix[outer_train], y[outer_train])
    probability = model.predict_proba(matrix[outer_val])[:, 1]
    return {
        "best_parameters": _jsonable(parameters),
        "best_inner_auc": trials[best]["mean_inner_auc"],
        "threshold": threshold,
        "threshold_source": "inner out-of-fold predictions on outer training rows",
        "preprocessing": preprocessing,
        "n_parameters": int(model.coef_.size + model.intercept_.size),
        "val": _metrics(y[outer_val], probability, threshold),
        "val_auc_by_category": _by_category(df, outer_val, y[outer_val], probability),
        "trials": trials,
    }


def _tune_mlp(
    df: pd.DataFrame,
    raw_blocks: Mapping[str, np.ndarray],
    block_names: Sequence[str],
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_val: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    candidates: Sequence[Mapping[str, Any]],
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    trials = []
    oof_by_trial = []
    for parameters in candidates:
        oof = np.full(len(y), np.nan)
        aucs, epochs = [], []
        n_parameters = 0
        for fold_index, (train_idx, eval_idx) in enumerate(folds):
            fold_seed = seed * 1000 + fold_index
            probability, selected_epoch, n_parameters = _select_epoch_and_score(
                df, raw_blocks, block_names, y, train_idx, eval_idx, parameters,
                seed=fold_seed, device=device, max_epochs=max_epochs, patience=patience,
            )
            oof[eval_idx] = probability
            aucs.append(float(roc_auc_score(y[eval_idx], probability)))
            epochs.append(int(selected_epoch))
        oof_by_trial.append(oof)
        trials.append({
            "parameters": _jsonable(parameters), "fold_auc": aucs,
            "selected_epochs": epochs, "n_parameters": n_parameters,
            "mean_inner_auc": float(np.mean(aucs)), "std_inner_auc": float(np.std(aucs)),
        })
    best = max(range(len(trials)), key=lambda i: (trials[i]["mean_inner_auc"], -trials[i]["std_inner_auc"]))
    threshold = _threshold(y[outer_train], oof_by_trial[best][outer_train])
    selected_epoch = max(1, int(np.median(trials[best]["selected_epochs"])))
    blocks, preprocessing = prepare_named_blocks(df, raw_blocks, outer_train, block_names)
    probability, n_parameters, loss = _fit_fixed_epoch_predict(
        blocks, y, outer_train, outer_val, candidates[best], seed, device, selected_epoch
    )
    return {
        "best_parameters": _jsonable(candidates[best]),
        "best_inner_auc": trials[best]["mean_inner_auc"],
        "selected_epoch": selected_epoch,
        "epoch_policy": "median of per-inner-fold monitor-selected epochs",
        "threshold": threshold,
        "threshold_source": "inner out-of-fold predictions on outer training rows",
        "preprocessing": preprocessing,
        "n_parameters": n_parameters,
        "final_train_loss": loss,
        "device": device,
        "val": _metrics(y[outer_val], probability, threshold),
        "val_auc_by_category": _by_category(df, outer_val, y[outer_val], probability),
        "trials": trials,
    }


def run_tuned_deep_seed(
    df: pd.DataFrame,
    raw_blocks: Mapping[str, np.ndarray],
    seed: int,
    ablation_map: Mapping[str, Sequence[str]],
    ablations: Sequence[str],
    device: str = "auto",
    inner_splits: int = 3,
    mlp_random_trials: int = 8,
    search_seed: int = 2026,
    max_epochs: int = 60,
    patience: int = 8,
    linear_grid: Sequence[Mapping[str, Any]] = DEFAULT_LINEAR_GRID,
) -> dict[str, Any]:
    """Tune linear and MLP fusion heads for one outer split seed."""
    resolved_device = choose_device(device)
    outer = split_indices(df, seed=seed)
    folds = inner_group_folds(df, outer["train"], seed=seed, n_splits=inner_splits)
    y = df.label.astype(int).to_numpy()
    candidates = mlp_candidates(mlp_random_trials, search_seed=search_seed)
    output: dict[str, Any] = {
        "seed": seed,
        "split_sizes": {name: int(len(rows)) for name, rows in outer.items()},
        "inner_splits": inner_splits,
        "test_evaluated": False,
        "ablations": {},
    }
    for name in ablations:
        if name not in ablation_map:
            raise ValueError(f"unknown ablation {name!r}")
        blocks = ablation_map[name]
        output["ablations"][name] = {
            "blocks": list(blocks),
            "linear_probe": _tune_linear(
                df, raw_blocks, blocks, y, outer["train"], outer["val"],
                folds, linear_grid, seed,
            ),
            "late_fusion_mlp": _tune_mlp(
                df, raw_blocks, blocks, y, outer["train"], outer["val"],
                folds, candidates, seed, resolved_device, max_epochs, patience,
            ),
        }
    return output


def aggregate_tuned_deep(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for ablation in runs[0]["ablations"]:
        output[ablation] = {}
        for model in ("linear_probe", "late_fusion_mlp"):
            output[ablation][model] = {}
            for metric in (
                "auc_roc", "pr_auc", "accuracy", "balanced_accuracy",
                "precision", "recall", "specificity", "f1",
            ):
                values = [run["ablations"][ablation][model]["val"][metric] for run in runs]
                output[ablation][model][metric] = {
                    "mean": float(np.mean(values)), "std": float(np.std(values)),
                    "values": [float(value) for value in values],
                }
            output[ablation][model]["selected_parameters"] = [
                run["ablations"][ablation][model]["best_parameters"] for run in runs
            ]
    return output
