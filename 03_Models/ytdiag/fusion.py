"""Linear probes and a small late-fusion MLP over frozen deep embeddings.

Only the projection/fusion layers train. Preprocessing is fitted on each
outer training split and validation rows are used solely for reported metrics.
The seeded development partitions overlap across runs, so this module does not
describe any one retrospective fold as a globally unseen test set.
"""
from __future__ import annotations

import random
import gc
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .baselines import _preprocessor
from .embed import EmbeddingSet, choose_device
from .features import select_columns
from .split import split_indices


ABLATIONS: dict[str, tuple[str, ...]] = {
    "thumbnail": ("thumbnail",),
    "text": ("text",),
    "thumbnail_text": ("thumbnail", "text"),
    "thumbnail_text_meta_sched": ("thumbnail", "text", "meta_sched"),
}


def _dense(value: Any) -> np.ndarray:
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def _embedding_block(bundle: EmbeddingSet, mask_names: Sequence[str]) -> np.ndarray:
    masks = np.column_stack([bundle.masks[name] for name in mask_names]).astype(np.float32)
    return np.column_stack([bundle.values, masks]).astype(np.float32)


def prepare_blocks(
    df: pd.DataFrame,
    thumbnail: EmbeddingSet,
    text: EmbeddingSet,
    train_idx: np.ndarray,
    block_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create train-fitted, model-ready blocks without validation leakage."""
    ids = df.video_id.astype(str).to_numpy()
    if not np.array_equal(ids, thumbnail.video_ids.astype(str)) or not np.array_equal(ids, text.video_ids.astype(str)):
        raise ValueError("embedding rows do not match canonical-table video_id order")

    raw: dict[str, np.ndarray] = {}
    if "thumbnail" in block_names:
        raw["thumbnail"] = _embedding_block(thumbnail, ("thumbnail_present",))
    if "text" in block_names:
        raw["text"] = _embedding_block(text, (
            "title_present", "description_present", "transcript_present",
            "transcript_usable", "transcript_quality_known",
        ))

    return prepare_named_blocks(df, raw, train_idx, block_names)


def prepare_named_blocks(
    df: pd.DataFrame,
    raw: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    block_names: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Scale arbitrary frozen blocks and optionally add train-fitted tabular data."""

    details: dict[str, Any] = {}
    blocks: dict[str, np.ndarray] = {}
    for name in block_names:
        if name in ("meta_sched", "engineered", "audio"):
            continue
        if name not in raw:
            raise ValueError(f"no frozen feature block named {name!r}")
        values = raw[name]
        scaler = StandardScaler().fit(values[train_idx])
        blocks[name] = scaler.transform(values).astype(np.float32)
        details[name] = {"input_dim": int(values.shape[1]), "scaler": "train-only StandardScaler"}

    for tabular_name, groups in (
        ("meta_sched", ("meta", "sched")),
        ("engineered", ("meta", "sched", "vis", "aud")),
        ("audio", ("aud",)),
    ):
        if tabular_name not in block_names:
            continue
        cols = select_columns(df, groups)
        all_nan = [c for c in cols if df[c].isna().all()]
        cols = [c for c in cols if c not in all_nan]
        prep = _preprocessor(cols, scale=True)
        prep.fit(df.iloc[train_idx][cols])
        blocks[tabular_name] = _dense(prep.transform(df[cols])).astype(np.float32)
        details[tabular_name] = {
            "input_dim": int(blocks[tabular_name].shape[1]),
            "source_columns": cols,
            "dropped_all_nan": all_nan,
            "preprocessing": "train-only median imputation, scaling, one-hot encoding",
        }
    return blocks, details


def _best_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda value: f1_score(y, probabilities >= value)))


def _metrics(
    y: np.ndarray, probabilities: np.ndarray, threshold: float,
) -> dict[str, float]:
    """Score predictions at a threshold selected without using these rows."""
    return {
        "auc_roc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "f1": float(f1_score(y, probabilities >= threshold)),
        "threshold": threshold,
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def _by_category(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    categories = df.meta__category.iloc[idx].astype(str).to_numpy()
    return {
        category: (float(roc_auc_score(y[categories == category], probabilities[categories == category]))
                   if len(set(y[categories == category])) == 2 else None)
        for category in sorted(set(categories))
    }


def _inner_split(df: pd.DataFrame, outer_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A channel-grouped 75/25 split wholly inside the outer training fold."""
    subset = df.iloc[outer_train]
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    folds = [test for _, test in splitter.split(
        np.zeros(len(subset)), subset.label.astype(int), subset.channel_id
    )]
    inner_val_local = folds[0]
    inner_train_local = np.concatenate(folds[1:])
    return outer_train[inner_train_local], outer_train[inner_val_local]


def linear_probe(
    blocks: Mapping[str, np.ndarray], y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
    inner_train_idx: np.ndarray, inner_val_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """L2 logistic probe with C selected only inside the outer training fold."""
    matrix = np.column_stack(list(blocks.values()))
    candidates = (0.001, 0.01, 0.1, 1.0)
    scores = {}
    inner_probabilities = {}
    for candidate in candidates:
        trial = LogisticRegression(max_iter=3000, C=candidate, random_state=0)
        trial.fit(matrix[inner_train_idx], y[inner_train_idx])
        candidate_probabilities = trial.predict_proba(matrix[inner_val_idx])[:, 1]
        inner_probabilities[candidate] = candidate_probabilities
        scores[candidate] = float(roc_auc_score(y[inner_val_idx], candidate_probabilities))
    selected = max(candidates, key=lambda value: (scores[value], -value))
    threshold = _best_threshold(y[inner_val_idx], inner_probabilities[selected])
    model = LogisticRegression(max_iter=3000, C=selected, random_state=0)
    model.fit(matrix[train_idx], y[train_idx])
    probabilities = model.predict_proba(matrix[val_idx])[:, 1]
    return probabilities, {
        "n_parameters": int(model.coef_.size + model.intercept_.size),
        "selected_C": float(selected),
        "threshold": float(threshold),
        "threshold_source": "inner validation",
        "inner_validation_auc_by_C": {str(key): value for key, value in scores.items()},
    }


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp_probe(
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    inner_train_idx: np.ndarray,
    inner_val_idx: np.ndarray,
    device: str = "auto",
    epochs: int = 60,
    batch_size: int = 64,
    projection_dim: int = 32,
    hidden_dim: int = 32,
    dropout: float = 0.50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select epoch on inner-train validation, refit, then score outer val."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(seed)
    resolved_device = choose_device(device)
    names = list(blocks)

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

    def fit(rows: np.ndarray, budget: int, monitor: np.ndarray | None = None) -> tuple[nn.Module, int, float, float | None]:
        _seed_everything(seed)
        model = LateFusion().to(resolved_device)
        dataset = TensorDataset(*[value[rows] for value in tensors], labels[rows])
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        positives = float(y[rows].sum())
        pos_weight = torch.tensor((len(rows) - positives) / positives, device=resolved_device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        final_loss = float("nan")
        best_auc = -float("inf")
        best_epoch = budget
        stale = 0
        for epoch in range(1, budget + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch in loader:
                inputs = [value.to(resolved_device) for value in batch[:-1]]
                target = batch[-1].to(resolved_device)
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), target)
                loss.backward()
                optimiser.step()
                total_loss += float(loss.detach().cpu()) * len(target)
                seen += len(target)
            final_loss = total_loss / seen
            if monitor is not None:
                model.eval()
                with torch.inference_mode():
                    logits = model([value[monitor].to(resolved_device) for value in tensors])
                    probabilities = torch.sigmoid(logits).float().cpu().numpy()
                score = float(roc_auc_score(y[monitor], probabilities))
                if score > best_auc + 1e-4:
                    best_auc, best_epoch, stale = score, epoch, 0
                else:
                    stale += 1
                if epoch >= 10 and stale >= patience:
                    break
        return model, best_epoch, final_loss, (best_auc if monitor is not None else None)

    inner_model, selected_epoch, _, inner_auc = fit(inner_train_idx, epochs, monitor=inner_val_idx)
    inner_model.eval()
    with torch.inference_mode():
        inner_logits = inner_model([
            value[inner_val_idx].to(resolved_device) for value in tensors
        ])
        inner_probabilities = torch.sigmoid(inner_logits).float().cpu().numpy()
    threshold = _best_threshold(y[inner_val_idx], inner_probabilities)
    del inner_model
    gc.collect()
    if resolved_device == "mps":
        torch.mps.empty_cache()
    elif resolved_device == "cuda":
        torch.cuda.empty_cache()
    model, _, final_loss, _ = fit(train_idx, selected_epoch)
    model.eval()
    with torch.inference_mode():
        logits = model([value[val_idx].to(resolved_device) for value in tensors])
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
    details = {
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "max_epochs": int(epochs), "selected_epoch": int(selected_epoch),
        "inner_validation_auc": float(inner_auc), "early_stopping_patience": int(patience),
        "batch_size": int(batch_size),
        "projection_dim": int(projection_dim), "hidden_dim": int(hidden_dim),
        "dropout": float(dropout), "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay), "loss": "positive-weighted BCEWithLogitsLoss",
        "threshold": float(threshold), "threshold_source": "inner validation",
        "final_train_loss": final_loss, "device": resolved_device,
    }
    del model
    gc.collect()
    if resolved_device == "mps":
        torch.mps.empty_cache()
    elif resolved_device == "cuda":
        torch.cuda.empty_cache()
    return probabilities, details


def run_seed(
    df: pd.DataFrame,
    thumbnail: EmbeddingSet,
    text: EmbeddingSet,
    seed: int,
    ablations: Sequence[str] = tuple(ABLATIONS),
    device: str = "auto",
    epochs: int = 60,
) -> dict[str, Any]:
    """Run all requested ablations on one channel-grouped outer split."""
    ids = df.video_id.astype(str).to_numpy()
    if not np.array_equal(ids, thumbnail.video_ids.astype(str)) or not np.array_equal(ids, text.video_ids.astype(str)):
        raise ValueError("embedding rows do not match canonical-table video_id order")
    raw = {
        "thumbnail": _embedding_block(thumbnail, ("thumbnail_present",)),
        "text": _embedding_block(text, (
            "title_present", "description_present", "transcript_present",
            "transcript_usable", "transcript_quality_known",
        )),
    }
    return run_named_blocks_seed(df, raw, seed, ABLATIONS, ablations, device, epochs)


def run_named_blocks_seed(
    df: pd.DataFrame,
    raw_blocks: Mapping[str, np.ndarray],
    seed: int,
    ablation_map: Mapping[str, Sequence[str]],
    ablations: Sequence[str],
    device: str = "auto",
    epochs: int = 60,
) -> dict[str, Any]:
    """Evaluate named frozen blocks under the shared nested validation protocol."""
    idx = split_indices(df, seed=seed)
    inner_train_idx, inner_val_idx = _inner_split(df, idx["train"], seed)
    y = df.label.astype(int).to_numpy()
    output: dict[str, Any] = {
        "seed": int(seed),
        "split_sizes": {name: int(len(rows)) for name, rows in idx.items()},
        "test_evaluated": False,
        "ablations": {},
    }
    for ablation in ablations:
        if ablation not in ablation_map:
            raise ValueError(f"unknown ablation {ablation!r}; choose from {list(ablation_map)}")
        block_names = ablation_map[ablation]
        blocks, preprocessing = prepare_named_blocks(df, raw_blocks, idx["train"], block_names)
        probe_p, probe_details = linear_probe(
            blocks, y, idx["train"], idx["val"], inner_train_idx, inner_val_idx
        )
        mlp_p, mlp_details = mlp_probe(
            blocks, y, idx["train"], idx["val"], seed=seed,
            inner_train_idx=inner_train_idx, inner_val_idx=inner_val_idx,
            device=device, epochs=epochs,
        )
        output["ablations"][ablation] = {
            "blocks": list(block_names),
            "preprocessing": preprocessing,
            "linear_probe": {
                "val": _metrics(y[idx["val"]], probe_p, probe_details["threshold"]),
                "val_auc_by_category": _by_category(df, idx["val"], y[idx["val"]], probe_p),
                **probe_details,
            },
            "late_fusion_mlp": {
                "val": _metrics(y[idx["val"]], mlp_p, mlp_details["threshold"]),
                "val_auc_by_category": _by_category(df, idx["val"], y[idx["val"]], mlp_p),
                **mlp_details,
            },
        }
    return output


def aggregate_runs(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± population SD across split/training seeds."""
    aggregate: dict[str, Any] = {}
    for ablation in runs[0]["ablations"]:
        aggregate[ablation] = {}
        for model in ("linear_probe", "late_fusion_mlp"):
            aggregate[ablation][model] = {}
            for metric in ("auc_roc", "pr_auc", "f1"):
                values = np.asarray([run["ablations"][ablation][model]["val"][metric] for run in runs])
                aggregate[ablation][model][metric] = {
                    "mean": float(values.mean()), "std": float(values.std(ddof=0)),
                    "values": values.tolist(),
                }
    return aggregate
