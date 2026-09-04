#!/usr/bin/env python3
"""Fit the frozen CLIP/text/metadata finalist and publish exact block attributions."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.attribution import fit_attributable_linear  # noqa: E402
from ytdiag.fusion import _inner_split, prepare_named_blocks  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402
from ytdiag.text_v2 import field_text_embeddings  # noqa: E402
from ytdiag.visual_ablation import (  # noqa: E402
    frame_mean_embeddings, thumbnail_embeddings_variant,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK_NAMES = (
    "thumbnail", "frames", "title", "description", "transcript", "metadata_schedule",
)


def _field_block(bundle, include_chunks: bool = False) -> np.ndarray:
    masks = [bundle.masks["field_present"]]
    if include_chunks:
        masks.append(np.log1p(bundle.masks["n_chunks"]))
    return np.column_stack([bundle.values, *masks]).astype(np.float32)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".attribution_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _example(df, row: int, local: int, result: dict, kind: str) -> dict:
    contributions = {
        name: float(result["contributions"][name][local]) for name in BLOCK_NAMES
    }
    return {
        "kind": kind,
        "video_id": str(df.video_id.iloc[row]),
        "title": str(df.asset__title.iloc[row])[:80],
        "category": str(df.meta__category.iloc[row]),
        "format": "Short" if bool(df.meta__is_short.iloc[row]) else "regular",
        "label": int(df.label.iloc[row]),
        "probability": float(result["probabilities"][local]),
        "predicted_label": int(result["probabilities"][local] >= result["threshold"]),
        "contributions": contributions,
        "top_positive_block": max(contributions, key=contributions.get),
        "top_negative_block": min(contributions, key=contributions.get),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "03_Models", "cache", "retrospective"))
    parser.add_argument("--output", default=os.path.join(
        ROOT, "05_Reports", "final_report", "results", "attribution.json",
    ))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    df = load_retrospective(args.data)
    df = df[df.label.notna()].reset_index(drop=True)
    thumbnail = thumbnail_embeddings_variant(
        df, args.cache_dir, "clip_base_center", device=args.device,
    )
    frames = frame_mean_embeddings(
        df, args.cache_dir, "clip_base_center", device=args.device,
    )
    fields, text_provenance = field_text_embeddings(df, args.cache_dir, device=args.device)
    raw = {
        "thumbnail": np.column_stack([
            thumbnail.values, thumbnail.masks["visual_present"],
        ]).astype(np.float32),
        "frames": np.column_stack([
            frames.values, frames.masks["visual_present"],
            np.log1p(frames.masks["images_aggregated"]),
        ]).astype(np.float32),
        "title": _field_block(fields["title"]),
        "description": _field_block(fields["description"]),
        "transcript": _field_block(fields["transcript"], include_chunks=True),
    }
    model_names = ("thumbnail", "frames", "title", "description", "transcript", "meta_sched")
    y = df.label.astype(int).to_numpy()
    runs = []
    examples = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        idx = split_indices(df, seed=seed)
        inner_train, inner_val = _inner_split(df, idx["train"], seed)
        blocks, preprocessing = prepare_named_blocks(df, raw, idx["train"], model_names)
        blocks["metadata_schedule"] = blocks.pop("meta_sched")
        result = fit_attributable_linear(
            blocks, y, idx["train"], idx["val"], inner_train, inner_val,
        )
        mean_absolute = {
            name: float(np.mean(np.abs(result["contributions"][name])))
            for name in BLOCK_NAMES
        }
        total_absolute = sum(mean_absolute.values())
        absolute_share = {
            name: float(value / total_absolute) for name, value in mean_absolute.items()
        }
        runs.append({
            "seed": seed,
            "split_sizes": {name: int(len(rows)) for name, rows in idx.items()},
            "metrics": result["metrics"],
            "selected_C": result["selected_C"],
            "threshold": result["threshold"],
            "threshold_source": result["threshold_source"],
            "intercept": result["intercept"],
            "mean_absolute_logit_contribution": mean_absolute,
            "share_of_absolute_logit_contribution": absolute_share,
            "preprocessing": preprocessing,
        })
        if seed == 0:
            truth = y[idx["val"]]
            predicted = result["probabilities"] >= result["threshold"]
            cases = {
                "true positive": np.flatnonzero((truth == 1) & predicted),
                "true negative": np.flatnonzero((truth == 0) & ~predicted),
                "false positive": np.flatnonzero((truth == 0) & predicted),
                "false negative": np.flatnonzero((truth == 1) & ~predicted),
            }
            selectors = {
                "true positive": np.argmax,
                "true negative": np.argmin,
                "false positive": np.argmax,
                "false negative": np.argmin,
            }
            for kind, local_rows in cases.items():
                if not len(local_rows):
                    continue
                probabilities = result["probabilities"][local_rows]
                local = int(local_rows[selectors[kind](probabilities)])
                examples.append(_example(df, int(idx["val"][local]), local, result, kind))

    aggregate = {}
    for metric in ("auc_roc", "pr_auc", "f1"):
        values = np.asarray([run["metrics"][metric] for run in runs])
        aggregate[metric] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "values": values.tolist(),
        }
    block_summary = {}
    for name in BLOCK_NAMES:
        values = np.asarray([run["share_of_absolute_logit_contribution"][name] for run in runs])
        block_summary[name] = {
            "mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "values": values.tolist(),
        }
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": (
            "five repeated channel-grouped development splits; C and F1 threshold selected "
            "inside each training partition; validation only; no independent retrospective test"
        ),
        "explanation_scope": (
            "exact decomposition of the linear model logit by standardized input block; "
            "predictive attribution, not causal attribution"
        ),
        "n_labelled": int(len(df)),
        "blocks": list(BLOCK_NAMES),
        "aggregate": aggregate,
        "share_of_absolute_logit_contribution": block_summary,
        "qualitative_seed": 0,
        "qualitative_examples": examples,
        "runs": runs,
        "embedding_provenance": {
            "thumbnail": thumbnail.provenance,
            "frames": frames.provenance,
            "text": text_provenance,
        },
    }
    _atomic_json(args.output, payload)
    print(json.dumps({
        "output": args.output,
        "aggregate": aggregate,
        "share_of_absolute_logit_contribution": block_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
