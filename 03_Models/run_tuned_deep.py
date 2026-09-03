"""Nested tuning of linear and MLP heads over cached frozen embeddings.

The encoders are never fine-tuned or rerun unless their cache is absent. This
command has no test-evaluation option.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.deep_tuning import aggregate_tuned_deep, run_tuned_deep_seed  # noqa: E402
from ytdiag.embed import EmbeddingSet, thumbnail_embeddings  # noqa: E402
from ytdiag.text_v2 import field_text_embeddings  # noqa: E402


ABLATIONS = {
    "thumbnail_text_fields_meta_sched": (
        "thumbnail", "title", "description", "transcript", "meta_sched",
    ),
    "thumbnail_text_fields_engineered": (
        "thumbnail", "title", "description", "transcript", "engineered",
    ),
}


def _field_block(bundle: EmbeddingSet, include_chunks: bool = False) -> np.ndarray:
    masks = [bundle.masks["field_present"]]
    if include_chunks:
        masks.append(np.log1p(bundle.masks["n_chunks"]))
    return np.column_stack([bundle.values, *masks]).astype(np.float32)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tuned_deep_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "03_Models", "cache", "retrospective"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "04_Experiments", "runs", "tuned_deep"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--ablations", default=",".join(ABLATIONS))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--mlp-random-trials", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    df = load_retrospective(args.data)
    df = df[df.label.notna()].reset_index(drop=True)
    fields, text_provenance = field_text_embeddings(
        df, args.cache_dir, batch_size=16, device=args.device, force=False,
    )
    thumbnail = thumbnail_embeddings(
        df, args.cache_dir, batch_size=64, device=args.device, force=False,
    )
    raw_blocks = {
        "title": _field_block(fields["title"]),
        "description": _field_block(fields["description"]),
        "transcript": _field_block(fields["transcript"], include_chunks=True),
        "thumbnail": np.column_stack([
            thumbnail.values, thumbnail.masks["thumbnail_present"],
        ]).astype(np.float32),
    }
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    ablations = [value.strip() for value in args.ablations.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        run = run_tuned_deep_seed(
            df, raw_blocks, seed, ABLATIONS, ablations, device=args.device,
            inner_splits=args.inner_splits, mlp_random_trials=args.mlp_random_trials,
            search_seed=args.search_seed, max_epochs=args.max_epochs, patience=args.patience,
        )
        runs.append(run)
        print(f"seed {seed} (test untouched)")
        for name, result in run["ablations"].items():
            linear = result["linear_probe"]["val"]["auc_roc"]
            mlp = result["late_fusion_mlp"]["val"]["auc_roc"]
            print(f"  {name:40s} linear={linear:.3f} MLP={mlp:.3f}")
    aggregate = aggregate_tuned_deep(runs)
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": (
            f"channel-grouped 60/20/20 outer split; {args.inner_splits} inner grouped folds; "
            "per-fold grouped monitor split for epoch selection; outer validation only; test untouched"
        ),
        "n_labelled": len(df), "seeds": seeds,
        "configuration": {
            "ablations": ablations, "mlp_random_trials": args.mlp_random_trials,
            "mlp_candidates_including_fixed": args.mlp_random_trials + 1,
            "search_seed": args.search_seed, "max_epochs": args.max_epochs,
            "patience": args.patience,
        },
        "embedding_provenance": {
            "text": text_provenance, "thumbnail": thumbnail.provenance,
        },
        "runs": runs, "aggregate": aggregate,
    }
    path = os.path.join(args.out_dir, "results.json")
    _atomic_json(path, payload)
    print("\nmean ± population SD outer-validation AUC")
    for name, models in aggregate.items():
        linear = models["linear_probe"]["auc_roc"]
        mlp = models["late_fusion_mlp"]["auc_roc"]
        print(f"  {name:40s} linear={linear['mean']:.3f}±{linear['std']:.3f} "
              f"MLP={mlp['mean']:.3f}±{mlp['std']:.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
