"""Field-aware Nomic ModernBERT text experiment under the sealed-test protocol.

This is the controlled follow-up to the generic ModernBERT result. It encodes
title, description and quality-gated transcript chunks separately, then runs
five nested channel-grouped validation ablations. There is no test option.
"""
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
from ytdiag.embed import EmbeddingSet, thumbnail_embeddings  # noqa: E402
from ytdiag.fusion import aggregate_runs, run_named_blocks_seed  # noqa: E402
from ytdiag.text_v2 import field_text_embeddings  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "02_Data", "processed")
DEFAULT_CACHE = os.path.join(ROOT, "03_Models", "cache", "retrospective")
DEFAULT_OUT = os.path.join(ROOT, "04_Experiments", "runs", "text_v2")
BASELINES = os.path.join(ROOT, "05_Reports", "final_report", "results", "baselines.json")
TIER1 = os.path.join(ROOT, "05_Reports", "final_report", "results", "deep_multimodal.json")

ABLATIONS = {
    "title_description": ("title", "description"),
    "transcript": ("transcript",),
    "text_fields": ("title", "description", "transcript"),
    "text_fields_meta_sched": ("title", "description", "transcript", "meta_sched"),
    "thumbnail_text_fields_meta_sched": (
        "thumbnail", "title", "description", "transcript", "meta_sched",
    ),
    "text_fields_engineered": ("title", "description", "transcript", "engineered"),
    "thumbnail_text_fields_engineered": (
        "thumbnail", "title", "description", "transcript", "engineered",
    ),
}


def _field_block(bundle: EmbeddingSet, include_chunks: bool = False) -> np.ndarray:
    masks = [bundle.masks["field_present"]]
    if include_chunks:
        masks.append(np.log1p(bundle.masks["n_chunks"]))
    return np.column_stack([bundle.values, *masks]).astype(np.float32)


def _references() -> dict:
    with open(BASELINES, encoding="utf-8") as f:
        baselines = json.load(f)
    with open(TIER1, encoding="utf-8") as f:
        tier1 = json.load(f)
    rows = {row["features"]: row for row in baselines["rows"]}
    seed_ids = baselines.get("seed_ids", list(range(baselines["seeds"])))
    return {
        "metadata_schedule_xgboost": {
            **rows["+ schedule"]["auc"]["xgboost"], "seed_ids": seed_ids,
        },
        "full_engineered_xgboost": {
            **rows["+ audio"]["auc"]["xgboost"], "seed_ids": seed_ids,
        },
        "tier1_full_linear": tier1["aggregate"]["thumbnail_text_meta_sched"]["linear_probe"]["auc_roc"],
        "tier1_full_mlp": tier1["aggregate"]["thumbnail_text_meta_sched"]["late_fusion_mlp"]["auc_roc"],
    }


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".text_v2_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--force-embeddings", action="store_true")
    ap.add_argument("--only-embeddings", action="store_true")
    args = ap.parse_args()

    df = load_retrospective(args.data)
    excluded = int(df.label.isna().sum())
    df = df[df.label.notna()].reset_index(drop=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    print(f"loaded {len(df)} labelled videos ({excluded} unlabelled excluded)")
    fields, text_provenance = field_text_embeddings(
        df, args.cache_dir, batch_size=args.batch_size, device=args.device,
        force=args.force_embeddings,
    )
    thumbnail = thumbnail_embeddings(
        df, args.cache_dir, batch_size=args.image_batch_size, device=args.device,
        force=False,
    )
    if args.only_embeddings:
        return

    raw_blocks = {
        "title": _field_block(fields["title"]),
        "description": _field_block(fields["description"]),
        "transcript": _field_block(fields["transcript"], include_chunks=True),
        "thumbnail": np.column_stack([
            thumbnail.values, thumbnail.masks["thumbnail_present"],
        ]).astype(np.float32),
    }
    ablations = [value.strip() for value in args.ablations.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        run = run_named_blocks_seed(
            df, raw_blocks, seed, ABLATIONS, ablations, device=args.device, epochs=args.epochs,
        )
        runs.append(run)
        print(f"seed {seed} split={run['split_sizes']} (test not evaluated)")
        for name, result in run["ablations"].items():
            linear = result["linear_probe"]["val"]["auc_roc"]
            mlp = result["late_fusion_mlp"]["val"]["auc_roc"]
            print(f"  {name:36s} linear={linear:.3f}  MLP={mlp:.3f}")
    aggregate = aggregate_runs(runs)
    references = _references()
    for name in ("metadata_schedule_xgboost", "full_engineered_xgboost"):
        if references[name].get("seed_ids") != seeds:
            raise ValueError(
                f"saved {name} does not match requested seeds {seeds}; "
                f"rerun make_tables.py for those seeds"
            )
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "channel-grouped 60/20/20; nested training-fold selection; validation only; test untouched",
        "n_labelled": int(len(df)), "seeds": seeds,
        "configuration": {"ablations": ablations, "epochs": args.epochs, "batch_size": args.batch_size},
        "text_provenance": text_provenance,
        "field_coverage": {
            field: {"present": int(bundle.masks["field_present"].sum()),
                    "mean_chunks_when_present": float(bundle.masks["n_chunks"][bundle.masks["field_present"] > 0].mean())}
            for field, bundle in fields.items()
        },
        "runs": runs, "aggregate": aggregate, "references": references,
    }
    payload["auc_mean_deltas"] = {
        ablation: {
            model: {name: metrics["auc_roc"]["mean"] - reference["mean"]
                    for name, reference in references.items()}
            for model, metrics in models.items()
        } for ablation, models in aggregate.items()
    }
    path = os.path.join(args.out_dir, "results.json")
    _atomic_json(path, payload)
    print("\nmean ± population SD validation AUC")
    for ablation, models in aggregate.items():
        linear, mlp = models["linear_probe"]["auc_roc"], models["late_fusion_mlp"]["auc_roc"]
        print(f"  {ablation:36s} linear={linear['mean']:.3f}±{linear['std']:.3f}  "
              f"MLP={mlp['mean']:.3f}±{mlp['std']:.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
