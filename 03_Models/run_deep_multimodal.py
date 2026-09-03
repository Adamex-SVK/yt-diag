"""Frozen DINOv2 + ModernBERT ablations and a small late-fusion classifier.

The command reports validation results over channel-grouped splits.  It has no
test-evaluation option by design: the project test set remains sealed until the
architecture and report protocol are frozen.

Example:
  .venv/bin/python 03_Models/run_deep_multimodal.py --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.embed import text_embeddings, thumbnail_embeddings  # noqa: E402
from ytdiag.fusion import ABLATIONS, aggregate_runs, run_seed  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "02_Data", "processed")
DEFAULT_CACHE = os.path.join(ROOT, "03_Models", "cache", "retrospective")
DEFAULT_OUT = os.path.join(ROOT, "04_Experiments", "runs", "deep_multimodal")
DEFAULT_BASELINES = os.path.join(ROOT, "05_Reports", "final_report", "results", "baselines.json")


def _baseline_comparison(path: str, aggregate: dict) -> dict:
    if not os.path.exists(path):
        return {"status": "baseline summary not found", "path": path}
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    by_name = {row["features"]: row for row in saved["rows"]}
    references = {
        "metadata_schedule_xgboost": by_name["+ schedule"]["auc"]["xgboost"],
        "full_engineered_xgboost": by_name["+ audio"]["auc"]["xgboost"],
    }
    deltas = {}
    for ablation, models in aggregate.items():
        deltas[ablation] = {}
        for model, metrics in models.items():
            auc = metrics["auc_roc"]["mean"]
            deltas[ablation][model] = {
                name: float(auc - reference["mean"]) for name, reference in references.items()
            }
    return {
        "path": os.path.relpath(path, ROOT),
        "protocol": saved.get("protocol"),
        "references": references,
        "auc_mean_deltas": deltas,
    }


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".results_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _print_run(run: dict) -> None:
    print(f"seed {run['seed']} split={run['split_sizes']} (test not evaluated)")
    for name, result in run["ablations"].items():
        linear = result["linear_probe"]["val"]["auc_roc"]
        mlp = result["late_fusion_mlp"]["val"]["auc_roc"]
        print(f"  {name:28s} linear={linear:.3f}  MLP={mlp:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ablations", default=",".join(ABLATIONS),
                    help=f"comma-separated subset of: {','.join(ABLATIONS)}")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--image-batch-size", type=int, default=32)
    ap.add_argument("--text-batch-size", type=int, default=8)
    ap.add_argument("--text-max-length", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--force-embeddings", action="store_true")
    ap.add_argument("--only-embeddings", action="store_true")
    args = ap.parse_args()

    df = load_retrospective(args.data)
    excluded = int(df.label.isna().sum())
    df = df[df.label.notna()].reset_index(drop=True)
    print(f"loaded {len(df)} labelled videos ({excluded} unlabelled excluded)")
    thumbnail = thumbnail_embeddings(
        df, args.cache_dir, batch_size=args.image_batch_size, device=args.device,
        force=args.force_embeddings,
    )
    text = text_embeddings(
        df, args.cache_dir, max_length=args.text_max_length, batch_size=args.text_batch_size,
        device=args.device, force=args.force_embeddings,
    )
    if args.only_embeddings:
        return

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    ablations = [value.strip() for value in args.ablations.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        run = run_seed(
            df, thumbnail, text, seed=seed, ablations=ablations,
            device=args.device, epochs=args.epochs,
        )
        runs.append(run)
        _print_run(run)
    aggregate = aggregate_runs(runs)
    results = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "channel-grouped 60/20/20; validation only; test untouched",
        "n_labelled": int(len(df)),
        "seeds": seeds,
        "configuration": {
            "ablations": ablations, "epochs": args.epochs,
            "text_max_length": args.text_max_length,
            "image_batch_size": args.image_batch_size, "text_batch_size": args.text_batch_size,
        },
        "embedding_provenance": {
            "thumbnail": thumbnail.provenance, "text": text.provenance,
        },
        "runs": runs,
        "aggregate": aggregate,
    }
    results["baseline_comparison"] = _baseline_comparison(DEFAULT_BASELINES, aggregate)
    path = os.path.join(args.out_dir, "results.json")
    _atomic_json(path, results)
    print("\nmean ± population SD validation AUC")
    for ablation, models in aggregate.items():
        linear = models["linear_probe"]["auc_roc"]
        mlp = models["late_fusion_mlp"]["auc_roc"]
        print(f"  {ablation:28s} linear={linear['mean']:.3f}±{linear['std']:.3f}  "
              f"MLP={mlp['mean']:.3f}±{mlp['std']:.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
