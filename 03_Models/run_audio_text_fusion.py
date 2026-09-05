"""Combine engineered audio (tree-friendly, real isolated signal per
run_audio_ablation.py) with frozen field-aware ModernBERT text embeddings
(weak but non-zero signal per run_text_v2.py) in one fusion model.

Motivation: these are two different, weakly-to-moderately-informative signal
sources that have never been combined. Audio's isolated lift (0.632 +/- 0.028,
not yet statistically significant per run_ablation_significance.py) used
XGBoost on tabular eGeMAPS features. Text's best isolated result (0.571,
all three fields via ModernBERT, linear/MLP fusion) used a completely
different model family. This checks whether combining them in the SAME
fusion architecture (ytdiag/fusion.py) does better than either alone,
against the same metadata+schedule and metadata+schedule+audio references.

Deliberately excludes thumbnail/frames: both were already shown to dilute
audio when bundled (run_audio_ablation.py's whole premise), so combining
audio with the *weaker*, still-untested-together signal (text) in isolation
is the fair next step, not a full six-block kitchen sink.

Unrelated to the frozen multimodal finalist (final_model_policy.json, frozen
2026-09-03): purely additive, does not touch the sealed test split.

Example:
  .venv/Scripts/python 03_Models/run_audio_text_fusion.py --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.embed import EmbeddingSet  # noqa: E402
from ytdiag.fusion import aggregate_runs, run_named_blocks_seed  # noqa: E402
from ytdiag.text_v2 import field_text_embeddings  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "02_Data", "processed")
DEFAULT_CACHE = os.path.join(ROOT, "03_Models", "cache", "retrospective")
DEFAULT_OUT = os.path.join(ROOT, "04_Experiments", "runs", "audio_text_fusion")

ABLATIONS = {
    "text_fields_meta_sched": ("title", "description", "transcript", "meta_sched"),
    "audio_meta_sched": ("audio", "meta_sched"),
    "text_fields_audio_meta_sched": ("title", "description", "transcript", "audio", "meta_sched"),
}


def _field_block(bundle: EmbeddingSet, include_chunks: bool = False) -> np.ndarray:
    masks = [bundle.masks["field_present"]]
    if include_chunks:
        masks.append(np.log1p(bundle.masks["n_chunks"]))
    return np.column_stack([bundle.values, *masks]).astype(np.float32)


def _atomic_json(path: str, payload: dict) -> None:
    import tempfile
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".audio_text_fusion_", suffix=".json", dir=os.path.dirname(path))
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
    if args.only_embeddings:
        return

    raw_blocks = {
        "title": _field_block(fields["title"]),
        "description": _field_block(fields["description"]),
        "transcript": _field_block(fields["transcript"], include_chunks=True),
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
            print(f"  {name:32s} linear={linear:.3f}  MLP={mlp:.3f}")

    aggregate = aggregate_runs(runs)
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": (
            "repeated channel-grouped 60/20/20 development splits; nested "
            "training-fold selection; validation only. Post-freeze exploratory "
            "combination, mirrors run_text_v2.py's protocol -- not part of "
            "final_model_policy.json."
        ),
        "n_labelled": int(len(df)), "seeds": seeds,
        "configuration": {"ablations": ablations, "epochs": args.epochs, "batch_size": args.batch_size},
        "text_provenance": text_provenance,
        "runs": runs, "aggregate": aggregate,
        "references": {
            "metadata_schedule_xgboost_tuned": {"mean": 0.619, "std": 0.022},
            "metadata_schedule_audio_xgboost_isolated": {"mean": 0.632, "std": 0.028},
        },
    }
    path = os.path.join(args.out_dir, "results.json")
    _atomic_json(path, payload)
    print("\nmean +/- population SD validation AUC")
    for ablation, models in aggregate.items():
        linear, mlp = models["linear_probe"]["auc_roc"], models["late_fusion_mlp"]["auc_roc"]
        print(f"  {ablation:32s} linear={linear['mean']:.3f}+/-{linear['std']:.3f}  "
              f"MLP={mlp['mean']:.3f}+/-{mlp['std']:.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
