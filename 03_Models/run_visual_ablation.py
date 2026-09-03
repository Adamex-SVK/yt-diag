"""Compare controlled frozen visual representations on grouped validation.

The script has no test-evaluation path. It compares model capacity, framing,
pooling, a conventional CNN, language-supervised vision, and stored-frame
aggregation while reusing the exact five outer splits used by other models.
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
from ytdiag.embed import thumbnail_embeddings  # noqa: E402
from ytdiag.fusion import aggregate_runs, run_named_blocks_seed  # noqa: E402
from ytdiag.text_v2 import field_text_embeddings  # noqa: E402
from ytdiag.visual_ablation import (  # noqa: E402
    VISUAL_SPECS, frame_mean_embeddings, thumbnail_embeddings_variant,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(ROOT, "02_Data", "processed")
DEFAULT_CACHE = os.path.join(ROOT, "03_Models", "cache", "retrospective")
DEFAULT_OUT = os.path.join(ROOT, "04_Experiments", "runs", "visual_ablation")

DEFAULT_VARIANTS = (
    "dino_small_center_cls", "dino_small_center_mean", "dino_small_fit_cls",
    "dino_base_center_cls", "clip_base_center", "resnet50_center",
    "frames_dino_small_mean", "thumbnail_frames_dino_small",
    "thumbnail_text_fields_meta_sched", "clip_text_fields_meta_sched",
    "frames_text_fields_meta_sched", "thumbnail_frames_text_fields_meta_sched",
)

TEXT_FUSIONS = {
    "thumbnail_text_fields_meta_sched",
    "clip_text_fields_meta_sched",
    "frames_text_fields_meta_sched",
    "thumbnail_frames_text_fields_meta_sched",
}


def _field_block(bundle, include_chunks: bool = False) -> np.ndarray:
    masks = [bundle.masks["field_present"]]
    if include_chunks:
        masks.append(np.log1p(bundle.masks["n_chunks"]))
    return np.column_stack([bundle.values, *masks]).astype(np.float32)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".visual_", suffix=".json", dir=os.path.dirname(path))
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
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = set(requested) - set(DEFAULT_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    frame_requested = bool({
        "frames_dino_small_mean", "thumbnail_frames_dino_small",
        "frames_text_fields_meta_sched", "thumbnail_frames_text_fields_meta_sched",
    } & set(requested))
    text_requested = bool(TEXT_FUSIONS & set(requested))

    df = load_retrospective(args.data)
    df = df[df.label.notna()].reset_index(drop=True)
    current = thumbnail_embeddings(df, args.cache_dir, batch_size=args.batch_size, device=args.device)
    blocks = {
        "dino_small_center_cls": np.column_stack([
            current.values, current.masks["thumbnail_present"],
        ]).astype(np.float32),
    }
    provenance = {"dino_small_center_cls": current.provenance}
    representation_requests = set(requested) & set(VISUAL_SPECS)
    if "clip_text_fields_meta_sched" in requested:
        representation_requests.add("clip_base_center")
    for name in representation_requests:
        if name in {"dino_small_center_cls", "frames_dino_small_mean", "thumbnail_frames_dino_small"}:
            continue
        bundle = thumbnail_embeddings_variant(
            df, args.cache_dir, name, batch_size=args.batch_size, device=args.device,
            force=args.force_embeddings,
        )
        blocks[name] = np.column_stack([
            bundle.values, bundle.masks["visual_present"],
        ]).astype(np.float32)
        provenance[name] = bundle.provenance
    if frame_requested:
        frames = frame_mean_embeddings(
            df, args.cache_dir, batch_size=args.frame_batch_size, device=args.device,
            force=args.force_embeddings,
        )
        blocks["frames_dino_small_mean"] = np.column_stack([
            frames.values, frames.masks["visual_present"],
            np.log1p(frames.masks["images_aggregated"]),
        ]).astype(np.float32)
        provenance["frames_dino_small_mean"] = frames.provenance

    if text_requested:
        fields, text_provenance = field_text_embeddings(
            df, args.cache_dir, batch_size=16, device=args.device, force=False,
        )
        blocks.update({
            "title": _field_block(fields["title"]),
            "description": _field_block(fields["description"]),
            "transcript": _field_block(fields["transcript"], include_chunks=True),
        })
        provenance["text_fields"] = text_provenance

    ablation_map = {
        name: (name,) for name in requested if name != "thumbnail_frames_dino_small"
    }
    if "thumbnail_frames_dino_small" in requested:
        ablation_map["thumbnail_frames_dino_small"] = (
            "dino_small_center_cls", "frames_dino_small_mean",
        )
    shared_text = ("title", "description", "transcript")
    if "thumbnail_text_fields_meta_sched" in requested:
        ablation_map["thumbnail_text_fields_meta_sched"] = (
            "dino_small_center_cls", *shared_text, "meta_sched",
        )
    if "clip_text_fields_meta_sched" in requested:
        ablation_map["clip_text_fields_meta_sched"] = (
            "clip_base_center", *shared_text, "meta_sched",
        )
    if "frames_text_fields_meta_sched" in requested:
        ablation_map["frames_text_fields_meta_sched"] = (
            "frames_dino_small_mean", *shared_text, "meta_sched",
        )
    if "thumbnail_frames_text_fields_meta_sched" in requested:
        ablation_map["thumbnail_frames_text_fields_meta_sched"] = (
            "dino_small_center_cls", "frames_dino_small_mean", *shared_text, "meta_sched",
        )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        run = run_named_blocks_seed(
            df, blocks, seed, ablation_map, requested, device=args.device, epochs=args.epochs,
        )
        runs.append(run)
        print(f"seed {seed} (test not evaluated)")
        for name, result in run["ablations"].items():
            linear = result["linear_probe"]["val"]["auc_roc"]
            mlp = result["late_fusion_mlp"]["val"]["auc_roc"]
            print(f"  {name:36s} linear={linear:.3f} MLP={mlp:.3f}")

    aggregate = aggregate_runs(runs)
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "channel-grouped 60/20/20; nested training selection; validation only; test untouched",
        "n_labelled": int(len(df)), "seeds": seeds, "variants": requested,
        "embedding_provenance": provenance, "runs": runs, "aggregate": aggregate,
    }
    path = os.path.join(args.out_dir, "results.json")
    _atomic_json(path, payload)
    print("\nmean +/- population SD validation AUC")
    for name, models in aggregate.items():
        linear = models["linear_probe"]["auc_roc"]
        mlp = models["late_fusion_mlp"]["auc_roc"]
        print(f"  {name:36s} linear={linear['mean']:.3f}+/-{linear['std']:.3f} "
              f"MLP={mlp['mean']:.3f}+/-{mlp['std']:.3f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
