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
    "frames_dino_base_mean", "thumbnail_frames_dino_base",
    "frames_clip_base_mean", "thumbnail_frames_clip",
    "frames_resnet50_mean", "thumbnail_frames_resnet50",
    "thumbnail_text_fields_meta_sched", "clip_text_fields_meta_sched",
    "frames_text_fields_meta_sched", "thumbnail_frames_text_fields_meta_sched",
    "dino_base_thumbnail_frames_text_fields_meta_sched",
    "clip_thumbnail_frames_text_fields_meta_sched",
    "resnet50_thumbnail_frames_text_fields_meta_sched",
)

TEXT_FUSIONS = {
    "thumbnail_text_fields_meta_sched",
    "clip_text_fields_meta_sched",
    "frames_text_fields_meta_sched",
    "thumbnail_frames_text_fields_meta_sched",
    "dino_base_thumbnail_frames_text_fields_meta_sched",
    "clip_thumbnail_frames_text_fields_meta_sched",
    "resnet50_thumbnail_frames_text_fields_meta_sched",
}

BACKBONES = {
    "dino_small": ("dino_small_center_cls", "frames_dino_small_mean"),
    "dino_base": ("dino_base_center_cls", "frames_dino_base_mean"),
    "clip": ("clip_base_center", "frames_clip_base_mean"),
    "resnet50": ("resnet50_center", "frames_resnet50_mean"),
}

COMPOSITE_ABLATIONS = {
    "thumbnail_frames_dino_small": ("dino_small_center_cls", "frames_dino_small_mean"),
    "thumbnail_frames_dino_base": ("dino_base_center_cls", "frames_dino_base_mean"),
    "thumbnail_frames_clip": ("clip_base_center", "frames_clip_base_mean"),
    "thumbnail_frames_resnet50": ("resnet50_center", "frames_resnet50_mean"),
}

FULL_FUSIONS = {
    "thumbnail_frames_text_fields_meta_sched":
        ("dino_small_center_cls", "frames_dino_small_mean"),
    "dino_base_thumbnail_frames_text_fields_meta_sched":
        ("dino_base_center_cls", "frames_dino_base_mean"),
    "clip_thumbnail_frames_text_fields_meta_sched":
        ("clip_base_center", "frames_clip_base_mean"),
    "resnet50_thumbnail_frames_text_fields_meta_sched":
        ("resnet50_center", "frames_resnet50_mean"),
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
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    requested = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = set(requested) - set(DEFAULT_VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    requested_set = set(requested)
    needed_backbones = set()
    for backbone, (thumbnail_key, frame_key) in BACKBONES.items():
        related = {thumbnail_key, frame_key, f"thumbnail_frames_{backbone}"}
        related.update(name for name, pair in FULL_FUSIONS.items() if frame_key in pair)
        if related & requested_set:
            needed_backbones.add(backbone)
    if "thumbnail_text_fields_meta_sched" in requested_set:
        needed_backbones.add("dino_small")
    if "clip_text_fields_meta_sched" in requested_set:
        needed_backbones.add("clip")
    frame_requested = any(
        BACKBONES[backbone][1] in requested_set
        or f"thumbnail_frames_{backbone}" in requested_set
        or any(name in requested_set and BACKBONES[backbone][1] in pair
               for name, pair in FULL_FUSIONS.items())
        for backbone in needed_backbones
    )
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
    representation_requests.update(
        BACKBONES[backbone][0] for backbone in needed_backbones
    )
    for name in representation_requests:
        if name == "dino_small_center_cls":
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
        for backbone in sorted(needed_backbones):
            thumbnail_key, frame_key = BACKBONES[backbone]
            needs_frames = (
                frame_key in requested_set
                or f"thumbnail_frames_{backbone}" in requested_set
                or any(name in requested_set and frame_key in pair
                       for name, pair in FULL_FUSIONS.items())
            )
            if not needs_frames:
                continue
            frames = frame_mean_embeddings(
                df, args.cache_dir, spec_name=thumbnail_key,
                batch_size=args.frame_batch_size, device=args.device,
                force=args.force_embeddings,
            )
            blocks[frame_key] = np.column_stack([
                frames.values, frames.masks["visual_present"],
                np.log1p(frames.masks["images_aggregated"]),
            ]).astype(np.float32)
            provenance[frame_key] = frames.provenance

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

    composite_names = set(COMPOSITE_ABLATIONS) | set(FULL_FUSIONS)
    composite_names.update({"thumbnail_text_fields_meta_sched", "clip_text_fields_meta_sched",
                            "frames_text_fields_meta_sched"})
    ablation_map = {name: (name,) for name in requested if name not in composite_names}
    for name, pair in COMPOSITE_ABLATIONS.items():
        if name in requested_set:
            ablation_map[name] = pair
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
    for name, pair in FULL_FUSIONS.items():
        if name in requested_set:
            ablation_map[name] = (*pair, *shared_text, "meta_sched")
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
