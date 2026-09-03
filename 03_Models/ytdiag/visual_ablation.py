"""Controlled frozen visual-encoder ablations for thumbnails and stored frames.

Each cache records the exact model revision, preprocessing, pooling and source
fingerprint. The helpers only extract representations; evaluation remains in
the shared channel-grouped fusion pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .embed import EmbeddingSet, _load_cache, _save_cache, choose_device


@dataclass(frozen=True)
class VisualSpec:
    """Complete definition of one frozen image representation."""

    name: str
    model: str
    revision: str
    family: str
    resize_shortest_edge: int
    crop_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    framing: str = "center_crop"
    pooling: str = "cls"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

VISUAL_SPECS = {
    "dino_small_center_mean": VisualSpec(
        "dino_small_center_mean", "facebook/dinov2-small",
        "ed25f3a31f01632728cabb09d1542f84ab7b0056", "dino",
        256, 224, IMAGENET_MEAN, IMAGENET_STD, pooling="patch_mean",
    ),
    "dino_small_fit_cls": VisualSpec(
        "dino_small_fit_cls", "facebook/dinov2-small",
        "ed25f3a31f01632728cabb09d1542f84ab7b0056", "dino",
        224, 224, IMAGENET_MEAN, IMAGENET_STD, framing="fit_pad",
    ),
    "dino_base_center_cls": VisualSpec(
        "dino_base_center_cls", "facebook/dinov2-base",
        "f9e44c814b77203eaa57a6bdbbd535f21ede1415", "dino",
        256, 224, IMAGENET_MEAN, IMAGENET_STD,
    ),
    "clip_base_center": VisualSpec(
        "clip_base_center", "openai/clip-vit-base-patch32",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "clip",
        224, 224, CLIP_MEAN, CLIP_STD,
    ),
    "resnet50_center": VisualSpec(
        "resnet50_center", "microsoft/resnet-50",
        "34c2154c194f829b11125337b98c8f5f9965ff19", "resnet",
        232, 224, IMAGENET_MEAN, IMAGENET_STD, pooling="avg",
    ),
}


def preprocess_image(path: str, spec: VisualSpec) -> np.ndarray:
    """Apply the recorded crop or aspect-preserving fit policy."""
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        if spec.framing == "center_crop":
            scale = spec.resize_shortest_edge / min(width, height)
            resized = image.resize(
                (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC,
            )
            width, height = resized.size
            left = (width - spec.crop_size) // 2
            top = (height - spec.crop_size) // 2
            image = resized.crop((left, top, left + spec.crop_size, top + spec.crop_size))
        elif spec.framing == "fit_pad":
            scale = spec.crop_size / max(width, height)
            resized = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.BICUBIC,
            )
            fill = tuple(round(value * 255) for value in spec.mean)
            image = Image.new("RGB", (spec.crop_size, spec.crop_size), fill)
            image.paste(
                resized,
                ((spec.crop_size - resized.width) // 2, (spec.crop_size - resized.height) // 2),
            )
        else:
            raise ValueError(f"unknown framing policy: {spec.framing}")
        array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(spec.mean, dtype=np.float32)
    std = np.asarray(spec.std, dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)


def _source_fingerprint(rows: Sequence[Sequence[str | None]]) -> str:
    digest = hashlib.sha256()
    for paths in rows:
        for path in paths:
            if path and os.path.exists(path):
                stat = os.stat(path)
                digest.update(os.path.abspath(path).encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            else:
                digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_provenance(spec: VisualSpec, aggregation: str, fingerprint: str) -> dict[str, Any]:
    """Build JSON-stable provenance so a written cache can be read again."""
    return {
        "schema": 1,
        "kind": "visual_ablation",
        # Normalise tuples before comparing with a payload read from JSON.
        "spec": json.loads(json.dumps(asdict(spec))),
        "aggregation": aggregation,
        "source_fingerprint": fingerprint,
    }


def _load_model(spec: VisualSpec, device: str):
    if spec.family == "dino":
        from transformers import AutoModel

        return AutoModel.from_pretrained(spec.model, revision=spec.revision).to(device).eval()
    if spec.family == "clip":
        from transformers import CLIPVisionModelWithProjection

        return CLIPVisionModelWithProjection.from_pretrained(
            spec.model, revision=spec.revision,
        ).to(device).eval()
    if spec.family == "resnet":
        from transformers import ResNetModel

        return ResNetModel.from_pretrained(spec.model, revision=spec.revision).to(device).eval()
    raise ValueError(f"unknown visual model family: {spec.family}")


def _encode(model, pixels, spec: VisualSpec):
    output = model(pixel_values=pixels)
    if spec.family == "dino":
        if spec.pooling == "cls":
            return output.last_hidden_state[:, 0]
        if spec.pooling == "patch_mean":
            return output.last_hidden_state[:, 1:].mean(dim=1)
    elif spec.family == "clip":
        return output.image_embeds
    elif spec.family == "resnet":
        return output.pooler_output.flatten(1)
    raise ValueError(f"unsupported family/pooling: {spec.family}/{spec.pooling}")


def _extract(
    video_ids: np.ndarray,
    paths_by_video: Sequence[Sequence[str | None]],
    spec: VisualSpec,
    cache_path: str,
    aggregation: str,
    batch_size: int,
    device: str,
    force: bool,
    progress: Callable[[str], None],
) -> EmbeddingSet:
    provenance = _cache_provenance(
        spec, aggregation, _source_fingerprint(paths_by_video),
    )
    if not force:
        cached = _load_cache(cache_path, video_ids, provenance)
        if cached is not None:
            progress(f"{spec.name}/{aggregation}: cache hit")
            return cached

    import torch

    resolved_device = choose_device(device)
    progress(f"{spec.name}/{aggregation}: loading {spec.model} on {resolved_device}")
    model = _load_model(spec, resolved_device)
    valid = [
        (row, path)
        for row, paths in enumerate(paths_by_video)
        for path in paths
        if path and os.path.exists(path)
    ]
    sums: np.ndarray | None = None
    counts = np.zeros(len(video_ids), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(valid), batch_size):
            batch = valid[start:start + batch_size]
            pixels = torch.from_numpy(np.stack([
                preprocess_image(path, spec) for _, path in batch
            ])).to(resolved_device)
            encoded = _encode(model, pixels, spec).float().cpu().numpy()
            if sums is None:
                sums = np.zeros((len(video_ids), encoded.shape[1]), dtype=np.float32)
            for (row, _), vector in zip(batch, encoded):
                sums[row] += vector
                counts[row] += 1
            if start == 0 or start + batch_size >= len(valid) or (start // batch_size + 1) % 50 == 0:
                progress(f"{spec.name}/{aggregation}: {min(start + batch_size, len(valid))}/{len(valid)} images")
    if sums is None:
        raise ValueError("no readable images found")
    present = counts > 0
    sums[present] /= counts[present, None]
    bundle = EmbeddingSet(
        video_ids.astype(str), sums,
        {"visual_present": present.astype(np.float32), "images_aggregated": counts},
        provenance,
    )
    _save_cache(cache_path, bundle)
    del model
    if resolved_device == "mps":
        torch.mps.empty_cache()
    return bundle


def thumbnail_embeddings_variant(
    df: pd.DataFrame,
    cache_dir: str,
    spec_name: str,
    batch_size: int = 32,
    device: str = "auto",
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> EmbeddingSet:
    """Extract one representation per stored thumbnail."""
    spec = VISUAL_SPECS[spec_name]
    paths = [[value if isinstance(value, str) else None] for value in df.asset__thumbnail_path]
    return _extract(
        df.video_id.astype(str).to_numpy(), paths, spec,
        os.path.join(cache_dir, f"visual_{spec.name}_thumbnail.npz"),
        "single_thumbnail", batch_size, device, force, progress,
    )


def frame_mean_embeddings(
    df: pd.DataFrame,
    cache_dir: str,
    batch_size: int = 64,
    device: str = "auto",
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> EmbeddingSet:
    """Mean-pool frozen DINOv2-small CLS vectors across each videos stored frames."""
    spec = VisualSpec(
        "dino_small_center_cls", "facebook/dinov2-small",
        "ed25f3a31f01632728cabb09d1542f84ab7b0056", "dino",
        256, 224, IMAGENET_MEAN, IMAGENET_STD,
    )
    paths = []
    for value in df.asset__frames_dir:
        if not isinstance(value, str) or not os.path.isdir(value):
            paths.append([])
            continue
        paths.append([
            os.path.join(value, name) for name in sorted(os.listdir(value))
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
    return _extract(
        df.video_id.astype(str).to_numpy(), paths, spec,
        os.path.join(cache_dir, "visual_dino_small_center_cls_frames_mean.npz"),
        "mean_over_stored_frames", batch_size, device, force, progress,
    )
