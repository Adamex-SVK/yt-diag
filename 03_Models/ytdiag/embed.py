"""Frozen thumbnail and text encoders with provenance-checked disk caches.

The encoders are intentionally feature extractors, not fine-tuned models: with
only 1,854 labelled videos an end-to-end transformer would mostly learn the
training channels.  Every cache records immutable model revisions, preprocessing
parameters, ordered video ids, and a fingerprint of the source assets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image


DINO_MODEL = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
TEXT_MODEL = "answerdotai/ModernBERT-base"
TEXT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
CACHE_SCHEMA = 1

# Pinned facebook/dinov2-small preprocessor_config.json at DINO_REVISION.
DINO_PREPROCESS = {
    "convert_rgb": True,
    "resize_shortest_edge": 256,
    "center_crop": [224, 224],
    "resample": "bicubic",
    "rescale_factor": 1.0 / 255.0,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass(frozen=True)
class EmbeddingSet:
    """One frozen representation per canonical-table row."""

    video_ids: np.ndarray
    values: np.ndarray
    masks: dict[str, np.ndarray]
    provenance: dict[str, Any]


def choose_device(requested: str = "auto") -> str:
    """Resolve auto -> CUDA, MPS, or CPU without importing torch at module load."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compose_text(row: pd.Series) -> tuple[str, dict[str, float]]:
    """Build the encoder document and explicit source-availability masks.

    Title, description, and the *cleaned* transcript path selected by the
    retrospective adapter are labelled so the encoder can distinguish fields.
    A missing transcript is not represented by fabricated words.
    """
    title = row.get("asset__title")
    description = row.get("asset__description")
    title = title.strip() if isinstance(title, str) else ""
    description = description.strip() if isinstance(description, str) else ""
    transcript = ""
    transcript_path = row.get("asset__transcript_path")
    if isinstance(transcript_path, str) and transcript_path:
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"transcript path does not exist: {transcript_path}")
        with open(transcript_path, encoding="utf-8", errors="strict") as f:
            transcript = f.read().strip()
    fields = []
    if title:
        fields.append(f"Title: {title}")
    if transcript:
        fields.append(f"Transcript: {transcript}")
    if description:
        fields.append(f"Description: {description}")
    quality = row.get("asset__transcript_usable")
    quality_known = float(pd.notna(quality))
    return "\n\n".join(fields), {
        "title_present": float(bool(title)),
        "description_present": float(bool(description)),
        "transcript_present": float(bool(transcript)),
        "transcript_usable": float(quality) if quality_known else float(bool(transcript)),
        "transcript_quality_known": quality_known,
    }


def _sha256_file(path: str, digest: "hashlib._Hash") -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)


def _thumbnail_fingerprint(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for _, row in df.iterrows():
        digest.update(str(row.video_id).encode())
        path = row.get("asset__thumbnail_path")
        if isinstance(path, str) and path and os.path.exists(path):
            _sha256_file(path, digest)
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _text_inputs(df: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray], str]:
    texts: list[str] = []
    raw_masks: dict[str, list[float]] = {
        "title_present": [], "description_present": [], "transcript_present": [],
        "transcript_usable": [], "transcript_quality_known": [],
    }
    digest = hashlib.sha256()
    for _, row in df.iterrows():
        text, masks = compose_text(row)
        texts.append(text)
        digest.update(str(row.video_id).encode())
        digest.update(text.encode("utf-8"))
        for name, value in masks.items():
            raw_masks[name].append(value)
            digest.update(bytes([int(value)]))
    return texts, {k: np.asarray(v, dtype=np.float32) for k, v in raw_masks.items()}, digest.hexdigest()


def _save_cache(path: str, bundle: EmbeddingSet) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "video_ids": np.asarray(bundle.video_ids, dtype=str),
        "values": np.asarray(bundle.values, dtype=np.float32),
        "provenance": np.asarray(json.dumps(bundle.provenance, sort_keys=True)),
    }
    payload.update({f"mask__{k}": np.asarray(v, dtype=np.float32) for k, v in bundle.masks.items()})
    fd, tmp = tempfile.mkstemp(prefix=".embeddings_", suffix=".npz", dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _load_cache(path: str, video_ids: np.ndarray, expected: dict[str, Any]) -> EmbeddingSet | None:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_ids = data["video_ids"].astype(str)
            provenance = json.loads(str(data["provenance"]))
            if not np.array_equal(cached_ids, video_ids.astype(str)) or provenance != expected:
                return None
            masks = {k.removeprefix("mask__"): data[k].astype(np.float32) for k in data.files if k.startswith("mask__")}
            return EmbeddingSet(cached_ids, data["values"].astype(np.float32), masks, provenance)
    except (KeyError, ValueError, json.JSONDecodeError, OSError):
        return None


def _batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _preprocess_dino(path: str) -> np.ndarray:
    """Exact pinned DINOv2 resize/crop/rescale/normalise without torchvision."""
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        width, height = image.size
        scale = DINO_PREPROCESS["resize_shortest_edge"] / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
        width, height = resized.size
        crop_w, crop_h = DINO_PREPROCESS["center_crop"]
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        image = resized.crop((left, top, left + crop_w, top + crop_h))
        array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(DINO_PREPROCESS["mean"], dtype=np.float32)
    std = np.asarray(DINO_PREPROCESS["std"], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1)


def thumbnail_embeddings(
    df: pd.DataFrame,
    cache_dir: str,
    batch_size: int = 32,
    device: str = "auto",
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> EmbeddingSet:
    """Return frozen DINOv2 CLS embeddings in canonical-table row order."""
    video_ids = df.video_id.astype(str).to_numpy()
    fingerprint = _thumbnail_fingerprint(df)
    provenance = {
        "schema": CACHE_SCHEMA, "kind": "thumbnail", "model": DINO_MODEL,
        "revision": DINO_REVISION, "pooling": "cls", "preprocess": DINO_PREPROCESS,
        "source_fingerprint": fingerprint,
    }
    cache_path = os.path.join(cache_dir, "thumbnail_dinov2_small.npz")
    if not force:
        cached = _load_cache(cache_path, video_ids, provenance)
        if cached is not None:
            progress(f"thumbnail embeddings: cache hit ({cache_path})")
            return cached

    import torch
    from transformers import AutoModel

    resolved_device = choose_device(device)
    progress(f"thumbnail embeddings: loading {DINO_MODEL}@{DINO_REVISION[:8]} on {resolved_device}")
    model = AutoModel.from_pretrained(DINO_MODEL, revision=DINO_REVISION).to(resolved_device).eval()
    values = np.zeros((len(df), int(model.config.hidden_size)), dtype=np.float32)
    present = np.zeros(len(df), dtype=np.float32)
    paths = [p if isinstance(p, str) and p and os.path.exists(p) else None for p in df.asset__thumbnail_path]
    valid = [(i, path) for i, path in enumerate(paths) if path is not None]
    for batch_no, (_, batch) in enumerate(_batched(valid, batch_size), start=1):
        indices, batch_paths = zip(*batch)
        pixels = torch.from_numpy(np.stack([_preprocess_dino(p) for p in batch_paths])).to(resolved_device)
        with torch.inference_mode():
            encoded = model(pixel_values=pixels).last_hidden_state[:, 0]
        values[list(indices)] = encoded.float().cpu().numpy()
        present[list(indices)] = 1.0
        if batch_no % 10 == 0 or batch_no * batch_size >= len(valid):
            progress(f"thumbnail embeddings: {min(batch_no * batch_size, len(valid))}/{len(valid)}")
    bundle = EmbeddingSet(video_ids, values, {"thumbnail_present": present}, provenance)
    _save_cache(cache_path, bundle)
    return bundle


def text_embeddings(
    df: pd.DataFrame,
    cache_dir: str,
    max_length: int = 1024,
    batch_size: int = 8,
    device: str = "auto",
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> EmbeddingSet:
    """Return frozen masked-mean ModernBERT embeddings in table row order."""
    video_ids = df.video_id.astype(str).to_numpy()
    texts, masks, fingerprint = _text_inputs(df)
    provenance = {
        "schema": CACHE_SCHEMA, "kind": "text", "model": TEXT_MODEL,
        "revision": TEXT_REVISION, "pooling": "masked_mean", "max_length": int(max_length),
        "attention_implementation": "sdpa",
        # Title and spoken content are deliberately ahead of description when
        # the combined document exceeds the fixed token budget.
        "field_order": ["title", "cleaned_transcript", "description"],
        "source_fingerprint": fingerprint,
    }
    cache_path = os.path.join(cache_dir, f"text_modernbert_base_{max_length}.npz")
    partial_path = os.path.join(cache_dir, f"text_modernbert_base_{max_length}.partial.npz")
    if not force:
        cached = _load_cache(cache_path, video_ids, provenance)
        if cached is not None:
            progress(f"text embeddings: cache hit ({cache_path})")
            return cached

    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = choose_device(device)
    progress(f"text embeddings: loading {TEXT_MODEL}@{TEXT_REVISION[:8]} on {resolved_device}")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
    model = AutoModel.from_pretrained(
        TEXT_MODEL, revision=TEXT_REVISION, attn_implementation="sdpa"
    ).to(resolved_device).eval()
    values = np.zeros((len(df), int(model.config.hidden_size)), dtype=np.float32)
    complete = np.zeros(len(df), dtype=np.float32)
    if not force:
        partial = _load_cache(partial_path, video_ids, provenance)
        if partial is not None and "row_complete" in partial.masks and partial.values.shape == values.shape:
            values = partial.values
            complete = partial.masks["row_complete"]
            progress(f"text embeddings: resuming {int(complete.sum())}/{len(df)} completed rows")

    pending = [(index, text) for index, text in enumerate(texts) if not complete[index]]

    def save_partial() -> None:
        _save_cache(partial_path, EmbeddingSet(
            video_ids, values, {**masks, "row_complete": complete}, provenance
        ))

    try:
        for batch_no, (_, indexed_batch) in enumerate(_batched(pending, batch_size), start=1):
            indices, batch = zip(*indexed_batch)
            tokens = tokenizer(list(batch), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {k: v.to(resolved_device) for k, v in tokens.items()}
            with torch.inference_mode():
                hidden = model(**tokens).last_hidden_state
                attention = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1)
            values[list(indices)] = pooled.float().cpu().numpy()
            complete[list(indices)] = 1.0
            if batch_no % 10 == 0 or int(complete.sum()) == len(texts):
                save_partial()
                progress(f"text embeddings: {int(complete.sum())}/{len(texts)}")
    except BaseException:
        save_partial()
        raise
    bundle = EmbeddingSet(video_ids, values, masks, provenance)
    _save_cache(cache_path, bundle)
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    return bundle
