"""Field-aware, embedding-trained text representations for the Tier-2 experiment.

Title, description, and quality-gated transcript are encoded separately with
Nomic's ModernBERT embedding model. Long transcripts are represented by up to
four evenly distributed overlapping chunks rather than a single leading
truncation. Encoder weights remain frozen.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from .embed import EmbeddingSet, _batched, _load_cache, _save_cache, choose_device


TEXT_V2_MODEL = "nomic-ai/modernbert-embed-base"
TEXT_V2_REVISION = "d556a88e332558790b210f7bdbe87da2fa94a8d8"
TEXT_V2_DIM = 256
TEXT_V2_CONFIG = {
    "prefix": "search_document: ",
    "pooling": "attention_masked_mean_then_l2",
    "matryoshka_dim": TEXT_V2_DIM,
    "title": {"max_length": 128, "overlap": 0, "max_chunks": 1},
    "description": {"max_length": 384, "overlap": 0, "max_chunks": 1},
    "transcript": {"max_length": 384, "overlap": 64, "max_chunks": 4},
}


def _field_value(row: pd.Series, field: str) -> str:
    if field == "transcript":
        path = row.get("asset__transcript_path")
        if not isinstance(path, str) or not path:
            return ""
        if not os.path.exists(path):
            raise FileNotFoundError(f"quality-gated transcript path does not exist: {path}")
        with open(path, encoding="utf-8", errors="strict") as f:
            return f.read().strip()
    value = row.get(f"asset__{field}")
    return value.strip() if isinstance(value, str) else ""


def _evenly_spaced(values: list[int], maximum: int) -> list[int]:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
    return [values[index] for index in indices]


def _field_chunks(tokenizer: Any, field: str, text: str) -> list[list[int]]:
    """Token ids with prefix/special tokens under the frozen field policy."""
    if not text:
        return []
    config = TEXT_V2_CONFIG[field]
    def encode(value: str) -> list[int]:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            # Tokenize the raw long document without the high-level wrapper's
            # misleading >8192 warning; only <=384-token chunks reach the model.
            return backend.encode(value, add_special_tokens=False).ids
        return tokenizer.encode(value, add_special_tokens=False)

    prefix = encode(f"{TEXT_V2_CONFIG['prefix']}{field}: ")
    content = encode(text)
    capacity = int(config["max_length"]) - tokenizer.num_special_tokens_to_add(False) - len(prefix)
    if capacity <= 0:
        raise ValueError(f"{field} token budget is smaller than its prefix")
    overlap = int(config["overlap"])
    step = capacity - overlap
    if step <= 0:
        raise ValueError(f"{field} overlap must be smaller than content capacity")
    last = max(0, len(content) - capacity)
    starts = list(range(0, last + 1, step)) or [0]
    if starts[-1] != last:
        starts.append(last)
    starts = _evenly_spaced(starts, int(config["max_chunks"]))
    chunks = [prefix + content[start:start + capacity] for start in starts]
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return [tokenizer.build_inputs_with_special_tokens(chunk) for chunk in chunks]
    # transformers 5's TokenizersBackend removed that compatibility method;
    # this pinned ModernBERT tokenizer uses the standard CLS ... SEP pair.
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("pinned tokenizer has no CLS/SEP special-token ids")
    return [[tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id] for chunk in chunks]


def _units(df: pd.DataFrame, tokenizer: Any) -> tuple[list[str], list[list[int]], list[tuple[int, str]], str]:
    unit_ids: list[str] = []
    token_ids: list[list[int]] = []
    owners: list[tuple[int, str]] = []
    digest = hashlib.sha256()
    digest.update(repr(TEXT_V2_CONFIG).encode())
    for row_index, (_, row) in enumerate(df.iterrows()):
        for field in ("title", "description", "transcript"):
            chunks = _field_chunks(tokenizer, field, _field_value(row, field))
            for chunk_index, chunk in enumerate(chunks):
                unit_id = f"{row.video_id}::{field}::{chunk_index}"
                unit_ids.append(unit_id)
                token_ids.append(chunk)
                owners.append((row_index, field))
                digest.update(unit_id.encode())
                digest.update(np.asarray(chunk, dtype=np.int32).tobytes())
    return unit_ids, token_ids, owners, digest.hexdigest()


def _aggregate_fields(
    video_ids: np.ndarray,
    unit_values: np.ndarray,
    owners: list[tuple[int, str]],
) -> dict[str, EmbeddingSet]:
    fields: dict[str, EmbeddingSet] = {}
    for field in ("title", "description", "transcript"):
        values = np.zeros((len(video_ids), TEXT_V2_DIM), dtype=np.float32)
        counts = np.zeros(len(video_ids), dtype=np.float32)
        for unit_index, (row_index, owner_field) in enumerate(owners):
            if owner_field == field:
                values[row_index] += unit_values[unit_index]
                counts[row_index] += 1
        present = counts > 0
        values[present] /= counts[present, None]
        norms = np.linalg.norm(values[present], axis=1, keepdims=True).clip(min=1e-12)
        values[present] /= norms
        fields[field] = EmbeddingSet(
            video_ids, values,
            {"field_present": present.astype(np.float32), "n_chunks": counts},
            {"kind": f"text_v2_{field}", "model": TEXT_V2_MODEL, "revision": TEXT_V2_REVISION,
             "config": TEXT_V2_CONFIG[field]},
        )
    return fields


def field_text_embeddings(
    df: pd.DataFrame,
    cache_dir: str,
    batch_size: int = 16,
    device: str = "auto",
    force: bool = False,
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, EmbeddingSet], dict[str, Any]]:
    """Encode and cache field/chunk units, returning one table-aligned set per field."""
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEXT_V2_MODEL, revision=TEXT_V2_REVISION)
    unit_ids, token_ids, owners, fingerprint = _units(df, tokenizer)
    unit_ids_array = np.asarray(unit_ids, dtype=str)
    provenance = {
        "schema": 1, "kind": "text_v2_units", "model": TEXT_V2_MODEL,
        "revision": TEXT_V2_REVISION, "config": TEXT_V2_CONFIG,
        "attention_implementation": "sdpa", "source_fingerprint": fingerprint,
        "n_units": len(unit_ids),
    }
    cache_path = os.path.join(cache_dir, "text_v2_nomic_units.npz")
    partial_path = os.path.join(cache_dir, "text_v2_nomic_units.partial.npz")
    cached = None if force else _load_cache(cache_path, unit_ids_array, provenance)
    if cached is not None:
        progress(f"text-v2 embeddings: cache hit ({cache_path})")
        return _aggregate_fields(df.video_id.astype(str).to_numpy(), cached.values, owners), provenance

    resolved_device = choose_device(device)
    progress(f"text-v2 embeddings: loading {TEXT_V2_MODEL}@{TEXT_V2_REVISION[:8]} on {resolved_device}")
    model = AutoModel.from_pretrained(
        TEXT_V2_MODEL, revision=TEXT_V2_REVISION, attn_implementation="sdpa"
    ).to(resolved_device).eval()
    values = np.zeros((len(unit_ids), TEXT_V2_DIM), dtype=np.float32)
    complete = np.zeros(len(unit_ids), dtype=np.float32)
    if not force:
        partial = _load_cache(partial_path, unit_ids_array, provenance)
        if partial is not None and "row_complete" in partial.masks and partial.values.shape == values.shape:
            values, complete = partial.values, partial.masks["row_complete"]
            progress(f"text-v2 embeddings: resuming {int(complete.sum())}/{len(unit_ids)} units")
    pending = [(index, token_ids[index]) for index in range(len(unit_ids)) if not complete[index]]

    def save_partial() -> None:
        _save_cache(partial_path, EmbeddingSet(
            unit_ids_array, values, {"row_complete": complete}, provenance
        ))

    try:
        for batch_no, (_, indexed_batch) in enumerate(_batched(pending, batch_size), start=1):
            indices, batch_tokens = zip(*indexed_batch)
            batch = tokenizer.pad(
                [{"input_ids": ids} for ids in batch_tokens], padding=True, return_tensors="pt"
            )
            batch = {name: value.to(resolved_device) for name, value in batch.items()}
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                attention = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1)
                pooled = functional.normalize(pooled[:, :TEXT_V2_DIM], p=2, dim=1)
            values[list(indices)] = pooled.float().cpu().numpy()
            complete[list(indices)] = 1.0
            if batch_no % 10 == 0 or int(complete.sum()) == len(unit_ids):
                save_partial()
                progress(f"text-v2 embeddings: {int(complete.sum())}/{len(unit_ids)} units")
    except BaseException:
        save_partial()
        raise
    bundle = EmbeddingSet(unit_ids_array, values, {"row_complete": complete}, provenance)
    _save_cache(cache_path, bundle)
    if os.path.exists(partial_path):
        os.unlink(partial_path)
    return _aggregate_fields(df.video_id.astype(str).to_numpy(), values, owners), provenance
