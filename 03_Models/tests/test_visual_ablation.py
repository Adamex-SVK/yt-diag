"""Offline checks for visual-ablation preprocessing and protocol plumbing."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ytdiag.embed import EmbeddingSet, _load_cache, _save_cache  # noqa: E402
from ytdiag.visual_ablation import (  # noqa: E402
    VISUAL_SPECS, _cache_provenance, preprocess_image,
)


def test_all_preprocessors_return_finite_chw() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = os.path.join(temporary, "wide.png")
        Image.new("RGB", (400, 200), (220, 30, 80)).save(path)
        for spec in VISUAL_SPECS.values():
            pixels = preprocess_image(path, spec)
            assert pixels.shape == (3, spec.crop_size, spec.crop_size)
            assert np.isfinite(pixels).all()


def test_fit_pad_preserves_full_wide_image_and_uses_neutral_padding() -> None:
    spec = VISUAL_SPECS["dino_small_fit_cls"]
    with tempfile.TemporaryDirectory() as temporary:
        path = os.path.join(temporary, "wide.png")
        image = Image.new("RGB", (400, 100), (255, 0, 0))
        image.putpixel((0, 50), (0, 255, 0))
        image.putpixel((399, 50), (0, 0, 255))
        image.save(path)
        pixels = preprocess_image(path, spec)
    # Padding is filled with the normalisation mean and is therefore near zero.
    assert np.abs(pixels[:, 0, :]).max() < 0.02
    # The fitted image occupies the middle of the canvas instead of being cropped.
    assert np.abs(pixels[:, spec.crop_size // 2, :]).max() > 1.0


def test_visual_cache_provenance_survives_json_round_trip() -> None:
    ids = np.array(["a", "b"])
    provenance = _cache_provenance(
        VISUAL_SPECS["dino_small_center_mean"], "single_thumbnail", "abc",
    )
    bundle = EmbeddingSet(
        ids, np.ones((2, 3), dtype=np.float32),
        {"visual_present": np.ones(2, dtype=np.float32)}, provenance,
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = os.path.join(temporary, "cache.npz")
        _save_cache(path, bundle)
        assert _load_cache(path, ids, provenance) is not None


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"{name} OK")
    print("ALL VISUAL ABLATION TESTS PASSED")
