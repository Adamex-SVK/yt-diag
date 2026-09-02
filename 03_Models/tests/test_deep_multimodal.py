"""Offline checks for frozen encoders' plumbing and late fusion (no downloads)."""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.embed import (  # noqa: E402
    DINO_PREPROCESS, EmbeddingSet, _load_cache, _preprocess_dino, _save_cache, compose_text,
)
from ytdiag.fusion import run_seed  # noqa: E402
from ytdiag.synthetic import generate  # noqa: E402
from ytdiag.text_v2 import _aggregate_fields, _evenly_spaced, _field_chunks  # noqa: E402


def test_compose_text_preserves_fields_and_masks():
    with tempfile.TemporaryDirectory() as tmp:
        transcript = os.path.join(tmp, "transcript_clean.txt")
        with open(transcript, "w", encoding="utf-8") as f:
            f.write("the cleaned spoken words")
        text, masks = compose_text(pd.Series({
            "asset__title": "A title", "asset__description": "",
            "asset__transcript_path": transcript, "asset__transcript_usable": 1,
        }))
    assert text == "Title: A title\n\nTranscript: the cleaned spoken words"
    assert masks == {
        "title_present": 1.0, "description_present": 0.0, "transcript_present": 1.0,
        "transcript_usable": 1.0, "transcript_quality_known": 1.0,
    }


def test_dino_preprocessing_is_pinned_and_chw():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "wide.png")
        Image.new("RGB", (400, 200), (255, 0, 0)).save(path)
        pixels = _preprocess_dino(path)
    assert pixels.shape == (3, 224, 224)
    expected_red = (1.0 - DINO_PREPROCESS["mean"][0]) / DINO_PREPROCESS["std"][0]
    assert np.allclose(pixels[0], expected_red)


def test_embedding_cache_requires_exact_provenance_and_order():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.npz")
        bundle = EmbeddingSet(
            np.array(["a", "b"]), np.ones((2, 3), dtype=np.float32),
            {"present": np.array([1, 0], dtype=np.float32)}, {"schema": 1, "model": "test"},
        )
        _save_cache(path, bundle)
        loaded = _load_cache(path, np.array(["a", "b"]), bundle.provenance)
        assert loaded is not None and np.array_equal(loaded.values, bundle.values)
        assert _load_cache(path, np.array(["b", "a"]), bundle.provenance) is None
        assert _load_cache(path, bundle.video_ids, {"schema": 2}) is None


def test_text_v2_chunks_cover_long_transcript_from_start_to_end():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(10, 10 + len(text.split())))

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def build_inputs_with_special_tokens(self, ids):
            return [1, *ids, 2]

    text = " ".join(f"word{i}" for i in range(3000))
    chunks = _field_chunks(FakeTokenizer(), "transcript", text)
    assert len(chunks) == 4 and all(len(chunk) <= 384 for chunk in chunks)
    assert chunks[0][0] == 1 and chunks[-1][-1] == 2
    # The evenly selected windows include both temporal ends.
    assert 10 in chunks[0] and (10 + 2999) in chunks[-1]
    assert _evenly_spaced(list(range(10)), 4) == [0, 3, 6, 9]


def test_text_v2_chunk_aggregation_is_table_aligned_and_normalised():
    ids = np.array(["a", "b"])
    units = np.array([[1, 0], [0, 1], [1, 1], [2, 0]], dtype=np.float32)
    owners = [(0, "title"), (0, "transcript"), (0, "transcript"), (1, "title")]
    # The real representation is 256-D; pad this fixture to that contract.
    units = np.pad(units, ((0, 0), (0, 254)))
    fields = _aggregate_fields(ids, units, owners)
    assert fields["title"].masks["field_present"].tolist() == [1, 1]
    assert fields["transcript"].masks["n_chunks"].tolist() == [2, 0]
    assert np.isclose(np.linalg.norm(fields["transcript"].values[0]), 1.0)
    assert np.all(fields["description"].values == 0)


def test_late_fusion_runs_without_touching_test():
    with tempfile.TemporaryDirectory() as tmp:
        df = load_retrospective(generate(tmp, n=300, seed=7))
    ids = df.video_id.astype(str).to_numpy()
    y = df.label.to_numpy(dtype=np.float32)
    rng = np.random.default_rng(3)
    thumb_values = rng.normal(size=(len(df), 12)).astype(np.float32)
    text_values = rng.normal(size=(len(df), 16)).astype(np.float32)
    thumb_values[:, 0] += 2.0 * y
    text_values[:, 0] += 2.0 * y
    thumbnail = EmbeddingSet(ids, thumb_values, {"thumbnail_present": np.ones(len(df))}, {"kind": "test"})
    text = EmbeddingSet(ids, text_values, {
        "title_present": np.ones(len(df)), "description_present": np.ones(len(df)),
        "transcript_present": np.ones(len(df)), "transcript_usable": np.ones(len(df)),
        "transcript_quality_known": np.ones(len(df)),
    }, {"kind": "test"})
    result = run_seed(
        df, thumbnail, text, seed=0, ablations=["thumbnail_text"], device="cpu", epochs=2
    )
    assert result["test_evaluated"] is False
    ablation = result["ablations"]["thumbnail_text"]
    assert ablation["linear_probe"]["val"]["auc_roc"] > 0.65
    assert "test" not in ablation["linear_probe"]
    assert ablation["late_fusion_mlp"]["n_parameters"] > 0


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"{name} OK")
    print("ALL DEEP MULTIMODAL TESTS PASSED")
