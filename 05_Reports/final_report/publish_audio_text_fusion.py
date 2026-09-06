"""Publish the post-freeze audio+text fusion experiment and splice its
paragraph into main.tex.

Companion to publish_audio_ablation.py / publish_visual_engineered_ablation.py.
Tests whether combining engineered audio with frozen ModernBERT text
embeddings in one fusion model beats either alone. It does not: see
run_audio_text_fusion.py.
"""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "audio_text_fusion", "results.json")
SUMMARY = os.path.join(HERE, "results", "audio_text_fusion.json")
MAIN_TEX = os.path.join(HERE, "main.tex")


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".audio_text_fusion_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compact(full: dict) -> dict:
    if full.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("the report requires the frozen five-seed run (0..4)")
    for run in full["runs"]:
        if run.get("test_evaluated", False) is not False:
            raise ValueError("refusing to publish a run that evaluated test")
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "aggregate": full["aggregate"],
        "references": full["references"],
    }


def render(summary: dict) -> str:
    aggregate = summary["aggregate"]
    text_only = aggregate["text_fields_meta_sched"]
    audio_only = aggregate["audio_meta_sched"]
    combined = aggregate["text_fields_audio_meta_sched"]
    ref_meta = summary["references"]["metadata_schedule_xgboost_tuned"]
    ref_audio = summary["references"]["metadata_schedule_audio_xgboost_isolated"]
    lines = [
        "\\emph{Combining audio with text:} field-aware text "
        "(\\S\\ref{tab:textv2}) and audio, individually weak-to-moderate, fused "
        "via the same linear/MLP architecture used for text and vision reach "
        f"${text_only['linear_probe']['auc_roc']['mean']:.3f}"
        f"\\pm{text_only['linear_probe']['auc_roc']['std']:.3f}$/"
        f"${text_only['late_fusion_mlp']['auc_roc']['mean']:.3f}"
        f"\\pm{text_only['late_fusion_mlp']['auc_roc']['std']:.3f}$ (text) and "
        f"${audio_only['linear_probe']['auc_roc']['mean']:.3f}"
        f"\\pm{audio_only['linear_probe']['auc_roc']['std']:.3f}$/"
        f"${audio_only['late_fusion_mlp']['auc_roc']['mean']:.3f}"
        f"\\pm{audio_only['late_fusion_mlp']['auc_roc']['std']:.3f}$ (audio) -- "
        f"both well below audio's ${ref_audio['mean']:.3f}$ under tuned "
        "XGBoost, so its edge is tied to tree-based modelling of tabular "
        "features, not portable to this fusion architecture. Combined, they "
        f"reach only ${combined['linear_probe']['auc_roc']['mean']:.3f}"
        f"\\pm{combined['linear_probe']['auc_roc']['std']:.3f}$/"
        f"${combined['late_fusion_mlp']['auc_roc']['mean']:.3f}"
        f"\\pm{combined['late_fusion_mlp']['auc_roc']['std']:.3f}$, below tuned "
        f"metadata+schedule (${ref_meta['mean']:.3f}$) and below audio alone -- "
        "text adds nothing, echoing the earlier visual+audio bundling pattern. "
        "\\note{PROVISIONAL}{Hypothesis test, not a search; it failed. Does not "
        "change the frozen finalist or the audio-ablation finding.}",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED audio-text-fusion>>>"
    end = "%% <<<END GENERATED audio-text-fusion>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError(
            "audio-text-fusion generated markers not found in main.tex -- "
            "add the markers once (see main.tex's other GENERATED blocks) before running this"
        )
    return pattern.sub(lambda _match: f"{begin}\n{body}\n{end}", document)


def main() -> None:
    with open(RUN_RESULTS, encoding="utf-8") as stream:
        summary = compact(json.load(stream))
    _atomic_json(SUMMARY, summary)
    with open(MAIN_TEX, encoding="utf-8") as stream:
        document = stream.read()
    updated = splice(document, render(summary))
    with open(MAIN_TEX + ".tmp", "w", encoding="utf-8") as stream:
        stream.write(updated)
    os.replace(MAIN_TEX + ".tmp", MAIN_TEX)
    print(f"saved {SUMMARY}")
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
