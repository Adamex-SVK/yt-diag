"""Publish the post-freeze engineered-visual ablation and splice its paragraph
into main.tex.

Companion to publish_audio_ablation.py, same conventions. This checks whether
audio's isolate-and-tune lift (run_audio_ablation.py) generalises to the
engineered visual block, or whether audio specifically carries signal that
visual does not. It does not: see run_visual_engineered_ablation.py.
"""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "visual_engineered_ablation", "results.json")
SIGNIFICANCE_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "ablation_significance", "results.json")
SUMMARY = os.path.join(HERE, "results", "visual_engineered_ablation.json")
MAIN_TEX = os.path.join(HERE, "main.tex")


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".visual_engineered_", suffix=".json", dir=os.path.dirname(path))
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
    for run in full["reference_meta_sched"]["runs"] + full["full_visual_15col"]["runs"]:
        if run["tuned"].get("test_evaluated", False) is not False:
            raise ValueError("refusing to publish a run that evaluated test")
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "reference_meta_sched": {"aggregate": full["reference_meta_sched"]["aggregate"]},
        "full_visual_15col": {"aggregate": full["full_visual_15col"]["aggregate"]},
        "gain_share_within_visual": full["gain_share_within_visual"],
    }


def render(summary: dict, significance: dict) -> str:
    reference = summary["reference_meta_sched"]["aggregate"]["xgboost"]
    full = summary["full_visual_15col"]["aggregate"]["xgboost"]
    shares = sorted(summary["gain_share_within_visual"].values(), reverse=True)
    spread = f"{shares[0] * 100:.1f}\\%\\ldots{shares[-1] * 100:.1f}\\%" if shares else "n/a"
    sig = significance["visual_engineered_vs_reference"]
    ci_low, ci_high = sig["bootstrap_95ci"]
    lines = [
        "\\paragraph{Post-freeze visual ablation.} The audio result above raises "
        "an obvious question: was audio's lift a property of \\emph{isolating} a "
        "block, or a property of audio specifically? The engineered visual block "
        "(15 thumbnail and frame-aggregate colour/face columns) had only "
        "appeared bundled with audio in the tuned ladder, or alone but untuned "
        "in the fixed ladder ($0.505$--$0.519$). Giving it the identical "
        "isolate-and-tune treatment answers the question: metadata+schedule "
        f"XGBoost reaches ${reference['mean']:.3f}\\pm{reference['std']:.3f}$ on "
        f"these seeds; adding the 15-column visual block reaches only "
        f"${full['mean']:.3f}\\pm{full['std']:.3f}$ -- \\emph{{below}} the "
        "reference, not above it. Gain-based importance across the 15 columns "
        f"is flat ({spread} of the visual gain each), consistent with noise "
        "rather than a masked signal. Unlike audio's lift, this drop is "
        f"statistically solid: the same paired bootstrap gives a 95\\% CI of "
        f"$[{ci_low:+.3f}, {ci_high:+.3f}]$, entirely below zero. Isolation was "
        "therefore not the fix on its own; audio specifically carries signal "
        "that engineered visual features do not.",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED visual-engineered-ablation>>>"
    end = "%% <<<END GENERATED visual-engineered-ablation>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError(
            "visual-engineered-ablation generated markers not found in main.tex -- "
            "add the markers once (see main.tex's other GENERATED blocks) before running this"
        )
    return pattern.sub(lambda _match: f"{begin}\n{body}\n{end}", document)


def main() -> None:
    with open(RUN_RESULTS, encoding="utf-8") as stream:
        summary = compact(json.load(stream))
    with open(SIGNIFICANCE_RESULTS, encoding="utf-8") as stream:
        significance = json.load(stream)
    _atomic_json(SUMMARY, summary)
    with open(MAIN_TEX, encoding="utf-8") as stream:
        document = stream.read()
    updated = splice(document, render(summary, significance))
    with open(MAIN_TEX + ".tmp", "w", encoding="utf-8") as stream:
        stream.write(updated)
    os.replace(MAIN_TEX + ".tmp", MAIN_TEX)
    print(f"saved {SUMMARY}")
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
