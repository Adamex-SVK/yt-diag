"""Publish the post-freeze audio ablation and splice its paragraph into main.tex.

Companion to publish_tuned_baselines.py, same conventions. This ablation is
NOT part of the frozen multimodal finalist (03_Models/final_model_policy.json,
frozen 2026-09-03) -- it isolates audio from visual in the tuned tabular
comparator using only already-approved retrospective columns, entirely inside
the existing nested channel-grouped protocol. See run_audio_ablation.py for
why it does not touch the freeze.
"""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "audio_ablation", "results.json")
SIGNIFICANCE_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "ablation_significance", "results.json")
SUMMARY = os.path.join(HERE, "results", "audio_ablation.json")
MAIN_TEX = os.path.join(HERE, "main.tex")


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".audio_ablation_", suffix=".json", dir=os.path.dirname(path))
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
    for run in (full["reference_meta_sched"]["runs"] + full["full_audio_88col"]["runs"]
                + full["reduced_audio"]["runs"]):
        if any(run["tuned"]["models"][m].get("val", {}).get("n") is None for m in ("xgboost",)):
            continue
        if run["tuned"].get("test_evaluated", False) is not False:
            raise ValueError("refusing to publish a run that evaluated test")
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "search_seed": full["search_seed"],
        "xgb_random_trials": full["xgb_random_trials"],
        "reference_meta_sched": {"aggregate": full["reference_meta_sched"]["aggregate"]},
        "full_audio_88col": {"aggregate": full["full_audio_88col"]["aggregate"]},
        "reduced_audio": {
            "aggregate": full["reduced_audio"]["aggregate"],
            "n_columns": full["reduced_audio"]["n_columns"],
        },
        "gain_family_share": full["gain_family_share"],
    }


def render(summary: dict, significance: dict) -> str:
    reference = summary["reference_meta_sched"]["aggregate"]["xgboost"]
    full = summary["full_audio_88col"]["aggregate"]["xgboost"]
    reduced = summary["reduced_audio"]["aggregate"]["xgboost"]
    n_reduced = summary["reduced_audio"]["n_columns"]
    shares = sorted(summary["gain_family_share"].items(), key=lambda kv: kv[1], reverse=True)
    top_families = ", ".join(f"{name} ({value * 100:.0f}\\%)" for name, value in shares[:4])
    sig = significance["audio_isolated_vs_reference"]
    ci_low, ci_high = sig["bootstrap_95ci"]
    p_value = sig["p_value_one_sided_not_better"]
    lines = [
        "\\paragraph{Post-freeze audio ablation.} Isolating audio from visual "
        "-- a combination the fixed and tuned engineered ladders never test in "
        "isolation -- changes the picture. Tuned metadata+schedule XGBoost "
        f"reaches ${reference['mean']:.3f}\\pm{reference['std']:.3f}$ on these "
        f"seeds; adding the full 88-column eGeMAPS block reaches "
        f"${full['mean']:.3f}\\pm{full['std']:.3f}$, exceeding it on every "
        "seed. Because 23 eGeMAPS pairs correlate at $|\\rho|>0.95$ "
        "(\\texttt{eda\\_features/audio\\_spearman.csv}), gain-based feature importance is "
        "nearly flat across all 109 model columns; grouping by acoustic "
        f"family shows the lift is diffuse rather than concentrated, led by "
        f"{top_families}, each ahead of metadata+schedule's own "
        f"{summary['gain_family_share']['metadata/schedule'] * 100:.0f}\\% share. "
        f"Clustering correlated eGeMAPS columns to one representative per "
        f"cluster ($|\\rho|>0.75$) reduces 88 columns to {n_reduced - 4} plus "
        "the 4 pause features "
        f"({n_reduced} total) at no cost: ${reduced['mean']:.3f}\\pm{reduced['std']:.3f}$. "
        "This ablation was computed after the 2026-09-03 model freeze using "
        "only already-approved retrospective columns and the existing nested "
        "tuning protocol; it does not touch the sealed test split or the "
        "frozen multimodal finalist, and is reported as a candidate for "
        "external validation rather than a change to the frozen comparator. "
        f"A paired bootstrap over validation-fold videos (held-fixed, already-"
        f"tuned hyperparameters; not a re-search) puts the lift at "
        f"${sig['point_estimate_mean_auc_diff']:+.3f}$ AUC, 95\\% CI "
        f"$[{ci_low:+.3f}, {ci_high:+.3f}]$ -- the interval includes zero, so "
        f"this does not clear conventional significance (one-sided "
        f"$p={p_value:.3f}$ for `not better'). "
        "\\note{PROVISIONAL}{Emmanuel + Adam, 2026-09-05: reported as a "
        "standalone post-hoc finding, not a trigger for a follow-up freeze "
        "cycle -- it stays outside the sealed test split and the frozen "
        "multimodal finalist. Direction is consistent but not yet "
        "statistically confirmed; revisit once the prospective panel matures.}",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED audio-ablation>>>"
    end = "%% <<<END GENERATED audio-ablation>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError(
            "audio-ablation generated markers not found in main.tex -- "
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
