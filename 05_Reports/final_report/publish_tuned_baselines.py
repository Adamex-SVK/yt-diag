"""Publish the nested tabular tuning run and splice its table into main.tex."""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(
    ROOT, "04_Experiments", "runs", "tuned_tabular_baselines", "results.json"
)
SUMMARY = os.path.join(HERE, "results", "tuned_baselines.json")
MAIN_TEX = os.path.join(HERE, "main.tex")
FEATURES = (
    ("meta+sched", "Metadata + schedule"),
    ("meta+sched+vis+aud", "Full engineered"),
)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tuned_", suffix=".json", dir=os.path.dirname(path))
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
    for key, _ in FEATURES:
        feature = full["feature_sets"].get(key)
        if feature is None:
            raise ValueError(f"missing tuned feature set {key}")
        if any(run["tuned"].get("test_evaluated") is not False for run in feature["runs"]):
            raise ValueError("refusing to publish a run that evaluated test")
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "search_seed": full["search_seed"],
        "xgb_random_trials": full["xgb_random_trials"],
        "feature_sets": {
            key: {
                "groups": full["feature_sets"][key]["groups"],
                "aggregate": full["feature_sets"][key]["aggregate"],
                "untuned_reference": full["feature_sets"][key]["untuned_reference"],
            }
            for key, _ in FEATURES
        },
    }


def render(summary: dict) -> str:
    best = summary["feature_sets"]["meta+sched"]["aggregate"]["xgboost"]
    full = summary["feature_sets"]["meta+sched+vis+aud"]["aggregate"]["xgboost"]
    lines = [
        "\\paragraph{Nested structured tuning.} Four channel-grouped folds "
        "inside each training partition select hyperparameters and the F1 "
        "threshold. Metadata+schedule "
        f"XGBoost improves from ${summary['feature_sets']['meta+sched']['untuned_reference']['xgboost']['mean']:.3f}"
        f"\\pm{summary['feature_sets']['meta+sched']['untuned_reference']['xgboost']['std']:.3f}$ to "
        f"${best['auc_roc']['mean']:.3f}\\pm{best['auc_roc']['std']:.3f}$. The tuned full engineered model reaches "
        f"${full['auc_roc']['mean']:.3f}\\pm{full['auc_roc']['std']:.3f}$; visual and audio "
        "features therefore do not improve the strongest tuned structured model. "
        "These five overlapping "
        "split-seed comparisons are descriptive, not confidence intervals or a significance test.",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED tuned-baselines>>>"
    end = "%% <<<END GENERATED tuned-baselines>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError("tuned-baselines generated markers not found in main.tex")
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
