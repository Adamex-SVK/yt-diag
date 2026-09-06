"""Publish the nested deep-head tuning follow-up and update main.tex."""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "tuned_deep", "results.json")
TEXT_RESULTS = os.path.join(HERE, "results", "text_v2.json")
BASELINE_RESULTS = os.path.join(HERE, "results", "tuned_baselines.json")
SUMMARY = os.path.join(HERE, "results", "tuned_deep.json")
MAIN_TEX = os.path.join(HERE, "main.tex")
ABLATION = "thumbnail_text_fields_meta_sched"


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tuned_deep_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compact(full: dict, text: dict, baselines: dict) -> dict:
    if full.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("the report requires outer seeds 0..4")
    if any(run.get("test_evaluated") is not False for run in full["runs"]):
        raise ValueError("refusing to publish a run that evaluated test")
    fixed = text["aggregate"][ABLATION]
    tuned = full["aggregate"][ABLATION]
    structured = baselines["feature_sets"]["meta+sched"]["aggregate"]["xgboost"]["auc_roc"]
    comparisons = {}
    for model in ("linear_probe", "late_fusion_mlp"):
        selected = tuned[model]["auc_roc"]
        original = fixed[model]["auc_roc"]
        differences = [value - reference for value, reference in zip(
            selected["values"], original["values"]
        )]
        vs_structured = [value - reference for value, reference in zip(
            selected["values"], structured["values"]
        )]
        comparisons[model] = {
            "fixed": original,
            "tuned_minus_fixed_mean": float(sum(differences) / len(differences)),
            "tuned_wins_vs_fixed": int(sum(value > 0 for value in differences)),
            "tuned_minus_structured_mean": float(sum(vs_structured) / len(vs_structured)),
            "tuned_wins_vs_structured": int(sum(value > 0 for value in vs_structured)),
        }
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"], "n_labelled": full["n_labelled"],
        "seeds": full["seeds"], "configuration": full["configuration"],
        "embedding_provenance": full["embedding_provenance"],
        "aggregate": {ABLATION: tuned},
        "comparisons": comparisons,
        "tuned_metadata_schedule_xgboost": structured,
    }


def render(summary: dict) -> str:
    tuned = summary["aggregate"][ABLATION]
    comparison = summary["comparisons"]["late_fusion_mlp"]
    linear = tuned["linear_probe"]["auc_roc"]
    mlp = tuned["late_fusion_mlp"]["auc_roc"]
    fixed = comparison["fixed"]
    structured = summary["tuned_metadata_schedule_xgboost"]
    return (
        "\\paragraph{Fusion-head robustness.} A nested search over 12 linear "
        "and nine MLP configurations does not rescue the field-aware model: the "
        f"best linear probe reaches ${linear['mean']:.3f}\\pm{linear['std']:.3f}$, "
        f"the best MLP ${mlp['mean']:.3f}\\pm{mlp['std']:.3f}$ against "
        f"${fixed['mean']:.3f}\\pm{fixed['std']:.3f}$ for the pre-specified head "
        f"({comparison['tuned_wins_vs_fixed']}/5 paired wins) -- still "
        f"${abs(comparison['tuned_minus_structured_mean']):.3f}$ below tuned "
        f"metadata+schedule XGBoost (${structured['mean']:.3f}\\pm{structured['std']:.3f}$)."
    )


def splice(document: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED tuned-deep>>>"
    end = "%% <<<END GENERATED tuned-deep>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError("tuned-deep generated markers not found")
    return pattern.sub(lambda _match: f"{begin}\n{body}\n{end}", document)


def main() -> None:
    with open(RUN_RESULTS, encoding="utf-8") as stream:
        full = json.load(stream)
    with open(TEXT_RESULTS, encoding="utf-8") as stream:
        text = json.load(stream)
    with open(BASELINE_RESULTS, encoding="utf-8") as stream:
        baselines = json.load(stream)
    summary = compact(full, text, baselines)
    _atomic_json(SUMMARY, summary)
    with open(MAIN_TEX, encoding="utf-8") as stream:
        document = stream.read()
    with open(MAIN_TEX + ".tmp", "w", encoding="utf-8") as stream:
        stream.write(splice(document, render(summary)))
    os.replace(MAIN_TEX + ".tmp", MAIN_TEX)
    print(f"saved {SUMMARY}")
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
