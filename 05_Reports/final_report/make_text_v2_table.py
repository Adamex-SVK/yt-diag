"""Publish the field-aware text-v2 run and splice its table into main.tex."""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "text_v2", "results.json")
SUMMARY = os.path.join(HERE, "results", "text_v2.json")
TUNED_SUMMARY = os.path.join(HERE, "results", "tuned_baselines.json")
MAIN_TEX = os.path.join(HERE, "main.tex")

ROWS = (
    ("transcript", "Transcript"),
    ("text_fields", "All three text fields"),
    ("thumbnail_text_fields_meta_sched", "+ metadata + thumbnail"),
)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".text_v2_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def compact(full: dict, tuned: dict) -> dict:
    expected = (
        "repeated channel-grouped 60/20/20 development splits; "
        "nested training-fold selection; validation only"
    )
    if full.get("protocol") != expected:
        raise ValueError(f"unexpected evaluation protocol: {full.get('protocol')}")
    if full.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("the report requires the frozen five-seed run (0..4)")
    if any(run.get("test_evaluated") is not False for run in full["runs"]):
        raise ValueError("refusing to publish a run that evaluated test")
    if tuned.get("seeds") != full["seeds"]:
        raise ValueError("tuned baseline and text run use different outer seeds")
    references = {
        "metadata_schedule_xgboost": tuned["feature_sets"]["meta+sched"]["aggregate"]["xgboost"]["auc_roc"],
        "full_engineered_xgboost": tuned["feature_sets"]["meta+sched+vis+aud"]["aggregate"]["xgboost"]["auc_roc"],
    }
    for name, reference in references.items():
        if len(reference.get("values", [])) != 5:
            raise ValueError(f"{name} lacks exact paired seed values")
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "metric_note": "AUC and PR-AUC are valid; historical F1 used a validation-selected threshold and is not reported. The finalist F1 in attribution.json uses inner-training thresholds.",
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "configuration": full["configuration"],
        "text_provenance": full["text_provenance"],
        "field_coverage": full["field_coverage"],
        "aggregate": full["aggregate"],
        "references": references,
        "reference_policy": "nested tuned structured baselines",
    }


def render(summary: dict) -> str:
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\begin{tabular}{l cc}", "\\toprule",
        "Inputs & Linear probe & Late-fusion MLP \\\\", "\\midrule",
    ]
    for key, label in ROWS:
        cells = []
        for model in ("linear_probe", "late_fusion_mlp"):
            auc = summary["aggregate"][key][model]["auc_roc"]
            cells.append(f"{auc['mean']:.3f} \\tiny{{$\\pm$ {auc['std']:.3f}}}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Validation AUC-ROC for the field-aware text follow-up, mean "
        "$\\pm$ s.d. over five channel-grouped split seeds. Title, description "
        "and quality-gated transcript chunks are encoded separately; model "
        "selection occurs inside each training partition.}",
        "\\label{tab:textv2}", "\\end{table}",
    ]
    aggregate = summary["aggregate"]
    all_text = aggregate["text_fields"]
    text_meta = aggregate["text_fields_meta_sched"]["late_fusion_mlp"]["auc_roc"]
    best = aggregate["thumbnail_text_fields_meta_sched"]["late_fusion_mlp"]["auc_roc"]
    engineered_fusion = aggregate["thumbnail_text_fields_engineered"]["late_fusion_mlp"]["auc_roc"]
    comparisons = []
    for name in ("metadata_schedule_xgboost", "full_engineered_xgboost"):
        reference = summary["references"][name]
        differences = [value - base for value, base in zip(best["values"], reference["values"])]
        comparisons.append((reference, sum(differences) / len(differences), sum(value > 0 for value in differences)))
    meta, meta_delta, meta_wins = comparisons[0]
    engineered, engineered_delta, engineered_wins = comparisons[1]
    lines += [
        "",
        "Field separation raises the all-text MLP to "
        f"${all_text['late_fusion_mlp']['auc_roc']['mean']:.3f}\\pm"
        f"{all_text['late_fusion_mlp']['auc_roc']['std']:.3f}$ AUC; metadata and schedule raise it to "
        f"${text_meta['mean']:.3f}\\pm{text_meta['std']:.3f}$. Adding the thumbnail reaches "
        f"${best['mean']:.3f}\\pm{best['std']:.3f}$, ${meta_delta:+.3f}$ versus tuned "
        f"metadata+schedule XGBoost (${meta['mean']:.3f}\\pm{meta['std']:.3f}$) with "
        f"{meta_wins}/5 paired wins. Adding engineered visual/audio features reaches only "
        f"${engineered_fusion['mean']:.3f}\\pm{engineered_fusion['std']:.3f}$. These overlapping "
        "split comparisons are descriptive rather than confidence intervals.",
    ]
    return "\n".join(lines)


def splice(tex: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED text-v2>>>"
    end = "%% <<<END GENERATED text-v2>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(tex):
        raise ValueError("text-v2 generated markers not found in main.tex")
    return pattern.sub(lambda _match: f"{begin}\n{body}\n{end}", tex)


def main() -> None:
    with open(RUN_RESULTS, encoding="utf-8") as stream:
        full = json.load(stream)
    with open(TUNED_SUMMARY, encoding="utf-8") as stream:
        tuned = json.load(stream)
    summary = compact(full, tuned)
    _atomic_json(SUMMARY, summary)
    with open(MAIN_TEX, encoding="utf-8") as stream:
        tex = stream.read()
    updated = splice(tex, render(summary))
    with open(MAIN_TEX + ".tmp", "w", encoding="utf-8") as stream:
        stream.write(updated)
    os.replace(MAIN_TEX + ".tmp", MAIN_TEX)
    print(f"saved {SUMMARY}")
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
