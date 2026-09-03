"""Publish the deep-model run summary and splice its table into main.tex.

The full per-seed run stays in gitignored 04_Experiments/runs; this script
validates it, writes a compact tracked evidence file, and updates only the
generated deep-results region in the paper.
"""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "deep_multimodal", "results.json")
SUMMARY = os.path.join(HERE, "results", "deep_multimodal.json")
MAIN_TEX = os.path.join(HERE, "main.tex")

LABELS = {
    "thumbnail": "Thumbnail (DINOv2)",
    "text": "Text (ModernBERT)",
    "thumbnail_text": "Thumbnail + text",
    "thumbnail_text_meta_sched": "+ metadata + schedule",
}


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".deep_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def compact(full: dict) -> dict:
    """Keep evidence needed for every reported number, including seed values."""
    if full.get("protocol") != "channel-grouped 60/20/20; validation only; test untouched":
        raise ValueError(f"unexpected evaluation protocol: {full.get('protocol')}")
    if full.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("the report requires the frozen five-seed run (0..4)")
    if any(run.get("test_evaluated") is not False for run in full["runs"]):
        raise ValueError("refusing to publish a development run that evaluated test")
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "configuration": full["configuration"],
        "embedding_provenance": full["embedding_provenance"],
        "aggregate": full["aggregate"],
        "baseline_comparison": full["baseline_comparison"],
    }


def render(summary: dict) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l cc}",
        "\\toprule",
        "Inputs & Linear probe & Late-fusion MLP \\\\",
        "\\midrule",
    ]
    for key, models in summary["aggregate"].items():
        cells = []
        for model in ("linear_probe", "late_fusion_mlp"):
            auc = models[model]["auc_roc"]
            cells.append(f"{auc['mean']:.3f} \\tiny{{$\\pm$ {auc['std']:.3f}}}")
        lines.append(f"{LABELS[key]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Validation AUC-ROC for frozen deep representations, mean "
        "$\\pm$ s.d. over five channel-grouped split seeds. ModernBERT uses "
        "title, cleaned transcript and description up to 1{,}024 tokens. Model "
        "selection occurs inside the training fold; test is untouched.}",
        "\\label{tab:deep}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def splice(tex: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED deep-model>>>"
    end = "%% <<<END GENERATED deep-model>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(tex):
        raise ValueError("deep-model generated markers not found in main.tex")
    return pattern.sub(lambda _match: f"{begin}\n{body}\n{end}", tex)


def main() -> None:
    with open(RUN_RESULTS, encoding="utf-8") as f:
        summary = compact(json.load(f))
    _atomic_json(SUMMARY, summary)
    with open(MAIN_TEX, encoding="utf-8") as f:
        tex = f.read()
    updated = splice(tex, render(summary))
    with open(MAIN_TEX + ".tmp", "w", encoding="utf-8") as f:
        f.write(updated)
    os.replace(MAIN_TEX + ".tmp", MAIN_TEX)
    print(f"saved {SUMMARY}")
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
