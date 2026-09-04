#!/usr/bin/env python3
"""Publish the finalist metric table and exact-attribution result into main.tex."""
from __future__ import annotations

import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))


def _cell(metric: dict) -> str:
    return f"{metric['mean']:.3f} \\tiny{{$\\pm$ {metric['std']:.3f}}}"


def render(attribution: dict, tuned: dict) -> str:
    clip = attribution["aggregate"]
    xgb = tuned["feature_sets"]["meta+sched"]["aggregate"]["xgboost"]
    shares = attribution["share_of_absolute_logit_contribution"]
    ordered = sorted(shares, key=lambda name: shares[name]["mean"], reverse=True)
    share_text = ", ".join(
        f"{name.replace('_', '+')} {100 * shares[name]['mean']:.1f}\\%" for name in ordered
    )
    first = attribution["runs"][0]["preprocessing"]
    meta_dims = [run["preprocessing"]["meta_sched"]["input_dim"] for run in attribution["runs"]]
    dimension_text = (
        f"{first['thumbnail']['input_dim']} thumbnail, {first['frames']['input_dim']} frames, "
        f"{first['title']['input_dim']}--{first['transcript']['input_dim']} per text field, "
        f"versus {min(meta_dims)}--{max(meta_dims)} transformed metadata/schedule inputs"
    )
    return "\n".join([
        "\\begin{figure}[t]",
        "\\centering",
        "\\includegraphics[width=0.96\\columnwidth]{figures/attribution.pdf}",
        "\\caption{Exact attribution for the CLIP--text linear fusion. Top: mean absolute-contribution shares. Bottom: signed contributions for four seed-0 examples; positive values favour the top-quartile class.}",
        "\\label{fig:attribution}",
        "\\end{figure}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{l ccc}",
        "\\toprule",
        "Finalist & AUC-ROC & PR-AUC & F1 \\\\",
        "\\midrule",
        f"Structured XGBoost & {_cell(xgb['auc_roc'])} & {_cell(xgb['pr_auc'])} & {_cell(xgb['f1'])} \\\\",
        f"CLIP--text linear fusion & {_cell(clip['auc_roc'])} & {_cell(clip['pr_auc'])} & {_cell(clip['f1'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Repeated channel-grouped development performance. Hyperparameters and F1 thresholds are selected inside each training partition. These rows are model-selection evidence, not an independent test estimate.}",
        "\\label{tab:finalists}",
        "\\end{table}",
        "",
        "\\paragraph{Predictive attribution and error analysis.} The linear fusion permits an exact decomposition of every logit by input block (Figure~\\ref{fig:attribution}). Mean absolute-contribution shares are " + share_text + ". They are not modality-importance estimates: block dimensionality (" + dimension_text + "), correlated coordinates, standardisation and the shared L2 penalty all affect the allocation. In seed 0, both a true-positive and false-positive latte-art Short receive their largest positive contribution from frames. The model recognises similar content but cannot explain their different outcomes; attribution reveals what moved the prediction, not what moved viewers.",
    ])


def main() -> None:
    with open(os.path.join(HERE, "results", "attribution.json"), encoding="utf-8") as stream:
        attribution = json.load(stream)
    with open(os.path.join(HERE, "results", "tuned_baselines.json"), encoding="utf-8") as stream:
        tuned = json.load(stream)
    path = os.path.join(HERE, "main.tex")
    with open(path, encoding="utf-8") as stream:
        document = stream.read()
    begin = "%% <<<BEGIN GENERATED attribution>>>"
    end = "%% <<<END GENERATED attribution>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError("attribution generated markers not found")
    replacement = f"{begin}\n{render(attribution, tuned)}\n{end}"
    document = pattern.sub(lambda _match: replacement, document)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(document)
    print(f"updated {path}")


if __name__ == "__main__":
    main()
