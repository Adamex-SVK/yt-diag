"""
Regenerate the results tables inside main.tex from saved experiment output.

Numbers in the report are never typed by hand. This runs the baselines over
several seeds and rewrites the region between the <<<BEGIN GENERATED ...>>> and
<<<END GENERATED ...>>> markers IN PLACE -- in place rather than via \\input,
because the AAAI style requires the source to be a single file.

Single-seed numbers are not reported: with ~370 validation rows, logistic
regression on metadata alone ranges from 0.496 to 0.660 across seeds, so a
lone figure would be an artefact of the split rather than a result.

    .venv/bin/python 05_Reports/final_report/make_tables.py
    .venv/bin/python 05_Reports/final_report/make_tables.py --seeds 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "03_Models"))

MAIN_TEX = os.path.join(HERE, "main.tex")
# The multi-seed numbers are persisted, not just rendered: a table in a PDF is
# not a record. Anything quoted in the report must be traceable to this file.
RESULTS_JSON = os.path.join(HERE, "results", "baselines.json")
DATA = os.path.join(ROOT, "02_Data", "processed")

# The ablation ladder. Order matters: it reads as an argument, from the
# no-content floor up to every engineered feature we have.
LADDER: list[tuple[str, tuple[str, ...]]] = [
    ("Metadata", ("meta",)),
    ("+ schedule", ("meta", "sched")),
    ("+ visual", ("meta", "sched", "vis")),
    ("+ audio", ("meta", "sched", "vis", "aud")),
    ("Visual only", ("vis",)),
    ("Audio only", ("aud",)),
]

PRETTY = {"dummy_prior": "Dummy", "logistic_regression": "Logistic",
          "xgboost": "XGBoost", "hist_gradient_boosting": "HistGB"}


def collect(seeds: int) -> tuple[list[dict[str, Any]], list[str], int]:
    """Run every ladder rung; return rows, model names, and labelled sample count."""
    from ytdiag.adapters import load_retrospective
    from ytdiag.baselines import run_baselines

    df = load_retrospective(DATA)
    n_labelled = int(df.label.notna().sum())
    rows, model_names = [], []
    for pretty, groups in LADDER:
        per: dict[str, list[float]] = {}
        n_feat = None
        for seed in range(seeds):
            r = run_baselines(df, groups, seed=seed)
            n_feat = r["n_features"]
            for name, v in r["models"].items():
                per.setdefault(name, []).append(v["val"]["auc_roc"])
        model_names = list(per)
        rows.append({"label": pretty, "n_features": n_feat,
                     "auc": {m: (float(np.mean(v)), float(np.std(v))) for m, v in per.items()},
                     "auc_values": {m: [float(value) for value in values]
                                    for m, values in per.items()}})
        print(f"  {pretty:14s} n_feat={n_feat:3d}  " +
              "  ".join(f"{PRETTY.get(m, m)} {np.mean(v):.3f}+-{np.std(v):.3f}"
                        for m, v in per.items() if m != "dummy_prior"))
    return rows, model_names, n_labelled


def render(rows: list[dict[str, Any]], model_names: list[str], seeds: int,
           n_labelled: int) -> str:
    """LaTeX for the baseline table. booktabs, two-column safe."""
    models = [m for m in model_names if m != "dummy_prior"]
    dummy = rows[0]["auc"].get("dummy_prior", (0.5, 0.0))[0]
    head = " & ".join(["Features", "$d$"] + [PRETTY.get(m, m) for m in models])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{l r {'c' * len(models)}}}",
        "\\toprule",
        head + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        cells = []
        for m in models:
            mean, std = r["auc"].get(m, (float("nan"), float("nan")))
            cells.append(f"{mean:.3f} \\tiny{{$\\pm$ {std:.3f}}}")
        lines.append(f"{r['label']} & {r['n_features']} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\midrule",
        f"\\multicolumn{{{2 + len(models)}}}{{l}}{{\\small Dummy (class prior): "
        f"{dummy:.3f}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Validation AUC-ROC on the retrospective cohort "
        f"($n={n_labelled}$ labelled), mean $\\pm$ s.d. over {seeds} seeds. "
        "$d$ is the number of input columns actually fitted, after all-NaN "
        "columns are dropped. Splits are channel-grouped; the test split is "
        "untouched. The visual block alone performs at chance, which is the "
        "expected consequence of stratifying video format out of the label.}",
        "\\label{tab:baselines}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _cct_provenance() -> dict[str, Any]:
    """Which colour-temperature policy the stored visual features carry.

    Recorded with the results because a CCT recomputation changes the vis__
    block underneath them: numbers produced against different versions are not
    comparable. Scan every file so a mixed-policy dataset fails loudly."""
    import glob
    versions, methods = set(), set()
    for p in glob.glob(os.path.join(DATA, "*", "*", "visual_features.json")):
        try:
            with open(p, encoding="utf-8") as f:
                vis = json.load(f)
            versions.add(vis.get("cct_version"))
            methods.add(vis.get("cct_method"))
        except (OSError, json.JSONDecodeError):
            continue
    if len(versions) > 1 or len(methods) > 1:
        raise ValueError(f"mixed CCT policies: versions={versions}, methods={methods}")
    return {"version": next(iter(versions), None), "method": next(iter(methods), None)}


def splice(tex: str, key: str, body: str) -> str:
    """Replace the region between the generated markers for `key`."""
    begin, end = f"%% <<<BEGIN GENERATED {key}>>>", f"%% <<<END GENERATED {key}>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(tex):
        sys.exit(f"markers for '{key}' not found in main.tex -- add:\n{begin}\n{end}")
    # lambda, not a replacement string: LaTeX is full of backslashes and
    # re.sub would read \c, \t etc. as escape sequences
    return pattern.sub(lambda _m: f"{begin}\n{body}\n{end}", tex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"running the ablation ladder over {args.seeds} seeds:")
    rows, model_names, n_labelled = collect(args.seeds)
    body = render(rows, model_names, args.seeds, n_labelled)

    if args.dry_run:
        print("\n" + body)
        return
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    cct = _cct_provenance()
    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seeds": args.seeds, "seed_ids": list(range(args.seeds)),
        "n_labelled": n_labelled,
        "protocol": "channel-grouped 60/20/20; VALIDATION only, test untouched",
        "cct_version": cct["version"], "cct_method": cct["method"],
        "rows": [{"features": r["label"], "n_features": r["n_features"],
                  "auc": {m: {"mean": mu, "std": sd,
                              "values": r["auc_values"][m]}
                          for m, (mu, sd) in r["auc"].items()}}
                 for r in rows],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {RESULTS_JSON}")

    with open(MAIN_TEX, encoding="utf-8") as f:
        tex = f.read()
    out = splice(tex, "baselines", body)
    tmp = MAIN_TEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(out)
    os.replace(tmp, MAIN_TEX)
    print(f"\nwrote the baselines table into {MAIN_TEX}")


if __name__ == "__main__":
    main()
