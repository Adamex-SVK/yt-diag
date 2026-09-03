"""Nested hyperparameter tuning for the retrospective tabular baselines.

Example:
  .venv/bin/python 03_Models/run_tuned_baselines.py \
    --feature-sets 'meta,sched;meta,sched,vis,aud' --seeds 0,1,2,3,4

All selection occurs inside each outer training fold. Results are reported on
outer validation. The unused fold within a run is not a globally unseen test
because rows are reassigned across development seeds.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.baselines import run_baselines  # noqa: E402
from ytdiag.tuning import run_tuned_baselines  # noqa: E402


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in runs[0]["tuned"]["models"]:
        output[model] = {}
        for metric in ("auc_roc", "pr_auc", "accuracy", "balanced_accuracy",
                       "precision", "recall", "specificity", "f1"):
            values = [run["tuned"]["models"][model]["val"][metric] for run in runs]
            output[model][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(value) for value in values],
            }
        output[model]["selected_parameters"] = [
            run["tuned"]["models"][model]["best_parameters"] for run in runs
        ]
    return output


def _untuned_summary(result: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(model["val"]["auc_roc"])
        for name, model in result["models"].items()
        if name in {"logistic_regression", "xgboost"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument(
        "--feature-sets",
        default="meta,sched;meta,sched,vis,aud",
        help="semicolon-separated feature sets; groups within a set use commas",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--xgb-trials", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--name", default="tuned_tabular_baselines")
    args = parser.parse_args()

    feature_sets = [
        tuple(group.strip() for group in value.split(",") if group.strip())
        for value in args.feature_sets.split(";") if value.strip()
    ]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    df = load_retrospective(args.data)
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": (
            f"channel-grouped 60/20/20 outer split; {args.inner_splits} channel-grouped inner folds; "
            "hyperparameters selected by inner AUC; threshold selected from inner OOF predictions; "
            "outer validation reported; no globally unseen retrospective test"
        ),
        "n_labelled": int(df.label.notna().sum()),
        "seeds": seeds,
        "search_seed": args.search_seed,
        "xgb_random_trials": args.xgb_trials,
        "feature_sets": {},
    }

    for groups in feature_sets:
        key = "+".join(groups)
        print(f"\n{key}")
        runs = []
        for seed in seeds:
            tuned = run_tuned_baselines(
                df, groups, seed=seed, inner_splits=args.inner_splits,
                xgb_trials=args.xgb_trials, search_seed=args.search_seed,
            )
            untuned = run_baselines(df, groups, seed=seed)
            runs.append({"seed": seed, "tuned": tuned, "untuned_auc": _untuned_summary(untuned)})
            scores = "  ".join(
                f"{name} {model['val']['auc_roc']:.3f}"
                for name, model in tuned["models"].items()
            )
            print(f"  seed {seed}: {scores}")
        aggregate = _aggregate(runs)
        untuned_aggregate = {}
        for model in ("logistic_regression", "xgboost"):
            values = [run["untuned_auc"][model] for run in runs]
            untuned_aggregate[model] = {
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "values": values,
            }
            tuned_auc = aggregate[model]["auc_roc"]
            print(
                f"  {model}: tuned {tuned_auc['mean']:.3f} ± {tuned_auc['std']:.3f}; "
                f"untuned {np.mean(values):.3f} ± {np.std(values):.3f}"
            )
        payload["feature_sets"][key] = {
            "groups": list(groups), "runs": runs, "aggregate": aggregate,
            "untuned_reference": untuned_aggregate,
        }

    output_dir = os.path.join(ROOT, "04_Experiments", "runs", args.name)
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, "results.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
