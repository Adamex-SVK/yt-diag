"""Post-freeze visual ablation: give engineered visual features the same
isolate-and-tune treatment that run_audio_ablation.py gave audio.

Motivation: the existing "full engineered" row (meta+sched+vis+aud, tuned
XGBoost 0.614 +/- 0.023) bundles visual and audio together. Isolating audio
alone (run_audio_ablation.py) revealed a real lift the bundle was hiding
(0.632 +/- 0.028). Visual-only has so far only been reported once, untuned,
in the fixed feature ladder (0.519/0.505 AUC, see results/tuned_baselines.json)
-- it never got the matching nested-tuned isolation. This script closes that
gap so "does content beat metadata" is answered fairly for both engineered
blocks, not just the one that happened to win.

Named "visual_engineered" (not "visual_ablation") to avoid colliding with the
existing run_visual_ablation.py, which is the unrelated frozen-encoder
(DINOv2/CLIP/ResNet-50) backbone comparison behind the report's visual-ablation
table -- a different script, a different question, deliberately kept separate.

The engineered visual block (ytdiag/features.py VIS_COLUMNS) is 15 columns --
thumbnail + frame-aggregate colour/face statistics -- entirely distinct from
the frozen DINOv2/CLIP embeddings used by the deep fusion model. It is small
enough that no correlation-clustering step is needed (contrast with the 88
collinear eGeMAPS columns).

Unrelated to the frozen multimodal finalist (final_model_policy.json, frozen
2026-09-03): uses only already-approved retrospective vis__ columns, runs
entirely inside the existing nested channel-grouped tuning protocol
(ytdiag/tuning.py, covered by test_tuning.py), and never touches the sealed
test split or the prospective panel.

Example:
  .venv/Scripts/python 03_Models/run_visual_engineered_ablation.py --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.baselines import run_baselines, _preprocessor  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402
from ytdiag.tuning import run_tuned_baselines  # noqa: E402


def _gain_importance(df: pd.DataFrame, groups: tuple, seed: int, parameters: dict) -> dict[str, float]:
    """Gain-based importances for one tuned model, refit on its outer-train rows."""
    from ytdiag.features import select_columns
    from xgboost import XGBClassifier

    columns = select_columns(df, groups)
    columns = [c for c in columns if not df[c].isna().all()]
    outer = split_indices(df, seed=seed)
    prep = _preprocessor(columns, scale=False)
    Xt = prep.fit_transform(df[columns].iloc[outer["train"]])
    feature_names = prep.get_feature_names_out()
    y = df.label.astype(int).to_numpy()
    clf = XGBClassifier(**parameters, eval_metric="logloss", random_state=seed, n_jobs=4)
    clf.fit(Xt, y[outer["train"]])
    scores = clf.get_booster().get_score(importance_type="gain")
    return {feature_names[int(key[1:])]: value for key, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--xgb-trials", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--name", default="visual_engineered_ablation")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    df_full = load_retrospective(args.data)
    df_full = df_full[df_full.label.notna()].reset_index(drop=True)

    def tuned_run(df: pd.DataFrame, groups: tuple) -> dict[str, Any]:
        runs = []
        for seed in seeds:
            tuned = run_tuned_baselines(
                df, groups, seed=seed, inner_splits=args.inner_splits,
                xgb_trials=args.xgb_trials, search_seed=args.search_seed,
            )
            untuned = run_baselines(df, groups, seed=seed)
            runs.append({"seed": seed, "tuned": tuned, "untuned_auc": {
                name: float(model["val"]["auc_roc"])
                for name, model in untuned["models"].items()
                if name in {"logistic_regression", "xgboost"}
            }})
        aggregate = {}
        for model in ("logistic_regression", "xgboost"):
            values = [run["tuned"]["models"][model]["val"]["auc_roc"] for run in runs]
            aggregate[model] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                 "values": values}
        return {"groups": list(groups), "runs": runs, "aggregate": aggregate}

    print("meta+sched (reference)")
    reference = tuned_run(df_full, ("meta", "sched"))
    print(f"  xgboost {reference['aggregate']['xgboost']['mean']:.3f} +/- {reference['aggregate']['xgboost']['std']:.3f}")

    print("meta+sched+vis (isolated engineered visual, 15 columns)")
    full_visual = tuned_run(df_full, ("meta", "sched", "vis"))
    print(f"  xgboost {full_visual['aggregate']['xgboost']['mean']:.3f} +/- {full_visual['aggregate']['xgboost']['std']:.3f}")

    print("computing gain-based importance per seed")
    gain_by_seed = [
        _gain_importance(df_full, ("meta", "sched", "vis"), seed, run["tuned"]["models"]["xgboost"]["best_parameters"])
        for seed, run in zip(seeds, full_visual["runs"])
    ]
    vis_totals: dict[str, float] = {}
    for gains in gain_by_seed:
        for column, value in gains.items():
            if "vis__" in column:
                vis_totals[column] = vis_totals.get(column, 0.0) + value
    total_gain = sum(vis_totals.values()) or 1.0
    vis_share = {name: value / total_gain for name, value in sorted(vis_totals.items(), key=lambda kv: -kv[1])}

    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": (
            "channel-grouped 60/20/20 outer split; four channel-grouped inner folds; "
            "hyperparameters and F1 threshold selected by inner AUC; outer validation "
            "reported; test untouched. Post-freeze exploratory ablation, mirrors "
            "run_audio_ablation.py -- not part of final_model_policy.json."
        ),
        "n_labelled": int(len(df_full)),
        "seeds": seeds,
        "search_seed": args.search_seed,
        "xgb_random_trials": args.xgb_trials,
        "reference_meta_sched": reference,
        "full_visual_15col": full_visual,
        "gain_share_within_visual": vis_share,
    }
    output_dir = os.path.join(ROOT, "04_Experiments", "runs", args.name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
