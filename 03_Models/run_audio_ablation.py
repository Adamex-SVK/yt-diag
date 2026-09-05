"""Post-freeze audio ablation: isolate audio from visual in the tuned tabular
comparator, then check whether the 88-column eGeMAPS block is redundant.

Motivation: the existing "full engineered" row (meta+sched+vis+aud, tuned
XGBoost 0.614 +/- 0.023, see results/tuned_baselines.json) bundles visual and
audio together, so it cannot show what audio alone contributes -- visual
features may be diluting it. This script isolates meta+sched+aud, then
reduces the 88 eGeMAPS columns to a non-redundant subset (existing EDA found
23 eGeMAPS pairs with |Spearman rho| > 0.95, see eda_features/audio_spearman.csv)
and checks the reduced set reproduces the full-audio result.

This is unrelated to the frozen multimodal finalist (final_model_policy.json,
frozen 2026-09-03): it uses only the already-approved retrospective aud__
columns, runs entirely inside the existing nested channel-grouped tuning
protocol (ytdiag/tuning.py, covered by test_tuning.py), and never touches the
sealed test split or the prospective panel. It does not modify or supersede
the frozen structured comparator -- it is a separate, explicitly-labelled
ablation for the team to evaluate before deciding whether it earns inclusion.

Example:
  .venv/bin/python 03_Models/run_audio_ablation.py --seeds 0,1,2,3,4
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
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.baselines import run_baselines, _preprocessor  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402
from ytdiag.tuning import run_tuned_baselines  # noqa: E402

SPEARMAN_CSV = os.path.join(ROOT, "02_Data", "eda_features", "audio_spearman.csv")
PAUSE_COLUMNS = ["aud__pause_count", "aud__pause_total_pause_sec",
                  "aud__pause_pause_ratio", "aud__pause_mean_pause_sec"]

FAMILY_RULES = [
    (r"pause", "pauses/dead-air"),
    (r"^meta__|^sched__", "metadata/schedule"),
    (r"F0semitone|logRelF0|jitter", "pitch (F0) dynamics"),
    (r"loudness|equivalentSoundLevel", "loudness"),
    (r"HNRdBACF|shimmer", "voice quality (HNR/shimmer)"),
    (r"mfcc", "MFCC (timbre)"),
    (r"F1|F2|F3", "formants"),
    (r"spectralFlux|alphaRatio|hammarberg|slope", "spectral balance/slope"),
    (r"VoicedSegment|UnvoicedSegment", "voicing rate/segment length"),
]


def _family_of(column: str) -> str:
    name = column.replace("num__aud__egemaps__", "").replace("num__aud__", "").replace("num__", "")
    for pattern, family in FAMILY_RULES:
        if re.search(pattern, name):
            return family
    return "other/uncategorized"


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


def _reduce_egemaps(gain_by_seed: list[dict[str, float]]) -> list[str]:
    """One representative eGeMAPS column per correlation cluster (|rho|>0.75),
    keeping the member with the highest mean gain across seeds."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = pd.read_csv(SPEARMAN_CSV, index_col=0)
    names = list(corr.columns)
    dist = 1 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    linkage_matrix = linkage(squareform(dist, checks=False), method="average")
    clusters = fcluster(linkage_matrix, t=0.25, criterion="distance")  # |rho| > 0.75

    def mean_gain(egemaps_name: str) -> float:
        col = f"num__aud__egemaps__{egemaps_name}"
        return float(np.mean([d.get(col, 0.0) for d in gain_by_seed]))

    members_by_cluster: dict[int, list[str]] = {}
    for name, cluster in zip(names, clusters):
        members_by_cluster.setdefault(cluster, []).append(name)
    representatives = [max(members, key=mean_gain) for members in members_by_cluster.values()]
    return [f"aud__egemaps__{name}" for name in representatives]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--xgb-trials", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--name", default="audio_ablation")
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

    print("meta+sched+aud (full 88-column eGeMAPS)")
    full_audio = tuned_run(df_full, ("meta", "sched", "aud"))
    print(f"  xgboost {full_audio['aggregate']['xgboost']['mean']:.3f} +/- {full_audio['aggregate']['xgboost']['std']:.3f}")

    print("computing gain-based importance per seed for redundancy reduction")
    gain_by_seed = [
        _gain_importance(df_full, ("meta", "sched", "aud"), seed, run["tuned"]["models"]["xgboost"]["best_parameters"])
        for seed, run in zip(seeds, full_audio["runs"])
    ]
    family_totals: dict[str, float] = {}
    for gains in gain_by_seed:
        for column, value in gains.items():
            family_totals[_family_of(column)] = family_totals.get(_family_of(column), 0.0) + value
    family_share = {name: value / sum(family_totals.values()) for name, value in family_totals.items()}

    reduced_egemaps = _reduce_egemaps(gain_by_seed)
    reduced_columns = reduced_egemaps + PAUSE_COLUMNS
    aud_all = [c for c in df_full.columns if c.startswith("aud__")]
    df_reduced = df_full.drop(columns=[c for c in aud_all if c not in reduced_columns])
    print(f"meta+sched+aud (correlation-reduced, {len(reduced_columns)} of {len(aud_all)} columns)")
    reduced_audio = tuned_run(df_reduced, ("meta", "sched", "aud"))
    print(f"  xgboost {reduced_audio['aggregate']['xgboost']['mean']:.3f} +/- {reduced_audio['aggregate']['xgboost']['std']:.3f}")

    payload = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": (
            "channel-grouped 60/20/20 outer split; four channel-grouped inner folds; "
            "hyperparameters and F1 threshold selected by inner AUC; outer validation "
            "reported; test untouched. Post-freeze exploratory ablation -- not part of "
            "final_model_policy.json."
        ),
        "n_labelled": int(len(df_full)),
        "seeds": seeds,
        "search_seed": args.search_seed,
        "xgb_random_trials": args.xgb_trials,
        "reference_meta_sched": reference,
        "full_audio_88col": full_audio,
        "reduced_audio": {**reduced_audio, "n_columns": len(reduced_columns), "columns": reduced_columns},
        "gain_family_share": family_share,
    }
    output_dir = os.path.join(ROOT, "04_Experiments", "runs", args.name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
