"""Paired bootstrap significance test for the post-freeze audio and
engineered-visual ablations.

Every AUC comparison so far in this report (audio-ablation, visual-ablation,
visual-engineered-ablation) is "mean +/- std over 5 overlapping channel-grouped
splits" -- explicitly disclaimed in main.tex as descriptive, not a significance
test, because the splits are not independent samples (videos repeat across
seeds). This script adds an actual paired test: for each seed, refit the
reference (meta+sched) and the augmented (meta+sched+X) model with the same
tuned hyperparameters already selected by run_audio_ablation.py /
run_visual_engineered_ablation.py, score both on the identical outer-validation
rows for that seed (paired by construction), then bootstrap-resample videos
*within* each seed's validation fold to build a distribution of the mean AUC
difference across the 5 seeds.

This only re-scores with already-selected hyperparameters -- it does not
re-tune, so it cannot manufacture a better result by searching harder. It
answers one question honestly: given the model we already reported, how much
of the observed AUC gap could plausibly be sampling noise?

Example:
  .venv/Scripts/python 03_Models/run_ablation_significance.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.baselines import _preprocessor  # noqa: E402
from ytdiag.features import select_columns  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "04_Experiments", "runs")


def _fit_predict(df: pd.DataFrame, groups: tuple, seed: int, parameters: dict,
                  train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    """Refit one already-tuned XGBoost model and return validation probabilities."""
    from xgboost import XGBClassifier

    columns = select_columns(df, groups)
    columns = [c for c in columns if not df[c].isna().all()]
    prep = _preprocessor(columns, scale=False)
    Xt_train = prep.fit_transform(df[columns].iloc[train_idx])
    Xt_val = prep.transform(df[columns].iloc[val_idx])
    y = df.label.astype(int).to_numpy()
    clf = XGBClassifier(**parameters, eval_metric="logloss", random_state=seed, n_jobs=4)
    clf.fit(Xt_train, y[train_idx])
    return clf.predict_proba(Xt_val)[:, 1]


def _paired_bootstrap(per_seed: list[dict], n_boot: int, rng: np.random.Generator) -> dict:
    """Cluster-bootstrap by channel within each seed's validation fold, then
    average the 5 seed-level AUC differences per draw.

    Videos are not exchangeable: they cluster by channel (shared subscriber
    count, audience, production style -- the same reason splits are
    channel-grouped in the first place, see ytdiag/split.py). Resampling
    individual videos, as an earlier version of this script did, treats
    within-channel videos as independent draws and understates the true
    sampling variance. Here each draw resamples *channels* with replacement
    (same number of unique channels as the fold) and takes every video
    belonging to a sampled channel, so channel-level correlation is preserved
    in the resampling distribution.

    The reported one-sided value is the bootstrap-based proportion of draws
    at or below zero -- a descriptive tail probability under this resampling
    scheme, not a null-centered hypothesis-test p-value (the bootstrap
    distribution is centered on the observed point estimate, not on zero).
    Report it as such."""
    point_diffs = [roc_auc_score(s["y"], s["p_alt"]) - roc_auc_score(s["y"], s["p_ref"]) for s in per_seed]
    point_estimate = float(np.mean(point_diffs))

    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        seed_diffs = np.empty(len(per_seed), dtype=float)
        for i, s in enumerate(per_seed):
            unique_channels = s["unique_channels"]
            n_channels = len(unique_channels)
            for _ in range(50):  # retry until both classes are present
                sampled_channels = unique_channels[rng.integers(0, n_channels, size=n_channels)]
                sample = np.concatenate([s["by_channel"][c] for c in sampled_channels])
                if len(np.unique(s["y"][sample])) == 2:
                    break
            auc_ref = roc_auc_score(s["y"][sample], s["p_ref"][sample])
            auc_alt = roc_auc_score(s["y"][sample], s["p_alt"][sample])
            seed_diffs[i] = auc_alt - auc_ref
        draws[b] = float(np.mean(seed_diffs))

    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    p_not_better = float(np.mean(draws <= 0))
    return {
        "point_estimate_mean_auc_diff": point_estimate,
        "bootstrap_95ci": [float(ci_low), float(ci_high)],
        "bootstrap_tail_prob_not_better": p_not_better,
        "n_bootstrap": n_boot,
        "per_seed_point_diffs": point_diffs,
        "resampling_unit": "channel",
    }


def _load_run(name: str) -> dict:
    path = os.path.join(RUNS_DIR, name, "results.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _compare(df: pd.DataFrame, seeds: list[int], reference_runs: list[dict],
             augmented_runs: list[dict], augmented_groups: tuple, n_boot: int,
             rng: np.random.Generator) -> dict:
    per_seed = []
    for seed, ref_run, alt_run in zip(seeds, reference_runs, augmented_runs):
        outer = split_indices(df, seed=seed)
        ref_params = ref_run["tuned"]["models"]["xgboost"]["best_parameters"]
        alt_params = alt_run["tuned"]["models"]["xgboost"]["best_parameters"]
        p_ref = _fit_predict(df, ("meta", "sched"), seed, ref_params, outer["train"], outer["val"])
        p_alt = _fit_predict(df, augmented_groups, seed, alt_params, outer["train"], outer["val"])
        y_val = df.label.astype(int).to_numpy()[outer["val"]]
        channel_val = df.channel_id.to_numpy()[outer["val"]]
        unique_channels = np.unique(channel_val)
        by_channel = {c: np.flatnonzero(channel_val == c) for c in unique_channels}
        per_seed.append({
            "seed": seed, "y": y_val, "p_ref": p_ref, "p_alt": p_alt,
            "unique_channels": unique_channels, "by_channel": by_channel,
        })
    return _paired_bootstrap(per_seed, n_boot, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--rng-seed", type=int, default=2026)
    args = parser.parse_args()

    df_full = load_retrospective(args.data)
    df_full = df_full[df_full.label.notna()].reset_index(drop=True)
    rng = np.random.default_rng(args.rng_seed)

    audio = _load_run("audio_ablation")
    seeds = audio["seeds"]
    print(f"Audio: meta+sched vs meta+sched+aud, {len(seeds)} seeds, {args.n_boot} bootstrap draws/seed")
    audio_result = _compare(
        df_full, seeds, audio["reference_meta_sched"]["runs"], audio["full_audio_88col"]["runs"],
        ("meta", "sched", "aud"), args.n_boot, rng,
    )
    print(f"  point estimate: {audio_result['point_estimate_mean_auc_diff']:+.4f} AUC")
    print(f"  95% CI: [{audio_result['bootstrap_95ci'][0]:+.4f}, {audio_result['bootstrap_95ci'][1]:+.4f}]")
    print(f"  bootstrap tail prob (not better): {audio_result['bootstrap_tail_prob_not_better']:.4f}")

    visual = _load_run("visual_engineered_ablation")
    seeds_v = visual["seeds"]
    print(f"\nVisual (engineered): meta+sched vs meta+sched+vis, {len(seeds_v)} seeds, {args.n_boot} bootstrap draws/seed")
    visual_result = _compare(
        df_full, seeds_v, visual["reference_meta_sched"]["runs"], visual["full_visual_15col"]["runs"],
        ("meta", "sched", "vis"), args.n_boot, rng,
    )
    print(f"  point estimate: {visual_result['point_estimate_mean_auc_diff']:+.4f} AUC")
    print(f"  95% CI: [{visual_result['bootstrap_95ci'][0]:+.4f}, {visual_result['bootstrap_95ci'][1]:+.4f}]")
    print(f"  bootstrap tail prob (not better): {visual_result['bootstrap_tail_prob_not_better']:.4f}")

    output = {
        "method": (
            "Cluster bootstrap over channels within each of the 5 channel-grouped "
            "seeds' validation folds independently (videos are not resampled "
            "individually -- they cluster by channel, the same reason splits are "
            "channel-grouped); the mean AUC difference across seeds is recomputed "
            "per bootstrap draw. Hyperparameters are held fixed at their "
            "already-tuned values (no re-tuning inside the bootstrap) so this "
            "cannot manufacture significance by searching harder. The reported "
            "tail probability is descriptive (bootstrap distribution centered on "
            "the point estimate, not on zero), not a null-centered hypothesis-test "
            "p-value."
        ),
        "n_boot": args.n_boot,
        "rng_seed": args.rng_seed,
        "audio_isolated_vs_reference": audio_result,
        "visual_engineered_vs_reference": visual_result,
    }
    output_path = os.path.join(RUNS_DIR, "ablation_significance", "results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
