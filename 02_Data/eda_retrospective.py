"""
Exploratory data analysis of the CLEANED retrospective dataset -- the evidence
base for the label-design decisions recorded in 02_Data/eda.md.

Reads only committed artefacts (02_Data/processed/ via the canonical adapter +
cleaning_manifest.csv, and 02_Data/tracking/ for the live view curves) and
writes figures + a machine-readable stats dump. Nothing here mutates data.

The three questions this exists to answer, in order of stakes:

1. SHORTCUT CEILING -- how much of log(views) is predictable from the four
   confounds a model gets for free (subscriber count, video age at collection,
   duration, is_short)? Everything above that ceiling is what content features
   could plausibly add; everything below it a metadata baseline gets anyway.

2. AGE FLOOR -- compute_labels_v2.py excludes videos younger than
   --min-age-days (default 30), which would discard 46% of the collection.
   The v1 label needed an age floor because views-per-day decays with age.
   v2 ranks WITHIN age bands, so the floor's justification has to be re-earned:
   the test is whether a young video's rank is already informative, measured
   on the prospective panel's real twice-daily curves (rank_stability()), and
   whether age bands actually neutralise age (residual_age_corr()).

3. SHORTS -- 52% of the collection is Shorts, and Shorts take ~2.5x the median
   views of regular videos. Sharing a quartile cell, they would absorb the
   viral label; shorts_vs_regular() quantifies the over-representation.

Usage:
    .venv/bin/python 02_Data/eda_retrospective.py                 # figures + stats
    .venv/bin/python 02_Data/eda_retrospective.py --no-figures    # stats only (fast)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_Models"))
from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.features import select_columns  # noqa: E402

DATA_DIR = os.path.join(ROOT, "02_Data", "processed")
TRACKING_DIR = os.path.join(ROOT, "02_Data", "tracking")
OUT_DIR = os.path.join(ROOT, "02_Data", "eda")
CATEGORIES = ("comedy", "howto", "product_reviews", "vlogs")
# the four things a model knows without looking at any content
CONFOUNDS = ["meta__channel_follower_count", "age_days", "meta__duration_sec", "meta__is_short"]


def load(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Canonical table joined with the cleaning manifest's quality columns.

    Left join, so a video the manifest never saw keeps its canonical row with
    NaN quality columns instead of vanishing from the counts. Two derived
    columns are added: age_days (days between upload and collection, NaN where
    the published_at backfill never landed for that video) and log_views =
    log1p(view_count) -- views are power-law (pooled skew 24.4), so every
    regression here is on the log scale.
    """
    df = load_retrospective(data_dir)
    man = pd.read_csv(os.path.join(data_dir, "cleaning_manifest.csv"))
    keep = ["video_id", "age_days_at_collection", "duration_sec", "transcript_kind",
            "transcript_usable", "transcript_words", "no_speech", "audio_present",
            "thumb_subhd", "frames_portrait", "vis_cct_valid", "lang_declared"]
    d = df.merge(man[keep], on="video_id", how="left")
    d["age_days"] = pd.to_numeric(d.age_days_at_collection, errors="coerce")
    d["log_views"] = np.log1p(d.view_count)
    return d


# ----------------------------------------------------------- 1. shortcut ceiling

def shortcut_ceiling(d: pd.DataFrame, seed: int = 0) -> dict[str, dict[str, Any]]:
    """R^2 of log(views) from the confounds alone, out-of-fold. This is the
    floor any content model must beat to be saying anything about content.

    Keyed "all" plus one entry per category; a scope with fewer than 50 rows
    that have log_views is omitted entirely rather than reported noisily, so a
    missing category key means "not enough data", not "R^2 of zero". r2_oof is
    unitless and may go negative (predictions worse than the mean).
    spearman_<confound> is that confound's rank correlation with log(views) on
    its own, or None when fewer than 11 rows carry both values.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    out = {}
    for scope, sub in [("all", d)] + [(c, d[d.meta__category == c]) for c in CATEGORIES]:
        s = sub.dropna(subset=["log_views"])
        X = s[CONFOUNDS].astype(float)
        y = s.log_views.to_numpy()
        if len(s) < 50:
            continue
        # grouped by channel: without it, several videos of one channel land on
        # both sides of a fold and the ceiling is flattered
        cv = GroupKFold(n_splits=min(5, s.channel_id.nunique()))
        pred = cross_val_predict(HistGradientBoostingRegressor(random_state=seed),
                                 X, y, cv=cv, groups=s.channel_id)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[scope] = {"n": int(len(s)), "r2_oof": round(1 - ss_res / ss_tot, 4)}
        for c in CONFOUNDS:  # each confound alone, as a rank correlation
            x = s[[c, "log_views"]].dropna()
            out[scope][f"spearman_{c}"] = round(float(x.corr(method="spearman").iloc[0, 1]), 3) if len(x) > 10 else None
    return out


# ------------------------------------------------------------- 2. the age floor

def rank_stability(tracking_dir: str, max_day: int = 5) -> dict[str, Any]:
    """Does a young video's view RANK already predict its later rank? Measured
    on the prospective panel (twice-daily snapshots), which is the only place
    in this project where the same video is observed at two ages.

    Views are interpolated at exactly 24*n hours of age, and a video is dropped
    unless its snapshots bracket every day from 1 to max_day -- nothing is ever
    extrapolated, so n_videos is the fully-observed subset, not the cohort size.
    Only the labelled main arm (sampling_arm == "date_window") is ranked; the
    comparison arms are a different population. Per day, by_day["day<n>"] gives
    spearman_vs_final (rho against day max_day), viral_quartile_agreement (the
    fraction of a category's day-n top quartile still in its day-max_day top
    quartile, averaged over categories with >=20 videos, None if none qualify),
    and median_view_multiple_to_final (a ratio, ~2.0 at day 1 -- the rank settles
    within a day, the raw count does not, which is why the label must stay a
    within-band ranking). Stability past day 5 is untested: the cohort is not
    old enough yet.
    """
    snaps = pd.read_csv(os.path.join(tracking_dir, "video_snapshots.csv"))
    cohort = pd.read_csv(os.path.join(tracking_dir, "cohort.csv"))
    ok = snaps[snaps.status == "ok"].merge(
        cohort[["video_id", "category", "sampling_arm"]], on="video_id")
    ok = ok[ok.sampling_arm == "date_window"]  # main arm only: the labelled population

    def at(g, hours):
        g = g.sort_values("age_hours")
        if g.age_hours.max() < hours or g.age_hours.min() > hours:
            return np.nan  # never extrapolate
        return float(np.interp(hours, g.age_hours, g.view_count))

    days = list(range(1, max_day + 1))
    curves = ok.groupby("video_id").apply(
        lambda g: pd.Series({**{f"d{n}": at(g, 24 * n) for n in days},
                             "category": g.category.iloc[0]}), include_groups=False).dropna()
    res = {"n_videos": int(len(curves)), "max_day": max_day, "by_day": {}}
    for n in days[:-1]:
        rho = float(curves[f"d{n}"].corr(curves[f"d{max_day}"], method="spearman"))
        # would the top-quartile LABEL be the same if assigned at day n?
        agree = []
        for _, g in curves.groupby("category"):
            if len(g) < 20:
                continue
            k = max(1, len(g) // 4)
            agree.append(len(set(g[f"d{n}"].nlargest(k).index) & set(g[f"d{max_day}"].nlargest(k).index)) / k)
        res["by_day"][f"day{n}"] = {
            "spearman_vs_final": round(rho, 3),
            "viral_quartile_agreement": round(float(np.mean(agree)), 3) if agree else None,
            "median_view_multiple_to_final": round(float((curves[f"d{max_day}"] / curves[f"d{n}"]).median()), 2),
        }
    return res


def assign_label(g: pd.DataFrame, n_age_bands: int,
                 n_size_bands: int) -> tuple[pd.Series, list[int]]:
    """compute_labels_v2's assignment, vectorised: exactly the top floor(n/4)
    of each (age-band x size-band) cell by log views. Bands are equal-count
    quantile bands assigned BY VALUE (identical values never split), matching
    the labeler; ties at the cutoff break by video_id as it does.

    Returns (label, cell_sizes). label is a float Series of 1.0/0.0 aligned to
    g.index. cell_sizes is the number of videos in each cell, returned alongside
    because flooring distorts the intended 25% in small cells: max(1, ...)
    forces a cell of three or fewer to 33-100%, while cells of 5-7 fall to
    14-20% because int() rounds the quarter down. The label is only trustworthy
    where the cells stayed big, so the sizes have to be inspected with it.
    """
    ab = _bands_by_value(g.age_days, n_age_bands)
    sb = _bands_by_value(g.meta__channel_follower_count, n_size_bands)
    label = pd.Series(0, index=g.index, dtype=float)
    sizes = []
    for cell, idx in g.groupby([ab, sb]).groups.items():
        sub = g.loc[idx].sort_values(["log_views", "video_id"])
        sizes.append(len(sub))
        k = max(1, int(len(sub) * 0.25))
        label.loc[sub.index[len(sub) - k:]] = 1.0
    return label, sizes


def _bands_by_value(s, n_bands):
    """Equal-count bands where identical values always share a band -- the
    labeler's rule (API subscriber counts are rounded, so ties are common)."""
    order = np.sort(s.to_numpy(dtype=float))
    first_rank, seen = {}, 0
    for i, v in enumerate(order):
        if v not in first_rank:
            first_rank[v] = i
    n = len(order)
    return s.map(lambda v: first_rank[float(v)] * n_bands // n)


def label_leakage(d: pd.DataFrame, n_age_bands: int, n_size_bands: int = 4,
                  min_age: Optional[float] = None,
                  seed: int = 0) -> Optional[dict[str, Any]]:
    """Can the LABEL be predicted from the confounds alone? Cross-validated
    AUC, grouped by channel. 0.5 = the stratification did its job.

    This replaces averaging per-cell correlations, which becomes pure noise as
    cells shrink. Two separate concerns:
      auc_subs -- subscriber count IS a model input, so any signal here is a
                  direct shortcut the baseline will find immediately;
      auc_age  -- video age is NOT an input, but it leaks in through audio
                  timbre (r~0.36) and thumbnail resolution, so signal here is
                  reachable label noise rather than a clean confound.

    min_age is the age floor in DAYS at collection (None = no floor). Labelling
    is per category and a category under 40 usable rows is skipped; if that
    leaves nothing, the return is None rather than an empty result. min_cell /
    median_cell / cells_under_20 count VIDEOS per cell pooled across categories
    (the labeler warns under 20). An auc_* is None only where a config produced
    a single class. Format is deliberately not a stratifier here, so auc_is_short
    measures the leak that label_leakage_stratified() closes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    frames, all_sizes = [], []
    for cat in CATEGORIES:
        g = d[d.meta__category == cat].dropna(
            subset=["log_views", "age_days", "meta__channel_follower_count"]).copy()
        if min_age is not None:
            g = g[g.age_days >= min_age]
        if len(g) < 40:
            continue
        g["label"], sizes = assign_label(g, n_age_bands, n_size_bands)
        all_sizes += sizes
        frames.append(g)
    if not frames:
        return None
    lab = pd.concat(frames)

    def auc(features):
        X = lab[features].astype(float)
        y = lab.label.to_numpy()
        if len(np.unique(y)) < 2:
            return None
        cv = GroupKFold(n_splits=min(5, lab.channel_id.nunique()))
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=seed),
                              X, y, cv=cv, groups=lab.channel_id, method="predict_proba")[:, 1]
        return round(float(roc_auc_score(y, p)), 3)

    return {"config": f"{n_age_bands}age x {n_size_bands}size" + (f", min_age={min_age}" if min_age else ""),
            "n_labelled": int(len(lab)), "viral_rate": round(float(lab.label.mean()), 3),
            "min_cell": int(min(all_sizes)), "median_cell": int(np.median(all_sizes)),
            "cells_under_20": int(sum(1 for s in all_sizes if s < 20)),
            "auc_age": auc(["age_days"]), "auc_subs": auc(["meta__channel_follower_count"]),
            "auc_both": auc(["age_days", "meta__channel_follower_count"]),
            # format is NOT stratified here -- this is the leak the recommendation closes
            "auc_is_short": auc(["meta__is_short"]),
            "auc_all_confounds": auc(CONFOUNDS),
            "per_category_n": {c: int((lab.meta__category == c).sum()) for c in CATEGORIES}}


def label_configs(d: pd.DataFrame) -> list[dict[str, Any]]:
    """The configuration sweep the label decision is made from.

    Up to 12 label_leakage() results -- age floors of 0/7/14/30 days crossed
    with 4/6/8 age bands at a fixed 4 size bands -- with unlabellable configs
    dropped, so the list can be shorter than the grid and order is the only way
    to tell which config a row is (each result carries its own "config" string).
    """
    out = []
    for min_age in (None, 7, 14, 30):
        for nb in (4, 6, 8):
            r = label_leakage(d, nb, 4, min_age)
            if r:
                out.append(r)
    return out


def label_leakage_stratified(d: pd.DataFrame, n_age_bands: int, n_size_bands: int,
                             min_age: Optional[float] = None,
                             seed: int = 0) -> Optional[dict[str, Any]]:
    """Same, but with is_short as a THIRD stratifier dimension. Shorts and
    regular videos are two populations (every Short here is <=180s, and 96% of
    Shorts thumbnails are pillarboxed), so sharing a quartile cell hands Shorts
    a ~2x prior on the viral label -- which the vision branch can then read
    straight off the frame geometry without learning anything about content.

    Same keys as label_leakage() minus auc_both, plus shorts_over_representation:
    the Shorts share of the viral label divided by their share of the labelled
    population, so 1.0 means the format prior is gone and 2.0 means Shorts take
    twice their due (None if the labelled set contains no Shorts). min_age is in
    DAYS. The third dimension halves every cell, so the age and size band counts
    have to come down with it -- read median_cell and cells_under_20 before
    trusting a config. Returns None if no (category, format) group cleared 8
    videos.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    frames, sizes = [], []
    for cat in CATEGORIES:
        g = d[d.meta__category == cat].dropna(
            subset=["log_views", "age_days", "meta__channel_follower_count", "meta__is_short"]).copy()
        if min_age is not None:
            g = g[g.age_days >= min_age]
        for _, sub in g.groupby("meta__is_short"):
            if len(sub) < 8:
                continue
            sub = sub.copy()
            sub["label"], sz = assign_label(sub, n_age_bands, n_size_bands)
            sizes += sz
            frames.append(sub)
    if not frames:
        return None
    lab = pd.concat(frames)

    def auc(feats):
        X = lab[feats].astype(float)
        y = lab.label.to_numpy()
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y,
                              cv=GroupKFold(n_splits=5), groups=lab.channel_id,
                              method="predict_proba")[:, 1]
        return round(float(roc_auc_score(y, p)), 3)

    share = float((lab.meta__is_short == 1).mean())
    viral_short = float((lab[lab.label == 1].meta__is_short == 1).mean())
    return {"config": f"{n_age_bands}age x {n_size_bands}size x 2format"
                      + (f", min_age={min_age}" if min_age else ""),
            "n_labelled": int(len(lab)), "viral_rate": round(float(lab.label.mean()), 3),
            "min_cell": int(min(sizes)), "median_cell": int(np.median(sizes)),
            "cells_under_20": int(sum(1 for s in sizes if s < 20)),
            "auc_age": auc(["age_days"]), "auc_subs": auc(["meta__channel_follower_count"]),
            "auc_is_short": auc(["meta__is_short"]),
            "auc_all_confounds": auc(CONFOUNDS),
            "shorts_over_representation": round(viral_short / share, 2) if share else None,
            "per_category_n": {c: int((lab.meta__category == c).sum()) for c in CATEGORIES}}


def format_leakage(d: pd.DataFrame) -> dict[str, Any]:
    """How readable is is_short off the content itself? If the format bit is
    trivially recoverable AND carries a label prior, the vision branch scores
    without learning content -- the single most important thing to control.

    frames_portrait_matches_is_short and the duration_180s_rule recall and
    precision are fractions in [0, 1]; the spearman_* entries are rank
    correlations in [-1, 1], and two are strongly NEGATIVE on the current data
    (thumb_brightness -0.736 -- Shorts thumbnails are pillarboxed and dark --
    and thumb_saturation -0.145), so a reader assuming [0, 1] would misread the
    second-strongest format tell as near-zero. max_duration_of_a_short_sec is
    SECONDS and is a property of THIS collection rather than a rule:
    the prospective cohort contains a verified 250s Short, so the duration_180s
    recall/precision pair describes the retrospective set and must not be ported
    to the tracker as a classifier. spearman_* correlations are computed only
    for visual columns that exist in d, so a key can be absent.
    """
    out = {}
    g = d.dropna(subset=["meta__is_short"])
    fp = g.dropna(subset=["frames_portrait"])
    out["frames_portrait_matches_is_short"] = round(float((fp.frames_portrait == fp.meta__is_short).mean()), 4)
    dur = g.dropna(subset=["meta__duration_sec"])
    out["max_duration_of_a_short_sec"] = float(dur[dur.meta__is_short == 1].meta__duration_sec.max())
    rule = dur.meta__duration_sec <= 180
    tp = int((rule & (dur.meta__is_short == 1)).sum())
    out["duration_180s_rule"] = {
        "recall": round(tp / int((dur.meta__is_short == 1).sum()), 4),
        "precision": round(tp / max(1, int(rule.sum())), 4)}
    for c in ("vis__thumb_brightness", "vis__thumb_saturation", "vis__frames_mean_brightness"):
        if c in g:
            x = g[[c, "meta__is_short"]].dropna()
            out[f"spearman_{c}_vs_is_short"] = round(float(x.corr(method="spearman").iloc[0, 1]), 3)
    return out


# ---------------------------------------------------------------- 3. the Shorts

def shorts_vs_regular(d: pd.DataFrame) -> dict[str, Any]:
    """If Shorts and regular videos share a quartile cell, do Shorts absorb the
    viral label? Compared against their population share, per category.

    over_representation is the Shorts share of the category's top log-views
    quartile divided by their share of the category: 1.0 is proportionate, 2.0
    is twice their due, None if the category has no Shorts. Categories under 20
    usable videos are skipped. Median views are raw view counts, reported per
    category on purpose -- pooled, Shorts take ~2.5x the median, but that
    reverses inside howto, so the pooled ratio is a Simpson artifact. This is
    the unstratified quartile, i.e. the leak, not the recommended label.
    """
    out = {"population": {}, "viral_share": {}}
    for cat in CATEGORIES:
        g = d[(d.meta__category == cat) & d.meta__is_short.notna()].dropna(subset=["log_views"])
        if len(g) < 20:
            continue
        share = float((g.meta__is_short == 1).mean())
        k = max(1, len(g) // 4)
        top = g.nlargest(k, "log_views")
        out["population"][cat] = {"n": int(len(g)), "shorts": int((g.meta__is_short == 1).sum()),
                                  "regular": int((g.meta__is_short == 0).sum()),
                                  "shorts_share": round(share, 3),
                                  "median_views_shorts": float(g[g.meta__is_short == 1].view_count.median()),
                                  "median_views_regular": float(g[g.meta__is_short == 0].view_count.median())}
        out["viral_share"][cat] = {"viral_n": int(k),
                                   "shorts_in_viral": round(float((top.meta__is_short == 1).mean()), 3),
                                   "over_representation": round(float((top.meta__is_short == 1).mean()) / share, 2) if share else None}
    return out


# ------------------------------------------------------------------- 4. matrix

def feature_matrix_profile(d: pd.DataFrame) -> dict[str, Any]:
    """What the tabular baselines will actually consume.

    Missingness figures are fractions of rows in [0, 1], per prefix group and
    per column. constant_columns are numeric columns with at most one distinct
    non-null value -- they carry nothing and will not standardise.
    egemaps_pairs_abs_corr_over_0.95 counts unordered pairs, not columns.
    video_age_is_an_input is a guard expected to stay False: age_days is derived
    in load() for this analysis and carries no group prefix, so select_columns()
    should never hand it to a model -- age is a confound the label bands out,
    not a feature.
    """
    cols = select_columns(d, ("meta", "sched", "vis", "aud"))
    X = d[cols]
    num = X.select_dtypes(include=[np.number])
    miss = X.isna().mean()
    by_prefix = {}
    for p in ("meta", "sched", "vis", "aud"):
        c = [x for x in cols if x.startswith(p + "__")]
        if c:
            by_prefix[p] = {"n_columns": len(c), "mean_missing": round(float(X[c].isna().mean().mean()), 4),
                            "max_missing": round(float(X[c].isna().mean().max()), 4)}
    const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    egemaps = [c for c in num.columns if c.startswith("aud__egemaps__")]
    corr_pairs = 0
    if len(egemaps) > 1:
        cm = num[egemaps].corr().abs().to_numpy()
        iu = np.triu_indices_from(cm, k=1)
        corr_pairs = int((cm[iu] > 0.95).sum())
    return {"n_input_columns": len(cols), "by_prefix": by_prefix,
            "constant_columns": const,
            "columns_missing_over_50pct": [c for c in cols if miss[c] > 0.5],
            "egemaps_pairs_abs_corr_over_0.95": corr_pairs,
            "egemaps_n": len(egemaps),
            "video_age_is_an_input": any("age_days" == c for c in cols)}


# ------------------------------------------------------------------- 5. figures

def figures(d: pd.DataFrame, stats: dict[str, Any], out_dir: str) -> list[str]:
    """The four figures of 02_Data/eda.md; returns the paths written, in order.

    Reads the already-computed stats dict (keys rank_stability, label_configs,
    label_configs_format_stratified, shorts) instead of recomputing anything, so
    a figure cannot disagree with eda_stats.json. Forces the Agg backend before
    importing pyplot -- nothing here needs a display, and matplotlib is imported
    inside the function so --no-figures does not pay for it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    made = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for cat in CATEGORIES:
        g = d[d.meta__category == cat]
        axes[0].hist(np.log10(g.view_count.clip(lower=1)), bins=40, alpha=0.5, label=cat)
    axes[0].set_xlabel("log10(views)"); axes[0].set_ylabel("videos")
    axes[0].set_title("View distribution by category"); axes[0].legend(fontsize=8)
    axes[1].hist(np.log10(d.age_days.clip(lower=0.5)), bins=40, color="#444")
    for x, lab in [(np.log10(7), "7d"), (np.log10(30), "30d"), (np.log10(365), "1y")]:
        axes[1].axvline(x, ls="--", lw=1, color="crimson")
        axes[1].text(x, axes[1].get_ylim()[1] * 0.92, lab, color="crimson", fontsize=8, ha="center")
    axes[1].set_xlabel("log10(age at collection, days)"); axes[1].set_title("Video age at collection")
    fig.tight_layout(); p = os.path.join(out_dir, "views_and_age.png"); fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p)

    rs = stats["rank_stability"]["by_day"]
    days = sorted(int(k.replace("day", "")) for k in rs)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(days, [rs[f"day{n}"]["spearman_vs_final"] for n in days], "o-", label="rank corr vs day 5")
    ax.plot(days, [rs[f"day{n}"]["viral_quartile_agreement"] for n in days], "s-", label="viral quartile agreement")
    ax.set_ylim(0.8, 1.005); ax.set_xlabel("age of the early observation (days)")
    ax.set_title("View rank settles within days (prospective panel)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(out_dir, "rank_stability.png"); fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p)

    # What actually moves the needle is the FORMAT stratifier, not the age
    # floor or the band count -- so plot the confound that is left, on a scale
    # wide enough that near-chance differences look near-chance.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    ax = axes[0]
    for min_age in (None, 7, 14, 30):
        tag = f", min_age={min_age}" if min_age else ""
        sel = [c for c in stats["label_configs"]
               if (c["config"].endswith(tag) if min_age else "min_age" not in c["config"])]
        if not sel:
            continue
        xs = [int(c["config"].split("age")[0]) for c in sel]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array([c["auc_age"] for c in sel])[order], "o-",
                label=f"min_age={min_age or 0}d (n={sel[0]['n_labelled']})")
    ax.set_xlabel("age bands"); ax.set_ylabel("AUC of predicting the label from the confound")
    ax.set_title("Age floor and band count: all near chance", fontsize=10)

    ax = axes[1]
    fs = stats["label_configs_format_stratified"]
    unstrat = [c for c in stats["label_configs"] if c["config"] == "4age x 4size"]
    names, fmt_auc, all_auc = [], [], []
    if unstrat:
        names.append("4x4\n(no format stratum)")
        fmt_auc.append(unstrat[0]["auc_is_short"])
        all_auc.append(unstrat[0]["auc_all_confounds"])
    for c in fs:
        cfg = c["config"]
        bands = cfg.split(" x 2format")[0].replace("age x ", "x").replace("size", "")
        tail = "\n+ format, min-age 7" if "min_age" in cfg else "\n+ format stratum"
        names.append(bands.strip() + tail)
        fmt_auc.append(c["auc_is_short"]); all_auc.append(c["auc_all_confounds"])
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, fmt_auc, w, label="from is_short alone", color="#c44")
    ax.bar(x + w / 2, all_auc, w, label="from all 4 confounds", color="#48a")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_title("Stratifying on format is what removes the leak", fontsize=10)
    for a in axes:
        a.axhline(0.5, ls="--", lw=1, color="grey")
        a.set_ylim(0.40, 0.75); a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper right")
    axes[0].text(7.9, 0.512, "0.5 = confound neutralised", fontsize=7, color="grey", ha="right")
    fig.tight_layout(); p = os.path.join(out_dir, "label_configs.png"); fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p)

    fig, ax = plt.subplots(figsize=(7.2, 4))
    cats = list(stats["shorts"]["viral_share"])
    x = np.arange(len(cats)); w = 0.38
    ax.bar(x - w / 2, [stats["shorts"]["population"][c]["shorts_share"] for c in cats], w, label="share of videos")
    ax.bar(x + w / 2, [stats["shorts"]["viral_share"][c]["shorts_in_viral"] for c in cats], w, label="share of viral quartile")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8); ax.set_ylabel("fraction that are Shorts")
    ax.set_title("Shorts absorb more than their share of the viral label")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(out_dir, "shorts_viral_share.png"); fig.savefig(p, dpi=130); plt.close(fig)
    made.append(p)
    return made


def main() -> None:
    """CLI entry: compute every statistic, write eda_stats.json (plus the
    figures unless --no-figures) into --out-dir, and print the headline numbers.

    Read-only with respect to the dataset: it writes nothing outside --out-dir,
    so it is safe to re-run against the live cohort at any time.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    d = load(args.data_dir)
    stats = {
        "n_videos": int(len(d)),
        "by_category": {c: int((d.meta__category == c).sum()) for c in CATEGORIES},
        "shortcut_ceiling": shortcut_ceiling(d),
        "rank_stability": rank_stability(TRACKING_DIR),
        "label_configs": label_configs(d),
        # the recommendation of 02_Data/eda.md, plus the neighbours it was chosen over
        "label_configs_format_stratified": [c for c in (
            label_leakage_stratified(d, 3, 3, None),
            label_leakage_stratified(d, 3, 3, 7),
            label_leakage_stratified(d, 4, 4, None),
            label_leakage_stratified(d, 2, 2, None)) if c],
        "format_leakage": format_leakage(d),
        "shorts": shorts_vs_regular(d),
        "feature_matrix": feature_matrix_profile(d),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "eda_stats.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    sc = stats["shortcut_ceiling"]["all"]
    print(f"videos: {stats['n_videos']}  {stats['by_category']}")
    print(f"shortcut ceiling (confounds only, out-of-fold R^2 on log views): {sc['r2_oof']:.3f}")
    print(f"  per category: " + ", ".join(
        f"{c}={stats['shortcut_ceiling'][c]['r2_oof']:.3f}" for c in CATEGORIES if c in stats["shortcut_ceiling"]))
    rs = stats["rank_stability"]
    print(f"rank stability (n={rs['n_videos']}): " + ", ".join(
        f"d{k[-1]}->d{rs['max_day']} rho={v['spearman_vs_final']}, quartile {v['viral_quartile_agreement']:.0%}"
        for k, v in rs["by_day"].items()))
    print("label configs -- AUC of predicting the label from the confound alone (0.5 = neutralised):")
    print(f"  {'config':28s} {'n':>5s} {'viral':>6s} {'age':>6s} {'subs':>6s} {'both':>6s} {'<20':>4s}")
    for c in stats["label_configs"]:
        print(f"  {c['config']:28s} {c['n_labelled']:5d} {c['viral_rate']:6.3f} "
              f"{str(c['auc_age']):>6s} {str(c['auc_subs']):>6s} {str(c['auc_both']):>6s} {c['cells_under_20']:4d}")
    print("Shorts over-representation in the viral quartile: " + ", ".join(
        f"{c}={v['over_representation']}x" for c, v in stats["shorts"]["viral_share"].items()))
    print("with is_short as a third stratifier (the recommendation is the first row):")
    print(f"  {'config':34s} {'n':>5s} {'medCell':>8s} {'<20':>4s} {'AUCfmt':>7s} {'AUC4c':>6s} {'over':>6s}")
    for c in stats["label_configs_format_stratified"]:
        print(f"  {c['config']:34s} {c['n_labelled']:5d} {c['median_cell']:8d} {c['cells_under_20']:4d} "
              f"{c['auc_is_short']:7.3f} {c['auc_all_confounds']:6.3f} {c['shorts_over_representation']:5.2f}x")
    fl = stats["format_leakage"]
    print(f"format leakage: frames_portrait recovers is_short {fl['frames_portrait_matches_is_short']:.1%}; "
          f"duration<=180s rule recall {fl['duration_180s_rule']['recall']:.3f} / "
          f"precision {fl['duration_180s_rule']['precision']:.3f}")
    fm = stats["feature_matrix"]
    print(f"feature matrix: {fm['n_input_columns']} columns, {len(fm['constant_columns'])} constant, "
          f"{fm['egemaps_pairs_abs_corr_over_0.95']} eGeMAPS pairs |r|>0.95")

    if not args.no_figures:
        for p in figures(d, stats, args.out_dir):
            print(f"figure: {p}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
