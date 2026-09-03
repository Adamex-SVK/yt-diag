"""Reproducible descriptive EDA for the retrospective feature modalities.

The core tables deliberately do not inspect the target label. Their purpose is
to describe coverage and distributions and detect format/category confounding.
The dashboard's view-association panels are the explicit exception: they use
only the seed-0, channel-grouped outer training partition and are descriptive,
never a full-dataset feature-selection scan.

Outputs (under ``02_Data/eda_features/``):
  dataset_by_category_format.csv
  visual_by_category_format.csv
  audio_speech_by_category_format.csv
  category_format_pivot.csv
  modality_coverage_by_category_format.csv
  interpretable_spearman_train_seed0.csv
  feature_outcome_associations_train_seed0.csv
  audio_spearman.csv
  eda_features_stats.json
  duration_by_category_format.png
  visual_profiles_by_format.png
  audio_speech_by_format.png
  correlation_heatmap_train_seed0.png
  audio_correlation_clustered.png
  modality_coverage_heatmap.png
  shortcut_scatterplots_train_seed0.png
  outcome_associations_train_seed0.png

Usage:
    .venv/bin/python 02_Data/eda_features.py
    .venv/bin/python 02_Data/eda_features.py --no-figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "03_Models"))
from ytdiag.adapters import load_retrospective  # noqa: E402

DATA_DIR = os.path.join(ROOT, "02_Data", "processed")
OUT_DIR = os.path.join(ROOT, "02_Data", "eda_features")
FORMATS = ("regular", "Shorts")

# A readable, interpretable subset for the ordinary correlation heatmap. The
# 88 eGeMAPS columns get their own clustered panel below; mixing them into this
# matrix would make both views unreadable and let one modality dominate it.
DASHBOARD_FEATURES = {
    "log1p views": "__log_views",
    "log1p subscribers": "__log_subscribers",
    "log1p age (days)": "__log_age",
    "log1p duration (s)": "__log_duration",
    "Shorts indicator": "meta__is_short",
    "thumbnail CCT (K)": "vis__thumb_cct",
    "frame CCT (K)": "vis__frames_mean_cct",
    "thumbnail brightness (crop)": "thumb_crop_brightness",
    "frame brightness": "vis__frames_mean_brightness",
    "thumbnail saturation": "vis__thumb_saturation",
    "thumbnail contrast": "vis__thumb_contrast",
    "thumbnail face present": "vis__thumb_has_face",
    "thumbnail face area": "vis__thumb_max_face_area_ratio",
    "frame face prevalence": "vis__frames_has_face_ratio",
    "pause ratio": "aud__pause_pause_ratio",
    "pauses/min": "pause_count_per_min",
    "sound level (dBp)": "aud__egemaps__equivalentSoundLevel_dBp",
    "loudness variability": "aud__egemaps__loudness_sma3_stddevNorm",
    "edited speech density": "speaking_rate_wpm",
}


def load(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Canonical table joined to cleaning and non-model derived diagnostics."""
    d = load_retrospective(data_dir)
    # The canonical adapter carries the target for modelling. Remove it before
    # any join, summary or helper can inspect it: this module is descriptive,
    # not a full-dataset feature-selection pass.
    d = d.drop(columns=["label"], errors="ignore")
    manifest = pd.read_csv(os.path.join(data_dir, "cleaning_manifest.csv"))
    derived = pd.read_csv(os.path.join(data_dir, "derived_features.csv"))
    man_cols = [
        "video_id", "age_days_at_collection", "transcript_usable",
        "transcript_words", "audio_present", "vis_cct_valid",
        "vis_cct_policy_valid", "vis_cct_thumbnail_valid",
        "vis_cct_frame_coverage", "vis_cct_frames_complete",
    ]
    d = d.merge(manifest[man_cols], on="video_id", how="left")
    # Canonical adapter calls this `meta__category`; derived CSV keeps the raw
    # folder name. Video ids are unique in the completed cohort, so merge by id
    # and retain the derived category only as a consistency audit.
    d = d.merge(derived, on="video_id", how="left", suffixes=("", "_derived"))
    mismatch = d["category"].notna() & (d["category"] != d["meta__category"])
    if mismatch.any():
        raise ValueError(f"derived category disagrees for {int(mismatch.sum())} videos")
    d["format"] = d["meta__is_short"].map({0.0: "regular", 1.0: "Shorts"})
    d["pause_count_per_min"] = (
        pd.to_numeric(d["aud__pause_pause_count"], errors="coerce")
        / (pd.to_numeric(d["meta__duration_sec"], errors="coerce") / 60.0)
    ).replace([np.inf, -np.inf], np.nan)
    return d


def _q(s: pd.Series, q: float) -> float | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return None if x.empty else float(x.quantile(q))


def _summary(s: pd.Series, prefix: str, mean: bool = False) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    out: dict[str, Any] = {
        f"{prefix}_n": int(len(x)),
        f"{prefix}_median": _q(x, 0.5),
        f"{prefix}_q1": _q(x, 0.25),
        f"{prefix}_q3": _q(x, 0.75),
    }
    if mean:
        out[f"{prefix}_mean"] = None if x.empty else float(x.mean())
    return out


def _groups(d: pd.DataFrame):
    """Overall-format and category-format groups; never misleading pooled only."""
    known = d[d["format"].notna()]
    for fmt in FORMATS:
        yield "all", fmt, known[known["format"] == fmt]
    for (cat, fmt), g in known.groupby(["meta__category", "format"], sort=True):
        yield str(cat), str(fmt), g


def dataset_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, fmt, g in _groups(d):
        row = {"category": cat, "format": fmt, "n": int(len(g))}
        for col, name in [
            ("meta__duration_sec", "duration_sec"),
            ("view_count", "views"),
            ("meta__channel_follower_count", "subscribers"),
            ("age_days_at_collection", "age_days"),
        ]:
            row.update(_summary(g[col], name))
        row["transcript_usable_pct"] = float(g["transcript_usable"].mean() * 100)
        row["audio_present_pct"] = float(g["audio_present"].mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def visual_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("vis__thumb_cct", "thumb_cct_k", True),
        ("vis__frames_mean_cct", "frames_cct_k", True),
        ("vis__thumb_brightness", "thumb_brightness_raw", False),
        ("thumb_crop_brightness", "thumb_brightness_crop", False),
        ("vis__frames_mean_brightness", "frames_brightness", False),
        ("vis__thumb_saturation", "thumb_saturation", False),
        ("vis__frames_mean_saturation", "frames_saturation", False),
        ("vis__thumb_contrast", "thumb_contrast", False),
        ("vis__frames_mean_contrast", "frames_contrast", False),
        ("vis__frames_has_face_ratio", "frames_face_ratio", False),
    ]
    for cat, fmt, g in _groups(d):
        row = {"category": cat, "format": fmt, "n": int(len(g))}
        for col, name, include_mean in specs:
            row.update(_summary(g[col], name, mean=include_mean))
        face = pd.to_numeric(g["vis__thumb_has_face"], errors="coerce")
        row["thumb_face_measured_n"] = int(face.notna().sum())
        row["thumb_face_present_pct"] = (None if face.notna().sum() == 0
                                          else float(face.mean() * 100))
        with_face = g[face == 1]
        for col, name in [
            ("vis__thumb_face_count", "thumb_face_count_given_face"),
            ("vis__thumb_max_face_area_ratio", "thumb_max_face_area_given_face"),
            ("vis__thumb_face_centrality", "thumb_face_centrality_given_face"),
        ]:
            row.update(_summary(with_face[col], name))
        row["cct_policy_valid_pct"] = float(g["vis_cct_policy_valid"].mean() * 100)
        row["cct_feature_valid_pct"] = float(g["vis_cct_valid"].mean() * 100)
        row["cct_thumbnail_valid_pct"] = float(g["vis_cct_thumbnail_valid"].mean() * 100)
        row["cct_frames_complete_pct"] = float(g["vis_cct_frames_complete"].mean() * 100)
        row.update(_summary(g["vis_cct_frame_coverage"], "cct_frame_coverage"))
        row["heuristic_thumb_crop_pct"] = float(g["thumb_was_cropped"].mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def audio_speech_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("aud__pause_pause_ratio", "pause_ratio"),
        ("pause_count_per_min", "pauses_per_min"),
        ("aud__pause_mean_pause_sec", "mean_pause_sec"),
        ("aud__egemaps__equivalentSoundLevel_dBp", "sound_level_dbp"),
        ("aud__egemaps__loudness_sma3_amean", "loudness_mean"),
        ("aud__egemaps__loudness_sma3_stddevNorm", "loudness_variability"),
        ("speaking_rate_wpm", "edited_speech_density_wpm"),
    ]
    for cat, fmt, g in _groups(d):
        row = {"category": cat, "format": fmt, "n": int(len(g))}
        for col, name in specs:
            row.update(_summary(g[col], name, mean=True))
        pause_count = pd.to_numeric(g["aud__pause_pause_count"], errors="coerce")
        row["pause_measured_n"] = int(pause_count.notna().sum())
        measured_pause_count = pause_count.dropna()
        row["zero_pause_pct"] = (None if measured_pause_count.empty
                                  else float((measured_pause_count == 0).mean() * 100))
        row["audio_present_pct"] = float(g["audio_present"].mean() * 100)
        row["speech_density_coverage_pct"] = float(g["speaking_rate_wpm"].notna().mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def training_partition(d: pd.DataFrame, data_dir: str = DATA_DIR, seed: int = 0) -> pd.DataFrame:
    """Return only one pre-specified channel-grouped outer training partition.

    The descriptive loader has already removed ``label``. We briefly load the
    canonical labelled table only to recover the exact split membership, then
    select those video ids from the richer descriptive frame. Neither outer
    validation nor outer test rows are inspected by an outcome plot.
    """
    from ytdiag.split import split_indices

    canonical = load_retrospective(data_dir)
    labelled = canonical[canonical["label"].notna()].reset_index(drop=True)
    split = split_indices(labelled, seed=seed)
    train_ids = set(labelled.iloc[split["train"]]["video_id"].astype(str))
    selected = d[d["video_id"].astype(str).isin(train_ids)].copy().reset_index(drop=True)
    if len(selected) != len(split["train"]):
        raise ValueError("descriptive and canonical tables disagree on seed-0 training video ids")
    return selected


def dashboard_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Named numeric columns for readable rank correlations and scatterplots."""
    frame = d.copy()

    def log_nonnegative(column: str) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        return np.log1p(values.where(values >= 0))

    frame["__log_views"] = log_nonnegative("view_count")
    frame["__log_subscribers"] = log_nonnegative("meta__channel_follower_count")
    frame["__log_age"] = log_nonnegative("age_days_at_collection")
    frame["__log_duration"] = log_nonnegative("meta__duration_sec")
    return pd.DataFrame({
        name: pd.to_numeric(frame[column], errors="coerce")
        for name, column in DASHBOARD_FEATURES.items()
    })


def correlation_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman correlations and their non-missing pair counts."""
    values = dashboard_frame(d)
    correlation = values.corr(method="spearman", min_periods=30)
    present = values.notna().astype(np.int64)
    pair_counts = present.T.dot(present)
    return correlation, pair_counts


def modality_coverage_table(d: pd.DataFrame) -> pd.DataFrame:
    """Coverage percentages by format and category × format."""
    rows = []
    for category, fmt, group in _groups(d):
        thumbnail = group["asset__thumbnail_path"].map(
            lambda value: isinstance(value, str) and bool(value) and os.path.exists(value)
        )
        frames = group["asset__frames_dir"].map(
            lambda value: isinstance(value, str) and bool(value) and os.path.isdir(value)
        )
        columns = {
            "thumbnail_pct": thumbnail,
            "frames_pct": frames,
            "audio_pct": group["audio_present"].fillna(False).astype(bool),
            "usable_transcript_pct": group["transcript_usable"].fillna(False).astype(bool),
            "valid_cct_pct": group["vis_cct_valid"].fillna(False).astype(bool),
            "pause_measurement_pct": group["aud__pause_pause_count"].notna(),
            "speech_density_pct": group["speaking_rate_wpm"].notna(),
            "face_measurement_pct": group["vis__thumb_has_face"].notna(),
        }
        row: dict[str, Any] = {"category": category, "format": fmt, "n": int(len(group))}
        row.update({name: float(values.mean() * 100) for name, values in columns.items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    category_order = {"all": 0}
    category_order.update({value: index + 1 for index, value in enumerate(
        sorted(result.loc[result["category"] != "all", "category"].unique())
    )})
    format_order = {value: index for index, value in enumerate(FORMATS)}
    result["__category_order"] = result["category"].map(category_order)
    result["__format_order"] = result["format"].map(format_order)
    result = result.sort_values(["__category_order", "__format_order"]).drop(
        columns=["__category_order", "__format_order"]
    )
    numeric = [column for column in result if column.endswith("_pct")]
    result[numeric] = result[numeric].round(1)
    return result.reset_index(drop=True)


def category_format_pivot(d: pd.DataFrame) -> pd.DataFrame:
    """One compact, genuinely pivoted category × format summary table."""
    coverage = modality_coverage_table(d)
    rows = []
    known = d[d["format"].notna()]
    for (category, fmt), group in known.groupby(["meta__category", "format"], sort=True):
        face = pd.to_numeric(group["vis__thumb_has_face"], errors="coerce")
        row = {
            "category": str(category), "format": str(fmt), "n": int(len(group)),
            "views_median": _q(group["view_count"], 0.5),
            "subscribers_median": _q(group["meta__channel_follower_count"], 0.5),
            "duration_sec_median": _q(group["meta__duration_sec"], 0.5),
            "thumb_brightness_median": _q(group["thumb_crop_brightness"], 0.5),
            "frames_brightness_median": _q(group["vis__frames_mean_brightness"], 0.5),
            "thumb_cct_k_median": _q(group["vis__thumb_cct"], 0.5),
            "thumb_face_present_pct": float(face.mean() * 100) if face.notna().any() else None,
            "pause_ratio_median": _q(group["aud__pause_pause_ratio"], 0.5),
            "speech_density_wpm_median": _q(group["speaking_rate_wpm"], 0.5),
        }
        match = coverage[(coverage["category"] == str(category)) & (coverage["format"] == str(fmt))]
        for column in ("thumbnail_pct", "audio_pct", "usable_transcript_pct", "valid_cct_pct"):
            row[column] = float(match.iloc[0][column]) if len(match) else None
        rows.append(row)
    long = pd.DataFrame(rows)
    metrics = [column for column in long.columns if column not in ("category", "format")]
    wide = long.pivot_table(index="category", columns="format", values=metrics, aggfunc="first")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([metrics, FORMATS]))
    wide.columns = [f"{metric}__{fmt.lower()}" for metric, fmt in wide.columns]
    wide = wide.reset_index()
    numeric = [column for column in wide if column != "category"]
    wide[numeric] = wide[numeric].round(2)
    for fmt in FORMATS:
        column = f"n__{fmt.lower()}"
        wide[column] = wide[column].astype("Int64")
    return wide


def audio_correlation_table(d: pd.DataFrame) -> pd.DataFrame:
    """Spearman matrix for eGeMAPS only; pauses belong in the small matrix."""
    columns = sorted(column for column in d if column.startswith("aud__egemaps__"))
    numeric = d[columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(method="spearman", min_periods=30)
    correlation.index = [name.removeprefix("aud__egemaps__") for name in correlation.index]
    correlation.columns = [name.removeprefix("aud__egemaps__") for name in correlation.columns]
    return correlation


def outcome_association_table(d: pd.DataFrame) -> pd.DataFrame:
    """Training-only Spearman associations, pooled and important subgroups."""
    values = dashboard_frame(d)
    outcome = values["log1p views"]

    def rho(rows: pd.Series, feature: str) -> tuple[float | None, int]:
        pair = pd.concat([outcome[rows], values.loc[rows, feature]], axis=1).dropna()
        if len(pair) < 20 or pair.iloc[:, 1].nunique() < 2:
            return None, int(len(pair))
        return float(pair.corr(method="spearman").iloc[0, 1]), int(len(pair))

    scopes: dict[str, pd.Series] = {
        "overall": pd.Series(True, index=d.index),
        "regular": d["format"].eq("regular"),
        "shorts": d["format"].eq("Shorts"),
    }
    for category in sorted(d["meta__category"].dropna().astype(str).unique()):
        scopes[f"category_{category}"] = d["meta__category"].astype(str).eq(category)
    rows = []
    for feature in values.columns:
        if feature == "log1p views":
            continue
        row: dict[str, Any] = {"feature": feature}
        for scope, mask in scopes.items():
            value, n = rho(mask, feature)
            row[f"rho_{scope}"] = value
            row[f"n_{scope}"] = n
        regular, shorts = row["rho_regular"], row["rho_shorts"]
        row["format_sign_reversal"] = bool(
            regular is not None and shorts is not None and np.sign(regular) != np.sign(shorts)
        )
        overall = row["rho_overall"]
        row["pooled_vs_format_reversal"] = bool(
            overall is not None and any(
                value is not None and np.sign(overall) != np.sign(value)
                for value in (regular, shorts)
            )
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rho_overall", key=lambda value: value.abs(), ascending=False)


def _format_auc(d: pd.DataFrame, col: str) -> float | None:
    from sklearn.metrics import roc_auc_score
    x = d[[col, "meta__is_short"]].dropna()
    if x.empty or x["meta__is_short"].nunique() < 2:
        return None
    auc = float(roc_auc_score(x["meta__is_short"], x[col]))
    return max(auc, 1.0 - auc)  # direction-free separability


def cct_provenance(data_dir: str) -> dict[str, Any]:
    versions: dict[str, int] = {}
    methods: dict[str, int] = {}
    valid_frames, total_frames, valid_thumbs = 0, 0, 0
    files = glob.glob(os.path.join(data_dir, "*", "*", "visual_features.json"))
    for path in files:
        with open(path, encoding="utf-8") as f:
            vis = json.load(f)
        versions[str(vis.get("cct_version"))] = versions.get(str(vis.get("cct_version")), 0) + 1
        methods[str(vis.get("cct_method"))] = methods.get(str(vis.get("cct_method")), 0) + 1
        valid_thumbs += int(bool(vis.get("cct_thumbnail_valid")))
        valid_frames += int(vis.get("cct_frames_valid") or 0)
        total_frames += int(vis.get("cct_frames_total") or 0)
    return {
        "visual_files": len(files), "versions": versions, "methods": methods,
        "valid_thumbnail_cct": valid_thumbs,
        "valid_frame_cct": valid_frames, "total_frames": total_frames,
        "frame_cct_valid_pct": 100 * valid_frames / total_frames if total_frames else None,
    }


def make_figures(d: pd.DataFrame, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    known = d[d["format"].notna()].copy()
    categories = sorted(known["meta__category"].dropna().unique())
    positions, values, labels, colours = [], [], [], []
    pos = 1
    for cat in categories:
        for fmt, colour in [("regular", "#4C78A8"), ("Shorts", "#F58518")]:
            x = pd.to_numeric(known.loc[(known.meta__category == cat) &
                                        (known["format"] == fmt), "meta__duration_sec"],
                              errors="coerce").dropna()
            if len(x):
                positions.append(pos); values.append(np.log10(x)); labels.append(f"{cat}\n{fmt}"); colours.append(colour)
            pos += 1
        pos += 0.5
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False)
    for box, colour in zip(bp["boxes"], colours): box.set_facecolor(colour)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel("log10 duration (seconds)")
    ax.set_title("Video length differs by both category and format")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "duration_by_category_format.png"), dpi=180); plt.close(fig)

    fmt_rows = visual_table(d)
    fmt_rows = fmt_rows[fmt_rows.category == "all"].set_index("format")
    metrics = ["thumb_cct_k_median", "frames_cct_k_median", "thumb_brightness_raw_median",
               "thumb_brightness_crop_median", "frames_brightness_median", "thumb_face_present_pct"]
    titles = ["Thumbnail CCT (K)", "Frame CCT (K)", "Raw thumb brightness", "Crop brightness",
              "Frame brightness", "Thumbnail with face (%)"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [fmt_rows.loc[f, metric] for f in FORMATS]
        ax.bar(FORMATS, vals, color=["#4C78A8", "#F58518"]); ax.set_title(title)
    fig.suptitle("Visual profiles must be interpreted within video format")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "visual_profiles_by_format.png"), dpi=180); plt.close(fig)

    aud_rows = audio_speech_table(d)
    aud_rows = aud_rows[aud_rows.category == "all"].set_index("format")
    metrics = ["pause_ratio_median", "pauses_per_min_median", "sound_level_dbp_median",
               "loudness_variability_median", "edited_speech_density_wpm_median", "zero_pause_pct"]
    titles = ["Pause ratio", "Pauses/min", "Equivalent level (dBp)", "Loudness variability",
              "Edited speech density (wpm)", "Zero detected pauses (%)"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [aud_rows.loc[f, metric] for f in FORMATS]
        ax.bar(FORMATS, vals, color=["#4C78A8", "#F58518"]); ax.set_title(title)
    fig.suptitle("Audio and transcript measurements differ sharply by format")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "audio_speech_by_format.png"), dpi=180); plt.close(fig)


def make_dashboard_figures(
    d: pd.DataFrame,
    train: pd.DataFrame,
    correlation: pd.DataFrame,
    audio_correlation: pd.DataFrame,
    coverage: pd.DataFrame,
    associations: pd.DataFrame,
    out_dir: str,
) -> None:
    """Correlation, missingness, shortcut and outcome-association panels."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    # Readable lower-triangle matrix. Correlations involving log views are
    # training-partition-only like the rest of this matrix.
    matrix = correlation.to_numpy(float)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    shown = np.ma.masked_where(mask, matrix)
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    labels = list(correlation.columns)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for row in range(len(labels)):
        for column in range(row + 1):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(value) > 0.55 else "black")
    ax.set_title("Interpretable-feature Spearman correlations\nseed-0 channel-grouped training partition only")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correlation_heatmap_train_seed0.png"), dpi=190)
    plt.close(fig)

    # Cluster |rho| so redundant acoustic families appear as blocks. The
    # signed values retain inverse relationships inside those blocks.
    absolute = audio_correlation.abs().fillna(0.0).to_numpy(float).copy()
    np.fill_diagonal(absolute, 1.0)
    distance = np.clip(1.0 - absolute, 0.0, 1.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    clustered = audio_correlation.iloc[order, order]
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(clustered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    step = max(1, len(clustered) // 14)
    ticks = np.arange(0, len(clustered), step)

    def short_audio(name: str) -> str:
        return (name.replace("_sma3nz", "").replace("_sma3", "")
                    .replace("_amean", " mean").replace("_stddevNorm", " sd"))

    tick_labels = [short_audio(clustered.index[index]) for index in ticks]
    ax.set_xticks(ticks, tick_labels, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(ticks, tick_labels, fontsize=6)
    ax.set_title(f"Clustered eGeMAPS correlation matrix ({len(clustered)} features)\n"
                 "clustering distance = 1 − |Spearman ρ|")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "audio_correlation_clustered.png"), dpi=190)
    plt.close(fig)

    coverage_columns = [
        "thumbnail_pct", "frames_pct", "audio_pct", "usable_transcript_pct",
        "valid_cct_pct", "pause_measurement_pct", "speech_density_pct",
        "face_measurement_pct",
    ]
    coverage_labels = [
        "thumbnail", "frames", "audio", "usable transcript", "valid CCT",
        "pause measurement", "speech density", "face measurement",
    ]
    coverage_plot = coverage.copy()
    coverage_plot["scope"] = coverage_plot["category"] + " · " + coverage_plot["format"]
    values = coverage_plot[coverage_columns].to_numpy(float)
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(coverage_labels)), coverage_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(coverage_plot)), coverage_plot["scope"])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.0f}%", ha="center", va="center",
                    fontsize=7, color="white" if values[row, column] > 68 else "black")
    ax.set_title("Modality and measurement coverage by category and format")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="rows with measurement (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "modality_coverage_heatmap.png"), dpi=190)
    plt.close(fig)

    # The two strongest known shortcuts, faceted so pooled Simpson effects are
    # visible rather than silently averaged away.
    categories = sorted(train["meta__category"].dropna().astype(str).unique())
    fig, axes = plt.subplots(2, len(categories), figsize=(4 * len(categories), 7), sharey=True)
    colours = {"regular": "#4C78A8", "Shorts": "#F58518"}
    for column, category in enumerate(categories):
        group = train[train["meta__category"].astype(str) == category]
        y = np.log1p(pd.to_numeric(group["view_count"], errors="coerce"))
        x_specs = [
            (np.log1p(pd.to_numeric(group["meta__channel_follower_count"], errors="coerce")),
             "log1p subscribers"),
            (np.log1p(pd.to_numeric(group["age_days_at_collection"], errors="coerce")),
             "log1p age at collection (days)"),
        ]
        for row, (x, xlabel) in enumerate(x_specs):
            ax = axes[row, column]
            for fmt in FORMATS:
                choose = group["format"].eq(fmt) & x.notna() & y.notna()
                ax.scatter(x[choose], y[choose], s=11, alpha=0.35, color=colours[fmt],
                           edgecolors="none", label=fmt)
            pair = pd.concat([x, y], axis=1).dropna()
            rho = pair.corr(method="spearman").iloc[0, 1] if len(pair) >= 20 else np.nan
            ax.text(0.03, 0.95, f"ρ={rho:.2f}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlabel(xlabel)
            if column == 0:
                ax.set_ylabel("log1p views")
            if row == 0:
                ax.set_title(category)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=2, frameon=False)
    fig.suptitle("View-count shortcuts differ across category and format\n"
                 "seed-0 channel-grouped training partition only", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(os.path.join(out_dir, "shortcut_scatterplots_train_seed0.png"), dpi=190,
                bbox_inches="tight")
    plt.close(fig)

    ordered = associations.sort_values("rho_overall", key=lambda values: values.abs())
    y_positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 8))
    for column, label, colour, marker in [
        ("rho_overall", "pooled", "#333333", "o"),
        ("rho_regular", "regular", "#4C78A8", "s"),
        ("rho_shorts", "Shorts", "#F58518", "^"),
    ]:
        ax.scatter(ordered[column], y_positions, label=label, color=colour, marker=marker, s=34)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_yticks(y_positions, ordered["feature"])
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ with log1p views")
    ax.set_title("Feature–outcome associations can reverse by format\n"
                 "seed-0 channel-grouped training partition; descriptive only")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "outcome_associations_train_seed0.png"), dpi=190)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = load(args.data_dir)
    train = training_partition(d, args.data_dir, seed=0)
    correlation, pair_counts = correlation_tables(train)
    coverage = modality_coverage_table(d)
    pivot = category_format_pivot(d)
    audio_correlation = audio_correlation_table(d)
    associations = outcome_association_table(train)
    tables = {
        "dataset_by_category_format.csv": dataset_table(d),
        "visual_by_category_format.csv": visual_table(d),
        "audio_speech_by_category_format.csv": audio_speech_table(d),
        "category_format_pivot.csv": pivot,
        "modality_coverage_by_category_format.csv": coverage,
        "interpretable_spearman_train_seed0.csv": correlation,
        "interpretable_spearman_pair_counts_train_seed0.csv": pair_counts,
        "feature_outcome_associations_train_seed0.csv": associations,
        "audio_spearman.csv": audio_correlation,
    }
    indexed_matrices = {
        "interpretable_spearman_train_seed0.csv",
        "interpretable_spearman_pair_counts_train_seed0.csv",
        "audio_spearman.csv",
    }
    for name, frame in tables.items():
        frame.to_csv(
            os.path.join(args.out_dir, name), index=name in indexed_matrices,
            index_label="feature" if name in indexed_matrices else None,
            lineterminator="\n",
        )

    audio_values = audio_correlation.abs().to_numpy(float)
    audio_pairs = audio_values[np.triu_indices_from(audio_values, k=1)]
    strongest = associations.iloc[0]

    stats = {
        "n_videos": int(len(d)),
        "n_unknown_format": int(d["format"].isna().sum()),
        "dashboard": {
            "correlation": "pairwise Spearman; seed-0 channel-grouped training partition only",
            "training_seed": 0,
            "training_rows": int(len(train)),
            "training_channels": int(train["channel_id"].nunique()),
            "interpretable_features": list(DASHBOARD_FEATURES),
            "strongest_absolute_pooled_view_association": {
                "feature": strongest["feature"],
                "rho": float(strongest["rho_overall"]),
            },
            "egemaps_features": int(len(audio_correlation)),
            "egemaps_pairs_abs_rho_over_0_95": int(np.nansum(audio_pairs > 0.95)),
            "format_sign_reversals": int(associations["format_sign_reversal"].sum()),
            "pooled_vs_format_sign_reversals": int(associations["pooled_vs_format_reversal"].sum()),
            "outcome_panels_policy": "descriptive only; never used to select features or tune models",
        },
        "cct": cct_provenance(args.data_dir),
        "thumbnail_format_auc": {
            "raw_brightness_direction_free": _format_auc(d, "vis__thumb_brightness"),
            "heuristic_crop_brightness_direction_free": _format_auc(d, "thumb_crop_brightness"),
        },
        "scope": "core tables are label-blind; view associations use only the seed-0 channel-grouped training partition and are descriptive, never feature-selection evidence",
        "selected_for_eda": [
            "duration by category and format", "CCT with validity coverage",
            "brightness/saturation/contrast", "face presence and conditional geometry",
            "pause diagnostics", "whole-track acoustic level and variability",
            "edited transcript word density", "mean frame-sampling interval",
            "category-format pivot table", "modality coverage heatmap",
            "interpretable Spearman heatmap on the seed-0 training partition",
            "clustered eGeMAPS redundancy heatmap",
            "training-only shortcut scatterplots and outcome associations",
        ],
        "not_selected": {
            "music amount": "caption tags and VAD do not quantify music reliably",
            "cutting pace": "20 sparse frames miss cuts and confound change with sample spacing",
            "single-scene": "zero threshold crossings among sparse frames does not establish one scene",
            "audio quality score": "eGeMAPS describes mixed speech/music/ambience, not recording quality",
            "full-dataset label association scans": "feature selection on the full dataset would contaminate evaluation; dashboard associations are training-only and descriptive",
        },
        "known_measurement_limits": {
            "pauses": "VAD can classify music as voice; interpret as VAD silence, especially for Shorts",
            "speaking_rate": "cleaned transcript words per video minute; edited speech density, not articulation rate",
            "thumbnail_crop": "column-variance heuristic, not a verified pillarbox detector",
            "faces": "OpenCV Haar detections, not identity, gaze, or composition ground truth",
        },
    }
    with open(os.path.join(args.out_dir, "eda_features_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    if not args.no_figures:
        make_figures(d, args.out_dir)
        make_dashboard_figures(
            d, train, correlation, audio_correlation, coverage, associations, args.out_dir,
        )
    print(f"wrote {len(tables)} tables, stats, and " + ("no figures" if args.no_figures else "8 figures"))
    print(f"to {args.out_dir}")


if __name__ == "__main__":
    main()
