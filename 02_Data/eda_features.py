"""Reproducible descriptive EDA for the retrospective feature modalities.

This script deliberately does not inspect the target label. Its purpose is to
describe coverage and distributions, detect format/category confounding, and
decide whether a measurement is fit for later modelling. Outcome-driven EDA
must be performed on training data only after the split is frozen.

Outputs (under ``02_Data/eda_features/``):
  dataset_by_category_format.csv
  visual_by_category_format.csv
  audio_speech_by_category_format.csv
  eda_features_stats.json
  duration_by_category_format.png
  visual_profiles_by_format.png
  audio_speech_by_format.png

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = load(args.data_dir)
    tables = {
        "dataset_by_category_format.csv": dataset_table(d),
        "visual_by_category_format.csv": visual_table(d),
        "audio_speech_by_category_format.csv": audio_speech_table(d),
    }
    for name, frame in tables.items():
        frame.to_csv(os.path.join(args.out_dir, name), index=False, lineterminator="\n")

    stats = {
        "n_videos": int(len(d)),
        "n_unknown_format": int(d["format"].isna().sum()),
        "cct": cct_provenance(args.data_dir),
        "thumbnail_format_auc": {
            "raw_brightness_direction_free": _format_auc(d, "vis__thumb_brightness"),
            "heuristic_crop_brightness_direction_free": _format_auc(d, "thumb_crop_brightness"),
        },
        "scope": "descriptive only; target label was dropped before analysis and never inspected",
        "selected_for_eda": [
            "duration by category and format", "CCT with validity coverage",
            "brightness/saturation/contrast", "face presence and conditional geometry",
            "pause diagnostics", "whole-track acoustic level and variability",
            "edited transcript word density", "mean frame-sampling interval",
        ],
        "not_selected": {
            "music amount": "caption tags and VAD do not quantify music reliably",
            "cutting pace": "20 sparse frames miss cuts and confound change with sample spacing",
            "single-scene": "zero threshold crossings among sparse frames does not establish one scene",
            "audio quality score": "eGeMAPS describes mixed speech/music/ambience, not recording quality",
            "label association scans": "feature selection on the full dataset would contaminate evaluation; use training data only",
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
    print(f"wrote {len(tables)} tables, stats, and " + ("no figures" if args.no_figures else "3 figures"))
    print(f"to {args.out_dir}")


if __name__ == "__main__":
    main()
