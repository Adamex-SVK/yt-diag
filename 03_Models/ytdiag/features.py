"""Feature registry -- the code counterpart of 02_Data/FEATURES.md.

Canonical-table column naming: ``<group>__<name>``. Selecting features for a
model means selecting GROUPS; the registry refuses to hand out label-only
columns as inputs, which is the mistake that produced the v1 label bugs.

Groups (see FEATURES.md for definitions and roles):
  meta   structured metadata that exists at upload time (model input;
         `channel_follower_count` and `category` double as label-v2
         stratifiers -- see the FEATURES.md caveat)
  sched  publish-time features derived from a full timestamp (model input;
         retrospective rows have them only after backfill_published_at.py)
  vis    visual engineered features, thumbnail + frame aggregates
  aud    audio/prosodic engineered features (eGeMAPS 88 + pauses)
  asset  paths/text handed to the deep stage (thumbnail, frames dir,
         transcript, title/description) -- not tabular inputs themselves
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

GROUPS = ("meta", "sched", "vis", "aud", "asset")

# Columns that may NEVER be a model input (FEATURES.md role "label-only").
LABEL_ONLY = ("view_count", "like_count", "comment_count", "views_at_7d",
              "views_at_30d", "outcome_views", "label")
IDENTIFIERS = ("video_id", "channel_id", "dataset")

META_COLUMNS = [
    "meta__duration_sec", "meta__definition_hd", "meta__title_length",
    "meta__description_length", "meta__tag_count", "meta__channel_follower_count",
    "meta__caption_available",   # uploader-provided captions (contentDetails.caption)
    "meta__auto_captions",       # YouTube auto-captions existed at collection (retrospective only)
    "meta__category",
    # uploader-declared language (defaultAudioLanguage, else defaultLanguage);
    # categorical, often missing. Mostly a FILTER candidate (English-only
    # transcripts) rather than a predictor -- see FEATURES.md 1.
    "meta__language",
    # channel maturity: age at publish (days), upload count, first-upload flag
    # (prospective: count at first observation; retrospective: current count,
    # post-outcome -- coarse)
    "meta__channel_age_days", "meta__channel_video_count", "meta__is_first_upload",
    # definitive Shorts verdict from the /shorts/<id> URL test (yt_shorts.py):
    # 1/0/NaN. Primarily a FILTER (the main arms exclude Shorts); constant
    # within a Shorts-free cohort, so rarely a useful predictor.
    "meta__is_short",
]
SCHED_COLUMNS = ["sched__hour_sin", "sched__hour_cos", "sched__weekday", "sched__is_weekend"]
VIS_THUMB = ["cct", "brightness", "saturation", "contrast", "has_face",
             "face_count", "max_face_area_ratio", "face_centrality"]
VIS_FRAMES = ["frames_mean_cct", "frames_std_cct", "frames_mean_brightness",
              "frames_mean_saturation", "frames_mean_contrast",
              "frames_has_face_ratio", "frames_mean_max_face_area_ratio"]
VIS_COLUMNS = [f"vis__thumb_{c}" for c in VIS_THUMB] + [f"vis__{c}" for c in VIS_FRAMES]
AUD_PAUSES = ["pause_count", "total_pause_sec", "pause_ratio", "mean_pause_sec"]
# eGeMAPS names are taken from the data (88 columns, openSMILE naming).
ASSET_COLUMNS = ["asset__thumbnail_path", "asset__frames_dir", "asset__transcript_path",
                 "asset__title", "asset__description"]


def group_of(column: str) -> Optional[str]:
    """Group prefix of a canonical column, or None when it has no `<group>__`
    prefix. None is the load-bearing case: identifiers and label-only columns
    are deliberately left unprefixed, so they belong to no group and can never
    be picked up by a group selection."""
    return column.split("__", 1)[0] if "__" in column else None


def select_columns(df: pd.DataFrame, groups: Sequence[str]) -> list[str]:
    """Tabular input columns for the requested groups, in a stable order.
    Raises if a label-only column would be selected (it never should be --
    label-only columns are not prefixed, but be explicit)."""
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)} (known: {GROUPS})")
    cols = [c for c in df.columns if group_of(c) in groups and group_of(c) != "asset"]
    bad = [c for c in cols if c in LABEL_ONLY]
    if bad:
        raise ValueError(f"label-only columns selected as inputs: {bad}")
    return cols


def available_groups(df: pd.DataFrame) -> list[str]:
    """Which groups have at least one non-null value in this table -- the
    honest way to know what a dataset supports (Adam's has aud/vis, the
    prospective one has sched/asset-thumbnail but no aud)."""
    out = []
    for g in GROUPS:
        cols = [c for c in df.columns if group_of(c) == g]
        if cols and df[cols].notna().any().any():
            out.append(g)
    return out
