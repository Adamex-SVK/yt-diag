"""
SUPERSEDED -- this is label v1, replaced by compute_labels_v2.py (CHANGELOG.md
2026-08-26) and kept only as a reference for what the original CLAUDE.md label
was. Do not run it on the real cohort. Two flaws found in review: views/day
assumes linear view accumulation, so the score decays with video age and the
label partly encodes how new a video is; and the subscriber-count denominator
is post-outcome (a viral video grows its own denominator, and subs is itself a
model feature -- target leakage), with the subs=1 fallback below near-
deterministically labeling hidden-subscriber videos viral.

Label computation for YT-Diag (CLAUDE.md label definition):

    "viral" = top quartile of views-per-day-since-upload, normalized by
    channel subscriber count, within category.
    "typical" = everything else in the same category.

Deliberately NOT part of collect_and_extract.py: this is a relative,
within-category ranking, so it can only be computed once a category's whole
collected cohort is on disk -- a single video's label depends on every other
video in its category, unlike every other feature in this pipeline.

Usage:
    source .venv/bin/activate
    python3 02_Data/compute_labels.py --category comedy
    python3 02_Data/compute_labels.py --category all
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from typing import Any, Optional

OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
QUARTILE_CUTOFF = 0.75


def is_done(video_dir: str) -> bool:
    """True once collect_and_extract.py has fully processed this video
    directory: the .done marker is written last, after every artifact. v1
    (superseded); compute_labels_v2.py has the same check."""
    return os.path.exists(os.path.join(video_dir, ".done"))


def load_metadata(video_dir: str) -> dict[str, Any]:
    """collect_and_extract.py's metadata.json for one video. v1 (superseded):
    unlike v2 this raises out of label_category on an unreadable file instead
    of recording the video as an exclusion."""
    with open(os.path.join(video_dir, "metadata.json")) as f:
        return json.load(f)


def compute_rate(meta: dict[str, Any]) -> tuple[Optional[dict[str, Any]], list[str]]:
    """views-per-day-since-upload, normalized by subscriber count.
    Returns (rate, warnings) -- warnings flag assumptions made for missing
    or degenerate inputs, so they're visible in the output, not silent.

    v1, SUPERSEDED: both quantities below are the reason (see the module
    docstring) -- views_per_day decays with age, and dividing by a
    post-outcome subscriber count is reverse causality.

    rate is None (with the reason in warnings) when upload_date or
    collected_at is missing, i.e. the rate is uncomputable and the caller
    must skip the video. Otherwise its keys are: days_since_upload (whole
    days, floored at 1 so the division is safe -- a same-day collection
    reports 1, not 0), views_per_day (views per day, lifetime average),
    normalized_rate (views per day per subscriber, the ranked score; with
    subs substituted by 1 when the count is hidden or zero, which makes that
    video's score incomparable to the rest of the cohort)."""
    warnings = []
    upload_date = meta.get("upload_date")
    collected_at = meta.get("collected_at")
    view_count = meta.get("view_count") or 0
    subs = meta.get("channel_follower_count")

    if not upload_date or not collected_at:
        return None, ["missing upload_date or collected_at -- cannot compute rate"]

    uploaded = datetime.datetime.strptime(upload_date, "%Y%m%d")
    collected = datetime.datetime.strptime(collected_at.split("T")[0], "%Y-%m-%d")
    days = max((collected - uploaded).days, 1)
    if (collected - uploaded).days < 1:
        warnings.append("video collected <1 day after upload -- views/day is unstable this early")

    views_per_day = view_count / days

    if not subs:
        subs = 1
        warnings.append("channel_follower_count missing/zero -- normalized by 1, not comparable to videos with real subscriber counts")

    normalized = views_per_day / subs
    return {"days_since_upload": days, "views_per_day": views_per_day, "normalized_rate": normalized}, warnings


def label_category(category: str) -> list[dict[str, Any]]:
    """Label one category under OUT_DIR (v1 has no --data-dir option) and
    write label.json per video plus labels_summary.csv. Returns the labeled
    rows, or an empty list when the category has no processed/ folder or no
    .done videos -- an empty list is "nothing to rank", never "all typical".

    v1, SUPERSEDED by compute_labels_v2.py. Two things to know before reading
    any number this produced: the whole category is ranked as one pool (no age
    or channel-size strata, so the ranking is confounded by both), and the
    cutoff is percentile >= QUARTILE_CUTOFF, which includes the boundary video
    and therefore labels slightly more than a quarter viral (26 of 100, 6 of
    20). v2 takes exactly the top floor(n/4) per cell instead."""
    cat_dir = os.path.join(OUT_DIR, category)
    if not os.path.isdir(cat_dir):
        print(f"{category}: no processed/ folder, skipping")
        return []

    rows = []
    for video_id in sorted(os.listdir(cat_dir)):
        video_dir = os.path.join(cat_dir, video_id)
        if not is_done(video_dir):
            continue
        meta = load_metadata(video_dir)
        rate_info, warnings = compute_rate(meta)
        if rate_info is None:
            print(f"  {video_id}: SKIPPED -- {warnings}")
            continue
        rows.append({"video_id": video_id, "video_dir": video_dir, "warnings": warnings, **rate_info})

    if not rows:
        print(f"{category}: 0 labelable videos")
        return []

    rows.sort(key=lambda r: r["normalized_rate"])
    n = len(rows)
    for i, r in enumerate(rows):
        percentile = (i + 1) / n
        r["category_percentile"] = percentile
        r["label"] = "viral" if percentile >= QUARTILE_CUTOFF else "typical"

    for r in rows:
        label_out = {
            "days_since_upload": r["days_since_upload"],
            "views_per_day": r["views_per_day"],
            "normalized_rate": r["normalized_rate"],
            "category_percentile": r["category_percentile"],
            "label": r["label"],
            "warnings": r["warnings"],
        }
        with open(os.path.join(r["video_dir"], "label.json"), "w") as f:
            json.dump(label_out, f, indent=2)

    summary_path = os.path.join(cat_dir, "labels_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "days_since_upload", "views_per_day", "normalized_rate", "category_percentile", "label", "warnings"])
        for r in rows:
            writer.writerow([r["video_id"], r["days_since_upload"], r["views_per_day"], r["normalized_rate"],
                              r["category_percentile"], r["label"], "; ".join(r["warnings"])])

    viral_count = sum(1 for r in rows if r["label"] == "viral")
    print(f"{category}: {n} videos labeled ({viral_count} viral, {n - viral_count} typical) -> {summary_path}")
    return rows


def main() -> None:
    """CLI entry point for the v1 label -- SUPERSEDED, run
    compute_labels_v2.py on the real cohort instead. --category all labels
    every subdirectory of OUT_DIR as its own pool. Overwrites the same
    label.json / labels_summary.csv files v2 writes, so running this after v2
    silently replaces v2 labels with v1 ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    if args.category == "all":
        categories = sorted(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else []
    else:
        categories = [args.category]

    for cat in categories:
        label_category(cat)


if __name__ == "__main__":
    main()
