"""
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
import argparse
import csv
import datetime
import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
QUARTILE_CUTOFF = 0.75


def is_done(video_dir):
    return os.path.exists(os.path.join(video_dir, ".done"))


def load_metadata(video_dir):
    with open(os.path.join(video_dir, "metadata.json")) as f:
        return json.load(f)


def compute_rate(meta):
    """views-per-day-since-upload, normalized by subscriber count.
    Returns (rate, warnings) -- warnings flag assumptions made for missing
    or degenerate inputs, so they're visible in the output, not silent."""
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


def label_category(category):
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


def main():
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
