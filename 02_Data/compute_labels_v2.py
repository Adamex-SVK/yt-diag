"""
Label computation v2 for YT-Diag -- supersedes compute_labels.py (v1),
pending team sign-off (see CHANGELOG.md 2026-08-26).

Why v1 was replaced (both problems spotted by Emmanuel in review):

1. v1's views-per-day-since-upload assumes linear view accumulation, but
   YouTube views are front-loaded -- so the score decays with video age and
   the label partly encodes "how new is this video" instead of "did this
   content perform". Age-correlated model features could then predict the
   label without learning content.
2. v1 normalized by CURRENT subscriber count, which is post-outcome: a viral
   video itself grows subs, inflating its own denominator (reverse
   causality). Subs is also one of the 12 metadata features, so the same
   post-outcome number sat in both the label denominator and the model
   input -- a target-leakage shortcut. And v1's subs=1 fallback for
   hidden/missing counts produced enormous scores that near-deterministically
   landed in the viral quartile.

v2 design -- stratify instead of divide:

    score  = log(1 + view_count)
    label  = "viral" if score is in the top quartile of the video's
             (category x age-band x channel-size-band) cell, else "typical"

  - Age bands and channel-size bands are equal-count quantile bands computed
    within each category (subscriber count is used only as a COARSE
    stratifier, never as a divisor -- post-viral sub growth only mislabels a
    video if it pushed the channel across a band boundary, a rare and
    disclosable event, vs. continuously rescaling every score in v1).
  - Because every cell has ~25% positives by construction, category, video
    age, and channel size are marginally uninformative about the label --
    the model cannot reconstruct the label from metadata stratifiers and is
    forced to find signal in content.
  - This implements the proposal's own definition ("typical = similarly
    sized channels, same category") more literally than v1 did.
  - Excluded from the cohort (recorded in labels_excluded.csv, never
    silently imputed): videos younger than --min-age-days (views/day too
    unstable, decay curve too steep), missing upload/collection dates,
    missing view counts, and hidden/missing subscriber counts.

NOT claimed: this is not Wu et al. (2018)'s relative engagement (that metric
is rank-percentile of average watch percentage calibrated against DURATION
at a fixed 30-day horizon -- watch data we cannot get from snapshots). Only
the calibrate-by-ranking-within-strata method is borrowed. Cite it as
"inspired by", never as "following".

Like v1, this is deliberately separate from collect_and_extract.py: the
label is a relative ranking, computable only once a category's whole cohort
is on disk.

Band counts default to 4x4 (=16 cells, ~125 videos/cell at the 2,000/category
target). Revisit --age-bands/--size-bands after the real age and subscriber
distributions are known (EDA), not before.

Usage:
    source .venv/bin/activate
    python3 02_Data/compute_labels_v2.py --category comedy
    python3 02_Data/compute_labels_v2.py --category all
    python3 02_Data/compute_labels_v2.py --category all --min-age-days 90
"""
import argparse
import csv
import datetime
import json
import math
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
QUARTILE_CUTOFF = 0.75
DEFAULT_MIN_AGE_DAYS = 30
DEFAULT_AGE_BANDS = 4
DEFAULT_SIZE_BANDS = 4
SMALL_CELL_WARN = 20  # quartile ranking gets noisy below this many videos


def is_done(video_dir):
    return os.path.exists(os.path.join(video_dir, ".done"))


def load_metadata(video_dir):
    with open(os.path.join(video_dir, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def extract_row(video_id, video_dir, meta, min_age_days):
    """Returns (row, None) if the video enters the labeling cohort, or
    (None, reason) if it is excluded. Exclusions are explicit and recorded --
    never imputed around (v1's subs=1 fallback is exactly what this replaces)."""
    upload_date = meta.get("upload_date")
    collected_at = meta.get("collected_at")
    view_count = meta.get("view_count")
    subs = meta.get("channel_follower_count")

    if not upload_date or not collected_at:
        return None, "missing upload_date or collected_at"
    if view_count is None:
        return None, "missing view_count"
    if not subs:  # None or 0 -- hidden subscriber count either way
        return None, "hidden/missing channel_follower_count"

    uploaded = datetime.datetime.strptime(upload_date, "%Y%m%d")
    collected = datetime.datetime.strptime(collected_at.split("T")[0], "%Y-%m-%d")
    days = (collected - uploaded).days
    if days < min_age_days:
        return None, f"younger than --min-age-days {min_age_days} ({days}d old at collection)"

    return {
        "video_id": video_id,
        "video_dir": video_dir,
        "days_since_upload": days,
        "view_count": view_count,
        "subs": subs,
        "log_views": math.log1p(view_count),
    }, None


def assign_bands(rows, key, n_bands, band_field):
    """Equal-count quantile bands: sort by key, band i*n_bands//n. Ties broken
    by video_id so the assignment is deterministic across runs."""
    ordered = sorted(rows, key=lambda r: (r[key], r["video_id"]))
    n = len(ordered)
    for i, r in enumerate(ordered):
        r[band_field] = i * n_bands // n


def label_category(category, cat_dir, args):
    rows, excluded = [], []
    for video_id in sorted(os.listdir(cat_dir)):
        video_dir = os.path.join(cat_dir, video_id)
        if not is_done(video_dir):
            continue
        try:
            meta = load_metadata(video_dir)
        except (OSError, json.JSONDecodeError) as e:
            excluded.append({"video_id": video_id, "reason": f"unreadable metadata.json ({e})"})
            continue
        row, reason = extract_row(video_id, video_dir, meta, args.min_age_days)
        if row is None:
            excluded.append({"video_id": video_id, "reason": reason})
        else:
            rows.append(row)

    if not rows:
        print(f"{category}: 0 labelable videos ({len(excluded)} excluded)")
        return

    n_age_bands = min(args.age_bands, len(rows))
    n_size_bands = min(args.size_bands, len(rows))
    assign_bands(rows, "days_since_upload", n_age_bands, "age_band")
    assign_bands(rows, "subs", n_size_bands, "size_band")

    cells = {}
    for r in rows:
        cells.setdefault((r["age_band"], r["size_band"]), []).append(r)

    small_cells = 0
    for cell_rows in cells.values():
        cell_rows.sort(key=lambda r: (r["log_views"], r["video_id"]))
        cell_n = len(cell_rows)
        if cell_n < SMALL_CELL_WARN:
            small_cells += 1
        for i, r in enumerate(cell_rows):
            r["cell_size"] = cell_n
            r["cell_percentile"] = (i + 1) / cell_n
            r["label"] = "viral" if r["cell_percentile"] >= QUARTILE_CUTOFF else "typical"

    for r in rows:
        label_out = {
            "label_version": 2,
            "label": r["label"],
            "days_since_upload": r["days_since_upload"],
            "view_count": r["view_count"],
            "channel_follower_count": r["subs"],
            "log_views": r["log_views"],
            "age_band": r["age_band"],
            "size_band": r["size_band"],
            "cell_size": r["cell_size"],
            "cell_percentile": r["cell_percentile"],
        }
        with open(os.path.join(r["video_dir"], "label.json"), "w", encoding="utf-8") as f:
            json.dump(label_out, f, indent=2)

    summary_path = os.path.join(cat_dir, "labels_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "days_since_upload", "view_count", "channel_follower_count",
                         "log_views", "age_band", "size_band", "cell_size", "cell_percentile", "label"])
        for r in sorted(rows, key=lambda r: r["video_id"]):
            writer.writerow([r["video_id"], r["days_since_upload"], r["view_count"], r["subs"],
                             f"{r['log_views']:.4f}", r["age_band"], r["size_band"],
                             r["cell_size"], f"{r['cell_percentile']:.4f}", r["label"]])

    excluded_path = os.path.join(cat_dir, "labels_excluded.csv")
    with open(excluded_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "reason"])
        for e in excluded:
            writer.writerow([e["video_id"], e["reason"]])

    n = len(rows)
    viral = sum(1 for r in rows if r["label"] == "viral")
    cell_sizes = sorted(len(c) for c in cells.values())
    print(f"{category}: {n} labeled ({viral} viral = {viral / n:.1%}), "
          f"{len(excluded)} excluded -> {summary_path}")
    print(f"  {len(cells)} cells ({n_age_bands} age x {n_size_bands} size bands), "
          f"sizes min/median/max = {cell_sizes[0]}/{cell_sizes[len(cell_sizes) // 2]}/{cell_sizes[-1]}")
    if small_cells:
        print(f"  WARNING: {small_cells} cells smaller than {SMALL_CELL_WARN} videos -- "
              f"quartile ranking is noisy there; consider fewer --age-bands/--size-bands")
    if excluded:
        reasons = {}
        for e in excluded:
            key = e["reason"].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  excluded {count}: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-dir", default=OUT_DIR,
                        help="processed/ directory to label (default: 02_Data/processed)")
    parser.add_argument("--min-age-days", type=int, default=DEFAULT_MIN_AGE_DAYS,
                        help="exclude videos younger than this at collection time")
    parser.add_argument("--age-bands", type=int, default=DEFAULT_AGE_BANDS)
    parser.add_argument("--size-bands", type=int, default=DEFAULT_SIZE_BANDS)
    args = parser.parse_args()

    if args.category == "all":
        categories = sorted(os.listdir(args.data_dir)) if os.path.isdir(args.data_dir) else []
    else:
        categories = [args.category]

    for cat in categories:
        cat_dir = os.path.join(args.data_dir, cat)
        if not os.path.isdir(cat_dir):
            print(f"{cat}: no processed/ folder, skipping")
            continue
        label_category(cat, cat_dir, args)


if __name__ == "__main__":
    main()
