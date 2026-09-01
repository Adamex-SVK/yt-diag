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
    label  = "viral" for exactly the top floor(n/4) videos (min 1) of the
             video's (category x age-band x channel-size-band) cell of size
             n, else "typical". Exact view-count ties at the cutoff are
             broken deterministically by video_id and reported as a warning
             (a percentile >= 0.75 rule was rejected: it over-labels --
             26/100, 6/20 -- because it includes the boundary video).

  - Age bands and channel-size bands are equal-count quantile bands computed
    within each category, assigned BY VALUE: identical values always share
    a band (the API rounds subscriber counts to ~3 significant figures, so
    exact ties are common -- rank-splitting could put videos with the same
    subscriber count, even videos of the same channel, into different
    "channel-size" bands).
  - Subscriber count is used only as a COARSE stratifier, never as a
    divisor. This reduces (does NOT eliminate) the reverse causality of
    current-subs being post-outcome: a video is only mislabeled if its own
    success pushed the channel across a band boundary. How often that
    happens should be quantified on the real data (sensitivity check:
    rerun with each viral video's subs halved and count band changes).
  - Within the labeled cohort every cell has ~25% positives by
    construction, so the band IDs themselves carry no marginal label
    information. Stratification reduces direct shortcuts from the
    peer-group variables; it does not guarantee that model performance
    comes from content -- models still see exact age and subscriber values
    (residual within-band gradients remain), plus duration, upload timing,
    tags, and collection-query artifacts.
  - This is a retrospective, cohort-relative ranking: band edges and
    cutoffs are computed over the whole collected cohort BEFORE any
    train/test split, so labels are transductive by construction. Disclose
    this in the report; the alternative (fit bins/thresholds on the train
    split only) is noted in evaluation planning as a sensitivity analysis.
  - This implements the proposal's own definition ("typical = similarly
    sized channels, same category") more literally than v1 did.
  - Excluded from the cohort (recorded in labels_excluded.csv, never
    silently imputed): videos younger than --min-age-days (views/day too
    unstable, decay curve too steep), missing upload/collection dates,
    missing view counts, and hidden/missing subscriber counts.
  - log(1+views) is monotonic, so it never changes the within-cell ranking;
    it is kept because the stored score is also a readable EDA quantity.

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
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
from typing import Any, Optional

OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
VIRAL_FRACTION = 0.25  # exactly floor(n * fraction) viral per cell, min 1
# Defaults revised 2026-09-01 from the evidence in 02_Data/eda.md:
#   min age 30 -> 0: the floor is a v1 inheritance (views-per-day decays with
#     age); v2 ranks WITHIN age bands, so it was paying 861 videos (46%, and a
#     biased 46% -- 64% Shorts vs 43% retained) for nothing measurable. With
#     format stratified, the four-confound label AUC is 0.576 at min-age 0 vs
#     0.572 at 7. The prospective panel confirms the premise directly: view
#     RANK is ~90% settled within 24h (day-1 vs day-5 Spearman 0.946).
#   4x4 -> 3x3: the format dimension halves every cell, and 4x4x2 leaves a
#     median cell of 13 with 102 cells under SMALL_CELL_WARN. 2x2x2 has clean
#     cells but lets the confounds back in (AUC 0.660). 3x3x2 is the knee.
DEFAULT_MIN_AGE_DAYS = 0
DEFAULT_AGE_BANDS = 3
DEFAULT_SIZE_BANDS = 3
SMALL_CELL_WARN = 20  # quartile ranking gets noisy below this many videos


def is_done(video_dir: str) -> bool:
    """True once collect_and_extract.py has fully processed this video
    directory. The .done marker is written last, after metadata/thumbnail/
    frames/transcript, so a directory without it may still be mid-write:
    labeling one would rank a video on a partial or absent view count."""
    return os.path.exists(os.path.join(video_dir, ".done"))


def load_metadata(video_dir: str) -> dict[str, Any]:
    """collect_and_extract.py's metadata.json for one video, as written (no
    key is required here -- extract_row decides what is missing). Raises
    OSError / json.JSONDecodeError; the caller records those videos as
    exclusions rather than dropping them silently."""
    with open(os.path.join(video_dir, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def extract_row(
    video_id: str,
    video_dir: str,
    meta: dict[str, Any],
    min_age_days: int,
    stratify_format: bool = True,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Returns (row, None) if the video enters the labeling cohort, or
    (None, reason) if it is excluded. Exclusions are explicit and recorded --
    never imputed around (v1's subs=1 fallback is exactly what this replaces).

    Row keys: video_id, video_dir, days_since_upload (whole days; both sides
    are dates only -- yt-dlp's upload_date has no time -- so it carries up to
    a day of rounding), view_count, subs (channel_follower_count at
    collection time), log_views = log(1 + view_count), and -- when
    stratify_format is on -- is_short ("true"/"false", read from
    metadata_extra.json, which the labeler does not otherwise touch).

    A video with no definitive Shorts verdict is EXCLUDED rather than guessed
    at: it cannot be placed in a format stratum, and duration is not a
    substitute (a verified 250-second Short exists in the prospective cohort).
    Five videos are affected, all of them deleted or private."""
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

    is_short = ""
    if stratify_format:
        try:
            with open(os.path.join(video_dir, "metadata_extra.json"), encoding="utf-8") as f:
                is_short = (json.load(f) or {}).get("is_short", "")
        except (OSError, json.JSONDecodeError):
            is_short = ""
        if is_short not in ("true", "false"):
            return None, "no definitive is_short verdict (cannot place in a format stratum)"

    return {
        "is_short": is_short,
        "video_id": video_id,
        "video_dir": video_dir,
        "days_since_upload": days,
        "view_count": view_count,
        "subs": subs,
        "log_views": math.log1p(view_count),
    }, None


def assign_bands(
    rows: list[dict[str, Any]],
    key: str,
    n_bands: int,
    band_field: str,
) -> None:
    """Writes an integer band index in 0..n_bands-1 into r[band_field] for
    every r in rows, in place (rows is mutated; nothing is returned).

    Equal-count quantile bands assigned BY VALUE: every video shares the
    band of its value's first occurrence in sort order, so identical values
    (common for subscriber counts, which the API rounds to ~3 significant
    figures) can never be split across bands. Band sizes become uneven when
    ties straddle a quantile edge -- acceptable, and preferable to placing
    two videos with the same subscriber count in different "size" bands."""
    ordered = sorted(r[key] for r in rows)
    n = len(ordered)
    first_rank = {}
    for i, v in enumerate(ordered):
        if v not in first_rank:
            first_rank[v] = i
    for r in rows:
        r[band_field] = first_rank[r[key]] * n_bands // n


def label_category(category: str, cat_dir: str, args: argparse.Namespace) -> None:
    """Label one category's collected cohort and write the result to disk:
    label.json per video, plus labels_summary.csv and labels_excluded.csv in
    cat_dir. Prints a summary; returns nothing. args supplies min_age_days,
    age_bands and size_bands.

    Every band edge and cell cutoff is derived from the videos on disk at the
    moment of the run, so the whole category must be relabeled together: a
    re-run after more videos are collected can flip an already-written label,
    and labeling a half-collected category ranks videos against a cohort that
    does not exist yet.

    Cells are (age_band x size_band) within the category and each gets exactly
    floor(cell_n * VIRAL_FRACTION) viral videos, minimum 1 -- so a cell of 1-3
    videos force-labels its top video viral. That is what the SMALL_CELL_WARN
    message is warning about."""
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
        row, reason = extract_row(video_id, video_dir, meta, args.min_age_days,
                                  stratify_format=not args.no_format_stratum)
        if row is None:
            excluded.append({"video_id": video_id, "reason": reason})
        else:
            rows.append(row)

    if not rows:
        print(f"{category}: 0 labelable videos ({len(excluded)} excluded)")
        return

    # Bands are quantiles WITHIN a format stratum, not across both: Shorts and
    # regular videos have different age and subscriber distributions, so pooled
    # band edges would put a Short and a long-form video in a cell that is only
    # nominally comparable (eda.md 3).
    strata = {}
    for r in rows:
        strata.setdefault(r.get("is_short", ""), []).append(r)
    n_age_bands = min(args.age_bands, min(len(v) for v in strata.values()))
    n_size_bands = min(args.size_bands, min(len(v) for v in strata.values()))
    for fmt_rows in strata.values():
        assign_bands(fmt_rows, "days_since_upload", n_age_bands, "age_band")
        assign_bands(fmt_rows, "subs", n_size_bands, "size_band")

    cells = {}
    for r in rows:
        cells.setdefault((r["age_band"], r["size_band"], r.get("is_short", "")), []).append(r)

    small_cells = boundary_tie_cells = 0
    for cell_rows in cells.values():
        cell_rows.sort(key=lambda r: (r["log_views"], r["video_id"]))
        cell_n = len(cell_rows)
        if cell_n < SMALL_CELL_WARN:
            small_cells += 1
        k = max(1, int(cell_n * VIRAL_FRACTION))
        cutoff_idx = cell_n - k
        if cutoff_idx > 0 and cell_rows[cutoff_idx]["log_views"] == cell_rows[cutoff_idx - 1]["log_views"]:
            boundary_tie_cells += 1
            for r in cell_rows:
                if r["log_views"] == cell_rows[cutoff_idx]["log_views"]:
                    r["warnings"] = r.get("warnings", []) + [
                        "view count tied at the viral cutoff -- label decided by video_id tiebreak"]
        for i, r in enumerate(cell_rows):
            r["cell_size"] = cell_n
            r["cell_percentile"] = (i + 1) / cell_n
            r["label"] = "viral" if i >= cutoff_idx else "typical"

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
            "is_short": r.get("is_short", ""),
            "cell_size": r["cell_size"],
            "cell_percentile": r["cell_percentile"],
            "warnings": r.get("warnings", []),
        }
        with open(os.path.join(r["video_dir"], "label.json"), "w", encoding="utf-8") as f:
            json.dump(label_out, f, indent=2)

    summary_path = os.path.join(cat_dir, "labels_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "days_since_upload", "view_count", "channel_follower_count",
                         "log_views", "age_band", "size_band", "is_short", "cell_size", "cell_percentile", "label",
                         "warnings"])
        for r in sorted(rows, key=lambda r: r["video_id"]):
            writer.writerow([r["video_id"], r["days_since_upload"], r["view_count"], r["subs"],
                             f"{r['log_views']:.4f}", r["age_band"], r["size_band"], r.get("is_short", ""),
                             r["cell_size"], f"{r['cell_percentile']:.4f}", r["label"],
                             "; ".join(r.get("warnings", []))])

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
    n_strata = len({r.get("is_short", "") for r in rows})
    print(f"  {len(cells)} cells ({n_age_bands} age x {n_size_bands} size x {n_strata} format), "
          f"sizes min/median/max = {cell_sizes[0]}/{cell_sizes[len(cell_sizes) // 2]}/{cell_sizes[-1]}")
    if small_cells:
        print(f"  WARNING: {small_cells} cells smaller than {SMALL_CELL_WARN} videos -- "
              f"quartile ranking is noisy there; consider fewer --age-bands/--size-bands")
    if boundary_tie_cells:
        print(f"  WARNING: {boundary_tie_cells} cells have exact view-count ties at the viral "
              f"cutoff (label decided by video_id tiebreak) -- affected videos flagged in the "
              f"warnings column of {summary_path}")
    if excluded:
        reasons = {}
        for e in excluded:
            key = e["reason"].split(" (")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  excluded {count}: {reason}")


def main() -> None:
    """CLI entry point. --category all labels every subdirectory of
    --data-dir as its own cohort: bands and cutoffs are per-category by
    design, so a video is never ranked against another category's videos.
    A named category with no processed/ folder is reported and skipped, not
    an error -- collection order across categories is not guaranteed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-dir", default=OUT_DIR,
                        help="processed/ directory to label (default: 02_Data/processed)")
    parser.add_argument("--min-age-days", type=int, default=DEFAULT_MIN_AGE_DAYS,
                        help="exclude videos younger than this at collection time")
    parser.add_argument("--no-format-stratum", action="store_true",
                        help="do NOT stratify on is_short. Only for the sensitivity analysis: "
                             "unstratified, Shorts take the viral label at ~2x their share and "
                             "the format bit is 99%% readable off the frames (eda.md 3-4)")
    parser.add_argument("--age-bands", type=int, default=DEFAULT_AGE_BANDS)
    parser.add_argument("--size-bands", type=int, default=DEFAULT_SIZE_BANDS)
    args = parser.parse_args()

    if args.category == "all":
        # directories only: processed/ also holds cleaning_manifest.csv
        categories = sorted(d for d in os.listdir(args.data_dir)
                            if os.path.isdir(os.path.join(args.data_dir, d))) \
            if os.path.isdir(args.data_dir) else []
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
