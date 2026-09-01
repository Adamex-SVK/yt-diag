"""
Temporal-dynamics diagnostics from the stored frames: how much the picture
appears to change across the 20 sparse samples.

Nothing in the collected feature set measures this. `visual_features.json`
describes each frame's appearance (brightness, colour, faces) and averages it,
which cannot distinguish a fast-cut montage from a locked-off talking head with
the same average brightness. These frames cannot recover cutting pace because
most cuts occur between samples.

THE TRAP THIS MODULE EXISTS TO AVOID. The 20 frames are deliberately NOT evenly
spaced: 12 land inside the first 60 s and 8 cover the rest (see
compute_frame_timestamps). So a raw difference between consecutive frames is a
difference over 5 seconds early in a long video and over several minutes later
in it. Comparing those numbers, or averaging them, measures the sampling
schedule as much as the content. Raw and time-normalised diagnostics are kept
to document that failure; neither is approved as a cross-video editing-rate
feature.

Difference metric: mean absolute difference of 64x64 greyscale frames, 0-255.
Downsampling is deliberate -- at full resolution the measure is dominated by
compression noise and camera grain rather than by content change.

    .venv/bin/python 02_Data/motion_features.py --limit 200   # try it
    .venv/bin/python 02_Data/motion_features.py               # all 1,860
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_and_extract import compute_frame_timestamps  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
OUT_CSV = os.path.join(DATA_DIR, "motion_features.csv")
THUMB_SIZE = (64, 64)

# MEASURED 2026-09-01 AND ABANDONED: cutting pace is NOT recoverable from these
# frames, and this constant records why rather than hiding it.
#
# The plan was to count shot changes by thresholding the difference between
# consecutive sampled frames. Calibrating on 400 videos killed it: the mean
# consecutive difference has median 43.9 and p10 20.2 (0-255 scale), so a
# "large" difference is the NORMAL case, not an event. At a threshold of 25,
# 83% of videos exceed it on average.
#
# The reason is the sampling, not the metric. The 20 frames span seconds to
# minutes apart, so almost every consecutive pair already straddles several
# cuts. Counting them measures the sampling schedule, not the editing. Real cut
# detection needs dense frames (~1 fps), which would mean re-downloading every
# video.
#
# What IS measurable from 20 sparse frames is how VARIED a video looks overall,
# which separates a slideshow or locked-off camera from anything edited. That
# is what this module reports.
STATIC_MAD = 5.0        # below this a video is visually near-frozen

# MEASURED 2026-09-01. Two of the fields below are duration proxies, which is
# recorded here because using either as a content feature would smuggle
# duration into a model that already has it:
#
#   mad_per_second_mean   spearman with duration = -0.829  DO NOT USE.
#       The time normalisation in the docstring above sounded right and is
#       wrong: frame difference SATURATES -- two unrelated frames differ by
#       roughly a fixed ceiling however far apart they are -- so dividing by an
#       ever-larger gap just reconstructs 1/duration. It is kept only so the
#       saturation is visible rather than rediscovered.
#
#   visual_diversity      spearman with duration = +0.202 pooled,
#       +0.075 within regular but +0.391 within Shorts, where duration spans
#       4-180 s and therefore changes the frame spacing a lot. Usable, but
#       condition on duration and format before drawing any conclusion from it.
#
# `is_static` means only "low change across the sparse samples". It is retained
# for auditing, not registered as a model feature or equated with one scene.


def _frame_array(path: str, np: Any) -> Optional[Any]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize(THUMB_SIZE, Image.BILINEAR),
                              dtype=np.float32)
    except Exception:
        return None


def video_motion(video_dir: str, duration: Optional[float]) -> Optional[dict[str, Any]]:
    """Motion statistics for one video, or None if its frames are unreadable.

    Returns raw and time-normalised change rates. `duration` in seconds is
    needed for the time normalisation and for the per-minute rates; pass None
    and those fields come back None rather than silently wrong.
    """
    import numpy as np
    frames_dir = os.path.join(video_dir, "frames")
    if not os.path.isdir(frames_dir):
        return None
    names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    arrs = [a for a in (_frame_array(os.path.join(frames_dir, n), np) for n in names) if a is not None]
    if len(arrs) < 2:
        return None

    consecutive = [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]
    allpairs = [float(np.abs(arrs[j] - arrs[i]).mean())
                for i in range(len(arrs)) for j in range(i + 1, len(arrs))]

    # time gaps actually separating consecutive samples
    gaps, per_sec = None, None
    if duration and duration > 0:
        ts = compute_frame_timestamps(float(duration), len(arrs))
        gaps = [max(ts[i + 1] - ts[i], 1e-6) for i in range(len(ts) - 1)]
        per_sec = [d / g for d, g in zip(consecutive, gaps)]

    return {
        "n_frames": len(arrs),
        # raw: comparable only between videos of similar length
        "mad_consecutive_mean": statistics.fmean(consecutive),
        "mad_consecutive_max": max(consecutive),
        "mad_allpairs_mean": statistics.fmean(allpairs),
        # time-normalised diagnostic; DO NOT interpret as editing rate
        "mad_per_second_mean": statistics.fmean(per_sec) if per_sec else None,
        "mad_per_second_max": max(per_sec) if per_sec else None,
        "is_static": statistics.fmean(allpairs) < STATIC_MAD,
    }


def _rows(data_dir: str, limit: Optional[int]) -> list[tuple[str, str, str, Optional[float]]]:
    out = []
    for category in sorted(os.listdir(data_dir)):
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for vid in sorted(os.listdir(cat_dir)):
            d = os.path.join(cat_dir, vid)
            if not os.path.exists(os.path.join(d, ".done")):
                continue
            dur = None
            try:
                with open(os.path.join(d, "metadata.json"), encoding="utf-8") as f:
                    dur = json.load(f).get("duration")
            except (OSError, json.JSONDecodeError):
                pass
            out.append((category, vid, d, dur))
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calibrate", action="store_true",
                    help="print the consecutive-difference distribution and exit, "
                         "documenting why shot-change thresholding was rejected")
    args = ap.parse_args()

    todo = _rows(args.data_dir, args.limit)
    print(f"{len(todo)} videos")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(video_motion, d, dur): (c, v) for c, v, d, dur in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            c, v = futs[fut]
            r = fut.result()
            if r:
                results.append({"video_id": v, "category": c, **r})
            if i % 250 == 0:
                print(f"  {i}/{len(todo)}")

    if args.calibrate:
        vals = sorted(r["mad_consecutive_mean"] for r in results)
        print("\nmean consecutive-frame MAD percentiles:")
        for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
            print(f"  p{p:<3d} {vals[int(p / 100 * (len(vals) - 1))]:7.2f}")
        print("\nNote: a 'large' consecutive difference is the median case here, which is "
              "why\nshot-change counting was abandoned -- see the STATIC_MAD comment.")
        return

    if not results:
        sys.exit("no videos produced motion features")
    results.sort(key=lambda r: (r["category"], r["video_id"]))
    fields = list(results[0])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(results)

    div = sorted(r["mad_allpairs_mean"] for r in results)
    static = sum(1 for r in results if r["is_static"])

    print(f"\n{len(results)} videos processed")
    print(f"  all-pairs MAD     median {div[len(div)//2]:.1f}  "
          f"p10 {div[len(div)//10]:.1f}  p90 {div[9*len(div)//10]:.1f}")
    print(f"  low-change samples (all-pairs MAD < 5): {static} ({static/len(results):.1%})")
    print("  (shot-change counting is deliberately NOT reported -- see STATIC_MAD note)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
