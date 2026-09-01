"""
Recompute the colour-temperature features from the images already on disk.

WHY: the original `_correlated_color_temp` set X=r, Y=g, Z=b -- skipping the
sRGB->XYZ matrix entirely -- so its x,y were normalised RGB fractions, not CIE
chromaticity. McCamy's denominator (0.1858 - y) then approached zero for
green-deficient images and the cubic diverged: 83 of the 1,860 collected videos
carried CCT values up to 5.0e9, and 27 were NEGATIVE Kelvin. After
standardisation those columns were effectively constant, so the visual feature
block was carrying three dead columns into every baseline. Version 2 fixed the
colour conversion but retained McCamy and only approximated Duv. Version 3
directly finds the closest point on a 1%-resolution Planckian-locus table over
one consistently supported range.

WHAT THIS DOES NOT NEED: no video re-download and no frame re-extraction. CCT
is a per-image statistic and every thumbnail and frame is still on disk, so
this reads ~39,000 JPEGs and rewrites three numbers per video. Nothing else in
visual_features.json is touched -- face detection, brightness, saturation and
contrast are left exactly as collected.

EVERY value is replaced, not only the 83 extreme ones. The old values are
uniformly wrong (the matrix was missing for all of them); the 83 are merely the
ones where being wrong became visible. Two corrections apply at once:
  1. the sRGB->XYZ matrix, and
  2. averaging in LINEAR light rather than over gamma-encoded code values.
So new values are NOT comparable to old ones and no mixed analysis is valid.

Provenance is written into each file (`cct_version`, `cct_recomputed_at_utc`)
along with explicit validity counts, so a row whose CCT is missing is
distinguishable from one that was never recomputed.

    .venv/bin/python 02_Data/recompute_cct.py --dry-run   # report, write nothing
    .venv/bin/python 02_Data/recompute_cct.py
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_and_extract import (CCT_METHOD, CCT_VALID_K, CCT_VERSION,  # noqa: E402
                                 correlated_color_temp, population_std,
                                 srgb_to_linear)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
VIS_NAME = "visual_features.json"
_LUT = None


def _lut(np: Any) -> Any:
    """256-entry uint8 -> linear-light table, built once per process."""
    global _LUT
    if _LUT is None:
        _LUT = np.array([srgb_to_linear(i / 255.0) for i in range(256)], dtype=np.float64)
    return _LUT


def image_cct(path: str) -> Optional[float]:
    """CCT in Kelvin for one image, or None if it is unreadable or its mean
    colour falls outside CCT_VALID_K.

    The mean is taken in LINEAR light: averaging gamma-encoded pixels and then
    converting would weight dark pixels far too heavily.

    Uses PIL rather than OpenCV so this runs on a MODELLING machine: cv2 lives
    in the collection stack (requirements.txt), pillow in requirements-ml.txt.
    PIL gives RGB directly, where cv2 would give BGR."""
    import numpy as np
    from PIL import Image
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception:
        return None  # unreadable/corrupt: reported as invalid, never guessed
    lin = _lut(np)[arr].reshape(-1, 3).mean(axis=0)
    return correlated_color_temp((float(lin[0]), float(lin[1]), float(lin[2])))


def video_cct(video_dir: str) -> dict[str, Any]:
    """Recomputed CCT fields for one video directory.

    frames_mean_cct / frames_std_cct are computed over the frames that yielded
    a VALID temperature, so `cct_frames_valid` must be read alongside them: a
    mean over 3 of 20 frames is not the same measurement as a mean over 20.
    std is None for fewer than two valid frames (undefined, not zero)."""
    thumb = next((os.path.join(video_dir, f)
                  for f in ("thumbnail.webp", "thumbnail.jpg", "thumbnail.png")
                  if os.path.exists(os.path.join(video_dir, f))), None)
    thumb_cct = image_cct(thumb) if thumb else None

    frames_dir = os.path.join(video_dir, "frames")
    frame_ccts = []
    n_frames = 0
    if os.path.isdir(frames_dir):
        for name in sorted(os.listdir(frames_dir)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            n_frames += 1
            c = image_cct(os.path.join(frames_dir, name))
            if c is not None:
                frame_ccts.append(c)
    return {
        "thumbnail_cct": thumb_cct,
        "frames_mean_cct": sum(frame_ccts) / len(frame_ccts) if frame_ccts else None,
        "frames_std_cct": population_std(frame_ccts),
        "cct_thumbnail_valid": thumb_cct is not None,
        "cct_frames_valid": len(frame_ccts),
        "cct_frames_total": n_frames,
    }


def _atomic_write_json(path: str, obj: dict[str, Any]) -> None:
    """Write via tmp + os.replace so a crash cannot truncate a feature file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def done_videos(data_dir: str) -> list[tuple[str, str, str]]:
    """(category, video_id, video_dir) for every completed video."""
    out = []
    for category in sorted(os.listdir(data_dir)):
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue  # processed/ also holds cleaning_manifest.csv
        for vid in sorted(os.listdir(cat_dir)):
            d = os.path.join(cat_dir, vid)
            if os.path.exists(os.path.join(d, ".done")):
                out.append((category, vid, d))
    return out


def apply_to_video(video_dir: str, dry_run: bool = False) -> Optional[dict[str, Any]]:
    """Recompute one video and merge the result into its visual_features.json.

    Returns a dict of before/after values for reporting, or None if the file is
    missing or unreadable (left untouched -- a broken feature file is a
    separate problem from a wrong CCT)."""
    path = os.path.join(video_dir, VIS_NAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            vis = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    new = video_cct(video_dir)
    before = {"thumb": (vis.get("thumbnail") or {}).get("cct"),
              "mean": vis.get("frames_mean_cct"), "std": vis.get("frames_std_cct")}

    if not dry_run:
        vis.setdefault("thumbnail", {})["cct"] = new["thumbnail_cct"]
        vis["frames_mean_cct"] = new["frames_mean_cct"]
        vis["frames_std_cct"] = new["frames_std_cct"]
        vis["cct_thumbnail_valid"] = new["cct_thumbnail_valid"]
        vis["cct_frames_valid"] = new["cct_frames_valid"]
        vis["cct_frames_total"] = new["cct_frames_total"]
        vis["cct_version"] = CCT_VERSION
        vis["cct_method"] = CCT_METHOD
        vis["cct_recomputed_at_utc"] = datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _atomic_write_json(path, vis)
    return {"before": before, "after": new}


def _in_range(v: Optional[float]) -> bool:
    return v is not None and CCT_VALID_K[0] <= v <= CCT_VALID_K[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    videos = done_videos(args.data_dir)
    print(f"{len(videos)} videos ({len(videos) * 21} images) -- "
          f"{'DRY RUN, writing nothing' if args.dry_run else 'writing in place'}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(apply_to_video, d, args.dry_run): (c, v)
                for c, v, d in videos}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            r = fut.result()
            if r:
                results.append(r)
            if i % 250 == 0:
                print(f"  {i}/{len(videos)}")

    old_bad = sum(1 for r in results
                  if not _in_range(r["before"]["thumb"]) or not _in_range(r["before"]["mean"]))
    new_thumb_ok = sum(1 for r in results if r["after"]["cct_thumbnail_valid"])
    new_mean_ok = sum(1 for r in results if r["after"]["frames_mean_cct"] is not None)
    partial = [r for r in results
               if r["after"]["cct_frames_total"] and
               r["after"]["cct_frames_valid"] < r["after"]["cct_frames_total"]]
    means = [r["after"]["frames_mean_cct"] for r in results if r["after"]["frames_mean_cct"]]

    print(f"\nprocessed {len(results)} videos")
    print(f"  BEFORE: {old_bad} videos had a thumbnail or frame-mean CCT outside "
          f"{CCT_VALID_K[0]:.0f}-{CCT_VALID_K[1]:.0f} K")
    print(f"  AFTER:  0 by construction -- out-of-range now yields None, not a number")
    print(f"  thumbnail CCT valid: {new_thumb_ok}/{len(results)}; "
          f"frame-mean CCT valid: {new_mean_ok}/{len(results)}")
    print(f"  videos with >=1 frame yielding no valid CCT: {len(partial)}")
    if means:
        means.sort()
        print(f"  frames_mean_cct: min {means[0]:.0f}  median {means[len(means)//2]:.0f}  "
              f"max {means[-1]:.0f} K")
    if args.dry_run:
        print("\n(dry run -- rerun without --dry-run to write)")


if __name__ == "__main__":
    main()
