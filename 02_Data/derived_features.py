"""
Derived features, each one added because the descriptive EDA of 2026-09-01
showed an existing feature was measuring the wrong thing.

Every feature here exists to fix a specific, measured defect. The evidence is
recorded beside each so nobody has to re-derive why it is here:

1. HEURISTIC CONTENT-CROP THUMBNAIL STATISTICS. `vis__thumb_brightness`
   separates Shorts from regular videos at AUC 0.925, largely because vertical
   source images are often padded inside a 16:9 thumbnail. A conservative
   column-variance crop reduces the format AUC to 0.559. It is an image-content
   heuristic, not a reliable pillarbox detector, so its crop flag must not be
   interpreted as thumbnail geometry ground truth.

2. HASHTAG-STRIPPED TITLE LENGTH. Shorts titles look LONGER than regular ones
   (median 71 vs 59 characters) purely because half a Shorts title is hashtags
   (median 50.9% of characters). Strip them and the ordering inverts: 31 vs 57
   characters. `meta__title_length` means two different things by format.

3. SPEAKING RATE. Words per minute from the CLEANED transcript. Using the raw
   transcript would triple it (the auto-caption duplication artefact). Regular
   video sits at 151 wpm, inside the normal 120-160 human band; Shorts sit at
   190 and vlogs Shorts at 224, because jump-cut editing removes the silence
   between sentences. So this measures edited speech DENSITY, not articulation.

4. MEAN FRAME SAMPLING INTERVAL. Every video gets exactly 20 frames whatever its
   length, so the interval spans 0.2 s to 600 s -- a 3,000x range. Frame
   variability statistics are therefore partly a measure of how far apart the
   samples were taken (interval vs frames_std_cct: +0.24 within Shorts, +0.21
   within regular). Exposed as an explicit covariate rather than left implicit.

    .venv/bin/python 02_Data/derived_features.py --limit 100   # try it
    .venv/bin/python 02_Data/derived_features.py               # all 1,860
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import statistics
from typing import Any, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
OUT_CSV = os.path.join(DATA_DIR, "derived_features.csv")

# Candidate content columns retain enough vertical variance. This often removes
# blurred side padding but also crops some ordinary thumbnails, so it must not
# be described as a pillarbox classifier.
PILLARBOX_VAR_RATIO = 0.35   # column variance below this share of the max = bar
MIN_CONTENT_FRAC = 0.15      # never crop away more than 85% of the width

_HASHTAG = re.compile(r"#\w+")


def content_column_range(gray: Any, np: Any) -> tuple[int, int]:
    """(left, right) column bounds of the real picture inside a thumbnail.

    Returns the full width when no defensible crop is found. This is a heuristic
    based on vertical variance, not a semantic padding/geometry detector."""
    col_var = gray.var(axis=0)
    if col_var.max() <= 0:
        return 0, gray.shape[1]
    keep = col_var >= PILLARBOX_VAR_RATIO * col_var.max()
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return 0, gray.shape[1]
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    if (hi - lo) < MIN_CONTENT_FRAC * gray.shape[1]:
        return 0, gray.shape[1]      # implausible crop: treat as un-padded
    return lo, hi


def thumbnail_content_stats(path: str) -> Optional[dict[str, Any]]:
    """Brightness, saturation and contrast of the thumbnail's CONTENT region.

    Brightness and contrast are the mean and standard deviation of HSV value
    (0-255); saturation is mean HSV S. Identical definitions to the collector's,
    so the only difference from vis__thumb_* is the crop."""
    import numpy as np
    from PIL import Image
    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
            hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    except Exception:
        return None
    lo, hi = content_column_range(gray, np)
    crop = hsv[:, lo:hi, :]
    width = gray.shape[1]
    return {
        "thumb_content_frac": (hi - lo) / width,
        "thumb_was_cropped": int((hi - lo) < 0.92 * width),
        "thumb_crop_brightness": float(crop[:, :, 2].mean()),
        "thumb_crop_saturation": float(crop[:, :, 1].mean()),
        "thumb_crop_contrast": float(crop[:, :, 2].std()),
    }


def title_text_features(title: Optional[str]) -> dict[str, Any]:
    """Title length with hashtags removed, plus the hashtag count itself.

    Raw title length is not comparable across formats: a median 50.9% of a
    Shorts title's characters are hashtags. `#shorts` is counted separately
    because it is a near-perfect format tell, not a stylistic choice."""
    if not isinstance(title, str):
        return {"title_chars_nohash": None, "title_words_nohash": None,
                "title_hashtag_count": None, "title_has_shorts_tag": None,
                "title_hashtag_char_frac": None}
    tags = _HASHTAG.findall(title)
    stripped = _HASHTAG.sub("", title).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    return {
        "title_chars_nohash": len(stripped),
        "title_words_nohash": len(stripped.split()),
        "title_hashtag_count": len(tags),
        "title_has_shorts_tag": int(any(t.lower() == "#shorts" for t in tags)),
        "title_hashtag_char_frac": (1 - len(stripped) / len(title)) if title else None,
    }


def speaking_rate_wpm(words: Optional[float], duration: Optional[float]) -> Optional[float]:
    """Words per minute from the CLEANED transcript word count.

    None when either input is missing or the duration is zero. Measures edited
    speech density, not articulation rate: jump-cut Shorts routinely exceed any
    physically plausible speaking rate because the silence has been removed."""
    if not words or not duration or duration <= 0:
        return None
    return float(words) / (float(duration) / 60.0)


def _read_json(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def video_features(category: str, vid: str, video_dir: str,
                   manifest_row: dict[str, str]) -> dict[str, Any]:
    """Every derived feature for one video."""
    meta = _read_json(os.path.join(video_dir, "metadata.json")) or {}
    duration = meta.get("duration")
    if duration is None:
        extra = _read_json(os.path.join(video_dir, "metadata_extra.json")) or {}
        duration = extra.get("duration_sec_api")

    row: dict[str, Any] = {"video_id": vid, "category": category}
    thumb = next((os.path.join(video_dir, f)
                  for f in ("thumbnail.webp", "thumbnail.jpg", "thumbnail.png")
                  if os.path.exists(os.path.join(video_dir, f))), None)
    stats = thumbnail_content_stats(thumb) if thumb else None
    row.update(stats or {k: None for k in
                         ("thumb_content_frac", "thumb_was_cropped", "thumb_crop_brightness",
                          "thumb_crop_saturation", "thumb_crop_contrast")})
    row.update(title_text_features(meta.get("title")))

    words = manifest_row.get("transcript_words")
    usable = manifest_row.get("transcript_usable") == "1"
    try:
        words_f = float(words) if words not in (None, "") else None
    except ValueError:
        words_f = None
    # Only rate speech when the cleaned transcript passed the quality policy.
    # Usable text can be native or machine-translated English; the value is
    # edited transcript density, not literal vocal articulation speed.
    row["speaking_rate_wpm"] = speaking_rate_wpm(words_f, duration) if usable else None
    row["mean_frame_interval_sec"] = (float(duration) / 20.0
                                      if duration and duration > 0 else None)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest_path = os.path.join(args.data_dir, "cleaning_manifest.csv")
    if not os.path.exists(manifest_path):
        sys.exit(f"need {manifest_path} -- run clean_retrospective.py first")
    manifest = {r["video_id"]: r for r in csv.DictReader(open(manifest_path, encoding="utf-8"))}

    todo = []
    for category in sorted(os.listdir(args.data_dir)):
        cat_dir = os.path.join(args.data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for vid in sorted(os.listdir(cat_dir)):
            d = os.path.join(cat_dir, vid)
            if os.path.exists(os.path.join(d, ".done")):
                todo.append((category, vid, d))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} videos")

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(video_features, c, v, d, manifest.get(v, {})) for c, v, d in todo]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            rows.append(fut.result())
            if i % 250 == 0:
                print(f"  {i}/{len(todo)}")

    rows.sort(key=lambda r: (r["category"], r["video_id"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(vals) if vals else float("nan")

    cropped = sum(1 for r in rows if r.get("thumb_was_cropped"))
    wpm_n = sum(1 for r in rows if r.get("speaking_rate_wpm") is not None)
    print(f"\n{len(rows)} videos")
    print(f"  heuristic thumb crops  : {cropped} ({cropped/len(rows):.1%})")
    print(f"  median crop brightness : {med('thumb_crop_brightness'):.1f}  "
          f"(raw thumb_brightness medians were 144 regular vs 80 Shorts)")
    print(f"  median title chars     : {med('title_chars_nohash'):.0f} without hashtags")
    print(f"  speaking rate          : {med('speaking_rate_wpm'):.0f} wpm over {wpm_n} videos")
    print(f"  mean-frame-gap proxy   : {med('mean_frame_interval_sec'):.1f} s (median)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
