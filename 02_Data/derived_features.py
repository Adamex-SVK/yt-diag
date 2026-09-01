"""
Derived features, each one added because the descriptive EDA of 2026-09-01
showed an existing feature was measuring the wrong thing.

Every feature here exists to fix a specific, measured defect. The evidence is
recorded beside each so nobody has to re-derive why it is here:

1. CONTENT-CROP THUMBNAIL STATISTICS. 91.4% of Shorts thumbnails are a vertical
   frame letterboxed into a 16:9 box with blurred, darkened bars either side.
   `vis__thumb_brightness` therefore measures the bars: it separates Shorts from
   regular videos at AUC 0.925, and it produces a sign reversal (pooled
   brightness-vs-views -0.033 while both strata are positive). Measured on the
   CONTENT REGION only, the format leak collapses to AUC 0.480 -- formats become
   statistically indistinguishable -- and the reversal disappears (pooled +0.124,
   agreeing with regular +0.116 and Shorts +0.137). This turns a format detector
   back into a perceptual feature.

2. HASHTAG-STRIPPED TITLE LENGTH. Shorts titles look LONGER than regular ones
   (median 71 vs 59 characters) purely because half a Shorts title is hashtags
   (median 50.9% of characters). Strip them and the ordering inverts: 31 vs 57
   characters. `meta__title_length` means two different things by format.

3. SPEAKING RATE. Words per minute from the CLEANED transcript. Using the raw
   transcript would triple it (the auto-caption duplication artefact). Regular
   video sits at 151 wpm, inside the normal 120-160 human band; Shorts sit at
   190 and vlogs Shorts at 224, because jump-cut editing removes the silence
   between sentences. So this measures edited speech DENSITY, not articulation.

4. STATIC-VIDEO FLAG. 10.8% of the collection is single-scene: locked-camera
   stand-up, one-angle tutorials, held photo cards. Nothing in the shipped
   feature set can distinguish those from an edited video with the same average
   brightness, and an attribution layer must not tell a locked-camera comedian
   to improve their pacing.

5. FRAME SAMPLING INTERVAL. Every video gets exactly 20 frames whatever its
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
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_and_extract import compute_frame_timestamps  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
OUT_CSV = os.path.join(DATA_DIR, "derived_features.csv")

# A pillarbox column is near-uniform vertically (the bar is a blurred, heavily
# smoothed stretch of the frame) and much darker than the content. Detect it by
# per-column vertical variance rather than by assuming 9:16, so a 4:3 letterbox
# or an unpadded vertical thumbnail is handled by the same rule.
PILLARBOX_VAR_RATIO = 0.35   # column variance below this share of the max = bar
MIN_CONTENT_FRAC = 0.15      # never crop away more than 85% of the width

# A consecutive-frame difference above this is a scene change. Calibrated on the
# real distribution: the floor rate moves only 4.0% -> 22.6% across T=30..60, so
# the flag is not sensitive to the exact value.
SCENE_MAD = 45.0

_HASHTAG = re.compile(r"#\w+")


def content_column_range(gray: Any, np: Any) -> tuple[int, int]:
    """(left, right) column bounds of the real picture inside a thumbnail.

    Returns the full width when no pillarbox is detected. Bars are found by
    vertical variance: a blurred bar barely changes down a column, while real
    content does."""
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
        "thumb_is_pillarboxed": int((hi - lo) < 0.92 * width),
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


def _frame_gray(path: str, np: Any) -> Optional[Any]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    except Exception:
        return None


def scene_features(video_dir: str, duration: Optional[float]) -> dict[str, Any]:
    """Scene-change count and the static flag, from the 20 stored frames.

    `scene_changes` counts consecutive pairs differing by more than SCENE_MAD.
    It is a LOWER BOUND and not a cutting rate: the frames are seconds to
    minutes apart, so cuts between two samples are invisible. Zero of them is
    still meaningful -- it identifies genuinely single-scene video."""
    import numpy as np
    frames_dir = os.path.join(video_dir, "frames")
    out: dict[str, Any] = {"scene_changes": None, "is_single_scene": None,
                           "frame_interval_sec": None}
    if duration and duration > 0:
        out["frame_interval_sec"] = float(duration) / 20.0
    if not os.path.isdir(frames_dir):
        return out
    names = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    arrs = [a for a in (_frame_gray(os.path.join(frames_dir, n), np) for n in names) if a is not None]
    if len(arrs) < 2:
        return out
    consec = [float(np.abs(arrs[i + 1] - arrs[i]).mean()) for i in range(len(arrs) - 1)]
    n = sum(1 for d in consec if d >= SCENE_MAD)
    out["scene_changes"] = n
    out["is_single_scene"] = int(n == 0)
    return out


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
                         ("thumb_content_frac", "thumb_is_pillarboxed", "thumb_crop_brightness",
                          "thumb_crop_saturation", "thumb_crop_contrast")})
    row.update(title_text_features(meta.get("title")))

    words = manifest_row.get("transcript_words")
    usable = manifest_row.get("transcript_usable") == "1"
    try:
        words_f = float(words) if words not in (None, "") else None
    except ValueError:
        words_f = None
    # only rate speech we believe is real: an unusable transcript is
    # hallucinated or machine-translated text, and its wpm is meaningless
    row["speaking_rate_wpm"] = speaking_rate_wpm(words_f, duration) if usable else None
    row.update(scene_features(video_dir, duration))
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
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(vals) if vals else float("nan")

    pill = sum(1 for r in rows if r.get("thumb_is_pillarboxed"))
    single = sum(1 for r in rows if r.get("is_single_scene"))
    wpm_n = sum(1 for r in rows if r.get("speaking_rate_wpm") is not None)
    print(f"\n{len(rows)} videos")
    print(f"  pillarboxed thumbnails : {pill} ({pill/len(rows):.1%})")
    print(f"  median crop brightness : {med('thumb_crop_brightness'):.1f}  "
          f"(raw thumb_brightness medians were 144 regular vs 80 Shorts)")
    print(f"  median title chars     : {med('title_chars_nohash'):.0f} without hashtags")
    print(f"  speaking rate          : {med('speaking_rate_wpm'):.0f} wpm over {wpm_n} videos")
    print(f"  single-scene videos    : {single} ({single/len(rows):.1%})")
    print(f"  median frame interval  : {med('frame_interval_sec'):.1f} s")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
