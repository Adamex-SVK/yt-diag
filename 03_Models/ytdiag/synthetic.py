"""Synthetic retrospective dataset -- a fake 02_Data/processed/ tree with the
exact file layout collect_and_extract.py + compute_labels_v2.py produce, and
a PLANTED signal so the pipeline can be validated end to end before Adam's
data lands: a latent 'views' depends on a few features (duration, face in
thumbnail, publish hour, channel size) plus noise; the label is the
within-category top quartile of that latent, like the real v2 label.

Usage:
    python3 -m ytdiag.synthetic --out /tmp/fake_processed --n 1200
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
from typing import Any

CATEGORIES = ["comedy", "howto", "vlogs", "product_reviews"]
EGEMAPS_NAMES = [f"egemaps_feat_{i:02d}" for i in range(88)]  # real names come from openSMILE


def make_video(
    rng: random.Random,
    category: str,
    channel: dict[str, Any],
    days_old: int,
    collected: datetime.datetime,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], int]:
    """The four JSON payloads one video directory holds (metadata, visual,
    audio, metadata_extra) plus its latent view count, which `generate` ranks
    into the label.

    The views are not noise around a constant: they are driven by a planted
    signal (face in thumbnail, evening publish hour, over-long duration,
    channel size) that a working pipeline must partially recover, mixed with
    gaussian noise loud enough that it cannot be recovered fully. So AUC near
    0.5 on this data means the pipeline is broken, and AUC near 1.0 means
    something leaks -- neither is a property of real videos.

    Units: `days_old` days, `collected` the pretend collection time (naive UTC),
    `duration` seconds, the returned count absolute views at collection. Views
    grow with both channel size and age, which is exactly why the label has to
    be cut within a category rather than globally.
    """
    duration = int(rng.lognormvariate(6.3, 0.7))  # ~550s median
    subs = channel["subs"]
    has_face = rng.random() < 0.55
    hour = rng.randrange(24)
    upload = collected - datetime.timedelta(days=days_old, hours=rng.random() * 24)
    published_at = upload.replace(hour=hour, minute=rng.randrange(60), second=0)
    # planted signal: faces + evening hours + moderate duration help; noise dominates
    signal = (0.6 * has_face + 0.4 * (18 <= hour <= 22) - 0.3 * (duration > 1500)
              + 0.8 * (subs ** 0.6) / 1000 + rng.gauss(0, 1.2))
    views = int(max(1, (subs ** 0.8) * (days_old ** 0.5) * (2.718 ** signal) * 0.05))
    meta = {
        "id": None, "title": f"{category} video {rng.randrange(10**6)}",
        "description": "lorem ipsum " * rng.randrange(1, 40), "tags": ["a"] * rng.randrange(0, 15),
        "categories": [category], "upload_date": upload.strftime("%Y%m%d"),
        "duration": duration, "view_count": views, "like_count": views // 40,
        "comment_count": views // 400, "channel_id": channel["id"],
        "channel_follower_count": subs, "license": "standard",
        "definition": "hd" if rng.random() < 0.9 else "sd",
        "collected_at": collected.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta["title_length"] = len(meta["title"]); meta["description_length"] = len(meta["description"])
    meta["tag_count"] = len(meta["tags"])
    thumb = {"cct": rng.gauss(5500, 900), "brightness": rng.gauss(120, 30), "saturation": rng.gauss(90, 30),
             "contrast": rng.gauss(55, 12), "has_face": has_face, "face_count": int(has_face) + (rng.random() < 0.2),
             "max_face_area_ratio": rng.random() * 0.3 if has_face else 0.0,
             "face_centrality": rng.random() * 0.4 if has_face else None}
    visual = {"thumbnail": thumb, "frames_mean_cct": rng.gauss(5400, 700), "frames_std_cct": rng.random() * 800,
              "frames_mean_brightness": rng.gauss(110, 25), "frames_mean_saturation": rng.gauss(85, 25),
              "frames_mean_contrast": rng.gauss(50, 10), "frames_has_face_ratio": rng.random(),
              "frames_mean_max_face_area_ratio": rng.random() * 0.2}
    audio = {"egemaps": {n: rng.gauss(0, 1) for n in EGEMAPS_NAMES},
             "pauses": {"pause_count": rng.randrange(0, 80), "total_pause_sec": rng.random() * 60,
                        "pause_ratio": rng.random() * 0.3, "mean_pause_sec": rng.random() * 2}}
    extra = {"published_at_utc": published_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "caption_available": rng.random() < 0.7, "youtube_category_id": "23",
             "channel_country": rng.choice(["US", "GB", "DE", "IN", ""]),
             "default_language": "", "default_audio_language": rng.choice(["en", "en", "en", "de", ""]),
             "channel_created_at": (upload - datetime.timedelta(days=rng.randint(1, 3000))).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "channel_video_count": str(rng.choice([1, 2, 5, 20, 100, 800])),
             "is_short": "true" if duration <= 60 and rng.random() < 0.9 else "false",
             "status": "ok"}
    return meta, visual, audio, extra, views


def generate(out_dir: str, n: int = 1200, seed: int = 0) -> str:
    """Write `n` fake videos under `out_dir` in the exact processed/ layout and
    return `out_dir`, so `load_retrospective(generate(...))` is a full pipeline
    check without touching real data or the network.

    Deterministic for a given `seed` -- the tests assert on numbers produced by
    a specific seed, so changing the draw order changes those assertions.

    Videos are spread over only `n // 4` channels because the channel-grouped
    split has nothing to hold out otherwise, and ages are drawn from 31-600 days
    so every video clears compute_labels_v2's 30-day minimum. The label is the
    top floor(n_category/4) of the latent views within a category (so roughly
    n/16 positives overall, not n/4) -- a stand-in for v2's
    category x age-band x size-band stratification, not a reproduction of it,
    and without v2's min-1 floor (a category with fewer than 4 videos gets no
    positive at all).

    Every video gets a `.done` marker and the five JSON/text artifacts, but NO
    thumbnail image and NO frames/ directory -- so asset__thumbnail_path and
    asset__frames_dir come back None on every row, and this tree exercises the
    adapter's missing-asset branches rather than the all-present path.
    """
    rng = random.Random(seed)
    collected = datetime.datetime(2026, 8, 25, 12, 0, 0)
    channels = [{"id": f"UCsynth{i:05d}", "subs": int(10 ** rng.uniform(2.5, 6.5))} for i in range(n // 4)]
    per_cat = {c: [] for c in CATEGORIES}
    for i in range(n):
        category = CATEGORIES[i % 4]
        channel = rng.choice(channels)
        vid = f"syn{i:06d}"
        d = os.path.join(out_dir, category, vid)
        os.makedirs(d, exist_ok=True)
        meta, visual, audio, extra, views = make_video(rng, category, channel, rng.randint(31, 600), collected)
        meta["id"] = vid
        for name, obj in (("metadata.json", meta), ("visual_features.json", visual),
                          ("audio_features.json", audio), ("metadata_extra.json", extra),
                          ("transcript_info.json", {"source": "auto_captions", "had_auto_captions": True})):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                json.dump(obj, f)
        with open(os.path.join(d, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write("synthetic transcript words " * 30)
        open(os.path.join(d, ".done"), "w").close()
        per_cat[category].append((vid, d, views))
    # label: within-category top quartile of views (stand-in for compute_labels_v2)
    for category, rows in per_cat.items():
        rows.sort(key=lambda r: r[2])
        k = len(rows) // 4
        for i, (vid, d, views) in enumerate(rows):
            label = "viral" if i >= len(rows) - k else "typical"
            with open(os.path.join(d, "label.json"), "w", encoding="utf-8") as f:
                json.dump({"label_version": 2, "label": label, "view_count": views}, f)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(generate(a.out, a.n, a.seed))
