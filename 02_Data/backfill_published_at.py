"""
One-shot backfill of publish timestamps (+ two more drifted fields) for the
already-collected retrospective dataset (FEATURES.md 6).

collect_and_extract.py keeps only yt-dlp's upload_date (YYYYMMDD -- date, no
time), so the planned publish-time features (UTC hour sin/cos, weekday,
time-of-day bucket) have nothing to compute from. This script fills the gap
from the official API in batches of 50 ids per call:

  videos.list  (snippet,contentDetails) -> published_at_utc (full ISO
               timestamp), caption_available (the "caption availability"
               feature #12, which never made it into metadata.json),
               youtube_category_id
  channels.list (snippet,statistics)    -> channel_country (for timezone
               conversion; self-declared, often missing -- record, never
               guess), hidden_subscriber_count

Cost: ~160 videos.list + ~<=160 channels.list batched calls for the full
8,000-video dataset -- trivial under any quota interpretation.

Writes metadata_extra.json into each video's directory rather than editing
Adam's metadata.json in place -- collection may still be running, and a
separate file cannot corrupt or race the collector's own writes. Resumable:
video dirs that already have metadata_extra.json are skipped.

Run this on the machine that holds 02_Data/processed/ (Adam's server).

Usage:
    python3 02_Data/backfill_published_at.py
    python3 02_Data/backfill_published_at.py --category comedy
    python3 02_Data/backfill_published_at.py --data-dir /path/to/processed

Requires in project-root .env: YOUTUBE_API_KEY=...
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
EXTRA_NAME = "metadata_extra.json"


def load_api_key():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        sys.exit(f"No {env_path} -- create it with YOUTUBE_API_KEY=<your key>")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("YOUTUBE_API_KEY="):
                return line.strip().split("=", 1)[1]
    sys.exit(f"YOUTUBE_API_KEY not found in {env_path}")


def api_get(endpoint, params):
    """Single YouTube Data API v3 GET. Isolated so tests can stub it."""
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def pending_videos(data_dir, only_category):
    """(video_id, video_dir, channel_id) for every .done video without an
    extra file yet. channel_id comes from the existing metadata.json."""
    pending = []
    if not os.path.isdir(data_dir):
        sys.exit(f"No data directory at {data_dir} -- run collect_and_extract.py first")
    for category in sorted(os.listdir(data_dir)):
        if only_category and category != only_category:
            continue
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for video_id in sorted(os.listdir(cat_dir)):
            video_dir = os.path.join(cat_dir, video_id)
            if not os.path.exists(os.path.join(video_dir, ".done")):
                continue
            if os.path.exists(os.path.join(video_dir, EXTRA_NAME)):
                continue  # resumable: already backfilled
            try:
                with open(os.path.join(video_dir, "metadata.json"), encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            pending.append((video_id, video_dir, meta.get("channel_id") or ""))
    return pending


def chunked(items, size=50):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="backfill only this category")
    parser.add_argument("--data-dir", default=OUT_DIR)
    parser.add_argument("--pacing", type=float, default=0.2, help="seconds between API calls")
    args = parser.parse_args()

    api_key = load_api_key()
    pending = pending_videos(args.data_dir, args.category)
    print(f"{len(pending)} videos to backfill")
    if not pending:
        return

    # Channel countries first, one lookup per unique channel, reused across
    # all of that channel's videos.
    channel_info = {}
    failed_channels = set()
    channel_ids = sorted({c for _, _, c in pending if c})
    for batch in chunked(channel_ids):
        try:
            data = api_get("channels", {"part": "snippet,statistics", "id": ",".join(batch),
                                        "maxResults": "50", "key": api_key})
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            # A transient failure must not become permanent missing data:
            # videos of these channels are skipped (no file written), so a
            # re-run retries them instead of finding them "done".
            failed_channels.update(batch)
            print(f"WARNING: channels.list batch failed ({e}) -- those channels' videos stay pending for a re-run")
            continue
        for item in data.get("items", []):
            channel_info[item["id"]] = {
                "channel_country": (item.get("snippet") or {}).get("country", ""),
                "hidden_subscriber_count": (item.get("statistics") or {}).get("hiddenSubscriberCount", ""),
            }
        time.sleep(args.pacing)

    written = missing = skipped_pending = 0
    for batch in chunked(pending):
        kept = [(v, d, c) for v, d, c in batch if c not in failed_channels]
        skipped_pending += len(batch) - len(kept)
        if not kept:
            continue
        batch = kept
        ids = ",".join(vid for vid, _, _ in batch)
        try:
            data = api_get("videos", {"part": "snippet,contentDetails", "id": ids,
                                      "maxResults": "50", "key": api_key})
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"WARNING: videos.list batch failed ({e}) -- those videos stay pending for a re-run")
            time.sleep(args.pacing)
            continue
        returned = {item["id"]: item for item in data.get("items", [])}
        for vid, video_dir, channel_id in batch:
            item = returned.get(vid)
            if item is None:
                # Deleted/private since collection: record that explicitly so
                # the re-run skip logic doesn't retry it forever.
                extra = {"published_at_utc": None, "status": "missing_from_api"}
                missing += 1
            else:
                snippet = item.get("snippet") or {}
                content = item.get("contentDetails") or {}
                extra = {
                    "published_at_utc": snippet.get("publishedAt"),
                    "caption_available": content.get("caption") == "true",
                    "youtube_category_id": snippet.get("categoryId", ""),
                    "status": "ok",
                }
            extra.update(channel_info.get(channel_id, {}))
            extra["backfilled_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(os.path.join(video_dir, EXTRA_NAME), "w", encoding="utf-8") as f:
                json.dump(extra, f, indent=2)
            written += 1
        time.sleep(args.pacing)

    print(f"Done: {written} written ({missing} missing from API), "
          f"{len(pending) - written} still pending "
          f"({skipped_pending} held back by failed channel batches -- re-run to retry)")


if __name__ == "__main__":
    main()
