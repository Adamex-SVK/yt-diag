"""
One-shot backfill of publish timestamps (+ the other fields metadata.json never captured) for the
already-collected retrospective dataset (FEATURES.md 6).

collect_and_extract.py keeps only yt-dlp's upload_date (YYYYMMDD -- date, no
time), so the planned publish-time features (UTC hour sin/cos, weekday,
time-of-day bucket) have nothing to compute from. This script fills the gap
from the official API in batches of 50 ids per call:

  videos.list  (snippet,contentDetails) -> published_at_utc (full ISO
               timestamp), caption_available (the "caption availability"
               feature #12, which never made it into metadata.json),
               youtube_category_id, default_language / default_audio_language
               (uploader-declared, often empty)
  channels.list (snippet,statistics)    -> channel_country (for timezone
               conversion; self-declared, often missing -- record, never
               guess), hidden_subscriber_count, channel_created_at and
               channel_video_count (channel age / first-upload features)

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yt_shorts  # noqa: E402  -- definitive Shorts check shared with track_new_videos.py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
EXTRA_NAME = "metadata_extra.json"
_known_verdicts = {}  # video_id -> "true"/"false" already in metadata_extra.json (never re-probed or overwritten)


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
    _known_verdicts.clear()  # rebuilt from the files on every scan
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
            extra_path = os.path.join(video_dir, EXTRA_NAME)
            if os.path.exists(extra_path):
                try:
                    with open(extra_path, encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if existing.get("is_short") in ("true", "false"):
                    _known_verdicts[video_id] = existing["is_short"]
                if "published_at_utc" in existing or existing.get("status") == "missing_from_api":
                    continue  # resumable: API fields already backfilled (a shorts-only file is not enough)
            try:
                with open(os.path.join(video_dir, "metadata.json"), encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            pending.append((video_id, video_dir, meta.get("channel_id") or ""))
    return pending


def _write_extra(video_dir, extra):
    """Merge into an existing metadata_extra.json (e.g. one written by
    --shorts-only) rather than overwriting it."""
    path = os.path.join(video_dir, EXTRA_NAME)
    merged = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                merged = json.load(f)
        except (OSError, json.JSONDecodeError):
            merged = {}
    merged.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def shorts_only(data_dir, only_category, recheck=False):
    """Definitive Shorts verdict for every .done video without one, written
    (merged) into metadata_extra.json. No API quota: one page fetch each."""
    todo = []
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
            path = os.path.join(video_dir, EXTRA_NAME)
            existing = {}
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        existing = json.load(f)
                except (OSError, json.JSONDecodeError):
                    existing = {}
            if existing.get("is_short") == "false" or (existing.get("is_short") == "true" and not recheck):
                continue
            todo.append((video_id, video_dir))
    print(f"{len(todo)} videos to classify via the /shorts/ URL test")
    verdicts = yt_shorts.classify_many([v for v, _ in todo])
    aborted = bool(verdicts.pop("__aborted__", None))
    if aborted:
        print("WARNING: check run aborted after consecutive failures -- consent bypass or network broken? "
              "No verdicts withdrawn; re-run on a stable connection.")
    written = unknown = inconclusive = 0
    for vid, video_dir in todo:
        v = verdicts.get(vid)
        if v is None:  # inconclusive probe (consent page, network, rate limit): never write or withdraw
            inconclusive += 1
            continue
        if not v:  # resolved unknown: the page says the video is deleted/private
            unknown += 1
            if recheck and not aborted:  # withdraw only on a HEALTHY recheck; an unstable run withdraws nothing
                _write_extra(video_dir, {"is_short": "", "is_short_checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            continue
        _write_extra(video_dir, {"is_short": v, "is_short_checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        written += 1
    print(f"Done: {written} verdicts written, {unknown} unknown (deleted/private"
          + (", withdrawn)" if recheck and not aborted else ")") + f", {inconclusive} inconclusive (re-run to retry)")


def chunked(items, size=50):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", help="backfill only this category")
    parser.add_argument("--data-dir", default=OUT_DIR)
    parser.add_argument("--pacing", type=float, default=0.2, help="seconds between API calls")
    parser.add_argument("--shorts-only", action="store_true",
                        help="only run the definitive /shorts/ URL test (no API key needed) and record is_short "
                             "for every .done video lacking a verdict; API fields are left for a later full run")
    parser.add_argument("--recheck", action="store_true",
                        help="with --shorts-only: re-examine videos currently is_short=true (unavailable videos "
                             "used to be misread as Shorts); a deleted/private page withdraws the old verdict, "
                             "but only on a healthy (non-aborted) run -- inconclusive probes never do")
    args = parser.parse_args()

    if args.shorts_only:
        shorts_only(args.data_dir, args.category, recheck=args.recheck)
        return

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
                # channel age / first-upload features. videoCount is the CURRENT
                # count (post-outcome for old videos) -- coarse, disclose.
                "channel_created_at": (item.get("snippet") or {}).get("publishedAt", ""),
                "channel_video_count": (item.get("statistics") or {}).get("videoCount", ""),
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
        # only probe videos that exist and have no definitive verdict yet
        verdicts = yt_shorts.classify_many([vid for vid, _, _ in batch if vid in returned and vid not in _known_verdicts])
        verdicts.pop("__aborted__", None)
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
                    # uploader-declared language fields (often empty; audio
                    # language is set more often) -- yt-dlp had them, the
                    # collector's metadata.json dropped them
                    "default_language": snippet.get("defaultLanguage", ""),
                    "default_audio_language": snippet.get("defaultAudioLanguage", ""),
                    "status": "ok",
                }
            extra.update(channel_info.get(channel_id, {}))
            verdict = _known_verdicts.get(vid) or verdicts.get(vid)
            if verdict:  # definitive only -- an unknown never overwrites a stored verdict
                extra["is_short"] = verdict
                extra["is_short_checked_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            extra["backfilled_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_extra(video_dir, extra)
            written += 1
        time.sleep(args.pacing)

    print(f"Done: {written} written ({missing} missing from API), "
          f"{len(pending) - written} still pending "
          f"({skipped_pending} held back by failed channel batches -- re-run to retry)")


if __name__ == "__main__":
    main()
