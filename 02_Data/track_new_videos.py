"""
Prospective tracking of newborn videos (FEATURES.md 6, design agreed 2026-08-26).

Builds the fixed-horizon outcome dataset the retrospective snapshot can never
give us: videos discovered at age < ~1 day, then views/likes/comments and
channel stats snapshotted daily for --track-days (default 30). From the
snapshot curves, views_at_7d (preliminary outcome) and views_at_30d (primary)
are interpolated from the observations bracketing each horizon (daily
samples rarely land at exactly 168/720h; the terminal at/after-horizon
sample guarantees day 30 is bracketed, never extrapolated), plus
subscriber_count_at_first_observation -- which
is NOT day-0 subs (even an hours-old observation may include video-driven
gains, and public counts are rounded to ~3 significant figures; that is why
hiddenSubscriberCount and the observation age are stored too).

This is a FIXED, CAPPED validation cohort (--max-cohort, default 12,000 =
the budget for the 4 main categories; the backup category adds the same
per-category cap on top, so the true main-arm ceiling is 15,000 -- see
DEFAULT_MAX_COHORT), not an unbounded sweep of everything discovery can
return: quota is
a discovery ceiling, not a target sample size, and the multimodal extraction
pipeline could never process 5,000 new videos a day anyway.

One idempotent daily "tick" does everything:
  1. discover  -- search.list per category (same categoryId+keyword mapping
     as collect_and_extract.py), order=date, publishedAfter = now - 24h,
     until the per-category / cohort caps are reached
  2. snapshot  -- videos.list (batched 50 per call, statistics+snippet) for
     every cohort video still inside --track-days whose last snapshot is
     older than --min-gap-hours; channels.list (batched 50) for their
     channels' subscriberCount / hiddenSubscriberCount / country

Designed for an OS scheduler (Windows Task Scheduler / cron), NOT a
long-running sleep loop: every run records exact observed_at_utc timestamps,
so late, missed, or doubled runs never corrupt anything -- a missed day is a
gap in the curve, a doubled run is skipped by the min-gap check. State is
the whole 02_Data/tracking/ directory (gitignored): four CSVs (cohort,
video/channel/thumbnail snapshots), discovery_state.json, and thumbnails/
-- when moving machines, copy ALL of it alongside this script, or thumbnail
history is lost and the discovery window resets.

Quota per tick: discovery is the expensive part -- up to 48 search.list
calls at the defaults (8 query arms x 2 duration filters x 3 pages; far
less once the big categories hit their caps and skip discovery), which is
nearly half of a 100-search-calls/day budget, hence the run-discovery-
once-a-day advice above. Snapshots are cheap: ~(cohort/50) videos.list +
~(channels/50) channels.list, ~100 one-unit calls/day at full cohort.

Usage:
    python3 02_Data/track_new_videos.py                # one full tick
    python3 02_Data/track_new_videos.py --no-discover  # snapshot only
    python3 02_Data/track_new_videos.py --max-cohort 12000 --track-days 30

Requires in project-root .env: YOUTUBE_API_KEY=... (use a dedicated key /
Google Cloud project so tracking never competes with collection quota).
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACK_DIR = os.path.join(os.path.dirname(__file__), "tracking")
COHORT_PATH = os.path.join(TRACK_DIR, "cohort.csv")
VIDEO_SNAPSHOTS_PATH = os.path.join(TRACK_DIR, "video_snapshots.csv")
CHANNEL_SNAPSHOTS_PATH = os.path.join(TRACK_DIR, "channel_snapshots.csv")
LOG_PATH = os.path.join(TRACK_DIR, "tracker_log.txt")
LOCK_PATH = os.path.join(TRACK_DIR, "tick.lock")
# A whole tick takes minutes; a lock older than this is a crashed tick's
# leftover and may be broken. Chosen far above any real tick duration and
# far below the scheduler interval.
STALE_LOCK_HOURS = 2.0

# Derived from collect_and_extract.py CATEGORIES (tickets #3/#4; the
# bare-categoryId-returns-0 quirk means every category needs a keyword),
# extended 2026-08-26 with two scarcity measures:
#   - product_reviews gets extra query arms ("unboxing", "first impressions")
#     -- at q="product review" alone it admitted only 9 main-arm videos/day,
#     which would leave its 750 cap ~2/3 empty at the end of the 30-day
#     window. Every video's exact source query is recorded in
#     discovery_source, so the slices stay auditable.
#   - tech_reviews (Science & Technology, categoryId 28 -- a ticket-#4
#     candidate for the product-reviews mapping) is a BACKUP category,
#     accumulating from day 2 as insurance: if product_reviews stays too
#     thin/noisy, the team can swap it in without losing cohort history.
#     Swap is a team decision with Adam; until then it is tracked alongside.
CATEGORIES = {
    "comedy": {"categoryId": "23", "queries": ["comedy"]},
    "howto": {"categoryId": "26", "queries": ["tutorial"]},
    "vlogs": {"categoryId": "22", "queries": ["vlog"]},
    "product_reviews": {"categoryId": "24", "queries": ["product review", "unboxing", "first impressions"]},
    "tech_reviews": {"categoryId": "28", "queries": ["review", "unboxing"]},
}
# Per-category cap divides max_cohort by the MAIN categories only, so the
# backup category gets the same absolute cap without shrinking the others.
MAIN_CATEGORY_COUNT = 4

# Raised 3,000 -> 12,000 on 2026-08-27 (team decision: storage/RAM are not
# binding, quota is use-it-or-lose-it, and a larger clean panel lets the
# post-deadline extension SELECT its extraction subset instead of taking
# whatever a small pool offers).
#
# ACCOUNTING (be precise -- an earlier doc claimed "12,000 total"):
# --max-cohort is the budget for the 4 MAIN categories; each gets
# max_cohort/4 (3,000). The backup category (tech_reviews) gets the SAME
# per-category cap ON TOP, so the main-arm ceiling is max_cohort * 5/4 =
# 15,000 at this default, plus the ~742 grandfathered short_form videos.
# Quota at that ceiling: ~315 videos.list + ~290 channels.list per
# snapshot pass, x2 ticks/day ~= 1,200 one-unit calls/day (~12% of
# budget); search spend is independent of the cap (~48 calls/day). The
# old wall-time constraint (serial thumbnail downloads) was removed by
# parallelizing them (THUMB_WORKERS below). Scarce categories won't reach
# their caps anyway; the raise mainly deepens comedy/howto/vlogs.
DEFAULT_MAX_COHORT = 12000
THUMB_WORKERS = 8  # concurrent CDN downloads; hashing/writes stay single-threaded
DEFAULT_TRACK_DAYS = 30
DEFAULT_MIN_GAP_HOURS = 12
# A video whose snapshots all landed BEFORE the horizon stays eligible for
# one terminal sample at/after it (else a video first seen at hour ~10 gets
# its last sample at day ~29.x and views_at_30d would need extrapolation).
# The grace bound stops long-dead videos from resurrecting after downtime.
TERMINAL_GRACE_DAYS = 3
DEFAULT_DISCOVER_WINDOW_HOURS = 24
# Quota safety cap per (query arm x duration filter) per tick; 50/page.
# With 8 query arms x 2 duration filters, 3 pages bounds a tick at ~48
# search calls worst-case (far less once the big categories hit their caps
# and skip discovery entirely). For twice-daily scheduling, run the second
# tick with --no-discover so search quota is spent once a day.
DEFAULT_DISCOVER_PAGES = 3

COHORT_FIELDS = ["video_id", "category", "channel_id", "published_at_utc",
                 "discovered_at_utc", "discovery_source", "sampling_arm",
                 "window_start_utc", "window_end_utc", "search_rank", "duration_sec"]
# Found 2026-08-26 on the day-1 cohort: at order=date, 94% of fresh uploads
# in our categories were <=180s (Shorts ceiling; comedy was 100%). Shorts
# have different distribution surfaces and virality dynamics than the
# videos this project models, so the MAIN arm requires duration >= 4min:
# discovery uses search.list's server-side videoDuration=medium/long
# filters, and every candidate's actual duration is verified via
# contentDetails before admission. Day-1 sub-4min videos were retagged to
# arm "short_form" -- still snapshotted (their growth curves are a free
# comparison dataset) but excluded from the caps.
MIN_DURATION_SEC = 240
SEARCH_DURATION_FILTERS = ("medium", "long")  # 4-20min, >20min
DISCOVERY_STATE_PATH = os.path.join(TRACK_DIR, "discovery_state.json")
# Windows overlap on purpose: publishedAfter/Before are boundary-inclusive
# and search indexing of brand-new videos can lag, so each tick re-covers
# the tail of the previous window and dedupes globally by video_id.
WINDOW_LOOKBACK_HOURS = 2.0
VIDEO_SNAP_FIELDS = ["video_id", "observed_at_utc", "age_hours",
                     "view_count", "like_count", "comment_count", "status",
                     "youtube_category_id", "title"]
CHANNEL_SNAP_FIELDS = ["channel_id", "observed_at_utc", "subscriber_count",
                       "hidden_subscriber_count", "channel_video_count", "country"]
# Thumbnails (and titles, via the snapshot rows) are captured because
# creators CHANGE them after upload -- a common optimization tactic. The
# retrospective dataset can only ever see the current thumbnail; this cohort
# captures the FIRST-OBSERVED one (discovery can lag publication by up to
# ~24h, so pre-discovery changes are invisible -- near-publish, not
# guaranteed at-publish) plus every subsequent change (a potential feature:
# "creator swapped the thumbnail after a weak first day"). A new image file
# is stored only when its hash differs from the last stored version.
THUMBS_DIR = os.path.join(TRACK_DIR, "thumbnails")
THUMBS_CSV_PATH = os.path.join(TRACK_DIR, "thumbnail_snapshots.csv")
THUMB_FIELDS = ["video_id", "observed_at_utc", "sha256", "quality", "file", "changed"]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_api_key():
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        sys.exit(f"No {env_path} -- create it with YOUTUBE_API_KEY=<your key> "
                 f"(a dedicated key is recommended so tracking never competes with collection quota)")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("YOUTUBE_API_KEY="):
                return line.strip().split("=", 1)[1]
    sys.exit(f"YOUTUBE_API_KEY not found in {env_path}")


def api_get(endpoint, params):
    """Single YouTube Data API v3 GET. Isolated so tests can stub it."""
    heartbeat()  # every API call proves this tick is alive, not wedged
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)


def parse_duration(d):
    """ISO8601 duration (PT#H#M#S) -> seconds, or None if unparseable."""
    import re
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


# ---------------------------------------------------------------------------
# State -- three plain CSVs, append-only for snapshots
# ---------------------------------------------------------------------------

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path, fields, rows):
    ensure_fields(path, fields)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def ensure_fields(path, fields):
    """Schema migration for append-only CSVs: when a release adds columns,
    rewrite the existing file once with the new header (old rows get empty
    values for the new columns) so appends stay aligned."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header == fields:
        return
    old_rows = read_csv(path)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in old_rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, path)
    log(f"migrated {os.path.basename(path)} to new schema ({len(fields)} columns)")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(api_key, cohort, args):
    """Bounded-window discovery, order=date only. Each tick covers
    [previous window end - lookback, now] -- an explicit, recorded sampling
    frame -- and pages through every nextPageToken inside it (bounded by
    --discover-pages as a quota safety cap). order=viewCount is deliberately
    never used for the main cohort: it selects on an early version of the
    outcome and would bias an underperformance model toward established
    successes. A popularity-enrichment arm, if ever added, must carry its
    own sampling_arm value and stay out of the representative test set."""
    now = utcnow()
    window_end = iso(now)
    if os.path.exists(DISCOVERY_STATE_PATH):
        with open(DISCOVERY_STATE_PATH, encoding="utf-8") as f:
            prev_end = json.load(f).get("last_window_end_utc")
        window_start = iso(parse_iso(prev_end) - datetime.timedelta(hours=WINDOW_LOOKBACK_HOURS))
    else:
        window_start = iso(now - datetime.timedelta(hours=args.discover_window_hours))

    per_category_cap = args.max_cohort // MAIN_CATEGORY_COUNT
    # Caps count only the MAIN arm -- short_form videos are tracked but must
    # never crowd regular videos out of the panel. Each category (backup
    # included) is bounded by its own cap; there is no global cut that could
    # let a high-volume category starve a scarce one.
    counts = {}
    for row in cohort:
        if row.get("sampling_arm") == "date_window":
            counts[row["category"]] = counts.get(row["category"], 0) + 1
    known = {row["video_id"] for row in cohort}

    new_rows = []
    dropped_short = 0
    failed_calls = 0
    for category, spec in CATEGORIES.items():
        if counts.get(category, 0) >= per_category_cap:
            log(f"discover {category}: at cap ({per_category_cap}), skipping")
            continue
        candidates = []
        for q in spec["queries"]:
            for duration_filter in SEARCH_DURATION_FILTERS:
                page_token = None
                rank = 0
                for _ in range(args.discover_pages):
                    params = {
                        "part": "snippet",
                        "type": "video",
                        "order": "date",
                        "publishedAfter": window_start,
                        "publishedBefore": window_end,
                        "maxResults": "50",
                        "relevanceLanguage": "en",
                        "videoCategoryId": spec["categoryId"],
                        "videoDuration": duration_filter,
                        "q": q,
                        "key": api_key,
                    }
                    if page_token:
                        params["pageToken"] = page_token
                    try:
                        data = api_get("search", params)
                    except (urllib.error.URLError, json.JSONDecodeError) as e:
                        failed_calls += 1
                        log(f"discover {category}/{q}/{duration_filter}: search failed ({e}) -- continuing")
                        break
                    for item in data.get("items", []):
                        vid = (item.get("id") or {}).get("videoId")
                        snippet = item.get("snippet") or {}
                        rank += 1
                        if not vid or vid in known:
                            continue
                        known.add(vid)
                        candidates.append({
                            "video_id": vid,
                            "category": category,
                            "channel_id": snippet.get("channelId", ""),
                            "published_at_utc": snippet.get("publishedAt", ""),
                            "discovered_at_utc": iso(now),
                            "discovery_source": f"search:{spec['categoryId']}:{q}:{duration_filter}",
                            "sampling_arm": "date_window",
                            "window_start_utc": window_start,
                            "window_end_utc": window_end,
                            "search_rank": rank,
                        })
                    page_token = data.get("nextPageToken")
                    if not page_token:
                        break
                    time.sleep(0.2)

        # Verify durations before admission -- the search filter should
        # already exclude <4min, but the recorded duration_sec must be the
        # authoritative contentDetails value, not an inference.
        durations = {}
        for batch in chunked(candidates):
            ids = ",".join(c["video_id"] for c in batch)
            try:
                data = api_get("videos", {"part": "contentDetails", "id": ids,
                                          "maxResults": "50", "key": api_key})
            except (urllib.error.URLError, json.JSONDecodeError) as e:
                failed_calls += 1
                log(f"discover {category}: duration check failed ({e}) -- batch not admitted this tick")
                continue
            for item in data.get("items", []):
                durations[item["id"]] = parse_duration((item.get("contentDetails") or {}).get("duration"))
            time.sleep(0.2)
        # Admit newest-first across ALL query arms and duration filters --
        # otherwise, near the cap, earlier arms and medium-length videos
        # would always win admission by loop order, making the capped sample
        # depend on iteration order instead of the order=date frame.
        for c in sorted(candidates, key=lambda r: (r["published_at_utc"], r["video_id"]), reverse=True):
            dur = durations.get(c["video_id"])
            if dur is None or dur < args.min_duration_sec:
                dropped_short += 1
                continue
            if counts.get(category, 0) >= per_category_cap:
                break
            c["duration_sec"] = dur
            counts[category] = counts.get(category, 0) + 1
            new_rows.append(c)
        log(f"discover {category}: cohort now {counts.get(category, 0)}/{per_category_cap}")

    if dropped_short:
        log(f"discover: {dropped_short} sub-{args.min_duration_sec}s candidates rejected "
            f"(Shorts slipping past the search duration filter)")
    if new_rows:
        append_rows(COHORT_PATH, COHORT_FIELDS, new_rows)
    if failed_calls:
        # Do NOT advance the checkpoint past a partially-failed window: the
        # 2h lookback would not re-cover it, permanently dropping whatever
        # the failed calls would have returned. Next tick re-covers the
        # whole window; global dedup makes the re-coverage free.
        log(f"discover: {failed_calls} API calls failed -- checkpoint NOT advanced, "
            f"next tick re-covers [{window_start} ..]")
    else:
        # Atomic replace: an interruption mid-write must never leave invalid
        # JSON that would crash every later tick before it can do anything.
        tmp = DISCOVERY_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"last_window_end_utc": window_end}, f)
        os.replace(tmp, DISCOVERY_STATE_PATH)
    log(f"discover: window [{window_start} .. {window_end}], +{len(new_rows)} videos, "
        f"cohort total {len(cohort) + len(new_rows)}/{args.max_cohort}")
    return cohort + new_rows


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def chunked(items, size=50):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def snapshot(api_key, cohort, args):
    now = utcnow()
    last_seen = {}
    for row in read_csv(VIDEO_SNAPSHOTS_PATH):
        last_seen[row["video_id"]] = row["observed_at_utc"]  # file is chronological

    due = []
    for row in cohort:
        try:
            published = parse_iso(row["published_at_utc"])
        except ValueError:
            continue  # unparseable publish time; discovery recorded it, nothing to age against
        age = now - published
        if age > datetime.timedelta(days=args.track_days):
            # Past the horizon: still due exactly once more, for the
            # terminal at/after-horizon sample -- unless that sample already
            # exists, or the grace window is over.
            last = last_seen.get(row["video_id"])
            has_terminal = last and (parse_iso(last) - published) >= datetime.timedelta(days=args.track_days)
            if has_terminal or age > datetime.timedelta(days=args.track_days + TERMINAL_GRACE_DAYS):
                continue
        last = last_seen.get(row["video_id"])
        if last and (now - parse_iso(last)) < datetime.timedelta(hours=args.min_gap_hours):
            continue  # doubled run -- min-gap makes ticks idempotent
        due.append(row)

    if not due:
        log("snapshot: nothing due")
        return

    snap_rows = []
    thumb_jobs = []
    for batch in chunked(due):
        ids = ",".join(r["video_id"] for r in batch)
        try:
            data = api_get("videos", {"part": "statistics,snippet", "id": ids,
                                      "maxResults": "50", "key": api_key})
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            log(f"snapshot: videos.list failed for a batch ({e}) -- those videos get no row this tick")
            continue
        returned = {item["id"]: item for item in data.get("items", [])}
        observed = iso(utcnow())
        for r in batch:
            item = returned.get(r["video_id"])
            stats = (item or {}).get("statistics", {})
            snippet = (item or {}).get("snippet", {})
            age_hours = (parse_iso(observed) - parse_iso(r["published_at_utc"])).total_seconds() / 3600.0
            snap_rows.append({
                "video_id": r["video_id"],
                "observed_at_utc": observed,
                "age_hours": f"{age_hours:.2f}",
                "view_count": stats.get("viewCount", ""),
                "like_count": stats.get("likeCount", ""),
                "comment_count": stats.get("commentCount", ""),
                # "missing" = deleted/private since discovery -- kept in the
                # cohort (attrition is itself an outcome worth reporting)
                "status": "ok" if item else "missing",
                # authoritative per-video category (self-selected by the
                # creator; discovery already filters on it, this verifies)
                "youtube_category_id": snippet.get("categoryId", ""),
                # recorded per snapshot on purpose: creators change titles
                # post-upload -- consecutive rows reveal every change
                "title": snippet.get("title", ""),
            })
            if item:
                thumb_jobs.append((r["video_id"], snippet.get("thumbnails") or {}, observed))
        time.sleep(0.2)
    append_rows(VIDEO_SNAPSHOTS_PATH, VIDEO_SNAP_FIELDS, snap_rows)
    snapshot_thumbnails(thumb_jobs)

    chan_rows = []
    channel_ids = sorted({r["channel_id"] for r in due if r["channel_id"]})
    for batch in chunked(channel_ids):
        try:
            data = api_get("channels", {"part": "statistics,snippet", "id": ",".join(batch),
                                        "maxResults": "50", "key": api_key})
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            log(f"snapshot: channels.list failed for a batch ({e})")
            continue
        observed = iso(utcnow())
        for item in data.get("items", []):
            stats = item.get("statistics", {})
            chan_rows.append({
                "channel_id": item["id"],
                "observed_at_utc": observed,
                "subscriber_count": stats.get("subscriberCount", ""),
                "hidden_subscriber_count": stats.get("hiddenSubscriberCount", ""),
                "channel_video_count": stats.get("videoCount", ""),
                "country": (item.get("snippet") or {}).get("country", ""),
            })
        time.sleep(0.2)
    append_rows(CHANNEL_SNAPSHOTS_PATH, CHANNEL_SNAP_FIELDS, chan_rows)

    missing = sum(1 for r in snap_rows if r["status"] == "missing")
    log(f"snapshot: {len(snap_rows)} video rows ({missing} missing), "
        f"{len(chan_rows)} channel rows, {len(cohort) - len(due)} videos skipped (aged out or not due)")


# Unique per process: release_lock only ever removes a lock this process
# wrote, so a tick whose lock was (wrongly or rightly) taken over can never
# delete the new owner's lock on its way out.
_LOCK_TOKEN = f"{os.getpid()}:{os.urandom(8).hex()}"


def acquire_lock():
    """One tick at a time: two overlapping invocations (scheduled + manual,
    or a stalled tick meeting the next one) would both read the same state,
    both consider everything due, and append duplicate rows / overwrite
    state concurrently. O_EXCL creation is atomic on every platform.
    Staleness is judged by mtime age, and the RUNNING tick heartbeats the
    lock (see heartbeat()) after every API batch -- so an old mtime means
    the owner has done nothing for STALE_LOCK_HOURS, i.e. it is dead or
    wedged beyond usefulness, not merely slow."""
    os.makedirs(TRACK_DIR, exist_ok=True)
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, _LOCK_TOKEN.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(LOCK_PATH)
            except OSError:
                continue  # holder just released it; retry the create
            if age > STALE_LOCK_HOURS * 3600:
                try:
                    os.remove(LOCK_PATH)
                except OSError:
                    pass
                continue
            return False


def heartbeat():
    """Refresh the lock mtime so a long-but-alive tick is never mistaken
    for a crashed one. Called after every API/download batch."""
    try:
        os.utime(LOCK_PATH)
    except OSError:
        pass


def release_lock():
    try:
        with open(LOCK_PATH, encoding="utf-8") as f:
            if f.read() != _LOCK_TOKEN:
                return  # not our lock (taken over after a stall) -- leave it
    except OSError:
        return
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def fetch_url(url):
    """Plain HTTP GET returning bytes. Isolated so tests can stub it."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def snapshot_thumbnails(thumb_jobs):
    """Download each due video's current thumbnail (best available quality
    from the snippet already fetched -- no extra API quota, only CDN GETs),
    hash it, and store the image ONLY when the hash differs from the last
    stored version for that video. Every check is logged as a CSV row, so
    thumbnail-change events and their timing are recoverable exactly."""
    if not thumb_jobs:
        return
    import hashlib

    os.makedirs(THUMBS_DIR, exist_ok=True)
    last_hash = {}
    versions = {}
    for row in read_csv(THUMBS_CSV_PATH):
        last_hash[row["video_id"]] = row["sha256"]  # chronological file
        if row["changed"] == "true":
            versions[row["video_id"]] = versions.get(row["video_id"], 0) + 1

    def fetch_one(job):
        """Runs in a worker thread: pick best-quality URL, download, return
        (video_id, observed, quality, blob-or-None). No shared state is
        touched here -- hashing and all writes happen in the main thread."""
        video_id, thumbnails, observed = job
        heartbeat()
        url = quality = None
        for q in ("maxres", "standard", "high", "medium", "default"):
            if q in thumbnails and thumbnails[q].get("url"):
                url, quality = thumbnails[q]["url"], q
                break
        if not url:
            return video_id, observed, None, None
        try:
            return video_id, observed, quality, fetch_url(url)
        except (urllib.error.URLError, OSError):
            return video_id, observed, quality, None

    rows = []
    changed = failed = 0
    # Downloads are the slow part (~0.3s each; a 15k cohort would take over
    # an hour serially). Parallelize ONLY the network fetch; pool.map
    # preserves job order, so CSV output stays deterministic. Consume the
    # iterator lazily INSIDE the pool context: materializing it would hold
    # every downloaded image at once (15k maxres blobs ~ 2GB) instead of
    # hashing and releasing each as it arrives.
    import concurrent.futures

    def fetched():
        with concurrent.futures.ThreadPoolExecutor(max_workers=THUMB_WORKERS) as pool:
            yield from pool.map(fetch_one, thumb_jobs)

    for video_id, observed, quality, blob in fetched():
        if quality is None:
            continue  # no thumbnail URL in the snippet
        if blob is None:
            failed += 1
            continue
        digest = hashlib.sha256(blob).hexdigest()
        is_new = last_hash.get(video_id) != digest
        fname = ""
        if is_new:
            n = versions.get(video_id, 0)
            fname = f"{video_id}_v{n}.jpg"
            with open(os.path.join(THUMBS_DIR, fname), "wb") as f:
                f.write(blob)
            versions[video_id] = n + 1
            changed += 1
        last_hash[video_id] = digest
        rows.append({
            "video_id": video_id,
            "observed_at_utc": observed,
            "sha256": digest,
            "quality": quality,
            "file": fname,  # empty = unchanged since last stored version
            "changed": "true" if is_new else "false",
        })
    append_rows(THUMBS_CSV_PATH, THUMB_FIELDS, rows)
    log(f"thumbnails: {len(rows)} checked, {changed} new/changed images stored, {failed} download failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cohort", type=int, default=DEFAULT_MAX_COHORT)
    parser.add_argument("--track-days", type=int, default=DEFAULT_TRACK_DAYS)
    parser.add_argument("--min-gap-hours", type=float, default=DEFAULT_MIN_GAP_HOURS)
    parser.add_argument("--discover-window-hours", type=float, default=DEFAULT_DISCOVER_WINDOW_HOURS)
    parser.add_argument("--discover-pages", type=int, default=DEFAULT_DISCOVER_PAGES)
    parser.add_argument("--min-duration-sec", type=int, default=MIN_DURATION_SEC,
                        help="main-arm admission floor (Shorts protection)")
    parser.add_argument("--no-discover", action="store_true",
                        help="snapshot only (e.g. after the cohort is full/frozen)")
    args = parser.parse_args()

    if not acquire_lock():
        print(f"another tick is already running ({LOCK_PATH}) -- exiting without touching state")
        return
    try:
        api_key = load_api_key()
        cohort = read_csv(COHORT_PATH)
        log(f"tick start: cohort {len(cohort)} videos")
        if not args.no_discover:
            cohort = discover(api_key, cohort, args)
        snapshot(api_key, cohort, args)
        log("tick done")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
