"""
Collection + feature-extraction pipeline for YT-Diag (wayfinder ticket #1,
Priority 4 in data_retrieval.md).

For each target category: discover video IDs via the YouTube Data API
search.list, then for each video use yt-dlp + ffmpeg to pull metadata,
thumbnail, auto-captions, frames, and audio-derived features -- then DELETE
the downloaded video (and extracted audio) file. The video itself is never
kept; only its derived features are.

Frame sampling and modality choice follow the locked Tier-1 architecture
(YT-Diag Architecture Digest, 2026-08-14): 16-24 frames, denser in the first
60s where engagement signal concentrates -- NOT uniform across the full
duration. Visual (color temperature, brightness/saturation/contrast, face
presence/area) and audio (eGeMAPS prosody) engineered features follow
additional_features.md 1:1 -- extracted now, while the video/audio is still
on disk, rather than re-downloading later if the metadata baseline shows
headroom for them.

Designed to be copied onto a second machine (e.g. a school computer) and run
unattended: every step is idempotent and resumable via a per-video ".done"
marker, so a crash or a killed process just means re-running the same
command picks up where it left off.

Per the addendum in cc_availability_scan_findings.md, the CC-only license
constraint was explicitly dropped by team decision -- discovery below does
NOT filter by videoLicense.

Category mapping (ticket #4, closed 2026-08-20) and dataset size (ticket #3,
closed 2026-08-20, 8,000 total / 2,000 per category) are both locked -- see
CATEGORIES and the --target default below.

Usage:
    source .venv/bin/activate
    python3 02_Data/collect_and_extract.py --category comedy --target 2000
    python3 02_Data/collect_and_extract.py --category comedy --target 2000 --resume
    python3 02_Data/collect_and_extract.py --category all --target 2000

Transcript: auto-captions are used if they pass a quality heuristic
(word count > 50, <=20% [Music]/[Applause]-style tag lines -- data_retrieval.md
#4.3); otherwise falls back to Whisper (small.en) on the audio track already
on disk. mediapipe is optional (face detection falls back to OpenCV's
bundled Haar cascade if it isn't installed).

Requires on PATH: yt-dlp, ffmpeg, ffprobe.
Requires in project-root .env: YOUTUBE_API_KEY=... (discovery only).
Requires in venv (requirements.txt): opencv-python-headless, numpy,
opensmile, openai-whisper.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Optional
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(__file__), "processed")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "collection_manifest.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "collection_log.txt")

# Ticket #4 (closed 2026-08-20): product_reviews = Entertainment (24) +
# keyword workaround -- bare categoryId=24/22 return 0 due to a search.list
# filter quirk (cc_availability_scan_findings.md). Confirmed 2026-08-22 that
# this quirk has spread to EVERY bare categoryId query (comedy=23, howto=26
# now also return 0 with no q) -- verified directly against the live API,
# not just observed once. Every category now carries a keyword as a result.
CATEGORIES = {
    "comedy": {"categoryId": "23", "q": "comedy"},
    "howto": {"categoryId": "26", "q": "tutorial"},
    "vlogs": {"categoryId": "22", "q": "vlog"},
    "product_reviews": {"categoryId": "24", "q": "product review"},
}

# Ticket #3 (closed 2026-08-20): 8,000 total, uniform 2,000/category.
DEFAULT_TARGET_PER_CATEGORY = 2000

# Locked in the Architecture Digest (2026-08-14): 16-24 frames, denser in
# the first DENSE_WINDOW_SEC seconds.
FRAME_COUNT = 20
DENSE_WINDOW_SEC = 60.0
DENSE_FRACTION = 0.6  # share of frames spent inside the dense window
MAX_HEIGHT = 720
SEARCH_PAGE_SIZE = 50
PAUSE_THRESHOLD_SEC = 0.3  # additional_features.md 2.3: silence >300ms = a pause

# Confirmed 2026-08-20: YouTube's current anti-bot JS challenge makes
# yt-dlp<2026.08.19 fail video downloads outright ("unable to download video
# data: HTTP Error 403"); the fix is a recent yt-dlp (requirements.txt) plus
# telling it to fetch the challenge-solver component. Applied to every
# yt-dlp call, not just download_video(), since this is a moving target on
# YouTube's side and metadata/captions calls could start needing it too.
YT_DLP_EXTRA_ARGS = ["--remote-components", "ejs:github"]

# Confirmed 2026-08-22: firing requests for consecutive videos with no
# pause is exactly the pattern that triggered a hard bot-check block
# ("Sign in to confirm you're not a bot") on a school network -- 10 videos
# in ~30s, every single one rejected. Default pacing is deliberately
# conservative; override with --pacing if a given network tolerates faster.
VIDEO_PACING_SEC = 5.0

_optional_import_warned = set()


_io_lock = threading.Lock()  # guards log/manifest file writes across worker threads


def log(msg: str) -> None:
    """Timestamped line to stdout and appended to collection_log.txt.

    Held under _io_lock because worker threads log concurrently: the log is the
    only post-hoc account of a multi-day unattended run, and interleaved partial
    writes would make it unreadable exactly when a run went wrong.
    """
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with _io_lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Found 2026-08-23: an overnight --workers 8 run collapsed after ~11.5
# hours -- yt-dlp's cookies-based session broke (explicit "Sign in to
# confirm your age" / "does not look like a Netscape format cookies file"
# errors), and once that happened, YouTube returned a generic
# "Video unavailable" for nearly every subsequent video instead of the
# real per-video reason. Nothing detected this, so the run burned through
# its entire remaining ~1,100-video queue as instant fake failures (up to
# 145/minute -- far faster than real processing) instead of stopping. This
# guards against that: any failure whose message matches a known
# auth-break signature bumps a shared counter; a real success (or a
# failure that ISN'T an auth-break signature) resets it. Crossing the
# threshold sets _abort_event, and process_video() skips remaining jobs
# instead of quietly mislabeling the rest of the queue as failed.
AUTH_BREAK_THRESHOLD = 6
_AUTH_BREAK_PATTERNS = (
    "sign in to confirm your age",
    "sign in to confirm you're not a bot",
    "sign in to confirm you’re not a bot",
    "does not look like a netscape format cookies file",
    "use --cookies-from-browser or --cookies for the authentication",
)
_auth_break_lock = threading.Lock()
_auth_break_count = 0
_abort_event = threading.Event()


def _is_auth_break_error(err_text):
    lowered = err_text.lower()
    return any(pat in lowered for pat in _AUTH_BREAK_PATTERNS)


def _note_result(success, err_text=""):
    """Track consecutive auth-break-signature failures across worker
    threads; trip _abort_event once AUTH_BREAK_THRESHOLD is hit in a row."""
    global _auth_break_count
    with _auth_break_lock:
        if success or not _is_auth_break_error(err_text):
            _auth_break_count = 0
            return
        _auth_break_count += 1
        count = _auth_break_count
    if count == AUTH_BREAK_THRESHOLD and not _abort_event.is_set():
        _abort_event.set()
        log(f"ABORT: {count} consecutive auth-break-signature failures "
            f"(cookies session likely dead/blocked) -- skipping all remaining "
            f"queued videos instead of burning through them as fake failures. "
            f"Refresh cookies.txt and re-run with --resume.")


def check_dependencies() -> None:
    """Verify yt-dlp/ffmpeg/ffprobe/deno are on PATH before any work starts.

    sys.exit()s with an install hint rather than raising: a missing binary would
    otherwise surface as thousands of identical CalledProcessError rows in the
    manifest, i.e. a whole run's worth of videos marked "failed" for a reason
    that has nothing to do with the videos.
    """
    for tool in ("yt-dlp", "ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"Missing required tool on PATH: {tool}. Install it before running this script.")
    if shutil.which("deno") is None:
        # Confirmed 2026-08-20: yt-dlp's YouTube extractor now requires a JS
        # runtime for some videos ("No supported JavaScript runtime could be
        # found") and fails the download without one. Not optional in
        # practice -- caught this via a real failed download, not a guess.
        sys.exit("Missing required tool on PATH: deno (yt-dlp needs a JS runtime for YouTube extraction). "
                  "Install with: brew install deno (macOS) or see https://deno.land for other platforms.")


def load_api_key() -> str:
    """YOUTUBE_API_KEY from the project-root .env.

    Used for discovery only -- yt-dlp never sees it -- so a run driven by
    --input-ids does not call this at all. Raises RuntimeError if the key is
    missing, which is the right failure: without discovery there are no jobs.
    """
    env_path = os.path.join(ROOT, ".env")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("YOUTUBE_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"YOUTUBE_API_KEY not found in {env_path}")


def warn_once(name: str, msg: str) -> None:
    """Log `msg` the first time this `name` is seen, then never again.

    Optional-dependency warnings live on a per-video code path, so without this
    a single missing package emits one line per video -- 8,000 copies that bury
    the real failures. `name` is the dedup key, not the message text.

    Not locked: a duplicate warning under a thread race is harmless, and taking
    _io_lock here would nest it inside log()'s own acquisition of the same lock.
    """
    if name not in _optional_import_warned:
        _optional_import_warned.add(name)
        log(f"WARNING: {msg}")


# ---------------------------------------------------------------------------
# Discovery (YouTube Data API search.list -- video IDs only, no license filter)
# ---------------------------------------------------------------------------

def discover_video_ids(api_key: str, category_id: Optional[str], q: Optional[str],
                       target_count: int) -> list[str]:
    """Up to `target_count` unique video IDs from search.list, in API order.

    Returns FEWER than target_count without error, and that is the normal case,
    not a failure: search.list stops issuing pageTokens long before the
    pageInfo.totalResults estimate is exhausted (cc_availability_scan_findings.md
    measured the gap). Callers must size the run off len() of this, never off
    what they asked for.

    IDs are deduplicated across pages because search.list repeats them, and both
    filters are omitted when falsy -- but note a bare videoCategoryId with no `q`
    returns 0 results on the live API, which is why every CATEGORIES entry
    carries a keyword.
    """
    seen = []
    seen_set = set()
    page_token = None
    while len(seen) < target_count:
        params = {
            "part": "id",
            "type": "video",
            "maxResults": str(SEARCH_PAGE_SIZE),
            "key": api_key,
            # data_retrieval.md #2.3 bias-mitigation table: constrain to
            # English explicitly. Confirmed necessary 2026-08-20 -- a
            # non-English test video (no English auto-captions) fell through
            # to Whisper small.en, an English-only model, and produced a
            # garbage transcript instead of a clean fallback.
            "relevanceLanguage": "en",
        }
        if category_id:
            params["videoCategoryId"] = category_id
        if q:
            params["q"] = q
        if page_token:
            params["pageToken"] = page_token
        url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.load(resp)
        items = data.get("items", [])
        for item in items:
            vid = item.get("id", {}).get("videoId")
            if vid and vid not in seen_set:
                seen_set.add(vid)
                seen.append(vid)
        page_token = data.get("nextPageToken")
        if not page_token or not items:
            break
        time.sleep(0.2)
    return seen[:target_count]


# ---------------------------------------------------------------------------
# Manifest (human-readable progress log, one row per video)
# ---------------------------------------------------------------------------

def manifest_append(video_id: str, category: str, status: str, error: str = "") -> None:
    """Append one row to collection_manifest.csv, writing the header if the file
    is new. `status` is "done" or "failed"; `error` is the truncated stderr.

    Append-only and locked across threads on purpose: this is the audit trail
    used to reconcile what was attempted against what landed on disk, so a video
    that fails twice across two runs correctly appears twice.
    """
    with _io_lock:
        is_new = not os.path.exists(MANIFEST_PATH)
        with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["video_id", "category", "status", "error", "timestamp"])
            writer.writerow([video_id, category, status, error, time.strftime("%Y-%m-%d %H:%M:%S")])


def is_done(video_dir: str) -> bool:
    """True if the .done marker exists -- the only signal --resume trusts.

    process_video() writes .done last, after the temp video and audio are
    deleted, so its presence means every artefact is on disk. A directory that
    exists but has no marker is therefore a partial video, and is redone from
    scratch rather than repaired.
    """
    return os.path.exists(os.path.join(video_dir, ".done"))


# ---------------------------------------------------------------------------
# Per-video processing -- metadata, thumbnail, captions, video/frames
# ---------------------------------------------------------------------------

def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """subprocess.run with check=True and captured text output.

    Every external-tool call goes through here so that a non-zero exit raises
    CalledProcessError with stderr attached to it -- process_video() writes that
    stderr into the manifest, and _note_result() pattern-matches it to detect a
    dead cookies session. Both depend on the message surviving; a bare
    subprocess.run that let output go to the terminal would lose it.
    """
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def fetch_metadata(video_id: str, video_dir: str) -> dict[str, Any]:
    """Write the kept subset of yt-dlp's --dump-json to metadata.json, and
    return the same dict.

    Only the baseline's fields are kept; the rest of the dump is mostly playback
    URLs that expire within hours and would be dead weight in the dataset.

    Every count here (view/like/comment/channel_follower) is a value AS OF
    collected_at, not a property of the video: they keep rising afterwards, so
    any label computed from them must age-normalise against collected_at rather
    than against the time the model is trained. `duration` is seconds. Fields
    absent from the dump are stored as None, so a null in metadata.json means
    "YouTube did not report it" (likes hidden by the creator, for instance), not
    zero.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    result = run(["yt-dlp", *YT_DLP_EXTRA_ARGS, "--dump-json", "--skip-download", url])
    data = json.loads(result.stdout)
    keep = {
        "id": data.get("id"),
        "title": data.get("title"),
        "description": data.get("description"),
        "tags": data.get("tags"),
        "categories": data.get("categories"),
        "upload_date": data.get("upload_date"),
        "duration": data.get("duration"),
        "view_count": data.get("view_count"),
        "like_count": data.get("like_count"),
        "comment_count": data.get("comment_count"),
        "channel_id": data.get("channel_id"),
        "channel_follower_count": data.get("channel_follower_count"),
        "license": data.get("license"),
        "definition": data.get("definition"),  # HD/SD -- data_retrieval.md #5.4 feature 9
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Derived metadata-baseline columns (data_retrieval.md #5.4 features 6-8)
    # -- computed here, not left for a later preprocessing pass, so
    # metadata.json alone is a complete row for the LR/XGBoost baseline.
    keep["title_length"] = len(keep["title"] or "")
    keep["description_length"] = len(keep["description"] or "")
    keep["tag_count"] = len(keep["tags"] or [])
    with open(os.path.join(video_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2)
    return keep


def fetch_thumbnail(video_id: str, video_dir: str) -> None:
    """Download the thumbnail into video_dir as thumbnail.<ext>.

    The extension is whatever YouTube served (.webp as often as .jpg), so
    consumers must glob for "thumbnail.*" -- process_video() and the adapters
    both do. Raises CalledProcessError on failure, which fails the whole video:
    a thumbnail is a required modality, not an optional extra.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    run([
        "yt-dlp", *YT_DLP_EXTRA_ARGS, "--write-thumbnail", "--skip-download",
        "-o", os.path.join(video_dir, "thumbnail.%(ext)s"),
        url,
    ])


def fetch_captions(video_id: str, video_dir: str) -> Optional[str]:
    """Returns the path to captions.en.srt if yt-dlp produced one, else None
    -- absence is expected for many videos, not fatal on its own (see
    transcribe() below for the Whisper fallback that covers that case)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        run([
            "yt-dlp", *YT_DLP_EXTRA_ARGS, "--write-auto-subs", "--sub-langs", "en", "--skip-download",
            "--convert-subs", "srt",
            "-o", os.path.join(video_dir, "captions.%(ext)s"),
            url,
        ])
    except subprocess.CalledProcessError:
        return None
    srt_path = os.path.join(video_dir, "captions.en.srt")
    return srt_path if os.path.exists(srt_path) else None


_SRT_TAG_RE = None  # set on first use, avoids importing re at module load for a one-off


def srt_to_text(srt_path: str) -> list[str]:
    """Caption LINES with SRT index and timestamp lines stripped -- a list, not
    a joined string.

    Callers join it themselves because the per-line split is load-bearing:
    caption_quality_ok() counts bare [Music]-style tag lines against the line
    total, and process_video() drops those same lines before joining.

    Side effect callers depend on: compiles the module-level _SRT_TAG_RE on
    first use. Anything touching _SRT_TAG_RE directly must have called this
    first, or it is still None.

    Decode errors are ignored rather than raised -- auto-caption SRTs are
    occasionally mis-encoded, and losing a character is cheaper than losing the
    video's entire transcript.
    """
    import re
    global _SRT_TAG_RE
    if _SRT_TAG_RE is None:
        _SRT_TAG_RE = re.compile(r"^\[[^\]]+\]$")
    lines = []
    with open(srt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.isdigit() or "-->" in line:
                continue
            lines.append(line)
    return lines


def caption_quality_ok(srt_path: str) -> bool:
    """data_retrieval.md #4.3 heuristic: word count > 50, no more than 20%
    of lines are bare [Music]/[Applause]-style tags.

    False means "use Whisper instead", not "this video has no speech" -- an
    empty or unreadable file also returns False. The 50-word floor is what keeps
    a music video's two spoken lines from being accepted as a transcript.
    """
    lines = srt_to_text(srt_path)
    if not lines:
        return False
    tag_lines = sum(1 for l in lines if _SRT_TAG_RE.match(l))
    word_count = sum(len(l.split()) for l in lines)
    tag_ratio = tag_lines / len(lines)
    return word_count > 50 and tag_ratio <= 0.2


_whisper_pool = None  # queue.Queue of loaded model instances, sized to --workers
_whisper_available = True  # set False by preload_whisper() if openai-whisper isn't installed

# History: originally one shared model instance behind a global lock, so
# transcription -- the slowest single step per video -- ran fully serialized
# regardless of --workers. That was a deliberate RAM tradeoff on a normal
# machine (~1GB+ per extra instance). On the real collection hardware (Dell
# PowerEdge R740, 96 threads, 1.5TB RAM -- see compute_scaling.md) that
# tradeoff wastes almost all of the machine: a real --workers 8 run averaged
# ~63s/video end-to-end even with downloads/frames/features overlapping,
# because every video still had to wait its turn for the one Whisper
# instance -- extrapolating to ~5-6 days for the full 8,000-video run, i.e.
# --workers barely helped. Switched to a small pool of N instances (found
# 2026-08-22) so transcription runs with the same concurrency as everything
# else. Each instance gets a slice of the CPU (torch.set_num_threads is a
# process-global intraop budget, so it's divided by N up front, not fought
# over at call time) to avoid N instances each trying to grab all 96 threads.


def preload_whisper(num_instances: int = 1) -> None:
    """Load Whisper/torch BEFORE any worker threads exist -- must be called
    from main(), single-threaded, prior to creating the ThreadPoolExecutor.

    Confirmed 2026-08-22: putting `import whisper` behind a lock inside
    transcribe_with_whisper() was NOT enough -- failure rate on a real run
    went from ~15% (3 workers) to ~25-35% (8 workers), scaling with thread
    count even with the import serialized. That points to a known Windows-
    specific issue, not a simple import race: some native DLLs (torch's
    C10/MKL among them) use thread-local storage in a way that Windows must
    retroactively initialize for every thread already alive in the process
    at LoadLibrary time, with a hard limit -- so loading torch's DLL for the
    first time while N worker threads already exist can fail purely from
    thread *count*, independent of whether the import call itself is
    serialized. Loading it once in the main thread, before any worker
    threads are spawned, sidesteps this entirely -- once the DLL is
    resident, later loads from any thread just bump a refcount, no re-init.
    Loading all `num_instances` copies here (still single-threaded, still
    before the ThreadPoolExecutor exists) keeps that guarantee for all of
    them, not just the first.

    A missing openai-whisper is degradation, not failure: _whisper_available
    goes False after one warning, and every video that would have needed the
    fallback finishes with transcript_source "none" and no transcript.txt. Such
    a run still marks those videos .done, so the gap is invisible afterwards
    except in transcript_info.json -- check the warning at the top of the log."""
    global _whisper_pool, _whisper_available
    try:
        import torch
        import whisper
    except ImportError:
        _whisper_available = False
        warn_once("whisper", "openai-whisper not installed -- videos with missing/poor auto-captions "
                              "will have NO transcript. Install it (requirements.txt) to get the fallback.")
        return
    num_instances = max(1, num_instances)
    threads_per_instance = max(1, (os.cpu_count() or 1) // num_instances)
    torch.set_num_threads(threads_per_instance)
    log(f"loading {num_instances} Whisper small.en instance(s) "
        f"({threads_per_instance} CPU threads each, before workers start)...")
    _whisper_pool = queue.Queue()
    for _ in range(num_instances):
        _whisper_pool.put(whisper.load_model("small.en"))


def transcribe_with_whisper(audio_path: str) -> Optional[str]:
    """data_retrieval.md #4.2/#4.3 fallback for missing/poor auto-captions.
    Assumes preload_whisper() already ran in main() before any worker
    threads started. Borrows one instance from the pool for the duration of
    the call so up to `num_instances` transcriptions run concurrently.

    None and "" mean different things and process_video() records the
    difference: None is "openai-whisper is not installed, no attempt was made"
    (transcript_source "none"), while "" is "Whisper ran and heard nothing"
    (transcript_source "whisper", no transcript.txt written).

    The instance is returned to the pool in a finally block -- a transcription
    that raises must not permanently shrink the pool, since after
    `num_instances` such failures every remaining video would block forever on
    an empty queue.

    English-only (small.en), which is why discovery constrains
    relevanceLanguage=en: a non-English video reaching here produces confident
    nonsense rather than an empty result.
    """
    if not _whisper_available:
        return None
    model = _whisper_pool.get()
    try:
        result = model.transcribe(audio_path, fp16=False)
    finally:
        _whisper_pool.put(model)
    return result["text"].strip()


def download_video(video_id: str, tmp_path: str) -> None:
    """Download the video muxed with its audio to `tmp_path`, capped at
    MAX_HEIGHT pixels of vertical resolution.

    The file is temporary by contract, not by convenience: process_video()
    deletes it on both the success and failure paths, so no video file is ever
    retained (the TOS constraint in the module docstring). Anything that needs
    pixels or audio must extract it before that deletion.

    Raises CalledProcessError on failure; the stderr it carries is what the
    auth-break breaker reads to tell a dead cookies session from a genuinely
    unavailable video.
    """
    # bestvideo+bestaudio (not bestvideo alone) -- audio track is required
    # for the eGeMAPS/prosody features below, not just the video frames.
    url = f"https://www.youtube.com/watch?v={video_id}"
    run([
        "yt-dlp", *YT_DLP_EXTRA_ARGS, "-f", f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]",
        "--merge-output-format", "mp4",
        "-o", tmp_path,
        url,
    ])


def probe_duration(video_path: str) -> float:
    """Duration of the file on disk, in SECONDS.

    Deliberately measured from the downloaded container rather than reused from
    metadata.json's `duration`: the two disagree when yt-dlp merges a slightly
    short stream, and frame timestamps computed from the advertised length would
    then seek past the end and hand ffmpeg a failing frame.

    Raises ValueError if ffprobe reports no parseable duration (some live-stream
    remnants), which fails the video rather than sampling it at nonsense times.
    """
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ])
    return float(result.stdout.strip())


def compute_frame_timestamps(duration: float, frame_count: int) -> list[float]:
    """16-24 frames, denser in the first DENSE_WINDOW_SEC seconds (locked
    in the Architecture Digest: 'engagement signal concentrates' there).
    Falls back to plain uniform sampling for videos shorter than the
    dense window, where the split is meaningless.

    `duration` and the returned timestamps are both in SECONDS. Positions are
    bin midpoints, so no timestamp is ever 0.0 or exactly `duration` -- both
    ends of a video are routinely black or a fade, and a black frame's mean
    colour is not a feature.

    Consequence for downstream models: the frames are NOT uniformly spaced, so
    frame index is not proportional to time. Anything treating the 20 frames as
    an evenly sampled sequence is wrong for any video longer than
    DENSE_WINDOW_SEC.
    """
    if duration <= DENSE_WINDOW_SEC:
        return [(i + 0.5) / frame_count * duration for i in range(frame_count)]

    dense_count = max(1, round(frame_count * DENSE_FRACTION))
    sparse_count = frame_count - dense_count
    dense_ts = [(i + 0.5) / dense_count * DENSE_WINDOW_SEC for i in range(dense_count)]
    remaining = duration - DENSE_WINDOW_SEC
    sparse_ts = [DENSE_WINDOW_SEC + (i + 0.5) / sparse_count * remaining for i in range(sparse_count)] \
        if sparse_count > 0 else []
    return dense_ts + sparse_ts


def extract_frames(video_path: str, frames_dir: str, frame_count: int) -> list[str]:
    """Write `frame_count` JPEGs as frames/frame_NN.jpg and return their paths
    in time order (index order == sampling order, but see
    compute_frame_timestamps(): the spacing is not uniform).

    One ffmpeg call per frame, with -ss BEFORE -i so ffmpeg seeks to the nearest
    keyframe instead of decoding the file from the start; a single-pass select
    filter would decode every frame of a 20-minute video to keep 20 of them.
    The cost is `frame_count` process spawns per video, which is why this is
    still the slow step on a low-core machine.

    Raises CalledProcessError if any single frame fails, so a video is never
    marked done with a short frame set that a later consumer would read as a
    complete one.
    """
    os.makedirs(frames_dir, exist_ok=True)
    duration = probe_duration(video_path)
    timestamps = compute_frame_timestamps(duration, frame_count)
    paths = []
    for i, timestamp in enumerate(timestamps):
        out_path = os.path.join(frames_dir, f"frame_{i:02d}.jpg")
        run([
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
            "-frames:v", "1", "-q:v", "2", out_path,
        ])
        paths.append(out_path)
    return paths


# ---------------------------------------------------------------------------
# Visual engineered features (additional_features.md 1.1-1.3): color
# temperature, brightness/saturation/contrast, face presence/area.
# ---------------------------------------------------------------------------

# Correlated colour temperature is only meaningful near the Planckian locus.
# The locus approximation below is valid over exactly this range; keeping one
# shared bound prevents the old 1500-vs-1667 K mismatch.
CCT_VALID_K = (1667.0, 25000.0)
CCT_VERSION = 3
CCT_METHOD = "nearest_planckian_locus_cie1960uv_1pct_lut"

# CCT is the temperature at the closest point on the Planckian locus in CIE
# 1960 UCS. Duv is the signed distance to that closest point. Saturated colours
# sit far off the locus, so reject them instead of assigning plausible-looking
# temperatures to colours (such as green) that no blackbody produces.
MAX_DUV = 0.05

# sRGB (IEC 61966-2-1) linear-RGB -> CIE XYZ, D65 white point.
_SRGB_TO_XYZ = ((0.4124564, 0.3575761, 0.1804375),
                (0.2126729, 0.7151522, 0.0721750),
                (0.0193339, 0.1191920, 0.9503041))


def srgb_to_linear(c: float) -> float:
    """Undo the sRGB transfer function for one channel, 0-1 in and out.

    Pixel values are gamma-ENCODED; averaging or matrix-multiplying them
    directly treats a perceptual code value as a light intensity. This is the
    step the original CCT implementation omitted."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _planckian_xy(temp_k: float) -> "tuple[float, float]":
    """Approximate CIE 1931 xy of a blackbody at temp_k (Kang et al. cubics,
    valid 1667-25000 K) -- used only to measure how far a colour sits from the
    locus and interpolate the nearest temperature."""
    t = temp_k
    if t <= 4000.0:
        x = (-0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2
             + 0.8776956e3 / t + 0.179910)
    else:
        x = (-3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2
             + 0.2226347e3 / t + 0.240390)
    if t <= 2222.0:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return x, y


def _uv_1960(x: float, y: float) -> "tuple[float, float]":
    """CIE 1931 xy -> CIE 1960 UCS uv, the space Duv is defined in."""
    d = -2.0 * x + 12.0 * y + 3.0
    if d == 0:
        return float("nan"), float("nan")
    return 4.0 * x / d, 6.0 * y / d


_planckian_lut_cache = None


def _planckian_lut() -> "list[tuple[float, float, float]]":
    """Planckian-locus (K, u, v) table with no step wider than 1% in K.

    Projecting onto its line segments finds the closest locus point rather
    than assuming an approximate CCT first and measuring distance only there.
    The table is built once and needs no optional colour-science dependency,
    so collection and modelling machines execute the identical method.
    """
    global _planckian_lut_cache
    if _planckian_lut_cache is None:
        lo, hi = CCT_VALID_K
        temperatures = [lo]
        while temperatures[-1] < hi:
            temperatures.append(min(hi, temperatures[-1] * 1.01))
        _planckian_lut_cache = [
            (temp, *_uv_1960(*_planckian_xy(temp))) for temp in temperatures
        ]
    return _planckian_lut_cache


def _nearest_planckian_cct_duv_uv(u: float, v: float) -> "tuple[float, float]":
    """Return (CCT, signed Duv) at the closest point on the locus polyline.

    Positive Duv is above the Planckian locus and negative is below it. CCT is
    interpolated in reciprocal temperature, as in Robertson-style tables.
    """
    best = None
    locus = _planckian_lut()
    for (t0, u0, v0), (t1, u1, v1) in zip(locus, locus[1:]):
        du, dv = u1 - u0, v1 - v0
        length2 = du * du + dv * dv
        alpha = ((u - u0) * du + (v - v0) * dv) / length2
        alpha = min(1.0, max(0.0, alpha))
        qu, qv = u0 + alpha * du, v0 + alpha * dv
        ru, rv = u - qu, v - qv
        distance2 = ru * ru + rv * rv
        if best is None or distance2 < best[0]:
            reciprocal_t = (1.0 - alpha) / t0 + alpha / t1
            cct = 1.0 / reciprocal_t
            # The LUT runs warm-to-cool (u decreases); its right-hand normal
            # points above the locus, which is the positive-Duv convention.
            cross = du * rv - dv * ru
            sign = 1.0 if cross <= 0.0 else -1.0
            best = (distance2, cct, sign)
    assert best is not None
    return best[1], best[2] * best[0] ** 0.5


def correlated_color_temp_and_duv(
    mean_linear_rgb: "tuple[float, float, float]",
) -> "Optional[tuple[float, float]]":
    """Nearest-locus (CCT in kelvin, signed Duv), or None for black.

    Takes LINEAR RGB in 0-1 (see srgb_to_linear), averaged in linear space --
    not the mean of gamma-encoded pixel values. Version 3 replaces McCamy's
    increasingly biased high-temperature cubic and its approximate distance
    check with a direct closest-point search on the Planckian locus.
    """
    r, g, b = mean_linear_rgb
    X = _SRGB_TO_XYZ[0][0] * r + _SRGB_TO_XYZ[0][1] * g + _SRGB_TO_XYZ[0][2] * b
    Y = _SRGB_TO_XYZ[1][0] * r + _SRGB_TO_XYZ[1][1] * g + _SRGB_TO_XYZ[1][2] * b
    Z = _SRGB_TO_XYZ[2][0] * r + _SRGB_TO_XYZ[2][1] * g + _SRGB_TO_XYZ[2][2] * b
    denom = X + Y + Z
    if denom <= 0:
        return None  # pure black: no chromaticity to speak of
    x = X / denom
    y = Y / denom
    u, v = _uv_1960(x, y)
    if not math.isfinite(u) or not math.isfinite(v):
        return None
    return _nearest_planckian_cct_duv_uv(u, v)


def correlated_color_temp(mean_linear_rgb: "tuple[float, float, float]") -> "Optional[float]":
    """CCT in kelvin, or None when black or farther than MAX_DUV from the locus."""
    result = correlated_color_temp_and_duv(mean_linear_rgb)
    if result is None:
        return None
    cct, duv = result
    return cct if abs(duv) <= MAX_DUV else None


def population_std(values: "list[float]") -> "Optional[float]":
    """Population standard deviation shared by collection and recomputation."""
    if not values:
        return None
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _face_detector():
    # mediapipe>=1.0 removed the legacy `solutions` API this was written
    # against (replaced by the Tasks API) -- exactly the "tool churn" risk
    # data_retrieval.md already flagged for yt-dlp. Caught broadly (not just
    # ImportError) so an API-shape change degrades to the sanctioned Haar
    # cascade fallback instead of crashing every video.
    try:
        import mediapipe as mp
        detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        def detect(bgr_image):
            import cv2
            rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            boxes = []
            if result.detections:
                for det in result.detections:
                    box = det.location_data.relative_bounding_box
                    boxes.append((box.width * box.height, box.xmin + box.width / 2, box.ymin + box.height / 2))
            return boxes
        return detect
    except (ImportError, AttributeError) as e:
        warn_once("mediapipe", f"mediapipe face detection unavailable ({e}) -- falling back to OpenCV Haar cascade "
                                "(per additional_features.md 1.3, sanctioned fallback, less accurate)")
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)

        def detect(bgr_image):
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            boxes = []
            for (fx, fy, fw, fh) in faces:
                area_ratio = (fw * fh) / (w * h)
                cx = (fx + fw / 2) / w
                cy = (fy + fh / 2) / h
                boxes.append((area_ratio, cx, cy))
            return boxes
        return detect


_linearise_lut = None


def _linearise_table(np):
    """256-entry uint8 -> linear-light lookup table, built once.

    Pixels are uint8, so the sRGB transfer function has only 256 possible
    inputs; a table turns a per-pixel pow() into an index."""
    global _linearise_lut
    if _linearise_lut is None:
        _linearise_lut = np.array([srgb_to_linear(i / 255.0) for i in range(256)],
                                  dtype=np.float64)
    return _linearise_lut


def _analyze_image(cv2, path, detect_faces):
    import numpy as np
    img = cv2.imread(path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_bgr = img.reshape(-1, 3).mean(axis=0)
    # linear-space mean: gamma-encoded pixels are code values, not intensities
    lin = _linearise_table(np)[img[:, :, :3]].reshape(-1, 3).mean(axis=0)  # BGR order
    cct = correlated_color_temp((float(lin[2]), float(lin[1]), float(lin[0])))
    brightness = float(hsv[:, :, 2].mean())
    saturation = float(hsv[:, :, 1].mean())
    contrast = float(hsv[:, :, 2].std())
    boxes = detect_faces(img)
    max_face = max(boxes, key=lambda b: b[0]) if boxes else None
    return {
        "cct": cct,
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "has_face": bool(boxes),
        "face_count": len(boxes),
        "max_face_area_ratio": max_face[0] if max_face else 0.0,
        "face_centrality": (
            math.hypot(max_face[1] - 0.5, max_face[2] - 0.5) if max_face else None
        ),
    }


def extract_visual_features(thumbnail_path: Optional[str], frame_paths: list[str],
                            video_dir: str) -> Optional[dict[str, Any]]:
    """Write visual_features.json (a "thumbnail" block plus across-frame
    aggregates) and return it. None means opencv is not installed and NO file
    was written -- distinct from a written file whose values are None.

    Units, none of which are 0-1 unless said so: *_cct is McCamy's
    correlated-colour-temperature approximation, nominally Kelvin, but computed
    by _correlated_color_temp() from mean RGB fed in as CIE XYZ directly (no
    sRGB->XYZ matrix, no gamma linearisation) -- so read it as a monotone
    warm/cool index, not a calibrated temperature. brightness and saturation are
    the means of the HSV V and S channels on the 0-255 byte scale; contrast is
    the standard deviation of V on that same 0-255 scale; *_face_area_ratio is a
    fraction of frame area; frames_has_face_ratio is the fraction of readable
    frames with at least one face.

    A None aggregate means no frame yielded a value for that key -- it is not
    zero, and averaging it as zero would push a video toward "cold and dark".
    Aggregates are also taken only over the frames OpenCV could actually decode
    and, per key, only over non-None values, so a mean may be over fewer than
    len(frame_paths) frames; that count is not recorded anywhere, so denominators
    here are not comparable across videos.

    "thumbnail" is None when no thumbnail file was found or it failed to decode,
    which is why consumers must not assume the block exists.
    """
    try:
        import cv2
    except ImportError:
        warn_once("opencv", "opencv-python(-headless) not installed -- skipping visual engineered features "
                             "(color temp/brightness/saturation/face). Install it to get these.")
        return None

    detect_faces = _face_detector()
    thumb = _analyze_image(cv2, thumbnail_path, detect_faces) if thumbnail_path and os.path.exists(thumbnail_path) else None
    frames = [a for a in (_analyze_image(cv2, p, detect_faces) for p in frame_paths) if a is not None]

    def _mean(key, source):
        vals = [s[key] for s in source if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _std(key, source):
        vals = [s[key] for s in source if s.get(key) is not None]
        if not vals:
            return None
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    features = {
        "thumbnail": thumb,
        "frames_mean_cct": _mean("cct", frames),
        "frames_std_cct": _std("cct", frames),
        "frames_mean_brightness": _mean("brightness", frames),
        "frames_mean_saturation": _mean("saturation", frames),
        "frames_mean_contrast": _mean("contrast", frames),
        "frames_has_face_ratio": _mean("has_face", [{"has_face": 1.0 if f["has_face"] else 0.0} for f in frames]),
        "frames_mean_max_face_area_ratio": _mean("max_face_area_ratio", frames),
    }
    with open(os.path.join(video_dir, "visual_features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    return features


# ---------------------------------------------------------------------------
# Audio / prosodic features (additional_features.md 2): eGeMAPS via
# opensmile, extracted from the audio track of the video already on disk.
# ---------------------------------------------------------------------------

def extract_audio_track(video_path: str, audio_path: str) -> None:
    """Demux to 16 kHz mono signed-16-bit PCM WAV at `audio_path`.

    That exact format is a hard requirement, not a default: webrtcvad
    (_pause_stats) accepts only 8/16/32/48 kHz mono 16-bit PCM and raises
    otherwise, and Whisper resamples everything to 16 kHz anyway, so writing it
    once here serves both consumers. wave.open() in _pause_stats also assumes
    2-byte samples when it slices frames.

    Raises CalledProcessError when the video has no audio stream at all, which
    fails the whole video.
    """
    run([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ])


def extract_audio_features(audio_path: str, video_dir: str) -> Optional[dict[str, Any]]:
    """Write audio_features.json as {"egemaps": {...}, "pauses": {...}} and
    return it. None means opensmile is not installed and no file was written.

    "egemaps" is the eGeMAPSv02 Functionals set: ~88 named scalars summarising
    the WHOLE track with no time axis, so it describes a video's average voice,
    never its dynamics. Anything wanting "the intro is loud, the rest is flat"
    has to come from somewhere else.

    "pauses" is None inside an otherwise-complete result when webrtcvad is
    missing -- a partial success, not a failure, so callers must null-check the
    inner key as well as the return value.

    Dumped with default=float because opensmile returns numpy scalars, which
    json cannot encode; that also silently coerces any NaN to the literal NaN,
    which is not valid JSON for strict parsers.
    """
    try:
        import opensmile
    except ImportError:
        warn_once("opensmile", "opensmile not installed -- skipping audio/prosody features "
                                "(pitch, loudness, pauses, eGeMAPS). Install it to get these.")
        return None

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    egemaps = smile.process_file(audio_path).iloc[0].to_dict()

    pause_stats = _pause_stats(audio_path)

    features = {"egemaps": egemaps, "pauses": pause_stats}
    with open(os.path.join(video_dir, "audio_features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, default=float)
    return features


def _pause_stats(audio_path):
    """Pause count/ratio via webrtcvad (additional_features.md 2.3),
    independent of eGeMAPS -- eGeMAPS' own voicing stats cover pitch/energy,
    this adds the explicit pause-length distribution."""
    try:
        import webrtcvad
        import wave
    except ImportError:
        warn_once("webrtcvad", "webrtcvad not installed -- skipping explicit pause-length stats "
                                "(eGeMAPS voicing features still cover part of this if opensmile is installed)")
        return None

    vad = webrtcvad.Vad(2)
    with wave.open(audio_path, "rb") as wf:
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    frame_ms = 30
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
    frames = [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
    frames = [f for f in frames if len(f) == frame_bytes]

    silence_run = 0.0
    pause_lengths = []
    for f in frames:
        is_speech = vad.is_speech(f, sample_rate)
        if is_speech:
            if silence_run >= PAUSE_THRESHOLD_SEC:
                pause_lengths.append(silence_run)
            silence_run = 0.0
        else:
            silence_run += frame_ms / 1000.0
    if silence_run >= PAUSE_THRESHOLD_SEC:
        pause_lengths.append(silence_run)

    total_duration = len(frames) * frame_ms / 1000.0
    total_pause = sum(pause_lengths)
    return {
        "pause_count": len(pause_lengths),
        "total_pause_sec": total_pause,
        "pause_ratio": (total_pause / total_duration) if total_duration else None,
        "mean_pause_sec": (total_pause / len(pause_lengths)) if pause_lengths else None,
    }


# ---------------------------------------------------------------------------

def process_video(video_id: str, category: str, frame_count: int, resume: bool) -> None:
    """Run the whole per-video pipeline into processed/<category>/<video_id>/
    and record the outcome in the manifest.

    Returns None on success, on failure, and on a skip -- deliberately, since a
    ThreadPoolExecutor task that raises would only surface when the future is
    read, and nothing reads these futures. Everything from fetch_metadata()
    onward is caught, logged, written to the manifest as "failed" and fed to
    _note_result(), so one poisonous video cannot end a multi-day run. The
    manifest, not a return value, is how a caller learns what happened.

    Ordering is the resumability contract: the temp video and audio are deleted
    on BOTH paths (so a crash never leaves full videos on disk), and .done is
    written last, after those deletions. A directory without .done is therefore
    always safe to overwrite, which is what makes --resume idempotent.

    `resume` only skips already-.done work; it never repairs a partial
    directory. Skips are also silent when the auth-break breaker has tripped --
    those videos are left for a later run rather than being recorded as failures
    they did not have.
    """
    video_dir = os.path.join(OUT_DIR, category, video_id)
    if resume and is_done(video_dir):
        log(f"skip (already done): {video_id}")
        return
    if _abort_event.is_set():
        log(f"skip (aborted -- cookies session appears dead): {video_id}")
        return
    os.makedirs(video_dir, exist_ok=True)
    tmp_video_path = os.path.join(video_dir, "_tmp_video.mp4")
    tmp_audio_path = os.path.join(video_dir, "_tmp_audio.wav")
    try:
        log(f"processing {video_id} ({category})")
        fetch_metadata(video_id, video_dir)
        fetch_thumbnail(video_id, video_dir)
        srt_path = fetch_captions(video_id, video_dir)
        download_video(video_id, tmp_video_path)

        frame_paths = extract_frames(tmp_video_path, os.path.join(video_dir, "frames"), frame_count)
        thumbnail_path = next(
            (os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.startswith("thumbnail.")),
            None,
        )
        extract_visual_features(thumbnail_path, frame_paths, video_dir)

        extract_audio_track(tmp_video_path, tmp_audio_path)
        extract_audio_features(tmp_audio_path, video_dir)

        # Transcript: prefer auto-captions if they pass the quality
        # heuristic, else fall back to Whisper on the audio track we
        # already have on disk (data_retrieval.md #4.3).
        if srt_path and caption_quality_ok(srt_path):
            transcript_text = " ".join(l for l in srt_to_text(srt_path) if not _SRT_TAG_RE.match(l))
            transcript_source = "auto_captions"
        else:
            transcript_text = transcribe_with_whisper(tmp_audio_path)
            transcript_source = "whisper" if transcript_text is not None else "none"
        if transcript_text:
            with open(os.path.join(video_dir, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(transcript_text)
        with open(os.path.join(video_dir, "transcript_info.json"), "w", encoding="utf-8") as f:
            json.dump({"source": transcript_source, "had_auto_captions": srt_path is not None}, f, indent=2)

        os.remove(tmp_video_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        with open(os.path.join(video_dir, ".done"), "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        manifest_append(video_id, category, "done")
        log(f"done {video_id}")
        _note_result(success=True)
    except subprocess.CalledProcessError as e:
        for p in (tmp_video_path, tmp_audio_path):
            if os.path.exists(p):
                os.remove(p)
        err = (e.stderr or str(e))[:500]
        manifest_append(video_id, category, "failed", err)
        log(f"FAILED {video_id}: {err}")
        _note_result(success=False, err_text=err)
    except Exception as e:
        for p in (tmp_video_path, tmp_audio_path):
            if os.path.exists(p):
                os.remove(p)
        err = str(e)[:500]
        manifest_append(video_id, category, "failed", err)
        log(f"FAILED {video_id}: {e}")
        _note_result(success=False, err_text=err)


# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point: parse args, build the job list (discovery or
    --input-ids), and process it across --workers threads.

    Three orderings here are load-bearing rather than stylistic:

    - --cookies/--cookies-from-browser are appended to the module-level
      YT_DLP_EXTRA_ARGS, so they reach every yt-dlp call in the process, not
      just the download. This mutates module state; a second main() in one
      process would append them twice.
    - preload_whisper() runs before the ThreadPoolExecutor is created. See its
      docstring -- doing it lazily inside a worker raises the failure rate to
      25-35% on Windows purely from the number of live threads.
    - Submissions are staggered by --pacing/--workers instead of being queued
      back-to-back, because burst traffic is precisely what triggered YouTube's
      bot check on a school network. The sleep is in the submitting thread, so
      it paces starts, not throughput.

    Discovery asks for --target IDs per category but search.list usually returns
    fewer; the run processes what it got and does not retry to reach the target.
    If the auth-break breaker trips, submission stops early and the remainder of
    the queue is deliberately left unsubmitted for a later --resume.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=list(CATEGORIES) + ["all"])
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_PER_CATEGORY, help="videos to collect per category")
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--resume", action="store_true", help="skip videos already marked .done")
    parser.add_argument("--input-ids", help="path to a CSV (video_id,category) to skip discovery")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                         help="e.g. edge, firefox -- pass a real logged-in browser session to yt-dlp. Confirmed "
                              "2026-08-22: without this, YouTube's bot-check ('Sign in to confirm you're not a "
                              "bot') can reject every single video on a network it doesn't trust (observed on a "
                              "school network). NOTE: does NOT work with current Chrome -- its App-Bound "
                              "Encryption breaks external DB decryption (yt-dlp issue #10927); use --cookies "
                              "with an exported cookies.txt instead if your browser is Chrome.")
    parser.add_argument("--cookies", metavar="PATH",
                         help="path to a Netscape-format cookies.txt (e.g. exported via the 'Get cookies.txt "
                              "LOCALLY' Chrome extension). Use this instead of --cookies-from-browser when the "
                              "browser is Chrome -- sidesteps issue #10927 since the export goes through "
                              "Chrome's own extension API, not external DB decryption.")
    parser.add_argument("--pacing", type=float, default=VIDEO_PACING_SEC,
                         help="target seconds between video *submissions* (divided across --workers) -- firing "
                              "requests back-to-back is exactly the pattern that triggers the bot-check above")
    parser.add_argument("--workers", type=int, default=3,
                         help="videos processed concurrently. Confirmed 2026-08-22: single-threaded is far too "
                              "slow for 8,000 videos (~50s/video observed -> ~5 days). Start conservative (3) on "
                              "a normal machine and watch for renewed 429/bot-check errors before raising it; "
                              "drop to 1 if they reappear.")
    parser.add_argument("--whisper-instances", type=int, default=None,
                         help="concurrent Whisper model instances (default: same as --workers). Each instance "
                              "gets cpu_count/instances threads. Only lower this below --workers on a RAM- or "
                              "core-constrained machine -- transcription was found 2026-08-22 to be the dominant "
                              "bottleneck when serialized behind a single instance (~63s/video even at --workers "
                              "8, extrapolating to ~5-6 days for 8,000 videos), so on a high-core/high-RAM box "
                              "this should normally just match --workers.")
    args = parser.parse_args()

    if args.cookies_from_browser:
        YT_DLP_EXTRA_ARGS.extend(["--cookies-from-browser", args.cookies_from_browser])
    if args.cookies:
        YT_DLP_EXTRA_ARGS.extend(["--cookies", args.cookies])

    check_dependencies()
    os.makedirs(OUT_DIR, exist_ok=True)
    # must happen before any worker threads exist -- see preload_whisper() docstring
    preload_whisper(args.whisper_instances if args.whisper_instances is not None else args.workers)

    targets = list(CATEGORIES) if args.category == "all" else [args.category]

    if args.input_ids:
        with open(args.input_ids, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            jobs = [(row["video_id"], row["category"]) for row in reader if row["category"] in targets]
    else:
        api_key = load_api_key()
        jobs = []
        for cat in targets:
            cfg = CATEGORIES[cat]
            log(f"discovering {args.target} video IDs for category={cat}")
            ids = discover_video_ids(api_key, cfg["categoryId"], cfg["q"], args.target)
            log(f"discovered {len(ids)} IDs for {cat}")
            jobs.extend((vid, cat) for vid in ids)

    log(f"processing {len(jobs)} videos with {args.workers} worker(s)")
    stagger = args.pacing / max(args.workers, 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for video_id, category in jobs:
            if _abort_event.is_set():
                log(f"stopping submission early -- {len(jobs) - len(futures)} videos left unsubmitted "
                    f"(cookies session dead, see ABORT line above). Refresh cookies.txt and re-run with --resume.")
                break
            futures.append(executor.submit(process_video, video_id, category, args.frames, args.resume))
            time.sleep(stagger)  # stagger submissions so `workers` tasks don't all start in one burst
        concurrent.futures.wait(futures)

    log("run complete" if not _abort_event.is_set() else "run aborted (cookies session dead)")


if __name__ == "__main__":
    main()
