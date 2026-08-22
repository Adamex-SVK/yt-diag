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
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
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


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def check_dependencies():
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


def load_api_key():
    env_path = os.path.join(ROOT, ".env")
    with open(env_path) as f:
        for line in f:
            if line.startswith("YOUTUBE_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"YOUTUBE_API_KEY not found in {env_path}")


def warn_once(name, msg):
    if name not in _optional_import_warned:
        _optional_import_warned.add(name)
        log(f"WARNING: {msg}")


# ---------------------------------------------------------------------------
# Discovery (YouTube Data API search.list -- video IDs only, no license filter)
# ---------------------------------------------------------------------------

def discover_video_ids(api_key, category_id, q, target_count):
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

def manifest_append(video_id, category, status, error=""):
    is_new = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["video_id", "category", "status", "error", "timestamp"])
        writer.writerow([video_id, category, status, error, time.strftime("%Y-%m-%d %H:%M:%S")])


def is_done(video_dir):
    return os.path.exists(os.path.join(video_dir, ".done"))


# ---------------------------------------------------------------------------
# Per-video processing -- metadata, thumbnail, captions, video/frames
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kwargs)


def fetch_metadata(video_id, video_dir):
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
    with open(os.path.join(video_dir, "metadata.json"), "w") as f:
        json.dump(keep, f, indent=2)
    return keep


def fetch_thumbnail(video_id, video_dir):
    url = f"https://www.youtube.com/watch?v={video_id}"
    run([
        "yt-dlp", *YT_DLP_EXTRA_ARGS, "--write-thumbnail", "--skip-download",
        "-o", os.path.join(video_dir, "thumbnail.%(ext)s"),
        url,
    ])


def fetch_captions(video_id, video_dir):
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


def srt_to_text(srt_path):
    """Strips SRT index/timestamp lines, returns plain caption text."""
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


def caption_quality_ok(srt_path):
    """data_retrieval.md #4.3 heuristic: word count > 50, no more than 20%
    of lines are bare [Music]/[Applause]-style tags."""
    lines = srt_to_text(srt_path)
    if not lines:
        return False
    tag_lines = sum(1 for l in lines if _SRT_TAG_RE.match(l))
    word_count = sum(len(l.split()) for l in lines)
    tag_ratio = tag_lines / len(lines)
    return word_count > 50 and tag_ratio <= 0.2


_whisper_model = None


def transcribe_with_whisper(audio_path):
    """data_retrieval.md #4.2/#4.3 fallback for missing/poor auto-captions.
    Lazy-loaded, reused across the whole run -- loading small.en per video
    would dominate runtime otherwise."""
    global _whisper_model
    try:
        import whisper
    except ImportError:
        warn_once("whisper", "openai-whisper not installed -- videos with missing/poor auto-captions "
                              "will have NO transcript. Install it (requirements.txt) to get the fallback.")
        return None
    if _whisper_model is None:
        log("loading Whisper small.en (first use this run)...")
        _whisper_model = whisper.load_model("small.en")
    result = _whisper_model.transcribe(audio_path, fp16=False)
    return result["text"].strip()


def download_video(video_id, tmp_path):
    # bestvideo+bestaudio (not bestvideo alone) -- audio track is required
    # for the eGeMAPS/prosody features below, not just the video frames.
    url = f"https://www.youtube.com/watch?v={video_id}"
    run([
        "yt-dlp", *YT_DLP_EXTRA_ARGS, "-f", f"bestvideo[height<={MAX_HEIGHT}]+bestaudio/best[height<={MAX_HEIGHT}]",
        "--merge-output-format", "mp4",
        "-o", tmp_path,
        url,
    ])


def probe_duration(video_path):
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ])
    return float(result.stdout.strip())


def compute_frame_timestamps(duration, frame_count):
    """16-24 frames, denser in the first DENSE_WINDOW_SEC seconds (locked
    in the Architecture Digest: 'engagement signal concentrates' there).
    Falls back to plain uniform sampling for videos shorter than the
    dense window, where the split is meaningless."""
    if duration <= DENSE_WINDOW_SEC:
        return [(i + 0.5) / frame_count * duration for i in range(frame_count)]

    dense_count = max(1, round(frame_count * DENSE_FRACTION))
    sparse_count = frame_count - dense_count
    dense_ts = [(i + 0.5) / dense_count * DENSE_WINDOW_SEC for i in range(dense_count)]
    remaining = duration - DENSE_WINDOW_SEC
    sparse_ts = [DENSE_WINDOW_SEC + (i + 0.5) / sparse_count * remaining for i in range(sparse_count)] \
        if sparse_count > 0 else []
    return dense_ts + sparse_ts


def extract_frames(video_path, frames_dir, frame_count):
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

def _correlated_color_temp(mean_rgb):
    """McCamy's approximation (additional_features.md 1.1). mean_rgb in
    0-255, RGB order."""
    r, g, b = (c / 255.0 for c in mean_rgb)
    total = r + g + b
    if total == 0:
        return None
    X = r
    Y = g
    Z = b
    denom = X + Y + Z
    x = X / denom
    y = Y / denom
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
    return cct


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


def _analyze_image(cv2, path, detect_faces):
    img = cv2.imread(path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_bgr = img.reshape(-1, 3).mean(axis=0)
    mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
    cct = _correlated_color_temp(mean_rgb)
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


def extract_visual_features(thumbnail_path, frame_paths, video_dir):
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
    with open(os.path.join(video_dir, "visual_features.json"), "w") as f:
        json.dump(features, f, indent=2)
    return features


# ---------------------------------------------------------------------------
# Audio / prosodic features (additional_features.md 2): eGeMAPS via
# opensmile, extracted from the audio track of the video already on disk.
# ---------------------------------------------------------------------------

def extract_audio_track(video_path, audio_path):
    run([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ])


def extract_audio_features(audio_path, video_dir):
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
    with open(os.path.join(video_dir, "audio_features.json"), "w") as f:
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

def process_video(video_id, category, frame_count, resume):
    video_dir = os.path.join(OUT_DIR, category, video_id)
    if resume and is_done(video_dir):
        log(f"skip (already done): {video_id}")
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
            with open(os.path.join(video_dir, "transcript.txt"), "w") as f:
                f.write(transcript_text)
        with open(os.path.join(video_dir, "transcript_info.json"), "w") as f:
            json.dump({"source": transcript_source, "had_auto_captions": srt_path is not None}, f, indent=2)

        os.remove(tmp_video_path)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        with open(os.path.join(video_dir, ".done"), "w") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))
        manifest_append(video_id, category, "done")
        log(f"done {video_id}")
    except subprocess.CalledProcessError as e:
        for p in (tmp_video_path, tmp_audio_path):
            if os.path.exists(p):
                os.remove(p)
        err = (e.stderr or str(e))[:500]
        manifest_append(video_id, category, "failed", err)
        log(f"FAILED {video_id}: {err}")
    except Exception as e:
        for p in (tmp_video_path, tmp_audio_path):
            if os.path.exists(p):
                os.remove(p)
        manifest_append(video_id, category, "failed", str(e)[:500])
        log(f"FAILED {video_id}: {e}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, choices=list(CATEGORIES) + ["all"])
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_PER_CATEGORY, help="videos to collect per category")
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    parser.add_argument("--resume", action="store_true", help="skip videos already marked .done")
    parser.add_argument("--input-ids", help="path to a CSV (video_id,category) to skip discovery")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                         help="e.g. chrome, edge, firefox -- pass a real logged-in browser session to yt-dlp. "
                              "Confirmed 2026-08-22: without this, YouTube's bot-check ('Sign in to confirm "
                              "you're not a bot') can reject every single video on a network it doesn't trust "
                              "(observed on a school network). Try this first if downloads are failing.")
    parser.add_argument("--pacing", type=float, default=VIDEO_PACING_SEC,
                         help="seconds to wait between videos -- firing requests back-to-back is exactly the "
                              "pattern that triggers the bot-check above")
    args = parser.parse_args()

    if args.cookies_from_browser:
        YT_DLP_EXTRA_ARGS.extend(["--cookies-from-browser", args.cookies_from_browser])

    check_dependencies()
    os.makedirs(OUT_DIR, exist_ok=True)

    targets = list(CATEGORIES) if args.category == "all" else [args.category]

    if args.input_ids:
        with open(args.input_ids) as f:
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

    log(f"processing {len(jobs)} videos")
    for video_id, category in jobs:
        process_video(video_id, category, args.frames, args.resume)
        time.sleep(args.pacing)

    log("run complete")


if __name__ == "__main__":
    main()
