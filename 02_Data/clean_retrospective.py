"""
Data cleaning for the retrospective dataset (02_Data/processed/) -- the audit
step between collection and labels/EDA/models. Never modifies Adam's collected
files (metadata.json, transcript.txt, captions.srt, images); everything it
produces is additive: a per-video quality manifest, rebuilt transcripts under a
NEW name, and merged fields in metadata_extra.json (our file).

Grounded in the 2026-08-31 six-aspect audit (see CHANGELOG):
  - 98% of auto-caption transcripts carry a ~3x rolling-window duplication
    (each phrase appears in consecutive SRT cues); word counts are inflated
    ~3x and any text feature computed on them is distorted.
  - 544 declared-non-English videos got machine-TRANSLATED English auto
    captions (the collector only ever fetched captions.en.srt); Whisper ran
    English-only (small.en), so its output for non-English audio is
    hallucinated English. The manifest classifies every transcript's kind
    rather than pretending the text modality is uniform.
  - 23 done videos lack transcript.txt (failed whisper runs); 14 of them have
    an unused captions.en.srt that a rebuild can salvage.
  - 1 video has duration=null, 1 has channel_follower_count=null with a
    public channel -- both recoverable from the API (--api-fixups).
  - 11 is_short=false verdicts look doubtful (<=60s or portrait frames);
    --reprobe-shorts re-runs the definitive /shorts/ URL test on them.

Modes (composable; default with no flags = manifest only):
    python3 02_Data/clean_retrospective.py                    # write cleaning_manifest.csv + report
    python3 02_Data/clean_retrospective.py --fix-transcripts  # rebuild transcript_clean.txt from SRTs, then manifest
    python3 02_Data/clean_retrospective.py --api-fixups       # backfill null duration/subs into metadata_extra.json (needs .env, ~2 quota units)
    python3 02_Data/clean_retrospective.py --reprobe-shorts   # re-probe doubtful is_short=false verdicts (no quota)

Outputs:
  02_Data/processed/cleaning_manifest.csv   one row per .done video: quality
      flags + transcript classification. Descriptive, not destructive: which
      rows a given model/labeling cohort excludes is decided (and documented)
      downstream, with the manifest as evidence.
  <video_dir>/transcript_clean.txt          de-duplicated transcript rebuilt
      from captions.en.srt (only where that is the better source; the raw
      transcript.txt is never edited).

Transcript kind (decreasing precedence; text = clean if present else raw):
  missing               no transcript text at all
  sparse                under 50 words after de-duplication (the collection
                        pipeline's own quality bar)
  whisper_on_non_english  declared non-English audio transcribed by the
                        English-only Whisper -- hallucinated English
  non_english_text      text is not English (low Latin-script or stopword
                        rate) -- e.g. rare SRTs that carry native language
  translated_en         declared non-English, auto captions -- YouTube's
                        machine TRANSLATION to English (usable, but
                        translationese; keep separable for ablations)
  native_en             declared English, English text
transcript_usable = kind in (native_en, translated_en) AND final dup8 < 0.5
AND bracket-tag lines < 50% AND pause_ratio < 0.8 (review 2026-08-31: at the
0.5 no-speech mark three long clean native_en transcripts were wrongly vetoed;
>=0.8 separates the truly speech-free population cleanly -- the informational
no_speech column keeps the 0.5 threshold).
Thresholds are stated here once and revisited in EDA, not scattered.

All writes are atomic (tmp + os.replace) and a metadata_extra.json that exists
but does not parse is a FATAL error, never merged over -- a crash mid-write
must stay visible and recoverable, not silently cement data loss (review
2026-08-31).
"""
from __future__ import annotations
import argparse
import csv
import datetime
import io
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterator, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yt_shorts  # noqa: E402  -- definitive Shorts verdict, shared with tracker/backfill

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
MANIFEST_NAME = "cleaning_manifest.csv"
CLEAN_NAME = "transcript_clean.txt"
EXTRA_NAME = "metadata_extra.json"

# 50 words of REAL content, measured after de-duplication and tag removal.
# Deliberately NOT the collector's bar: collect_and_extract.py required >50
# words of raw auto-caption text, which at the ~3x rolling-window inflation is
# only ~17 real words -- far too little to be a usable transformer input. The
# manifest keeps `transcript_words`, so any consumer can re-threshold.
MIN_WORDS = 50
DUP8_USABLE = 0.5       # above this, text is still repetition-dominated
TAG_TOKENS_USABLE = 0.5  # [Music]/[Applause] tokens as a fraction of all tokens
NO_SPEECH_PAUSE_RATIO = 0.5   # informational flag
NO_SPEECH_VETO = 0.8          # usability veto -- see module docstring
EN_STOPWORD_RATE = 0.15  # below this, Latin text is likely not English
LATIN_RATIO_EN = 0.8
CCT_RANGE = (1667.0, 25000.0)  # version-3 Planckian-locus support (K)
CCT_VERSION = 3
CCT_METHOD = "nearest_planckian_locus_cie1960uv_1pct_lut"

_STOPWORDS = frozenset(
    "the a an and or but of to in on for with is are was were be been it this "
    "that you i we they he she at as by from have has had not so what all".split())
_TAG_LINE = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")
_SRT_TIME = re.compile(r"\d\d:\d\d:\d\d[,.]\d\d\d\s*-->")
_SRT_MARKUP = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------- transcripts

def parse_srt_text_lines(path: str, keep_tags: bool = False) -> list[str]:
    """Cue text lines of an SRT, in order: markup and the literal WebVTT/ASS
    hard-space escape stripped, YouTube's `>>` speaker prefix removed, and
    bracket-only cues ([Music], [Applause]) dropped unless keep_tags -- the
    collector stripped those too, and counting them as words inflates every
    text statistic.

    A pure-digit line is an SRT *index* only when the next non-empty line is a
    timestamp. Dropping every digit line by shape alone silently deleted the
    content of counting/price/measurement cues (407 caption lines across 57
    videos -- review 2026-08-31); the lookahead keeps the robustness against
    malformed numbering while costing no real text."""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = [_SRT_MARKUP.sub("", l).replace("\\h", " ") for l in f]
    raw = [re.sub(r"\s+", " ", l).strip() for l in raw]
    raw = [l for l in raw if l]
    lines = []
    for i, line in enumerate(raw):
        if _SRT_TIME.search(line):
            continue
        if re.fullmatch(r"\d+", line) and _SRT_TIME.search(raw[i + 1] if i + 1 < len(raw) else ""):
            continue  # index line: a digit followed by its cue's timestamp
        line = re.sub(r"^(?:>>\s*)+", "", line).strip()
        if not line or (not keep_tags and _TAG_LINE.match(line)):
            continue
        lines.append(line)
    return lines


def dedup_rolling_lines(lines: list[str]) -> list[str]:
    """Collapse YouTube auto-caption rolling windows: consecutive cues repeat
    the previous cue's tail line before appending one new line. A line is
    dropped if it equals the previously kept line, or if the previously kept
    line ends with it / it starts with the previously kept line (partial
    progressive cues).

    Overlaps must fall on a WORD boundary: a bare substring test deleted
    genuinely distinct cues ('solution' followed by 'on', 'This' followed by
    'is' -- review 2026-08-31)."""
    kept = []
    for line in lines:
        if kept:
            prev = kept[-1]
            if line == prev:
                continue
            if prev.endswith(" " + line):
                continue
            if line.startswith(prev + " "):
                kept[-1] = line  # progressive cue grew -- keep the longer text
                continue
        kept.append(line)
    return kept


def words_of(text: str) -> list[str]:
    """Whitespace tokens -- the one word definition behind MIN_WORDS,
    transcript_words and dup8_ratio, so those three always count the same
    thing. Punctuation stays attached and case is preserved; script_profile
    normalises separately for the checks where token identity matters."""
    return text.split()


def dup8_ratio(words: list[str]) -> float:
    """Fraction of 8-grams that are repeats of an earlier 8-gram (0 for <9
    words). The audit's repetition metric: ~0.35 on raw auto captions."""
    if len(words) < 9:
        return 0.0
    grams = [tuple(words[i:i + 8]) for i in range(len(words) - 7)]
    return 1.0 - len(set(grams)) / len(grams)


def script_profile(text: str) -> tuple[float, float]:
    """(latin_ratio among letters, english_stopword_rate among tokens), both
    fractions in 0..1. Text with no letters at all scores (0.0, 0.0), i.e.
    indistinguishable from non-Latin script; that only reaches a verdict in
    classify_transcript after the MIN_WORDS gate, where it is a genuine 'not
    English' rather than an empty-input artefact."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0, 0.0
    latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
    tokens = [w.strip(".,!?\"'()[]").lower() for w in text.split()]
    tokens = [t for t in tokens if t]
    stop = sum(1 for t in tokens if t in _STOPWORDS) / len(tokens) if tokens else 0.0
    return latin / len(letters), stop


_BRACKETED = re.compile(r"[\[(][^\])]*[\])]")


def tag_token_ratio(text: str) -> float:
    """Fraction of tokens sitting inside [Music]/(laughs)-style brackets.

    Counted over TOKENS, not lines: SRT-derived text is one line per cue while
    whisper output is a single paragraph, so a per-line ratio measured two
    incommensurable things and never fired (review 2026-08-31)."""
    if not text:
        return 0.0
    total = len(text.split())
    if not total:
        return 0.0
    tagged = sum(len(m.split()) for m in _BRACKETED.findall(text))
    return min(1.0, tagged / total)


def rebuild_transcript(video_dir: str) -> Optional[str]:
    """transcript_clean.txt from captions.en.srt where the SRT is the better
    source: the transcript came from auto captions (rolling-window artifact),
    or the whisper output is missing/under the word bar. Returns the action
    taken ('rebuilt', 'salvaged', None). Idempotent: rewrites are harmless."""
    srt = os.path.join(video_dir, "captions.en.srt")
    if not os.path.exists(srt):
        return None
    try:
        tinfo = _read_json(os.path.join(video_dir, "transcript_info.json")) or {}
    except CorruptJSON:
        tinfo = {}  # unknown source: the salvage branch below is the safe default
    tpath = os.path.join(video_dir, "transcript.txt")
    source = tinfo.get("source")
    if source == "auto_captions":
        action = "rebuilt"
    else:  # whisper (or unknown): only salvage if whisper output is absent/sparse
        if os.path.exists(tpath):
            with open(tpath, encoding="utf-8", errors="replace") as f:
                if len(words_of(f.read())) >= MIN_WORDS:
                    return None
        action = "salvaged"
    lines = dedup_rolling_lines(parse_srt_text_lines(srt))
    _atomic_write_text(os.path.join(video_dir, CLEAN_NAME),
                       "\n".join(lines) + ("\n" if lines else ""))
    return action


# ------------------------------------------------------------------- helpers

class CorruptJSON(Exception):
    """The file exists but does not parse -- never the same as 'absent'."""


def _read_json(path):
    """None means ABSENT. A file that exists but does not parse raises, so a
    corrupted file can never masquerade as a missing one and get silently
    replaced by an empty dict (review 2026-08-31). I/O errors propagate."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptJSON(f"{path}: {e}") from e


def _read_text(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def done_videos(data_dir: str) -> Iterator[tuple[str, str, str]]:
    """Yield (category, video_id, video_dir) for every COMPLETED video, sorted
    by category then id so runs are reproducible.

    The `.done` sentinel is the collector's last write for a video; a directory
    without it is a half-downloaded video whose missing files would otherwise
    show up in the manifest as data-quality defects rather than as an
    unfinished download."""
    for category in sorted(os.listdir(data_dir)):
        cat_dir = os.path.join(data_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for vid in sorted(os.listdir(cat_dir)):
            d = os.path.join(cat_dir, vid)
            if os.path.exists(os.path.join(d, ".done")):
                yield category, vid, d


def _atomic_write_text(path, text):
    """Crash-safe write: a kill mid-write leaves the old file intact, never a
    truncated fragment (review 2026-08-31 -- a fragment would silently shadow
    healthy data downstream)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _merge_extra(video_dir, fields):
    path = os.path.join(video_dir, EXTRA_NAME)
    try:
        merged = _read_json(path) or {}
    except CorruptJSON as e:  # NEVER merge over it -- the backfill fields
        sys.exit(f"FATAL: {e}\nrefusing to overwrite an unparseable "  # inside are expensive/irrecoverable
                 f"{EXTRA_NAME}; restore it (git checkout) before re-running")
    merged.update(fields)
    _atomic_write_text(path, json.dumps(merged, indent=2))


def _language(extra):
    for v in (extra.get("default_audio_language"), extra.get("default_language")):
        if isinstance(v, str) and v.strip():
            base = v.strip().split("-")[0].lower()
            if base not in ("zxx", "und"):
                return base
    return ""


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


# ---------------------------------------------------------------- API fixups

def load_api_key() -> str:
    """YOUTUBE_API_KEY from the repo-root .env (never committed). Exits instead
    of returning empty: --api-fixups has nothing to do without a key, and a
    silent no-key run would look like 'nothing to fix'."""
    env_path = os.path.join(ROOT, ".env")
    if not os.path.exists(env_path):
        sys.exit(f"No {env_path} -- create it with YOUTUBE_API_KEY=<your key>")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("YOUTUBE_API_KEY="):
                return line.strip().split("=", 1)[1]
    sys.exit(f"YOUTUBE_API_KEY not found in {env_path}")


def _api_get(endpoint, params):
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def _iso8601_duration_sec(s):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m or not any(m.groups()):
        return None
    d, h, mi, sec = (int(g) if g else 0 for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + sec


def api_fixups(data_dir: str) -> None:
    """Recover metadata.json nulls that the labeler/features need (duration,
    channel_follower_count) from the API into metadata_extra.json -- Adam's
    metadata.json is never edited. Generic: finds every affected done row."""
    need_dur, need_subs = [], []
    for category, vid, d in done_videos(data_dir):
        # a corrupt file here must stop the run, not be quietly written over
        meta = _read_json(os.path.join(d, "metadata.json")) or {}
        extra = _read_json(os.path.join(d, EXTRA_NAME)) or {}
        if extra.get("status") != "ok":
            continue  # gone from the API -- nothing to recover
        if meta.get("duration") is None and "duration_sec_api" not in extra \
                and not extra.get("duration_api_unrecoverable"):
            need_dur.append((vid, d))
        if not meta.get("channel_follower_count") and "channel_follower_count_api" not in extra \
                and not extra.get("subs_api_unrecoverable"):
            need_subs.append((vid, d, meta.get("channel_id")))
    if not need_dur and not need_subs:
        print("api-fixups: nothing to fix")
        return
    key = load_api_key()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if need_dur:
        ids = ",".join(v for v, _ in need_dur)
        items = {i["id"]: i for i in _api_get("videos", {"part": "contentDetails", "id": ids, "key": key}).get("items", [])}
        for vid, d in need_dur:
            item = items.get(vid)
            dur = _iso8601_duration_sec(((item or {}).get("contentDetails") or {}).get("duration"))
            if dur:  # 0 = live/premiere content, not a real duration -- leave it missing
                _merge_extra(d, {"duration_sec_api": dur, "fixup_at_utc": stamp})
                print(f"api-fixups: {vid} duration_sec_api={dur}")
            elif item is not None:  # definitive non-answer: memoize so re-runs stop asking
                _merge_extra(d, {"duration_api_unrecoverable": True, "fixup_at_utc": stamp})
                print(f"api-fixups: {vid} API duration is {dur!r} (live content?) -- recorded as unrecoverable")
            else:
                print(f"api-fixups: {vid} not returned by the API -- left pending (transient?)")
    if need_subs:
        ids = ",".join(c for _, _, c in need_subs if c)
        items = {i["id"]: i for i in _api_get("channels", {"part": "statistics", "id": ids, "key": key}).get("items", [])}
        for vid, d, ch in need_subs:
            item = items.get(ch)
            stats = (item or {}).get("statistics") or {}
            subs = stats.get("subscriberCount")
            if subs is not None and not stats.get("hiddenSubscriberCount"):
                _merge_extra(d, {"channel_follower_count_api": int(subs), "fixup_at_utc": stamp})
                print(f"api-fixups: {vid} channel_follower_count_api={subs}")
            elif item is not None:  # channel exists but hides the count: memoize
                _merge_extra(d, {"subs_api_unrecoverable": True, "fixup_at_utc": stamp})
                print(f"api-fixups: {vid} subscriber count hidden at the API -- recorded as unrecoverable")
            else:
                print(f"api-fixups: {vid} channel not returned by the API -- left pending (transient?)")


# ------------------------------------------------------------- shorts reprobe

def reprobe_shorts(data_dir: str) -> None:
    """Re-run the definitive /shorts/ URL test (no API quota) on doubtful
    is_short=false rows: <=60s duration, or portrait frames. A definitive
    verdict (either way) replaces the stored one; inconclusive probes change
    nothing (yt_shorts semantics)."""
    suspects = []
    for category, vid, d in done_videos(data_dir):
        extra = _read_json(os.path.join(d, EXTRA_NAME)) or {}
        if extra.get("is_short") != "false":
            continue
        meta = _read_json(os.path.join(d, "metadata.json")) or {}
        # the API fixup is the authority when yt-dlp had no duration
        dur = meta.get("duration")
        if dur is None:
            dur = extra.get("duration_sec_api")
        portrait = False
        size = _image_size(os.path.join(d, "frames", "frame_00.jpg"))
        if size:
            portrait = size[1] > size[0]
        if (dur is not None and dur <= 60) or portrait:
            suspects.append((vid, d))
    print(f"reprobe-shorts: {len(suspects)} doubtful is_short=false rows")
    if not suspects:
        return
    verdicts = yt_shorts.classify_many([v for v, _ in suspects])
    aborted = bool(verdicts.pop("__aborted__", None))
    if aborted:
        print("reprobe-shorts: WARNING -- probe run ABORTED after consecutive failures "
              "(network/consent trouble); unprobed rows are counted as inconclusive below")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    flipped = confirmed = inconclusive = 0
    for vid, d in suspects:
        v = verdicts.get(vid)
        if v is None or v == "":  # inconclusive or dead page: never overwrite
            inconclusive += 1
            continue
        if v == "true":
            flipped += 1
            print(f"reprobe-shorts: {vid} flipped false -> true")
            _merge_extra(d, {"is_short": v, "is_short_checked_at_utc": stamp})
        else:  # verdict unchanged: no write -- rewriting an identical verdict
            confirmed += 1  # just multiplies crash exposure on the extra file
    print(f"reprobe-shorts: {confirmed} confirmed false (not rewritten), {flipped} flipped to true, "
          f"{inconclusive} inconclusive (unchanged)")


# ------------------------------------------------------------------ manifest

def classify_transcript(
    final_text: Optional[str],
    source: Optional[str],
    lang: str,
    speechless: bool,
) -> tuple[str, bool, int, float, float, float, float]:
    """Returns (kind, usable, words, dup8, latin_ratio, en_stopword_rate,
    tag_token_ratio) -- `kind` one of the six values listed in the module
    docstring, the last four fractions in 0..1, `words` a count after tag
    removal. `final_text` None means no transcript exists at all (kind
    'missing'); `lang` "" means the video declares no audio language, which is
    treated as English because that is what the collector's English-only
    pipeline assumed.

    `speechless` is the usability veto (pause_ratio >= NO_SPEECH_VETO), not
    the informational no_speech flag -- see the module docstring.

    Word counts, script and stopword rates are all measured on the text with
    bracket tags removed: [Music]/[Applause] are not words, and counting them
    inflated transcript_words and depressed en_stopword_rate (review
    2026-08-31)."""
    tags = tag_token_ratio(final_text or "")
    speech = _BRACKETED.sub(" ", final_text or "") if final_text else ""
    words = words_of(speech)
    n = len(words)
    dup8 = dup8_ratio(words)
    latin, stop = script_profile(speech)
    english = latin >= LATIN_RATIO_EN and stop >= EN_STOPWORD_RATE
    if final_text is None:
        kind = "missing"
    elif n < MIN_WORDS:
        kind = "sparse"
    elif lang and lang != "en" and source == "whisper":
        kind = "whisper_on_non_english"
    elif not english:
        kind = "non_english_text"
    elif lang and lang != "en":
        kind = "translated_en"
    else:
        kind = "native_en"
    usable = (kind in ("native_en", "translated_en") and dup8 < DUP8_USABLE
              and tags < TAG_TOKENS_USABLE and not speechless)
    return kind, usable, n, dup8, latin, stop, tags


def build_manifest(data_dir: str) -> list[dict[str, Any]]:
    """One row per .done video, in done_videos order, ready for csv.DictWriter
    (every row carries the same keys in the same order).

    Values are CSV-shaped, not Python-shaped: an absent number is the empty
    string, not 0, so a 0 in the CSV means measured-as-0 (duration, pause_ratio
    and age_days all have legitimate near-zero values). channel_follower_count
    is the one exception -- a falsy count and an absent one both come out ""
    with flag_subs_missing=1. Duration and subscriber count prefer
    metadata.json and fall back to the --api-fixups values, with a *_source
    column recording which one was used.

    Purely descriptive: nothing here excludes a video. Which rows a labeling
    cohort drops is decided downstream, with this manifest as the evidence."""
    rows = []
    for category, vid, d in done_videos(data_dir):
        unreadable = []

        def _safe(name, default=None):
            """A corrupt file is recorded as evidence in the manifest, not
            silently indistinguishable from an absent one."""
            try:
                got = _read_json(os.path.join(d, name))
            except CorruptJSON:
                unreadable.append(name)
                return default
            return default if got is None else got

        meta = _safe("metadata.json", {})
        extra = _safe(EXTRA_NAME, {})
        vis = _safe("visual_features.json", {})
        aud = _safe("audio_features.json")
        tinfo = _safe("transcript_info.json", {})

        duration = meta.get("duration")
        if duration is None:
            duration = extra.get("duration_sec_api")
        subs = meta.get("channel_follower_count")
        if not subs:
            subs = extra.get("channel_follower_count_api")

        age_days = ""
        try:
            up = datetime.datetime.strptime(meta.get("upload_date") or "", "%Y%m%d")
            col = datetime.datetime.strptime((meta.get("collected_at") or "").split("T")[0], "%Y-%m-%d")
            age_days = (col - up).days
        except (ValueError, TypeError):  # null or malformed dates -> blank, not a crash
            pass

        thumb = next((os.path.join(d, f) for f in ("thumbnail.webp", "thumbnail.jpg", "thumbnail.png")
                      if os.path.exists(os.path.join(d, f))), None)
        tsize = _image_size(thumb) if thumb else None
        fsize = _image_size(os.path.join(d, "frames", "frame_00.jpg"))

        def _in_or_missing(v, lo, hi):
            return v is None or (isinstance(v, (int, float))
                                 and math.isfinite(v) and lo <= v <= hi)
        def _present_in(v, lo, hi):
            return (isinstance(v, (int, float)) and math.isfinite(v)
                    and lo <= v <= hi)
        thumb_cct = (vis.get("thumbnail") or {}).get("cct")
        frame_mean_cct = vis.get("frames_mean_cct")
        frame_std_cct = vis.get("frames_std_cct")
        cct_frames_valid = vis.get("cct_frames_valid")
        cct_frames_total = vis.get("cct_frames_total")
        counts_valid = (isinstance(cct_frames_valid, int)
                        and isinstance(cct_frames_total, int)
                        and 0 <= cct_frames_valid <= cct_frames_total)
        cct_policy_valid = (
            vis.get("cct_version") == CCT_VERSION
            and vis.get("cct_method") == CCT_METHOD
            and counts_valid
            and _in_or_missing(thumb_cct, *CCT_RANGE)
            and _in_or_missing(frame_mean_cct, *CCT_RANGE)
            # A spread is not itself a temperature; require only a finite,
            # nonnegative value within the full locus span.
            and _in_or_missing(frame_std_cct, 0.0, CCT_RANGE[1])
        )
        cct_thumbnail_valid = (cct_policy_valid
                               and bool(vis.get("cct_thumbnail_valid"))
                               and _present_in(thumb_cct, *CCT_RANGE))
        cct_frames_have_valid = (cct_policy_valid and cct_frames_valid > 0
                                 and _present_in(frame_mean_cct, *CCT_RANGE))
        # This is the model-ready aggregate triplet. Partial frame coverage is
        # allowed but exposed separately; a missing thumbnail or frame mean is
        # never silently called valid.
        cct_feature_valid = (cct_thumbnail_valid and cct_frames_have_valid
                             and _present_in(frame_std_cct, 0.0, CCT_RANGE[1]))
        cct_frames_complete = (cct_policy_valid and cct_frames_total > 0
                               and cct_frames_valid == cct_frames_total)
        cct_frame_coverage = (cct_frames_valid / cct_frames_total
                              if counts_valid and cct_frames_total else "")

        pause_ratio = ((aud or {}).get("pauses") or {}).get("pause_ratio")
        no_speech = isinstance(pause_ratio, (int, float)) and pause_ratio > NO_SPEECH_PAUSE_RATIO
        speechless = isinstance(pause_ratio, (int, float)) and pause_ratio >= NO_SPEECH_VETO

        raw = _read_text(os.path.join(d, "transcript.txt"))
        clean = _read_text(os.path.join(d, CLEAN_NAME))
        final_text = clean if clean is not None else raw
        final_source = ("srt_dedup" if clean is not None
                        else (tinfo.get("source") or "none") if raw is not None else "none")
        # the KIND follows the text actually used: a salvaged SRT is auto
        # captions regardless of what the collection pipeline ran last
        effective_source = "auto_captions" if clean is not None else tinfo.get("source")
        lang = _language(extra)
        kind, usable, n_words, dup8, latin, stop, tags = classify_transcript(
            final_text, effective_source, lang, speechless)
        # a clean file dramatically shorter than a healthy raw one is the
        # signature of a truncated write fragment -- flag, never silently trust
        raw_words = len(words_of(raw)) if raw is not None else 0
        shadow_suspect = (clean is not None and raw_words >= MIN_WORDS
                          and n_words < 0.15 * raw_words)

        rows.append({
            "video_id": vid, "category": category, "channel_id": meta.get("channel_id", ""),
            "api_status": extra.get("status", ""), "is_short": extra.get("is_short", ""),
            "flag_unreadable_json": ";".join(unreadable),
            "duration_sec": duration if duration is not None else "",
            # provenance: collection-time yt-dlp value vs an API fixup -- a
            # consumer must be able to tell them apart, not guess
            "duration_source": ("metadata" if meta.get("duration") is not None
                                else "api_fixup" if duration is not None else "none"),
            "age_days_at_collection": age_days,
            "view_count": meta.get("view_count", ""),
            "channel_follower_count": subs if subs else "",
            "subs_source": ("metadata" if meta.get("channel_follower_count")
                            else "api_fixup" if subs else "none"),
            "flag_duration_missing": int(duration is None),
            "flag_subs_missing": int(not subs),
            "flag_likes_missing": int(meta.get("like_count") is None),
            "flag_comments_missing": int(meta.get("comment_count") is None),
            "thumb_width": tsize[0] if tsize else "", "thumb_height": tsize[1] if tsize else "",
            "thumb_subhd": int(bool(tsize) and (tsize[0] < 1280 or tsize[1] < 720)),
            "frames_portrait": int(bool(fsize) and fsize[1] > fsize[0]),
            # `vis_cct_valid` is retained as the concise model-ready flag;
            # policy conformance and each source of missingness are separate.
            "vis_cct_valid": int(cct_feature_valid),
            "vis_cct_policy_valid": int(cct_policy_valid),
            "vis_cct_thumbnail_valid": int(cct_thumbnail_valid),
            "vis_cct_frames_valid": cct_frames_valid if counts_valid else "",
            "vis_cct_frames_total": cct_frames_total if counts_valid else "",
            "vis_cct_frame_coverage": cct_frame_coverage,
            "vis_cct_frames_complete": int(cct_frames_complete),
            "vis_cct_any_missing": int(not cct_thumbnail_valid
                                       or not cct_frames_complete),
            "audio_present": int(aud is not None),
            "pause_ratio": pause_ratio if pause_ratio is not None else "",
            "no_speech": int(no_speech),
            "had_auto_captions": int(bool(tinfo.get("had_auto_captions"))),
            "transcript_source": tinfo.get("source", ""),
            "srt_present": int(os.path.exists(os.path.join(d, "captions.en.srt"))),
            "transcript_present": int(raw is not None),
            "transcript_clean_present": int(clean is not None),
            "transcript_final_source": final_source,
            "transcript_raw_words": raw_words,
            "flag_clean_shadows_raw": int(shadow_suspect),
            "transcript_words": n_words, "transcript_dup8": f"{dup8:.4f}",
            "latin_ratio": f"{latin:.3f}", "en_stopword_rate": f"{stop:.3f}",
            "tag_token_ratio": f"{tags:.3f}",
            "lang_declared": lang,
            "transcript_kind": kind, "transcript_usable": int(usable),
        })
    return rows


def write_manifest(rows: list[dict[str, Any]], data_dir: str) -> str:
    """Write rows to <data_dir>/cleaning_manifest.csv; returns the path.

    The header comes from rows[0], so `rows` must be non-empty and homogeneous
    (build_manifest builds every row from one literal; main exits on an empty
    manifest rather than truncating the file to nothing). Rendered in memory
    first and then written atomically: a crash must leave the previous
    manifest, not a half-CSV that pandas would parse without complaint."""
    path = os.path.join(data_dir, MANIFEST_NAME)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    _atomic_write_text(path, buf.getvalue())
    return path


def report(rows: list[dict[str, Any]]) -> None:
    """Print the audit summary for a manifest to stdout; writes nothing.

    Every line is informational except the last: flag_clean_shadows_raw means a
    rebuilt transcript is a suspected truncated write fragment, which needs a
    re-run of --fix-transcripts before the manifest is trusted. Requires a
    non-empty `rows` (the usable percentage divides by len)."""
    n = len(rows)
    def count(pred):
        return sum(1 for r in rows if pred(r))
    print(f"manifest: {n} done videos")
    print(f"  api gone: {count(lambda r: r['api_status'] == 'missing_from_api')}; "
          f"is_short true/false/unknown: {count(lambda r: r['is_short'] == 'true')}/"
          f"{count(lambda r: r['is_short'] == 'false')}/{count(lambda r: r['is_short'] == '')}")
    print(f"  duration missing: {count(lambda r: r['flag_duration_missing'])}, "
          f"subs missing: {count(lambda r: r['flag_subs_missing'])}, "
          f"cct feature invalid: {count(lambda r: not r['vis_cct_valid'])}, "
          f"cct partial/missing: {count(lambda r: r['vis_cct_any_missing'])}, "
          f"no audio features: {count(lambda r: not r['audio_present'])}, "
          f"no-speech: {count(lambda r: r['no_speech'])}")
    kinds = {}
    for r in rows:
        kinds[r["transcript_kind"]] = kinds.get(r["transcript_kind"], 0) + 1
    print(f"  transcript kinds: {dict(sorted(kinds.items(), key=lambda kv: -kv[1]))}")
    print(f"  transcript usable: {count(lambda r: r['transcript_usable'])} "
          f"({count(lambda r: r['transcript_usable']) / n:.1%})")
    print(f"  rebuilt from srt: {count(lambda r: r['transcript_clean_present'])}")
    shadow = count(lambda r: r["flag_clean_shadows_raw"])
    if shadow:
        print(f"  WARNING: {shadow} rebuilt transcripts are <15% of a healthy raw transcript -- "
              f"likely truncated write fragments; re-run --fix-transcripts and re-check")


def main() -> None:
    """CLI entry point. The mode flags are composable and always run BEFORE the
    manifest is rebuilt, so a single invocation ends with a manifest that
    describes the post-fix state rather than the state the run started in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--fix-transcripts", action="store_true",
                        help="rebuild transcript_clean.txt from captions.en.srt where the SRT is the better source")
    parser.add_argument("--api-fixups", action="store_true",
                        help="recover null duration/subscriber count from the API into metadata_extra.json")
    parser.add_argument("--reprobe-shorts", action="store_true",
                        help="re-run the /shorts/ URL test on doubtful is_short=false rows (<=60s or portrait frames)")
    args = parser.parse_args()

    if args.api_fixups:
        api_fixups(args.data_dir)
    if args.reprobe_shorts:
        reprobe_shorts(args.data_dir)
    if args.fix_transcripts:
        actions = {"rebuilt": 0, "salvaged": 0}
        for category, vid, d in done_videos(args.data_dir):
            a = rebuild_transcript(d)
            if a:
                actions[a] += 1
        print(f"fix-transcripts: {actions['rebuilt']} rebuilt from srt, {actions['salvaged']} salvaged "
              f"(missing/sparse whisper output with an unused srt)")

    rows = build_manifest(args.data_dir)
    if not rows:
        sys.exit("no done videos found")
    path = write_manifest(rows, args.data_dir)
    report(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
