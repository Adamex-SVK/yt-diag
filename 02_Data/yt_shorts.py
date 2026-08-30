"""Definitive YouTube Shorts check, shared by track_new_videos.py and
backfill_published_at.py (added 2026-08-29).

The Data API exposes no Shorts flag; duration is only a proxy (a Short can be
up to 180s since Oct 2024, and a 2-minute horizontal video is not a Short).
The definitive test is the URL: https://www.youtube.com/shorts/<id> answers
200 for a Short and 303 -> /watch?v=<id> for a regular video. One plain HTTP
request per video, no API quota.

Two pitfalls verified 2026-08-29 from Germany:
  1. Without a consent cookie EVERY request 302s to consent.youtube.com,
     which a naive redirect check would read as "not a Short". A CONSENT/SOCS
     cookie bypasses that; a consent redirect is still classified as unknown
     ("") and retried once, never as a verdict.
  2. A DELETED/private video also answers 200 at /shorts/<id> -- there is no
     /watch page to redirect to -- so status alone calls every unavailable
     video a Short (65 long videos answered 200 in the first cohort pass;
     all but one -- a genuine 250-second Short -- were deleted or private
     pages). The page body's ytInitialPlayerResponse.playabilityStatus
     separates them: ERROR ("Video unavailable"), or LOGIN_REQUIRED without
     the bot-check reason / with an empty "<title> - YouTube" (private video)
     => resolved unknown; OK, or the "confirm you're not a bot" interstitial
     (which still carries the video) => Short; a 200 with no player payload at
     all is not a watch page => inconclusive (see page_has_video/classify_ex).
Be polite: this is a page fetch, not an API call; YouTube starts bot-checking
after a few thousand fast requests (harmless to the verdict, but rude).
"""
import re
import concurrent.futures
import time
import urllib.error
import urllib.request

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CONSENT_COOKIE = "CONSENT=YES+cb.20240101-01-p0.en+FX+000; SOCS=CAI"
WORKERS = 4
PACING_SEC = 0.3  # per request, per worker -- plain page fetches, be polite
BODY_BYTES = 2_000_000  # the player response sits in the first ~1.3MB of the page
_PLAYABILITY = re.compile(r'"playabilityStatus":\{"status":"([A-Z_]+)"(?:,"reason":"([^"]{0,120})")?')
_TITLE = re.compile(r"<title>([^<]*)</title>")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def fetch_status(url):
    """(status, location, body) without following redirects; the body is
    read only for 200 responses. Isolated for tests."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                                               "Cookie": CONSENT_COOKIE})
    try:
        with _opener.open(req, timeout=30) as resp:
            return resp.status, "", resp.read(BODY_BYTES).decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location") or "", ""


def playability(body):
    """(status, reason) from the embedded player response -- e.g. ('OK', ''),
    ('ERROR', 'Video unavailable'), ('LOGIN_REQUIRED', 'Private video'),
    ('LOGIN_REQUIRED', "Sign in to confirm you're not a bot") -- or
    (None, '') if the page carries none."""
    m = _PLAYABILITY.search(body or "")
    return (m.group(1), m.group(2) or "") if m else (None, "")


def page_has_video(body):
    """A 200 at /shorts/<id> only means 'no /watch page to redirect to'. The
    page proves a video exists unless the player response says the video is
    unavailable (ERROR), or private/removed (LOGIN_REQUIRED without the
    bot-check reason -- 'Private video' or no reason at all, with an empty
    '<title> - YouTube'). Found 2026-08-29: 6 private videos of 4-40 minutes
    had passed as Shorts. YouTube's bot-check (LOGIN_REQUIRED, 'Sign in to
    confirm you're not a bot') is NOT unavailability -- the redirect signal
    already decided, the player payload is just gated."""
    status, reason = playability(body)
    if status == "ERROR":
        return False
    if status == "LOGIN_REQUIRED" and "bot" not in reason.lower():
        return False
    t = _TITLE.search(body or "")
    if t and t.group(1).strip() in ("- YouTube", "YouTube") and status != "LOGIN_REQUIRED":
        return False  # empty title: no video rendered
    return True


RATE_LIMIT_STATUSES = (429, 403, 503)
BACKOFF_SEC = (5.0, 30.0)  # per retry attempt after a rate-limit response
ABORT_AFTER_CONSECUTIVE_UNKNOWN = 25  # consecutive INCONCLUSIVE probes: a blocked run must stop hammering ...
GRACE_PAUSE_SEC = 60.0  # ... but a transient burst (rate-limit, brief outage) gets ONE pause-and-resume first


def classify_ex(video_id, retries=1):
    """(verdict, resolved). verdict: 'true' (Short), 'false' (regular video),
    '' (unknown). resolved=True when YouTube answered definitively -- including
    "this video is deleted/private/unavailable" (no verdict possible, but the
    probe itself worked); False when the probe was inconclusive (consent page,
    network error, rate limit, unexpected status) and a retry may succeed."""
    for attempt in range(retries + 1):
        try:
            status, location, body = fetch_status(f"https://www.youtube.com/shorts/{video_id}")
        except Exception:
            status, location, body = None, "", ""
        if status == 200:
            if playability(body)[0] is None:
                pass  # no player payload: not a watch page (interstitial, empty body) -> inconclusive, retry
            elif not page_has_video(body):
                return "", True  # deleted / private / unavailable: definitive non-answer
            else:
                return "true", True
        if status in (301, 302, 303, 307, 308) and "/watch" in location:
            return "false", True
        if attempt < retries:  # inconclusive (no payload, consent redirect, error status, exception): retry
            # rate-limited: back off hard before the retry; otherwise a short pause
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC) - 1)] if status in RATE_LIMIT_STATUSES else 1.0)
    return "", False


def classify(video_id, retries=1):
    """'true' (Short), 'false' (regular video), '' (unknown: consent page,
    network error, deleted/private/unavailable page, unexpected status)."""
    return classify_ex(video_id, retries)[0]


def classify_many(video_ids, workers=WORKERS, on_item=None):
    """{video_id: verdict} for many ids, fetched concurrently, order-free.
    verdict: 'true' / 'false' (definitive), '' (resolved unknown: the page says
    the video is deleted/private/unavailable -- no verdict is possible), or
    None (INCONCLUSIVE probe: consent page, network error, rate limit, no
    player payload -- retry later; callers must never write or withdraw
    anything on None). on_item (optional) is called after every fetch -- the
    tracker passes its lock heartbeat so a long check is never mistaken for a
    dead tick. Breaker: after ABORT_AFTER_CONSECUTIVE_UNKNOWN consecutive
    inconclusive probes every worker pauses once for GRACE_PAUSE_SEC; if the
    run of failures recurs after the pause the remaining ids are returned as
    None without fetching and the result carries an '__aborted__' marker, so
    a blocked run stops hammering and the caller can log it."""
    import threading
    ids = list(dict.fromkeys(video_ids))
    if not ids:
        return {}
    state = {"consecutive_unknown": 0, "abort": False, "paused_once": False, "pause_until": 0.0, "epoch": 0}
    guard = threading.Lock()

    def one(vid):
        while True:
            with guard:
                if state["abort"]:
                    return vid, None  # skipped after the abort: inconclusive, not a verdict
                wait = state["pause_until"] - time.time()
                epoch = state["epoch"]
            if wait <= 0:
                break
            time.sleep(min(wait, 5.0))  # every worker holds during the grace pause
        v, resolved = classify_ex(vid)
        with guard:
            # only INCONCLUSIVE probes count towards the breaker: a run of deleted/private
            # videos is a definitive answer, not a broken connection (found 2026-08-30:
            # the morning heal aborted on 60 dead videos and wasted 11 minutes). Probes
            # started before a pause was triggered belong to the old epoch and must not
            # eat into the fresh post-pause budget.
            if resolved:
                state["consecutive_unknown"] = 0
            elif epoch == state["epoch"]:
                state["consecutive_unknown"] += 1
            if state["consecutive_unknown"] >= ABORT_AFTER_CONSECUTIVE_UNKNOWN:
                if state["paused_once"]:
                    state["abort"] = True  # trouble persisted through the pause: stop hammering
                else:  # transient burst: pause every worker once, then resume
                    state["paused_once"] = True
                    state["consecutive_unknown"] = 0
                    state["epoch"] += 1
                    state["pause_until"] = time.time() + GRACE_PAUSE_SEC
        if on_item:
            on_item()
        time.sleep(PACING_SEC)
        return vid, (v if resolved else None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        out = dict(pool.map(one, ids))
    if state["abort"]:
        out["__aborted__"] = "true"  # marker for callers; not a video id
    return out
