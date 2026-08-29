"""Definitive YouTube Shorts check, shared by track_new_videos.py and
backfill_published_at.py (added 2026-08-29).

The Data API exposes no Shorts flag; duration is only a proxy (a Short can be
up to 180s since Oct 2024, and a 2-minute horizontal video is not a Short).
The definitive test is the URL: https://www.youtube.com/shorts/<id> answers
200 for a Short and 303 -> /watch?v=<id> for a regular video. One plain HTTP
request per video, no API quota.

Pitfall verified 2026-08-29 from Germany: without a consent cookie EVERY
request 302s to consent.youtube.com, which a naive redirect check would read
as "not a Short". A CONSENT/SOCS cookie bypasses that; a consent redirect is
still classified as unknown ("") and retried once, never as a verdict.
"""
import concurrent.futures
import time
import urllib.error
import urllib.request

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
CONSENT_COOKIE = "CONSENT=YES+cb.20240101-01-p0.en+FX+000; SOCS=CAI"
WORKERS = 8
PACING_SEC = 0.05  # per request, per worker -- plain page fetches, be polite


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def fetch_status(url):
    """(status, location) without following redirects. Isolated for tests."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                                               "Cookie": CONSENT_COOKIE})
    try:
        with _opener.open(req, timeout=20) as resp:
            return resp.status, ""
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Location") or ""


def classify(video_id, retries=1):
    """'true' (Short), 'false' (regular video), '' (unknown: consent page,
    network error, deleted/private, unexpected status)."""
    for attempt in range(retries + 1):
        try:
            status, location = fetch_status(f"https://www.youtube.com/shorts/{video_id}")
        except Exception:
            status, location = None, ""
        if status == 200:
            return "true"
        if status in (301, 302, 303, 307, 308) and "/watch" in location:
            return "false"
        if attempt < retries:
            time.sleep(1.0)  # consent page / transient error: retry once
    return ""


def classify_many(video_ids, workers=WORKERS):
    """{video_id: verdict} for many ids, fetched concurrently, order-free."""
    ids = list(dict.fromkeys(video_ids))
    if not ids:
        return {}

    def one(vid):
        v = classify(vid)
        time.sleep(PACING_SEC)
        return vid, v

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, ids))
