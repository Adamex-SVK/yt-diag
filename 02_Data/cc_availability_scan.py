"""
CC-license availability scan for YT-Diag (wayfinder ticket #2).

For each candidate category, page through YouTube Data API v3 search.list
with videoLicense=creativeCommon to get a REAL count of retrievable CC
video IDs (not just the totalResults estimate, which Google's own docs
note can be inaccurate for large result sets).

Usage: source .venv/bin/activate && python3 02_Data/cc_availability_scan.py
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional
import urllib.parse
import urllib.request

CAP = 500  # stop paging once we've confirmed this many are retrievable
PAGE_SIZE = 50

CANDIDATES = [
    {"label": "comedy_skits (categoryId=23)", "categoryId": "23"},
    {"label": "howto_style (categoryId=26)", "categoryId": "26"},
    # categoryId=22 alone + videoLicense=creativeCommon reproducibly returns 0
    # results from the API (confirmed independent of query params) — this is a
    # known videoCategoryId quirk in search.list, not a bug in this script.
    # Workaround: pair the category filter with a broad keyword query.
    {"label": "people_blogs / vlogs (categoryId=22, bare filter — known-broken)", "categoryId": "22"},
    {"label": "people_blogs / vlogs (categoryId=22 + q=vlog workaround)", "categoryId": "22", "q": "vlog"},
    {"label": "product_reviews via science_tech (categoryId=28)", "categoryId": "28"},
    # categoryId=24 alone has the same bare-filter quirk as categoryId=22.
    {"label": "product_reviews via entertainment (categoryId=24, bare filter — known-broken)", "categoryId": "24"},
    {"label": "product_reviews via entertainment (categoryId=24 + q=product review)", "categoryId": "24", "q": "product review"},
    {"label": "product_reviews via keyword search (no category)", "categoryId": None, "q": "product review"},
]


def load_api_key() -> str:
    """YOUTUBE_API_KEY from the project-root .env.

    Path is resolved relative to this file, not the cwd, so the script gives
    the same answer wherever it is invoked from. Raises RuntimeError when the
    key is absent -- there is no unauthenticated fallback for search.list.
    """
    with open(os.path.join(os.path.dirname(__file__), "..", ".env")) as f:
        for line in f:
            if line.startswith("YOUTUBE_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("YOUTUBE_API_KEY not found in .env")


def search_page(key: str, category_id: Optional[str], q: Optional[str],
                page_token: Optional[str] = None) -> dict[str, Any]:
    """One raw decoded search.list page, always with videoLicense=creativeCommon
    -- that filter is the entire point of this scan and is never optional.

    category_id and q are each omitted from the request when falsy, which is how
    CANDIDATES expresses "keyword only, no category" and "bare category filter".
    HTTP/network errors propagate rather than returning an empty page: a failed
    request counted as zero results would silently understate CC availability,
    the one number this script exists to measure.
    """
    params = {
        "part": "id",
        "type": "video",
        "videoLicense": "creativeCommon",
        "maxResults": str(PAGE_SIZE),
        "key": key,
    }
    if category_id:
        params["videoCategoryId"] = category_id
    if q:
        params["q"] = q
    if page_token:
        params["pageToken"] = page_token
    url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.load(resp)


def scan_candidate(key: str, candidate: dict[str, Optional[str]]) -> dict[str, Any]:
    """Page one CANDIDATES entry until CAP distinct IDs are seen or the API
    stops handing out pages.

    Returns {"label", "retrievable_count", "hit_cap", "totalResults_estimate",
    "api_calls_used"}. retrievable_count counts DISTINCT videoIds actually
    returned; totalResults_estimate is carried alongside it only as evidence of
    the gap, since Google documents it as an approximation and it runs orders of
    magnitude above what search.list will actually page out. hit_cap=True means
    retrievable_count is a floor (CAP), not a total -- the scan stopped early.

    api_calls_used is returned because quota, not time, is the binding
    constraint: search.list costs 100 units per call against a 10,000/day free
    budget, i.e. ~100 pages per day for the whole scan.
    """
    seen_ids = set()
    total_results_estimate = None
    page_token = None
    calls = 0
    while len(seen_ids) < CAP:
        data = search_page(key, candidate.get("categoryId"), candidate.get("q"), page_token)
        calls += 1
        if total_results_estimate is None:
            total_results_estimate = data.get("pageInfo", {}).get("totalResults")
        items = data.get("items", [])
        for item in items:
            vid = item.get("id", {}).get("videoId")
            if vid:
                seen_ids.add(vid)
        page_token = data.get("nextPageToken")
        if not page_token or not items:
            break
        time.sleep(0.2)
    return {
        "label": candidate["label"],
        "retrievable_count": len(seen_ids),
        "hit_cap": len(seen_ids) >= CAP,
        "totalResults_estimate": total_results_estimate,
        "api_calls_used": calls,
    }


def main() -> None:
    """Scan every CANDIDATES entry, print one line each, and write
    cc_availability_scan_results.json next to this file.

    Historical, not part of the live pipeline: the team dropped the CC-only
    constraint after this ran (cc_availability_scan_findings.md addendum,
    2026-08-14), so collect_and_extract.py does not filter by videoLicense.
    Kept as the record of what that decision was based on -- re-running it costs
    most of a day's search.list quota, which is why the total is printed.
    """
    key = load_api_key()
    results = []
    total_calls = 0
    for candidate in CANDIDATES:
        r = scan_candidate(key, candidate)
        total_calls += r["api_calls_used"]
        results.append(r)
        print(f"{r['label']}: retrievable={r['retrievable_count']}"
              f"{'+ (cap hit)' if r['hit_cap'] else ''}, "
              f"totalResults_estimate={r['totalResults_estimate']}, "
              f"calls={r['api_calls_used']}")

    print(f"\nTotal search.list calls used: {total_calls} / 100 daily free quota")

    out_path = os.path.join(os.path.dirname(__file__), "cc_availability_scan_results.json")
    with open(out_path, "w") as f:
        json.dump({"scan_date": time.strftime("%Y-%m-%d"), "cap_per_candidate": CAP, "results": results}, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
