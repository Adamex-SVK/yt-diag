"""Verify track_new_videos.py and backfill_published_at.py logic with a
stubbed API: discovery windows/caps/dedup, snapshot min-gap and age-out,
provenance fields, backfill writes and resumability."""
import argparse
import csv
import datetime
import importlib.util
import json
import os
import shutil
import sys
import tempfile

SCRATCH = tempfile.gettempdir()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 02_Data/


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- tracker ----------------
trk = load_module("trk", os.path.join(REPO, "track_new_videos.py"))
import tempfile
# unique per run: concurrent runs (e.g. reviewers running the suite while a
# dev run is in flight) must never share state
tdir = tempfile.mkdtemp(prefix="fake_tracking_", dir=SCRATCH)
trk.TRACK_DIR = tdir
trk.COHORT_PATH = os.path.join(tdir, "cohort.csv")
trk.VIDEO_SNAPSHOTS_PATH = os.path.join(tdir, "video_snapshots.csv")
trk.CHANNEL_SNAPSHOTS_PATH = os.path.join(tdir, "channel_snapshots.csv")
trk.LOG_PATH = os.path.join(tdir, "tracker_log.txt")
trk.DISCOVERY_STATE_PATH = os.path.join(tdir, "discovery_state.json")

NOW = [datetime.datetime(2026, 8, 26, 12, 0, 0, tzinfo=datetime.timezone.utc)]
trk.utcnow = lambda: NOW[0]

calls = []


def fake_api(endpoint, params):
    calls.append((endpoint, dict(params)))
    if endpoint == "search":
        cat = params["videoCategoryId"]
        assert params["videoDuration"] in ("medium", "long")  # Shorts filter always on
        if params["videoDuration"] == "long":
            return {"items": []}
        # two fresh videos per category per DAY -- overlapping windows on the
        # same day return the same ids, like the real API would. Id ..._1 is
        # a Short that slipped past the duration filter (45s).
        stamp = params["publishedBefore"][:10]
        items = []
        for i in range(6 if MANY_ADMISSIBLE[0] else 2):
            vid = f"v{cat}_{stamp}_{i}"
            items.append({"id": {"videoId": vid},
                          "snippet": {"channelId": f"ch{cat}",
                                      "publishedAt": trk.iso(NOW[0] - datetime.timedelta(hours=3))}})
        # a duplicate of the first item, to exercise dedup
        items.append(items[0])
        return {"items": items}
    if endpoint == "videos":
        ids = params["id"].split(",")
        parts = set(params["part"].split(","))
        items = []
        for v in ids:
            if v.startswith("gone"):
                continue
            it = {"id": v}
            if "statistics" in parts:
                it["statistics"] = {"viewCount": "100", "likeCount": "5", "commentCount": "1"}
            if "contentDetails" in parts:
                it["contentDetails"] = {"duration": "PT45S" if v.endswith("_1") else "PT10M30S",
                                        "definition": "hd", "caption": "true"}
            if "snippet" in parts:
                it["snippet"] = {"categoryId": "23", "title": f"Title of {v}",
                                 "description": DESCRIPTIONS.get(v, f"desc of {v}"), "tags": ["x", "y", "z"],
                                 "defaultAudioLanguage": "en",
                                 "thumbnails": {"high": {"url": f"http://cdn/{v}.jpg"}}}
            items.append(it)
        return {"items": items}
    if endpoint == "channels":
        ids = params["id"].split(",")
        parts = set(params["part"].split(","))
        items = []
        for c in ids:
            it = {"id": c}
            if "statistics" in parts:
                it["statistics"] = {"subscriberCount": "12300", "hiddenSubscriberCount": "false", "videoCount": "42"}
            if "snippet" in parts:
                it["snippet"] = {"country": "US", "publishedAt": "2020-01-01T00:00:00Z"}
            items.append(it)
        return {"items": items}
    raise AssertionError(endpoint)


trk.api_get = fake_api
SHORT_IDS = set()  # ids the fake /shorts/ URL test calls Shorts
REAL_CLASSIFY_MANY = trk.yt_shorts.classify_many  # kept for the classifier test below
CLASSIFY_CALLS = []  # ids requested from the fake /shorts/ test, per call
def fake_classify_many(ids, workers=8, on_item=None):
    ids = list(ids); CLASSIFY_CALLS.append(ids)
    return {v: ("true" if v in SHORT_IDS else "false") for v in ids}
trk.yt_shorts.classify_many = fake_classify_many
MANY_ADMISSIBLE = [False]  # when set, the fake search yields 4 admissible (10-minute) videos per category
DESCRIPTIONS = {}  # video_id -> current fake description (mutated to simulate edits)
THUMB_BYTES = {}  # url -> current fake image bytes


def fake_fetch(url):
    return THUMB_BYTES.get(url, b"IMG:" + url.encode())


trk.fetch_url = fake_fetch
trk.THUMBS_DIR = os.path.join(tdir, "thumbnails")
trk.THUMBS_CSV_PATH = os.path.join(tdir, "thumbnail_snapshots.csv")
trk.TEXTS_DIR = os.path.join(tdir, "texts")
trk.TEXTS_CSV_PATH = os.path.join(tdir, "text_snapshots.csv")
args = argparse.Namespace(max_cohort=3000, track_days=30, min_gap_hours=12,
                          discover_window_hours=24, discover_pages=5, min_duration_sec=240)

# Tick 1
cohort = trk.read_csv(trk.COHORT_PATH)
cohort = trk.discover("KEY", cohort, args)
trk.snapshot("KEY", cohort, args)
rows = trk.read_csv(trk.COHORT_PATH)
assert len(rows) == 5, len(rows)  # 1 admitted per category (incl. backup); _1 Shorts rejected, dupes deduped across query arms
assert not any(r["video_id"].endswith("_1") for r in rows)
r = rows[0]
assert r["sampling_arm"] == "date_window" and r["window_start_utc"] and r["search_rank"], r
assert r["duration_sec"] == "630", r
assert (r["definition"], r["caption_available"], r["default_audio_language"]) == ("hd", "true", "en"), r
assert all(x["is_short"] == "false" for x in rows), "definitive Shorts verdict not recorded at admission"
snaps = trk.read_csv(trk.VIDEO_SNAPSHOTS_PATH)
assert len(snaps) == 5 and all(s["status"] == "ok" for s in snaps)
assert abs(float(snaps[0]["age_hours"]) - 3.0) < 0.1, snaps[0]
chans = trk.read_csv(trk.CHANNEL_SNAPSHOTS_PATH)
assert len(chans) == 5 and chans[0]["country"] == "US" and chans[0]["subscriber_count"] == "12300"
assert chans[0]["channel_created_at"] == "2020-01-01T00:00:00Z"
assert snaps[0]["youtube_category_id"] == "23" and snaps[0]["title"].startswith("Title of")
assert snaps[0]["tag_count"] == "3" and int(snaps[0]["description_length"]) > 0
texts = trk.read_csv(trk.TEXTS_CSV_PATH)
assert len(texts) == 5 and all(t["changed"] == "true" for t in texts)
assert json.load(open(os.path.join(trk.TEXTS_DIR, texts[0]["file"])))["tags"] == ["x", "y", "z"]
thumbs = trk.read_csv(trk.THUMBS_CSV_PATH)
assert len(thumbs) == 5 and all(t["changed"] == "true" for t in thumbs)
assert os.path.exists(os.path.join(trk.THUMBS_DIR, thumbs[0]["file"]))
print("tick 1 OK: 5 admitted (Shorts rejected at admission), snapshots + category/title + thumbnails stored")

# Tick 2, one hour later: same window tail -> same ids -> nothing new; min-gap blocks snapshots
NOW[0] += datetime.timedelta(hours=1)
cohort = trk.read_csv(trk.COHORT_PATH)
cohort = trk.discover("KEY", cohort, args)
trk.snapshot("KEY", cohort, args)
assert len(trk.read_csv(trk.COHORT_PATH)) == 5
assert len(trk.read_csv(trk.VIDEO_SNAPSHOTS_PATH)) == 5  # unchanged
state = json.load(open(trk.DISCOVERY_STATE_PATH))
assert state["last_window_end_utc"] == trk.iso(NOW[0])
print("tick 2 OK: doubled run added nothing, min-gap suppressed snapshots, window state advanced")

# Tick 3, next day: new window ids discovered, everyone snapshotted again;
# plant an aged-out video and a deleted one first; swap one thumbnail
NOW[0] += datetime.timedelta(hours=23)
first_vid = trk.read_csv(trk.COHORT_PATH)[0]["video_id"]
THUMB_BYTES[f"http://cdn/{first_vid}.jpg"] = b"NEW IMAGE"
DESCRIPTIONS[first_vid] = "EDITED description"
trk.append_rows(trk.COHORT_PATH, trk.COHORT_FIELDS, [
    # 35d old, no snapshots: beyond the 30+3d grace -> never sampled again
    {"video_id": "old00000001", "category": "comedy", "channel_id": "ch23",
     "published_at_utc": trk.iso(NOW[0] - datetime.timedelta(days=35)),
     "discovered_at_utc": trk.iso(NOW[0]), "discovery_source": "manual",
     "sampling_arm": "date_window", "window_start_utc": "", "window_end_utc": "", "search_rank": "", "duration_sec": "600"},
    {"video_id": "gone0000001", "category": "comedy", "channel_id": "ch23",
     "published_at_utc": trk.iso(NOW[0] - datetime.timedelta(hours=5)),
     "discovered_at_utc": trk.iso(NOW[0]), "discovery_source": "manual",
     "sampling_arm": "date_window", "window_start_utc": "", "window_end_utc": "", "search_rank": "", "duration_sec": "600"},
    # a comparison-arm row: tracked (snapshotted) but never part of the main arm
    {"video_id": "hindi000001", "category": "comedy", "channel_id": "ch23",
     "published_at_utc": trk.iso(NOW[0] - datetime.timedelta(hours=6)),
     "discovered_at_utc": trk.iso(NOW[0]), "discovery_source": "manual",
     "sampling_arm": "non_english", "window_start_utc": "", "window_end_utc": "", "search_rank": "", "duration_sec": "600",
     "definition": "hd", "caption_available": "false", "default_language": "", "default_audio_language": "hi"},
    # 30.5d old, last snapshot at day 29: past horizon but NO at/after-horizon
    # sample yet -> due exactly once more (the terminal sample)
    {"video_id": "final000001", "category": "comedy", "channel_id": "ch23",
     "published_at_utc": trk.iso(NOW[0] - datetime.timedelta(days=30, hours=12)),
     "discovered_at_utc": trk.iso(NOW[0]), "discovery_source": "manual",
     "sampling_arm": "date_window", "window_start_utc": "", "window_end_utc": "", "search_rank": "", "duration_sec": "600"},
])
trk.append_rows(trk.VIDEO_SNAPSHOTS_PATH, trk.VIDEO_SNAP_FIELDS, [
    {"video_id": "final000001", "observed_at_utc": trk.iso(NOW[0] - datetime.timedelta(days=1, hours=12)),
     "age_hours": f"{29*24:.2f}", "view_count": "500", "like_count": "1", "comment_count": "0",
     "status": "ok", "youtube_category_id": "23", "title": "t"},
])
cohort = trk.read_csv(trk.COHORT_PATH)
cohort = trk.discover("KEY", cohort, args)
trk.snapshot("KEY", cohort, args)
snaps = trk.read_csv(trk.VIDEO_SNAPSHOTS_PATH)
snapped_ids = [s["video_id"] for s in snaps[5:]]
assert "old00000001" not in snapped_ids, "beyond-grace video was snapshotted"
term = [x for x in trk.read_csv(trk.VIDEO_SNAPSHOTS_PATH) if x["video_id"] == "final000001"]
assert len(term) == 2 and float(term[1]["age_hours"]) >= 30 * 24, term

gone = [s for s in snaps if s["video_id"] == "gone0000001"]
assert len(gone) == 1 and gone[0]["status"] == "missing" and gone[0]["view_count"] == ""
assert len(trk.read_csv(trk.COHORT_PATH)) == 14  # 5 + 4 planted + 5 new-window
assert "hindi000001" in snapped_ids, "comparison-arm row was not snapshotted"
gone_row = [s for s in snaps if s["video_id"] == "gone0000001"][0]
assert gone_row["description_length"] == "" and gone_row["tag_count"] == ""
assert not any(t["video_id"] == "gone0000001" for t in trk.read_csv(trk.TEXTS_CSV_PATH)), "missing video got a text row"
assert "hindi000001" not in {r["video_id"] for r in trk.read_csv(trk.COHORT_PATH) if r["sampling_arm"] == "date_window"}
thumbs = trk.read_csv(trk.THUMBS_CSV_PATH)
tick1_ids = {x["video_id"] for x in thumbs[:5]}
swapped = [t for t in thumbs[5:] if t["video_id"] == first_vid]
assert len(swapped) == 1 and swapped[0]["changed"] == "true" and swapped[0]["file"].endswith("_v1.jpg")
assert all(t["changed"] == "false" for t in thumbs[5:]
           if t["video_id"] in tick1_ids and t["video_id"] != first_vid)
assert open(os.path.join(trk.THUMBS_DIR, swapped[0]["file"]), "rb").read() == b"NEW IMAGE"
texts = trk.read_csv(trk.TEXTS_CSV_PATH)
tchg = [t for t in texts[5:] if t["video_id"] == first_vid]
assert len(tchg) == 1 and tchg[0]["changed"] == "true" and tchg[0]["file"].endswith("_v1.json")
assert json.load(open(os.path.join(trk.TEXTS_DIR, tchg[0]["file"])))["description"] == "EDITED description"
assert all(t["changed"] == "false" for t in texts[5:] if t["video_id"] in tick1_ids and t["video_id"] != first_vid)
print("tick 3 OK: day-2 snapshots, beyond-grace skip, TERMINAL at/after-horizon sample taken, missing video, thumbnail swap detected")

# Schema-migration check: old-schema CSV gains the new columns on next append
mig = os.path.join(tdir, "mig.csv")
with open(mig, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "b"]); w.writerow(["1", "2"])
trk.append_rows(mig, ["a", "b", "c"], [{"a": "3", "b": "4", "c": "5"}])
rows = trk.read_csv(mig)
assert rows[0] == {"a": "1", "b": "2", "c": ""} and rows[1]["c"] == "5"
print("migration OK: old rows kept with empty new column, new rows aligned")

# Cap test: fresh state, max_cohort 4 -> 1 per category
shutil.rmtree(tdir); os.makedirs(tdir)
args_small = argparse.Namespace(max_cohort=4, track_days=30, min_gap_hours=12,
                                discover_window_hours=24, discover_pages=5, min_duration_sec=240)
cohort = trk.discover("KEY", [], args_small)
per_cat = {}
for r in trk.read_csv(trk.COHORT_PATH):
    per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
assert all(v == 1 for v in per_cat.values()) and len(per_cat) == 5, per_cat
# regression (review 2026-08-29): with MORE admissible candidates than the cap, the cap must still hold
# and the /shorts/ check must not probe hundreds of needless ids
shutil.rmtree(tdir); os.makedirs(tdir)
MANY_ADMISSIBLE[0] = True; CLASSIFY_CALLS.clear()
trk.discover("KEY", [], args_small)  # cap 1/category, 5 admissible each
per_cat = {}
for r in trk.read_csv(trk.COHORT_PATH):
    per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
assert all(v == 1 for v in per_cat.values()) and len(per_cat) == 5, per_cat
assert all(len(c) <= 3 for c in CLASSIFY_CALLS), [len(c) for c in CLASSIFY_CALLS]  # remaining(1) + margin(2), not all 5
MANY_ADMISSIBLE[0] = False
print("cap test OK: cap = max-cohort/4 -> exactly 1 per category, backup uncapped by main total")

# Failure test: a failed search call must NOT advance the discovery checkpoint
import urllib.error
state_before = json.load(open(trk.DISCOVERY_STATE_PATH))
def failing_api(endpoint, params):
    if endpoint == "search":
        raise urllib.error.URLError("boom")
    return fake_api(endpoint, params)
trk.api_get = failing_api
NOW[0] += datetime.timedelta(hours=6)
trk.discover("KEY", trk.read_csv(trk.COHORT_PATH), args)  # caps not reached -> searches fire and fail
assert json.load(open(trk.DISCOVERY_STATE_PATH)) == state_before, "checkpoint advanced past a failed window"
trk.api_get = fake_api
print("failure test OK: checkpoint frozen when search calls fail")

# Lock test: second acquire fails while held; stale lock is taken over
import time as _time
trk.LOCK_PATH = os.path.join(tdir, "tick.lock")
assert trk.acquire_lock() is True
assert trk.acquire_lock() is False, "lock acquired twice concurrently"
os.utime(trk.LOCK_PATH, (_time.time() - 3 * 3600, _time.time() - 3 * 3600))  # 3h old = stale
assert trk.acquire_lock() is True, "stale lock not taken over"
trk.release_lock()
assert trk.acquire_lock() is True  # clean reacquire after release
trk.release_lock()
# Token safety: release must never remove a lock owned by another process
open(trk.LOCK_PATH, "w").write("9999:someothertoken")
trk.release_lock()
assert os.path.exists(trk.LOCK_PATH), "released a lock owned by another process"
os.remove(trk.LOCK_PATH)
print("lock test OK: exclusive while held, stale takeover, clean reacquire, token-guarded release")

# --backfill-static: rows admitted before the static columns existed get filled in place
shutil.rmtree(tdir); os.makedirs(tdir)
old_fields = [f for f in trk.COHORT_FIELDS if f not in trk.STATIC_FIELDS]
with open(trk.COHORT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=old_fields); w.writeheader()
    for i in range(3):
        w.writerow({"video_id": f"legacy{i}", "category": "comedy", "channel_id": "chL",
                    "published_at_utc": trk.iso(NOW[0]), "discovered_at_utc": trk.iso(NOW[0]),
                    "discovery_source": "manual", "sampling_arm": "date_window", "window_start_utc": "",
                    "window_end_utc": "", "search_rank": "", "duration_sec": "600"})
def missing_one_api(endpoint, params):  # legacy2 is deleted/private: not returned by the API
    out = fake_api(endpoint, params)
    if endpoint == "videos":
        out["items"] = [it for it in out["items"] if it["id"] != "legacy2"]
    return out
trk.api_get = missing_one_api
trk.backfill_static("KEY")
rows = trk.read_csv(trk.COHORT_PATH)
assert len(rows) == 3
assert all(r["definition"] == "hd" and r["caption_available"] == "true" and r["default_audio_language"] == "en"
           for r in rows if r["video_id"] != "legacy2"), rows
assert rows[2]["definition"] == "" and rows[2]["sampling_arm"] == "date_window", rows[2]  # gone video: empty, arm untouched
trk.backfill_static("KEY")  # idempotent: only legacy2 retried, nothing else changes
assert trk.read_csv(trk.COHORT_PATH) == rows
# schema migration never drops columns an older code version doesn't know about
with open(trk.COHORT_PATH, newline="") as f:
    hdr = next(csv.reader(f))
with open(trk.COHORT_PATH, newline="") as f:
    old = list(csv.DictReader(f))
with open(trk.COHORT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=hdr + ["future_col"]); w.writeheader()
    for r in old:
        r["future_col"] = "keep-me"; w.writerow(r)
trk.append_rows(trk.COHORT_PATH, trk.COHORT_FIELDS, [{k: "" for k in trk.COHORT_FIELDS} | {"video_id": "new1"}])
after = trk.read_csv(trk.COHORT_PATH)
assert after[0]["future_col"] == "keep-me" and after[-1]["video_id"] == "new1" and after[-1]["future_col"] == "", after[-1]
trk.api_get = fake_api
print("backfill-static OK: deleted video stays empty with arm untouched, idempotent re-run, unknown columns preserved")

# Language gate: candidates whose declared language is not en* are rejected at admission
shutil.rmtree(tdir); os.makedirs(tdir)
def hindi_api(endpoint, params):
    out = fake_api(endpoint, params)
    if endpoint == "videos":
        for it in out["items"]:
            it["snippet"]["defaultAudioLanguage"] = "hi"
    return out
trk.api_get = hindi_api
cohort = trk.discover("KEY", [], args)
assert cohort == [] and not os.path.exists(trk.COHORT_PATH), "non-English candidates were admitted"
trk.api_get = fake_api
assert trk.is_english({"default_audio_language": "en-GB"}) and trk.is_english({"default_language": "en"})
assert not trk.is_english({"default_audio_language": ""}) and not trk.is_english({"default_audio_language": "id"})
# non-language codes fall through to the metadata language, mirroring the adapter's _language()
assert trk.is_english({"default_audio_language": "zxx", "default_language": "en-US"})
assert not trk.is_english({"default_audio_language": "und", "default_language": ""})
assert trk.declared_language({"default_audio_language": "zxx", "default_language": "hi"}) == "hi"
print("language-gate test OK: hi/id/undeclared rejected, en/en-GB admitted")

# Definitive Shorts verdict gate: duration passes but the /shorts/ URL test says Short -> rejected
shutil.rmtree(tdir); os.makedirs(tdir)
SHORT_IDS.update(f"v{c}_{trk.iso(NOW[0])[:10]}_0" for c in ("23", "26", "22", "24", "28"))  # ids are stamped with NOW's date
cohort = trk.discover("KEY", [], args)
assert cohort == [], "a video the URL test calls a Short was admitted"
SHORT_IDS.clear()
# --check-shorts fills is_short for rows lacking it
with open(trk.COHORT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[x for x in trk.COHORT_FIELDS if x != "is_short"]); w.writeheader()
    w.writerow({"video_id": "legacyS", "category": "comedy", "channel_id": "chL", "published_at_utc": trk.iso(NOW[0]),
                "discovered_at_utc": trk.iso(NOW[0]), "discovery_source": "manual", "sampling_arm": "short_form",
                "window_start_utc": "", "window_end_utc": "", "search_rank": "", "duration_sec": "30",
                "definition": "hd", "caption_available": "false", "default_language": "", "default_audio_language": "en"})
SHORT_IDS.add("legacyS")
trk.check_shorts()
assert trk.read_csv(trk.COHORT_PATH)[0]["is_short"] == "true"
SHORT_IDS.clear()
print("shorts-verdict test OK: URL-test Shorts rejected at admission; --check-shorts fills legacy rows")

# yt_shorts.classify branches, with the network stubbed at fetch_status
ys = trk.yt_shorts
responses = {}
ys.fetch_status = lambda url: responses[url.rsplit("/", 1)[1]]
responses["S1"] = (200, "", '{"playabilityStatus":{"status":"OK"}}')            # a real Short
responses["S2"] = (200, "", '<title>Cool short - YouTube</title>{"playabilityStatus":{"status":"LOGIN_REQUIRED","reason":"Sign in to confirm you\u2019re not a bot"}}')  # real Short behind a bot-check
responses["P1"] = (200, "", '<title> - YouTube</title>{"playabilityStatus":{"status":"LOGIN_REQUIRED","reason":"Private video"}}')  # private
responses["P2"] = (200, "", '<title> - YouTube</title>{"playabilityStatus":{"status":"LOGIN_REQUIRED"}}')  # removed, no reason
responses["E1"] = (200, "", '<title> - YouTube</title>{"playabilityStatus":{"status":"OK"}}')  # empty title, no video rendered
responses["R1"] = (303, "https://www.youtube.com/watch?v=R1", "")               # regular video
responses["D1"] = (200, "", '{"playabilityStatus":{"status":"ERROR","reason":"Video unavailable"}}')  # deleted
responses["C1"] = (302, "https://consent.youtube.com/m?continue=x", "")          # consent page, no cookie effect
assert ys.classify("S1", retries=0) == "true" and ys.classify("S2", retries=0) == "true"
assert ys.classify("R1", retries=0) == "false"
assert ys.classify("D1", retries=0) == "" and ys.classify("C1", retries=0) == ""
assert ys.classify("P1", retries=0) == "" and ys.classify("P2", retries=0) == "" and ys.classify("E1", retries=0) == ""  # private/removed pages
assert REAL_CLASSIFY_MANY(["S1", "R1", "D1"], workers=2) == {"S1": "true", "R1": "false", "D1": ""}
# circuit breaker: a burst of INCONCLUSIVE probes pauses once and RESUMES; a persistent outage aborts;
# a run of deleted/private pages is a definitive answer and never trips it
ys.GRACE_PAUSE_SEC = 0.2; ys.PACING_SEC = 0.0
assert ys.classify_ex("D1", retries=0) == ("", True) and ys.classify_ex("C1", retries=0) == ("", False)  # dead page resolved; consent inconclusive
flaky_calls = {"n": 0}
def flaky(url):  # first 30 fetches are network failures (burst), then everything is a regular video
    flaky_calls["n"] += 1
    return (None, "", "") if flaky_calls["n"] <= 30 else (303, "https://www.youtube.com/watch?v=x", "")
ys.fetch_status = flaky
out = REAL_CLASSIFY_MANY([f"b{i:02d}" for i in range(60)], workers=2)
assert "__aborted__" not in out and sum(1 for v in out.values() if v == "false") >= 25, "breaker did not resume after the grace pause"
ys.fetch_status = lambda url: (None, "", "")  # persistent outage
out = REAL_CLASSIFY_MANY([f"c{i:02d}" for i in range(80)], workers=2)
assert out.pop("__aborted__") == "true" and all(v is None for v in out.values()), "persistent outage must abort; skipped ids are inconclusive (None)"
ys.fetch_status = lambda url: (200, "", '{"playabilityStatus":{"status":"ERROR"}}')  # 80 deleted videos in a row
out = REAL_CLASSIFY_MANY([f"d{i:02d}" for i in range(80)], workers=2)
assert "__aborted__" not in out and all(v == "" for v in out.values()), "dead pages must not trip the breaker"
ys.fetch_status = lambda url: (200, "", "<html><title>Before you continue</title></html>")  # 200 without a player payload
assert ys.classify_ex("X1", retries=0) == ("", False), "a 200 without ytInitialPlayerResponse is inconclusive, not a Short"
ys.fetch_status = lambda url: responses[url.rsplit("/", 1)[1]]
# --recheck-shorts withdraws a stale "true" that is now unavailable, keeps a confirmed one
shutil.rmtree(tdir); os.makedirs(tdir)
with open(trk.COHORT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=trk.COHORT_FIELDS); w.writeheader()
    for vid, verdict in (("S1", "true"), ("D1", "true"), ("R1", "false")):
        w.writerow({**{k: "" for k in trk.COHORT_FIELDS}, "video_id": vid, "category": "comedy", "sampling_arm": "short_form", "is_short": verdict})
trk.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {v: ys.classify(v, retries=0) for v in ids}
trk.check_shorts(recheck_true=True)
got = {r["video_id"]: r["is_short"] for r in trk.read_csv(trk.COHORT_PATH)}
assert got == {"S1": "true", "D1": "", "R1": "false"}, got
# an ABORTED recheck (network down, consent bypass broken) must withdraw nothing
trk.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {**{v: "" for v in ids}, "__aborted__": "true"}
with open(trk.COHORT_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=trk.COHORT_FIELDS); w.writeheader()
    w.writerow({**{k: "" for k in trk.COHORT_FIELDS}, "video_id": "S9", "category": "comedy", "sampling_arm": "short_form", "is_short": "true"})
trk.check_shorts(recheck_true=True)
assert trk.read_csv(trk.COHORT_PATH)[0]["is_short"] == "true", "aborted recheck withdrew a verdict"
trk.yt_shorts.classify_many = fake_classify_many
# an INCONCLUSIVE probe (None) on a HEALTHY recheck must not withdraw anything either
before = {r["video_id"]: r["is_short"] for r in trk.read_csv(trk.COHORT_PATH)}
trk.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {v: None for v in ids}
trk.check_shorts(recheck_true=True)
assert {r["video_id"]: r["is_short"] for r in trk.read_csv(trk.COHORT_PATH)} == before, "inconclusive probes must never withdraw verdicts"
trk.yt_shorts.classify_many = fake_classify_many
print("classifier test OK: OK/LOGIN_REQUIRED -> Short, /watch redirect -> regular, ERROR page or consent -> unknown; recheck withdraws stale verdicts, never on an aborted run")

# Unknown verdict at admission: admitted with blank is_short, then healed by the per-tick check_shorts(limit)
shutil.rmtree(tdir); os.makedirs(tdir)
trk.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {v: "" for v in ids}
cohort = trk.discover("KEY", [], args)
assert len(cohort) == 5 and all(r["is_short"] == "" for r in cohort), "unknown verdicts must still admit (duration already >= 4min)"
trk.yt_shorts.classify_many = fake_classify_many
assert trk.check_shorts(limit=3) == 3
assert sum(1 for r in trk.read_csv(trk.COHORT_PATH) if r["is_short"] == "false") == 3  # bounded healing
# a row appended DURING a long check must survive the rewrite (fresh re-read before os.replace)
def appending_classify(ids, workers=8, on_item=None):
    trk.append_rows(trk.COHORT_PATH, trk.COHORT_FIELDS, [{**{k: "" for k in trk.COHORT_FIELDS}, "video_id": "late1", "category": "comedy", "sampling_arm": "date_window"}])
    return {v: "false" for v in ids}
trk.yt_shorts.classify_many = appending_classify
trk.check_shorts()
assert any(r["video_id"] == "late1" for r in trk.read_csv(trk.COHORT_PATH)), "row appended during check_shorts was clobbered"
trk.yt_shorts.classify_many = fake_classify_many
# the heal (scheduled tick / --check-shorts) skips rows whose LATEST snapshot is missing (unverifiable); --recheck-shorts still probes them
blank = lambda vid: {**{k: "" for k in trk.COHORT_FIELDS}, "video_id": vid, "category": "comedy", "sampling_arm": "date_window"}
trk.append_rows(trk.COHORT_PATH, trk.COHORT_FIELDS, [blank("dead1"), blank("alive1")])
snap = lambda vid, st, t: {**{k: "" for k in trk.VIDEO_SNAP_FIELDS}, "video_id": vid, "observed_at_utc": t, "status": st}
if not os.path.exists(trk.VIDEO_SNAPSHOTS_PATH):
    with open(trk.VIDEO_SNAPSHOTS_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=trk.VIDEO_SNAP_FIELDS).writeheader()
trk.append_rows(trk.VIDEO_SNAPSHOTS_PATH, trk.VIDEO_SNAP_FIELDS,
                [snap("dead1", "ok", "2026-08-28T00:00:00Z"), snap("dead1", "missing", "2026-08-29T00:00:00Z"),
                 snap("alive1", "missing", "2026-08-28T00:00:00Z"), snap("alive1", "ok", "2026-08-29T00:00:00Z")])
CLASSIFY_CALLS.clear(); trk.check_shorts()
# late1 (appended mid-check, never snapshotted) is unverified-but-alive, so it is probed too
assert len(CLASSIFY_CALLS) == 1 and set(CLASSIFY_CALLS[0]) == {"late1", "alive1"}, f"heal must probe only live rows, got {CLASSIFY_CALLS}"
assert {r["video_id"]: r["is_short"] for r in trk.read_csv(trk.COHORT_PATH)}["dead1"] == "", "dead row must stay unknown"
CLASSIFY_CALLS.clear(); trk.check_shorts(recheck_true=True)
assert CLASSIFY_CALLS == [["dead1"]], f"recheck must still probe dead rows, got {CLASSIFY_CALLS}"
print("unknown-verdict test OK: blank admitted, healed per tick with a limit, concurrent appends preserved, dead rows skipped by the heal only")

# Per-category page budgets: with an endless result stream, search calls per
# category must equal queries x 2 duration filters x that category's pages
shutil.rmtree(tdir); os.makedirs(tdir)
seq = [0]
def endless_api(endpoint, params):
    if endpoint == "search":
        calls.append((endpoint, dict(params)))
        items = []
        for _ in range(50):
            seq[0] += 1
            items.append({"id": {"videoId": f"e{seq[0]:09d}"},
                          "snippet": {"channelId": "chE", "publishedAt": trk.iso(NOW[0] - datetime.timedelta(hours=3))}})
        return {"items": items, "nextPageToken": "more"}
    if endpoint == "videos":  # duration check: everything long-form
        return {"items": [{"id": v, "contentDetails": {"duration": "PT10M"}} for v in params["id"].split(",")]}
    return fake_api(endpoint, params)
calls.clear()
trk.api_get = endless_api
args_budget = argparse.Namespace(max_cohort=1_000_000, track_days=30, min_gap_hours=8,
                                 discover_window_hours=24, discover_pages=None, min_duration_sec=240)
trk.discover("KEY", [], args_budget)
per_cat = {}
for _, prm in calls:
    per_cat[prm["videoCategoryId"]] = per_cat.get(prm["videoCategoryId"], 0) + 1
expected = {spec["categoryId"]: 2 * sum(spec["queries"].values()) for spec in trk.CATEGORIES.values()}
assert per_cat == expected, (per_cat, expected)
assert sum(per_cat.values()) == 80, sum(per_cat.values())
trk.api_get = fake_api
print(f"page-budget test OK: search calls per category {per_cat} == 2 x sum(per-arm pages); worst case {sum(per_cat.values())}")

# Bounded-FIFO thumbnail streaming: order preserved under out-of-order
# completion, and never more than 2 x THUMB_WORKERS results alive at once
import random, threading
shutil.rmtree(tdir); os.makedirs(tdir)
trk.THUMBS_DIR = os.path.join(tdir, "thumbnails")
trk.THUMBS_CSV_PATH = os.path.join(tdir, "thumbnail_snapshots.csv")
alive = [0]; peak = [0]; guard = threading.Lock()
random.seed(7)
def slow_fetch(url):
    with guard:
        alive[0] += 1; peak[0] = max(peak[0], alive[0])
    _time.sleep(random.uniform(0.0, 0.02))  # scramble completion order
    with guard:
        alive[0] -= 1
    return b"IMG:" + url.encode()
trk.fetch_url = slow_fetch
jobs = [(f"ord{i:03d}", {"high": {"url": f"http://cdn/ord{i:03d}.jpg"}}, trk.iso(NOW[0])) for i in range(60)]
trk.snapshot_thumbnails(jobs)
order = [r["video_id"] for r in trk.read_csv(trk.THUMBS_CSV_PATH)]
assert order == [j[0] for j in jobs], "thumbnail rows out of job order"
assert peak[0] <= trk.THUMB_WORKERS, f"more concurrent downloads than workers: {peak[0]}"
trk.fetch_url = fake_fetch
print(f"streaming test OK: 60 jobs, CSV order == job order, peak concurrent downloads {peak[0]} <= {trk.THUMB_WORKERS}")

# Texts are captured BEFORE thumbnails: a thumbnail-stage failure must not lose the near-publish text
shutil.rmtree(tdir); os.makedirs(tdir)
trk.THUMBS_DIR = os.path.join(tdir, "thumbnails"); trk.THUMBS_CSV_PATH = os.path.join(tdir, "thumbnail_snapshots.csv")
trk.TEXTS_DIR = os.path.join(tdir, "texts"); trk.TEXTS_CSV_PATH = os.path.join(tdir, "text_snapshots.csv")
def boom(url): raise RuntimeError("cdn down")
trk.fetch_url = boom
cohort = trk.discover("KEY", [], args)
try:
    trk.snapshot("KEY", cohort, args)
except RuntimeError:
    pass
assert len(trk.read_csv(trk.TEXTS_CSV_PATH)) == 5, "texts not stored before the thumbnail stage failed"
trk.fetch_url = fake_fetch
assert trk.arm_counts(cohort + [{"sampling_arm": "non_english"}, {"sampling_arm": "short_form"}, {"sampling_arm": "non_english"}]) == {"non_english": 2, "short_form": 1}
print("ordering test OK: texts captured despite thumbnail failure; arm_counts groups non-main arms")

# ---------------- backfill ----------------
bf = load_module("bf", os.path.join(REPO, "backfill_published_at.py"))
pdir = tempfile.mkdtemp(prefix="fake_processed_bf_", dir=SCRATCH)
shutil.rmtree(pdir)  # generator below recreates it
for vid, ch in [("aaa", "chA"), ("bbb", "chA"), ("gone1", "chB")]:
    d = os.path.join(pdir, "comedy", vid)
    os.makedirs(d)
    json.dump({"channel_id": ch}, open(os.path.join(d, "metadata.json"), "w"))
    open(os.path.join(d, ".done"), "w").close()


def fake_bf_api(endpoint, params):
    if endpoint == "videos":
        ids = [v for v in params["id"].split(",") if not v.startswith("gone")]
        return {"items": [{"id": v,
                           "snippet": {"publishedAt": "2026-07-01T18:30:00Z", "categoryId": "23",
                                       "defaultAudioLanguage": "en-GB"},
                           "contentDetails": {"caption": "true" if v == "aaa" else "false"}}
                          for v in ids]}
    if endpoint == "channels":
        return {"items": [{"id": c, "snippet": {"country": "DE", "publishedAt": "2019-05-05T00:00:00Z"},
                           "statistics": {"hiddenSubscriberCount": "false", "videoCount": "77"}}
                          for c in params["id"].split(",")]}
    raise AssertionError(endpoint)


bf.api_get = fake_bf_api
bf.load_api_key = lambda: "KEY"
bf.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {v: ("true" if v == "bbb" else "false") for v in ids}
# --shorts-only first: verdicts only, no API key needed, and the full run must still fill API fields afterwards
sys.argv = ["backfill_published_at.py", "--data-dir", pdir, "--shorts-only"]
bf.main()
assert json.load(open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json")))["is_short"] == "true"
assert len(bf.pending_videos(pdir, None)) == 3, "shorts-only files must not count as backfilled"
# --recheck withdraws a "true" that the classifier can no longer confirm
bf.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {v: "" for v in ids}
sys.argv = ["backfill_published_at.py", "--data-dir", pdir, "--shorts-only", "--recheck"]
bf.main()
assert json.load(open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json")))["is_short"] == ""
# an aborted --recheck must not withdraw the stored verdict either
bf.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {**{v: "" for v in ids}, "__aborted__": "true"}
json.dump({"is_short": "true"}, open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json"), "w"))
sys.argv = ["backfill_published_at.py", "--data-dir", pdir, "--shorts-only", "--recheck"]
bf.main()
assert json.load(open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json")))["is_short"] == "true", "aborted recheck withdrew a verdict"
json.dump({"is_short": ""}, open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json"), "w"))
bf.yt_shorts.classify_many = lambda ids, workers=8, on_item=None: {v: ("true" if v == "bbb" else "false") for v in ids}
sys.argv = ["backfill_published_at.py", "--data-dir", pdir, "--shorts-only"]
bf.main()  # unknowns are re-examined by a normal run
assert json.load(open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json")))["is_short"] == "true"
# the full API run must neither re-probe bbb nor let an unknown overwrite its stored verdict
BF_PROBED = []
def bf_unknown_classify(ids, workers=8, on_item=None):
    BF_PROBED.extend(ids); return {v: "" for v in ids}
bf.yt_shorts.classify_many = bf_unknown_classify
# drop aaa's stored verdict so the full run has exactly one video to probe (and gets an unknown for it)
_p = os.path.join(pdir, "comedy", "aaa", "metadata_extra.json"); _e = json.load(open(_p)); _e.pop("is_short", None); json.dump(_e, open(_p, "w"))
sys.argv = ["backfill_published_at.py", "--data-dir", pdir, "--pacing", "0"]
bf.main()
aaa = json.load(open(os.path.join(pdir, "comedy", "aaa", "metadata_extra.json")))
assert aaa["published_at_utc"] == "2026-07-01T18:30:00Z" and aaa["caption_available"] is True
assert aaa["channel_country"] == "DE" and aaa["status"] == "ok"
assert aaa["default_audio_language"] == "en-GB" and aaa["default_language"] == ""
assert aaa["channel_created_at"] == "2019-05-05T00:00:00Z" and aaa["channel_video_count"] == "77"
assert not aaa.get("is_short"), aaa.get("is_short")  # probed, came back unknown -> no verdict written
bbb = json.load(open(os.path.join(pdir, "comedy", "bbb", "metadata_extra.json")))
assert bbb["caption_available"] is False
assert bbb["is_short"] == "true" and "is_short_checked_at_utc" in bbb  # shorts-only verdict merged, not overwritten by the full run
assert BF_PROBED == ["aaa"], BF_PROBED  # only the video without a stored verdict is probed; bbb never re-probed
gone = json.load(open(os.path.join(pdir, "comedy", "gone1", "metadata_extra.json")))
assert gone["status"] == "missing_from_api" and gone["published_at_utc"] is None
assert bf.pending_videos(pdir, None) == []  # resumable: nothing left
print("backfill OK: timestamps/caption/country written, missing video marked, second run finds 0 pending")

print("\nALL TRACKER + BACKFILL CHECKS PASSED")
