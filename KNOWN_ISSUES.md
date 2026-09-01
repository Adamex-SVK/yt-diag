# Known issues

Real defects found by reading, not yet fixed. Each says what breaks, when it can actually happen,
and why it was left. Deliberately separate from `CHANGELOG.md`, which records what *was* done.

Most of these surfaced during the 2026-09-01 documentation pass: writing an honest docstring for a
function forces you to state its contract, and a contract you cannot state truthfully is usually a
bug. None was fixed at the time, because a documentation pass that also changes behaviour cannot be
verified as behaviour-preserving.

---

## Live tracker (`02_Data/track_new_videos.py`)

**1. The Shorts heal can starve.** `check_shorts()` takes `todo[:limit]` — always the *first* 200
blank-`is_short` rows in `cohort.csv` order. In the heal path a **resolved** unknown (the page says
the video is gone) is deliberately not written, so that row stays blank and re-occupies a heal slot
on every tick, with rows further down the file queued behind it forever.
`latest_missing_ids()` only clears rows a snapshot pass has already marked `status=missing`, so a
video that is gone at `/shorts/` but still `ok` in the snapshots blocks a slot indefinitely.
*Impact today:* the 66 unverified rows are all deleted/private and already excluded, so the heal
reports "nothing to classify" — the starvation is latent. It would bite if a large batch of
resolved-unknown rows ever entered the front of the file.
*Fix:* record a resolved unknown explicitly (e.g. `is_short_unverifiable=true`) so it leaves the
todo set, rather than relying on the snapshot pass to clear it.

**2. A malformed `discovery_state.json` bricks every subsequent tick.** If the file exists but
lacks `last_window_end_utc`, `prev_end` is `None` and `parse_iso(None)` raises `TypeError` —
killing the tick *before* discovery **and** before snapshotting. The atomic write makes it
unlikely, but a hand-edit or an externally truncated file is exactly the failure the atomic replace
was added to prevent, and this path defeats it.
*Fix:* treat a missing key like a missing file (fall back to the default window) and log loudly.

---

## Prospective adapter (`03_Models/ytdiag/adapters.py`)

**3. `title` is missing from the schema-tolerance loop.** `load_prospective()` back-fills absent
`description_length` / `tag_count` columns on `video_snapshots.csv`, but not `title`, which was
added in the same 2026-08-27 schema change. A snapshots file written before that change (an
archived copy — the live file's header already has it) raises `AttributeError` at `first.title`
instead of degrading to NaN like its two siblings.

**4. `track__title_changes` can be −1.** `int(ok.title.dropna().nunique() - 1)` yields −1 — a
nonsense change count rather than NaN — for a video whose `ok` snapshots all have an empty title.
`track__thumbnail_changes` has the same shape and is latent only because the tracker guarantees the
first thumbnail row is `changed=true`. *Verified on the live cohort: the minimum is 0 for all three
`track__` columns today*, so nothing is negative in practice.

---

## Cleaning (`02_Data/clean_retrospective.py`)

**5. A genuine zero subscriber count is recorded as "missing".** `build_manifest` writes
`subs if subs else ""` and `flag_subs_missing = int(not subs)`, and `api_fixups` selects on
`if not meta.get("channel_follower_count")` — all truthiness tests, so a real 0-subscriber channel
is indistinguishable from an absent value.
*Impact today: zero rows.* All 1,860 videos have a numeric count (minimum 2). But the tracking
cohort *does* contain 168 channels reporting 0 subscribers with `hiddenSubscriberCount=false`, so
this becomes reachable the moment a 0-sub channel enters `processed/`. Note the labeler
(`compute_labels_v2.py`) uses the same truthiness convention, so the manifest is at least
consistent with it — fixing one means fixing both.

---

## Collection (`02_Data/collect_and_extract.py`)

**6. ~~The colour-temperature feature is not the formula it claims.~~ FIXED 2026-09-01.**
`_correlated_color_temp` set X=r, Y=g, Z=b and skipped the sRGB→XYZ matrix entirely, so its `x,y`
were normalised RGB fractions rather than CIE chromaticity; McCamy's denominator `(0.1858 − y)`
then approached zero for green-deficient images, producing values to 5.0 × 10⁹ K and negative
Kelvin. Replaced with a correct implementation (`correlated_color_temp` in `collect_and_extract.py`)
that gamma-decodes sRGB, applies the D65 matrix, averages in **linear light**, and gates on
**Duv ≤ 0.05** — because a temperature only describes a colour that is actually near the Planckian
locus. All 39,060 images were recomputed in place by `02_Data/recompute_cct.py`, and
`02_Data/tests/test_cct.py` pins the result to known references (D65 white = 6503.5 K against a
textbook 6504). This was **not** cosmetic: correcting it moved the full-feature XGBoost baseline
from 0.628 to 0.642 on seed 0.

---

## Superseded code (`02_Data/compute_labels.py`, v1)

**7. Silent zero-imputation.** `compute_rate` does `view_count = meta.get("view_count") or 0`, so a
video with a missing view count is ranked at the bottom of its category and labelled `typical`
rather than excluded — the same class of silent imputation as the `subs=1` fallback that got v1
replaced. v1 is retained only as a reference for the CHANGELOG narrative and **must not be run**;
`compute_labels_v2.py` is the current labeler.
