# CC-License Availability Scan — Findings

_Run 2026-08-14. Resolves the empirical half of wayfinder ticket #2
("Stand up the collection environment and run the CC-license availability
scan"), feeding tickets #3 (dataset size) and #4 (product-reviews category
mapping). Script: `cc_availability_scan.py`, raw output:
`cc_availability_scan_results.json`._

## Method

For each candidate, paginate `search.list` with `videoLicense=creativeCommon`
via the real YouTube Data API v3 (not yt-dlp) up to 500 unique video IDs or
until results run dry, whichever comes first. This counts *retrievable*
videos, not the `totalResults` field — that estimate turned out to be wildly
unreliable (see below), so it's recorded for comparison only, never trusted
as the answer.

## Results

| Candidate | Retrievable (real) | `totalResults` estimate | Read |
|---|---|---|---|
| Comedy/skits (categoryId=23) | **102** | 3,509 | Estimate off by ~34x |
| Howto & Style (categoryId=26) | **95** | 5,605 | Estimate off by ~59x |
| People & Blogs / vlogs, category filter alone (categoryId=22) | **0** | 0 | API quirk, not scarcity — see below |
| People & Blogs / vlogs + `q=vlog` workaround | **523+ (hit the pagination cap)** | 982,771 | Workaround unlocks volume; estimate meaningless |
| Product reviews via Science & Tech (categoryId=28) | **107** | 4,184 | Estimate off by ~39x |
| Product reviews via Entertainment, category filter alone (categoryId=24) | **0** | 0 | Same API quirk as categoryId=22 |
| Product reviews via Entertainment + `q=product review` | **410** | 607,258 | Workaround unlocks volume |
| Product reviews via keyword only, no category (`q=product review`) | **210** | 1,000,000 | API caps `totalResults` display at 1M for broad queries — meaningless |

## Two findings, not one

**1. `totalResults` cannot be trusted at all.** It overstates real availability
by 30–1,000,000x across every candidate tested. `data_retrieval.md`'s ~49M
CC-videos-on-YouTube figure (State of the Commons 2017) was already flagged
as an unreliable upper bound; this confirms the API's own per-query estimate
is equally unreliable. **Any dataset-size decision must use the retrievable
column, never `totalResults`.**

**2. `videoCategoryId` + `videoLicense=creativeCommon` alone reproducibly
returns zero results for categoryId=22 (People & Blogs) and categoryId=24
(Entertainment)** — confirmed independent of `order`, page size, or any other
param (see debug trace in ticket #2). This is a known category-filter quirk
in `search.list`, not a real absence of CC-licensed content in those
categories: pairing the same category filter with *any* text query (`q=vlog`,
`q=product review`) immediately returns hundreds of results. **Category-only
filtering is not usable for vlogs or Entertainment; a keyword-assisted query
is required as a workaround for those two categories specifically** —
comedy (23), howto (26), and Science & Tech (28) all work fine with the bare
category filter.

## What this means for the dataset-size and category-mapping decisions

- **Pure category+license filtering is genuinely scarce**: ~95–110 real CC
  videos for comedy, howto, and Science & Tech. That is well short of even
  the original 500/category target, let alone the 5,000–8,000 scaling
  ambition in `compute_scaling.md`.
- **Keyword-assisted queries recover much more volume** (400–500+, capped by
  this scan's own pagination limit rather than confirmed exhaustion) — but
  change what's being sampled: results are now ranked by search relevance to
  the keyword, not purely "belongs to category X," which risks topic drift
  and duplicate/near-duplicate content the category filter alone wouldn't
  surface.
- **This scan did not exhaustively test the ceiling.** `search.list`
  pagination itself appears to cap out well under 1,000 real results per
  single query regardless of `totalResults`. Reaching a genuinely larger
  pool (toward 5,000–8,000) would require slicing each category across
  multiple keyword variants and/or `publishedBefore`/`publishedAfter` date
  windows and deduplicating — not attempted here, since this scan's job was
  feasibility, not the actual collection pipeline.
- **Quota used:** 38 of 100 daily free `search.list` calls. Re-running with
  more candidate keyword variants (to test the 5,000–8,000 ceiling properly)
  is affordable within the same daily budget if ticket #3 wants a second,
  deeper pass before locking a number.

## Addendum, 2026-08-14: team decision to drop the CC-only constraint

Adam made an explicit, informed decision (his call, on the record) to **not
restrict frame sampling and audio-derived features to CC-licensed videos**,
accepting that this falls outside YouTube's TOS for the video/audio-download
step specifically (metadata, thumbnails, and captions were never
TOS-restricted regardless of license — only downloading the actual media
file is). This was surfaced and pushed back on before being accepted, per
`AGENTS.md`'s instruction to be explicit about uncertainty and log the
reasoning behind decisions, not just the outcome.

**What this changes:** the CC-scarcity numbers above (95–110 per category via
bare category filter) are no longer the binding constraint on dataset size —
supply of non-CC public video is effectively unlimited relative to any
target this project needs. The binding constraint shifts entirely to
**collection/frame-extraction/audio-processing throughput within the
remaining calendar time** (`compute_scaling.md` §1's point that "more compute
doesn't buy more calendar time" still applies).

**What does NOT change:** the `videoCategoryId=22` (vlogs) and `=24`
(Entertainment) bare-filter-returns-zero quirk is independent of the
`videoLicense` parameter — confirmed in the original debug trace
(`categoryId=22, no q, no license: totalResults=0`). The keyword-assisted
workaround (`q=vlog`, etc.) is still required for those two categories
regardless of the CC decision. This does not resolve ticket #4.
