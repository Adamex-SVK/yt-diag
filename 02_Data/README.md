# 02_Data — README

## Purpose
Dataset documentation, YouTube API data collection, preprocessing pipelines, and exploratory data analysis (EDA).

## Data source

**YouTube Data API v3** — 4 categories, ~500 videos each, ~2,000 total:
- Comedy/skits
- Tutorials & how-to
- Vlogs
- Product reviews

## Per-video features

| Signal | Format | Encoder |
|--------|--------|---------|
| 12 metadata features | Structured (duration, publish hour/day, tag count, title/desc length, category, channel subs/age) | Raw → baseline models |
| Thumbnail | 1280×720 JPG | Pretrained vision backbone |
| 16–24 frames | Sampled uniformly across video | Same vision backbone + temporal transformer |
| Transcript | YouTube auto-captions (~70-80% coverage) + Whisper fallback | Pretrained text transformer |
| Title + description | Text | Pretrained text transformer |

**Label**: views/likes/comments used ONLY for label construction (never model input).

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `data_collection_plan.md` | API query strategy, rate limits, category selection | TBD |
| `label_definition.md` | Viral threshold, normalization, within-category ranking | TBD |
| `preprocessing.md` | Cleaning, tokenization, frame extraction pipeline | TBD |
| `eda.md` | Distributions, category stats, missing data report | TBD |

## Status
- **Data collection**: not started
- **Preprocessing pipeline**: not started
- **EDA**: not started

## Notes
- Actual data files (CSV, frames, transcripts) go in the GitHub repo, not here.
- CC-licensed filter required for frame extraction (YouTube TOS constraint).
- YouTube API quota: plan collection carefully — 2,000 videos × multiple API calls per video.
