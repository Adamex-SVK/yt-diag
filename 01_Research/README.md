# 01_Research — README

## Purpose
Literature review, related work, and notes on academic papers relevant to YT-Diag.

## Contents
_One `.md` per paper. Each file: key idea, method, results, relevance to our project._

## Core papers (from proposal)

| Paper | File | Status |
|-------|------|--------|
| Wu, Rizoiu & Xie (2018) — "Beyond Views" | `wu2018_beyond_views.md` | TBD |
| Rajaram & Manchanda (2020) — "Unboxing Engagement" | `rajaram2020_unboxing.md` | TBD |
| Abu-El-Haija et al. (2016) — "YouTube-8M" | `abu2016_youtube8m.md` | TBD |

## Additional topics to research

- Video virality prediction literature
- Multimodal fusion architectures (text + vision + temporal)
- Attention-based attribution / explainability methods
- YouTube Data API v3 best practices and rate limits
- Whisper transcription for fallback captions
- Relative engagement metrics beyond view count

## Completed Research

| Topic | File | Date |
|-------|------|------|
| Data Retrieval Strategy | `2026-08-09_initial_research/data_retrieval.md` | 2026-08-09 |
| Evaluation, Baselines & Planning | `2026-08-09_initial_research/evaluation_and_planning.md` | 2026-08-10 |
| **Model Architectures** | `2026-08-09_initial_research/model_architectures.md` | 2026-08-10 |
| **Milestones & To-Do List** | `2026-08-09_initial_research/MILESTONES.md` | 2026-08-09 |
| **Executive Summary** | `2026-08-09_initial_research/SUMMARY.md` | 2026-08-09 |
| Compute Scaling (A40 + 96 cores upgrade path) | `2026-08-14_scaling_and_features/compute_scaling.md` | 2026-08-14 |
| Additional Features (color temp, face area, voice prosody) | `2026-08-14_scaling_and_features/additional_features.md` | 2026-08-14 |

## Status
- **Structure**: done
- **Research**: initial round complete — 3 parallel agents covered data retrieval, model architectures, and evaluation & planning
- **Deliverables**: `MILESTONES.md` (master task list) and `SUMMARY.md` (executive overview) synthesized from all three reports
- **Remaining**: core paper summaries (Wu et al., Rajaram & Manchanda, Abu-El-Haija), additional architecture exploration if needed

## Notes
- The Wu et al. (2018) codebase is at https://github.com/avalanchesiqi/youtube-engagement — we adapt from this.
- Tag each note with relevance to our specific architecture decisions (not just general interest).
- Pay special attention to how existing work handles the "cold-start" prediction problem.
