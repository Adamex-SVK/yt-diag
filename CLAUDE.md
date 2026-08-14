# CLAUDE.md — DLDM Project Context

_This file gives any AI session full context on our DLDM course project without re-briefing. Read this first._

## What this project is

**YT-Diag**: a multimodal deep learning system that, given any public YouTube video, predicts its likely performance within its own content niche and produces a short, ranked, human-readable explanation of the content features most likely responsible for underperformance.

Group project for **Deep Learning and Decision Making (DLDM)** at TUM. Full proposal: `00_Project_Brief/DLDM_Final_Project_Proposal.pdf`.

## Team

- **Adam Michalik** (03821559) — go54lix@tum.de
- **Emmanuel Gyabaah** (03811884) — emmanuel.gyabaah@tum.de

## Technical approach (three modalities, fused)

1. **Text transformer** (pretrained) — title, description, transcript (YouTube auto-captions + Whisper fallback)
2. **Vision backbone** (pretrained) — thumbnail (1280×720 JPG)
3. **Temporal transformer** (lightweight) — 16–24 frames sampled uniformly across the full video, encoded with the same vision backbone, aggregated temporally (following Abu-El-Haija et al. 2016)

Fused via cross-attention or concatenation → trained end-to-end to predict **within-category viral/non-viral label**. An attention- or ablation-based attribution layer (inspired by Rajaram & Manchanda 2020) surfaces which specific features likely depressed a video's predicted performance.

## Data

- **Source**: YouTube Data API v3
- **Scope**: 4 categories (comedy/skits, tutorials & how-to, vlogs, product reviews) — chosen because virality drivers differ by category
- **Target**: ~500 videos per category, ~2,000 total
- **Label**: "viral" = top quartile of views-per-day-since-upload, normalized by channel subscriber count, within category; "typical" = similarly sized channels, same category
- **Per video**: 12 structured metadata features + thumbnail + 16–24 frames + transcript + title/description
- **Key constraint**: Full video files for frame sampling only for CC-licensed videos (YouTube TOS) — component scoped accordingly or limitation disclosed

## Baselines & validation

- Logistic regression and XGBoost on structured metadata alone (to test whether deep multimodal signal adds real predictive value beyond scheduling/format)
- Adapt (not adopt wholesale) Wu et al. (2018) codebase: https://github.com/avalanchesiqi/youtube-engagement
- Wu et al. defines the relative engagement metric and content/channel-feature baselines reused as non-deep-learning comparison point

## Key challenges (acknowledged in proposal)

1. Frame sampling sits outside YouTube TOS unless CC-licensed → scope or disclose
2. No universal definition of "virality" → defensible, stated threshold required
3. External factors (platform promotion, timing, luck) cap achievable accuracy → report honestly
4. Fusing four heterogeneous signal types into one coherently trained model within a one-month timeline

## Project phases

1. **Problem definition** — done (proposal submitted) → `00_Project_Brief/`
2. **Literature review & research** — related work, baselines, state of the art → `01_Research/`
3. **Data acquisition & preprocessing** — YouTube API, cleaning, EDA → `02_Data/`
4. **Model development** — architecture design, training, iteration → `03_Models/`
5. **Experiments & evaluation** — metrics, ablation, comparison → `04_Experiments/`
6. **Reporting** — paper, presentation, deliverables → `05_Reports/`
7. **Team coordination** — meetings, decisions, task tracking → `06_Meeting_Notes/`

## AI usage rules (mandatory for all teammates)

1. **Every AI session that changes anything in this folder MUST log it in `CHANGELOG.md`.** No exceptions. The changelog is the single source of truth for what happened.
2. Before making changes, read `AGENTS.md` for the full protocol.
3. Use `/changelog-summary` (or ask the AI directly) to get a summary of recent changes, what's new in a folder, or the current state of any part of the project.
4. Each folder has a `README.md` that describes its contents — keep these updated when adding/removing files.

## Repository

A GitHub repo is planned. Until it exists, this OneDrive folder is the canonical project root. Once the repo is created, code lives there and this folder keeps docs, briefs, and meeting notes.

## References (from proposal)

- Abu-El-Haija et al. (2016). YouTube-8M: A large-scale video classification benchmark. _arXiv:1609.08675_.
- Rajaram & Manchanda (2020). Unboxing engagement in YouTube influencer videos: An attention-based approach. _arXiv:2012.12311_.
- Wu, Rizoiu & Xie (2018). Beyond views: Measuring and predicting engagement in online videos. _ICWSM 2018_.

---

_Last updated: 2026-08-09. Update this file as the project evolves — it should stay the single source of truth alongside `AGENTS.md`._
