# 03_Models — README

## Purpose
Model code, training configurations, and design rationale. Baselines and the deep multimodal model share ONE pipeline (`ytdiag/`); every variant and ablation is a feature-group configuration, never a second implementation.

## Design: one pipeline, feature groups as the unit of configuration

Two datasets feed the project and they do **not** have the same features — Adam's retrospective collection (`02_Data/processed/`: metadata, thumbnail, frames, transcript, visual + audio engineered features, v2 label) and Emmanuel's prospective tracker (`02_Data/tracking/`: metadata, publish timestamp, thumbnail history, daily view curves → fixed-horizon outcomes; no frames/transcript/audio). Rather than two pipelines, each source has an **adapter** that maps it onto one canonical table whose columns are prefixed by feature **group**:

| Group | Prefix | What | Retrospective | Prospective |
|---|---|---|---|---|
| metadata | `meta__` | duration, HD, title/description length, tag count, subscriber count, uploader captions + auto-captions flags, category, uploader-declared language (primary subtag), channel age / upload count / first-upload flag, definitive Shorts verdict (`is_short`, a filter) | ✅ (language after backfill) | ✅ (static fields since 2026-08-27 + `--backfill-static`) |
| schedule | `sched__` | publish hour (sin/cos), weekday, weekend — from a full UTC timestamp | after `backfill_published_at.py` | ✅ |
| visual engineered | `vis__` | thumbnail + frame aggregates (`visual_features.json`) | ✅ | ❌ |
| audio engineered | `aud__` | eGeMAPS 88 + pause stats (`audio_features.json`) | ✅ | ❌ |
| assets | `asset__` | paths/text for the deep stage: thumbnail, frames dir, transcript, title/description | ✅ | thumbnail + title + description (from `texts/`) |
| tracker-only | `track__` | thumbnail / title / text (title+description+tags) change counts, snapshot count | ❌ | ✅ |

Roles from `02_Data/FEATURES.md` are enforced in code: label-only columns (`view_count`, outcomes, `label`) can never be selected as inputs; `asset__` columns are never tabular inputs. `features.available_groups(df)` reports honestly what a loaded table supports.

**Label**: retrospective rows carry the v2 stratified label from `compute_labels_v2.py` (`label.json`). Prospective rows get `outcome_views` interpolated at `--horizon-days` from the bracketing snapshots (NaN until a video reaches the horizon — never extrapolated) and a within-category top-quartile label computed **only among main-arm (`date_window`) videos** that reached it, and only for categories with ≥ 8 such videos (comparison arms `short_form`/`non_english` never get labels; a stand-in for the v2 stratification, documented as such). **Upload-time policy**: title/description/tags/thumbnail features come from the FIRST observation (falling back to the first stored text version for rows snapshotted before those columns existed); edits are exposed only as `track__*` counts.

**Split**: channel-grouped, label-stratified 60/20/20 via `StratifiedGroupKFold` — no channel spans splits within a run (asserted). The original fixed baselines use validation for their displayed F1 threshold. The tuned workflow selects both hyperparameters and threshold on inner, channel-grouped training folds, leaving outer validation for evaluation. Across seeds, however, rows change roles; these are repeated development splits and not an independent test.

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `ytdiag/features.py` | Feature registry: groups, roles, column selection with label-only guard | Done |
| `ytdiag/adapters.py` | `load_retrospective(processed_dir)`, `load_prospective(tracking_dir, horizon_days)` → canonical table | Done — retrospective verified on synthetic tree; prospective verified on the live tracker (2,953+ rows, thumbnails resolved) |
| `ytdiag/split.py` | Channel-grouped stratified split | Done |
| `ytdiag/baselines.py` | Dummy floor / logistic regression / XGBoost (HistGradientBoosting fallback) on any group selection; AUC-ROC, PR-AUC, F1 at val-tuned threshold, per-category AUC | Done |
| `ytdiag/tuning.py` | Nested channel-grouped tuning for logistic regression and XGBoost; training-OOF threshold selection; outer-validation evaluation only | Done |
| `ytdiag/synthetic.py` | Synthetic retrospective tree with planted signal (exact file layout of the collection pipeline) | Done |
| `run_baselines.py` | CLI: `--source retrospective\|prospective`, `--groups meta,sched,vis,aud`, `--synthetic N`, `--test` | Done |
| `run_tuned_baselines.py` | Five-seed nested tuning CLI; compares tuned and fixed models on identical outer splits; deliberately has no test switch | Done |
| `tests/test_pipeline.py` | Registry/adapter/split/baseline checks on synthetic data + live prospective adapter smoke test | Passing |
| `tests/test_tuning.py` | Inner-fold channel isolation, coverage, reproducibility, and threshold provenance | Passing |
| `ytdiag/embed.py` | Frozen, revision-pinned DINOv2 ViT-S/14 thumbnail and ModernBERT-base title/clean-transcript/description embeddings; source-fingerprinted resumable caches and modality/quality masks | Done for Tier 1 thumbnail/text |
| `ytdiag/text_v2.py` | Field-aware, revision-pinned Nomic ModernBERT embeddings: separate title/description/transcript blocks and distributed long-transcript chunks | Done for targeted text v2 |
| `ytdiag/fusion.py` | Train-only preprocessing, linear probes, and a small block-projection/late-fusion MLP with nested selection | Done for Tier 1 and text v2 |
| `ytdiag/deep_tuning.py` | Nested frozen-head search with fold-local preprocessing and grouped monitor splits for epoch selection | Done |
| `run_deep_multimodal.py` | Multi-seed deep-model CLI; validation only, deliberately has no test-evaluation switch | Done |
| `run_text_v2.py` | Five-seed field-aware text ablations and exact paired structured-baseline comparison; no test switch | Done |
| `run_tuned_deep.py` | Five-seed tuning of expanded linear and MLP fusion heads over cached frozen embeddings; no test switch | Done |
| `ytdiag/visual_ablation.py` | Revision-pinned DINOv2-S/B, CLIP ViT-B/32 and ResNet-50 encoders for thumbnails and frames; aspect/pooling controls and mean frame aggregation | Done |
| `run_visual_ablation.py` | Five-seed controlled visual and multimodal ablation runner; validation only, no test switch | Done |
| `tests/test_deep_multimodal.py` | Offline preprocessing/cache/fusion regression tests (no model downloads) | Done |
| `tests/test_deep_tuning.py` | Frozen candidate search, nested execution, and epoch/threshold provenance | Passing |
| `tests/test_visual_ablation.py` | Preprocessing shape/content and cache-provenance regression checks | Passing |
| temporal frame model + attribution | Learned temporal aggregation and example-level explanations | Not implemented |

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt   # once; torch uses MPS on Apple silicon
.venv/bin/python 03_Models/run_baselines.py --synthetic 1200 --groups meta            # pipeline check, no data needed
.venv/bin/python 03_Models/run_baselines.py --source retrospective --data 02_Data/processed --groups meta
.venv/bin/python 03_Models/run_baselines.py --source retrospective --data 02_Data/processed --groups meta,sched,vis,aud
.venv/bin/python 03_Models/run_baselines.py --source prospective --data 02_Data/tracking --horizon-days 7 --groups meta,sched
.venv/bin/python 03_Models/run_tuned_baselines.py --feature-sets 'meta,sched;meta,sched,vis,aud' --seeds 0,1,2,3,4
.venv/bin/python 03_Models/run_deep_multimodal.py --seeds 0,1,2,3,4
.venv/bin/python 03_Models/run_text_v2.py --seeds 0,1,2,3,4
.venv/bin/python 03_Models/run_tuned_deep.py --seeds 0,1,2,3,4 --ablations thumbnail_text_fields_meta_sched
.venv/bin/python 03_Models/run_visual_ablation.py --seeds 0,1,2,3,4
(cd 03_Models && ../.venv/bin/python tests/test_pipeline.py)
(cd 03_Models && ../.venv/bin/python tests/test_tuning.py)
(cd 03_Models && ../.venv/bin/python tests/test_deep_multimodal.py)
(cd 03_Models && ../.venv/bin/python tests/test_deep_tuning.py)
(cd 03_Models && ../.venv/bin/python tests/test_visual_ablation.py)
```
Results land in `04_Experiments/runs/<name>/results.json` (gitignored).

## Status
- **Fixed baseline pipeline**: built and verified on synthetic data, then run on all 1,854 labelled retrospective videos over five channel-grouped validation seeds. Full engineered XGBoost reaches AUC 0.605 ± 0.030.
- **Tuned tabular baselines**: 12 logistic configurations and 17 XGBoost configurations (the fixed baseline plus 16 reproducibly sampled candidates) are selected by four inner channel-grouped folds for each outer seed. Metadata+schedule XGBoost is strongest at AUC 0.619 ± 0.022, up from 0.589 ± 0.028 fixed and above tuned full engineered XGBoost at 0.614 ± 0.023.
- **Deep model Tier 1**: implemented and run — frozen DINOv2 thumbnail + frozen ModernBERT title/clean-transcript/description representations, evaluated as thumbnail-only, text-only, thumbnail+text, and thumbnail+text+metadata/schedule. The cleaning manifest gates transcripts (1,404 usable; 456 withheld while title/description remain). Best full deep result is linear AUC 0.570 ± 0.051 (MLP 0.554 ± 0.022), below both structured baselines. Encoder caches live under gitignored `03_Models/cache/`; full run results live under gitignored `04_Experiments/runs/deep_multimodal/`; tracked evidence lives in `05_Reports/final_report/results/deep_multimodal.json`.
- **Targeted text v2**: separate title, description, and distributed transcript-chunk embeddings improve the best thumbnail+text+metadata/schedule MLP to AUC 0.600 ± 0.026. Against the tuned comparators it is -0.019 versus metadata+schedule XGBoost (1/5 paired wins) and -0.014 versus full engineered XGBoost (3/5 wins). Content carries signal but does not beat the strongest baseline.
- **Nested deep-head tuning**: an expanded 12-configuration linear grid remains 0.587 ± 0.034. Nine MLP heads varying widths, dropout, learning rate, weight decay, batch size and class weighting reach 0.598 ± 0.037 versus the fixed head's 0.600 ± 0.026, winning only 2/5 paired splits. The extra search does not improve the content model.
- **Controlled visual ablation**: thumbnails and all 20 frames are encoded independently with DINOv2-small/base, CLIP ViT-B/32 and ResNet-50. CLIP is the strongest visual representation: frame-only linear AUC is 0.605 ± 0.028 and CLIP thumbnail+frames reaches 0.606 ± 0.040. DINOv2-base frames reach MLP AUC 0.602 ± 0.026, while ResNet-50 remains weak at 0.548 ± 0.018. The best observed full fusion uses CLIP thumbnail/frames with field-aware text, metadata and schedule and reaches linear AUC 0.613 ± 0.044, versus 0.619 ± 0.022 for tuned metadata+schedule XGBoost.

## Notes
- The visual ablation isolates backbone, model size, crop/padding policy, token pooling and frame aggregation. Exact input-block attribution is implemented for the linear CLIP finalist; a learned temporal frame model remains untested.
- The five seeded channel-grouped partitions are repeated development splits, not a sealed test: rows change roles across seeds. Independent evaluation is reserved for the prospective cohort after fixed-horizon outcomes mature.
- Synthetic AUCs are meaningless as performance estimates; the generator plants a deliberately strong signal so pipeline bugs show up as AUC ≈ 0.5.
- `requirements-ml.txt` is separate from `requirements.txt` (the collection stack) on purpose — the training machine doesn't need yt-dlp/Whisper.
