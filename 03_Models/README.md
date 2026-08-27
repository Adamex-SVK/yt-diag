# 03_Models — README

## Purpose
Model code, training configurations, and design rationale. Baselines and the deep multimodal model share ONE pipeline (`ytdiag/`); every variant and ablation is a feature-group configuration, never a second implementation.

## Design: one pipeline, feature groups as the unit of configuration

Two datasets feed the project and they do **not** have the same features — Adam's retrospective collection (`02_Data/processed/`: metadata, thumbnail, frames, transcript, visual + audio engineered features, v2 label) and Emmanuel's prospective tracker (`02_Data/tracking/`: metadata, publish timestamp, thumbnail history, daily view curves → fixed-horizon outcomes; no frames/transcript/audio). Rather than two pipelines, each source has an **adapter** that maps it onto one canonical table whose columns are prefixed by feature **group**:

| Group | Prefix | What | Retrospective | Prospective |
|---|---|---|---|---|
| metadata | `meta__` | duration, HD, title/description length, tag count, subscriber count, uploader captions + auto-captions flags, category, uploader-declared language (primary subtag), channel age / upload count / first-upload flag | ✅ (language after backfill) | ✅ (static fields since 2026-08-27 + `--backfill-static`) |
| schedule | `sched__` | publish hour (sin/cos), weekday, weekend — from a full UTC timestamp | after `backfill_published_at.py` | ✅ |
| visual engineered | `vis__` | thumbnail + frame aggregates (`visual_features.json`) | ✅ | ❌ |
| audio engineered | `aud__` | eGeMAPS 88 + pause stats (`audio_features.json`) | ✅ | ❌ |
| assets | `asset__` | paths/text for the deep stage: thumbnail, frames dir, transcript, title/description | ✅ | thumbnail + title + description (from `texts/`) |
| tracker-only | `track__` | thumbnail/title change counts, snapshot count | ❌ | ✅ |

Roles from `02_Data/FEATURES.md` are enforced in code: label-only columns (`view_count`, outcomes, `label`) can never be selected as inputs; `asset__` columns are never tabular inputs. `features.available_groups(df)` reports honestly what a loaded table supports.

**Label**: retrospective rows carry the v2 stratified label from `compute_labels_v2.py` (`label.json`). Prospective rows get `outcome_views` interpolated at `--horizon-days` from the bracketing snapshots (NaN until a video reaches the horizon — never extrapolated) and a within-category top-quartile label among videos that reached it (a stand-in for the v2 stratification, documented as such).

**Split**: channel-grouped, label-stratified 60/20/20 via `StratifiedGroupKFold` — no channel spans splits (asserted). Val is used for thresholds and model selection; the **test split is evaluated only with `--test`, once, at the end**.

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `ytdiag/features.py` | Feature registry: groups, roles, column selection with label-only guard | Done |
| `ytdiag/adapters.py` | `load_retrospective(processed_dir)`, `load_prospective(tracking_dir, horizon_days)` → canonical table | Done — retrospective verified on synthetic tree; prospective verified on the live tracker (2,953+ rows, thumbnails resolved) |
| `ytdiag/split.py` | Channel-grouped stratified split | Done |
| `ytdiag/baselines.py` | Dummy floor / logistic regression / XGBoost (HistGradientBoosting fallback) on any group selection; AUC-ROC, PR-AUC, F1 at val-tuned threshold, per-category AUC | Done |
| `ytdiag/synthetic.py` | Synthetic retrospective tree with planted signal (exact file layout of the collection pipeline) | Done |
| `run_baselines.py` | CLI: `--source retrospective\|prospective`, `--groups meta,sched,vis,aud`, `--synthetic N`, `--test` | Done |
| `tests/test_pipeline.py` | Registry/adapter/split/baseline checks on synthetic data + live prospective adapter smoke test | Passing |
| deep model (`ytdiag/embed.py`, `ytdiag/fusion.py`) | Frozen DINOv2 ViT-S/14 (thumbnail, frames) + ModernBERT-base (title/description/transcript) embeddings cached to disk → late-fusion MLP over selected groups → Integrated Gradients attribution | **Next** |

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-ml.txt   # once; torch uses MPS on Apple silicon
.venv/bin/python 03_Models/run_baselines.py --synthetic 1200 --groups meta            # pipeline check, no data needed
.venv/bin/python 03_Models/run_baselines.py --source retrospective --data 02_Data/processed --groups meta
.venv/bin/python 03_Models/run_baselines.py --source retrospective --data 02_Data/processed --groups meta,sched,vis,aud
.venv/bin/python 03_Models/run_baselines.py --source prospective --data 02_Data/tracking --horizon-days 7 --groups meta,sched
(cd 03_Models && ../.venv/bin/python tests/test_pipeline.py)
```
Results land in `04_Experiments/runs/<name>/results.json` (gitignored).

## Status
- **Baseline pipeline**: built and verified 2026-08-27 on synthetic data (planted signal recovered: LR/XGB AUC ≈ 0.95 vs dummy 0.50 — a plumbing check, not a performance forecast). Real retrospective run pending Adam's collected data + `compute_labels_v2.py`; prospective baselines become possible from 2026-09-02, when the first cohort videos reach the 7-day horizon.
- **Deep model**: not started — next.

## Notes
- The four planned ablations (evaluation_and_planning.md) are group selections on this pipeline: full vs `meta`-only baselines; full minus text; full minus vision; full minus attribution.
- Synthetic AUCs are meaningless as performance estimates; the generator plants a deliberately strong signal so pipeline bugs show up as AUC ≈ 0.5.
- `requirements-ml.txt` is separate from `requirements.txt` (the collection stack) on purpose — the training machine doesn't need yt-dlp/Whisper.
