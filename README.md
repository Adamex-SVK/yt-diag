# YT-Diag

Given a public YouTube video, predict how it is likely to perform **within its own niche**, and
produce a short, ranked, human-readable explanation of the content features most likely holding it
back.

Group project for **Deep Learning and Decision Making (DLDM)** at TUM.
Adam Michalik (03821559) · Emmanuel Gyabaah (03811884)

---

## Start here

| If you want to… | Read |
|---|---|
| Understand the problem and the modelling approach | [`CLAUDE.md`](CLAUDE.md) — project identity, three modalities, references |
| Know what every feature means and whether it may be a model input | [`02_Data/FEATURES.md`](02_Data/FEATURES.md) — **the feature dictionary, single source of truth** |
| Know what the data actually looks like, and why the label is defined as it is | [`02_Data/eda.md`](02_Data/eda.md) |
| Know what changed and when | [`CHANGELOG.md`](CHANGELOG.md) — every change is logged, no exceptions |
| Know what is currently broken | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) — real defects found but not yet fixed |
| Find where a file lives | [`CLAUDE-MAP.md`](CLAUDE-MAP.md) |

## The two datasets

This is the thing to understand first, because almost every design decision follows from it. There
are **two** datasets and they do not have the same features.

**1. The retrospective archive** — `02_Data/processed/`
1,860 already-published videos across four categories, collected once (22–25 Aug 2026). Each video
is a folder holding metadata, a thumbnail, 20 sampled frames, a transcript, and two bundles of
engineered features. This is the only source with frames, audio and transcripts, so it is what the
deep multimodal model trains on. Outcome = the view count observed at collection.

**2. The prospective panel** — `02_Data/tracking/`
A live, fixed-cap cohort of newly published videos, discovered daily and snapshotted **twice a day for 30
days** by two launchd agents (09:05 and 21:05). On 2026-09-02 it held 12,738 videos (10,443 in the
date-window arm, plus `short_form` and `non_english` comparison arms); the frozen date-window ceiling is
20,000 videos (4,000 for each of four principal categories plus 4,000 for the backup category). No frames or audio — but it has something the
archive never can: a *growth curve* per video, and the first-observed near-publish thumbnail and
title, before creators edit them.

They meet in `03_Models/ytdiag/adapters.py`, which maps both onto one canonical table.

## Running the pipeline

Two environments on purpose: the collection stack (yt-dlp, ffmpeg, Whisper, openSMILE) is heavy and
only needed on a collection machine; the ML stack is what you need to model.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-ml.txt      # modelling + EDA
.venv/bin/pip install -r requirements.txt         # ONLY on a collection machine
git lfs pull                                      # the 39,060 images (~2.8 GB)
```

Order matters. Later steps assume earlier ones ran.

```bash
# 1. COLLECT (already done -- 1,860 videos; needs yt-dlp, ffmpeg, deno, a cookies.txt)
.venv/bin/python 02_Data/collect_and_extract.py --category comedy --target 500

# 2. BACKFILL the fields yt-dlp never captured, from the official API   (done 2026-08-30)
.venv/bin/python 02_Data/backfill_published_at.py

# 3. CLEAN -- rebuild transcripts, write the per-video quality manifest  (done 2026-08-31)
.venv/bin/python 02_Data/clean_retrospective.py --fix-transcripts

# 4. EDA -- stats + figures behind 02_Data/eda.md                        (done 2026-09-01)
.venv/bin/python 02_Data/eda_retrospective.py

# 5. LABEL -- materialise the frozen retrospective target                (done 2026-09-01)
.venv/bin/python 02_Data/compute_labels_v2.py --category all

# 6. BASELINES -- metadata-only models, the bar the deep model must beat
.venv/bin/python 03_Models/run_baselines.py --source retrospective --groups meta
.venv/bin/python 03_Models/run_baselines.py --source retrospective --groups meta,sched,vis,aud

# 7. FROZEN CONTENT -- cached Tier-1 and field-aware text-v2 experiments
.venv/bin/python 03_Models/run_deep_multimodal.py --seeds 0,1,2,3,4
.venv/bin/python 03_Models/run_text_v2.py --seeds 0,1,2,3,4
```

The prospective tracker runs itself on a schedule; a manual tick must pass `--no-discover` so it
cannot spend the day's search quota:

```bash
.venv/bin/python 02_Data/track_new_videos.py --no-discover
```

### Report

```bash
# once: a real pdflatex in $HOME, no sudo (AAAI requires PDFLaTeX)
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
~/Library/TinyTeX/bin/universal-darwin/tlmgr install psnfss booktabs xcolor \
    graphics graphics-def epstopdf-pkg times helvetic courier amsmath
brew install poppler          # optional: enables the AAAI font checks in build.sh

.venv/bin/python 05_Reports/final_report/make_tables.py --seeds 5   # numbers first
.venv/bin/python 05_Reports/final_report/make_deep_table.py
.venv/bin/python 05_Reports/final_report/make_text_v2_table.py
./05_Reports/final_report/build.sh                                  # then compile
```

`build.sh` runs the full pdflatex → bibtex → pdflatex ×2 cycle and then *checks* AAAI
compliance rather than assuming it: no Type 3 fonts, everything embedded, US letter.
`05_Reports/final_report/claim_evidence.md` maps every number in the report to the
command that reproduces it.

### Notebook walkthrough

```bash
.venv/bin/python -m ipykernel install --user --name ytdiag --display-name "YT-Diag (.venv)"
.venv/bin/python -m jupyter lab notebooks/00_project_walkthrough.ipynb
```

The notebook's first cell **bootstraps any Python environment**: it locates the repo by
walking up from wherever it is opened, installs whatever is missing into that kernel
(not a different interpreter), and says plainly if the Git-LFS images are still pointer
files. Verified from an empty venv on a different Python version — so a teammate or
grader needs only the clone.

`notebooks/00_project_walkthrough.ipynb` runs the whole pipeline on real data and lets you
**check the claims** in `02_Data/eda.md` and `02_Data/eda_features.md` yourself: the
transcript repetition, frame sampling, colour-temperature fix, label-blind category ×
format profiles and pivot tables for visual/audio/speech features, correlation and
modality-coverage heatmaps, training-only shortcut/outcome plots, feature include/exclude decisions, the
Shorts confound, channel-split leak, baselines, and both tracked five-seed frozen deep-model
results. It is a thin *interface* — every cell
imports from `02_Data/` and `03_Models/` and calls the real functions, so it can never
drift from the code that produces the results. Edit `notebooks/build_walkthrough.py` and
regenerate rather than editing cells.

### Tests

```bash
.venv/bin/python tests/test_docs_and_compat.py                          # repo-wide guards
(cd 02_Data   && ../.venv/bin/python tests/test_clean_retrospective.py)
(cd 02_Data   && ../.venv/bin/python tests/test_tracker_backfill.py)
.venv/bin/python 02_Data/tests/test_eda_features.py
(cd 03_Models && ../.venv/bin/python tests/test_pipeline.py)
(cd 03_Models && ../.venv/bin/python tests/test_deep_multimodal.py)
```

The suites use stubbed APIs, synthetic embeddings and temp directories — no network, no key,
no risk to real data.
`tests/test_docs_and_compat.py` is the guard that keeps documentation from rotting: it fails if a
public function loses its docstring or type annotations, and — more importantly — if any script the
launchd agents run stops being **Python 3.9 compatible**. Those agents invoke `/usr/bin/python3`
(3.9.6 on this Mac), not the venv, so a `str | None` annotation in the tracker is a silent
production break: the tick dies at 09:05 and that day's snapshot is lost.

## Where things are

```
02_Data/          acquire, clean, label
  collect_and_extract.py    download → 20 frames → features → delete the video
  backfill_published_at.py  fields yt-dlp missed, from the YouTube API
  clean_retrospective.py    transcript rebuild + quality manifest
  compute_labels_v2.py      the "viral" label (stratified; see eda.md before changing)
  track_new_videos.py       the live daily tracker
  eda_retrospective.py      the analysis behind eda.md
  yt_shorts.py              definitive Shorts check (shared, no API quota)
  processed/                the retrospective dataset (Git LFS)
  tracking/                 the live cohort's CSVs (gitignored)

03_Models/        one pipeline, for baselines and the deep model alike
  ytdiag/adapters.py        both datasets → one canonical table
  ytdiag/features.py        feature groups; refuses to hand out label-only columns
  ytdiag/split.py           channel-grouped train/val/test
  ytdiag/baselines.py       dummy floor, logistic regression, XGBoost
  ytdiag/embed.py           frozen DINOv2/ModernBERT embeddings + safe caches
  ytdiag/fusion.py          linear probes + regularised late-fusion MLP
  run_baselines.py          the CLI
  run_deep_multimodal.py    five-seed frozen-content ablations

04_Experiments/   run outputs (gitignored)
05_Reports/       paper, slides, deliverables
06_Meeting_Notes/ decisions and coordination
```

## Three things that will bite you if you don't know them

**1. Never model raw view counts.** A model given only subscriber count, video age, duration and
Shorts-status — no content whatsoever — predicts `log(views)` at out-of-fold R² = 0.584. Views are
mostly a channel-size readout. That is why the target is a *binary label, ranked within a peer
group of similar channel size, age and format*, and why no R² on views may be quoted as evidence
that content features work.

**2. The Shorts/regular format bit leaks everywhere.** `frames_portrait` recovers it for 99% of
videos, and raw thumbnail brightness predicts format at direction-free AUC 0.925, largely because
vertical source imagery is often padded/composed differently in a 16:9 thumbnail. A heuristic
content crop lowers that to 0.559 but is not a verified pillarbox detector. The label stratifies
on format to stop that inflating a score — and every vision result still needs a format-only
baseline printed beside it.

**3. Splits must be grouped by channel.** A plain random split lets a model reach AUC 0.86 by
memorising which channel a video came from. `ytdiag/split.py` enforces this and asserts it.

## Current state (2026-09-03)

- **Data collection** — done: 1,860 videos (comedy 464, howto 373, product_reviews 482, vlogs 541).
  Short of the original 8,000 target because `search.list` pagination caps out well below its
  advertised result counts; disclosed as a limitation.
- **Prospective tracking** — live and healthy since 2026-08-26; first 7-day outcomes 2026-09-02,
  first 30-day outcomes 2026-09-25.
- **Cleaning** — done. **EDA** — done, with three label changes recommended and awaiting sign-off.
- **Baselines** — complete over five channel-grouped validation seeds. The fixed full-engineered
  XGBoost reaches AUC 0.605 ± 0.030; nested tuning makes metadata+schedule XGBoost the strongest
  comparator at 0.619 ± 0.022.
- **Frozen deep Tier 1** — complete for thumbnail + quality-gated text. The best full linear fusion reaches
  0.570 ± 0.051 and the MLP 0.554 ± 0.022, so neither beats the structured baselines.
- **Field-aware text v2** — complete. Separate field/chunk embeddings lift the best MLP to
  0.600 ± 0.026, but it is 0.019 below tuned metadata+schedule XGBoost and wins only one of five
  paired splits.
- **Deep-head tuning** — complete. Nested selection across nine MLP heads reaches 0.598 ± 0.037,
  slightly below and more variable than the pre-specified 0.600 ± 0.026 head.
- **Visual ablation** — complete for thumbnails and all 20 frames with DINOv2-small/base, CLIP and
  ResNet-50. CLIP is strongest visually: frame-only linear AUC is 0.605 ± 0.028 and adding its
  thumbnail reaches 0.606 ± 0.040. The best observed full fusion combines CLIP thumbnail/frames,
  field-aware text, metadata and schedule at 0.613 ± 0.044, still 0.006 below tuned
  metadata+schedule XGBoost and ahead on two of five paired splits. ResNet-50 remains weak.

These are repeated development results, not a final test estimate. Although each split is
channel-disjoint internally, changing the split seed reassigns videos: every nominal retrospective
test row appears in training or validation under another seed. The prospective cohort will provide
the genuinely unseen external evaluation after its fixed-horizon outcomes mature.

## A note on conventions

- **Every change is logged in `CHANGELOG.md`.** It is the single source of truth for what happened.
- Comments here explain **why**, not what. In ML the code usually runs fine and the *answer* is
  wrong — the bugs live in assumptions about the data, so that is what the prose defends.
- Feature *groups* are the unit of configuration. Every ablation is a `--groups` value, never a
  second copy of the pipeline.
- `02_Data/FEATURES.md` decides what may be a model input. Code enforces it: `features.py` raises
  if a label-only column is ever selected.
