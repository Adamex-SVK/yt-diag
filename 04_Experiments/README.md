# 04_Experiments — README

## Purpose
Experiment tracking: configurations, run logs, results, and analysis. One subfolder per experiment.

## Planned experiments

1. **Metadata-only baselines** — logistic regression and XGBoost on 12 structured features. Establishes whether multimodal signal adds value.
2. **Text-only model** — pretrained transformer on title + description + transcript. Tests text signal alone.
3. **Vision-only model** — vision backbone on thumbnail + sampled frames. Tests visual signal alone.
4. **Full multimodal (frozen encoders)** — all three modalities, fusion layer trained, encoders frozen. Simpler, faster.
5. **Full multimodal (fine-tuned)** — optional end-to-end follow-up whose value must be established empirically; it is substantially heavier and more prone to overfitting here.
6. **Ablation studies** — remove one modality at a time to measure marginal contribution.
7. **Attribution quality** — human evaluation of explanation quality (does the attribution layer surface plausible reasons?).

## Experiment template

Each experiment subfolder: `YYYY-MM-DD_Experiment_Name/`
- `config.json` — model, data, hyperparams
- `results.json` — metrics (accuracy, F1, precision, recall, AUC)
- `notes.md` — what we changed, what we expected, what actually happened

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `results_summary.md` | Running comparison table across all experiments | TBD |

## Current baseline status (2026-09-01)

The retrospective tabular ladder has been run on the v3 CCT dataset with five
channel-grouped 60/20/20 split seeds. These are validation results only; the
test split is untouched. The tracked source of truth is
`05_Reports/final_report/results/baselines.json`, including CCT version/method.

| Feature set | Logistic AUC | XGBoost AUC |
|---|---:|---:|
| Metadata + schedule | 0.597 ± 0.055 | 0.589 ± 0.028 |
| Metadata + schedule + visual | 0.597 ± 0.050 | 0.577 ± 0.026 |
| Full engineered (meta + schedule + visual + audio) | 0.587 ± 0.042 | 0.605 ± 0.030 |
| Visual only | 0.519 ± 0.020 | 0.505 ± 0.012 |

Values are mean ± standard deviation across split seeds, not confidence
intervals.

## Nested tabular tuning (2026-09-02)

The fixed baselines above remain the reproducible reference. A separate nested
workflow tunes 12 logistic configurations and 17 XGBoost configurations per
outer split (the fixed XGBoost configuration plus 16 reproducibly sampled
candidates). Selection uses mean AUC across four channel-grouped folds wholly
inside the outer training rows. The operating threshold is selected from inner
out-of-fold training predictions; outer validation is evaluation-only and test
remains untouched.

| Feature set | Model | Fixed AUC | Tuned AUC | Paired tuned wins |
|---|---|---:|---:|---:|
| Metadata + schedule | Logistic | 0.597 ± 0.055 | 0.590 ± 0.048 | 1/5 |
| Metadata + schedule | XGBoost | 0.589 ± 0.028 | **0.619 ± 0.022** | 4/5 |
| Full engineered | Logistic | 0.587 ± 0.042 | 0.595 ± 0.032 | 3/5 |
| Full engineered | XGBoost | 0.605 ± 0.030 | 0.614 ± 0.023 | 3/5 |

The gain is specific to XGBoost rather than evidence that tuning always helps.
The strongest tuned model uses metadata+schedule only; adding engineered visual
and audio features changes its AUC by -0.005. Results are descriptive over five
overlapping split seeds, not confidence intervals or a significance test. Full
trial histories and thresholded metrics are in the gitignored
`runs/tuned_tabular_baselines/results.json` and are reproducible with
`03_Models/run_tuned_baselines.py`.

## Frozen deep multimodal result (2026-09-02)

Tier 1 uses revision-pinned, frozen DINOv2 ViT-S/14 thumbnail embeddings and
ModernBERT-base embeddings of title + cleaned transcript + description. A
linear probe and a regularised late-fusion MLP were evaluated over the same five
channel-grouped split seeds. Logistic C and MLP epoch selection occur on an
inner channel-grouped split of the training fold; outer validation is not used
for model selection and test remains untouched.

| Inputs | Linear probe AUC | Late-fusion MLP AUC |
|---|---:|---:|
| Thumbnail | 0.563 ± 0.032 | 0.547 ± 0.019 |
| Text | 0.523 ± 0.038 | 0.512 ± 0.055 |
| Thumbnail + text | 0.551 ± 0.041 | 0.531 ± 0.033 |
| Thumbnail + text + metadata/schedule | 0.570 ± 0.051 | 0.554 ± 0.022 |

This is a negative result: neither deep configuration beats tuned
metadata+schedule XGBoost (0.619 ± 0.022), and both trail tuned full engineered
XGBoost (0.614 ± 0.023). The full linear fusion ranges from 0.496 to 0.632 across seeds, so adding
a frame transformer now would compound split sensitivity rather than answer the core
question. Full run details are gitignored under
`04_Experiments/runs/deep_multimodal/results.json`; the compact tracked evidence
is `05_Reports/final_report/results/deep_multimodal.json`.

The final run is quality-gated: `cleaning_manifest.csv` exposes only 1,404
usable transcripts. The other 456 transcript files are not encoded, although
their title and description remain. This replaced a provisional run that had
mistaken file existence for transcript validity.

## Field-aware text follow-up (2026-09-02)

The pre-declared follow-up replaces generic concatenation with an
embedding-trained ModernBERT encoder, separate title/description/transcript
fields, and up to four evenly distributed overlapping transcript chunks. The
encoder is frozen and revision-pinned; model selection remains nested inside
the outer training fold and the test split remains untouched.

| Inputs | Linear probe AUC | Late-fusion MLP AUC |
|---|---:|---:|
| Title + description | 0.531 ± 0.045 | 0.528 ± 0.036 |
| Transcript | 0.562 ± 0.028 | 0.537 ± 0.016 |
| All text fields | 0.568 ± 0.018 | 0.571 ± 0.004 |
| All text + metadata/schedule | 0.581 ± 0.023 | 0.581 ± 0.020 |
| Thumbnail + all text + metadata/schedule | 0.587 ± 0.033 | **0.600 ± 0.026** |

The strongest v2 model is -0.019 AUC below tuned metadata+schedule XGBoost and
wins only 1/5 paired seeds. It is -0.014 below tuned full engineered XGBoost
while winning 3/5 seeds. These five overlapping split-seed comparisons
are descriptive, not confidence intervals or significance tests. Adding the
engineered block to v2 does not help. The controlled visual experiment below
tests frame aggregation and alternative thumbnail representations directly.

## Nested deep-head tuning (2026-09-02)

The strongest field-aware frozen-content configuration was subjected to a
post-hoc robustness search without rerunning or training either encoder. For
each of five outer channel-grouped seeds, three inner grouped folds select among
12 linear configurations and nine MLP heads. Every fold fits its own scalers;
each MLP fold uses another grouped monitor split to select its epoch. The search
varies projection and hidden widths, dropout, learning rate, weight decay, batch
size and positive-class weighting.

| Head | Pre-specified AUC | Nested tuned AUC | Tuned wins |
|---|---:|---:|---:|
| Linear probe | 0.587 ± 0.033 | 0.587 ± 0.034 | 1/5 |
| Late-fusion MLP | **0.600 ± 0.026** | 0.598 ± 0.037 | 2/5 |

The MLP search helps seeds 0 and 1 but hurts seeds 2–4, increasing variability.
It remains 0.021 AUC below tuned metadata+schedule XGBoost and wins only 2/5
paired splits. This is not evidence that tuning damaged a true effect; it says
the inner folds cannot select a head that generalises consistently across held-
out channels. Escalating to end-to-end encoder fine-tuning is not justified.
Full trials are under gitignored `runs/tuned_deep/results.json`.

## Controlled visual ablation (2026-09-03)

The visual follow-up changes one representation choice at a time while keeping
the labelled cohort, five channel-grouped outer splits and downstream heads
fixed. It compares DINOv2 model size, CLS versus patch-mean pooling, centre crop
versus aspect-preserving padding, language-supervised CLIP, convolutional
ResNet-50, and mean aggregation of the 20 stored video frames.

| Frozen visual input | Linear probe AUC | Late-fusion MLP AUC |
|---|---:|---:|
| DINOv2-S thumbnail, CLS | 0.563 ± 0.032 | 0.547 ± 0.019 |
| DINOv2-S thumbnail, patch mean | 0.565 ± 0.035 | 0.520 ± 0.041 |
| DINOv2-S full-thumbnail pad, CLS | 0.556 ± 0.051 | 0.541 ± 0.039 |
| DINOv2-B thumbnail, CLS | 0.551 ± 0.031 | 0.547 ± 0.050 |
| CLIP ViT-B/32 thumbnail | **0.570 ± 0.047** | 0.548 ± 0.037 |
| ResNet-50 thumbnail | 0.520 ± 0.013 | 0.521 ± 0.045 |
| DINOv2-S mean of 20 frames | 0.582 ± 0.039 | **0.589 ± 0.023** |

DINOv2-base being worse than DINOv2-small rules out backbone size as the main
explanation for the weak thumbnail model. CLIP provides a small thumbnail-only
gain, but sampled frames contribute more. The best full frozen fusion
(thumbnail + frames + field-aware text + metadata/schedule) reaches
0.609 ± 0.026, 0.010 below tuned metadata+schedule XGBoost and higher on one of
five matched splits. Full per-seed evidence is gitignored under
`runs/visual_ablation/results.json`; the compact report evidence is
`05_Reports/final_report/results/visual_ablation.json`. Test remains untouched.

## Notes
- Always log: what you changed, what you expected, what actually happened.
- Report honestly on accuracy ceiling (external factors cap it).
- Keep `results_summary.md` updated after every experiment run.
