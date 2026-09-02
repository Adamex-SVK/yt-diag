# 04_Experiments — README

## Purpose
Experiment tracking: configurations, run logs, results, and analysis. One subfolder per experiment.

## Planned experiments

1. **Metadata-only baselines** — logistic regression and XGBoost on 12 structured features. Establishes whether multimodal signal adds value.
2. **Text-only model** — pretrained transformer on title + description + transcript. Tests text signal alone.
3. **Vision-only model** — vision backbone on thumbnail + sampled frames. Tests visual signal alone.
4. **Full multimodal (frozen encoders)** — all three modalities, fusion layer trained, encoders frozen. Simpler, faster.
5. **Full multimodal (fine-tuned)** — all three modalities, end-to-end fine-tuned. Expected best but heaviest.
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
intervals. Deep text/vision/fusion experiments remain not started.

## Notes
- Always log: what you changed, what you expected, what actually happened.
- Report honestly on accuracy ceiling (external factors cap it).
- Keep `results_summary.md` updated after every experiment run.
