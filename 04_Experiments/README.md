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

## Status
- **All experiments**: not started

## Notes
- Always log: what you changed, what you expected, what actually happened.
- Report honestly on accuracy ceiling (external factors cap it).
- Keep `results_summary.md` updated after every experiment run.
