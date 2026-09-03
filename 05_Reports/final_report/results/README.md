# Report result evidence

## Purpose

Small, versioned, machine-readable summaries behind quantitative report tables.
Full experiment logs and model caches stay gitignored elsewhere.

## Contents

| File | Source | Status |
|---|---|---|
| `baselines.json` | `make_tables.py --seeds 5` | Five-seed validation baseline ladder |
| `tuned_baselines.json` | `publish_tuned_baselines.py`, from `04_Experiments/runs/tuned_tabular_baselines/results.json` | Five-seed nested tabular tuning |
| `deep_multimodal.json` | `make_deep_table.py`, from `04_Experiments/runs/deep_multimodal/results.json` | Five-seed frozen thumbnail/text ablations |
| `text_v2.json` | `make_text_v2_table.py`, from the text-v2 run plus `tuned_baselines.json` | Five-seed field-aware text/chunk ablations against tuned comparators |
| `tuned_deep.json` | `publish_tuned_deep.py`, from `04_Experiments/runs/tuned_deep/results.json` | Nested frozen fusion-head tuning follow-up |
| `visual_ablation.json` | `publish_visual_ablation.py`, from `04_Experiments/runs/visual_ablation/results.json` | Frozen thumbnail-backbone, preprocessing, pooling, and frame ablations |
| `attribution.json` | `run_attribution.py` from cached finalist embeddings | Exact linear-logit block contributions, threshold-safe finalist metrics, and four qualitative examples |
| `split_exposure.json` | `audit_split_exposure.py` | Cross-seed proof that no retrospective row remained globally unseen |

All summaries use repeated channel-grouped development splits. No independent
retrospective test exists because rows change roles across seeds; the frozen
prospective external-test policy is in `03_Models/final_model_policy.json`.
