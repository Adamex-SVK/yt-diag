# Claim → evidence register

Every quantitative claim in `main.tex` must appear here with a command that reproduces it.
If a number is in the report and not in this table, it is unsourced and must not ship.

**Status meanings.** `confirmed` = reproduced from code output, command given.
`provisional` = a decision we made, defensible but not team-signed-off.
`pending` = the experiment has not run. `limitation` = a stated threat to validity.

Regenerate the baseline table (never retype it):
```bash
.venv/bin/python 05_Reports/final_report/make_tables.py --seeds 5
.venv/bin/python 05_Reports/final_report/make_deep_table.py
.venv/bin/python 05_Reports/final_report/make_text_v2_table.py
.venv/bin/python 05_Reports/final_report/publish_visual_ablation.py
```

---

## Dataset

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| 1,860 videos; comedy 464 / howto 373 / product_reviews 482 / vlogs 541 | confirmed | `eda_retrospective.py` → `by_category` in `02_Data/eda/eda_stats.json` | Data |
| 1,319 distinct channels, 81.7% contributing one video | confirmed | notebook §7 | Data, Method |
| 12,738 prospective videos as of 2026-09-02; 10,443 in the date-window arm | confirmed | dated tracker audit; top-level `README.md` | Data |
| 20,000 date-window ceiling = 4 principal categories × 4,000 plus 1 backup category × 4,000 | confirmed | tracker collection policy; top-level `README.md` | Data |
| 20 frames/video, 12 inside the first 60 s | confirmed | `compute_frame_timestamps()`; notebook §3 | Data |
| Collection fell short of the 8,000 target (pagination cap) | limitation | `02_Data/cc_availability_scan_findings.md` | Data |

## Data repair

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| 98% of auto-caption transcripts repeat each phrase ~3× | confirmed | notebook §2 (113-video sample) | Data |
| dup8 median 0.349 → 0.000; word ratio 0.352 | confirmed | notebook §2 | Data |
| Transcript kinds 944 / 467 / 391 / 37 / 12 / 9 | confirmed | `02_Data/processed/cleaning_manifest.csv`, `transcript_kind` | Data |
| Native-English-only leaves comedy 77, vlogs 64 | confirmed | `02_Data/eda.md` §7 | Data (limitation) |
| Old CCT reached 5.0×10⁹ K, 27 negative | confirmed | pre-fix values; `KNOWN_ISSUES.md` §6 | Data |
| Corrected CCT (v3): nearest-Planckian-locus, 1% LUT, signed Duv | confirmed | `cd 02_Data && ../.venv/bin/python tests/test_cct.py` | Experiment |
| v3 recovers locus reference points exactly where McCamy drifts (25,000 K: 25,000 vs 19,504) | confirmed | `_nearest_planckian_cct_duv_uv` vs McCamy, see CHANGELOG | Experiment |
| v3 frame-mean distribution 2,553–15,679 K, median 5,666; all 1,860 files carry the same v3 method | confirmed | `.venv/bin/python 02_Data/recompute_cct.py --dry-run`; `eda_features_stats.json` | Experiment |
| v3 coverage: 1,854/1,860 thumbnails and 36,652/37,200 frames valid | confirmed | `02_Data/eda_features/eda_features_stats.json` | Experiment |
| v3 visual/full baselines use five channel-grouped validation splits; test untouched | confirmed | `results/baselines.json` (`cct_version` recorded) | Experiment |
| Pure green would report 6069 K without the Duv gate | confirmed | notebook §4 | Data |

## Target and label

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| **Shortcut ceiling R² = 0.584**, channel-grouped | confirmed | `eda_retrospective.py::shortcut_ceiling`; notebook §9 | Methodology |
| It is a *diagnostic*, not a leakage-free estimate — subscriber count is observed retrospectively, and a hit raises its own channel's count | limitation | reasoning, stated in Methodology | Methodology |
| Random folds give 0.643 — the leaky comparison, not ours | confirmed | notebook §9 discussion | Method |
| Subscribers alone reach R² ≈ 0.43 | confirmed | `eda_stats.json → shortcut_ceiling` | Method |
| 1,854 labelled, 438 viral (23.6%) | confirmed | `.venv/bin/python 02_Data/compute_labels_v2.py --category all` | Method |
| 3×3×2 cells, median cell 24, 22 cells < 20 | confirmed | `eda_stats.json → label_configs_format_stratified` | Method |
| Age floor cost 861 videos, changed AUC 0.576 → 0.572 | confirmed | `eda.md` §2 | Method |
| Day-1 vs day-5 rank Spearman 0.946; 89.3% quartile agreement | confirmed | `eda_stats.json → rank_stability` | Method |
| Stability beyond day 5 untested (panel too young) | limitation | — re-check after 2026-09-25 | Method |
| Format unstratified: label AUC from `is_short` 0.596 → 0.447 | confirmed | `eda_stats.json`, both config blocks | Method |
| 4×4×2 → median cell 13, 102 small cells; 2×2×2 → AUC 0.660 | confirmed | `eda_stats.json → label_configs_format_stratified` | Method |
| Three label changes revise the locked proposal definition | **signoff** | Adam | Method |
| Only 83 of 438 ever-viral videos are viral in all 16 configs | confirmed | EDA config sweep | Limitations |

## Leakage and protocol

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| `frames_portrait` recovers `is_short` for 99.0% | confirmed | `eda_stats.json → format_leakage`; notebook §6 | Method |
| Raw thumbnail brightness predicts format at direction-free AUC 0.925; heuristic content crop reduces it to 0.559 | confirmed | `02_Data/eda_features/eda_features_stats.json` | Method |
| Training-partition dashboard: subscriber count is the strongest pooled association with log views (Spearman rho = 0.705); duration reverses from weakly negative pooled to positive within both formats | confirmed | `02_Data/eda_features.py`; `eda_features/feature_outcome_associations_train_seed0.csv`; notebook section 5 | Experiment (descriptive) |
| Usable transcript coverage differs by format (92.4% regular vs 60.2% Shorts); 23 eGeMAPS pairs have absolute Spearman rho above 0.95 | confirmed | `eda_features/modality_coverage_by_category_format.csv`; `eda_features/audio_spearman.csv`; notebook section 5 | Experiment (descriptive) |
| Channel features alone: AUC 0.644 under random folds | confirmed | notebook §7 | Method |
| No channel spans splits (asserted at runtime) | confirmed | `ytdiag/split.py`; notebook §7 | Method |
| ~6–7 positives per category per test split | confirmed | `eda.md` §5 | Method (limitation) |

## Results

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| Baseline table (all cells) | confirmed | `make_tables.py --seeds 5` — writes `results/baselines.json` **and** into `main.tex` | Experiment |
| Nested tuned metadata/schedule XGBoost = 0.619 ± 0.022; tuned full engineered = 0.614 ± 0.023 | confirmed | `run_tuned_baselines.py`; `publish_tuned_baselines.py`; `results/tuned_baselines.json` | Experiment |
| Tuning and F1-threshold selection use only four inner channel-grouped training folds; outer validation is evaluation-only and test is untouched | confirmed | `ytdiag/tuning.py`; `test_tuning.py`; exact trials in gitignored run result | Method, Experiment |
| Visual block alone ≈ chance (0.505) | confirmed | same | Experiments |
| LR on metadata ranges 0.496–0.660 across seeds | confirmed | same | Experiments |
| Frozen thumbnail/text/full linear AUC = 0.563/0.523/0.570; full MLP = 0.554 | confirmed | `run_deep_multimodal.py --seeds 0,1,2,3,4`; `results/deep_multimodal.json` | Experiments |
| Full deep fusion does not beat tuned metadata+schedule XGBoost (0.619) or tuned engineered XGBoost (0.614) | confirmed | `results/deep_multimodal.json`; `results/tuned_baselines.json` | Experiments |
| Only 1,404/1,860 transcripts pass the cleaning-manifest usability gate; unusable files are withheld from the encoder | confirmed | `cleaning_manifest.csv`; `test_adapter_withholds_manifest_unusable_transcript` | Method, Experiments |
| Full linear fusion ranges 0.496–0.632 across seeds | confirmed | `results/deep_multimodal.json → aggregate.thumbnail_text_meta_sched.linear_probe` | Experiments |
| Text-v2 all-text linear/MLP AUC = 0.568/0.571; text+meta/schedule MLP = 0.581 | confirmed | `run_text_v2.py --seeds 0,1,2,3,4`; `results/text_v2.json` | Experiments |
| Text-v2 thumbnail+text+meta/schedule MLP = 0.600 ± 0.026 | confirmed | `results/text_v2.json → aggregate.thumbnail_text_fields_meta_sched` | Experiments |
| Best text-v2 model vs tuned metadata/schedule XGBoost: -0.019 mean, 1/5 paired wins; vs tuned full engineered: -0.014, 3/5 wins | confirmed | `results/text_v2.json`, exact seed arrays under `aggregate` and `references` | Experiments |
| Nested tuned deep linear/MLP = 0.587 ± 0.034 / 0.598 ± 0.037; MLP does not improve the pre-specified 0.600 ± 0.026 head (2/5 wins) | confirmed | `run_tuned_deep.py`; `publish_tuned_deep.py`; `results/tuned_deep.json` | Experiments |
| Deep tuning uses fold-local preprocessing, inner grouped hyperparameter folds, and a further grouped epoch-monitor split; test untouched | confirmed | `ytdiag/deep_tuning.py`; `test_deep_tuning.py` | Method, Experiments |
| Thumbnail linear AUC: DINOv2-S 0.563 ± 0.032, DINOv2-B 0.551 ± 0.031, CLIP ViT-B/32 0.570 ± 0.047, ResNet-50 0.520 ± 0.013 | confirmed | `run_visual_ablation.py --seeds 0,1,2,3,4`; `publish_visual_ablation.py`; `results/visual_ablation.json` | Experiments |
| Mean DINOv2-S representation over 20 frames reaches 0.582 ± 0.039 linear / 0.589 ± 0.023 MLP | confirmed | same visual-ablation evidence | Experiments |
| Thumbnail + frames + field-aware text + metadata/schedule MLP = 0.609 ± 0.026; -0.010 vs tuned metadata/schedule XGBoost and 1/5 paired wins | confirmed | `results/visual_ablation.json`, exact seed arrays under `aggregate` and `comparison` | Abstract, Experiments, Conclusion |
| Visual ablation changes backbone, encoder size, crop/padding, pooling, and frame aggregation while reusing the same five grouped splits; test untouched | confirmed | `ytdiag/visual_ablation.py`; `run_visual_ablation.py`; `test_visual_ablation.py` | Method, Experiments |
| **Test-set metrics** | **pending** | evaluate the frozen selected model ONCE after the modelling policy is final | Experiments |
| **Attribution examples** | **pending** | Integrated Gradients not implemented | Experiments |

---

## Before submission

- [ ] Flip `\drafttrue` → `\draftfalse` in `main.tex`; confirm every `\note` disappears
- [ ] Section page budget from the course guideline: Introduction ≤1p, Related Work ≤0.5p, Methodology ≤2p, Experiment ≤2p, Conclusion one paragraph, Abstract ~250 words
- [ ] Add the 3-minute presentation video link to the report
- [ ] Add the GitHub/drive link for dataset + source code (the guideline requires it)
- [ ] Re-run `make_tables.py` so the table matches the final code
- [ ] Every `pending` row above is either resolved or removed from the report
- [ ] Compile in Overleaf and check: two columns, no page numbers, Times, fonts embedded
- [x] `references.bib` populated for every cited work
- [ ] Adam's sign-off recorded on the three label changes
