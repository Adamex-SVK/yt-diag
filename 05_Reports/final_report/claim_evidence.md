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

## Pre-processing

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| dup8 median 0.349 → 0.000; word ratio 0.352 | confirmed | notebook §2 | Data |
| Transcript kinds 944 / 467 / 391 / 37 / 12 / 9 | confirmed | `02_Data/processed/cleaning_manifest.csv`, `transcript_kind` | Data |
| Native-English-only leaves comedy 77, vlogs 64 | confirmed | `02_Data/eda.md` §7 | Data (limitation) |
| CCT (v3): linear-light sRGB-to-XYZ, nearest Planckian-locus lookup, estimates with Duv > 0.05 missing | confirmed | `cd 02_Data && ../.venv/bin/python tests/test_cct.py` | Experiment |
| v3 frame-mean distribution 2,553–15,679 K, median 5,666; all 1,860 files carry the same v3 method | confirmed | `.venv/bin/python 02_Data/recompute_cct.py --dry-run`; `eda_features_stats.json` | Experiment |
| v3 coverage: 1,854/1,860 thumbnails and 36,652/37,200 frames valid | confirmed | `02_Data/eda_features/eda_features_stats.json` | Experiment |
| v3 visual/full baselines use five repeated channel-grouped development splits | confirmed | `results/baselines.json` (`cct_version` recorded) | Experiment |

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
| Three label changes revise the locked proposal definition | **confirmed** — signed off by Adam 2026-09-05 (no age floor: 861-video cost vs. AUC 0.576→0.572; format stratification: AUC 0.596→0.447; three bands: 4-band gives 102 undersized cells, 2-band leaves AUC 0.660 residual confounding) | Adam | Method |
| Only 83 of 438 videos ever labelled top-quartile retain that label in all 16 configurations | confirmed | EDA config sweep | Limitations |

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
| Tuning and F1-threshold selection use only four inner channel-grouped training folds; outer validation is evaluation-only within each run | confirmed | `ytdiag/tuning.py`; `test_tuning.py`; exact trials in gitignored run result | Method, Experiment |
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
| Deep tuning uses fold-local preprocessing, inner grouped hyperparameter folds, and a further grouped epoch-monitor split | confirmed | `ytdiag/deep_tuning.py`; `test_deep_tuning.py` | Method, Experiments |
| Thumbnail linear AUC: DINOv2-S 0.563 ± 0.032, DINOv2-B 0.551 ± 0.031, CLIP ViT-B/32 0.570 ± 0.047, ResNet-50 0.520 ± 0.013 | confirmed | `run_visual_ablation.py --seeds 0,1,2,3,4`; `publish_visual_ablation.py`; `results/visual_ablation.json` | Experiments |
| All-frame visual results: DINOv2-S MLP 0.589 ± 0.023; DINOv2-B MLP 0.602 ± 0.026; CLIP linear 0.605 ± 0.028; ResNet-50 MLP 0.548 ± 0.018 | confirmed | same visual-ablation evidence | Experiments |
| CLIP thumbnail + frames linear = 0.606 ± 0.040 | confirmed | `results/visual_ablation.json → aggregate.thumbnail_frames_clip` | Experiments |
| CLIP thumbnail + frames + field-aware text + metadata/schedule linear = 0.613 ± 0.044; -0.006 vs tuned metadata/schedule XGBoost and 2/5 paired wins | confirmed | `results/visual_ablation.json`, exact seed arrays under `aggregate` and `comparison` | Abstract, Experiments, Conclusion |
| Visual ablation changes backbone, encoder size, crop/padding, pooling, and frame aggregation over the same five grouped development splits | confirmed | `ytdiag/visual_ablation.py`; `run_visual_ablation.py`; `test_visual_ablation.py` | Method, Experiments |
| Every nominal retrospective test row appears in training or validation under at least one other seed, so there is no globally sealed retrospective test set | confirmed | `results/split_exposure.json`; `audit_split_exposure.py` | Evaluation protocol, Limitations |
| Linear CLIP fusion logits are exactly decomposed into six standardized input-block contributions | confirmed | `results/attribution.json`; `ytdiag/attribution.py`; reconstruction test | Method, Experiment |
| **Test-set metrics** | **pending** | evaluate the frozen selected model ONCE after the modelling policy is final | Experiments |
| **Attribution examples** | **pending** | Integrated Gradients not implemented | Experiments |
| Post-freeze audio ablation: meta+sched 0.614 ± 0.022 → meta+sched+aud (88-col eGeMAPS) 0.632 ± 0.028, tuned XGBoost, same 5 seeds | confirmed | `run_audio_ablation.py --seeds 0,1,2,3,4`; `publish_audio_ablation.py`; `results/audio_ablation.json` | Experiments |
| Correlation-reduced audio (49 eGeMAPS representatives, |ρ|>0.75 clustering, + 4 pause features = 53 cols) reproduces the lift within noise: 0.633 ± 0.030 | confirmed | same run; `results/audio_ablation.json → reduced_audio` | Experiments |
| Gain-based importance is diffuse across acoustic families (formants 16%, spectral slope 15%, MFCC 15%, F0 pitch 15%), each ahead of metadata+schedule's own 12% share — not concentrated in a few columns | confirmed | `results/audio_ablation.json → gain_family_share` | Experiments |
| **Not part of the frozen finalist** — computed 2026-09 after `final_model_policy.json` froze (2026-09-03); uses only already-approved retrospective `aud__` columns inside the existing nested channel-grouped protocol; does not touch the sealed test split or the prospective panel | limitation | `run_audio_ablation.py` docstring | Experiments, Limitations |
| Resolved 2026-09-05 (Emmanuel + Adam): reported as a standalone post-hoc finding, not a trigger for a follow-up freeze cycle | confirmed | `main.tex` `\note{PROVISIONAL}` on the audio-ablation paragraph (softened 2026-09-06, see significance test below) | Experiments |
| Same isolate-and-tune treatment applied to engineered visual (15-column `vis__` block, not the deep DINOv2/CLIP embeddings): meta+sched 0.614 ± 0.022 → meta+sched+vis 0.591 ± 0.028 — *below* the reference, not above it | confirmed | `run_visual_engineered_ablation.py --seeds 0,1,2,3,4`; `publish_visual_engineered_ablation.py`; `results/visual_engineered_ablation.json` | Experiments, Limitations |
| Gain-based importance across the 15 visual columns is flat (3.5%–8.7% each), consistent with noise rather than a masked signal — rules out isolation-alone as the explanation for audio's lift | confirmed | `results/visual_engineered_ablation.json → gain_share_within_visual` | Experiments |
| Paired bootstrap significance (videos resampled within each seed's validation fold, hyperparameters held fixed at their already-tuned values — no re-search): audio's lift is +0.018 AUC, 95% CI [-0.009, +0.044], one-sided p=0.089 for "not better" — does **not** clear conventional significance, though directionally consistent | confirmed | `run_ablation_significance.py`; `04_Experiments/runs/ablation_significance/results.json → audio_isolated_vs_reference` | Experiments, Limitations |
| Same test on the visual-engineered drop: -0.023 AUC, 95% CI [-0.042, -0.004] — entirely below zero, i.e. statistically solid, unlike audio's inconclusive lift | confirmed | `run_ablation_significance.py`; `results.json → visual_engineered_vs_reference` | Experiments |
| Post-freeze audio+text fusion: engineered audio + frozen ModernBERT text embeddings, combined in the same linear/MLP fusion architecture as the frozen finalist. Text+meta 0.581/0.578 (linear/MLP), audio+meta in this architecture 0.608/0.585 (below audio's 0.632 under tuned XGBoost), combined 0.578/0.597 — below metadata alone (0.619) and below audio alone in either architecture. Text adds nothing; audio's lift is tied to tree-based modelling of tabular features, not portable to this fusion architecture | confirmed | `run_audio_text_fusion.py --seeds 0,1,2,3,4`; `publish_audio_text_fusion.py`; `results/audio_text_fusion.json` | Experiments |
| Conclusion and Limitations now state the audio-isolation lift is directionally consistent but not statistically significant (previously stated flatly as an improvement, inconsistent with the Experiment section's own softened `PROVISIONAL` note) | confirmed | `main.tex` Conclusion + Limitations, matches the significance test above | Conclusion, Limitations |

---

## Before submission

- [ ] Flip `\drafttrue` → `\draftfalse` in `main.tex`; confirm every `\note` disappears
- [ ] Section page budget from the course guideline: Introduction ≤1p, Related Work ≤0.5p, Methodology ≤2p, Experiment ≤2p, Conclusion one paragraph, Abstract ~250 words. Checked 2026-09-06 with a throwaway Tectonic compile (not the submission toolchain — draft `\note`s and `\pdfinfo` were stripped for this check only): Introduction and Methodology comfortable, Conclusion compliant (one paragraph), Abstract 255 words. Related Work still runs slightly past 0.5p even after folding its third subsection in. Experiment was ~2.7–2.8p before trimming the three post-freeze ablation paragraphs (they repeated the same bootstrap-protocol framing three times and restated the audio-isolation numbers in both Limitations and Conclusion); tightened to ~2.4–2.5p, still somewhat over. Re-verify against the real Overleaf compile, not this estimate.
- [ ] Add the 3-minute presentation video link to the report
- [ ] Add the GitHub/drive link for dataset + source code (the guideline requires it)
- [ ] Re-run `make_tables.py` so the table matches the final code
- [ ] Every `pending` row above is either resolved or removed from the report
- [ ] Compile in Overleaf and check: two columns, no page numbers, Times, fonts embedded
- [x] `references.bib` populated for every cited work
- [x] Adam's sign-off recorded on the three label changes (2026-09-05)
