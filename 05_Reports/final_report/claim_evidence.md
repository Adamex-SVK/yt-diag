# Claim → evidence register

Every quantitative claim in `main.tex` must appear here with a command that reproduces it.
If a number is in the report and not in this table, it is unsourced and must not ship.

**Status meanings.** `confirmed` = reproduced from code output, command given.
`provisional` = a decision we made, defensible but not team-signed-off.
`pending` = the experiment has not run. `limitation` = a stated threat to validity.

Regenerate the baseline table (never retype it):
```bash
.venv/bin/python 05_Reports/final_report/make_tables.py --seeds 5
```

---

## Dataset

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| 1,860 videos; comedy 464 / howto 373 / product_reviews 482 / vlogs 541 | confirmed | `eda_retrospective.py` → `by_category` in `02_Data/eda/eda_stats.json` | Data |
| 1,319 distinct channels, 81.7% contributing one video | confirmed | notebook §7 | Data, Method |
| 11,256 prospective cohort videos | confirmed | `wc -l 02_Data/tracking/cohort.csv` | Data |
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
| v3 distribution 2,553–25,000 K, median 5,678; all 1,860 files `cct_version=3` | confirmed | `.venv/bin/python 02_Data/recompute_cct.py --dry-run` | Experiment |
| v3 changed baselines only within seed noise | confirmed | `results/baselines.json` (`cct_version` recorded) | Experiment |
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
| 96.3% of Shorts thumbnails are pillarboxed | confirmed | `eda.md` §4 | Method |
| Channel features alone: AUC 0.644 under random folds | confirmed | notebook §7 | Method |
| No channel spans splits (asserted at runtime) | confirmed | `ytdiag/split.py`; notebook §7 | Method |
| ~6–7 positives per category per test split | confirmed | `eda.md` §5 | Method (limitation) |

## Results

| Claim | Status | Reproduce with | In report |
|---|---|---|---|
| Baseline table (all cells) | confirmed | `make_tables.py --seeds 5` — writes `results/baselines.json` **and** into `main.tex` | Experiment |
| Visual block alone ≈ chance (0.505) | confirmed | same | Experiments |
| LR on metadata ranges 0.496–0.660 across seeds | confirmed | same | Experiments |
| **Test-set metrics** | **pending** | `run_baselines.py --test` — run ONCE, at the end | Experiments |
| **Deep multimodal model** | **pending** | not yet implemented | Experiments |
| **Four planned ablations** | **pending** | only the feature ladder has run | Experiments |
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
- [ ] `references.bib` populated (currently three entries needed: Wu 2018, Abu-El-Haija 2016, Rajaram 2020)
- [ ] Adam's sign-off recorded on the three label changes
