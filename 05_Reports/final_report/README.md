# Final report

## Purpose

AAAI-formatted project paper and the scripts/evidence that keep its quantitative
claims reproducible.

## Contents

| File | Purpose | Status |
|---|---|---|
| `main.tex` | Single-file report source required by the course template | Active draft |
| `claim_evidence.md` | Claim-to-command register; every reported number belongs here | Active |
| `make_tables.py` | Recomputes and inserts the compact structured baseline ladder | Passing |
| `publish_tuned_baselines.py` | Validates nested tuning and publishes compact comparison evidence | Passing |
| `make_deep_table.py` | Validates the five-seed deep run, publishes compact evidence, and inserts its table | Passing |
| `make_text_v2_table.py` | Validates and publishes the field-aware text follow-up | Passing |
| `publish_tuned_deep.py` | Publishes the nested frozen-head tuning robustness check | Passing |
| `publish_visual_ablation.py` | Publishes the controlled visual-backbone and sampled-frame ablation | Passing |
| `publish_attribution.py` | Publishes finalist metrics, exact block attribution, and error analysis | Passing |
| `make_figures.py` | Generates report figures from tracked analysis outputs | Passing |
| `results/` | Compact tracked experiment evidence used by the report | Fixed/tuned baseline + text + tuned-head + visual + attribution results present |
| `build.sh` | Multi-pass LaTeX build plus font/page checks | Passing; five pages including references |

Generated LaTeX files and `main.pdf` are ignored. Set `\draftfalse` only for the
submission build after every pending claim is resolved or removed.
