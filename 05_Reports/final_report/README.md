# Final report

## Purpose

AAAI-formatted project paper and the scripts/evidence that keep its quantitative
claims reproducible.

## Contents

| File | Purpose | Status |
|---|---|---|
| `main.tex` | Single-file report source required by the course template | Active draft |
| `claim_evidence.md` | Claim-to-command register; every reported number belongs here | Active |
| `make_tables.py` | Recomputes and inserts the structured baseline table | Passing |
| `publish_tuned_baselines.py` | Validates nested tuning, publishes compact evidence, and inserts the tuned-comparison table | Passing |
| `make_deep_table.py` | Validates the five-seed deep run, publishes compact evidence, and inserts its table | Passing |
| `make_text_v2_table.py` | Validates and publishes the field-aware text follow-up | Passing |
| `publish_tuned_deep.py` | Publishes the nested frozen-head tuning robustness check | Passing |
| `make_figures.py` | Generates report figures from tracked analysis outputs | Passing |
| `results/` | Compact tracked experiment evidence used by the report | Fixed/tuned baseline + Tier-1 + text-v2 + tuned-head results present |
| `build.sh` | Multi-pass LaTeX build plus font/page checks | Passing; draft is five pages |

Generated LaTeX files and `main.pdf` are ignored. Set `\draftfalse` only for the
submission build after every pending claim is resolved or removed.
