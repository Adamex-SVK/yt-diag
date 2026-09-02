# Notebooks

## Purpose

Runnable, line-by-line interfaces to the tested data and modelling pipeline.
Notebook cells demonstrate the real modules; they do not duplicate cleaning,
label, split, or model logic.

## Contents

| File | Purpose | Status |
|---|---|---|
| `build_walkthrough.py` | Generates the walkthrough deterministically; edit this rather than the JSON notebook | Done |
| `00_project_walkthrough.ipynb` | Executed end-to-end walkthrough of cleaning evidence, EDA, labels, baselines, Tier 1, and text v2 | Passing |

## Usage

```bash
.venv/bin/python notebooks/build_walkthrough.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/00_project_walkthrough.ipynb --ExecutePreprocessor.timeout=600
```

The notebook reads the tracked deep-model summaries instead of regenerating
encoder embeddings; the full accelerated experiment is run through
`03_Models/run_deep_multimodal.py` and `03_Models/run_text_v2.py`.
