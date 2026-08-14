# 03_Models — README

## Purpose
Model architecture designs, training configurations, hyperparameter choices. One subfolder per model variant.

## Architecture overview

YT-Diag fuses **three modalities** into a single end-to-end trained model:

| Component | Input | Architecture | Notes |
|-----------|-------|-------------|-------|
| Text encoder | Title + description + transcript | Pretrained transformer (e.g. BERT/RoBERTa) | Shared across title/desc/transcript or separate? |
| Vision encoder | Thumbnail (1280×720) | Pretrained CNN/ViT | Same backbone as frame encoder |
| Temporal encoder | 16–24 sampled frames | Same vision backbone + lightweight temporal transformer | Following Abu-El-Haija et al. (2016) |
| Fusion layer | All three embeddings | Cross-attention or concatenation | Key design decision |
| Classification head | Fused representation | FC → sigmoid | Binary: viral / non-viral |
| Attribution layer | Internal representations | Attention-based or ablation-based | Inspired by Rajaram & Manchanda (2020) |

## Baselines (non-deep-learning)

- Logistic regression on 12 metadata features
- XGBoost on 12 metadata features
- Adapted from Wu et al. (2018) codebase

## Key design decisions (TBD)

1. Which pretrained text transformer? (BERT, RoBERTa, DistilBERT — trade off accuracy vs. compute)
2. Which vision backbone? (ResNet, ViT, CLIP — CLIP interesting because it bridges text+vision)
3. Fusion strategy: early (concatenate embeddings), mid (cross-attention), or late (ensemble)?
4. Attribution method: integrated gradients, attention weights, or SHAP on final layer?
5. Training strategy: joint end-to-end vs. frozen encoders + trainable fusion head?

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `architecture_overview.md` | Full architecture diagram and design rationale | TBD |
| `baseline/` | Baseline model configs and results | TBD |
| `multimodal/` | Multimodal model variants | TBD |
| `attribution/` | Attribution layer experiments | TBD |

## Status
- **Architecture design**: not started (proposal committed to high-level approach only)
- **Implementation**: not started

## Notes
- Document WHY each choice, not just WHAT. Future us will forget.
- Track all hyperparameter decisions and their rationale.
- Checkpoints go in GitHub repo — document paths here.
