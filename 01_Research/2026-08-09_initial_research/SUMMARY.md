# YT-Diag — Research Summary: What's Ahead

_Generated 2026-08-09 from three parallel research agents (data retrieval, model architectures, evaluation & planning). Full reports in this folder._

---

## TL;DR — What you need to know right now

**We have a solid plan.** The YT-Diag proposal is ambitious but feasible for two students in one month. The three research agents confirmed the approach and surfaced practical constraints. Here's the executive summary.

---

## 1. Data Retrieval — The Biggest Immediate Bottleneck

**Key finding: Use yt-dlp, not the official YouTube API for data collection.**

The official API charges **50 quota units per caption operation** — at 10,000 units/day, you'd need ~21 days just to collect 2,000 videos. yt-dlp bypasses this entirely by using YouTube's internal API (the same one the web client uses), with **no quota**.

**What to do this week:**
1. Install yt-dlp and test on 5 videos
2. Run a quick scan to check CC-licensed video availability in our 4 target categories (comedy, tutorials, vlogs, product reviews)
3. Resolve the "product reviews" category — it's not a native YouTube category; we may need to substitute Science & Technology or Entertainment

**Pipeline:** Discovery (API search for CC IDs) → Metadata (yt-dlp) → Subtitles (yt-dlp, Whisper as fallback for ~10-30%) → Thumbnails (yt-dlp) → Frame extraction (decord, 16–24 frames) → Cleanup

**Storage:** ~5–10 GB total. Manageable on any laptop.

---

## 2. Model Architecture — The Recommended Stack

**Key finding: Freeze everything except the fusion head. Start simple.**

The research agents converged on a clear tiered recommendation:

### 🥇 Tier 1: The Pragmatic Stack (start here)

| Component | Choice | Why |
|-----------|--------|-----|
| Text encoder | **ModernBERT-base** (frozen) | 8,192-token context handles long transcripts without chunking; 2–4x faster than DeBERTa |
| Vision backbone | **DINOv2 ViT-S/14** (frozen) | Best frozen features; only 21M params; reused for both thumbnails AND frames |
| Temporal | Same DINOv2 + small attention pooling | Only ~1–2M extra params; reuses vision backbone |
| Fusion | Late fusion (concatenate 3 projections → MLP) | ~3M trainable params total; easy to ablate; won't overfit on 2,000 samples |
| Attribution | Integrated Gradients via PyTorch Captum | Axiomatically grounded; model-agnostic; produces ranked feature importance |

**GPU:** Fits on any GPU with ≥6 GB VRAM (RTX 3060, T4, Colab). ~2–4 hours to train.

### 🥈 Tier 2 (if time permits): LoRA fine-tuning + 1 cross-attention layer
### 🥉 Tier 3 (only if weeks 1–2 go perfectly): TimeSformer for temporal

**Fallback if multimodal doesn't converge:** Text-only ModernBERT on title+description+transcript. Rajaram & Manchanda found text dominates over visuals for engagement prediction — a text-only model might get 80% of the way.

---

## 3. Evaluation & Planning — The 4-Week Timeline

### Metrics: AUC-ROC is the primary metric
- AUC-ROC is threshold-invariant, standard in comparable literature, works at 3:1 class ratio
- Report F1, precision, recall as supplementary
- Report honestly on the accuracy ceiling (external factors cap performance)
- **The delta between deep multimodal and metadata-only baselines is the real story — not the absolute AUC**

### Baselines: Logistic Regression + XGBoost on 12 metadata features
- Adapt Wu et al. (2018) codebase (feature extraction logic is reusable; needs Python 3.10+ update)
- Add a Dummy Classifier as the absolute floor

### Validation: 60/20/20 stratified split with channel grouping
- Prevent same-channel videos from leaking across splits
- 5-fold CV on training set for hyperparameter tuning
- Test set touched exactly ONCE at the end

### Four priority ablations:
1. Full model vs. metadata-only baselines (the central claim)
2. Full model vs. –Text (text is likely the strongest modality)
3. Full model vs. –All vision (does vision add anything?)
4. Full model vs. –Attribution (is the attribution layer pulling weight?)

---

## 4. The 4-Week Plan

| Week | Dates | Theme | Key Deliverable |
|------|-------|-------|----------------|
| **Week 1** | Aug 11–17 | Data collection + baselines | All 2,000 videos collected. LR + XGBoost baselines reporting AUC-ROC |
| **Week 2** | Aug 18–24 | Deep model implementation | Full multimodal model training. First complete run with loss curves |
| **Week 3** | Aug 25–31 | Tuning + ablations | Best model locked. All four ablations complete. Results drafted |
| **Week 4** | Sep 1–7 | Final evaluation + report | Test set evaluated once. Report + slides submitted |

### Critical path (what blocks everything else):
```
Data collection → Data preprocessing → Feature extraction → 
Baselines → Deep model → Ablations → Results → Report
```

### What can be parallelized:
- Adam and Emmanuel can split modalities (text vs. vision pipelines)
- Report writing can begin in Week 1 (lit review), not Week 4
- Baselines and deep model development can partially overlap

---

## 5. Realistic Targets

| Outcome | AUC-ROC | What it means |
|---------|---------|---------------|
| **Minimum viable** | > 0.65 (≥0.05 gap over baselines) | Beats metadata-only. Valid result. |
| **Good** | 0.70–0.78 | Clear multimodal advantage. Competitive with literature. |
| **Excellent** | > 0.78 | Strong evidence for the approach. Unlikely given 2,000 videos and external noise. |

---

## 6. What to do RIGHT NOW (today/tomorrow)

1. **Set up the Python environment** and GitHub repo
2. **Install yt-dlp** and test it on 5 YouTube videos — confirm you can extract metadata, captions, thumbnails
3. **Scan CC availability** in our 4 categories using the official API (100 free search calls/day)
4. **Decide the product reviews category question** — substitute or keyword-based?
5. **Read the three full research files** in this folder:
   - `data_retrieval.md` — detailed collection strategy
   - `model_architectures.md` — tiered architecture recommendations with citations
   - `evaluation_and_planning.md` — metrics, baselines, week-by-week breakdown
6. **Open `MILESTONES.md`** — that's your master to-do list. Start at the top.

---

## 7. Files in this research round

| File | What it covers |
|------|---------------|
| `data_retrieval.md` | YouTube API quota model, yt-dlp strategy, CC filtering, frame extraction, transcript handling, metadata features |
| `model_architectures.md` | Text encoder comparison, vision backbone selection, temporal transformer, fusion strategies, attribution methods, GPU estimates, tiered recommendations |
| `evaluation_and_planning.md` | Metrics for imbalanced classification, baseline implementation, validation strategy, ablation design, comparable projects, week-by-week plan, risk register, MVS definition |
| `MILESTONES.md` | Master to-do list with task owners, priorities, and exit criteria per week |
| `SUMMARY.md` | This file — executive summary |

---

_Generated by three parallel AI research agents on 2026-08-09. Full details and citations in the companion files above._
