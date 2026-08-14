# YT-Diag — Milestones & To-Do List

_Generated 2026-08-09 from three parallel research agents. This is the team's master task list. Update as tasks are completed or new priorities emerge._

---

## How to use this

- Tasks are in **priority order within each week** — work top-to-bottom
- **P0** = must do this week or everything stalls. **P1** = important but can slip a day. **P2** = nice-to-have.
- After completing a task, mark it `[x]` and log it in `CHANGELOG.md`

---

## 🚨 Immediate (before Week 1)

| # | Task | Owner | Why it's urgent |
|---|------|-------|-----------------|
| 1 | [ ] Set up Python environment (conda/pip, `requirements.txt`) | Adam | Everything depends on this |
| 2 | [ ] Create GitHub repo for code | Adam | Code needs a home |
| 3 | [ ] Install yt-dlp as Python library — test on 5 videos | Emmanuel | Confirms collection method works |
| 4 | [ ] Scan CC availability per category using API search (100 calls/day free) | Emmanuel | Critical go/no-go: if CC videos are too sparse, we need to adjust categories NOW |

---

## Week 1 (Aug 11–17): Data Collection & Pipeline Foundations

**Goal**: All ~2,000 videos collected and preprocessed. Baselines running.

### P0 — Must Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 1 | [ ] Write data collection script using **yt-dlp Python API**: collect ~500 CC-licensed video IDs per category | Emmanuel | 4 |
| 2 | [ ] Download metadata via yt-dlp: title, description, view/like/comment counts, duration, upload date, channel subs, category, tags | Emmanuel | 3 |
| 3 | [ ] Download thumbnails (1280×720 JPG) | Emmanuel | 1 |
| 4 | [ ] Download auto-captions / transcripts via yt-dlp `--write-auto-subs` | Adam | 3 |
| 5 | [ ] Download CC-licensed videos for frame extraction (~500 of 2,000) | Adam | 2 |
| 6 | [ ] Extract 16–24 frames per video using **decord** (uniform sampling, hybrid: 8 from first 60s + 8 across rest) | Adam | 3 |
| 7 | [ ] Compute "viral" label: top quartile views-per-day-since-upload, normalized by channel sub count, within each category | Emmanuel | 2 |
| 8 | [ ] Extract 12 structured metadata features (Wu et al. feature set) | Emmanuel | 3 |
| 9 | [ ] Implement train/val/test split: 60/20/20, stratified by category + label, channel grouping to prevent leakage | Adam | 1 |
| 10 | [ ] Implement **Logistic Regression baseline** on 12 metadata features → report AUC-ROC | Emmanuel | 2 |
| 11 | [ ] Implement **XGBoost baseline** on 12 metadata features → report AUC-ROC | Emmanuel | 2 |

### P1 — Should Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 12 | [ ] Write EDA notebook: class distribution, category breakdown, feature correlations, missing data report | Adam | 2 |
| 13 | [ ] Quality-check captions: word count heuristic → flag poor-quality ones for Whisper fallback | Adam | 1 |
| 14 | [ ] Set up experiment tracking (Weights & Biases or TensorBoard) | Adam | 1 |

### P2 — Nice to Have

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 15 | [ ] Begin literature review notes for report introduction/related work | Either | 2 |
| 16 | [ ] Set up Whisper `small.en` — install, test on 2–3 videos with poor captions | Adam | 1 |

### Week 1 Exit Criteria

- [ ] All data collected and cleaned
- [ ] Labels computed (viral/typical)
- [ ] Train/val/test splits ready, no channel contamination
- [ ] LR and XGBoost baselines running with AUC-ROC reported
- [ ] EDA notebook complete

---

## Week 2 (Aug 18–24): Deep Model Implementation

**Goal**: Full multimodal model implemented and training. First training run complete.

### P0 — Must Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 1 | [ ] **Text pipeline**: tokenize titles + descriptions + transcripts → feed through **ModernBERT-base** (frozen) → 256-dim projection | Adam | 5 |
| 2 | [ ] **Thumbnail pipeline**: encode thumbnails through **DINOv2 ViT-S/14** (frozen) → 256-dim projection | Emmanuel | 3 |
| 3 | [ ] **Temporal pipeline**: encode 16–24 frames through same DINOv2 ViT-S/14 (frozen) → attention pooling → 256-dim projection | Emmanuel | 4 |
| 4 | [ ] **Fusion module**: concatenate three 256-dim embeddings → 2-layer MLP → binary logit (late fusion) | Both (pair) | 3 |
| 5 | [ ] **Training loop**: DataLoader, BCE loss, AdamW optimizer, LR schedule, early stopping | Emmanuel | 3 |
| 6 | [ ] **First training run** — debug NaN losses, overfitting, GPU memory | Both | 2 |

### P1 — Should Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 7 | [ ] **Attribution module**: Integrated Gradients via PyTorch Captum on the fused model | Adam | 3 |
| 8 | [ ] Pre-compute and cache all frame embeddings (avoid on-the-fly encoding during training) | Emmanuel | 1 |
| 9 | [ ] Experiment: try training with and without temporal modality, compare validation AUC | Either | 1 |

### P2 — Nice to Have

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 10 | [ ] Start drafting report methodology section (architecture diagram, training details) | Either | 2 |
| 11 | [ ] Try **hybrid fusion** (add 1 cross-attention layer before concat) — only if late fusion works | Adam | 2 |

### Week 2 Exit Criteria

- [ ] Full multimodal model trains without errors
- [ ] Loss curves show convergence (no NaN, no wild oscillation)
- [ ] At least one complete training run with validation AUC reported

---

## Week 3 (Aug 25–31): Tuning & Ablations

**Goal**: Best model found. All four priority ablations complete. No more model changes after Sunday.

### P0 — Must Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 1 | [ ] Hyperparameter sweep: learning rate, batch size, dropout, fusion dim, frame count → best config on validation | Adam | 5 |
| 2 | [ ] Train final best model on full training set, evaluate on validation, select classification threshold | Emmanuel | 2 |
| 3 | [ ] **Ablation 1**: Full model vs. metadata-only (XGBoost + LR) — compare AUC-ROC, F1, precision/recall | Emmanuel | 1 |
| 4 | [ ] **Ablation 2**: Full model vs. –Text (remove text modality, retrain) | Adam | 3 |
| 5 | [ ] **Ablation 3**: Full model vs. –All vision (remove thumbnail + temporal, retrain) | Adam | 3 |
| 6 | [ ] **Ablation 4**: Full model vs. –Attribution (replace attention with simple concat, retrain) | Emmanuel | 3 |

### P1 — Should Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 7 | [ ] Qualitative attribution check: on 10–20 test videos, do top-3 surfaced features make intuitive sense? | Both | 2 |
| 8 | [ ] Draft results section: all tables, figures, ablation charts | Either | 3 |
| 9 | [ ] Draft discussion: limitations (accuracy ceiling, CC-only frames, external factors) | Either | 1 |

### Week 3 Exit Criteria

- [ ] Best model selected and locked (no more changes)
- [ ] All four priority ablations complete with results tables
- [ ] Results and discussion sections drafted

---

## Week 4 (Sep 1–7): Final Evaluation, Report & Submission

**Goal**: Test set evaluated exactly once. Report and presentation submitted.

### P0 — Must Do

| # | Task | Owner | Est. Hours |
|---|------|-------|------------|
| 1 | [ ] **FINAL test set evaluation** — run ONCE, record all metrics, lock numbers | Both | 1 |
| 2 | [ ] Write report: abstract, introduction, related work, methodology, results, discussion, conclusion | Both | 10 |
| 3 | [ ] Create presentation slides (10–12 slides) | Both | 4 |
| 4 | [ ] Rehearse presentation, time it, fix pacing | Both | 2 |
| 5 | [ ] Clean up code repo: README with reproduction instructions, requirements.txt, remove dead code | Adam | 2 |
| 6 | [ ] Submit: report + slides + code | Adam | 1 |

### Week 4 Exit Criteria

- [ ] Report submitted
- [ ] Presentation ready and rehearsed
- [ ] Code repository clean with README

---

## 🛡️ Minimum Viable Submission (if time runs short)

If we hit major setbacks, this is the absolute floor for a passing submission:

1. **Data**: ≥500 videos across ≥2 categories, all modalities collected
2. **Baselines**: LR + XGBoost on metadata, AUC-ROC reported
3. **Deep model**: ≥2 modalities fused (e.g., text + metadata), trained and evaluated
4. **Ablation**: ≥1 ablation showing deep vs. metadata-only baseline
5. **Report**: Intro, related work, methodology, results, limitations, conclusion
6. **Code**: Runnable with README

---

## ⚠️ Risk Watchlist

| Risk | Signal to watch | Mitigation |
|------|----------------|------------|
| CC videos too sparse | Immediate scan results show <300 per category | Drop to 3 categories, or accept non-CC metadata-only |
| yt-dlp blocked/throttled | Collection stalls after ~100 videos | Switch to official API with multiple keys |
| GPU unavailable | Can't train on Colab/TUM cluster | Use smaller models (DistilBERT, ResNet-18), reduce batch size |
| Model doesn't converge | Loss flatlines or oscillates after 10 epochs | Freeze ALL encoders, simplify to text-only, add regularization |
| Model doesn't beat baselines | AUC gap < 0.02 | Honest negative result is valid; analyze why; focus report on learnings |
| Team member unavailable | No commits for 3+ days | Cross-train on all parts; don't split into silos |

---

_Last updated: 2026-08-09. Check off tasks as completed. Update priorities if research reveals new constraints._
