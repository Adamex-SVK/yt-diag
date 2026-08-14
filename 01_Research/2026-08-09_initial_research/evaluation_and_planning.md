# Evaluation, Baselines & Project Planning — YT-Diag

_Research date: 2026-08-10. Sources: Wu et al. (2018) GitHub repo, scikit-learn docs, machinelearningmastery.com, Rajaram & Manchanda (2020), YouTube Data API docs, prior data retrieval research._

---

## 1. Metrics for Imbalanced Binary Classification

### 1.1 The Problem

YT-Diag labels "viral" = top quartile within category, "typical" = bottom three quartiles. This creates a **~1:3 class ratio** (roughly 25% positive, 75% negative). While not "severely" imbalanced (not 1:100 or worse), this ratio is enough to make **accuracy a misleading metric**: a dummy classifier predicting "typical" for every video would achieve 75% accuracy.

### 1.2 Recommended Primary Metrics

**For model selection and hyperparameter tuning, use AUC-ROC as the primary metric.** Rationale:

| Metric | Pros for YT-Diag | Cons |
|--------|-----------------|------|
| **AUC-ROC** | Threshold-invariant; summarizes ranking quality across all thresholds; standard in comparable literature (Rajaram & Manchanda 2020); works well at 3:1 ratio | Can be optimistic with very small minority classes (not our case at ~500 positives per category) |
| **F1-score** | Single interpretable number; balances precision and recall; commonly reported | Threshold-dependent; needs a fixed threshold choice |
| **Precision-Recall AUC** | Focuses on the positive (viral) class; more informative than ROC AUC when the positive class is rare | At 3:1 ratio, ROC AUC and PR AUC are both valid; PR AUC becomes more important at more extreme ratios |
| **Balanced Accuracy** | Accounts for class imbalance; equal to (sensitivity + specificity)/2 | Less standard in the literature; not threshold-invariant |

**Recommendation**: Report **AUC-ROC as the primary metric** for model comparison, supplemented by **F1-score, precision, and recall at the optimal threshold** found on the validation set. Also report a **confusion matrix** and **per-category breakdown** to show where the model succeeds and fails.

### 1.3 Handling the Accuracy Ceiling

External factors (platform promotion, algorithmic recommendations, luck/timing) place a hard ceiling on achievable accuracy. The proposal already acknowledges this. Concrete strategies:

1. **Report a "no-skill" baseline**: Always predicting the majority class ("typical") yields 75% accuracy, 0.5 AUC-ROC. Any model must beat these convincingly.
2. **Report an "upper bound" estimate**: Use human inter-annotator agreement on a small labeled subset (e.g., can two humans agree on whether a video "should have" gone viral based on content alone?). If humans agree 80% of the time, that's a plausible ceiling.
3. **Frame as improvement over metadata-only baselines**: The key claim isn't "we predict virality perfectly" — it's "deep multimodal signal adds predictive value beyond what structured metadata alone can capture." The gap between the deep model and XGBoost/logistic regression on metadata is the real finding.
4. **Honest limitations section**: Explicitly list what the model cannot capture (algorithm changes, platform promotion, external events, creator's existing audience dynamics).

### 1.4 What Wu et al. (2018) Uses

Wu et al. (2018) is primarily a **regression** problem: they predict engagement metrics (watch time, view count) rather than a binary viral/typical label. Their codebase uses:

- **Mean Absolute Error (MAE)** and **Symmetric Mean Absolute Percentage Error (SMAPE)** for regression evaluation
- Their predictors output continuous engagement scores, not class labels
- Their "relative engagement" metric (views normalized by channel size and time) is conceptually closest to our viral/typical label

**Key adaptation note**: Wu et al.'s prediction task is regression, ours is classification. We can adapt their feature engineering pipeline (metadata features, temporal patterns) but need our own classification evaluation framework. The structure of their `engagement_prediction/` scripts (feature extraction → train → evaluate) is reusable, but the output layer and loss function must change from regression to binary cross-entropy.

Sources: Wu et al. (2018) GitHub repo (`avalanchesiqi/youtube-engagement`); scikit-learn Metrics and Scoring documentation; machinelearningmastery.com "Tour of Evaluation Metrics for Imbalanced Classification."

---

## 2. Baselines

### 2.1 What Wu et al. Codebase Provides

The `engagement_prediction/` directory in the Wu et al. repo includes:

| Component | What it does | Adaptability |
|-----------|-------------|-------------|
| `run_all_predictors.sh` | Orchestrates feature extraction, training, evaluation | Framework is reusable; Python version (3.6) needs updating |
| Feature engineering | Extracts ~30 features from video metadata: duration, definition (HD/SD), category, day-of-week, hour-of-upload, title length, description length, channel age, channel total views, etc. | **Directly reusable** — these are the 12 structured metadata features we already planned to use |
| Regression models | Linear regression, Random Forest, Gradient Boosting (likely via scikit-learn ~2017) | Must convert to classification; XGBoost is the modern equivalent |
| Temporal analysis | Engagement decay curves, watch-time patterns | Useful for understanding the data but not directly for our baselines |

**Adaptation plan**: Extract the feature engineering logic from `wrangling/` and `engagement_prediction/`, update to Python 3.10+/modern scikit-learn, and add classification wrappers. The 12 structured features for our baselines map directly to what Wu et al. extract — this is why we cited them as the baseline source.

### 2.2 Baseline Models (Priority Order)

| # | Model | Rationale | Effort |
|---|-------|-----------|--------|
| 1 | **Logistic Regression** | Simplest linear baseline; interpretable coefficients show which metadata features matter most; required by proposal | Very low |
| 2 | **XGBoost** | Tree-based, handles non-linear interactions, built-in feature importance; required by proposal | Low |
| 3 | **Random Forest** | Alternative tree-based model; less tuning-sensitive than XGBoost; useful for comparison | Low |
| 4 | **Dummy Classifier (stratified)** | Predicts "viral" 25% of the time randomly; the absolute floor any model must beat | Trivial |

### 2.3 Alternative/Supplementary Baselines

| Model | Why consider | Priority for 1-month timeline |
|-------|-------------|------------------------------|
| **MLP on metadata only** | Tests whether a simple neural network on the same 12 features beats trees — isolates "deep" signal from "multimodal" signal | Medium (easy to add once metadata pipeline is built) |
| **Text-only model** (fine-tuned BERT on title+description) | Ablation: what does text alone achieve? | Medium (Week 3 ablation) |
| **Thumbnail-only model** (fine-tuned ResNet/ViT) | Ablation: what does vision alone achieve? | Medium (Week 3 ablation) |
| **Gradient Boosting (sklearn)** | Comparison point to XGBoost; sometimes more stable on small data | Low |

### 2.4 Baseline Evaluation Protocol

All baselines must be evaluated on the **same train/val/test split** as the deep model. Report:

- AUC-ROC, F1, precision, recall
- Confusion matrix
- Feature importance (for tree-based models) — these are directly interpretable and valuable for the report

Sources: Wu et al. (2018) GitHub repo; scikit-learn documentation; XGBoost documentation.

---

## 3. Validation Strategy

### 3.1 Train/Val/Test Split Design

For ~2,000 videos across 4 categories (500 per category):

**Recommended split: 60/20/20 stratified by category and label**

| Split | Videos | Purpose |
|-------|--------|---------|
| Train | ~1,200 | Model training |
| Validation | ~400 | Hyperparameter tuning, early stopping, threshold selection |
| Test | ~400 | Final evaluation — used exactly once at the end |

**Why stratified?**
- Stratify by **category** to ensure each split has equal representation of all 4 categories (125 per category in validation, 125 in test)
- Stratify by **label** (viral/typical) to maintain the ~1:3 ratio in every split

### 3.2 Cross-Validation

For 2,000 videos, **5-fold stratified cross-validation on the training set** is affordable and recommended:

- Train on 960, validate on 240, repeat 5×
- Report mean ± std of AUC-ROC across folds
- Use this for hyperparameter tuning (grid search or Optuna)

**Do not** cross-validate the test set — the test set is touched exactly once at the very end.

### 3.3 Ensuring Real-World Representativeness

| Risk | Mitigation |
|------|-----------|
| Temporal leakage (training on future videos, testing on old ones) | Split **by video**, not randomly. Consider a time-based split: train on videos uploaded before a cutoff date, test on later videos — this better simulates real deployment. |
| Category imbalance in splits | Stratify by category (see above) |
| Channel contamination (videos from the same channel in both train and test) | Group videos by `channelId`; ensure all videos from one channel go to the same split. This prevents the model from learning channel-specific patterns and overfitting. |
| Label definition sensitivity | Test robustness by varying the "viral" threshold: try top 20% and top 30% in addition to top 25%. Report whether conclusions hold. |

**Recommendation**: Use a **time-based split** if upload dates are available. If not, use random stratified split with channel grouping.

Sources: scikit-learn `StratifiedKFold`, `train_test_split`; general ML best practices.

---

## 4. Ablation Studies

### 4.1 Goal

Convincingly demonstrate that **each modality contributes predictive value beyond metadata alone** and beyond the other modalities.

### 4.2 Recommended Ablation Design: "Remove-One"

Starting from the full multimodal model, remove one modality at a time and measure the drop in AUC-ROC:

| Model Variant | Modalities Active | Hypothesis |
|---------------|-------------------|------------|
| **Full model** | Metadata + Text + Thumbnail + Temporal frames | Best performance |
| **– Temporal** | Metadata + Text + Thumbnail | Temporal frames add signal beyond static thumbnail |
| **– Thumbnail** | Metadata + Text + Temporal | Thumbnail is the most concentrated visual signal; does temporal compensate? |
| **– Text** | Metadata + Thumbnail + Temporal | Text (title + description + transcript) is the richest signal — expect largest drop |
| **– All vision** | Metadata + Text | How much do visual modalities contribute over text + metadata? |
| **– All deep (metadata only)** | Metadata (12 features) | Is the deep multimodal model better than XGBoost on the same features? This is the KEY comparison. |

### 4.3 Attribution Layer Ablation

The proposal mentions an "attention- or ablation-based attribution layer" inspired by Rajaram & Manchanda (2020). To ablate this:

1. **Train with attribution module → train without it**: Replace attention-weighted fusion with simple concatenation + MLP. Compare AUC-ROC.
2. **Compare attention weights to feature importance from XGBoost**: Do the attention weights surface the same features as important? Agreement with XGBoost feature importance adds credibility.
3. **Human evaluation**: On 10–20 test videos, show the top-3 features surfaced by the attribution layer. Do they make intuitive sense to a human? Qualitative sanity check.

### 4.4 Minimum Ablation Set for 1-Month Timeline

Given time constraints, prioritize these **four ablations** (in order):

| Priority | Ablation | Why non-negotiable |
|----------|----------|-------------------|
| **1** | Full model vs. metadata-only (XGBoost/LR) | This IS the central claim of the paper |
| **2** | Full model vs. –Text | Text is likely the strongest modality (Rajaram & Manchanda found "what is said" dominates over visuals) |
| **3** | Full model vs. –All vision (thumbnail + temporal) | Tests whether vision adds anything at all |
| **4** | Full model vs. –Attribution (simple concat fusion) | Tests whether the attribution mechanism is pulling weight or just ornament |

Additional ablations (temporal-only removal, thumbnail-only removal) are nice-to-have if time permits but not required for a credible submission.

Sources: Rajaram & Manchanda (2020) "Unboxing Engagement in YouTube Influencer Videos"; general ML ablation methodology.

---

## 5. Comparable Projects & Realistic Targets

### 5.1 Related Systems

| Project | Task | Data Size | Modalities | Performance | Relevance |
|---------|------|-----------|------------|-------------|-----------|
| **Wu et al. (2018)** | Regression: predict engagement (watch time, views) | 5.3M tweeted videos | Metadata only (30+ features) | SMAPE ~0.3–0.4 depending on model; R² not reported directly in repo | Baseline source; regression task, not classification |
| **Rajaram & Manchanda (2020)** | Classification/regression: predict engagement of influencer videos | Not specified in abstract | Text (transcript), audio, video images (frames) | "Strong out-of-sample prediction"; specific AUC not in abstract | **Closest comparable system** — multimodal, attention-based, YouTube videos. Text > visuals > audio for engagement prediction |
| **YouTube-8M (Abu-El-Haija et al. 2016)** | Video classification benchmark | 8M videos, 4,800 classes | Visual + audio features (pre-extracted) | State-of-the-art at time: ~80–85% mAP depending on architecture | Only tangentially relevant — classification of video topic, not virality prediction |

### 5.2 What Accuracy/AUC Do They Achieve?

- **Wu et al. (2018)**: Reports MAE and SMAPE for regression, not classification metrics. Their best model (Gradient Boosting Regression) achieves SMAPE of ~0.30–0.35 on watch-time prediction. Not directly comparable to our AUC-ROC.
- **Rajaram & Manchanda (2020)**: The abstract says "strong out-of-sample prediction" but the specific AUC/accuracy numbers are in the full 50-page paper (not fetched). Based on their methodology, a multimodal attention model on influencer videos, expect AUC-ROC in the **0.70–0.85 range** depending on the engagement metric.
- **General literature**: Virality prediction is known to be hard. Papers reporting >0.90 AUC-ROC typically have a looser definition of "viral" or use post-hoc features (early-view velocity) that wouldn't be available at upload time.

### 5.3 Realistic Target for YT-Diag

| Target | What it means | Confidence |
|--------|---------------|------------|
| **Minimum viable: AUC-ROC > 0.65** | Beats metadata-only baseline by a meaningful margin (≥0.05 AUC improvement) | High — even a modest deep model should beat linear classifiers on rich multimodal data |
| **Good result: AUC-ROC 0.70–0.78** | Clear improvement over baselines; multimodal fusion works | Medium — achievable with careful training, but not guaranteed at 2,000 videos |
| **Excellent result: AUC-ROC > 0.78** | Strong evidence for multimodal approach; competitive with published literature | Low — limited by 2,000-video dataset size and external noise ceiling |

**Crucially, the absolute number matters less than the gap between the deep multimodal model and the metadata-only baselines.** A paper showing AUC-ROC of 0.70 for the full model vs. 0.62 for XGBoost is more convincing than one showing 0.82 vs. 0.80. The **delta** is the story.

### 5.4 Previous DLDM Course Benchmarks

No previous DLDM course project benchmarks are available in the project folder or online. **Action: ask the course instructors/TAs about typical grade boundaries for project metrics** — they may have informal expectations (e.g., "beating baselines is sufficient for a passing grade, novelty + strong results for top marks"). This should be done in Week 1.

Sources: Wu et al. (2018) GitHub repo and paper; Rajaram & Manchanda (2020) arXiv abstract; Abu-El-Haija et al. (2016) YouTube-8M paper.

---

## 6. Project Plan & Milestones (4 Weeks, 2 Students)

### 6.1 Critical Path Analysis

The **critical path** (longest chain of dependent tasks) is:

```
Data collection → Data preprocessing → Feature extraction pipeline → 
Baseline implementation → Deep model implementation → Training & tuning → 
Ablation studies → Results analysis → Report writing
```

**Parallelizable work**:
- Adam and Emmanuel can work on different modalities simultaneously (text vs. vision pipelines)
- Baseline implementation can start as soon as metadata features are extracted
- Report writing (literature review, introduction, methodology) can begin in Week 1 alongside technical work
- Presentation slides can be drafted while final experiments run

### 6.2 Week 1 (Aug 11–17): Data Collection & Pipeline Foundations

**Goal**: All 2,000 videos collected and preprocessed. Metadata features extracted. Baselines running.

| Priority | Task | Owner | Est. Hours | Depends On |
|----------|------|-------|------------|------------|
| **P0** | Set up Python environment (conda/pip, requirements.txt) and GitHub repo | Adam | 2 | Nothing |
| **P0** | Write data collection script using yt-dlp: collect ~2,000 video IDs across 4 categories | Emmanuel | 4 | Nothing |
| **P0** | Download metadata (title, description, view count, likes, comments, duration, upload date, channel subs, category) | Emmanuel | 3 | Collection script |
| **P0** | Download thumbnails (1280×720 JPG) | Emmanuel | 1 | Collection script |
| **P0** | Download auto-captions / transcripts | Adam | 3 | Collection script |
| **P0** | Download CC-licensed videos for frame extraction (~500 of the 2,000) | Adam | 2 | Collection script |
| **P1** | Extract 16–24 frames per video using decord | Adam | 3 | Downloaded videos |
| **P1** | Compute "viral" label: top quartile of views-per-day-since-upload normalized by channel subscriber count, within each category | Emmanuel | 2 | Metadata collected |
| **P1** | Extract 12 structured metadata features (Wu et al. feature set) | Emmanuel | 3 | Metadata collected |
| **P1** | Implement train/val/test split (60/20/20, stratified by category + label) | Adam | 1 | Labels computed |
| **P2** | Implement logistic regression baseline on metadata | Emmanuel | 2 | Metadata features + split |
| **P2** | Implement XGBoost baseline on metadata | Emmanuel | 2 | Metadata features + split |
| **P2** | Write EDA notebook: class distribution, category breakdown, feature correlations, missing data report | Adam | 2 | Metadata collected |
| **P3** | Begin literature review notes for report introduction/related work | Either | 2 | Nothing |

**Week 1 exit criteria**: All data collected and cleaned. Labels computed. Train/val/test splits ready. LR and XGBoost baselines running with AUC-ROC reported. EDA notebook complete.

**⚠️ Risk: yt-dlp rate limiting or YouTube blocking.** Mitigation: start collection on Day 1. If blocked, use official API with multiple keys as fallback (see data_retrieval.md Section 1.5).

### 6.3 Week 2 (Aug 18–24): Deep Model Implementation

**Goal**: Full multimodal model implemented and training. First training run complete.

| Priority | Task | Owner | Est. Hours | Depends On |
|----------|------|-------|------------|------------|
| **P0** | Text pipeline: tokenize titles + descriptions + transcripts; feed through pre-trained transformer (DistilBERT or MiniLM for efficiency) | Adam | 5 | Transcripts downloaded |
| **P0** | Vision pipeline (thumbnail): pre-trained ResNet-50 or ViT-B/16, extract embedding | Emmanuel | 3 | Thumbnails downloaded |
| **P0** | Temporal pipeline: encode 16–24 frames through same vision backbone, temporal aggregation (mean pooling or lightweight transformer) | Emmanuel | 4 | Frames extracted |
| **P0** | Metadata pipeline: MLP on 12 structured features → embedding vector | Adam | 2 | Metadata features extracted |
| **P1** | Fusion module: cross-attention or concatenation of all modality embeddings → classification head | Adam + Emmanuel (pair) | 3 | All modality pipelines |
| **P1** | Attribution module: attention-based feature importance layer (inspired by Rajaram & Manchanda 2020) | Adam | 3 | Fusion module |
| **P1** | Training loop: DataLoader, loss function (binary cross-entropy), optimizer (AdamW), learning rate schedule, early stopping | Emmanuel | 3 | Full model defined |
| **P1** | First full training run — debug, check for NaN losses, overfitting, GPU memory issues | Both | 2 | Training loop |
| **P2** | Set up experiment tracking (Weights & Biases or TensorBoard) | Adam | 1 | Training loop |
| **P3** | Start drafting report methodology section | Either | 2 | Architecture defined |

**Week 2 exit criteria**: Full multimodal model trains without errors. One complete training run with loss curves showing convergence. No NaN losses or obvious bugs.

**⚠️ Risk: GPU memory issues with multiple pre-trained models.** Mitigation: use smaller backbones (DistilBERT not BERT-large; ResNet-18 not ResNet-152). Use gradient checkpointing. Process modalities in separate forward passes if needed.

### 6.4 Week 3 (Aug 25–31): Hyperparameter Tuning & Ablations

**Goal**: Best model found. All four priority ablation studies complete.

| Priority | Task | Owner | Est. Hours | Depends On |
|----------|------|-------|------------|------------|
| **P0** | Hyperparameter tuning: learning rate, batch size, dropout, fusion dimension, number of temporal frames | Adam | 5 | Working training loop |
| **P0** | Train final best model on full training set, evaluate on validation set, select threshold | Emmanuel | 2 | Tuned hyperparams |
| **P1** | Ablation 1: Full model vs. metadata-only (XGBoost + LR) — already have baseline numbers from Week 1, just need to finalize comparison table | Emmanuel | 1 | Baselines + best model |
| **P1** | Ablation 2: Full model vs. –Text (remove text modality, retrain) | Adam | 3 | Best model config |
| **P1** | Ablation 3: Full model vs. –All vision (remove thumbnail + temporal, retrain) | Adam | 3 | Best model config |
| **P1** | Ablation 4: Full model vs. –Attribution (replace attention fusion with simple concat, retrain) | Emmanuel | 3 | Best model config |
| **P2** | Additional ablations if time: –Temporal only, –Thumbnail only | Either | 3 | P1 ablations done |
| **P2** | Qualitative analysis: select 10–20 test videos, run attribution, manually inspect whether surfaced features make sense | Both | 2 | Attribution module |
| **P3** | Draft results section, generate all figures and tables | Either | 3 | Ablations done |
| **P3** | Begin drafting discussion/conclusion | Either | 1 | Results taking shape |

**Week 3 exit criteria**: Best model selected and locked. All four priority ablations complete with results tables. No more model changes after Sunday — everything from here is analysis and writing.

**⚠️ Risk: Ablation results are inconclusive or negative (deep model doesn't beat baselines).** Mitigation: this is still a valid paper — "we hypothesized multimodal signal would help; it didn't; here's our analysis of why" is honest science. Frame as a negative result with careful analysis. Do NOT cherry-pick or p-hack.

### 6.5 Week 4 (Sep 1–7): Final Evaluation, Report & Presentation

**Goal**: Test set evaluated exactly once. Report and presentation submitted.

| Priority | Task | Owner | Est. Hours | Depends On |
|----------|------|-------|------------|------------|
| **P0** | **FINAL test set evaluation** — run exactly once, record all metrics, lock numbers | Both | 1 | Best model locked |
| **P0** | Write results section with all tables and figures | Adam | 4 | Test results + ablation results |
| **P0** | Write discussion: limitations (accuracy ceiling, CC-only frame limitation, external factors), interpretation of ablation findings | Emmanuel | 3 | Results written |
| **P0** | Write methodology section (architecture diagrams, training details, hyperparameters) | Adam | 3 | Architecture defined |
| **P0** | Write introduction + related work (review of Wu et al., Rajaram & Manchanda, YouTube-8M) | Emmanuel | 3 | Literature review notes |
| **P0** | Write abstract + conclusion | Both (pair) | 1 | Everything else written |
| **P1** | Create presentation slides (10–12 slides: problem, approach, architecture, results, ablation, limitations) | Both | 4 | Report draft |
| **P1** | Rehearse presentation, time it | Both | 2 | Slides done |
| **P2** | Format references, proofread, final formatting pass | Either | 2 | Report complete |
| **P2** | Submit report + slides + code (with README reproduction instructions) | Adam | 1 | Everything done |

**Week 4 exit criteria**: Report submitted. Presentation ready. Code repository clean with README.

**⚠️ Risk: Writing takes longer than expected.** Mitigation: start writing in Week 1 (literature review), Week 2 (methodology), and Week 3 (results). Week 4 should be polishing, not drafting from scratch.

### 6.6 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Data collection blocked/throttled by YouTube | Medium | High | Use multiple API keys; start Day 1; have fallback plan for smaller dataset |
| GPU unavailability or quota limits | Low | High | Use Google Colab Pro/GPU; use smaller model variants; reduce batch size |
| Deep model doesn't converge | Medium | Medium | Simplify architecture; reduce modalities; use stronger regularization |
| Deep model doesn't beat baselines | Medium | Medium | Honest negative result is still valid; analyze why; focus report on what was learned |
| Team member unavailable (illness, etc.) | Low | High | Cross-train: both should understand the full pipeline, not just their half |
| 2,000 videos insufficient for deep learning | Medium | Medium | Use heavy data augmentation (especially for frames); use pre-trained backbones; acknowledge limitation |
| CC-licensed video scarcity in target categories | Medium | Medium | Adjust categories if needed (see data_retrieval.md Section 2.2); disclose limitation |

### 6.7 Minimum Viable Submission (MVS)

If time runs short, the **absolute minimum** for a passing submission is:

1. **Data**: At least 500 videos across ≥2 categories, with all modalities collected
2. **Baselines**: Logistic regression AND XGBoost on metadata, with AUC-ROC reported
3. **Deep model**: At least TWO modalities fused (e.g., text + metadata), trained and evaluated
4. **Ablation**: At least ONE ablation showing deep model vs. metadata-only baseline
5. **Report**: Introduction, related work, methodology, results (with tables), limitations, conclusion
6. **Code**: Runnable with README reproduction instructions

Anything beyond this (3rd/4th modality, attribution layer, all ablations, polished presentation) improves the grade but the MVS is achievable even with major setbacks.

---

## References

- Wu, S., Rizoiu, M.-A., & Xie, L. (2018). Beyond Views: Measuring and Predicting Engagement in Online Videos. _ICWSM 2018_. Code: https://github.com/avalanchesiqi/youtube-engagement
- Rajaram, P., & Manchanda, P. (2020). Unboxing Engagement in YouTube Influencer Videos: An Attention-Based Approach. _arXiv:2012.12311_.
- Abu-El-Haija, S., et al. (2016). YouTube-8M: A Large-Scale Video Classification Benchmark. _arXiv:1609.08675_.
- Scikit-learn Developers. (2026). Metrics and scoring: quantifying the quality of predictions. https://scikit-learn.org/stable/modules/model_evaluation.html
- Brownlee, J. (2021). Tour of Evaluation Metrics for Imbalanced Classification. Machine Learning Mastery. https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

---

_Last updated: 2026-08-10. Update as experiments reveal new constraints or findings._
