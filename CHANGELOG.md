# CHANGELOG — DLDM Project

_Every change to this project must be logged here. Format: `- **[Author]** Description. (Files: ...)`_

---

### 2026-08-09

- **[Adam]** Initialized project structure: created folder hierarchy, `CLAUDE.md`, `AGENTS.md`, `CLAUDE-MAP.md`, `CHANGELOG.md`, and `.agents/skills/` with changelog and folder-summary skills. All folder READMEs stubbed. (Files: all root-level files and subfolder READMEs)
- **[AI (via Adam)]** Created data retrieval research document covering: YouTube Data API v3 quota model and collection strategy, CC licensing filtering and sampling, frame extraction tools (decord/OpenCV/ffmpeg), transcript handling (auto-captions + Whisper fallback), and metadata feature mapping. (Files: `01_Research/2026-08-09_initial_research/data_retrieval.md`)

### 2026-08-10

- **[AI (via Adam)]** Created comprehensive evaluation, baselines, and project planning research document. Covers: (1) metrics for imbalanced binary classification (AUC-ROC primary, F1/PR-AUC supplementary, accuracy ceiling strategies), (2) baseline models and Wu et al. (2018) codebase adaptability assessment, (3) validation strategy (60/20/20 stratified split, 5-fold CV, time-based split considerations), (4) ablation study design (remove-one with 4 priority ablations), (5) comparable projects and realistic performance targets (AUC-ROC 0.70–0.78), (6) detailed 4-week project plan with task assignments, critical path analysis, risk register, and minimum viable submission definition. (Files: `01_Research/2026-08-09_initial_research/evaluation_and_planning.md`)
- **[AI (via Adam)]** Created comprehensive model architectures research document. Covers: (1) text encoder comparison (BERT/RoBERTa/DistilBERT/ModernBERT — recommended: ModernBERT-base for its 8K context, 2–4x speed, and Pareto improvements), (2) vision backbone comparison (ResNet/ViT/CLIP/DINOv2 — recommended: DINOv2 ViT-S/14 frozen), (3) temporal transformer options (YouTube-8M pooling, TimeSformer, VideoMAE, per-frame DINOv2 + aggregator — recommended: shared frozen DINOv2 + attention pooling), (4) fusion strategy (early/mid/late — recommended: late fusion with projection layers), (5) attribution/explainability (attention weights, Integrated Gradients via Captum, SHAP, LIME — recommended: Integrated Gradients as primary), (6) practical GPU memory (~3–5 GB frozen, ~6–8 GB fine-tuned) and 4-week timeline feasibility. Includes 3-tier recommendation stack (pragmatic → moderate → ambitious) with explicit scaling-down flags and fallback options. (Files: `01_Research/2026-08-09_initial_research/model_architectures.md`)
- **[Adam]** Created master milestones to-do list (`MILESTONES.md`) and executive summary (`SUMMARY.md`) synthesizing all three parallel research agents. Milestones organized by week with P0/P1/P2 priorities, task owners, and exit criteria. Includes immediate pre-week-1 actions, minimum viable submission definition, and risk watchlist. (Files: `01_Research/2026-08-09_initial_research/MILESTONES.md`, `01_Research/2026-08-09_initial_research/SUMMARY.md`, `01_Research/README.md`)

### 2026-08-14

- **[AI (via Adam)]** Initialized the project as a git repo and pushed to GitHub (`Adamex-SVK/yt-diag`, private), completing the MILESTONES.md pre-Week-1 task "Create GitHub repo for code." Added `.gitignore` scoped for large data/model artifacts under `02_Data`, `03_Models`, `04_Experiments`. (Files: `.gitignore`)
