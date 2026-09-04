# Update: Adam's session — 2026-09-04

**For**: Emmanuel
**From**: Adam (via AI)
**Nothing here touches `main` yet** — everything is on separate branches, pushed, waiting on review/merge.

## What changed

### 1. Fixed a real (but report-inert) leakage bug in `baselines.py`
Branch: `fix/baseline-threshold-leakage` (based on `feat/final-attribution-protocol`)

`run_baselines()` was picking the F1 decision threshold on the same validation rows it then reported F1 on — classic threshold leakage. Fixed by carving an inner train/val split out of the outer training fold (mirrors `fusion.py`'s `_inner_split`), picking the threshold there, then fitting the final model on the full outer-train fold and scoring on val/test as before.

**Confirmed this changes nothing already published**: `make_tables.py`, `run_tuned_baselines.py`, and `run_audio_ablation.py` all only ever read `auc_roc` from this function's output, never `f1` — so no number in the report moves. `test_pipeline.py` and `test_tuning.py` both pass. Done on its own branch specifically so it wouldn't collide with your `final-attribution-protocol` work in progress.

**Action for you**: review and merge whenever convenient — it's a correctness fix with zero downstream impact, not urgent.

### 2. Expanded Related Work with 5 new citations
Branch: `feat/final-attribution-protocol` (built on your existing commit `b1d13c4eb`)

Ran a literature search (Consensus AI) and reviewed ~25 candidate papers for genuine relevance (not keyword overlap) — 18 were screened out. Added 5 that fill real gaps:

- **Hessel, Lee & Mimno 2017** — ranks near-simultaneous Reddit posts to isolate content from timing/author confounds; closest existing precedent to our peer-relative label design.
- **Rosenblatt et al. 2024** (*Nature Communications*) — quantifies leakage from out-of-fold confound correction and grouped-sample splitting; anchors our channel-grouped protocol with an actual citation instead of just code comments.
- **Liu et al. 2026** — 2026 survey naming our exact frozen-encoder-to-fusion paradigm as the dominant approach in the field.
- **Wang et al. 2023** — same ResNet-50/XGBoost backbones we use; finds context features alone beat content modalities individually.
- **Lin & Lee 2024** — 2024 SMP Challenge winner; documents same-creator posts having highly correlated popularity, direct empirical justification for our channel-grouped splits.

Also added a short new **"Confound control and evaluation protocol"** subsection (cites Rosenblatt et al.), and **fixed the existing Rajaram & Manchanda (2020) sentence**, which mischaracterized the paper as combining "video and creator characteristics" — it actually fuses text/audio/visual signals via attention. Caught by cross-referencing our own `01_Research/2026-08-09_initial_research/model_architectures.md` notes.

**Heads up**: author given names on the 5 new bib entries were reconstructed from initials printed on the PDFs — plausible, not confirmed. Worth a 2-minute check against each paper's title page before final typesetting.

**Action for you**: review and merge — it's additive to your Related Work text, nothing existing was restructured.

### 3. Reviewed Methodology + Experiment sections for scientific accuracy (no edits)
Read on `feat/audio-fusion-block` (your audio-ablation branch, since that's the current tip).

**Clean bill of health.** Spot-checked essentially every reported number in Methodology and Experiment against its source JSON (`baselines.json`, `tuned_baselines.json`, `audio_ablation.json`, `attribution.json`) — the shortcut ceiling, all four baseline ladders, the tuned comparison, the audio-ablation percentages, the six-block attribution shares, the finalist table. Every one matched to the decimal place. The methodology is honest about its own limits (repeated-split-not-independent-test caveat, transductive label bands, subscriber-count reverse-causality) rather than overclaiming.

One tiny gap, not an error: the label formula (`rank_c(v)(s(v)) > |c(v)| - k_c(v)`) doesn't state a tie-breaking rule for videos with identical view counts in the same cell. Cosmetic — flag if you think it's worth a clause.

### 4. Cross-checked `claim_evidence.md` against the results JSONs — found 2 small issues **for you to fix**
These are in `claim_evidence.md` itself, not in `main.tex`, and I didn't touch your branch — noting them here instead:

- **Line 102**: says gain-based importance is "F0 pitch 15%" — the actual JSON value (`audio_ablation.json → gain_family_share → "pitch (F0) dynamics"`) is 14.47%, which rounds to **14%**. `main.tex` itself already has this right ("pitch (F0) dynamics (14%)") — just the evidence table row is off by one point.
- **Line 99**: "Attribution examples — pending (Integrated Gradients not implemented)" is stale. We pivoted to exact linear-block decomposition instead of IG; it's already marked `confirmed` two rows above (line 97), and `main.tex` already contains a qualitative seed-0 TP/FP example. This pending row should be resolved/removed, not left open.

## Still open (not blocked on me)

- **Merge order**: three branches now diverge from the same base (`b1d13c4eb`) — `fix/baseline-threshold-leakage`, `feat/final-attribution-protocol` (now includes the Related Work work), and `feat/audio-fusion-block`. Whoever merges first should probably go final-attribution-protocol → audio-fusion-block → the tiny baseline fix, but that's your call on ordering given the audio-ablation SIGNOFF note below.
- **Your SIGNOFF note in `main.tex`**: "does this warrant a follow-up freeze cycle, or does it stay a reported post-hoc finding?" — still open, needs both of us.
- **Repo visibility**: `github.com/Adamex-SVK/yt-diag` (cited in the report's Availability paragraph as a public link) currently 404s for an unauthenticated fetch — looks private. Needs to go public (or the availability text needs to change) before submission.
- **Frozen external test-set evaluation** — still pending on your end; nothing downstream of it (final proofread) can start until it lands.

## Not touched
`05_Reports/presentation/` (saw it appear as untracked while I was working — assumed it's yours, left it alone).
