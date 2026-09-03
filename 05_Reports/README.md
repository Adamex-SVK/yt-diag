# 05_Reports — README

## Purpose
Project deliverables: final paper, presentation slides, figures, and supplementary materials.

## Expected deliverables

- **Final paper/report** — describing YT-Diag: problem, method, experiments, results, limitations
- **Presentation slides** — for course presentation
- **Figures** — architecture diagram, results charts, attribution examples, confusion matrices
- **Supplementary materials** — anything beyond the page limit

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `final_report/` | **The report.** AAAI two-column, single-file `main.tex` (the style forbids `\input`, so do not split it into sections). `claim_evidence.md` maps every number to a reproducing command; publisher scripts insert baselines, Tier-1, and text-v2 results. Draft markers vanish when `\drafttrue` becomes `\draftfalse`. | Active draft; validation results populated |
| `paper/` | Draft and final versions of the project paper | superseded by `final_report/` |
| `presentation/` | Slides for course presentation | TBD |
| `figures/` | Exported publication-quality figures | TBD |

## Status
- **All deliverables**: not started

## Notes
- Keep drafts versioned (`paper_v1.md`, `paper_v2.md`) — better yet, use the GitHub repo for version control.
- Check grading rubric (in `00_Project_Brief/`) before finalizing.
- Honesty about limitations (TOS constraint, accuracy ceiling) is explicitly committed to in the proposal.
