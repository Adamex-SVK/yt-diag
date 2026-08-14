# 00_Project_Brief — README

## Purpose
Course requirements, problem statement, grading rubric, and deliverable specifications for the DLDM project.

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `DLDM_Final_Project_Proposal.pdf` | Submitted project proposal — YT-Diag system | **done** |

## Project summary (from proposal)

**YT-Diag**: given any public YouTube video, predict its likely performance within its content niche and produce a ranked, human-readable explanation of the content features most likely responsible for underperformance.

- **Three modalities fused**: text transformer (title/description/transcript), vision backbone (thumbnail), temporal transformer (sampled frames)
- **4 content categories**: comedy/skits, tutorials & how-to, vlogs, product reviews
- **~2,000 videos** via YouTube Data API v3
- **Baselines**: logistic regression + XGBoost on metadata only (adapted from Wu et al. 2018)
- **Key challenge**: frame sampling limited to CC-licensed videos (YouTube TOS)

See `CLAUDE.md` for full project context.

## Status
- **Proposal**: submitted
- **Next**: data collection via YouTube API, literature review, baseline implementation

## Notes
- Extract grading rubric and deliverable deadlines here once available from the course.
- Keep the proposal PDF as the authoritative reference for what we committed to.
