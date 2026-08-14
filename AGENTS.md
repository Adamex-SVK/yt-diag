# AGENTS.md — AI Agent Instructions for DLDM Project

_Every AI session in this project MUST follow these rules. This file is the contract between the team (Adam & Emmanuel) and any AI assistant._

**Project**: YT-Diag — multimodal deep learning system for diagnosing YouTube video underperformance.
**Team**: Adam Michalik (go54lix@tum.de) & Emmanuel Gyabaah (emmanuel.gyabaah@tum.de).

---

## 1. Changelog Protocol (NON-NEGOTIABLE)

**Every change made to this project — by a human or an AI — must be logged in `CHANGELOG.md`.**

### When to log

- Adding a new file or folder
- Modifying an existing file (content change, not just formatting)
- Deleting a file
- Running an experiment (log the config, result, and where outputs went)
- Making a design decision
- Updating a README or this AGENTS.md itself

### What to log

Every entry in `CHANGELOG.md` must follow this format:

```markdown
### YYYY-MM-DD

- **[Author]** Brief description of what changed and why. (Files affected: `path/to/file`)
```

- **Author**: your name or "AI (via [your name])" if an AI did the work under your direction.
- **Why**: always include the reason — not just "added file X" but "added file X because we needed a place to track Y."
- **Files affected**: comma-separated relative paths from the project root.

### How to log

**If you're a human**: add the entry directly to `CHANGELOG.md`.

**If you're an AI agent**: after completing any change, append the entry to `CHANGELOG.md` BEFORE ending the session. If the session is purely read-only (question, summary, search), no log entry is needed.

---

## 2. Folder Summarization

### Every folder must have a `README.md`

Each `README.md` describes:
- **Purpose**: what this folder is for
- **Contents**: key files and what they are
- **Status**: what's done, what's in progress, what's TBD

AIs must keep these READMEs updated when they add/remove/change files in a folder.

### How to ask for a summary

Any teammate can ask: _"Summarize what's in folder X"_ or _"What's new since [date]?"_ or _"Give me the current state of the project."_

The AI should:
1. Read `CHANGELOG.md` for recent entries
2. Read the relevant `README.md` files
3. Produce a structured summary with: what's new, what's in progress, what's blocked

---

## 3. Project Structure Rules

- `00_Project_Brief/` — course requirements, problem statement, grading rubric. **Read this first.**
- `01_Research/` — literature review, related work, notes on papers. One `.md` per paper/topic.
- `02_Data/` — dataset documentation, preprocessing scripts, EDA. Actual data files live in the GitHub repo or are `.gitignore`'d.
- `03_Models/` — model architecture designs, training scripts, saved checkpoints (or pointers to them). One subfolder per model variant.
- `04_Experiments/` — experiment configs, run logs, results tables. One subfolder per experiment.
- `05_Reports/` — draft report, final paper, presentation slides, figures.
- `06_Meeting_Notes/` — one `.md` per meeting, named `YYYY-MM-DD_Meeting_Topic.md`.

---

## 4. Code & GitHub

- Code lives in the GitHub repo (once created). This folder tracks: briefs, research notes, experiment logs, meeting notes, reports.
- When the repo is created, add its URL to `CLAUDE.md`.
- Code-related decisions (architecture choices, hyperparameter changes) are logged in `CHANGELOG.md` with a reference to the relevant commit or PR.

---

## 5. AI Behavior Rules

1. **Always read `CLAUDE.md` first** — it's the project context.
2. **Always read this `AGENTS.md`** — it's the rules of engagement.
3. **Log every change** to `CHANGELOG.md` before ending a session that modified anything.
4. **Keep READMEs updated** — if you add a file to a folder, update its `README.md`.
5. **Be explicit about uncertainty** — if the project brief is unclear, say so. Don't invent plausible-sounding answers for things that should be decided by the team.
6. **When summarizing**, distinguish between: confirmed facts (from the project brief), team decisions (from meeting notes), and AI suggestions.

---

## 6. Custom Skills

This project includes two AI skills (in `.agents/skills/`):

### `/changelog-summary`
Summarize recent changes, what's new since a date, or the current state of a folder. The AI reads `CHANGELOG.md` and relevant `README.md` files and produces a structured summary.

### `/folder-summary`
Deep-dive summary of a specific folder: all files, their purpose, what's done vs. TBD. More detailed than a README — meant for onboarding or handoff.

---

_Last updated: 2026-08-09. Update as team workflows evolve._
