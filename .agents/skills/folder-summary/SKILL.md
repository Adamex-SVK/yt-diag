# folder-summary

Deep-dive summary of a specific folder: all files, their purpose, what's done vs. TBD. More detailed than a README — meant for onboarding or handoff.

## Trigger

User says any of:
- "Summarize folder [name]"
- "What's in [folder]?"
- "Give me a deep dive on [folder]"
- `/folder-summary [folder]`

## Steps

1. List all files in the requested folder (use `list_dir`).
2. Read the folder's `README.md` for context.
3. Read any `.md` files in the folder for content (skim headers, not full text unless needed).
4. Check `CHANGELOG.md` for entries mentioning files in this folder.
5. Produce a structured summary:

```markdown
## Folder: [path]

### Purpose
[from README.md]

### Contents
| File | Purpose | Status |
|------|---------|--------|
| ... | ... | done / in-progress / stub |

### Recent changes
- [date] [author] — description

### What's done
- ...

### What's in progress
- ...

### What's TBD
- ...
```

## Rules

- If the folder has many files (15+), group them logically rather than listing every file individually.
- Mark files as "stub" if they're placeholders with no real content yet.
- Reference `CHANGELOG.md` for the "Recent changes" section.
