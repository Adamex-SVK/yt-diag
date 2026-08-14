# changelog-summary

Summarize recent changes, what's new since a date, or the current state of a folder in the DLDM project.

## Trigger

User says any of:
- "What's new?"
- "Summarize changes since [date]"
- "What happened this week?"
- "What's the current state of [folder]?"
- "Give me a project update"
- `/changelog-summary`

## Steps

1. Read `CHANGELOG.md` from the project root.
2. If a date is specified, filter entries from that date onward.
3. If a folder is specified, filter entries that mention files in that folder.
4. If the user asks for "current state" of a folder, also read that folder's `README.md`.
5. Produce a structured summary:

```markdown
## Summary: [what was asked]

### New since [date] (or "All recent changes")
- **[date] [author]** — description (files)
- ...

### Current state of [folder] (if asked)
- What's done
- What's in progress
- What's TBD or blocked
```

## Rules

- Always read `CHANGELOG.md` first — don't rely on memory.
- Distinguish between: confirmed facts (from project brief), team decisions (from meeting notes), AI suggestions.
- Keep it concise but complete. If there are 20+ entries, group by folder or theme.
