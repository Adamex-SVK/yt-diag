# YT-Diag — Slide Outline (3-minute, solo)

Companion to `script.md`. Originally 5 slides for a 3-minute talk, built on
the "Broadside" deck theme (dark editorial canvas, single fire-orange
accent).

Note (2026-09-06): the actual deck now has 12 slides after adding dataset,
architecture, transformer-internals, five-inputs, fusion-math, attribution,
and rationale slides on request. This table still documents the original
5-slide cut; a rewrite is needed to describe the current 12-slide structure
before this file is trusted as the source of truth. Until then, treat the
published artifact (`ytdiag_deck.html`) as authoritative for slide content.

| # | Slide | Layout | Content |
|---|---|---|---|
| 1 | Cover | `slide--cover` (orange) | "does content actually explain why a video performed?" Adam Michalik, Emmanuel Gyabaah, DLDM 2026 |
| 2 | The question | `slide--statement` (dark, orange accent line) | Peer-relative performance beyond metadata; shortcut ceiling 58.4% named as the bar to clear |
| 3 | Headline result | `slide--chart` (dark) | Bar chart: structured XGBoost 0.619±0.022 vs. best content fusion 0.613±0.044 (2/5 wins) |
| 4 | One honest finding | custom stat+CI diagram (dark) | Audio-isolation lift +0.018 AUC, 95% CI [−0.009, +0.044], crosses zero, "not yet significant" |
| 5 | Conclusion | `slide--end` (orange) | Content is informative, not yet reliably better; prospective panel is the real next test |

## Design notes

- Numbers here must stay byte-identical to `script.md` and `main.tex`. If a
  number changes upstream, update all three in the same pass.
- Slide 4's CI whisker crossing zero is a custom diagram; most other slides
  use Broadside's built-in `stats`/`chart`/`statement`/`end` layout patterns
  as-is, per the theme's own "don't invent new layouts" rule.
- No em dashes anywhere in slide copy or script text, per explicit
  instruction (2026-09-06). Use periods, colons, or "because"/"so" instead.
- Cut from the 8-minute draft originally: dataset composition, architecture
  diagram, visual-ablation ruled-out claims, attribution walkthrough,
  limitations list. Several of these were later added back into the 12-slide
  deck once the professor-facing methodology depth was requested; see the
  live artifact for what actually survived.
- If time runs short again, the honest fallback order is to cut the deepest
  methodology slides first (transformer internals, fusion math) since the
  headline result and its honesty caveat are the load-bearing content.
