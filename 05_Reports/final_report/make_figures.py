"""
Generate the report's figures as PDFs (vector, embeddable by pdflatex).

The AAAI style forbids .eps, so everything here is written as PDF. Figures are
generated rather than drawn by hand for the same reason the tables are: a
figure that is redrawn from the data cannot silently disagree with it.

    .venv/bin/python 05_Reports/final_report/make_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
# AAAI forbids Type 3 fonts ("No type 3 fonts may be used (even in
# illustrations)"), and matplotlib emits Type 3 in PDF by default. 42 selects
# TrueType. build.sh fails the build if this regresses.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(HERE, "figures")

INK = "#111318"
MUTED = "#6b7280"
RETRO = "#b45309"   # retrospective path
PROSP = "#0f766e"   # prospective path
SHARED = "#3730a3"  # where they converge


def _box(ax, x, y, w, h, text, edge, face="white", fontsize=7.2, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.1, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=3, linespacing=1.35, weight=weight)


def _arrow(ax, x1, y1, x2, y2, color=MUTED, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                                 linewidth=1.0, color=color, zorder=1,
                                 shrinkA=1.5, shrinkB=1.5))


def pipeline_overview(path: str) -> None:
    """The methodology overview figure the course guideline requires.

    Vertical three-tier flow: the two cohorts sit side by side, converge on one
    canonical table, and that table feeds the label and the models. Laid out
    this way so no arrow crosses a box -- the one structural idea of the project
    is that both sources reach a single table, and a diagram that is hard to
    trace obscures exactly that.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- tier 1: the two cohorts, side by side
    _box(ax, 0.02, 0.62, 0.44, 0.30, "", RETRO, face="#fdf8f1", fontsize=7)
    ax.text(0.24, 0.885, "RETROSPECTIVE  n=1,860, collected once",
            ha="center", fontsize=6.8, color=RETRO, weight="bold")
    _box(ax, 0.045, 0.755, 0.185, 0.10, "search +\nyt-dlp download", RETRO, fontsize=6.6)
    _box(ax, 0.255, 0.755, 0.185, 0.10, "extract, then\ndelete the video", RETRO, fontsize=6.6)
    _box(ax, 0.045, 0.645, 0.395, 0.085,
         "thumbnail · 20 frames · transcript · audio", RETRO, fontsize=6.6)
    _arrow(ax, 0.231, 0.805, 0.254, 0.805, RETRO)
    _arrow(ax, 0.242, 0.753, 0.242, 0.732, RETRO)

    _box(ax, 0.54, 0.62, 0.44, 0.30, "", PROSP, face="#f0f7f6", fontsize=7)
    ax.text(0.76, 0.885, "PROSPECTIVE  n=11,256, twice daily / 30 d",
            ha="center", fontsize=6.8, color=PROSP, weight="bold")
    _box(ax, 0.565, 0.755, 0.185, 0.10, "discovery at\nage < 24 h", PROSP, fontsize=6.6)
    _box(ax, 0.775, 0.755, 0.185, 0.10, "snapshot views,\ntitle, thumbnail", PROSP, fontsize=6.6)
    _box(ax, 0.565, 0.645, 0.395, 0.085,
         "growth curve · first-observed assets", PROSP, fontsize=6.6)
    _arrow(ax, 0.751, 0.805, 0.774, 0.805, PROSP)
    _arrow(ax, 0.762, 0.753, 0.762, 0.732, PROSP)

    # ---- tier 2: the convergence
    _box(ax, 0.235, 0.40, 0.53, 0.115,
         # matplotlib renders plain text -- no LaTeX escaping here
         "CANONICAL TABLE    meta__   sched__   vis__   aud__   asset__",
         SHARED, face="#eeeefb", fontsize=7.0, weight="bold")
    _arrow(ax, 0.242, 0.618, 0.36, 0.518, RETRO)
    _arrow(ax, 0.762, 0.618, 0.645, 0.518, PROSP)

    # ---- tier 3: what the table produces
    _box(ax, 0.045, 0.13, 0.27, 0.16,
         "LABEL\ntop quartile within\ncategory × age ×\nsize × format", SHARED, fontsize=6.6)
    _box(ax, 0.365, 0.13, 0.27, 0.16,
         "BASELINES\nlogistic regression,\ngradient boosting\non column groups", SHARED, fontsize=6.6)
    _box(ax, 0.685, 0.13, 0.27, 0.16,
         "MULTIMODAL MODEL\nfrozen encoders +\nlate fusion +\nattribution", SHARED, fontsize=6.6)
    _arrow(ax, 0.36, 0.398, 0.18, 0.295, SHARED)
    _arrow(ax, 0.5, 0.398, 0.5, 0.295, SHARED)
    _arrow(ax, 0.645, 0.398, 0.82, 0.295, SHARED)

    ax.text(0.5, 0.055,
            "Both cohorts map onto one table, so every model and ablation is a choice of "
            "column groups rather than a second pipeline.",
            ha="center", fontsize=6.6, color=MUTED, style="italic")

    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def shortcut_ceiling_figure(path: str) -> None:
    """Why the target is a within-cell rank rather than a view count.

    Reads the EDA stats dump rather than hard-coded numbers.
    """
    import json
    stats_path = os.path.join(ROOT, "02_Data", "eda", "eda_stats.json")
    if not os.path.exists(stats_path):
        print(f"  skipped shortcut figure: {stats_path} missing (run eda_retrospective.py)")
        return
    stats = json.load(open(stats_path))
    sc = stats["shortcut_ceiling"]
    cats = [c for c in ("all", "comedy", "howto", "product_reviews", "vlogs") if c in sc]
    vals = [sc[c]["r2_oof"] for c in cats]
    labels = [c.replace("product_reviews", "product rev.").replace("all", "ALL") for c in cats]

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    colors = [SHARED if c == "all" else MUTED for c in cats]
    ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("out-of-fold $R^2$", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.2, color=INK)
    ax.set_title("log(views) from four non-content variables", fontsize=7.2, pad=4)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    p1 = os.path.join(FIGDIR, "pipeline.pdf")
    pipeline_overview(p1)
    print(f"wrote {p1}")
    p2 = os.path.join(FIGDIR, "shortcut_ceiling.pdf")
    shortcut_ceiling_figure(p2)
    if os.path.exists(p2):
        print(f"wrote {p2}")


if __name__ == "__main__":
    main()
