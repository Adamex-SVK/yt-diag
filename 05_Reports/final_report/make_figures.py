"""
Generate the report's figures as PDFs (vector, embeddable by pdflatex).

The AAAI style forbids .eps, so everything here is written as PDF. Figures are
generated rather than drawn by hand for the same reason the tables are: a
figure that is redrawn from the data cannot silently disagree with it.

    .venv/bin/python 05_Reports/final_report/make_figures.py
"""
from __future__ import annotations

import os
import json
import sys

import matplotlib
matplotlib.use("Agg")
# AAAI forbids Type 3 fonts ("No type 3 fonts may be used (even in
# illustrations)"), and matplotlib emits Type 3 in PDF by default. 42 selects
# TrueType. build.sh fails the build if this regresses.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(HERE, "figures")

INK = "#111318"
MUTED = "#6b7280"
RETRO = "#b45309"   # retrospective path
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


def architecture_overview(path: str) -> None:
    """The frozen multimodal finalist and its structured comparator."""
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    inputs = [
        (0.04, 0.79, "thumbnail", "CLIP\nViT-B/32"),
        (0.04, 0.59, "20 frames", "shared CLIP\n+ mean pool"),
        (0.04, 0.39, "title · description\ntranscript chunks", "ModernBERT\nper field"),
        (0.04, 0.19, "metadata\n+ schedule", "impute · scale\none-hot"),
    ]
    ax.text(0.105, 0.97, "INPUT", ha="center", fontsize=7, color=MUTED, weight="bold")
    ax.text(0.345, 0.97, "FROZEN / TRAIN-FIT TRANSFORM", ha="center", fontsize=7,
            color=MUTED, weight="bold")
    for _, y, input_text, encoder_text in inputs:
        _box(ax, 0.02, y - 0.07, 0.17, 0.13, input_text, RETRO, face="#fdf8f1", fontsize=6.7)
        _box(ax, 0.245, y - 0.07, 0.20, 0.13, encoder_text, SHARED,
             face="#eeeefb", fontsize=6.7)
        _arrow(ax, 0.192, y, 0.242, y, SHARED)
        _arrow(ax, 0.448, y, 0.53, 0.58, SHARED)

    _box(ax, 0.535, 0.47, 0.18, 0.22,
         "STANDARDIZE\nEACH BLOCK\n\nconcatenate", SHARED, face="#eeeefb",
         fontsize=7.0, weight="bold")
    _arrow(ax, 0.718, 0.58, 0.77, 0.58, SHARED)
    _box(ax, 0.775, 0.47, 0.19, 0.22,
         "L2 LOGISTIC\nFUSION\n\np(top quartile)", SHARED, face="#eeeefb",
         fontsize=7.0, weight="bold")
    _box(ax, 0.535, 0.12, 0.43, 0.16,
         "EXACT PREDICTIVE EXPLANATION\nlogit = intercept + Σ block contribution",
         RETRO, face="#fdf8f1", fontsize=6.9)
    _arrow(ax, 0.87, 0.465, 0.84, 0.285, RETRO)
    ax.text(0.23, 0.055,
            "Comparator: metadata + schedule → nested-tuned XGBoost",
            ha="center", fontsize=6.8, color=INK, weight="bold")

    fig.tight_layout(pad=0.2)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def attribution_figure(path: str) -> None:
    """Global block shares plus four seed-0 prediction explanations."""
    result_path = os.path.join(HERE, "results", "attribution.json")
    if not os.path.exists(result_path):
        print(f"  skipped attribution figure: {result_path} missing")
        return
    with open(result_path, encoding="utf-8") as stream:
        result = json.load(stream)
    names = result["blocks"]
    labels = [name.replace("metadata_schedule", "metadata + schedule") for name in names]
    shares = [result["share_of_absolute_logit_contribution"][name]["mean"] for name in names]
    examples = result["qualitative_examples"]
    matrix = [[example["contributions"][name] for name in names] for example in examples]
    row_labels = [f"{example['kind']}\n{example['video_id']}" for example in examples]

    fig, (left, right) = plt.subplots(2, 1, figsize=(3.35, 3.15), gridspec_kw={"height_ratios": [1, 1.15]})
    order = np.argsort(shares)
    left.barh(np.arange(len(names)), np.asarray(shares)[order] * 100, color=SHARED, alpha=0.88)
    left.set_yticks(np.arange(len(names)), np.asarray(labels)[order], fontsize=5.9)
    left.set_xlabel("mean share of |logit contribution| (%)", fontsize=6.4)
    left.tick_params(axis="x", labelsize=6)
    left.grid(axis="x", alpha=0.2, linewidth=0.5)
    left.set_title("(a) Share across five development splits", fontsize=7.0)
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)

    bound = max(abs(np.asarray(matrix).min()), abs(np.asarray(matrix).max()))
    image = right.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    right.set_xticks(np.arange(len(names)), labels, rotation=24, ha="right", fontsize=5.4)
    right.set_yticks(np.arange(len(examples)), row_labels, fontsize=5.5)
    right.set_title("(b) Exact logit contributions for seed-0 examples", fontsize=7.2)
    for row in range(len(examples)):
        for col in range(len(names)):
            value = matrix[row][col]
            right.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=5.2,
                       color="white" if abs(value) > bound * 0.52 else INK)
    fig.tight_layout(pad=0.25, h_pad=0.55)
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
    p1 = os.path.join(FIGDIR, "architecture.pdf")
    architecture_overview(p1)
    print(f"wrote {p1}")
    p2 = os.path.join(FIGDIR, "shortcut_ceiling.pdf")
    shortcut_ceiling_figure(p2)
    if os.path.exists(p2):
        print(f"wrote {p2}")
    p3 = os.path.join(FIGDIR, "attribution.pdf")
    attribution_figure(p3)
    if os.path.exists(p3):
        print(f"wrote {p3}")


if __name__ == "__main__":
    main()
