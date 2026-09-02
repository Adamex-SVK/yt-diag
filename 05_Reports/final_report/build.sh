#!/bin/sh
# Build the report. Runs the full pdflatex -> bibtex -> pdflatex x2 cycle,
# because a single pass leaves citations and the table reference unresolved.
#
# TinyTeX is used rather than MacTeX: it installs into $HOME with no sudo, and
# it provides a real pdflatex, which is what the AAAI style requires ("Your
# .tex file must compile in PDFLaTeX"). Install once with:
#
#   curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
#   ~/Library/TinyTeX/bin/universal-darwin/tlmgr install psnfss booktabs \
#       xcolor graphics graphics-def epstopdf-pkg times helvetic courier amsmath
#
# Regenerate the results table BEFORE building, so the numbers match the code:
#   .venv/bin/python 05_Reports/final_report/make_tables.py --seeds 5
set -e

cd "$(dirname "$0")"
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"

command -v pdflatex >/dev/null 2>&1 || {
    echo "pdflatex not found. Install TinyTeX (see the header of this script)." >&2
    exit 1
}

echo "==> pass 1"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "==> bibtex"
bibtex main >/dev/null || echo "    (bibtex reported an issue -- check main.blg)"
echo "==> pass 2"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
echo "==> pass 3 (resolves cross-references)"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

echo
echo "--- warnings worth reading ---"
grep -E "LaTeX Warning|Overfull \\\\hbox" main.log | grep -v "Font Warning" | head -20 || true

# AAAI compliance, checked rather than assumed. Needs poppler (brew install poppler).
if command -v pdffonts >/dev/null 2>&1; then
    t3=$(pdffonts main.pdf | awk 'NR>2 && $2=="Type" && $3=="3"' | wc -l | tr -d ' ')
    noemb=$(pdffonts main.pdf | awk 'NR>2 && $4=="no"' | wc -l | tr -d ' ')
    pages=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
    size=$(pdfinfo main.pdf | awk -F: '/^Page size/{print $2}' | sed 's/^ *//')
    echo
    echo "--- AAAI checks ---"
    echo "  pages          : $pages   (confirm against the course page limit)"
    echo "  page size      : $size"
    echo "  Type 3 fonts   : $t3  (must be 0)"
    echo "  non-embedded   : $noemb  (must be 0)"
    [ "$t3" = "0" ] && [ "$noemb" = "0" ] || { echo "  FAILED font requirements" >&2; exit 1; }
fi

echo
echo "built main.pdf"
grep -q '\\drafttrue' main.tex && printf 'NOTE: still in DRAFT mode -- draft markers are rendered.\n      Set \\draftfalse in main.tex before submitting.\n'
