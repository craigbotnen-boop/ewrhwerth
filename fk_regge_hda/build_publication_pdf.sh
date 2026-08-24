#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python make_figures.py
cp manuscript_v1.tex manuscript_publication.tex
python - <<'PY'
from pathlib import Path
p = Path('manuscript_publication.tex')
s = p.read_text()
if '\\usepackage[section]{placeins}' not in s:
    s = s.replace('\\usepackage{mathrsfs}', '\\usepackage{mathrsfs}\n\\usepackage[section]{placeins}')
s = s.replace('width=0.72\\linewidth', 'width=0.62\\linewidth')
p.write_text(s)
PY
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript_publication.tex

echo "Built manuscript_publication.pdf"
