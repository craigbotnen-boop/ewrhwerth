#!/usr/bin/env bash
set -euo pipefail
echo "== JOSS PREFLIGHT: CORE ASSETS =="
req=("paper.md" "paper.bib" "README.md" "LICENSE")
for f in "${req[@]}"; do
  [[ -f "$f" ]] && echo "PASS: $f" || echo "FAIL: $f (REQUIRED)"
done

echo -e "\n== JOSS PREFLIGHT: CITATION SYNC =="
# Extract keys like [@Smith2020]
grep -oE '\[@[A-Za-z0-9_]+' paper.md | sed 's/\[@//' | sort -u > cited.txt
# Extract keys from @article{Smith2020,
grep -oE '^@[A-Za-z]+\{[^,]+' paper.bib | sed -E 's/^@[A-Za-z]+\{//' | sort -u > bibbed.txt

missing=$(comm -23 cited.txt bibbed.txt)
if [[ -z "$missing" ]]; then
  echo "PASS: All citations in paper.md are present in paper.bib."
else
  echo "FAIL: Missing keys in paper.bib:"
  echo "$missing"
fi
rm cited.txt bibbed.txt
