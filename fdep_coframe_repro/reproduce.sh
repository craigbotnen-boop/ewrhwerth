#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
rm -f results/p8.json results/p9.json results/p10.json results/p12.json results/p13.json results/p14.json results/p17.json results/p15.json results/summary.json results/main_tables.md
for stage in p8 p9 p10 p12 p13 p14 p17; do
  echo "== $stage =="
  python -u run_stage.py "$stage"
done
echo "== p15 refinement =="
python -u run_p15_all.py
python -u assemble_results.py
