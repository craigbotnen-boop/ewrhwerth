#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--include-pseudo', action='store_true', help='also run the slower fresh-seed pseudo-constraint suite')
args = parser.parse_args()

def run(name):
    print(f'==> {name}', flush=True)
    subprocess.run([sys.executable, str(HERE / name)], check=True)

if args.include_pseudo:
    run('fk48_cleanroom_replication.py')
run('fk48_stationary_canonical_reproducer.py')

checks = json.loads((HERE / 'expected_fk48_outputs.json').read_text())
canon = json.loads((HERE / 'fk48_stationary_canonical_results.json').read_text())
failures = []

def within(label, value, target, tol):
    if abs(value-target) > tol:
        failures.append(f'{label}: got {value}, expected {target} ± {tol}')

within('stationary H_g exponent', canon['fits']['gauge_hessian_frobenius']['exponent'], checks['stationary_gauge_hessian_exponent']['target'], checks['stationary_gauge_hessian_exponent']['abs_tol'])
within('Kgg exponent', canon['fits']['Kgg_frobenius']['exponent'], checks['Kgg_exponent']['target'], checks['Kgg_exponent']['abs_tol'])
within('Kgp exponent', canon['fits']['Kgp_operator_norm']['exponent'], checks['Kgp_exponent']['target'], checks['Kgp_exponent']['abs_tol'])
max_stationarity = max(r['stationarity_norm'] for r in canon['refinement'])
if max_stationarity > checks['max_stationarity_norm']:
    failures.append(f'max stationarity norm {max_stationarity} > {checks["max_stationarity_norm"]}')
min_gap = min(r['physical_gap'] for r in canon['refinement'])
if min_gap < checks['min_physical_gap']:
    failures.append(f'min physical gap {min_gap} < {checks["min_physical_gap"]}')
max_restricted = max(r['restricted_response_norm'] for r in canon['refinement'])
if max_restricted > checks['max_restricted_response_norm']:
    failures.append(f'max restricted response {max_restricted} > {checks["max_restricted_response_norm"]}')

if args.include_pseudo:
    base = json.loads((HERE / 'fk48_cleanroom_results.json').read_text())
    within('pseudo exponent', base['gauge_obstruction_fit']['exponent'], checks['pseudo_constraint_exponent']['target'], checks['pseudo_constraint_exponent']['abs_tol'])

if failures:
    print('\nREPRODUCIBILITY CHECK: FAIL')
    for f in failures:
        print(' -', f)
    raise SystemExit(1)
print('\nREPRODUCIBILITY CHECK: PASS')
