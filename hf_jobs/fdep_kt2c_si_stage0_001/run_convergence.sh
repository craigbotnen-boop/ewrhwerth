#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_convergence}"
mkdir -p "$WORK/qe_tmp" "$WORK/outputs" "$WORK/reports"
cd "$WORK"

python - <<'PY'
from pathlib import Path
import hashlib
import urllib.request

url = "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
path = Path("Si.UPF")
expected = "669fb75395a9d26973b0ea1ce8223bbcb30d3396c5d48bf5e794d1243c52375a"
with urllib.request.urlopen(url, timeout=120) as response:
    path.write_bytes(response.read())
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"PSEUDOPOTENTIAL_HASH_FAIL expected={expected} actual={actual}")
print(f"PSEUDOPOTENTIAL_HASH_PASS {actual}")

cutoffs = [40, 50, 60, 70]
kmeshes = [4, 6, 8, 10]

def write_input(path, prefix, ecut, k):
    path.write_text(f"""&CONTROL
  calculation = 'scf'
  prefix = '{prefix}'
  pseudo_dir = '{Path.cwd()}'
  outdir = '{Path.cwd() / 'qe_tmp' / prefix}'
  tprnfor = .true.
  tstress = .true.
  verbosity = 'high'
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = {ecut}
  ecutrho = {8*ecut}
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
  electron_maxstep = 200
  mixing_beta = 0.3
/
ATOMIC_SPECIES
Si 28.0855 Si.UPF
CELL_PARAMETERS angstrom
0.0000000000 2.7155000000 2.7155000000
2.7155000000 0.0000000000 2.7155000000
2.7155000000 2.7155000000 0.0000000000
ATOMIC_POSITIONS crystal
Si 0.0000000000 0.0000000000 0.0000000000
Si 0.2500000000 0.2500000000 0.2500000000
K_POINTS automatic
{k} {k} {k} 0 0 0
""")

for ecut in cutoffs:
    write_input(Path('outputs') / f'cutoff_{ecut:03d}Ry.in', f'cutoff_{ecut}', ecut, 8)
for k in kmeshes:
    write_input(Path('outputs') / f'kmesh_{k:02d}.in', f'kmesh_{k}', 60, k)
PY

export OMP_NUM_THREADS=1
MPI_RANKS="${CPU_CORES:-1}"

run_case() {
  local input="$1"
  local output="${input%.in}.out"
  local label
  label="$(basename "${input%.in}")"
  mkdir -p "qe_tmp/${label}"
  echo "START_CASE ${label}"
  mpirun --bind-to none -np "$MPI_RANKS" pw.x < "$input" > "$output"
  grep -q "JOB DONE." "$output"
  python - "$label" "$output" <<'PY'
from pathlib import Path
import re, sys
label, output = sys.argv[1], Path(sys.argv[2])
text = output.read_text(errors='replace')
energy = re.findall(r"!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry", text)
force = re.findall(r"Total force\s+=\s+([\-0-9.Ee+]+)", text)
if not energy or 'JOB DONE.' not in text:
    raise SystemExit(f'{label}: PARSE_FAIL')
print(f"CASE_PASS {label} ENERGY_RY={energy[-1]} TOTAL_FORCE_RY_BOHR={force[-1] if force else 'NA'}")
PY
}

for input in outputs/cutoff_*Ry.in; do run_case "$input"; done
for input in outputs/kmesh_*.in; do run_case "$input"; done

python - <<'PY'
from pathlib import Path
import hashlib
import json
import os
import re

RY_TO_EV = 13.605693122994
BOHR_TO_A = 0.529177210903
FORCE_CONV = RY_TO_EV / BOHR_TO_A
N_ATOMS = 2
ENERGY_TOL_MEV_ATOM = 1.0
FORCE_TOL_EV_A = 1.0e-3

root = Path.cwd()

def parse(path):
    text = path.read_text(errors='replace')
    energy = re.findall(r"!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry", text)
    force = re.findall(r"Total force\s+=\s+([\-0-9.Ee+]+)", text)
    qe_version = re.findall(r"Program PWSCF v\.([^\s]+)", text)
    if not energy or 'JOB DONE.' not in text:
        raise RuntimeError(f'Incomplete output {path}')
    return {
        'file': str(path),
        'energy_Ry': float(energy[-1]),
        'energy_eV_per_atom': float(energy[-1]) * RY_TO_EV / N_ATOMS,
        'total_force_Ry_per_Bohr': float(force[-1]) if force else None,
        'total_force_eV_per_A': float(force[-1]) * FORCE_CONV if force else None,
        'qe_version': qe_version[0] if qe_version else None,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }

cutoff = []
for path in sorted(root.glob('outputs/cutoff_*Ry.out')):
    match = re.search(r'cutoff_(\d+)Ry', path.name)
    row = parse(path)
    row['ecutwfc_Ry'] = int(match.group(1))
    cutoff.append(row)
cutoff.sort(key=lambda x: x['ecutwfc_Ry'])

kmesh = []
for path in sorted(root.glob('outputs/kmesh_*.out')):
    match = re.search(r'kmesh_(\d+)', path.name)
    row = parse(path)
    row['kmesh'] = [int(match.group(1))] * 3
    kmesh.append(row)
kmesh.sort(key=lambda x: x['kmesh'][0])

for rows in (cutoff, kmesh):
    previous = None
    for row in rows:
        row['delta_from_previous_meV_per_atom'] = None
        if previous is not None:
            row['delta_from_previous_meV_per_atom'] = abs(
                row['energy_eV_per_atom'] - previous['energy_eV_per_atom']
            ) * 1000.0
        previous = row

cutoff_last_delta = cutoff[-1]['delta_from_previous_meV_per_atom']
kmesh_last_delta = kmesh[-1]['delta_from_previous_meV_per_atom']
cutoff_force_ok = all(
    row['total_force_eV_per_A'] is None or row['total_force_eV_per_A'] <= FORCE_TOL_EV_A
    for row in cutoff
)
kmesh_force_ok = all(
    row['total_force_eV_per_A'] is None or row['total_force_eV_per_A'] <= FORCE_TOL_EV_A
    for row in kmesh
)
cutoff_pass = cutoff_last_delta is not None and cutoff_last_delta <= ENERGY_TOL_MEV_ATOM and cutoff_force_ok
kmesh_pass = kmesh_last_delta is not None and kmesh_last_delta <= ENERGY_TOL_MEV_ATOM and kmesh_force_ok

result = {
    'document_code': 'FDEP_KT2C_SI_CONVERGENCE_EXECUTION_001',
    'job_id': os.environ.get('JOB_ID'),
    'status': 'PASS' if cutoff_pass and kmesh_pass else 'FAIL',
    'pseudopotential_sha256': hashlib.sha256((root / 'Si.UPF').read_bytes()).hexdigest(),
    'mpi_ranks': int(os.environ.get('CPU_CORES', '1')),
    'thresholds': {
        'energy_meV_per_atom': ENERGY_TOL_MEV_ATOM,
        'total_force_eV_per_A': FORCE_TOL_EV_A,
    },
    'gates': {
        'KT2C_G05_CUTOFF_CONVERGENCE': 'PASS' if cutoff_pass else 'FAIL',
        'KT2C_G06_KPOINT_CONVERGENCE': 'PASS' if kmesh_pass else 'FAIL',
    },
    'cutoff_ladder': cutoff,
    'kmesh_ladder': kmesh,
    'claim_boundary': {
        'software_pilot_convergence': 'PASS' if cutoff_pass and kmesh_pass else 'FAIL',
        'structural_relaxation': 'NOT_EXECUTED',
        'fc2_fc3': 'NOT_EXECUTED',
        'bte': 'NOT_EXECUTED',
        'physical_claim_upgrade': 'BLOCKED',
    },
}
(root / 'reports' / 'convergence_result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if result['status'] != 'PASS':
    raise SystemExit(1)
PY

echo "HF_CONVERGENCE_OVERALL: PASS"
