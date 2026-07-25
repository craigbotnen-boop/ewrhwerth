#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_vc_relax_003}"
mkdir -p "$WORK/qe_tmp" "$WORK/reports" "$WORK/outputs"
cd "$WORK"

python - <<'PY'
from pathlib import Path
import hashlib, urllib.request
url='https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF'
path=Path('Si.UPF')
expected='669fb75395a9d26973b0ea1ce8223bbcb30d3396c5d48bf5e794d1243c52375a'
with urllib.request.urlopen(url, timeout=120) as r:
    path.write_bytes(r.read())
actual=hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f'PSEUDOPOTENTIAL_HASH_FAIL expected={expected} actual={actual}')
print('PSEUDOPOTENTIAL_HASH_PASS', actual)
PY

cat > outputs/si_vc_relax.in <<EOF
&CONTROL
  calculation = 'vc-relax'
  prefix = 'si_vc_relax'
  pseudo_dir = '$WORK'
  outdir = '$WORK/qe_tmp'
  tprnfor = .true.
  tstress = .true.
  verbosity = 'high'
  nstep = 100
  etot_conv_thr = 1.0d-6
  forc_conv_thr = 1.0d-5
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = 80
  ecutrho = 640
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
  electron_maxstep = 200
  mixing_beta = 0.3
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press = 0.0
  press_conv_thr = 0.5
  cell_dofree = 'all'
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
14 14 14 0 0 0
EOF

export OMP_NUM_THREADS=1
MPI_RANKS="${CPU_CORES:-1}"
mpirun --bind-to none -np "$MPI_RANKS" pw.x < outputs/si_vc_relax.in > outputs/si_vc_relax.out
grep -q 'JOB DONE.' outputs/si_vc_relax.out

python - <<'PY'
from pathlib import Path
import hashlib, json, math, os, re
from ase.io import read, write

root=Path.cwd()
out=root/'outputs/si_vc_relax.out'
text=out.read_text(errors='replace')
energy=[float(x) for x in re.findall(r'!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry', text)]
force=[float(x) for x in re.findall(r'Total force\s+=\s+([\-0-9.Ee+]+)', text)]
pressure=[float(x) for x in re.findall(r'P=\s*([\-0-9.Ee+]+)', text)]
if not energy or 'JOB DONE.' not in text:
    raise SystemExit('RELAX_PARSE_FAIL')

atoms=read(out, format='espresso-out', index=-1)
write(root/'outputs/Si_relaxed.xyz', atoms)
cell=atoms.cell.array.tolist()
positions=atoms.get_scaled_positions(wrap=False).tolist()
volume=float(atoms.get_volume())
lengths=[float(x) for x in atoms.cell.lengths()]
angles=[float(x) for x in atoms.cell.angles()]

final_force=force[-1] if force else None
final_pressure=pressure[-1] if pressure else None
force_pass=final_force is not None and final_force <= 1.0e-5
pressure_pass=final_pressure is not None and abs(final_pressure) <= 0.5
finite=all(math.isfinite(v) for row in cell for v in row) and math.isfinite(volume) and volume>0
status='PASS' if force_pass and pressure_pass and finite else 'FAIL'

result={
  'document_code':'FDEP_KT2C_SI_VC_RELAX_EXECUTION_003',
  'job_id':os.environ.get('JOB_ID'),
  'status':status,
  'qe_version':'7.5',
  'production_settings':{'ecutwfc_Ry':80,'ecutrho_Ry':640,'kmesh':[14,14,14]},
  'thresholds':{'final_total_force_Ry_per_Bohr':1.0e-5,'absolute_pressure_kbar':0.5},
  'final':{
    'energy_Ry':energy[-1],
    'total_force_Ry_per_Bohr':final_force,
    'pressure_kbar':final_pressure,
    'cell_A':cell,
    'cell_lengths_A':lengths,
    'cell_angles_deg':angles,
    'volume_A3':volume,
    'scaled_positions':positions,
  },
  'gates':{
    'KT2C_G07_RELAXATION_JOB_DONE':'PASS',
    'KT2C_G07_FINAL_FORCE':'PASS' if force_pass else 'FAIL',
    'KT2C_G07_FINAL_PRESSURE':'PASS' if pressure_pass else 'FAIL',
    'KT2C_G07_FINAL_STRUCTURE_FINITE':'PASS' if finite else 'FAIL',
  },
  'hashes':{
    'input_sha256':hashlib.sha256((root/'outputs/si_vc_relax.in').read_bytes()).hexdigest(),
    'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'relaxed_xyz_sha256':hashlib.sha256((root/'outputs/Si_relaxed.xyz').read_bytes()).hexdigest(),
    'pseudopotential_sha256':hashlib.sha256((root/'Si.UPF').read_bytes()).hexdigest(),
  },
  'claim_boundary':{
    'software_pilot_relaxation':status,
    'displacement_campaign':'NOT_EXECUTED',
    'fc2_fc3':'NOT_EXECUTED',
    'bte':'NOT_EXECUTED',
    'physical_claim_upgrade':'BLOCKED',
  },
}
(root/'reports/vc_relax_result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
if status != 'PASS':
    raise SystemExit(1)
PY

echo 'HF_VC_RELAX_OVERALL: PASS'
