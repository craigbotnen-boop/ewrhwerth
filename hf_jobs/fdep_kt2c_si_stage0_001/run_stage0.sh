#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_stage0}"
mkdir -p "$WORK/qe_tmp" "$WORK/reports"
cd "$WORK"

PSEUDO_URL="https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
PSEUDO_FILE="Si.pbe-n-rrkjus_psl.1.0.0.UPF"
PSEUDO_SHA256="669fb75395a9d26973b0ea1ce8223bbcb30d3396c5d48bf5e794d1243c52375a"

python - <<'PY'
from pathlib import Path
import hashlib
import urllib.request

url = "https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF"
path = Path("Si.pbe-n-rrkjus_psl.1.0.0.UPF")
expected = "669fb75395a9d26973b0ea1ce8223bbcb30d3396c5d48bf5e794d1243c52375a"
with urllib.request.urlopen(url, timeout=120) as response:
    path.write_bytes(response.read())
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"PSEUDOPOTENTIAL_HASH_FAIL expected={expected} actual={actual}")
print(f"PSEUDOPOTENTIAL_HASH_PASS {actual}")
PY

{
  echo "JOB_ID=${JOB_ID:-UNSET}"
  echo "ACCELERATOR=${ACCELERATOR:-UNSET}"
  echo "CPU_CORES=${CPU_CORES:-UNSET}"
  echo "MEMORY=${MEMORY:-UNSET}"
  python --version
  pw.x --version 2>&1 | head -40
  phono3py --version
  phonopy --version
} | tee reports/software_versions.txt

cat > si_stage0.in <<EOF
&CONTROL
  calculation = 'scf'
  prefix = 'si_stage0'
  pseudo_dir = '$WORK'
  outdir = '$WORK/qe_tmp'
  tprnfor = .true.
  tstress = .true.
  verbosity = 'high'
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'fixed'
/
&ELECTRONS
  conv_thr = 1.0d-10
  electron_maxstep = 200
  mixing_beta = 0.3
/
ATOMIC_SPECIES
Si 28.0855 $PSEUDO_FILE
CELL_PARAMETERS angstrom
0.0000000000 2.7155000000 2.7155000000
2.7155000000 0.0000000000 2.7155000000
2.7155000000 2.7155000000 0.0000000000
ATOMIC_POSITIONS crystal
Si 0.0000000000 0.0000000000 0.0000000000
Si 0.2500000000 0.2500000000 0.2500000000
K_POINTS automatic
8 8 8 0 0 0
EOF

pw.x -in si_stage0.in > si_stage0.out
grep -q "JOB DONE." si_stage0.out

python - <<'PY'
from pathlib import Path
import hashlib
import json
import os
import re

root = Path.cwd()
text = (root / "si_stage0.out").read_text(errors="replace")
energy = re.findall(r"!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry", text)
forces = re.findall(r"Total force\s+=\s+([\-0-9.Ee+]+)", text)
iterations = re.findall(r"iteration #\s*([0-9]+)", text)
if "JOB DONE." not in text or not energy:
    raise SystemExit("HF_STAGE0_SCF_FAIL")
result = {
    "document_code": "FDEP_KT2C_HF_STAGE0_EXECUTION_001",
    "job_id": os.environ.get("JOB_ID"),
    "status": "PASS",
    "pseudopotential_sha256": hashlib.sha256((root / "Si.pbe-n-rrkjus_psl.1.0.0.UPF").read_bytes()).hexdigest(),
    "final_energy_Ry": float(energy[-1]),
    "final_total_force_Ry_per_Bohr": float(forces[-1]) if forces else None,
    "last_scf_iteration": int(iterations[-1]) if iterations else None,
    "input_sha256": hashlib.sha256((root / "si_stage0.in").read_bytes()).hexdigest(),
    "output_sha256": hashlib.sha256((root / "si_stage0.out").read_bytes()).hexdigest(),
    "claim_boundary": {
        "real_qe_scf": "PASS",
        "cutoff_convergence": "NOT_EXECUTED",
        "kpoint_convergence": "NOT_EXECUTED",
        "fc2_fc3": "NOT_EXECUTED",
        "bte": "NOT_EXECUTED",
        "physical_claim_upgrade": "BLOCKED"
    }
}
(root / "reports" / "stage0_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

echo "HF_STAGE0_OVERALL: PASS"
