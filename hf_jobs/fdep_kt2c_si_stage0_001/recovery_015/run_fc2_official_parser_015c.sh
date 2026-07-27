#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_fc2_official_015c}"
MPI_RANKS="${MPI_RANKS:-8}"
mkdir -p "$WORK"
cd "$WORK"

python - <<'PY'
import urllib.request
url='https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/05521fdc3fcbe4b8b0aa104421061bc9f2750a7b/hf_jobs/fdep_kt2c_si_stage0_001/run_displacements_007.sh'
urllib.request.urlretrieve(url,'/tmp/run_displacements_007.sh')
PY
PATH=/opt/conda/bin:$PATH bash /tmp/run_displacements_007.sh "$WORK"

mkdir -p outputs reports qe_tmp/fc2_00001
export OMP_NUM_THREADS=1

echo "FC2_ONLY_OFFICIAL_PARSER MPI_RANKS=$MPI_RANKS"
echo "START_FORCE_CASE fc2_00001"
mpirun --bind-to none -np "$MPI_RANKS" pw.x < force_inputs_fc2/fc2_00001.in > outputs/fc2_00001.out
grep -q 'JOB DONE.' outputs/fc2_00001.out

python - <<'PY'
from pathlib import Path
import hashlib, json, os, re
import numpy as np
from phonopy.interface.qe import parse_set_of_forces

inp = Path('force_inputs_fc2/fc2_00001.in')
out = Path('outputs/fc2_00001.out')
text = out.read_text(errors='replace')
energies = re.findall(r'!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry', text)
totals = re.findall(r'Total force\s+=\s+([\-0-9.Ee+]+)', text)
if not energies or 'JOB DONE.' not in text:
    raise SystemExit('QE_COMPLETION_FAIL fc2_00001')
parsed = parse_set_of_forces(54, [out], verbose=False)
if len(parsed) != 1:
    raise SystemExit(f'OFFICIAL_FORCE_PARSE_COUNT_FAIL fc2_00001 count={len(parsed)}')
forces = np.asarray(parsed[0], dtype=float)
if forces.shape != (54, 3) or not np.isfinite(forces).all():
    raise SystemExit(f'OFFICIAL_FORCE_PARSE_SHAPE_FAIL fc2_00001 shape={forces.shape}')
max_abs = float(np.max(np.abs(forces)))
sum_norm = float(np.linalg.norm(forces, axis=1).sum())
reported_total = float(totals[-1]) if totals else None
if max_abs <= 1e-5:
    if reported_total is None or reported_total > 1e-5:
        raise SystemExit(f'FORCE_SCALE_SANITY_FAIL fc2_00001 max_abs={max_abs} reported_total={reported_total}')
    scale_class = 'NEAR_ZERO_CONSISTENT'
else:
    scale_class = 'NORMAL'
receipt = {
    'document_code': 'FDEP_KT2C_SI_FC2_OFFICIAL_PARSER_015C',
    'case': 'fc2_00001',
    'status': 'PASS',
    'atom_count': 54,
    'parser': 'phonopy.interface.qe.parse_set_of_forces',
    'parser_returns_drift_corrected_qe_force_units': True,
    'energy_Ry': float(energies[-1]),
    'reported_total_force_Ry_per_Bohr': reported_total,
    'parsed_force_max_abs_Ry_per_Bohr': max_abs,
    'parsed_force_sum_norm_Ry_per_Bohr': sum_norm,
    'force_scale_class': scale_class,
    'forces_Ry_per_Bohr': forces.tolist(),
    'input_sha256': hashlib.sha256(inp.read_bytes()).hexdigest(),
    'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'provenance': {
        'phono3py_disp_yaml_sha256': hashlib.sha256(Path('phono3py_disp.yaml').read_bytes()).hexdigest(),
        'runner_commit': 'REPLACED_AT_LAUNCH'
    },
    'claim_boundary': {
        'official_fc2_force_extraction': 'PASS',
        'fc2_fc3_construction': 'NOT_EXECUTED',
        'bte': 'NOT_EXECUTED',
        'physical_claim_upgrade': 'BLOCKED'
    }
}
p = Path('reports/fc2_00001_official_receipt.json')
p.write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
print('OFFICIAL_CASE_RECEIPT', json.dumps(receipt, separators=(',', ':')))
print('OFFICIAL_FC2_RECEIPT_SHA256', hashlib.sha256(p.read_bytes()).hexdigest())
PY

echo "END_FORCE_CASE fc2_00001"
echo "HF_OFFICIAL_FC2_OVERALL: PASS"
