#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_force_pilot_011}"
mkdir -p "$WORK"
cd "$WORK"

python - <<'PY'
import urllib.request
url='https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/05521fdc3fcbe4b8b0aa104421061bc9f2750a7b/hf_jobs/fdep_kt2c_si_stage0_001/run_displacements_007.sh'
urllib.request.urlretrieve(url,'/tmp/run_displacements_007.sh')
PY
PATH=/opt/conda/bin:$PATH bash /tmp/run_displacements_007.sh "$WORK"

mkdir -p outputs reports qe_tmp
export OMP_NUM_THREADS=1
MPI_RANKS=16
printf 'MPI_RANKS_SELECTED=%s CPU_CORES_REPORTED=%s\n' "$MPI_RANKS" "${CPU_CORES:-UNSET}"

run_case(){
  local input="$1"
  local output="$2"
  local prefix="$3"
  mkdir -p "$WORK/qe_tmp/$prefix" "$(dirname "$output")"
  echo "START_FORCE_CASE $prefix"
  mpirun --bind-to none -np "$MPI_RANKS" pw.x < "$input" > "$output"
  grep -q 'JOB DONE.' "$output"
  python - "$output" "$prefix" <<'PY'
from pathlib import Path
import hashlib, json, re, sys
path=Path(sys.argv[1]); prefix=sys.argv[2]
text=path.read_text(errors='replace')
all_rows=re.findall(r'atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)', text)
energy=re.findall(r'!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry', text)
total=re.findall(r'Total force\s+=\s+([\-0-9.Ee+]+)', text)
expected=16 if prefix.startswith('fc3') else 54
if len(all_rows) < expected or len(all_rows) % expected != 0 or not energy or 'JOB DONE.' not in text:
    raise SystemExit(f'FORCE_PARSE_FAIL {prefix} rows={len(all_rows)} expected_multiple={expected}')
rows=all_rows[-expected:]
indices=[int(r[0]) for r in rows]
if indices != list(range(1, expected+1)):
    raise SystemExit(f'FINAL_FORCE_BLOCK_INDEX_FAIL {prefix} indices={indices}')
receipt={
    'case':prefix,
    'status':'PASS',
    'atom_count':expected,
    'printed_force_blocks':len(all_rows)//expected,
    'selected_force_block':'final',
    'energy_Ry':float(energy[-1]),
    'total_force_Ry_per_Bohr':float(total[-1]) if total else None,
    'forces_Ry_per_Bohr':[[float(x),float(y),float(z)] for _,x,y,z in rows],
    'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
}
Path(f'reports/{prefix}_force_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('FORCE_RECEIPT_JSON',json.dumps(receipt,separators=(',',':')))
PY
}

run_case force_inputs_fc3/fc3_00001.in outputs/fc3_00001.out fc3_00001
run_case force_inputs_fc2/fc2_00001.in outputs/fc2_00001.out fc2_00001

python - <<'PY'
from pathlib import Path
import json, os
r3=json.loads(Path('reports/fc3_00001_force_receipt.json').read_text())
r2=json.loads(Path('reports/fc2_00001_force_receipt.json').read_text())
result={
  'document_code':'FDEP_KT2C_SI_FORCE_PILOT_011',
  'status':'PASS',
  'job_id':os.environ.get('JOB_ID'),
  'mpi_ranks':16,
  'parser_rule':'final_complete_N_atom_force_block',
  'fc3_case':{k:v for k,v in r3.items() if k!='forces_Ry_per_Bohr'},
  'fc2_case':{k:v for k,v in r2.items() if k!='forces_Ry_per_Bohr'},
  'gates':{
    'KT2C_G12_FC3_FORCE_CASE':'PASS',
    'KT2C_G13_FC2_FORCE_CASE':'PASS',
  },
  'claim_boundary':{
    'force_pilot':'PASS',
    'full_force_campaign':'NOT_EXECUTED',
    'fc2_fc3_construction':'NOT_EXECUTED',
    'bte':'NOT_EXECUTED',
    'physical_claim_upgrade':'BLOCKED',
  },
}
Path('reports/force_pilot_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY

echo 'HF_FORCE_PILOT_OVERALL: PASS'
