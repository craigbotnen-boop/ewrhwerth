#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_force_recovery_015b}"
START_INDEX="${START_INDEX:?START_INDEX required}"
END_INDEX="${END_INDEX:?END_INDEX required}"
RUN_FC2="${RUN_FC2:-0}"
MPI_RANKS="${MPI_RANKS:-8}"
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
printf 'OFFICIAL_PARSER_BATCH_RANGE=%s-%s RUN_FC2=%s MPI_RANKS=%s\n' "$START_INDEX" "$END_INDEX" "$RUN_FC2" "$MPI_RANKS"

run_case(){
  local input="$1" output="$2" prefix="$3" expected="$4"
  mkdir -p "$WORK/qe_tmp/$prefix" "$(dirname "$output")"
  echo "START_FORCE_CASE $prefix"
  mpirun --bind-to none -np "$MPI_RANKS" pw.x < "$input" > "$output"
  grep -q 'JOB DONE.' "$output"
  python - "$input" "$output" "$prefix" "$expected" <<'PY'
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
from phonopy.interface.qe import parse_set_of_forces
inp=Path(sys.argv[1]); out=Path(sys.argv[2]); prefix=sys.argv[3]; expected=int(sys.argv[4])
text=out.read_text(errors='replace')
energies=re.findall(r'!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry',text)
totals=re.findall(r'Total force\s+=\s+([\-0-9.Ee+]+)',text)
if not energies or 'JOB DONE.' not in text:
    raise SystemExit(f'QE_COMPLETION_FAIL {prefix}')
parsed=parse_set_of_forces(expected,[out],verbose=False)
if len(parsed)!=1:
    raise SystemExit(f'OFFICIAL_FORCE_PARSE_COUNT_FAIL {prefix} count={len(parsed)}')
forces=np.asarray(parsed[0],dtype=float)
if forces.shape!=(expected,3) or not np.isfinite(forces).all():
    raise SystemExit(f'OFFICIAL_FORCE_PARSE_SHAPE_FAIL {prefix} shape={forces.shape}')
max_abs=float(np.max(np.abs(forces)))
sum_norm=float(np.linalg.norm(forces,axis=1).sum())
reported_total=float(totals[-1]) if totals else None
near_zero=max_abs <= 1e-5
if near_zero and (reported_total is None or reported_total > 1e-4):
    raise SystemExit(f'FORCE_SCALE_INCONSISTENCY_FAIL {prefix} max_abs={max_abs} reported_total={reported_total}')
scale_class='NEAR_ZERO_CONSISTENT' if near_zero else 'NORMAL'
receipt={
  'case':prefix,'status':'PASS','atom_count':expected,
  'parser':'phonopy.interface.qe.parse_set_of_forces',
  'parser_returns_drift_corrected_qe_force_units':True,
  'energy_Ry':float(energies[-1]),
  'reported_total_force_Ry_per_Bohr':reported_total,
  'parsed_force_max_abs_Ry_per_Bohr':max_abs,
  'parsed_force_sum_norm_Ry_per_Bohr':sum_norm,
  'force_scale_class':scale_class,
  'forces_Ry_per_Bohr':forces.tolist(),
  'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
  'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
}
Path(f'reports/{prefix}.json').write_text(json.dumps(receipt,separators=(',',':'))+'\n')
print('OFFICIAL_CASE_RECEIPT',json.dumps(receipt,separators=(',',':')))
PY
  echo "END_FORCE_CASE $prefix"
}

for idx in $(seq "$START_INDEX" "$END_INDEX"); do
  tag=$(printf '%05d' "$idx")
  run_case "force_inputs_fc3/fc3_${tag}.in" "outputs/fc3_${tag}.out" "fc3_${tag}" 16
done

if [ "$RUN_FC2" = "1" ]; then
  run_case force_inputs_fc2/fc2_00001.in outputs/fc2_00001.out fc2_00001 54
fi

python - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
rows=[json.loads(p.read_text()) for p in sorted(Path('reports').glob('fc*.json'))]
result={
 'document_code':'FDEP_KT2C_SI_FORCE_BATCH_OFFICIAL_PARSER_015B',
 'status':'PASS','job_id':os.environ.get('JOB_ID'),
 'start_index':int(os.environ['START_INDEX']),'end_index':int(os.environ['END_INDEX']),
 'run_fc2':os.environ.get('RUN_FC2','0')=='1','case_count':len(rows),'cases':rows,
 'gates':{'KT2C_G14A_OFFICIAL_QE_FORCE_EXTRACTION':'PASS','KT2C_G14B_NEAR_ZERO_CONSISTENCY':'PASS'},
 'claim_boundary':{'official_force_extraction':'PASS','fc2_fc3_construction':'NOT_EXECUTED','bte':'NOT_EXECUTED','physical_claim_upgrade':'BLOCKED'}
}
p=Path('reports/batch_official_receipt.json'); p.write_text(json.dumps(result,separators=(',',':'))+'\n')
sha=hashlib.sha256(p.read_bytes()).hexdigest()
print('OFFICIAL_BATCH_RECEIPT_JSON',json.dumps(result,separators=(',',':')))
cp=subprocess.run(['curl','--fail','--silent','--show-error','-F',f'file=@{p}','https://tmpfiles.org/api/v1/upload'],check=True,text=True,capture_output=True)
payload=json.loads(cp.stdout)
print('OFFICIAL_BATCH_UPLOAD_JSON',json.dumps({'status':'PASS','page_url':payload['data']['url'],'sha256':sha,'size_bytes':p.stat().st_size},separators=(',',':')))
PY

echo 'HF_OFFICIAL_FORCE_BATCH_OVERALL: PASS'
