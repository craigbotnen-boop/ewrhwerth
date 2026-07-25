#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_force_batch_012}"
START_INDEX="${START_INDEX:?START_INDEX required}"
END_INDEX="${END_INDEX:?END_INDEX required}"
RUN_FC2="${RUN_FC2:-0}"
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
printf 'BATCH_RANGE=%s-%s RUN_FC2=%s MPI_RANKS=%s\n' "$START_INDEX" "$END_INDEX" "$RUN_FC2" "$MPI_RANKS"

run_case(){
  local input="$1" output="$2" prefix="$3" expected="$4"
  mkdir -p "$WORK/qe_tmp/$prefix" "$(dirname "$output")"
  echo "START_FORCE_CASE $prefix"
  mpirun --bind-to none -np "$MPI_RANKS" pw.x < "$input" > "$output"
  grep -q 'JOB DONE.' "$output"
  python - "$output" "$prefix" "$expected" <<'PY'
from pathlib import Path
import hashlib,json,re,sys
path=Path(sys.argv[1]); prefix=sys.argv[2]; expected=int(sys.argv[3])
text=path.read_text(errors='replace')
all_rows=re.findall(r'atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)',text)
energy=re.findall(r'!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry',text)
total=re.findall(r'Total force\s+=\s+([\-0-9.Ee+]+)',text)
if len(all_rows)<expected or len(all_rows)%expected or not energy or 'JOB DONE.' not in text:
    raise SystemExit(f'FORCE_PARSE_FAIL {prefix} rows={len(all_rows)} expected_multiple={expected}')
rows=all_rows[-expected:]
if [int(r[0]) for r in rows] != list(range(1,expected+1)):
    raise SystemExit(f'FINAL_FORCE_BLOCK_INDEX_FAIL {prefix}')
receipt={'case':prefix,'status':'PASS','atom_count':expected,'printed_force_blocks':len(all_rows)//expected,'energy_Ry':float(energy[-1]),'total_force_Ry_per_Bohr':float(total[-1]) if total else None,'forces_Ry_per_Bohr':[[float(x),float(y),float(z)] for _,x,y,z in rows],'input_sha256':hashlib.sha256(Path(sys.argv[1].replace('outputs/','force_inputs_fc3/').replace('.out','.in')).read_bytes()).hexdigest() if prefix.startswith('fc3') else None,'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
Path(f'reports/{prefix}.json').write_text(json.dumps(receipt,separators=(',',':'))+'\n')
print('CASE_RECEIPT',json.dumps(receipt,separators=(',',':')))
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
import json,os
rows=[json.loads(p.read_text()) for p in sorted(Path('reports').glob('fc*.json'))]
result={'document_code':'FDEP_KT2C_SI_FORCE_BATCH_012','status':'PASS','job_id':os.environ.get('JOB_ID'),'start_index':int(os.environ['START_INDEX']),'end_index':int(os.environ['END_INDEX']),'run_fc2':os.environ.get('RUN_FC2','0')=='1','case_count':len(rows),'cases':rows,'gates':{'KT2C_G14_FORCE_BATCH':'PASS'},'claim_boundary':{'force_batch':'PASS','complete_force_campaign':'NOT_EXECUTED','fc2_fc3_construction':'NOT_EXECUTED','bte':'NOT_EXECUTED','physical_claim_upgrade':'BLOCKED'}}
print('BATCH_RECEIPT_JSON',json.dumps(result,separators=(',',':')))
PY

echo 'HF_FORCE_BATCH_OVERALL: PASS'
