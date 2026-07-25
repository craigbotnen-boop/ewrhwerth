#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_fc_build_bte_013}"
mkdir -p "$WORK"/{outputs,reports,qe_tmp}
cd "$WORK"

python - <<'PY'
import urllib.request
url='https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/05521fdc3fcbe4b8b0aa104421061bc9f2750a7b/hf_jobs/fdep_kt2c_si_stage0_001/run_displacements_007.sh'
urllib.request.urlretrieve(url,'/tmp/run_displacements_007.sh')
PY
PATH=/opt/conda/bin:$PATH bash /tmp/run_displacements_007.sh "$WORK"

mkdir -p outputs reports qe_tmp
export OMP_NUM_THREADS=1
MPI_RANKS=8
printf 'MPI_RANKS_SELECTED=%s CPU_CORES_REPORTED=%s\n' "$MPI_RANKS" "${CPU_CORES:-UNSET}"
: > reports/force_campaign.jsonl

run_case(){
  local input="$1" output="$2" prefix="$3" expected="$4"
  mkdir -p "$WORK/qe_tmp/$prefix" "$(dirname "$output")"
  printf 'START_FORCE_CASE %s\n' "$prefix"
  mpirun --bind-to none -np "$MPI_RANKS" pw.x < "$input" > "$output"
  grep -q 'JOB DONE.' "$output"
  python - "$input" "$output" "$prefix" "$expected" <<'PY'
from pathlib import Path
import hashlib, json, re, sys
inp=Path(sys.argv[1]); out=Path(sys.argv[2]); prefix=sys.argv[3]; expected=int(sys.argv[4])
text=out.read_text(errors='replace')
all_rows=re.findall(r'atom\s+(\d+)\s+type\s+\d+\s+force\s+=\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)\s+([\-0-9.Ee+]+)',text)
energies=re.findall(r'!\s+total energy\s+=\s+([\-0-9.Ee+]+)\s+Ry',text)
totals=re.findall(r'Total force\s+=\s+([\-0-9.Ee+]+)',text)
if len(all_rows)<expected or len(all_rows)%expected or not energies or 'JOB DONE.' not in text:
    raise SystemExit(f'FORCE_PARSE_FAIL {prefix} rows={len(all_rows)} expected_multiple={expected}')
rows=all_rows[-expected:]
if [int(r[0]) for r in rows] != list(range(1,expected+1)):
    raise SystemExit(f'FINAL_FORCE_BLOCK_INDEX_FAIL {prefix}')
receipt={
  'case':prefix,'status':'PASS','atom_count':expected,
  'printed_force_blocks':len(all_rows)//expected,
  'energy_Ry':float(energies[-1]),
  'total_force_Ry_per_Bohr':float(totals[-1]) if totals else None,
  'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
  'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
}
with Path('reports/force_campaign.jsonl').open('a') as f:
    f.write(json.dumps(receipt,separators=(',',':'))+'\n')
print('CASE_RECEIPT',json.dumps(receipt,separators=(',',':')))
PY
  printf 'END_FORCE_CASE %s\n' "$prefix"
}

for idx in $(seq 1 57); do
  tag=$(printf '%05d' "$idx")
  run_case "force_inputs_fc3/fc3_${tag}.in" "outputs/fc3_${tag}.out" "fc3_${tag}" 16
done
run_case force_inputs_fc2/fc2_00001.in outputs/fc2_00001.out fc2_00001 54

python - <<'PY'
from pathlib import Path
import json
rows=[json.loads(x) for x in Path('reports/force_campaign.jsonl').read_text().splitlines() if x.strip()]
fc3=[r for r in rows if r['case'].startswith('fc3_')]
fc2=[r for r in rows if r['case'].startswith('fc2_')]
if len(fc3)!=57 or len(fc2)!=1 or any(r['status']!='PASS' for r in rows):
    raise SystemExit(f'FORCE_CAMPAIGN_COMPLETENESS_FAIL fc3={len(fc3)} fc2={len(fc2)}')
print('FORCE_CAMPAIGN_COMPLETENESS_PASS fc3=57 fc2=1')
PY

phono3py-init --cf3 outputs/fc3_*.out > reports/collect_fc3.log 2>&1
phono3py-init --cf2 outputs/fc2_00001.out > reports/collect_fc2.log 2>&1
test -s FORCES_FC3
test -s FORCES_FC2
printf 'FORCE_COLLECTION_PASS\n'

phono3py --fc-symmetry > reports/fc_build.log 2>&1
test -s fc2.hdf5
test -s fc3.hdf5
printf 'FORCE_CONSTANT_BUILD_PASS\n'

python - <<'PY'
from pathlib import Path
import hashlib, h5py, json, math, numpy as np

def read_dataset(path,key):
    with h5py.File(path,'r') as h:
        if key not in h:
            raise SystemExit(f'MISSING_DATASET {path}:{key} keys={list(h.keys())}')
        return h[key][:], list(h.keys())

fc2,keys2=read_dataset('fc2.hdf5','fc2')
fc3,keys3=read_dataset('fc3.hdf5','fc3')
finite2=bool(np.isfinite(fc2).all())
finite3=bool(np.isfinite(fc3).all())
if not finite2 or not finite3 or fc2.size==0 or fc3.size==0:
    raise SystemExit('FORCE_CONSTANT_FINITE_FAIL')
fc2_max=float(np.max(np.abs(fc2)))
fc3_max=float(np.max(np.abs(fc3)))
fc2_drift=float(np.max(np.abs(fc2.sum(axis=1)))) if fc2.ndim>=2 else math.nan
fc3_drift_axis1=float(np.max(np.abs(fc3.sum(axis=1)))) if fc3.ndim>=3 else math.nan
fc3_drift_axis2=float(np.max(np.abs(fc3.sum(axis=2)))) if fc3.ndim>=3 else math.nan
result={
 'document_code':'FDEP_KT2C_SI_FORCE_CONSTANT_BUILD_013',
 'status':'PASS',
 'fc2_shape':list(fc2.shape),'fc3_shape':list(fc3.shape),
 'fc2_hdf5_keys':keys2,'fc3_hdf5_keys':keys3,
 'fc2_all_finite':finite2,'fc3_all_finite':finite3,
 'fc2_max_abs_eV_per_A2':fc2_max,'fc3_max_abs_eV_per_A3':fc3_max,
 'translational_diagnostics':{
   'fc2_max_abs_sum_over_supercell_atom':fc2_drift,
   'fc3_max_abs_sum_axis1':fc3_drift_axis1,
   'fc3_max_abs_sum_axis2':fc3_drift_axis2,
 },
 'hashes':{
   'FORCES_FC3_sha256':hashlib.sha256(Path('FORCES_FC3').read_bytes()).hexdigest(),
   'FORCES_FC2_sha256':hashlib.sha256(Path('FORCES_FC2').read_bytes()).hexdigest(),
   'fc3_hdf5_sha256':hashlib.sha256(Path('fc3.hdf5').read_bytes()).hexdigest(),
   'fc2_hdf5_sha256':hashlib.sha256(Path('fc2.hdf5').read_bytes()).hexdigest(),
 },
 'gates':{
   'KT2C_G15_FORCE_COLLECTION':'PASS',
   'KT2C_G16_FC2_CONSTRUCTION':'PASS',
   'KT2C_G17_FC3_CONSTRUCTION':'PASS',
   'KT2C_G18_FORCE_CONSTANT_FINITE':'PASS',
 },
}
Path('reports/fc_build_result.json').write_text(json.dumps(result,indent=2)+'\n')
print('FC_BUILD_RECEIPT_JSON',json.dumps(result,separators=(',',':')))
PY

export OMP_NUM_THREADS=8
phono3py --fc3 --fc2 --mesh="9 9 9" --br --ts 300 > reports/bte_9x9x9_300K.log 2>&1
KAPPA_FILE=$(find . -maxdepth 1 -type f -name 'kappa-m*.hdf5' | sort | head -n1)
test -n "$KAPPA_FILE"
test -s "$KAPPA_FILE"

python - "$KAPPA_FILE" <<'PY'
from pathlib import Path
import hashlib, h5py, json, math, sys, numpy as np
kpath=Path(sys.argv[1])
with h5py.File(kpath,'r') as h:
    keys=list(h.keys())
    if 'kappa' not in h or 'temperature' not in h:
        raise SystemExit(f'BTE_DATASET_MISSING keys={keys}')
    kappa=np.asarray(h['kappa'][:],dtype=float)
    temp=np.asarray(h['temperature'][:],dtype=float)
    frequency=np.asarray(h['frequency'][:],dtype=float) if 'frequency' in h else None
finite=bool(np.isfinite(kappa).all() and np.isfinite(temp).all())
if not finite or kappa.size==0 or temp.size==0:
    raise SystemExit('BTE_FINITE_FAIL')
result={
 'document_code':'FDEP_KT2C_SI_BTE_SOFTWARE_PILOT_013',
 'status':'PASS',
 'mesh':[9,9,9],
 'temperatures_K':temp.tolist(),
 'kappa_W_per_mK_voigt':kappa.tolist(),
 'kappa_shape':list(kappa.shape),
 'hdf5_keys':keys,
 'frequency_min_THz':float(np.min(frequency)) if frequency is not None else None,
 'frequency_max_THz':float(np.max(frequency)) if frequency is not None else None,
 'hashes':{
   'kappa_hdf5_sha256':hashlib.sha256(kpath.read_bytes()).hexdigest(),
   'fc3_hdf5_sha256':hashlib.sha256(Path('fc3.hdf5').read_bytes()).hexdigest(),
   'fc2_hdf5_sha256':hashlib.sha256(Path('fc2.hdf5').read_bytes()).hexdigest(),
 },
 'gates':{
   'KT2C_G19_BTE_RTA_EXECUTION':'PASS',
   'KT2C_G20_BTE_OUTPUT_FINITE':'PASS',
 },
 'claim_boundary':{
   'software_pilot_force_constants':'PASS',
   'software_pilot_bte':'PASS',
   'mesh_convergence':'NOT_EXECUTED',
   'experimental_validation':'NOT_EXECUTED',
   'physical_claim_upgrade':'BLOCKED',
 },
}
Path('reports/bte_result.json').write_text(json.dumps(result,indent=2)+'\n')
print('BTE_RECEIPT_JSON',json.dumps(result,separators=(',',':')))
PY

python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path.cwd()
with tarfile.open('fdep_kt2c_fc_bte_013.tar.gz','w:gz') as tf:
    for name in ['Si_relaxed.in','phono3py_disp.yaml','FORCES_FC3','FORCES_FC2','fc3.hdf5','fc2.hdf5','reports']:
        tf.add(name)
p=Path('fdep_kt2c_fc_bte_013.tar.gz')
receipt={'archive':p.name,'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print('FINAL_ARCHIVE_RECEIPT_JSON',json.dumps(receipt,separators=(',',':')))
PY

echo 'HF_FC_BUILD_BTE_OVERALL: PASS'
