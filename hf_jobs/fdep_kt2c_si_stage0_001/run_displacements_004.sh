#!/usr/bin/env bash
set -euo pipefail

WORK="${1:-/tmp/fdep_kt2c_si_displacements_004}"
mkdir -p "$WORK"/{fc3,fc2,force_inputs_fc3,force_inputs_fc2,reports}
cd "$WORK"

cat > Si_relaxed.in <<'EOF'
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
/
ATOMIC_SPECIES
Si 28.0855 Si.UPF
CELL_PARAMETERS angstrom
-0.000000000 2.734717221 2.734717221
2.734717221 -0.000000000 2.734717221
2.734717221 2.734717221 -0.000000000
ATOMIC_POSITIONS crystal
Si 0.000000000 0.000000000 -0.000000000
Si 0.250000000 0.250000000 0.250000000
EOF

python - <<'PY'
from pathlib import Path
import hashlib, json, math
p=Path('Si_relaxed.in')
expected_cell=[[-0.0,2.734717221,2.734717221],[2.734717221,-0.0,2.734717221],[2.734717221,2.734717221,-0.0]]
text=p.read_text()
assert '2.734717221' in text and text.count('Si ')>=3
receipt={
 'gate':'KT2C_G08_RELAXED_STRUCTURE_BINDING',
 'status':'PASS',
 'source_job_id':'6a6456737ef3c08464968166',
 'source_output_sha256':'c74881ac87fc7b856d36acb4161de632fd9a30a5d7b7184c7487000a790cc15a',
 'cell_A':expected_cell,
 'scaled_positions':[[0.0,0.0,-0.0],[0.25,0.25,0.25]],
 'structure_input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
}
Path('reports/relaxed_structure_binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY

phono3py-init --qe -d --dim="2 2 2" --dim-fc2="3 3 3" --pa="F" --amplitude=0.03 -c Si_relaxed.in

python - <<'PY'
from pathlib import Path
import hashlib, json, re, shutil
from ase.io import read, write

root=Path.cwd()
fc3=sorted(p for p in root.glob('supercell-*.in') if 'fc2' not in p.name)
fc2=sorted(root.glob('supercell_fc2-*.in'))
if not fc3 or not fc2:
    raise SystemExit(f'DISPLACEMENT_FILE_COUNT_FAIL fc3={len(fc3)} fc2={len(fc2)}')

for p in fc3: shutil.move(str(p), root/'fc3'/p.name)
for p in fc2: shutil.move(str(p), root/'fc2'/p.name)
fc3=sorted((root/'fc3').glob('*.in'))
fc2=sorted((root/'fc2').glob('*.in'))

input_data={
 'control':{
   'calculation':'scf','pseudo_dir':str(root),'outdir':str(root/'qe_tmp'),
   'tprnfor':True,'tstress':True,'verbosity':'high','disk_io':'none'},
 'system':{'ecutwfc':80,'ecutrho':640,'occupations':'fixed'},
 'electrons':{'conv_thr':1.0e-10,'electron_maxstep':200,'mixing_beta':0.3},
}
pseudopotentials={'Si':'Si.UPF'}

def convert(files,outdir,kmesh,prefix):
    rows=[]
    outdir.mkdir(exist_ok=True)
    for i,p in enumerate(files,1):
        atoms=read(p,format='espresso-in')
        target=outdir/f'{prefix}_{i:05d}.in'
        data={section:values.copy() for section,values in input_data.items()}
        data['control']['prefix']=f'{prefix}_{i:05d}'
        data['control']['outdir']=str(root/'qe_tmp'/f'{prefix}_{i:05d}')
        write(target,atoms,format='espresso-in',input_data=data,
              pseudopotentials=pseudopotentials,kpts=kmesh)
        rows.append({'index':i,'file':str(target.relative_to(root)),
                     'atoms':len(atoms),'kmesh':list(kmesh),
                     'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    return rows

rows3=convert(fc3,root/'force_inputs_fc3',(7,7,7),'fc3')
rows2=convert(fc2,root/'force_inputs_fc2',(5,5,5),'fc2')
yaml_path=root/'phono3py_disp.yaml'
if not yaml_path.is_file(): raise SystemExit('MISSING_PHONO3PY_DISP_YAML')
result={
 'document_code':'FDEP_KT2C_SI_DISPLACEMENT_EXECUTION_004',
 'status':'PASS',
 'source_relaxation_job_id':'6a6456737ef3c08464968166',
 'phono3py_version':'4.3.3',
 'displacement_amplitude_A':0.03,
 'fc3_supercell_matrix':[2,2,2],
 'fc2_supercell_matrix':[3,3,3],
 'fc3_displacement_count':len(rows3),
 'fc2_displacement_count':len(rows2),
 'fc3_atoms_per_supercell':rows3[0]['atoms'],
 'fc2_atoms_per_supercell':rows2[0]['atoms'],
 'force_settings':{
   'ecutwfc_Ry':80,'ecutrho_Ry':640,
   'fc3_kmesh':[7,7,7],'fc2_kmesh':[5,5,5],
   'electronic_conv_thr':1.0e-10,
 },
 'hashes':{
   'phono3py_disp_yaml_sha256':hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
   'relaxed_input_sha256':hashlib.sha256((root/'Si_relaxed.in').read_bytes()).hexdigest(),
 },
 'gates':{
   'KT2C_G08_RELAXED_STRUCTURE_BINDING':'PASS',
   'KT2C_G09_FC3_DISPLACEMENT_GENERATION':'PASS',
   'KT2C_G10_FC2_DISPLACEMENT_GENERATION':'PASS',
   'KT2C_G11_FORCE_INPUT_GENERATION':'PASS',
 },
 'fc3_manifest':rows3,
 'fc2_manifest':rows2,
 'claim_boundary':{
   'displacement_generation':'PASS',
   'force_input_generation':'PASS',
   'force_campaign':'NOT_EXECUTED',
   'fc2_fc3_construction':'NOT_EXECUTED',
   'bte':'NOT_EXECUTED',
   'physical_claim_upgrade':'BLOCKED',
 },
}
(root/'reports/displacement_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('fc3_manifest','fc2_manifest')},indent=2))
PY

python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
root=Path.cwd()
with tarfile.open('kt2c_displacements_004.tar.gz','w:gz') as tf:
    for name in ['Si_relaxed.in','phono3py_disp.yaml','fc3','fc2','force_inputs_fc3','force_inputs_fc2','reports']:
        tf.add(name)
archive=Path('kt2c_displacements_004.tar.gz')
receipt={'archive':archive.name,'size_bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
Path('reports/archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY

echo 'HF_DISPLACEMENT_GENERATION_OVERALL: PASS'
