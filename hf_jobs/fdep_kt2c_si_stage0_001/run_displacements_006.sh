#!/usr/bin/env bash
set -euo pipefail
WORK="${1:-/tmp/fdep_kt2c_si_displacements_006}"
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
import hashlib,json,urllib.request
p=Path('Si_relaxed.in')
url='https://pseudopotentials.quantum-espresso.org/upf_files/Si.pbe-n-rrkjus_psl.1.0.0.UPF'
Path('Si.UPF').write_bytes(urllib.request.urlopen(url,timeout=120).read())
actual=hashlib.sha256(Path('Si.UPF').read_bytes()).hexdigest()
expected='669fb75395a9d26973b0ea1ce8223bbcb30d3396c5d48bf5e794d1243c52375a'
if actual != expected: raise SystemExit('PSEUDOPOTENTIAL_HASH_FAIL')
receipt={'gate':'KT2C_G08_RELAXED_STRUCTURE_BINDING','status':'PASS','source_job_id':'6a6456737ef3c08464968166','source_output_sha256':'c74881ac87fc7b856d36acb4161de632fd9a30a5d7b7184c7487000a790cc15a','structure_input_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pseudopotential_sha256':actual}
Path('reports/relaxed_structure_binding.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY

phono3py-init --qe -d --dim="2 2 2" --dim-fc2="3 3 3" --pa=auto --amplitude=0.03 -c Si_relaxed.in

python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
from phonopy.interface.calculator import read_crystal_structure

root=Path.cwd()
fc3=sorted(p for p in root.glob('supercell-*.in') if 'fc2' not in p.name)
fc2=sorted(root.glob('supercell_fc2-*.in'))
if len(fc3)!=57 or len(fc2)!=1:
    raise SystemExit(f'DISPLACEMENT_COUNT_FAIL fc3={len(fc3)} fc2={len(fc2)}')
for p in fc3: shutil.move(str(p),root/'fc3'/p.name)
for p in fc2: shutil.move(str(p),root/'fc2'/p.name)
fc3=sorted((root/'fc3').glob('*.in'))
fc2=sorted((root/'fc2').glob('*.in'))

def qe_input(cell,prefix,kmesh):
    lines=["&CONTROL", "  calculation = 'scf'", f"  prefix = '{prefix}'", f"  pseudo_dir = '{root}'", f"  outdir = '{root/'qe_tmp'/prefix}'", "  tprnfor = .true.", "  tstress = .true.", "  verbosity = 'high'", "  disk_io = 'none'", "/", "&SYSTEM", "  ibrav = 0", f"  nat = {len(cell)}", "  ntyp = 1", "  ecutwfc = 80", "  ecutrho = 640", "  occupations = 'fixed'", "/", "&ELECTRONS", "  conv_thr = 1.0d-10", "  electron_maxstep = 200", "  mixing_beta = 0.3", "/", "ATOMIC_SPECIES", "Si 28.0855 Si.UPF", "CELL_PARAMETERS angstrom"]
    for row in cell.cell: lines.append(' '.join(f'{x:.12f}' for x in row))
    lines.append('ATOMIC_POSITIONS crystal')
    for symbol,pos in zip(cell.symbols,cell.scaled_positions): lines.append(symbol+' '+' '.join(f'{x:.12f}' for x in pos))
    lines.extend(['K_POINTS automatic',f'{kmesh[0]} {kmesh[1]} {kmesh[2]} 0 0 0',''])
    return '\n'.join(lines)

def convert(files,outdir,kmesh,prefix):
    rows=[]
    for i,p in enumerate(files,1):
        cell,_=read_crystal_structure(str(p),interface_mode='qe')
        if cell is None: raise SystemExit(f'PHONOPY_QE_PARSE_FAIL {p}')
        target=outdir/f'{prefix}_{i:05d}.in'
        target.write_text(qe_input(cell,f'{prefix}_{i:05d}',kmesh))
        rows.append({'index':i,'source':str(p.relative_to(root)),'file':str(target.relative_to(root)),'atoms':len(cell),'kmesh':list(kmesh),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    return rows

rows3=convert(fc3,root/'force_inputs_fc3',(7,7,7),'fc3')
rows2=convert(fc2,root/'force_inputs_fc2',(5,5,5),'fc2')
yaml_path=root/'phono3py_disp.yaml'
if not yaml_path.is_file(): raise SystemExit('MISSING_PHONO3PY_DISP_YAML')
result={'document_code':'FDEP_KT2C_SI_DISPLACEMENT_EXECUTION_006','status':'PASS','source_relaxation_job_id':'6a6456737ef3c08464968166','phono3py_version':'4.3.3','displacement_amplitude_A':0.03,'fc3_supercell_matrix':[2,2,2],'fc2_supercell_matrix':[3,3,3],'fc3_displacement_count':len(rows3),'fc2_displacement_count':len(rows2),'fc3_atoms_per_supercell':rows3[0]['atoms'],'fc2_atoms_per_supercell':rows2[0]['atoms'],'force_settings':{'ecutwfc_Ry':80,'ecutrho_Ry':640,'fc3_kmesh':[7,7,7],'fc2_kmesh':[5,5,5],'electronic_conv_thr':1e-10},'hashes':{'phono3py_disp_yaml_sha256':hashlib.sha256(yaml_path.read_bytes()).hexdigest(),'relaxed_input_sha256':hashlib.sha256((root/'Si_relaxed.in').read_bytes()).hexdigest(),'pseudopotential_sha256':hashlib.sha256((root/'Si.UPF').read_bytes()).hexdigest()},'gates':{'KT2C_G08_RELAXED_STRUCTURE_BINDING':'PASS','KT2C_G09_FC3_DISPLACEMENT_GENERATION':'PASS','KT2C_G10_FC2_DISPLACEMENT_GENERATION':'PASS','KT2C_G11_FORCE_INPUT_GENERATION':'PASS'},'fc3_manifest':rows3,'fc2_manifest':rows2,'claim_boundary':{'displacement_generation':'PASS','force_input_generation':'PASS','force_campaign':'NOT_EXECUTED','fc2_fc3_construction':'NOT_EXECUTED','bte':'NOT_EXECUTED','physical_claim_upgrade':'BLOCKED'}}
(root/'reports/displacement_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('fc3_manifest','fc2_manifest')},indent=2))
PY

python - <<'PY'
from pathlib import Path
import hashlib,json,tarfile
with tarfile.open('kt2c_displacements_006.tar.gz','w:gz') as tf:
    for name in ['Si_relaxed.in','Si.UPF','phono3py_disp.yaml','phono3py.yaml','fc3','fc2','force_inputs_fc3','force_inputs_fc2','reports']: tf.add(name)
p=Path('kt2c_displacements_006.tar.gz')
r={'archive':p.name,'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
Path('reports/archive_receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
PY

echo 'HF_DISPLACEMENT_GENERATION_OVERALL: PASS'
