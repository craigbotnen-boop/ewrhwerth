#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from pathlib import Path
import urllib.request
url='https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/1ef305a86c1a618b82e4c79f23687349fd2d4b05/hf_jobs/fdep_kt2c_si_stage0_001/run_displacements_004.sh'
text=urllib.request.urlopen(url,timeout=120).read().decode()
text=text.replace('--pa="F"','--pa=auto')
text=text.replace('DISPLACEMENT_EXECUTION_004','DISPLACEMENT_EXECUTION_005')
text=text.replace('displacements_004','displacements_005')
text=text.replace('kt2c_displacements_004','kt2c_displacements_005')
path=Path('/tmp/run_displacements_corrected.sh')
path.write_text(text)
path.chmod(0o755)
print('PRIMITIVE_AXIS_CORRECTION: PASS')
PY
exec bash /tmp/run_displacements_corrected.sh "${1:-/tmp/fdep_kt2c_si_displacements_005}"
