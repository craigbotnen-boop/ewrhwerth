from __future__ import annotations

import base64
import hashlib
import runpy
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

BASE = "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/three-paper-suite/.tmp"
FILES = [
    "p13b_spec_cipher_0.b64",
    "p13b_spec_cipher_0r.b64",
    "p13b_spec_cipher_1.b64",
    "p13b_spec_cipher_2.b64",
]
PART_SHA = [
    "81e95229ec89b51a5aa6acdefa9b89ddaf91b01057badc9b8e2559018c0ba58f",
    "8d1dc5b356bb385840b8291152b79c660efa480666b827a44a4f0756081a61c2",
    "e35a4e93f337961171660b6e198fe1f70afcc3d713b243f4443ef85283b77361",
    "4b927c8d9f4c2faaf5972c99ef33eb0b45968384ba1f7a749fe746eb30ec1a1b",
]
CT_SHA = "01a05b6e29550f4f1ac340a17cd13260f7f38966b21f8ac3c27a71727640b5ef"
V1_SHA = "c164f560846e698fab2586d2a4df39b455dea74c4be695e9b2322580bbcf4bae"
V2_SHA = "d77f2df6b50228958d465ce0922194c623de34dce96603864a26e2136761d9ea"
KEY = base64.b64decode("CEZqC/Xj5MUYe2BKBWcCrm0GUjmkb0259+2W9dk7Hhc=")
NONCE = base64.b64decode("xBh26buYZCDrWw0q")
AAD = b"P13B_SPEC_RUNNER_V1"

parts = []
for i, (name, expected) in enumerate(zip(FILES, PART_SHA)):
    text = requests.get(f"{BASE}/{name}", timeout=60).text.strip()
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("TRANSFER_PART", i, len(text), digest, flush=True)
    if digest != expected:
        raise RuntimeError(f"part {i} hash mismatch")
    parts.append(text)
ct = base64.b64decode("".join(parts), validate=True)
if hashlib.sha256(ct).hexdigest() != CT_SHA:
    raise RuntimeError("ciphertext hash mismatch")
source = ChaCha20Poly1305(KEY).decrypt(NONCE, ct, AAD)
if hashlib.sha256(source).hexdigest() != V1_SHA:
    raise RuntimeError("v1 runner hash mismatch")
text = source.decode()
old = '''    # exact Hermitian hull
    for i in range(D):
      A[i,i]=A[i,i].real
      for j in range(i):
        m=(A[i,j]+A[j,i].conjugate())/2; r=max(Rad[i,j]+abs(A[i,j]-m),Rad[j,i]+abs(A[j,i].conjugate()-m))
        A[i,j]=m; A[j,i]=m.conjugate(); Rad[i,j]=Rad[j,i]=math.nextafter(r,math.inf)
'''
new = '''    # Exact Hessian quadratic form: Hermitian symmetrization.
    # The anti-Hermitian midpoint part contributes identically zero to the
    # real quadratic form and is therefore projected out, not charged as
    # interval uncertainty. Independent Arb radii average under the linear
    # map H -> (H+H*)/2.
    for i in range(D):
      A[i,i]=A[i,i].real
      for j in range(i):
        m=(A[i,j]+A[j,i].conjugate())/2
        r=math.nextafter((Rad[i,j]+Rad[j,i])/2,math.inf)
        A[i,j]=m; A[j,i]=m.conjugate(); Rad[i,j]=Rad[j,i]=r
'''
if text.count(old) != 1:
    raise RuntimeError("audited patch target count is not one")
patched = text.replace(old, new).encode()
sha = hashlib.sha256(patched).hexdigest()
print("V1_RUNNER_SHA", V1_SHA, flush=True)
print("PATCHED_V2_RUNNER_SHA", sha, flush=True)
if sha != V2_SHA:
    raise RuntimeError("patched v2 hash mismatch")
runner = Path("/tmp/arb_formal_spectral_runner_v2.py")
runner.write_bytes(patched)
compile(patched, str(runner), "exec")
print("AUTHENTICATED_PATCHED_RECONSTRUCTION_PASS", flush=True)
runpy.run_path(str(runner), run_name="__main__")
