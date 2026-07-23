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
    "p13b_spec_cipher_1r.b64",
    "p13b_spec_cipher_2.b64",
    "p13b_spec_cipher_2r.b64",
]
EXPECTED_PART_SHA = [
    "81e95229ec89b51a5aa6acdefa9b89ddaf91b01057badc9b8e2559018c0ba58f",
    "8d1dc5b356bb385840b8291152b79c660efa480666b827a44a4f0756081a61c2",
    "0aa90d24c935cf9464f961178cb3774e7b01bb5dce53749a35189fd374d88473",
    "bf61284e485b65fee09fed6a70dac79bb94baa4ae9dac5eb7317a1cf68d428cb",
    "aba445079adeff35e5ae1be30b053a46143027cdbccbd19370f9d314d3c18b77",
    "cd8972cad02a86ece7e23d7315d63bb2a32cbfadce92316e63935d5198b716b8",
]
EXPECTED_CIPHERTEXT_SHA = "01a05b6e29550f4f1ac340a17cd13260f7f38966b21f8ac3c27a71727640b5ef"
EXPECTED_PLAINTEXT_SHA = "c164f560846e698fab2586d2a4df39b455dea74c4be695e9b2322580bbcf4bae"
KEY = base64.b64decode("CEZqC/Xj5MUYe2BKBWcCrm0GUjmkb0259+2W9dk7Hhc=")
NONCE = base64.b64decode("xBh26buYZCDrWw0q")
AAD = b"P13B_SPEC_RUNNER_V1"

parts: list[str] = []
for index, (filename, expected) in enumerate(zip(FILES, EXPECTED_PART_SHA)):
    response = requests.get(f"{BASE}/{filename}", timeout=60)
    response.raise_for_status()
    text = response.text.strip()
    digest = hashlib.sha256(text.encode()).hexdigest()
    print("TRANSFER_PART", index, len(text), digest, flush=True)
    if digest != expected:
        raise RuntimeError(f"part {index} hash mismatch")
    parts.append(text)

ciphertext = base64.b64decode("".join(parts), validate=True)
ct_sha = hashlib.sha256(ciphertext).hexdigest()
print("CIPHERTEXT_SHA", ct_sha, flush=True)
if ct_sha != EXPECTED_CIPHERTEXT_SHA:
    raise RuntimeError("ciphertext hash mismatch")

plaintext = ChaCha20Poly1305(KEY).decrypt(NONCE, ciphertext, AAD)
pt_sha = hashlib.sha256(plaintext).hexdigest()
print("RUNNER_SHA", pt_sha, flush=True)
if pt_sha != EXPECTED_PLAINTEXT_SHA:
    raise RuntimeError("runner hash mismatch")

runner = Path("/tmp/arb_formal_spectral_runner.py")
runner.write_bytes(plaintext)
compile(plaintext, str(runner), "exec")
print("AUTHENTICATED_RECONSTRUCTION_PASS", flush=True)
runpy.run_path(str(runner), run_name="__main__")
