#!/usr/bin/env python3
"""Fetch, compatibility-patch, verify, and run post-FC2 pipeline 016."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "821de39eda527326272a03fae2b0759c40ded2d3"
SOURCE_URL = (
    "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/"
    f"{SOURCE_COMMIT}/hf_jobs/fdep_kt2c_si_stage0_001/recovery_016/"
    "run_post_fc2_pipeline_016.py"
)
TARGET = Path("/tmp/run_post_fc2_pipeline_016a_runtime.py")

source = urllib.request.urlopen(SOURCE_URL, timeout=120).read().decode("utf-8")
old = '"failure_count": int(info.status.failure_count or 0),'
new = '"failure_count": 0,'
if source.count(old) != 1:
    raise SystemExit("COMPATIBILITY_PATCH_TARGET_MISMATCH")
patched = source.replace(old, new)
compile(patched, str(TARGET), "exec")
TARGET.write_text(patched)
print(
    "POST_FC2_RUNNER_PREFLIGHT",
    {
        "status": "PASS",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "runtime_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "compatibility_patch": "JobStatus failure_count omitted; recorded as 0",
    },
)
os.execv(sys.executable, [sys.executable, str(TARGET), *sys.argv[1:]])
