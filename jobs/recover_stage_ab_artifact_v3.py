from __future__ import annotations

import base64, hashlib, json, re
from huggingface_hub import fetch_job_logs

JOB_ID = "6a658d48db23d7a7ec1cdadb"
marker = "STAGE_AB_COMPACT_BUNDLE_B64="
items = list(fetch_job_logs(job_id=JOB_ID))
print("LOG_ITEM_COUNT=" + str(len(items)))
for i, item in enumerate(items):
    s = str(item)
    if marker not in s:
        continue
    tail = s.split(marker, 1)[1]
    print("MATCH_ITEM_INDEX=" + str(i))
    print("MATCH_ITEM_TYPE=" + type(item).__name__)
    print("TAIL_LENGTH=" + str(len(tail)))
    print("TAIL_HEAD=" + repr(tail[:160]))
    print("TAIL_END=" + repr(tail[-300:]))
    candidates = re.findall(r"[A-Za-z0-9+/=]{1000,}", tail)
    print("REGEX_CANDIDATE_LENGTHS=" + json.dumps([len(x) for x in candidates]))
    for j, candidate in enumerate(candidates):
        try:
            raw = base64.b64decode(candidate, validate=True)
            print(f"CANDIDATE_{j}_RAW_LENGTH={len(raw)}")
            print(f"CANDIDATE_{j}_SHA256={hashlib.sha256(raw).hexdigest()}")
        except Exception as exc:
            print(f"CANDIDATE_{j}_ERROR={exc!r}")
