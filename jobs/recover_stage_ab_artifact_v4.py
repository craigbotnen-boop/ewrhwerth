from __future__ import annotations

import base64, hashlib, io, json, zipfile
from huggingface_hub import fetch_job_logs

JOB_ID = "6a658d48db23d7a7ec1cdadb"
EXPECTED_ZIP_SHA256 = "29385a620ed3e2c9fb23b1ddb2638f081bb81e22e4881bdbd13674b09be277ab"
EXPECTED_RECEIPT_SHA256 = "ce12e53f115098124d9b0f64397ca87cc805c41ff8751498fc5fbe0ae4cefe9a"
START = "STAGE_AB_COMPACT_BUNDLE_B64="
STOP = "STAGE_AB_PERSISTENCE_WARNING="

items = [str(x) for x in fetch_job_logs(job_id=JOB_ID)]
start_i = next(i for i, s in enumerate(items) if START in s)
parts = [items[start_i].split(START, 1)[1]]
for s in items[start_i + 1:]:
    if STOP in s:
        parts.append(s.split(STOP, 1)[0])
        break
    parts.append(s)
b64 = "".join(parts).strip()
raw = base64.b64decode(b64, validate=True)
zip_sha = hashlib.sha256(raw).hexdigest()
if zip_sha != EXPECTED_ZIP_SHA256:
    raise RuntimeError(f"ZIP SHA mismatch: {zip_sha}, b64_len={len(b64)}, parts={len(parts)}")

with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
    if zf.testzip() is not None:
        raise RuntimeError("ZIP CRC failure")
    names = zf.namelist()
    receipt_raw = zf.read("STAGE_AB_EXECUTION_RECEIPT.json")
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    if receipt_sha != EXPECTED_RECEIPT_SHA256:
        raise RuntimeError(f"Receipt SHA mismatch: {receipt_sha}")
    receipt = json.loads(receipt_raw)
    manifest = json.loads(zf.read("RESULTS_MANIFEST.json"))
    compact = json.loads(zf.read("COMPACT_RESULTS.json"))
    summary = json.loads(zf.read("STAGE_AB_001/SCAN_SUMMARY.json"))
    source = json.loads(zf.read("STAGE_AB_001/SOURCE_BINDING.json"))
    layers = json.loads(zf.read("STAGE_AB_001/STRUCTURAL_LAYER_SUMMARY.json"))
    compact_member_checks = []
    for row in manifest:
        path = row["path"]
        if path not in names:
            continue
        data = zf.read(path)
        ok = len(data) == row["size_bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
        compact_member_checks.append({"path": path, "status": "PASS" if ok else "FAIL"})

candidate_keys = {(x["layer"], x["coordinate"]) for x in summary["controlling_candidates"]}
candidate_rows = [r for r in compact["top_energy_rows_100"] if (r["layer"], r["coordinate"]) in candidate_keys]
if len(candidate_rows) != len(candidate_keys):
    raise RuntimeError(f"Recovered {len(candidate_rows)} of {len(candidate_keys)} candidate rows")

report = {
    "recovery_status": "PASS",
    "source_job_id": JOB_ID,
    "log_event_parts": len(parts),
    "compact_zip_sha256": zip_sha,
    "compact_zip_size_bytes": len(raw),
    "zip_crc": "PASS",
    "zip_members": names,
    "execution_receipt_sha256": receipt_sha,
    "receipt": receipt,
    "scan_summary": summary,
    "source_binding_core": compact["source_binding_core"],
    "snapshot_file_count": compact["snapshot_file_count"],
    "snapshot_total_bytes": compact["snapshot_total_bytes"],
    "controlling_candidate_rows": candidate_rows,
    "layer_summary": layers,
    "compact_member_manifest_checks": compact_member_checks,
    "stage_c_executed": compact["stage_c_executed"],
}
print("RECOVERY_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
