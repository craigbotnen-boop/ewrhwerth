import base64, csv, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi

BUNDLE_SHA256 = "30ac530f3c11fc8ff272800ac0968dee057ca0a6d392a8e3e01bcc3a0e7d05a2"
TRANSFER_BASE = "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/c3a137142c82ba8dd6489599902612e72639b34c/.tmp/fdep_cmc_003_bundle_b64"
TRANSFER_PARTS = ["00", "01", "02a", "02b", "02c", "03", "04", "05"]
REPO_ID = "craigbotnen/fdep-cmc-t0-local-003"
WORK = Path("/tmp/FDEP_CMC_HF_003_SINGLE")
PACKAGE = WORK / "package"
STORE = WORK / "store"
BUNDLE = WORK / "FDEP_CMC_ANTIGRAVITY_LOCAL_ACQUISITION_003_BUNDLE.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    print("RUN", json.dumps([str(x) for x in cmd]), flush=True)
    p = subprocess.run([str(x) for x in cmd], text=True)
    print("RETURN_CODE", p.returncode, flush=True)
    if p.returncode:
        raise SystemExit(p.returncode)


if WORK.exists():
    shutil.rmtree(WORK)
PACKAGE.mkdir(parents=True)
STORE.mkdir(parents=True)
free = shutil.disk_usage(WORK).free
print("FREE_BYTES", free, flush=True)
if free < 4 * 1024**3:
    raise SystemExit("BLOCKED_INSUFFICIENT_DISK")

parts = []
for name in TRANSFER_PARTS:
    url = f"{TRANSFER_BASE}/{name}.txt"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    parts.append(r.text.strip())
BUNDLE.write_bytes(base64.b64decode("".join(parts), validate=True))
actual = sha256_file(BUNDLE)
print("BUNDLE_BYTES", BUNDLE.stat().st_size, flush=True)
print("BUNDLE_SHA256", actual, flush=True)
if actual != BUNDLE_SHA256:
    raise SystemExit("BLOCKED_BUNDLE_HASH_MISMATCH")

with zipfile.ZipFile(BUNDLE) as z:
    bad = z.testzip()
    if bad:
        raise SystemExit(f"BLOCKED_ZIP_INTEGRITY:{bad}")
    z.extractall(PACKAGE)

acquire = PACKAGE / "local_fdep_cmc_acquire_unique_003.py"
validate = PACKAGE / "validate_fdep_cmc_local_store_003.py"
manifest = PACKAGE / "FDEP_CMC_T0_SNAPSHOT_HASH_BINDING_001_MODEL_MANIFEST.csv"
run([sys.executable, acquire, "--self-test"])
run([
    sys.executable, acquire,
    "--expected-manifest", manifest,
    "--store-root", STORE,
    "--workers", "1",
    "--retries", "4",
    "--allow-insecure-tls",
])
seal = STORE / "FDEP_CMC_LOCAL_PRESERVATION_SEAL_003.json"
run([
    sys.executable, validate,
    "--expected-manifest", manifest,
    "--store-root", STORE,
    "--seal-output", seal,
])

pkgdest = STORE / "acquisition_package_003"
pkgdest.mkdir(exist_ok=True)
shutil.copy2(BUNDLE, pkgdest / BUNDLE.name)
for p in PACKAGE.iterdir():
    if p.is_file():
        shutil.copy2(p, pkgdest / p.name)

local_binding = STORE / "FDEP_CMC_LOCAL_BINDING_104.csv"
remote_binding = STORE / "FDEP_CMC_HF_REMOTE_BINDING_104.csv"
with local_binding.open(newline="", encoding="utf-8") as src, remote_binding.open("w", newline="", encoding="utf-8") as dst:
    rows = list(csv.DictReader(src))
    fields = [
        "model_id", "source_archive_filename", "snapshot_member", "snapshot_sha256",
        "snapshot_bytes", "repository", "path_in_repo", "hf_uri"
    ]
    w = csv.DictWriter(dst, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        d = r["snapshot_sha256"]
        rel = f"objects/{d[:2]}/{d}.dat.gz"
        w.writerow({
            "model_id": r["model_id"],
            "source_archive_filename": r["source_archive_filename"],
            "snapshot_member": r["snapshot_member"],
            "snapshot_sha256": d,
            "snapshot_bytes": r["snapshot_bytes"],
            "repository": REPO_ID,
            "path_in_repo": rel,
            "hf_uri": f"hf://datasets/{REPO_ID}/{rel}",
        })

(STORE / "README.md").write_text("""---
pretty_name: FDEP CMC t=0 Local Preservation 003
license: other
---
# FDEP CMC t=0 Local Preservation 003

Private preservation repository for the 50 byte-distinct CMC t=0 snapshot objects representing 104 frozen model bindings. Bundle 003, frozen byte counts and SHA-256 hashes, exact ordered 62-field header, and Validator 003 were used. CMC TLS verification was disabled because its certificate was expired; no object was accepted without matching its frozen byte count and SHA-256. Graph extraction and graph execution remain unauthorized.
""", encoding="utf-8")

api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
print("HF_ACCOUNT", who.get("name"), flush=True)
api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
print("HF_REPO_CREATED", REPO_ID, flush=True)
api.upload_large_folder(
    repo_id=REPO_ID,
    repo_type="dataset",
    folder_path=STORE,
    num_workers=4,
    ignore_patterns=[".cache/**"],
    print_report=True,
    print_report_every=60,
)

info = api.repo_info(REPO_ID, repo_type="dataset", files_metadata=True)
first_commit = info.sha
files = {s.rfilename: s for s in info.siblings}
required = [
    "FDEP_CMC_LOCAL_PRESERVATION_SEAL_003.json",
    "FDEP_CMC_LOCAL_BINDING_104.csv",
    "FDEP_CMC_HF_REMOTE_BINDING_104.csv",
    "FDEP_CMC_LOCAL_ACQUISITION_AUDIT.json",
]
missing = [x for x in required if x not in files]
if missing:
    raise SystemExit("BLOCKED_REMOTE_REQUIRED_FILES_MISSING:" + json.dumps(missing))
object_paths = []
for p in (STORE / "objects").rglob("*.dat.gz"):
    rel = p.relative_to(STORE).as_posix()
    if rel not in files:
        raise SystemExit("BLOCKED_REMOTE_OBJECT_MISSING:" + rel)
    object_paths.append(rel)
if len(object_paths) != 50:
    raise SystemExit(f"BLOCKED_REMOTE_OBJECT_COUNT:{len(object_paths)}")

receipt = {
    "campaign": "FDEP_CMC_T0_HF_REMOTE_PRESERVATION_003",
    "status": "PASS_HF_REMOTE_UPLOAD_FILESET",
    "repository": REPO_ID,
    "private": True,
    "object_count": len(object_paths),
    "unique_snapshot_bytes": sum(p.stat().st_size for p in (STORE / "objects").rglob("*.dat.gz")),
    "bundle_sha256": BUNDLE_SHA256,
    "local_preservation_seal_sha256": sha256_file(seal),
    "local_binding_sha256": sha256_file(local_binding),
    "remote_binding_sha256": sha256_file(remote_binding),
    "object_upload_commit": first_commit,
    "remote_required_files_present": True,
    "remote_object_paths_present": True,
    "transport_tls_verified": False,
    "transport_caveat": "CMC certificate expired; each accepted object matched frozen byte count and SHA-256.",
    "graph_execution_authorized": False,
}
receipt_path = WORK / "FDEP_CMC_HF_REMOTE_PRESERVATION_RECEIPT_003.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
api.upload_file(
    path_or_fileobj=str(receipt_path),
    path_in_repo=receipt_path.name,
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="Add HF remote preservation receipt 003",
)
final_info = api.repo_info(REPO_ID, repo_type="dataset")
print(json.dumps({
    "FINAL_STATUS": "PASS_HF_REMOTE_PRESERVATION",
    "repository": REPO_ID,
    "object_count": len(object_paths),
    "unique_snapshot_bytes": receipt["unique_snapshot_bytes"],
    "local_preservation_seal_sha256": receipt["local_preservation_seal_sha256"],
    "remote_receipt_sha256": sha256_file(receipt_path),
    "object_upload_commit": first_commit,
    "final_repo_commit": final_info.sha,
    "graph_execution_authorized": False,
}, indent=2), flush=True)
