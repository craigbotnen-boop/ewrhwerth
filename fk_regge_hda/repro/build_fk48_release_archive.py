#!/usr/bin/env python3
"""Build the deterministic FK48 v1.0.1 reproducibility archive.

The scientific payload is byte-identical to Git commit
c80a4bd03349bfba8580be68cfeeccb4472687c8.  v1.0.1 changes packaging only:
all tar/gzip metadata is normalized so repeated builds produce identical bytes.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "FK48_REPRODUCIBILITY_PACKAGE_v1.0.1.tar.gz"
MANIFEST = HERE / "SHA256SUMS.txt"
SOURCE_DATE_EPOCH = 1787559124  # 2026-08-24T08:12:04Z, c80a4bd commit time

PAYLOAD = [
    "PAYLOAD_SOURCE_COMMIT.txt",
    "README.md",
    "REPRO_FREEZE_v1.md",
    "build_fk48_release_archive.py",
    "expected_fk48_outputs.json",
    "fk48_cleanroom_replication.py",
    "fk48_cleanroom_results.json",
    "fk48_stationary_canonical_reproducer.py",
    "fk48_stationary_canonical_results.json",
    "requirements_fk48.txt",
    "run_fk48_repro.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    lines = [f"{sha256(HERE / name)}  {name}\n" for name in PAYLOAD]
    MANIFEST.write_text("".join(lines), encoding="utf-8", newline="\n")


def add_bytes(tf: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    ti.mode = mode
    ti.uid = 0
    ti.gid = 0
    ti.uname = "root"
    ti.gname = "root"
    ti.mtime = SOURCE_DATE_EPOCH
    tf.addfile(ti, io.BytesIO(data))


def build() -> str:
    write_manifest()
    names = sorted(PAYLOAD + [MANIFEST.name])
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in names:
            data = (HERE / name).read_bytes()
            mode = 0o755 if name in {"run_fk48_repro.py", "build_fk48_release_archive.py"} else 0o644
            add_bytes(tf, name, data, mode)
    with ARCHIVE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            gz.write(tar_buffer.getvalue())
    return sha256(ARCHIVE)


if __name__ == "__main__":
    digest = build()
    print(f"{digest}  {ARCHIVE.name}")
