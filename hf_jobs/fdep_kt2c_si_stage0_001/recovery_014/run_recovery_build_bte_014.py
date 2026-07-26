#!/usr/bin/env python3
"""Recover authenticated KT2C force receipts, build FC2/FC3, and run BTE pilot.

No Quantum ESPRESSO calculations are executed by this runner.
All outputs are uploaded to a private Hugging Face dataset repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from huggingface_hub import HfApi

ROOT = Path(os.environ.get("KT2C_WORK", "/tmp/fdep_kt2c_recovery_014")).resolve()
ART = ROOT / "artifacts"
REPORTS = ART / "reports"
RECEIPTS = ART / "receipts"
REPO_ID = os.environ.get("KT2C_ARTIFACT_REPO", "craigbotnen/fdep-kt2c-si-artifacts")
JOB_ID = os.environ.get("JOB_ID", "local")
PATH_IN_REPO = f"recovery_014/{JOB_ID}"
RAW_BASE = "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/main/hf_jobs/fdep_kt2c_si_stage0_001/recovery_014"
EXPECTED_DISP_SHA = "3b4e2edd7f51af178acdaf277d550798ff41ffe1b54d34862de130cc0cc92962"

RECEIPT_FILES = [
    "force_pilot_fc3_00001_fc2_00001.json",
    "force_batch_02_15.json",
    "force_batch_16_29.json",
    "force_batch_30_43.json",
    "force_batch_44_57.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run_stream(command: list[str], log_path: Path, timeout: int | None = None) -> None:
    print("RUN", json.dumps(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait(timeout=timeout)
        except Exception:
            process.kill()
            process.wait()
            raise
    if return_code != 0:
        raise RuntimeError(f"COMMAND_FAIL return_code={return_code} command={command}")


def make_displacements() -> None:
    structure = """&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
/
ATOMIC_SPECIES
Si 28.0855 Si.UPF
CELL_PARAMETERS angstrom
-0.000000000 2.734717221 2.734717221
2.734717221 -0.000000000 2.734717221
2.734717221 2.734717221 -0.000000000
ATOMIC_POSITIONS crystal
Si 0.000000000 0.000000000 -0.000000000
Si 0.250000000 0.250000000 0.250000000
"""
    (ROOT / "Si_relaxed.in").write_text(structure)
    run_stream(
        [
            "phono3py-init",
            "--qe",
            "-d",
            "--dim=2 2 2",
            "--dim-fc2=3 3 3",
            "--pa=auto",
            "--amplitude=0.03",
            "-c",
            "Si_relaxed.in",
        ],
        REPORTS / "displacement_generation.log",
        timeout=900,
    )
    disp = ROOT / "phono3py_disp.yaml"
    if not disp.is_file():
        raise RuntimeError("MISSING phono3py_disp.yaml")
    actual = sha256(disp)
    print(f"DISPLACEMENT_SHA256={actual}", flush=True)
    if actual != EXPECTED_DISP_SHA:
        raise RuntimeError(
            f"DISPLACEMENT_HASH_FAIL expected={EXPECTED_DISP_SHA} actual={actual}"
        )
    shutil.copy2(ROOT / "Si_relaxed.in", ART / "Si_relaxed.in")
    shutil.copy2(disp, ART / "phono3py_disp.yaml")
    if (ROOT / "phono3py.yaml").is_file():
        shutil.copy2(ROOT / "phono3py.yaml", ART / "phono3py.yaml")


def download_receipts() -> list[dict[str, Any]]:
    all_cases: list[dict[str, Any]] = []
    receipt_hashes: dict[str, str] = {}
    for name in RECEIPT_FILES:
        target = RECEIPTS / name
        urllib.request.urlretrieve(f"{RAW_BASE}/{name}", target)
        data = json.loads(target.read_text())
        if data.get("status") != "PASS":
            raise RuntimeError(f"RECEIPT_STATUS_FAIL {name}")
        cases = data.get("cases")
        if not isinstance(cases, list):
            raise RuntimeError(f"RECEIPT_CASES_MISSING {name}")
        all_cases.extend(cases)
        receipt_hashes[name] = sha256(target)
    write_json(REPORTS / "receipt_file_hashes.json", receipt_hashes)
    return all_cases


def validate_and_write_forces(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    for case in cases:
        name = case.get("case")
        if not isinstance(name, str) or name in by_name:
            raise RuntimeError(f"DUPLICATE_OR_INVALID_CASE {name}")
        by_name[name] = case

    expected_fc3 = [f"fc3_{i:05d}" for i in range(1, 58)]
    missing = [name for name in expected_fc3 if name not in by_name]
    extra_fc3 = sorted(
        name for name in by_name if name.startswith("fc3_") and name not in expected_fc3
    )
    if missing or extra_fc3 or "fc2_00001" not in by_name:
        raise RuntimeError(
            f"FORCE_RECEIPT_COMPLETENESS_FAIL missing={missing} extra={extra_fc3} "
            f"fc2_present={'fc2_00001' in by_name}"
        )

    forces3 = np.asarray(
        [by_name[name]["forces_Ry_per_Bohr"] for name in expected_fc3], dtype=float
    )
    forces2 = np.asarray([by_name["fc2_00001"]["forces_Ry_per_Bohr"]], dtype=float)
    if forces3.shape != (57, 16, 3):
        raise RuntimeError(f"FC3_FORCE_SHAPE_FAIL {forces3.shape}")
    if forces2.shape != (1, 54, 3):
        raise RuntimeError(f"FC2_FORCE_SHAPE_FAIL {forces2.shape}")
    if not np.isfinite(forces3).all() or not np.isfinite(forces2).all():
        raise RuntimeError("FORCE_FINITE_FAIL")
    if any(by_name[name].get("status") != "PASS" for name in expected_fc3):
        raise RuntimeError("FC3_RECEIPT_STATUS_FAIL")
    if by_name["fc2_00001"].get("status") != "PASS":
        raise RuntimeError("FC2_RECEIPT_STATUS_FAIL")

    np.savetxt(ROOT / "FORCES_FC3", forces3.reshape(-1, 3), fmt="%.12e")
    np.savetxt(ROOT / "FORCES_FC2", forces2.reshape(-1, 3), fmt="%.12e")
    shutil.copy2(ROOT / "FORCES_FC3", ART / "FORCES_FC3")
    shutil.copy2(ROOT / "FORCES_FC2", ART / "FORCES_FC2")

    manifest = {
        "document_code": "FDEP_KT2C_SI_FORCE_RECOVERY_014",
        "status": "PASS",
        "source_jobs": sorted(
            {
                "6a64f805db23d7a7ec1cca6d",
                "6a6512b77ef3c084649691af",
                "6a6512c57ef3c084649691b2",
                "6a6512d6db23d7a7ec1cd13b",
                "6a6512e2db23d7a7ec1cd13f",
            }
        ),
        "fc3_case_count": 57,
        "fc2_case_count": 1,
        "fc3_force_shape": list(forces3.shape),
        "fc2_force_shape": list(forces2.shape),
        "fc3_all_finite": bool(np.isfinite(forces3).all()),
        "fc2_all_finite": bool(np.isfinite(forces2).all()),
        "hashes": {
            "FORCES_FC3_sha256": sha256(ROOT / "FORCES_FC3"),
            "FORCES_FC2_sha256": sha256(ROOT / "FORCES_FC2"),
            "phono3py_disp_yaml_sha256": sha256(ROOT / "phono3py_disp.yaml"),
        },
        "gates": {
            "KT2C_G14_FORCE_CAMPAIGN": "PASS",
            "KT2C_G15_DURABLE_FORCE_RECOVERY": "PASS",
        },
    }
    write_json(REPORTS / "force_recovery_result.json", manifest)
    print("FORCE_RECOVERY_RECEIPT_JSON", json.dumps(manifest, separators=(",", ":")), flush=True)
    return manifest


def build_force_constants() -> dict[str, Any]:
    import phono3py
    from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

    print(f"PHONO3PY_VERSION={phono3py.__version__}", flush=True)
    ph3 = phono3py.load(
        phono3py_yaml=str(ROOT / "phono3py_disp.yaml"),
        forces_fc3_filename=(str(ROOT / "FORCES_FC3"), str(ROOT / "phono3py_disp.yaml")),
        forces_fc2_filename=(str(ROOT / "FORCES_FC2"), str(ROOT / "phono3py_disp.yaml")),
        calculator="qe",
        produce_fc=True,
        is_symmetry=True,
        symmetrize_fc=True,
        is_compact_fc=False,
        is_nac=False,
        log_level=1,
    )
    if ph3.fc2 is None or ph3.fc3 is None:
        raise RuntimeError(f"FORCE_CONSTANT_NONE fc2={ph3.fc2 is None} fc3={ph3.fc3 is None}")
    fc2 = np.asarray(ph3.fc2, dtype=float)
    fc3 = np.asarray(ph3.fc3, dtype=float)
    print(f"FC2_SHAPE={fc2.shape} FC3_SHAPE={fc3.shape}", flush=True)
    if not np.isfinite(fc2).all() or not np.isfinite(fc3).all():
        raise RuntimeError("FORCE_CONSTANT_FINITE_FAIL")

    fc2_path = ROOT / "fc2.hdf5"
    fc3_path = ROOT / "fc3.hdf5"
    write_fc2_to_hdf5(fc2, filename=str(fc2_path), compression="gzip")
    write_fc3_to_hdf5(fc3, filename=str(fc3_path), compression="gzip")
    if not fc2_path.is_file() or not fc3_path.is_file():
        raise RuntimeError("FORCE_CONSTANT_WRITE_FAIL")
    shutil.copy2(fc2_path, ART / "fc2.hdf5")
    shutil.copy2(fc3_path, ART / "fc3.hdf5")

    fc2_drift = float(np.max(np.abs(fc2.sum(axis=1))))
    fc3_drift = {
        "axis0": float(np.max(np.abs(fc3.sum(axis=0)))),
        "axis1": float(np.max(np.abs(fc3.sum(axis=1)))),
        "axis2": float(np.max(np.abs(fc3.sum(axis=2)))),
    }
    fc2_pair_residual = float(np.max(np.abs(fc2 - fc2.transpose(1, 0, 3, 2))))
    fc3_swap01 = float(np.max(np.abs(fc3 - fc3.transpose(1, 0, 2, 4, 3, 5))))
    fc3_swap12 = float(np.max(np.abs(fc3 - fc3.transpose(0, 2, 1, 3, 5, 4))))

    result = {
        "document_code": "FDEP_KT2C_SI_FORCE_CONSTANT_BUILD_014",
        "status": "PASS",
        "phono3py_version": phono3py.__version__,
        "fc2_shape": list(fc2.shape),
        "fc3_shape": list(fc3.shape),
        "fc2_all_finite": bool(np.isfinite(fc2).all()),
        "fc3_all_finite": bool(np.isfinite(fc3).all()),
        "fc2_max_abs_eV_per_A2": float(np.max(np.abs(fc2))),
        "fc3_max_abs_eV_per_A3": float(np.max(np.abs(fc3))),
        "diagnostics_not_thresholded": {
            "fc2_max_abs_sum_over_atom_axis": fc2_drift,
            "fc3_max_abs_sum_over_atom_axes": fc3_drift,
            "fc2_pair_permutation_residual": fc2_pair_residual,
            "fc3_swap01_residual": fc3_swap01,
            "fc3_swap12_residual": fc3_swap12,
        },
        "hashes": {
            "fc2_hdf5_sha256": sha256(fc2_path),
            "fc3_hdf5_sha256": sha256(fc3_path),
        },
        "gates": {
            "KT2C_G16_FC2_CONSTRUCTION": "PASS",
            "KT2C_G17_FC3_CONSTRUCTION": "PASS",
            "KT2C_G18_FORCE_CONSTANT_FINITE": "PASS",
            "KT2C_G18A_STRICT_ASR_THRESHOLD": "NOT_EXECUTED",
            "KT2C_G18B_ZONE_WIDE_STABILITY": "NOT_EXECUTED",
        },
    }
    write_json(REPORTS / "force_constant_build_result.json", result)
    print("FC_BUILD_RECEIPT_JSON", json.dumps(result, separators=(",", ":")), flush=True)
    return result


def run_bte_pilot() -> dict[str, Any]:
    command = [
        "phono3py",
        str(ROOT / "phono3py_disp.yaml"),
        "--fc3",
        "--fc2",
        "--mesh=9 9 9",
        "--br",
        "--ts=300",
    ]
    run_stream(command, REPORTS / "bte_9x9x9_300K.log", timeout=7200)
    kappa_files = sorted(ROOT.glob("kappa-m*.hdf5"))
    if not kappa_files:
        raise RuntimeError("BTE_KAPPA_FILE_MISSING")
    kappa_path = kappa_files[0]
    with h5py.File(kappa_path, "r") as handle:
        keys = sorted(handle.keys())
        if "kappa" not in handle or "temperature" not in handle:
            raise RuntimeError(f"BTE_DATASET_MISSING keys={keys}")
        kappa = np.asarray(handle["kappa"][:], dtype=float)
        temperatures = np.asarray(handle["temperature"][:], dtype=float)
        frequency = np.asarray(handle["frequency"][:], dtype=float) if "frequency" in handle else None
        gamma = np.asarray(handle["gamma"][:], dtype=float) if "gamma" in handle else None
    if not np.isfinite(kappa).all() or not np.isfinite(temperatures).all():
        raise RuntimeError("BTE_FINITE_FAIL")
    shutil.copy2(kappa_path, ART / kappa_path.name)
    result = {
        "document_code": "FDEP_KT2C_SI_BTE_SOFTWARE_PILOT_014",
        "status": "PASS",
        "mesh": [9, 9, 9],
        "temperatures_K": temperatures.tolist(),
        "kappa_W_per_mK_voigt": kappa.tolist(),
        "kappa_shape": list(kappa.shape),
        "hdf5_keys": keys,
        "frequency_min_THz": float(np.min(frequency)) if frequency is not None else None,
        "frequency_max_THz": float(np.max(frequency)) if frequency is not None else None,
        "gamma_all_finite": bool(np.isfinite(gamma).all()) if gamma is not None else None,
        "hashes": {"kappa_hdf5_sha256": sha256(kappa_path)},
        "gates": {
            "KT2C_G19_BTE_RTA_EXECUTION": "PASS",
            "KT2C_G20_BTE_OUTPUT_FINITE": "PASS",
            "KT2C_G21_BTE_MESH_CONVERGENCE": "NOT_EXECUTED",
            "KT2C_G22_BENCHMARK_VALIDATION": "NOT_EXECUTED",
        },
        "claim_boundary": {
            "software_pilot_force_constants": "PASS",
            "software_pilot_bte": "PASS",
            "strict_stability_gate": "NOT_EXECUTED",
            "mesh_convergence": "NOT_EXECUTED",
            "experimental_validation": "NOT_EXECUTED",
            "physical_claim_upgrade": "BLOCKED",
        },
    }
    write_json(REPORTS / "bte_result.json", result)
    print("BTE_RECEIPT_JSON", json.dumps(result, separators=(",", ":")), flush=True)
    return result


def create_archive_and_manifest(overall_status: str, error: str | None) -> tuple[Path, dict[str, Any]]:
    manifest_files: dict[str, dict[str, Any]] = {}
    for path in sorted(ART.rglob("*")):
        if path.is_file():
            manifest_files[str(path.relative_to(ART))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "document_code": "FDEP_KT2C_SI_RECOVERY_PACKET_014",
        "status": overall_status,
        "job_id": JOB_ID,
        "artifact_repo": REPO_ID,
        "path_in_repo": PATH_IN_REPO,
        "error": error,
        "files": manifest_files,
        "physical_claim_upgrade": "BLOCKED",
    }
    write_json(ART / "MANIFEST.json", manifest)
    archive = ROOT / "FDEP_KT2C_SI_RECOVERY_PACKET_014.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(ART, arcname="FDEP_KT2C_SI_RECOVERY_PACKET_014")
    archive_receipt = {
        "archive": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256(archive),
    }
    write_json(ART / "ARCHIVE_RECEIPT.json", archive_receipt)
    shutil.copy2(archive, ART / archive.name)
    return archive, archive_receipt


def upload_artifacts() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN_MISSING")
    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(ART),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=PATH_IN_REPO,
        commit_message=f"Upload KT2C recovery 014 artifacts from job {JOB_ID}",
    )
    return str(commit)


def main() -> int:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ART.mkdir(parents=True)
    REPORTS.mkdir(parents=True)
    RECEIPTS.mkdir(parents=True)
    os.chdir(ROOT)
    overall_status = "FAIL"
    error_text: str | None = None
    exit_code = 1
    try:
        make_displacements()
        cases = download_receipts()
        validate_and_write_forces(cases)
        build_force_constants()
        run_bte_pilot()
        overall_status = "PASS"
        exit_code = 0
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        traceback_text = traceback.format_exc()
        print("RECOVERY_EXCEPTION", error_text, flush=True)
        print(traceback_text, flush=True)
        write_json(
            REPORTS / "failure.json",
            {"status": "FAIL", "error": error_text, "traceback": traceback_text},
        )
    finally:
        archive, archive_receipt = create_archive_and_manifest(overall_status, error_text)
        print("ARCHIVE_RECEIPT_JSON", json.dumps(archive_receipt, separators=(",", ":")), flush=True)
        try:
            upload_result = upload_artifacts()
            print(f"DURABLE_UPLOAD_PASS {upload_result}", flush=True)
        except Exception as upload_exc:
            print(f"DURABLE_UPLOAD_FAIL {type(upload_exc).__name__}: {upload_exc}", flush=True)
            traceback.print_exc()
            exit_code = 1
    if exit_code == 0:
        print("HF_KT2C_RECOVERY_BUILD_BTE_OVERALL: PASS", flush=True)
    else:
        print("HF_KT2C_RECOVERY_BUILD_BTE_OVERALL: FAIL", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
