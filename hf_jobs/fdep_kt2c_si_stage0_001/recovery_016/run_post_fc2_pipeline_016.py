#!/usr/bin/env python3
"""Fail-closed reconstruction of KT2C Si FC2/FC3 and 9x9x9 RTA pilot.

This script performs no electronic-structure calculation. It reconstructs the
57 FC3 and one FC2 force sets from immutable Hugging Face Job logs, validates
them against the authoritative regenerated displacement inputs, constructs raw
and symmetry-projected force constants, runs harmonic stability diagnostics,
and conditionally executes a 9x9x9 single-mode RTA thermal-conductivity pilot.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from huggingface_hub import HfApi
from phonopy import Phonopy
from phonopy.physical_units import get_physical_units
from phono3py import load as load_phono3py
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5

DOCUMENT_CODE = "FDEP_KT2C_SI_POST_FC2_PIPELINE_016"
ACTIVE_FC2_JOB = "6a667e5ddb23d7a7ec1ced9a"
SOURCE_JOBS = [
    "6a662bc6db23d7a7ec1ce68c",  # valid FC3 1--4
    "6a662bdfdb23d7a7ec1ce68e",  # FC3 16--29
    "6a662bf7db23d7a7ec1ce690",  # FC3 30--43
    "6a662c1adb23d7a7ec1ce692",  # FC3 44--57
    ACTIVE_FC2_JOB,              # corrected FC3 5--15 and FC2
]
DISPLACEMENT_SCRIPT_URL = (
    "https://raw.githubusercontent.com/craigbotnen-boop/ewrhwerth/"
    "05521fdc3fcbe4b8b0aa104421061bc9f2750a7b/"
    "hf_jobs/fdep_kt2c_si_stage0_001/run_displacements_007.sh"
)
DISPLACEMENT_SCRIPT_COMMIT = "05521fdc3fcbe4b8b0aa104421061bc9f2750a7b"
EXPECTED_PHONO3PY_VERSION = "4.3.3"
EXPECTED_PHONOPY_VERSION = "4.3.1"
STABILITY_MESH = (21, 21, 21)
BTE_MESH = (9, 9, 9)
BTE_TEMPERATURES_K = (300.0,)
SEVERE_IMAGINARY_THRESHOLD_THZ = -0.10
GAMMA_ACOUSTIC_TOLERANCE_THZ = 0.10

ROOT = Path(os.environ.get("KT2C_WORK", "/tmp/kt2c_post_fc2_016")).resolve()
ART = ROOT / "artifacts"
REPORTS = ART / "reports"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def max_abs(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def relative_change(new: np.ndarray, old: np.ndarray) -> float:
    denom = float(np.linalg.norm(old.ravel()))
    return float(np.linalg.norm((new - old).ravel()) / max(denom, 1e-300))


def collect_receipts(api: HfApi) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    source_summary: dict[str, Any] = {}
    marker = "OFFICIAL_CASE_RECEIPT "

    for job_id in SOURCE_JOBS:
        info = api.inspect_job(job_id=job_id)
        count = 0
        cases: list[str] = []
        for entry in api.fetch_job_logs(job_id=job_id, follow=False):
            text = str(entry)
            if marker not in text:
                continue
            payload = text.split(marker, 1)[1].strip()
            row = json.loads(payload)
            case = str(row["case"])
            row = dict(row)
            row["source_job_id"] = job_id
            if case in rows:
                old = rows[case]
                comparable = ("case", "status", "atom_count", "parser", "forces_Ry_per_Bohr", "input_sha256", "output_sha256")
                if any(old.get(k) != row.get(k) for k in comparable):
                    raise RuntimeError(f"CONFLICTING_DUPLICATE_RECEIPT {case}")
            else:
                rows[case] = row
            count += 1
            cases.append(case)
        source_summary[job_id] = {
            "job_stage": str(info.status.stage),
            "failure_count": int(info.status.failure_count or 0),
            "receipt_lines": count,
            "cases": cases,
        }

    return rows, source_summary


def validate_receipts(rows: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_fc3 = [f"fc3_{i:05d}" for i in range(1, 58)]
    expected_all = expected_fc3 + ["fc2_00001"]
    missing = [case for case in expected_all if case not in rows]
    extra = sorted(set(rows) - set(expected_all))
    if missing:
        if missing == ["fc2_00001"]:
            raise RuntimeError("BLOCKED_FC2_RECEIPT_NOT_COMPLETE")
        raise RuntimeError(f"MISSING_FORCE_RECEIPTS {missing}")
    if extra:
        raise RuntimeError(f"UNEXPECTED_FORCE_RECEIPTS {extra}")

    parser = "phonopy.interface.qe.parse_set_of_forces"
    for case in expected_all:
        row = rows[case]
        expected_atoms = 54 if case.startswith("fc2_") else 16
        if row.get("status") != "PASS":
            raise RuntimeError(f"NONPASS_RECEIPT {case} {row.get('status')}")
        if int(row.get("atom_count", -1)) != expected_atoms:
            raise RuntimeError(f"ATOM_COUNT_MISMATCH {case}")
        if row.get("parser") != parser:
            raise RuntimeError(f"PARSER_MISMATCH {case}")
        forces = np.asarray(row.get("forces_Ry_per_Bohr"), dtype=float)
        if forces.shape != (expected_atoms, 3) or not np.isfinite(forces).all():
            raise RuntimeError(f"FORCE_ARRAY_INVALID {case} shape={forces.shape}")
        for key in ("input_sha256", "output_sha256"):
            value = str(row.get(key, ""))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise RuntimeError(f"HASH_INVALID {case} {key}")

    ordered = [rows[c] for c in expected_all]
    summary = {
        "status": "PASS",
        "fc3_count": 57,
        "fc2_count": 1,
        "total_count": 58,
        "unique_case_count": len(rows),
        "parser": parser,
    }
    return ordered, summary


def regenerate_authority() -> Path:
    authority = ROOT / "authority"
    if authority.exists():
        shutil.rmtree(authority)
    authority.mkdir(parents=True)
    script = ROOT / "run_displacements_007.sh"
    urllib.request.urlretrieve(DISPLACEMENT_SCRIPT_URL, script)
    script.chmod(0o755)
    subprocess.run(["bash", str(script), str(authority)], check=True)
    return authority


def validate_input_hashes(authority: Path, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    checked: list[dict[str, str]] = []
    for i in range(1, 58):
        case = f"fc3_{i:05d}"
        path = authority / "force_inputs_fc3" / f"{case}.in"
        actual = sha256_file(path)
        expected = rows[case]["input_sha256"]
        checked.append({"case": case, "sha256": actual})
        if actual != expected:
            mismatches.append({"case": case, "expected": expected, "actual": actual})
    case = "fc2_00001"
    path = authority / "force_inputs_fc2" / f"{case}.in"
    actual = sha256_file(path)
    expected = rows[case]["input_sha256"]
    checked.append({"case": case, "sha256": actual})
    if actual != expected:
        mismatches.append({"case": case, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"INPUT_AUTHORITY_HASH_MISMATCH {mismatches}")
    return {
        "status": "PASS",
        "checked_count": len(checked),
        "displacement_script_commit": DISPLACEMENT_SCRIPT_COMMIT,
        "phono3py_disp_yaml_sha256": sha256_file(authority / "phono3py_disp.yaml"),
        "checked_inputs": checked,
    }


def fc2_diagnostics(fc: np.ndarray) -> dict[str, float]:
    return {
        "max_abs": max_abs(fc),
        "translation_drift_sum_atom0_max_abs": max_abs(fc.sum(axis=0)),
        "translation_drift_sum_atom1_max_abs": max_abs(fc.sum(axis=1)),
        "permutation_residual_max_abs": max_abs(fc - fc.transpose(1, 0, 3, 2)),
        "frobenius_norm": float(np.linalg.norm(fc.ravel())),
    }


def fc3_diagnostics(fc: np.ndarray) -> dict[str, Any]:
    permutation_residuals: dict[str, float] = {}
    for perm in itertools.permutations((0, 1, 2)):
        axes = perm + tuple(i + 3 for i in perm)
        key = "".join(str(i) for i in perm)
        permutation_residuals[key] = max_abs(fc - fc.transpose(axes))
    return {
        "max_abs": max_abs(fc),
        "translation_drift_sum_atom0_max_abs": max_abs(fc.sum(axis=0)),
        "translation_drift_sum_atom1_max_abs": max_abs(fc.sum(axis=1)),
        "translation_drift_sum_atom2_max_abs": max_abs(fc.sum(axis=2)),
        "permutation_residual_max_abs_by_permutation": permutation_residuals,
        "permutation_residual_global_max_abs": max(permutation_residuals.values()),
        "frobenius_norm": float(np.linalg.norm(fc.ravel())),
    }


def build_force_constants(authority: Path, rows: dict[str, dict[str, Any]]) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import phonopy
    import phono3py

    if phono3py.__version__ != EXPECTED_PHONO3PY_VERSION:
        raise RuntimeError(f"PHONO3PY_VERSION_MISMATCH {phono3py.__version__}")
    if phonopy.__version__ != EXPECTED_PHONOPY_VERSION:
        raise RuntimeError(f"PHONOPY_VERSION_MISMATCH {phonopy.__version__}")

    ph3 = load_phono3py(authority / "phono3py_disp.yaml", produce_fc=False, log_level=1)
    if ph3.calculator != "qe":
        raise RuntimeError(f"CALCULATOR_MISMATCH {ph3.calculator}")
    if len(ph3.supercells_with_displacements) != 57:
        raise RuntimeError("FC3_DISPLACEMENT_COUNT_MISMATCH")
    if len(ph3.phonon_supercells_with_displacements) != 1:
        raise RuntimeError("FC2_DISPLACEMENT_COUNT_MISMATCH")

    # Native QE units are required here: Ry/Bohr, matching phono3py_disp.yaml in Bohr.
    ph3.forces = np.asarray(
        [rows[f"fc3_{i:05d}"]["forces_Ry_per_Bohr"] for i in range(1, 58)],
        dtype=float,
    )
    ph3.phonon_forces = np.asarray([rows["fc2_00001"]["forces_Ry_per_Bohr"]], dtype=float)

    ph3.produce_fc3(is_compact_fc=False, fc_calculator="traditional")
    raw_fc3_native = np.array(ph3.fc3, dtype=float, copy=True, order="C")
    ph3.produce_fc2(is_compact_fc=False, fc_calculator="traditional")
    raw_fc2_native = np.array(ph3.fc2, dtype=float, copy=True, order="C")

    ph3.fc3 = raw_fc3_native.copy()
    ph3.fc2 = raw_fc2_native.copy()
    ph3.symmetrize_fc3(use_symfc_projector=True)
    ph3.symmetrize_fc2(use_symfc_projector=True)
    sym_fc3_native = np.array(ph3.fc3, dtype=float, copy=True, order="C")
    sym_fc2_native = np.array(ph3.fc2, dtype=float, copy=True, order="C")

    expected_shapes = {
        "fc2": (54, 54, 3, 3),
        "fc3": (16, 16, 16, 3, 3, 3),
    }
    for name, arr, shape in (
        ("raw_fc2", raw_fc2_native, expected_shapes["fc2"]),
        ("sym_fc2", sym_fc2_native, expected_shapes["fc2"]),
        ("raw_fc3", raw_fc3_native, expected_shapes["fc3"]),
        ("sym_fc3", sym_fc3_native, expected_shapes["fc3"]),
    ):
        if arr.shape != shape or not np.isfinite(arr).all():
            raise RuntimeError(f"FORCE_CONSTANT_INVALID {name} shape={arr.shape}")

    pu = get_physical_units()
    fc2_to_ev_a2 = float(pu.Rydberg / pu.Bohr**2)
    fc3_to_ev_a3 = float(pu.Rydberg / pu.Bohr**3)
    raw_fc2_std = raw_fc2_native * fc2_to_ev_a2
    sym_fc2_std = sym_fc2_native * fc2_to_ev_a2
    raw_fc3_std = raw_fc3_native * fc3_to_ev_a3
    sym_fc3_std = sym_fc3_native * fc3_to_ev_a3

    write_fc2_to_hdf5(raw_fc2_std, filename=str(ART / "fc2.raw.hdf5"), physical_unit="eV/angstrom^2")
    write_fc2_to_hdf5(sym_fc2_std, filename=str(ART / "fc2.sym.hdf5"), physical_unit="eV/angstrom^2")
    write_fc2_to_hdf5(sym_fc2_std, filename=str(ART / "fc2.hdf5"), physical_unit="eV/angstrom^2")
    write_fc3_to_hdf5(raw_fc3_std, filename=str(ART / "fc3.raw.hdf5"))
    write_fc3_to_hdf5(sym_fc3_std, filename=str(ART / "fc3.sym.hdf5"))
    write_fc3_to_hdf5(sym_fc3_std, filename=str(ART / "fc3.hdf5"))

    for path, dataset, shape in (
        (ART / "fc2.raw.hdf5", "fc2", expected_shapes["fc2"]),
        (ART / "fc2.sym.hdf5", "fc2", expected_shapes["fc2"]),
        (ART / "fc2.hdf5", "fc2", expected_shapes["fc2"]),
        (ART / "fc3.raw.hdf5", "fc3", expected_shapes["fc3"]),
        (ART / "fc3.sym.hdf5", "fc3", expected_shapes["fc3"]),
        (ART / "fc3.hdf5", "fc3", expected_shapes["fc3"]),
    ):
        with h5py.File(path, "r") as h5:
            if dataset not in h5 or h5[dataset].shape != shape or not np.isfinite(h5[dataset][...]).all():
                raise RuntimeError(f"HDF5_VALIDATION_FAIL {path.name}")

    diagnostics = {
        "status": "PASS",
        "native_units": {"fc2": "Ry/Bohr^2", "fc3": "Ry/Bohr^3"},
        "stored_units": {"fc2": "eV/Angstrom^2", "fc3": "eV/Angstrom^3"},
        "conversion_factors": {
            "fc2_Ry_per_Bohr2_to_eV_per_A2": fc2_to_ev_a2,
            "fc3_Ry_per_Bohr3_to_eV_per_A3": fc3_to_ev_a3,
        },
        "raw_fc2_native": fc2_diagnostics(raw_fc2_native),
        "sym_fc2_native": fc2_diagnostics(sym_fc2_native),
        "raw_fc3_native": fc3_diagnostics(raw_fc3_native),
        "sym_fc3_native": fc3_diagnostics(sym_fc3_native),
        "symmetry_projection_change": {
            "fc2_relative_frobenius": relative_change(sym_fc2_native, raw_fc2_native),
            "fc2_max_abs_element_change_native": max_abs(sym_fc2_native - raw_fc2_native),
            "fc3_relative_frobenius": relative_change(sym_fc3_native, raw_fc3_native),
            "fc3_max_abs_element_change_native": max_abs(sym_fc3_native - raw_fc3_native),
        },
        "shapes": {
            "fc2_full": list(sym_fc2_native.shape),
            "fc3_full": list(sym_fc3_native.shape),
        },
    }
    return ph3, raw_fc2_native, sym_fc2_native, raw_fc3_native, sym_fc3_native, diagnostics


def run_stability(ph3: Any, sym_fc2_native: np.ndarray) -> dict[str, Any]:
    phonon = Phonopy(
        ph3.unitcell,
        supercell_matrix=ph3.phonon_supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
        calculator="qe",
        log_level=1,
    )
    phonon.force_constants = sym_fc2_native
    phonon.run_mesh(
        STABILITY_MESH,
        is_mesh_symmetry=False,
        is_gamma_center=True,
        with_eigenvectors=False,
        with_group_velocities=False,
    )
    mesh = phonon.get_mesh_dict()
    qpoints = np.asarray(mesh["qpoints"], dtype=float)
    frequencies = np.asarray(mesh["frequencies"], dtype=float)
    if not np.isfinite(frequencies).all():
        raise RuntimeError("NONFINITE_HARMONIC_FREQUENCY")
    flat_index = int(np.argmin(frequencies))
    q_index, band_index = np.unravel_index(flat_index, frequencies.shape)
    min_frequency = float(frequencies[q_index, band_index])
    q_min = qpoints[q_index].tolist()

    away = np.linalg.norm(qpoints - np.rint(qpoints), axis=1) > 1e-10
    away_freq = frequencies[away]
    min_away = float(np.min(away_freq)) if away_freq.size else min_frequency
    severe_count = int(np.count_nonzero(frequencies < SEVERE_IMAGINARY_THRESHOLD_THZ))

    phonon.run_qpoints([[0.0, 0.0, 0.0]])
    gamma = np.sort(np.asarray(phonon.get_qpoints_dict()["frequencies"], dtype=float)[0])
    gamma_acoustic = gamma[:3]
    gamma_acoustic_max_abs = max_abs(gamma_acoustic)

    status = "PASS" if severe_count == 0 and gamma_acoustic_max_abs <= GAMMA_ACOUSTIC_TOLERANCE_THZ else "FAIL"
    return {
        "status": status,
        "criterion": {
            "severe_imaginary_threshold_THz": SEVERE_IMAGINARY_THRESHOLD_THZ,
            "gamma_acoustic_max_abs_tolerance_THz": GAMMA_ACOUSTIC_TOLERANCE_THZ,
        },
        "mesh": list(STABILITY_MESH),
        "mesh_qpoint_count": int(len(qpoints)),
        "minimum_frequency_THz": min_frequency,
        "minimum_qpoint_reduced": q_min,
        "minimum_band_index_zero_based": int(band_index),
        "minimum_frequency_away_from_gamma_THz": min_away,
        "severe_negative_mode_count": severe_count,
        "gamma_frequencies_THz": gamma.tolist(),
        "gamma_acoustic_frequencies_THz": gamma_acoustic.tolist(),
        "gamma_acoustic_max_abs_THz": gamma_acoustic_max_abs,
    }


def run_rta_pilot(ph3: Any, stability: dict[str, Any]) -> dict[str, Any]:
    if stability["status"] != "PASS":
        return {
            "status": "BLOCKED_STABILITY_FAIL",
            "solver": "single-mode relaxation-time approximation",
            "mesh": list(BTE_MESH),
        }

    ph3.mesh_numbers = BTE_MESH
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        is_LBTE=False,
        temperatures=BTE_TEMPERATURES_K,
        is_isotope=False,
        write_kappa=True,
        log_level=1,
    )
    tc = ph3.thermal_conductivity
    if tc is None or tc.kappa is None:
        raise RuntimeError("RTA_KAPPA_MISSING")
    kappa = np.asarray(tc.kappa, dtype=float)
    if not np.isfinite(kappa).all():
        raise RuntimeError("RTA_KAPPA_NONFINITE")

    copied: list[str] = []
    for path in ROOT.glob("kappa-*.hdf5"):
        target = ART / path.name
        shutil.copy2(path, target)
        copied.append(target.name)
    for path in Path.cwd().glob("kappa-*.hdf5"):
        target = ART / path.name
        if not target.exists():
            shutil.copy2(path, target)
            copied.append(target.name)

    return {
        "status": "PASS",
        "solver": "single-mode relaxation-time approximation",
        "is_LBTE": False,
        "claim_level": "SMOKE_TEST_PILOT_NOT_MESH_CONVERGED",
        "mesh": list(BTE_MESH),
        "temperatures_K": list(BTE_TEMPERATURES_K),
        "kappa_shape": list(kappa.shape),
        "kappa_W_per_mK": kappa.tolist(),
        "output_files": sorted(set(copied)),
    }


def create_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(p for p in ART.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS.txt", "package_receipt.json"}):
        files.append({
            "path": str(path.relative_to(ART)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in files)
    (ART / "SHA256SUMS.txt").write_text(sums)
    return {"status": "PASS", "file_count": len(files), "files": files}


def create_archive() -> dict[str, Any]:
    archive = ROOT / "FDEP_KT2C_SI_POST_FC2_PIPELINE_016.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(ART, arcname=ART.name)
    target = ART / archive.name
    shutil.copy2(archive, target)
    return {
        "filename": target.name,
        "size_bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def upload_artifacts(api: HfApi) -> dict[str, Any]:
    who = api.whoami()
    owner = who.get("name") or who.get("fullname")
    if not owner:
        raise RuntimeError("HF_IDENTITY_UNAVAILABLE")
    repo_id = f"{owner}/kt2c-si-recovery-016"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(ART),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="production",
        commit_message=f"Upload {DOCUMENT_CODE} artifacts",
    )
    return {"status": "PASS", "repo_id": repo_id, "repo_type": "dataset", "path_in_repo": "production", "private": True}


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    if ART.exists():
        shutil.rmtree(ART)
    REPORTS.mkdir(parents=True)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    top: dict[str, Any] = {
        "document_code": DOCUMENT_CODE,
        "started_at": now_iso(),
        "status": "RUNNING",
        "claim_boundary": {
            "official_force_extraction": "UNDER_VALIDATION",
            "fc2_fc3_construction": "NOT_EXECUTED",
            "stability_analysis": "NOT_EXECUTED",
            "rta_pilot": "NOT_EXECUTED",
            "transport_mesh_convergence": "NOT_EXECUTED",
            "physical_claim_upgrade": "BLOCKED",
        },
    }

    try:
        rows, source_summary = collect_receipts(api)
        ordered, receipt_validation = validate_receipts(rows)
        write_json(REPORTS / "source_jobs.json", source_summary)
        write_json(REPORTS / "official_force_receipts.json", {"cases": ordered})
        write_json(REPORTS / "receipt_validation.json", receipt_validation)
        top["claim_boundary"]["official_force_extraction"] = "PASS"

        authority = regenerate_authority()
        authority_validation = validate_input_hashes(authority, rows)
        write_json(REPORTS / "authority_validation.json", authority_validation)
        shutil.copy2(authority / "phono3py_disp.yaml", ART / "phono3py_disp.yaml")
        shutil.copy2(authority / "phono3py.yaml", ART / "phono3py.yaml")
        shutil.copy2(authority / "reports" / "displacement_result.json", REPORTS / "displacement_result.json")

        ph3, raw_fc2, sym_fc2, raw_fc3, sym_fc3, fc_diag = build_force_constants(authority, rows)
        write_json(REPORTS / "force_constant_diagnostics.json", fc_diag)
        top["claim_boundary"]["fc2_fc3_construction"] = "PASS"

        stability = run_stability(ph3, sym_fc2)
        write_json(REPORTS / "zone_wide_stability.json", stability)
        top["claim_boundary"]["stability_analysis"] = stability["status"]

        rta = run_rta_pilot(ph3, stability)
        write_json(REPORTS / "rta_9x9x9_pilot.json", rta)
        top["claim_boundary"]["rta_pilot"] = rta["status"]

        top.update({
            "finished_at": now_iso(),
            "status": "PASS" if stability["status"] == "PASS" and rta["status"] == "PASS" else "FAIL_CLOSED",
            "receipt_validation": receipt_validation,
            "authority_validation": authority_validation,
            "force_constant_summary": {
                "fc2_raw_shape": list(raw_fc2.shape),
                "fc2_sym_shape": list(sym_fc2.shape),
                "fc3_raw_shape": list(raw_fc3.shape),
                "fc3_sym_shape": list(sym_fc3.shape),
            },
            "stability": stability,
            "rta_pilot": rta,
        })
        write_json(REPORTS / "production_receipt.json", top)
        manifest = create_manifest()
        write_json(ART / "package_receipt.json", {"document_code": DOCUMENT_CODE, "manifest": manifest})
        archive = create_archive()
        top["archive"] = archive
        write_json(REPORTS / "production_receipt.json", top)
        upload = upload_artifacts(api)
        top["upload"] = upload
        write_json(REPORTS / "production_receipt.json", top)
        # Final receipt upload after adding upload metadata.
        api.upload_file(
            path_or_fileobj=str(REPORTS / "production_receipt.json"),
            path_in_repo="production/reports/production_receipt.json",
            repo_id=upload["repo_id"],
            repo_type="dataset",
            commit_message="Finalize production receipt",
        )
        print("POST_FC2_PIPELINE_RECEIPT", json.dumps(top, separators=(",", ":")))
        print("KT2C_POST_FC2_PIPELINE_OVERALL:", top["status"])
        return 0 if top["status"] == "PASS" else 2

    except Exception as exc:
        top.update({
            "finished_at": now_iso(),
            "status": "BLOCKED" if str(exc).startswith("BLOCKED_") else "FAIL",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        })
        write_json(REPORTS / "production_receipt.json", top)
        try:
            create_manifest()
            upload = upload_artifacts(api)
            top["upload"] = upload
            write_json(REPORTS / "production_receipt.json", top)
            api.upload_file(
                path_or_fileobj=str(REPORTS / "production_receipt.json"),
                path_in_repo="production/reports/production_receipt.json",
                repo_id=upload["repo_id"],
                repo_type="dataset",
                commit_message="Upload fail-closed production receipt",
            )
        except Exception as upload_exc:
            top["upload_error"] = {"type": type(upload_exc).__name__, "message": str(upload_exc)}
        print("POST_FC2_PIPELINE_RECEIPT", json.dumps(top, separators=(",", ":")))
        print("KT2C_POST_FC2_PIPELINE_OVERALL:", top["status"])
        return 3


if __name__ == "__main__":
    sys.exit(main())
