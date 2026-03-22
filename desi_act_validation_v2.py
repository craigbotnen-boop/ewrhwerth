#!/usr/bin/env python3
"""
desi_act_validation_v2.py — NaMaster cross-correlation with provenance.
========================================================================
Part of the SWORD P1 production pipeline.
Fixes applied (per review):
  1. LMAX capped at min(1500, 3*NSIDE-1)
  2. Mandatory ACT mask — hard fail if missing
  3. Common apodized mask (DESI x ACT intersection)
  4. Shared make_binning() for primary and null paths
  5. SNR labeled as smoke-test lower bound, not publishable
  6. Provenance hashes captured from environment
"""

import os
import json
import datetime
import numpy as np
import healpy as hp
import pymaster as nmt

# ---------------------------------------------------------------------------
# Provenance: capture hashes exported by run_sword_validation.sh
# ---------------------------------------------------------------------------
PROV_HASHES = {
    "kappa_alm": os.environ.get("PROV_HASH_KAPPA", "NOT_SET"),
    "curl_alm":  os.environ.get("PROV_HASH_CURL",  "NOT_SET"),
    "mask":      os.environ.get("PROV_HASH_MASK",   "NOT_SET"),
    "desi_npz":  os.environ.get("PROV_HASH_DESI",   "NOT_SET"),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NSIDE    = 512
LMAX_MAX = 3 * NSIDE - 1          # = 1535
LMAX_USE = min(1500, LMAX_MAX)     # conservative cap below Nyquist
LMIN_USE = 20
DELL     = 20

# Paths — alm_to_map_v2.py writes these into the working directory
ACT_KAPPA_MAP = "act_kappa_nside512.fits"
ACT_CURL_MAP  = "act_curl_nside512.fits"
ACT_MASK_FILE = "act_mask_nside512.fits"

# DESI tomographic bin (absolute path through WSL mount)
DESI_NPZ = "/mnt/c/Users/magika/Downloads/dump/desi_dr1_tomo_maps/DESI_LRG_z0.40_0.55_nside512.npz"

# Outputs
OUT_NPZ  = "p1_bin1_validation_v2.npz"
OUT_JSON = "p1_bin1_validation_summary_v2.json"


# ---------------------------------------------------------------------------
# Binning — single function used by BOTH primary and null paths
# ---------------------------------------------------------------------------
def make_binning():
    """Consistent ell-binning for all cross-correlation measurements."""
    bpw_edges = np.arange(LMIN_USE, LMAX_USE + DELL, DELL)
    if bpw_edges[-1] < LMAX_USE:
        bpw_edges = np.append(bpw_edges, LMAX_USE)
    return nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" SWORD P1 — NaMaster Validation v2")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 0. Pre-flight: mandatory mask check
    # ------------------------------------------------------------------
    if not os.path.exists(ACT_MASK_FILE):
        raise FileNotFoundError(
            f"Mandatory ACT mask missing: {ACT_MASK_FILE}. "
            "Run alm_to_map_v2.py first."
        )

    # ------------------------------------------------------------------
    # 1. Load maps
    # ------------------------------------------------------------------
    print(f"Loading DESI map: {DESI_NPZ}")
    d = np.load(DESI_NPZ)
    delta_g  = d["delta_g"].astype(np.float64)
    mask_lss = d["mask_lss"].astype(np.float64)

    npix = len(delta_g)
    assert hp.npix2nside(npix) == NSIDE, "DESI map NSIDE mismatch!"

    print("Loading ACT kappa, curl, and mask...")
    kappa    = hp.read_map(ACT_KAPPA_MAP, verbose=False).astype(np.float64)
    curl     = hp.read_map(ACT_CURL_MAP,  verbose=False).astype(np.float64)
    mask_act = hp.read_map(ACT_MASK_FILE, verbose=False).astype(np.float64)
    assert len(kappa) == npix, "ACT map pixel count mismatch!"

    # ------------------------------------------------------------------
    # 2. Build common apodized mask (DESI ∩ ACT)
    # ------------------------------------------------------------------
    print("Building and apodizing common mask...")
    good = (np.isfinite(delta_g) & np.isfinite(mask_lss)
            & np.isfinite(kappa) & np.isfinite(curl))

    mask = np.zeros_like(mask_lss)
    mask[good] = mask_lss[good] * mask_act[good]

    # C1 apodization at 1 degree
    mask_apo = nmt.mask_apodization(mask, 1.0, apotype="C1")

    f_sky_raw = np.mean(mask > 0)
    f_sky_apo = np.mean(mask_apo > 0)
    print(f"  f_sky (raw intersection): {f_sky_raw:.4f}")
    print(f"  f_sky (after apodization): {f_sky_apo:.4f}")

    # ------------------------------------------------------------------
    # 3. Define NaMaster spin-0 fields (all share the SAME mask)
    # ------------------------------------------------------------------
    print(f"Initializing NmtFields (lmax={LMAX_USE})...")
    f_kappa = nmt.NmtField(mask_apo, [kappa],   spin=0)
    f_curl  = nmt.NmtField(mask_apo, [curl],    spin=0)
    f_delta = nmt.NmtField(mask_apo, [delta_g], spin=0)

    # ------------------------------------------------------------------
    # 4. Shared binning
    # ------------------------------------------------------------------
    b = make_binning()
    ell_eff = b.get_effective_ells()
    print(f"Bandpowers: {len(ell_eff)} bins, ell = [{ell_eff[0]:.0f}, {ell_eff[-1]:.0f}]")

    # ------------------------------------------------------------------
    # 5. Mode-coupling workspace (computed once, reused for null)
    # ------------------------------------------------------------------
    print("Computing mode-coupling workspace...")
    w = nmt.NmtWorkspace.from_fields(f_kappa, f_delta, b)

    # ------------------------------------------------------------------
    # 6. Primary cross-spectrum: kappa × delta_g
    # ------------------------------------------------------------------
    print("Computing primary cross-spectrum (kappa x delta_g)...")
    cl_kd = nmt.compute_full_master(f_kappa, f_delta, b, workspace=w)[0]

    # ------------------------------------------------------------------
    # 7. Null cross-spectrum: curl × delta_g (same workspace & binning)
    # ------------------------------------------------------------------
    print("Computing null cross-spectrum (curl x delta_g)...")
    cl_cd = nmt.compute_full_master(f_curl, f_delta, b, workspace=w)[0]

    # ------------------------------------------------------------------
    # 8. Summary metrics
    # ------------------------------------------------------------------
    finite = np.isfinite(cl_kd) & np.isfinite(cl_cd)
    mean_primary = float(np.mean(np.abs(cl_kd[finite])))
    mean_curl    = float(np.mean(np.abs(cl_cd[finite])))

    summary = {
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "desi_npz":     DESI_NPZ,
        "nside":        NSIDE,
        "lmax_use":     LMAX_USE,
        "lmin_use":     LMIN_USE,
        "dell":         DELL,
        "n_bandpowers": int(len(ell_eff)),
        "f_sky_raw":    float(f_sky_raw),
        "f_sky_apodized": float(f_sky_apo),
        "all_finite":   bool(np.all(finite)),
        "mean_abs_primary": mean_primary,
        "mean_abs_curl":    mean_curl,
        "snr_smoketest_lower_bound": (
            mean_primary / mean_curl if mean_curl > 0 else 0.0
        ),
        "provenance_hashes": PROV_HASHES,
    }

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    np.savez(OUT_NPZ,
             ell_eff=ell_eff,
             cl_kappa_delta=cl_kd,
             cl_curl_delta=cl_cd)

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(" Validation v2 Complete")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {OUT_NPZ}, {OUT_JSON}")


if __name__ == "__main__":
    main()
