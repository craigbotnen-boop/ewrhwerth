"""SWORD P1000 Colab runner with install guards and map preflight checks.

This script is intended for single-cell style Colab execution.
"""

import os
import sys
import time


# ---------- THREAD / ENCODING DISCIPLINE ----------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"


# ---------- CONFIG ----------
SCRIPT = "sword_p1000_CLEAN.py"
KAPPA = "kappa_map_realspace_nside1024.npy"
DELTA_G = "delta_s_patch_map.npy"
MASK = "combined_mask_SGC_nside1024.npy"
OUTDIR = "p1000_results"
LMIN, LMAX, DELL = 8, 2000, 50
N_NULL = 100  # bump to 1000 for production after validation


print("=" * 60)
print("STEP 1: Installing system libs + Python packages")
print("=" * 60)
os.system(
    "apt-get update -qq && apt-get install -y -qq "
    "libgsl-dev libfftw3-dev libcfitsio-dev pkg-config"
)

# NOTE: pip wheels usually work on Colab; if pip fails, do NOT try condacolab inside
# this same cell.
ret = os.system("pip install -q healpy pymaster numpy scipy tqdm")
if ret != 0:
    print("\nFATAL: pip install failed. Recommended fix:")
    print("  1) Runtime > Restart runtime")
    print(
        "  2) In a NEW cell: !pip install -q condacolab && "
        "python -c \"import condacolab; condacolab.install()\""
    )
    print("  3) After restart: !conda install -y -c conda-forge namaster healpy")
    sys.exit(1)

# Smoke test imports
try:
    import healpy as hp
    import numpy as np
    import pymaster as nmt

    print(f"  pymaster {nmt.__version__}, healpy {hp.__version__} — OK")
except ImportError as e:
    print(f"FATAL import: {e}")
    sys.exit(1)


print("\n" + "=" * 60)
print("STEP 2: Checking uploaded files")
print("=" * 60)
required = [SCRIPT, KAPPA, DELTA_G, MASK]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    print(f"MISSING FILES: {missing}")
    sys.exit(1)
print("  All files present — OK")


print("\n" + "=" * 60)
print("STEP 2b: Preflight validate maps (NSIDE, finiteness, mask overlap)")
print("=" * 60)

kappa = np.load(KAPPA)
tr = np.load(DELTA_G)
mask = np.load(MASK)


def preflight(name, arr):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name}: expected 1D healpix map, got shape {arr.shape}")
    nside = hp.npix2nside(arr.size)
    finite = np.isfinite(arr).mean()
    print(
        f"{name}: npix={arr.size:,} nside={nside} "
        f"dtype={arr.dtype} finite_frac={finite:.4f}"
    )
    return nside


nside_k = preflight("kappa", kappa)
nside_t = preflight("tracer", tr)
nside_m = preflight("mask", mask)

if not (nside_k == nside_t == nside_m):
    raise ValueError(f"NSIDE mismatch: kappa={nside_k}, tracer={nside_t}, mask={nside_m}")

# mask sanity: ensure there is overlap with finite kappa/tracer
mask_bool = (mask > 0.5) & np.isfinite(kappa) & np.isfinite(tr)
print(f"mask_true_frac (finite overlap) = {mask_bool.mean():.4f}")
if mask_bool.mean() < 0.01:
    raise ValueError("Mask overlap is tiny (<1%). Likely wrong mask or wrong sky patch / ordering.")

# Reduce memory footprint for downstream script (optional but helps Colab stability)
if kappa.dtype != np.float32:
    np.save("kappa_f32.npy", kappa.astype(np.float32))
    KAPPA = "kappa_f32.npy"
if tr.dtype != np.float32:
    np.save("tracer_f32.npy", tr.astype(np.float32))
    DELTA_G = "tracer_f32.npy"
if mask.dtype != np.float32:
    np.save("mask_f32.npy", mask.astype(np.float32))
    MASK = "mask_f32.npy"


print("\n" + "=" * 60)
print("STEP 3: Running OBS mode (observed cross-spectrum)")
print("=" * 60)
os.makedirs(OUTDIR, exist_ok=True)

cmd_obs = (
    f"python -u {SCRIPT} --mode obs --out_cl {OUTDIR}/cl_obs.npy "
    f"--kappa_path {KAPPA} --tracer_path {DELTA_G} --mask_path {MASK} "
    f"--lmin {LMIN} --lmax {LMAX} --bin_width {DELL}"
)
print(f"  CMD: {cmd_obs}")
t0 = time.time()
ret = os.system(cmd_obs)
print(f"  Elapsed: {(time.time() - t0) / 60:.1f} min")
if ret != 0:
    print("OBS mode FAILED.")
    sys.exit(1)


print("\n" + "=" * 60)
print(f"STEP 4: Running {N_NULL} NULL surrogates")
print("=" * 60)
t_start = time.time()

for seed in range(N_NULL):
    expected_out = os.path.join(OUTDIR, f"cl_null_seed{seed:04d}.npy")
    if os.path.exists(expected_out):
        continue

    cmd_null = (
        f"python -u {SCRIPT} --mode null --out_cl {expected_out} "
        f"--kappa_path {KAPPA} --tracer_path {DELTA_G} --mask_path {MASK} "
        f"--lmin {LMIN} --lmax {LMAX} --bin_width {DELL} --seed {seed}"
    )
    ret = os.system(cmd_null)
    if ret != 0:
        print(f"NULL mode FAILED at seed={seed}")
        sys.exit(1)

    if (seed + 1) % 10 == 0:
        elapsed_h = (time.time() - t_start) / 3600
        rate = (seed + 1) / elapsed_h if elapsed_h > 0 else 0
        eta_h = (N_NULL - seed - 1) / rate if rate > 0 else 0
        print(f"  {seed + 1}/{N_NULL} done | {elapsed_h:.2f}h elapsed | ETA {eta_h:.1f}h")


print("\n" + "=" * 60)
print("STEP 5: Zip results for download")
print("=" * 60)
os.system(f"zip -r {OUTDIR}.zip {OUTDIR}/ >/dev/null")
print(f"\nDONE. Download {OUTDIR}.zip")
