# sword_p1000_colab.py
# SWORD P1000 Null-Test Pipeline — Google Colab Edition
#
# Cross-correlates DESI LRG galaxy overdensity with ACT CMB lensing
# using NaMaster (MASTER algorithm) and validates detection significance
# via 1000 Gaussian random field null simulations.
#
# Requirements: Upload these FITS files to the Colab runtime before running:
#   - desi_lrg_map.fits   (DESI LRG galaxy overdensity, HEALPix)
#   - act_kappa_map.fits   (ACT CMB lensing convergence, HEALPix)
#   - footprint_mask.fits  (joint survey footprint binary mask, HEALPix)

# ── Cell 1: Install system dependencies ──────────────────────────────────────
# Separate cells so Colab shows clear error output per stage.

# !apt-get update -qq && apt-get install -y -qq \
#     libgsl-dev libfftw3-dev libcfitsio-dev \
#     build-essential gfortran pkg-config \
#     autoconf automake libtool

# ── Cell 2: Install Python packages ──────────────────────────────────────────

# !pip install -q --upgrade pip setuptools wheel cython
# !pip install -q numpy healpy
# !pip install -q pymaster --no-cache-dir

# ── Cell 3: Pipeline ─────────────────────────────────────────────────────────

import os
import sys
import time
import numpy as np
import healpy as hp
import pymaster as nmt

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
NSIMS = 1000
L_MAX = 2048        # Target multipole resolution matching DESI/ACT
BIN_WIDTH = 50      # Bandpower bin width in multipole space

# ---------------------------------------------------------------------------
# 2. Load Map Artifacts
# ---------------------------------------------------------------------------
REQUIRED_FILES = [
    'footprint_mask.fits',
    'desi_lrg_map.fits',
    'act_kappa_map.fits',
]

missing = [f for f in REQUIRED_FILES if not os.path.isfile(f)]
if missing:
    sys.exit(
        "Missing FITS files — upload them to the Colab runtime first:\n  "
        + "\n  ".join(missing)
    )

print("Loading maps...")
mask = hp.read_map('footprint_mask.fits')
map_desi = hp.read_map('desi_lrg_map.fits')
map_act = hp.read_map('act_kappa_map.fits')

nside = hp.get_nside(mask)
lmax_use = min(L_MAX, 3 * nside - 1)  # Cannot exceed Nyquist limit of map
print(f"NSIDE={nside}, using lmax={lmax_use}")

# ---------------------------------------------------------------------------
# 3. Setup NaMaster Fields and Bandpower Bins
# ---------------------------------------------------------------------------
bins = nmt.NmtBin.from_nside_linear(nside, n_lb=BIN_WIDTH)

# Spin-0 fields
field_desi = nmt.NmtField(mask, [map_desi])
field_act = nmt.NmtField(mask, [map_act])

# Compute mode-coupling matrix (expensive — done once and reused)
print("Computing mode-coupling matrix...")
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(field_desi, field_act, bins)

# ---------------------------------------------------------------------------
# 4. Observed Cross-Spectrum
# ---------------------------------------------------------------------------
# decouple_cell returns shape (n_cls, n_bins); index [0] for spin-0 x spin-0
cl_obs = w.decouple_cell(nmt.compute_coupled_cell(field_desi, field_act))[0]
print(f"Observed cross-spectrum computed ({len(cl_obs)} bandpowers).")

# ---------------------------------------------------------------------------
# 5. Estimate Input Power Spectrum for Mocks
# ---------------------------------------------------------------------------
# Use NaMaster-decoupled auto-spectrum of the ACT map so the mask bias is
# removed.  hp.anafast on a masked map would underestimate power.
w_auto = nmt.NmtWorkspace()
w_auto.compute_coupling_matrix(field_act, field_act, bins)
cl_act_binned = w_auto.decouple_cell(
    nmt.compute_coupled_cell(field_act, field_act)
)[0]

# Interpolate binned Cl back to per-ell for synfast
ell_eff = bins.get_effective_ells()
ells_full = np.arange(lmax_use + 1)
# Simple log-linear interpolation; zero the monopole/dipole
cls_for_synfast = np.zeros(lmax_use + 1)
valid = ell_eff > 1  # skip ell=0,1 bins if present
cls_for_synfast[2:] = np.interp(
    ells_full[2:],
    ell_eff[valid],
    cl_act_binned[valid],
    left=0.0,
    right=0.0,
)
# Ensure non-negative (interpolation can produce small negatives)
cls_for_synfast = np.maximum(cls_for_synfast, 0.0)

# ---------------------------------------------------------------------------
# 6. P1000 Null Suite
# ---------------------------------------------------------------------------
cl_ensemble = []

start_time = time.time()
print(f"Starting {NSIMS} null simulations...")

for i in range(NSIMS):
    # Generate Gaussian random field mock lensing map
    mock_kappa = hp.synfast(cls_for_synfast, nside, lmax=lmax_use, verbose=False)
    f_mock = nmt.NmtField(mask, [mock_kappa])

    # Cross-correlate mock lensing × real galaxy overdensity
    # Reuse the precomputed workspace (same mask geometry)
    cl_mock = w.decouple_cell(nmt.compute_coupled_cell(field_desi, f_mock))[0]
    cl_ensemble.append(cl_mock)

    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (NSIMS - i - 1) / rate
        print(f"  Sim {i+1}/{NSIMS}  |  {elapsed:.0f}s elapsed  |  ~{eta:.0f}s remaining")

cl_ensemble = np.array(cl_ensemble)  # shape (NSIMS, n_bins)
elapsed_total = time.time() - start_time
print(f"Simulations finished in {elapsed_total:.0f}s.")

# ---------------------------------------------------------------------------
# 7. Statistical Validation
# ---------------------------------------------------------------------------
mean_null = np.mean(cl_ensemble, axis=0)
std_null = np.std(cl_ensemble, axis=0)

# Avoid division by zero in bins with no statistical power
safe = std_null > 0
deviation = np.zeros_like(cl_obs)
deviation[safe] = (cl_obs[safe] - mean_null[safe]) / std_null[safe]

snr = np.sqrt(np.sum(deviation**2))

# ---------------------------------------------------------------------------
# 8. Save Outputs
# ---------------------------------------------------------------------------
np.save('cl_ensemble_null.npy', cl_ensemble)
np.save('cl_obs.npy', cl_obs)

with open('summary_1000.txt', 'w') as f:
    f.write("SWORD P1000 Validation Run Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"NSIMS:              {NSIMS}\n")
    f.write(f"NSIDE:              {nside}\n")
    f.write(f"lmax:               {lmax_use}\n")
    f.write(f"Bin width:          {BIN_WIDTH}\n")
    f.write(f"Bandpowers:         {len(cl_obs)}\n")
    f.write(f"Signal-to-Noise:    {snr:.2f} sigma\n")
    f.write(f"Wall time:          {elapsed_total:.0f}s\n")
    f.write(f"Timestamp:          {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"\nValidation complete.  SNR = {snr:.2f} sigma")
print("Saved: cl_ensemble_null.npy, cl_obs.npy, summary_1000.txt")
