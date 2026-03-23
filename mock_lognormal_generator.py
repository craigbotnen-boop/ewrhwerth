#!/usr/bin/env python3
"""
DESI-like log-normal mock generator for SWORD pipeline validation.

Builds a LCDM log-normal mock of the DESI DR1 v1.5 NGC LRG sample.
Outputs two FITS files that mimic real pipeline inputs:
  - mock_LRG_NGC_clustering.dat.fits  (RA/DEC catalogue)
  - mock_LRG_NGC_randoms_0.fits       (matching random catalogue)

Interpretation after running through Stage-1 -> Stage-2:
  - If kappa x d_s(mock) >= 7-8 sigma: statistic likely repackages overdensity/mask geometry
  - If kappa x d_s(mock) <= 2-3 sigma: strong evidence real-data 9 sigma is genuine topology
"""

import healpy as hp
import numpy as np
import pymaster as nmt
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
import camb
import os

# ---- USER CONFIG ------------------------------------------------------------
NSIDE        = 1024                # match production pipeline
MASK_FITS    = '/content/drive/MyDrive/SWORD/mask_act_dr6_lensing_v1.fits'
OUT_DIR      = '/content/drive/MyDrive/SWORD/mock_outputs/'
SEED         = 42
N_GAL_DESI   = 262415 * 355 / 305  # target galaxy count (tweak density here)
ELL_MAX      = 3 * NSIDE
# -----------------------------------------------------------------------------

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load survey mask (1=observed, 0=outside)
mask = hp.read_map(MASK_FITS)
mask = hp.ud_grade(mask, NSIDE)
mask = (mask > 0.5).astype(float)
wsky = np.sum(mask) / (12 * NSIDE**2)  # f_sky

# 2. Build angular Cl for galaxies (bias^2 * Cl_matter)
pars = camb.CAMBparams()
pars.set_cosmology(
    H0=cosmo.H0.value,
    ombh2=cosmo.Ob0 * cosmo.h**2,
    omch2=cosmo.Odm0 * cosmo.h**2,
    mnu=0.06,
)
pars.InitPower.set_params(ns=0.96, As=2.1e-9)
pars.set_for_lmax(ELL_MAX, lens_potential_accuracy=1)
results = camb.get_results(pars)
cls_th = results.get_total_cl()[1:ELL_MAX + 1, 0]  # TT as matter proxy
# Crude galaxy bias so that sigma8_g ~ 1
cls_gal = 1.8**2 * cls_th

# 3. Generate log-normal realisation using NaMaster
lgn = nmt.synfast_log_normal(
    1, cls_gal[np.newaxis, :], f_sky=wsky,
    lmax=ELL_MAX, nside=NSIDE, mask=mask,
    beta=1e-4, seed=SEED,
)[0]

delta_map = lgn * mask  # enforce mask exactly

# 4. Poisson sample galaxy counts
mean_ngal = N_GAL_DESI / wsky       # mean per steradian
pix_area = hp.nside2pixarea(NSIDE)   # sr
lam_pix = mean_ngal * pix_area * (1 + delta_map)
lam_pix[lam_pix < 0] = 0
N_pix = np.random.poisson(lam_pix)

# 5. Convert counts -> RA/DEC catalogue
pix_idx = np.repeat(np.arange(len(N_pix)), N_pix)
theta, phi = hp.pix2ang(NSIDE, pix_idx)
ra, dec = np.degrees(phi), 90 - np.degrees(theta)

n_gal = len(ra)
n_rand = int(20 * n_gal)

cat = Table({
    'RA': ra,
    'DEC': dec,
    'Z': np.random.uniform(0.6, 1.0, n_gal),
    'WEIGHT': np.ones(n_gal),
    'WEIGHT_SYS': np.ones(n_gal),
    'WEIGHT_COMP': np.ones(n_gal),
    'WEIGHT_ZFAIL': np.ones(n_gal),
})

rand = Table({
    'RA': np.random.uniform(0, 360, n_rand),
    'DEC': np.degrees(np.arcsin(np.random.uniform(-1, 1, n_rand))),
    'Z': np.random.uniform(0.6, 1.0, n_rand),
    'WEIGHT': np.ones(n_rand),
})

cat.write(OUT_DIR + 'mock_LRG_NGC_clustering.dat.fits', overwrite=True)
rand.write(OUT_DIR + 'mock_LRG_NGC_randoms_0.fits', overwrite=True)
hp.write_map(OUT_DIR + 'mock_delta_map.fits', delta_map, overwrite=True)

print(f"Mock catalogue: {n_gal:,} galaxies  |  Randoms: {n_rand:,}")
print(f"f_sky = {wsky:.4f}")
print(f"Files written to {OUT_DIR}")
