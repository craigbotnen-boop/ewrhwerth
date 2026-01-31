"""
Surrogate-null cross-spectrum pipeline using NaMaster.

Generates phase-randomized kappa (convergence) maps that preserve the
auto-power spectrum C_ell^{kk} while destroying any cross-correlation
with a tracer field.  This provides a defensible constraint-style null
test for the cross-spectrum pipeline when survey-specific ACT
simulations are unavailable.

Phase randomization method
--------------------------
For each (l, m>0) mode the amplitude |a_{lm}| is preserved and the
phase is replaced with a fresh uniform draw on [0, 2*pi).  Modes with
m=0 are real by the reality constraint and are left untouched.  This
guarantees uniform random phases while exactly preserving the angular
power spectrum.

Author : Craig W. Botnen
Licence: MIT
"""

import os
import numpy as np
import healpy as hp
import pymaster as nmt
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# Phase-randomized surrogate generator
# ---------------------------------------------------------------------------

def get_surrogate_kappa(m, rng, lmax=None):
    """Return a phase-randomized HEALPix map preserving C_ell.

    Parameters
    ----------
    m : array_like
        Input HEALPix map (RING ordering).
    rng : numpy.random.Generator
        Random number generator instance.
    lmax : int, optional
        Maximum multipole for the SHT.  Defaults to 3*nside - 1.

    Returns
    -------
    surrogate : ndarray
        Phase-randomized HEALPix map with the same nside and C_ell.
    """
    if lmax is None:
        lmax = 3 * hp.get_nside(m) - 1

    alm = hp.map2alm(m, lmax=lmax)
    lmax2 = hp.Alm.getlmax(len(alm))

    for ell in range(1, lmax2 + 1):
        for mm in range(1, ell + 1):
            idx = hp.Alm.getidx(lmax2, ell, mm)
            amp = np.abs(alm[idx])
            ph = rng.uniform(0, 2 * np.pi)
            alm[idx] = amp * (np.cos(ph) + 1j * np.sin(ph))

    return hp.alm2map(alm, hp.get_nside(m), verbose=False)


# ---------------------------------------------------------------------------
# NaMaster surrogate-null cross-spectrum pipeline
# ---------------------------------------------------------------------------

def run_surrogate_null(
    kappa_path,
    tracer_path,
    mask_path,
    out_dir="nmt_outputs",
    nsims=100,
    aposize=1.0,
    apotype="C1",
    bin_width=20,
    ridge=1e-12,
    seed=12345,
):
    """Run surrogate-null cross-spectrum test and persist results.

    Parameters
    ----------
    kappa_path : str
        Path to the real kappa convergence map (.npy, RING order).
    tracer_path : str
        Path to the tracer / delta_s map (.npy, RING order).
    mask_path : str
        Path to the binary survey mask (.npy, RING order).
    out_dir : str
        Directory for output files.
    nsims : int
        Number of surrogate realisations.
    aposize : float
        Apodisation scale in degrees.
    apotype : str
        NaMaster apodisation type.
    bin_width : int
        Multipole bin width for NmtBin.from_nside_linear.
    ridge : float
        Tikhonov ridge for covariance inversion.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    results : dict
        Keys: ells, cl_xk_real, cl_sims, chi2_real, chi2_sims,
        p_surrogate, snr.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = default_rng(seed)

    # -- load data ----------------------------------------------------------
    mask = np.load(mask_path).astype(float)
    kappa_real = np.load(kappa_path).astype(float)
    xmap = np.load(tracer_path).astype(float)

    kappa_real = np.where(np.isfinite(kappa_real), kappa_real, 0.0)
    xmap = np.where(np.isfinite(xmap), xmap, 0.0)

    # -- infer nside from the map itself ------------------------------------
    nside = hp.get_nside(kappa_real)
    lmax = 3 * nside - 1

    # -- apodise mask and build binning -------------------------------------
    mask_apo = nmt.mask_apodization(mask, aposize=aposize, apotype=apotype)
    b = nmt.NmtBin.from_nside_linear(nside, bin_width)
    ells = b.get_effective_ells()

    # -- workspace (mode-coupling matrix, cached) ---------------------------
    f_x = nmt.NmtField(mask_apo, [xmap])
    f_k = nmt.NmtField(mask_apo, [kappa_real])

    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f_x, f_k, b)

    # -- real data cross-spectrum -------------------------------------------
    cl_xk_real = w.decouple_cell(
        nmt.compute_coupled_cell(f_x, f_k)
    )[0]

    # -- surrogate loop -----------------------------------------------------
    print(
        f"Surrogate Null: {nsims} phase-randomized kappa maps "
        f"(preserve C_ell^{{kk}})"
    )
    cl_sims = np.zeros((nsims, len(ells)), dtype=float)

    for i in range(nsims):
        ksim = get_surrogate_kappa(kappa_real, rng, lmax=lmax)
        f_ksim = nmt.NmtField(mask_apo, [ksim])
        cl_sims[i] = w.decouple_cell(
            nmt.compute_coupled_cell(f_x, f_ksim)
        )[0]
        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{nsims}]")

    # -- chi-squared --------------------------------------------------------
    mu = cl_sims.mean(axis=0)
    cov = np.cov(cl_sims, rowvar=False, ddof=1)
    cinv = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))

    diff = cl_xk_real - mu
    chi2_real = float(diff @ cinv @ diff)
    chi2_sims = np.einsum(
        "bi,ij,bj->b", cl_sims - mu, cinv, cl_sims - mu
    )
    p_surrogate = float(np.mean(chi2_sims >= chi2_real))
    snr = float(np.sqrt(max(chi2_real, 0.0)))

    # -- persist ------------------------------------------------------------
    np.save(os.path.join(out_dir, "ells.npy"), ells)
    np.save(os.path.join(out_dir, "cl_xk_real.npy"), cl_xk_real)
    np.save(os.path.join(out_dir, "cl_xk_sims.npy"), cl_sims)
    np.save(os.path.join(out_dir, "chi2_sims.npy"), chi2_sims)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"nsims_used={nsims}\n")
        f.write(f"chi2_real={chi2_real:.12e}\n")
        f.write(f"snr_sqrt_chi2={snr:.6f}\n")
        f.write(f"p_surrogate={p_surrogate:.6f}\n")
        f.write("null_type=phase_randomized_kappa_preserve_Cl\n")

    print("=" * 55)
    print(f"chi2_real     = {chi2_real:.6e}")
    print(f"SNR ~ sqrt(chi2) = {snr:.4f}")
    print(f"p_surrogate   = {p_surrogate:.4f}")
    print(f"Wrote: {out_dir}/summary.txt")
    print("=" * 55)

    return {
        "ells": ells,
        "cl_xk_real": cl_xk_real,
        "cl_sims": cl_sims,
        "chi2_real": chi2_real,
        "chi2_sims": chi2_sims,
        "p_surrogate": p_surrogate,
        "snr": snr,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Surrogate-null NaMaster cross-spectrum test"
    )
    parser.add_argument("--kappa", required=True, help="Path to kappa map (.npy)")
    parser.add_argument("--tracer", required=True, help="Path to tracer map (.npy)")
    parser.add_argument("--mask", required=True, help="Path to mask (.npy)")
    parser.add_argument("--out-dir", default="nmt_outputs", help="Output directory")
    parser.add_argument("--nsims", type=int, default=100, help="Number of surrogates")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    args = parser.parse_args()

    run_surrogate_null(
        kappa_path=args.kappa,
        tracer_path=args.tracer,
        mask_path=args.mask,
        out_dir=args.out_dir,
        nsims=args.nsims,
        seed=args.seed,
    )
