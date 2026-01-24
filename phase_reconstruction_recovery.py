#!/usr/bin/env python3
import argparse
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")  # headless-safe on cluster
import matplotlib.pyplot as plt

# Phase randomization: preserve |delta_k| (hence P(k)), destroy phases.
def phase_randomize_real_field(delta, seed=0):
    rng = np.random.default_rng(seed)
    F = np.fft.rfftn(delta)
    amp = np.abs(F)
    phase = rng.uniform(0, 2*np.pi, size=F.shape)

    # Enforce real constraints at DC/Nyquist bins
    nx, ny, nz = delta.shape
    kx_special = [0] + ([nx//2] if nx % 2 == 0 else [])
    ky_special = [0] + ([ny//2] if ny % 2 == 0 else [])
    kz_special = [0] + ([nz//2] if nz % 2 == 0 else [])

    for kx in kx_special:
        for ky in ky_special:
            for kz in kz_special:
                if kz < F.shape[2]:
                    phase[kx, ky, kz] = 0.0

    Fp = amp * np.exp(1j * phase)
    out = np.fft.irfftn(Fp, s=delta.shape)
    out = out - out.mean()
    out = out / (out.std() + 1e-12)
    return out

def cic_mesh(positions, weights, boxsize, boxcenter, nmesh):
    """
    Simple CIC onto a cubic mesh.
    positions: (N,3) in same units as boxsize/boxcenter
    """
    pos = positions - (boxcenter - boxsize/2.0)
    # normalize to [0, boxsize)
    grid = (pos / boxsize) * nmesh
    # clip inside box
    m = np.all((grid >= 0) & (grid < nmesh), axis=1)
    grid = grid[m]
    w = weights[m]

    ix = np.floor(grid).astype(int)
    dx = grid - ix

    mesh = np.zeros((nmesh, nmesh, nmesh), dtype=np.float64)

    for ox in (0, 1):
        wx = (1 - dx[:, 0]) if ox == 0 else dx[:, 0]
        i0 = (ix[:, 0] + ox)
        for oy in (0, 1):
            wy = (1 - dx[:, 1]) if oy == 0 else dx[:, 1]
            j0 = (ix[:, 1] + oy)
            for oz in (0, 1):
                wz = (1 - dx[:, 2]) if oz == 0 else dx[:, 2]
                k0 = (ix[:, 2] + oz)
                ww = w * wx * wy * wz
                # keep in bounds (non-periodic; consistent with survey box crop)
                ok = (i0 >= 0) & (i0 < nmesh) & (j0 >= 0) & (j0 < nmesh) & (k0 >= 0) & (k0 < nmesh)
                np.add.at(mesh, (i0[ok], j0[ok], k0[ok]), ww[ok])

    return mesh

def density_contrast(data_mesh, rand_mesh, alpha=None, eps=1e-12, min_rand=1.0):
    """
    delta = (n_data - alpha n_rand) / (alpha n_rand)
    alpha defaults to sum(data)/sum(rand)
    Handles empty random cells by masking.
    """
    if alpha is None:
        alpha = (data_mesh.sum() + eps) / (rand_mesh.sum() + eps)

    sel = (alpha * rand_mesh)
    mask = sel > min_rand  # cells with sufficient random support

    delta = np.zeros_like(data_mesh, dtype=np.float64)
    delta[mask] = (data_mesh[mask] - sel[mask]) / (sel[mask] + eps)

    # standardize only on supported cells
    mu = delta[mask].mean()
    sig = delta[mask].std() + 1e-12
    delta[mask] = (delta[mask] - mu) / sig

    return delta, float(alpha)

def voxel_graph_from_delta(delta, q=0.97, connectivity=6):
    """
    Keep top-q voxels of delta as "web".
    Build adjacency graph (6-neighbor).
    """
    thr = np.quantile(delta, q)
    mask = delta >= thr
    idx = np.argwhere(mask)
    n = idx.shape[0]
    if n < 200:
        raise RuntimeError(f"Too few web voxels ({n}). Lower q (e.g., 0.95).")

    key = {(i, j, k): t for t, (i, j, k) in enumerate(idx)}
    nx, ny, nz = delta.shape

    if connectivity == 6:
        nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    else:
        nbrs = [(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1) if not (a==b==c==0)]

    rows, cols = [], []
    for t, (i, j, k) in enumerate(idx):
        for di, dj, dk in nbrs:
            ni, nj, nk = i+di, j+dj, k+dk
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                u = key.get((ni, nj, nk))
                if u is not None:
                    rows.append(t); cols.append(u)

    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (A + A.T) * 0.5
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg, format="csr") - A
    return L, n

def spectral_dimension(L, tvals):
    """
    d_s(t) = -2 d ln P(t) / d ln t, P(t)=mean(exp(-t lambda))
    For n<=~5000, use dense eigvals.
    """
    n = L.shape[0]
    if n > 5000:
        raise RuntimeError(f"Graph too large (n={n}). Raise q (e.g., 0.98) or reduce nmesh.")

    lam = np.linalg.eigvalsh(L.toarray())
    lam = np.maximum(lam, 0.0)
    P = np.array([np.mean(np.exp(-t * lam)) for t in tvals])
    logt = np.log(tvals)
    logP = np.log(P + 1e-300)
    dlogP = np.gradient(logP, logt)
    ds = -2.0 * dlogP
    return P, ds

def plateau_stat(tvals, ds, tmin=None, tmax=None):
    """
    Robust median on [tmin, tmax]. If not provided, take middle 50% of log-t range.
    """
    if tmin is None or tmax is None:
        lt = np.log(tvals)
        lo = np.quantile(lt, 0.25)
        hi = np.quantile(lt, 0.75)
        m = (lt >= lo) & (lt <= hi)
    else:
        m = (tvals >= tmin) & (tvals <= tmax)
    return float(np.median(ds[m]))

def run_pyrecon_shift(data_pos, data_w, rand_pos, rand_w, f, bias, nmesh, boxsize, boxcenter, smoothing_radius, reciso):
    """
    Phase-informed reconstruction via DESI's pyrecon (BAO reconstruction).
    We use MultiGridReconstruction by default.
    """
    try:
        from pyrecon import MultiGridReconstruction
    except Exception as e:
        raise RuntimeError(
            "pyrecon is required for reconstruction. Install from cosmodesi/pyrecon."
        ) from e

    recon = MultiGridReconstruction(
        f=f, bias=bias, los=None,
        nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter
    )
    recon.assign_data(data_pos, data_w)
    recon.assign_randoms(rand_pos, rand_w)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    recon.run()

    data_pos_rec = recon.read_shifted_positions(data_pos)
    # RecSym default (field not specified or None?) -> RecSym uses disp+shift
    # Actually pyrecon default acts as RecSym.
    # If reciso is True, we want RecIso, which is removal of displacement from randoms?
    # Pyrecon Readme:
    # RecSym: read_shifted_positions(data, field='disp+shift') - default
    # RecIso: read_shifted_positions(data, field='disp+shift'), randoms field='disp'

    # Wait, the user snippet says:
    # rand_pos_rec = recon.read_shifted_positions(rand_pos) # RecSym default
    # if args.reciso: rand_pos_rec = recon.read_shifted_positions(rand_pos, field='disp')

    if reciso:
         rand_pos_rec = recon.read_shifted_positions(rand_pos, field='disp')
    else:
         rand_pos_rec = recon.read_shifted_positions(rand_pos)

    return data_pos_rec, rand_pos_rec

def main():
    ap = argparse.ArgumentParser(description="Final Kill-Switch: Phase-Reconstruction Recovery test")
    ap.add_argument("--data-pos", required=True, help="npy file (N,3) data positions")
    ap.add_argument("--rand-pos", required=True, help="npy file (M,3) random positions")
    ap.add_argument("--data-w", default=None, help="optional npy file (N,) weights")
    ap.add_argument("--rand-w", default=None, help="optional npy file (M,) weights")

    ap.add_argument("--nmesh", type=int, default=128)
    ap.add_argument("--boxsize", type=float, default=2000.0)
    ap.add_argument("--boxcenter", type=float, default=0.0, help="if scalar, uses [c,c,c]")
    ap.add_argument("--q", type=float, default=0.97, help="top-quantile for web voxels")
    ap.add_argument("--connectivity", type=int, default=6, choices=[6, 26])
    ap.add_argument("--min-rand", type=float, default=1.0, help="min expected randoms per cell for delta mask (selection function threshold)")

    ap.add_argument("--hon1-seed", type=int, default=0)
    ap.add_argument("--f", type=float, default=0.8, help="growth rate f for reconstruction")
    ap.add_argument("--bias", type=float, default=2.0, help="linear bias for reconstruction")
    ap.add_argument("--smooth", type=float, default=15.0, help="Reconstruction smoothing radius (Mpc/h)")
    ap.add_argument("--reciso", action="store_true", help="Use RecIso for randoms (field='disp')")

    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--out", default="phase_recon_recovery_out.json")
    args = ap.parse_args()

    data_pos = np.load(args.data_pos)
    rand_pos = np.load(args.rand_pos)
    data_w = np.ones(data_pos.shape[0]) if args.data_w is None else np.load(args.data_w)
    rand_w = np.ones(rand_pos.shape[0]) if args.rand_w is None else np.load(args.rand_w)

    boxcenter = np.array([args.boxcenter]*3) if np.isscalar(args.boxcenter) else np.array(args.boxcenter)
    nmesh = args.nmesh

    # 1) Build observed density mesh and delta
    data_mesh = cic_mesh(data_pos, data_w, args.boxsize, boxcenter, nmesh)
    rand_mesh = cic_mesh(rand_pos, rand_w, args.boxsize, boxcenter, nmesh)
    delta_data, alpha = density_contrast(data_mesh, rand_mesh, min_rand=args.min_rand)

    # 2) HON-1: phase randomize delta at fixed |F(k)|
    delta_hon1 = phase_randomize_real_field(delta_data, seed=args.hon1_seed)

    # 3) Reconstruction: run pyrecon to shift positions (phase-informed displacement field)
    data_pos_rec, rand_pos_rec = run_pyrecon_shift(
        data_pos, data_w, rand_pos, rand_w,
        f=args.f, bias=args.bias,
        nmesh=nmesh, boxsize=args.boxsize, boxcenter=boxcenter,
        smoothing_radius=args.smooth, reciso=args.reciso
    )
    data_mesh_rec = cic_mesh(data_pos_rec, data_w, args.boxsize, boxcenter, nmesh)
    rand_mesh_rec = cic_mesh(rand_pos_rec, rand_w, args.boxsize, boxcenter, nmesh)
    delta_recon, _ = density_contrast(data_mesh_rec, rand_mesh_rec, alpha=alpha, min_rand=args.min_rand)

    # 4) Build graphs and compute d_s(t)
    tvals = np.logspace(-2, 1.2, 24)  # dimensionless times; consistent across comparisons

    def compute(delta):
        L, n = voxel_graph_from_delta(delta, q=args.q, connectivity=args.connectivity)
        P, ds = spectral_dimension(L, tvals)
        ds_star = plateau_stat(tvals, ds, args.tmin, args.tmax)
        return {"n_nodes": int(n), "ds": ds.tolist(), "ds_star": ds_star}

    res_data = compute(delta_data)
    res_hon1 = compute(delta_hon1)
    res_recon = compute(delta_recon)

    out = {
        "alpha": alpha,
        "params": vars(args),
        "ds_star": {
            "data": res_data["ds_star"],
            "hon1": res_hon1["ds_star"],
            "recon": res_recon["ds_star"],
        },
        "nodes": {
            "data": res_data["n_nodes"],
            "hon1": res_hon1["n_nodes"],
            "recon": res_recon["n_nodes"],
        },
        "tvals": tvals.tolist(),
        "reconstruction_convention": "RecIso (field='disp' for randoms)" if args.reciso else "RecSym (default pyrecon behavior)",
        "selection_function": {
            "min_rand_threshold": args.min_rand,
            "description": f"Voxels with alpha*n_rand < {args.min_rand} are masked (set to zero in delta field)"
        },
        "notes": {
            "hon1": "Phase randomization applied to survey-windowed delta field on finite box (not periodic cosmological volume).",
            "delta_masking": f"Cells with insufficient random support (< {args.min_rand} expected randoms) are masked to avoid division-by-zero artifacts.",
            "reconstruction": "BAO reconstruction via pyrecon uses linear displacement field to approximately invert Zel'dovich dynamics.",
            "audit_grade": "For referee-proof results, run ensemble over multiple HON-1 seeds, RANN realizations, and smoothing scales."
        }
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    # Plot d_s(t)
    plt.figure()
    plt.semilogx(tvals, np.array(res_data["ds"]), label="DATA")
    plt.semilogx(tvals, np.array(res_hon1["ds"]), label="HON-1 (phase-rand)")
    plt.semilogx(tvals, np.array(res_recon["ds"]), label="RECON (pyrecon)")
    plt.xlabel("Diffusion time t (arb.)")
    plt.ylabel("d_s(t)")
    plt.title("Phase-Reconstruction Recovery Kill-Switch")
    plt.legend()
    plt.tight_layout()
    plt.savefig('phase_reconstruction_recovery.png')
    print("Plot saved to phase_reconstruction_recovery.png")

    print("[RESULT] d_s* DATA =", res_data["ds_star"])
    print("[RESULT] d_s* HON1 =", res_hon1["ds_star"])
    print("[RESULT] d_s* RECON=", res_recon["ds_star"])
    print(f"[WROTE] {args.out}")

if __name__ == "__main__":
    main()
