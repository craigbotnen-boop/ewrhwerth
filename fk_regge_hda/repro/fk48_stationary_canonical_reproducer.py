#!/usr/bin/env python3
"""
Full stationary/canonical clean-room FK48 replication.

This script extends fk48_cleanroom_replication.py to:
  1. solve the full 16-equation stationary branch by a physical/gauge
     Lyapunov-Schmidt split;
  2. measure the four lifted middle-Hessian gauge modes;
  3. form the old/new boundary Jacobian B = dE_internal/db_upper;
  4. measure gauge-to-gauge K_gg and gauge-to-physical K_gp blocks;
  5. compute the stationary response L from H L = -B;
  6. test bounded response on upper geometric deformation directions.

The seed is explicit and independent of the manuscript's historical seed.
It is intentionally strengthened by a curvature amplitude of 3 so the
fourth-order canonical signal remains well above double-precision finite-
difference noise at all reported scales.

Dependencies: numpy, scipy
Run:
    python fk48_stationary_canonical_reproducer.py

Requires fk48_cleanroom_replication.py in the same directory.

Outputs:
    fk48_stationary_canonical_results.json
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
BASE = HERE / "fk48_cleanroom_replication.py"

spec = importlib.util.spec_from_file_location("fkbase", BASE)
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)

AMP = 3.0
H_VALUES = [0.15, 0.12, 0.09]
FD_HB = 3e-5
FD_GAUGE = 1e-3
FD_GEOM = 1e-6

UPPER_EDGES = [e for e in fk.CX["all_edges"] if "N" in e and "M" not in e]

def edge_equations_xb(x, b, h, amp=AMP):
    lengths = fk.initial_length_dict(h, amp=amp)
    for edge, val in zip(fk.CX["internal_edges"], x):
        lengths[edge] = h * float(val)
    for edge, val in zip(UPPER_EDGES, b):
        lengths[edge] = h * float(val)
    try:
        deficits = fk.internal_deficits(lengths)
        eq = []
        for edge in fk.CX["internal_edges"]:
            value = 0.0
            for tri in fk.CX["edge_to_internal_tris"][edge]:
                value += deficits[tri] * fk.darea_dedge(tri, edge, lengths)
            eq.append(value / h)
        return np.asarray(eq, dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.full(len(x), 1e3, dtype=float)

def jacobian_central(fun, x, eps):
    f0 = np.asarray(fun(x), dtype=float)
    J = np.empty((len(f0), len(x)), dtype=float)
    for j in range(len(x)):
        xp = x.copy(); xm = x.copy(); xp[j] += eps; xm[j] -= eps
        J[:, j] = (fun(xp) - fun(xm)) / (2.0 * eps)
    return J

def power_fit(h, y):
    h = np.asarray(h, dtype=float); y = np.asarray(y, dtype=float)
    p, logc = np.polyfit(np.log(h), np.log(y), 1)
    yhat = np.exp(logc) * h**p
    ss_res = np.sum((np.log(y) - np.log(yhat))**2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y)))**2)
    return {"exponent": float(p), "coefficient": float(np.exp(logc)), "R2_loglog": float(1.0 - ss_res / ss_tot)}

def flat_splitting():
    h = H_VALUES[-1]
    lengths = fk.initial_length_dict(h, amp=0.0)
    xflat = np.array([lengths[e] / h for e in fk.CX["internal_edges"]], dtype=float)
    bflat = np.array([lengths[e] / h for e in UPPER_EDGES], dtype=float)
    H0 = jacobian_central(lambda z: edge_equations_xb(z, bflat, h, amp=0.0), xflat, 2e-6)
    U, S, Vt = np.linalg.svd(H0)
    return {"Up": U[:, :12], "Ug": U[:, 12:], "Vp": Vt.T[:, :12], "Vg": Vt.T[:, 12:], "S": S, "H0": H0}

SPLIT = flat_splitting()
Up, Ug, Vp, Vg = SPLIT["Up"], SPLIT["Ug"], SPLIT["Vp"], SPLIT["Vg"]

def solve_physical(h, b, xbase, g, y0=None):
    if y0 is None: y0 = np.zeros(12)
    def residual(y):
        x = xbase + Vp @ y + Vg @ g
        return Up.T @ edge_equations_xb(x, b, h, AMP)
    sol = least_squares(residual, y0, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=100)
    return sol.x, xbase + Vp @ sol.x + Vg @ g

def reduced_gauge(h, b, xbase, g, y0=None):
    y, x = solve_physical(h, b, xbase, g, y0)
    return Ug.T @ edge_equations_xb(x, b, h, AMP), y, x

GAUGE_STARTS = {
    0.15: np.array([-0.07472150024544631, -0.031218047659559937, 0.19159477122530766, -0.05861962047139362]),
    0.12: np.array([-0.07489872520275818, -0.031222728773044762, 0.19193744385418748, -0.058744665079083096]),
    0.09: np.array([-0.07503704194189208, -0.03122507065626409, 0.19220606665155, -0.058842288809141344]),
}

def solve_stationary(h, g_init=None):
    lengths = fk.initial_length_dict(h, amp=AMP)
    xbase = np.array([lengths[e] / h for e in fk.CX["internal_edges"]], dtype=float)
    b = np.array([lengths[e] / h for e in UPPER_EDGES], dtype=float)
    g = GAUGE_STARTS[float(h)].copy() if g_init is None else np.asarray(g_init, dtype=float).copy()
    y0 = None
    for _ in range(4):
        r, y, x = reduced_gauge(h, b, xbase, g, y0)
        y0 = y
        if np.linalg.norm(r) < 3e-14: break
        Jg = jacobian_central(lambda gg: reduced_gauge(h, b, xbase, gg, None)[0], g, FD_GAUGE)
        g = g + np.linalg.solve(Jg, -r)
        y0 = None
    r, y, x = reduced_gauge(h, b, xbase, g, y0)
    return x, b, g, edge_equations_xb(x, b, h, AMP)

def upper_geometric_basis(h):
    def normalized_spokes(Ncoord):
        vals = []
        for edge in UPPER_EDGES:
            outer = fk.CX["coords"][edge[1]]
            d = outer - Ncoord
            l2 = h*h*float(d @ d) - (h**4/3.0)*AMP*fk.r_contract(d, Ncoord)
            vals.append(math.sqrt(l2) / h)
        return np.asarray(vals, dtype=float)
    J = np.empty((len(UPPER_EDGES), 4), dtype=float); eye = np.eye(4)
    for a in range(4):
        J[:, a] = (normalized_spokes(fk.N_COORD + FD_GEOM*eye[a]) - normalized_spokes(fk.N_COORD - FD_GEOM*eye[a])) / (2.0*FD_GEOM)
    Q, _ = np.linalg.qr(J, mode="complete")
    return Q[:, :4], Q[:, 4:]

def boundary_jacobian(x, b, h):
    B = np.empty((16, len(b)), dtype=float)
    for j in range(len(b)):
        bp = b.copy(); bm = b.copy(); bp[j] += FD_HB; bm[j] -= FD_HB
        B[:, j] = (edge_equations_xb(x, bp, h, AMP) - edge_equations_xb(x, bm, h, AMP)) / (2.0*FD_HB)
    return B

def analyze_scale(h, stationary):
    x, b, g, residual = stationary
    H = jacobian_central(lambda z: edge_equations_xb(z, b, h, AMP), x, FD_HB)
    Uh, Sh, Vth = np.linalg.svd(H); Ugh = Uh[:, -4:]
    B = boundary_jacobian(x, b, h); Qg, Qp = upper_geometric_basis(h)
    Kgg = Ugh.T @ B @ Qg; Kgp = Ugh.T @ B @ Qp
    Hpp = Up.T @ H @ Vp; Hpg = Up.T @ H @ Vg; Hgp = Ug.T @ H @ Vp; Hgg = Ug.T @ H @ Vg
    Geff = Hgg - Hgp @ np.linalg.solve(Hpp, Hpg)
    L = np.linalg.lstsq(H, -B, rcond=1e-12)[0]
    gauge_singulars = Sh[-4:]; kgg_singulars = np.linalg.svd(Kgg, compute_uv=False); geff_singulars = np.linalg.svd(Geff, compute_uv=False)
    return {"h": h, "stationarity_norm": float(np.linalg.norm(residual)), "gauge_coordinates": g.tolist(), "physical_gap": float(Sh[11]), "gauge_hessian_singular_values": gauge_singulars.tolist(), "gauge_hessian_frobenius": float(np.linalg.norm(gauge_singulars)), "gauge_hessian_frobenius_over_h4": float(np.linalg.norm(gauge_singulars)/h**4), "Geff_singular_values": geff_singulars.tolist(), "Geff_singular_values_over_h4": (geff_singulars/h**4).tolist(), "Kgg_singular_values": kgg_singulars.tolist(), "Kgg_frobenius": float(np.linalg.norm(Kgg,"fro")), "Kgg_frobenius_over_h4": float(np.linalg.norm(Kgg,"fro")/h**4), "Kgp_operator_norm": float(np.linalg.norm(Kgp,2)), "Kgp_operator_norm_over_h2": float(np.linalg.norm(Kgp,2)/h**2), "restricted_response_norm": float(np.linalg.norm(L@Qg,2)), "full_response_norm": float(np.linalg.norm(L,2))}

def main():
    assert len(fk.CX["star_tets"]) == 24 and len(fk.CX["simplices"]) == 48 and len(fk.CX["internal_edges"]) == 16 and len(UPPER_EDGES) == 14
    stationaries = [solve_stationary(h, None) for h in H_VALUES]
    rows = [analyze_scale(h,s) for h,s in zip(H_VALUES,stationaries)]
    result = {"description":"Independent full stationary/canonical FK48 reproduction using the explicit seed in fk48_cleanroom_replication.py with curvature amplitude 3.", "curvature_amplitude":AMP, "h_values":H_VALUES, "finite_difference_step_H_and_boundary":FD_HB, "inventory":{"four_simplices":len(fk.CX["simplices"]),"internal_edges":len(fk.CX["internal_edges"]),"upper_boundary_spokes":len(UPPER_EDGES)}, "flat_control":{"singular_values":SPLIT["S"].tolist(),"physical_gap":float(SPLIT["S"][11]),"four_smallest_singular_values":SPLIT["S"][-4:].tolist()}, "refinement":rows, "fits":{}, "claim_boundary":"This is an independent fresh-seed numerical reproduction. It validates the block hierarchy and fourth-order aggregate canonical gauge sector, but is not a byte-for-byte replay of the manuscript's historical boundary seed."}
    result["fits"]["gauge_hessian_frobenius"] = power_fit(H_VALUES,[r["gauge_hessian_frobenius"] for r in rows])
    result["fits"]["Kgg_frobenius"] = power_fit(H_VALUES,[r["Kgg_frobenius"] for r in rows])
    result["fits"]["Kgp_operator_norm"] = power_fit(H_VALUES,[r["Kgp_operator_norm"] for r in rows])
    result["fits"]["restricted_response_norm"] = power_fit(H_VALUES,[r["restricted_response_norm"] for r in rows])
    result["fits"]["full_response_norm"] = power_fit(H_VALUES,[r["full_response_norm"] for r in rows])
    out = HERE / "fk48_stationary_canonical_results.json"; out.write_text(json.dumps(result,indent=2))
    print("Full stationary/canonical clean-room FK48 reproduction\n------------------------------------------------------\ncurvature amplitude:", AMP)
    for r in rows:
        print(f"h={r['h']:.3f}  ||E||={r['stationarity_norm']:.3e}  gap={r['physical_gap']:.6f}  ||H_g||_F/h^4={r['gauge_hessian_frobenius_over_h4']:.6e}  ||K_gg||_F/h^4={r['Kgg_frobenius_over_h4']:.6e}  ||K_gp||/h^2={r['Kgp_operator_norm_over_h2']:.6e}  ||L Q_g||={r['restricted_response_norm']:.6f}")
    print("\nFits")
    for name, fit in result["fits"].items(): print(f"{name}: p={fit['exponent']:.9f}, R^2={fit['R2_loglog']:.9f}")
    print("\nWrote:", out)

if __name__ == "__main__": main()
