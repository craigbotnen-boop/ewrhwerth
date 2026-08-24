#!/usr/bin/env python3
"""
Clean-room FK48 Regge pseudo-constraint replication.

Purpose
-------
Independently reconstruct a two-step Freudenthal-Kuhn (FK) length-Regge
complex with 48 four-simplices, generate a fresh explicitly specified
algebraic-curvature boundary seed, solve the 16 internal Regge edge equations
in least-squares form, and test whether the unresolved pseudo-constraint
obstruction:
  (i) lies in the four flat vertex-displacement gauge directions, and
  (ii) scales as O(h^4).

This script does NOT attempt to reproduce the original manuscript seed.
It is an independent control with a fully specified seed.

Dependencies
------------
Python 3.10+
numpy
scipy

Run
---
python fk48_cleanroom_replication.py

Outputs
-------
fk48_cleanroom_results.json
"""

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

H_VALUES = [0.09, 0.065, 0.045]

RICCI = np.array([
    [ 0.120,  0.030, -0.020,  0.025],
    [ 0.030, -0.080,  0.018, -0.015],
    [-0.020,  0.018,  0.050,  0.022],
    [ 0.025, -0.015,  0.022,  0.020],
], dtype=float)

O_COORD = np.array([ 0.20, -0.15,  0.10, -0.80], dtype=float)
M_REF   = np.array([ 0.00,  0.00,  0.00,  0.00], dtype=float)
N_COORD = np.array([-0.25,  0.30, -0.20,  0.90], dtype=float)


def fk_star_tetrahedra():
    tets = set()
    origin = (0, 0, 0)
    eye = np.eye(3, dtype=int)
    for low_tuple in itertools.product([-1, 0], repeat=3):
        low = np.array(low_tuple, dtype=int)
        for perm in itertools.permutations(range(3)):
            verts = [low.copy()]
            cur = low.copy()
            for ax in perm:
                cur = cur + eye[ax]
                verts.append(cur.copy())
            tet = tuple(sorted(tuple(map(int, v)) for v in verts))
            if origin in tet:
                tets.add(tet)
    return sorted(tets)


def build_complex():
    star_tets = fk_star_tetrahedra()
    outer = sorted({v for tet in star_tets for v in tet if v != (0, 0, 0)})
    offset_to_label = {v: f"v{i}" for i, v in enumerate(outer)}
    simplices = []
    for tet in star_tets:
        nbrs = [offset_to_label[v] for v in tet if v != (0, 0, 0)]
        assert len(nbrs) == 3
        simplices.append(tuple(["O", "M"] + nbrs))
        simplices.append(tuple(["M", "N"] + nbrs))

    edge_counts = Counter()
    tri_counts = Counter()
    tet_counts = Counter()
    for s in simplices:
        for e in itertools.combinations(s, 2):
            edge_counts[tuple(sorted(e))] += 1
        for t in itertools.combinations(s, 3):
            tri_counts[tuple(sorted(t))] += 1
        for q in itertools.combinations(s, 4):
            tet_counts[tuple(sorted(q))] += 1

    boundary_tets = {q for q, c in tet_counts.items() if c == 1}
    boundary_tris = {tuple(sorted(t)) for q in boundary_tets for t in itertools.combinations(q, 3)}
    internal_tris = set(tri_counts) - boundary_tris
    boundary_edges = {tuple(sorted(e)) for q in boundary_tets for e in itertools.combinations(q, 2)}
    internal_edges = sorted(set(edge_counts) - boundary_edges)

    coords = {}
    for i, v in enumerate(outer):
        coords[f"v{i}"] = np.array([v[0], v[1], v[2], 0.0], dtype=float)
    coords["O"] = O_COORD.copy()
    coords["M"] = M_REF.copy()
    coords["N"] = N_COORD.copy()

    edge_to_internal_tris = {
        e: [t for t in internal_tris if set(e).issubset(t)]
        for e in internal_edges
    }
    return {
        "star_tets": star_tets,
        "outer": outer,
        "simplices": simplices,
        "all_edges": sorted(edge_counts),
        "internal_tris": internal_tris,
        "internal_edges": internal_edges,
        "edge_to_internal_tris": edge_to_internal_tris,
        "coords": coords,
    }

CX = build_complex()


def riemann_from_ricci(ric):
    ric = np.asarray(ric, dtype=float)
    scal = np.trace(ric)
    delta = np.eye(4)
    R = np.zeros((4, 4, 4, 4), dtype=float)
    for a, b, c, d in itertools.product(range(4), repeat=4):
        R[a, b, c, d] = (
            0.5 * (
                delta[a, c] * ric[b, d]
                + delta[b, d] * ric[a, c]
                - delta[a, d] * ric[b, c]
                - delta[b, c] * ric[a, d]
            )
            - (scal / 6.0) * (
                delta[a, c] * delta[b, d]
                - delta[a, d] * delta[b, c]
            )
        )
    return R

RIEMANN = riemann_from_ricci(RICCI)


def r_contract(d, u, R=RIEMANN):
    return float(np.einsum("a,b,c,d,abcd", d, u, d, u, R))


def edge_len_rnc(a, b, h, amp=1.0):
    x = CX["coords"][a]
    y = CX["coords"][b]
    d = y - x
    l2 = h * h * float(d @ d) - (h**4 / 3.0) * amp * r_contract(d, x)
    if l2 <= 0:
        raise ValueError(f"Nonpositive edge square for {(a,b)}: {l2}")
    return math.sqrt(l2)


def initial_length_dict(h, amp=1.0):
    return {e: edge_len_rnc(e[0], e[1], h, amp=amp) for e in CX["all_edges"]}


def simplex_coords_from_lengths(simplex, lengths):
    D = np.zeros((5, 5), dtype=float)
    for i, j in itertools.combinations(range(5), 2):
        e = tuple(sorted((simplex[i], simplex[j])))
        D[i, j] = D[j, i] = lengths[e]
    B = np.zeros((4, 4), dtype=float)
    for i in range(1, 5):
        for j in range(1, 5):
            B[i - 1, j - 1] = 0.5 * (D[0, i] ** 2 + D[0, j] ** 2 - D[i, j] ** 2)
    evals, evecs = np.linalg.eigh(B)
    if evals.min() <= 1e-13:
        raise ValueError(f"Nonpositive simplex Gram spectrum: {evals}")
    X = np.zeros((5, 4), dtype=float)
    X[1:] = evecs @ np.diag(np.sqrt(evals))
    return X


def simplex_dihedral_angles(simplex, lengths):
    X = simplex_coords_from_lengths(simplex, lengths)
    normals = []
    for missing in range(5):
        ids = [i for i in range(5) if i != missing]
        p0 = X[ids[0]]
        mat = np.stack([X[i] - p0 for i in ids[1:]], axis=0)
        _, _, vh = np.linalg.svd(mat)
        n = vh[-1]
        n /= np.linalg.norm(n)
        if float(n @ (X[missing] - p0)) > 0:
            n = -n
        normals.append(n)
    out = {}
    for i, j in itertools.combinations(range(5), 2):
        c = float(np.clip(-normals[i] @ normals[j], -1.0, 1.0))
        theta = math.acos(c)
        hinge = tuple(sorted(simplex[k] for k in range(5) if k not in (i, j)))
        out[hinge] = theta
    return out


def internal_deficits(lengths):
    angle_sum = {t: 0.0 for t in CX["internal_tris"]}
    for simplex in CX["simplices"]:
        angles = simplex_dihedral_angles(simplex, lengths)
        for tri, theta in angles.items():
            if tri in angle_sum:
                angle_sum[tri] += theta
    return {t: 2.0 * math.pi - s for t, s in angle_sum.items()}


def triangle_area(tri, lengths):
    a, b, c = [lengths[tuple(sorted(e))] for e in itertools.combinations(tri, 2)]
    sem = 0.5 * (a + b + c)
    return math.sqrt(max(sem * (sem - a) * (sem - b) * (sem - c), 0.0))


def darea_dedge(tri, edge, lengths):
    a = lengths[edge]
    other = [
        lengths[tuple(sorted(e))]
        for e in itertools.combinations(tri, 2)
        if tuple(sorted(e)) != edge
    ]
    b, c = other
    A = triangle_area(tri, lengths)
    return a * (b * b + c * c - a * a) / (8.0 * A)


def edge_equations_x(x, h, amp=1.0):
    lengths = initial_length_dict(h, amp=amp)
    for edge, val in zip(CX["internal_edges"], x):
        lengths[edge] = h * float(val)
    try:
        deficits = internal_deficits(lengths)
        equations = []
        for edge in CX["internal_edges"]:
            val = 0.0
            for tri in CX["edge_to_internal_tris"][edge]:
                val += deficits[tri] * darea_dedge(tri, edge, lengths)
            equations.append(val / h)
        return np.asarray(equations, dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.full(len(x), 1e3, dtype=float)


def jacobian_central(fun, x, eps=1e-6):
    f0 = fun(x)
    J = np.empty((len(f0), len(x)), dtype=float)
    for j in range(len(x)):
        xp = x.copy(); xm = x.copy()
        xp[j] += eps; xm[j] -= eps
        J[:, j] = (fun(xp) - fun(xm)) / (2.0 * eps)
    return J


def fit_power(h, y):
    h = np.asarray(h, dtype=float); y = np.asarray(y, dtype=float)
    p, logc = np.polyfit(np.log(h), np.log(y), 1)
    yhat = np.exp(logc) * h**p
    ss_res = np.sum((np.log(y) - np.log(yhat)) ** 2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y))) ** 2)
    return float(p), float(np.exp(logc)), float(1.0 - ss_res / ss_tot)


def main():
    assert len(CX["star_tets"]) == 24
    assert len(CX["outer"]) == 14
    assert len(CX["simplices"]) == 48
    assert len(CX["internal_edges"]) == 16

    h_ref = H_VALUES[0]
    flat_lengths = initial_length_dict(h_ref, amp=0.0)
    x_flat = np.array([flat_lengths[e] / h_ref for e in CX["internal_edges"]], dtype=float)
    f_flat = edge_equations_x(x_flat, h_ref, amp=0.0)
    J_flat = jacobian_central(lambda z: edge_equations_x(z, h_ref, amp=0.0), x_flat, eps=1e-6)
    U, S, Vt = np.linalg.svd(J_flat)
    U_phys = U[:, :12]
    U_gauge = U[:, -4:]

    results = {
        "description": "Independent clean-room FK48 length-Regge pseudo-constraint replication with a fresh explicit Weyl-free Ricci seed.",
        "seed": {
            "ricci": RICCI.tolist(),
            "old_vertex": O_COORD.tolist(),
            "middle_reference": M_REF.tolist(),
            "new_vertex": N_COORD.tolist(),
            "world_function": "l^2=h^2|d|^2-(h^4/3) R(d,u,d,u)"
        },
        "inventory": {
            "fk_spatial_tetrahedra": len(CX["star_tets"]),
            "outer_neighbors": len(CX["outer"]),
            "four_simplices": len(CX["simplices"]),
            "internal_edges": len(CX["internal_edges"]),
            "internal_triangles": len(CX["internal_tris"])
        },
        "flat_control": {
            "equation_norm": float(np.linalg.norm(f_flat)),
            "singular_values": S.tolist(),
            "physical_gap_12th": float(S[11]),
            "four_smallest_singular_values": S[-4:].tolist()
        },
        "refinement": []
    }

    previous = None
    gauge_norms = []
    for h in H_VALUES:
        lengths0 = initial_length_dict(h, amp=1.0)
        x0 = np.array([lengths0[e] / h for e in CX["internal_edges"]], dtype=float)
        if previous is not None:
            x0 = previous.copy()
        sol = least_squares(
            lambda z: edge_equations_x(z, h, amp=1.0), x0,
            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=300
        )
        x = sol.x; previous = x.copy()
        f = edge_equations_x(x, h, amp=1.0)
        gauge_part = U_gauge.T @ f
        phys_part = U_phys.T @ f
        J = jacobian_central(lambda z: edge_equations_x(z, h, amp=1.0), x, eps=2e-6)
        svals = np.linalg.svd(J, compute_uv=False)
        gnorm = float(np.linalg.norm(gauge_part)); gauge_norms.append(gnorm)
        results["refinement"].append({
            "h": h,
            "least_squares_success": bool(sol.success),
            "nfev": int(sol.nfev),
            "total_residual_norm": float(np.linalg.norm(f)),
            "gauge_projected_residual_norm": gnorm,
            "physical_projected_residual_norm": float(np.linalg.norm(phys_part)),
            "physical_fraction_of_total": float(np.linalg.norm(phys_part) / np.linalg.norm(f)),
            "gauge_residual_over_h4": float(gnorm / h**4),
            "four_smallest_jacobian_singular_values": svals[-4:].tolist(),
            "internal_x_solution": x.tolist()
        })

    exponent, coeff, r2 = fit_power(H_VALUES, gauge_norms)
    results["gauge_obstruction_fit"] = {"exponent": exponent, "coefficient": coeff, "R2_loglog": r2}
    out = Path(__file__).with_name("fk48_cleanroom_results.json")
    out.write_text(json.dumps(results, indent=2))

    print("FK48 clean-room replication")
    print("Inventory:", results["inventory"])
    print("Flat equation norm:", results["flat_control"]["equation_norm"])
    print("Flat physical gap:", results["flat_control"]["physical_gap_12th"])
    for row in results["refinement"]:
        print(f"h={row['h']:.3f} ||E_g||={row['gauge_projected_residual_norm']:.12e} ||E_g||/h^4={row['gauge_residual_over_h4']:.12e} phys/total={row['physical_fraction_of_total']:.3e}")
    print(f"Gauge obstruction fit: p={exponent:.12f}, R^2={r2:.12f}")
    print("Wrote:", out)

if __name__ == "__main__":
    main()
