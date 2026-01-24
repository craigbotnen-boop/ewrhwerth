#!/usr/bin/env python3
"""
Fractal Discrete Scale Invariance (DSI) Analysis

This module implements comparative analysis of hierarchical fingerprints in
self-similar fractals (Sierpiński Carpet, Sierpiński Gasket) via entropy
gradient force ratios.

Key Components:
1. Sparse CSR fractal generation (memory-efficient for L > 1000)
2. BFS shell counting using graph distance (honors intrinsic metric)
3. Entropy gradient force ratio F(r) = S(r+1) - S(r)
4. Log-periodic oscillation detection (DSI signature)
5. Euclidean 2D grid baseline (control for discrete artifacts)

Scientific Context:
- Sierpiński Carpet: scaling factor λ = 3, df = log(8)/log(3) ≈ 1.893
- Sierpiński Gasket: scaling factor λ = 2, df = log(3)/log(2) ≈ 1.585
- Expected log-periodic wavelength: Δlog(r) = log(λ)

References:
- Hambly et al. (1994): Heat kernel asymptotics on fractals
- Barlow & Bass (1999): Random walks on fractal graphs
- Sornette (1998): Discrete scale invariance and complex dimensions
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Headless mode for cluster compatibility


# ==============================================================================
# Sierpiński Carpet Generation (Sparse CSR)
# ==============================================================================

def sierpinski_carpet_sparse(level, remove_center=True):
    """
    Generate Sierpiński Carpet as sparse CSR adjacency matrix.

    Uses iterative subdivision: each filled square -> 8 squares (remove center).

    Parameters
    ----------
    level : int
        Subdivision level (L = 3^level grid size)
    remove_center : bool
        If True, remove center square (standard carpet). If False, keep filled.

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Adjacency matrix (4-neighbor connectivity on 2D grid)
    coords : ndarray (N, 2)
        Node coordinates in 2D
    L : int
        Grid size (3^level)
    """
    L = 3 ** level

    # Build filled/removed mask via recursive subdivision
    def is_filled(i, j, lev):
        """Check if cell (i,j) is filled at level lev."""
        if lev == 0:
            return True
        # Divide into 3x3 grid
        cell_size = 3 ** (lev - 1)
        sub_i = i // cell_size
        sub_j = j // cell_size
        # Remove center cell (1,1) if remove_center=True
        if remove_center and sub_i == 1 and sub_j == 1:
            return False
        # Recurse
        return is_filled(i % cell_size, j % cell_size, lev - 1)

    # Generate filled cells
    filled = []
    for i in range(L):
        for j in range(L):
            if is_filled(i, j, level):
                filled.append((i, j))

    N = len(filled)
    print(f"Carpet L={L}: {N} nodes (fill fraction: {N/(L*L):.4f})")

    # Map (i,j) -> node index
    coord_to_idx = {(i, j): idx for idx, (i, j) in enumerate(filled)}
    coords = np.array(filled, dtype=np.float64)

    # Build sparse adjacency (4-neighbor)
    A = lil_matrix((N, N), dtype=np.float64)
    for idx, (i, j) in enumerate(filled):
        # Check 4 neighbors
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if (ni, nj) in coord_to_idx:
                neighbor_idx = coord_to_idx[(ni, nj)]
                A[idx, neighbor_idx] = 1.0

    return A.tocsr(), coords, L


# ==============================================================================
# Sierpiński Gasket Generation (3-copy corner identification)
# ==============================================================================

def sierpinski_gasket_sparse(level):
    """
    Generate Sierpiński Gasket as sparse CSR adjacency matrix.

    Uses "three copies + identify corners" PCF (post-critically finite) method:
    - Start with 3 nodes forming a triangle
    - At each level: make 3 copies, identify corner nodes

    This preserves the finite ramification property critical for walk dimension.

    Parameters
    ----------
    level : int
        Subdivision level (N ≈ 3^(level+1)/2 nodes)

    Returns
    -------
    A : scipy.sparse.csr_matrix
        Adjacency matrix
    coords : ndarray (N, 2)
        Approximate 2D embedding coordinates (for visualization)
    """
    if level == 0:
        # Base triangle: 3 nodes, 3 edges
        A = csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64))
        coords = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        return A, coords

    # Recursive construction
    A_prev, coords_prev = sierpinski_gasket_sparse(level - 1)
    N_prev = A_prev.shape[0]

    # Identify the 3 corner nodes (first 3 nodes by construction)
    corners = [0, 1, 2]

    # Make 3 copies
    N_new = 3 * N_prev - 3  # Subtract 3 because corners are shared

    # Build new adjacency
    A_new = lil_matrix((N_new, N_new), dtype=np.float64)
    coords_new = []

    # Node mapping: copy i, node j in old graph -> new index
    node_map = {}
    next_idx = 0

    # First, assign indices to shared corners
    for c in corners:
        node_map[(0, c)] = next_idx  # All copies share the same corner
        node_map[(1, c)] = next_idx
        node_map[(2, c)] = next_idx
        next_idx += 1

    # Assign indices to non-corner nodes in each copy
    for copy_idx in range(3):
        for j in range(N_prev):
            if j not in corners:
                node_map[(copy_idx, j)] = next_idx
                next_idx += 1

    # Copy edges from previous level
    A_prev_coo = A_prev.tocoo()
    for u, v in zip(A_prev_coo.row, A_prev_coo.col):
        for copy_idx in range(3):
            u_new = node_map[(copy_idx, u)]
            v_new = node_map[(copy_idx, v)]
            A_new[u_new, v_new] = 1.0

    # Generate approximate coordinates (for visualization)
    # Place 3 copies in triangle configuration
    offsets = [
        np.array([0, 0]),
        np.array([1, 0]),
        np.array([0.5, np.sqrt(3)/2])
    ]
    scale = 0.5 ** level

    for copy_idx in range(3):
        for j in range(N_prev):
            idx_new = node_map[(copy_idx, j)]
            if idx_new >= len(coords_new):
                coords_new.append(offsets[copy_idx] + scale * coords_prev[j])

    coords_new = np.array(coords_new)

    print(f"Gasket level={level}: {N_new} nodes")

    return A_new.tocsr(), coords_new


# ==============================================================================
# BFS Shell Counting (Graph Distance Metric)
# ==============================================================================

def bfs_shell_counts(A, center_node, r_max):
    """
    Compute shell occupancy N(r) using BFS from center node.

    Uses graph distance (intrinsic metric), not Euclidean distance.
    This is critical for fractals where Euclidean distance is misleading.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Adjacency matrix (symmetric, unweighted)
    center_node : int
        Starting node index
    r_max : int
        Maximum radius to explore

    Returns
    -------
    shells : ndarray of shape (r_max+1,)
        Number of nodes at each graph distance r
    """
    N = A.shape[0]
    distance = np.full(N, -1, dtype=np.int32)
    distance[center_node] = 0

    queue = deque([center_node])
    shells = np.zeros(r_max + 1, dtype=np.int32)
    shells[0] = 1

    while queue:
        u = queue.popleft()
        r_u = distance[u]
        if r_u >= r_max:
            continue

        # Explore neighbors
        for v in A.indices[A.indptr[u]:A.indptr[u+1]]:
            if distance[v] == -1:
                distance[v] = r_u + 1
                shells[r_u + 1] += 1
                queue.append(v)

    return shells


# ==============================================================================
# Euclidean 2D Grid Baseline (Control)
# ==============================================================================

def euclidean_grid_shell_counts(L, r_max):
    """
    BFS shell counts on standard 2D grid (control baseline).

    Uses same discrete graph-distance logic as fractal analysis.

    Parameters
    ----------
    L : int
        Grid size (L x L)
    r_max : int
        Maximum radius

    Returns
    -------
    shells : ndarray
        Shell occupancy N(r)
    """
    # Build 2D grid adjacency (4-neighbor)
    N = L * L
    A = dok_matrix((N, N), dtype=np.float64)

    for i in range(L):
        for j in range(L):
            u = i * L + j
            # Right neighbor
            if j + 1 < L:
                v = i * L + (j + 1)
                A[u, v] = A[v, u] = 1.0
            # Down neighbor
            if i + 1 < L:
                v = (i + 1) * L + j
                A[u, v] = A[v, u] = 1.0

    A = A.tocsr()

    # BFS from center
    center = (L // 2) * L + (L // 2)
    return bfs_shell_counts(A, center, r_max)


# ==============================================================================
# Entropy Gradient Force Ratio F(r)
# ==============================================================================

def entropy_gradient_force_ratio(shells):
    """
    Compute entropy gradient "force" F(r) = S(r+1) - S(r).

    Where S(r) = log(N(r)) is the shell entropy.

    F(r) measures the rate of change of accessible states as a function of
    graph distance. Log-periodic oscillations in F(r) are signatures of
    discrete scale invariance (DSI).

    Parameters
    ----------
    shells : ndarray
        Shell occupancy N(r) from BFS

    Returns
    -------
    r : ndarray
        Radii where F(r) is defined (1 to len(shells)-2)
    F : ndarray
        Force ratio F(r)
    """
    # Avoid log(0) by masking empty shells
    mask = shells > 0
    S = np.zeros_like(shells, dtype=np.float64)
    S[mask] = np.log(shells[mask])

    # F(r) = S(r+1) - S(r)
    F = np.diff(S)
    r = np.arange(1, len(F) + 1)

    # Only return valid radii (where both shells are non-empty)
    valid = (shells[:-1] > 0) & (shells[1:] > 0)

    return r[valid], F[valid]


# ==============================================================================
# Log-Periodic Oscillation Fitting
# ==============================================================================

def fit_log_periodic(r, F, lambda_scale):
    """
    Fit log-periodic oscillation model to force ratio.

    Model: F(r) = A * r^β * (1 + B * cos(2π * log(r) / log(λ) + φ))

    Returns residuals after removing the power-law trend.

    Parameters
    ----------
    r : ndarray
        Radii
    F : ndarray
        Force ratios
    lambda_scale : float
        Expected scaling factor (3 for Carpet, 2 for Gasket)

    Returns
    -------
    r : ndarray
        Radii (trimmed to positive F values)
    residuals : ndarray
        Detrended residuals
    period : float
        Log-periodic wavelength Δlog(r) = log(λ)
    """
    # Simple detrending: remove power-law fit
    mask = F != 0
    r_fit = r[mask]
    F_fit = F[mask]

    if len(r_fit) < 10:
        return r_fit, F_fit, np.log(lambda_scale)

    # Log-log linear fit: log(F) = β * log(r) + const
    log_r = np.log(r_fit)
    log_F = np.log(np.abs(F_fit))

    # Robust linear fit (least squares)
    p = np.polyfit(log_r, log_F, 1)
    beta = p[0]

    # Remove trend
    F_trend = np.exp(np.polyval(p, log_r))
    residuals = F_fit / F_trend - 1.0  # Fractional residuals

    period = np.log(lambda_scale)

    return r_fit, residuals, period


# ==============================================================================
# 3-Panel Comparative Plot
# ==============================================================================

def plot_dsi_comparison(results, output_file='fractal_dsi_comparison.png'):
    """
    Generate 3-panel plot comparing DSI signatures across fractals.

    Parameters
    ----------
    results : dict
        Dictionary with keys 'carpet', 'gasket', 'euclidean', each containing
        {'r': radii, 'F': force ratios, 'residuals': detrended, 'period': wavelength}
    output_file : str
        Output filename
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Raw Force Ratios
    ax = axes[0]
    for label, style in [('carpet', 'o-'), ('gasket', 's-'), ('euclidean', '^--')]:
        if label in results:
            r = results[label]['r']
            F = results[label]['F']
            ax.loglog(r, np.abs(F), style, label=label.capitalize(), alpha=0.7, markersize=4)

    ax.set_xlabel('Graph Distance r')
    ax.set_ylabel('|F(r)| = |S(r+1) - S(r)|')
    ax.set_title('A: Raw Entropy Gradient Force')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Detrended Residuals
    ax = axes[1]
    for label, style in [('carpet', 'o-'), ('gasket', 's-')]:
        if label in results:
            r = results[label]['r']
            res = results[label]['residuals']
            ax.semilogx(r, res, style, label=label.capitalize(), alpha=0.7, markersize=3)

    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Graph Distance r')
    ax.set_ylabel('Fractional Residual (F / F_trend - 1)')
    ax.set_title('B: Log-Periodic Oscillations (Detrended)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Periodogram / Crossing Analysis
    ax = axes[2]
    for label in ['carpet', 'gasket']:
        if label in results:
            r = results[label]['r']
            res = results[label]['residuals']
            period = results[label]['period']

            # Simple crossing count (residual changes sign)
            crossings = np.where(np.diff(np.sign(res)))[0]
            log_r = np.log(r)
            log_r_crossings = log_r[crossings]

            # Expected spacing
            expected_spacing = period

            if len(log_r_crossings) > 1:
                spacings = np.diff(log_r_crossings)
                ax.hist(spacings, bins=10, alpha=0.5, label=f'{label} (expect {expected_spacing:.2f})')

    ax.set_xlabel('Δlog(r) between crossings')
    ax.set_ylabel('Count')
    ax.set_title('C: Crossing Interval Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"[SAVED] {output_file}")


# ==============================================================================
# Main Comparative Analysis
# ==============================================================================

def run_comparative_analysis(carpet_level=4, gasket_level=6, r_max=100):
    """
    Run full comparative DSI analysis.

    Parameters
    ----------
    carpet_level : int
        Sierpiński Carpet subdivision level (L = 3^level)
    gasket_level : int
        Sierpiński Gasket subdivision level
    r_max : int
        Maximum BFS radius

    Returns
    -------
    results : dict
        Comparative results for all fractals
    """
    results = {}

    # 1. Sierpiński Carpet
    print("\n" + "="*60)
    print("Generating Sierpiński Carpet...")
    print("="*60)
    A_carpet, coords_carpet, L_carpet = sierpinski_carpet_sparse(carpet_level)
    center_carpet = A_carpet.shape[0] // 2  # Approximate center
    shells_carpet = bfs_shell_counts(A_carpet, center_carpet, r_max)
    r_carpet, F_carpet = entropy_gradient_force_ratio(shells_carpet)
    r_fit_carpet, res_carpet, period_carpet = fit_log_periodic(r_carpet, F_carpet, lambda_scale=3)

    results['carpet'] = {
        'r': r_fit_carpet,
        'F': F_carpet[:len(r_fit_carpet)],
        'residuals': res_carpet,
        'period': period_carpet,
        'shells': shells_carpet
    }

    # 2. Sierpiński Gasket
    print("\n" + "="*60)
    print("Generating Sierpiński Gasket...")
    print("="*60)
    A_gasket, coords_gasket = sierpinski_gasket_sparse(gasket_level)
    center_gasket = 0  # Start from corner node
    shells_gasket = bfs_shell_counts(A_gasket, center_gasket, r_max)
    r_gasket, F_gasket = entropy_gradient_force_ratio(shells_gasket)
    r_fit_gasket, res_gasket, period_gasket = fit_log_periodic(r_gasket, F_gasket, lambda_scale=2)

    results['gasket'] = {
        'r': r_fit_gasket,
        'F': F_gasket[:len(r_fit_gasket)],
        'residuals': res_gasket,
        'period': period_gasket,
        'shells': shells_gasket
    }

    # 3. Euclidean Grid Baseline
    print("\n" + "="*60)
    print("Generating Euclidean Grid Baseline...")
    print("="*60)
    L_euclid = L_carpet  # Match carpet size
    shells_euclid = euclidean_grid_shell_counts(L_euclid, r_max)
    r_euclid, F_euclid = entropy_gradient_force_ratio(shells_euclid)

    results['euclidean'] = {
        'r': r_euclid,
        'F': F_euclid,
        'residuals': np.zeros_like(F_euclid),  # No oscillations expected
        'period': 0.0,
        'shells': shells_euclid
    }

    # 4. Generate Comparative Plot
    print("\n" + "="*60)
    print("Generating 3-Panel DSI Comparison Plot...")
    print("="*60)
    plot_dsi_comparison(results)

    # 5. Summary Statistics
    print("\n" + "="*60)
    print("DSI SIGNATURE SUMMARY")
    print("="*60)

    for label in ['carpet', 'gasket']:
        res = results[label]['residuals']
        period = results[label]['period']

        # Count zero-crossings
        crossings = np.sum(np.diff(np.sign(res)) != 0)

        # Measure oscillation amplitude
        amplitude = np.std(res)

        print(f"\n{label.upper()}:")
        print(f"  Expected period: Δlog(r) = {period:.4f}")
        print(f"  Observed zero-crossings: {crossings}")
        print(f"  Oscillation amplitude (std): {amplitude:.4f}")

    return results


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fractal DSI Analysis")
    parser.add_argument('--carpet-level', type=int, default=4,
                        help='Sierpiński Carpet level (L=3^level)')
    parser.add_argument('--gasket-level', type=int, default=6,
                        help='Sierpiński Gasket level')
    parser.add_argument('--r-max', type=int, default=100,
                        help='Maximum BFS radius')
    parser.add_argument('--output', default='fractal_dsi_comparison.png',
                        help='Output plot filename')

    args = parser.parse_args()

    results = run_comparative_analysis(
        carpet_level=args.carpet_level,
        gasket_level=args.gasket_level,
        r_max=args.r_max
    )

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output: fractal_dsi_comparison.png")
    print("\nNext steps:")
    print("1. Check Panel B for log-periodic oscillations")
    print("2. Verify Panel C crossing intervals match expected periods:")
    print(f"   - Carpet: Δlog(r) ≈ {np.log(3):.3f}")
    print(f"   - Gasket: Δlog(r) ≈ {np.log(2):.3f}")
    print("3. If confirmed, this is the DSI signature for PRE/JSTAT submission")
