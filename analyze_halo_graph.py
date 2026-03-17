"""
Halo Graph Analysis Pipeline

Generates a synthetic dark-matter halo catalog with cosmic-web topology
(filaments + nodes + voids), builds a proximity graph, saves it as an
edgelist, then runs:
  1. Spectral dimension (d_s) analysis
  2. Correlation dimension (D_2) analysis
  3. Forensic validation suite (whitening, radial null, phase surrogate)

Supports pilot (2k) through stress-test (100k+) scales on a single
workstation with >= 8 GB RAM.

Usage:
  python analyze_halo_graph.py                    # 2k pilot (default)
  python analyze_halo_graph.py --n_halos 100000   # 100k stress test
"""

import argparse
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

# Local modules
from diffusion_spectral_dimension import (
    compute_spectral_dimension,
    compute_spectral_dimension_hutchinson,
    plot_spectral_dimension,
)
from correlation_dimension import compute_D2_full, plot_D2_diagnostic
from forensic_validation_suite import run_forensic_validation, plot_forensic_validation


def auto_keig(n_nodes):
    """Auto-select number of eigenpairs: min(512, max(128, int(sqrt(N))))."""
    return min(512, max(128, int(np.sqrt(n_nodes))))


def graph_sanity_check(G, label="Graph"):
    """Print degree stats and GCC fraction before launching heavy compute."""
    N = G.number_of_nodes()
    M = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    gcc_size = max(len(c) for c in nx.connected_components(G))
    gcc_frac = gcc_size / N

    print(f"  --- {label} sanity check ---")
    print(f"  Nodes: {N:,}  Edges: {M:,}")
    print(f"  Degree — mean: {np.mean(degrees):.1f}  "
          f"median: {np.median(degrees):.0f}  "
          f"min: {min(degrees)}  max: {max(degrees)}")
    print(f"  GCC: {gcc_size:,} / {N:,} ({gcc_frac:.1%})")

    if gcc_frac < 0.9:
        print(f"  WARNING: GCC covers only {gcc_frac:.1%} of nodes — "
              "consider increasing k_nn or linking_length")
    if np.mean(degrees) < 4:
        print(f"  WARNING: Mean degree {np.mean(degrees):.1f} is low — "
              "graph may be too sparse for reliable d_s")

    return {'mean_deg': np.mean(degrees), 'gcc_frac': gcc_frac}


# ============================================================================
# 1. Generate synthetic halo catalog with cosmic-web structure
# ============================================================================

def generate_cosmic_web_halos(
    n_halos=2000,
    box_size=250.0,       # Mpc/h
    n_filaments=40,
    n_nodes=8,
    filament_thickness=5.0,
    node_radius=12.0,
    void_fraction=0.3,
    seed=42
):
    """
    Generate a synthetic halo catalog with cosmic-web-like structure.

    Halos are distributed along:
    - Filaments: 1D structures connecting cluster nodes
    - Nodes: dense clusters at filament intersections
    - Field: sparse halos in low-density regions (sheet/void)

    Vectorized for efficiency at N >= 100k.
    Returns xyz positions in comoving Mpc/h.
    """
    rng = np.random.default_rng(seed)

    # Scale filament count with N for larger catalogs
    if n_halos > 10000:
        n_filaments = max(n_filaments, int(np.sqrt(n_halos) / 3))
        n_nodes = max(n_nodes, int(np.sqrt(n_halos) / 15))

    # -- Cluster nodes (massive halos) -- vectorized
    n_node_halos = int(n_halos * 0.15)
    node_centers = rng.uniform(box_size * 0.15, box_size * 0.85, size=(n_nodes, 3))
    center_idx = rng.integers(0, n_nodes, size=n_node_halos)
    node_halos = node_centers[center_idx] + rng.normal(0, node_radius, size=(n_node_halos, 3))

    # -- Filaments (connecting random pairs of nodes) -- vectorized
    n_filament_halos = int(n_halos * 0.50)
    filament_pairs = []
    for _ in range(n_filaments):
        i, j = rng.choice(n_nodes, size=2, replace=False)
        filament_pairs.append((node_centers[i], node_centers[j]))

    per_filament = n_filament_halos // n_filaments
    filament_halos = np.empty((n_filaments * per_filament, 3))

    for f_idx, (start, end) in enumerate(filament_pairs):
        direction = end - start
        length = np.linalg.norm(direction)
        direction /= length

        perp1 = np.cross(direction, rng.standard_normal(3))
        perp1 /= np.linalg.norm(perp1) + 1e-10
        perp2 = np.cross(direction, perp1)
        perp2 /= np.linalg.norm(perp2) + 1e-10

        t_vals = rng.uniform(0, 1, size=per_filament)
        base = start[None, :] + t_vals[:, None] * (end - start)[None, :]
        scatter = (rng.normal(0, filament_thickness, size=(per_filament, 1)) * perp1[None, :] +
                   rng.normal(0, filament_thickness, size=(per_filament, 1)) * perp2[None, :])

        sl = slice(f_idx * per_filament, (f_idx + 1) * per_filament)
        filament_halos[sl] = base + scatter

    # -- Field / sheet halos (sparse, filling voids) --
    n_placed = n_node_halos + len(filament_halos)
    n_field = n_halos - n_placed
    field_halos = rng.uniform(0, box_size, size=(max(n_field, 0), 3))

    # Concatenate and trim
    xyz = np.vstack([node_halos, filament_halos, field_halos])[:n_halos]

    # Wrap into box
    xyz = xyz % box_size

    return xyz, node_centers


def build_halo_graph(xyz, linking_length=None, k_nn=12):
    """
    Build a proximity graph from halo positions.

    Uses a hybrid approach:
    - k-NN edges for local connectivity
    - Friends-of-friends (FoF) linking for halos within linking_length

    Vectorized edge construction for N >= 100k.
    Returns a NetworkX graph.
    """
    N = xyz.shape[0]
    tree = cKDTree(xyz)

    # k-NN graph — vectorized edge list
    k_query = min(k_nn + 1, N)
    distances, indices = tree.query(xyz, k=k_query)

    # Build edge array: (i, j) for all k-NN pairs, skip self (column 0)
    row = np.repeat(np.arange(N), k_query - 1)
    col = indices[:, 1:].ravel()
    valid = col < N
    edges_knn = set(zip(row[valid].tolist(), col[valid].tolist()))

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges_knn)

    # Optionally add FoF links
    if linking_length is not None:
        pairs = tree.query_pairs(linking_length)
        G.add_edges_from(pairs)

    return G


def save_edgelist(G, filepath):
    """Save graph as whitespace-delimited edgelist with metadata header."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    with open(filepath, 'w') as f:
        f.write(f"# n={n} m={m}\n")
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    print(f"Saved edgelist: {filepath} ({n} nodes, {m} edges)")


def load_edgelist(filepath):
    """Load edgelist and return graph."""
    G = nx.read_edgelist(filepath, nodetype=int)
    print(f"Loaded: {filepath} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


# ============================================================================
# 2. Analysis pipeline
# ============================================================================

def run_full_analysis(xyz, output_prefix="halo_graph", keig=None, n_null=None):
    """Run spectral dimension, correlation dimension, and forensic validation."""

    N = len(xyz)
    results = {}

    # Auto-select keig
    if keig is None:
        keig = auto_keig(N)
    print(f"\nAnalysis config: N={N:,}, keig={keig}")

    # --- Spectral Dimension ---
    # Solver selection:
    #   N <= 50k  → Lanczos eigsh (fast, exact low-mode spectrum)
    #   N > 50k   → Hutchinson trace estimator (O(1) memory per probe,
    #               avoids the sparse LU factorization that OOMs at 250k)
    # The Hutchinson path uses 50 Rademacher probes × 60 time points
    # via expm_multiply (Krylov mat-vec), peak RSS stays under 1 GB.
    use_hutchinson = N > 50000
    solver_name = "Hutchinson" if use_hutchinson else "Lanczos eigsh"
    print(f"\n  Solver: {solver_name}")

    print("\n" + "=" * 70)
    print("SPECTRAL DIMENSION ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    if use_hutchinson:
        sd_result = compute_spectral_dimension_hutchinson(
            xyz, k=15, n_vectors=50, n_times=60,
            t_min=0.01, t_max=200.0, verbose=True
        )
    else:
        sd_result = compute_spectral_dimension(
            xyz, k=15, n_eigs=keig, n_times=60,
            t_min=0.01, t_max=200.0, verbose=True
        )

    dt = time.time() - t0
    print(f"Time: {dt:.1f}s")
    results['spectral_dimension'] = sd_result

    # --- Correlation Dimension ---
    print("\n" + "=" * 70)
    print("CORRELATION DIMENSION ANALYSIS")
    print("=" * 70)
    t0 = time.time()

    # Scale bootstrap iterations with N (fewer for large N since each is slower)
    B = 50 if N <= 10000 else 20
    d2_result = compute_D2_full(
        xyz, use_guard_region=True,
        bootstrap=True, B=B, verbose=True
    )
    dt = time.time() - t0
    print(f"Time: {dt:.1f}s")
    results['correlation_dimension'] = d2_result

    # --- Forensic Validation ---
    print("\n" + "=" * 70)
    print("FORENSIC VALIDATION SUITE")
    print("=" * 70)
    t0 = time.time()

    # Scale null draws: 30 for pilot, 10 for large (each draw is a full d_s compute)
    if n_null is None:
        n_null = 30 if N <= 5000 else 10
    fv_result = run_forensic_validation(
        xyz, n_null_draws=n_null, k=15, verbose=True,
        use_hutchinson=use_hutchinson,
    )
    dt = time.time() - t0
    print(f"Time: {dt:.1f}s")
    results['forensic_validation'] = fv_result

    return results


def create_summary_figure(xyz, results, output_prefix="halo_graph"):
    """Create a multi-panel summary figure."""

    fig = plt.figure(figsize=(16, 12))

    # Panel 1: 3D scatter of halo positions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    subsample = np.random.choice(len(xyz), min(1000, len(xyz)), replace=False)
    ax1.scatter(xyz[subsample, 0], xyz[subsample, 1], xyz[subsample, 2],
                s=1, alpha=0.3, c='navy')
    ax1.set_title(f'Halo Positions (N={len(xyz)})')
    ax1.set_xlabel('x [Mpc/h]')
    ax1.set_ylabel('y [Mpc/h]')
    ax1.set_zlabel('z [Mpc/h]')

    # Panel 2: Heat kernel trace
    sd = results['spectral_dimension']
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.loglog(sd.t_values, sd.P_t, 'b.-', markersize=3)
    if np.isfinite(sd.d_s):
        t_ref = sd.t_values
        mid = len(t_ref) // 2
        P_ref = sd.P_t[mid] * (t_ref / t_ref[mid]) ** (-sd.d_s / 2)
        ax2.loglog(t_ref, P_ref, 'r--', alpha=0.7,
                   label=f'$t^{{-d_s/2}}$, $d_s={sd.d_s:.2f}$')
    ax2.set_xlabel('Diffusion time $t$')
    ax2.set_ylabel('$P(t)$')
    ax2.set_title('Heat Kernel Trace')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Local spectral dimension
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.semilogx(sd.t_values, sd.local_d_s, 'b.-', markersize=3)
    if sd.plateau_mask.any():
        ax3.semilogx(sd.t_values[sd.plateau_mask],
                     sd.local_d_s[sd.plateau_mask], 'go', alpha=0.5, markersize=6)
    if np.isfinite(sd.d_s):
        ax3.axhline(sd.d_s, color='r', linestyle='--',
                    label=f'$d_s = {sd.d_s:.2f} \\pm {sd.d_s_std:.2f}$')
        ax3.axhspan(sd.d_s - sd.d_s_std, sd.d_s + sd.d_s_std, color='r', alpha=0.1)
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$d_s(t)$')
    ax3.set_title('Spectral Dimension')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 5)

    # Panel 4: Correlation integral
    d2 = results['correlation_dimension']
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.loglog(d2.r_centers, d2.C, 'b.-', markersize=3)
    if d2.plateau_mask.any():
        ax4.loglog(d2.r_centers[d2.plateau_mask], d2.C[d2.plateau_mask],
                   'go', markersize=6, alpha=0.5)
    ax4.set_xlabel('$r$ [Mpc/h]')
    ax4.set_ylabel('$C(r)$')
    ax4.set_title('Correlation Integral')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Local D2
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.semilogx(d2.r_centers, d2.D2, 'b.-', markersize=3)
    if d2.plateau_mask.any():
        ax5.semilogx(d2.r_centers[d2.plateau_mask], d2.D2[d2.plateau_mask],
                     'go', markersize=6, alpha=0.5)
    if np.isfinite(d2.D2_estimate):
        ax5.axhline(d2.D2_estimate, color='r', linestyle='--',
                    label=f'$D_2 = {d2.D2_estimate:.2f} \\pm {d2.D2_std:.2f}$')
        ax5.axhspan(d2.D2_estimate - d2.D2_std, d2.D2_estimate + d2.D2_std,
                    color='r', alpha=0.1)
    ax5.set_xlabel('$r$ [Mpc/h]')
    ax5.set_ylabel('$D_2(r)$')
    ax5.set_title('Correlation Dimension')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 5)

    # Panel 6: Forensic validation summary
    fv = results['forensic_validation']
    ax6 = fig.add_subplot(2, 3, 6)
    labels = ['Original', 'Whitened', 'Radial\nNull', 'Phase\nNull']
    values = [fv.d_s_original, fv.d_s_whitened,
              fv.d_s_radial_null_mean, fv.d_s_phase_null_mean]
    errors = [fv.d_s_original_std, fv.d_s_whitened_std,
              fv.d_s_radial_null_std, fv.d_s_phase_null_std]
    colors = ['steelblue',
              'green' if fv.whitening_passed else 'red',
              'green' if fv.radial_null_passed else 'orange',
              'green' if fv.phase_null_passed else 'orange']
    ax6.bar(labels, values, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax6.set_ylabel('$d_s$')
    ax6.set_title(f'Forensic: {fv.verdict}\n(Confidence: {fv.confidence})')

    plt.suptitle('Halo Graph Pilot Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    pdf_path = f'{output_prefix}_analysis.pdf'
    png_path = f'{output_prefix}_analysis.png'
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {pdf_path}, {png_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Halo Graph Analysis Pipeline')
    parser.add_argument('--n_halos', type=int, default=2000,
                        help='Number of halos to generate (default: 2000)')
    parser.add_argument('--keig', type=int, default=None,
                        help='Number of eigenpairs (default: auto)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_prefix', type=str, default=None,
                        help='Output file prefix (default: halo_graph_N)')
    args = parser.parse_args()

    N_HALOS = args.n_halos
    if args.output_prefix is None:
        prefix = f"halo_graph_{N_HALOS // 1000}k" if N_HALOS >= 1000 else f"halo_graph_{N_HALOS}"
    else:
        prefix = args.output_prefix
    EDGELIST_PATH = f'{prefix}.edgelist'

    keig = args.keig if args.keig else auto_keig(N_HALOS)

    print("=" * 70)
    print(f"HALO GRAPH ANALYSIS — N={N_HALOS:,} halos, keig={keig}")
    print("=" * 70)

    total_t0 = time.time()

    # Step 1: Generate halo catalog
    print("\n[Step 1] Generating synthetic halo catalog...")
    t0 = time.time()
    xyz, node_centers = generate_cosmic_web_halos(n_halos=N_HALOS, seed=args.seed)
    print(f"  Generated {len(xyz):,} halos in {time.time()-t0:.1f}s")
    print(f"  Box: [{xyz.min(axis=0).round(1)} ... {xyz.max(axis=0).round(1)}] Mpc/h")

    # Step 2: Build proximity graph
    print("\n[Step 2] Building halo graph (k-NN + FoF)...")
    t0 = time.time()

    tree = cKDTree(xyz)
    nn_dist, _ = tree.query(xyz, k=2)
    mean_nn = nn_dist[:, 1].mean()
    linking_length = mean_nn * 1.5
    print(f"  Mean NN distance: {mean_nn:.2f} Mpc/h")
    print(f"  FoF linking length: {linking_length:.2f} Mpc/h")

    G = build_halo_graph(xyz, linking_length=linking_length, k_nn=12)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Degree sanity check
    graph_sanity_check(G, label="Halo Graph")

    # Step 3: Save edgelist
    print(f"\n[Step 3] Saving edgelist...")
    save_edgelist(G, EDGELIST_PATH)
    print(f"  File: {os.path.abspath(EDGELIST_PATH)}")

    # Step 4: Run full analysis
    print("\n[Step 4] Running analysis pipeline...")
    results = run_full_analysis(xyz, output_prefix=prefix, keig=keig)

    # Step 5: Summary figure
    print("\n[Step 5] Creating summary figure...")
    create_summary_figure(xyz, results, output_prefix=prefix)

    # Print final summary
    sd = results['spectral_dimension']
    d2 = results['correlation_dimension']
    fv = results['forensic_validation']
    total_dt = time.time() - total_t0

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Halo count:             {N_HALOS:,}")
    print(f"  Graph edges:            {G.number_of_edges():,}")
    print(f"  keig (auto):            {keig}")
    print(f"  Spectral dimension:     d_s = {sd.d_s:.3f} +/- {sd.d_s_std:.3f}")
    print(f"  Correlation dimension:  D_2 = {d2.D2_estimate:.3f} +/- {d2.D2_std:.3f}")
    print(f"  Forensic verdict:       {fv.verdict} ({fv.confidence})")
    print(f"  Whitening gate:         {'PASS' if fv.whitening_passed else 'FAIL'}")
    print(f"  Radial null (z):        {fv.z_score_radial:.2f} ({'PASS' if fv.radial_null_passed else 'FAIL'})")
    print(f"  Phase null (z):         {fv.z_score_phase:.2f} ({'PASS' if fv.phase_null_passed else 'FAIL'})")
    print(f"  Total wall time:        {total_dt:.0f}s ({total_dt/60:.1f} min)")
    print("=" * 70)
