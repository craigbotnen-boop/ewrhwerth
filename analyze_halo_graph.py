"""
Halo Graph Analysis Pipeline

Generates a pilot-scale synthetic dark-matter halo catalog with cosmic-web
topology (filaments + nodes + voids), builds a proximity graph, saves it as
an edgelist, then runs:
  1. Spectral dimension (d_s) analysis
  2. Correlation dimension (D_2) analysis
  3. Forensic validation suite (whitening, radial null, phase surrogate)

Designed to run on a machine with >= 4 GB RAM and 4 cores.
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

# Local modules
from diffusion_spectral_dimension import compute_spectral_dimension, plot_spectral_dimension
from correlation_dimension import compute_D2_full, plot_D2_diagnostic
from forensic_validation_suite import run_forensic_validation, plot_forensic_validation


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

    Returns xyz positions in comoving Mpc/h.
    """
    rng = np.random.default_rng(seed)

    halos = []

    # -- Cluster nodes (massive halos) --
    n_node_halos = int(n_halos * 0.15)
    node_centers = rng.uniform(box_size * 0.15, box_size * 0.85, size=(n_nodes, 3))

    for i in range(n_node_halos):
        center = node_centers[rng.integers(n_nodes)]
        pos = center + rng.normal(0, node_radius, size=3)
        halos.append(pos)

    # -- Filaments (connecting random pairs of nodes) --
    n_filament_halos = int(n_halos * 0.50)
    filament_pairs = []
    for _ in range(n_filaments):
        i, j = rng.choice(n_nodes, size=2, replace=False)
        filament_pairs.append((node_centers[i], node_centers[j]))

    per_filament = n_filament_halos // n_filaments
    for start, end in filament_pairs:
        direction = end - start
        length = np.linalg.norm(direction)
        direction /= length

        # Perpendicular directions
        perp1 = np.cross(direction, rng.standard_normal(3))
        perp1 /= np.linalg.norm(perp1) + 1e-10
        perp2 = np.cross(direction, perp1)
        perp2 /= np.linalg.norm(perp2) + 1e-10

        for _ in range(per_filament):
            t = rng.uniform(0, 1)
            pos = start + t * (end - start)
            # Add transverse scatter (filament thickness)
            pos += rng.normal(0, filament_thickness) * perp1
            pos += rng.normal(0, filament_thickness) * perp2
            halos.append(pos)

    # -- Field / sheet halos (sparse, filling voids) --
    n_field = n_halos - len(halos)
    field_halos = rng.uniform(0, box_size, size=(n_field, 3))
    halos.extend(field_halos.tolist())

    xyz = np.array(halos[:n_halos])

    # Wrap into box
    xyz = xyz % box_size

    return xyz, node_centers


def build_halo_graph(xyz, linking_length=None, k_nn=12):
    """
    Build a proximity graph from halo positions.

    Uses a hybrid approach:
    - k-NN edges for local connectivity
    - Friends-of-friends (FoF) linking for halos within linking_length

    Returns a NetworkX graph.
    """
    N = xyz.shape[0]
    tree = cKDTree(xyz)

    # k-NN graph
    distances, indices = tree.query(xyz, k=min(k_nn + 1, N))

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Add k-NN edges
    for i in range(N):
        for j_idx in range(1, min(k_nn + 1, indices.shape[1])):
            j = indices[i, j_idx]
            if j < N:
                G.add_edge(i, j)

    # Optionally add FoF links
    if linking_length is not None:
        pairs = tree.query_pairs(linking_length)
        G.add_edges_from(pairs)

    return G


def save_edgelist(G, filepath):
    """Save graph as whitespace-delimited edgelist."""
    nx.write_edgelist(G, filepath, data=False)
    print(f"Saved edgelist: {filepath} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def load_edgelist(filepath):
    """Load edgelist and return graph."""
    G = nx.read_edgelist(filepath, nodetype=int)
    print(f"Loaded: {filepath} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


# ============================================================================
# 2. Analysis pipeline
# ============================================================================

def run_full_analysis(xyz, output_prefix="halo_graph"):
    """Run spectral dimension, correlation dimension, and forensic validation."""

    results = {}

    # --- Spectral Dimension ---
    print("\n" + "=" * 70)
    print("SPECTRAL DIMENSION ANALYSIS")
    print("=" * 70)
    t0 = time.time()
    sd_result = compute_spectral_dimension(
        xyz, k=15, n_eigs=200, n_times=60,
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
    d2_result = compute_D2_full(
        xyz, use_guard_region=True,
        bootstrap=True, B=50, verbose=True
    )
    dt = time.time() - t0
    print(f"Time: {dt:.1f}s")
    results['correlation_dimension'] = d2_result

    # --- Forensic Validation (reduced draws for pilot) ---
    print("\n" + "=" * 70)
    print("FORENSIC VALIDATION SUITE")
    print("=" * 70)
    t0 = time.time()
    fv_result = run_forensic_validation(
        xyz, n_null_draws=30, k=15, verbose=True
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
    print("=" * 70)
    print("HALO GRAPH PILOT ANALYSIS")
    print("=" * 70)

    EDGELIST_PATH = 'halo_graph.edgelist'
    N_HALOS = 2000  # Pilot size — fits in < 1 GB RAM

    # Step 1: Generate halo catalog
    print("\n[Step 1] Generating synthetic halo catalog...")
    t0 = time.time()
    xyz, node_centers = generate_cosmic_web_halos(n_halos=N_HALOS, seed=42)
    print(f"  Generated {len(xyz)} halos in {time.time()-t0:.1f}s")
    print(f"  Box: [{xyz.min(axis=0).round(1)} ... {xyz.max(axis=0).round(1)}] Mpc/h")

    # Step 2: Build proximity graph
    print("\n[Step 2] Building halo graph (k-NN + FoF)...")
    t0 = time.time()

    # Estimate mean inter-halo separation for linking length
    tree = cKDTree(xyz)
    nn_dist, _ = tree.query(xyz, k=2)
    mean_nn = nn_dist[:, 1].mean()
    linking_length = mean_nn * 1.5  # ~1.5x mean nearest-neighbor
    print(f"  Mean NN distance: {mean_nn:.2f} Mpc/h")
    print(f"  FoF linking length: {linking_length:.2f} Mpc/h")

    G = build_halo_graph(xyz, linking_length=linking_length, k_nn=12)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Step 3: Save edgelist
    print(f"\n[Step 3] Saving edgelist...")
    save_edgelist(G, EDGELIST_PATH)

    # Quick stats
    print(f"\n  --- Edgelist summary ---")
    print(f"  File: {os.path.abspath(EDGELIST_PATH)}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.1f}")

    # Step 4: Run full analysis
    print("\n[Step 4] Running analysis pipeline...")
    results = run_full_analysis(xyz, output_prefix="halo_graph")

    # Step 5: Summary figure
    print("\n[Step 5] Creating summary figure...")
    create_summary_figure(xyz, results, output_prefix="halo_graph")

    # Print final summary
    sd = results['spectral_dimension']
    d2 = results['correlation_dimension']
    fv = results['forensic_validation']

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Halo count:             {N_HALOS}")
    print(f"  Graph edges:            {G.number_of_edges()}")
    print(f"  Spectral dimension:     d_s = {sd.d_s:.3f} +/- {sd.d_s_std:.3f}")
    print(f"  Correlation dimension:  D_2 = {d2.D2_estimate:.3f} +/- {d2.D2_std:.3f}")
    print(f"  Forensic verdict:       {fv.verdict} ({fv.confidence})")
    print(f"  Whitening gate:         {'PASS' if fv.whitening_passed else 'FAIL'}")
    print(f"  Radial null (z):        {fv.z_score_radial:.2f} ({'PASS' if fv.radial_null_passed else 'FAIL'})")
    print(f"  Phase null (z):         {fv.z_score_phase:.2f} ({'PASS' if fv.phase_null_passed else 'FAIL'})")
    print("=" * 70)
