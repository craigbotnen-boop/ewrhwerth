"""
Correlation Dimension (D_2) Analysis Module

Implements fast, numerically stable computation of the correlation dimension
using KDTree-based pair counting and guard-region edge correction.

Key design principles:
1. Use scipy.spatial.cKDTree.count_neighbors for O(N log N) cumulative counts
2. Guard-region trimming for edge correction (no expensive node-random pairs)
3. Bootstrap resamples NODES only (geometry correction is deterministic)
4. Plateau selection rule prevents log-derivative blow-up from near-zero counts

Reference: Grassberger & Procaccia (1983), Physica D 9, 189-208
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional, NamedTuple
import warnings


class D2Result(NamedTuple):
    """Container for correlation dimension results."""
    r_centers: np.ndarray      # Geometric mean of bin edges (length M-1)
    C: np.ndarray              # Correlation integral at r_centers
    D2: np.ndarray             # Local correlation dimension estimate
    plateau_mask: np.ndarray   # Boolean mask for valid plateau region
    D2_estimate: float         # Robust D2 estimate from plateau
    D2_std: float              # Uncertainty from bootstrap (if computed)


def corr_integral_D2(
    xyz_nodes: np.ndarray,
    r_edges: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute correlation integral and local D2 using KDTree pair counting.

    This is O(N log N) per radius edge, suitable for N ~ 200-10000 nodes.

    Parameters
    ----------
    xyz_nodes : ndarray of shape (N, 3)
        Node positions in 3D space.
    r_edges : ndarray of shape (M,)
        Monotonically increasing radii defining bin edges.
    weights : ndarray of shape (N,), optional
        Node weights (not currently used, placeholder for future).

    Returns
    -------
    r_centers : ndarray of shape (M-1,)
        Geometric mean of consecutive bin edges.
    C : ndarray of shape (M-1,)
        Normalized correlation integral C(r) = N_pairs(<r) / N_total_pairs.
    D2 : ndarray of shape (M-1,)
        Local correlation dimension d(log C)/d(log r).

    Notes
    -----
    Uses cumulative pair counts at edges, then evaluates C at centers.
    The correlation integral is normalized by N*(N-1)/2 (total possible pairs).
    """
    N = xyz_nodes.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 nodes, got {N}")

    tree = cKDTree(xyz_nodes)

    # Cumulative pair counts at each edge
    # count_neighbors returns total pairs (i,j) with i != j, so divide by 2
    cum = np.array([
        tree.count_neighbors(tree, r, cumulative=True)
        for r in r_edges
    ], dtype=np.float64)

    # Subtract self-pairs (N of them) and divide by 2 for i<j convention
    cum = (cum - N) / 2.0

    # Geometric mean for bin centers (more appropriate for log-space)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])

    # Correlation integral: use counts at upper edge of each bin
    # C(r_center[i]) approximated by cum[i+1] / total_pairs
    total_pairs = N * (N - 1) / 2.0
    C = cum[1:] / total_pairs

    # Numerical safety: clip to avoid log(0)
    eps = 1e-15
    C_safe = np.maximum(C, eps)

    # Local dimension via numerical gradient
    # D2 = d(log C) / d(log r)
    log_r = np.log(r_centers)
    log_C = np.log(C_safe)
    D2 = np.gradient(log_C, log_r)

    return r_centers, C, D2


def apply_guard_region(
    xyz_nodes: np.ndarray,
    r_guard: float,
    box_min: Optional[np.ndarray] = None,
    box_max: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove nodes within r_guard of the analysis box boundary.

    This is the simplest and most defensible edge correction for D2:
    it avoids the need for expensive node-random pair counting entirely.

    Parameters
    ----------
    xyz_nodes : ndarray of shape (N, 3)
        Node positions.
    r_guard : float
        Guard distance from boundary.
    box_min : ndarray of shape (3,), optional
        Lower corner of analysis box. Default: min of node coords.
    box_max : ndarray of shape (3,), optional
        Upper corner of analysis box. Default: max of node coords.

    Returns
    -------
    xyz_interior : ndarray of shape (N_interior, 3)
        Nodes in the interior region.
    interior_mask : ndarray of shape (N,)
        Boolean mask indicating interior nodes.

    Notes
    -----
    Reports should explicitly state the guard width used.
    A typical choice is r_guard = r_max (maximum analysis radius).
    """
    if box_min is None:
        box_min = xyz_nodes.min(axis=0)
    if box_max is None:
        box_max = xyz_nodes.max(axis=0)

    # Node is interior if it's > r_guard from all boundaries
    dist_to_lower = xyz_nodes - box_min
    dist_to_upper = box_max - xyz_nodes

    interior_mask = (
        (dist_to_lower > r_guard).all(axis=1) &
        (dist_to_upper > r_guard).all(axis=1)
    )

    return xyz_nodes[interior_mask], interior_mask


def identify_plateau(
    C: np.ndarray,
    D2: np.ndarray,
    C_min: float = 1e-4,
    C_max: float = 0.5,
    D2_max: float = 10.0,
    min_points: int = 5
) -> np.ndarray:
    """
    Identify the scaling plateau where D2 is reliable.

    The plateau is where:
    - C(r) is neither too small (noise) nor saturated
    - D2 is finite and not absurdly large
    - There are enough points for a robust estimate

    Parameters
    ----------
    C : ndarray
        Correlation integral values.
    D2 : ndarray
        Local dimension estimates.
    C_min : float
        Minimum C(r) to avoid noise-dominated regime.
    C_max : float
        Maximum C(r) to avoid saturation.
    D2_max : float
        Maximum plausible D2 (anything larger is likely numerical garbage).
    min_points : int
        Minimum number of valid points required.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask for plateau region.
    """
    mask = (
        (C > C_min) &
        (C < C_max) &
        np.isfinite(D2) &
        (D2 > 0) &
        (D2 < D2_max)
    )

    if mask.sum() < min_points:
        warnings.warn(
            f"Plateau has only {mask.sum()} points (need {min_points}). "
            "D2 estimate may be unreliable."
        )

    return mask


def robust_D2_estimate(D2: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute robust D2 from plateau region.

    Uses median and MAD-based scale estimate for robustness.

    Parameters
    ----------
    D2 : ndarray
        Local dimension estimates.
    mask : ndarray of bool
        Plateau mask.

    Returns
    -------
    D2_med : float
        Median D2 in plateau.
    D2_mad : float
        Median absolute deviation (scaled to match std for normal).
    """
    if mask.sum() == 0:
        return np.nan, np.nan

    D2_plateau = D2[mask]
    D2_med = np.median(D2_plateau)

    # MAD scaled to match std for normal distribution
    mad = np.median(np.abs(D2_plateau - D2_med))
    D2_mad = 1.4826 * mad  # scale factor for normal distribution

    return D2_med, D2_mad


def bootstrap_D2(
    xyz_nodes: np.ndarray,
    r_edges: np.ndarray,
    B: int = 50,
    seed: int = 0,
    C_min: float = 1e-4,
    C_max: float = 0.5,
    D2_max: float = 10.0,
    min_plateau_points: int = 5
) -> Tuple[float, float, np.ndarray]:
    """
    Bootstrap uncertainty estimate for D2.

    CRITICAL: Only resamples NODES, not random points.
    The geometry correction (if using guard region) is deterministic
    for a fixed box and should not be recomputed per bootstrap.

    Parameters
    ----------
    xyz_nodes : ndarray of shape (N, 3)
        Node positions (already edge-corrected if applicable).
    r_edges : ndarray of shape (M,)
        Radii bin edges.
    B : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    C_min, C_max : float
        Plateau bounds for C(r).
    D2_max : float
        Maximum plausible D2.
    min_plateau_points : int
        Minimum points required in plateau.

    Returns
    -------
    D2_median : float
        Median of bootstrap D2 estimates.
    D2_std : float
        Standard deviation of bootstrap estimates.
    D2_samples : ndarray of shape (B,)
        Individual bootstrap estimates (may contain NaN).

    Notes
    -----
    This is O(B * N * M * log N), which for N=228, M=50, B=50 is very fast.
    Compare to the broken version that recomputes N_node x N_random distances
    inside the loop: O(B * N_node * N_random) with N_random ~ 100k.
    """
    rng = np.random.default_rng(seed)
    N = xyz_nodes.shape[0]

    D2_samples = np.empty(B, dtype=np.float64)

    for i in range(B):
        # Resample nodes with replacement
        idx = rng.integers(0, N, size=N)
        xyz_boot = xyz_nodes[idx]

        # Compute correlation integral for bootstrap sample
        r_centers, C, D2 = corr_integral_D2(xyz_boot, r_edges)

        # Find plateau and extract robust estimate
        mask = identify_plateau(C, D2, C_min, C_max, D2_max, min_plateau_points)

        if mask.sum() < min_plateau_points:
            D2_samples[i] = np.nan
        else:
            D2_samples[i] = np.nanmedian(D2[mask])

    # Compute summary statistics, ignoring NaN
    D2_median = np.nanmedian(D2_samples)
    D2_std = np.nanstd(D2_samples, ddof=1)

    return D2_median, D2_std, D2_samples


def compute_D2_full(
    xyz_nodes: np.ndarray,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    n_bins: int = 50,
    use_guard_region: bool = True,
    bootstrap: bool = True,
    B: int = 50,
    seed: int = 42,
    verbose: bool = True
) -> D2Result:
    """
    Complete D2 analysis pipeline with proper edge correction and uncertainty.

    Parameters
    ----------
    xyz_nodes : ndarray of shape (N, 3)
        Node positions in 3D.
    r_min : float, optional
        Minimum radius. Default: mean nearest-neighbor distance / 2.
    r_max : float, optional
        Maximum radius. Default: 1/4 of box diagonal.
    n_bins : int
        Number of radial bins.
    use_guard_region : bool
        If True, apply guard-region edge correction.
    bootstrap : bool
        If True, compute bootstrap uncertainty.
    B : int
        Number of bootstrap iterations.
    seed : int
        Random seed.
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    D2Result
        Named tuple with all results.
    """
    N_original = xyz_nodes.shape[0]

    # Auto-determine radial range if not specified
    if r_min is None or r_max is None:
        tree = cKDTree(xyz_nodes)
        nn_dist, _ = tree.query(xyz_nodes, k=2)
        mean_nn = nn_dist[:, 1].mean()

        box_size = xyz_nodes.max(axis=0) - xyz_nodes.min(axis=0)
        box_diag = np.linalg.norm(box_size)

        if r_min is None:
            r_min = mean_nn / 2
        if r_max is None:
            # Use min dimension / 4 for better guard-region compatibility
            # (diagonal / 4 often exceeds box size in one dimension)
            r_max = min(box_size) / 4

    if verbose:
        print(f"D2 Analysis: N={N_original} nodes")
        print(f"Radial range: [{r_min:.4f}, {r_max:.4f}]")

    # Apply guard-region edge correction
    if use_guard_region:
        xyz_interior, interior_mask = apply_guard_region(xyz_nodes, r_max)
        N_interior = xyz_interior.shape[0]

        if verbose:
            print(f"Guard region (r={r_max:.4f}): {N_interior}/{N_original} nodes retained")

        if N_interior < 10:
            warnings.warn(
                f"Only {N_interior} nodes after guard trimming. "
                "Consider reducing r_max or disabling guard region."
            )
            # Fall back to using all nodes
            xyz_analysis = xyz_nodes
        else:
            xyz_analysis = xyz_interior
    else:
        xyz_analysis = xyz_nodes

    # Create log-spaced radial bins
    r_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)

    # Compute correlation integral and D2
    r_centers, C, D2 = corr_integral_D2(xyz_analysis, r_edges)

    # Identify plateau
    plateau_mask = identify_plateau(C, D2)
    D2_est, D2_mad = robust_D2_estimate(D2, plateau_mask)

    if verbose:
        print(f"Plateau: {plateau_mask.sum()} points")
        print(f"D2 estimate (MAD): {D2_est:.2f} +/- {D2_mad:.2f}")

    # Bootstrap uncertainty
    if bootstrap:
        D2_med, D2_std, _ = bootstrap_D2(
            xyz_analysis, r_edges, B=B, seed=seed
        )
        if verbose:
            print(f"D2 estimate (bootstrap): {D2_med:.2f} +/- {D2_std:.2f}")
    else:
        D2_std = D2_mad

    return D2Result(
        r_centers=r_centers,
        C=C,
        D2=D2,
        plateau_mask=plateau_mask,
        D2_estimate=D2_est,
        D2_std=D2_std
    )


# ============================================================================
# Optional: RR-based edge correction (compute ONCE, reuse)
# ============================================================================

def compute_RR_correction(
    xyz_nodes: np.ndarray,
    r_edges: np.ndarray,
    n_random: int = 10000,
    box_min: Optional[np.ndarray] = None,
    box_max: Optional[np.ndarray] = None,
    seed: int = 42
) -> np.ndarray:
    """
    Compute random-random pair counts for edge correction (ONCE).

    This is for advanced use when guard-region trimming is insufficient.
    Compute this ONCE and reuse across all bootstrap iterations.

    CRITICAL: Do NOT call this inside a bootstrap loop!

    Parameters
    ----------
    xyz_nodes : ndarray of shape (N, 3)
        Node positions (used only to infer box if not provided).
    r_edges : ndarray of shape (M,)
        Radii bin edges.
    n_random : int
        Number of random points to generate.
    box_min, box_max : ndarray of shape (3,), optional
        Analysis box bounds.
    seed : int
        Random seed.

    Returns
    -------
    RR : ndarray of shape (M-1,)
        Normalized random-random pair counts in each bin.

    Usage
    -----
    # Compute RR once
    RR = compute_RR_correction(xyz, r_edges)

    # In bootstrap loop, only resample nodes:
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        DD = compute_node_pairs(xyz[idx], r_edges)  # fast
        C_corrected = DD / RR  # edge correction applied
    """
    if box_min is None:
        box_min = xyz_nodes.min(axis=0)
    if box_max is None:
        box_max = xyz_nodes.max(axis=0)

    rng = np.random.default_rng(seed)

    # Generate uniform random points in the box
    xyz_random = rng.uniform(box_min, box_max, size=(n_random, 3))

    # Build KDTree and count pairs at each radius
    tree = cKDTree(xyz_random)
    cum = np.array([
        tree.count_neighbors(tree, r, cumulative=True)
        for r in r_edges
    ], dtype=np.float64)

    # Convert to per-bin counts
    cum = (cum - n_random) / 2.0  # remove self-pairs
    RR = cum[1:] - cum[:-1]  # differential counts

    # Normalize
    total_pairs = n_random * (n_random - 1) / 2.0
    RR = RR / total_pairs

    # Avoid division by zero
    RR = np.maximum(RR, 1e-15)

    return RR


# ============================================================================
# Visualization functions with CORRECT bin center handling
# ============================================================================

def plot_correlation_integral(
    result: D2Result,
    ax=None,
    show_plateau: bool = True,
    **kwargs
):
    """
    Plot correlation integral C(r) vs r.

    IMPORTANT: Uses r_centers (length M-1) consistently with C (length M-1).
    This prevents the 39 vs 40 mismatch.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Both arrays have the same length - no mismatch possible
    ax.loglog(result.r_centers, result.C, 'b.-', label='C(r)', **kwargs)

    if show_plateau and result.plateau_mask.any():
        ax.loglog(
            result.r_centers[result.plateau_mask],
            result.C[result.plateau_mask],
            'go', markersize=8, label='Plateau', alpha=0.5
        )

    ax.set_xlabel('r')
    ax.set_ylabel('C(r)')
    ax.set_title('Correlation Integral')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_local_dimension(
    result: D2Result,
    ax=None,
    show_plateau: bool = True,
    show_estimate: bool = True,
    **kwargs
):
    """
    Plot local dimension D2(r) vs r.

    IMPORTANT: Uses r_centers (length M-1) consistently with D2 (length M-1).
    This prevents the 39 vs 40 mismatch.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Both arrays have the same length - no mismatch possible
    ax.semilogx(result.r_centers, result.D2, 'b.-', label='$D_2(r)$', **kwargs)

    if show_plateau and result.plateau_mask.any():
        ax.semilogx(
            result.r_centers[result.plateau_mask],
            result.D2[result.plateau_mask],
            'go', markersize=8, label='Plateau', alpha=0.5
        )

    if show_estimate and np.isfinite(result.D2_estimate):
        ax.axhline(
            result.D2_estimate, color='r', linestyle='--',
            label=f'$D_2 = {result.D2_estimate:.2f} \\pm {result.D2_std:.2f}$'
        )
        ax.axhspan(
            result.D2_estimate - result.D2_std,
            result.D2_estimate + result.D2_std,
            color='r', alpha=0.1
        )

    ax.set_xlabel('r')
    ax.set_ylabel('$D_2(r)$')
    ax.set_title('Local Correlation Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 6)  # Reasonable range for physical systems

    return ax


def plot_D2_diagnostic(result: D2Result, figsize=(12, 5)):
    """
    Create diagnostic plot with correlation integral and local dimension.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    plot_correlation_integral(result, ax=ax1)
    plot_local_dimension(result, ax=ax2)

    plt.tight_layout()
    return fig, (ax1, ax2)


# ============================================================================
# Demo / test function
# ============================================================================

def demo_D2_analysis():
    """
    Demonstrate D2 analysis on synthetic point sets.
    """
    import matplotlib.pyplot as plt

    np.random.seed(42)

    # Generate test cases with known dimensions
    # 1. Random points in 3D box (D2 ~ 3)
    xyz_random = np.random.uniform(0, 100, size=(228, 3))

    # 2. Points on a 2D surface embedded in 3D (D2 ~ 2)
    theta = np.random.uniform(0, 2*np.pi, 228)
    phi = np.random.uniform(0, np.pi, 228)
    r_sphere = 50
    xyz_surface = np.column_stack([
        r_sphere * np.sin(phi) * np.cos(theta),
        r_sphere * np.sin(phi) * np.sin(theta),
        r_sphere * np.cos(phi)
    ])
    xyz_surface += np.random.normal(0, 1, xyz_surface.shape)  # Add noise

    # 3. Points along a 1D filament (D2 ~ 1)
    t = np.linspace(0, 100, 228)
    xyz_filament = np.column_stack([
        t,
        5 * np.sin(t / 10),
        5 * np.cos(t / 10)
    ])
    xyz_filament += np.random.normal(0, 0.5, xyz_filament.shape)

    # Analyze each
    cases = [
        ('3D Random (expect D2~3)', xyz_random),
        ('2D Surface (expect D2~2)', xyz_surface),
        ('1D Filament (expect D2~1)', xyz_filament),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for i, (name, xyz) in enumerate(cases):
        print(f"\n{'='*50}")
        print(f"{name}")
        print('='*50)

        result = compute_D2_full(
            xyz,
            use_guard_region=True,
            bootstrap=True,
            B=30,
            verbose=True
        )

        plot_correlation_integral(result, ax=axes[i, 0])
        plot_local_dimension(result, ax=axes[i, 1])
        axes[i, 0].set_title(f'{name}\nCorrelation Integral')
        axes[i, 1].set_title(f'{name}\n$D_2 = {result.D2_estimate:.2f} \\pm {result.D2_std:.2f}$')

    plt.tight_layout()
    plt.savefig('D2_diagnostic_demo.pdf')
    plt.savefig('D2_diagnostic_demo.png', dpi=150)
    print("\nSaved: D2_diagnostic_demo.pdf, D2_diagnostic_demo.png")
    plt.close()


if __name__ == '__main__':
    demo_D2_analysis()
