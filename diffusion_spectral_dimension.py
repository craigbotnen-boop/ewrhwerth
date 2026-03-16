"""
Diffusion-based Spectral Dimension (d_s) Computation

Computes spectral dimension from heat kernel trace on k-NN graphs:
    P(t) ~ t^(-d_s/2)  =>  d_s = -2 * d(log P)/d(log t)

Key design choices to avoid d_s = 0.00:
1. Ensure graph connectivity (auto-increase k if disconnected)
2. Extract largest connected component if fragmented
3. Filter zero/tiny eigenvalues before trace computation
4. Use normalized Laplacian for numerical stability

Reference: Burioni & Cassi (2005), J. Phys. A 38, R45-R78
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from typing import Tuple, Optional, NamedTuple
import warnings


class SpectralDimensionResult(NamedTuple):
    """Container for spectral dimension results."""
    d_s: float                  # Estimated spectral dimension
    d_s_std: float              # Uncertainty estimate
    t_values: np.ndarray        # Diffusion times
    P_t: np.ndarray             # Heat kernel trace P(t)
    local_d_s: np.ndarray       # Local d_s at each t
    plateau_mask: np.ndarray    # Valid fitting region
    n_components: int           # Number of connected components
    largest_component_size: int # Size of largest component used
    k_used: int                 # Actual k used for k-NN


def _build_knn_graph(
    xyz: np.ndarray,
    k: int,
    symmetric: bool = True
) -> csr_matrix:
    """
    Build k-nearest neighbor adjacency matrix.

    Parameters
    ----------
    xyz : ndarray of shape (N, D)
        Point coordinates.
    k : int
        Number of nearest neighbors.
    symmetric : bool
        If True, symmetrize the adjacency (i->j implies j->i).

    Returns
    -------
    A : csr_matrix of shape (N, N)
        Sparse adjacency matrix (binary weights).
    """
    N = xyz.shape[0]
    tree = cKDTree(xyz)

    # Query k+1 neighbors (includes self)
    distances, indices = tree.query(xyz, k=min(k + 1, N))

    # Build sparse adjacency
    row_idx = np.repeat(np.arange(N), k)
    col_idx = indices[:, 1:k+1].flatten()  # Skip self (index 0)

    # Handle case where k+1 > N
    valid = col_idx < N
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    data = np.ones(len(row_idx), dtype=np.float64)
    A = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))

    if symmetric:
        A = A + A.T
        A.data = np.minimum(A.data, 1.0)  # Binary adjacency

    return A


def _check_connectivity(A: csr_matrix) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Check graph connectivity and return component info.

    Returns
    -------
    n_components : int
        Number of connected components.
    labels : ndarray
        Component label for each node.
    sizes : ndarray
        Size of each component.
    """
    n_components, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    return n_components, labels, sizes


def _extract_largest_component(
    xyz: np.ndarray,
    A: csr_matrix,
    labels: np.ndarray,
    sizes: np.ndarray
) -> Tuple[np.ndarray, csr_matrix]:
    """Extract the largest connected component."""
    largest_label = np.argmax(sizes)
    mask = labels == largest_label

    xyz_lcc = xyz[mask]

    # Extract subgraph
    idx = np.where(mask)[0]
    A_lcc = A[idx][:, idx]

    return xyz_lcc, A_lcc, mask


def _build_graph_laplacian(
    A: csr_matrix,
    normalized: bool = True
) -> csr_matrix:
    """
    Build graph Laplacian from adjacency matrix.

    Parameters
    ----------
    A : csr_matrix
        Adjacency matrix.
    normalized : bool
        If True, use normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.
        This is more numerically stable and gives d_s in [0, 2] for
        regular lattices.

    Returns
    -------
    L : csr_matrix
        Graph Laplacian.
    """
    degrees = np.asarray(A.sum(axis=1)).flatten()

    # Handle isolated nodes (degree 0)
    degrees = np.maximum(degrees, 1e-10)

    if normalized:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        L = diags(np.ones(A.shape[0])) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Unnormalized: L = D - A
        D = diags(degrees)
        L = D - A

    return L.tocsr()


def _compute_eigenvalues(
    L: csr_matrix,
    n_eigs: int = 100
) -> np.ndarray:
    """
    Compute smallest eigenvalues of Laplacian.

    For heat kernel trace, we need the smallest eigenvalues (excluding
    the trivial zero eigenvalue from constant eigenvector).

    Parameters
    ----------
    L : csr_matrix
        Graph Laplacian.
    n_eigs : int
        Number of eigenvalues to compute.

    Returns
    -------
    eigenvalues : ndarray
        Sorted eigenvalues (ascending), smallest first.
    """
    N = L.shape[0]
    n_eigs = min(n_eigs, N - 2)  # eigsh needs k < N

    if n_eigs < 1:
        warnings.warn(f"Graph too small (N={N}) for eigendecomposition")
        return np.array([0.0])

    # For small graphs, use dense method (more reliable)
    if N <= 500:
        L_dense = L.toarray()
        eigenvalues = np.linalg.eigvalsh(L_dense)
        eigenvalues = np.sort(eigenvalues)
        # Return smallest n_eigs (including near-zero)
        return eigenvalues[:n_eigs]

    # For larger graphs, use sparse eigsh with shift-invert mode.
    # sigma=0 + which='LM' finds eigenvalues nearest zero via (L - 0*I)^{-1},
    # which converges much faster than which='SM' for large sparse Laplacians.
    # The 'SM' mode requires many ARPACK iterations and can stall at N > 100k.
    try:
        eigenvalues, _ = eigsh(L, k=n_eigs, sigma=0.0, which='LM')
        eigenvalues = np.sort(np.abs(eigenvalues))
    except Exception as e:
        # Shift-invert can fail if L is exactly singular (zero eigenvalue).
        # Fall back to which='SM' with relaxed tolerance.
        warnings.warn(f"Shift-invert eigsh failed ({e}); falling back to which='SM'")
        try:
            eigenvalues, _ = eigsh(L, k=n_eigs, which='SM', tol=1e-4)
            eigenvalues = np.sort(np.abs(eigenvalues))
        except Exception as e2:
            warnings.warn(f"Sparse eigendecomposition failed: {e2}. Using dense.")
            L_dense = L.toarray()
            eigenvalues = np.linalg.eigvalsh(L_dense)
            eigenvalues = np.sort(eigenvalues)[:n_eigs]

    return eigenvalues


def _heat_kernel_trace(
    eigenvalues: np.ndarray,
    t_values: np.ndarray,
    lambda_min: float = 1e-8
) -> np.ndarray:
    """
    Compute heat kernel trace P(t) = sum_i exp(-lambda_i * t).

    Parameters
    ----------
    eigenvalues : ndarray
        Laplacian eigenvalues.
    t_values : ndarray
        Diffusion times.
    lambda_min : float
        Minimum eigenvalue to include (filters numerical zeros).

    Returns
    -------
    P_t : ndarray
        Heat kernel trace at each time.
    """
    # Filter out near-zero eigenvalues (these cause P(t) = const)
    # The zero eigenvalue corresponds to the constant eigenvector
    valid_eigs = eigenvalues[eigenvalues > lambda_min]

    if len(valid_eigs) == 0:
        warnings.warn("No non-zero eigenvalues found! Graph may be disconnected.")
        return np.ones_like(t_values)

    # Compute trace: P(t) = sum_i exp(-lambda_i * t)
    # Shape: (n_times, n_eigs)
    exp_terms = np.exp(-np.outer(t_values, valid_eigs))
    P_t = exp_terms.sum(axis=1)

    return P_t


def _estimate_spectral_dimension(
    t_values: np.ndarray,
    P_t: np.ndarray,
    t_min_frac: float = 0.1,
    t_max_frac: float = 0.7
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Estimate d_s from heat kernel trace via log-log slope.

    d_s = -2 * d(log P)/d(log t)

    Parameters
    ----------
    t_values : ndarray
        Diffusion times.
    P_t : ndarray
        Heat kernel trace.
    t_min_frac, t_max_frac : float
        Fraction of time range to use for fitting (avoid early/late artifacts).

    Returns
    -------
    d_s : float
        Estimated spectral dimension.
    d_s_std : float
        Standard error from linear fit.
    local_d_s : ndarray
        Local d_s at each time point.
    plateau_mask : ndarray
        Boolean mask for fitting region.
    """
    log_t = np.log(t_values)
    log_P = np.log(np.maximum(P_t, 1e-300))

    # Local slope via gradient
    local_slope = np.gradient(log_P, log_t)
    local_d_s = -2 * local_slope

    # Define plateau region
    t_range = t_values.max() - t_values.min()
    t_lo = t_values.min() + t_min_frac * t_range
    t_hi = t_values.min() + t_max_frac * t_range

    plateau_mask = (t_values >= t_lo) & (t_values <= t_hi)
    plateau_mask &= np.isfinite(local_d_s)
    plateau_mask &= (local_d_s > 0) & (local_d_s < 10)  # Sanity bounds

    if plateau_mask.sum() < 3:
        warnings.warn("Insufficient plateau points for d_s estimation")
        return np.nan, np.nan, local_d_s, plateau_mask

    # Robust estimate: median in plateau
    d_s = np.median(local_d_s[plateau_mask])
    d_s_std = np.std(local_d_s[plateau_mask], ddof=1) / np.sqrt(plateau_mask.sum())

    return d_s, d_s_std, local_d_s, plateau_mask


def compute_spectral_dimension(
    xyz: np.ndarray,
    k: int = 10,
    k_max: int = 50,
    auto_connect: bool = True,
    n_eigs: int = 100,
    n_times: int = 50,
    t_min: float = 0.01,
    t_max: float = 100.0,
    normalized_laplacian: bool = True,
    verbose: bool = True
) -> SpectralDimensionResult:
    """
    Compute spectral dimension d_s from point cloud.

    The spectral dimension characterizes diffusion on the graph:
        P(t) ~ t^(-d_s/2)

    For Euclidean space: d_s = d (ambient dimension)
    For fractals: d_s < d (anomalous diffusion)

    Parameters
    ----------
    xyz : ndarray of shape (N, D)
        Point coordinates.
    k : int
        Initial k for k-NN graph.
    k_max : int
        Maximum k to try if graph is disconnected.
    auto_connect : bool
        If True, automatically increase k until graph is connected.
    n_eigs : int
        Number of Laplacian eigenvalues to compute.
    n_times : int
        Number of diffusion time points.
    t_min, t_max : float
        Range of diffusion times.
    normalized_laplacian : bool
        Use normalized Laplacian (recommended).
    verbose : bool
        Print diagnostic info.

    Returns
    -------
    SpectralDimensionResult
        Named tuple with d_s estimate and diagnostics.

    Notes
    -----
    Common causes of d_s = 0.00:
    1. Disconnected graph (k too small)
    2. All eigenvalues near zero (numerical issues)
    3. Too few points (N < 20)

    This implementation addresses all three via:
    - Auto-connectivity check and k adjustment
    - Largest connected component extraction
    - Eigenvalue filtering
    """
    N = xyz.shape[0]

    if N < 10:
        warnings.warn(f"Very few points (N={N}). d_s estimate will be unreliable.")

    if verbose:
        print(f"Spectral Dimension Analysis: N={N} points")

    # Build k-NN graph with connectivity check
    k_used = k
    A = _build_knn_graph(xyz, k_used)
    n_components, labels, sizes = _check_connectivity(A)

    if verbose:
        print(f"Initial k={k_used}: {n_components} connected component(s)")

    # Auto-increase k if disconnected
    if auto_connect and n_components > 1:
        while n_components > 1 and k_used < k_max:
            k_used += 5
            A = _build_knn_graph(xyz, k_used)
            n_components, labels, sizes = _check_connectivity(A)
            if verbose:
                print(f"  k={k_used}: {n_components} component(s)")

        if n_components > 1:
            warnings.warn(
                f"Graph still disconnected with k={k_used}. "
                f"Using largest component ({sizes.max()}/{N} nodes)."
            )

    # Extract largest component if still disconnected
    largest_size = N
    if n_components > 1:
        xyz_use, A_use, mask = _extract_largest_component(xyz, A, labels, sizes)
        largest_size = xyz_use.shape[0]
        if verbose:
            print(f"Using largest component: {largest_size} nodes")
    else:
        xyz_use, A_use = xyz, A

    # Build Laplacian
    L = _build_graph_laplacian(A_use, normalized=normalized_laplacian)

    # Compute eigenvalues
    n_eigs_use = min(n_eigs, largest_size - 2)
    if verbose:
        print(f"Computing {n_eigs_use} eigenvalues...")

    eigenvalues = _compute_eigenvalues(L, n_eigs=n_eigs_use)

    if verbose:
        n_nonzero = np.sum(eigenvalues > 1e-8)
        print(f"Non-zero eigenvalues: {n_nonzero}/{len(eigenvalues)}")
        if n_nonzero > 0:
            print(f"Eigenvalue range: [{eigenvalues[eigenvalues > 1e-8].min():.4e}, "
                  f"{eigenvalues.max():.4e}]")

    # Compute heat kernel trace
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    P_t = _heat_kernel_trace(eigenvalues, t_values)

    # Estimate spectral dimension
    d_s, d_s_std, local_d_s, plateau_mask = _estimate_spectral_dimension(t_values, P_t)

    if verbose:
        if np.isfinite(d_s):
            print(f"Spectral dimension: d_s = {d_s:.3f} +/- {d_s_std:.3f}")
        else:
            print("WARNING: Could not estimate d_s (check eigenvalues)")

    return SpectralDimensionResult(
        d_s=d_s,
        d_s_std=d_s_std,
        t_values=t_values,
        P_t=P_t,
        local_d_s=local_d_s,
        plateau_mask=plateau_mask,
        n_components=n_components,
        largest_component_size=largest_size,
        k_used=k_used
    )


# ============================================================================
# Hutchinson trace estimator for large graphs
# ============================================================================

def hutchinson_trace_estimate(
    L: csr_matrix,
    t_values: np.ndarray,
    n_vectors: int = 30,
    seed: int = 42
) -> np.ndarray:
    """
    Estimate heat kernel trace using Hutchinson's method.

    For large graphs where full eigendecomposition is expensive.
    Uses random vectors: Tr(exp(-tL)) ≈ E[z^T exp(-tL) z]

    Parameters
    ----------
    L : csr_matrix
        Graph Laplacian.
    t_values : ndarray
        Diffusion times.
    n_vectors : int
        Number of random probe vectors.
    seed : int
        Random seed.

    Returns
    -------
    P_t : ndarray
        Estimated heat kernel trace.
    """
    from scipy.sparse.linalg import expm_multiply

    rng = np.random.default_rng(seed)
    N = L.shape[0]

    P_t = np.zeros(len(t_values))

    for _ in range(n_vectors):
        # Rademacher random vector
        z = rng.choice([-1.0, 1.0], size=N)

        for i, t in enumerate(t_values):
            # exp(-tL) @ z via Krylov subspace method
            exp_Lz = expm_multiply(-t * L, z)
            P_t[i] += z @ exp_Lz

    P_t /= n_vectors
    return P_t


def compute_spectral_dimension_hutchinson(
    xyz: np.ndarray,
    k: int = 10,
    n_vectors: int = 50,
    n_times: int = 50,
    t_min: float = 0.01,
    t_max: float = 100.0,
    verbose: bool = True,
    **knn_kwargs
) -> SpectralDimensionResult:
    """
    Compute spectral dimension using Hutchinson trace estimator.

    More scalable for large graphs (N > 10000).
    """
    N = xyz.shape[0]

    if verbose:
        print(f"Spectral Dimension (Hutchinson): N={N} points")

    # Build graph
    A = _build_knn_graph(xyz, k)
    n_components, labels, sizes = _check_connectivity(A)

    if n_components > 1:
        xyz, A, _ = _extract_largest_component(xyz, A, labels, sizes)
        if verbose:
            print(f"Using largest component: {A.shape[0]} nodes")

    L = _build_graph_laplacian(A, normalized=True)

    # Hutchinson trace estimate
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_times)

    if verbose:
        print(f"Estimating trace with {n_vectors} random vectors...")

    P_t = hutchinson_trace_estimate(L, t_values, n_vectors=n_vectors)

    # Estimate d_s
    d_s, d_s_std, local_d_s, plateau_mask = _estimate_spectral_dimension(t_values, P_t)

    if verbose:
        print(f"Spectral dimension: d_s = {d_s:.3f} +/- {d_s_std:.3f}")

    return SpectralDimensionResult(
        d_s=d_s,
        d_s_std=d_s_std,
        t_values=t_values,
        P_t=P_t,
        local_d_s=local_d_s,
        plateau_mask=plateau_mask,
        n_components=n_components,
        largest_component_size=A.shape[0],
        k_used=k
    )


# ============================================================================
# Visualization
# ============================================================================

def plot_spectral_dimension(result: SpectralDimensionResult, figsize=(12, 5)):
    """Plot heat kernel trace and local d_s."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Heat kernel trace
    ax1.loglog(result.t_values, result.P_t, 'b.-')

    # Reference slopes
    if np.isfinite(result.d_s):
        t_ref = result.t_values
        P_ref = result.P_t[len(result.P_t)//2] * (t_ref / t_ref[len(t_ref)//2])**(-result.d_s/2)
        ax1.loglog(t_ref, P_ref, 'r--', alpha=0.7,
                   label=f'$t^{{-d_s/2}}$, $d_s={result.d_s:.2f}$')

    ax1.set_xlabel('Diffusion time $t$')
    ax1.set_ylabel('Heat kernel trace $P(t)$')
    ax1.set_title('Heat Kernel Decay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Local d_s
    ax2.semilogx(result.t_values, result.local_d_s, 'b.-')

    if result.plateau_mask.any():
        ax2.semilogx(result.t_values[result.plateau_mask],
                     result.local_d_s[result.plateau_mask],
                     'go', markersize=8, alpha=0.5, label='Plateau')

    if np.isfinite(result.d_s):
        ax2.axhline(result.d_s, color='r', linestyle='--',
                    label=f'$d_s = {result.d_s:.2f} \\pm {result.d_s_std:.2f}$')
        ax2.axhspan(result.d_s - result.d_s_std, result.d_s + result.d_s_std,
                    color='r', alpha=0.1)

    ax2.set_xlabel('Diffusion time $t$')
    ax2.set_ylabel('Local $d_s(t)$')
    ax2.set_title('Spectral Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)

    plt.tight_layout()
    return fig, (ax1, ax2)


# ============================================================================
# Demo
# ============================================================================

def demo_spectral_dimension():
    """Test on synthetic data with known d_s."""
    import matplotlib.pyplot as plt

    np.random.seed(42)

    print("="*60)
    print("SPECTRAL DIMENSION DEMO")
    print("="*60)

    # Test cases
    cases = []

    # 1. 3D uniform random (d_s ~ 3 for large N, but finite-size effects)
    xyz_3d = np.random.uniform(0, 100, size=(500, 3))
    cases.append(('3D Random (expect d_s~2-3)', xyz_3d))

    # 2. 2D surface (d_s ~ 2)
    theta = np.random.uniform(0, 2*np.pi, 500)
    phi = np.random.uniform(0, np.pi, 500)
    xyz_2d = 50 * np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    xyz_2d += np.random.normal(0, 1, xyz_2d.shape)
    cases.append(('2D Sphere (expect d_s~2)', xyz_2d))

    # 3. 1D filament (d_s ~ 1)
    t = np.linspace(0, 100, 500)
    xyz_1d = np.column_stack([t, 5*np.sin(t/10), 5*np.cos(t/10)])
    xyz_1d += np.random.normal(0, 0.5, xyz_1d.shape)
    cases.append(('1D Filament (expect d_s~1)', xyz_1d))

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for i, (name, xyz) in enumerate(cases):
        print(f"\n{name}")
        print("-"*40)

        result = compute_spectral_dimension(xyz, k=15, verbose=True)

        # Plot
        ax1, ax2 = axes[i]

        ax1.loglog(result.t_values, result.P_t, 'b.-')
        ax1.set_xlabel('t')
        ax1.set_ylabel('P(t)')
        ax1.set_title(f'{name}\nHeat Kernel')
        ax1.grid(True, alpha=0.3)

        ax2.semilogx(result.t_values, result.local_d_s, 'b.-')
        if result.plateau_mask.any():
            ax2.semilogx(result.t_values[result.plateau_mask],
                         result.local_d_s[result.plateau_mask], 'go', alpha=0.5)
        if np.isfinite(result.d_s):
            ax2.axhline(result.d_s, color='r', linestyle='--')
        ax2.set_xlabel('t')
        ax2.set_ylabel('$d_s(t)$')
        ax2.set_title(f'$d_s = {result.d_s:.2f} \\pm {result.d_s_std:.2f}$')
        ax2.set_ylim(0, 5)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spectral_dimension_demo.pdf')
    plt.savefig('spectral_dimension_demo.png', dpi=150)
    print("\nSaved: spectral_dimension_demo.pdf/png")
    plt.close()


if __name__ == '__main__':
    demo_spectral_dimension()
