"""
Kernel Testing Suite for Synthetic Data Analysis

Tests three kernel types (inverse-distance, Gaussian, quantile-normalized) across
four synthetic universes (Poisson, ΛCDM-like, Filamentary, GBM-surrogate) using
URC-style μ-sweep analysis of the shattering index σ(μ).

Key Components:
1. Synthetic Universes: Poisson (random), ΛCDM-like (clustered), Filamentary
2. GBM Surrogate: Gene expression network with block correlation structure
3. Kernels: Inverse-distance, Gaussian, Quantile-normalized
4. Analysis: σ(μ) curves and critical threshold μ_c estimation

The shattering index σ(μ) measures network fragmentation as:
    σ(μ) = 1 - GCC/N
where GCC is the size of the giant connected component and edges satisfy K_ij > μ.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# -----------------------------
# 1. Utilities: union-find & sigma
# -----------------------------
class UnionFind:
    """
    Disjoint-set data structure for tracking connected components.

    Uses path compression and union-by-size for near-O(1) amortized operations.
    """
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        """Find root with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Union by size."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def sigma_from_threshold(K, mu):
    """
    Compute shattering index for given threshold.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Symmetric interaction matrix (zero diagonal).
    mu : float
        Threshold value for edge inclusion.

    Returns
    -------
    sigma : float
        Shattering index σ = 1 - GCC/N, where GCC is the giant
        connected component size. Range: [0, 1], where 0 means
        fully connected and 1 means fully fragmented.

    Notes
    -----
    Edge rule: edge exists if K_ij > mu (strict inequality).
    """
    N = K.shape[0]
    # Get upper-triangular edges above threshold
    i, j = np.triu_indices(N, k=1)
    mask = K[i, j] > mu
    if mask.sum() == 0:
        return 1.0
    uf = UnionFind(N)
    for a, b in zip(i[mask], j[mask]):
        uf.union(a, b)
    # Largest component
    roots, counts = np.unique([uf.find(x) for x in range(N)], return_counts=True)
    gcc = counts.max()
    return 1.0 - gcc / N


def mu_sweep(K, mu_values):
    """
    Compute σ(μ) curve over a range of thresholds.

    Parameters
    ----------
    K : ndarray of shape (N, N)
        Interaction matrix.
    mu_values : ndarray
        Array of threshold values to test.

    Returns
    -------
    sigma_values : ndarray
        Shattering index for each threshold.
    """
    return np.array([sigma_from_threshold(K, mu) for mu in mu_values])


def estimate_mu_c(mu_values, sigma_values):
    """
    Estimate critical threshold μ_c where σ crosses 0.5.

    Parameters
    ----------
    mu_values : ndarray
        Threshold values.
    sigma_values : ndarray
        Corresponding shattering indices.

    Returns
    -------
    mu_c : float
        Approximate critical threshold (percolation transition point).
    """
    idx = np.argmin(np.abs(sigma_values - 0.5))
    return mu_values[idx]


# -----------------------------
# 2. Synthetic universes
# -----------------------------
def universe_poisson(N=500, box=100.0):
    """
    Generate Poisson (completely random) point distribution.

    Parameters
    ----------
    N : int
        Number of points.
    box : float
        Side length of cubic volume.

    Returns
    -------
    points : ndarray of shape (N, 3)
        Uniformly random 3D coordinates.
    """
    return np.random.uniform(0, box, size=(N, 3))


def universe_lcdm_like(N=500, box=100.0, n_clusters=8):
    """
    Generate ΛCDM-like clustered distribution.

    Uses mixture of Gaussians to simulate dark matter halos
    with smooth background.

    Parameters
    ----------
    N : int
        Number of points.
    box : float
        Side length of cubic volume.
    n_clusters : int
        Number of Gaussian clusters.

    Returns
    -------
    points : ndarray of shape (N, 3)
        Clustered 3D point distribution.
    """
    centers = np.random.uniform(0, box, size=(n_clusters, 3))
    pts = []
    per_cluster = N // n_clusters
    for c in centers:
        pts.append(c + np.random.normal(scale=box/20, size=(per_cluster, 3)))
    pts = np.vstack(pts)
    if pts.shape[0] < N:
        extra = np.random.uniform(0, box, size=(N - pts.shape[0], 3))
        pts = np.vstack([pts, extra])
    return pts


def universe_filamentary(N=500, box=100.0, n_filaments=10):
    """
    Generate filamentary cosmic web structure.

    Points distributed along random line segments with small
    perpendicular dispersion.

    Parameters
    ----------
    N : int
        Number of points.
    box : float
        Side length of cubic volume.
    n_filaments : int
        Number of filament segments.

    Returns
    -------
    points : ndarray of shape (N, 3)
        Filamentary 3D point distribution.
    """
    pts = []
    per_fil = N // n_filaments
    for _ in range(n_filaments):
        p0 = np.random.uniform(0, box, size=3)
        p1 = np.random.uniform(0, box, size=3)
        t = np.random.uniform(0, 1, size=(per_fil, 1))
        line = p0 + t * (p1 - p0)
        noise = np.random.normal(scale=box/100, size=(per_fil, 3))
        pts.append(line + noise)
    pts = np.vstack(pts)
    if pts.shape[0] < N:
        extra = np.random.uniform(0, box, size=(N - pts.shape[0], 3))
        pts = np.vstack([pts, extra])
    return pts


# -----------------------------
# 3. GBM surrogate
# -----------------------------
def gbm_surrogate(N_genes=400, n_blocks=5, in_block_corr=0.8, out_block_corr=0.1, seed=0):
    """
    Generate GBM (Glioblastoma) gene expression surrogate network.

    Creates block-structured correlation matrix representing modular
    gene regulation. Interaction strength is absolute correlation.

    Parameters
    ----------
    N_genes : int
        Number of genes.
    n_blocks : int
        Number of functional modules.
    in_block_corr : float
        Within-module correlation.
    out_block_corr : float
        Between-module correlation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    K : ndarray of shape (N_genes, N_genes)
        Symmetric correlation-based interaction matrix.
    """
    rng = np.random.default_rng(seed)
    # Block structure correlation matrix
    block_size = N_genes // n_blocks
    C = np.full((N_genes, N_genes), out_block_corr)
    for b in range(n_blocks):
        i0 = b * block_size
        i1 = (b + 1) * block_size if b < n_blocks - 1 else N_genes
        C[i0:i1, i0:i1] = in_block_corr
    np.fill_diagonal(C, 1.0)
    # Ensure positive semi-definite by adding small jitter
    C += np.eye(N_genes) * 1e-3
    # Sample expression matrix ~ N(0, C)
    L = np.linalg.cholesky(C)
    X = rng.normal(size=(N_genes, 50))  # 50 samples
    expr = L @ X  # genes x samples
    # Use correlation as interaction strength
    R = np.corrcoef(expr)
    np.fill_diagonal(R, 0.0)
    return R


# -----------------------------
# 4. Kernels
# -----------------------------
def K_from_points(points, kernel_type="inv", lam=10.0):
    """
    Construct interaction kernel from spatial point distribution.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        3D spatial coordinates.
    kernel_type : str
        Kernel function: 'inv', 'gauss', or 'quantile'.
    lam : float
        Length scale for Gaussian kernel.

    Returns
    -------
    K : ndarray of shape (N, N)
        Symmetric interaction matrix (zero diagonal).

    Kernel Types
    ------------
    inv : K_ij = 1/d_ij (inverse-distance, gravity-like)
    gauss : K_ij = exp(-d_ij^2 / (2*λ^2)) (Gaussian RBF)
    quantile : K_ij = 1 - quantile(d_ij) (rank-normalized)
    """
    d = squareform(pdist(points))  # pairwise distances
    np.fill_diagonal(d, np.inf)

    if kernel_type == "inv":
        K = 1.0 / d
        K[np.isinf(K)] = 0.0
    elif kernel_type == "gauss":
        K = np.exp(-d**2 / (2 * lam**2))
    elif kernel_type == "quantile":
        # Map distances to [0,1] via rank, then invert so small d -> large K
        flat = d[np.isfinite(d)].ravel()
        ranks = np.argsort(np.argsort(flat)).astype(float)
        q = ranks / (len(flat) - 1 + 1e-9)
        K_flat = 1.0 - q  # high for small distances
        K = np.zeros_like(d)
        K[np.isfinite(d)] = K_flat
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    np.fill_diagonal(K, 0.0)
    return K


def K_from_gbm_corr(R, kernel_type="inv"):
    """
    Construct kernel from GBM correlation matrix.

    Parameters
    ----------
    R : ndarray of shape (N, N)
        Correlation matrix (zero diagonal).
    kernel_type : str
        Kernel transformation: 'inv', 'gauss', or 'quantile'.

    Returns
    -------
    K : ndarray of shape (N, N)
        Kernel-transformed interaction matrix.

    Transformations
    ---------------
    inv : K = |R| (absolute correlation)
    gauss : K = exp(-(1-|R|)^2 / (2*λ^2)) (similarity kernel)
    quantile : K = quantile-normalized |R| (rank-based)
    """
    A = np.abs(R)
    np.fill_diagonal(A, 0.0)

    if kernel_type == "inv":
        K = A.copy()
    elif kernel_type == "gauss":
        lam = 0.2
        K = np.exp(-(1.0 - A)**2 / (2 * lam**2))
    elif kernel_type == "quantile":
        flat = A.ravel()
        ranks = np.argsort(np.argsort(flat)).astype(float)
        q = ranks / (len(flat) - 1 + 1e-9)
        K = q.reshape(A.shape)
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    np.fill_diagonal(K, 0.0)
    return K


# -----------------------------
# 5. Run the full suite
# -----------------------------
def run_suite():
    """
    Execute complete kernel testing suite.

    Generates 4×3 panel plot showing σ(μ) curves for:
    - 4 universes: Poisson, ΛCDM-like, Filamentary, GBM-surrogate
    - 3 kernels: inverse-distance, Gaussian, quantile-normalized

    Prints table of estimated critical thresholds μ_c for each case.
    """
    np.random.seed(42)

    # Universes
    U_names = ["Poisson", "LCDM-like", "Filamentary", "GBM-surrogate"]
    pts_poisson = universe_poisson()
    pts_lcdm = universe_lcdm_like()
    pts_fil = universe_filamentary()
    R_gbm = gbm_surrogate()

    # Kernels
    K_types = ["inv", "gauss", "quantile"]
    K_labels = {
        "inv": "Inverse-distance / |corr|",
        "gauss": "Gaussian",
        "quantile": "Quantile-normalized"
    }

    # μ sweep range
    mu_grid = np.linspace(0.0, 1.0, 30)

    fig, axes = plt.subplots(len(U_names), len(K_types), figsize=(14, 12),
                             sharex=True, sharey=True)
    mu_c_table = []

    for ui, uname in enumerate(U_names):
        for ki, ktype in enumerate(K_types):
            ax = axes[ui, ki]

            if uname == "Poisson":
                K = K_from_points(pts_poisson, kernel_type=ktype)
            elif uname == "LCDM-like":
                K = K_from_points(pts_lcdm, kernel_type=ktype)
            elif uname == "Filamentary":
                K = K_from_points(pts_fil, kernel_type=ktype)
            elif uname == "GBM-surrogate":
                K = K_from_gbm_corr(R_gbm, kernel_type=ktype)
            else:
                continue

            sigmas = mu_sweep(K, mu_grid)
            mu_c = estimate_mu_c(mu_grid, sigmas)
            mu_c_table.append((uname, ktype, float(mu_c)))

            ax.plot(mu_grid, sigmas, 'r-', lw=2)
            ax.axvline(mu_c, color='k', ls='--', alpha=0.5)
            if ui == 0:
                ax.set_title(K_labels[ktype])
            if ki == 0:
                ax.set_ylabel(f"{uname}\nσ(μ)")
            if ui == len(U_names) - 1:
                ax.set_xlabel("μ (threshold on K_ij)")
            ax.grid(alpha=0.3)

    plt.suptitle("Unified Sigmoid: σ(μ) Across Universes and Kernels", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure instead of showing (for headless environments)
    plt.savefig("kernel_test_results.png", dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'kernel_test_results.png'")

    print("\nApproximate μ_c per universe and kernel:")
    print("-" * 50)
    for uname, ktype, muc in mu_c_table:
        print(f"{uname:15s} | {ktype:9s} | μ_c ≈ {muc:.3f}")
    print("-" * 50)


if __name__ == "__main__":
    run_suite()
