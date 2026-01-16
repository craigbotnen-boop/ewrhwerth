"""
DESI Return Probability and Spectral Dimension Analysis
========================================================

Drop-in replacement for spectral dimension estimation that:
1. Uses DESI DR1 LSS random catalogs + weights as the control (rather than uniform Poisson)
2. Adds boundary/selection excision diagnostic driven by the randoms
3. Replaces kNN with an epsilon-ball (fixed physical radius) graph for operator stability

References:
-----------
- DESI DR1 LSS documentation: https://data.desi.lbl.gov/doc/releases/dr1/
- Graph Laplacian construction dependence: Ting-Huang-Jordan, arXiv:1101.5435
- CDT spectral dimension: Coumbe et al., JHEP03(2015)151

Author: Generated for spectral dimension analysis pipeline
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

# Optional dependencies with graceful fallback
try:
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    logging.warning("astropy not available; using approximate comoving distances")

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Catalog:
    """
    Container for galaxy/random catalog data.

    Attributes
    ----------
    ra_deg : np.ndarray
        Right ascension in degrees
    dec_deg : np.ndarray
        Declination in degrees
    z : np.ndarray
        Spectroscopic redshift
    w : np.ndarray
        Total weight (data: WEIGHT_COMP * WEIGHT_SYS * WEIGHT_ZFAIL * WEIGHT_FKP or equivalent;
                      randoms: typically 1.0 or WEIGHT_FKP if provided)
    name : str
        Catalog identifier for logging
    """
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    z: np.ndarray
    w: np.ndarray
    name: str = "catalog"

    def __post_init__(self):
        # Ensure all arrays have consistent length
        n = len(self.ra_deg)
        assert len(self.dec_deg) == n, "dec_deg length mismatch"
        assert len(self.z) == n, "z length mismatch"
        assert len(self.w) == n, "w length mismatch"
        logger.info(f"Catalog '{self.name}' loaded with {n} objects")

    def select(self, mask: np.ndarray) -> 'Catalog':
        """Return a new Catalog with only selected objects."""
        return Catalog(
            ra_deg=self.ra_deg[mask],
            dec_deg=self.dec_deg[mask],
            z=self.z[mask],
            w=self.w[mask],
            name=self.name
        )


@dataclass
class SpectralDimensionResult:
    """
    Container for spectral dimension analysis results.

    All fields are JSON-serializable for audit logging.
    """
    n_raw: int                          # Objects in z-slice before excision
    n_used: int                         # Objects after boundary excision
    n_edges: int                        # Number of edges in epsilon-ball graph
    mean_degree: float                  # Mean node degree
    boundary_score_threshold: float     # Quantile threshold used
    eps_mpc: float                      # Epsilon-ball radius in Mpc
    zmin: float                         # Redshift slice lower bound
    zmax: float                         # Redshift slice upper bound
    t_grid: np.ndarray                  # Diffusion time values
    P_t: np.ndarray                     # Return probability P(t) = tr(exp(-tL))/n
    d_s_t: np.ndarray                   # Spectral dimension d_s(t)
    plateau_t_min: Optional[float]      # Detected plateau window start
    plateau_t_max: Optional[float]      # Detected plateau window end
    d_s_plateau: Optional[float]        # Mean d_s in plateau window
    d_s_plateau_std: Optional[float]    # Std of d_s in plateau window

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Convert numpy arrays to lists
        for key in ['t_grid', 'P_t', 'd_s_t']:
            if d[key] is not None:
                d[key] = d[key].tolist()
        return d

    def save_json(self, path: str):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")


# =============================================================================
# Coordinate Transformations
# =============================================================================

def _approx_comoving_distance(z: np.ndarray, Om0: float = 0.3111) -> np.ndarray:
    """
    Approximate comoving distance in Mpc/h using Mattig formula.
    For z < 0.5, error is < 2% compared to full integral.
    """
    # Speed of light in km/s
    c = 299792.458
    # Hubble constant with h=1
    H0 = 100.0  # km/s/Mpc * h

    # Simple approximation: chi ~ (c/H0) * z * (1 - z*(1+Om0)/4)
    # Better: use numerical integration of 1/E(z)
    from scipy.integrate import cumulative_trapezoid

    z_grid = np.linspace(0, np.max(z) * 1.1, 1000)
    E_z = np.sqrt(Om0 * (1 + z_grid)**3 + (1 - Om0))
    chi_grid = (c / H0) * cumulative_trapezoid(1.0 / E_z, z_grid, initial=0)

    # Interpolate to input redshifts
    return np.interp(z, z_grid, chi_grid)


def comoving_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray,
                       cosmo=None, Om0: float = 0.3111) -> np.ndarray:
    """
    Convert RA/DEC/Z to comoving Cartesian coordinates in Mpc/h.

    Parameters
    ----------
    ra_deg : np.ndarray
        Right ascension in degrees
    dec_deg : np.ndarray
        Declination in degrees
    z : np.ndarray
        Redshift
    cosmo : astropy.cosmology object, optional
        If provided, use astropy for precise distances
    Om0 : float
        Matter density parameter (used if cosmo is None)

    Returns
    -------
    X : np.ndarray, shape (N, 3)
        Comoving Cartesian coordinates [x, y, z] in Mpc/h
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    if cosmo is not None and HAS_ASTROPY:
        # Use astropy for precise comoving distance
        chi = cosmo.comoving_distance(z).to(u.Mpc).value
        # Scale by h to get Mpc/h
        h = cosmo.H0.value / 100.0
        chi = chi * h
    else:
        chi = _approx_comoving_distance(z, Om0=Om0)

    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc = chi * np.sin(dec)

    return np.column_stack([x, y, zc])


# =============================================================================
# Boundary/Selection Diagnostics
# =============================================================================

def boundary_score_from_randoms(X_data: np.ndarray, X_rand: np.ndarray,
                                 m: int = 32) -> np.ndarray:
    """
    Compute boundary/selection score for data points using random catalog geometry.

    Large distances to the m-th nearest random imply:
    - Low local selection density (incompleteness)
    - Proximity to survey mask boundary
    - Edge effects that will bias the graph Laplacian

    Parameters
    ----------
    X_data : np.ndarray, shape (N_data, 3)
        Data point coordinates in comoving Mpc/h
    X_rand : np.ndarray, shape (N_rand, 3)
        Random catalog coordinates (same selection function as data)
    m : int
        Use distance to m-th nearest random neighbor

    Returns
    -------
    score : np.ndarray, shape (N_data,)
        Boundary score (larger = more boundary-dominated)
    """
    logger.info(f"Computing boundary scores using {X_rand.shape[0]} randoms, m={m}")

    tree = cKDTree(X_rand)
    dists, _ = tree.query(X_data, k=m)

    # Return distance to m-th neighbor (last column)
    if dists.ndim == 1:
        return dists
    return dists[:, -1]


def excise_by_boundary_score(X: np.ndarray, w: np.ndarray, score: np.ndarray,
                             q: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Excise boundary-dominated points based on boundary score quantile.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Point coordinates
    w : np.ndarray, shape (N,)
        Point weights
    score : np.ndarray, shape (N,)
        Boundary scores from boundary_score_from_randoms
    q : float
        Quantile threshold; keep points with score <= quantile(score, q)
        E.g., q=0.90 removes the worst 10% of boundary-dominated points

    Returns
    -------
    X_kept : np.ndarray
        Coordinates of kept points
    w_kept : np.ndarray
        Weights of kept points
    keep_mask : np.ndarray
        Boolean mask of kept points
    threshold : float
        Score threshold used
    """
    threshold = np.quantile(score, q)
    keep_mask = score <= threshold
    n_excised = np.sum(~keep_mask)

    logger.info(f"Boundary excision: q={q:.2f}, threshold={threshold:.2f} Mpc/h, "
                f"excised {n_excised}/{len(score)} points ({100*n_excised/len(score):.1f}%)")

    return X[keep_mask], w[keep_mask], keep_mask, threshold


# =============================================================================
# Graph Construction
# =============================================================================

def build_eps_ball_graph(X: np.ndarray, eps: float,
                         weights: Optional[np.ndarray] = None) -> sparse.csr_matrix:
    """
    Build unweighted epsilon-ball adjacency graph.

    Connects all pairs of points within distance eps.
    Using fixed radius (rather than kNN) ensures the graph scale is
    physically interpretable and avoids density-dependent operator drift.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Point coordinates in Mpc/h
    eps : float
        Neighborhood radius in Mpc/h
    weights : np.ndarray, optional
        Point weights (currently unused; reserved for weighted Laplacian)

    Returns
    -------
    A : sparse.csr_matrix
        Symmetric adjacency matrix

    Raises
    ------
    ValueError
        If no edges found (eps too small) or graph is disconnected
    """
    logger.info(f"Building epsilon-ball graph with eps={eps:.1f} Mpc/h on {X.shape[0]} points")

    tree = cKDTree(X)
    pairs = tree.query_pairs(r=eps, output_type='ndarray')

    n = X.shape[0]
    if pairs.size == 0:
        raise ValueError(
            f"No edges found with eps={eps} Mpc/h. "
            "Increase eps or check coordinate units (should be Mpc/h)."
        )

    i = pairs[:, 0]
    j = pairs[:, 1]
    data = np.ones(len(i), dtype=np.float64)

    A = sparse.coo_matrix((data, (i, j)), shape=(n, n))
    A = A + A.T  # Symmetrize
    A = A.tocsr()

    # Compute basic graph statistics
    degrees = np.array(A.sum(axis=1)).ravel()
    n_edges = len(i)
    mean_deg = np.mean(degrees)
    n_isolated = np.sum(degrees == 0)

    logger.info(f"Graph: {n_edges} edges, mean degree={mean_deg:.1f}, "
                f"isolated nodes={n_isolated}")

    if n_isolated > 0:
        raise ValueError(
            f"Graph has {n_isolated} isolated nodes. "
            "Increase eps or reduce boundary excision aggressiveness."
        )

    return A


def normalized_laplacian(A: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Compute symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}

    The normalized Laplacian has eigenvalues in [0, 2] and is preferred
    for spectral dimension analysis as it removes degree bias.

    Parameters
    ----------
    A : sparse.csr_matrix
        Symmetric adjacency matrix

    Returns
    -------
    L : sparse.csr_matrix
        Normalized Laplacian
    """
    deg = np.array(A.sum(axis=1)).ravel()

    if np.any(deg == 0):
        raise ValueError("Cannot compute normalized Laplacian with isolated nodes")

    inv_sqrt_deg = 1.0 / np.sqrt(deg)
    Dinv2 = sparse.diags(inv_sqrt_deg)

    L = sparse.identity(A.shape[0], format='csr') - Dinv2 @ A @ Dinv2

    return L


# =============================================================================
# Trace Estimation and Spectral Dimension
# =============================================================================

def hutchinson_trace_exp(L: sparse.csr_matrix, t_grid: np.ndarray,
                         R: int = 64, seed: int = 0,
                         verbose: bool = True) -> np.ndarray:
    """
    Estimate tr(exp(-tL)) for each t using Hutchinson trace estimator.

    Uses Rademacher random vectors and expm_multiply for matrix exponential.
    This is O(R * |E| * len(t_grid)) where |E| is number of edges.

    Parameters
    ----------
    L : sparse.csr_matrix
        Normalized Laplacian
    t_grid : np.ndarray
        Diffusion time values
    R : int
        Number of Rademacher random vectors (more = lower variance)
    seed : int
        Random seed for reproducibility
    verbose : bool
        Log progress

    Returns
    -------
    traces : np.ndarray
        Estimated tr(exp(-tL)) for each t
    """
    rng = np.random.default_rng(seed)
    n = L.shape[0]

    # Rademacher vectors: entries in {-1, +1}
    G = rng.choice([-1.0, 1.0], size=(n, R)).astype(np.float64)

    traces = np.zeros(len(t_grid))

    for idx, t in enumerate(t_grid):
        if verbose and idx % 10 == 0:
            logger.info(f"Computing trace estimate: t={t:.3f} ({idx+1}/{len(t_grid)})")

        # Compute exp(-tL) @ G using Krylov subspace method
        Y = expm_multiply((-t) * L, G)

        # Hutchinson estimator: tr(M) ≈ (1/R) Σ_r g_r^T M g_r
        traces[idx] = np.mean(np.sum(G * Y, axis=0))

    return traces


def spectral_dimension_from_Pt(t_grid: np.ndarray, P_t: np.ndarray,
                               smooth_window: int = 3) -> np.ndarray:
    """
    Extract spectral dimension d_s(t) from return probability P(t).

    Uses the relation: P(t) ~ t^{-d_s/2}
    Therefore: d_s(t) = -2 * d(log P) / d(log t)

    Parameters
    ----------
    t_grid : np.ndarray
        Diffusion time values
    P_t : np.ndarray
        Return probability P(t) = tr(exp(-tL))/n
    smooth_window : int
        Window size for local linear regression (reduces noise)

    Returns
    -------
    d_s : np.ndarray
        Spectral dimension at each t
    """
    lt = np.log(t_grid)
    lp = np.log(np.maximum(P_t, 1e-30))  # Avoid log(0)

    d_s = np.zeros_like(P_t)

    for i in range(len(t_grid)):
        i0 = max(0, i - smooth_window)
        i1 = min(len(t_grid), i + smooth_window + 1)

        if i1 - i0 >= 2:
            # Local linear fit in log-log space
            slope, _ = np.polyfit(lt[i0:i1], lp[i0:i1], deg=1)
            d_s[i] = -2.0 * slope
        else:
            d_s[i] = np.nan

    return d_s


def detect_plateau(t_grid: np.ndarray, d_s: np.ndarray,
                   stability_threshold: float = 0.3,
                   min_window_decades: float = 0.5) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Detect plateau region in d_s(t) curve.

    Parameters
    ----------
    t_grid : np.ndarray
        Diffusion times
    d_s : np.ndarray
        Spectral dimension values
    stability_threshold : float
        Maximum relative variation in d_s to count as plateau
    min_window_decades : float
        Minimum width of plateau in log10(t) decades

    Returns
    -------
    t_min, t_max : float or None
        Plateau time window bounds
    d_s_mean, d_s_std : float or None
        Mean and std of d_s in plateau
    """
    valid = ~np.isnan(d_s) & (d_s > 0) & (d_s < 10)
    if np.sum(valid) < 5:
        return None, None, None, None

    lt = np.log10(t_grid[valid])
    ds_valid = d_s[valid]

    best_window = None
    best_score = np.inf

    # Scan for stable windows
    for i_start in range(len(lt)):
        for i_end in range(i_start + 3, len(lt)):
            window_width = lt[i_end] - lt[i_start]
            if window_width < min_window_decades:
                continue

            ds_window = ds_valid[i_start:i_end+1]
            mean_ds = np.mean(ds_window)
            std_ds = np.std(ds_window)

            if mean_ds > 0 and std_ds / mean_ds < stability_threshold:
                # Score: prefer wider windows with lower variance
                score = std_ds / mean_ds - 0.1 * window_width
                if score < best_score:
                    best_score = score
                    best_window = (
                        10**lt[i_start],
                        10**lt[i_end],
                        mean_ds,
                        std_ds
                    )

    if best_window is None:
        return None, None, None, None

    return best_window


# =============================================================================
# Main Pipeline
# =============================================================================

def run_ds_pipeline(data: Catalog, rand: Catalog,
                    zmin: float, zmax: float,
                    eps_mpc: float = 50.0,
                    boundary_m: int = 32,
                    boundary_quantile: float = 0.90,
                    t_grid: Optional[np.ndarray] = None,
                    R: int = 64,
                    seed: int = 0,
                    cosmo=None) -> SpectralDimensionResult:
    """
    End-to-end spectral dimension pipeline.

    Steps:
    1. Select redshift slice
    2. Convert to comoving coordinates
    3. Compute boundary scores using randoms
    4. Excise boundary-dominated points
    5. Build epsilon-ball graph
    6. Compute normalized Laplacian
    7. Estimate P(t) via Hutchinson trace
    8. Extract d_s(t) from P(t)
    9. Detect plateau window

    Parameters
    ----------
    data : Catalog
        Galaxy data catalog
    rand : Catalog
        Random catalog with same selection function
    zmin, zmax : float
        Redshift slice bounds
    eps_mpc : float
        Epsilon-ball radius in Mpc/h
    boundary_m : int
        Number of random neighbors for boundary score
    boundary_quantile : float
        Quantile for boundary excision (0.90 = remove worst 10%)
    t_grid : np.ndarray, optional
        Diffusion time grid (default: logspace(-1, 2, 40))
    R : int
        Number of Hutchinson probe vectors
    seed : int
        Random seed
    cosmo : astropy.cosmology, optional
        Cosmology for distance calculation

    Returns
    -------
    result : SpectralDimensionResult
        Complete analysis results
    """
    logger.info(f"=" * 60)
    logger.info(f"Running spectral dimension pipeline")
    logger.info(f"z-slice: [{zmin:.3f}, {zmax:.3f})")
    logger.info(f"eps = {eps_mpc:.1f} Mpc/h, boundary_q = {boundary_quantile:.2f}")
    logger.info(f"=" * 60)

    # Set default cosmology if not provided
    if cosmo is None and HAS_ASTROPY:
        cosmo = FlatLambdaCDM(H0=67.66, Om0=0.3111, Ob0=0.049)
        logger.info("Using default Planck 2018 cosmology")

    # Select redshift slice
    sel_d = (data.z >= zmin) & (data.z < zmax)
    sel_r = (rand.z >= zmin) & (rand.z < zmax)

    n_data_raw = int(np.sum(sel_d))
    n_rand_raw = int(np.sum(sel_r))
    logger.info(f"In z-slice: {n_data_raw} data, {n_rand_raw} randoms")

    if n_data_raw < 100:
        raise ValueError(f"Too few data points ({n_data_raw}) in z-slice")
    if n_rand_raw < 1000:
        raise ValueError(f"Too few random points ({n_rand_raw}) in z-slice")

    # Convert to comoving coordinates
    Xd = comoving_cartesian(data.ra_deg[sel_d], data.dec_deg[sel_d],
                            data.z[sel_d], cosmo)
    Xr = comoving_cartesian(rand.ra_deg[sel_r], rand.dec_deg[sel_r],
                            rand.z[sel_r], cosmo)

    wd = data.w[sel_d]

    # Boundary excision
    score = boundary_score_from_randoms(Xd, Xr, m=boundary_m)
    Xd_excised, wd_excised, keep_mask, threshold = excise_by_boundary_score(
        Xd, wd, score, q=boundary_quantile
    )

    # Build graph
    A = build_eps_ball_graph(Xd_excised, eps=eps_mpc)
    L = normalized_laplacian(A)

    # Graph statistics
    n_edges = A.nnz // 2
    mean_degree = np.mean(np.array(A.sum(axis=1)).ravel())

    # Diffusion time grid
    if t_grid is None:
        # Default: span ~3 decades, should cover graph-scale to system-scale diffusion
        t_grid = np.logspace(-1, 2, 40)

    # Trace estimation
    tr_exp = hutchinson_trace_exp(L, t_grid, R=R, seed=seed)
    P_t = tr_exp / L.shape[0]

    # Extract spectral dimension
    d_s_t = spectral_dimension_from_Pt(t_grid, P_t, smooth_window=3)

    # Detect plateau
    t_min, t_max, ds_mean, ds_std = detect_plateau(t_grid, d_s_t)

    if t_min is not None:
        logger.info(f"Plateau detected: t in [{t_min:.2f}, {t_max:.2f}], "
                    f"d_s = {ds_mean:.3f} +/- {ds_std:.3f}")
    else:
        logger.warning("No clear plateau detected in d_s(t)")

    result = SpectralDimensionResult(
        n_raw=n_data_raw,
        n_used=Xd_excised.shape[0],
        n_edges=n_edges,
        mean_degree=mean_degree,
        boundary_score_threshold=threshold,
        eps_mpc=eps_mpc,
        zmin=zmin,
        zmax=zmax,
        t_grid=t_grid,
        P_t=P_t,
        d_s_t=d_s_t,
        plateau_t_min=t_min,
        plateau_t_max=t_max,
        d_s_plateau=ds_mean,
        d_s_plateau_std=ds_std
    )

    return result


def compare_data_randoms(data: Catalog, rand: Catalog,
                         zmin: float, zmax: float,
                         eps_mpc: float = 50.0,
                         boundary_m: int = 32,
                         boundary_quantile: float = 0.90,
                         t_grid: Optional[np.ndarray] = None,
                         R: int = 64,
                         seed: int = 0,
                         cosmo=None) -> Dict[str, Any]:
    """
    Run pipeline on both data and randoms (as data), compute differential signal.

    This is the proper control: randoms encode the selection function,
    so Delta_ds = ds_data - ds_rand isolates clustering signal.

    Parameters
    ----------
    (Same as run_ds_pipeline)

    Returns
    -------
    comparison : dict
        Contains 'data', 'rand' SpectralDimensionResult dicts,
        and 'delta_ds', 'delta_ds_plateau' differential measurements
    """
    logger.info("=" * 70)
    logger.info("COMPARISON: Running pipeline on DATA and RANDOMS")
    logger.info("=" * 70)

    # Run on actual data
    logger.info("\n>>> Processing DATA catalog")
    result_data = run_ds_pipeline(
        data, rand, zmin, zmax, eps_mpc, boundary_m, boundary_quantile,
        t_grid, R, seed, cosmo
    )

    # Run on randoms-as-data (use a subsample as "data", rest as control)
    # Split randoms 50/50 for data vs control
    logger.info("\n>>> Processing RANDOMS catalog (as data, for null test)")
    n_rand = len(rand.ra_deg)
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(n_rand)
    split = n_rand // 2

    rand_as_data = Catalog(
        ra_deg=rand.ra_deg[perm[:split]],
        dec_deg=rand.dec_deg[perm[:split]],
        z=rand.z[perm[:split]],
        w=rand.w[perm[:split]],
        name="randoms_as_data"
    )
    rand_as_control = Catalog(
        ra_deg=rand.ra_deg[perm[split:]],
        dec_deg=rand.dec_deg[perm[split:]],
        z=rand.z[perm[split:]],
        w=rand.w[perm[split:]],
        name="randoms_as_control"
    )

    result_rand = run_ds_pipeline(
        rand_as_data, rand_as_control, zmin, zmax, eps_mpc, boundary_m,
        boundary_quantile, t_grid, R, seed + 1, cosmo
    )

    # Compute differential
    delta_ds = result_data.d_s_t - result_rand.d_s_t

    # Plateau-averaged differential
    delta_ds_plateau = None
    delta_ds_plateau_err = None
    if result_data.plateau_t_min is not None and result_rand.plateau_t_min is not None:
        # Use intersection of plateau windows
        t_min_common = max(result_data.plateau_t_min, result_rand.plateau_t_min)
        t_max_common = min(result_data.plateau_t_max, result_rand.plateau_t_max)

        if t_max_common > t_min_common:
            mask = (result_data.t_grid >= t_min_common) & (result_data.t_grid <= t_max_common)
            delta_ds_plateau = float(np.mean(delta_ds[mask]))
            delta_ds_plateau_err = float(np.std(delta_ds[mask]) / np.sqrt(np.sum(mask)))

            logger.info(f"\nDifferential plateau: Delta_ds = {delta_ds_plateau:.3f} +/- {delta_ds_plateau_err:.3f}")

    return {
        'data': result_data.to_dict(),
        'rand': result_rand.to_dict(),
        'delta_ds': delta_ds.tolist(),
        'delta_ds_plateau': delta_ds_plateau,
        'delta_ds_plateau_err': delta_ds_plateau_err
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_convergence_diagnostic(result_data: SpectralDimensionResult,
                                result_rand: SpectralDimensionResult,
                                output_path: str = "ds_convergence.pdf"):
    """
    Generate the single convergence figure needed for the appendix.

    Shows:
    - Top left: P(t) for data and randoms
    - Top right: d_s(t) for data and randoms
    - Bottom left: Delta d_s(t) = d_s_data - d_s_rand
    - Bottom right: Summary statistics
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    t = result_data.t_grid

    # Top left: Return probability P(t)
    ax = axs[0, 0]
    ax.loglog(t, result_data.P_t, 'b-', lw=2, label='Data')
    ax.loglog(t, result_rand.P_t, 'r--', lw=2, label='Randoms')
    ax.set_xlabel(r'Diffusion time $t$')
    ax.set_ylabel(r'Return probability $P(t)$')
    ax.set_title('Return Probability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Spectral dimension d_s(t)
    ax = axs[0, 1]
    ax.semilogx(t, result_data.d_s_t, 'b-', lw=2, label='Data')
    ax.semilogx(t, result_rand.d_s_t, 'r--', lw=2, label='Randoms')
    ax.axhline(3, color='gray', ls=':', label='$d_s=3$ (Euclidean)')

    # Shade plateau regions
    if result_data.plateau_t_min is not None:
        ax.axvspan(result_data.plateau_t_min, result_data.plateau_t_max,
                   alpha=0.2, color='blue', label='Data plateau')
    if result_rand.plateau_t_min is not None:
        ax.axvspan(result_rand.plateau_t_min, result_rand.plateau_t_max,
                   alpha=0.2, color='red', label='Rand plateau')

    ax.set_xlabel(r'Diffusion time $t$')
    ax.set_ylabel(r'Spectral dimension $d_s(t)$')
    ax.set_title('Spectral Dimension')
    ax.set_ylim(0, 6)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Bottom left: Differential Delta d_s
    ax = axs[1, 0]
    delta_ds = result_data.d_s_t - result_rand.d_s_t
    ax.semilogx(t, delta_ds, 'k-', lw=2)
    ax.axhline(0, color='gray', ls='--')
    ax.fill_between(t, delta_ds, 0, where=delta_ds > 0, alpha=0.3, color='green')
    ax.fill_between(t, delta_ds, 0, where=delta_ds < 0, alpha=0.3, color='red')
    ax.set_xlabel(r'Diffusion time $t$')
    ax.set_ylabel(r'$\Delta d_s = d_{s,\mathrm{data}} - d_{s,\mathrm{rand}}$')
    ax.set_title('Differential Spectral Dimension')
    ax.grid(True, alpha=0.3)

    # Bottom right: Summary text
    ax = axs[1, 1]
    ax.axis('off')

    summary_text = f"""
Analysis Summary
================

Redshift range: [{result_data.zmin:.3f}, {result_data.zmax:.3f})
ε-ball radius: {result_data.eps_mpc:.1f} Mpc/h

DATA:
  N (raw):  {result_data.n_raw}
  N (used): {result_data.n_used}
  Edges:    {result_data.n_edges}
  <k>:      {result_data.mean_degree:.1f}
  d_s (plateau): {result_data.d_s_plateau:.3f} ± {result_data.d_s_plateau_std:.3f}
                 (t ∈ [{result_data.plateau_t_min:.1f}, {result_data.plateau_t_max:.1f}])

RANDOMS (null):
  N (used): {result_rand.n_used}
  d_s (plateau): {result_rand.d_s_plateau:.3f} ± {result_rand.d_s_plateau_std:.3f}

DIFFERENTIAL:
  Δd_s (plateau avg): {result_data.d_s_plateau - result_rand.d_s_plateau:.3f}
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Convergence figure saved to {output_path}")


# =============================================================================
# DESI Catalog Loaders
# =============================================================================

def load_desi_lrg_catalog(filepath: str, tracer: str = 'LRG') -> Catalog:
    """
    Load DESI DR1 LSS galaxy catalog.

    Expected columns (DESI DR1 LRG format):
    - RA, DEC: sky coordinates in degrees
    - Z: spectroscopic redshift
    - WEIGHT_COMP, WEIGHT_SYS, WEIGHT_ZFAIL: systematic weights
    - WEIGHT_FKP: FKP weight for optimal clustering estimation

    Total weight: w_tot = WEIGHT_COMP * WEIGHT_SYS * WEIGHT_ZFAIL * WEIGHT_FKP

    Parameters
    ----------
    filepath : str
        Path to FITS file
    tracer : str
        Tracer type for logging

    Returns
    -------
    catalog : Catalog
    """
    if not HAS_FITS:
        raise ImportError("astropy.io.fits required for DESI catalog loading")

    logger.info(f"Loading DESI {tracer} catalog from {filepath}")

    with fits.open(filepath) as hdul:
        data = hdul[1].data

        ra = data['RA']
        dec = data['DEC']
        z = data['Z']

        # Compute total weight
        w_comp = data.get('WEIGHT_COMP', np.ones(len(ra)))
        w_sys = data.get('WEIGHT_SYS', np.ones(len(ra)))
        w_zfail = data.get('WEIGHT_ZFAIL', np.ones(len(ra)))
        w_fkp = data.get('WEIGHT_FKP', np.ones(len(ra)))

        w_tot = w_comp * w_sys * w_zfail * w_fkp

    return Catalog(
        ra_deg=np.asarray(ra, dtype=np.float64),
        dec_deg=np.asarray(dec, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        w=np.asarray(w_tot, dtype=np.float64),
        name=f"DESI_{tracer}"
    )


def load_desi_random_catalog(filepath: str, tracer: str = 'LRG') -> Catalog:
    """
    Load DESI DR1 LSS random catalog.

    Random catalogs have the same angular/radial selection as data
    but uniformly sample the selection function (no clustering).

    Expected columns:
    - RA, DEC: sky coordinates
    - Z: redshift (sampled from n(z))
    - WEIGHT_FKP: FKP weight (optional)

    Parameters
    ----------
    filepath : str
        Path to FITS file
    tracer : str
        Tracer type for logging

    Returns
    -------
    catalog : Catalog
    """
    if not HAS_FITS:
        raise ImportError("astropy.io.fits required for DESI catalog loading")

    logger.info(f"Loading DESI {tracer} random catalog from {filepath}")

    with fits.open(filepath) as hdul:
        data = hdul[1].data

        ra = data['RA']
        dec = data['DEC']
        z = data['Z']

        # Randoms typically have uniform weight or just FKP
        w_fkp = data.get('WEIGHT_FKP', np.ones(len(ra)))

    return Catalog(
        ra_deg=np.asarray(ra, dtype=np.float64),
        dec_deg=np.asarray(dec, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        w=np.asarray(w_fkp, dtype=np.float64),
        name=f"DESI_{tracer}_randoms"
    )


def load_catalog_from_arrays(ra: np.ndarray, dec: np.ndarray, z: np.ndarray,
                             weights: Optional[np.ndarray] = None,
                             name: str = "custom") -> Catalog:
    """
    Create Catalog from numpy arrays (for testing or custom data).
    """
    if weights is None:
        weights = np.ones(len(ra))

    return Catalog(
        ra_deg=np.asarray(ra, dtype=np.float64),
        dec_deg=np.asarray(dec, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        w=np.asarray(weights, dtype=np.float64),
        name=name
    )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for spectral dimension analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='DESI Spectral Dimension Analysis with Randoms-Based Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on DESI LRG catalogs
  python desi_return_probability.py \\
      --data LRG_clustering.dat.fits \\
      --rand LRG_clustering.ran.fits \\
      --zmin 0.4 --zmax 0.6 \\
      --eps 50 \\
      --output results_z0.4-0.6.json

  # Run comparison (data vs randoms null test)
  python desi_return_probability.py \\
      --data LRG_clustering.dat.fits \\
      --rand LRG_clustering.ran.fits \\
      --zmin 0.4 --zmax 0.6 \\
      --compare \\
      --output comparison_z0.4-0.6.json
"""
    )

    parser.add_argument('--data', type=str, required=True,
                        help='Path to data catalog FITS file')
    parser.add_argument('--rand', type=str, required=True,
                        help='Path to random catalog FITS file')
    parser.add_argument('--zmin', type=float, required=True,
                        help='Minimum redshift')
    parser.add_argument('--zmax', type=float, required=True,
                        help='Maximum redshift')
    parser.add_argument('--eps', type=float, default=50.0,
                        help='Epsilon-ball radius in Mpc/h (default: 50)')
    parser.add_argument('--boundary-m', type=int, default=32,
                        help='Number of random neighbors for boundary score (default: 32)')
    parser.add_argument('--boundary-q', type=float, default=0.90,
                        help='Boundary excision quantile (default: 0.90)')
    parser.add_argument('--R', type=int, default=64,
                        help='Number of Hutchinson probe vectors (default: 64)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison: data vs randoms null test')
    parser.add_argument('--output', type=str, default='ds_results.json',
                        help='Output JSON file (default: ds_results.json)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output PDF for convergence plot (requires --compare)')
    parser.add_argument('--tracer', type=str, default='LRG',
                        help='Tracer type (default: LRG)')

    args = parser.parse_args()

    # Load catalogs
    data = load_desi_lrg_catalog(args.data, tracer=args.tracer)
    rand = load_desi_random_catalog(args.rand, tracer=args.tracer)

    # Set up cosmology
    cosmo = None
    if HAS_ASTROPY:
        cosmo = FlatLambdaCDM(H0=67.66, Om0=0.3111, Ob0=0.049)

    if args.compare:
        # Run comparison
        results = compare_data_randoms(
            data, rand,
            zmin=args.zmin, zmax=args.zmax,
            eps_mpc=args.eps,
            boundary_m=args.boundary_m,
            boundary_quantile=args.boundary_q,
            R=args.R,
            seed=args.seed,
            cosmo=cosmo
        )

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Comparison results saved to {args.output}")

        # Generate plot if requested
        if args.plot:
            # Reconstruct result objects for plotting
            result_data = SpectralDimensionResult(
                n_raw=results['data']['n_raw'],
                n_used=results['data']['n_used'],
                n_edges=results['data']['n_edges'],
                mean_degree=results['data']['mean_degree'],
                boundary_score_threshold=results['data']['boundary_score_threshold'],
                eps_mpc=results['data']['eps_mpc'],
                zmin=results['data']['zmin'],
                zmax=results['data']['zmax'],
                t_grid=np.array(results['data']['t_grid']),
                P_t=np.array(results['data']['P_t']),
                d_s_t=np.array(results['data']['d_s_t']),
                plateau_t_min=results['data']['plateau_t_min'],
                plateau_t_max=results['data']['plateau_t_max'],
                d_s_plateau=results['data']['d_s_plateau'],
                d_s_plateau_std=results['data']['d_s_plateau_std']
            )
            result_rand = SpectralDimensionResult(
                n_raw=results['rand']['n_raw'],
                n_used=results['rand']['n_used'],
                n_edges=results['rand']['n_edges'],
                mean_degree=results['rand']['mean_degree'],
                boundary_score_threshold=results['rand']['boundary_score_threshold'],
                eps_mpc=results['rand']['eps_mpc'],
                zmin=results['rand']['zmin'],
                zmax=results['rand']['zmax'],
                t_grid=np.array(results['rand']['t_grid']),
                P_t=np.array(results['rand']['P_t']),
                d_s_t=np.array(results['rand']['d_s_t']),
                plateau_t_min=results['rand']['plateau_t_min'],
                plateau_t_max=results['rand']['plateau_t_max'],
                d_s_plateau=results['rand']['d_s_plateau'],
                d_s_plateau_std=results['rand']['d_s_plateau_std']
            )
            plot_convergence_diagnostic(result_data, result_rand, args.plot)
    else:
        # Run single analysis
        result = run_ds_pipeline(
            data, rand,
            zmin=args.zmin, zmax=args.zmax,
            eps_mpc=args.eps,
            boundary_m=args.boundary_m,
            boundary_quantile=args.boundary_q,
            R=args.R,
            seed=args.seed,
            cosmo=cosmo
        )

        result.save_json(args.output)


if __name__ == '__main__':
    main()
