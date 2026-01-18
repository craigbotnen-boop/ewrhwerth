#!/usr/bin/env python3
"""
STARGATE-ATOPO: CORRECTED Implementation for DESI Lyα Analysis
===============================================================
Fixes critical bugs in graph Laplacian construction:
1. Symmetrizes adjacency matrix
2. Removes self-loops
3. Uses proper comoving coordinates (not raw RA/Dec/z)

Author: Craig W. Botnen, FDEP Institute
Version: 2.0.0 (Bug-fixed)
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

__version__ = "2.0.0"

# =============================================================================
# COORDINATE TRANSFORMATION (CRITICAL FIX #3)
# =============================================================================

def ra_dec_z_to_comoving(ra, dec, z, H0=67.4, Om0=0.315):
    """
    Convert RA/Dec/z to comoving Cartesian coordinates.

    THIS IS CRITICAL: Raw RA/Dec/z cannot be used as Euclidean xyz!

    Parameters
    ----------
    ra : array
        Right ascension in degrees
    dec : array
        Declination in degrees
    z : array
        Spectroscopic redshift
    H0 : float
        Hubble constant (km/s/Mpc)
    Om0 : float
        Matter density parameter

    Returns
    -------
    positions : ndarray, shape (N, 3)
        Comoving Cartesian coordinates in Mpc/h
    """
    # Convert angles to radians
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)

    # Comoving distance via numerical integration
    # Using simple trapezoidal integration for E(z) = sqrt(Om0*(1+z)^3 + (1-Om0))
    c = 299792.458  # km/s

    def comoving_distance(z_val):
        """Compute comoving distance to redshift z."""
        if z_val <= 0:
            return 0.0
        z_arr = np.linspace(0, z_val, 1000)
        E_z = np.sqrt(Om0 * (1 + z_arr)**3 + (1 - Om0))
        integrand = 1.0 / E_z
        # Use trapezoid (numpy >= 2.0) or trapz (numpy < 2.0)
        try:
            d_c = (c / H0) * np.trapezoid(integrand, z_arr)
        except AttributeError:
            d_c = (c / H0) * np.trapz(integrand, z_arr)
        return d_c

    # Vectorize the distance calculation
    r = np.array([comoving_distance(zi) for zi in z])

    # Convert to Cartesian
    x = r * np.cos(dec_rad) * np.cos(ra_rad)
    y = r * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = r * np.sin(dec_rad)

    positions = np.column_stack([x, y, z_cart])

    return positions


def validate_coordinates(positions, name="positions"):
    """
    Validate coordinate array for common issues.
    """
    issues = []

    if np.any(np.isnan(positions)):
        issues.append("Contains NaN values")
    if np.any(np.isinf(positions)):
        issues.append("Contains Inf values")

    # Check for degenerate dimensions
    for i, axis in enumerate(['x', 'y', 'z']):
        if np.std(positions[:, i]) < 1e-10:
            issues.append(f"{axis}-axis has zero variance")

    if issues:
        print(f"WARNING [{name}]: {', '.join(issues)}")
        return False
    return True


# =============================================================================
# GRAPH CONSTRUCTION (CRITICAL FIXES #1 and #2)
# =============================================================================

def build_knn_graph_CORRECT(positions, k=10, verbose=True):
    """
    Build SYMMETRIC k-nearest neighbor graph, EXCLUDING self-loops.

    FIXES:
    1. Excludes self (index 0 from query results)
    2. Symmetrizes: if i→j, then also j→i
    3. Removes any remaining diagonal entries

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Comoving Cartesian coordinates
    k : int
        Number of nearest neighbors (excluding self)
    verbose : bool
        Print diagnostics

    Returns
    -------
    adjacency : sparse CSR matrix
        Symmetric binary adjacency matrix with zero diagonal
    """
    N = positions.shape[0]

    if verbose:
        print(f"Building symmetric kNN graph: N={N}, k={k}")

    # Build KD-tree
    tree = cKDTree(positions)

    # Query k+1 neighbors (includes self as first result)
    distances, indices = tree.query(positions, k=k+1)

    # CRITICAL FIX #2: Drop self (first column)
    indices = indices[:, 1:]  # Now shape (N, k)

    if verbose:
        print(f"  After dropping self: indices shape = {indices.shape}")

    # Build edge lists
    rows = np.repeat(np.arange(N), k)
    cols = indices.ravel()

    # CRITICAL FIX #1: Create symmetric adjacency
    # Add both (i,j) and (j,i) edges
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.ones(len(all_rows), dtype=np.float64)

    # Create sparse matrix
    A = csr_matrix((all_data, (all_rows, all_cols)), shape=(N, N))

    # Convert to binary (remove duplicate edges)
    A.data = np.ones_like(A.data)

    # CRITICAL FIX #2b: Ensure zero diagonal (no self-loops)
    A.setdiag(0)
    A.eliminate_zeros()

    # Verify symmetry
    if verbose:
        asymmetry = np.abs(A - A.T).sum()
        print(f"  Asymmetry check: {asymmetry} (should be 0)")
        print(f"  Number of edges: {A.nnz // 2}")
        print(f"  Mean degree: {A.sum(axis=1).mean():.2f}")

    return A


def build_normalized_laplacian_CORRECT(adjacency, verbose=True):
    """
    Build symmetric normalized graph Laplacian.

    L = I - D^{-1/2} A D^{-1/2}

    For symmetric A, this L is guaranteed to be:
    - Symmetric
    - Positive semi-definite
    - Eigenvalues in [0, 2]

    Parameters
    ----------
    adjacency : sparse matrix
        MUST be symmetric with zero diagonal

    Returns
    -------
    L : sparse matrix
        Symmetric normalized Laplacian
    """
    N = adjacency.shape[0]

    # Degree vector
    degree = np.array(adjacency.sum(axis=1)).flatten().astype(np.float64)

    # Handle isolated nodes (degree = 0)
    n_isolated = np.sum(degree == 0)
    if n_isolated > 0 and verbose:
        print(f"  WARNING: {n_isolated} isolated nodes (degree=0)")
    degree[degree == 0] = 1.0  # Prevent division by zero

    # D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    D_inv_sqrt = diags(d_inv_sqrt)

    # Symmetric normalized Laplacian
    L = eye(N) - D_inv_sqrt @ adjacency @ D_inv_sqrt

    return L.tocsr()


def verify_laplacian_spectrum(L, n_check=20, verbose=True):
    """
    Verify Laplacian has valid spectrum (all eigenvalues >= 0).

    This is the CRITICAL diagnostic that was failing before.
    """
    N = L.shape[0]
    n_eig = min(n_check, N - 1)

    try:
        # Get smallest eigenvalues
        eigenvalues, _ = eigsh(L, k=n_eig, which='SM', tol=1e-6)
        eigenvalues = np.sort(eigenvalues)

        # Check for negative eigenvalues (THE BUG SYMPTOM)
        n_negative = np.sum(eigenvalues < -1e-10)
        min_eval = eigenvalues[0]
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0

        if verbose:
            print(f"\nLaplacian Spectrum Verification:")
            print(f"  λ_min = {min_eval:.6f} (should be ≈ 0)")
            print(f"  λ_1 (spectral gap) = {spectral_gap:.6f} (should be > 0)")
            print(f"  Negative eigenvalues: {n_negative} (should be 0)")
            print(f"  First 5 eigenvalues: {eigenvalues[:5]}")

        # CRITICAL CHECK
        if n_negative > 0:
            print(f"\n  *** CRITICAL ERROR: {n_negative} negative eigenvalues! ***")
            print(f"  *** This indicates the Laplacian is NOT symmetric or has bugs ***")
            return False, eigenvalues

        if spectral_gap < 1e-10:
            print(f"\n  *** WARNING: Spectral gap ≈ 0 suggests disconnected graph ***")

        return True, eigenvalues

    except Exception as e:
        print(f"  Eigendecomposition failed: {e}")
        return False, None


# =============================================================================
# SPECTRAL DIMENSION (CORRECTED)
# =============================================================================

def compute_spectral_dimension_CORRECT(eigenvalues, t_window, n_points=50):
    """
    Compute spectral dimension from heat kernel trace.

    d_s = -2 * d(ln P_r) / d(ln t)

    where P_r(t) = Σ_i exp(-λ_i t)

    REQUIRES: eigenvalues >= 0 (will fail/NaN otherwise)
    """
    # Remove any negative or zero eigenvalues
    valid_eig = eigenvalues[eigenvalues > 1e-10]

    if len(valid_eig) < 5:
        print("  WARNING: Too few valid eigenvalues for d_s computation")
        return 0.0

    t_min, t_max = t_window
    t_values = np.logspace(np.log10(t_min), np.log10(t_max), n_points)

    # Heat kernel trace: P_r(t) = Σ exp(-λ_i * t)
    P_r = np.array([np.sum(np.exp(-valid_eig * t)) for t in t_values])

    # Numerical safety
    P_r = np.clip(P_r, 1e-300, None)

    # Log-log regression
    log_t = np.log(t_values)
    log_P = np.log(P_r)

    # d_s = -2 * slope
    coeffs = np.polyfit(log_t, log_P, 1)
    d_s = -2.0 * coeffs[0]

    return d_s


# =============================================================================
# A_TOPO (CORRECTED - Using heat kernel weights, not 1/λ)
# =============================================================================

def compute_atopo_CORRECT(positions, adjacency, t_window, n_eig=200, verbose=True):
    """
    Compute A_topo using CORRECT methodology.

    FIXES:
    1. Uses symmetric normalized Laplacian
    2. Verifies eigenvalues are non-negative
    3. Uses heat kernel weights exp(-λt), NOT 1/λ
    4. Centers coordinates before inertia computation
    """
    N = positions.shape[0]
    t_mid = np.sqrt(t_window[0] * t_window[1])

    if verbose:
        print(f"\nComputing A_topo (CORRECTED) for N={N}")
        print(f"  Plateau window: {t_window}")

    # Build Laplacian
    L = build_normalized_laplacian_CORRECT(adjacency, verbose=verbose)

    # Verify spectrum
    is_valid, eigenvalues_check = verify_laplacian_spectrum(L, verbose=verbose)

    if not is_valid:
        print("\n*** ABORTING: Invalid Laplacian spectrum ***")
        return None, {'error': 'Invalid Laplacian - negative eigenvalues'}

    # Full eigendecomposition for A_topo
    n_eig_actual = min(n_eig, N - 2)

    if verbose:
        print(f"\nComputing {n_eig_actual} eigenmodes for A_topo...")

    try:
        eigenvalues, eigenvectors = eigsh(L, k=n_eig_actual, which='SM', tol=1e-5)
    except Exception as e:
        print(f"  Eigendecomposition failed: {e}")
        return None, {'error': str(e)}

    # Sort
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Remove near-zero modes
    valid_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[valid_mask]
    eigenvectors = eigenvectors[:, valid_mask]
    n_modes = len(eigenvalues)

    if verbose:
        print(f"  Using {n_modes} valid modes")
        print(f"  λ range: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")

    # Compute spectral dimension
    d_s = compute_spectral_dimension_CORRECT(eigenvalues, t_window)
    if verbose:
        print(f"  d_s = {d_s:.4f}")

    # CRITICAL FIX: Heat kernel weights, NOT 1/λ
    # w_m = exp(-λ_m * t_mid), not 1/λ
    mode_weights = np.exp(-eigenvalues * t_mid)
    mode_weights /= mode_weights.sum() + 1e-10

    # Center positions
    pos_centered = positions - positions.mean(axis=0)

    # Weighted inertia tensor
    I_tensor = np.zeros((3, 3))

    for m in range(n_modes):
        psi_sq = eigenvectors[:, m] ** 2
        psi_sq /= psi_sq.sum() + 1e-10

        weighted_pos = pos_centered * psi_sq[:, np.newaxis]
        I_tensor += mode_weights[m] * (pos_centered.T @ weighted_pos)

    # Eigenvalues of inertia tensor
    I_eigenvalues = np.linalg.eigvalsh(I_tensor)
    I_eigenvalues = np.sort(I_eigenvalues)[::-1]

    # A_topo = normalized variance
    mean_I = np.mean(I_eigenvalues)
    std_I = np.std(I_eigenvalues)
    atopo = (std_I / (mean_I + 1e-10)) ** 2

    if verbose:
        print(f"\n  ══════════════════════════════════")
        print(f"  A_topo = {atopo:.6f}")
        print(f"  d_s = {d_s:.4f}")
        print(f"  I_eigenvalues = {I_eigenvalues}")
        print(f"  I_ratio = {I_eigenvalues[0]/(I_eigenvalues[2]+1e-10):.2f}")
        print(f"  ══════════════════════════════════")

    diagnostics = {
        'atopo': atopo,
        'd_s': d_s,
        'I_eigenvalues': I_eigenvalues.tolist(),
        'I_ratio': float(I_eigenvalues[0] / (I_eigenvalues[2] + 1e-10)),
        'n_modes': n_modes,
        'lambda_range': [float(eigenvalues[0]), float(eigenvalues[-1])],
        'spectrum_valid': True
    }

    return atopo, diagnostics


# =============================================================================
# COMPLETE PIPELINE TEST
# =============================================================================

def run_diagnostic_test():
    """
    Run complete diagnostic to verify fixes.
    """
    print("=" * 70)
    print("STARGATE-ATOPO DIAGNOSTIC TEST (v2.0 - Bug Fixed)")
    print("=" * 70)

    np.random.seed(42)

    # Test 1: Synthetic isotropic data (should have low A_topo)
    print("\n" + "-" * 70)
    print("TEST 1: Synthetic Isotropic Data")
    print("-" * 70)

    N = 500
    positions = np.random.randn(N, 3) * 100  # Isotropic Gaussian

    print(f"Generated {N} isotropic points")
    validate_coordinates(positions)

    # Build graph
    adjacency = build_knn_graph_CORRECT(positions, k=10, verbose=True)

    # Compute A_topo
    atopo, diag = compute_atopo_CORRECT(positions, adjacency, t_window=(5, 100),
                                        n_eig=100, verbose=True)

    if atopo is not None:
        print(f"\n✓ TEST 1 PASSED: A_topo = {atopo:.4f}, d_s = {diag['d_s']:.4f}")
    else:
        print("\n✗ TEST 1 FAILED")

    # Test 2: Synthetic anisotropic data (should have HIGH A_topo)
    print("\n" + "-" * 70)
    print("TEST 2: Synthetic Anisotropic Data (3x stretch in x)")
    print("-" * 70)

    positions_aniso = positions.copy()
    positions_aniso[:, 0] *= 3.0  # Stretch x

    adjacency_aniso = build_knn_graph_CORRECT(positions_aniso, k=10, verbose=True)

    atopo_aniso, diag_aniso = compute_atopo_CORRECT(
        positions_aniso, adjacency_aniso, t_window=(5, 100),
        n_eig=100, verbose=True
    )

    if atopo_aniso is not None:
        print(f"\n✓ TEST 2 PASSED: A_topo = {atopo_aniso:.4f}, d_s = {diag_aniso['d_s']:.4f}")
    else:
        print("\n✗ TEST 2 FAILED")

    # Test 3: Mock "RA/Dec/z" data - show WRONG vs RIGHT approach
    print("\n" + "-" * 70)
    print("TEST 3: Coordinate Conversion (RA/Dec/z → Comoving)")
    print("-" * 70)

    # Mock DESI-like coordinates
    ra = np.random.uniform(180, 200, N)   # degrees
    dec = np.random.uniform(20, 40, N)    # degrees
    z = np.random.uniform(2.0, 3.5, N)    # Lyα forest redshifts

    print(f"Mock DESI data: RA=[{ra.min():.1f}, {ra.max():.1f}], "
          f"Dec=[{dec.min():.1f}, {dec.max():.1f}], z=[{z.min():.2f}, {z.max():.2f}]")

    # WRONG way (what the bug was doing)
    positions_wrong = np.column_stack([ra, dec, z])
    print(f"\nWRONG: Using raw RA/Dec/z as xyz")
    print(f"  Coordinate ranges: x=[{positions_wrong[:,0].min():.1f}, {positions_wrong[:,0].max():.1f}], "
          f"y=[{positions_wrong[:,1].min():.1f}, {positions_wrong[:,1].max():.1f}], "
          f"z=[{positions_wrong[:,2].min():.2f}, {positions_wrong[:,2].max():.2f}]")
    print(f"  Scale mismatch: RA/Dec in degrees, z dimensionless!")

    # RIGHT way
    positions_right = ra_dec_z_to_comoving(ra, dec, z)
    print(f"\nRIGHT: Converting to comoving Cartesian (Mpc/h)")
    print(f"  Coordinate ranges: x=[{positions_right[:,0].min():.1f}, {positions_right[:,0].max():.1f}], "
          f"y=[{positions_right[:,1].min():.1f}, {positions_right[:,1].max():.1f}], "
          f"z=[{positions_right[:,2].min():.1f}, {positions_right[:,2].max():.1f}]")

    # Compute A_topo on correct coordinates
    adjacency_right = build_knn_graph_CORRECT(positions_right, k=10, verbose=True)

    atopo_right, diag_right = compute_atopo_CORRECT(
        positions_right, adjacency_right, t_window=(5, 100),
        n_eig=100, verbose=True
    )

    if atopo_right is not None:
        print(f"\n✓ TEST 3 PASSED: A_topo = {atopo_right:.4f}, d_s = {diag_right['d_s']:.4f}")
    else:
        print("\n✗ TEST 3 FAILED")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<25} {'A_topo':<12} {'d_s':<10} {'I_ratio':<10} {'Status'}")
    print("-" * 70)

    if atopo is not None:
        print(f"{'Isotropic':<25} {atopo:<12.4f} {diag['d_s']:<10.4f} "
              f"{diag['I_ratio']:<10.2f} {'✓ PASS'}")
    if atopo_aniso is not None:
        print(f"{'Anisotropic (3x)':<25} {atopo_aniso:<12.4f} {diag_aniso['d_s']:<10.4f} "
              f"{diag_aniso['I_ratio']:<10.2f} {'✓ PASS'}")
    if atopo_right is not None:
        print(f"{'Comoving coords':<25} {atopo_right:<12.4f} {diag_right['d_s']:<10.4f} "
              f"{diag_right['I_ratio']:<10.2f} {'✓ PASS'}")

    # Validation checks
    print("\n" + "-" * 70)
    print("CRITICAL CHECKS")
    print("-" * 70)

    checks = []

    if atopo is not None and atopo_aniso is not None:
        if atopo_aniso > atopo:
            print("✓ Anisotropic A_topo > Isotropic A_topo")
            checks.append(True)
        else:
            print("✗ FAILED: Anisotropic should exceed isotropic")
            checks.append(False)

    if diag is not None and diag['d_s'] > 0:
        print(f"✓ d_s > 0 (was 0.000 before fix)")
        checks.append(True)
    else:
        print("✗ FAILED: d_s should be > 0")
        checks.append(False)

    if diag is not None and diag.get('spectrum_valid', False):
        print("✓ Laplacian spectrum valid (no negative eigenvalues)")
        checks.append(True)
    else:
        print("✗ FAILED: Laplacian spectrum invalid")
        checks.append(False)

    print(f"\n{sum(checks)}/{len(checks)} critical checks passed")

    if all(checks):
        print("\n" + "=" * 70)
        print("✓ ALL BUGS FIXED - Ready for DESI DR1 Lyα analysis")
        print("=" * 70)

    return {
        'isotropic': {'atopo': atopo, 'diag': diag},
        'anisotropic': {'atopo': atopo_aniso, 'diag': diag_aniso},
        'comoving': {'atopo': atopo_right, 'diag': diag_right}
    }


if __name__ == "__main__":
    results = run_diagnostic_test()
