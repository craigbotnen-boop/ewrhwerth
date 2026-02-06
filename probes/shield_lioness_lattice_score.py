#!/usr/bin/env python3
"""
Shield LIONESS Lattice Score Engine
====================================
Universal Dual-Metric Engine for patient-specific lattice score computation.

This script computes two candidate survival-governing metrics per patient
using the LIONESS (Linear Interpolation to Obtain Network Estimates for
Single Samples) framework:

  Metric A: Heat-kernel trace at t=1.0 (lattice_score_t1)
  Metric B: Spectral dimension (spectral_dimension)

The "winner" is the metric that reproduces historical variance patterns.

Usage:
    1. Set TEST_MODE = True for synthetic verification
    2. Set TEST_MODE = False and configure EXPR_GZ / CLIN_TSV for real data
    3. Run: python shield_lioness_lattice_score.py

Output:
    lattice_scores_universal_N{count}.csv
"""

import os
import sys
import time
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST_MODE = True  # Set False for real TCGA data

EXPR_GZ = "tcga_RSEM_gene_tpm.gz"       # Path to expression matrix (gzipped)
CLIN_TSV = "tcga_gbm_survival_clin_pool.tsv"  # Path to clinical labels

# Heat-kernel evaluation point for Metric A
T_EVAL = 1.0

# Number of top-variance genes to retain for network construction
N_TOP_GENES = 500

# LIONESS: correlation-based network estimation
LIONESS_EDGE_METHOD = "pearson"

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def compute_heat_kernel_trace(laplacian_eigenvalues, t):
    """
    Compute the heat-kernel trace: Tr(e^{-tL}) = sum_i exp(-t * lambda_i).

    Parameters
    ----------
    laplacian_eigenvalues : array-like
        Eigenvalues of the graph Laplacian.
    t : float
        Diffusion time.

    Returns
    -------
    float
        Heat-kernel trace value.
    """
    return np.sum(np.exp(-t * np.asarray(laplacian_eigenvalues)))


def compute_spectral_dimension(laplacian_eigenvalues, t_range=(0.1, 10.0),
                                n_points=50):
    """
    Estimate spectral dimension from heat-kernel trace decay.

    d_s = -2 * d(log Z(t)) / d(log t)

    where Z(t) = Tr(e^{-tL}).

    Parameters
    ----------
    laplacian_eigenvalues : array-like
        Eigenvalues of the graph Laplacian.
    t_range : tuple
        (t_min, t_max) for log-spaced evaluation.
    n_points : int
        Number of evaluation points.

    Returns
    -------
    float
        Estimated spectral dimension (median of local slopes).
    """
    eigs = np.asarray(laplacian_eigenvalues)
    ts = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n_points)
    log_Z = np.array([np.log(np.sum(np.exp(-t * eigs)) + 1e-300) for t in ts])
    log_t = np.log(ts)

    # Local slopes via finite differences
    slopes = np.diff(log_Z) / np.diff(log_t)
    d_s = -2.0 * np.median(slopes)
    return d_s


def correlation_matrix(X):
    """
    Pearson correlation matrix for genes, given samples-by-genes matrix X.
    Returns a (genes x genes) correlation matrix.
    """
    # Transpose so we correlate genes across samples
    X_t = X.T  # genes x samples
    X_centered = X_t - X_t.mean(axis=1, keepdims=True)
    norms = np.sqrt((X_centered ** 2).sum(axis=1, keepdims=True))
    norms = np.where(norms == 0, 1.0, norms)
    X_normed = X_centered / norms
    return X_normed @ X_normed.T


def graph_laplacian_eigenvalues(corr_matrix, threshold=0.0):
    """
    Compute normalized Laplacian eigenvalues from a correlation matrix.

    Parameters
    ----------
    corr_matrix : ndarray
        Symmetric correlation matrix (N x N).
    threshold : float
        Minimum absolute correlation to form an edge.

    Returns
    -------
    ndarray
        Sorted eigenvalues of the normalized Laplacian.
    """
    adj = np.abs(corr_matrix)
    np.fill_diagonal(adj, 0.0)
    adj[adj < threshold] = 0.0

    degree = adj.sum(axis=1)
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)

    L_norm = np.eye(len(adj)) - D_inv_sqrt @ adj @ D_inv_sqrt
    eigenvalues = np.linalg.eigvalsh(L_norm)
    eigenvalues = np.clip(eigenvalues, 0, None)  # Numerical stability
    return np.sort(eigenvalues)


def lioness_single_sample(X_cohort, sample_idx):
    """
    LIONESS: estimate the network for a single sample by linear
    interpolation between the cohort network and the leave-one-out network.

    L_i = N * C_all - (N-1) * C_{-i}

    Parameters
    ----------
    X_cohort : ndarray
        Full cohort expression matrix (samples x genes).
    sample_idx : int
        Index of the sample to estimate.

    Returns
    -------
    ndarray
        Patient-specific correlation matrix.
    """
    N = X_cohort.shape[0]
    C_all = correlation_matrix(X_cohort)

    mask = np.ones(N, dtype=bool)
    mask[sample_idx] = False
    C_loo = correlation_matrix(X_cohort[mask])

    # LIONESS interpolation
    C_patient = N * C_all - (N - 1) * C_loo
    return C_patient


# ---------------------------------------------------------------------------
# Synthetic Verification
# ---------------------------------------------------------------------------

def run_synthetic_test():
    """
    Verify the engine on a synthetic cohort with known properties.
    """
    print("running SYNTHETIC verification...")
    np.random.seed(42)

    n_samples = 20
    n_genes = 50
    X_synth = np.random.randn(n_samples, n_genes)

    # Add structured signal to first 10 genes
    signal = np.random.randn(n_samples, 1)
    X_synth[:, :10] += 2.0 * signal

    results = []
    for i in range(n_samples):
        C_patient = lioness_single_sample(X_synth, i)

        # Use the gene-gene correlation submatrix
        # For synthetic test, compute gene-gene correlations from patient network
        eigs = graph_laplacian_eigenvalues(C_patient, threshold=0.1)

        trace_t1 = compute_heat_kernel_trace(eigs, T_EVAL)
        d_s = compute_spectral_dimension(eigs)

        results.append({
            "sample_idx": i,
            "lattice_score_t1": trace_t1,
            "spectral_dimension": d_s,
        })

    # Validate
    traces = np.array([r["lattice_score_t1"] for r in results])
    dims = np.array([r["spectral_dimension"] for r in results])

    print(f"  Trace at t=1.0:  mean={traces.mean():.4f}  std={traces.std():.4f}")
    print(f"  Spectral dim:    mean={dims.mean():.4f}  std={dims.std():.4f}")

    if traces.std() > 0 and dims.std() > 0:
        print("[PASS] Metrics computed successfully.")
    else:
        print("[FAIL] Zero variance detected in one or both metrics.")
        sys.exit(1)

    return results


# ---------------------------------------------------------------------------
# Full TCGA Run
# ---------------------------------------------------------------------------

def run_full_cohort():
    """
    Run LIONESS on the full TCGA-GBM cohort.
    Requires: EXPR_GZ (gzipped expression matrix) and CLIN_TSV (clinical data).
    """
    import gzip
    import csv

    # Load clinical data
    print(f"Loading clinical data from {CLIN_TSV}...")
    if not os.path.exists(CLIN_TSV):
        print(f"ERROR: {CLIN_TSV} not found. Upload it first.")
        sys.exit(1)

    clin_samples = []
    with open(CLIN_TSV, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clin_samples.append(row)
    print(f"  Clinical records: {len(clin_samples)}")

    # Load expression data
    print(f"Loading expression data from {EXPR_GZ}...")
    if not os.path.exists(EXPR_GZ):
        print(f"ERROR: {EXPR_GZ} not found. Upload it first.")
        sys.exit(1)

    open_fn = gzip.open if EXPR_GZ.endswith(".gz") else open
    with open_fn(EXPR_GZ, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        sample_ids = header[1:]  # First column is gene name
        gene_names = []
        expr_data = []
        for row in reader:
            gene_names.append(row[0])
            expr_data.append([float(x) for x in row[1:]])

    expr_matrix = np.array(expr_data).T  # samples x genes
    print(f"  Expression matrix: {expr_matrix.shape[0]} samples x {expr_matrix.shape[1]} genes")

    # Select top-variance genes
    gene_var = expr_matrix.var(axis=0)
    top_idx = np.argsort(gene_var)[-N_TOP_GENES:]
    X = expr_matrix[:, top_idx]
    print(f"  Using top {N_TOP_GENES} genes by variance")

    # Run LIONESS per patient
    n_patients = X.shape[0]
    print(f"\nRunning LIONESS for {n_patients} patients...")
    results = []
    t_start = time.time()

    for i in range(n_patients):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_patients - i - 1) / rate if rate > 0 else 0
            print(f"  Patient {i+1}/{n_patients}  ({rate:.1f} pts/sec, ETA {eta:.0f}s)")

        C_patient = lioness_single_sample(X, i)
        eigs = graph_laplacian_eigenvalues(C_patient, threshold=0.1)

        trace_t1 = compute_heat_kernel_trace(eigs, T_EVAL)
        d_s = compute_spectral_dimension(eigs)

        results.append({
            "sample_id": sample_ids[i] if i < len(sample_ids) else f"sample_{i}",
            "lattice_score_t1": trace_t1,
            "spectral_dimension": d_s,
        })

    # Write output CSV
    out_file = f"lattice_scores_universal_N{n_patients}.csv"
    with open(out_file, "w") as f:
        f.write("sample_id,lattice_score_t1,spectral_dimension\n")
        for r in results:
            f.write(f"{r['sample_id']},{r['lattice_score_t1']:.6f},{r['spectral_dimension']:.6f}\n")

    traces = np.array([r["lattice_score_t1"] for r in results])
    dims = np.array([r["spectral_dimension"] for r in results])

    print(f"\n== RESULTS ==")
    print(f"  Output: {out_file}")
    print(f"  Trace at t=1.0:  mean={traces.mean():.4f}  std={traces.std():.4f}")
    print(f"  Spectral dim:    mean={dims.mean():.4f}  std={dims.std():.4f}")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if TEST_MODE:
        run_synthetic_test()
    else:
        run_full_cohort()
