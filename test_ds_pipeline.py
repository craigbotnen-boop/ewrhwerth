#!/usr/bin/env python
"""
Test/Demo script for DESI Return Probability Pipeline

Generates synthetic catalogs with known clustering properties to verify
the spectral dimension pipeline works correctly before applying to real DESI data.

Usage:
    python test_ds_pipeline.py

Outputs:
    - test_ds_results.json: Full comparison results
    - test_ds_convergence.pdf: Diagnostic figure for appendix
"""

import numpy as np
import json
import logging

from desi_return_probability import (
    Catalog,
    load_catalog_from_arrays,
    run_ds_pipeline,
    compare_data_randoms,
    plot_convergence_diagnostic,
    SpectralDimensionResult,
    comoving_cartesian
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_catalog_uniform(n: int, ra_range=(150, 210), dec_range=(20, 50),
                                       z_range=(0.3, 0.8), seed=42) -> Catalog:
    """
    Generate a uniform random catalog (no clustering).
    This mimics DESI randoms that sample the selection function.
    """
    rng = np.random.default_rng(seed)

    # Uniform in RA
    ra = rng.uniform(ra_range[0], ra_range[1], n)

    # Uniform in sin(dec) for uniform sky density
    sin_dec_min = np.sin(np.deg2rad(dec_range[0]))
    sin_dec_max = np.sin(np.deg2rad(dec_range[1]))
    sin_dec = rng.uniform(sin_dec_min, sin_dec_max, n)
    dec = np.rad2deg(np.arcsin(sin_dec))

    # Redshift with n(z) ~ z^2 * exp(-z/0.3) shape (realistic galaxy distribution)
    z = rng.gamma(shape=3, scale=0.15, size=n)
    z = np.clip(z, z_range[0], z_range[1])

    # Unit weights
    w = np.ones(n)

    return load_catalog_from_arrays(ra, dec, z, w, name="synthetic_uniform")


def generate_synthetic_catalog_clustered(n: int, ra_range=(150, 210), dec_range=(20, 50),
                                         z_range=(0.3, 0.8), n_clusters=20,
                                         cluster_radius_deg=2.0, seed=43) -> Catalog:
    """
    Generate a clustered catalog by adding Gaussian clusters to a uniform background.

    This mimics galaxy clustering: most points are in dense structures,
    with some uniform field distribution.
    """
    rng = np.random.default_rng(seed)

    # Start with uniform background (30% of points)
    n_bg = int(0.3 * n)
    n_clustered = n - n_bg

    # Background
    ra_bg = rng.uniform(ra_range[0], ra_range[1], n_bg)
    sin_dec_min = np.sin(np.deg2rad(dec_range[0]))
    sin_dec_max = np.sin(np.deg2rad(dec_range[1]))
    sin_dec = rng.uniform(sin_dec_min, sin_dec_max, n_bg)
    dec_bg = np.rad2deg(np.arcsin(sin_dec))
    z_bg = rng.gamma(shape=3, scale=0.15, size=n_bg)
    z_bg = np.clip(z_bg, z_range[0], z_range[1])

    # Cluster centers
    ra_centers = rng.uniform(ra_range[0] + 5, ra_range[1] - 5, n_clusters)
    dec_centers = rng.uniform(dec_range[0] + 5, dec_range[1] - 5, n_clusters)
    z_centers = rng.uniform(z_range[0] + 0.1, z_range[1] - 0.1, n_clusters)

    # Generate clustered points
    points_per_cluster = n_clustered // n_clusters
    ra_cl, dec_cl, z_cl = [], [], []

    for i in range(n_clusters):
        # Angular scatter
        dra = rng.normal(0, cluster_radius_deg, points_per_cluster)
        ddec = rng.normal(0, cluster_radius_deg, points_per_cluster)
        # Redshift scatter (tighter, ~0.01 in z)
        dz = rng.normal(0, 0.01, points_per_cluster)

        ra_cl.extend(ra_centers[i] + dra / np.cos(np.deg2rad(dec_centers[i])))
        dec_cl.extend(dec_centers[i] + ddec)
        z_cl.extend(z_centers[i] + dz)

    ra_cl = np.array(ra_cl)
    dec_cl = np.array(dec_cl)
    z_cl = np.array(z_cl)

    # Clip to bounds
    ra_cl = np.clip(ra_cl, ra_range[0], ra_range[1])
    dec_cl = np.clip(dec_cl, dec_range[0], dec_range[1])
    z_cl = np.clip(z_cl, z_range[0], z_range[1])

    # Combine
    ra = np.concatenate([ra_bg, ra_cl])
    dec = np.concatenate([dec_bg, dec_cl])
    z = np.concatenate([z_bg, z_cl])
    w = np.ones(len(ra))

    return load_catalog_from_arrays(ra, dec, z, w, name="synthetic_clustered")


def run_test_pipeline():
    """Run full test pipeline with synthetic data."""
    logger.info("=" * 70)
    logger.info("SYNTHETIC DATA TEST FOR DESI RETURN PROBABILITY PIPELINE")
    logger.info("=" * 70)

    # Generate synthetic catalogs
    n_data = 10000
    n_rand = 50000

    logger.info(f"\nGenerating synthetic catalogs: {n_data} data, {n_rand} randoms")

    # Clustered data (mimics galaxy survey)
    data = generate_synthetic_catalog_clustered(n_data, seed=123)

    # Uniform randoms (mimics selection function)
    rand = generate_synthetic_catalog_uniform(n_rand, seed=456)

    # Analysis parameters
    zmin, zmax = 0.4, 0.6
    eps_mpc = 40.0  # Slightly smaller epsilon for denser synthetic data

    logger.info(f"\nAnalysis parameters:")
    logger.info(f"  z-slice: [{zmin}, {zmax})")
    logger.info(f"  eps: {eps_mpc} Mpc/h")

    # Run comparison (data vs randoms null test)
    results = compare_data_randoms(
        data, rand,
        zmin=zmin, zmax=zmax,
        eps_mpc=eps_mpc,
        boundary_m=32,
        boundary_quantile=0.90,
        R=64,
        seed=0
    )

    # Save JSON results
    output_json = "test_ds_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_json}")

    # Generate convergence diagnostic figure
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

    output_pdf = "test_ds_convergence.pdf"
    plot_convergence_diagnostic(result_data, result_rand, output_pdf)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Data d_s plateau: {results['data']['d_s_plateau']:.3f} +/- {results['data']['d_s_plateau_std']:.3f}")
    logger.info(f"Rand d_s plateau: {results['rand']['d_s_plateau']:.3f} +/- {results['rand']['d_s_plateau_std']:.3f}")

    if results['delta_ds_plateau'] is not None:
        logger.info(f"Delta d_s: {results['delta_ds_plateau']:.3f} +/- {results['delta_ds_plateau_err']:.3f}")

        # Simple significance test
        if results['delta_ds_plateau_err'] > 0:
            significance = abs(results['delta_ds_plateau']) / results['delta_ds_plateau_err']
            logger.info(f"Significance: {significance:.1f}σ")

            if significance > 3:
                logger.info("PASS: Clustering signal detected at >3σ")
            else:
                logger.info("CAUTION: Clustering signal <3σ (may need larger sample or different ε)")
    else:
        logger.info("No common plateau window found between data and randoms")

    logger.info("\nTest complete. Check outputs:")
    logger.info(f"  - {output_json}")
    logger.info(f"  - {output_pdf}")


def test_boundary_score():
    """Quick test of boundary score functionality."""
    logger.info("\n--- Testing boundary score computation ---")

    # Create simple test case
    n_data = 1000
    n_rand = 5000

    # Put data in center, randoms uniformly distributed
    rng = np.random.default_rng(0)

    # Data concentrated in center
    data_ra = 180 + rng.normal(0, 5, n_data)
    data_dec = 35 + rng.normal(0, 5, n_data)
    data_z = 0.5 + rng.normal(0, 0.05, n_data)

    # Randoms uniform over larger area
    rand_ra = rng.uniform(160, 200, n_rand)
    rand_dec = rng.uniform(20, 50, n_rand)
    rand_z = rng.uniform(0.3, 0.7, n_rand)

    # Convert to Cartesian
    X_data = comoving_cartesian(data_ra, data_dec, data_z)
    X_rand = comoving_cartesian(rand_ra, rand_dec, rand_z)

    # Import boundary function
    from desi_return_probability import boundary_score_from_randoms

    scores = boundary_score_from_randoms(X_data, X_rand, m=32)

    logger.info(f"Boundary scores: min={scores.min():.2f}, max={scores.max():.2f}, "
                f"median={np.median(scores):.2f}")

    # Points near edges (large deviation from center) should have higher scores
    center_mask = (np.abs(data_ra - 180) < 3) & (np.abs(data_dec - 35) < 3)
    edge_mask = ~center_mask

    if np.sum(center_mask) > 0 and np.sum(edge_mask) > 0:
        logger.info(f"Center region mean score: {np.mean(scores[center_mask]):.2f}")
        logger.info(f"Edge region mean score: {np.mean(scores[edge_mask]):.2f}")

        if np.mean(scores[edge_mask]) > np.mean(scores[center_mask]):
            logger.info("PASS: Edge points have higher boundary scores (as expected)")
        else:
            logger.info("WARNING: Boundary score pattern unexpected")


def test_epsilon_ball_graph():
    """Test epsilon-ball graph construction."""
    logger.info("\n--- Testing epsilon-ball graph ---")

    from desi_return_probability import build_eps_ball_graph, normalized_laplacian

    # Simple 3D grid
    x = np.linspace(0, 100, 10)
    y = np.linspace(0, 100, 10)
    z = np.linspace(0, 100, 10)
    xx, yy, zz = np.meshgrid(x, y, z)
    X = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    logger.info(f"Test grid: {X.shape[0]} points")

    # Test different epsilon values
    for eps in [15, 20, 30]:
        try:
            A = build_eps_ball_graph(X, eps=eps)
            L = normalized_laplacian(A)

            degrees = np.array(A.sum(axis=1)).ravel()
            logger.info(f"eps={eps}: edges={A.nnz//2}, mean_deg={np.mean(degrees):.1f}, "
                        f"min_deg={np.min(degrees)}, max_deg={np.max(degrees)}")
        except ValueError as e:
            logger.info(f"eps={eps}: {e}")


if __name__ == '__main__':
    # Run basic tests first
    test_boundary_score()
    test_epsilon_ball_graph()

    # Run full pipeline test
    run_test_pipeline()
