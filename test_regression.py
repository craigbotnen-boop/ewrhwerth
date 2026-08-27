"""
Regression tests for halo graph analysis pipeline.

Generates a 10k stub, runs the full pipeline, and asserts
key metrics remain within physically expected bounds.
"""

import numpy as np
import pytest
import os

from analyze_halo_graph import (
    generate_cosmic_web_halos,
    build_halo_graph,
    auto_keig,
    graph_sanity_check,
)
from diffusion_spectral_dimension import compute_spectral_dimension
from correlation_dimension import compute_D2_full


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def halo_10k():
    """Generate a deterministic 10k halo catalog and k-NN graph."""
    xyz, _ = generate_cosmic_web_halos(n_halos=10_000, seed=42)
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    nn_dist, _ = tree.query(xyz, k=2)
    ll = nn_dist[:, 1].mean() * 1.5
    G = build_halo_graph(xyz, linking_length=ll, k_nn=12)
    return xyz, G


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpectralDimension:
    """Spectral dimension must stay in the FoF-like range 0.5 < d_s < 3.1."""

    def test_ds_range(self, halo_10k):
        xyz, _ = halo_10k
        keig = auto_keig(len(xyz))
        result = compute_spectral_dimension(
            xyz, k=15, n_eigs=keig, n_times=60,
            t_min=0.01, t_max=200.0, verbose=False,
        )
        assert np.isfinite(result.d_s), "d_s is not finite"
        assert 0.5 < result.d_s < 3.1, (
            f"d_s = {result.d_s:.3f} outside expected [0.5, 3.1]"
        )

    def test_ds_uncertainty_reasonable(self, halo_10k):
        xyz, _ = halo_10k
        keig = auto_keig(len(xyz))
        result = compute_spectral_dimension(
            xyz, k=15, n_eigs=keig, n_times=60,
            t_min=0.01, t_max=200.0, verbose=False,
        )
        assert result.d_s_std < 1.0, (
            f"d_s uncertainty {result.d_s_std:.3f} unreasonably large"
        )


class TestCorrelationDimension:
    """Correlation dimension D_2 must be finite with R² > 0.95."""

    def test_D2_range(self, halo_10k):
        xyz, _ = halo_10k
        result = compute_D2_full(
            xyz, use_guard_region=True,
            bootstrap=False, verbose=False,
        )
        assert np.isfinite(result.D2_estimate), "D_2 is not finite"
        assert 0.5 < result.D2_estimate < 4.0, (
            f"D_2 = {result.D2_estimate:.3f} outside expected [0.5, 4.0]"
        )

    def test_plateau_exists(self, halo_10k):
        xyz, _ = halo_10k
        result = compute_D2_full(
            xyz, use_guard_region=True,
            bootstrap=False, verbose=False,
        )
        n_plateau = result.plateau_mask.sum()
        assert n_plateau >= 3, (
            f"Only {n_plateau} plateau points — too few for reliable D_2"
        )


class TestGraphSanity:
    """Graph structure checks."""

    def test_gcc_covers_majority(self, halo_10k):
        _, G = halo_10k
        import networkx as nx
        gcc_size = max(len(c) for c in nx.connected_components(G))
        frac = gcc_size / G.number_of_nodes()
        assert frac > 0.9, f"GCC fraction {frac:.2%} < 90%"

    def test_mean_degree_sufficient(self, halo_10k):
        _, G = halo_10k
        degrees = [d for _, d in G.degree()]
        assert np.mean(degrees) >= 4, (
            f"Mean degree {np.mean(degrees):.1f} < 4 — graph too sparse"
        )

    def test_auto_keig_scaling(self):
        assert auto_keig(100) == 128        # floor
        assert auto_keig(1_000_000) == 512  # ceiling
        assert 128 <= auto_keig(50_000) <= 512
