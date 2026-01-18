#!/usr/bin/env python3
"""
DESI DR1 Lyα HEALPix Campaign Runner
=====================================

Production script for running A_topo analysis across HEALPix tiles
with full forensic validation (whitening gate, radial null, FDR correction).

Usage:
    python run_lya_tiles.py --delta-dir /path/to/Delta --n-tiles 50 --n-null 200

Author: Craig W. Botnen, FDEP Institute
"""

import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

# Import our validated modules
from stargate_atopo import (
    ra_dec_z_to_comoving,
    build_knn_graph_CORRECT,
    build_normalized_laplacian_CORRECT,
    compute_atopo_CORRECT,
    validate_coordinates
)
from forensic_validation_suite import mahalanobis_whiten, radial_shuffle


# =============================================================================
# DATA LOADING
# =============================================================================

def load_lya_metadata(delta_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load RA, Dec, Z from DESI Lyα delta file HDU2 (METADATA).

    Parameters
    ----------
    delta_file : str
        Path to delta-*.fits.gz file

    Returns
    -------
    ra, dec, z : arrays
        Sightline coordinates
    """
    try:
        from astropy.io import fits
    except ImportError:
        raise ImportError("astropy required: pip install astropy")

    with fits.open(delta_file) as hdu:
        # METADATA is typically HDU2
        if 'METADATA' in hdu:
            meta = hdu['METADATA'].data
        elif len(hdu) > 2:
            meta = hdu[2].data
        else:
            raise ValueError(f"Cannot find METADATA in {delta_file}")

        ra = meta['RA']
        dec = meta['DEC']
        z = meta['Z']

    return ra, dec, z


def discover_delta_files(delta_dir: str, pattern: str = "delta-*.fits.gz") -> List[str]:
    """Find all delta files in directory."""
    files = sorted(glob.glob(os.path.join(delta_dir, pattern)))
    if not files:
        # Try subdirectories
        files = sorted(glob.glob(os.path.join(delta_dir, "**", pattern), recursive=True))
    return files


# =============================================================================
# RADIAL NULL SHUFFLE
# =============================================================================

def null_shuffle_radial_from_nhat(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Radial-distance shuffle preserving angular positions (n-hat).

    This is the DESI-specific null: keeps sightline directions fixed,
    only shuffles the radial (redshift) component.
    """
    return radial_shuffle(xyz, rng=rng)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def compute_z_score(observed: float, null_samples: np.ndarray) -> Tuple[float, float]:
    """Compute Z-score and empirical p-value."""
    null_mean = np.mean(null_samples)
    null_std = np.std(null_samples, ddof=1)

    if null_std < 1e-10:
        return 0.0, 1.0

    z_score = (observed - null_mean) / null_std

    # Two-tailed empirical p-value
    n_extreme = np.sum(np.abs(null_samples - null_mean) >= np.abs(observed - null_mean))
    p_value = (n_extreme + 1) / (len(null_samples) + 1)

    return z_score, p_value


def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Returns boolean array of which hypotheses are rejected.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH threshold: p[i] <= (i+1) * alpha / n
    thresholds = (np.arange(1, n+1) / n) * alpha

    # Find largest i where p[i] <= threshold[i]
    valid = sorted_p <= thresholds
    if not np.any(valid):
        return np.zeros(n, dtype=bool)

    max_valid = np.max(np.where(valid)[0])

    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_idx[:max_valid+1]] = True

    return rejected


def robust_H_index(z_scores: np.ndarray) -> float:
    """
    Heterogeneity index using MAD-scaled Z-scores.

    Low H-index indicates spatially homogeneous signal.
    """
    median_z = np.median(z_scores)
    mad = np.median(np.abs(z_scores - median_z))

    if mad < 1e-10:
        return 0.0

    # H = std(Z) / MAD-scale
    return np.std(z_scores, ddof=1) / (1.4826 * mad)


# =============================================================================
# TILE ANALYSIS
# =============================================================================

@dataclass
class TileResult:
    """Result container for a single tile."""
    tile_id: str
    n_sightlines: int
    atopo_observed: float
    d_s: float
    lambda2: float
    atopo_null_mean: float
    atopo_null_std: float
    z_score: float
    p_value: float
    is_significant: bool  # After FDR correction
    spectrum_valid: bool


def analyze_tile(
    xyz: np.ndarray,
    tile_id: str,
    k: int = 15,
    t_window: Tuple[float, float] = (5, 100),
    n_modes: int = 80,
    n_null: int = 100,
    seed: int = 42,
    verbose: bool = False
) -> Optional[TileResult]:
    """
    Analyze a single HEALPix tile.

    Parameters
    ----------
    xyz : ndarray (N, 3)
        Comoving coordinates
    tile_id : str
        Tile identifier
    k : int
        k-NN parameter
    t_window : tuple
        Diffusion time window (t_min, t_max)
    n_modes : int
        Number of eigenmodes for A_topo
    n_null : int
        Number of null shuffles
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    TileResult or None if analysis fails
    """
    N = xyz.shape[0]

    if N < 50:
        if verbose:
            print(f"  Tile {tile_id}: Skipping (N={N} < 50)")
        return None

    # Validate coordinates
    if not validate_coordinates(xyz, name=tile_id):
        return None

    rng = np.random.default_rng(seed)

    # Build graph and compute observed A_topo
    adj = build_knn_graph_CORRECT(xyz, k=k, verbose=False)
    atopo_obs, diag = compute_atopo_CORRECT(
        xyz, adj, t_window=t_window, n_eig=n_modes, verbose=False
    )

    if atopo_obs is None:
        if verbose:
            print(f"  Tile {tile_id}: A_topo computation failed")
        return None

    d_s = diag.get('d_s', 0.0)
    spectrum_valid = diag.get('spectrum_valid', False)

    # Compute lambda2 (algebraic connectivity)
    from scipy.sparse.linalg import eigsh
    L = build_normalized_laplacian_CORRECT(adj, verbose=False)
    try:
        evals, _ = eigsh(L, k=2, which='SM', tol=1e-4)
        lambda2 = float(sorted(evals)[1])
    except Exception:
        lambda2 = 0.0

    # Null distribution via radial shuffle
    null_atopos = []
    for i in range(n_null):
        xyz_null = null_shuffle_radial_from_nhat(xyz, rng)
        adj_null = build_knn_graph_CORRECT(xyz_null, k=k, verbose=False)
        atopo_null, _ = compute_atopo_CORRECT(
            xyz_null, adj_null, t_window=t_window, n_eig=n_modes, verbose=False
        )
        if atopo_null is not None:
            null_atopos.append(atopo_null)

    if len(null_atopos) < 10:
        if verbose:
            print(f"  Tile {tile_id}: Too few valid null samples")
        return None

    null_atopos = np.array(null_atopos)
    z_score, p_value = compute_z_score(atopo_obs, null_atopos)

    if verbose:
        print(f"  Tile {tile_id}: N={N}, A_topo={atopo_obs:.4f}, Z={z_score:.2f}, p={p_value:.4f}")

    return TileResult(
        tile_id=tile_id,
        n_sightlines=N,
        atopo_observed=atopo_obs,
        d_s=d_s,
        lambda2=lambda2,
        atopo_null_mean=np.mean(null_atopos),
        atopo_null_std=np.std(null_atopos, ddof=1),
        z_score=z_score,
        p_value=p_value,
        is_significant=False,  # Updated after FDR
        spectrum_valid=spectrum_valid
    )


# =============================================================================
# CAMPAIGN RUNNER
# =============================================================================

def run_campaign(
    delta_files: List[str],
    n_tiles: int = 50,
    n_null: int = 200,
    k: int = 15,
    t_window: Tuple[float, float] = (5, 100),
    n_modes: int = 80,
    whiten: bool = False,
    fdr_alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[TileResult], dict]:
    """
    Run HEALPix campaign across multiple tiles.

    Parameters
    ----------
    delta_files : list of str
        Paths to delta files (each = one tile)
    n_tiles : int
        Maximum number of tiles to process
    n_null : int
        Number of null shuffles per tile
    k : int
        k-NN parameter
    t_window : tuple
        Diffusion time window
    n_modes : int
        Number of eigenmodes
    whiten : bool
        Apply Mahalanobis whitening before analysis
    fdr_alpha : float
        FDR significance level
    seed : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    results : list of TileResult
    summary : dict with campaign statistics
    """
    if verbose:
        print("="*70)
        print(f"DESI DR1 Lyα HEALPix Campaign")
        print(f"  Tiles: {min(n_tiles, len(delta_files))}")
        print(f"  Null draws: {n_null}")
        print(f"  Whitening: {whiten}")
        print("="*70)

    results = []

    for i, delta_file in enumerate(delta_files[:n_tiles]):
        tile_id = Path(delta_file).stem

        if verbose:
            print(f"\n[{i+1}/{min(n_tiles, len(delta_files))}] Processing {tile_id}")

        try:
            ra, dec, z = load_lya_metadata(delta_file)
            xyz = ra_dec_z_to_comoving(ra, dec, z)

            if whiten:
                xyz, _, _ = mahalanobis_whiten(xyz)

            result = analyze_tile(
                xyz, tile_id,
                k=k, t_window=t_window, n_modes=n_modes,
                n_null=n_null, seed=seed + i, verbose=verbose
            )

            if result is not None:
                results.append(result)

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            continue

    if not results:
        return [], {'error': 'No valid tiles processed'}

    # Apply FDR correction
    p_values = np.array([r.p_value for r in results])
    rejected = bh_fdr(p_values, alpha=fdr_alpha)

    for i, result in enumerate(results):
        result.is_significant = rejected[i]

    # Compute summary statistics
    z_scores = np.array([r.z_score for r in results])

    summary = {
        'n_tiles_processed': len(results),
        'n_significant_fdr': int(np.sum(rejected)),
        'fdr_alpha': fdr_alpha,
        'median_z_score': float(np.median(z_scores)),
        'mean_z_score': float(np.mean(z_scores)),
        'std_z_score': float(np.std(z_scores, ddof=1)),
        'max_z_score': float(np.max(z_scores)),
        'min_z_score': float(np.min(z_scores)),
        'H_index': float(robust_H_index(z_scores)),
        'fraction_significant': float(np.sum(rejected) / len(results)),
        'whitened': whiten
    }

    if verbose:
        print("\n" + "="*70)
        print("CAMPAIGN SUMMARY")
        print("="*70)
        print(f"  Tiles processed: {summary['n_tiles_processed']}")
        print(f"  Significant (FDR): {summary['n_significant_fdr']} ({100*summary['fraction_significant']:.1f}%)")
        print(f"  Median Z-score: {summary['median_z_score']:.2f}")
        print(f"  H-index: {summary['H_index']:.2f}")
        print("="*70)

    return results, summary


def save_results(results: List[TileResult], summary: dict, output_path: str):
    """Save results to CSV and JSON."""
    import csv

    # Save tile results to CSV
    csv_path = output_path if output_path.endswith('.csv') else output_path + '.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Save summary to JSON
    json_path = csv_path.replace('.csv', '_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DESI DR1 Lyα HEALPix Campaign',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline run (unwhitened)
  python run_lya_tiles.py --delta-dir /path/to/Delta --n-tiles 50 --n-null 200

  # Whitened run (geometry-controlled)
  python run_lya_tiles.py --delta-dir /path/to/Delta --n-tiles 50 --n-null 200 --whiten

  # Quick test
  python run_lya_tiles.py --delta-dir /path/to/Delta --n-tiles 5 --n-null 20
        """
    )

    parser.add_argument('--delta-dir', type=str, required=True,
                        help='Directory containing delta-*.fits.gz files')
    parser.add_argument('--n-tiles', type=int, default=50,
                        help='Number of tiles to process (default: 50)')
    parser.add_argument('--n-null', type=int, default=200,
                        help='Number of null shuffles per tile (default: 200)')
    parser.add_argument('--k', type=int, default=15,
                        help='k-NN parameter (default: 15)')
    parser.add_argument('--t-min', type=float, default=5.0,
                        help='Minimum diffusion time (default: 5)')
    parser.add_argument('--t-max', type=float, default=100.0,
                        help='Maximum diffusion time (default: 100)')
    parser.add_argument('--n-modes', type=int, default=80,
                        help='Number of eigenmodes (default: 80)')
    parser.add_argument('--whiten', action='store_true',
                        help='Apply Mahalanobis whitening')
    parser.add_argument('--fdr-alpha', type=float, default=0.05,
                        help='FDR significance level (default: 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--out', type=str, default='campaign_results.csv',
                        help='Output file path (default: campaign_results.csv)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Discover delta files
    delta_files = discover_delta_files(args.delta_dir)

    if not delta_files:
        print(f"ERROR: No delta files found in {args.delta_dir}")
        sys.exit(1)

    print(f"Found {len(delta_files)} delta files")

    # Run campaign
    results, summary = run_campaign(
        delta_files,
        n_tiles=args.n_tiles,
        n_null=args.n_null,
        k=args.k,
        t_window=(args.t_min, args.t_max),
        n_modes=args.n_modes,
        whiten=args.whiten,
        fdr_alpha=args.fdr_alpha,
        seed=args.seed,
        verbose=not args.quiet
    )

    if results:
        save_results(results, summary, args.out)
    else:
        print("No results to save")
        sys.exit(1)


# =============================================================================
# SYNTHETIC TEST MODE
# =============================================================================

def run_synthetic_test():
    """Run campaign on synthetic data for testing."""
    print("="*70)
    print("SYNTHETIC TEST MODE")
    print("="*70)

    np.random.seed(42)

    # Generate 5 synthetic "tiles"
    results = []

    for i in range(5):
        N = np.random.randint(200, 500)

        # Create structured data (filamentary)
        t = np.linspace(0, 100, N)
        xyz = np.column_stack([
            t + np.random.normal(0, 5, N),
            20 * np.sin(t / 10) + np.random.normal(0, 5, N),
            20 * np.cos(t / 10) + np.random.normal(0, 5, N)
        ])

        result = analyze_tile(
            xyz, f"synthetic_{i:03d}",
            k=15, t_window=(5, 100), n_modes=50,
            n_null=30, seed=42 + i, verbose=True
        )

        if result is not None:
            results.append(result)

    if results:
        # Apply FDR
        p_values = np.array([r.p_value for r in results])
        rejected = bh_fdr(p_values, alpha=0.05)

        for i, r in enumerate(results):
            r.is_significant = rejected[i]

        z_scores = np.array([r.z_score for r in results])

        print("\n" + "="*70)
        print("SYNTHETIC TEST RESULTS")
        print("="*70)
        print(f"\n{'Tile':<20} {'N':<8} {'A_topo':<10} {'Z':<8} {'p':<10} {'Sig'}")
        print("-"*70)
        for r in results:
            sig = "***" if r.is_significant else ""
            print(f"{r.tile_id:<20} {r.n_sightlines:<8} {r.atopo_observed:<10.4f} "
                  f"{r.z_score:<8.2f} {r.p_value:<10.4f} {sig}")

        print(f"\nMedian Z: {np.median(z_scores):.2f}")
        print(f"H-index: {robust_H_index(z_scores):.2f}")
        print(f"Significant (FDR): {np.sum(rejected)}/{len(results)}")


if __name__ == '__main__':
    # If no args, run synthetic test
    if len(sys.argv) == 1:
        run_synthetic_test()
    else:
        main()
