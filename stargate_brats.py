# stargate_brats.py
#
# STARGATE analysis of brain tumor segmentation data (BraTS challenge)
# Extracts tumor topology from MRI segmentations and computes spectral dimension
# to test the "malignant collapse" hypothesis (d_s deviation from 4/3)
#
# Dataset: BraTS 2021 (or later) - available via Synapse
# Download: https://www.synapse.org/#!Synapse:syn25829067
#
# Prerequisites:
#   pip install nibabel numpy scipy matplotlib pandas

from __future__ import annotations

import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.ndimage import label, binary_dilation, generate_binary_structure

try:
    import nibabel as nib
except ImportError:
    raise ImportError("Install nibabel: pip install nibabel")

import matplotlib.pyplot as plt


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TumorSample:
    """Container for a single tumor segmentation analysis."""
    sample_id: str
    seg_path: str
    n_voxels: int = 0
    n_nodes: int = 0
    n_edges: int = 0
    ds_hat: float = np.nan
    ds_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    t_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    window: Tuple[int, int] = (0, 0)
    neg_rate: float = 0.0
    delta_ao: float = np.nan  # Deviation from 4/3
    lcc_fraction: float = 0.0  # Fraction of voxels in largest connected component


@dataclass
class BraTSDataset:
    """Container for the full BraTS analysis."""
    root: Path
    samples: List[TumorSample] = field(default_factory=list)
    healthy_baseline: Optional[float] = None


# =============================================================================
# BraTS Data Loading
# =============================================================================

def discover_brats_samples(
    brats_root: str,
    seg_suffix: str = "_seg.nii.gz"
) -> BraTSDataset:
    """
    Discovers BraTS segmentation files in the standard directory structure.

    Expected structure:
    brats_root/
    ├── BraTS2021_00000/
    │   ├── BraTS2021_00000_flair.nii.gz
    │   ├── BraTS2021_00000_t1.nii.gz
    │   ├── BraTS2021_00000_t1ce.nii.gz
    │   ├── BraTS2021_00000_t2.nii.gz
    │   └── BraTS2021_00000_seg.nii.gz  <-- This is what we need
    └── BraTS2021_00001/
        └── ...
    """
    root = Path(brats_root)
    if not root.exists():
        raise FileNotFoundError(f"BraTS root not found: {root}")

    dataset = BraTSDataset(root=root)

    # Find all segmentation files
    seg_pattern = f"**/*{seg_suffix}"
    seg_files = sorted(glob.glob(str(root / seg_pattern), recursive=True))

    for seg_path in seg_files:
        sample_id = Path(seg_path).parent.name
        sample = TumorSample(
            sample_id=sample_id,
            seg_path=seg_path
        )
        dataset.samples.append(sample)

    print(f"Discovered {len(dataset.samples)} BraTS samples")
    return dataset


def load_tumor_mask(
    seg_path: str,
    tumor_labels: Tuple[int, ...] = (1, 2, 4)  # BraTS: NCR/NET, ED, ET
) -> np.ndarray:
    """
    Loads the segmentation and extracts the tumor mask.

    BraTS label convention:
    - 0: Background
    - 1: Necrotic and Non-Enhancing Tumor (NCR/NET)
    - 2: Peritumoral Edema (ED)
    - 4: GD-Enhancing Tumor (ET)  [Note: label 3 is not used]

    Returns binary mask where any tumor tissue = True.
    """
    img = nib.load(seg_path)
    data = np.asarray(img.dataobj, dtype=np.int16)

    # Combine all tumor labels into binary mask
    mask = np.isin(data, tumor_labels)

    return mask


# =============================================================================
# Graph Construction (3D Voxel Connectivity)
# =============================================================================

def tumor_mask_to_graph(
    mask: np.ndarray,
    connectivity: int = 26,  # 6, 18, or 26 for 3D
    min_nodes: int = 100
) -> Optional[sp.csr_matrix]:
    """
    Converts a 3D binary tumor mask to a sparse adjacency matrix.

    Parameters
    ----------
    mask : ndarray
        3D binary mask (D, H, W)
    connectivity : int
        Neighborhood connectivity: 6 (face), 18 (face+edge), 26 (face+edge+corner)
    min_nodes : int
        Minimum number of tumor voxels required

    Returns
    -------
    A : csr_matrix or None
        Sparse adjacency matrix of the largest connected component
    """
    # Extract coordinates of tumor voxels
    coords = np.argwhere(mask)
    n_total = len(coords)

    if n_total < min_nodes:
        return None

    # Find largest connected component
    struct = generate_binary_structure(3, 1 if connectivity == 6 else 2 if connectivity == 18 else 3)
    labeled, n_components = label(mask, structure=struct)

    if n_components == 0:
        return None

    # Find largest component
    component_sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
    largest_label = np.argmax(component_sizes) + 1
    lcc_mask = labeled == largest_label

    # Get LCC coordinates
    coords = np.argwhere(lcc_mask)
    n_nodes = len(coords)

    if n_nodes < min_nodes:
        return None

    # Build index map: (d, h, w) -> node index
    D, H, W = mask.shape
    idx_map = -np.ones((D, H, W), dtype=np.int32)
    for k, (d, h, w) in enumerate(coords):
        idx_map[d, h, w] = k

    # Define neighbor offsets based on connectivity
    if connectivity == 6:
        # Face neighbors only
        offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    elif connectivity == 18:
        # Face + edge neighbors
        offsets = []
        for dd in [-1, 0, 1]:
            for dh in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if abs(dd) + abs(dh) + abs(dw) <= 2 and (dd, dh, dw) != (0, 0, 0):
                        offsets.append((dd, dh, dw))
    else:  # 26
        # All neighbors including corners
        offsets = []
        for dd in [-1, 0, 1]:
            for dh in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if (dd, dh, dw) != (0, 0, 0):
                        offsets.append((dd, dh, dw))

    # Build edge list
    rows, cols = [], []
    for d, h, w in coords:
        u = idx_map[d, h, w]
        for dd, dh, dw in offsets:
            nd, nh, nw = d + dd, h + dh, w + dw
            if 0 <= nd < D and 0 <= nh < H and 0 <= nw < W:
                v = idx_map[nd, nh, nw]
                if v >= 0:
                    rows.append(u)
                    cols.append(v)

    # Create sparse adjacency matrix
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Ensure symmetry (should already be, but be safe)
    A = ((A + A.T) > 0).astype(np.float32)

    lcc_fraction = n_nodes / n_total if n_total > 0 else 0

    return A, n_nodes, lcc_fraction


# =============================================================================
# STARGATE Spectral Analysis (Validated Core)
# =============================================================================

def normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """Computes L = I - D^{-1/2} A D^{-1/2}."""
    deg = np.array(A.sum(axis=1)).flatten()
    deg_safe = np.where(deg > 0, deg, 1.0)
    inv_sqrt = 1.0 / np.sqrt(deg_safe)
    D_inv_sqrt = sp.diags(inv_sqrt)
    L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()


def hutchinson_trace(
    L: sp.csr_matrix,
    t_grid: np.ndarray,
    n_probe: int = 64,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Estimates P(t) = (1/N) Tr[exp(-tL)] using Hutchinson's stochastic trace estimator.

    Returns (P_array, negativity_rate)
    """
    n = L.shape[0]
    rng = np.random.default_rng(seed)

    # Rademacher random vectors
    V = rng.choice([-1.0, 1.0], size=(n, n_probe)).astype(np.float64)

    P = np.zeros(len(t_grid), dtype=np.float64)
    neg_count = 0

    for i, t in enumerate(t_grid):
        # Y = exp(-tL) @ V
        Y = expm_multiply((-t) * L, V)

        # Trace estimate: (1/n_probe) * sum_j (V_j^T @ Y_j)
        tr_est = np.sum(V * Y) / n_probe

        if tr_est < 0:
            neg_count += 1

        P[i] = tr_est / n  # Normalize by number of nodes

    # Clip for numerical stability
    P = np.clip(P, 1e-300, None)
    neg_rate = neg_count / len(t_grid)

    return P, neg_rate


def spectral_dimension_curve(t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Computes d_s(t) = -2 * d(log P) / d(log t)."""
    log_t = np.log(t)
    log_P = np.log(P)
    d_log_P = np.gradient(log_P, log_t)
    return -2.0 * d_log_P


def find_plateau(
    t: np.ndarray,
    ds: np.ndarray,
    search_range: Tuple[float, float] = (1.0, 50.0),
    window_size: int = 8,
    ds_bounds: Tuple[float, float] = (0.5, 4.0)
) -> Tuple[float, Tuple[int, int]]:
    """
    Finds the most stable plateau in d_s(t) within the search range.

    Returns (ds_hat, (start_idx, end_idx))
    """
    # Mask to search range
    mask = (t >= search_range[0]) & (t <= search_range[1])
    valid_indices = np.where(mask)[0]

    if len(valid_indices) < window_size:
        # Fall back to full range
        valid_indices = np.arange(len(ds))

    best_score = np.inf
    best_window = (valid_indices[0], valid_indices[-1])
    best_ds = np.nan

    for i in range(len(valid_indices) - window_size + 1):
        start = valid_indices[i]
        end = valid_indices[i + window_size - 1]

        segment = ds[start:end+1]
        mu = np.mean(segment)

        # Check bounds
        if not (ds_bounds[0] <= mu <= ds_bounds[1]):
            continue

        # Score: variance + drift penalty
        var = np.var(segment)
        drift = np.abs(np.polyfit(np.arange(len(segment)), segment, 1)[0])
        score = var + 10.0 * drift**2

        if score < best_score:
            best_score = score
            best_window = (start, end)
            best_ds = mu

    return best_ds, best_window


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def analyze_sample(
    sample: TumorSample,
    t_min: float = 0.1,
    t_max: float = 100.0,
    n_t: int = 50,
    n_probe: int = 64,
    connectivity: int = 26,
    min_nodes: int = 100,
    seed: int = 42
) -> TumorSample:
    """
    Runs the complete STARGATE analysis on a single tumor sample.
    """
    # Load tumor mask
    try:
        mask = load_tumor_mask(sample.seg_path)
    except Exception as e:
        print(f"  Failed to load {sample.sample_id}: {e}")
        return sample

    sample.n_voxels = int(np.sum(mask))

    if sample.n_voxels < min_nodes:
        print(f"  {sample.sample_id}: Insufficient voxels ({sample.n_voxels})")
        return sample

    # Build graph
    result = tumor_mask_to_graph(mask, connectivity=connectivity, min_nodes=min_nodes)
    if result is None:
        print(f"  {sample.sample_id}: Graph construction failed")
        return sample

    A, n_nodes, lcc_fraction = result
    sample.n_nodes = n_nodes
    sample.n_edges = A.nnz // 2
    sample.lcc_fraction = lcc_fraction

    # Compute Laplacian
    L = normalized_laplacian(A)

    # Spectral analysis
    t_grid = np.geomspace(t_min, t_max, n_t)
    P, neg_rate = hutchinson_trace(L, t_grid, n_probe=n_probe, seed=seed)
    ds_curve = spectral_dimension_curve(t_grid, P)

    sample.t_grid = t_grid
    sample.ds_curve = ds_curve
    sample.neg_rate = neg_rate

    # Find plateau
    ds_hat, window = find_plateau(t_grid, ds_curve)
    sample.ds_hat = ds_hat
    sample.window = window

    # Compute deviation from AO
    sample.delta_ao = ds_hat - (4.0 / 3.0) if not np.isnan(ds_hat) else np.nan

    print(f"  {sample.sample_id}: n={n_nodes}, d_s={ds_hat:.3f}, Δ_AO={sample.delta_ao:+.3f}")

    return sample


def run_brats_analysis(
    brats_root: str,
    output_dir: str = "brats_results",
    max_samples: Optional[int] = None,
    **kwargs
) -> BraTSDataset:
    """
    Runs STARGATE analysis on all samples in a BraTS dataset.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    # Discover samples
    dataset = discover_brats_samples(brats_root)

    if max_samples is not None:
        dataset.samples = dataset.samples[:max_samples]

    print(f"\nAnalyzing {len(dataset.samples)} samples...")

    # Analyze each sample
    for i, sample in enumerate(dataset.samples):
        print(f"[{i+1}/{len(dataset.samples)}] Processing {sample.sample_id}")
        dataset.samples[i] = analyze_sample(sample, **kwargs)

    # Aggregate results
    valid_samples = [s for s in dataset.samples if not np.isnan(s.ds_hat)]

    if valid_samples:
        ds_values = [s.ds_hat for s in valid_samples]
        delta_values = [s.delta_ao for s in valid_samples]

        print(f"\n{'='*60}")
        print("BraTS STARGATE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Valid samples: {len(valid_samples)} / {len(dataset.samples)}")
        print(f"d_s: median={np.median(ds_values):.3f}, mean={np.mean(ds_values):.3f}, std={np.std(ds_values):.3f}")
        print(f"Δ_AO: median={np.median(delta_values):+.3f}, mean={np.mean(delta_values):+.3f}")
        print(f"Range: [{min(ds_values):.3f}, {max(ds_values):.3f}]")

        # Save results
        results_df = pd.DataFrame([{
            'sample_id': s.sample_id,
            'n_voxels': s.n_voxels,
            'n_nodes': s.n_nodes,
            'n_edges': s.n_edges,
            'lcc_fraction': s.lcc_fraction,
            'ds_hat': s.ds_hat,
            'delta_ao': s.delta_ao,
            'neg_rate': s.neg_rate
        } for s in valid_samples])

        results_df.to_csv(os.path.join(output_dir, "brats_spectral_results.csv"), index=False)

        # Summary statistics
        summary = {
            'n_samples': len(valid_samples),
            'ds_median': float(np.median(ds_values)),
            'ds_mean': float(np.mean(ds_values)),
            'ds_std': float(np.std(ds_values)),
            'delta_ao_median': float(np.median(delta_values)),
            'ao_target': 4.0/3.0,
            'hypothesis': 'malignant_collapse' if np.median(delta_values) < -0.1 else 'near_ao'
        }

        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

    return dataset


# =============================================================================
# Visualization
# =============================================================================

def plot_brats_results(
    dataset: BraTSDataset,
    output_dir: str = "brats_results"
) -> None:
    """
    Generates publication-quality figures for BraTS analysis.
    """
    valid_samples = [s for s in dataset.samples if not np.isnan(s.ds_hat)]

    if not valid_samples:
        print("No valid samples to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: d_s(t) curves (sample of 10)
    ax1 = axes[0]
    n_show = min(10, len(valid_samples))
    for s in valid_samples[:n_show]:
        ax1.plot(s.t_grid, s.ds_curve, alpha=0.5, lw=1)

    ax1.axhline(4/3, color='black', linestyle='--', label='AO target (4/3)')
    ax1.axhline(2.0, color='gray', linestyle=':', alpha=0.5, label='Euclidean (2.0)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Diffusion time t')
    ax1.set_ylabel('$d_s(t)$')
    ax1.set_title('A: Spectral dimension curves')
    ax1.legend()
    ax1.set_ylim(0, 4)

    # Panel B: Distribution of d_s
    ax2 = axes[1]
    ds_values = [s.ds_hat for s in valid_samples]
    ax2.hist(ds_values, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(4/3, color='red', linestyle='--', lw=2, label='AO target')
    ax2.axvline(np.median(ds_values), color='blue', linestyle='-', lw=2, label=f'Median ({np.median(ds_values):.2f})')
    ax2.set_xlabel('$\\hat{d}_s$')
    ax2.set_ylabel('Count')
    ax2.set_title('B: Distribution of tumor $d_s$')
    ax2.legend()

    # Panel C: Δ_AO vs tumor size
    ax3 = axes[2]
    sizes = [s.n_nodes for s in valid_samples]
    deltas = [s.delta_ao for s in valid_samples]
    ax3.scatter(sizes, deltas, alpha=0.6, edgecolors='black')
    ax3.axhline(0, color='red', linestyle='--', label='AO baseline')
    ax3.set_xscale('log')
    ax3.set_xlabel('Number of nodes (tumor size)')
    ax3.set_ylabel('$\\Delta_{AO}$ (deviation from 4/3)')
    ax3.set_title('C: Transport deficit vs. tumor size')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "brats_figure.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "brats_figure.pdf"))
    print(f"Saved figures to {output_dir}")
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="STARGATE analysis of BraTS tumor data")
    parser.add_argument("brats_root", help="Path to BraTS dataset root")
    parser.add_argument("--output", default="brats_results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--n-probe", type=int, default=64, help="Number of Hutchinson probes")
    parser.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26],
                        help="3D voxel connectivity")
    parser.add_argument("--t-min", type=float, default=0.1, help="Minimum diffusion time")
    parser.add_argument("--t-max", type=float, default=100.0, help="Maximum diffusion time")
    parser.add_argument("--n-t", type=int, default=50, help="Number of time points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    dataset = run_brats_analysis(
        args.brats_root,
        output_dir=args.output,
        max_samples=args.max_samples,
        n_probe=args.n_probe,
        connectivity=args.connectivity,
        t_min=args.t_min,
        t_max=args.t_max,
        n_t=args.n_t,
        seed=args.seed
    )

    plot_brats_results(dataset, args.output)


if __name__ == "__main__":
    # Import pandas here to avoid issues if not installed
    import pandas as pd
    main()
