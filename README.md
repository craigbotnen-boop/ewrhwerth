# STARGATE: Spectral Transport Analysis for Universal Scaling

This repository contains the complete implementation of the STARGATE framework for cross-domain validation of the Asymptotic Optimality (AO) scaling hypothesis (d_s → 4/3).

## Three-Paper Suite

### 1. **Physical Review E** - Mathematical Foundations
- Validated spectral dimension measurement methodology
- Benchmark results on synthetic graphs (lattices, trees, scale-free networks)
- Proof-of-concept applications to social networks

### 2. **Physical Review Letters** - BraTS Biological Validation ⭐
- **Complete and ready to run**
- Brain tumor segmentation analysis using publicly available BraTS dataset
- Tests the "malignant collapse" hypothesis: pathological networks deviate from d_s = 4/3
- Full pipeline from MRI data to spectral dimension measurement

### 3. **Universe Research Compendium** - Cosmological Extension
- Analysis of cosmic structure formation (DECADE First Light dataset)
- Dark matter halo connectivity and spectral dimensionality
- Tests universality of AO scaling across biological and cosmological domains

---

## Quick Start: BraTS Analysis

### Prerequisites

```bash
pip install nibabel numpy scipy matplotlib pandas
```

### Download BraTS Dataset

1. Visit [BraTS 2021 on Synapse](https://www.synapse.org/#!Synapse:syn25829067)
2. Register and download the training dataset
3. Extract to a local directory (e.g., `./BraTS2021_Training`)

### Run Analysis

```bash
# Analyze all samples
python stargate_brats.py ./BraTS2021_Training --output brats_results

# Analyze first 50 samples (faster)
python stargate_brats.py ./BraTS2021_Training --output brats_results --max-samples 50

# Adjust spectral parameters
python stargate_brats.py ./BraTS2021_Training \
    --output brats_results \
    --n-probe 128 \
    --connectivity 26 \
    --t-min 0.1 \
    --t-max 100.0 \
    --n-t 60
```

### Output

The analysis produces:

- **`brats_spectral_results.csv`**: Per-sample spectral dimension measurements
- **`summary.json`**: Aggregate statistics (median d_s, Δ_AO, hypothesis verdict)
- **`brats_figure.png`**: Publication-ready three-panel figure
  - Panel A: Spectral dimension curves d_s(t)
  - Panel B: Distribution histogram of d_s values
  - Panel C: Transport deficit (Δ_AO) vs tumor size

---

## Methodology Overview

### Spectral Dimension Measurement

For a network with adjacency matrix A:

1. **Compute normalized Laplacian**: L = I - D^(-1/2) A D^(-1/2)
2. **Estimate return probability**: P(t) = (1/N) Tr[exp(-tL)] using Hutchinson's method
3. **Extract spectral dimension**: d_s(t) = -2 d(log P) / d(log t)
4. **Find plateau**: Identify stable region of d_s(t) → d̂_s
5. **Compute deviation**: Δ_AO = d̂_s - 4/3

### The 4/3 Hypothesis

**Asymptotic Optimality (AO)** predicts that transport-efficient networks converge to:

```
d_s → 4/3
```

This universal scaling law should hold across:
- Social networks (verified in PRE)
- Biological networks (BraTS analysis)
- Cosmological structures (DECADE extension)

---

## BraTS Analysis Details

### Graph Construction

- Extracts 3D tumor mask from MRI segmentations (labels 1, 2, 4)
- Finds largest connected component (LCC)
- Builds voxel connectivity graph with 26-neighborhood (face+edge+corner)

### Validated Spectral Core

- Normalized Laplacian with degree-based regularization
- Hutchinson trace estimation with Rademacher probes
- Gradient-based d_s(t) calculation with log-log smoothing
- Plateau detection with variance + drift penalty

### Expected Results

If the AO hypothesis holds for healthy tissue:
- **Healthy baseline**: d_s ≈ 4/3 (1.33)
- **Malignant tumors**: d_s < 4/3 (transport deficit, Δ_AO < 0)
- **Interpretation**: Pathological disruption of optimal transport efficiency

---

## Repository Structure

```
.
├── stargate_brats.py          # Complete BraTS analysis pipeline
├── stargate_decade.py          # DECADE First Light skeleton (WIP)
├── README.md                   # This file
└── brats_results/             # Output directory (generated)
    ├── brats_spectral_results.csv
    ├── summary.json
    ├── brats_figure.png
    └── brats_figure.pdf
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{stargate_pre,
  title={Spectral Transport Analysis for Network Scaling},
  journal={Physical Review E},
  note={In preparation}
}

@article{stargate_prl,
  title={Universal Scaling in Brain Tumor Networks},
  journal={Physical Review Letters},
  note={In preparation}
}

@article{stargate_urc,
  title={Cosmic Structure and the 4/3 Conjecture},
  journal={Universe Research Compendium},
  note={In preparation}
}
```

---

## Development Status

- ✅ **PRE**: Mathematical foundations validated
- ✅ **PRL (BraTS)**: Complete and ready to run
- 🚧 **URC (DECADE)**: Skeleton implemented, awaiting data retrieval logic

---

## Contact

For questions about the BraTS analysis or STARGATE methodology, please open an issue in this repository.

---

## License

Research code for academic use. See individual papers for licensing terms upon publication.
