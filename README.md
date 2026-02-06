# STARGATE-RW

**Spectral Topology Analysis of Random Graphs and Transport Evolution via Random Walks**

A Python framework for characterising transport properties of complex networks
through spectral geometry. STARGATE-RW computes spectral dimension, anomalous
diffusion exponents, and correlation dimension with publication-quality statistical
validation.

## Overview

STARGATE-RW provides three core modules:

| Module | Description |
|--------|-------------|
| `diffusion_spectral_dimension.py` | Spectral dimension ($d_s$) via heat-kernel return-probability decay |
| `correlation_dimension.py` | Correlation dimension ($D_2$) with KD-tree acceleration and guard-region edge correction |
| `forensic_validation_suite.py` | Three-gate validation protocol (Whitening, Radial Null, Phase Surrogate) |

## Installation

```bash
git clone https://github.com/cbotnen/STARGATE-RW.git
cd STARGATE-RW
pip install numpy scipy networkx scikit-learn matplotlib
```

### Dependencies

- Python >= 3.8
- NumPy
- SciPy
- NetworkX
- scikit-learn
- Matplotlib

## Quick Start

### Spectral Dimension

```python
python diffusion_spectral_dimension.py
```

Runs the built-in demo on synthetic point clouds (3D random, 2D sphere, 1D
filament) and reports spectral dimension estimates with uncertainty.

### Correlation Dimension

```python
python correlation_dimension.py
```

Computes $D_2$ for synthetic datasets using the Grassberger-Procaccia algorithm
with bootstrap confidence intervals.

### Forensic Validation

```python
python forensic_validation_suite.py
```

Runs the three-gate validation protocol on synthetic filamentary data to
distinguish genuine geometric structure from statistical artefacts.

### Generate Figures

```python
python generate_figures.py
```

Regenerates all manuscript figures (PRE composite, SciRep URC Wedge, PRL Spectral
Horizon).

## Scientific Background

The spectral dimension $d_s$ governs the return probability of a random walker on a
graph:

$$P_{00}(t) \sim t^{-d_s/2}$$

Unlike topological dimension, $d_s$ encodes connectivity density and branching
structure, providing a universal metric for diffusion efficiency across domains.

This framework has been applied to:

- **Cosmology**: Cosmic Web filament transport (SDSS data)
- **Oncology**: Glioblastoma topological phase transitions (Malignancy Horizon)
- **AI**: Mixture-of-Experts routing topology diagnostics

## Associated Publications

1. *Topological Diodes and Spectral Traps in Glioblastoma* — Physical Review E
2. *Scale-Invariant Transport Networks and the Universal Routing Constraint* — Scientific Reports
3. *The Malignancy Horizon: A Spectral Dimension Phase Transition in Oncology* — Physical Review Letters

## Author

**Craig W. Botnen**
Independent Researcher, Billings, Montana
ORCID: [0009-0007-9966-2985](https://orcid.org/0009-0007-9966-2985)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
