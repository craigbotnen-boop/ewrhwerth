---
title: 'STARGATE-RW: Spectral Topology Analysis of Random Graphs and Transport Evolution via Random Walks'
tags:
  - Python
  - spectral geometry
  - random walks
  - network science
  - glioblastoma
  - anomalous diffusion
  - spectral dimension
authors:
  - name: Craig W. Botnen
    orcid: 0009-0007-9966-2985
    affiliation: 1
affiliations:
  - name: Independent Researcher, Billings, Montana 59101, USA
    index: 1
date: 6 February 2026
bibliography: paper.bib
---

# Summary

STARGATE-RW (Spectral Topology Analysis of Random Graphs and Transport Evolution
via Random Walks) is a Python framework for characterising transport properties of
complex networks through spectral geometry. The software computes the spectral
dimension $d_s$, anomalous diffusion exponent $\alpha_{\mathrm{RW}}$, and
correlation dimension $D_2$ of arbitrary graphs, enabling researchers to quantify
topological anisotropy and manifold scaling in datasets ranging from cosmological
structure to clinical neuroimaging.

The package provides three core modules:

- **Spectral dimension estimation** via heat-kernel return-probability decay
  ($P_{00}(t) \sim t^{-d_s/2}$), supporting both exact eigendecomposition and
  stochastic Lanczos quadrature for large networks.
- **Correlation dimension analysis** using $O(N \log N)$ KD-tree pair counting
  with guard-region edge correction and bootstrap uncertainty quantification.
- **Forensic validation suite** implementing a three-gate protocol (Whitening,
  Radial Null, Phase Surrogate) to separate genuine geometric structure from
  statistical artefacts.

# Statement of Need

Spectral geometry provides a powerful lens for understanding transport on complex
networks, yet existing tools either require exact eigendecomposition—prohibitively
expensive at $O(N^3)$ for large graphs—or lack the statistical validation needed
for publication-quality results.

STARGATE-RW addresses this gap by implementing a high-performance spectral
dimension instrument that scales to massive datasets. It allows researchers to
quantify topological anisotropy and manifold scaling (spectral dimension) in large
networks, bridging the gap between theoretical physics (e.g., Cosmic Web structure
[@alexander1982; @havlin2002]) and clinical bioinformatics (e.g., patient-specific
gene manifolds and tumor microarchitecture [@stupp2005]).

The software has been applied to three domains:

1. **Cosmology**: Characterising transport on Cosmic Web filaments derived from
   SDSS galaxy catalogues, revealing that large-scale structure inhabits a narrow
   stability band—the Universal Routing Constraint (URC) Wedge [@alexander1982].
2. **Oncology**: Identifying a topological phase transition (the "Malignancy
   Horizon") in glioblastoma, where a spectral dimension gap of
   $\Delta d_s \approx 3.5$ between necrotic core and enhancing rim creates an
   entropic trap for therapeutic agents [@stupp2005; @jain2001].
3. **AI routing**: Benchmarking Mixture-of-Experts routing topologies against the
   URC Wedge to diagnose transport bottlenecks in neural network architectures.

No existing Python package combines spectral dimension estimation, correlation
dimension analysis, and forensic validation in a single framework with the
numerical safeguards (guard-region correction, plateau detection, bootstrap
resampling) required for rigorous scientific analysis.

# Mathematics

The spectral dimension is extracted from the return probability of a random walk on
a graph $G$:

$$P_{00}(t) \sim t^{-d_s/2}$$

where $P_{00}(t)$ is the probability that a walker returns to its starting node at
time $t$ [@alexander1982]. The anomalous diffusion exponent
$\alpha_{\mathrm{RW}}$ is measured from the mean squared displacement:

$$\langle r^2(t) \rangle \sim t^{\alpha_{\mathrm{RW}}}$$

For fractal networks at the percolation threshold, these quantities satisfy the
Alexander–Orbach conjecture $\alpha_{\mathrm{RW}} \approx d_s / 2$ [@alexander1982;
@havlin2002]. The correlation dimension $D_2$ is estimated via the Grassberger–
Procaccia algorithm with $O(N \log N)$ KD-tree acceleration [@grassberger1983].

# Acknowledgements

The author acknowledges the open-source scientific Python ecosystem (NumPy, SciPy,
NetworkX, scikit-learn) upon which this software is built. The author also
acknowledges the use of large language models for code generation and LaTeX
typesetting during development; all scientific claims and interpretations remain the
sole responsibility of the author.

# References
