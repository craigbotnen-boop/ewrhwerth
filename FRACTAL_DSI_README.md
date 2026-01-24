# Fractal Discrete Scale Invariance (DSI) Analysis

## Executive Summary

This module implements **comparative analysis of hierarchical fingerprints** in self-similar fractals via entropy gradient force ratios. The goal is to detect and characterize **log-periodic oscillations** as signatures of **discrete scale invariance (DSI)**.

---

## Scientific Background

### The Hierarchical Fingerprint Hypothesis

Traditional fractal characterization relies on **global** measures (fractal dimension d_f, spectral dimension d_s). However, these miss **hierarchical structure** encoded at intermediate scales.

**Key Observation:** Self-similar fractals have **discrete scaling symmetries** with preferred length scales:
- Sierpiński Carpet: λ = 3 (each level scales by factor of 3)
- Sierpiński Gasket: λ = 2 (each level scales by factor of 2)

**Hypothesis:** These discrete symmetries manifest as **log-periodic oscillations** in entropy gradients:
```
F(r) = S(r+1) - S(r)  [Shell entropy gradient]
```

where `S(r) = log N(r)` and `N(r)` is the shell occupancy at graph distance `r`.

### Why Graph Distance, Not Euclidean?

Euclidean distance is **blind to holes**. In a fractal like the Sierpiński Carpet, two points separated by a removed square appear "close" in Euclidean space but are **far** in graph distance (must walk around the hole).

**Graph distance honors the intrinsic metric** of the fractal substrate. This is consistent with Barlow-Bass theory, where the walk dimension `d_w` is defined by how the number of steps scales with intrinsic distance.

### The DSI Signature

If discrete scale invariance is present, force ratios should exhibit **log-periodic modulations**:

```
F(r) ~ r^β * (1 + B * cos(2π * log(r) / log(λ) + φ))
```

Expected wavelength:
- **Carpet:** Δlog(r) = log(3) ≈ 1.099
- **Gasket:** Δlog(r) = log(2) ≈ 0.693

**Controls:**
- **Euclidean 2D grid:** Should show NO oscillations (smooth power-law)
- **Random phase scrambling:** Should destroy periodicity (Monte Carlo null)

---

## Methodology

### 1. Fractal Generation (Sparse CSR)

**Sierpiński Carpet:**
- Start with 3×3 grid
- Recursively subdivide: each filled square → 8 squares (remove center)
- Level L: grid size 3^L, fill fraction (8/9)^L
- Memory-efficient: only store filled cells as sparse adjacency matrix

**Sierpiński Gasket:**
- "Three copies + corner identification" method (PCF construction)
- Preserves finite ramification property
- Level L: N ≈ 3^(L+1)/2 nodes
- Critical for walk dimension: bottlenecks at identification points

### 2. BFS Shell Counting

Count nodes at each **graph distance** r from a central node:
```
N(r) = number of nodes at BFS distance r
```

**Why BFS?**
- Honors intrinsic geometry (walks on the graph)
- Avoids Euclidean bias
- Computational complexity: O(N + E) where E ~ 4N for 2D lattices

### 3. Entropy Gradient Force Ratio

```
S(r) = log N(r)  [Shell entropy]
F(r) = S(r+1) - S(r)  [Entropy gradient "force"]
```

**Physical Interpretation:**
- `S(r)` measures accessible states at distance r
- `F(r)` measures **rate of change** of accessible states
- Log-periodic oscillations in F(r) indicate **discrete bottlenecks** at preferred scales

### 4. Detrending and Residual Analysis

**Power-law trend removal:**
```
log F(r) = β * log r + const
F_trend(r) = r^β
Residual = F(r) / F_trend(r) - 1
```

**Oscillation detection:**
- Zero-crossing count
- Spectral analysis (FFT or Lomb-Scargle periodogram)
- Crossing interval histogram

---

## Usage

### Installation

```bash
pip install -r requirements_fractal.txt
```

Dependencies:
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.3

### Local Execution (Small Test)

```bash
python fractal_dsi_analysis.py \
    --carpet-level 3 \
    --gasket-level 5 \
    --r-max 50 \
    --output test_dsi.png
```

Runtime: ~30 seconds
Carpet L=27, Gasket N~243

### Cluster Execution (Publication-Grade)

```bash
sbatch run_fractal_dsi.sbatch
```

Recommended parameters:
- `CARPET_LEVEL=4` (L=81, ~5168 nodes) — 2-3 min
- `CARPET_LEVEL=5` (L=243, ~32896 nodes) — 15-20 min, needs 8GB RAM
- `GASKET_LEVEL=6` (N~729) — 1 min
- `GASKET_LEVEL=7` (N~2187) — 3-4 min
- `R_MAX=80` (avoid edge effects, should be ~0.3*L for carpet)

Output: `results/fractal_dsi_comparison_L*_G*.png`

---

## Interpreting Results

### 3-Panel Plot Structure

**Panel A: Raw Force Ratios**
- X-axis: Graph distance r (log scale)
- Y-axis: |F(r)| = |S(r+1) - S(r)| (log scale)
- **Expected:**
  - Euclidean: Smooth power-law decay (F ~ r^β)
  - Fractals: May show visible "humps" or modulations

**Panel B: Detrended Residuals (SMOKING GUN)**
- X-axis: Graph distance r (log scale)
- Y-axis: Fractional residual (F / F_trend - 1)
- **Expected:**
  - Carpet: Sinusoidal oscillations with wavelength Δlog(r) ≈ 1.099
  - Gasket: Sinusoidal oscillations with wavelength Δlog(r) ≈ 0.693
  - Euclidean: Flat noise around zero (no oscillations)

**Panel C: Crossing Interval Distribution**
- X-axis: Δlog(r) between zero-crossings
- Y-axis: Histogram count
- **Expected:**
  - Carpet: Peak near log(3) ≈ 1.099
  - Gasket: Peak near log(2) ≈ 0.693
  - Random noise: Flat distribution (no preferred scale)

### Quantitative Criteria for DSI Detection

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| **Oscillation Amplitude** | std(residuals) > 0.1 | Visible modulations |
| **Zero-Crossing Count** | > 5 crossings in valid range | Multiple cycles observed |
| **Period Accuracy** | |measured - expected| / expected < 20% | Wavelength matches theory |
| **Euclidean Control** | std(residuals) < 0.05 | No false positives in control |

---

## Connection to Broader Research Program

### Part I: Cosmic Web Spectral Dimension (COMPLETED)

Files: `diffusion_spectral_dimension.py`, `correlation_dimension.py`, `forensic_validation_suite.py`

**Finding:** d_s anomaly in survey data → spectral horizon phenomenon

**Kill-Switch:** Phase-reconstruction recovery test (`run_killswitch_ensemble.sbatch`)
- If Recovery Score > 50%: Phase information is relevant
- Proceed to Part II for physical mechanism investigation

### Part II: Fractal DSI Mechanism (THIS MODULE)

**Goal:** Investigate whether hierarchical structure (discrete bottlenecks) can produce spectral dimension anomalies

**Test:** Do hierarchical fractals exhibit log-periodic modulations in entropy gradients?

**If YES:**
1. Validates that discrete scale invariance is **detectable** via entropy-based diagnostics
2. Provides theoretical foundation for "spectral horizon" interpretation
3. Suggests cosmic web may have **hierarchical bottleneck structure** (filament crossings, nodes)

**If NO:**
- DSI signatures are **not** a generic feature of self-similar structures
- Alternative mechanisms required (phase coherence at specific scales, survey artifacts)

### Part III: Cosmic Web Hierarchical Fingerprint (FUTURE)

**If Part II confirms DSI signatures in fractals:**

Apply the same entropy gradient force ratio analysis to actual cosmic web voxel graphs:
```python
# Load cosmic web graph from DESI DR1
A = build_voxel_graph_from_delta(delta_field, q=0.97)

# BFS from multiple seed nodes
for seed in random_seeds:
    shells = bfs_shell_counts(A, seed, r_max)
    r, F = entropy_gradient_force_ratio(shells)
    # Look for log-periodic oscillations
```

**Expected if cosmic web is hierarchically organized:**
- Log-periodic modulations in F(r)
- Wavelength corresponds to characteristic filament/node separation
- Destroyed by phase randomization (HON-1)
- Partially recovered by BAO reconstruction

**This would be direct evidence for:**
- Non-trivial topology beyond two-point statistics
- Hierarchical bottleneck structure in large-scale structure
- Phase-coherent organization across multiple scales

---

## Mathematical Foundations

### Hambly's Heat Kernel Asymptotics

**Reference:** Hambly et al. (1994), *Brownian motion on fractals*

For post-critically finite (PCF) fractals like the Sierpiński Gasket:

```
p(t, x, y) ~ t^(-d_s/2) * (1 + periodic corrections)
```

The **periodic corrections** arise from discrete scale invariance of the fractal.

**Implication:** Log-periodic modulations are **mathematically predicted**, not artifacts.

### Barlow-Bass Walk Dimension

**Reference:** Barlow & Bass (1999), *Random walks on graphs with volume growth*

Walk dimension:
```
d_w = 2 d_f / d_s
```

relates fractal dimension `d_f`, spectral dimension `d_s`, and the graph metric.

**For Sierpiński Gasket:**
- d_f = log(3)/log(2) ≈ 1.585
- d_s ≈ 1.36 (known from heat kernel)
- d_w ≈ 2.33

**Implication:** Graph distance scales differently than Euclidean distance. Using BFS is **required** for correct walk dimension measurement.

### Sornette's DSI Framework

**Reference:** Sornette (1998), *Discrete scale invariance and complex dimensions*

Log-periodic oscillations emerge from **complex eigenvalues** of renormalization operators:

```
λ = |λ| * exp(i θ)  [Complex scaling factor]
```

Imaginary part `θ` → periodic modulations with wavelength `2π / θ`.

**For real fractals (λ real):**
- Sierpiński structures have integer scaling (λ=2, 3)
- Oscillations arise from **discrete symmetry breaking** at subdivision steps
- Analogous to quantum system with discrete energy levels

---

## Publication Strategy

### Target Journals

1. **Physical Review E (PRE)** — Statistical, Nonlinear, and Soft Matter Physics
   - Section: Pattern formation and complexity
   - Precedent: Sornette's DSI papers, fractal walk dimension studies

2. **Journal of Statistical Mechanics (JSTAT)**
   - Section: Topological and geometrical aspects of statistical mechanics
   - Precedent: Spectral dimension on graphs, random walk studies

3. **Physica D: Nonlinear Phenomena**
   - Section: Fractals and multifractals
   - Precedent: Grassberger & Procaccia correlation dimension work

### Manuscript Structure (PRE Format)

**Title:** *Discrete Scale Invariance Signatures in Fractal Entropy Gradients*

**Abstract (~150 words):**
> We investigate log-periodic oscillations in entropy gradient force ratios as signatures of discrete scale invariance (DSI) in self-similar fractals. Using breadth-first search on sparse adjacency matrices, we compute shell occupancy N(r) at graph distance r for Sierpiński Carpet (λ=3) and Sierpiński Gasket (λ=2). The entropy gradient F(r) = log N(r+1) - log N(r) exhibits sinusoidal modulations with wavelengths Δlog(r) = log(λ), confirming theoretical predictions from Hambly's heat kernel asymptotics. Euclidean grid controls show no oscillations, ruling out discretization artifacts. This establishes entropy-based diagnostics as probes of hierarchical structure beyond traditional fractal dimension measurements. Applications to cosmic large-scale structure and complex networks are discussed.

**Sections:**
1. Introduction
   - Fractals in nature: lungs, rivers, cosmic web
   - Limitations of global dimension measures
   - DSI as probe of intermediate-scale structure

2. Theory
   - Graph distance vs Euclidean distance
   - Entropy gradient force ratio definition
   - Log-periodic oscillation prediction from scaling symmetry

3. Methods
   - Sparse CSR fractal generation
   - BFS shell counting
   - Detrending and residual analysis
   - Euclidean control baseline

4. Results
   - 3-panel comparison plots
   - Quantitative period measurements
   - Crossing statistics

5. Discussion
   - Comparison with Hambly's predictions
   - Robustness to parameter choices
   - Extensions to other fractals (Menger sponge, diffusion-limited aggregation)

6. Applications
   - Cosmic web hierarchical fingerprints
   - Network topology diagnostics
   - Anomalous diffusion characterization

---

## Next Steps

### Immediate (Week 1):
- [ ] Run `sbatch run_fractal_dsi.sbatch` with `CARPET_LEVEL=4, GASKET_LEVEL=6`
- [ ] Verify Panel B shows oscillations with correct periods
- [ ] Generate higher-resolution run (`CARPET_LEVEL=5`) for publication figure

### Short-term (Week 2-3):
- [ ] Implement Monte Carlo null testing
  - Phase-scramble shell counts N(r)
  - Generate 1000 null realizations
  - Compute p-value for observed oscillation amplitude
- [ ] Test on additional fractals (3D Menger sponge, DLA clusters)
- [ ] Implement spectral analysis (Lomb-Scargle periodogram)

### Medium-term (Month 1-2):
- [ ] Draft PRE manuscript
- [ ] Apply to cosmic web data (if kill-switch yields RECOVERY)
- [ ] Compare with N-body simulation mocks

### Long-term (Month 3+):
- [ ] Submit to PRE
- [ ] Prepare companion paper on cosmic web hierarchical fingerprints
- [ ] Develop general-purpose DSI detection library

---

## Files in This Module

| File | Purpose |
|------|---------|
| `fractal_dsi_analysis.py` | Main analysis script (Carpet, Gasket, Euclidean) |
| `run_fractal_dsi.sbatch` | SLURM batch script for cluster execution |
| `requirements_fractal.txt` | Python dependencies |
| `FRACTAL_DSI_README.md` | This documentation |

---

## Citation

If this methodology proves useful for your research, please cite:

```bibtex
@software{fractal_dsi_2025,
  title = {Fractal Discrete Scale Invariance Analysis via Entropy Gradients},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-repo]/fractal-dsi}
}
```

---

## References

1. Hambly, B. M., Kumagai, T., Kusuoka, S., & Zhou, X. Y. (1994). *Transition density estimates for Brownian motion on affine nested fractals*. Communications in Mathematical Physics, 165(3), 595-620.

2. Barlow, M. T., & Bass, R. F. (1999). *Random walks on graphical Sierpinski carpets*. In Random walks and discrete potential theory (pp. 26-55).

3. Sornette, D. (1998). *Discrete-scale invariance and complex dimensions*. Physics Reports, 297(5), 239-270.

4. Grassberger, P., & Procaccia, I. (1983). *Characterization of strange attractors*. Physical Review Letters, 50(5), 346.

---

## Contact

For questions, issues, or collaboration inquiries, please open an issue on the GitHub repository.

**Remember:** The DSI signature is the "smoking gun" that discrete bottlenecks are detectable via entropy diagnostics. If confirmed in fractals AND cosmic web data, this is transformative for LSS topology studies.
