# 🎯 The Kill-Switch: Phase-Reconstruction Recovery Test

## Executive Summary

This is the **final diagnostic** to determine whether cosmic web spectral dimension anomalies represent genuine physics or measurement artifacts.

**The Question:** Is the connectivity of the universe encoded in the **phases** of Fourier modes, or is the observed spectral dimension anomaly a phantom produced by graph construction, survey windowing, or other systematics?

**The Test:** Three conditions are compared:
1. **DATA** — Original survey density field
2. **HON-1** — Phase-randomized field (destroys phases, preserves power spectrum)
3. **RECON** — BAO-reconstructed field (attempts to recover phase information)

**The Verdict:**
- If **RECON** recovers the signal → **phases matter** → discovery stands
- If **RECON** fails like **HON-1** → **phases don't matter** → retire the theory

---

## Scientific Background

### The Spectral Dimension Anomaly

Recent analysis of large-scale structure surveys has revealed an unexpected feature in the spectral dimension of the cosmic web voxel graph:

```
d_s(t) = -2 d ln P(t) / d ln t
```

where `P(t) = ⟨exp(-t λ_i)⟩` is the return probability on the graph Laplacian spectrum.

**Observation:** The plateau value `d_s*` measured from survey data appears **anomalously low** compared to simulations, suggesting a "spectral horizon" — a breakdown of effective dimensionality at large scales.

### The Phase Coherence Hypothesis

Standard cosmological observables (power spectrum `P(k)`, correlation function `ξ(r)`) depend only on the **amplitudes** of Fourier modes. They are blind to phase information.

**Hypothesis:** The cosmic web's topological connectivity is encoded in the **phases** of `δ_k`, not just the amplitudes. If true:
- Phase randomization should **destroy** the spectral dimension signal
- Phase-informed reconstruction should **recover** it

### Why This Matters

If the anomaly is **phase-driven**, it represents:
- A new cosmological observable inaccessible to traditional two-point statistics
- Evidence that structure formation imprints coherent phase relationships
- Potential constraints on dark matter models and modified gravity theories

If the anomaly is **not phase-driven**, it is likely:
- An artifact of graph construction methodology
- A consequence of survey geometry and masking
- A finite-volume effect or numerical artifact

**This test is the tie-breaker.**

---

## Methodology

### 1. Phase Randomization (HON-1)

**Higher-Order Null Level 1:** Destroy phase information while preserving the power spectrum.

```python
δ̃_k = |δ_k| · exp(i φ_random)
```

This operation:
- Preserves `P(k) = ⟨|δ_k|²⟩` exactly
- Destroys all phase relationships between modes
- Maintains Gaussianity if the original field was Gaussian

**Critical Detail:** For real-space fields, special care is required at DC and Nyquist frequencies to enforce conjugate symmetry `δ*_{-k} = δ_k`.

### 2. BAO Reconstruction (RECON)

**Phase-informed displacement field recovery** using iterative reconstruction algorithms (pyrecon).

The reconstruction solves for the displacement field `Ψ` that approximately inverts Zel'dovich dynamics:

```
δ_obs(x) ≈ δ_L(x - Ψ(x))
```

where `δ_L` is the linear-theory density field. This operation:
- **Recovers phase coherence** encoded in the large-scale density field
- Sharpens BAO features (primary purpose for surveys like DESI)
- Provides a "deconvolved" view of the initial conditions

**Two conventions tested:**
- **RecSym:** Symmetric reconstruction (default pyrecon behavior)
- **RecIso:** Isotropic reconstruction (alternative for non-periodic boxes)

### 3. Recovery Score Calculation

```
Recovery Score = 100 × (d_s*[RECON] - d_s*[HON1]) / (d_s*[DATA] - d_s*[HON1])
```

**Interpretation:**
- **100%:** Perfect recovery — RECON fully restores the original signal
- **0%:** No recovery — RECON performs identically to phase randomization
- **> 100%:** Over-recovery — reconstruction overshoots (possible with aggressive smoothing)

---

## Usage

### Prerequisites

Install required packages:
```bash
pip install numpy scipy matplotlib pyrecon
```

The `pyrecon` package is from DESI collaboration:
```bash
git clone https://github.com/cosmodesi/pyrecon.git
cd pyrecon
pip install -e .
```

### Prepare Input Data

You need survey data in `.npy` format:

```python
# Data galaxy/halo positions (N, 3) in Mpc/h
data_pos = np.array([[x1, y1, z1], [x2, y2, z2], ...])
np.save("survey_data_positions.npy", data_pos)

# Random catalog positions (M, 3) in Mpc/h
rand_pos = np.array([[x1, y1, z1], [x2, y2, z2], ...])
np.save("survey_random_positions.npy", rand_pos)

# Optional: weights (e.g., FKP weights, completeness corrections)
data_w = np.ones(N)  # or your actual weights
rand_w = np.ones(M)
np.save("survey_data_weights.npy", data_w)
np.save("survey_random_weights.npy", rand_w)
```

### Run on Cluster

1. **Edit `run_killswitch.sbatch`:**
   - Update `#SBATCH` directives for your cluster (partition, time, memory)
   - Set paths to your data files (lines 45-48)
   - Adjust cosmological parameters (`F`, `BIAS`) to match your survey

2. **Submit job:**
   ```bash
   sbatch run_killswitch.sbatch
   ```

3. **Monitor progress:**
   ```bash
   tail -f killswitch_*.out
   ```

### Run Locally (for testing with small datasets)

```bash
python phase_reconstruction_recovery.py \
    --data-pos survey_data_positions.npy \
    --rand-pos survey_random_positions.npy \
    --nmesh 64 \
    --boxsize 1000.0 \
    --q 0.95 \
    --f 0.8 \
    --bias 2.0 \
    --smooth 15.0 \
    --out killswitch_result.json

python analyze_killswitch_results.py --input killswitch_result.json
```

---

## Interpreting Results

### Verdict Thresholds

| Recovery Score | Verdict | Meaning |
|---------------|---------|---------|
| **> 50%** | 🟢 **RECOVERY** | **The discovery stands.** Phase coherence is essential. Publish with confidence. |
| **20-50%** | 🟡 **INCONCLUSIVE** | Partial phase dependence detected. Investigate confounding factors before publication. |
| **< 20%** | 🔴 **KILL** | **Retire the theory.** Phases don't matter. The anomaly is an artifact. Do not publish. |

### Example Outputs

**Scenario 1: Discovery Confirmed**
```
Recovery Score: 87.3%
Verdict: RECOVERY

d_s* DATA  = 1.84
d_s* HON-1 = 2.52  [+0.68 from phase destruction]
d_s* RECON = 1.91  [-0.61 from phase recovery]
```
**Interpretation:** Reconstruction recovers 87% of the phase-dependent signal. The spectral dimension anomaly is genuine physics.

**Scenario 2: Theory Rejected**
```
Recovery Score: 8.2%
Verdict: KILL

d_s* DATA  = 1.84
d_s* HON-1 = 2.52  [+0.68 from phase destruction]
d_s* RECON = 2.47  [-0.05 from phase recovery]
```
**Interpretation:** Reconstruction fails to recover the signal. The anomaly is not phase-driven — likely an artifact of graph construction or survey masking.

**Scenario 3: Needs Investigation**
```
Recovery Score: 35.1%
Verdict: INCONCLUSIVE

d_s* DATA  = 1.84
d_s* HON-1 = 2.52  [+0.68 from phase destruction]
d_s* RECON = 2.28  [-0.24 from phase recovery]
```
**Interpretation:** Partial recovery suggests phase information plays a role, but confounding factors (wrong reconstruction parameters, masking effects, resolution limitations) are present.

---

## Troubleshooting

### Common Issues

1. **"Too few web voxels" error:**
   - Lower the quantile threshold: `--q 0.95` or `--q 0.93`
   - Increase mesh resolution: `--nmesh 256`

2. **"Graph too large" error:**
   - Raise the quantile threshold: `--q 0.98`
   - Reduce mesh resolution: `--nmesh 64`

3. **Unexpected Recovery Score > 100%:**
   - Reconstruction may be over-smoothing
   - Try smaller smoothing radius: `--smooth 10.0`
   - Check if reconstruction parameters (f, bias) match your survey

4. **Recovery Score near zero despite visual differences:**
   - Verify that `d_s*` plateau identification is robust
   - Try explicit `--tmin` and `--tmax` for plateau window
   - Check for numerical instabilities in spectral dimension calculation

### Parameter Tuning Guide

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `nmesh` | 64-256 | Higher = finer resolution, more memory |
| `q` | 0.93-0.98 | Higher = fewer voxels, sparser graph |
| `f` | 0.5-1.0 | Growth rate (cosmology-dependent) |
| `bias` | 1.5-3.0 | Linear bias (tracer-dependent) |
| `smooth` | 10-25 Mpc/h | Reconstruction smoothing scale |

---

## Physical Interpretation of Verdicts

### If RECOVERY (> 50%):

The cosmic web's spectral dimension anomaly is driven by **coherent phase relationships** in the Fourier decomposition of the density field. This has profound implications:

1. **New Cosmological Observable:** Traditional analyses (power spectrum, correlation function) miss critical information encoded in phases.

2. **Structure Formation Imprint:** The gravitational collapse process creates non-random phase coherence that survives to z=0.

3. **Topological Information:** Graph-based diagnostics (spectral dimension, persistent homology) capture information inaccessible to traditional statistics.

4. **Theoretical Puzzles:**
   - Why do phases remain coherent after non-linear evolution?
   - What is the physical origin of the spectral horizon?
   - Can this constrain dark matter/modified gravity models?

**Action:** Prepare manuscript, investigate physical mechanisms, apply to other surveys.

### If INCONCLUSIVE (20-50%):

The signal shows **partial phase dependence**, but confounding factors complicate interpretation:

**Possible Explanations:**
- Survey geometry effects not fully corrected by alpha normalization
- Non-linear evolution not fully captured by linear reconstruction
- Resolution limitations or graph construction choices
- Incomplete reconstruction (wrong parameters or methodology)

**Action:** Systematic investigation of each confounding factor before publication. This regime demands **maximum scientific honesty** — don't cherry-pick the interpretation.

### If KILL (< 20%):

The spectral dimension anomaly is **NOT driven by phase coherence**. It is an artifact.

**Likely Culprits:**
1. **Graph Construction Bias:** The voxel graph connectivity is artificially imposed by quantile thresholding and neighbor rules
2. **Survey Masking:** Non-periodic boundary conditions and irregular survey geometry create spurious topology
3. **Finite-Size Effects:** The eigenspectrum of small graphs is dominated by edge effects
4. **Numerical Artifacts:** Logarithmic derivatives amplify noise; plateau identification is subjective

**Action:** **Do not publish as cosmological discovery.** Archive the analysis as a lesson in the dangers of interpreting graph-based diagnostics in windowed survey data.

**The honest map must remain honest.** This kill-switch has provided the definitive answer.

---

## Citation

If this methodology proves useful for your research, please cite:

```bibtex
@software{killswitch2025,
  title = {Phase-Reconstruction Recovery Kill-Switch for Cosmic Web Topology},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-repo]/killswitch}
}
```

---

## Files in This Package

| File | Purpose |
|------|---------|
| `phase_reconstruction_recovery.py` | Main analysis script (DATA, HON-1, RECON comparison) |
| `run_killswitch.sbatch` | SLURM batch script for cluster deployment |
| `analyze_killswitch_results.py` | Verdict generator (computes Recovery Score) |
| `KILLSWITCH_README.md` | This documentation |

---

## Acknowledgments

This methodology builds on:
- **pyrecon** (DESI collaboration) for BAO reconstruction algorithms
- **Spectral dimension theory** from fractal geometry and quantum gravity
- **Graph Laplacian methods** from topological data analysis

The "kill-switch" philosophy: **Delegate the final judgment to the data, not to hope.**

---

## Contact

For questions, issues, or collaboration inquiries, please open an issue on the GitHub repository.

**Remember:** The sober truth is always better than a beautiful lie.
