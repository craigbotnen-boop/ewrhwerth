#!/usr/bin/env python3
"""
Kill-Switch Verdict Generator

This script analyzes the output from phase_reconstruction_recovery.py and
computes the **Recovery Score**, the ultimate metric that determines whether
cosmic web spectral dimension anomalies are real physics or artifacts.

Recovery Score = 100 * (d_s*[RECON] - d_s*[HON1]) / (d_s*[DATA] - d_s*[HON1])

Interpretation:
  >50%  → RECOVERY:      The discovery stands. Phase coherence is essential.
  20-50% → INCONCLUSIVE: Partial phase dependence; confounding factors present.
  <20%  → KILL:          Retire the theory. Phases don't matter.
"""

import argparse
import json
import sys
import numpy as np


def compute_recovery_score(ds_data, ds_hon1, ds_recon):
    """
    Compute the phase recovery score.

    If reconstruction recovers the original signal (d_s*[RECON] ≈ d_s*[DATA]),
    then the numerator ≈ denominator and Recovery Score ≈ 100%.

    If reconstruction fails like phase randomization (d_s*[RECON] ≈ d_s*[HON1]),
    then the numerator ≈ 0 and Recovery Score ≈ 0%.

    Parameters
    ----------
    ds_data : float
        Spectral dimension plateau value for original DATA
    ds_hon1 : float
        Spectral dimension plateau value for HON-1 (phase-randomized)
    ds_recon : float
        Spectral dimension plateau value for RECON (BAO-reconstructed)

    Returns
    -------
    recovery_score : float
        Recovery percentage (0-100+, can exceed 100% if recon overshoots)
    """
    denominator = ds_data - ds_hon1
    numerator = ds_recon - ds_hon1

    if abs(denominator) < 1e-6:
        # Degenerate case: DATA and HON1 are identical
        # This means phase randomization had no effect → phases don't matter
        return 0.0

    recovery_score = 100.0 * numerator / denominator
    return recovery_score


def generate_verdict(recovery_score):
    """
    Generate the kill-switch verdict based on recovery score thresholds.

    Returns
    -------
    verdict : str
        One of: "RECOVERY", "INCONCLUSIVE", "KILL"
    interpretation : str
        Detailed physical interpretation of the result
    """
    if recovery_score > 50.0:
        verdict = "RECOVERY"
        interpretation = """
🟢 **THE DISCOVERY STANDS.**

Phase coherence is the load-bearing ingredient of the cosmic web connectivity anomaly.
BAO reconstruction successfully recovers the spectral dimension signal, demonstrating
that the phases of Fourier modes encode essential information about the web's topology.

This is a genuine physical signal, not a measurement artifact.

**Scientific Implications:**
- The connectivity of the universe is written into the phases of δ_k
- Standard cosmological observables (P(k), ξ(r)) are insufficient to capture web topology
- Phase information is critical for understanding large-scale structure formation
- The "spectral horizon" phenomenon is a robust cosmological observable

**Next Steps:**
1. Prepare manuscript with this diagnostic as a robustness test
2. Investigate the physical origin of phase coherence in cosmic web formation
3. Explore implications for structure formation theory and dark matter models
4. Apply this methodology to other surveys (DESI, Euclid, etc.)
"""

    elif recovery_score >= 20.0:
        verdict = "INCONCLUSIVE"
        interpretation = """
🟡 **PARTIAL PHASE DEPENDENCE DETECTED.**

The signal shows some recovery with BAO reconstruction, suggesting phase information
plays a role, but other factors are contributing to the anomaly:

**Possible Contributing Factors:**
- Survey geometry and masking effects
- Non-linear structure formation (fingers-of-god, velocity dispersion)
- Incomplete reconstruction (wrong f, bias, or smoothing scale)
- Edge effects in graph construction from windowed density field
- Resolution limitations (try higher nmesh or different q threshold)

**Diagnostic Actions:**
1. Verify reconstruction parameters (f, bias) match your survey's cosmology
2. Test different smoothing scales (10-20 Mpc/h range)
3. Try RecIso vs RecSym conventions (--reciso flag)
4. Check for systematic errors in CIC meshing or density contrast calculation
5. Run with multiple HON-1 seeds to verify phase randomization stability
6. Compare results at different spatial resolutions (nmesh=64, 128, 256)

**Interpretation:**
The true signal likely lies between pure artifact and pure phase coherence.
More detailed analysis is needed before publication.
"""

    else:  # recovery_score < 20.0
        verdict = "KILL"
        interpretation = """
🔴 **RETIRE THE THEORY.**

Phase coherence does NOT explain the spectral dimension anomaly. BAO reconstruction
fails to recover the signal, performing no better than random phase scrambling.

**What This Means:**
- The cosmic web connectivity anomaly is NOT driven by phase information
- The signal is likely an artifact of graph construction, survey masking, or analysis choices
- The "spectral horizon" is a phantom phenomenon, not genuine physics

**Possible Artifact Sources:**
1. **Graph Construction Bias:**
   - Quantile threshold (q) creates artificial topology
   - Connectivity choice (6 vs 26) affects spectral properties
   - Finite mesh resolution introduces discretization artifacts

2. **Survey Window Effects:**
   - Non-periodic boundary conditions distort the density field
   - Masking creates spurious correlations in the voxel graph
   - Alpha correction (data/random normalization) introduces edge effects

3. **Spectral Dimension Calculation:**
   - Finite-size effects in eigenspectrum
   - Logarithmic derivative amplifies numerical noise
   - Plateau identification is subjective (choice of tmin, tmax)

**Recommendation:**
Do NOT publish this result as a cosmological discovery. Archive the analysis as a
cautionary tale about graph-based topology diagnostics in windowed survey data.

The honest map must remain honest. This kill-switch has spoken.
"""

    return verdict, interpretation


def format_summary_table(ds_data, ds_hon1, ds_recon, recovery_score, verdict):
    """
    Generate a clean summary table for the terminal output.
    """
    table = f"""
╔════════════════════════════════════════════════════════════════════╗
║           KILL-SWITCH VERDICT: PHASE-RECONSTRUCTION RECOVERY       ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Spectral Dimension Plateau (d_s*):                               ║
║    DATA (original)          : {ds_data:6.3f}                              ║
║    HON-1 (phase-randomized) : {ds_hon1:6.3f}                              ║
║    RECON (BAO-reconstructed): {ds_recon:6.3f}                              ║
║                                                                    ║
║  Phase Sensitivity:                                                ║
║    Δ(DATA - HON1) = {ds_data - ds_hon1:+6.3f}  [phase destruction impact]     ║
║    Δ(RECON - HON1) = {ds_recon - ds_hon1:+6.3f}  [phase recovery achieved]     ║
║                                                                    ║
║  ┌──────────────────────────────────────────────────────────────┐ ║
║  │  RECOVERY SCORE: {recovery_score:6.1f}%                                   │ ║
║  └──────────────────────────────────────────────────────────────┘ ║
║                                                                    ║
║  ╔════════════════════════════════════════════════════════════╗  ║
║  ║  VERDICT: {verdict:^48s}  ║  ║
║  ╚════════════════════════════════════════════════════════════╝  ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kill-switch results and generate verdict"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON file from phase_reconstruction_recovery.py"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to save detailed verdict report"
    )
    args = parser.parse_args()

    # Load results
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract spectral dimension plateau values
    ds_data = data["ds_star"]["data"]
    ds_hon1 = data["ds_star"]["hon1"]
    ds_recon = data["ds_star"]["recon"]

    # Compute recovery score
    recovery_score = compute_recovery_score(ds_data, ds_hon1, ds_recon)

    # Generate verdict
    verdict, interpretation = generate_verdict(recovery_score)

    # Display summary table
    print(format_summary_table(ds_data, ds_hon1, ds_recon, recovery_score, verdict))

    # Display interpretation
    print(interpretation)

    # Save detailed report if requested
    if args.output:
        report = {
            "verdict": verdict,
            "recovery_score_percent": recovery_score,
            "spectral_dimensions": {
                "data": ds_data,
                "hon1_phase_randomized": ds_hon1,
                "recon_bao": ds_recon
            },
            "deltas": {
                "phase_destruction_impact": ds_data - ds_hon1,
                "phase_recovery_achieved": ds_recon - ds_hon1
            },
            "interpretation": interpretation.strip(),
            "input_file": args.input,
            "input_params": data.get("params", {}),
            "thresholds": {
                "recovery": "> 50%",
                "inconclusive": "20-50%",
                "kill": "< 20%"
            }
        }

        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n[SAVED] Detailed verdict report: {args.output}")

    # Exit with appropriate code
    if verdict == "RECOVERY":
        sys.exit(0)  # Success
    elif verdict == "INCONCLUSIVE":
        sys.exit(2)  # Needs further investigation
    else:  # KILL
        sys.exit(1)  # Theory rejected


if __name__ == "__main__":
    main()
