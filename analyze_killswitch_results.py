#!/usr/bin/env python3
"""
Kill-Switch Verdict Generator

This script analyzes the output from phase_reconstruction_recovery.py and
computes the **Recovery Score**, the ultimate metric that determines whether
cosmic web spectral dimension anomalies are phase-driven.

Recovery Score = 100 * (1 - |d_s*[RECON] - d_s*[DATA]| / |d_s*[HON1] - d_s*[DATA]|)

Distance-based closure metric:
  100% = RECON closes the gap (RECON ≈ DATA)
  0%   = RECON does not move toward DATA (RECON ≈ HON1)
  <0%  = RECON moves away from DATA (worse than HON1)

Interpretation:
  >50%  → RECOVERY:      Phase information is relevant to this statistic.
  20-50% → INCONCLUSIVE: Partial phase dependence; confounding factors present.
  <20%  → KILL:          This reconstruction does not close the gap.
"""

import argparse
import json
import sys
import numpy as np


def compute_recovery_score(ds_data, ds_hon1, ds_recon):
    """
    Compute the phase recovery score using distance-based closure metric.

    This robust formula avoids sign ambiguities and treats the problem as:
    "How much of the gap between HON1 and DATA did RECON close?"

    100% = RECON is identical to DATA (Full Recovery)
    0%   = RECON is identical to HON-1 (No Recovery)
    < 0% = RECON moved further away from DATA than the Null was

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
        Recovery percentage (0-100 for partial recovery, can be <0 or >100)
    """
    total_gap = abs(ds_hon1 - ds_data)

    if total_gap < 1e-9:
        # Degenerate case: DATA and HON1 are identical
        # Phase randomization had no effect → phases don't matter
        return 0.0

    remaining_gap = abs(ds_recon - ds_data)

    # 1.0 means remaining_gap is 0 (Perfect recovery)
    # 0.0 means remaining_gap equals total_gap (No movement toward DATA)
    # Negative means RECON moved away from DATA (worse than HON1)
    score = (1.0 - (remaining_gap / total_gap)) * 100.0

    return score


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
🟢 **PHASE INFORMATION IS RELEVANT TO THIS STATISTIC.**

BAO reconstruction (pyrecon) moves d_s* from HON-1 toward DATA under the survey window,
closing >50% of the gap created by phase randomization. This demonstrates that phase
coherence in the density field is a load-bearing ingredient of the spectral dimension signal.

**Bounded Interpretation:**
- Phase information encoded in δ_k affects the graph-based spectral dimension
- Standard two-point statistics P(k), ξ(r) do not capture this phase dependence
- The reconstruction's displacement field (based on large-scale coherent flows) recovers
  topological connectivity properties that are destroyed by phase randomization

**Important Caveats:**
- This does NOT prove "we recovered the true primordial phases" (reconstruction is
  a specific linear displacement operation, not a full Bayesian phase posterior)
- The signal is measured under survey windowing and selection; edge effects may contribute
- Robustness should be verified across multiple RANN realizations and HON-1 seeds

**Next Steps:**
1. Verify stability across the ensemble (multiple seeds, RANN, smoothing scales)
2. Investigate physical origin: are phases coherent due to gravitational collapse dynamics?
3. Test on mock catalogs with known phase structure for validation
4. Apply to other surveys (DESI, Euclid) to confirm universality
5. Prepare manuscript with this diagnostic as a key robustness test
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
🔴 **THIS SPECIFIC RECONSTRUCTION DOES NOT CLOSE THE GAP.**

BAO reconstruction fails to recover the spectral dimension signal, performing no
better than random phase scrambling. The gap between DATA and HON-1 is not closed
by pyrecon's displacement-based phase recovery.

**Bounded Interpretation:**
- This specific reconstruction method (pyrecon with these parameters) does not
  move d_s* from HON-1 toward DATA
- Phase information may still be present but not accessible via linear displacement fields
- The anomaly may be driven by factors other than large-scale coherent phase structure

**This DOES NOT automatically prove:**
- "The signal is a pure artifact" (other phase recovery methods might work)
- "Survey masking is the sole cause" (requires dedicated masking tests)
- "Graph construction is artificial" (requires comparison with alternative methods)

**Most Likely Explanations (in order):**
1. **Phase structure is sub-linear scale:**
   - Small-scale coherence not recovered by large-scale displacement fields
   - Non-linear structure formation dominates over linear reconstruction

2. **Survey Window and Selection:**
   - Non-periodic boundaries create edge effects
   - Selection function (min_rand threshold) may be inadequate
   - Alpha correction introduces systematic topology changes

3. **Graph Construction Choices:**
   - Quantile threshold q creates artificial connectivity
   - Connectivity rule (6 vs 26 neighbors) affects eigenspectrum
   - Mesh resolution too coarse to capture relevant scales

**Recommendation:**
Do NOT publish the d_s* anomaly as a phase-driven cosmological discovery without
further controls. Required follow-up:
- Test alternative reconstruction methods (Gaussian random field realizations, N-body mocks)
- Investigate masking effects with controlled synthetic catalogs
- Validate graph construction against alternative topology diagnostics

The honest map must remain honest. This kill-switch indicates caution is warranted.
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
