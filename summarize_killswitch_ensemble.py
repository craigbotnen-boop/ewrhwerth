#!/usr/bin/env python3
"""
Ensemble Kill-Switch Summary Generator

This script aggregates results from multiple kill-switch runs (RANN × seed × smooth)
and computes statistical properties of the Recovery Score distribution.

The ensemble approach provides:
1. Mean and standard deviation of Recovery Score across realizations
2. Robustness check: Does the verdict change across parameter variations?
3. Statistical significance: Is the signal consistent or noisy?
4. Publication-ready summary with uncertainty quantification
"""

import argparse
import json
import sys
import glob
from pathlib import Path
import numpy as np


def compute_recovery_score(ds_data, ds_hon1, ds_recon):
    """
    Distance-based recovery score (same formula as analyze_killswitch_results.py).

    Returns
    -------
    float
        Recovery percentage (typically 0-100, but can be <0 or >100)
    """
    total_gap = abs(ds_hon1 - ds_data)
    if total_gap < 1e-9:
        return 0.0
    remaining_gap = abs(ds_recon - ds_data)
    score = (1.0 - (remaining_gap / total_gap)) * 100.0
    return score


def generate_ensemble_verdict(mean_score, std_score, min_score, max_score):
    """
    Generate ensemble verdict based on mean Recovery Score and consistency.

    Parameters
    ----------
    mean_score : float
        Mean recovery score across ensemble
    std_score : float
        Standard deviation of recovery scores
    min_score : float
        Minimum recovery score in ensemble
    max_score : float
        Maximum recovery score in ensemble

    Returns
    -------
    verdict : str
        One of: "RECOVERY", "INCONCLUSIVE", "KILL"
    confidence : str
        One of: "HIGH", "MEDIUM", "LOW"
    interpretation : str
        Detailed explanation
    """
    # Determine verdict based on mean score
    if mean_score > 50.0:
        verdict = "RECOVERY"
    elif mean_score >= 20.0:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "KILL"

    # Assess confidence based on consistency
    # High confidence: std < 10% and all runs agree on verdict
    # Medium confidence: std < 20% or some variation across runs
    # Low confidence: std > 20% or wildly inconsistent

    if std_score < 10.0 and (max_score - min_score) < 20.0:
        confidence = "HIGH"
        consistency_note = "All runs produce consistent Recovery Scores."
    elif std_score < 20.0:
        confidence = "MEDIUM"
        consistency_note = "Moderate variation across runs; verdict is stable."
    else:
        confidence = "LOW"
        consistency_note = "High variation across runs; results are sensitive to parameters."

    # Check if any runs cross verdict boundaries
    boundary_crossings = []
    for score in [min_score, max_score]:
        if score > 50.0:
            v = "RECOVERY"
        elif score >= 20.0:
            v = "INCONCLUSIVE"
        else:
            v = "KILL"
        if v != verdict:
            boundary_crossings.append(v)

    if boundary_crossings:
        confidence = "LOW"
        consistency_note = f"WARNING: Some runs yield different verdicts ({', '.join(set(boundary_crossings))})."

    # Generate interpretation
    interpretation = f"""
╔════════════════════════════════════════════════════════════════════╗
║                  ENSEMBLE KILL-SWITCH VERDICT                      ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Verdict: {verdict:^56s}  ║
║  Confidence: {confidence:^53s}  ║
║                                                                    ║
║  Recovery Score Statistics:                                        ║
║    Mean:   {mean_score:6.1f}% ± {std_score:5.1f}%                                          ║
║    Median: {np.nan:6.1f}% (computed from distribution)                     ║
║    Range:  [{min_score:6.1f}%, {max_score:6.1f}%]                                        ║
║                                                                    ║
║  {consistency_note:^67s}║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
"""

    return verdict, confidence, interpretation


def load_ensemble_results(input_dir):
    """
    Load all JSON results from ensemble runs.

    Returns
    -------
    results : list of dict
        List of result dictionaries from individual runs
    """
    pattern = str(Path(input_dir) / "killswitch_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No result files found matching: {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} ensemble result files")

    results = []
    failed = []

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)

            # Compute recovery score for this run
            ds_data = data["ds_star"]["data"]
            ds_hon1 = data["ds_star"]["hon1"]
            ds_recon = data["ds_star"]["recon"]
            recovery_score = compute_recovery_score(ds_data, ds_hon1, ds_recon)

            # Add recovery score to data
            data["recovery_score"] = recovery_score

            # Extract parameters for grouping
            params = data.get("params", {})
            data["rann"] = Path(fpath).stem.split("_")[1].replace("RANN", "")
            data["seed"] = params.get("hon1_seed", -1)
            data["smooth"] = params.get("smooth", -1.0)

            results.append(data)

        except Exception as e:
            print(f"WARNING: Failed to load {fpath}: {e}", file=sys.stderr)
            failed.append(fpath)

    print(f"Successfully loaded: {len(results)}/{len(files)}")
    if failed:
        print(f"Failed to load: {len(failed)} files", file=sys.stderr)

    return results


def compute_ensemble_statistics(results):
    """
    Compute statistical properties of the ensemble.

    Returns
    -------
    stats : dict
        Dictionary of statistical properties
    """
    # Extract arrays
    recovery_scores = np.array([r["recovery_score"] for r in results])
    ds_data = np.array([r["ds_star"]["data"] for r in results])
    ds_hon1 = np.array([r["ds_star"]["hon1"] for r in results])
    ds_recon = np.array([r["ds_star"]["recon"] for r in results])

    stats = {
        "n_runs": len(results),
        "recovery_score": {
            "mean": float(np.mean(recovery_scores)),
            "std": float(np.std(recovery_scores, ddof=1)),
            "median": float(np.median(recovery_scores)),
            "min": float(np.min(recovery_scores)),
            "max": float(np.max(recovery_scores)),
            "q25": float(np.percentile(recovery_scores, 25)),
            "q75": float(np.percentile(recovery_scores, 75)),
            "distribution": recovery_scores.tolist()
        },
        "ds_star": {
            "data": {
                "mean": float(np.mean(ds_data)),
                "std": float(np.std(ds_data, ddof=1))
            },
            "hon1": {
                "mean": float(np.mean(ds_hon1)),
                "std": float(np.std(ds_hon1, ddof=1))
            },
            "recon": {
                "mean": float(np.mean(ds_recon)),
                "std": float(np.std(ds_recon, ddof=1))
            }
        }
    }

    # Group by parameter variations
    stats["by_smooth"] = {}
    smooths = sorted(set(r["smooth"] for r in results))
    for sm in smooths:
        subset = [r["recovery_score"] for r in results if r["smooth"] == sm]
        stats["by_smooth"][f"{sm}"] = {
            "mean": float(np.mean(subset)),
            "std": float(np.std(subset, ddof=1)),
            "n": len(subset)
        }

    stats["by_rann"] = {}
    ranns = sorted(set(r["rann"] for r in results))
    for rann in ranns:
        subset = [r["recovery_score"] for r in results if r["rann"] == rann]
        stats["by_rann"][rann] = {
            "mean": float(np.mean(subset)),
            "std": float(np.std(subset, ddof=1)),
            "n": len(subset)
        }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ensemble kill-switch results"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing ensemble JSON outputs"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for ensemble summary"
    )
    args = parser.parse_args()

    # Load all ensemble results
    results = load_ensemble_results(args.input_dir)

    # Compute statistics
    stats = compute_ensemble_statistics(results)

    # Generate verdict
    verdict, confidence, interpretation = generate_ensemble_verdict(
        stats["recovery_score"]["mean"],
        stats["recovery_score"]["std"],
        stats["recovery_score"]["min"],
        stats["recovery_score"]["max"]
    )

    # Create summary report
    summary = {
        "verdict": verdict,
        "confidence": confidence,
        "statistics": stats,
        "notes": {
            "ensemble_size": stats["n_runs"],
            "description": "Audit-grade ensemble sweep over RANN realizations, HON-1 seeds, and smoothing scales",
            "recovery_score_formula": "100 * (1 - |d_s*[RECON] - d_s*[DATA]| / |d_s*[HON1] - d_s*[DATA]|)",
            "verdict_thresholds": {
                "recovery": "> 50%",
                "inconclusive": "20-50%",
                "kill": "< 20%"
            },
            "confidence_criteria": {
                "high": "std < 10% and no boundary crossings",
                "medium": "std < 20% or minor variations",
                "low": "std > 20% or verdict changes across runs"
            }
        }
    }

    # Save summary
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)

    # Display interpretation with updated median value
    interpretation = interpretation.replace(
        f"Median: {np.nan:6.1f}%",
        f"Median: {stats['recovery_score']['median']:6.1f}%"
    )
    print(interpretation)

    # Additional statistics
    print("\nDetailed Statistics:\n")
    print(f"Recovery Score Distribution:")
    print(f"  Mean:      {stats['recovery_score']['mean']:6.1f}% ± {stats['recovery_score']['std']:5.1f}%")
    print(f"  Median:    {stats['recovery_score']['median']:6.1f}%")
    print(f"  Range:     [{stats['recovery_score']['min']:6.1f}%, {stats['recovery_score']['max']:6.1f}%]")
    print(f"  IQR:       [{stats['recovery_score']['q25']:6.1f}%, {stats['recovery_score']['q75']:6.1f}%]")
    print()

    print("Breakdown by Smoothing Scale:")
    for sm, vals in stats["by_smooth"].items():
        print(f"  {sm:>5s} Mpc/h: {vals['mean']:6.1f}% ± {vals['std']:5.1f}%  (n={vals['n']})")
    print()

    print("Breakdown by RANN:")
    for rann, vals in stats["by_rann"].items():
        print(f"  RANN {rann}: {vals['mean']:6.1f}% ± {vals['std']:5.1f}%  (n={vals['n']})")
    print()

    print(f"\n[SAVED] Ensemble summary: {args.output}")

    # Exit with appropriate code based on verdict and confidence
    if verdict == "RECOVERY" and confidence == "HIGH":
        sys.exit(0)  # Strong discovery
    elif verdict == "KILL" and confidence == "HIGH":
        sys.exit(1)  # Strong rejection
    else:
        sys.exit(2)  # Inconclusive or low confidence


if __name__ == "__main__":
    main()
