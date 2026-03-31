"""
cox_shield_survival_v2.py
--------------------------
P9 SHIELD Cox survival engine — v2.0

Changes from v1:
  - Enforces R^2 >= 0.95 gate on physical network fits
  - Drops 'Unclassified' WHO2021 cases from Model B
  - Sets Oligodendroglioma as the reference category (favorable anchor)
  - Scales d_s by 10 → HR is interpreted per 0.1-unit change in d_s

Inputs (all in P9_SHIELD/):
    tcga_clinical_backbone.tsv   -- from build_tcga_clinical_backbone.py
    tcga_physics_results.csv     -- your local N=675 d_s output
                                    required columns: patient_id, d_s, r2

Outputs (printed to terminal + written to cox_summary_v2.txt):
    Model A HR (d_s, 95% CI, p)
    Model B HR (d_s adjusted, 95% CI, p)
    Concordance index for each model

Run from inside P9_SHIELD/:
    python cox_shield_survival_v2.py
"""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import warnings

warnings.filterwarnings("ignore")


def run_survival_sweep():
    print("--- P9 SHIELD SURVIVAL SWEEP v2.0 ---")

    # ── 1. Clinical backbone ──────────────────────────────────────────────────
    try:
        clin = pd.read_csv("tcga_clinical_backbone.tsv", sep="\t")
    except FileNotFoundError:
        clin = pd.read_csv("tcga_clinical_backbone.csv")

    # ── 2. Physical network results ───────────────────────────────────────────
    # Must contain: patient_id, d_s, r2
    try:
        phys = pd.read_csv("tcga_physics_results.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "[!] tcga_physics_results.csv not found.\n"
            "    Point this path at your local N=675 spectral-dimension output "
            "    (columns: patient_id, d_s, r2) and re-run."
        )

    df = pd.merge(
        clin, phys,
        left_on="bcr_patient_barcode",
        right_on="patient_id",
        how="inner"
    )

    # ── 3. R² gate ────────────────────────────────────────────────────────────
    initial_n = len(df)
    df = df[df["r2"] >= 0.95].copy()
    print(f"\nEnrollment after R² >= 0.95 gate: {len(df)} / {initial_n}")

    # ── 4. Survival endpoints ─────────────────────────────────────────────────
    df["OS_days"] = pd.to_numeric(df["OS.time"], errors="coerce")
    df["Event"]   = pd.to_numeric(df["OS"],      errors="coerce")
    df = df.dropna(subset=["OS_days", "Event", "d_s"])
    df = df[df["OS_days"] > 0].copy()

    # Scale: HR per 0.1-unit change in d_s
    df["d_s_scaled"] = df["d_s"] * 10

    print(f"Number of events      : {int(df['Event'].sum())}")
    print(f"d_s  mean ± SD        : {df['d_s'].mean():.3f} ± {df['d_s'].std():.3f}")

    summary_lines = [
        "P9 SHIELD Cox Survival Engine v2.0",
        "=" * 60,
        f"Post-gate N            : {len(df)}",
        f"Events                 : {int(df['Event'].sum())}",
        f"d_s mean ± SD          : {df['d_s'].mean():.3f} ± {df['d_s'].std():.3f}",
        "",
    ]

    # ── 5. Model A — Univariate ───────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("MODEL A: Continuous d_s vs OS (univariate)")
    print("=" * 50)

    df_a = df[["OS_days", "Event", "d_s_scaled"]].copy()
    cph_a = CoxPHFitter()
    try:
        cph_a.fit(df_a, duration_col="OS_days", event_col="Event")
        cph_a.print_summary(
            columns=["coef", "exp(coef)",
                     "exp(coef) lower 95%", "exp(coef) upper 95%", "p"],
            decimals=3
        )
        c_a = cph_a.concordance_index_
        print(f"Model A Concordance: {c_a:.3f}")

        r = cph_a.summary.loc["d_s_scaled"]
        summary_lines += [
            "Model A — Univariate: d_s (per 0.1 unit)",
            f"  HR = {r['exp(coef)']:.3f}  "
            f"95% CI [{r['exp(coef) lower 95%']:.3f}–{r['exp(coef) upper 95%']:.3f}]  "
            f"p = {r['p']:.4f}",
            f"  Concordance: {c_a:.3f}",
            "",
        ]
    except Exception as e:
        print(f"Model A error: {e}")
        summary_lines.append(f"Model A error: {e}\n")

    # ── 6. Model B — Multivariate ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("MODEL B: d_s + Age + WHO2021 Grade")
    print("  Reference: Oligodendroglioma (favorable anchor)")
    print("=" * 50)

    # Use the 'Grade' column from the backbone (simplified labels)
    grade_col = "Grade" if "Grade" in df.columns else "Grade_WHO2021"
    df_b = df.copy()

    # Normalise to lowercase for reliable filtering / reference setting
    df_b[grade_col] = df_b[grade_col].astype(str).str.strip().str.lower()
    df_b = df_b[df_b[grade_col] != "unclassified"].copy()
    df_b = df_b.dropna(subset=["Age", grade_col]).copy()
    print(f"Model B N (excl. Unclassified): {len(df_b)}")

    df_model_b = df_b[[
        "OS_days", "Event", "d_s_scaled", "Age", grade_col
    ]].copy()

    cph_b = CoxPHFitter()
    try:
        formula = (
            f"d_s_scaled + Age + "
            f"C({grade_col}, Treatment('oligodendroglioma'))"
        )
        cph_b.fit(
            df_model_b,
            duration_col="OS_days",
            event_col="Event",
            formula=formula
        )
        cph_b.print_summary(
            columns=["coef", "exp(coef)",
                     "exp(coef) lower 95%", "exp(coef) upper 95%", "p"],
            decimals=3
        )
        c_b = cph_b.concordance_index_
        print(f"Model B Concordance: {c_b:.3f}")

        r_b = cph_b.summary.loc["d_s_scaled"]
        summary_lines += [
            "Model B — Multivariate: d_s + Age + Grade (ref=Oligodendroglioma)",
            f"  N = {len(df_b)}",
            f"  d_s HR = {r_b['exp(coef)']:.3f}  "
            f"95% CI [{r_b['exp(coef) lower 95%']:.3f}–{r_b['exp(coef) upper 95%']:.3f}]  "
            f"p = {r_b['p']:.4f}",
            f"  Concordance: {c_b:.3f}",
            "",
        ]
    except Exception as e:
        print(f"Model B error: {e}")
        summary_lines.append(f"Model B error: {e}\n")

    # ── 7. Write summary ──────────────────────────────────────────────────────
    out = "\n".join(summary_lines)
    with open("cox_summary_v2.txt", "w") as f:
        f.write(out)
    print("\n--- Summary written to cox_summary_v2.txt ---")
    print(out)


if __name__ == "__main__":
    run_survival_sweep()
