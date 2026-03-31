"""
run_cox_survival.py
--------------------
Cox proportional-hazards survival engine for P9 SHIELD.

Inputs (all in the same directory):
    tcga_clinical_backbone.tsv  -- from build_tcga_clinical_backbone.py
    ds_values.tsv               -- two-column TSV: Patient_ID | d_s
                                   (spectral/diffusion dimension per patient)

Outputs:
    cox_model_A_results.tsv     -- univariate:  d_s vs OS
    cox_model_B_results.tsv     -- multivariate: d_s + Age + WHO2021 Grade
    cox_summary.txt             -- human-readable report for both models
    kaplan_meier_by_grade.png   -- KM curves stratified by WHO2021 Grade

Dependencies:
    pip install lifelines pandas matplotlib

Run from inside P9_SHIELD/:
    python run_cox_survival.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKBONE  = Path("tcga_clinical_backbone.tsv")
DS_VALUES = Path("ds_values.tsv")
OUT_A     = Path("cox_model_A_results.tsv")
OUT_B     = Path("cox_model_B_results.tsv")
OUT_SUM   = Path("cox_summary.txt")
OUT_KM    = Path("kaplan_meier_by_grade.png")

# ── 1. Load & merge ───────────────────────────────────────────────────────────
backbone = pd.read_csv(BACKBONE, sep="\t")
backbone["Patient_ID"] = backbone["Patient_ID"].astype(str).str.strip().str[:12]

if not DS_VALUES.exists():
    raise FileNotFoundError(
        f"{DS_VALUES} not found.\n"
        "Create a two-column TSV with headers Patient_ID and d_s "
        "from your spectral dimension pipeline, then re-run."
    )

ds = pd.read_csv(DS_VALUES, sep="\t")
ds["Patient_ID"] = ds["Patient_ID"].astype(str).str.strip().str[:12]

df = backbone.merge(ds[["Patient_ID", "d_s"]], on="Patient_ID", how="inner")
print(f"Post-merge N (backbone × d_s): {len(df)}")

# ── 2. Prepare survival columns ───────────────────────────────────────────────
df = df.dropna(subset=["OS_days", "Event", "d_s"]).copy()
df["OS_days"] = pd.to_numeric(df["OS_days"], errors="coerce")
df["Event"]   = pd.to_numeric(df["Event"],   errors="coerce")
df = df[df["OS_days"] > 0].copy()
print(f"Analysable rows (OS_days>0, d_s present): {len(df)}")

# ── 3. Grade encoding ─────────────────────────────────────────────────────────
# Treat "Unclassified" and NaN as a single reference category
grade_map = {
    "Glioblastoma":    "GBM",
    "Astrocytoma":     "Astrocytoma",
    "Oligodendroglioma": "Oligodendroglioma",
}
df["Grade_cat"] = df["Grade"].map(grade_map).fillna("Other")

# One-hot encode with GBM as reference (dropped)
grade_dummies = pd.get_dummies(df["Grade_cat"], prefix="Grade", drop_first=False)
for col in ["Grade_Astrocytoma", "Grade_Oligodendroglioma", "Grade_Other"]:
    if col not in grade_dummies.columns:
        grade_dummies[col] = 0
grade_dummies = grade_dummies[["Grade_Astrocytoma", "Grade_Oligodendroglioma", "Grade_Other"]]

df = pd.concat([df.reset_index(drop=True), grade_dummies.reset_index(drop=True)], axis=1)

# ── 4. Model A — Univariate: d_s vs OS ───────────────────────────────────────
print("\n── Model A: Univariate Cox (d_s ~ OS) ──────────────────────────────")
cph_A = CoxPHFitter()
cph_A.fit(
    df[["OS_days", "Event", "d_s"]],
    duration_col="OS_days",
    event_col="Event",
    show_progress=False
)
cph_A.print_summary()

res_A = cph_A.summary.copy()
res_A.to_csv(OUT_A, sep="\t")
print(f"Wrote {OUT_A}")

# ── 5. Model B — Multivariate: d_s + Age + Grade ─────────────────────────────
print("\n── Model B: Multivariate Cox (d_s + Age + Grade ~ OS) ──────────────")

model_b_cols = [
    "OS_days", "Event", "d_s", "Age",
    "Grade_Astrocytoma", "Grade_Oligodendroglioma", "Grade_Other"
]
df_B = df[model_b_cols].dropna().copy()
print(f"Model B N (complete cases): {len(df_B)}")

cph_B = CoxPHFitter()
cph_B.fit(
    df_B,
    duration_col="OS_days",
    event_col="Event",
    show_progress=False
)
cph_B.print_summary()

res_B = cph_B.summary.copy()
res_B.to_csv(OUT_B, sep="\t")
print(f"Wrote {OUT_B}")

# ── 6. Text summary ───────────────────────────────────────────────────────────
def fmt_row(name, hr, ci_low, ci_high, p):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return (f"  {name:<38} HR={hr:.3f}  95%CI [{ci_low:.3f}–{ci_high:.3f}]"
            f"  p={p:.4f}  {sig}")

lines = [
    "P9 SHIELD — Cox Survival Engine Results",
    "=" * 60,
    f"Analysable patients (Model A): {len(df)}",
    f"Analysable patients (Model B): {len(df_B)}",
    "",
    "Model A — Univariate: d_s",
    "-" * 40,
]
for cov in res_A.index:
    lines.append(fmt_row(
        cov,
        res_A.loc[cov, "exp(coef)"],
        res_A.loc[cov, "exp(coef) lower 95%"],
        res_A.loc[cov, "exp(coef) upper 95%"],
        res_A.loc[cov, "p"],
    ))

lines += [
    "",
    "Model B — Multivariate: d_s + Age + WHO2021 Grade",
    "  Reference: GBM",
    "-" * 40,
]
for cov in res_B.index:
    lines.append(fmt_row(
        cov,
        res_B.loc[cov, "exp(coef)"],
        res_B.loc[cov, "exp(coef) lower 95%"],
        res_B.loc[cov, "exp(coef) upper 95%"],
        res_B.loc[cov, "p"],
    ))

OUT_SUM.write_text("\n".join(lines))
print(f"\nWrote {OUT_SUM}")
print("\n" + "\n".join(lines))

# ── 7. Kaplan-Meier by Grade ──────────────────────────────────────────────────
grade_order  = ["GBM", "Astrocytoma", "Oligodendroglioma", "Other"]
grade_colors = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]

fig, ax = plt.subplots(figsize=(8, 5))
kmf = KaplanMeierFitter()

for grade, color in zip(grade_order, grade_colors):
    mask = df["Grade_cat"] == grade
    if mask.sum() < 5:
        continue
    kmf.fit(
        df.loc[mask, "OS_days"],
        event_observed=df.loc[mask, "Event"],
        label=f"{grade} (n={mask.sum()})"
    )
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

ax.set_title("Kaplan–Meier Overall Survival by WHO 2021 Grade\n(TCGA PanGlioma)", fontsize=12)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival probability")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(OUT_KM, dpi=150)
plt.close()
print(f"Wrote {OUT_KM}")

# ── 8. Log-rank test across grades ───────────────────────────────────────────
print("\n── Log-rank test (all grades) ──────────────────────────────────────")
lr = multivariate_logrank_test(
    df["OS_days"], df["Grade_cat"], df["Event"]
)
lr.print_summary()
print(f"Log-rank p = {lr.p_value:.6f}")
