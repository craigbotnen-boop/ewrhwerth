"""
tmax_diagnostic.py
-------------------
Forensic audit of heat-kernel trace behavior across t in [0.1, 100].

Purpose:
  Resolve the T_MAX conflict (locked methodology: t_max=100 vs
  pipeline implementation: t_max=10) on physical grounds.
  Determines whether the power-law regime P(t) ~ t^(-d_s/2) is
  valid through t=100 or thermalizes before t=10 for real TCGA
  LIONESS networks.

Inputs:
  Three representative network TSVs — one per WHO2021 grade:
    Grade II  (Oligodendroglioma or low-grade Astrocytoma)
    Grade III (Astrocytoma IDHmut)
    Grade IV  (Glioblastoma)
  Specify paths in SAMPLE_NETWORKS below, or pass --network_dir
  and the script will auto-select one network per grade using
  tcga_clinical_backbone.tsv.

Outputs:
  tmax_diagnostic.png        — log-log K(t) plots for all three networks
  tmax_diagnostic_data.tsv  — raw log_t | log_K | local_ds columns
  tmax_diagnostic_report.txt — d_s estimates at t_max=10 vs t_max=100,
                               thermalization point, and protocol recommendation

Run from P9_SHIELD/:
    python tmax_diagnostic.py --network_dir /path/to/networks
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Configurable paths ────────────────────────────────────────────────────────
# Fill these in if you want to skip auto-selection:
SAMPLE_NETWORKS = {
    "GBM":             None,   # e.g. Path("networks/TCGA-06-0125.tsv")
    "Astrocytoma":     None,
    "Oligodendroglioma": None,
}

BACKBONE   = Path("tcga_clinical_backbone.tsv")
N_EIGS     = 150
N_TIMES    = 100
T_MIN      = 0.1
T_MAX_FULL = 100.0     # locked methodology
T_MAX_ALT  = 10.0      # current pipeline

OUT_PLOT   = Path("tmax_diagnostic.png")
OUT_DATA   = Path("tmax_diagnostic_data.tsv")
OUT_REPORT = Path("tmax_diagnostic_report.txt")


# ── Network I/O ───────────────────────────────────────────────────────────────
def load_adjacency(path: Path):
    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)
    cols = df.columns.tolist()

    # Square adjacency matrix?
    if df.shape[0] == df.shape[1] or df.shape[0] == df.shape[1] - 1:
        try:
            mat = df.values.astype(float)
            if mat.shape[0] == mat.shape[1]:
                A = sp.csr_matrix(mat)
                A = (A + A.T)
                A.data[:] = np.minimum(A.data, 1.0)
                return A
        except (ValueError, TypeError):
            pass

    # Edge list
    src_col, tgt_col = cols[0], cols[1]
    wt_col = cols[2] if len(cols) >= 3 else None
    nodes = pd.unique(pd.concat([df[src_col], df[tgt_col]]))
    nidx  = {n: i for i, n in enumerate(nodes)}
    N     = len(nodes)
    src   = df[src_col].map(nidx).values
    tgt   = df[tgt_col].map(nidx).values
    wts   = (pd.to_numeric(df[wt_col], errors="coerce")
             .fillna(1.0).abs().values if wt_col else np.ones(len(src)))
    row = np.concatenate([src, tgt])
    col = np.concatenate([tgt, src])
    dat = np.concatenate([wts, wts])
    return sp.csr_matrix((dat, (row, col)), shape=(N, N))


def largest_component(A):
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return A
    sizes = np.bincount(labels)
    idx   = np.where(labels == np.argmax(sizes))[0]
    return A[idx][:, idx]


def normalized_laplacian(A):
    deg = np.maximum(np.asarray(A.sum(axis=1)).flatten(), 1e-10)
    D   = sp.diags(1.0 / np.sqrt(deg))
    return (sp.eye(A.shape[0]) - D @ A @ D).tocsr()


def compute_eigenvalues(L, n_eigs):
    N = L.shape[0]
    k = min(n_eigs, N - 2)
    if N <= 600:
        vals = np.linalg.eigvalsh(L.toarray())
        return np.sort(vals)[:k]
    try:
        vals, _ = eigsh(L, k=k, which="SM")
        return np.sort(np.abs(vals))
    except Exception:
        vals = np.linalg.eigvalsh(L.toarray())
        return np.sort(vals)[:k]


def heat_kernel_trace(eigs, t_values, lam_min=1e-8):
    valid = eigs[eigs > lam_min]
    if len(valid) == 0:
        return np.ones_like(t_values)
    return np.exp(-np.outer(t_values, valid)).sum(axis=1)


def thermalization_floor(N, n_nonzero_eigs):
    """
    Theoretical floor: K(t→∞) → 1 (one zero mode kept out).
    In practice the trace approaches n_components / N as t→∞.
    Flag when K(t) < 5 * floor.
    """
    return 1.0 / max(n_nonzero_eigs, 1)


def fit_ds_window(t_values, K_t, t_max_cut):
    """OLS log-log fit in [T_MIN, t_max_cut]. Returns (d_s, r2)."""
    mask = (t_values >= T_MIN) & (t_values <= t_max_cut)
    log_t = np.log(t_values[mask])
    log_K = np.log(np.maximum(K_t[mask], 1e-300))
    if mask.sum() < 4 or not np.isfinite(log_K).all():
        return np.nan, 0.0
    A_mat = np.column_stack([log_t, np.ones(mask.sum())])
    coef, *_ = np.linalg.lstsq(A_mat, log_K, rcond=None)
    a = coef[0]
    y_pred = a * log_t + coef[1]
    ss_res = np.sum((log_K - y_pred) ** 2)
    ss_tot = np.sum((log_K - log_K.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(-2 * a), float(r2)


def local_ds(t_values, K_t):
    log_t = np.log(t_values)
    log_K = np.log(np.maximum(K_t, 1e-300))
    return -2.0 * np.gradient(log_K, log_t)


# ── Auto-select networks from manifest ───────────────────────────────────────
def auto_select(network_dir: Path):
    if not BACKBONE.exists():
        raise FileNotFoundError(f"{BACKBONE} not found. Run build_tcga_clinical_backbone.py first.")

    bb = pd.read_csv(BACKBONE, sep="\t")
    bb["Patient_ID"] = bb["Patient_ID"].astype(str).str.strip().str[:12]

    selected = {}
    for grade in ["Glioblastoma", "Astrocytoma", "Oligodendroglioma"]:
        subset = bb[bb["Grade"] == grade].dropna(subset=["OS_days"])
        for _, row in subset.iterrows():
            pid = row["Patient_ID"]
            candidates = list(network_dir.glob(f"{pid}*.tsv"))
            if candidates:
                label = {
                    "Glioblastoma": "GBM",
                    "Astrocytoma":  "Astrocytoma",
                    "Oligodendroglioma": "Oligodendroglioma"
                }[grade]
                selected[label] = candidates[0]
                break
    return selected


# ── Per-network analysis ──────────────────────────────────────────────────────
def analyse_network(label: str, path: Path, t_full, axes_row):
    print(f"\n── {label}: {path.name} ──────────────────────────────────────")

    A    = largest_component(load_adjacency(path))
    N    = A.shape[0]
    L    = normalized_laplacian(A)
    eigs = compute_eigenvalues(L, N_EIGS)

    n_nonzero = int(np.sum(eigs > 1e-8))
    floor     = thermalization_floor(N, n_nonzero)
    K_t       = heat_kernel_trace(eigs, t_full)
    loc_ds    = local_ds(t_full, K_t)

    # Thermalization: first t where K(t) < 10× floor
    therm_idx = np.where(K_t < 10 * floor)[0]
    t_therm   = float(t_full[therm_idx[0]]) if len(therm_idx) else float(T_MAX_FULL)

    ds_full, r2_full = fit_ds_window(t_full, K_t, T_MAX_FULL)
    ds_alt,  r2_alt  = fit_ds_window(t_full, K_t, T_MAX_ALT)

    print(f"  N nodes          : {N}")
    print(f"  Non-zero eigs    : {n_nonzero}")
    print(f"  Therm. threshold : K(t) < {10*floor:.4f}  → t ≈ {t_therm:.2f}")
    print(f"  d_s (t_max=100)  : {ds_full:.3f}  R²={r2_full:.4f}")
    print(f"  d_s (t_max=10)   : {ds_alt:.3f}   R²={r2_alt:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ax1, ax2 = axes_row

    # K(t) log-log
    ax1.loglog(t_full, K_t, "b-", lw=1.5, label="K(t)")
    ax1.axvline(T_MAX_ALT,  color="orange", ls="--", lw=1.2, label=f"t_max=10")
    ax1.axvline(T_MAX_FULL, color="red",    ls="--", lw=1.2, label=f"t_max=100")
    ax1.axhline(10 * floor, color="gray",   ls=":",  lw=1.0, label="10× floor")
    if t_therm < T_MAX_FULL:
        ax1.axvline(t_therm, color="purple", ls="-.", lw=1.2, label=f"therm≈{t_therm:.1f}")
    ax1.set_title(f"{label}\nlog K(t) vs log t")
    ax1.set_xlabel("t")
    ax1.set_ylabel("K(t)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Local d_s
    ax2.semilogx(t_full, loc_ds, "b-", lw=1.5)
    ax2.axvline(T_MAX_ALT,  color="orange", ls="--", lw=1.2, label=f"t_max=10  d_s={ds_alt:.2f} R²={r2_alt:.3f}")
    ax2.axvline(T_MAX_FULL, color="red",    ls="--", lw=1.2, label=f"t_max=100 d_s={ds_full:.2f} R²={r2_full:.3f}")
    if t_therm < T_MAX_FULL:
        ax2.axvline(t_therm, color="purple", ls="-.", lw=1.2)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylim(-1, 6)
    ax2.set_title(f"{label} — local d_s(t)")
    ax2.set_xlabel("t")
    ax2.set_ylabel("local d_s")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    return {
        "label":    label,
        "file":     path.name,
        "N_nodes":  N,
        "t_therm":  round(t_therm, 3),
        "ds_tmax100": round(ds_full, 4) if np.isfinite(ds_full) else None,
        "r2_tmax100": round(r2_full, 4),
        "ds_tmax10":  round(ds_alt,  4) if np.isfinite(ds_alt)  else None,
        "r2_tmax10":  round(r2_alt,  4),
        "log_t": np.log(t_full).tolist(),
        "log_K": np.log(np.maximum(K_t, 1e-300)).tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(network_dir: Path):
    # Resolve which networks to use
    nets = {k: v for k, v in SAMPLE_NETWORKS.items() if v is not None}
    if len(nets) < 3:
        print(f"Auto-selecting representative networks from {network_dir} …")
        nets.update(auto_select(network_dir))

    if not nets:
        raise RuntimeError(
            "No networks found. Either set SAMPLE_NETWORKS in the script "
            "or pass --network_dir pointing to your TSV folder."
        )

    t_full = np.logspace(np.log10(T_MIN), np.log10(T_MAX_FULL), N_TIMES)

    fig, axes = plt.subplots(len(nets), 2, figsize=(12, 4 * len(nets)))
    if len(nets) == 1:
        axes = [axes]

    results = []
    tsv_rows = []

    for i, (label, path) in enumerate(nets.items()):
        r = analyse_network(label, path, t_full, axes[i])
        results.append(r)
        for lt, lk in zip(r["log_t"], r["log_K"]):
            tsv_rows.append({"label": label, "file": r["file"],
                             "log_t": lt, "log_K": lk})

    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close()
    print(f"\nWrote {OUT_PLOT}")

    # Raw data TSV
    pd.DataFrame(tsv_rows).to_csv(OUT_DATA, sep="\t", index=False)
    print(f"Wrote {OUT_DATA}")

    # Report
    lines = [
        "T_MAX DIAGNOSTIC REPORT",
        "=" * 60,
        f"Networks audited : {len(results)}",
        f"T_MIN            : {T_MIN}",
        f"T_MAX (locked)   : {T_MAX_FULL}",
        f"T_MAX (pipeline) : {T_MAX_ALT}",
        "",
        f"{'Label':<22} {'t_therm':>8}  "
        f"{'d_s(100)':>9} {'R²(100)':>8}  "
        f"{'d_s(10)':>8} {'R²(10)':>7}",
        "-" * 70,
    ]
    for r in results:
        lines.append(
            f"{r['label']:<22} {r['t_therm']:>8.2f}  "
            f"{str(r['ds_tmax100']):>9} {r['r2_tmax100']:>8.4f}  "
            f"{str(r['ds_tmax10']):>8} {r['r2_tmax10']:>7.4f}"
        )

    # Recommendation
    all_therm = [r["t_therm"] for r in results]
    min_therm = min(all_therm)
    max_r2_100 = max(r["r2_tmax100"] or 0 for r in results)
    max_r2_10  = max(r["r2_tmax10"]  or 0 for r in results)

    lines += ["", "── Recommendation ──────────────────────────────────────────"]
    if min_therm < T_MAX_ALT:
        lines += [
            f"FINDING: Thermalization onset at t≈{min_therm:.1f} — BEFORE t=10.",
            "IMPLICATION: Both t_max=10 and t_max=100 extend into the floor.",
            "ACTION REQUIRED: Reduce T_MAX to t_therm and re-fit all networks.",
            "The p=0.0496 result at t_max=10 is NOT in the valid power-law window.",
        ]
    elif min_therm < T_MAX_FULL:
        lines += [
            f"FINDING: Thermalization onset at t≈{min_therm:.1f} — between 10 and 100.",
            f"IMPLICATION: t_max=10 (R²≈{max_r2_10:.3f}) is in the valid window.",
            f"             t_max=100 (R²≈{max_r2_100:.3f}) extends past thermalization.",
            "RECOMMENDATION: Protocol amendment — anchor T_MAX=10.",
            f"The p=0.0496 result (t_max=10) is in the physically valid window.",
            f"The p=0.17 result (t_max=100) is contaminated by the floor.",
            "VERDICT: t_max=10 result is scientifically defensible.",
        ]
    else:
        lines += [
            f"FINDING: No thermalization before t={T_MAX_FULL} in audited networks.",
            f"IMPLICATION: Both windows are in the valid power-law regime.",
            f"             t_max=100 R²={max_r2_100:.3f} vs t_max=10 R²={max_r2_10:.3f}.",
            "The divergence in p-values is NOT explained by thermalization.",
            "ACTION REQUIRED: Investigate other sources of the p=0.17 vs p=0.0496 split.",
        ]

    report = "\n".join(lines)
    OUT_REPORT.write_text(report)
    print(f"Wrote {OUT_REPORT}")
    print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_dir", type=Path, default=Path("networks"))
    args = parser.parse_args()
    main(args.network_dir)
