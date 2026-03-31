"""
run_physics_sweep.py
---------------------
Batch spectral-dimension sweep over TCGA network TSV files.

For each network file:
  1. Loads edge list (source, target[, weight])
  2. Builds adjacency → normalized Laplacian
  3. Computes eigenvalues → heat kernel trace P(t)
  4. Fits log P(t) ~ -(d_s/2) log t to get d_s and R²
  5. Enforces R² >= R2_THRESHOLD gate (default 0.95)

Outputs:
  ds_values.tsv          -- Patient_ID | d_s | r2 | n_nodes | n_edges | k_used
  ds_sweep_rejected.tsv  -- files that failed the gate (for audit)
  ds_sweep_log.txt       -- per-file diagnostics

Network TSV format expected (auto-detected):
  Option A — edge list:   source  target  [weight]
  Option B — dense/wide adjacency matrix (square, header = node names)

Patient ID extracted from filename: first 12 characters matching TCGA-XX-XXXX
pattern, or the full stem if no TCGA barcode is found.

Usage (from inside P9_SHIELD/):
    python run_physics_sweep.py --network_dir /path/to/network/tsv/files

    # Or set NETWORK_DIR below and run without arguments:
    python run_physics_sweep.py

Dependencies:
    pip install numpy scipy pandas tqdm
    (diffusion_spectral_dimension.py must be on sys.path or in P9_SHIELD/)
"""

import sys
import re
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
# Set this to your local network directory if you prefer not to use --network_dir
NETWORK_DIR   = Path("networks")        # override with --network_dir
OUTFILE       = Path("ds_values.tsv")
REJECT_FILE   = Path("ds_sweep_rejected.tsv")
LOG_FILE      = Path("ds_sweep_log.txt")

R2_THRESHOLD  = 0.95   # gate
N_EIGS        = 100    # Laplacian eigenvalues per network
N_TIMES       = 60     # heat kernel time points
T_MIN         = 0.01
T_MAX         = 200.0
T_FIT_FRAC_LO = 0.10   # plateau window: lower 10%–70% of log-time range
T_FIT_FRAC_HI = 0.70

# ── Helpers ───────────────────────────────────────────────────────────────────
BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.IGNORECASE)


def patient_id_from_path(p: Path) -> str:
    m = BARCODE_RE.search(p.stem)
    return m.group(0).upper() if m else p.stem[:12]


def load_edge_list(path: Path):
    """
    Load a TSV as an edge list → scipy CSR adjacency matrix.
    Handles:
      - 2-column (source, target) → binary weights
      - 3-column (source, target, weight) → weighted
      - Square adjacency matrix (detected when #cols == #rows after pivot)
    Returns (A, node_list) where A is symmetric CSR.
    """
    df = pd.read_csv(path, sep="\t", header=0, low_memory=False)

    # ── Square adjacency matrix? ──────────────────────────────────────────────
    if df.shape[0] == df.shape[1] - 1 or df.shape[0] == df.shape[1]:
        try:
            mat = df.values.astype(float)
            n = mat.shape[0]
            if mat.shape == (n, n):
                A = csr_matrix(mat)
                A = (A + A.T)
                A.data = np.minimum(A.data, 1.0)
                return A, list(range(n))
        except (ValueError, TypeError):
            pass

    # ── Edge list ─────────────────────────────────────────────────────────────
    cols = df.columns.tolist()
    src_col, tgt_col = cols[0], cols[1]
    wt_col = cols[2] if len(cols) >= 3 else None

    nodes = pd.unique(pd.concat([df[src_col], df[tgt_col]]))
    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    src = df[src_col].map(node_idx).values
    tgt = df[tgt_col].map(node_idx).values
    wts = pd.to_numeric(df[wt_col], errors="coerce").fillna(1.0).abs().values \
          if wt_col else np.ones(len(src))

    # Symmetrise
    row = np.concatenate([src, tgt])
    col = np.concatenate([tgt, src])
    dat = np.concatenate([wts, wts])

    A = csr_matrix((dat, (row, col)), shape=(N, N))
    return A, list(nodes)


def ensure_connected(A: csr_matrix):
    """Return largest connected component."""
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return A, A.shape[0]
    sizes = np.bincount(labels)
    big = np.argmax(sizes)
    idx = np.where(labels == big)[0]
    return A[idx][:, idx], int(sizes[big])


def build_normalized_laplacian(A: csr_matrix) -> csr_matrix:
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg = np.maximum(deg, 1e-10)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    L = sp.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()


def compute_eigenvalues(L: csr_matrix, n_eigs: int) -> np.ndarray:
    N = L.shape[0]
    k = min(n_eigs, N - 2)
    if k < 1:
        return np.array([0.0])
    if N <= 600:
        vals = np.linalg.eigvalsh(L.toarray())
        return np.sort(vals)[:k]
    try:
        vals, _ = eigsh(L, k=k, which="SM")
        return np.sort(np.abs(vals))
    except Exception:
        vals = np.linalg.eigvalsh(L.toarray())
        return np.sort(vals)[:k]


def heat_kernel_trace(eigenvalues: np.ndarray, t_values: np.ndarray,
                      lam_min: float = 1e-8) -> np.ndarray:
    valid = eigenvalues[eigenvalues > lam_min]
    if len(valid) == 0:
        return np.ones_like(t_values)
    return np.exp(-np.outer(t_values, valid)).sum(axis=1)


def fit_power_law(t_values: np.ndarray, P_t: np.ndarray):
    """
    Fit log P(t) = -(d_s/2) log t + C in the plateau window.
    Returns (d_s, r2).
    """
    log_t = np.log(t_values)
    log_P = np.log(np.maximum(P_t, 1e-300))

    t_lo = t_values[0] * (t_values[-1] / t_values[0]) ** T_FIT_FRAC_LO
    t_hi = t_values[0] * (t_values[-1] / t_values[0]) ** T_FIT_FRAC_HI
    mask = (t_values >= t_lo) & (t_values <= t_hi) & np.isfinite(log_P)

    if mask.sum() < 4:
        return np.nan, 0.0

    x = log_t[mask]
    y = log_P[mask]

    # OLS: y = a*x + b
    A_mat = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A_mat, y, rcond=None)
    a, b = coef
    d_s = -2.0 * a

    y_pred = a * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(d_s), float(r2)


# ── Main sweep ────────────────────────────────────────────────────────────────
def sweep(network_dir: Path):
    files = sorted(network_dir.glob("*.tsv"))
    if not files:
        files = sorted(network_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .tsv/.txt files found in {network_dir}\n"
            "Set --network_dir to your local TCGA network folder."
        )

    print(f"Found {len(files)} network files in {network_dir}")
    print(f"R² gate: {R2_THRESHOLD}")

    t_values = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_TIMES)

    passed, rejected = [], []
    log_lines = []

    try:
        from tqdm import tqdm
        iterator = tqdm(files, desc="Physics sweep")
    except ImportError:
        iterator = files

    for i, fpath in enumerate(iterator):
        pid = patient_id_from_path(fpath)
        status = "OK"
        d_s = r2 = np.nan
        n_nodes = n_edges = k_used = 0

        try:
            A_raw, _ = load_edge_list(fpath)
            A_lcc, n_nodes = ensure_connected(A_raw)
            n_edges = A_lcc.nnz // 2

            if n_nodes < 10:
                raise ValueError(f"Too few nodes: {n_nodes}")

            L = build_normalized_laplacian(A_lcc)
            eigs = compute_eigenvalues(L, N_EIGS)
            P_t  = heat_kernel_trace(eigs, t_values)
            d_s, r2 = fit_power_law(t_values, P_t)

            if not np.isfinite(d_s) or d_s <= 0:
                raise ValueError(f"Non-physical d_s={d_s:.3f}")

            if r2 >= R2_THRESHOLD:
                passed.append(dict(Patient_ID=pid, d_s=round(d_s, 4),
                                   r2=round(r2, 4), n_nodes=n_nodes,
                                   n_edges=n_edges))
            else:
                status = f"REJECTED (R²={r2:.3f})"
                rejected.append(dict(Patient_ID=pid, d_s=round(d_s, 4),
                                     r2=round(r2, 4), n_nodes=n_nodes,
                                     n_edges=n_edges, file=fpath.name))

        except Exception as e:
            status = f"ERROR: {e}"
            rejected.append(dict(Patient_ID=pid, d_s=np.nan, r2=np.nan,
                                 n_nodes=n_nodes, n_edges=n_edges,
                                 file=fpath.name))

        log_lines.append(
            f"{i+1:4d}/{len(files)}  {pid}  d_s={d_s:.3f}  "
            f"R²={r2:.3f}  N={n_nodes}  E={n_edges}  [{status}]"
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    df_pass = pd.DataFrame(passed)
    df_pass.to_csv(OUTFILE, sep="\t", index=False)
    print(f"\nWrote {OUTFILE}  ({len(df_pass)} patients passed gate)")

    if rejected:
        df_rej = pd.DataFrame(rejected)
        df_rej.to_csv(REJECT_FILE, sep="\t", index=False)
        print(f"Wrote {REJECT_FILE}  ({len(df_rej)} rejected)")

    LOG_FILE.write_text("\n".join(log_lines))
    print(f"Wrote {LOG_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Sweep summary ────────────────────────────────────────────────")
    print(f"Total files       : {len(files)}")
    print(f"Passed R²≥{R2_THRESHOLD} : {len(df_pass)}")
    print(f"Rejected / errors : {len(rejected)}")
    if len(df_pass):
        print(f"d_s  mean ± SD    : {df_pass['d_s'].mean():.3f} ± "
              f"{df_pass['d_s'].std():.3f}")
        print(f"d_s  range        : [{df_pass['d_s'].min():.3f}, "
              f"{df_pass['d_s'].max():.3f}]")
        print(f"R²   mean         : {df_pass['r2'].mean():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch spectral-dimension sweep over TCGA network TSVs"
    )
    parser.add_argument(
        "--network_dir", type=Path, default=NETWORK_DIR,
        help="Directory containing network .tsv files"
    )
    args = parser.parse_args()

    if not args.network_dir.exists():
        print(f"[!] Network directory not found: {args.network_dir}")
        print("    Pass --network_dir /full/path/to/your/networks")
        sys.exit(1)

    sweep(args.network_dir)
