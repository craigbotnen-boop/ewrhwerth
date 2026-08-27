"""
STARGATE_Colab_L729.py

Purpose:
    - Reference / convenience runner for L=729 carpet walks.
    - Provides portable, interactive parity with the local archival campaign.

Governance:
    - NOT an archival execution path.
    - Results from this runner are for validation/reproduction only.
    - Archival provenance is defined by the local campaign with frozen config.

ValidationGate v2.0.5-ARCHIVAL (Constitutional)
Gate Configuration (Paper 0 Frozen):
    - R² >= 0.99 (strict linearity)
    - Floor Factor >= 50.0 (window-wise)
    - Min Points >= 15
    - Window Policy: LVW (Longest Valid Window)
"""

# ============================================================
# CELL 1: Configuration (RUN THIS FIRST)
# ============================================================

LEVEL = 6  # L = 3^6 = 729 (262,144 nodes)
N_REALIZATIONS = 10
N_WALKERS = 50000
N_STEPS = 1000
LAZY_FACTOR = 0.5

# Archival gates (Paper 0 frozen)
GATE_CONFIG = {
    "r2_min": 0.99,
    "floor_factor": 50.0,
    "min_pts": 15
}

# LVW grid definition (for metadata transparency)
LVW_GRID = {
    "t_start": {"min": 2, "max": 32, "step": 2},
    "t_end": {"offset_min": 20, "max": 800, "step": 10},
    "min_pts": 15
}

print(f"Configuration:")
print(f"  Level: {LEVEL} (L={3**LEVEL})")
print(f"  Nodes: ~{8**LEVEL} (estimate)")
print(f"  Realizations: {N_REALIZATIONS}")
print(f"  Walkers: {N_WALKERS}")
print(f"  Gates: R2>={GATE_CONFIG['r2_min']}, floor>={GATE_CONFIG['floor_factor']}")

# ============================================================
# CELL 2: Imports and Core Functions
# ============================================================

import numpy as np
import json
import hashlib
from datetime import datetime
from scipy import stats
import time

def generate_carpet_graph_sparse(level):
    """Generate Sierpinski Carpet as CSR adjacency matrix (memory efficient)."""
    print(f"Generating L={3**level} carpet...")
    start = time.time()

    def is_carpet(x, y):
        while x > 0 or y > 0:
            if x % 3 == 1 and y % 3 == 1:
                return False
            x //= 3
            y //= 3
        return True

    size = 3**level

    # Generate nodes
    print(f"  Enumerating nodes...")
    nodes = []
    for i in range(size):
        for j in range(size):
            if is_carpet(i, j):
                nodes.append((i, j))

    node_map = {pos: i for i, pos in enumerate(nodes)}
    n_nodes = len(nodes)
    print(f"  Nodes: {n_nodes}")

    # Build CSR components directly
    print(f"  Building adjacency...")
    row_ptr = [0]
    col_idx = []

    for idx, (i, j) in enumerate(nodes):
        neighbors = []
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (i + di, j + dj)
            if neighbor in node_map:
                neighbors.append(node_map[neighbor])
        neighbors.sort()
        col_idx.extend(neighbors)
        row_ptr.append(len(col_idx))

    # Convert to numpy
    indptr = np.array(row_ptr, dtype=np.int32)
    indices = np.array(col_idx, dtype=np.int32)
    n_edges = len(indices) // 2

    elapsed = time.time() - start
    print(f"  Edges: {n_edges}")
    print(f"  Generation time: {elapsed:.1f}s")

    return indptr, indices, n_nodes, n_edges


def run_lazy_walk_fast(indptr, indices, start_node, steps, n_walkers, lazy_factor=0.5):
    """Optimized lazy random walk."""
    p_t = np.zeros(steps)
    rng = np.random.default_rng()

    for _ in range(n_walkers):
        current = start_node
        p_t[0] += 1

        for t in range(1, steps):
            if rng.random() > lazy_factor:
                row_start = indptr[current]
                row_end = indptr[current + 1]
                if row_end > row_start:
                    current = indices[row_start + rng.integers(row_end - row_start)]
            if current == start_node:
                p_t[t] += 1

    return p_t / n_walkers


def evaluate_window(log_t, log_p, p_values, pi_origin, gate_config):
    """Evaluate window against archival gates."""
    min_pts = gate_config['min_pts']

    if len(log_t) < min_pts:
        return None, 'FAIL_MIN_PTS'

    valid_mask = np.isfinite(log_p) & np.isfinite(log_t) & (p_values > 0)
    if np.sum(valid_mask) < min_pts:
        return None, 'FAIL_MIN_PTS'

    log_t_c = log_t[valid_mask]
    log_p_c = log_p[valid_mask]
    p_c = p_values[valid_mask]

    slope, intercept, r_value, _, std_err = stats.linregress(log_t_c, log_p_c)
    r2 = r_value**2

    if r2 < gate_config['r2_min']:
        return {'r2': float(r2), 'ds': float(-2*slope)}, 'FAIL_R2'

    floor_ratios = p_c / pi_origin
    floor_ratio_min = float(np.min(floor_ratios))

    if floor_ratio_min < gate_config['floor_factor']:
        return {'r2': float(r2), 'floor_ratio_min': floor_ratio_min, 'ds': float(-2*slope)}, 'FAIL_FLOOR'

    ds = -2 * slope
    return {
        'is_valid': True,
        'ds': float(ds),
        'r2': float(r2),
        'floor_ratio_min': floor_ratio_min,
        'slope_se': float(std_err),
        'n_pts': int(np.sum(valid_mask)),
        'window_length': len(log_t)
    }, 'PASS'


def lvw_window_search(p_t, pi_origin, gate_config, max_t=None):
    """LVW (Longest Valid Window) search."""
    steps = len(p_t)
    if max_t is None:
        max_t = min(steps - 1, 800)

    candidates = []
    for t_start in range(2, min(33, max_t), 2):
        for t_end in range(t_start + 20, max_t, 10):
            if t_end - t_start >= gate_config['min_pts']:
                candidates.append((t_start, t_end))

    valid_windows = []

    for t_start, t_end in candidates:
        t = np.arange(t_start, t_end)
        p_window = p_t[t_start:t_end]

        nonzero_mask = p_window > 0
        if np.sum(nonzero_mask) < gate_config['min_pts']:
            continue

        log_t = np.log(t[nonzero_mask])
        log_p = np.log(p_window[nonzero_mask])

        result, status = evaluate_window(log_t, log_p, p_window[nonzero_mask], pi_origin, gate_config)

        if status == 'PASS':
            valid_windows.append({
                'window': (t_start, t_end),
                'length': t_end - t_start,
                'result': result
            })

    if not valid_windows:
        return None, 'NO_VALID_WINDOW'

    valid_windows.sort(key=lambda x: (x['length'], x['result']['r2']), reverse=True)

    winner = valid_windows[0]
    winner['result']['window'] = list(winner['window'])

    return winner['result'], 'LVW_PASS'


print("Core functions loaded.")

# ============================================================
# CELL 3: Run Production
# ============================================================

def run_production():
    print("="*70)
    print("STARGATE-UNIVERSAL ARCHIVAL PRODUCTION RUN (COLAB)")
    print("ValidationGate v2.0.5-ARCHIVAL")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    # Generate graph
    print(f"\n[1/3] Generating Sierpinski Carpet (L={3**LEVEL})...")
    indptr, indices, n_nodes, n_edges = generate_carpet_graph_sparse(LEVEL)
    m_undirected = n_edges

    print(f"\n      Gate: R2>={GATE_CONFIG['r2_min']}, floor>={GATE_CONFIG['floor_factor']}")
    print(f"      Walkers: {N_WALKERS}, Steps: {N_STEPS}")

    # Run realizations
    ledger = []
    ds_values = []
    fail_counts = {'FAIL_R2': 0, 'FAIL_FLOOR': 0, 'FAIL_MIN_PTS': 0, 'NO_VALID_WINDOW': 0}

    print(f"\n[2/3] Running {N_REALIZATIONS} realizations...")

    rng = np.random.default_rng()

    for r in range(N_REALIZATIONS):
        start_time = time.time()
        start_node = rng.integers(0, n_nodes)

        # Compute degree and pi_origin
        deg_origin = indptr[start_node + 1] - indptr[start_node]
        pi_origin = deg_origin / (2 * m_undirected)

        # Run walk
        p_t = run_lazy_walk_fast(indptr, indices, start_node, N_STEPS, N_WALKERS, LAZY_FACTOR)

        # LVW search
        result, status = lvw_window_search(p_t, pi_origin, GATE_CONFIG)

        # Record
        entry = {
            'realization': r,
            'start_node': int(start_node),
            'deg_origin': int(deg_origin),
            'pi_origin': float(pi_origin),
            'status': status
        }

        if status == 'LVW_PASS':
            entry.update(result)
            ds_values.append(result['ds'])
            status_str = f"PASS ds={result['ds']:.3f}"
        else:
            fail_counts[status] = fail_counts.get(status, 0) + 1
            status_str = status

        elapsed = time.time() - start_time
        print(f"  [R{r:03d}] {status_str} ({elapsed:.1f}s)")

        ledger.append(entry)

    # Aggregate
    print(f"\n[3/3] Aggregating results...")

    n_pass = len(ds_values)
    pass_rate = n_pass / N_REALIZATIONS

    if n_pass > 0:
        ds_mean = float(np.mean(ds_values))
        ds_sd = float(np.std(ds_values))
        ds_se = float(ds_sd / np.sqrt(n_pass))
    else:
        ds_mean = ds_sd = ds_se = None

    # Build output
    config_str = json.dumps(GATE_CONFIG, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    output = {
        'audit_metadata': {
            'engine_version': '2.0.5-ARCHIVAL',
            'config_hash': f"sha256:{config_hash}",
            'gate_config': GATE_CONFIG,
            'lvw_grid': LVW_GRID,
            'window_policy': 'LVW',
            'timestamp': datetime.now().isoformat(),
            'substrate': f"Sierpinski_Carpet_L{3**LEVEL}",
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'runtime': {
                'n_walkers': N_WALKERS,
                'n_steps': N_STEPS,
                'lazy_factor': LAZY_FACTOR
            },
            'runtime_mode': 'COLAB_REFERENCE'
        },
        'summary': {
            'n_realizations': N_REALIZATIONS,
            'n_pass': n_pass,
            'pass_rate': pass_rate,
            'ds_mean': ds_mean,
            'ds_sd': ds_sd,
            'ds_se': ds_se,
            'theoretical_ds': 1.805,
            'failure_breakdown': fail_counts
        },
        'per_realization_ledger': ledger
    }

    # Compute result hash
    output_json = json.dumps(output, sort_keys=True)
    output['audit_metadata']['result_sha256'] = hashlib.sha256(output_json.encode()).hexdigest()

    # Save
    output_path = f'archival_production_L{3**LEVEL}_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("ARCHIVAL PRODUCTION RUN COMPLETE")
    print("="*70)
    print(f"  Pass Rate:    {pass_rate:.1%} ({n_pass}/{N_REALIZATIONS})")
    if ds_mean:
        print(f"  ds_mean:      {ds_mean:.4f}")
        print(f"  ds_SD:        {ds_sd:.4f}")
        print(f"  ds_SE:        {ds_se:.4f}")
    print(f"  Theoretical:  ~1.805")
    print(f"\n  Output: {output_path}")
    print(f"  Result Hash: {output['audit_metadata']['result_sha256'][:16]}...")
    print("="*70)

    return output

# Run it!
results = run_production()

# ============================================================
# CELL 4: Download Results (Colab only)
# ============================================================

try:
    from google.colab import files
    files.download(f'archival_production_L{3**LEVEL}_results.json')
except ImportError:
    print("Not running in Colab - results saved locally.")
