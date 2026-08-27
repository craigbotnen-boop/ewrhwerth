"""
STARGATE-UNIVERSAL Archival Production Run
===========================================
ValidationGate v2.0.5-ARCHIVAL (Constitutional)

Gate Configuration (Paper 0 Frozen):
- R² ≥ 0.99 (strict linearity)
- Floor Factor ≥ 50.0 (window-wise)
- Min Points ≥ 15
- Window Policy: LVW (Longest Valid Window)

This script is self-authenticating via config_hash embedding.
"""

import numpy as np
import networkx as nx
from scipy import stats
import json
import hashlib
from datetime import datetime
import sys

# ============================================================
# LOAD AND HASH FROZEN CONFIG
# ============================================================

def load_and_hash_config(config_path):
    """Load config and compute canonical SHA256 hash."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Canonical hash (sorted keys, no whitespace)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config_hash = hashlib.sha256(canonical).hexdigest()

    return cfg, config_hash


# ============================================================
# CORE FUNCTIONS
# ============================================================

def generate_carpet_graph(level):
    """Generate Sierpinski Carpet as adjacency matrix."""
    def is_carpet(x, y):
        while x > 0 or y > 0:
            if x % 3 == 1 and y % 3 == 1:
                return False
            x //= 3
            y //= 3
        return True

    size = 3**level
    nodes = [(i, j) for i in range(size) for j in range(size) if is_carpet(i, j)]
    node_map = {pos: i for i, pos in enumerate(nodes)}

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))

    for (i, j) in nodes:
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (i + di, j + dj)
            if neighbor in node_map:
                G.add_edge(node_map[(i, j)], node_map[neighbor])

    return nx.adjacency_matrix(G).tocsr(), G.number_of_nodes(), G.number_of_edges()


def run_lazy_walk(adj_csr, start_node, steps, n_walkers, lazy_factor=0.5):
    """Lazy random walk with configurable stay probability."""
    p_t = np.zeros(steps)

    for _ in range(n_walkers):
        current = start_node
        p_t[0] += 1

        for t in range(1, steps):
            if np.random.rand() > lazy_factor:
                row_start = adj_csr.indptr[current]
                row_end = adj_csr.indptr[current + 1]
                neighbors = adj_csr.indices[row_start:row_end]
                if len(neighbors) > 0:
                    current = np.random.choice(neighbors)
            if current == start_node:
                p_t[t] += 1

    return p_t / n_walkers


def evaluate_window(log_t, log_p, p_values, pi_origin, gate_config):
    """Evaluate a single candidate window against archival gates."""
    min_pts = gate_config['min_pts']
    r2_min = gate_config['r2_min']
    floor_factor = gate_config['floor_factor']

    if len(log_t) < min_pts:
        return None, 'FAIL_MIN_PTS'

    valid_mask = np.isfinite(log_p) & np.isfinite(log_t) & (p_values > 0)
    if np.sum(valid_mask) < min_pts:
        return None, 'FAIL_MIN_PTS'

    log_t_c = log_t[valid_mask]
    log_p_c = log_p[valid_mask]
    p_c = p_values[valid_mask]

    # Linearity check
    slope, intercept, r_value, _, std_err = stats.linregress(log_t_c, log_p_c)
    r2 = r_value**2

    if r2 < r2_min:
        return {'r2': float(r2), 'ds': float(-2*slope)}, 'FAIL_R2'

    # Floor check: EVERY point must satisfy P(t)/pi >= floor_factor
    floor_ratios = p_c / pi_origin
    floor_ratio_min = float(np.min(floor_ratios))

    if floor_ratio_min < floor_factor:
        return {'r2': float(r2), 'floor_ratio_min': floor_ratio_min, 'ds': float(-2*slope)}, 'FAIL_FLOOR'

    # PASS
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
    """
    LVW (Longest Valid Window) search.

    Enumerate candidate windows and return the LONGEST one that passes.
    Ties break on highest R².
    """
    steps = len(p_t)
    if max_t is None:
        max_t = min(steps - 1, 512)

    # Candidate windows: start from early times, varying lengths
    # Window family: t_start in [2, 4, 6, 8, ..., 32], t_end varies
    candidates = []

    for t_start in range(2, min(33, max_t), 2):
        for t_end in range(t_start + 20, max_t, 10):  # At least 20 points
            if t_end - t_start >= gate_config['min_pts']:
                candidates.append((t_start, t_end))

    # Evaluate each candidate
    valid_windows = []

    for t_start, t_end in candidates:
        t = np.arange(t_start, t_end)
        p_window = p_t[t_start:t_end]

        # Skip if too many zeros
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

    # LVW: sort by length (desc), then R² (desc)
    valid_windows.sort(key=lambda x: (x['length'], x['result']['r2']), reverse=True)

    winner = valid_windows[0]
    winner['result']['window'] = list(winner['window'])

    return winner['result'], 'LVW_PASS'


def run_archival_production(config_path, level=4, n_realizations=100):
    """Run archival production with frozen config."""

    # Load and hash config
    cfg, config_hash = load_and_hash_config(config_path)
    gate = cfg['gate_config']
    runtime = cfg.get('runtime_defaults', {})

    n_walkers = runtime.get('n_walkers', 5000)
    n_steps = runtime.get('n_steps', 1000)
    lazy_factor = runtime.get('lazy_factor', 0.5)

    print("="*70)
    print("STARGATE-UNIVERSAL ARCHIVAL PRODUCTION RUN")
    print(f"ValidationGate {cfg['engine_version']}")
    print(f"Config Hash: sha256:{config_hash[:16]}...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    print(f"\nGate: R2>={gate['r2_min']}, floor>={gate['floor_factor']}, min_pts>={gate['min_pts']}")
    print(f"Window Policy: {cfg['window_policy']}")
    print(f"Runtime: walkers={n_walkers}, steps={n_steps}, lazy={lazy_factor}")

    # Generate substrate
    print(f"\n[1/3] Generating Sierpinski Carpet (L={3**level})...")
    adj, n_nodes, n_edges = generate_carpet_graph(level)
    m_undirected = n_edges

    print(f"      Nodes: {n_nodes}, Edges: {n_edges}")

    # Run realizations
    ledger = []
    ds_values = []
    fail_counts = {'FAIL_R2': 0, 'FAIL_FLOOR': 0, 'FAIL_MIN_PTS': 0, 'NO_VALID_WINDOW': 0}

    print(f"\n[2/3] Running {n_realizations} realizations...")

    for r in range(n_realizations):
        start_node = np.random.randint(0, n_nodes)

        # Compute degree and pi_origin
        row_start = adj.indptr[start_node]
        row_end = adj.indptr[start_node + 1]
        deg_origin = row_end - row_start
        pi_origin = deg_origin / (2 * m_undirected)

        # Run walk
        p_t = run_lazy_walk(adj, start_node, n_steps, n_walkers, lazy_factor)

        # LVW search
        result, status = lvw_window_search(p_t, pi_origin, gate)

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
        else:
            fail_counts[status] = fail_counts.get(status, 0) + 1

        ledger.append(entry)

        # Progress
        if (r + 1) % 10 == 0:
            n_pass = len(ds_values)
            print(f"      [{r+1:4d}/{n_realizations}] pass={n_pass} ({100*n_pass/(r+1):.1f}%)")

    # Aggregate
    print(f"\n[3/3] Aggregating results...")

    n_pass = len(ds_values)
    pass_rate = n_pass / n_realizations

    if n_pass > 0:
        ds_mean = float(np.mean(ds_values))
        ds_sd = float(np.std(ds_values))
        ds_se = float(ds_sd / np.sqrt(n_pass))
    else:
        ds_mean = ds_sd = ds_se = None

    # Build output
    output = {
        'audit_metadata': {
            'engine_version': cfg['engine_version'],
            'config_hash': f"sha256:{config_hash}",
            'gate_config': gate,
            'window_policy': cfg['window_policy'],
            'timestamp': datetime.now().isoformat(),
            'substrate': f"Sierpinski_Carpet_L{3**level}",
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'runtime': {
                'n_walkers': n_walkers,
                'n_steps': n_steps,
                'lazy_factor': lazy_factor
            }
        },
        'summary': {
            'n_realizations': n_realizations,
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
    output_path = f'archival_production_L{3**level}_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("ARCHIVAL PRODUCTION RUN COMPLETE")
    print("="*70)
    print(f"  Pass Rate:    {pass_rate:.1%} ({n_pass}/{n_realizations})")
    print(f"  ds_mean:      {ds_mean:.4f}" if ds_mean else "  ds_mean:      N/A")
    print(f"  ds_SD:        {ds_sd:.4f}" if ds_sd else "  ds_SD:        N/A")
    print(f"  ds_SE:        {ds_se:.4f}" if ds_se else "  ds_SE:        N/A")
    print(f"  Theoretical:  ~1.805")
    print(f"\n  Failures:")
    for k, v in fail_counts.items():
        if v > 0:
            print(f"    {k}: {v}")
    print(f"\n  Output: {output_path}")
    print(f"  Config Hash: sha256:{config_hash[:16]}...")
    print(f"  Result Hash: {output['audit_metadata']['result_sha256'][:16]}...")
    print("="*70)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='STARGATE-UNIVERSAL Archival Production Run')
    parser.add_argument('--level', type=int, default=4, help='Carpet level (L=3^level)')
    parser.add_argument('--realizations', type=int, default=10, help='Number of realizations')
    parser.add_argument('--config', type=str, default='stargate_gate_v2.0.5-ARCHIVAL.json', help='Config path')

    args = parser.parse_args()

    run_archival_production(args.config, level=args.level, n_realizations=args.realizations)
