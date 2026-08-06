# Local Wave-Driven Coefficient Adaptation on Metric Networks

Reproducibility materials for Submission Release 017.

## Contents

- `run_higher_order_solver.py`: second-order characteristic / SSP-RK3 experiment.
- `summary.json`: controlling parameters, convergence results, and projection audit.
- `grid_runs.csv`: per-grid timing, leakage, coefficient-change, and projection diagnostics.
- `convergence.csv`: nested-grid relative differences and observed orders.
- `sensor_history_h0p005.csv`: selected time histories.
- `reference_profiles_h0p0025.csv`: finest-grid final profiles.
- four PNG figures.
- `requirements.txt`: tested Python environment.

## Boundary and projection conventions

The external endpoint condition is the analytical reflecting condition

```text
p = 0  <=>  r = -ell
```

at every SSP-RK stage. The coefficient is clipped to `[a_min,a_max]` after each stage as a fail-safe. The maximum pre-projection correction was exactly `0.0` in double precision on every grid, so projection was inactive in the reported runs.

## Detection convention

First-arrival diagnostics use

```text
SUPPORT_THRESHOLD = 1e-10
```

for both characteristic amplitude and coefficient perturbation. These are threshold- and grid-dependent diagnostics, not sharp continuum fronts.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_higher_order_solver.py
```
