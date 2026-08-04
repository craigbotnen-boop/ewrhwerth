# Local Wave-Driven Coefficient Adaptation on Metric Networks

Reproducibility materials for the manuscript:

> Craig Botnen, *Local Wave-Driven Coefficient Adaptation on Metric Networks: Short-Time Well-Posedness and Finite Propagation*.

## Contents

- `run_higher_order_solver.py`: second-order characteristic / SSP-RK3 numerical experiment.
- `summary.json`: controlling numerical parameters and summary results.
- `grid_runs.csv`: per-grid timing, leakage, and coefficient-change diagnostics.
- `convergence.csv`: nested-grid relative differences and observed orders.
- `requirements.txt`: tested Python environment.

The full archived submission supplement also contains the finest-grid profiles and sensor-history table.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_higher_order_solver.py
```

The script writes the CSV tables, `summary.json`, and four PNG figures into the working directory.

## Detection convention

Reported first-arrival diagnostics use the fixed threshold

```text
SUPPORT_THRESHOLD = 1e-10
```

for both characteristic amplitude and coefficient perturbation. These times are threshold- and grid-dependent diagnostics, not estimates of a sharp continuum front.

## Numerical method

- second-order one-sided characteristic differences;
- three-stage SSP Runge-Kutta time stepping;
- grids `h = 0.04, 0.02, 0.01, 0.005, 0.0025`;
- uniform speed ceiling `sqrt(a_max) = 1.3`.

## Version

These files correspond to Submission Release 012.