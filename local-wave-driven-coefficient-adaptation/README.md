# Local Wave-Driven Coefficient Adaptation on Metric Networks

Reproducibility materials for:

> Craig W. Botnen, *Local Wave-Driven Coefficient Adaptation on Metric Networks: Short-Time Well-Posedness and Finite Propagation*.

## Contents

- `run_higher_order_solver.py`: second-order characteristic / SSP-RK3 numerical experiment.
- `summary.json`: numerical parameters and summary results.
- `grid_runs.csv`: per-grid timing, leakage, and coefficient-change diagnostics.
- `convergence.csv`: nested-grid relative differences and observed orders.
- `requirements.txt`: tested Python environment.
- `NUMERICAL_METHOD_RELEASE_013.md`: detailed discretization and diagnostic definitions.
- `PEER_REVIEW_RESPONSE_013.md`: record of the theorem-level revision that expanded the differentiated vertex estimates.

The archived submission supplement also contains finest-grid profiles, sensor histories, figures, and checksums.

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
- nominal CFL number `0.35`;
- uniform speed ceiling `sqrt(a_max) = 1.3`;
- coefficient projection to `[a_min,a_max]` at each Runge-Kutta stage;
- vertex scattering imposed at every Runge-Kutta stage.

## Version

These materials correspond to the numerical content used in the journal-targeted Submission Release 015. The mathematical proof is contained in the manuscript; this repository supports only the numerical illustration and reproducibility claims.
