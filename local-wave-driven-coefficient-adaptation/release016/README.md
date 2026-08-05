# Release 016 reflecting-boundary rerun

This folder records the final numerical correction for Craig W. Botnen, *Local Wave-Driven Coefficient Adaptation on Metric Networks: Short-Time Well-Posedness and Finite Propagation*.

## Correction

The analytical endpoint condition is `p=0`. Since `p=(r+ell)/2`, Release 016 imposes the reflecting characteristic relation

```text
r = -ell
```

at every external endpoint and at every SSP-RK3 stage. The differentiated stage trace is `r_t=-ell_t`.

## Files

- `run_reflecting_release_016.py`: wrapper applying the reflecting endpoint correction to the archived solver.
- `summary.json`: controlling parameters and finest-grid results.
- `grid_runs.csv`: five-grid leakage and arrival diagnostics.
- `convergence.csv`: nested-grid differences and observed orders.

The complete archived supplement contains the full modified solver, sensor histories, reference profiles, figures, environment file, and checksums.

## Result

The corrected rerun retains approximately second-order wave convergence on the finest pair. The reported finest-grid pre-cone wave magnitude is `1.470820559783418e-05`, and the structural magnitude is `6.727951529228449e-13`. First-arrival diagnostics continue to use threshold `1e-10`.

The numerical experiment is illustrative and is not used to prove the analytical theorems.
