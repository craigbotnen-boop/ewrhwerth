# Coefficient projection audit

The solver clips the coefficient to the invariant interval after each SSP-RK stage. Release 017 records the largest pre-clipping change:

```text
maximum_projection_correction_all_grids = 0.0
```

Thus no stage value left the coefficient box in double precision, and clipping was inactive. The convergence and leakage results correspond to the original stage evolution rather than a projection-modified scheme.
