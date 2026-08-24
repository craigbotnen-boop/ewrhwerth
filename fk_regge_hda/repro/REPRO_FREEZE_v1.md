# FK48 reproducibility freeze v1.0

Freeze date: 2026-08-24

## Clean-environment gate

From an empty directory containing only the declared package files and using the locked NumPy/SciPy environment, the command

```bash
python run_fk48_repro.py --include-pseudo
```

completed with

```text
REPRODUCIBILITY CHECK: PASS
```

## Frozen headline fits

- fresh-seed pseudo-constraint obstruction: p = 4.000808101334
- full stationary gauge-Hessian aggregate: p = 4.024762949
- canonical gauge-to-gauge K_gg aggregate: p = 4.039159580
- canonical gauge-to-physical K_gp control: p = 2.000569303
- maximum full stationarity residual in the reported stationary family: < 1.5e-14
- minimum physical spectral gap: > 4.26
- deformation-restricted stationary response remains O(1)

## Scope

This is an independent fresh-seed reproducibility control. It does not claim byte-for-byte reconstruction of the historical manuscript boundary seed. Its purpose is to reproduce the same stationary/canonical block hierarchy from fully specified public inputs.
