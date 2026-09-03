# FDEP coframe / teleparallel reproducibility bundle

This bundle reproduces the core numerical certificates from the P8–P17 coframe lane using the authenticated Packet-011 Freudenthal–Kuhn slab generator.

## One-command reproduction

```bash
./reproduce.sh
```

The command writes:

- `results/summary.json`
- `results/main_tables.md`

Use `python reproduce.py --quick` for a smaller smoke test that omits the finest P10/P15 levels.

## What is reproduced

- **P8**: `16 -> 10 -> 6` coframe/metric/Diff rank chain and 10-dimensional kernel.
- **P9**: `40 -> 16` edge-cochain/coframe map and `34 = 24 + 6 + 4` observable kernel decomposition.
- **P10**: Whitney coframe gluing on periodic FK slabs, Regge-compatible tangential metric continuity, and smooth `e`, `dE` convergence.
- **P12**: exact finite local-Lorentz invariance of `g=e^T eta e` and the nonlinear failure of the naive componentwise common-wave evolution.
- **P13**: flat inertial Lorentz links, covariant graph evolution, and covariant triangle torsion.
- **P14**: rank-three quadratic torsion-invariant response on the authenticated FK complex.
- **P15**: continuum-consistent fixed-link local-frame near-null coefficient ray approaching the TEGR target `(1/4, 1/2, -1)` under refinement.
- **P17**: exact integer certificate that the two audited mixed-torsion index notations define the same quadratic form on the 24-dimensional torsion space.

## Claim ceiling

This code does **not** derive the microscopic principle that would require fixed-link local-frame Hessian nullness. Therefore the TEGR result reproduced here is an **asymptotic conditional selector**, not an unconditional derivation of GR.

## Frozen runtime used for the certified run

- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0

`environment.txt` records these exact versions. P17 is an exact integer 24-component quadratic-form certificate and does not depend on floating-point tolerances.
