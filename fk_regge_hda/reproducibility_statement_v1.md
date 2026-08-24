# FK48 reproducibility statement v1

## Authoritative frozen state

- Repository: `craigbotnen-boop/ewrhwerth`
- Freeze branch: `fk-regge-hda-oh4-repro-v1-freeze`
- Freeze commit: `c80a4bd03349bfba8580be68cfeeccb4472687c8`
- Archive: `FK48_REPRODUCIBILITY_PACKAGE_v1.0.tar.gz`
- Archive SHA-256: `5d0469efaccb91e2c71672932dd087b58146b8777a72ccb9be316a83688fd786`

## Clean release gate

From an empty directory with the declared NumPy/SciPy environment,

```bash
python run_fk48_repro.py --include-pseudo
```

completed with

```text
REPRODUCIBILITY CHECK: PASS
```

for both the pseudo-constraint suite and the full stationary/canonical suite.

## Frozen headline outputs

- fresh-seed pseudo-constraint obstruction: `p ≈ 4.00081`;
- stationary gauge-Hessian aggregate: `p ≈ 4.0248`;
- canonical gauge-to-gauge block: `p ≈ 4.0392`;
- gauge-to-physical control: `p ≈ 2.00057`;
- full 16-equation stationarity residuals: approximately `1e-14`;
- physical spectral gap: greater than `4.26` over the reported scales;
- deformation-restricted stationary response remains bounded, while the unrestricted hostile control retains approximately `h^-2` growth.

## Claim boundary

This package is an independent fresh-seed reproducibility control for the block hierarchy and fourth-order canonical gauge sector. It does not claim byte-for-byte recovery of the historical manuscript boundary seed, exact finite-spacing diffeomorphism symmetry, or unrestricted off-shell Dirac-algebra closure.
