# Zenodo manual deposit metadata — FK48 reproducibility package v1.0

## Deposit route

Use a **manual Zenodo software upload**, not the GitHub auto-integration. This repository contains multiple unrelated research projects; the manual route isolates the FK48 reproducibility package and allows DOI reservation before publication.

## File to upload

- `FK48_REPRODUCIBILITY_PACKAGE_v1.0.tar.gz`
- SHA-256: `5d0469efaccb91e2c71672932dd087b58146b8777a72ccb9be316a83688fd786`

## Suggested metadata

**Title**

FK48 Regge HDA Reproducibility Package v1.0

**Resource type**

Software

**Version**

1.0.0

**Publication date**

2026-08-24

**Creator**

Craig Botnen

**Description**

Independent clean-room reproducibility package for the FK48 Regge hypersurface-deformation-algebra study. The package reconstructs a 48-four-simplex Freudenthal–Kuhn two-step length-Regge complex from fully specified inputs, solves the stationary internal Regge equations, and reproduces the canonical block hierarchy associated with fourth-order restoration of the gauge/deformation sector under refinement. The frozen release includes both the pseudo-constraint suite and the full stationary/canonical suite. The clean-environment one-command gate completes with `REPRODUCIBILITY CHECK: PASS`.

Headline fresh-seed outputs include a pseudo-constraint obstruction exponent near 4.00081, stationary gauge-Hessian aggregate exponent near 4.0248, canonical gauge-to-gauge exponent near 4.0392, gauge-to-physical control exponent near 2.00057, full stationarity residuals of order 1e-14, and an O(1) deformation-restricted stationary response.

This package is an independent fresh-seed reproduction of the manuscript's stationary/canonical block hierarchy. It is not a byte-for-byte reconstruction of the historical manuscript boundary seed and does not claim exact finite-spacing diffeomorphism symmetry or unrestricted off-shell Dirac-algebra closure.

**Keywords**

- Regge calculus
- discrete gravity
- hypersurface-deformation algebra
- canonical gravity
- pseudo-constraints
- Freudenthal–Kuhn triangulation
- reproducibility

## Related identifier

Repository: `https://github.com/craigbotnen-boop/ewrhwerth`

Frozen branch: `fk-regge-hda-oh4-repro-v1-freeze`

Frozen commit: `c80a4bd03349bfba8580be68cfeeccb4472687c8`

Recommended relation: **IsSupplementTo** the eventual manuscript record; after manuscript publication, add the paper DOI as the related identifier.

## DOI workflow

1. Start a new Zenodo upload and add the tarball above.
2. Fill the metadata using this file.
3. Use Zenodo's **Get a DOI now!** function to reserve the DOI before publication.
4. Insert the reserved DOI into `data_availability_CQG.md` and the LaTeX manuscript.
5. Recheck the archive SHA-256 before publishing the Zenodo record.
6. Publish the Zenodo record only after the final metadata and file hash are confirmed.
