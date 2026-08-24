# Data availability statement — CQG draft

All computational inputs and scripts required for the independent FK48 reproducibility control are frozen in the GitHub branch `fk-regge-hda-oh4-repro-v1-freeze` at commit `c80a4bd03349bfba8580be68cfeeccb4472687c8`.

The frozen one-command gate is

```bash
python run_fk48_repro.py --include-pseudo
```

which completed from a fresh environment with `REPRODUCIBILITY CHECK: PASS` for both the pseudo-constraint and full stationary/canonical suites.

The deterministic archive is named `FK48_REPRODUCIBILITY_PACKAGE_v1.0.tar.gz` with SHA-256

```text
5d0469efaccb91e2c71672932dd087b58146b8777a72ccb9be316a83688fd786
```

The frozen control is an independent fresh-seed reproduction of the stationary/canonical block hierarchy. It is not a byte-for-byte reconstruction of the historical manuscript boundary seed.

For final submission, replace this paragraph or append to it with the DOI assigned to the deposited archive/commit snapshot.