# FK48 clean-room reproducibility package

This package independently reconstructs the 48-four-simplex Freudenthal–Kuhn length-Regge control used to validate the manuscript's gauge-sector scaling mechanism.

## Files

- `fk48_cleanroom_replication.py` — fresh-seed pseudo-constraint control.
- `fk48_stationary_canonical_reproducer.py` — full stationary Lyapunov–Schmidt solve and canonical block extraction.
- `expected_fk48_outputs.json` — acceptance tolerances.
- `run_fk48_repro.py` — one-command stationary/canonical runner and gate checker.
- `requirements_fk48.txt` — tested Python dependency versions.

## Run the headline stationary/canonical gate

```bash
python -m pip install -r requirements_fk48.txt
python run_fk48_repro.py
```

The runner should end with `REPRODUCIBILITY CHECK: PASS`.

## Also run the slower independent pseudo-constraint control

```bash
python run_fk48_repro.py --include-pseudo
```

## Claim boundary

The seed is explicit and independent of the manuscript's historical seed. It independently validates the fourth-order aggregate stationary/canonical gauge block and the contrasting second-order gauge-to-physical block. It is not a byte-for-byte replay of historical data.
