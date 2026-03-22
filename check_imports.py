#!/usr/bin/env python3
"""Import gate: verifies pymaster and healpy are available before running the pipeline."""

import sys

def check():
    errors = []
    try:
        import pymaster as nmt  # noqa: F401
        print("NaMaster OK")
    except ImportError as e:
        errors.append(f"pymaster: {e}")

    try:
        import healpy as hp  # noqa: F401
        print("healpy OK")
    except ImportError as e:
        errors.append(f"healpy: {e}")

    if errors:
        print("\nImport gate FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print("\nInstall missing packages:  conda install -c conda-forge namaster healpy", file=sys.stderr)
        sys.exit(1)

    print("Import gate passed — ready to run pipeline.")

if __name__ == "__main__":
    check()
