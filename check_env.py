#!/usr/bin/env python3
"""Lightweight environment import check. No training, no dataset loading, no OPF solve."""

import importlib
import sys

REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "torch",
    "gurobipy",
    "openpyxl",
]

failed = []
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except Exception as exc:
        failed.append((pkg, str(exc)))
        print(f"[FAIL] {pkg}: {exc}")

if failed:
    print("\nEnvironment check failed.")
    sys.exit(1)

print("\nEnvironment check passed.")
