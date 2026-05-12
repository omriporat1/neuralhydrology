#!/usr/bin/env python3
"""Lightweight deterministic checks for the Flash-NH RBI calculation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.usgs_discharge_probe import calculate_probe_metrics, convert_units  # noqa: E402


def assert_close(actual: float | None, expected: float, tolerance: float = 1e-12) -> None:
    if actual is None:
        raise AssertionError(f"Expected {expected}, got None")
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual}")


def main() -> None:
    constant_series = pd.Series([1.0, 1.0, 1.0], index=pd.date_range("2024-01-01", periods=3, freq="h"))
    rbi, *_ = calculate_probe_metrics(constant_series)
    assert_close(rbi, 0.0)

    pulse_series = pd.Series([0.0, 10.0, 0.0], index=pd.date_range("2024-01-01", periods=3, freq="h"))
    rbi, *_ = calculate_probe_metrics(pulse_series)
    assert_close(rbi, 2.0)

    gap_series = pd.Series([0.0, 10.0, np.nan, 0.0], index=pd.date_range("2024-01-01", periods=4, freq="h"))
    rbi, *_ = calculate_probe_metrics(gap_series)
    assert_close(rbi, 1.0)

    cfs_series = pd.Series([0.0, 10.0, 0.0], index=pd.date_range("2024-01-01", periods=3, freq="h"))
    m3s_series, units_output, converted = convert_units(cfs_series, "ft3/s")
    rbi_cfs, *_ = calculate_probe_metrics(cfs_series)
    rbi_m3s, *_ = calculate_probe_metrics(m3s_series)
    assert_close(rbi_cfs, rbi_m3s)
    if units_output != "m3/s" or not converted:
        raise AssertionError("Expected ft3/s to convert to m3/s")

    zero_series = pd.Series([0.0, 0.0, 0.0], index=pd.date_range("2024-01-01", periods=3, freq="h"))
    rbi, *_ = calculate_probe_metrics(zero_series)
    if rbi is not None:
        raise AssertionError(f"Expected None for zero-denominator series, got {rbi}")

    print("RBI calculation checks passed")


if __name__ == "__main__":
    main()