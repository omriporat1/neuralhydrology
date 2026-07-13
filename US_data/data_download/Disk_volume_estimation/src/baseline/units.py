"""Discharge <-> area-normalized runoff-depth conversions (Milestone 2K-G-I I-B).

Implements the binding Stage 1 target transform
(docs/stage1_scientific_baseline_design.md §5/§5a;
docs/stage1_baseline_package_implementation_plan.md §10):

    q_mm_per_h = q_m3s * 3.6 / area_km2
    q_m3s      = q_mm_per_h * area_km2 / 3.6

Derivation of the 3.6 factor: 1 m^3/s = 3600 m^3/h; spread over
area_km2 * 1e6 m^2 gives 3600 / (area_km2 * 1e6) m/h = 3.6 / area_km2 mm/h.

Behavior contract:
- Computation is performed in float64 (inputs are cast up first, so float32
  containers do not silently degrade precision).
- Container structure is preserved: scalar -> float, numpy.ndarray ->
  numpy.ndarray, pandas.Series -> Series (index and name preserved),
  xarray.DataArray -> DataArray (dims and coords preserved). Normal
  library broadcasting between discharge and area applies.
- NaN discharge is preserved (NaN in -> NaN out). Zero discharge converts to
  exact zero. Negative discharge is converted arithmetically at this level —
  enforcing the cleaned-target policy (negatives already NaN) is the package
  builder's job, not this utility's.
- Infinite discharge (+/-inf) is rejected with ValueError: no Flash-NH data
  source can legitimately produce it, so it always indicates an upstream bug;
  NaN is the only permitted nonfinite discharge value.
- Area must be finite and strictly positive everywhere (scalar or array-like);
  zero, negative, NaN, and +/-inf areas raise ValueError. Invalid values are
  never silently replaced.
"""
from __future__ import annotations

import numpy as np


def _as_float64(x):
    """Cast to float64 preserving container type (ndarray/Series/DataArray/scalar)."""
    if hasattr(x, "astype"):
        return x.astype(np.float64)
    return float(x)


def _validate_area(area_km2) -> None:
    """Reject zero, negative, NaN, or infinite basin area (ValueError)."""
    values = np.asarray(area_km2, dtype=np.float64)
    n_nonfinite = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
    if n_nonfinite > 0:
        raise ValueError(
            f"area_km2 must be finite; found {n_nonfinite} NaN/inf value(s)"
        )
    n_nonpositive = int(np.count_nonzero(values <= 0.0))
    if n_nonpositive > 0:
        raise ValueError(
            f"area_km2 must be strictly positive; found {n_nonpositive} "
            f"value(s) <= 0"
        )


def _reject_infinite_discharge(q, name: str) -> None:
    """Reject +/-inf discharge values (ValueError); NaN is permitted."""
    values = np.asarray(q, dtype=np.float64)
    n_inf = int(np.count_nonzero(np.isinf(values)))
    if n_inf > 0:
        raise ValueError(
            f"{name} contains {n_inf} infinite value(s); NaN is the only "
            f"permitted nonfinite discharge value"
        )


def discharge_m3s_to_runoff_mm_per_h(q_m3s, area_km2):
    """Convert discharge [m^3/s] to equivalent runoff depth [mm/h].

    q_mm_per_h = q_m3s * 3.6 / area_km2

    q_m3s: scalar, numpy.ndarray, pandas.Series, or xarray.DataArray
        (NaN preserved; +/-inf rejected).
    area_km2: basin area [km^2], scalar or broadcastable array-like
        (must be finite and > 0 everywhere).
    Returns the same container type as q_m3s, in float64.
    """
    _validate_area(area_km2)
    _reject_infinite_discharge(q_m3s, "q_m3s")
    return _as_float64(q_m3s) * 3.6 / _as_float64(area_km2)


def runoff_mm_per_h_to_discharge_m3s(q_mm_per_h, area_km2):
    """Convert equivalent runoff depth [mm/h] back to discharge [m^3/s].

    q_m3s = q_mm_per_h * area_km2 / 3.6

    q_mm_per_h: scalar, numpy.ndarray, pandas.Series, or xarray.DataArray
        (NaN preserved; +/-inf rejected).
    area_km2: basin area [km^2], scalar or broadcastable array-like
        (must be finite and > 0 everywhere).
    Returns the same container type as q_mm_per_h, in float64.
    """
    _validate_area(area_km2)
    _reject_infinite_discharge(q_mm_per_h, "q_mm_per_h")
    return _as_float64(q_mm_per_h) * _as_float64(area_km2) / 3.6