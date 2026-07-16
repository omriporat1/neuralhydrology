"""Lead-target construction from an hourly discharge series
(Milestone 2K-G-I I-C1).

Builds a single package-ready lead target from a source discharge series
through one fixed, explicit two-step sequence:

    1. convert q_m3s -> mm/h via
       ``units.discharge_m3s_to_runoff_mm_per_h(q_m3s, area_km2)``
    2. shift the converted series by ``-lead_hours`` so that the value at
       package timestamp ``t`` is the converted observation at ``t + lead``.

Basin area is constant for a given series, so unit conversion and the
integer-hour shift are both linear operations that commute -- converting
then shifting is not numerically distinguished from shifting then
converting. This module defines and tests convert-then-shift as the single
public API for clarity and for attaching consistent metadata in one place;
later builder code should call this API rather than reimplementing the
transform, not because the alternative ordering would be wrong.

This module only operates on an already-regularized, strictly increasing,
unique, hourly ``pandas.DatetimeIndex`` supplied by the caller. It never
reindexes, resamples, or infers missing hours -- reconciling the source
schedule with the forcing/attribute join is the package builder's job, not
this utility's. Source NaNs (including NaNs introduced by forcing-gap
exclusion upstream) propagate through the shift unchanged; the last
``lead_hours`` entries of every target are NaN because no future
observation exists for them. No interpolation or filling is performed
anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baseline.units import discharge_m3s_to_runoff_mm_per_h

#: Binding Stage 1 lead set (config/stage1_scientific_baseline_v001.yaml
#: ``target.leads_hours``). Provided as a convenience default only -- the
#: functions below accept any positive integer lead.
DEFAULT_LEADS_HOURS = (1, 3, 6, 12)

#: Mirrors config/stage1_scientific_baseline_v001.yaml
#: ``target.variable_name_template``.
DEFAULT_VARIABLE_NAME_TEMPLATE = "qobs_mm_per_h_lead{lead:02d}"

DEFAULT_SOURCE_VARIABLE = "qobs_m3s"
DEFAULT_UNITS = "mm/h"

_HOUR = pd.Timedelta(hours=1)


class LeadTargetError(ValueError):
    """Raised for an invalid source time index or lead specification."""


@dataclass(frozen=True)
class LeadTargetMetadata:
    """Metadata the package builder needs to attach to a lead-target column."""

    variable_name: str
    lead_hours: int
    units: str
    source_variable: str
    long_name: str


def variable_name_for_lead(
    lead_hours: int, template: str = DEFAULT_VARIABLE_NAME_TEMPLATE
) -> str:
    """Render the package variable name for one lead, e.g. ``qobs_mm_per_h_lead06``."""
    return template.format(lead=lead_hours)


def _validate_hourly_index(index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise LeadTargetError(
            f"time index must be a pandas.DatetimeIndex, got {type(index)!r}"
        )
    n = len(index)
    if n < 2:
        raise LeadTargetError(
            f"time index must have at least 2 timestamps to validate hourly "
            f"spacing, got {n}"
        )
    if index.has_duplicates:
        raise LeadTargetError("time index must not contain duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise LeadTargetError(
            "time index must be strictly increasing (ascending); found a "
            "non-increasing or descending step"
        )
    deltas = index[1:] - index[:-1]
    irregular = np.asarray(deltas != _HOUR)
    if irregular.any():
        first_bad = int(np.flatnonzero(irregular)[0])
        raise LeadTargetError(
            f"time index must be strictly hourly with no gaps; found "
            f"{int(irregular.sum())} irregular step(s), first at position "
            f"{first_bad} ({index[first_bad]} -> {index[first_bad + 1]}, "
            f"delta {deltas[first_bad]})"
        )


def _validate_lead_hours(lead_hours) -> int:
    if isinstance(lead_hours, bool) or not isinstance(lead_hours, (int, np.integer)):
        raise LeadTargetError(f"lead_hours must be a positive int, got {lead_hours!r}")
    if lead_hours <= 0:
        raise LeadTargetError(f"lead_hours must be positive, got {int(lead_hours)}")
    return int(lead_hours)


def build_lead_target(
    q_m3s: pd.Series,
    area_km2,
    lead_hours: int,
    *,
    source_variable: str = DEFAULT_SOURCE_VARIABLE,
    variable_name_template: str = DEFAULT_VARIABLE_NAME_TEMPLATE,
    units: str = DEFAULT_UNITS,
) -> tuple[pd.Series, LeadTargetMetadata]:
    """Build one package-ready lead target from a source discharge series.

    q_m3s: pandas.Series of discharge [m^3/s] indexed by a strictly
        increasing, unique, gap-free hourly pandas.DatetimeIndex. Not
        reindexed or resampled -- an irregular index raises LeadTargetError.
    area_km2: basin area [km^2]; forwarded to
        units.discharge_m3s_to_runoff_mm_per_h (finite, > 0 required there).
    lead_hours: positive integer number of hours ahead.

    Returns (target_series, metadata):
      - target_series has the same index, length, and (unchanged) timestamps
        as q_m3s. target_series[t] == mm_per_h[t + lead_hours]. The last
        lead_hours entries are NaN (no future observation exists for them);
        any source NaN at t + lead_hours propagates to target[t]. No
        interpolation or filling is performed.
      - metadata carries variable_name, lead_hours, units, source_variable,
        and long_name for the package builder to attach to the column.
    """
    if not isinstance(q_m3s, pd.Series):
        raise LeadTargetError(f"q_m3s must be a pandas.Series, got {type(q_m3s)!r}")
    _validate_hourly_index(q_m3s.index)
    lead_hours = _validate_lead_hours(lead_hours)

    mm_per_h = discharge_m3s_to_runoff_mm_per_h(q_m3s, area_km2)
    target = mm_per_h.shift(-lead_hours)
    variable_name = variable_name_for_lead(lead_hours, variable_name_template)
    target.name = variable_name

    metadata = LeadTargetMetadata(
        variable_name=variable_name,
        lead_hours=lead_hours,
        units=units,
        source_variable=source_variable,
        long_name=(
            f"{lead_hours}-hour-ahead equivalent runoff depth, i.e. "
            f"{source_variable} converted to {units} and shifted {lead_hours} "
            f"hour(s) ahead of the package timestamp"
        ),
    )
    return target, metadata


def build_lead_targets(
    q_m3s: pd.Series,
    area_km2,
    leads_hours=DEFAULT_LEADS_HOURS,
    *,
    source_variable: str = DEFAULT_SOURCE_VARIABLE,
    variable_name_template: str = DEFAULT_VARIABLE_NAME_TEMPLATE,
    units: str = DEFAULT_UNITS,
) -> dict[str, tuple[pd.Series, LeadTargetMetadata]]:
    """Convenience wrapper: build_lead_target for each lead in leads_hours.

    Returns a dict keyed by rendered variable_name, values (series, metadata)
    exactly as returned by build_lead_target. The hourly-index and area
    validation happen independently for each lead (no shared mutable state);
    this is a plain loop, not a package builder.
    """
    result = {}
    for lead_hours in leads_hours:
        series, metadata = build_lead_target(
            q_m3s,
            area_km2,
            lead_hours,
            source_variable=source_variable,
            variable_name_template=variable_name_template,
            units=units,
        )
        result[metadata.variable_name] = (series, metadata)
    return result
