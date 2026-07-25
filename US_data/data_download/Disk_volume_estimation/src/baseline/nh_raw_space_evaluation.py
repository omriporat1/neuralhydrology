"""Raw-space (m^3/s) evaluation layer for a completed NeuralHydrology 1.13
run (Stage 1 initial full-population seed training, see
``docs/decision_log.md``'s 2026-07-25 entry).

NH's own evaluation output (``<period>_results.p``, read by
``nh_evaluation_check.py``) reports observations/predictions in the
area-normalized training scale (``qobs_mm_per_h_lead06``, mm/h-equivalent
runoff depth), never raw discharge. This module is the ONLY place in the
repository that reconverts those values back to raw m^3/s
(``docs/stage1_scientific_baseline_design.md``'s binding primary metric
space) and computes hydrological skill metrics on them.

Basin area is not stored anywhere in ``stage1_scientific_package_v002`` (no
``DRAIN_SQKM`` field in any NetCDF or manifest -- confirmed by inspection of
``package_netcdf.py``). Rather than requiring a new h2o->Moriah transfer of
the external static-attribute matrix or a separate area CSV, this module
self-derives each basin's area directly from data already present in its own
package NetCDF: the diagnostic raw ``qobs_m3s`` series and the built
lead-shifted target (e.g. ``qobs_mm_per_h_lead06``) satisfy, by construction
(see ``docs/stage1_scientific_baseline_design.md`` Sec 5a's own round-trip
audit requirement)::

    qobs_mm_per_h_lead{L}[t] == discharge_m3s_to_runoff_mm_per_h(qobs_m3s[t + L], area_km2)

so ``area_km2`` can be recovered algebraically from many (t) samples per
basin, reduced robustly (median + a relative-MAD consistency check) rather
than trusting a single sample.

Reuses :mod:`src.baseline.units` for the discharge<->runoff-depth conversion
contract (float64, NaN-preserving, infinite-discharge-rejecting,
positive-finite-area-enforcing) and
:class:`src.baseline.package_audit.AuditReport` / ``sha256_file`` for the
generic evidence-reporting idiom used by every other Stage 1 audit module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .units import runoff_mm_per_h_to_discharge_m3s

__all__ = [
    "RawSpaceEvaluationError",
    "BasinAreaResult",
    "derive_basin_area_km2",
    "derive_basin_area_km2_from_netcdf",
    "PeriodConversionResult",
    "convert_period_to_raw_space",
    "raw_space_metrics",
    "aggregate_raw_space_metrics",
    "evaluate_basin_raw_space",
]

DEFAULT_MIN_AREA_SAMPLES = 100
DEFAULT_MAX_RELATIVE_MAD = 1e-4


class RawSpaceEvaluationError(Exception):
    """Raised for a setup/contract problem (insufficient samples, an
    inconsistent algebraic round-trip, shape mismatch, etc.), never for an
    ordinary "this basin/epoch has poor skill" outcome."""


# ---------------------------------------------------------------------------
# Basin-area self-derivation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasinAreaResult:
    """Result of self-deriving one basin's area from its own package data."""

    basin_id: str
    area_km2: float
    n_samples: int
    relative_mad: float
    consistent: bool


def derive_basin_area_km2(
    qobs_m3s: np.ndarray,
    target_mm_per_h: np.ndarray,
    lead_hours: int,
    *,
    basin_id: str = "",
    min_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> BasinAreaResult:
    """Self-derive one basin's area [km^2] from its own hourly time series.

    ``qobs_m3s`` and ``target_mm_per_h`` (e.g. ``qobs_mm_per_h_lead06``) must
    be the SAME basin's full, aligned hourly series (same length, same
    timeline, no reindexing performed here -- callers own alignment).
    ``target_mm_per_h[i]`` is expected (by the Stage 1 package build
    contract) to equal the mm/h-conversion of ``qobs_m3s[i + lead_hours]``;
    inverting that identity for each valid ``i`` gives one area estimate:

        area_km2_estimate[i] = 3.6 * qobs_m3s[i + lead_hours] / target_mm_per_h[i]

    Only strictly positive, finite pairs are used. Raises
    :class:`RawSpaceEvaluationError` if fewer than ``min_samples`` valid
    pairs are available (an area estimate from too few samples is not
    trustworthy). Returns a :class:`BasinAreaResult` whose ``consistent``
    flag is ``False`` (never silently dropped -- callers must check it) when
    the relative median-absolute-deviation of the per-sample estimates
    exceeds ``max_relative_mad``, which would indicate the algebraic
    identity does not hold cleanly for this basin (e.g. a build-time area
    inconsistency) rather than ordinary floating-point noise.
    """
    qobs_m3s = np.asarray(qobs_m3s, dtype=np.float64)
    target_mm_per_h = np.asarray(target_mm_per_h, dtype=np.float64)
    if qobs_m3s.ndim != 1 or target_mm_per_h.ndim != 1:
        raise RawSpaceEvaluationError("qobs_m3s and target_mm_per_h must both be 1-D series")
    if len(qobs_m3s) != len(target_mm_per_h):
        raise RawSpaceEvaluationError(
            f"qobs_m3s (len {len(qobs_m3s)}) and target_mm_per_h (len {len(target_mm_per_h)}) "
            "must be the same length (same aligned timeline)"
        )
    if lead_hours <= 0:
        raise RawSpaceEvaluationError(f"lead_hours must be positive, got {lead_hours}")

    n = len(target_mm_per_h)
    usable_n = n - lead_hours
    if usable_n <= 0:
        raise RawSpaceEvaluationError(
            f"series length {n} is too short for lead_hours={lead_hours}"
        )

    target_head = target_mm_per_h[:usable_n]
    qobs_shifted = qobs_m3s[lead_hours:lead_hours + usable_n]

    valid = np.isfinite(target_head) & np.isfinite(qobs_shifted) & (target_head > 0.0) & (qobs_shifted > 0.0)
    n_valid = int(valid.sum())
    if n_valid < min_samples:
        raise RawSpaceEvaluationError(
            f"basin {basin_id!r}: only {n_valid} valid area-estimate sample(s) "
            f"(need >= {min_samples}); cannot derive a trustworthy area"
        )

    estimates = 3.6 * qobs_shifted[valid] / target_head[valid]
    median = float(np.median(estimates))
    mad = float(np.median(np.abs(estimates - median)))
    relative_mad = mad / median if median > 0.0 else float("inf")
    consistent = relative_mad <= max_relative_mad

    return BasinAreaResult(
        basin_id=basin_id,
        area_km2=median,
        n_samples=n_valid,
        relative_mad=relative_mad,
        consistent=consistent,
    )


def derive_basin_area_km2_from_netcdf(
    nc_path,
    *,
    basin_id: str,
    target_variable: str,
    lead_hours: int,
    min_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> BasinAreaResult:
    """I/O wrapper: reads ``qobs_m3s`` and ``target_variable`` from one
    package basin NetCDF (``time_series/<basin_id>.nc``) and calls
    :func:`derive_basin_area_km2`."""
    import xarray as xr

    nc_path = Path(nc_path)
    if not nc_path.is_file():
        raise RawSpaceEvaluationError(f"basin NetCDF not found: {nc_path}")
    with xr.open_dataset(nc_path) as ds:
        if "qobs_m3s" not in ds.data_vars:
            raise RawSpaceEvaluationError(f"{nc_path}: missing diagnostic variable 'qobs_m3s'")
        if target_variable not in ds.data_vars:
            raise RawSpaceEvaluationError(f"{nc_path}: missing target variable {target_variable!r}")
        qobs_m3s = ds["qobs_m3s"].values
        target_mm_per_h = ds[target_variable].values

    return derive_basin_area_km2(
        qobs_m3s,
        target_mm_per_h,
        lead_hours,
        basin_id=basin_id,
        min_samples=min_samples,
        max_relative_mad=max_relative_mad,
    )


# ---------------------------------------------------------------------------
# mm/h -> m^3/s conversion with consistent NaN masking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodConversionResult:
    """One basin/period's raw-space series plus the NaN-masking accounting
    needed to audit "consistent NaN masking" (item required by Part B)."""

    obs_m3s: np.ndarray
    sim_m3s: np.ndarray
    admitted_mask: np.ndarray
    n_admitted: int
    n_total: int
    n_sim_nonfinite_at_admitted: int


def convert_period_to_raw_space(
    obs_mm_per_h: np.ndarray,
    sim_mm_per_h: np.ndarray,
    area_km2: float,
) -> PeriodConversionResult:
    """Convert one basin/period's NH-native mm/h obs+sim series to raw m^3/s.

    "Admitted" samples are exactly those with a finite observation -- the
    same convention ``nh_evaluation_check.py`` already uses (a qobs-NaN hour
    is excluded from loss/metrics, but the model still emits an output for
    every window). Model outputs that are non-finite at an admitted position
    are counted (``n_sim_nonfinite_at_admitted``) rather than silently
    dropped, so callers can surface it as an evaluation-integrity error
    instead of a quietly-shrunk sample count.
    """
    obs_mm_per_h = np.asarray(obs_mm_per_h, dtype=np.float64)
    sim_mm_per_h = np.asarray(sim_mm_per_h, dtype=np.float64)
    if obs_mm_per_h.shape != sim_mm_per_h.shape:
        raise RawSpaceEvaluationError(
            f"obs shape {obs_mm_per_h.shape} != sim shape {sim_mm_per_h.shape}"
        )

    admitted_mask = np.isfinite(obs_mm_per_h)
    n_admitted = int(admitted_mask.sum())
    sim_at_admitted = sim_mm_per_h[admitted_mask]
    n_sim_nonfinite_at_admitted = int((~np.isfinite(sim_at_admitted)).sum())

    obs_m3s_full = np.full_like(obs_mm_per_h, np.nan, dtype=np.float64)
    sim_m3s_full = np.full_like(sim_mm_per_h, np.nan, dtype=np.float64)
    # Convert only at admitted positions; runoff_mm_per_h_to_discharge_m3s
    # rejects +/-inf but tolerates NaN, so non-finite sim values at admitted
    # positions safely become NaN in raw space rather than raising -- the
    # n_sim_nonfinite_at_admitted count above is the authoritative signal for
    # that condition, matching nh_evaluation_check.py's error-not-silent-drop
    # convention.
    obs_admitted = obs_mm_per_h[admitted_mask]
    sim_admitted = np.where(np.isfinite(sim_at_admitted), sim_at_admitted, np.nan)
    obs_m3s_full[admitted_mask] = runoff_mm_per_h_to_discharge_m3s(obs_admitted, area_km2)
    sim_m3s_full[admitted_mask] = runoff_mm_per_h_to_discharge_m3s(sim_admitted, area_km2)

    return PeriodConversionResult(
        obs_m3s=obs_m3s_full,
        sim_m3s=sim_m3s_full,
        admitted_mask=admitted_mask,
        n_admitted=n_admitted,
        n_total=int(obs_mm_per_h.size),
        n_sim_nonfinite_at_admitted=n_sim_nonfinite_at_admitted,
    )


# ---------------------------------------------------------------------------
# Raw-space metrics
# ---------------------------------------------------------------------------

def raw_space_metrics(obs_m3s: np.ndarray, sim_m3s: np.ndarray) -> dict:
    """Raw-space (m^3/s) skill metrics on a single basin/period's admitted
    samples. Both inputs must already be NaN-free and equal-length (callers
    pass ``PeriodConversionResult.obs_m3s[mask]`` / ``sim_m3s[mask]``, or
    equivalent). NSE/KGE/Pearson-r are NaN when fewer than 2 finite samples
    or the observation series has zero variance (undefined, not zero)."""
    obs = np.asarray(obs_m3s, dtype=np.float64)
    sim = np.asarray(sim_m3s, dtype=np.float64)
    if obs.shape != sim.shape:
        raise RawSpaceEvaluationError(f"obs shape {obs.shape} != sim shape {sim.shape}")
    finite = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[finite]
    sim = sim[finite]
    n = int(obs.size)

    result = {
        "n_samples": n,
        "nse": float("nan"),
        "kge": float("nan"),
        "kge_r": float("nan"),
        "kge_alpha": float("nan"),
        "kge_beta": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "pearson_r": float("nan"),
        "bias": float("nan"),
        "pbias": float("nan"),
    }
    if n < 2:
        return result

    obs_mean = float(np.mean(obs))
    error = sim - obs
    result["rmse"] = float(np.sqrt(np.mean(error ** 2)))
    result["mae"] = float(np.mean(np.abs(error)))
    result["bias"] = float(np.mean(error))

    obs_sum = float(np.sum(obs))
    result["pbias"] = float(100.0 * np.sum(error) / obs_sum) if obs_sum != 0.0 else float("nan")

    denom = float(np.sum((obs - obs_mean) ** 2))
    if denom > 0.0:
        result["nse"] = float(1.0 - np.sum(error ** 2) / denom)

    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))
    sim_mean = float(np.mean(sim))
    if obs_std > 0.0 and sim_std > 0.0:
        r = float(np.corrcoef(obs, sim)[0, 1])
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean if obs_mean != 0.0 else float("nan")
        result["pearson_r"] = r
        result["kge_r"] = r
        result["kge_alpha"] = alpha
        result["kge_beta"] = beta
        if np.isfinite(beta):
            result["kge"] = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return result


def evaluate_basin_raw_space(
    *,
    basin_id: str,
    obs_mm_per_h: np.ndarray,
    sim_mm_per_h: np.ndarray,
    area_km2: float,
) -> dict:
    """Convert one basin/period to raw space and compute its metrics in one
    step. Returns a flat dict merging :class:`PeriodConversionResult`
    accounting fields with :func:`raw_space_metrics`'s output."""
    conversion = convert_period_to_raw_space(obs_mm_per_h, sim_mm_per_h, area_km2)
    metrics = raw_space_metrics(
        conversion.obs_m3s[conversion.admitted_mask],
        conversion.sim_m3s[conversion.admitted_mask],
    )
    return {
        "basin_id": basin_id,
        "area_km2": area_km2,
        "n_total": conversion.n_total,
        "n_admitted": conversion.n_admitted,
        "n_sim_nonfinite_at_admitted": conversion.n_sim_nonfinite_at_admitted,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Cross-basin aggregation
# ---------------------------------------------------------------------------

_AGGREGATE_METRIC_NAMES = ("nse", "kge", "rmse", "mae", "pearson_r", "bias", "pbias")


def aggregate_raw_space_metrics(per_basin_results: Sequence[Mapping]) -> dict:
    """Aggregate a list of :func:`evaluate_basin_raw_space` dicts into
    per-metric median/mean/quartiles, finite-value counts, and total sample
    counts across the whole basin population."""
    n_basins = len(per_basin_results)
    aggregate: dict = {
        "n_basins": n_basins,
        "n_admitted_total": int(sum(r["n_admitted"] for r in per_basin_results)),
        "n_sim_nonfinite_at_admitted_total": int(
            sum(r["n_sim_nonfinite_at_admitted"] for r in per_basin_results)
        ),
        "metrics": {},
    }
    for metric_name in _AGGREGATE_METRIC_NAMES:
        values = np.array([r[metric_name] for r in per_basin_results], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        n_finite = int(finite_values.size)
        if n_finite == 0:
            aggregate["metrics"][metric_name] = {
                "n_finite_basins": 0,
                "median": float("nan"),
                "mean": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
            }
            continue
        aggregate["metrics"][metric_name] = {
            "n_finite_basins": n_finite,
            "median": float(np.median(finite_values)),
            "mean": float(np.mean(finite_values)),
            "q25": float(np.percentile(finite_values, 25)),
            "q75": float(np.percentile(finite_values, 75)),
        }
    return aggregate
