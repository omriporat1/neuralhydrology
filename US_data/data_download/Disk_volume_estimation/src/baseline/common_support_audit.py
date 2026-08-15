"""Secondary common-support fairness audit for a fixed set of already-completed
NH runs (Sequence-Length-A closure task 5, ``docs/decision_log.md``). The
primary comparison (task 4/6) uses each candidate's own natural support --
its own admitted-sample set, which shrinks with ``seq_length`` due to
lookback-warmup exclusion (see :mod:`src.baseline.nh_raw_space_evaluation`'s
``admitted_mask = isfinite(obs)``). This module answers a narrower secondary
question: restricted to the (basin_id, date) positions admitted by EVERY
candidate simultaneously, does the ranking/interpretation change?

Deliberately the smallest correct helper, not a general sample-intersection
framework: it adds exactly one new operation (per-basin admitted-mask
intersection across candidates, verified against a shared ``date``
coordinate) and reuses everything else --
:func:`src.baseline.nh_seed_evaluation.load_period_results`,
:func:`src.baseline.nh_raw_space_evaluation.derive_basin_area_km2_from_netcdf`,
:func:`src.baseline.nh_raw_space_evaluation.evaluate_basin_raw_space`, and
:func:`src.baseline.nh_raw_space_evaluation.aggregate_raw_space_metrics`
verbatim. No new metric math.

The intersection is computed by masking each candidate's observation array to
NaN outside the shared-admitted positions, then calling
``evaluate_basin_raw_space`` unchanged -- its own
``admitted_mask = isfinite(obs)`` then naturally reduces to exactly the
common-support subset, so no separate metric implementation is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .nh_raw_space_evaluation import (
    DEFAULT_MAX_RELATIVE_MAD,
    DEFAULT_MIN_AREA_SAMPLES,
    RawSpaceEvaluationError,
    aggregate_raw_space_metrics,
    derive_basin_area_km2_from_netcdf,
    evaluate_basin_raw_space,
)
from .nh_seed_evaluation import NHSeedEvaluationError, basin_netcdf_path, load_period_results

__all__ = [
    "CommonSupportAuditError",
    "basin_date_and_admitted",
    "common_support_admitted_mask",
    "common_support_metrics_for_run_period",
]


class CommonSupportAuditError(Exception):
    """Raised for a setup/contract problem (missing basin/candidate result,
    mismatched date coordinates across candidates), never for an ordinary
    poor-skill outcome."""


def basin_date_and_admitted(
    period_results: Mapping, basin_id: str, target_variable: str
) -> tuple:
    """Extracts one basin's ``date`` coordinate and obs/sim flat arrays from an
    already-loaded ``load_period_results(...)`` payload. Returns
    ``(date_values, obs_mm_per_h, sim_mm_per_h)``. Raises
    :class:`CommonSupportAuditError` if the basin or its target data_vars are
    missing (mirrors ``nh_seed_evaluation.raw_space_metrics_for_run_period``'s
    own missing-data handling, but raises rather than silently excluding,
    since a common-support audit over a fixed candidate set should never
    silently drop a basin one candidate happens to be missing)."""
    if basin_id not in period_results:
        raise CommonSupportAuditError(f"basin {basin_id!r} missing from period_results")
    freq_results = period_results[basin_id]
    obs_key = f"{target_variable}_obs"
    sim_key = f"{target_variable}_sim"
    for freq, freq_result in freq_results.items():
        xr_ds = freq_result.get("xr")
        if xr_ds is None or obs_key not in xr_ds.data_vars or sim_key not in xr_ds.data_vars:
            continue
        date_values = xr_ds.coords["date"].values
        obs_mm_per_h = xr_ds[obs_key].values.reshape(-1)
        sim_mm_per_h = xr_ds[sim_key].values.reshape(-1)
        return date_values, obs_mm_per_h, sim_mm_per_h
    raise CommonSupportAuditError(
        f"basin {basin_id!r}: no freq result with both {obs_key!r} and {sim_key!r} data_vars"
    )


def common_support_admitted_mask(
    date_and_obs_by_candidate: Mapping[str, tuple]
) -> np.ndarray:
    """Given ``{candidate_id: (date_values, obs_mm_per_h)}`` for one basin
    (same basin, multiple candidates), verifies every candidate's ``date``
    coordinate is identical (raises :class:`CommonSupportAuditError`
    otherwise -- a silent mismatched-timestamp pairing would invalidate the
    whole audit) and returns the elementwise AND of each candidate's
    ``isfinite(obs)`` admitted mask: the positions admitted by every
    candidate simultaneously."""
    candidate_ids = sorted(date_and_obs_by_candidate)
    if not candidate_ids:
        raise CommonSupportAuditError("no candidates supplied")
    reference_id = candidate_ids[0]
    reference_date, reference_obs = date_and_obs_by_candidate[reference_id]
    combined_mask = np.isfinite(reference_obs)
    for candidate_id in candidate_ids[1:]:
        date_values, obs_mm_per_h = date_and_obs_by_candidate[candidate_id]
        if date_values.shape != reference_date.shape or not np.array_equal(date_values, reference_date):
            raise CommonSupportAuditError(
                f"date coordinate mismatch between {reference_id!r} and {candidate_id!r}: "
                "cannot compute common support over mismatched timestamps"
            )
        combined_mask &= np.isfinite(obs_mm_per_h)
    return combined_mask


def common_support_metrics_for_run_period(
    *,
    period_results_by_candidate: Mapping[str, Mapping],
    package_root,
    target_variable: str,
    lead_hours: int,
    basin_ids: Sequence[str],
    min_area_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> dict:
    """Computes, for each candidate in ``period_results_by_candidate``, raw-space
    metrics restricted to the (basin, date) positions admitted by EVERY
    candidate simultaneously (the common-support subset), over
    ``basin_ids``. ``lead_hours`` is required for the same self-derived-area
    algebraic identity :func:`derive_basin_area_km2_from_netcdf` uses
    elsewhere (``target_mm_per_h[i] == 3.6 * qobs_m3s[i + lead_hours] /
    area_km2``); area is derived once per basin from the package NetCDF's own
    round-trip-consistent samples, independent of which candidate is being
    scored, so the same ``lead_hours`` applies to every candidate.

    Returns ``{"per_basin_common_support": [...], "by_candidate": {candidate_id:
    {"per_basin": [...], "aggregate": {...}}}, "basins_excluded": [...]}``.
    Each ``per_basin_common_support`` entry additionally reports
    ``n_common_admitted`` and, per candidate, ``n_natural_admitted`` (from that
    candidate's own unrestricted ``isfinite(obs)`` count) so the fraction of
    natural support retained under common-support restriction is directly
    inspectable without recomputation.
    """
    candidate_ids = sorted(period_results_by_candidate)
    per_basin_common_support = []
    basins_excluded = []
    per_candidate_per_basin: dict = {cid: [] for cid in candidate_ids}

    for basin_id in basin_ids:
        try:
            date_and_arrays = {
                cid: basin_date_and_admitted(period_results_by_candidate[cid], basin_id, target_variable)
                for cid in candidate_ids
            }
        except CommonSupportAuditError as exc:
            basins_excluded.append({"basin_id": basin_id, "reason": str(exc)})
            continue

        date_and_obs = {cid: (date_and_arrays[cid][0], date_and_arrays[cid][1]) for cid in candidate_ids}
        try:
            common_mask = common_support_admitted_mask(date_and_obs)
        except CommonSupportAuditError as exc:
            basins_excluded.append({"basin_id": basin_id, "reason": str(exc)})
            continue

        nc_path = basin_netcdf_path(package_root, basin_id)
        try:
            area_result = derive_basin_area_km2_from_netcdf(
                nc_path,
                basin_id=basin_id,
                target_variable=target_variable,
                lead_hours=lead_hours,
                min_samples=min_area_samples,
                max_relative_mad=max_relative_mad,
            )
        except RawSpaceEvaluationError as exc:
            basins_excluded.append({"basin_id": basin_id, "reason": f"area derivation failed: {exc}"})
            continue
        if not area_result.consistent:
            basins_excluded.append(
                {
                    "basin_id": basin_id,
                    "reason": (
                        f"area derivation inconsistent: relative_mad={area_result.relative_mad:.6g} "
                        f"> max_relative_mad={max_relative_mad:.6g}"
                    ),
                }
            )
            continue

        n_common_admitted = int(common_mask.sum())
        basin_row = {"basin_id": basin_id, "n_common_admitted": n_common_admitted}
        for cid in candidate_ids:
            _, obs_mm_per_h, sim_mm_per_h = date_and_arrays[cid]
            basin_row[f"n_natural_admitted__{cid}"] = int(np.isfinite(obs_mm_per_h).sum())

            obs_common = np.where(common_mask, obs_mm_per_h, np.nan)
            basin_metrics = evaluate_basin_raw_space(
                basin_id=basin_id,
                obs_mm_per_h=obs_common,
                sim_mm_per_h=sim_mm_per_h,
                area_km2=area_result.area_km2,
            )
            basin_metrics["freq"] = "1h"
            basin_metrics["area_n_samples"] = area_result.n_samples
            basin_metrics["area_relative_mad"] = area_result.relative_mad
            per_candidate_per_basin[cid].append(basin_metrics)

        per_basin_common_support.append(basin_row)

    by_candidate = {}
    for cid in candidate_ids:
        rows = per_candidate_per_basin[cid]
        by_candidate[cid] = {
            "per_basin": rows,
            "aggregate": aggregate_raw_space_metrics(rows) if rows else {
                "n_basins": 0,
                "n_admitted_total": 0,
                "n_sim_nonfinite_at_admitted_total": 0,
                "metrics": {},
            },
        }

    return {
        "candidate_ids": candidate_ids,
        "n_basins_requested": len(basin_ids),
        "n_basins_evaluated": len(per_basin_common_support),
        "n_basins_excluded": len(basins_excluded),
        "basins_excluded": basins_excluded,
        "per_basin_common_support": per_basin_common_support,
        "by_candidate": by_candidate,
    }
