"""v2 six-axis fixed-support (120h-floor common-support) contract: schema,
loader/validator, and evaluator (Section E, additive six-axis campaign
foundation).

Defines a FROZEN, versioned, checksummed evaluation-support artifact -- one
120h-floor common-support per-basin ``date``/admitted map -- that the v2
six-axis campaign's PRIMARY objective
(:data:`src.baseline.sweep_v2_six_axis_campaign.OBJECTIVE_ID_V2`,
``common120_raw_space_nse_v001``) must be computed on, so every v2 trial
across all seven ``seq_length`` candidates is scored on exactly the same
evaluation support. This is scientifically justified by monotone nesting: a
timestamp admitted by ``compute_history_valid(..., seq_length=120)`` has a
gap-free 120h lookback window, and every shorter-``seq_length`` lookback
window is a contiguous sub-window of that gap-free window, hence also
gap-free -- so the 120h-floor admitted set is a valid common support for
every candidate in the domain (see
:func:`src.baseline.validity_mask.compute_history_valid`/
:func:`compute_boundary_valid`). Natural support (each trial's own
admitted-sample set, which grows for shorter ``seq_length``) remains
available only as a secondary diagnostic
(:func:`evaluate_natural_support_raw_space_metrics`), never the objective.

Reuses, unmodified: :func:`src.baseline.nh_seed_evaluation.load_period_results`
/ :func:`basin_netcdf_path` / :func:`raw_space_metrics_for_run_period`,
:func:`src.baseline.nh_raw_space_evaluation.derive_basin_area_km2_from_netcdf`
/ :func:`evaluate_basin_raw_space` / :func:`aggregate_raw_space_metrics`. The
masking approach -- restrict ``obs`` to NaN outside the admitted support,
then call :func:`evaluate_basin_raw_space` unmodified so its own
``admitted_mask = isfinite(obs)`` naturally reduces to the support subset --
and the exact-``date``-coordinate-equality precedent both mirror the
already-qualified :mod:`src.baseline.common_support_audit` module (Sequence-
Length-A closure task 5); no new metric math is introduced here.

This module does NOT compute the frozen support artifact from real
production data (out of scope for this local-only foundation task); it
defines the schema/loader/validator/evaluator, exercised only against
synthetic fixtures in tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .nh_raw_space_evaluation import (
    DEFAULT_MAX_RELATIVE_MAD,
    DEFAULT_MIN_AREA_SAMPLES,
    RawSpaceEvaluationError,
    aggregate_raw_space_metrics,
    derive_basin_area_km2_from_netcdf,
    evaluate_basin_raw_space,
)
from .nh_seed_evaluation import basin_netcdf_path, load_period_results, raw_space_metrics_for_run_period
from .sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2, SEQ_LENGTH_MAX

__all__ = [
    "FixedSupportContractError",
    "CONTRACT_SCHEMA_NAME",
    "CONTRACT_SCHEMA_VERSION",
    "build_fixed_support_contract",
    "validate_fixed_support_contract",
    "write_fixed_support_contract",
    "load_fixed_support_contract",
    "evaluate_fixed_support_raw_space_metrics",
    "evaluate_natural_support_raw_space_metrics",
    "extract_v2_objective_from_fixed_support_result",
]


class FixedSupportContractError(ValueError):
    """Raised for a malformed/inconsistent fixed-support contract, an
    attempted overwrite, or an evaluation-time identity contradiction
    (basin/date/support mismatch, checksum mismatch, wrong
    ``objective_scope``). Never raised for an ordinary poor-skill outcome."""


CONTRACT_SCHEMA_NAME = "flashnh_stage1_v2_fixed_support_contract"
CONTRACT_SCHEMA_VERSION = 1

_REQUIRED_KEYS = {
    "schema_name",
    "schema_version",
    "contract_id",
    "seq_length_floor",
    "lead_hours",
    "target_variable",
    "period",
    "date_start",
    "date_end",
    "source_gap_policy_identity",
    "screening_basin_ids_sha256",
    "basin_ids",
    "date_dtype",
    "per_basin_support",
    "eligible_counts",
    "checksum_sha256",
}


def _serialize_date_array(date_values: np.ndarray) -> tuple:
    arr = np.asarray(date_values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return list(np.datetime_as_string(arr, unit="ns")), "datetime64"
    if np.issubdtype(arr.dtype, np.integer):
        return [int(v) for v in arr.tolist()], "int64"
    raise FixedSupportContractError(
        f"unsupported date coordinate dtype {arr.dtype!r}; expected datetime64 or integer"
    )


def _deserialize_date_array(values: list, date_dtype: str) -> np.ndarray:
    if date_dtype == "datetime64":
        return np.array(values, dtype="datetime64[ns]")
    if date_dtype == "int64":
        return np.array(values, dtype="int64")
    raise FixedSupportContractError(f"unsupported date_dtype {date_dtype!r}; expected 'datetime64' or 'int64'")


def _canonical_payload_for_checksum(payload: Mapping) -> bytes:
    body = {k: v for k, v in payload.items() if k != "checksum_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_fixed_support_contract(
    *,
    contract_id: str,
    lead_hours: int,
    target_variable: str,
    period: str,
    date_start: str,
    date_end: str,
    source_gap_policy_identity: str,
    screening_basin_ids_sha256: str,
    per_basin_date: Mapping[str, np.ndarray],
    per_basin_admitted: Mapping[str, np.ndarray],
    seq_length_floor: int = SEQ_LENGTH_MAX,
) -> dict:
    """Builds (but does not write) a fixed-support contract payload from
    per-basin ``(date coordinate, boolean 120h-floor-admitted mask)`` pairs
    -- typically ``per_basin_admitted[basin_id] =
    compute_history_valid(index, bad_hour_mask, seq_length_floor) &
    compute_boundary_valid(index, lead_hours)`` (:mod:`validity_mask`
    primitives) evaluated against the basin's own ``date`` coordinate.
    Persists, per basin, only the ADMITTED subset of the date coordinate
    (not the full timeline) plus its eligible count, so evaluation-time
    alignment reduces to an exact-membership check against the run's own
    ``date`` coordinate (mirrors :mod:`common_support_audit`'s exact
    date-coordinate-equality precedent).
    """
    basin_ids = sorted(per_basin_date)
    if basin_ids != sorted(per_basin_admitted):
        raise FixedSupportContractError(
            "per_basin_date and per_basin_admitted must share exactly the same basin_id set"
        )
    if not basin_ids:
        raise FixedSupportContractError("no basins supplied; cannot build an empty fixed-support contract")

    date_dtype: Optional[str] = None
    per_basin_support: dict = {}
    eligible_counts: dict = {}
    for basin_id in basin_ids:
        date_values = np.asarray(per_basin_date[basin_id])
        admitted = np.asarray(per_basin_admitted[basin_id], dtype=bool)
        if admitted.shape != date_values.shape:
            raise FixedSupportContractError(
                f"basin {basin_id!r}: admitted mask shape {admitted.shape} != date shape {date_values.shape}"
            )
        admitted_dates = date_values[admitted]
        serialized, this_dtype = _serialize_date_array(admitted_dates)
        if date_dtype is None:
            date_dtype = this_dtype
        elif this_dtype != date_dtype:
            raise FixedSupportContractError(
                f"basin {basin_id!r}: date dtype {this_dtype!r} != established {date_dtype!r} "
                "(all basins must share one date representation)"
            )
        per_basin_support[basin_id] = serialized
        eligible_counts[basin_id] = int(admitted.sum())

    payload = {
        "schema_name": CONTRACT_SCHEMA_NAME,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": contract_id,
        "seq_length_floor": int(seq_length_floor),
        "lead_hours": int(lead_hours),
        "target_variable": target_variable,
        "period": period,
        "date_start": date_start,
        "date_end": date_end,
        "source_gap_policy_identity": source_gap_policy_identity,
        "screening_basin_ids_sha256": screening_basin_ids_sha256,
        "basin_ids": basin_ids,
        "date_dtype": date_dtype,
        "per_basin_support": per_basin_support,
        "eligible_counts": eligible_counts,
    }
    payload["checksum_sha256"] = hashlib.sha256(_canonical_payload_for_checksum(payload)).hexdigest()
    return payload


def validate_fixed_support_contract(data: dict, *, expected_contract_id: str = OBJECTIVE_ID_V2) -> dict:
    """Strict schema/checksum validation. Raises :class:`FixedSupportContractError`
    on any missing/extra key, wrong schema/contract identity, wrong
    ``seq_length_floor``, inconsistent basin_id sets, inconsistent eligible
    counts, or checksum mismatch (a checksum mismatch means the payload was
    altered after checksumming -- never silently tolerated)."""
    if not isinstance(data, dict):
        raise FixedSupportContractError(f"contract must be a mapping, got {type(data).__name__}")
    missing = _REQUIRED_KEYS - set(data)
    extra = set(data) - _REQUIRED_KEYS
    if missing:
        raise FixedSupportContractError(f"contract missing required key(s): {sorted(missing)}")
    if extra:
        raise FixedSupportContractError(f"contract has unexpected extra key(s): {sorted(extra)}")

    if data["schema_name"] != CONTRACT_SCHEMA_NAME:
        raise FixedSupportContractError(f"schema_name must be {CONTRACT_SCHEMA_NAME!r}, got {data['schema_name']!r}")
    if data["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise FixedSupportContractError(
            f"schema_version must be {CONTRACT_SCHEMA_VERSION!r}, got {data['schema_version']!r}"
        )
    if data["contract_id"] != expected_contract_id:
        raise FixedSupportContractError(f"contract_id must be {expected_contract_id!r}, got {data['contract_id']!r}")
    if data["seq_length_floor"] != SEQ_LENGTH_MAX:
        raise FixedSupportContractError(
            f"seq_length_floor must be {SEQ_LENGTH_MAX!r} (the v2 domain ceiling), got {data['seq_length_floor']!r}"
        )

    basin_ids = data["basin_ids"]
    if not isinstance(basin_ids, list) or basin_ids != sorted(set(basin_ids)):
        raise FixedSupportContractError("basin_ids must be a sorted list of unique basin ids")
    if set(basin_ids) != set(data["per_basin_support"]) or set(basin_ids) != set(data["eligible_counts"]):
        raise FixedSupportContractError(
            "basin_ids must exactly match the per_basin_support and eligible_counts key sets"
        )
    for basin_id in basin_ids:
        n_expected = data["eligible_counts"][basin_id]
        n_actual = len(data["per_basin_support"][basin_id])
        if n_expected != n_actual:
            raise FixedSupportContractError(
                f"basin {basin_id!r}: eligible_counts={n_expected} does not match "
                f"len(per_basin_support)={n_actual}"
            )

    recomputed = hashlib.sha256(_canonical_payload_for_checksum(data)).hexdigest()
    if recomputed != data["checksum_sha256"]:
        raise FixedSupportContractError(
            f"checksum mismatch: recomputed {recomputed} != stored {data['checksum_sha256']} "
            "-- contract payload was altered after checksumming"
        )

    return data


def write_fixed_support_contract(data: dict, path) -> Path:
    """Validates, then writes ``data`` to ``path`` via atomic tmp-write +
    replace. Strict no-overwrite: refuses if ``path`` already exists (no
    force option -- a fixed-support artifact is never silently replaced)."""
    path = Path(path)
    validate_fixed_support_contract(data)
    if path.exists():
        raise FixedSupportContractError(f"refusing to overwrite existing fixed-support contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def load_fixed_support_contract(path, *, expected_contract_id: str = OBJECTIVE_ID_V2) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FixedSupportContractError(f"fixed-support contract not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_fixed_support_contract(data, expected_contract_id=expected_contract_id)


def evaluate_fixed_support_raw_space_metrics(
    *,
    run_dir,
    epoch: int,
    package_root,
    contract: dict,
    basin_ids: Optional[Sequence[str]] = None,
    require_full_screening_population: bool = False,
    min_area_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> dict:
    """Evaluates raw-space metrics restricted to ``contract``'s frozen
    120h-floor common support. The returned dict is tagged
    ``objective_scope="fixed_support"`` and is structurally distinct from
    :func:`evaluate_natural_support_raw_space_metrics`'s output -- the ONLY
    function permitted to feed the v2 primary objective is
    :func:`extract_v2_objective_from_fixed_support_result`, which refuses
    any result whose ``objective_scope`` is not ``"fixed_support"``.

    Raises :class:`FixedSupportContractError` (never silently
    excludes/realigns) if a contract basin is entirely missing from this
    run's results, or if this run's own ``date`` coordinate for a basin
    does not contain every one of the contract's admitted timestamps for
    that basin -- both are basin/date identity contradictions, not normal
    per-basin exclusions.
    """
    validate_fixed_support_contract(contract)
    target_variable = contract["target_variable"]
    lead_hours = contract["lead_hours"]
    requested = sorted(basin_ids) if basin_ids is not None else contract["basin_ids"]
    unknown = set(requested) - set(contract["basin_ids"])
    if unknown:
        raise FixedSupportContractError(f"basin_ids not present in the fixed-support contract: {sorted(unknown)}")
    if require_full_screening_population:
        if len(contract["basin_ids"]) != 400 or len(set(contract["basin_ids"])) != 400:
            raise FixedSupportContractError("production fixed-support contract must contain exactly 400 unique screening basins")
        if requested != contract["basin_ids"]:
            raise FixedSupportContractError("production fixed-support evaluation must use the complete frozen screening population")

    period_results = load_period_results(run_dir, contract["period"], epoch)

    per_basin = []
    excluded = []
    for basin_id in requested:
        if basin_id not in period_results:
            raise FixedSupportContractError(
                f"basin {basin_id!r} is part of the fixed-support contract but missing from this run's "
                "period_results -- basin-identity contradiction, not a normal exclusion"
            )
        freq_results = period_results[basin_id]
        obs_key, sim_key = f"{target_variable}_obs", f"{target_variable}_sim"
        xr_ds = None
        for freq_result in freq_results.values():
            candidate = freq_result.get("xr")
            if candidate is not None and obs_key in candidate.data_vars and sim_key in candidate.data_vars:
                xr_ds = candidate
                break
        if xr_ds is None:
            raise FixedSupportContractError(f"basin {basin_id!r}: no freq result with both {obs_key!r} and {sim_key!r}")

        run_date_values = np.asarray(xr_ds.coords["date"].values)
        support_dates = _deserialize_date_array(contract["per_basin_support"][basin_id], contract["date_dtype"])
        if len(np.unique(run_date_values)) != len(run_date_values):
            raise FixedSupportContractError(f"basin {basin_id!r}: run date coordinate contains duplicates")
        if len(np.unique(support_dates)) != len(support_dates):
            raise FixedSupportContractError(f"basin {basin_id!r}: frozen support contains duplicate timestamps")
        support_mask = np.isin(run_date_values, support_dates)
        n_matched = int(support_mask.sum())
        if n_matched != len(support_dates):
            raise FixedSupportContractError(
                f"basin {basin_id!r}: {len(support_dates)} contract support timestamps but only "
                f"{n_matched} found in this run's own date coordinate -- date/period contradiction "
                "(refusing to silently realign)"
            )

        obs_mm_per_h = xr_ds[obs_key].values.reshape(-1)
        sim_mm_per_h = xr_ds[sim_key].values.reshape(-1)
        obs_support = np.where(support_mask, obs_mm_per_h, np.nan)
        if not np.isfinite(obs_support[support_mask]).all():
            raise FixedSupportContractError(
                f"basin {basin_id!r}: frozen admitted timestamps are not naturally admitted observations"
            )

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
            excluded.append({"basin_id": basin_id, "reason": f"area derivation failed: {exc}"})
            continue
        if not area_result.consistent:
            excluded.append(
                {
                    "basin_id": basin_id,
                    "reason": f"area derivation inconsistent: relative_mad={area_result.relative_mad:.6g}",
                }
            )
            continue

        basin_metrics = evaluate_basin_raw_space(
            basin_id=basin_id,
            obs_mm_per_h=obs_support,
            sim_mm_per_h=sim_mm_per_h,
            area_km2=area_result.area_km2,
        )
        basin_metrics["freq"] = "1h"
        basin_metrics["n_fixed_support_eligible"] = len(support_dates)
        per_basin.append(basin_metrics)

    return {
        "objective_scope": "fixed_support",
        "contract_id": contract["contract_id"],
        "contract_checksum_sha256": contract["checksum_sha256"],
        "seq_length_floor": contract["seq_length_floor"],
        "n_basins_requested": len(requested),
        "n_basins_evaluated": len(per_basin),
        "n_basins_excluded": len(excluded),
        "basins_excluded": excluded,
        "per_basin": per_basin,
        "aggregate": aggregate_raw_space_metrics(per_basin)
        if per_basin
        else {"n_basins": 0, "n_admitted_total": 0, "n_sim_nonfinite_at_admitted_total": 0, "metrics": {}},
    }


def evaluate_natural_support_raw_space_metrics(
    *,
    run_dir,
    period: str,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    basin_ids: Optional[Sequence[str]] = None,
    min_area_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> dict:
    """Thin, distinctly-named (``objective_scope="natural_support"``)
    wrapper around :func:`nh_seed_evaluation.raw_space_metrics_for_run_period`
    -- secondary diagnostic only. Structurally rejected by
    :func:`extract_v2_objective_from_fixed_support_result`."""
    result = raw_space_metrics_for_run_period(
        run_dir=run_dir,
        period=period,
        epoch=epoch,
        package_root=package_root,
        target_variable=target_variable,
        lead_hours=lead_hours,
        basin_ids=basin_ids,
        min_area_samples=min_area_samples,
        max_relative_mad=max_relative_mad,
    )
    return {"objective_scope": "natural_support", **result}


def extract_v2_objective_from_fixed_support_result(result: dict) -> float:
    """The ONLY function permitted to produce the v2 primary objective
    number. Structurally refuses any result not shaped like an
    :func:`evaluate_fixed_support_raw_space_metrics` output --
    ``objective_scope`` must be ``"fixed_support"`` -- which is the
    mechanism that makes it structurally impossible to publish the
    natural-support diagnostic as the v2 optimizer objective."""
    if not isinstance(result, dict) or result.get("objective_scope") != "fixed_support":
        got = result.get("objective_scope") if isinstance(result, dict) else type(result).__name__
        raise FixedSupportContractError(
            f"v2 objective may only be extracted from a fixed_support-scoped result, got objective_scope={got!r}"
        )
    metrics = result.get("aggregate", {}).get("metrics", {})
    nse_stats = metrics.get("nse")
    if not nse_stats or "median" not in nse_stats:
        raise FixedSupportContractError(
            "fixed-support result has no aggregate.metrics.nse.median to publish as the v2 objective"
        )
    median_nse = nse_stats["median"]
    if median_nse is None or not np.isfinite(median_nse):
        raise FixedSupportContractError(f"fixed-support aggregate median NSE is not a finite value: {median_nse!r}")
    return float(median_nse)
