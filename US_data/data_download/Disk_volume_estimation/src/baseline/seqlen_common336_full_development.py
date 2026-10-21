"""Full-development (2,307-basin) Common-336 comparison of the nine
screening-selected sequence-length checkpoints: support contract, strict
completeness around the fixed-support scorer, and isolated evaluation staging.

Scientific scope (user-approved milestone "Prepare the nine-checkpoint
Common-336 full-development comparison", 2026-10-18): a PRELIMINARY transfer
check of whether the 400-basin screening pattern persists across the 2,307
development basins in 2024. No selection, promotion, HPO or new metric; the
nine checkpoints (L120/L240/L336 x seeds A/B/C) are frozen in
:data:`SELECTED_CHECKPOINTS`. This module only prepares; it runs nothing.

Three additive pieces:

1. Support contract (new schema :data:`CONTRACT_SCHEMA_NAME`, new id
   :data:`CONTRACT_ID`). The admission rule is exactly the screening rule of
   :mod:`seqlen_common336_support` (:func:`basin_support_arrays`, floor 336):
   a valid continuous 336 h history with no documented gap and no non-finite
   dynamic input, issue time inside the validation period, target time
   ``t + lead`` inside the period, finite target. The population is recorded
   under ``development_basin_ids_sha256`` -- never under the screening
   contract's ``screening_basin_ids_sha256``. Support is run-length encoded
   (as in the training-support artifact) because a per-hour list for 2,307
   basins would be ~0.6 GB of JSON. Per basin it records the admitted count,
   the natural L120/L240/L336 counts and the natural-but-not-common losses by
   cause. A zero-support basin is a build failure, never a drop.
2. Completeness (:func:`require_complete_scored_population`,
   :func:`evaluate_full_development_run`, :func:`evaluate_full_development_canary`,
   :func:`require_identical_support_across_runs`) around the unchanged
   :func:`~.fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`.
   Any exclusion (including an area-derivation failure, which that scorer
   soft-excludes), any non-finite metric or prediction at an admitted hour, or
   any disagreement with the contract's exact per-basin admitted hours fails
   the evaluation. The full path pins the 2,307-basin development membership;
   a canary must declare its subset. :func:`require_screening_support_reproduced`
   requires the support of the 400 screening basins to equal the qualified
   screening contract hour for hour.
3. Staging (:func:`stage_full_development_eval_run`): a new run directory per
   checkpoint holding sha-verified copies of the run's own ``config.yml``,
   selected checkpoint and own scaler, with only the validation basin file,
   ``validate_n_random_basins`` and run-directory plumbing changed (recorded
   machine-checkably). The source run directory is never written.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .devpop_common120_audit_contract import (
    DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    DEVELOPMENT_TRAIN_ROLE,
    EXPECTED_DEVELOPMENT_POPULATION_SIZE,
    canonical_membership_sha256,
)
from .fixed_support_contract_v2 import evaluate_fixed_support_raw_space_metrics, is_strict_int
from .seqlen_common336_support import (
    CAMPAIGN_ID,
    COMMON_SEQ_LENGTH_FLOOR,
    SUPPORT_POLICY_IDENTITY,
    Common336SupportError,
    _checksum,
    _period_bounds,
    _read_basin,
    _sha256_file,
    basin_support_arrays,
    decode_issue_intervals,
    encode_issue_intervals,
)
from .validity_mask import bad_hour_mask_from_timestamps

__all__ = [
    "CONTRACT_SCHEMA_NAME",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_ID",
    "FIXTURE_ONLY_ROLE",
    "ACCOUNTED_SEQ_LENGTHS",
    "PRODUCTION_DATE_START",
    "PRODUCTION_DATE_END",
    "PRODUCTION_OBSERVATION_CUTOFF",
    "PRODUCTION_TARGET_VARIABLE",
    "PRODUCTION_LEAD_HOURS",
    "PRODUCTION_DYNAMIC_INPUTS",
    "PRODUCTION_GAP_ARTIFACT_SHA256",
    "PRODUCTION_SCREENING_CONTRACT_FILE_SHA256",
    "PRODUCTION_SCREENING_CONTRACT_CHECKSUM_SHA256",
    "QUALIFICATION_RECORD",
    "STAGING_MANIFEST_FILENAME",
    "STAGED_VALIDATION_BASIN_FILENAME",
    "ALLOWED_STAGED_CONFIG_CHANGES",
    "FullDevelopmentError",
    "FullDevelopmentSupportBuildError",
    "SelectedCheckpoint",
    "SELECTED_CHECKPOINTS",
    "build_full_development_support",
    "compute_full_development_support",
    "assemble_full_development_contract",
    "validate_full_development_support_contract",
    "require_production_contract_identity",
    "full_development_support_dates",
    "admitted_hours_sha256",
    "require_screening_support_reproduced",
    "write_full_development_support_contract",
    "load_full_development_support_contract",
    "require_complete_scored_population",
    "require_full_development_scored_population",
    "evaluate_full_development_run",
    "evaluate_full_development_canary",
    "require_identical_support_across_runs",
    "stage_full_development_eval_run",
]

CONTRACT_SCHEMA_NAME = "flashnh_seqlen_common336_full_development_support_contract"
CONTRACT_SCHEMA_VERSION = 1
CONTRACT_ID = "common336_full_development_raw_space_v001"
#: Population role of a synthetic test contract. It can never satisfy
#: :func:`require_production_contract_identity`.
FIXTURE_ONLY_ROLE = "fixture_only_unqualified_population"
_ALLOWED_ROLES = (DEVELOPMENT_TRAIN_ROLE, FIXTURE_ONLY_ROLE)
#: The three compared lengths; the floor is their maximum.
ACCOUNTED_SEQ_LENGTHS = (120, 240, 336)

PRODUCTION_PERIOD = "validation"
PRODUCTION_DATE_START = "2024-01-01"
PRODUCTION_DATE_END = "2024-12-31"
PRODUCTION_OBSERVATION_CUTOFF = "2024-12-31T23:00:00"
PRODUCTION_TARGET_VARIABLE = "qobs_mm_per_h_lead06"
PRODUCTION_LEAD_HOURS = 6
PRODUCTION_DYNAMIC_INPUTS = ("mrms_qpe_1h_mm", "rtma_2t_K")
#: Package v002 ``masks/gap_timestamps.json`` (138 timestamps), as recorded by
#: the 2026-10-14 anchor/seq-length preflight. The production builder ALSO
#: checks it against the package manifest's own recorded value.
PRODUCTION_GAP_ARTIFACT_SHA256 = "93b58aa2cb1d0467447a45c08e95cb8926a536229f2aa755d42095ef9411a2d5"
#: The qualified campaign's frozen 400-basin screening contract
#: (``.scratch_local/seqlen_common336_seed3_v001/support/common336_screening_contract_v001.json``
#: on Moriah), as recorded in the retained bundle's ``campaign_manifest.json``.
PRODUCTION_SCREENING_CONTRACT_FILE_SHA256 = "7b885f30e5baf0ad4c45e00bb27890c6d6f2140c1e48e72191ff0fec325dc7d7"
PRODUCTION_SCREENING_CONTRACT_CHECKSUM_SHA256 = "951f09f03e55fcbe1a2819d1399edcb34879e7e5cb032a1ed8460b287bbfc337"

_SHA_KEYS = (
    "development_basin_ids_sha256", "screening_subset_artifact_sha256", "gap_artifact_sha256",
    "package_manifest_sha256", "package_file_checksums_sha256", "package_run_provenance_sha256",
    "development_split_sha256", "spatial_holdout_split_sha256",
)
_IDENTITY_KEYS = (
    "gap_artifact_sha256", "package_manifest_sha256", "package_file_checksums_sha256",
    "package_run_provenance_sha256", "development_split_sha256", "spatial_holdout_split_sha256",
)
_CONTRACT_KEYS = {
    "schema_name", "schema_version", "contract_id", "campaign_id", "evaluation_population_role",
    "seq_length_floor", "seq_lengths_accounted", "lead_hours", "target_variable", "dynamic_inputs",
    "period", "date_start", "date_end", "observation_cutoff", "support_policy_identity", "date_dtype",
    *_SHA_KEYS, "basin_ids", "per_basin_intervals", "eligible_counts", "per_basin_accounting",
    "checksum_sha256",
}
_LOSS_CAUSES = ("lost_timeline_start", "lost_documented_gap", "lost_nonfinite_input")
_ACCOUNTING_KEYS = frozenset(
    {"n_admitted", "n_nonfinite_input_hours_not_gap"}
    | {f"n_natural_L{L}" for L in ACCOUNTED_SEQ_LENGTHS}
    | {f"{cause}_L{L}" for cause in _LOSS_CAUSES for L in ACCOUNTED_SEQ_LENGTHS}
)
#: Per-basin raw-space metrics produced by ``evaluate_basin_raw_space``; all
#: must be finite for a basin to count as scored.
_REQUIRED_FINITE_METRICS = ("nse", "kge", "rmse", "mae", "pearson_r", "bias", "pbias")
_HOUR = pd.Timedelta(hours=1)


class FullDevelopmentError(Common336SupportError):
    """Malformed/inconsistent full-development contract, identity or
    membership mismatch, incomplete or non-finite scored population, or an
    unsafe staging request. Never raised for an ordinary poor-skill score."""


class FullDevelopmentSupportBuildError(FullDevelopmentError):
    """Support build failed for at least one basin (unreadable NetCDF,
    timeline mismatch, zero support). ``accounting`` carries the full
    per-basin evidence so a driver can still write it."""

    def __init__(self, message: str, accounting: Mapping):
        super().__init__(message)
        self.accounting = accounting


# --------------------------------------------------------------------------- #
# 1. Support contract
# --------------------------------------------------------------------------- #

def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and value == value.lower()
            and all(c in "0123456789abcdef" for c in value))


def _cutoff_for(date_end: str) -> str:
    return f"{date_end}T23:00:00"


def _production_identities() -> dict:
    from . import sweep_v1_campaign as sweep
    from . import sweep_v1_production_adapter as adapter

    return {
        "gap_artifact_sha256": PRODUCTION_GAP_ARTIFACT_SHA256,
        "package_manifest_sha256": adapter.PACKAGE_MANIFEST_SHA256,
        "package_file_checksums_sha256": adapter.PACKAGE_FILE_CHECKSUMS_SHA256,
        "package_run_provenance_sha256": adapter.PACKAGE_RUN_PROVENANCE_SHA256,
        "development_split_sha256": adapter.DEVELOPMENT_SPLIT_SHA256,
        "spatial_holdout_split_sha256": adapter.SPATIAL_HOLDOUT_SPLIT_SHA256,
        "screening_subset_artifact_sha256": sweep.SCREENING_ARTIFACT_SHA256,
    }


def build_full_development_support(*, package_root, splits_dir, screening_basin_ids_path, baseline_policy_path,
                                   progress=None) -> dict:
    """Production entry point: verify every identity, then build (never write)
    the contract for the 2,307 development basins. Returns
    ``{"contract", "accounting"}``; raises :class:`FullDevelopmentSupportBuildError`
    (with accounting attached) if any basin fails."""
    from . import sweep_v1_campaign as sweep
    from .devpop_common120_audit_contract import ExpectedPopulationSpec
    from .gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT, load_gap_timestamps_json
    from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership
    from .pilot_lead06_config import load_screening_basin_ids
    from .policy import load_stage1_baseline_policy
    from .sweep_v1_production_adapter import PreparationPaths, _verify_artifact_identities

    identities = dict(_verify_artifact_identities(PreparationPaths(
        Path(baseline_policy_path), Path(package_root), Path(splits_dir), Path(screening_basin_ids_path))))
    manifest = read_package_manifest(package_root)
    if manifest.get("gap_product_scope") != [MRMS_PRODUCT, RTMA_PRODUCT]:
        raise FullDevelopmentError("package gap_product_scope must be complete MRMS+RTMA")
    gap_path = Path(package_root) / "masks" / "gap_timestamps.json"
    gap_sha = _sha256_file(gap_path)
    if gap_sha != (manifest.get("gap_timestamp_artifact") or {}).get("sha256"):
        raise FullDevelopmentError("package gap-timestamp artifact checksum contradicts package manifest")
    if gap_sha != PRODUCTION_GAP_ARTIFACT_SHA256:
        raise FullDevelopmentError(f"gap artifact sha256 {gap_sha} != pinned {PRODUCTION_GAP_ARTIFACT_SHA256}")
    identities["gap_artifact_sha256"] = gap_sha

    membership = validate_full_population_basin_membership(manifest, splits_dir)
    population = ExpectedPopulationSpec.for_development_train(splits_dir)
    if sorted(population.basin_ids) != sorted(membership.development_basins):
        raise FullDevelopmentError("development split file and package-manifest membership disagree")
    load_screening_basin_ids(screening_basin_ids_path, development_basins=membership.development_basins,
                             expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)
    policy = load_stage1_baseline_policy(baseline_policy_path)
    validation = policy["temporal_split"]["validation"]
    if (str(validation["start"]), str(validation["end"])) != (PRODUCTION_DATE_START, PRODUCTION_DATE_END):
        raise FullDevelopmentError(f"policy validation period {validation} is not 2024")
    return compute_full_development_support(
        package_root=package_root, basin_ids=population.basin_ids, population_role=DEVELOPMENT_TRAIN_ROLE,
        date_start=PRODUCTION_DATE_START, date_end=PRODUCTION_DATE_END,
        gap_timestamps=load_gap_timestamps_json(gap_path), dynamic_inputs=PRODUCTION_DYNAMIC_INPUTS,
        target_variable=PRODUCTION_TARGET_VARIABLE, lead_hours=PRODUCTION_LEAD_HOURS,
        identities=identities, screening_subset_artifact_sha256=sweep.SCREENING_ARTIFACT_SHA256,
        progress=progress)


def compute_full_development_support(*, package_root, basin_ids: Sequence[str], population_role: str,
                                     date_start: str, date_end: str, gap_timestamps,
                                     dynamic_inputs: Sequence[str], target_variable: str, lead_hours: int,
                                     identities: Mapping[str, str], screening_subset_artifact_sha256: str,
                                     progress=None) -> dict:
    """Identity-agnostic core (synthetic tests call it with
    :data:`FIXTURE_ONLY_ROLE`). The development role is accepted only for the
    exact pinned 2,307-basin membership."""
    ids = [str(b) for b in basin_ids]
    if not ids or len(set(ids)) != len(ids):
        raise FullDevelopmentError("basin_ids must be non-empty and unique")
    membership_sha = canonical_membership_sha256(ids)
    if population_role not in _ALLOWED_ROLES:
        raise FullDevelopmentError(f"population_role must be one of {_ALLOWED_ROLES}, got {population_role!r}")
    if population_role == DEVELOPMENT_TRAIN_ROLE and (
            len(ids) != EXPECTED_DEVELOPMENT_POPULATION_SIZE or membership_sha != DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256):
        raise FullDevelopmentError(
            f"development role requires the pinned {EXPECTED_DEVELOPMENT_POPULATION_SIZE}-basin membership "
            f"{DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256}; got {len(ids)} basins / {membership_sha}")
    period_start, period_end = _period_bounds({"start": date_start, "end": date_end})
    gaps = list(gap_timestamps)
    variables = list(dynamic_inputs) + [target_variable]

    reference_dates, gap_mask = None, None
    intervals, counts, accounting, rows, failures = {}, {}, {}, [], []
    for i, basin in enumerate(sorted(ids)):
        try:
            dates, values = _read_basin(package_root, basin, variables, period_end, lead_hours)
            if reference_dates is not None and not dates.equals(reference_dates):
                raise Common336SupportError(f"{basin}: timeline differs from the package-native timeline")
        except Common336SupportError as exc:
            failures.append({"basin_id": basin, "reason": str(exc)})
            continue
        if reference_dates is None:
            reference_dates = dates
            gap_mask = bad_hour_mask_from_timestamps(dates, gaps, on_out_of_range="ignore")
        arrays = basin_support_arrays(dates=dates, gap_mask=gap_mask, dynamic_values=values[:-1],
                                      target_values=values[-1], lead_hours=lead_hours, period_start=period_start,
                                      period_end=period_end, seq_lengths=ACCOUNTED_SEQ_LENGTHS,
                                      floor=COMMON_SEQ_LENGTH_FLOOR)
        n_admitted = int(arrays.admitted.sum())
        record = {"n_admitted": n_admitted,
                  "n_nonfinite_input_hours_not_gap": arrays.n_nonfinite_input_hours_not_gap}
        for L in ACCOUNTED_SEQ_LENGTHS:
            record[f"n_natural_L{L}"] = int(arrays.natural[L].sum())
            record["lost_timeline_start_L%d" % L] = int(arrays.lost_timeline_start[L].sum())
            record["lost_documented_gap_L%d" % L] = int(arrays.lost_documented_gap[L].sum())
            record["lost_nonfinite_input_L%d" % L] = int(arrays.lost_nonfinite_input[L].sum())
        rows.append({"basin_id": basin, **record})
        if n_admitted == 0:
            failures.append({"basin_id": basin, "reason": "zero Common-336 support"})
            continue
        intervals[basin] = encode_issue_intervals(dates, arrays.admitted)
        counts[basin] = n_admitted
        accounting[basin] = record
        if progress is not None:
            progress(i + 1, len(ids), basin)

    report = _accounting_report(rows, failures, membership_sha=membership_sha, population_role=population_role,
                                date_start=date_start, date_end=date_end, n_gap_timestamps=len(gaps),
                                timeline=None if reference_dates is None else [str(reference_dates[0]),
                                                                               str(reference_dates[-1])])
    if failures:
        raise FullDevelopmentSupportBuildError(
            f"{len(failures)} basin(s) failed the full-development support build (first: {failures[:3]})", report)
    contract = assemble_full_development_contract(
        basin_ids=ids, population_role=population_role, date_start=date_start, date_end=date_end,
        dynamic_inputs=dynamic_inputs, target_variable=target_variable, lead_hours=lead_hours,
        identities=identities, screening_subset_artifact_sha256=screening_subset_artifact_sha256,
        per_basin_intervals=intervals, eligible_counts=counts, per_basin_accounting=accounting)
    return {"contract": contract, "accounting": report}


def _accounting_report(rows, failures, **context) -> dict:
    totals = {key: int(sum(r[key] for r in rows)) for key in sorted(_ACCOUNTING_KEYS)}
    retention = [r["n_admitted"] / r["n_natural_L120"] for r in rows if r["n_natural_L120"] > 0]
    return {
        **context, "n_basins_read": len(rows), "n_failures": len(failures), "failures": failures,
        "totals": totals,
        "retained_fraction_of_natural_L120": None if not retention else {
            "pooled": totals["n_admitted"] / totals["n_natural_L120"] if totals["n_natural_L120"] else None,
            "per_basin_min": float(np.min(retention)), "per_basin_median": float(np.median(retention)),
            "per_basin_max": float(np.max(retention))},
        "rows": rows,
    }


def assemble_full_development_contract(*, basin_ids: Sequence[str], population_role: str, date_start: str,
                                       date_end: str, dynamic_inputs: Sequence[str], target_variable: str,
                                       lead_hours: int, identities: Mapping[str, str],
                                       screening_subset_artifact_sha256: str,
                                       per_basin_intervals: Mapping[str, list], eligible_counts: Mapping[str, int],
                                       per_basin_accounting: Mapping[str, Mapping]) -> dict:
    """Checksum and validate a contract payload (no I/O)."""
    ids = sorted(str(b) for b in basin_ids)
    payload = {
        "schema_name": CONTRACT_SCHEMA_NAME, "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID, "campaign_id": CAMPAIGN_ID, "evaluation_population_role": population_role,
        "development_basin_ids_sha256": canonical_membership_sha256(ids),
        "screening_subset_artifact_sha256": screening_subset_artifact_sha256,
        "seq_length_floor": COMMON_SEQ_LENGTH_FLOOR, "seq_lengths_accounted": list(ACCOUNTED_SEQ_LENGTHS),
        "lead_hours": int(lead_hours), "target_variable": str(target_variable),
        "dynamic_inputs": [str(d) for d in dynamic_inputs], "period": PRODUCTION_PERIOD,
        "date_start": str(date_start), "date_end": str(date_end), "observation_cutoff": _cutoff_for(str(date_end)),
        "support_policy_identity": SUPPORT_POLICY_IDENTITY, "date_dtype": "datetime64",
        **{key: identities[key] for key in _IDENTITY_KEYS},
        "basin_ids": ids,
        "per_basin_intervals": {b: [list(p) for p in per_basin_intervals[b]] for b in ids},
        "eligible_counts": {b: eligible_counts[b] for b in ids},
        "per_basin_accounting": {b: dict(per_basin_accounting[b]) for b in ids},
    }
    payload["checksum_sha256"] = _checksum(payload)
    return validate_full_development_support_contract(payload)


def validate_full_development_support_contract(data: Mapping) -> dict:
    """Strict schema, membership, support-bound, accounting and checksum
    validation. Every support hour must be an issue time in
    ``[date_start 00:00, date_end 23:00 - lead]`` (so every target lies in the
    period), and every basin must have positive support."""
    if not isinstance(data, dict):
        raise FullDevelopmentError(f"contract must be a mapping, got {type(data).__name__}")
    missing, extra = _CONTRACT_KEYS - set(data), set(data) - _CONTRACT_KEYS
    if missing or extra:
        raise FullDevelopmentError(f"contract keys: missing {sorted(missing)}, extra {sorted(extra)}")
    fixed = {"schema_name": CONTRACT_SCHEMA_NAME, "schema_version": CONTRACT_SCHEMA_VERSION,
             "contract_id": CONTRACT_ID, "campaign_id": CAMPAIGN_ID, "seq_length_floor": COMMON_SEQ_LENGTH_FLOOR,
             "seq_lengths_accounted": list(ACCOUNTED_SEQ_LENGTHS), "period": PRODUCTION_PERIOD,
             "support_policy_identity": SUPPORT_POLICY_IDENTITY, "date_dtype": "datetime64"}
    for key, value in fixed.items():
        if data[key] != value or type(data[key]) is not type(value):
            raise FullDevelopmentError(f"contract {key} must be {value!r}, got {data[key]!r}")
    if data["evaluation_population_role"] not in _ALLOWED_ROLES:
        raise FullDevelopmentError(f"evaluation_population_role {data['evaluation_population_role']!r} not allowed")
    for key in _SHA_KEYS:
        if not _is_sha256(data[key]):
            raise FullDevelopmentError(f"{key} must be a lowercase 64-character SHA-256")
    if not is_strict_int(data["lead_hours"]) or data["lead_hours"] < 0:
        raise FullDevelopmentError(f"lead_hours must be a non-negative int, got {data['lead_hours']!r}")
    if not isinstance(data["target_variable"], str) or not data["target_variable"]:
        raise FullDevelopmentError("target_variable must be a non-empty string")
    dyn = data["dynamic_inputs"]
    if not isinstance(dyn, list) or not dyn or not all(isinstance(d, str) and d for d in dyn):
        raise FullDevelopmentError("dynamic_inputs must be a non-empty list of names")
    try:
        start, end = pd.Timestamp(data["date_start"]), pd.Timestamp(data["date_end"])
    except (TypeError, ValueError) as exc:
        raise FullDevelopmentError(f"unparseable period dates: {exc}") from exc
    if str(start.date()) != data["date_start"] or str(end.date()) != data["date_end"] or end < start:
        raise FullDevelopmentError("date_start/date_end must be YYYY-MM-DD with date_end >= date_start")
    if data["observation_cutoff"] != _cutoff_for(data["date_end"]):
        raise FullDevelopmentError(
            f"observation_cutoff must be the last period hour {_cutoff_for(data['date_end'])}, "
            f"got {data['observation_cutoff']!r}")

    basin_ids = data["basin_ids"]
    if not isinstance(basin_ids, list) or not basin_ids or basin_ids != sorted(set(basin_ids)):
        raise FullDevelopmentError("basin_ids must be a non-empty sorted list of unique ids")
    if canonical_membership_sha256(basin_ids) != data["development_basin_ids_sha256"]:
        raise FullDevelopmentError("development_basin_ids_sha256 does not match basin_ids")
    for key in ("per_basin_intervals", "eligible_counts", "per_basin_accounting"):
        if not isinstance(data[key], dict) or set(data[key]) != set(basin_ids):
            raise FullDevelopmentError(f"{key} keys must exactly match basin_ids")

    first_issue = start
    last_issue = end + pd.Timedelta(hours=23) - pd.Timedelta(hours=data["lead_hours"])
    for basin in basin_ids:
        count = data["eligible_counts"][basin]
        if not is_strict_int(count) or count <= 0:
            raise FullDevelopmentError(f"basin {basin!r}: support count must be a positive int, got {count!r}")
        total, previous_end = 0, None
        for pair in data["per_basin_intervals"][basin]:
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) for v in pair):
                raise FullDevelopmentError(f"basin {basin!r}: malformed interval {pair!r}")
            a, b = pd.Timestamp(pair[0]), pd.Timestamp(pair[1])
            if a.tzinfo is not None or b.tzinfo is not None or b < a or a != a.floor("h") or b != b.floor("h"):
                raise FullDevelopmentError(f"basin {basin!r}: interval {pair!r} must be tz-naive hourly with end >= start")
            if previous_end is not None and a <= previous_end + _HOUR:
                raise FullDevelopmentError(f"basin {basin!r}: intervals must be sorted, disjoint, non-adjacent")
            if a < first_issue or b > last_issue:
                raise FullDevelopmentError(
                    f"basin {basin!r}: interval {pair!r} outside admissible issue times "
                    f"[{first_issue}, {last_issue}] (targets must lie inside the period)")
            total += int((b - a) / _HOUR) + 1
            previous_end = b
        if total != count:
            raise FullDevelopmentError(f"basin {basin!r}: intervals cover {total} hours, eligible_counts says {count}")
        record = data["per_basin_accounting"][basin]
        if not isinstance(record, dict) or set(record) != _ACCOUNTING_KEYS:
            raise FullDevelopmentError(f"basin {basin!r}: accounting keys must be {sorted(_ACCOUNTING_KEYS)}")
        if not all(is_strict_int(v) and v >= 0 for v in record.values()):
            raise FullDevelopmentError(f"basin {basin!r}: accounting values must be non-negative ints")
        if record["n_admitted"] != count or record[f"n_natural_L{COMMON_SEQ_LENGTH_FLOOR}"] != count:
            raise FullDevelopmentError(f"basin {basin!r}: accounting n_admitted/n_natural_L336 != support count")
        for L in ACCOUNTED_SEQ_LENGTHS:
            lost = sum(record[f"{cause}_L{L}"] for cause in _LOSS_CAUSES)
            if record[f"n_natural_L{L}"] - count != lost:
                raise FullDevelopmentError(f"basin {basin!r}: L{L} natural - admitted != sum of losses by cause")
    if _checksum(data) != data["checksum_sha256"]:
        raise FullDevelopmentError("contract checksum mismatch -- payload altered after checksumming")
    return data


def require_production_contract_identity(contract: Mapping) -> None:
    """Pin a (validated) contract to the real 2,307-basin development
    population, package/split/gap identities, target, lead, inputs, 2024
    period and cutoff. A fixture contract can never pass."""
    expected = {
        "evaluation_population_role": DEVELOPMENT_TRAIN_ROLE,
        "development_basin_ids_sha256": DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
        "target_variable": PRODUCTION_TARGET_VARIABLE, "lead_hours": PRODUCTION_LEAD_HOURS,
        "dynamic_inputs": list(PRODUCTION_DYNAMIC_INPUTS), "date_start": PRODUCTION_DATE_START,
        "date_end": PRODUCTION_DATE_END, "observation_cutoff": PRODUCTION_OBSERVATION_CUTOFF,
        **_production_identities(),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise FullDevelopmentError(f"contract {key}={contract.get(key)!r} is not the production value {value!r}")
    if len(contract["basin_ids"]) != EXPECTED_DEVELOPMENT_POPULATION_SIZE:
        raise FullDevelopmentError(f"production contract must hold {EXPECTED_DEVELOPMENT_POPULATION_SIZE} basins")


def full_development_support_dates(contract: Mapping, basin_id: str) -> np.ndarray:
    """Admitted issue times of one basin as sorted ``datetime64[ns]`` (the
    fixed-support scorer's per-basin support accessor for this schema)."""
    return decode_issue_intervals(contract["per_basin_intervals"][basin_id])


def admitted_hours_sha256(dates) -> str:
    """Digest of an exact admitted-hour set: SHA-256 of the timestamps as
    little-endian int64 nanoseconds, in the given (contract, i.e. sorted) order."""
    values = np.asarray(dates, dtype="datetime64[ns]").astype("<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def require_screening_support_reproduced(contract: Mapping, screening_contract: Mapping) -> dict:
    """The full-development support restricted to the screening basins must
    equal, hour for hour, the qualified Common-336 screening contract, which
    is the authority. Target, lead, period, floor, support policy, screening
    subset and package/split identities must agree too. A production
    (development-role) contract is compared only against the pinned qualified
    screening contract. Any mismatch fails; nothing is reconciled or dropped."""
    from .fixed_support_contract_v2 import _deserialize_date_array, validate_fixed_support_contract
    from .seqlen_common336_support import SCREENING_CONTRACT_ID

    validate_full_development_support_contract(contract)
    validate_fixed_support_contract(screening_contract, expected_contract_id=SCREENING_CONTRACT_ID,
                                    expected_seq_length_floor=COMMON_SEQ_LENGTH_FLOOR)
    if contract["evaluation_population_role"] == DEVELOPMENT_TRAIN_ROLE and \
            screening_contract["checksum_sha256"] != PRODUCTION_SCREENING_CONTRACT_CHECKSUM_SHA256:
        raise FullDevelopmentError(
            f"screening contract checksum {screening_contract['checksum_sha256']} is not the qualified "
            f"{PRODUCTION_SCREENING_CONTRACT_CHECKSUM_SHA256}")
    pairs = [("lead_hours", "lead_hours"), ("target_variable", "target_variable"), ("period", "period"),
             ("date_start", "date_start"), ("date_end", "date_end"), ("seq_length_floor", "seq_length_floor"),
             ("support_policy_identity", "source_gap_policy_identity"),
             ("screening_subset_artifact_sha256", "screening_basin_ids_sha256"),
             *((key, key) for key in _IDENTITY_KEYS if key != "gap_artifact_sha256")]
    for full_key, screening_key in pairs:
        if contract[full_key] != screening_contract[screening_key]:
            raise FullDevelopmentError(f"{full_key}={contract[full_key]!r} != screening "
                                       f"{screening_key}={screening_contract[screening_key]!r}")
    if screening_contract["date_dtype"] != "datetime64":
        raise FullDevelopmentError("screening contract must carry datetime64 support")
    screening_ids = list(screening_contract["basin_ids"])
    outside = sorted(set(screening_ids) - set(contract["basin_ids"]))
    if outside:
        raise FullDevelopmentError(f"{len(outside)} screening basin(s) outside the full contract: {outside[:10]}")
    mismatches = []
    for basin in screening_ids:
        expected = _deserialize_date_array(screening_contract["per_basin_support"][basin], "datetime64")
        got = full_development_support_dates(contract, basin)
        if not np.array_equal(expected.astype("datetime64[ns]"), got):
            mismatches.append({"basin_id": basin, "n_screening": int(expected.size), "n_full": int(got.size),
                               "n_only_screening": int(np.setdiff1d(expected, got).size),
                               "n_only_full": int(np.setdiff1d(got, expected).size)})
    if mismatches:
        raise FullDevelopmentError(
            f"{len(mismatches)} of {len(screening_ids)} screening basins do not reproduce the qualified "
            f"Common-336 support: {mismatches[:5]}")
    return {"n_screening_basins": len(screening_ids),
            "n_admitted_total": int(sum(screening_contract["eligible_counts"][b] for b in screening_ids)),
            "screening_contract_checksum_sha256": screening_contract["checksum_sha256"],
            "full_contract_checksum_sha256": contract["checksum_sha256"]}


def write_full_development_support_contract(contract: dict, path) -> Path:
    path = Path(path)
    validate_full_development_support_contract(contract)
    if path.exists():
        raise FullDevelopmentError(f"refusing to overwrite existing full-development contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(contract, sort_keys=True, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_full_development_support_contract(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FullDevelopmentError(f"full-development contract not found: {path}")
    return validate_full_development_support_contract(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# 2. Strict completeness around the fixed-support scorer
# --------------------------------------------------------------------------- #

def require_complete_scored_population(result: Mapping, *, contract: Mapping, expected_basin_ids: Sequence[str]) -> None:
    """Fail unless ``result`` scored exactly ``expected_basin_ids`` (no
    exclusion of any kind), against this contract, with finite metrics, no
    non-finite simulation at any admitted hour, and every basin scored on
    exactly its contract support count."""
    expected = [str(b) for b in expected_basin_ids]
    if not expected or len(set(expected)) != len(expected):
        raise FullDevelopmentError("expected_basin_ids must be non-empty and unique")
    expected = sorted(expected)
    unknown = set(expected) - set(contract["basin_ids"])
    if unknown:
        raise FullDevelopmentError(f"expected basins outside the contract: {sorted(unknown)[:10]}")
    if result.get("contract_id") != contract["contract_id"] or \
            result.get("contract_checksum_sha256") != contract["checksum_sha256"]:
        raise FullDevelopmentError("result was not scored against this contract (id/checksum mismatch)")
    if result.get("seq_length_floor") != COMMON_SEQ_LENGTH_FLOOR:
        raise FullDevelopmentError(f"result seq_length_floor {result.get('seq_length_floor')!r} != 336")
    if result.get("observation_cutoff") != str(pd.Timestamp(contract["observation_cutoff"])):
        raise FullDevelopmentError(
            f"result observation_cutoff {result.get('observation_cutoff')!r} != contract {contract['observation_cutoff']}")
    excluded = result.get("basins_excluded")
    if result.get("n_basins_excluded") != 0 or excluded != []:
        raise FullDevelopmentError(f"scored population has exclusions (fail closed): {(excluded or [])[:5]}")
    rows = result.get("per_basin") or []
    ids = [row.get("basin_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise FullDevelopmentError("scored population contains duplicate basin rows")
    if sorted(ids) != expected:
        missing, extra = sorted(set(expected) - set(ids)), sorted(set(ids) - set(expected))
        raise FullDevelopmentError(
            f"scored basins != expected: {len(ids)} scored, {len(expected)} expected; "
            f"missing {missing[:10]} ({len(missing)}), extra {extra[:10]} ({len(extra)})")
    if result.get("n_basins_requested") != len(expected) or result.get("n_basins_evaluated") != len(expected):
        raise FullDevelopmentError(
            f"requested/evaluated counts {result.get('n_basins_requested')}/{result.get('n_basins_evaluated')} "
            f"!= {len(expected)}")
    for row in rows:
        basin = row["basin_id"]
        for name in _REQUIRED_FINITE_METRICS:
            value = row.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise FullDevelopmentError(f"basin {basin!r}: metric {name}={value!r} is not finite")
        if row.get("n_sim_nonfinite_at_admitted") != 0:
            raise FullDevelopmentError(
                f"basin {basin!r}: {row.get('n_sim_nonfinite_at_admitted')!r} non-finite prediction(s) at admitted hours")
        n_contract = contract["eligible_counts"][basin]
        if row.get("n_admitted") != n_contract or row.get("n_fixed_support_eligible") != n_contract:
            raise FullDevelopmentError(
                f"basin {basin!r}: scored n_admitted={row.get('n_admitted')!r}, "
                f"n_fixed_support_eligible={row.get('n_fixed_support_eligible')!r}, contract {n_contract}")
        if row.get("admitted_hours_sha256") != admitted_hours_sha256(full_development_support_dates(contract, basin)):
            raise FullDevelopmentError(f"basin {basin!r}: scored admitted hours are not the contract support hours")
    aggregate = result.get("aggregate") or {}
    if aggregate.get("n_basins") != len(expected) or aggregate.get("n_sim_nonfinite_at_admitted_total") != 0 or \
            aggregate.get("n_admitted_total") != sum(contract["eligible_counts"][b] for b in expected):
        raise FullDevelopmentError("aggregate counts disagree with the complete scored population")


def require_full_development_scored_population(result: Mapping, *, contract: Mapping) -> None:
    """The full 2,307-basin gate: production contract identity plus complete
    scoring of every development basin (1,000 or 2,306 basins fail)."""
    validate_full_development_support_contract(contract)
    require_production_contract_identity(contract)
    require_complete_scored_population(result, contract=contract, expected_basin_ids=contract["basin_ids"])
    scored = sorted(row["basin_id"] for row in result["per_basin"])
    if len(scored) != EXPECTED_DEVELOPMENT_POPULATION_SIZE or \
            canonical_membership_sha256(scored) != DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256:
        raise FullDevelopmentError("scored population is not the pinned 2,307-basin development membership")


_PASSTHROUGH_SCORER_KWARGS = frozenset(
    {"min_area_samples", "max_relative_mad", "return_admitted_series", "package_identity",
     "authenticated_period_results"})


def _score(*, run_dir, epoch, package_root, contract, basin_ids, scorer_kwargs) -> dict:
    from .nh_raw_space_evaluation import read_run_observation_cutoff

    unexpected = set(scorer_kwargs) - _PASSTHROUGH_SCORER_KWARGS
    if unexpected:
        raise FullDevelopmentError(f"unsupported scorer option(s) for full-development scoring: {sorted(unexpected)}")
    cutoff = pd.Timestamp(contract["observation_cutoff"])
    declared = read_run_observation_cutoff(run_dir)
    if declared != cutoff:
        raise FullDevelopmentError(f"{run_dir}: run declares observation cutoff {declared}, contract requires {cutoff}")
    kwargs = dict(scorer_kwargs)
    keep_series = bool(kwargs.pop("return_admitted_series", False))
    result = evaluate_fixed_support_raw_space_metrics(
        run_dir=run_dir, epoch=epoch, package_root=package_root, contract=contract, basin_ids=basin_ids,
        require_full_screening_population=False, contract_schema=CONTRACT_SCHEMA_NAME,
        expected_contract_id=CONTRACT_ID, expected_seq_length_floor=COMMON_SEQ_LENGTH_FLOOR,
        observation_cutoff=cutoff, return_admitted_series=True, **kwargs)
    # Record the exact admitted hours each basin was scored on (the scorer's
    # timestamp-matched admitted series), then drop the arrays unless asked.
    series = result["admitted_series_by_basin"] if keep_series else result.pop("admitted_series_by_basin")
    for row in result["per_basin"]:
        admitted = series.get(row["basin_id"])
        row["admitted_hours_sha256"] = None if admitted is None else admitted_hours_sha256(admitted.date)
    return result


def evaluate_full_development_run(*, run_dir, epoch: int, package_root, contract: dict, **scorer_kwargs) -> dict:
    """Score one staged run on the full 2,307-basin Common-336 support with the
    qualified raw-space scorer, then apply :func:`require_full_development_scored_population`."""
    validate_full_development_support_contract(contract)
    require_production_contract_identity(contract)
    result = _score(run_dir=run_dir, epoch=epoch, package_root=package_root, contract=contract, basin_ids=None,
                    scorer_kwargs=scorer_kwargs)
    require_full_development_scored_population(result, contract=contract)
    result["completeness"] = {"scope": "full_development", "n_expected": len(contract["basin_ids"]),
                              "expected_membership_sha256": contract["development_basin_ids_sha256"]}
    return result


def evaluate_full_development_canary(*, run_dir, epoch: int, package_root, contract: dict,
                                     canary_basin_ids: Sequence[str],
                                     fixture_only_allow_unqualified_contract: bool = False, **scorer_kwargs) -> dict:
    """Score an explicitly declared basin subset; fail on any omission,
    exclusion or non-finite value. The contract must be the production one
    unless ``fixture_only_allow_unqualified_contract`` (synthetic tests only,
    and then the contract must carry :data:`FIXTURE_ONLY_ROLE`)."""
    validate_full_development_support_contract(contract)
    if fixture_only_allow_unqualified_contract:
        if contract["evaluation_population_role"] != FIXTURE_ONLY_ROLE:
            raise FullDevelopmentError("fixture-only canary requires a fixture-only contract")
    else:
        require_production_contract_identity(contract)
    declared = [str(b) for b in canary_basin_ids]
    if not declared or len(set(declared)) != len(declared):
        raise FullDevelopmentError("canary_basin_ids must be a non-empty list of unique basin ids")
    unknown = set(declared) - set(contract["basin_ids"])
    if unknown:
        raise FullDevelopmentError(f"canary basins outside the contract: {sorted(unknown)[:10]}")
    result = _score(run_dir=run_dir, epoch=epoch, package_root=package_root, contract=contract,
                    basin_ids=sorted(declared), scorer_kwargs=scorer_kwargs)
    require_complete_scored_population(result, contract=contract, expected_basin_ids=declared)
    result["completeness"] = {"scope": "canary", "n_expected": len(declared),
                              "expected_membership_sha256": canonical_membership_sha256(declared)}
    return result


def require_identical_support_across_runs(results: Mapping[str, Mapping], *, contract: Mapping,
                                          expected_run_ids: Optional[Sequence[str]] = None) -> dict:
    """All runs scored the same basins on the same contract on exactly the
    same per-basin admitted hours (each equal to the contract's, compared by
    :func:`admitted_hours_sha256`, not by count). With ``expected_run_ids``
    the run set must match exactly."""
    if not results:
        raise FullDevelopmentError("no run results supplied")
    if expected_run_ids is not None and sorted(results) != sorted(expected_run_ids):
        raise FullDevelopmentError(
            f"run set mismatch: missing {sorted(set(expected_run_ids) - set(results))}, "
            f"extra {sorted(set(results) - set(expected_run_ids))}")
    contract_digest = {}
    reference = None
    for run_id in sorted(results):
        result = results[run_id]
        if result.get("contract_checksum_sha256") != contract["checksum_sha256"] or \
                result.get("observation_cutoff") != str(pd.Timestamp(contract["observation_cutoff"])):
            raise FullDevelopmentError(f"{run_id}: not scored against this contract/cutoff")
        if result.get("n_basins_excluded") != 0:
            raise FullDevelopmentError(f"{run_id}: has excluded basins")
        hours = {row["basin_id"]: (row["n_admitted"], row.get("admitted_hours_sha256")) for row in result["per_basin"]}
        if len(hours) != len(result["per_basin"]):
            raise FullDevelopmentError(f"{run_id}: duplicate basin rows")
        for basin, (n, digest) in hours.items():
            if basin not in contract["eligible_counts"]:
                raise FullDevelopmentError(f"{run_id}: basin {basin!r} is not in the contract")
            if basin not in contract_digest:
                contract_digest[basin] = admitted_hours_sha256(full_development_support_dates(contract, basin))
            if n != contract["eligible_counts"][basin] or digest != contract_digest[basin]:
                raise FullDevelopmentError(f"{run_id}: basin {basin!r} admitted hours differ from the contract")
        if reference is None:
            reference = (run_id, hours)
        elif hours != reference[1]:
            diff = sorted(set(hours) ^ set(reference[1]))[:5]
            raise FullDevelopmentError(f"{run_id} and {reference[0]} scored different basins: {diff}")
    return {"n_runs": len(results), "contract_checksum_sha256": contract["checksum_sha256"],
            "n_basins": len(reference[1]), "n_admitted_total": int(sum(n for n, _ in reference[1].values()))}


# --------------------------------------------------------------------------- #
# 3. Isolated evaluation staging
# --------------------------------------------------------------------------- #

#: The final adjudicated qualification record of the 15-run campaign (all 15
#: rows ``qualified=True``, including L120 seed A). The ledger re-adjudicates
#: the raw ``rescore_reproduces_stored`` check: nse/bias/rmse/mae/pbias/
#: n_admitted/area_km2 must be exactly equal, kge/pearson_r within 1e-12.
#: Paths are relative to the project root of the local analysis copy.
QUALIFICATION_RECORD = {
    "ledger_relpath": ".scratch_local/seqlen_common336_qual_20261017/analysis/out/qualification_ledger.csv",
    "ledger_sha256": "c807bff42594ba7b97c7c498b9d18c0a96070740b3e381829e1bb4c0837bd6db",
    "adjudication_script_relpath": ".scratch_local/seqlen_common336_qual_20261017/analysis/c336_ledger.py",
    "adjudication_script_sha256": "557e6cab48089bfecbcd8c2e1c6cb5608e4936475c5f28f21b275679921f85f8",
    "bundle_relpath": ".scratch_local/seqlen_common336_qual_20261017/bundle/bundle_c336_qual_v001.tgz",
    "bundle_sha256": "7947f9a33c0a6ecd2343bc78fa4732d4f6f89dbeede8c7358be1ec02463216e0",
}


@dataclass(frozen=True)
class SelectedCheckpoint:
    """One frozen screening-selected checkpoint (provenance verified locally
    2026-10-19 against :data:`QUALIFICATION_RECORD`):

    * ``qualification_index``/``epoch``: the ledger row (``qualified=True``)
      and its ``selected_epoch``.
    * ``model_sha256``: bundle ``qual_out/idxNN_*_qualification.json``
      ``checkpoints/<epoch>/model_sha256``.
    * ``config_sha256``: the run's NH-dumped ``config.yml``, retained byte for
      byte as bundle ``runs/<run_id>/nh_config.yml``. Not printed in the
      ledger: the ledger's ``config_sha256`` is the campaign-generated input
      ``config.yaml`` (also retained in the bundle), a different file.
    * ``scaler_sha256``: ``train_data/train_data_scaler.yml``, observed
      read-only on Moriah 2026-10-18; neither the ledger nor the bundle holds
      the scaler. One value per length, identical across the three seeds.

    Staging re-checks all three against the original files before copying."""

    qualification_index: int
    run_id: str
    nh_run_dir_name: str
    seq_length: int
    seed: int
    epoch: int
    model_sha256: str
    scaler_sha256: str
    config_sha256: str


_SEEDS = {"A": 967139, "B": 329911494, "C": 3023892610}
_SCALER = {120: "bed4374154ca3deccaabc0205ee1976acfb2534c50a578c3b2fc53d410d299ca",
           240: "59d23434c2d3f42e3cee68b54bcc7611708b93244b04e821d64d03c7d8a4c939",
           336: "b10bdd9fca41853673555640332360bd1f2d3002bcd72b253159af52f0371500"}


def _selected(idx, L, seed_label, epoch, stamp, model_sha, config_sha) -> SelectedCheckpoint:
    run_id = f"{CAMPAIGN_ID}__L{L:03d}__seed{seed_label}{_SEEDS[seed_label]}__mf12x50000__attempt001"
    return SelectedCheckpoint(qualification_index=idx, run_id=run_id, nh_run_dir_name=f"{run_id}_{stamp}",
                              seq_length=L, seed=_SEEDS[seed_label], epoch=epoch, model_sha256=model_sha,
                              scaler_sha256=_SCALER[L], config_sha256=config_sha)


SELECTED_CHECKPOINTS: tuple = (
    _selected(1, 120, "A", 4, "1610_130108", "9961a34253420eab355f4e8d43d2721d3dbe8a71e7153f26f93dd32ab5eb4430",
              "c9937d47b21d907dbc12c93952313281af71e1f3db9d5e9dd0707910c15db090"),
    _selected(3, 240, "A", 9, "1610_143831", "14e9230f76e0de3cb7cf5867212f7c5dd8808786ab488c5589b79c8c4ef69e24",
              "37b2c06ddd81868147b389486de2d611667557f9bcd024a9718ee1d2a7729993"),
    _selected(4, 336, "A", 11, "1610_143831", "8844fa71f1c10acadf239b6e30e1ee5ff48a709734459094a776309f87913af0",
              "cf04dadb0139c0c4d063719ec1dc91700e39b7f2510046f660963d0126170ad7"),
    _selected(6, 120, "B", 7, "1610_143831", "fe6bda1ef06ba1b963f9c74892389e7ff37047c16ce1ccb3f8a80eb57ad7933c",
              "13248a7e8939f8de393a1b08c943b3eec54223f3da37b5a1ed9f6bc3e5ffd702"),
    _selected(8, 240, "B", 8, "1610_143831", "1ae6a27773ca34247182a235eb556d07ac4770cfda79c74dad7859f7b53ac0ef",
              "953b8f04ac9b8cd4b772ada8261c30142217a3c104c90c5b08bc3eec616be804"),
    _selected(9, 336, "B", 9, "1610_143831", "90f46db7b7fd9673148aad2d309f9d0917a60fb1cddb55ff699038a134a5ee90",
              "e8c4775bbf57253f33670c90f0a0d3c7e3413a62a1468feddf371728c969713a"),
    _selected(11, 120, "C", 5, "1610_143831", "d7fc31403475e47035af339dec918716f05b90960272f1617b8e57f0306010f9",
              "c6508c43fb15a1cc9636b9449782b87d4f2af663ece8e7fa2fbee7becdd391cc"),
    _selected(13, 240, "C", 5, "1610_143831", "99927af685c4fb165a63940ac84bd83548f1677532c75accca7aa574d54d89e4",
              "fcbce02ce401d6cec6de091556f11652eb39eebc4d074f2bc3b5090e31c1fbde"),
    _selected(14, 336, "C", 12, "1610_143534", "7e34b061b5b3fe8d84b0c9da1bca06d92e6694cbc5e614ebabd091e0f92ddc05",
              "d7442f6bf164bcd23fadcc4ef922bd263fc5dacf932724e3994a2e3c14f2a5fb"),
)

STAGING_MANIFEST_FILENAME = "C336_FULL_DEVELOPMENT_EVAL_STAGING_MANIFEST.json"
STAGED_VALIDATION_BASIN_FILENAME = "c336_full_development_validation_basins.txt"
#: The only config keys staging may change. Everything else, including the
#: observation cutoff, data_dir, inputs, statics, model and training-support
#: key, must be identical to the source run's own config.yml.
ALLOWED_STAGED_CONFIG_CHANGES = ("validation_basin_file", "validate_n_random_basins", "run_dir", "train_dir",
                                 "img_log_dir")
_NH_DATE_FORMAT = "%d/%m/%Y"


def _yaml():
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    return yaml


def _read_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = _yaml().load(fh)
    if not isinstance(data, dict):
        raise FullDevelopmentError(f"{path}: config is not a mapping")
    return data


def _tree_snapshot(root: Path) -> dict:
    """Cheap no-write proof for a run directory: (size, mtime_ns) per file."""
    snapshot = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            stat = path.stat()
            snapshot[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _read_basin_list(path: Path) -> list:
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise FullDevelopmentError(f"{path}: basin list must be non-empty with unique ids")
    return ids


def _copy_and_verify(src: Path, dst: Path, *, expected_sha256: str, label: str) -> str:
    shutil.copy2(src, dst)
    observed = _sha256_file(dst)
    if observed != expected_sha256:
        raise FullDevelopmentError(f"staged {label} sha256 {observed} != expected {expected_sha256}")
    return observed


def _nh_date(value, key: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(str(value), format=_NH_DATE_FORMAT)
    except (TypeError, ValueError) as exc:
        raise FullDevelopmentError(f"config {key}={value!r} is not a {_NH_DATE_FORMAT} date") from exc


def _require_source_config_consistent(cfg: Mapping, *, checkpoint: SelectedCheckpoint, contract: Mapping) -> None:
    from .nh_raw_space_evaluation import OBSERVATION_CUTOFF_CONFIG_KEY, parse_observation_cutoff

    checks = {
        "experiment_name": (cfg.get("experiment_name"), checkpoint.run_id),
        "seq_length": (cfg.get("seq_length"), checkpoint.seq_length),
        "seed": (cfg.get("seed"), checkpoint.seed),
        OBSERVATION_CUTOFF_CONFIG_KEY: (parse_observation_cutoff(cfg.get(OBSERVATION_CUTOFF_CONFIG_KEY)),
                                        pd.Timestamp(contract["observation_cutoff"])),
        "validation_start_date": (_nh_date(cfg.get("validation_start_date"), "validation_start_date"),
                                  pd.Timestamp(contract["date_start"])),
        "validation_end_date": (_nh_date(cfg.get("validation_end_date"), "validation_end_date"),
                                pd.Timestamp(contract["date_end"])),
        "target_variables": (cfg.get("target_variables"), [contract["target_variable"]]),
        "dynamic_inputs": (cfg.get("dynamic_inputs"), contract["dynamic_inputs"]),
    }
    for key, (observed, expected) in checks.items():
        if observed != expected:
            raise FullDevelopmentError(f"source config {key}={observed!r} != expected {expected!r}")
    if cfg.get("use_basin_id_encoding"):
        raise FullDevelopmentError("source config uses basin-id encoding; the staged run would need id_to_int")
    if cfg.get("additional_feature_files") not in (None, "None", [], ""):
        raise FullDevelopmentError("source config declares additional_feature_files; refusing to stage")
    if "validation_basin_file" not in cfg or "validate_n_random_basins" not in cfg:
        raise FullDevelopmentError("source config lacks validation_basin_file/validate_n_random_basins")


def stage_full_development_eval_run(*, source_run_dir, out_run_dir, checkpoint: SelectedCheckpoint, contract: dict,
                                    development_basin_file, canary_basin_ids: Optional[Sequence[str]] = None,
                                    fixture_only_allow_unqualified_contract: bool = False) -> dict:
    """Stage one checkpoint for full-development (or declared-canary)
    evaluation in a NEW directory ``out_run_dir``; returns the manifest.

    ``development_basin_file`` must hold exactly the contract's basins. With
    ``canary_basin_ids`` the staged validation list is that declared subset;
    otherwise it is the full list. NH is then run as
    ``run_stage1_nh.py eval <out_run_dir> --period validation --epoch N``."""
    from .nh_raw_space_evaluation import OBSERVATION_CUTOFF_CONFIG_KEY
    from .nh_seed_evaluation import (
        EVALUATION_ONLY_MARKER_FILENAME,
        NHSeedEvaluationError,
        _raise_if_out_run_dir_collides_with_development_run_dir,
        weight_stem,
    )

    validate_full_development_support_contract(contract)
    if fixture_only_allow_unqualified_contract:
        if contract["evaluation_population_role"] != FIXTURE_ONLY_ROLE:
            raise FullDevelopmentError("fixture-only staging requires a fixture-only contract")
    else:
        require_production_contract_identity(contract)
        if checkpoint not in SELECTED_CHECKPOINTS:
            raise FullDevelopmentError(f"{checkpoint.run_id}: not one of the nine frozen selected checkpoints")
    source = Path(source_run_dir)
    out = Path(out_run_dir)
    if source.name != checkpoint.nh_run_dir_name:
        raise FullDevelopmentError(f"source run dir {source.name!r} != expected {checkpoint.nh_run_dir_name!r}")
    try:
        _raise_if_out_run_dir_collides_with_development_run_dir(out, source)
    except NHSeedEvaluationError as exc:
        raise FullDevelopmentError(str(exc)) from exc
    if out.exists():
        raise FullDevelopmentError(f"refusing to reuse or overwrite existing staging directory: {out}")

    stem = weight_stem(checkpoint.epoch)
    src_config = source / "config.yml"
    src_model = source / f"{stem}.pt"
    src_scaler = source / "train_data" / "train_data_scaler.yml"
    for path in (src_config, src_model, src_scaler):
        if not path.is_file():
            raise FullDevelopmentError(f"missing source artifact: {path}")
    before = _tree_snapshot(source)
    for path, expected, label in ((src_config, checkpoint.config_sha256, "config.yml"),
                                  (src_model, checkpoint.model_sha256, "checkpoint"),
                                  (src_scaler, checkpoint.scaler_sha256, "scaler")):
        observed = _sha256_file(path)
        if observed != expected:
            raise FullDevelopmentError(f"source {label} sha256 {observed} != frozen {expected}")
    source_cfg = _read_yaml(src_config)
    _require_source_config_consistent(source_cfg, checkpoint=checkpoint, contract=contract)

    dev_file = Path(development_basin_file)
    dev_ids = _read_basin_list(dev_file)
    if sorted(dev_ids) != contract["basin_ids"] or \
            canonical_membership_sha256(dev_ids) != contract["development_basin_ids_sha256"]:
        raise FullDevelopmentError(f"{dev_file}: basin membership != contract development membership")
    if canary_basin_ids is None:
        scope, staged_ids = "full_development", sorted(dev_ids)
    else:
        staged_ids = [str(b) for b in canary_basin_ids]
        if not staged_ids or len(set(staged_ids)) != len(staged_ids) or not set(staged_ids) <= set(dev_ids):
            raise FullDevelopmentError("canary_basin_ids must be a non-empty unique subset of the development basins")
        scope, staged_ids = "canary", sorted(staged_ids)

    out.mkdir(parents=True)
    (out / "train_data").mkdir()
    staged_model = _copy_and_verify(src_model, out / f"{stem}.pt", expected_sha256=checkpoint.model_sha256,
                                    label="checkpoint")
    staged_scaler = _copy_and_verify(src_scaler, out / "train_data" / "train_data_scaler.yml",
                                     expected_sha256=checkpoint.scaler_sha256, label="scaler")
    basin_path = out / STAGED_VALIDATION_BASIN_FILENAME
    basin_path.write_bytes("".join(f"{b}\n" for b in staged_ids).encode("utf-8"))

    staged_cfg = dict(source_cfg)
    staged_cfg["validation_basin_file"] = basin_path.resolve().as_posix()
    staged_cfg["validate_n_random_basins"] = len(staged_ids)
    staged_cfg["run_dir"] = out.resolve().as_posix()
    if "train_dir" in source_cfg:
        staged_cfg["train_dir"] = (out / "train_data").resolve().as_posix()
    if "img_log_dir" in source_cfg:
        staged_cfg["img_log_dir"] = (out / "img_log").resolve().as_posix()
    with (out / "config.yml").open("w", encoding="utf-8") as fh:
        _yaml().dump(staged_cfg, fh)

    reloaded = _read_yaml(out / "config.yml")
    changes = []
    for key in sorted(set(source_cfg) | set(reloaded)):
        if source_cfg.get(key, "<absent>") != reloaded.get(key, "<absent>"):
            changes.append({"key": key, "source": source_cfg.get(key, "<absent>"),
                            "staged": reloaded.get(key, "<absent>")})
    illegal = [c["key"] for c in changes if c["key"] not in ALLOWED_STAGED_CONFIG_CHANGES]
    if illegal or set(reloaded) != set(source_cfg):
        raise FullDevelopmentError(f"staged config changed non-whitelisted keys {illegal} or the key set")
    if reloaded["validate_n_random_basins"] != len(staged_ids) or \
            _read_basin_list(Path(reloaded["validation_basin_file"])) != staged_ids:
        raise FullDevelopmentError("staged validation basin file / validate_n_random_basins do not cover the scope")
    if reloaded.get(OBSERVATION_CUTOFF_CONFIG_KEY) != source_cfg.get(OBSERVATION_CUTOFF_CONFIG_KEY):
        raise FullDevelopmentError("observation cutoff not preserved")

    (out / EVALUATION_ONLY_MARKER_FILENAME).write_text(
        "EVALUATION-ONLY staged run directory for the Common-336 full-development\n"
        f"comparison (contract {CONTRACT_ID}, scope {scope}, {len(staged_ids)} basins).\n"
        f"Checkpoint {stem}.pt and the scaler are byte-identical copies of the screening-selected\n"
        f"run {checkpoint.run_id}. Do NOT train here. Results from this directory are\n"
        "valid only through seqlen_common336_full_development completeness checks.\n", encoding="utf-8")

    if _tree_snapshot(source) != before or _sha256_file(src_config) != checkpoint.config_sha256:
        raise FullDevelopmentError(f"source run directory changed during staging: {source}")
    manifest = {
        "schema_name": "flashnh_c336_full_development_eval_staging_manifest", "schema_version": 1,
        "staged_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope, "contract_id": contract["contract_id"],
        "contract_checksum_sha256": contract["checksum_sha256"],
        "evaluation_population_role": contract["evaluation_population_role"],
        "development_basin_ids_sha256": contract["development_basin_ids_sha256"],
        "development_basin_file": str(dev_file), "development_basin_file_sha256": _sha256_file(dev_file),
        "staged_basin_ids_sha256": canonical_membership_sha256(staged_ids), "n_staged_basins": len(staged_ids),
        "checkpoint": asdict(checkpoint),
        # Pointer only: the ledger lives in the local analysis copy, so it is
        # not re-read here; the model/config/scaler checks above are.
        "qualification_record": {**QUALIFICATION_RECORD, "ledger_index": checkpoint.qualification_index,
                                 "ledger_qualified": True, "ledger_selected_epoch": checkpoint.epoch},
        "source_run_dir": str(source),
        "source_sha256": {"config.yml": checkpoint.config_sha256, f"{stem}.pt": checkpoint.model_sha256,
                          "train_data/train_data_scaler.yml": checkpoint.scaler_sha256},
        "source_tree_unchanged": True, "n_source_files_snapshotted": len(before),
        "staged_run_dir": str(out),
        "staged_sha256": {"config.yml": _sha256_file(out / "config.yml"), f"{stem}.pt": staged_model,
                          "train_data/train_data_scaler.yml": staged_scaler,
                          STAGED_VALIDATION_BASIN_FILENAME: _sha256_file(basin_path)},
        "allowed_config_changes": list(ALLOWED_STAGED_CONFIG_CHANGES), "config_changes": changes,
        "observation_cutoff": contract["observation_cutoff"], "epoch": checkpoint.epoch,
        "evaluation_only_marker": EVALUATION_ONLY_MARKER_FILENAME,
    }
    (out / STAGING_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
