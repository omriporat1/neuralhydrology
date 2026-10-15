"""Common-336 (basin, issue-time) support for the three-seed sequence-length
experiment (L in {72, 120, 168, 240, 336} h).

Scientific contract (user-approved design, 2026-10-14):

    Every sequence length trains on, and is screened on, the SAME eligible
    (basin, issue-time) pairs. Eligibility is defined by a valid 336-hour
    input history; each model still consumes only its own L input hours.

    admitted_336(b, t) = history_ok_336(b, t)            # see below
                       & period_start <= t <= period_end
                       & t + lead <= period_end            # period target boundary
                       & finite(target[b, t])

    history_ok_336(b, t): the inclusive window [t - 335 h, t] lies inside the
    package-native timeline and contains no BAD hour for basin b, where

    bad(b, h) = h is a documented MRMS/RTMA archive gap (masks/gap_timestamps.json)
              | any configured dynamic input of basin b is non-finite at h.

Because a gap-free, finite 336 h window implies the same for every suffix of
length L <= 336, and the boundary/target clauses do not depend on L, this set
is a subset of every L's natural FlashNHDataset training lookup (NH's own
train-mode any-NaN-in-x_d exclusion plus Flash-NH concerns A/B/C).
:class:`~src.baseline.nh_dataset.FlashNHDataset` verifies that subset relation
at load time and fails closed if it does not hold.

Two artifacts, two consumers:

* Training support (new schema :data:`TRAINING_SUPPORT_SCHEMA_NAME`; run-length
  encoded per-basin issue-time intervals, sha256-checksummed). Producer:
  :func:`build_common336_support`. Consumer: ``FlashNHDataset`` when the NH
  config key :data:`COMMON_SUPPORT_CONFIG_KEY` is set (train period only).
* Screening contract: the EXISTING fixed-support schema
  (:mod:`fixed_support_contract_v2`), under the separate contract id
  :data:`SCREENING_CONTRACT_ID` and ``seq_length_floor`` 336. Consumer:
  :func:`~src.baseline.fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`
  called with ``expected_contract_id``/``expected_seq_length_floor`` set
  (:func:`evaluate_common336_screening`).

This is NOT the historical Common-120 contract (``common120_raw_space_nse_v001``)
and never relabels it. Common-120 fixed only the *screening* support, with a
120 h gap-only history floor and no basin-specific input-finiteness clause,
while each sequence length trained on its own natural sample pool; scoring a
L > 120 run on Common-120 is therefore not a common-support comparison. The
Common-120 artifacts, ids and v2 constants are unchanged by this module.

Reads are restricted to the verified development (training) and screening
basin lists and to the package timeline up to the validation end date; no
test-period value is materialized. Basins with zero admitted pairs are
reported as failures (:class:`Common336SupportError`), never dropped.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .fixed_support_contract_v2 import (
    build_fixed_support_contract,
    canonical_contract_checksum_payload,
    evaluate_fixed_support_raw_space_metrics,
    load_fixed_support_contract,
    validate_fixed_support_contract,
    write_fixed_support_contract,
)
from .validity_mask import _validate_hourly_index, bad_hour_mask_from_timestamps, compute_history_valid

__all__ = [
    "CAMPAIGN_ID",
    "COMMON_SEQ_LENGTH_FLOOR",
    "SEQ_LENGTHS",
    "SCREENING_CONTRACT_ID",
    "TRAINING_SUPPORT_ID",
    "TRAINING_SUPPORT_SCHEMA_NAME",
    "TRAINING_SUPPORT_SCHEMA_VERSION",
    "SUPPORT_POLICY_IDENTITY",
    "COMMON_SUPPORT_CONFIG_KEY",
    "Common336SupportError",
    "BasinSupportArrays",
    "basin_support_arrays",
    "encode_issue_intervals",
    "decode_issue_intervals",
    "build_training_support",
    "validate_training_support",
    "write_training_support",
    "load_training_support",
    "build_common336_support",
    "compute_common336_support",
    "write_common336_screening_contract",
    "load_common336_screening_contract",
    "evaluate_common336_screening",
    "screening_observation_cutoff",
]

CAMPAIGN_ID = "seqlen_common336_seed3_v001"
COMMON_SEQ_LENGTH_FLOOR = 336
SEQ_LENGTHS = (72, 120, 168, 240, 336)
SCREENING_CONTRACT_ID = "common336_raw_space_nse_v001"
TRAINING_SUPPORT_ID = "common336_training_support_v001"
TRAINING_SUPPORT_SCHEMA_NAME = "flashnh_seqlen_common_training_support"
TRAINING_SUPPORT_SCHEMA_VERSION = 1
SUPPORT_POLICY_IDENTITY = "seqlen_common336_gap_or_nonfinite_dynamic_input_history_v001"
# NH config key read by FlashNHDataset (registered on NH's Config by
# nh_register.register_flashnh_dataset()).
COMMON_SUPPORT_CONFIG_KEY = "flashnh_common_support_file"

_HOUR = pd.Timedelta(hours=1)
_ISO_UNIT = "s"

_TRAINING_REQUIRED_KEYS = {
    "schema_name", "schema_version", "support_id", "campaign_id", "seq_length_floor", "lead_hours",
    "target_variable", "dynamic_inputs", "period", "date_start", "date_end", "support_policy_identity",
    "gap_artifact_sha256", "package_manifest_sha256", "package_file_checksums_sha256",
    "package_run_provenance_sha256", "development_split_sha256", "spatial_holdout_split_sha256",
    "train_basin_ids_sha256", "basin_ids", "per_basin_intervals", "eligible_counts", "checksum_sha256",
}


class Common336SupportError(ValueError):
    """Malformed/inconsistent support input or artifact, identity mismatch,
    or a basin without admitted support (a reported failure, never a drop)."""


# --------------------------------------------------------------------------- #
# Pure per-basin support math
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class BasinSupportArrays:
    """Boolean vectors aligned to one basin's package-native timeline.

    ``natural[L]`` is the per-L natural admission (history window of L over
    the same bad-hour definition); ``admitted`` is ``natural[floor]``. The
    ``lost_*`` vectors partition ``natural[L] & ~admitted`` by first cause, in
    priority order: insufficient package history (timeline start), documented
    archive gap, other non-finite dynamic input."""

    dates: pd.DatetimeIndex
    admitted: np.ndarray
    natural: Mapping[int, np.ndarray]
    lost_timeline_start: Mapping[int, np.ndarray]
    lost_documented_gap: Mapping[int, np.ndarray]
    lost_nonfinite_input: Mapping[int, np.ndarray]
    n_nonfinite_input_hours_not_gap: int


def basin_support_arrays(
    *,
    dates: pd.DatetimeIndex,
    gap_mask: np.ndarray,
    dynamic_values: Sequence[np.ndarray],
    target_values: np.ndarray,
    lead_hours: int,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    seq_lengths: Sequence[int] = SEQ_LENGTHS,
    floor: int = COMMON_SEQ_LENGTH_FLOOR,
) -> BasinSupportArrays:
    _validate_hourly_index(dates)
    n = len(dates)
    gap = np.asarray(gap_mask, dtype=bool)
    if gap.shape != (n,):
        raise Common336SupportError(f"gap mask shape {gap.shape} != timeline ({n},)")
    if not dynamic_values:
        raise Common336SupportError("at least one dynamic input is required")
    nonfinite = np.zeros(n, dtype=bool)
    for values in dynamic_values:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (n,):
            raise Common336SupportError(f"dynamic input shape {arr.shape} != timeline ({n},)")
        nonfinite |= ~np.isfinite(arr)
    target = np.asarray(target_values, dtype=np.float64)
    if target.shape != (n,):
        raise Common336SupportError(f"target shape {target.shape} != timeline ({n},)")
    if floor not in seq_lengths or max(seq_lengths) != floor:
        raise Common336SupportError(f"floor {floor} must be the maximum of seq_lengths {tuple(seq_lengths)}")

    bad = gap | nonfinite
    lead = pd.Timedelta(hours=int(lead_hours))
    common = (np.asarray(dates >= period_start) & np.asarray(dates <= period_end)
              & np.asarray(dates + lead <= period_end) & np.isfinite(target))
    no_bad = np.zeros(n, dtype=bool)
    hist_floor = compute_history_valid(dates, bad, floor)
    hist_floor_gap_only = compute_history_valid(dates, gap, floor)
    hist_floor_no_bad = compute_history_valid(dates, no_bad, floor)  # timeline-start clause only
    admitted = common & hist_floor
    natural, lost_start, lost_gap, lost_nonfinite = {}, {}, {}, {}
    for L in seq_lengths:
        nat = common & compute_history_valid(dates, bad, int(L))
        lost = nat & ~admitted
        lost_start[L] = lost & ~hist_floor_no_bad
        lost_gap[L] = lost & hist_floor_no_bad & ~hist_floor_gap_only
        lost_nonfinite[L] = lost & hist_floor_gap_only
        natural[L] = nat
    return BasinSupportArrays(
        dates=dates, admitted=admitted, natural=natural, lost_timeline_start=lost_start,
        lost_documented_gap=lost_gap, lost_nonfinite_input=lost_nonfinite,
        n_nonfinite_input_hours_not_gap=int((nonfinite & ~gap & np.asarray(dates <= period_end)).sum()),
    )


# --------------------------------------------------------------------------- #
# Training-support artifact (run-length encoded issue-time intervals)
# --------------------------------------------------------------------------- #

def encode_issue_intervals(dates: pd.DatetimeIndex, admitted: np.ndarray) -> list:
    """Inclusive hourly ``[start_iso, end_iso]`` runs of admitted issue times."""
    admitted = np.asarray(admitted, dtype=bool)
    if admitted.shape != (len(dates),):
        raise Common336SupportError("admitted mask/timeline length mismatch")
    padded = np.concatenate(([False], admitted, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    values = dates.to_numpy(dtype="datetime64[ns]")
    iso_start = np.datetime_as_string(values[starts], unit=_ISO_UNIT)
    iso_end = np.datetime_as_string(values[ends], unit=_ISO_UNIT)
    return [[str(a), str(b)] for a, b in zip(iso_start, iso_end)]


def decode_issue_intervals(intervals: Sequence[Sequence[str]]) -> np.ndarray:
    """Admitted issue times as a sorted ``datetime64[ns]`` array."""
    pieces = []
    previous_end = None
    for pair in intervals:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise Common336SupportError(f"malformed interval {pair!r}")
        start, end = pd.Timestamp(pair[0]), pd.Timestamp(pair[1])
        if start.tzinfo is not None or end.tzinfo is not None or end < start:
            raise Common336SupportError(f"interval {pair!r} must be tz-naive UTC with end >= start")
        if previous_end is not None and start <= previous_end + _HOUR:
            raise Common336SupportError(f"intervals must be sorted, disjoint and non-adjacent at {pair!r}")
        pieces.append(pd.date_range(start, end, freq="h").to_numpy(dtype="datetime64[ns]"))
        previous_end = end
    if not pieces:
        return np.array([], dtype="datetime64[ns]")
    return np.concatenate(pieces)


def _checksum(payload: Mapping) -> str:
    return hashlib.sha256(canonical_contract_checksum_payload(payload)).hexdigest()


def build_training_support(*, per_basin_dates: Mapping[str, pd.DatetimeIndex],
                           per_basin_admitted: Mapping[str, np.ndarray], lead_hours: int,
                           target_variable: str, dynamic_inputs: Sequence[str], date_start: str,
                           date_end: str, identities: Mapping[str, str], train_basin_ids_sha256: str,
                           seq_length_floor: int = COMMON_SEQ_LENGTH_FLOOR) -> dict:
    basin_ids = sorted(per_basin_dates)
    if basin_ids != sorted(per_basin_admitted) or not basin_ids:
        raise Common336SupportError("per-basin dates/admitted must share one non-empty basin set")
    intervals, counts, zero = {}, {}, []
    for basin in basin_ids:
        admitted = np.asarray(per_basin_admitted[basin], dtype=bool)
        intervals[basin] = encode_issue_intervals(per_basin_dates[basin], admitted)
        counts[basin] = int(admitted.sum())
        if counts[basin] == 0:
            zero.append(basin)
    if zero:
        raise Common336SupportError(f"{len(zero)} training basin(s) have zero Common-336 support: {zero}")
    payload = {
        "schema_name": TRAINING_SUPPORT_SCHEMA_NAME, "schema_version": TRAINING_SUPPORT_SCHEMA_VERSION,
        "support_id": TRAINING_SUPPORT_ID, "campaign_id": CAMPAIGN_ID,
        "seq_length_floor": int(seq_length_floor), "lead_hours": int(lead_hours),
        "target_variable": str(target_variable), "dynamic_inputs": list(dynamic_inputs), "period": "train",
        "date_start": str(date_start), "date_end": str(date_end),
        "support_policy_identity": SUPPORT_POLICY_IDENTITY,
        **{key: identities[key] for key in (
            "gap_artifact_sha256", "package_manifest_sha256", "package_file_checksums_sha256",
            "package_run_provenance_sha256", "development_split_sha256", "spatial_holdout_split_sha256")},
        "train_basin_ids_sha256": train_basin_ids_sha256, "basin_ids": basin_ids,
        "per_basin_intervals": intervals, "eligible_counts": counts,
    }
    payload["checksum_sha256"] = _checksum(payload)
    return validate_training_support(payload)


def validate_training_support(data: Mapping) -> dict:
    if not isinstance(data, dict):
        raise Common336SupportError(f"training support must be a mapping, got {type(data).__name__}")
    missing, extra = _TRAINING_REQUIRED_KEYS - set(data), set(data) - _TRAINING_REQUIRED_KEYS
    if missing or extra:
        raise Common336SupportError(f"training support keys: missing {sorted(missing)}, extra {sorted(extra)}")
    expected = {"schema_name": TRAINING_SUPPORT_SCHEMA_NAME, "schema_version": TRAINING_SUPPORT_SCHEMA_VERSION,
                "support_id": TRAINING_SUPPORT_ID, "campaign_id": CAMPAIGN_ID,
                "seq_length_floor": COMMON_SEQ_LENGTH_FLOOR, "period": "train",
                "support_policy_identity": SUPPORT_POLICY_IDENTITY}
    for key, value in expected.items():
        if data[key] != value or type(data[key]) is not type(value):
            raise Common336SupportError(f"training support {key} must be {value!r}, got {data[key]!r}")
    basin_ids = data["basin_ids"]
    if not isinstance(basin_ids, list) or not basin_ids or basin_ids != sorted(set(basin_ids)):
        raise Common336SupportError("basin_ids must be a non-empty sorted list of unique ids")
    if set(basin_ids) != set(data["per_basin_intervals"]) or set(basin_ids) != set(data["eligible_counts"]):
        raise Common336SupportError("basin_ids must match per_basin_intervals and eligible_counts")
    for basin in basin_ids:
        count = data["eligible_counts"][basin]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise Common336SupportError(f"basin {basin!r}: eligible count must be a positive int, got {count!r}")
        total = sum(int((pd.Timestamp(e) - pd.Timestamp(s)) / _HOUR) + 1 for s, e in data["per_basin_intervals"][basin])
        if total != count:
            raise Common336SupportError(f"basin {basin!r}: intervals cover {total} hours, eligible_counts says {count}")
    if _checksum(data) != data["checksum_sha256"]:
        raise Common336SupportError("training support checksum mismatch -- payload altered after checksumming")
    return data


def write_training_support(data: dict, path) -> Path:
    path = Path(path)
    validate_training_support(data)
    if path.exists():
        raise Common336SupportError(f"refusing to overwrite existing training support: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def load_training_support(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise Common336SupportError(f"training support not found: {path}")
    return validate_training_support(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Package-reading producer
# --------------------------------------------------------------------------- #

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_basin(package_root, basin: str, variables: Sequence[str], last_date: pd.Timestamp,
                lead_hours: int = 0):
    """Package-native dates and values up to ``last_date`` only; later
    (test-period) values are never materialized. The LAST variable is the
    lead-shifted target, whose value at t is the observation at t + lead: it
    is read only up to ``last_date - lead`` and NaN-padded after, so no
    observation later than ``last_date`` is ever materialized either (those
    issue times are inadmissible by the period-target boundary clause)."""
    import xarray as xr

    from .nh_seed_evaluation import basin_netcdf_path

    path = basin_netcdf_path(package_root, basin)
    if not path.is_file():
        raise Common336SupportError(f"missing basin NetCDF: {path}")
    with xr.open_dataset(path) as data:
        coord = "date" if "date" in data.coords else "time" if "time" in data.coords else None
        if coord is None or any(v not in data for v in variables):
            raise Common336SupportError(f"{path}: required coordinate/variables missing")
        sub = data.sel({coord: slice(None, last_date)})
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(sub[coord].values)))
        values = [np.asarray(sub[v].values, dtype=np.float64) for v in variables[:-1]]
        target_last = last_date - pd.Timedelta(hours=int(lead_hours))
        target = np.asarray(data[variables[-1]].sel({coord: slice(None, target_last)}).values, dtype=np.float64)
        if target.ndim != 1 or len(target) > len(dates):
            raise Common336SupportError(f"{path}: {variables[-1]} is not one-dimensional and date-aligned")
        values.append(np.concatenate([target, np.full(len(dates) - len(target), np.nan)]))
    for name, arr in zip(variables, values):
        if arr.shape != (len(dates),):
            raise Common336SupportError(f"{path}: {name} is not one-dimensional and date-aligned")
    try:
        _validate_hourly_index(dates)
    except Exception as exc:
        raise Common336SupportError(f"{path}: invalid package-native timeline: {exc}") from exc
    return dates, values


def _year_counts(dates: pd.DatetimeIndex, mask: np.ndarray) -> dict:
    years = dates.year[np.asarray(mask, dtype=bool)]
    unique, counts = np.unique(np.asarray(years), return_counts=True)
    return {int(y): int(c) for y, c in zip(unique, counts)}


def _period_bounds(policy_period: Mapping) -> tuple[pd.Timestamp, pd.Timestamp]:
    # +23 h is the last hourly point of an inclusive end date (the same
    # convention as common120_support_builder / NH's date + 1 day - 1 s).
    return pd.Timestamp(policy_period["start"]), pd.Timestamp(policy_period["end"]) + pd.Timedelta(hours=23)


def build_common336_support(*, package_root, splits_dir, screening_basin_ids_path, baseline_policy_path,
                            train_basin_file, dynamic_inputs: Sequence[str],
                            target_variable: str = "qobs_mm_per_h_lead06", lead_hours: int = 6,
                            seq_lengths: Sequence[int] = SEQ_LENGTHS, basin_subset: Sequence[str] | None = None,
                            progress=None) -> dict:
    """Build (never write) both Common-336 artifacts plus per-basin accounting.

    ``basin_subset`` (calibration only) restricts the basins read; artifacts
    are then NOT built (a partial artifact must never exist), only accounting.
    Returns ``{"training_support", "screening_contract", "accounting"}``."""
    from . import sweep_v1_campaign as sweep
    from .gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT, load_gap_timestamps_json
    from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership
    from .pilot_lead06_config import load_screening_basin_ids
    from .policy import load_stage1_baseline_policy
    from .sweep_v1_production_adapter import PreparationPaths, _verify_artifact_identities

    identities = dict(_verify_artifact_identities(PreparationPaths(
        Path(baseline_policy_path), Path(package_root), Path(splits_dir), Path(screening_basin_ids_path))))
    manifest = read_package_manifest(package_root)
    if manifest.get("gap_product_scope") != [MRMS_PRODUCT, RTMA_PRODUCT]:
        raise Common336SupportError("package gap_product_scope must be complete MRMS+RTMA")
    gap_path = Path(package_root) / "masks" / "gap_timestamps.json"
    gap_sha = _sha256_file(gap_path)
    if gap_sha != (manifest.get("gap_timestamp_artifact") or {}).get("sha256"):
        raise Common336SupportError("package gap-timestamp artifact checksum contradicts package manifest")
    identities["gap_artifact_sha256"] = gap_sha

    membership = validate_full_population_basin_membership(manifest, splits_dir)
    train_bytes = Path(train_basin_file).read_bytes()
    train_sha = hashlib.sha256(train_bytes).hexdigest()
    train_basins = [line.strip() for line in train_bytes.decode("utf-8").splitlines() if line.strip()]
    if train_sha != identities["development_split_sha256"] or sorted(train_basins) != sorted(membership.development_basins):
        raise Common336SupportError("train basin file is not byte-identical to the verified development split")
    if len(train_basins) != 2307 or len(set(train_basins)) != 2307:
        raise Common336SupportError("training population must be exactly 2307 unique basins")
    screening = load_screening_basin_ids(screening_basin_ids_path, development_basins=membership.development_basins,
                                         expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)

    policy = load_stage1_baseline_policy(baseline_policy_path)
    return compute_common336_support(
        package_root=package_root, train_basins=train_basins, screening_basins=screening,
        train_period=_period_bounds(policy["temporal_split"]["training"]),
        validation_period=_period_bounds(policy["temporal_split"]["validation"]),
        gap_timestamps=load_gap_timestamps_json(gap_path), dynamic_inputs=dynamic_inputs,
        target_variable=target_variable, lead_hours=lead_hours, seq_lengths=seq_lengths, identities=identities,
        train_basin_ids_sha256=train_sha, screening_basin_ids_sha256=sweep.SCREENING_ARTIFACT_SHA256,
        basin_subset=basin_subset, progress=progress)


def compute_common336_support(*, package_root, train_basins: Sequence[str], screening_basins: Sequence[str],
                              train_period: tuple, validation_period: tuple, gap_timestamps,
                              dynamic_inputs: Sequence[str], target_variable: str, lead_hours: int,
                              seq_lengths: Sequence[int], identities: Mapping[str, str],
                              train_basin_ids_sha256: str, screening_basin_ids_sha256: str,
                              basin_subset: Sequence[str] | None = None, progress=None) -> dict:
    """Identity-agnostic core of :func:`build_common336_support` (which is the
    only production entry point: it verifies package/split/screening identity
    and the gap artifact first). ``*_period`` are inclusive hourly
    ``(first, last)`` timestamps."""
    train_start, train_end = (pd.Timestamp(v) for v in train_period)
    val_start, val_end = (pd.Timestamp(v) for v in validation_period)
    gaps = list(gap_timestamps)
    if identities.get("gap_artifact_sha256") is None:
        raise Common336SupportError("identities must include gap_artifact_sha256")

    train_set, screening_set = set(train_basins), set(screening_basins)
    basins = sorted(train_set | screening_set)
    if basin_subset is not None:
        unknown = set(basin_subset) - set(basins)
        if unknown:
            raise Common336SupportError(f"basin_subset outside the verified membership: {sorted(unknown)}")
        basins = sorted(set(basin_subset))

    variables = list(dynamic_inputs) + [target_variable]
    reference_dates, gap_mask = None, None
    train_dates, train_admitted, val_dates, val_admitted = {}, {}, {}, {}
    rows, failures = [], []
    for i, basin in enumerate(basins):
        try:
            dates, values = _read_basin(package_root, basin, variables, val_end, lead_hours)
            if reference_dates is not None and not dates.equals(reference_dates):
                raise Common336SupportError(f"{basin}: timeline differs from the package-native timeline")
        except Common336SupportError as exc:  # identity failure: reported, never dropped
            failures.append({"basin_id": basin, "role": "read", "reason": str(exc)})
            continue
        if reference_dates is None:
            reference_dates = dates
            gap_mask = bad_hour_mask_from_timestamps(dates, gaps, on_out_of_range="ignore")
        dyn, target = values[:-1], values[-1]
        for role, member, start, end, store_d, store_a in (
                ("train", basin in train_set, train_start, train_end, train_dates, train_admitted),
                ("validation", basin in screening_set, val_start, val_end, val_dates, val_admitted)):
            if not member:
                continue
            arrays = basin_support_arrays(dates=dates, gap_mask=gap_mask, dynamic_values=dyn, target_values=target,
                                          lead_hours=lead_hours, period_start=start, period_end=end,
                                          seq_lengths=seq_lengths)
            store_d[basin], store_a[basin] = reference_dates, arrays.admitted
            n_common = int(arrays.admitted.sum())
            if n_common == 0:
                failures.append({"basin_id": basin, "role": role, "reason": "zero Common-336 support"})
            row = {"basin_id": basin, "role": role, "n_common336": n_common,
                   "n_nonfinite_input_hours_not_gap": arrays.n_nonfinite_input_hours_not_gap,
                   "common336_by_year": _year_counts(dates, arrays.admitted)}
            for L in seq_lengths:
                row[f"n_natural_L{L}"] = int(arrays.natural[L].sum())
                row[f"lost_timeline_start_L{L}"] = int(arrays.lost_timeline_start[L].sum())
                row[f"lost_documented_gap_L{L}"] = int(arrays.lost_documented_gap[L].sum())
                row[f"lost_nonfinite_input_L{L}"] = int(arrays.lost_nonfinite_input[L].sum())
                row[f"natural_by_year_L{L}"] = _year_counts(dates, arrays.natural[L])
            rows.append(row)
        if progress is not None:
            progress(i + 1, len(basins), basin)

    accounting = {"identities": dict(identities), "train_basin_ids_sha256": train_basin_ids_sha256,
                  "screening_basin_ids_sha256": screening_basin_ids_sha256,
                  "n_train_basins": len(train_dates), "n_screening_basins": len(val_dates),
                  "train_period": [str(train_start), str(train_end)], "validation_period": [str(val_start), str(val_end)],
                  "timeline": None if reference_dates is None else [str(reference_dates[0]), str(reference_dates[-1])], "n_gap_timestamps": len(gaps),
                  "dynamic_inputs": list(dynamic_inputs), "target_variable": target_variable,
                  "lead_hours": int(lead_hours), "seq_lengths": list(seq_lengths), "rows": rows,
                  "failures": failures, "basin_subset": basin_subset is not None}
    if basin_subset is not None or failures:
        return {"training_support": None, "screening_contract": None, "accounting": accounting}

    training_support = build_training_support(
        per_basin_dates=train_dates, per_basin_admitted=train_admitted, lead_hours=lead_hours,
        target_variable=target_variable, dynamic_inputs=dynamic_inputs,
        date_start=str(train_start.date()), date_end=str(train_end.date()), identities=identities,
        train_basin_ids_sha256=train_basin_ids_sha256)
    screening_contract = build_fixed_support_contract(
        contract_id=SCREENING_CONTRACT_ID, lead_hours=lead_hours, target_variable=target_variable,
        period="validation", date_start=str(val_start.date()), date_end=str(val_end.date()),
        source_gap_policy_identity=SUPPORT_POLICY_IDENTITY,
        screening_basin_ids_sha256=screening_basin_ids_sha256,
        per_basin_date={b: d.to_numpy() for b, d in val_dates.items()}, per_basin_admitted=val_admitted,
        seq_length_floor=COMMON_SEQ_LENGTH_FLOOR,
        **{k: identities[k] for k in ("package_manifest_sha256", "package_file_checksums_sha256",
                                      "package_run_provenance_sha256", "development_split_sha256",
                                      "spatial_holdout_split_sha256")})
    validate_fixed_support_contract(screening_contract, expected_contract_id=SCREENING_CONTRACT_ID,
                                    expected_seq_length_floor=COMMON_SEQ_LENGTH_FLOOR)
    return {"training_support": training_support, "screening_contract": screening_contract, "accounting": accounting}


def write_common336_screening_contract(contract: dict, path) -> Path:
    return write_fixed_support_contract(contract, path, expected_contract_id=SCREENING_CONTRACT_ID,
                                        expected_seq_length_floor=COMMON_SEQ_LENGTH_FLOOR)


def load_common336_screening_contract(path) -> dict:
    return load_fixed_support_contract(path, expected_contract_id=SCREENING_CONTRACT_ID,
                                       expected_seq_length_floor=COMMON_SEQ_LENGTH_FLOOR)


def screening_observation_cutoff(contract: Mapping) -> pd.Timestamp:
    """The campaign's observation cutoff: the last hour of the screening
    contract's (validation) period, i.e. the last observation the campaign may
    read. Every Common-336 run config declares exactly this value."""
    return pd.Timestamp(contract["date_end"]) + pd.Timedelta(hours=23)


def evaluate_common336_screening(*, run_dir, epoch: int, package_root, contract: dict,
                                 require_full_screening_population: bool = True, **kwargs) -> dict:
    """The qualified fixed-support raw-space scorer, bound to the Common-336
    screening contract identity (same masking, area, inverse conversion and
    metric implementation as Common-120 scoring), with the campaign's
    observation cutoff REQUIRED: the run must declare
    ``flashnh_observation_cutoff`` equal to :func:`screening_observation_cutoff`,
    so area derivation never reads a sealed (post-validation) observation."""
    from .nh_raw_space_evaluation import read_run_observation_cutoff

    cutoff = screening_observation_cutoff(contract)
    declared = read_run_observation_cutoff(run_dir)
    if declared != cutoff:
        raise Common336SupportError(
            f"{run_dir}: run declares observation cutoff {declared}, Common-336 screening requires {cutoff}")
    return evaluate_fixed_support_raw_space_metrics(
        run_dir=run_dir, epoch=epoch, package_root=package_root, contract=contract,
        require_full_screening_population=require_full_screening_population,
        expected_contract_id=SCREENING_CONTRACT_ID, expected_seq_length_floor=COMMON_SEQ_LENGTH_FLOOR,
        observation_cutoff=cutoff, **kwargs)
