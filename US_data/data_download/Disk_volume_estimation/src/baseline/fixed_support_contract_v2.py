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
from dataclasses import dataclass
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
from .authenticated_period_results import (
    AuthenticatedPeriodResults,
    AuthenticatedPeriodResultsError,
    load_authenticated_period_results,
)
from .nh_seed_evaluation import (
    basin_netcdf_path,
    load_period_results,
    period_results_path,
    raw_space_metrics_for_run_period,
)
from .package_identity_qualification import (
    PackageIdentityError,
    QualifiedPackageIdentity,
    qualify_package_identity,
    verify_basin_time_series_file,
)
from .sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2, SEQ_LENGTH_MAX

__all__ = [
    "FixedSupportContractError",
    "CONTRACT_SCHEMA_NAME",
    "CONTRACT_SCHEMA_VERSION",
    "AdmittedSeries",
    "CanonicalPackageObservedSeries",
    "derive_canonical_package_observed_series",
    # Re-exported (RD1-C4-D1 finding A1) so a contract-bound consumer can
    # obtain the package-identity proof from the same module it obtains the
    # contract from, without a second import path to keep in sync.
    "PackageIdentityError",
    "QualifiedPackageIdentity",
    "qualify_package_identity",
    "verify_basin_time_series_file",
    # Re-exported (RD1-C4-D1 Correction Pass A, finding A3) so a consumer
    # obtains the authenticated evaluation-source object from the same
    # module it obtains the contract and the evaluator from.
    "AuthenticatedPeriodResults",
    "AuthenticatedPeriodResultsError",
    "load_authenticated_period_results",
    "fixture_only_unqualified_package",
    "SupportContractProvenance",
    "build_support_contract_provenance",
    "build_fixed_support_contract",
    "validate_fixed_support_contract",
    "write_fixed_support_contract",
    "load_fixed_support_contract",
    "evaluate_fixed_support_raw_space_metrics",
    "evaluate_natural_support_raw_space_metrics",
    "extract_v2_objective_from_fixed_support_result",
    # Neutral primitives shared additively with the devpop-audit contract family.
    "serialize_support_date_array",
    "deserialize_support_date_array",
    "canonical_contract_checksum_payload",
    "is_strict_int",
    "strict_int",
]


class FixedSupportContractError(ValueError):
    """Raised for a malformed/inconsistent fixed-support contract, an
    attempted overwrite, or an evaluation-time identity contradiction
    (basin/date/support mismatch, checksum mismatch, wrong
    ``objective_scope``). Never raised for an ordinary poor-skill outcome."""


CONTRACT_SCHEMA_NAME = "flashnh_stage1_v2_fixed_support_contract"
CONTRACT_SCHEMA_VERSION = 2

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
    "package_manifest_sha256",
    "package_file_checksums_sha256",
    "package_run_provenance_sha256",
    "development_split_sha256",
    "spatial_holdout_split_sha256",
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


def _canonicalize_timestamps_for_identity(values: np.ndarray, *, date_dtype: str, context: str) -> np.ndarray:
    """Normalizes a timestamp array to the one canonical, validated
    representation implied by the frozen contract's own ``date_dtype``,
    before it is used to build an identity lookup (``np.unique``/
    ``np.isin`` membership, or a Python ``dict`` keyed by timestamp
    scalars). Equivalent timestamps represented in different
    ``datetime64`` units (e.g. ``datetime64[D]`` vs ``datetime64[ns]``)
    compare equal under ``==``/``np.isin`` but are distinct Python scalar
    objects with unit-dependent hashes, so a dict keyed by one run's
    ``datetime64[D]`` values will not resolve lookups by the contract's
    always-``datetime64[ns]`` values (see :func:`_deserialize_date_array`)
    even though the underlying instants agree -- RD1-C4 review Finding 4.
    """
    arr = np.asarray(values)
    if date_dtype == "datetime64":
        if not np.issubdtype(arr.dtype, np.datetime64):
            raise FixedSupportContractError(
                f"{context}: expected a datetime64 date coordinate for a 'datetime64' contract, got {arr.dtype!r}"
            )
        return arr.astype("datetime64[ns]")
    if date_dtype == "int64":
        if not np.issubdtype(arr.dtype, np.integer):
            raise FixedSupportContractError(
                f"{context}: expected an integer date coordinate for an 'int64' contract, got {arr.dtype!r}"
            )
        return arr.astype("int64")
    raise FixedSupportContractError(f"{context}: unsupported date_dtype {date_dtype!r}; expected 'datetime64' or 'int64'")


def _canonical_payload_for_checksum(payload: Mapping) -> bytes:
    body = {k: v for k, v in payload.items() if k != "checksum_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------------- #
# Neutral primitives, exposed additively (same objects as the internal
# underscore helpers).  The development-population Common-120 *audit* contract
# family (:mod:`src.baseline.devpop_common120_audit_contract`) reuses these so
# the two contract families stay byte-compatible in how they serialize
# per-basin support and checksum, without any module importing an
# underscore-prefixed name across a package boundary.  These are pure
# encoding/accounting helpers -- no scientific math, no population semantics,
# no optimizer identity.
# --------------------------------------------------------------------------- #

serialize_support_date_array = _serialize_date_array
deserialize_support_date_array = _deserialize_date_array
canonical_contract_checksum_payload = _canonical_payload_for_checksum


def is_strict_int(value: object) -> bool:
    """True only for a genuine Python ``int`` that is not a ``bool``.

    ``True``/``False`` and floating-point values (even integral ones such as
    ``2307.0``) are rejected -- an accounting field that compares "equal" to an
    integer only after a bool/float coercion is treated as missing evidence.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def strict_int(value: object, *, name: str, minimum: Optional[int] = None) -> int:
    """Return ``value`` if it is a strict non-bool ``int`` (optionally
    ``>= minimum``); otherwise raise :class:`FixedSupportContractError`.
    """
    if not is_strict_int(value):
        raise FixedSupportContractError(
            f"{name} must be a strict integer (no bool, no float), got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise FixedSupportContractError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


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
    package_manifest_sha256: str,
    package_file_checksums_sha256: str,
    package_run_provenance_sha256: str,
    development_split_sha256: str,
    spatial_holdout_split_sha256: str,
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
        "package_manifest_sha256": package_manifest_sha256,
        "package_file_checksums_sha256": package_file_checksums_sha256,
        "package_run_provenance_sha256": package_run_provenance_sha256,
        "development_split_sha256": development_split_sha256,
        "spatial_holdout_split_sha256": spatial_holdout_split_sha256,
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
    for key in (
        "screening_basin_ids_sha256", "package_manifest_sha256", "package_file_checksums_sha256",
        "package_run_provenance_sha256", "development_split_sha256", "spatial_holdout_split_sha256",
    ):
        value = data[key]
        if not isinstance(value, str) or len(value) != 64 or value != value.lower() or any(c not in "0123456789abcdef" for c in value):
            raise FixedSupportContractError(f"{key} must be a lowercase 64-character SHA-256")

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


@dataclass(frozen=True)
class SupportContractProvenance:
    """Immutable snapshot of a fixed-support contract's own identity/
    provenance facts, copied only from an already-
    :func:`validate_fixed_support_contract`-validated payload (RD1-C4
    review Finding 6) -- never inferred from a path name or reconstructed
    downstream. C5/C6 can consume this directly instead of reopening the
    contract file."""

    schema_name: str
    schema_version: int
    contract_id: str
    checksum_sha256: str
    seq_length_floor: int
    period: str
    date_start: str
    date_end: str
    target_variable: str
    lead_hours: int
    screening_basin_ids_sha256: str
    source_gap_policy_identity: str
    package_manifest_sha256: str
    package_file_checksums_sha256: str
    package_run_provenance_sha256: str
    development_split_sha256: str
    spatial_holdout_split_sha256: str


def build_support_contract_provenance(contract: Mapping) -> SupportContractProvenance:
    """Builds an immutable :class:`SupportContractProvenance` from a
    contract mapping. Re-validates ``contract`` against the current schema
    first (:func:`validate_fixed_support_contract`) -- an unvalidated or
    malformed contract must fail closed here rather than silently
    propagate a partial/fabricated provenance record."""
    validated = validate_fixed_support_contract(contract)
    return SupportContractProvenance(
        schema_name=validated["schema_name"],
        schema_version=validated["schema_version"],
        contract_id=validated["contract_id"],
        checksum_sha256=validated["checksum_sha256"],
        seq_length_floor=validated["seq_length_floor"],
        period=validated["period"],
        date_start=validated["date_start"],
        date_end=validated["date_end"],
        target_variable=validated["target_variable"],
        lead_hours=validated["lead_hours"],
        screening_basin_ids_sha256=validated["screening_basin_ids_sha256"],
        source_gap_policy_identity=validated["source_gap_policy_identity"],
        package_manifest_sha256=validated["package_manifest_sha256"],
        package_file_checksums_sha256=validated["package_file_checksums_sha256"],
        package_run_provenance_sha256=validated["package_run_provenance_sha256"],
        development_split_sha256=validated["development_split_sha256"],
        spatial_holdout_split_sha256=validated["spatial_holdout_split_sha256"],
    )


def load_fixed_support_contract(path, *, expected_contract_id: str = OBJECTIVE_ID_V2) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FixedSupportContractError(f"fixed-support contract not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return validate_fixed_support_contract(data, expected_contract_id=expected_contract_id)


def _frozen_array_copy(value) -> np.ndarray:
    """Return an independent, read-only copy of ``value``.

    Never aliases (shares memory with) a caller-owned array, and never
    mutates the writeability of the caller's own original array -- only the
    returned copy is marked non-writeable (RD1-C4 review: defensive
    immutability of public scientific results)."""
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AdmittedSeries:
    """One basin's frozen-support admitted raw-space series, in the fixed-
    support contract's own ``per_basin_support[basin_id]`` timestamp order
    (not run/positional order). ``date``, ``obs_m3s``, and ``sim_m3s`` are
    one-dimensional and positionally aligned -- ``date[i]``/``obs_m3s[i]``/
    ``sim_m3s[i]`` describe the same admitted timestamp. ``obs_m3s`` is
    always finite (fail-closed elsewhere); ``sim_m3s`` may contain
    non-finite values, which remain represented by the corresponding
    metric row's ``n_sim_nonfinite_at_admitted`` -- never silently dropped.
    This is the RD1-C4 producer-side result contract; RD1-C4-B (Q98
    analysis/rendering) is the intended consumer.
    """

    basin_id: str
    date: np.ndarray
    obs_m3s: np.ndarray
    sim_m3s: np.ndarray

    def __post_init__(self) -> None:
        # Defensively protect public scientific results: these arrays are
        # returned to callers as part of an already-qualified, immutable
        # record -- store an independent, read-only copy of each (never a
        # view/alias of the caller-supplied array) so neither the caller's
        # own original array nor its writeability is affected, while the
        # stored copy itself can never be silently mutated after
        # construction (RD1-C4 review follow-on: the prior `np.asarray` +
        # `setflags` here aliased and froze the caller's own array whenever
        # it was already an ndarray of the right dtype).
        for name in ("date", "obs_m3s", "sim_m3s"):
            object.__setattr__(self, name, _frozen_array_copy(getattr(self, name)))


@dataclass(frozen=True)
class CanonicalPackageObservedSeries:
    """One basin's canonical raw-space observed series, read directly from
    the frozen package NetCDF's own ``qobs_m3s`` and lead-shift-aligned to
    the fixed-support contract's ``per_basin_support`` timestamps (RD1-C4
    reproducibility correction, see ``docs/decision_log.md``). This is the
    sole source of formal RD1-C4 Q98 evaluation targets/facts; per-run
    ``validation_results.p`` observations remain provenance/consistency
    audit inputs only and never determine this series or any Q98 fact.

    ``date``/``obs_m3s`` are one-dimensional, positionally aligned, and in
    the same order as the contract's own ``per_basin_support[basin_id]``.
    ``obs_m3s`` is always fully finite (fail-closed at construction).
    """

    basin_id: str
    date: np.ndarray
    obs_m3s: np.ndarray

    def __post_init__(self) -> None:
        for name in ("date", "obs_m3s"):
            object.__setattr__(self, name, _frozen_array_copy(getattr(self, name)))


#: Temporal coordinate name of a NeuralHydrology ``*_results.p`` per-basin
#: xarray dataset. This is NeuralHydrology's own convention for run results
#: and is deliberately NOT the frozen package's coordinate name -- the
#: package's authoritative coordinate comes from
#: ``QualifiedPackageIdentity.netcdf_time_coordinate`` (RD1-C4-D1 finding A3),
#: which resolves it through the package's own declared, recognized schema.
_RUN_RESULTS_DATE_COORDINATE = "date"

#: NeuralHydrology's own run-results layout carries a size-1 ``time_step``
#: axis alongside ``date`` (its per-step lookback bookkeeping dimension,
#: degenerate for the single-lead-time case this evaluator handles).
#: ``_require_exact_one_dimensional_variable`` accepts exactly this one
#: named, singleton extra dimension -- see RD1-C4-D1 benchmark job 46175818.
_RUN_RESULTS_SINGLETON_STEP_DIMENSION = "time_step"


@dataclass(frozen=True)
class _FixtureOnlyUnqualifiedPackage:
    """Test-fixture-only stand-in for a :class:`QualifiedPackageIdentity`.

    RD1-C4-D1 Correction Pass A, finding A2: package qualification is
    mandatory on every production path, so ``package_identity`` is a
    REQUIRED argument of :func:`derive_canonical_package_observed_series`
    with no default. Many in-repo unit fixtures legitimately build a bare
    two-file package directory with no manifests at all and only want to
    exercise the alignment algebra; they pass
    :func:`fixture_only_unqualified_package` instead.

    This type provides exactly one thing -- the temporal coordinate name to
    bind variable dimensions against -- and provides NO checksum, contract,
    root or basin proof. It is module-private and no module under
    ``src/baseline/`` other than this one may reference it; a guard test
    enforces that, which is what makes the convenience path unreachable
    from production.
    """

    netcdf_time_coordinate: str = _RUN_RESULTS_DATE_COORDINATE


def fixture_only_unqualified_package(
    *, netcdf_time_coordinate: str = _RUN_RESULTS_DATE_COORDINATE
) -> "_FixtureOnlyUnqualifiedPackage":
    """Build the test-fixture-only unqualified package marker.

    Never call this from production code: it deliberately proves nothing.
    See :class:`_FixtureOnlyUnqualifiedPackage`.
    """
    return _FixtureOnlyUnqualifiedPackage(netcdf_time_coordinate=netcdf_time_coordinate)


def _require_exact_one_dimensional_variable(
    dataset,
    variable_name: str,
    *,
    coordinate_name: str,
    context: str,
    allow_singleton_time_step: bool = False,
):
    """Return ``dataset[variable_name]``'s values, proven to be exactly a
    one-dimensional series along ``coordinate_name``.

    RD1-C4-D1 Correction Pass A, finding A3: the previous code flattened
    every package/run variable with ``.reshape(-1)`` and then paired it
    positionally with a coordinate vector. That silently accepted a variable
    laid out along an unrelated dimension of the same length, a transposed
    ``(time, x)`` variable, and any multidimensional variable whose element
    count happened to match -- producing a confidently wrong alignment
    rather than an error.

    This helper instead proves the layout before any value is read:

    * ``coordinate_name`` exists on the dataset and is itself a plain
      one-dimensional index coordinate (``dims == (coordinate_name,)``);
    * ``variable_name`` exists as a data variable;
    * its dims are EXACTLY ``(coordinate_name,)`` -- not a superset, not a
      permutation, not a same-length sibling dimension --

    with exactly one named exception, gated by ``allow_singleton_time_step``
    (RD1-C4-D1 benchmark job 46175818): real NeuralHydrology validation-results
    variables carry the variable as ``(coordinate_name, 'time_step')`` with
    ``time_step`` of size exactly 1. That axis is NeuralHydrology's own
    per-step bookkeeping dimension, degenerate here because this evaluator
    only ever reads a single lead time; it is proven singleton, and proven to
    have exactly one value per ``coordinate_name`` entry, before its sole
    element is explicitly selected. This is not generic squeezing: no other
    extra dimension name, no ``time_step`` of size other than 1, and no
    temporal-length mismatch, is accepted -- and the exception is only ever
    offered to callers that opt in, never to the frozen package's own
    authoritative reader.

    Nothing is flattened, reshaped, transposed or broadcast: a variable that
    is not already the expected one-dimensional series, or (when
    ``allow_singleton_time_step`` is set) exactly that one named
    singleton-``time_step`` variant, is rejected.
    """
    coords = getattr(dataset, "coords", {})
    if coordinate_name not in coords:
        raise FixedSupportContractError(
            f"{context}: dataset has no {coordinate_name!r} coordinate (available coordinates: "
            f"{sorted(map(str, coords))}) -- refusing to align against an unproven temporal coordinate"
        )
    coordinate = coords[coordinate_name]
    if tuple(coordinate.dims) != (coordinate_name,):
        raise FixedSupportContractError(
            f"{context}: coordinate {coordinate_name!r} has dims {tuple(map(str, coordinate.dims))}; expected "
            f"the plain one-dimensional index ({coordinate_name!r},)"
        )
    if variable_name not in dataset.data_vars:
        raise FixedSupportContractError(
            f"{context}: missing data variable {variable_name!r} (available: "
            f"{sorted(map(str, dataset.data_vars))})"
        )
    variable = dataset[variable_name]
    dims = tuple(str(d) for d in variable.dims)
    is_singleton_step_layout = allow_singleton_time_step and (
        dims == (coordinate_name, _RUN_RESULTS_SINGLETON_STEP_DIMENSION)
    )
    if dims != (coordinate_name,) and not is_singleton_step_layout:
        supported = (
            f"({coordinate_name!r},) and ({coordinate_name!r}, {_RUN_RESULTS_SINGLETON_STEP_DIMENSION!r}) with "
            f"a singleton {_RUN_RESULTS_SINGLETON_STEP_DIMENSION!r}"
            if allow_singleton_time_step
            else f"({coordinate_name!r},)"
        )
        raise FixedSupportContractError(
            f"{context}: data variable {variable_name!r} has dims {dims} but the only supported layout"
            f"{'s are' if allow_singleton_time_step else ' is'} {supported} -- refusing to flatten an "
            "unexpected, transposed or multidimensional variable into a positional series"
        )
    coordinate_length = np.asarray(coordinate.values).shape[0]
    values = np.asarray(variable.values)
    if is_singleton_step_layout:
        if values.ndim != 2 or values.shape[1] != 1:
            raise FixedSupportContractError(
                f"{context}: data variable {variable_name!r} declares dims {dims} but its values have shape "
                f"{values.shape}, not a singleton {_RUN_RESULTS_SINGLETON_STEP_DIMENSION!r} axis -- refusing "
                "to select an unproven element"
            )
        if values.shape[0] != coordinate_length:
            raise FixedSupportContractError(
                f"{context}: data variable {variable_name!r} has {values.shape[0]} {coordinate_name!r} rows "
                f"but the {coordinate_name!r} coordinate has {coordinate_length} -- refusing to select an "
                "unproven element against a mismatched temporal length"
            )
        values = values[:, 0]
    if values.ndim != 1:
        raise FixedSupportContractError(
            f"{context}: data variable {variable_name!r} declares dims {dims} but its values are "
            f"{values.ndim}-dimensional"
        )
    if values.shape[0] != coordinate_length:
        raise FixedSupportContractError(
            f"{context}: data variable {variable_name!r} has {values.shape[0]} values but the "
            f"{coordinate_name!r} coordinate has {coordinate_length} -- refusing to pair a value "
            "with an unproven date"
        )
    return values


def _require_qualified_package_identity(
    package_identity: QualifiedPackageIdentity,
    *,
    package_root,
    basin_id: str,
    contract: Mapping,
) -> None:
    """Fail closed unless ``package_identity`` actually qualifies THIS read.

    Three independent facts must hold, and each is proven, never assumed:
    the identity was qualified against this same fixed-support contract
    (checksum identity, so a stale identity from an older contract cannot
    be reused); the root about to be opened is the root that was qualified;
    and this basin's NetCDF still hashes to the package's own recorded
    checksum for that file. Raises :class:`FixedSupportContractError` so
    contract-bound callers see one exception family, chaining the
    underlying :class:`~.package_identity_qualification.PackageIdentityError`
    for the exact per-file evidence.
    """
    if not isinstance(package_identity, QualifiedPackageIdentity):
        raise FixedSupportContractError(
            f"basin {basin_id!r}: package_identity must be a QualifiedPackageIdentity, got "
            f"{type(package_identity).__name__} -- refusing to accept an unqualified identity claim"
        )
    if package_identity.contract_checksum_sha256 != contract["checksum_sha256"]:
        raise FixedSupportContractError(
            f"basin {basin_id!r}: package identity was qualified against contract checksum "
            f"{package_identity.contract_checksum_sha256} but this evaluation uses "
            f"{contract['checksum_sha256']} -- package/contract identity contradiction"
        )
    if package_identity.contract_id != contract["contract_id"]:
        raise FixedSupportContractError(
            f"basin {basin_id!r}: package identity was qualified against contract "
            f"{package_identity.contract_id!r} but this evaluation uses {contract['contract_id']!r}"
        )
    try:
        verify_basin_time_series_file(package_identity, basin_id, package_root=package_root)
    except PackageIdentityError as exc:
        raise FixedSupportContractError(
            f"basin {basin_id!r}: package identity verification failed before any value was read: {exc}"
        ) from exc


def derive_canonical_package_observed_series(
    *,
    package_root,
    basin_id: str,
    contract: dict,
    package_identity: "QualifiedPackageIdentity",
) -> CanonicalPackageObservedSeries:
    """Derives basin ``basin_id``'s canonical raw-space observed series from
    the frozen package NetCDF alone (no run/trial data), for the contract's
    frozen ``per_basin_support`` timestamps.

    Reuses the module's own already-qualified, documented algebraic lead-
    shift identity (see ``nh_raw_space_evaluation.py``'s module docstring)::

        qobs_mm_per_h_lead{L}[t] == discharge_m3s_to_runoff_mm_per_h(qobs_m3s[t + L], area_km2)

    i.e. the package's raw ``qobs_m3s`` value corresponding to a contract
    support timestamp ``t`` lives at the package's own date coordinate
    ``t + lead_hours``, not at ``t`` itself. No new metric math or unit
    conversion is introduced here: ``qobs_m3s`` is already raw m^3/s, so no
    area is needed to read it.

    Raises :class:`FixedSupportContractError` (never silently drops/
    realigns) if: the contract's own support timestamps for this basin
    contain a duplicate; the package NetCDF's own date coordinate contains
    a duplicate; any lead-shifted lookup date is absent from the package's
    date coordinate (lead-alignment failure); or any resulting aligned
    ``qobs_m3s`` value is non-finite (finite-mask disagreement).

    ``package_identity`` (RD1-C4-D1 findings A1/A2/A3) is REQUIRED and has
    no default. It binds this read to a
    :class:`~.package_identity_qualification.QualifiedPackageIdentity` that
    was proven, once per task, to be the package ``contract`` was built
    against. This function refuses to read values unless the identity was
    qualified against this same contract checksum and this same package
    root, and unless this basin's NetCDF still matches the package's own
    recorded checksum for that file -- i.e. identity is proven BEFORE any
    value is read. The identity is also the authority for the package's
    temporal coordinate name: the coordinate is never assumed, it is taken
    from the package's own declared, recognized NetCDF schema, and
    ``qobs_m3s`` must be laid out along exactly that coordinate.

    The only other accepted value is the module-private test-fixture marker
    from :func:`fixture_only_unqualified_package`, which supplies the
    coordinate name and nothing else. It exists for in-repo unit fixtures
    that build a bare package directory with no manifests; it is
    unreachable from production code and a guard test enforces that.
    """
    import xarray as xr

    validate_fixed_support_contract(contract)
    lead_hours = contract["lead_hours"]
    if isinstance(package_identity, _FixtureOnlyUnqualifiedPackage):
        coordinate_name = package_identity.netcdf_time_coordinate
    else:
        _require_qualified_package_identity(
            package_identity, package_root=package_root, basin_id=basin_id, contract=contract
        )
        coordinate_name = package_identity.netcdf_time_coordinate

    support_dates = _deserialize_date_array(contract["per_basin_support"][basin_id], contract["date_dtype"])
    support_dates = _canonicalize_timestamps_for_identity(
        support_dates,
        date_dtype=contract["date_dtype"],
        context=f"basin {basin_id!r}: frozen support timestamps",
    )
    if len(np.unique(support_dates)) != len(support_dates):
        raise FixedSupportContractError(f"basin {basin_id!r}: frozen support contains duplicate timestamps")

    nc_path = basin_netcdf_path(package_root, basin_id)
    with xr.open_dataset(nc_path) as ds:
        # Exact dimension/coordinate identity BEFORE any value is read
        # (RD1-C4-D1 finding A3): 'qobs_m3s' must be a plain series along
        # the package's own authoritative temporal coordinate. Nothing is
        # flattened, so an unrelated same-length dimension, a transposed
        # layout, or an unexpected multidimensional variable is rejected
        # rather than positionally mispaired.
        package_qobs_m3s = _require_exact_one_dimensional_variable(
            ds,
            "qobs_m3s",
            coordinate_name=coordinate_name,
            context=f"{nc_path}",
        )
        package_date_values = np.asarray(ds.coords[coordinate_name].values)
        if package_date_values.shape != package_qobs_m3s.shape:
            raise FixedSupportContractError(
                f"{nc_path}: coordinate {coordinate_name!r} has length {package_date_values.shape} but "
                f"'qobs_m3s' has length {package_qobs_m3s.shape}"
            )

    package_date_values = _canonicalize_timestamps_for_identity(
        package_date_values,
        date_dtype=contract["date_dtype"],
        context=f"basin {basin_id!r}: package {coordinate_name!r} coordinate",
    )
    if len(np.unique(package_date_values)) != len(package_date_values):
        raise FixedSupportContractError(
            f"basin {basin_id!r}: package {coordinate_name!r} coordinate contains duplicates"
        )

    if contract["date_dtype"] == "datetime64":
        lookup_dates = support_dates + np.timedelta64(lead_hours, "h")
    else:
        # int64 contracts (fast synthetic fixtures only) represent the date
        # coordinate as a plain hour-index, so the lead-shift is ordinary
        # integer addition -- the same instant-shift semantics, expressed in
        # the contract's own unit convention (see
        # ``_canonicalize_timestamps_for_identity``).
        lookup_dates = support_dates + lead_hours
    position_by_date = {date: idx for idx, date in enumerate(package_date_values)}
    try:
        positions = np.array([position_by_date[date] for date in lookup_dates])
    except KeyError as exc:
        raise FixedSupportContractError(
            f"basin {basin_id!r}: lead-shifted package lookup failed for {exc} (lead_hours={lead_hours}) -- "
            "package/contract support timestamp identity contradiction"
        ) from exc

    canonical_obs_m3s = package_qobs_m3s[positions]
    if not np.isfinite(canonical_obs_m3s).all():
        raise FixedSupportContractError(
            f"basin {basin_id!r}: lead-shifted package qobs_m3s contains non-finite values at admitted "
            "support timestamps"
        )

    return CanonicalPackageObservedSeries(basin_id=basin_id, date=support_dates, obs_m3s=canonical_obs_m3s)


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
    return_admitted_series: bool = False,
    package_identity: Optional[QualifiedPackageIdentity] = None,
    authenticated_period_results: Optional[AuthenticatedPeriodResults] = None,
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

    ``return_admitted_series``, if set, additionally returns
    ``result["admitted_series_by_basin"]``: a ``{basin_id: AdmittedSeries}``
    mapping for exactly the successfully evaluated basins (same keys as the
    ``per_basin`` metric rows), built from the same extraction/conversion
    pass as those metric rows -- not a second independent reconstruction.
    Defaults to ``False`` so the legacy result shape is unchanged.

    ``package_identity`` (RD1-C4-D1 finding A1), when supplied, binds every
    package NetCDF this evaluation opens to a
    :class:`~.package_identity_qualification.QualifiedPackageIdentity`
    proven once per task against ``contract`` -- see
    :func:`_require_qualified_package_identity`. Omitted, the historical
    unqualified behaviour is unchanged.

    ``authenticated_period_results`` (RD1-C4-D1 finding A3), when supplied,
    is used instead of loading the run's results pickle again. It must be an
    :class:`~.authenticated_period_results.AuthenticatedPeriodResults` --
    which can only be produced by an actual authenticated load of an actual
    file -- and it must have been authenticated for exactly this
    ``run_dir``/``contract["period"]``/``epoch``. It replaces the previous
    plain-``Mapping`` seam, which any fabricated ``dict`` satisfied.

    It exists so a caller that must evaluate one run's basins across several
    passes (for example the D1 observation diagnostic, which needs per-basin
    typed outcomes rather than one all-or-nothing call) loads and hashes
    that run's ~84.5 MB ``validation_results.p`` once rather than once per
    basin. When it is omitted this function performs that authenticated load
    itself, so the simulations it evaluates are bound to a named pickle and
    a recorded digest on every path. The observed source facts are returned
    in ``result["evaluation_source"]``.
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

    if authenticated_period_results is None:
        try:
            authenticated_period_results = load_authenticated_period_results(
                run_dir=run_dir, period=contract["period"], epoch=epoch
            )
        except AuthenticatedPeriodResultsError as exc:
            raise FixedSupportContractError(str(exc)) from exc
    else:
        if not isinstance(authenticated_period_results, AuthenticatedPeriodResults):
            raise FixedSupportContractError(
                "authenticated_period_results must be an AuthenticatedPeriodResults produced by "
                "load_authenticated_period_results(), got "
                f"{type(authenticated_period_results).__name__} -- refusing to evaluate simulations whose "
                "source pickle cannot be proven"
            )
        try:
            authenticated_period_results.require_bound_to(
                run_dir=run_dir, period=contract["period"], epoch=epoch
            )
        except AuthenticatedPeriodResultsError as exc:
            raise FixedSupportContractError(str(exc)) from exc
    period_results = authenticated_period_results

    per_basin = []
    excluded = []
    admitted_series_by_basin: dict = {}
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

        # Exact dimension/coordinate identity BEFORE any value is read
        # (RD1-C4-D1 finding A3). The run-results coordinate is
        # NeuralHydrology's own 'date', which is a different authority from
        # the frozen package's declared coordinate -- both are bound, neither
        # is assumed to be the other.
        obs_mm_per_h = _require_exact_one_dimensional_variable(
            xr_ds,
            obs_key,
            coordinate_name=_RUN_RESULTS_DATE_COORDINATE,
            context=f"basin {basin_id!r}: run results ({authenticated_period_results.results_path})",
            allow_singleton_time_step=True,
        )
        sim_mm_per_h = _require_exact_one_dimensional_variable(
            xr_ds,
            sim_key,
            coordinate_name=_RUN_RESULTS_DATE_COORDINATE,
            context=f"basin {basin_id!r}: run results ({authenticated_period_results.results_path})",
            allow_singleton_time_step=True,
        )
        run_date_values = np.asarray(xr_ds.coords[_RUN_RESULTS_DATE_COORDINATE].values)
        if not (run_date_values.shape == obs_mm_per_h.shape == sim_mm_per_h.shape):
            raise FixedSupportContractError(
                f"basin {basin_id!r}: run results coordinate/observation/simulation lengths disagree "
                f"({run_date_values.shape}, {obs_mm_per_h.shape}, {sim_mm_per_h.shape})"
            )
        support_dates = _deserialize_date_array(contract["per_basin_support"][basin_id], contract["date_dtype"])
        run_date_values = _canonicalize_timestamps_for_identity(
            run_date_values,
            date_dtype=contract["date_dtype"],
            context=f"basin {basin_id!r}: run date coordinate",
        )
        support_dates = _canonicalize_timestamps_for_identity(
            support_dates,
            date_dtype=contract["date_dtype"],
            context=f"basin {basin_id!r}: frozen support timestamps",
        )
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

        obs_support = np.where(support_mask, obs_mm_per_h, np.nan)
        if not np.isfinite(obs_support[support_mask]).all():
            raise FixedSupportContractError(
                f"basin {basin_id!r}: frozen admitted timestamps are not naturally admitted observations"
            )

        nc_path = basin_netcdf_path(package_root, basin_id)
        if package_identity is not None:
            _require_qualified_package_identity(
                package_identity, package_root=package_root, basin_id=basin_id, contract=contract
            )
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
            return_admitted_arrays=return_admitted_series,
        )
        if return_admitted_series:
            # ``evaluate_basin_raw_space``'s admitted arrays are ordered by
            # this run's own date coordinate (positional), not the frozen
            # contract's ``per_basin_support`` order -- reindex by exact
            # timestamp identity to the contract's canonical order.
            run_dates_admitted = run_date_values[support_mask]
            position_by_date = {date: idx for idx, date in enumerate(run_dates_admitted)}
            try:
                reorder = np.array([position_by_date[date] for date in support_dates])
            except KeyError as exc:
                raise FixedSupportContractError(
                    f"basin {basin_id!r}: admitted timestamp identity lookup failed for {exc} after "
                    "unit canonicalization -- run/contract support timestamp identity contradiction"
                ) from exc
            admitted_obs_m3s = basin_metrics.pop("_admitted_obs_m3s")[reorder]
            admitted_sim_m3s = basin_metrics.pop("_admitted_sim_m3s")[reorder]
            admitted_series_by_basin[basin_id] = AdmittedSeries(
                basin_id=basin_id,
                date=support_dates,
                obs_m3s=admitted_obs_m3s,
                sim_m3s=admitted_sim_m3s,
            )
        basin_metrics["freq"] = "1h"
        basin_metrics["n_fixed_support_eligible"] = len(support_dates)
        per_basin.append(basin_metrics)

    result = {
        "objective_scope": "fixed_support",
        "contract_id": contract["contract_id"],
        "contract_checksum_sha256": contract["checksum_sha256"],
        # The authenticated pickle these simulations actually came from
        # (RD1-C4-D1 finding A3) -- path, digest, run, period and epoch.
        "evaluation_source": authenticated_period_results.source_fields(),
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
    if return_admitted_series:
        result["admitted_series_by_basin"] = admitted_series_by_basin
    if require_full_screening_population:
        _require_complete_production_fixed_support_population(result, required_basin_ids=contract["basin_ids"])
    return result


def _require_complete_production_fixed_support_population(
    result: Mapping,
    *,
    required_basin_ids: Sequence[str],
) -> None:
    """Reject a v2 objective result unless all 400 frozen basins contributed.

    This is deliberately applied only to the explicitly production-shaped
    ``require_full_screening_population`` path. Small synthetic fixtures may
    still exercise the generic evaluator, but no partial population (including
    a finite aggregate over 399 basins) can enter the v2 objective route.
    """
    required = list(required_basin_ids)
    expected = set(required)
    if len(required) != 400 or len(expected) != 400:
        raise FixedSupportContractError("production fixed-support contract must contain exactly 400 unique screening basins")

    requested = result.get("n_basins_requested")
    evaluated = result.get("n_basins_evaluated")
    excluded = result.get("n_basins_excluded")
    per_basin = result.get("per_basin")
    if (requested, evaluated, excluded) != (400, 400, 0) or not isinstance(per_basin, list):
        raise FixedSupportContractError(
            "production fixed-support evaluation requires 400 requested, 400 evaluated, zero excluded basins, "
            "and a complete per-basin receipt"
        )

    evaluated_ids = [row.get("basin_id") if isinstance(row, Mapping) else None for row in per_basin]
    if len(evaluated_ids) != 400 or len(set(evaluated_ids)) != 400 or set(evaluated_ids) != expected:
        raise FixedSupportContractError(
            "production fixed-support evaluated basin IDs must be exactly the 400 unique frozen support-contract basins"
        )
    for row in per_basin:
        nse = row.get("nse")
        nonfinite_sim = row.get("n_sim_nonfinite_at_admitted")
        if isinstance(nse, bool) or not isinstance(nse, (int, float)) or not np.isfinite(nse):
            raise FixedSupportContractError("every production fixed-support basin must contribute a finite NSE")
        if nonfinite_sim != 0:
            raise FixedSupportContractError("production fixed-support basin has non-finite simulated values at admitted timestamps")

    aggregate = result.get("aggregate")
    nse_summary = aggregate.get("metrics", {}).get("nse", {}) if isinstance(aggregate, Mapping) else {}
    if not isinstance(aggregate, Mapping) or aggregate.get("n_basins") != 400 or nse_summary.get("n_finite_basins") != 400:
        raise FixedSupportContractError("production fixed-support aggregate does not represent all 400 finite basin NSE values")


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
