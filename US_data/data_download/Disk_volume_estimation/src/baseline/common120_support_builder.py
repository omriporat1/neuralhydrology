"""Authoritative, candidate-independent Common-120 support construction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from .fixed_support_contract_v2 import build_fixed_support_contract, validate_fixed_support_contract
from .nh_seed_evaluation import basin_netcdf_path
from .gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT, load_gap_timestamps_json
from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership
from .pilot_lead06_config import load_screening_basin_ids
from .policy import load_stage1_baseline_policy
from .policy_v2_six_axis import load_stage1_baseline_policy_v2_six_axis
from .sweep_v1_production_adapter import (
    DEVELOPMENT_SPLIT_SHA256,
    PACKAGE_FILE_CHECKSUMS_SHA256,
    PACKAGE_MANIFEST_SHA256,
    PACKAGE_RUN_PROVENANCE_SHA256,
    SPATIAL_HOLDOUT_SPLIT_SHA256,
    PreparationPaths,
    SweepV1PreparationError,
    _sha256,
    _verify_artifact_identities,
)
from . import sweep_v1_campaign as sweep
from .sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2, SEQ_LENGTH_MAX
from .validity_mask import bad_hour_mask_from_timestamps, compute_history_valid, _validate_hourly_index

__all__ = [
    "Common120SupportError",
    "Common120BuildResult",
    "build_common120_support",
    "build_common120_support_for_population",
    "build_common120_support_for_development_population",
]


class Common120SupportError(ValueError):
    pass


@dataclass(frozen=True)
class Common120BuildResult:
    contract: dict
    accounting: dict


def _dates_and_target(path: Path, target: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not path.is_file():
        raise Common120SupportError(f"missing basin NetCDF: {path}")
    with xr.open_dataset(path) as data:
        if "date" not in data.coords or target not in data:
            raise Common120SupportError(f"{path}: required date/target is missing")
        values = np.asarray(data[target].values)
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(data["date"].values)))
    if values.ndim != 1 or values.shape != (len(dates),):
        raise Common120SupportError(f"{path}: target must be one-dimensional and date-aligned")
    try:
        _validate_hourly_index(dates)
    except Exception as exc:
        raise Common120SupportError(f"{path}: invalid package-native timeline: {exc}") from exc
    return dates, values


def _common120_admitted_maps(*, basins, package_root, target, lead, start, end, gap_path,
                             raise_on_first_zero):
    """Apply the frozen Common-120 admitted-timestamp predicate to every basin
    of ``basins`` independently, against the package-native authoritative
    timeline.

    This is the single shared implementation of the scientific membership
    math -- ``history_valid_120(t) AND validation_start <= t <= validation_end
    AND t + lead <= validation_end AND finite(qobs[b, t])`` -- reused verbatim
    by both the frozen 400-basin screening builder and the additive
    full-development-population audit builder.  It performs no parallel math.

    The packaged gap-timestamp JSON is *parsed* here (``gap_path``), after the
    first basin's package-native timeline has been read -- this preserves the
    historical screening validation order (gap-artifact checksum -> first-basin
    read -> gap-JSON parse -> history mask), so that when both the first basin
    input and the gap input are invalid the missing/invalid-basin error is
    still the one that fires.  The gap-artifact *checksum* is verified by the
    caller before this function is entered.

    Returns
    ``(per_dates, per_admitted, counts, zero_support, n_global_valid, gaps)``.
    ``zero_support`` lists (in encounter order) every basin whose admitted set
    is empty; when ``raise_on_first_zero`` is true the first such basin raises
    immediately instead (preserving the historical single-basin message)."""
    if not basins:
        raise Common120SupportError("no basins supplied for Common-120 support construction")
    first_dates, _ = _dates_and_target(basin_netcdf_path(package_root, basins[0]), target)
    gaps = load_gap_timestamps_json(gap_path)
    try:
        bad = bad_hour_mask_from_timestamps(first_dates, gaps, on_out_of_range="ignore")
        history = compute_history_valid(first_dates, bad, SEQ_LENGTH_MAX)
    except Exception as exc:
        raise Common120SupportError(f"invalid packaged gap mask: {exc}") from exc
    global_valid = history & (first_dates >= start) & (first_dates <= end) & (first_dates + pd.Timedelta(hours=lead) <= end)
    per_dates, per_admitted, counts, zero_support = {}, {}, {}, []
    for basin in basins:
        dates, qobs = _dates_and_target(basin_netcdf_path(package_root, basin), target)
        if not dates.equals(first_dates):
            raise Common120SupportError(f"{basin}: timeline differs from package-native authoritative timeline")
        admitted = global_valid & np.isfinite(qobs)
        n_admitted = int(admitted.sum())
        if n_admitted == 0:
            if raise_on_first_zero:
                raise Common120SupportError(f"{basin}: zero Common-120 support")
            zero_support.append(basin)
        per_dates[basin], per_admitted[basin] = dates.to_numpy(), admitted
        counts[basin] = n_admitted
    return per_dates, per_admitted, counts, zero_support, int(global_valid.sum()), gaps


_AUDIT_PACKAGE_PAYLOAD_ARTIFACTS = (
    ("package_manifest_sha256", ("manifests", "package_manifest.json"), PACKAGE_MANIFEST_SHA256),
    ("package_file_checksums_sha256", ("manifests", "file_checksums.csv"), PACKAGE_FILE_CHECKSUMS_SHA256),
    ("package_run_provenance_sha256", ("run_provenance.json",), PACKAGE_RUN_PROVENANCE_SHA256),
)
_AUDIT_SPLIT_ARTIFACTS = (
    ("development_split_sha256", "development_train.txt", DEVELOPMENT_SPLIT_SHA256),
    ("spatial_holdout_split_sha256", "spatial_holdout_nonca.txt", SPATIAL_HOLDOUT_SPLIT_SHA256),
)


def _verify_audit_package_payload_identities(package_root) -> dict:
    """Byte-identity of the three authoritative package payload artifacts
    (manifest JSON, file-checksum CSV, run-provenance JSON), verified against
    the exact same authoritative constants the frozen v2 fixed-support
    preparation path pins.  These payloads are emitted by the package builder
    and are not line-ending sensitive, so a raw-bytes SHA-256 is the correct
    identity check -- identical to
    :func:`sweep_v1_production_adapter._verify_artifact_identities` for these
    three files.  No package identity check is weakened."""
    package_root = Path(package_root)
    verified: dict = {}
    for name, parts, expected in _AUDIT_PACKAGE_PAYLOAD_ARTIFACTS:
        try:
            actual = _sha256(package_root.joinpath(*parts))
        except SweepV1PreparationError as exc:
            raise Common120SupportError(str(exc)) from exc
        if actual != expected:
            raise Common120SupportError(
                f"audit package identity mismatch for {name}: {actual} != {expected}"
            )
        verified[name] = actual
    return verified


def _verify_audit_split_identities(splits_dir) -> dict:
    """**Line-ending-independent** identity of the development and
    spatial-holdout split membership.

    The existing production adapter verifies the split *files* by raw
    working-tree bytes -- LF on the committed blob, but CRLF once checked out
    on a ``core.autocrlf`` platform: scientifically identical membership,
    different bytes.  This audit-specific check instead parses the canonical
    STAID set (:func:`~src.baseline.splits.load_eligible_basins`, which strips
    line endings, sorts and de-duplicates) and compares its canonical
    membership SHA-256 (:func:`~src.baseline.devpop_common120_audit_contract.canonical_membership_sha256`)
    to the authoritative constants -- the LF-blob hashes -- so it succeeds
    identically for an LF or a CRLF checkout and still fails closed on any real
    membership change.  It reads only the committed split ID lists; it never
    touches a sealed observation, a prediction, or the scientific package."""
    from .splits import load_eligible_basins
    from .devpop_common120_audit_contract import canonical_membership_sha256

    splits_dir = Path(splits_dir)
    verified: dict = {}
    for name, filename, expected in _AUDIT_SPLIT_ARTIFACTS:
        path = splits_dir / filename
        try:
            ids = load_eligible_basins(path)
        except Exception as exc:
            raise Common120SupportError(f"could not read {name} list {path}: {exc}") from exc
        actual = canonical_membership_sha256(ids)
        if actual != expected:
            label = "development" if name.startswith("development") else "spatial-holdout"
            raise Common120SupportError(
                f"audit {label} split identity mismatch for {name}: line-ending-independent canonical "
                f"membership hash {actual} != authoritative {expected} -- split membership changed"
            )
        verified[name] = actual
    return verified


def _verify_audit_artifact_identities(paths: PreparationPaths) -> dict:
    """Audit-specific replacement for
    :func:`sweep_v1_production_adapter._verify_artifact_identities`, used only
    by the development-population Common-120 audit engine.

    Verifies the three package payload artifacts exactly as the production
    adapter does, but verifies the development and spatial-holdout split
    membership by parsed canonical STAID set (LF/CRLF-independent) rather than
    by platform-materialized raw bytes.  No package or split identity check is
    weakened and the existing production adapter is not modified.  Returns the
    same five ``*_sha256`` keys the generic verifier returns, so it is a
    drop-in for :func:`build_devpop_audit_contract`'s ``**identities``."""
    return {
        **_verify_audit_package_payload_identities(paths.package_root),
        **_verify_audit_split_identities(paths.splits_dir),
    }


def build_common120_support(*, package_root, splits_dir, screening_basin_ids_path,
                            baseline_policy_path, policy_overlay_path) -> Common120BuildResult:
    """Build, but never write, the frozen 400-basin issue-time support contract."""
    baseline = load_stage1_baseline_policy(baseline_policy_path)
    effective = load_stage1_baseline_policy_v2_six_axis(baseline_policy_path, policy_overlay_path)
    if effective["gap_policy"]["include_rtma_in_history_mask"] is not True:
        raise Common120SupportError("frozen production policy must include RTMA in the package gap mask")
    validation = baseline["temporal_split"]["validation"]
    # For hourly grids, +23h is the final point of an inclusive end date,
    # equivalent to NeuralHydrology's date + 1 day - 1 second convention.
    start, end = pd.Timestamp(validation["start"]), pd.Timestamp(validation["end"]) + pd.Timedelta(hours=23)
    target, lead = "qobs_mm_per_h_lead06", 6
    identities = _verify_artifact_identities(PreparationPaths(
        Path(baseline_policy_path), Path(package_root), Path(splits_dir), Path(screening_basin_ids_path)))
    manifest = read_package_manifest(package_root)
    scope = manifest.get("gap_product_scope")
    if scope != [MRMS_PRODUCT, RTMA_PRODUCT]:
        raise Common120SupportError(f"package gap_product_scope must be complete MRMS+RTMA, got {scope!r}")
    membership = validate_full_population_basin_membership(manifest, splits_dir)
    basins = load_screening_basin_ids(screening_basin_ids_path, development_basins=membership.development_basins,
                                     expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)
    if basins != sorted(set(basins)):
        raise Common120SupportError("screening basins must be exactly 400 sorted unique IDs")
    gap_path = Path(package_root) / "masks" / "gap_timestamps.json"
    declared_gap = (manifest.get("gap_timestamp_artifact") or {}).get("sha256")
    actual_gap = hashlib.sha256(gap_path.read_bytes()).hexdigest() if gap_path.is_file() else None
    if not isinstance(declared_gap, str) or actual_gap != declared_gap:
        raise Common120SupportError("package gap-timestamp artifact checksum contradicts package manifest")
    per_dates, per_admitted, counts, _zero, global_valid_count, gaps = _common120_admitted_maps(
        basins=basins, package_root=package_root, target=target, lead=lead,
        start=start, end=end, gap_path=gap_path, raise_on_first_zero=True,
    )
    contract = build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=lead, target_variable=target, period="validation",
        date_start=str(start.date()), date_end=str(end.date()), source_gap_policy_identity="stage1_policy_mrms_rtma_history_v001",
        screening_basin_ids_sha256=sweep.SCREENING_ARTIFACT_SHA256, per_basin_date=per_dates,
        per_basin_admitted=per_admitted, seq_length_floor=SEQ_LENGTH_MAX, **identities,
    )
    validate_fixed_support_contract(contract)
    values = np.array(list(counts.values()), dtype=int)
    accounting = {"n_basins": len(basins), "global_validation_issue_times": global_valid_count,
                  "total_retained": int(values.sum()), "per_basin_retained": counts,
                  "min_retained": int(values.min()), "max_retained": int(values.max()),
                  "median_retained": float(np.median(values)), "gap_count": len(gaps),
                  "package_identities": identities}
    return Common120BuildResult(contract=contract, accounting=accounting)


def build_common120_support_for_population(
    *, population, package_root, splits_dir, baseline_policy_path, policy_overlay_path,
) -> Common120BuildResult:
    """**Generic engine.**  Build, but never write, a Common-120 *audit*
    support contract (diagnostic identity
    ``common120_raw_space_nse_devpop_audit_v001``) over an *arbitrary*
    explicitly-identified :class:`~src.baseline.devpop_common120_audit_contract.ExpectedPopulationSpec`.

    This applies the identical frozen Common-120 predicate as
    :func:`build_common120_support` -- via the shared
    :func:`_common120_admitted_maps` -- to every basin of the supplied
    population instead of the frozen 400 screening basins.  It reuses the same
    gap-scope check, gap-checksum check, and package-native-timeline agreement
    check.  Package + split identity is verified through the audit-specific
    :func:`_verify_audit_artifact_identities`: the three package payload
    artifacts are byte-checked exactly as the production adapter does, but the
    development and spatial-holdout split membership is verified by parsed
    canonical STAID set (LF/CRLF-independent) rather than by
    platform-materialized raw bytes.  The existing production adapter and its
    frozen-screening consumer are untouched.

    It does **not** by itself assert the canonical 2,307-basin development
    identity: it verifies only that the supplied population equals the
    package-verified development membership for the given package.  The
    canonical entry point is
    :func:`build_common120_support_for_development_population`, which pins the
    real development identity before delegating here.  A generic / synthetic
    population therefore reaches this engine only through a test fixture whose
    monkeypatched membership matches it -- never through the canonical wrapper.

    It never touches the frozen screening artifact / identity JSON, never
    produces the v2 optimizer objective, and (in this milestone) is exercised
    only against synthetic fixtures -- it must not read the real scientific
    package or build the real 2,307-basin artifact.

    Every basin with zero Common-120 support is named explicitly; any such
    basin blocks the build."""
    from .devpop_common120_audit_contract import (
        DEVPOP_AUDIT_CONTRACT_ID,
        ExpectedPopulationSpec,
        build_devpop_audit_contract,
    )

    if not isinstance(population, ExpectedPopulationSpec):
        raise Common120SupportError("population must be a validated ExpectedPopulationSpec")
    if population.contract_id != DEVPOP_AUDIT_CONTRACT_ID:
        raise Common120SupportError("population must carry the diagnostic devpop-audit contract id")

    baseline = load_stage1_baseline_policy(baseline_policy_path)
    effective = load_stage1_baseline_policy_v2_six_axis(baseline_policy_path, policy_overlay_path)
    if effective["gap_policy"]["include_rtma_in_history_mask"] is not True:
        raise Common120SupportError("frozen production policy must include RTMA in the package gap mask")
    validation = baseline["temporal_split"]["validation"]
    start, end = pd.Timestamp(validation["start"]), pd.Timestamp(validation["end"]) + pd.Timedelta(hours=23)
    if str(start.date()) != population.date_start or str(end.date()) != population.date_end:
        raise Common120SupportError(
            "baseline validation window "
            f"[{start.date()}, {end.date()}] does not match the expected population window "
            f"[{population.date_start}, {population.date_end}]"
        )
    target, lead = "qobs_mm_per_h_lead06", 6

    identities = _verify_audit_artifact_identities(PreparationPaths(
        Path(baseline_policy_path), Path(package_root), Path(splits_dir),
        Path(splits_dir) / "development_train.txt"))
    manifest = read_package_manifest(package_root)
    scope = manifest.get("gap_product_scope")
    if scope != [MRMS_PRODUCT, RTMA_PRODUCT]:
        raise Common120SupportError(f"package gap_product_scope must be complete MRMS+RTMA, got {scope!r}")
    membership = validate_full_population_basin_membership(manifest, splits_dir)
    dev_basins = tuple(sorted(set(membership.development_basins)))
    if dev_basins != population.basin_ids:
        raise Common120SupportError(
            "explicit expected population does not equal the package-verified development membership"
        )

    gap_path = Path(package_root) / "masks" / "gap_timestamps.json"
    declared_gap = (manifest.get("gap_timestamp_artifact") or {}).get("sha256")
    actual_gap = hashlib.sha256(gap_path.read_bytes()).hexdigest() if gap_path.is_file() else None
    if not isinstance(declared_gap, str) or actual_gap != declared_gap:
        raise Common120SupportError("package gap-timestamp artifact checksum contradicts package manifest")

    per_dates, per_admitted, counts, zero_support, global_valid_count, gaps = _common120_admitted_maps(
        basins=list(population.basin_ids), package_root=package_root, target=target, lead=lead,
        start=start, end=end, gap_path=gap_path, raise_on_first_zero=False,
    )
    if zero_support:
        raise Common120SupportError(
            f"{len(zero_support)} development basin(s) have zero Common-120 support "
            f"and are named explicitly: {sorted(zero_support)}"
        )

    contract = build_devpop_audit_contract(
        population=population, target_variable=target,
        source_gap_policy_identity="stage1_policy_mrms_rtma_history_v001",
        per_basin_date=per_dates, per_basin_admitted=per_admitted,
        lead_hours=lead, seq_length_floor=SEQ_LENGTH_MAX, **identities,
    )
    values = np.array(list(counts.values()), dtype=int)
    accounting = {
        "audit_contract_id": DEVPOP_AUDIT_CONTRACT_ID,
        "diagnostic_only": True,
        "population_role": population.role,
        "n_basins": len(population.basin_ids),
        "expected_population_size": population.expected_size,
        "membership_ids_sha256": population.membership_ids_sha256,
        "global_validation_issue_times": global_valid_count,
        "total_retained": int(values.sum()),
        "per_basin_retained": counts,
        "min_retained": int(values.min()),
        "max_retained": int(values.max()),
        "median_retained": float(np.median(values)),
        "zero_support_basin_count": 0,
        "gap_count": len(gaps),
        "package_identities": identities,
    }
    return Common120BuildResult(contract=contract, accounting=accounting)


def build_common120_support_for_development_population(
    *, package_root, splits_dir, baseline_policy_path, policy_overlay_path,
) -> Common120BuildResult:
    """**Canonical entry point.**  Build, but never write, the
    full-development-population Common-120 *audit* support contract for the
    approved 2,307-basin development population.

    There is deliberately **no** ``population`` parameter: the population is
    constructed here from the committed split artifacts via
    :meth:`ExpectedPopulationSpec.for_development_train` and then re-pinned on
    every scientifically meaningful axis via
    :func:`assert_canonical_development_population` (role, 2,307 unique IDs,
    line-ending-independent canonical membership hash, full frozen 2024
    validation window, named provenance).  A generic / synthetic population
    cannot be injected here, so it can never be labelled canonical
    development-population audit behaviour.

    Construction reads only committed split artifacts -- never a sealed scope.
    The frozen screening artifact / identity JSON is never touched and the v2
    optimizer objective is never produced."""
    from .devpop_common120_audit_contract import (
        ExpectedPopulationSpec,
        assert_canonical_development_population,
        validate_canonical_devpop_audit_contract,
    )

    population = ExpectedPopulationSpec.for_development_train(splits_dir)
    assert_canonical_development_population(population)
    result = build_common120_support_for_population(
        population=population,
        package_root=package_root,
        splits_dir=splits_dir,
        baseline_policy_path=baseline_policy_path,
        policy_overlay_path=policy_overlay_path,
    )
    # Fail closed: the production-facing build result must satisfy the one
    # canonical contract boundary (every authoritative package / split /
    # provenance identity), not merely be well-formed.
    validate_canonical_devpop_audit_contract(result.contract)
    return result
