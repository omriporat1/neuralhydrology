"""RD1-C4-D1 section A1: canonical package identity binding.

Every test builds a complete synthetic package in ``tmp_path`` and then
breaks exactly one fact, so each identity check is shown to be individually
load-bearing rather than incidentally satisfied. No real Moriah product is
read.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from src.baseline.package_identity_qualification import (
    CANONICAL_OBSERVED_VARIABLE,
    PACKAGE_IDENTITY_ARTIFACTS,
    PackageIdentityError,
    qualify_package_identity,
    verify_basin_time_series_file,
)
from tests._rd1_c4_d1_support import (
    AREA_KM2,
    BUILDER_MANIFEST_SCHEMA_NAME,
    GAP_MASK_RELATIVE_PATH,
    LEAD_HOURS,
    TARGET_VARIABLE,
    build_contract,
    build_package,
    write_package_manifests,
)

BASINS = ["00000001", "00000002", "00000003"]


def _package(tmp_path, basin_ids=None):
    return build_package(tmp_path, basin_ids or BASINS)


def _edit_run_provenance(package_root, mutate):
    """Rewrite ``run_provenance.json`` in place, as ``_edit_manifest`` does
    for the manifest."""
    path = package_root / "run_provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"))


def _edit_manifest(package_root, mutate):
    """Rewrite ``package_manifest.json`` in place. Callers build the contract
    AFTER this so the three identity hashes still agree -- which isolates the
    check under test from the raw-bytes binding."""
    path = package_root / "manifests" / "package_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"))


# --- success ------------------------------------------------------------ #


def test_a_matching_package_qualifies_and_reports_a_complete_identity(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)

    identity = qualify_package_identity(package_root=package_root, contract=contract)

    assert identity.raw_target_variable == CANONICAL_OBSERVED_VARIABLE
    assert identity.target_variable == TARGET_VARIABLE
    assert identity.lead_hours == LEAD_HOURS
    assert identity.period == "validation"
    assert identity.n_package_basins == len(BASINS)
    assert identity.n_qualified_basins == len(BASINS)
    assert identity.timeline_window_verified is True
    assert set(identity.basin_time_series_sha256) == set(BASINS)
    assert identity.basin_time_series_relative_path["00000001"] == "time_series/00000001.nc"


def test_identity_fields_are_receipt_ready_and_exclude_the_per_basin_table(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    fields = identity.identity_fields()

    json.dumps(fields)  # must be serialisable without a custom encoder
    assert fields["contract_checksum_sha256"] == identity.contract_checksum_sha256
    assert "basin_time_series_sha256" not in fields
    assert "basin_time_series_relative_path" not in fields


def test_the_identity_result_is_immutable(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    with pytest.raises(Exception):
        identity.package_manifest_sha256 = "0" * 64
    with pytest.raises(TypeError):
        identity.basin_time_series_sha256["00000001"] = "0" * 64


def test_an_explicit_basin_subset_narrows_the_qualified_population(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)

    identity = qualify_package_identity(
        package_root=package_root, contract=contract, basin_ids=["00000002"]
    )
    assert identity.n_package_basins == 3
    assert identity.n_qualified_basins == 1
    assert set(identity.basin_time_series_sha256) == {"00000002"}


def test_qualification_does_not_hash_basin_files(tmp_path):
    """A1 requires a once-per-task global binding with no whole-package
    rehash. Corrupting a basin NetCDF (same size) therefore does NOT stop
    qualification -- it is caught at the point of use instead."""
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    target = package_root / "time_series" / "00000001.nc"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    target.write_bytes(bytes(payload))

    identity = qualify_package_identity(package_root=package_root, contract=contract)

    with pytest.raises(PackageIdentityError, match="not the file this package was built from"):
        verify_basin_time_series_file(identity, "00000001")


# --- the three raw-bytes identity hashes -------------------------------- #


@pytest.mark.parametrize(
    "field",
    ["package_manifest_sha256", "package_file_checksums_sha256", "package_run_provenance_sha256"],
)
def test_each_identity_hash_mismatch_fails_closed(tmp_path, field):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates, **{field: "0" * 64})

    with pytest.raises(PackageIdentityError, match=f"package identity mismatch for '{field}'"):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize("field,parts", PACKAGE_IDENTITY_ARTIFACTS)
def test_a_missing_identity_artifact_fails_closed(tmp_path, field, parts):
    """All three identity payloads are ALSO members of the producer's
    required layout, so ``package_audit.check_package_layout`` -- which now
    runs first -- is what reports their absence. Either message is a
    fail-closed refusal naming the missing artifact; what matters is that
    qualification cannot proceed without it."""
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    package_root.joinpath(*parts).unlink()

    with pytest.raises(
        PackageIdentityError,
        match="package layout is incomplete|package identity artifact missing",
    ):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_mutating_an_identity_artifact_after_the_contract_was_built_is_detected(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    provenance = package_root / "run_provenance.json"
    provenance.write_bytes(provenance.read_bytes() + b" ")

    with pytest.raises(PackageIdentityError, match="package_run_provenance_sha256"):
        qualify_package_identity(package_root=package_root, contract=contract)


# --- package root shape -------------------------------------------------- #


def test_a_missing_package_root_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    with pytest.raises(PackageIdentityError, match="package root is not a directory"):
        qualify_package_identity(package_root=tmp_path / "absent", contract=contract)


def test_a_package_root_without_time_series_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    for path in (package_root / "time_series").iterdir():
        path.unlink()
    (package_root / "time_series").rmdir()
    with pytest.raises(PackageIdentityError, match="no time_series/ directory"):
        qualify_package_identity(package_root=package_root, contract=contract)


# --- the package's metadata must agree with itself ---------------------- #


def test_manifest_and_file_checksums_sha_disagreement_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m["per_basin_time_series"][0].update({"sha256": "0" * 64}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="package's own metadata disagrees with itself"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_manifest_and_file_checksums_size_disagreement_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m["per_basin_time_series"][0].update({"size_bytes": 1}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="file_checksums.csv records"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_duplicate_checksum_row_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    checksums = package_root / "manifests" / "file_checksums.csv"
    lines = checksums.read_bytes().decode("utf-8").splitlines(keepends=True)
    checksums.write_bytes(("".join(lines) + lines[1]).encode("utf-8"))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="duplicate relative_path"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_an_unexpected_checksum_header_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    checksums = package_root / "manifests" / "file_checksums.csv"
    body = checksums.read_bytes().decode("utf-8").split("\n", 1)[1]
    checksums.write_bytes(("path,sha256,size_bytes,artifact_role\n" + body).encode("utf-8"))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="unexpected file_checksums.csv header"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_an_unreadable_manifest_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    (package_root / "manifests" / "package_manifest.json").write_bytes(b"{ not json")
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="unreadable package manifest"):
        qualify_package_identity(package_root=package_root, contract=contract)


# --- scientific identity ------------------------------------------------- #


def test_a_non_canonical_raw_target_variable_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(package_root, lambda m: m.update({"raw_target_variable": "discharge_cfs"}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="canonical\nRD1-C4 observed series|canonical"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_lead_hour_disagreement_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m.update({"lead_targets": [{"name": TARGET_VARIABLE, "lead_hours": 12}]}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="expected exactly one lead target"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_target_variable_disagreement_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m.update({"lead_targets": [{"name": "qobs_mm_per_h_lead01", "lead_hours": LEAD_HOURS}]}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="expected exactly one lead target"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_contract_basin_absent_from_the_package_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    # basin_count and the per-basin table are updated alongside basin_ids:
    # otherwise the manifest's own self-consistency checks fire first and
    # this test would prove nothing about contract-vs-package population.
    _edit_manifest(
        package_root,
        lambda m: m.update(
            {
                "basin_ids": BASINS[:2],
                "basin_count": 2,
                "per_basin_time_series": [
                    e for e in m["per_basin_time_series"] if e["basin_id"] in BASINS[:2]
                ],
            }
        ),
    )
    _edit_run_provenance(package_root, lambda p: p.update({"basin_count": 2}))
    contract = build_contract(package_root, BASINS, dates)

    # The now-orphaned time_series/00000003.nc checksum row is itself a
    # closed-world violation and is caught first; either way the package
    # cannot be qualified for a contract naming a basin it does not have.
    with pytest.raises(
        PackageIdentityError,
        match="contract basins absent from the package|unexpected checksum entr",
    ):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_missing_per_basin_entry_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m.update(
            {"per_basin_time_series": [e for e in m["per_basin_time_series"] if e["basin_id"] != "00000002"]}
        ),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(
        PackageIdentityError, match="per_basin_time_series does not cover basin_ids exactly"
    ):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_layout_convention_violation_is_refused(tmp_path):
    """The consumer resolves ``time_series/<basin>.nc``; a package that
    records a different path would be read from the wrong file."""
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m["per_basin_time_series"][0].update({"relative_path": "ts/00000001.nc"}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="package layout contradiction"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_requesting_a_basin_outside_the_contract_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="not present in the fixed-support contract"):
        qualify_package_identity(
            package_root=package_root, contract=contract, basin_ids=["99999999"]
        )


def test_a_contract_period_outside_the_package_timeline_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    narrow_end = str(np.datetime64(dates[10], "s"))
    _edit_manifest(package_root, lambda m: m["timeline"].update({"end": narrow_end}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="period identity contradiction"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_missing_timeline_key_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(package_root, lambda m: m["timeline"].pop("rows"))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="timeline is missing required key"):
        qualify_package_identity(package_root=package_root, contract=contract)


# --- per-basin verification at the point of use ------------------------- #


def test_verifying_a_basin_returns_the_recorded_checksum(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    digest = verify_basin_time_series_file(identity, "00000001")
    assert digest == identity.basin_time_series_sha256["00000001"]


def test_verifying_an_unqualified_basin_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root,
        contract=build_contract(package_root, BASINS, dates),
        basin_ids=["00000001"],
    )
    with pytest.raises(PackageIdentityError, match="refusing to read an unqualified basin"):
        verify_basin_time_series_file(identity, "00000002")


def test_a_deleted_basin_file_is_reported_at_the_point_of_use(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    (package_root / "time_series" / "00000001.nc").unlink()
    with pytest.raises(PackageIdentityError, match="package time series missing"):
        verify_basin_time_series_file(identity, "00000001")


def test_a_size_disagreement_is_reported_before_hashing(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    target = package_root / "time_series" / "00000001.nc"
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(PackageIdentityError, match="bytes but the package records"):
        verify_basin_time_series_file(identity, "00000001")


def test_a_qualified_identity_cannot_be_used_to_read_another_package(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    other_root, _, _ = build_package(tmp_path, BASINS, seed=99, package_name="other_package")

    with pytest.raises(PackageIdentityError, match="never qualified"):
        verify_basin_time_series_file(identity, "00000001", package_root=other_root)


def test_the_default_production_read_detects_post_qualification_mutation(tmp_path):
    """RD1-C4-D1 finding A5. The previous implementation memoised a basin's
    verification result, so a file mutated after its first read went on
    returning the recorded digest unless the caller passed ``force=True``.
    No production call site passed it. The default path -- the only one D1
    and the formal consumer use -- must now detect the mutation itself."""
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    assert verify_basin_time_series_file(identity, "00000001") == identity.basin_time_series_sha256[
        "00000001"
    ]

    target = package_root / "time_series" / "00000001.nc"
    original = target.read_bytes()
    payload = bytearray(original)
    payload[-1] ^= 0xFF  # same size: only a real re-hash can see this
    target.write_bytes(bytes(payload))

    with pytest.raises(PackageIdentityError, match="not the file this package was built from"):
        verify_basin_time_series_file(identity, "00000001")

    # Restoring the real bytes makes the same call succeed again, which
    # shows the refusal came from re-reading the file rather than from a
    # sticky failure flag.
    target.write_bytes(original)
    assert verify_basin_time_series_file(identity, "00000001") == identity.basin_time_series_sha256[
        "00000001"
    ]


def test_there_is_no_flag_that_skips_the_digest_recomputation():
    """The correction is structural, not merely a changed default: the
    ``force`` parameter and the module-level memo are gone, so no caller can
    opt back into a cached answer."""
    import inspect

    from src.baseline import package_identity_qualification as pq

    parameters = inspect.signature(pq.verify_basin_time_series_file).parameters
    assert set(parameters) == {"identity", "basin_id", "package_root"}
    assert not [name for name in vars(pq) if "MEMO" in name.upper()]


def test_every_qualified_read_is_proven_against_the_package_checksum(tmp_path):
    """Repeated reads are each proven, so a mutation is caught by whichever
    read follows it -- not only by the first read of the file."""
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    for _ in range(3):
        verify_basin_time_series_file(identity, "00000002")

    target = package_root / "time_series" / "00000002.nc"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 0xFF
    target.write_bytes(bytes(payload))

    with pytest.raises(PackageIdentityError, match="not the file this package was built from"):
        verify_basin_time_series_file(identity, "00000002")


# --- A4: the producer's own declared identity is parsed strictly -------- #


def test_an_unknown_builder_manifest_schema_name_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(package_root, lambda m: m.update({"schema_name": "some_other_builder_v9"}))
    _edit_run_provenance(
        package_root, lambda p: p.update({"package_schema_name": "some_other_builder_v9"})
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="the only\nbuilder-manifest identity|builder-manifest identity"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_wrong_builder_manifest_schema_version_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(package_root, lambda m: m.update({"schema_version": 2}))
    _edit_run_provenance(package_root, lambda p: p.update({"builder_schema_version": 2}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="builder-manifest schema_version"):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize(
    "key",
    ["package_role", "basin_count", "per_basin_time_series", "lead_targets", "gap_timestamp_artifact"],
)
def test_a_manifest_missing_a_producer_written_key_is_refused(tmp_path, key):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(package_root, lambda m: m.pop(key))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="missing required key '%s'" % key):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_package_role_that_contradicts_the_netcdf_schema_is_refused(tmp_path):
    """``package_role`` and the NetCDF schema are two independent producer
    declarations of the same fact, so they must agree."""
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root, lambda m: m.update({"package_role": "stage1_compact_scientific_package"})
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="but the producer assigns"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_an_unrecognized_netcdf_package_schema_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    for edit in (_edit_manifest, _edit_run_provenance):
        edit(package_root, lambda d: d.update({"netcdf_package_schema_name": "invented_v003"}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="does not declare a recognized NetCDF package schema"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_manifest_and_provenance_netcdf_schema_disagreement_is_refused(tmp_path):
    """Only the run-provenance record is changed, to a schema identity that
    is itself perfectly valid -- so this can only be caught by requiring the
    two independent declarations to agree."""
    package_root, dates, _ = _package(tmp_path)
    _edit_run_provenance(
        package_root,
        lambda p: p.update(
            {
                "netcdf_package_schema_name": "stage1_compact_scientific_package_v001",
                "netcdf_package_schema_version": 1,
                "netcdf_time_coordinate": "time",
            }
        ),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="run_provenance.json declares"):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize(
    "key", ["builder_module", "builder_schema_version", "created_at_utc", "dry_run", "qc_csv_enabled"]
)
def test_run_provenance_missing_a_producer_written_key_is_refused(tmp_path, key):
    package_root, dates, _ = _package(tmp_path)
    _edit_run_provenance(package_root, lambda p: p.pop(key))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="missing required key '%s'" % key):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_dry_run_package_is_never_consumed(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_run_provenance(package_root, lambda p: p.update({"dry_run": True}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="dry_run=True"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_foreign_builder_module_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_run_provenance(package_root, lambda p: p.update({"builder_module": "scratch.my_builder"}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="the authoritative"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_provenance_basin_count_disagreement_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_run_provenance(package_root, lambda p: p.update({"basin_count": 99}))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="basin-population identity contradiction"):
        qualify_package_identity(package_root=package_root, contract=contract)


# --- A4: the checksum manifest is closed-world and path-safe ------------ #


@pytest.mark.parametrize(
    "relative_path,expected",
    [
        ("/etc/passwd", "is absolute"),
        ("C:/windows/system32/x", "drive-qualified"),
        ("../../outside.nc", "escaping/traversing path"),
        ("time_series/./00000001x.nc", "escaping/traversing path"),
        ("time_series\\00000001x.nc", "contains a backslash"),
    ],
)
def test_an_unsafe_checksum_path_is_refused(tmp_path, relative_path, expected):
    package_root, dates, _ = _package(tmp_path)
    write_package_manifests(
        package_root,
        BASINS,
        dates,
        extra_checksum_rows=[
            {
                "relative_path": relative_path,
                "sha256": "a" * 64,
                "size_bytes": 1,
                "artifact_role": "authoritative_time_series",
            }
        ],
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match=re.escape(expected)):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_an_unexpected_checksum_entry_is_refused(tmp_path):
    """A row naming a plausible in-package file the producer would never
    write. Closed-world membership is what makes this visible."""
    package_root, dates, _ = _package(tmp_path)
    write_package_manifests(
        package_root,
        BASINS,
        dates,
        extra_checksum_rows=[
            {
                "relative_path": "time_series/99999999.nc",
                "sha256": "b" * 64,
                "size_bytes": 10,
                "artifact_role": "authoritative_time_series",
            }
        ],
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="unexpected checksum entr"):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize(
    "relative_path",
    ["attributes/attributes.csv", "basins/basin_ids.txt", GAP_MASK_RELATIVE_PATH, "time_series/00000002.nc"],
)
def test_a_missing_checksum_entry_is_refused(tmp_path, relative_path):
    package_root, dates, _ = _package(tmp_path)
    write_package_manifests(package_root, BASINS, dates, drop_checksum_rows=[relative_path])
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="missing checksum entr"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_wrong_artifact_role_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    write_package_manifests(
        package_root,
        BASINS,
        dates,
        drop_checksum_rows=[GAP_MASK_RELATIVE_PATH],
        extra_checksum_rows=[
            {
                "relative_path": GAP_MASK_RELATIVE_PATH,
                "sha256": "c" * 64,
                "size_bytes": 1,
                "artifact_role": "something_else",
            }
        ],
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="wrong artifact_role"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_malformed_checksum_digest_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    checksums = package_root / "manifests" / "file_checksums.csv"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[1] = "NOTAHASH"
    lines[1] = ",".join(parts)
    checksums.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="malformed sha256"):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize("size_text", ["-1", "01", "1.5", "size"])
def test_a_malformed_checksum_size_is_refused(tmp_path, size_text):
    package_root, dates, _ = _package(tmp_path)
    checksums = package_root / "manifests" / "file_checksums.csv"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[2] = size_text
    lines[1] = ",".join(parts)
    checksums.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    contract = build_contract(package_root, BASINS, dates)
    with pytest.raises(PackageIdentityError, match="malformed size_bytes"):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize("suffix,match", [("", "missing column"), (",extra", "more fields")])
def test_a_short_or_long_checksum_row_is_refused(tmp_path, suffix, match):
    package_root, dates, _ = _package(tmp_path)
    checksums = package_root / "manifests" / "file_checksums.csv"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    if suffix:
        lines[1] += suffix
    else:
        lines[1] = ",".join(lines[1].split(",")[:-1])
    checksums.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    contract = build_contract(package_root, BASINS, dates)
    with pytest.raises(PackageIdentityError, match=match):
        qualify_package_identity(package_root=package_root, contract=contract)


@pytest.mark.parametrize("value", [None, -1, True, "0"])
def test_malformed_gap_artifact_count_is_refused(tmp_path, value):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda manifest: manifest["gap_timestamp_artifact"].update({"count": value}),
    )
    contract = build_contract(package_root, BASINS, dates)
    with pytest.raises(PackageIdentityError, match="malformed gap_timestamp_artifact"):
        qualify_package_identity(package_root=package_root, contract=contract)


# --- A4: the fixed authoritative artifacts are proven against real bytes -- #


def test_a_mutated_authoritative_artifact_is_detected_during_qualification(tmp_path):
    """Unlike the per-basin NetCDFs, the three small fixed artifacts are
    hashed during qualification itself, so the checksum manifest is proven
    against real bytes rather than merely cross-referenced."""
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    target = package_root / "basins" / "basin_ids.txt"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 0x20  # same size
    target.write_bytes(bytes(payload))

    with pytest.raises(PackageIdentityError, match="do not match its own checksum manifest"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_resized_authoritative_artifact_is_detected_during_qualification(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    contract = build_contract(package_root, BASINS, dates)
    target = package_root / "attributes" / "attributes.csv"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(PackageIdentityError, match="file_checksums.csv records"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_gap_artifact_digest_disagreement_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m["gap_timestamp_artifact"].update({"sha256": "d" * 64}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="gap_timestamp_artifact sha256"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_a_gap_artifact_at_a_foreign_path_is_refused(tmp_path):
    package_root, dates, _ = _package(tmp_path)
    _edit_manifest(
        package_root,
        lambda m: m["gap_timestamp_artifact"].update({"relative_path": "masks/other.json"}),
    )
    contract = build_contract(package_root, BASINS, dates)

    with pytest.raises(PackageIdentityError, match="gap_timestamp_artifact relative_path"):
        qualify_package_identity(package_root=package_root, contract=contract)


def test_the_qualification_reports_the_producer_identity_it_actually_read(tmp_path):
    """A4 asked that qualification not overstate what it validated. These
    are the facts it genuinely parsed, so they are the facts it reports."""
    package_root, dates, _ = _package(tmp_path)
    identity = qualify_package_identity(
        package_root=package_root, contract=build_contract(package_root, BASINS, dates)
    )
    assert identity.manifest_schema_name == BUILDER_MANIFEST_SCHEMA_NAME
    assert identity.manifest_schema_version == 1
    assert identity.package_role == "stage1_scientific_package"
    assert identity.netcdf_package_schema_name == "stage1_scientific_package_v002"
    assert identity.netcdf_time_coordinate == "date"
    assert identity.netcdf_schema_historical_lineage_applied is False
    assert identity.run_provenance_builder_module == "src.baseline.package_builder"
    assert identity.run_provenance_dry_run is False
    # 3 fixed authoritative artifacts + one per basin, and nothing else.
    assert identity.n_checksum_entries == 3 + len(BASINS)


def test_the_fixture_area_is_only_a_magnitude_choice(tmp_path):
    """Guards against a reader assuming the fixture asserts a real basin
    area fact: AREA_KM2 is chosen so synthetic discharges are realistic in
    size, and nothing in identity qualification depends on it."""
    assert AREA_KM2 > 0
