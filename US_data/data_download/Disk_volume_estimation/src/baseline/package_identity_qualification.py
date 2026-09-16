"""Frozen-package identity qualification for contract-bound consumers.

RD1-C4-D1 finding A1: :func:`fixed_support_contract_v2.derive_canonical_package_observed_series`
(and every other contract-bound reader of the frozen scientific package)
historically opened ``package_root`` and trusted whatever NetCDF it found
there. Nothing proved that the opened directory was the package the fixed-
support contract was actually built against, and nothing proved that the
individual basin NetCDF being read still matched the package's own recorded
checksum for that file.

This module is the narrowest generic validator at the lowest authoritative
layer: it consumes a validated fixed-support contract plus a package root,
and either returns an immutable :class:`QualifiedPackageIdentity` or fails
closed with :class:`PackageIdentityError`. It introduces no scientific math
and no new identity convention -- every fact it checks is derived from an
artifact that already exists in the package, using the same interpretation
already established elsewhere in this repository.

Authoritative producers this module binds against
-------------------------------------------------

Every requirement below is read off the actual producer, never invented
(RD1-C4-D1 Correction Pass A, finding A4). The producers are:

``src.baseline.package_builder._build_manifest``
    writes ``manifests/package_manifest.json``.
``src.baseline.package_builder._collect_checksums`` / ``_write_manifests``
    write ``manifests/file_checksums.csv``.
``src.baseline.package_builder._write_run_provenance``
    writes ``run_provenance.json``.

The three contract hash fields mean exactly the raw-bytes SHA-256 of those
three payloads:

``package_manifest_sha256``
    Raw-bytes SHA-256 of ``<package_root>/manifests/package_manifest.json``,
    the UTF-8 ``json.dumps(manifest, indent=2, sort_keys=True)`` text
    ``_write_manifests`` emits.
``package_file_checksums_sha256``
    Raw-bytes SHA-256 of ``<package_root>/manifests/file_checksums.csv``,
    the UTF-8 ``relative_path,sha256,size_bytes,artifact_role`` table
    ``_write_manifests`` emits, sorted by relative path, LF-terminated.
``package_run_provenance_sha256``
    Raw-bytes SHA-256 of ``<package_root>/run_provenance.json``, the UTF-8
    ``json.dumps(provenance, indent=2, sort_keys=True)`` text
    ``_write_run_provenance`` emits.

That mapping is not invented here either: it is exactly
``sweep_v1_production_adapter._verify_artifact_identities`` and
``common120_support_builder._AUDIT_PACKAGE_PAYLOAD_ARTIFACTS``. All three
payloads are builder-written UTF-8 text and are not line-ending sensitive,
so a raw-bytes SHA-256 is the correct identity check for them.

What is proven, and what is deliberately not
--------------------------------------------

Proven here (all of it metadata-cost only):

* the three identity payloads hash to the contract's recorded values;
* the manifest parses strictly: it is a JSON object, it declares the
  builder-manifest schema identity the producer emits
  (``package_builder.SCHEMA_NAME``/``SCHEMA_VERSION``, independently
  redeclared by ``package_audit`` as
  ``HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME``/``_VERSION`` and reused from
  there), and carries every field this qualification depends on, each of
  the exact type the producer writes;
* ``run_provenance.json`` parses strictly, is a JSON object, declares the
  producer's own ``builder_module``, is not a ``dry_run`` package, and
  agrees with the manifest on builder-manifest schema identity and basin
  count. Fields the producer added later (``builder_manifest_schema_*``)
  are cross-checked when present and reported as absent, not manufactured,
  when the package predates them;
* the declared NetCDF package schema is a *recognized* identity, resolved
  through ``package_audit``'s own
  :func:`~.package_audit.resolve_expected_netcdf_package_schema_independent`
  and its independently-redeclared registry -- including that module's
  narrow, exact-lineage historical compact-v001 compatibility path, which
  the real frozen Common-120 package needs because its manifest and
  provenance predate the ``netcdf_package_schema_*``/
  ``netcdf_time_coordinate`` fields entirely. Both the manifest and the
  run-provenance record must resolve, and must resolve to the same schema.
  The resolved ``coordinate_name`` is the AUTHORITATIVE temporal coordinate
  every package read must then bind its variable dimensions to (finding A3);
* the manifest's ``package_role`` is the role the producer assigns to that
  resolved NetCDF schema (``package_builder._PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME``);
* ``file_checksums.csv`` is complete, well formed and closed-world: exactly
  one entry per manifest basin plus the producer's three fixed authoritative
  artifacts and nothing else, each with a lowercase 64-hex digest, a
  non-negative integer size, the producer's own ``artifact_role`` for that
  artifact family, and a relative, non-escaping, non-traversing,
  forward-slash POSIX path. Duplicate, missing, malformed, absolute,
  escaping, traversing, short, long and unexpected entries are all rejected
  -- across the whole table, not only the selected basins;
* the manifest's ``per_basin_time_series`` table covers exactly the
  manifest's own ``basin_ids`` once each and agrees with
  ``file_checksums.csv`` on path, digest and size for every one of them --
  again across the whole table, not only the selected basins;
* the manifest's ``gap_timestamp_artifact`` digest agrees with that file's
  ``file_checksums.csv`` row;
* the package's required layout is present
  (:func:`package_audit.check_package_layout`), and the three fixed
  authoritative artifacts hash to their recorded digests;
* the package's basin population contains every qualified basin, each at
  the ``time_series/<basin_id>.nc`` path the consumer actually resolves;
* the contract's target variable / lead-hours pair is one of the package's
  own ``lead_targets``, the package's ``raw_target_variable`` is the
  canonical observed variable, and the contract's period window lies inside
  the package's own timeline.

Deliberately NOT proven here, and never claimed:

* that the bytes of every *basin* file still match their recorded
  checksums. That is the independent package audit's job
  (:func:`package_audit.run_audit`), it rehashes the whole package, and it
  is far too expensive to repeat per task. The per-basin NetCDF proof is a
  separate, explicitly-called step (:func:`verify_basin_time_series_file`)
  so a caller performs it exactly once per basin actually consumed,
  immediately before reading that file's values;
* a closed-world on-disk file enumeration. ``package_audit`` has one
  (:func:`~.package_audit.check_exact_package_layout`), but its expected set
  is the six required metadata files plus one NetCDF per basin, which
  rejects the optional QC CSV artifacts a package built with
  ``qc_csv_enabled`` legitimately contains -- and the real frozen package is
  such a package. Applying it here would reject valid packages, so this
  module applies the open-world required-layout check instead and closes the
  world over the *checksum manifest*, which the producer does close.

Cost model for the per-basin proof (finding A5)
-----------------------------------------------

:func:`verify_basin_time_series_file` ALWAYS recomputes the file's SHA-256
from disk bytes. There is no memoisation and no ``force`` flag, because a
memo keyed on "this process already checked this basin" cannot detect a file
mutated after that check -- which is exactly the window a long-running task
leaves open, and exactly the defect RD1-C4-D1 finding A5 reports. Nor is a
size/mtime "filesystem identity" cache used: rechecking metadata is not
cryptographic verification, and this function's contract is cryptographic.

The cost is one full read of that basin's NetCDF per call. The correct way
to control it is to call the function once per basin per consumption, not to
cache its result. For the RD1-C4-D1 population this is ~3 hashes per basin
(the diagnostic's own guard, the canonical-observation read, and the
fixed-support evaluator read) x 400 basins x 24 trials, i.e. of order 10^4
file hashes against ~8x10^7 element comparisons -- a small fraction of the
task, and a deliberate, stated price for a proof that cannot go stale.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .package_audit import (
    AuditReport,
    HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME,
    HISTORICAL_BUILDER_MANIFEST_SCHEMA_VERSION,
    HISTORICAL_BUILDER_MODULE,
    check_package_layout,
    package_declares_historical_v001_compatibility_lineage,
    resolve_expected_netcdf_package_schema_independent,
    sha256_file,
)

__all__ = [
    "PackageIdentityError",
    "QualifiedPackageIdentity",
    "CANONICAL_OBSERVED_VARIABLE",
    "PACKAGE_IDENTITY_ARTIFACTS",
    "FIXED_AUTHORITATIVE_CHECKSUM_ENTRIES",
    "TIME_SERIES_ARTIFACT_ROLE",
    "EXPECTED_PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME",
    "qualify_package_identity",
    "verify_basin_time_series_file",
]


class PackageIdentityError(ValueError):
    """Raised when a package root cannot be proven to be the package a
    fixed-support contract was built against, or when a consumed basin
    NetCDF no longer matches the package's own recorded checksum. This is
    always a global/product identity contradiction -- never a normal
    per-basin scientific exclusion."""


#: The frozen package's own raw observed-discharge variable, i.e. the
#: variable the canonical RD1-C4 observed series is read from. Identical to
#: ``package_assembly.RAW_TARGET_VARIABLE`` / ``package_audit.RAW_TARGET_VARIABLE``;
#: restated here (rather than imported) only so this low-level validator
#: does not pull in the package-assembly import graph.
CANONICAL_OBSERVED_VARIABLE = "qobs_m3s"

#: ``(contract_field, relative_path_parts)`` for the three package-identity
#: payloads. Same mapping as ``sweep_v1_production_adapter`` and
#: ``common120_support_builder`` -- do not redefine it a fourth time.
PACKAGE_IDENTITY_ARTIFACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("package_manifest_sha256", ("manifests", "package_manifest.json")),
    ("package_file_checksums_sha256", ("manifests", "file_checksums.csv")),
    ("package_run_provenance_sha256", ("run_provenance.json",)),
)

#: Per-basin NetCDF path convention, identical to
#: ``nh_seed_evaluation.basin_netcdf_path``.
_TIME_SERIES_DIR = "time_series"

#: ``artifact_role`` the producer (``package_builder._collect_checksums``)
#: writes for every ``time_series/<basin_id>.nc`` row.
TIME_SERIES_ARTIFACT_ROLE = "authoritative_time_series"

#: The three non-basin rows ``package_builder._collect_checksums`` always
#: writes, with the producer's own ``artifact_role`` for each. Together with
#: one ``time_series`` row per manifest basin this is the COMPLETE expected
#: content of ``file_checksums.csv`` -- the producer writes nothing else, so
#: anything else present is an unexpected entry.
FIXED_AUTHORITATIVE_CHECKSUM_ENTRIES: Mapping[str, str] = MappingProxyType(
    {
        "attributes/attributes.csv": "authoritative_static_attributes",
        "basins/basin_ids.txt": "authoritative_basin_list",
        "masks/gap_timestamps.json": "authoritative_gap_mask",
    }
)

#: Mirrors ``package_builder._PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME``: the
#: closed mapping from the selected NetCDF package schema to the package's
#: own role identity. Redeclared (never imported) for the same reason
#: ``package_audit`` redeclares the schema registry -- so a bug in the
#: producer's own table cannot also blind this qualification. The v001 entry
#: is additionally asserted by ``package_audit.HISTORICAL_PACKAGE_ROLE``.
EXPECTED_PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME: Mapping[str, str] = MappingProxyType(
    {
        "stage1_compact_scientific_package_v001": "stage1_compact_scientific_package",
        "stage1_scientific_package_v002": "stage1_scientific_package",
    }
)

#: Exact header ``package_builder._write_manifests`` emits.
_CHECKSUM_CSV_HEADER = ["relative_path", "sha256", "size_bytes", "artifact_role"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NON_NEGATIVE_INT_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _relative_time_series_path(basin_id: str) -> str:
    return f"{_TIME_SERIES_DIR}/{basin_id}.nc"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PackageIdentityError(message)


def _is_strict_int(value: object) -> bool:
    """``True`` only for a real ``int``. ``bool`` is excluded: JSON ``true``
    parses to ``True``, which would otherwise silently pass as the integer
    ``1`` for a count or a version."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class QualifiedPackageIdentity:
    """Immutable proof that one package root is the package a specific
    fixed-support contract was built against.

    Produced only by :func:`qualify_package_identity`. Consumers must treat
    this object -- not a path string -- as the authority for "which package
    am I reading?", and must call :func:`verify_basin_time_series_file`
    before consuming any individual basin NetCDF's values.

    ``netcdf_time_coordinate`` is the AUTHORITATIVE temporal coordinate name
    for every basin NetCDF in this package, resolved through
    ``package_audit``'s own recognized-schema registry. A package reader must
    bind its variable dimensions to this name rather than assuming one
    (RD1-C4-D1 finding A3).

    ``basin_time_series_sha256``/``basin_time_series_size_bytes``/
    ``basin_time_series_relative_path`` are restricted to the qualified
    basin population (the contract's basins, or an explicitly requested
    subset of them) so a task never carries the whole package's file table
    around in memory or in a receipt. The whole table is still *validated*
    -- it is simply not retained.
    """

    package_root: str
    contract_id: str
    contract_checksum_sha256: str
    contract_schema_name: str
    contract_schema_version: int
    package_manifest_sha256: str
    package_file_checksums_sha256: str
    package_run_provenance_sha256: str
    manifest_schema_name: str
    manifest_schema_version: int
    package_role: str
    netcdf_package_schema_name: str
    netcdf_package_schema_version: int
    netcdf_time_coordinate: str
    netcdf_schema_historical_lineage_applied: bool
    run_provenance_builder_module: str
    run_provenance_created_at_utc: str
    run_provenance_dry_run: bool
    raw_target_variable: str
    target_variable: str
    lead_hours: int
    period: str
    contract_date_start: str
    contract_date_end: str
    timeline_start: str
    timeline_end: str
    timeline_rows: int
    timeline_frequency: str
    timeline_window_verified: bool
    n_package_basins: int
    n_qualified_basins: int
    n_checksum_entries: int
    basin_time_series_relative_path: Mapping[str, str]
    basin_time_series_sha256: Mapping[str, str]
    basin_time_series_size_bytes: Mapping[str, int]
    qualified_at_utc: str

    def __post_init__(self) -> None:
        for name in (
            "basin_time_series_relative_path",
            "basin_time_series_sha256",
            "basin_time_series_size_bytes",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def identity_fields(self) -> dict:
        """Flat, JSON-serializable identity record for receipts/manifests.

        Deliberately excludes the per-basin file table (that is bulk
        evidence, not identity) while including every global fact a
        downstream receipt must pin.

        Also deliberately excludes ``qualified_at_utc``. That field records
        WHEN this package was checked, not WHICH package was checked, so it
        is provenance rather than identity: including it would make two
        qualifications of the same package on the same contract compare
        unequal, and any consumer that pins a shard to an identity would
        then be unable to prove a resumed run is the same run. Callers that
        want the qualification time must read the attribute and record it
        beside the identity, not inside it.
        """
        return {
            "package_root": self.package_root,
            "contract_id": self.contract_id,
            "contract_checksum_sha256": self.contract_checksum_sha256,
            "contract_schema_name": self.contract_schema_name,
            "contract_schema_version": self.contract_schema_version,
            "package_manifest_sha256": self.package_manifest_sha256,
            "package_file_checksums_sha256": self.package_file_checksums_sha256,
            "package_run_provenance_sha256": self.package_run_provenance_sha256,
            "manifest_schema_name": self.manifest_schema_name,
            "manifest_schema_version": self.manifest_schema_version,
            "package_role": self.package_role,
            "netcdf_package_schema_name": self.netcdf_package_schema_name,
            "netcdf_package_schema_version": self.netcdf_package_schema_version,
            "netcdf_time_coordinate": self.netcdf_time_coordinate,
            "netcdf_schema_historical_lineage_applied": self.netcdf_schema_historical_lineage_applied,
            "run_provenance_builder_module": self.run_provenance_builder_module,
            "run_provenance_created_at_utc": self.run_provenance_created_at_utc,
            "run_provenance_dry_run": self.run_provenance_dry_run,
            "raw_target_variable": self.raw_target_variable,
            "target_variable": self.target_variable,
            "lead_hours": self.lead_hours,
            "period": self.period,
            "contract_date_start": self.contract_date_start,
            "contract_date_end": self.contract_date_end,
            "timeline_start": self.timeline_start,
            "timeline_end": self.timeline_end,
            "timeline_rows": self.timeline_rows,
            "timeline_frequency": self.timeline_frequency,
            "timeline_window_verified": self.timeline_window_verified,
            "n_package_basins": self.n_package_basins,
            "n_qualified_basins": self.n_qualified_basins,
            "n_checksum_entries": self.n_checksum_entries,
        }


# ---------------------------------------------------------------------------
# Strict payload parsing
# ---------------------------------------------------------------------------


def _load_json_object(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageIdentityError(f"{path}: unreadable {label}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path}: {label} is not a JSON object")
    return payload


def _reject_unsafe_relative_path(path_value: str, *, source: Path) -> None:
    """Reject any checksum-manifest path that is not a plain, relative,
    forward-slash path inside the package.

    The producer only ever emits ``PurePosixPath.as_posix()`` of a path
    relative to the package root, so anything absolute, drive-qualified,
    backslash-separated, empty, dot-segmented, or ``..``-traversing is a
    tampered or foreign entry -- never a legitimate one. Rejecting these is
    what keeps a checksum row from naming a file outside the package.
    """
    _require(path_value != "", f"{source}: empty relative_path in file_checksums.csv")
    _require(
        "\\" not in path_value,
        f"{source}: relative_path {path_value!r} contains a backslash; the producer writes POSIX paths only",
    )
    _require(
        not path_value.startswith("/"),
        f"{source}: relative_path {path_value!r} is absolute",
    )
    _require(
        not _WINDOWS_DRIVE_RE.match(path_value),
        f"{source}: relative_path {path_value!r} is drive-qualified (absolute)",
    )
    segments = path_value.split("/")
    for segment in segments:
        _require(
            segment not in ("", ".", ".."),
            f"{source}: relative_path {path_value!r} contains an empty, '.' or '..' segment "
            "(escaping/traversing path)",
        )


def _read_file_checksums_csv(path: Path) -> dict:
    """Strictly parse ``manifests/file_checksums.csv``.

    Validates the exact producer header, then every row's arity, path
    safety, digest form, size form and role presence, rejecting duplicates.
    Closed-world membership against the manifest is checked by the caller,
    which is the only place the expected basin population is known.
    """
    rows: dict = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        _require(
            list(reader.fieldnames or []) == _CHECKSUM_CSV_HEADER,
            f"{path}: unexpected file_checksums.csv header {reader.fieldnames!r} "
            f"(expected {_CHECKSUM_CSV_HEADER!r})",
        )
        for line_no, row in enumerate(reader, start=2):
            _require(
                None not in row,
                f"{path}: line {line_no} has more fields than the {len(_CHECKSUM_CSV_HEADER)}-column header",
            )
            missing = [name for name in _CHECKSUM_CSV_HEADER if row.get(name) is None]
            _require(
                not missing,
                f"{path}: line {line_no} is missing column(s) {missing} "
                f"(short row: {[row.get(n) for n in _CHECKSUM_CSV_HEADER]!r})",
            )
            rel = row["relative_path"]
            _reject_unsafe_relative_path(rel, source=path)
            _require(
                rel not in rows,
                f"{path}: duplicate relative_path {rel!r} in file_checksums.csv",
            )
            digest = row["sha256"]
            _require(
                bool(_SHA256_RE.match(digest)),
                f"{path}: line {line_no} ({rel!r}) has malformed sha256 {digest!r}; expected 64 lowercase "
                "hex characters",
            )
            size_text = row["size_bytes"]
            _require(
                bool(_NON_NEGATIVE_INT_RE.match(size_text)),
                f"{path}: line {line_no} ({rel!r}) has malformed size_bytes {size_text!r}; expected a "
                "non-negative decimal integer",
            )
            role = row["artifact_role"]
            _require(
                role != "",
                f"{path}: line {line_no} ({rel!r}) has an empty artifact_role",
            )
            rows[rel] = {
                "sha256": digest,
                "size_bytes": int(size_text),
                "artifact_role": role,
            }
    _require(rows != {}, f"{path}: file_checksums.csv contains no entries")
    return rows


def _resolve_netcdf_schema(
    manifest: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[dict, bool]:
    """Resolve the package's NetCDF schema identity through ``package_audit``.

    Reuses that module's own registry, its resolver and its narrow
    exact-lineage historical compact-v001 compatibility path rather than
    reimplementing any of them (RD1-C4-D1 finding A4: "partly duplicates
    existing package-audit schema logic"). Both the manifest and the
    run-provenance record must resolve, and must resolve identically.
    """
    report = AuditReport()
    historical = package_declares_historical_v001_compatibility_lineage(manifest, run_provenance)
    from_manifest = resolve_expected_netcdf_package_schema_independent(
        report,
        source_label="package_manifest",
        declared=manifest,
        historical_lineage_recognized=historical,
    )
    from_provenance = resolve_expected_netcdf_package_schema_independent(
        report,
        source_label="run_provenance",
        declared=run_provenance,
        historical_lineage_recognized=historical,
    )
    _require(
        report.error_count == 0 and from_manifest is not None and from_provenance is not None,
        f"{manifest_path}: the package does not declare a recognized NetCDF package schema: "
        f"{report.failed_messages()}",
    )
    _require(
        from_manifest == from_provenance,
        f"{manifest_path}: package_manifest.json declares NetCDF schema {from_manifest} but "
        f"run_provenance.json declares {from_provenance} -- the package's own metadata disagrees with itself",
    )
    return dict(from_manifest), bool(historical)


def _parse_manifest_strictly(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict:
    """Validate the builder-manifest fields this qualification depends on.

    Every field below is one ``package_builder._build_manifest`` actually
    writes, and is required with the exact type the producer writes. Fields
    the producer writes but this qualification does not consume are not
    required, and no field the producer does not write is invented.
    """
    _require(
        manifest.get("schema_name") == HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME,
        f"{manifest_path}: builder-manifest schema_name is {manifest.get('schema_name')!r}; the only "
        f"builder-manifest identity any producer in this repository emits is "
        f"{HISTORICAL_BUILDER_MANIFEST_SCHEMA_NAME!r}",
    )
    _require(
        manifest.get("schema_version") == HISTORICAL_BUILDER_MANIFEST_SCHEMA_VERSION,
        f"{manifest_path}: builder-manifest schema_version is {manifest.get('schema_version')!r}; expected "
        f"{HISTORICAL_BUILDER_MANIFEST_SCHEMA_VERSION!r}",
    )

    for key, predicate, description in (
        ("package_role", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("basin_count", _is_strict_int, "an integer"),
        ("basin_ids", lambda v: isinstance(v, list), "a list"),
        ("per_basin_time_series", lambda v: isinstance(v, list), "a list"),
        ("raw_target_variable", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("lead_targets", lambda v: isinstance(v, list), "a list"),
        ("timeline", lambda v: isinstance(v, dict), "a JSON object"),
        ("gap_timestamp_artifact", lambda v: isinstance(v, dict), "a JSON object"),
    ):
        _require(key in manifest, f"{manifest_path}: manifest is missing required key {key!r}")
        _require(
            predicate(manifest[key]),
            f"{manifest_path}: manifest key {key!r} must be {description}, got {type(manifest[key]).__name__}",
        )

    basin_ids = list(manifest["basin_ids"])
    non_str = [b for b in basin_ids if not isinstance(b, str)]
    _require(not non_str, f"{manifest_path}: manifest basin_ids contains non-string entries: {non_str[:5]}")
    duplicates = sorted({b for b in basin_ids if basin_ids.count(b) > 1})
    _require(not duplicates, f"{manifest_path}: manifest basin_ids contains duplicates: {duplicates[:10]}")
    _require(
        int(manifest["basin_count"]) == len(basin_ids),
        f"{manifest_path}: manifest basin_count {manifest['basin_count']!r} != len(basin_ids) "
        f"{len(basin_ids)} -- the package's own metadata disagrees with itself",
    )

    timeline = manifest["timeline"]
    for key, predicate, description in (
        ("start", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("end", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("rows", _is_strict_int, "an integer"),
        ("frequency", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
    ):
        _require(key in timeline, f"{manifest_path}: manifest timeline is missing required key {key!r}")
        _require(
            predicate(timeline[key]),
            f"{manifest_path}: manifest timeline key {key!r} must be {description}, got "
            f"{type(timeline[key]).__name__}",
        )
    _require(
        int(timeline["rows"]) > 0,
        f"{manifest_path}: manifest timeline rows is {timeline['rows']!r}; expected a positive row count",
    )

    lead_targets = list(manifest["lead_targets"])
    for entry in lead_targets:
        _require(
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and entry.get("name") != ""
            and _is_strict_int(entry.get("lead_hours"))
            and entry.get("lead_hours") > 0,
            f"{manifest_path}: malformed lead_targets entry {entry!r}; expected "
            "{'name': non-empty str, 'lead_hours': positive int}",
        )

    gap_artifact = manifest["gap_timestamp_artifact"]
    _require(
        isinstance(gap_artifact.get("relative_path"), str)
        and gap_artifact.get("relative_path") != ""
        and isinstance(gap_artifact.get("sha256"), str)
        and bool(_SHA256_RE.match(gap_artifact["sha256"]))
        and _is_strict_int(gap_artifact.get("count"))
        and gap_artifact["count"] >= 0,
        f"{manifest_path}: malformed gap_timestamp_artifact {gap_artifact!r}; expected producer fields "
        "relative_path, sha256 and non-negative integer count",
    )

    return {"basin_ids": basin_ids, "timeline": timeline, "lead_targets": lead_targets}


def _parse_run_provenance_strictly(
    run_provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    provenance_path: Path,
) -> dict:
    """Validate ``run_provenance.json`` and cross-check it against the manifest.

    The required set is exactly what a real, already-built package carries:
    ``builder_module``, ``builder_schema_version``, ``package_schema_name``,
    ``created_at_utc``, ``dry_run``, ``basin_count``, ``qc_csv_enabled``.
    Verified against the real frozen compact-v001 package's own
    ``run_provenance.json``, which contains those seven keys and no others.

    ``builder_manifest_schema_name``/``builder_manifest_schema_version`` were
    added to the producer later and are genuinely ABSENT from that real
    package, so they are cross-checked when present and neither required nor
    manufactured when absent. The ``netcdf_*`` fields are likewise absent
    there and are handled by :func:`_resolve_netcdf_schema`, which routes
    that exact case through ``package_audit``'s historical-lineage path.
    """
    for key, predicate, description in (
        ("builder_module", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("builder_schema_version", _is_strict_int, "an integer"),
        ("package_schema_name", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("created_at_utc", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
        ("dry_run", lambda v: isinstance(v, bool), "a boolean"),
        ("basin_count", _is_strict_int, "an integer"),
        ("qc_csv_enabled", lambda v: isinstance(v, bool), "a boolean"),
    ):
        _require(
            key in run_provenance,
            f"{provenance_path}: run provenance is missing required key {key!r} "
            "(written by package_builder._write_run_provenance)",
        )
        _require(
            predicate(run_provenance[key]),
            f"{provenance_path}: run provenance key {key!r} must be {description}, got "
            f"{type(run_provenance[key]).__name__}",
        )

    _require(
        run_provenance["builder_module"] == HISTORICAL_BUILDER_MODULE,
        f"{provenance_path}: builder_module is {run_provenance['builder_module']!r}; the authoritative "
        f"package producer is {HISTORICAL_BUILDER_MODULE!r}",
    )
    _require(
        run_provenance["dry_run"] is False,
        f"{provenance_path}: run provenance records dry_run=True -- a dry-run package is not a scientific "
        "product and must never be consumed",
    )
    _require(
        run_provenance["package_schema_name"] == manifest["schema_name"],
        f"{provenance_path}: package_schema_name {run_provenance['package_schema_name']!r} != manifest "
        f"schema_name {manifest['schema_name']!r} -- the package's own metadata disagrees with itself",
    )
    _require(
        run_provenance["builder_schema_version"] == manifest["schema_version"],
        f"{provenance_path}: builder_schema_version {run_provenance['builder_schema_version']!r} != manifest "
        f"schema_version {manifest['schema_version']!r}",
    )
    _require(
        run_provenance["basin_count"] == manifest["basin_count"],
        f"{provenance_path}: run provenance basin_count {run_provenance['basin_count']!r} != manifest "
        f"basin_count {manifest['basin_count']!r} -- basin-population identity contradiction",
    )

    for key, manifest_key in (
        ("builder_manifest_schema_name", "schema_name"),
        ("builder_manifest_schema_version", "schema_version"),
    ):
        if key in run_provenance:
            _require(
                run_provenance[key] == manifest[manifest_key],
                f"{provenance_path}: {key} {run_provenance[key]!r} != manifest {manifest_key} "
                f"{manifest[manifest_key]!r}",
            )

    return {
        "builder_module": str(run_provenance["builder_module"]),
        "created_at_utc": str(run_provenance["created_at_utc"]),
        "dry_run": bool(run_provenance["dry_run"]),
    }


def _parse_timeline_instant(value) -> Optional[np.datetime64]:
    try:
        return np.datetime64(str(value))
    except (TypeError, ValueError):
        return None


def qualify_package_identity(
    *,
    package_root,
    contract: Mapping[str, Any],
    basin_ids: Optional[Sequence[str]] = None,
) -> QualifiedPackageIdentity:
    """Prove that ``package_root`` is the package ``contract`` was built
    against, and return an immutable qualification result.

    Performs, in order, failing closed at the first contradiction:

    1. ``contract`` re-validated against the fixed-support contract schema.
    2. ``package_root`` exists, is a directory, contains a ``time_series/``
       directory, and passes ``package_audit.check_package_layout`` (every
       required top-level path and metadata file present).
    3. Each of the three package-identity payloads exists and its raw-bytes
       SHA-256 equals the contract's corresponding recorded hash.
    4. ``manifests/package_manifest.json`` and ``run_provenance.json`` parse
       strictly as JSON objects and carry, with the producer's exact types,
       every field this qualification consumes; run provenance is not a
       dry-run record and agrees with the manifest on builder-manifest
       schema identity and basin count.
    5. The NetCDF package schema is resolved through ``package_audit``'s own
       registry/resolver from BOTH the manifest and the run-provenance
       record, which must agree; the manifest's ``package_role`` is the role
       the producer assigns to that schema. The resolved ``coordinate_name``
       becomes ``netcdf_time_coordinate`` -- the authoritative temporal
       coordinate for every subsequent package read.
    6. ``manifests/file_checksums.csv`` parses strictly (header, arity, path
       safety, digest form, size form, role presence, no duplicates) and its
       entry set is exactly the producer's three fixed authoritative
       artifacts plus one ``time_series/<basin_id>.nc`` per manifest basin,
       each with the producer's own ``artifact_role``; nothing else.
    7. The manifest's ``per_basin_time_series`` table covers exactly the
       manifest's ``basin_ids`` once each, uses the
       ``time_series/<basin_id>.nc`` path the consumer resolves, and agrees
       with ``file_checksums.csv`` on digest and size for every basin. The
       ``gap_timestamp_artifact`` digest likewise agrees with its row.
    8. The three fixed authoritative artifacts are hashed and must match
       their recorded digests and sizes.
    9. Package/contract scientific identity: the manifest's
       ``raw_target_variable`` is the canonical observed variable; the
       manifest's ``lead_targets`` contains exactly the contract's
       ``(target_variable, lead_hours)`` pair; every qualified basin is in
       the package's own ``basin_ids``.
    10. Period containment: the contract's ``[date_start, date_end]`` window
        lies inside the package's own ``timeline``. Contracts whose
        ``date_dtype`` is not ``datetime64`` (synthetic integer-index test
        fixtures only) cannot express a real instant, so the window check is
        skipped and recorded as ``timeline_window_verified=False`` rather
        than silently claimed.

    Hashes exactly six small metadata files (the three identity payloads and
    the three fixed authoritative artifacts). Performs NO per-basin NetCDF
    hashing -- see :func:`verify_basin_time_series_file`. Intended to be
    called once per task, not once per trial and never once per basin.
    """
    from .fixed_support_contract_v2 import validate_fixed_support_contract

    validated = validate_fixed_support_contract(contract)
    root = Path(package_root)
    _require(root.is_dir(), f"package root is not a directory: {root}")
    _require((root / _TIME_SERIES_DIR).is_dir(), f"package root has no {_TIME_SERIES_DIR}/ directory: {root}")

    layout_report = AuditReport()
    check_package_layout(layout_report, root)
    _require(
        layout_report.error_count == 0,
        f"{root}: package layout is incomplete: {layout_report.failed_messages()}",
    )

    observed: dict = {}
    for field_name, parts in PACKAGE_IDENTITY_ARTIFACTS:
        artifact_path = root.joinpath(*parts)
        _require(
            artifact_path.is_file(),
            f"package identity artifact missing: {artifact_path} (contract field {field_name!r})",
        )
        actual = sha256_file(artifact_path)
        expected = validated[field_name]
        _require(
            actual == expected,
            f"package identity mismatch for {field_name!r}: {artifact_path} has sha256 {actual} but the "
            f"fixed-support contract {validated['contract_id']!r} records {expected} -- this package root is "
            "not the package this contract was built against",
        )
        observed[field_name] = actual

    manifest_path = root / "manifests" / "package_manifest.json"
    checksums_path = root / "manifests" / "file_checksums.csv"
    provenance_path = root / "run_provenance.json"

    manifest = _load_json_object(manifest_path, "package manifest")
    run_provenance = _load_json_object(provenance_path, "run provenance")

    parsed_manifest = _parse_manifest_strictly(manifest, manifest_path=manifest_path)
    parsed_provenance = _parse_run_provenance_strictly(
        run_provenance, manifest, provenance_path=provenance_path
    )
    netcdf_schema, historical_lineage_applied = _resolve_netcdf_schema(
        manifest, run_provenance, manifest_path=manifest_path
    )

    expected_role = EXPECTED_PACKAGE_ROLE_BY_NETCDF_SCHEMA_NAME.get(netcdf_schema["name"])
    _require(
        expected_role is not None,
        f"{manifest_path}: no package_role is defined for NetCDF schema {netcdf_schema['name']!r}",
    )
    _require(
        manifest["package_role"] == expected_role,
        f"{manifest_path}: package_role is {manifest['package_role']!r} but the producer assigns "
        f"{expected_role!r} to NetCDF schema {netcdf_schema['name']!r}",
    )

    package_basin_ids = parsed_manifest["basin_ids"]
    checksum_rows = _read_file_checksums_csv(checksums_path)

    # Closed-world checksum manifest: exactly the producer's fixed artifacts
    # plus one time series per manifest basin, each with the producer's role.
    expected_roles: dict = dict(FIXED_AUTHORITATIVE_CHECKSUM_ENTRIES)
    for basin_id in package_basin_ids:
        expected_roles[_relative_time_series_path(basin_id)] = TIME_SERIES_ARTIFACT_ROLE
    missing_rows = sorted(set(expected_roles) - set(checksum_rows))
    unexpected_rows = sorted(set(checksum_rows) - set(expected_roles))
    _require(
        not missing_rows,
        f"{checksums_path}: missing checksum entr{'y' if len(missing_rows) == 1 else 'ies'} "
        f"{missing_rows[:10]} ({len(missing_rows)} total) -- the package's checksum manifest is incomplete",
    )
    _require(
        not unexpected_rows,
        f"{checksums_path}: unexpected checksum entr{'y' if len(unexpected_rows) == 1 else 'ies'} "
        f"{unexpected_rows[:10]} ({len(unexpected_rows)} total) -- the producer writes one row per basin "
        "time series plus exactly "
        f"{sorted(FIXED_AUTHORITATIVE_CHECKSUM_ENTRIES)} and nothing else",
    )
    wrong_role = sorted(
        rel for rel, expected in expected_roles.items() if checksum_rows[rel]["artifact_role"] != expected
    )
    _require(
        not wrong_role,
        f"{checksums_path}: wrong artifact_role for {wrong_role[:5]} -- expected "
        f"{ {rel: expected_roles[rel] for rel in wrong_role[:5]} }, got "
        f"{ {rel: checksum_rows[rel]['artifact_role'] for rel in wrong_role[:5]} }",
    )

    # Manifest per-basin table must itself be complete and consistent with
    # the checksum manifest across the WHOLE package, not only the selected
    # basins (RD1-C4-D1 finding A4).
    manifest_entry_by_basin: dict = {}
    for entry in manifest["per_basin_time_series"]:
        _require(
            isinstance(entry, dict),
            f"{manifest_path}: malformed per_basin_time_series entry {entry!r}; expected a JSON object",
        )
        basin_id = entry.get("basin_id")
        _require(
            isinstance(basin_id, str) and basin_id != "",
            f"{manifest_path}: per_basin_time_series entry has a non-string basin_id {basin_id!r}",
        )
        _require(
            basin_id not in manifest_entry_by_basin,
            f"{manifest_path}: duplicate per_basin_time_series entry for basin {basin_id!r}",
        )
        for key, predicate, description in (
            ("relative_path", lambda v: isinstance(v, str) and v != "", "a non-empty string"),
            ("sha256", lambda v: isinstance(v, str) and bool(_SHA256_RE.match(v)), "64 lowercase hex chars"),
            ("size_bytes", lambda v: _is_strict_int(v) and v >= 0, "a non-negative integer"),
        ):
            _require(
                key in entry,
                f"{manifest_path}: per_basin_time_series entry for basin {basin_id!r} is missing {key!r}",
            )
            _require(
                predicate(entry[key]),
                f"{manifest_path}: per_basin_time_series entry for basin {basin_id!r} has {key}="
                f"{entry[key]!r}; expected {description}",
            )
        manifest_entry_by_basin[basin_id] = entry

    manifest_only = sorted(set(manifest_entry_by_basin) - set(package_basin_ids))
    basins_only = sorted(set(package_basin_ids) - set(manifest_entry_by_basin))
    _require(
        not manifest_only and not basins_only,
        f"{manifest_path}: per_basin_time_series does not cover basin_ids exactly "
        f"(only in per_basin_time_series: {manifest_only[:10]}; only in basin_ids: {basins_only[:10]}) -- "
        "the package's own metadata disagrees with itself",
    )

    for basin_id in package_basin_ids:
        entry = manifest_entry_by_basin[basin_id]
        rel = entry["relative_path"]
        expected_rel = _relative_time_series_path(basin_id)
        _require(
            rel == expected_rel,
            f"{manifest_path}: basin {basin_id!r} time series relative_path is {rel!r} but the consumer "
            f"resolves {expected_rel!r} (nh_seed_evaluation.basin_netcdf_path) -- package layout contradiction",
        )
        row = checksum_rows[rel]
        _require(
            row["sha256"] == entry["sha256"],
            f"basin {basin_id!r}: package manifest records sha256 {entry['sha256']} for {rel!r} but "
            f"file_checksums.csv records {row['sha256']} -- the package's own metadata disagrees with itself",
        )
        _require(
            int(row["size_bytes"]) == int(entry["size_bytes"]),
            f"basin {basin_id!r}: package manifest records size_bytes {entry['size_bytes']} for {rel!r} but "
            f"file_checksums.csv records {row['size_bytes']}",
        )

    gap_artifact = manifest["gap_timestamp_artifact"]
    gap_rel = gap_artifact.get("relative_path")
    _require(
        gap_rel == "masks/gap_timestamps.json",
        f"{manifest_path}: gap_timestamp_artifact relative_path is {gap_rel!r}; the producer writes "
        "'masks/gap_timestamps.json'",
    )
    _require(
        gap_artifact.get("sha256") == checksum_rows[gap_rel]["sha256"],
        f"{manifest_path}: gap_timestamp_artifact sha256 {gap_artifact.get('sha256')!r} != "
        f"file_checksums.csv {checksum_rows[gap_rel]['sha256']!r} -- the package's own metadata disagrees "
        "with itself",
    )

    # The three fixed authoritative artifacts are small and are hashed here,
    # once per qualification, so the checksum manifest is proven against real
    # bytes for everything except the per-basin NetCDFs (which are proven
    # individually, at point of use, by verify_basin_time_series_file).
    for rel in sorted(FIXED_AUTHORITATIVE_CHECKSUM_ENTRIES):
        artifact_path = root / rel
        _require(artifact_path.is_file(), f"{root}: authoritative artifact missing: {rel}")
        row = checksum_rows[rel]
        actual_size = artifact_path.stat().st_size
        _require(
            actual_size == int(row["size_bytes"]),
            f"{artifact_path} is {actual_size} bytes but file_checksums.csv records {row['size_bytes']}",
        )
        actual_digest = sha256_file(artifact_path)
        _require(
            actual_digest == row["sha256"],
            f"{artifact_path} has sha256 {actual_digest} but file_checksums.csv records {row['sha256']} -- "
            "the package's authoritative artifacts do not match its own checksum manifest",
        )

    contract_basin_ids = list(validated["basin_ids"])
    if basin_ids is None:
        qualified_basin_ids = contract_basin_ids
    else:
        qualified_basin_ids = sorted(basin_ids)
        unknown = set(qualified_basin_ids) - set(contract_basin_ids)
        _require(not unknown, f"basin_ids not present in the fixed-support contract: {sorted(unknown)}")
    missing_from_package = sorted(set(qualified_basin_ids) - set(package_basin_ids))
    _require(
        not missing_from_package,
        f"{manifest_path}: contract basins absent from the package's own basin_ids: "
        f"{missing_from_package[:10]} ({len(missing_from_package)} total) -- basin-population identity "
        "contradiction",
    )

    _require(
        manifest["raw_target_variable"] == CANONICAL_OBSERVED_VARIABLE,
        f"{manifest_path}: raw_target_variable is {manifest['raw_target_variable']!r} but the canonical "
        f"RD1-C4 observed series is read from {CANONICAL_OBSERVED_VARIABLE!r}",
    )

    lead_targets = parsed_manifest["lead_targets"]
    matching = [
        entry
        for entry in lead_targets
        if entry.get("name") == validated["target_variable"]
        and entry.get("lead_hours") == validated["lead_hours"]
    ]
    _require(
        len(matching) == 1,
        f"{manifest_path}: expected exactly one lead target with name="
        f"{validated['target_variable']!r} and lead_hours={validated['lead_hours']!r}, found {len(matching)} "
        f"(package lead targets: {[(e.get('name'), e.get('lead_hours')) for e in lead_targets]}) -- "
        "target/lead identity contradiction",
    )

    relative_path = {b: _relative_time_series_path(b) for b in qualified_basin_ids}
    sha256_by_basin = {b: checksum_rows[relative_path[b]]["sha256"] for b in qualified_basin_ids}
    size_by_basin = {b: int(checksum_rows[relative_path[b]]["size_bytes"]) for b in qualified_basin_ids}

    timeline = parsed_manifest["timeline"]
    timeline_window_verified = False
    if validated["date_dtype"] == "datetime64":
        bounds = {
            "package timeline start": _parse_timeline_instant(timeline["start"]),
            "package timeline end": _parse_timeline_instant(timeline["end"]),
            "contract date_start": _parse_timeline_instant(validated["date_start"]),
            "contract date_end": _parse_timeline_instant(validated["date_end"]),
        }
        unparsed = sorted(name for name, value in bounds.items() if value is None)
        _require(
            not unparsed,
            f"{manifest_path}: could not parse {unparsed} as instants for the contract period containment "
            "check -- refusing to claim an unverified period binding",
        )
        _require(
            bounds["contract date_start"] <= bounds["contract date_end"],
            f"contract {validated['contract_id']!r}: date_start {validated['date_start']!r} is after "
            f"date_end {validated['date_end']!r}",
        )
        _require(
            bounds["package timeline start"] <= bounds["contract date_start"]
            and bounds["contract date_end"] <= bounds["package timeline end"],
            f"contract period [{validated['date_start']}, {validated['date_end']}] is not contained in the "
            f"package timeline [{timeline['start']}, {timeline['end']}] -- period identity contradiction",
        )
        timeline_window_verified = True

    return QualifiedPackageIdentity(
        package_root=root.resolve().as_posix(),
        contract_id=validated["contract_id"],
        contract_checksum_sha256=validated["checksum_sha256"],
        contract_schema_name=validated["schema_name"],
        contract_schema_version=int(validated["schema_version"]),
        package_manifest_sha256=observed["package_manifest_sha256"],
        package_file_checksums_sha256=observed["package_file_checksums_sha256"],
        package_run_provenance_sha256=observed["package_run_provenance_sha256"],
        manifest_schema_name=str(manifest["schema_name"]),
        manifest_schema_version=int(manifest["schema_version"]),
        package_role=str(manifest["package_role"]),
        netcdf_package_schema_name=str(netcdf_schema["name"]),
        netcdf_package_schema_version=int(netcdf_schema["version"]),
        netcdf_time_coordinate=str(netcdf_schema["coordinate_name"]),
        netcdf_schema_historical_lineage_applied=historical_lineage_applied,
        run_provenance_builder_module=parsed_provenance["builder_module"],
        run_provenance_created_at_utc=parsed_provenance["created_at_utc"],
        run_provenance_dry_run=parsed_provenance["dry_run"],
        raw_target_variable=str(manifest["raw_target_variable"]),
        target_variable=validated["target_variable"],
        lead_hours=int(validated["lead_hours"]),
        period=validated["period"],
        contract_date_start=validated["date_start"],
        contract_date_end=validated["date_end"],
        timeline_start=str(timeline["start"]),
        timeline_end=str(timeline["end"]),
        timeline_rows=int(timeline["rows"]),
        timeline_frequency=str(timeline["frequency"]),
        timeline_window_verified=timeline_window_verified,
        n_package_basins=len(package_basin_ids),
        n_qualified_basins=len(qualified_basin_ids),
        n_checksum_entries=len(checksum_rows),
        basin_time_series_relative_path=relative_path,
        basin_time_series_sha256=sha256_by_basin,
        basin_time_series_size_bytes=size_by_basin,
        qualified_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def verify_basin_time_series_file(
    identity: QualifiedPackageIdentity,
    basin_id: str,
    *,
    package_root=None,
) -> str:
    """Prove that one basin's package NetCDF still matches the package's own
    recorded checksum, returning the verified SHA-256.

    Call this immediately before consuming that basin's values. ``package_root``
    defaults to ``identity.package_root``; when supplied it must resolve to
    the same directory the identity was qualified against (a caller must not
    be able to qualify one package and then read another).

    This ALWAYS recomputes the digest from disk bytes. There is no memo and
    no ``force`` flag (RD1-C4-D1 finding A5): a "already checked this basin"
    memo silently accepts a file mutated after that check, and a size/mtime
    cache would only recheck metadata, which is not the cryptographic proof
    this function's contract promises. The cost is one full read of the
    basin's NetCDF per call; see this module's docstring for the cost model.

    Raises :class:`PackageIdentityError` for an unqualified basin, a
    disagreeing package root, a missing file, a size disagreement, or a
    checksum disagreement.
    """
    expected = identity.basin_time_series_sha256.get(basin_id)
    _require(
        expected is not None,
        f"basin {basin_id!r} is not part of the qualified package population "
        f"({identity.n_qualified_basins} basins) -- refusing to read an unqualified basin",
    )
    root = Path(identity.package_root) if package_root is None else Path(package_root)
    _require(
        root.resolve().as_posix() == identity.package_root,
        f"package root {root} does not match the qualified package root {identity.package_root} -- "
        "refusing to read values from a package that was never qualified",
    )
    rel = identity.basin_time_series_relative_path[basin_id]
    path = root / rel
    _require(path.is_file(), f"basin {basin_id!r}: package time series missing: {path}")
    actual_size = path.stat().st_size
    expected_size = identity.basin_time_series_size_bytes[basin_id]
    _require(
        actual_size == expected_size,
        f"basin {basin_id!r}: {path} is {actual_size} bytes but the package records {expected_size}",
    )
    actual = sha256_file(path)
    _require(
        actual == expected,
        f"basin {basin_id!r}: {path} has sha256 {actual} but the package records {expected} -- "
        "the consumed basin NetCDF is not the file this package was built from",
    )
    return actual
