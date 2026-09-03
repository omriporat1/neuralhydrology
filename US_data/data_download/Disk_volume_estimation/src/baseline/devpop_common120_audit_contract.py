"""Full-development-population Common-120 audit contract family (SHARED-A1).

This module is the *foundation* for the [SHARED] screening-400 vs
full-development-population audit.  It defines a **separate, immutable**
contract/identity family whose only job is to represent -- and later let a
qualified builder construct -- one fixed Common-120 support over an
*explicitly identified* evaluation population (the approved 2,307-basin
development population, ``config/stage1_baseline_splits_v001/development_train.txt``).

It is deliberately **not** part of the v2 optimizer path:

* the diagnostic contract identifier is
  :data:`DEVPOP_AUDIT_CONTRACT_ID` (``common120_raw_space_nse_devpop_audit_v001``),
  which is asserted at import time to differ from
  :data:`~src.baseline.sweep_v2_six_axis_campaign.OBJECTIVE_ID_V2`
  (``common120_raw_space_nse_v001``);
* the contract carries its own schema name
  (:data:`DEVPOP_AUDIT_SCHEMA_NAME`), so
  :func:`~src.baseline.fixed_support_contract_v2.validate_fixed_support_contract`
  structurally rejects it (wrong ``schema_name`` / ``contract_id`` / extra
  keys);
* an audit *result* built against this contract is tagged
  ``objective_scope="devpop_audit"``, so
  :func:`~src.baseline.fixed_support_contract_v2.extract_v2_objective_from_fixed_support_result`
  (which only accepts ``"fixed_support"``) structurally refuses it;
* this module exposes **no** function that returns a bare optimizer-objective
  float and **no** ``flashnh/``-prefixed W&B metric name, and imports nothing
  from the W&B bridge / sweep-execution layer.

Scope of *this* milestone (SHARED-A1): schema + explicit population identity +
generalized strict completeness validation + a synthetic-only builder
foundation.  It does **not** run checkpoint inference, prepare an
evaluation run, emit audit rows, contact W&B/Moriah/Slurm, or construct the
real 2,307-basin artifact.  Every code path here is exercised only against
synthetic in-memory fixtures.

The scientific predicate is unchanged and is **not** re-implemented here: the
frozen Common-120 membership rule
(``history_valid_120(t) AND validation_start <= t <= validation_end AND
t + 6h <= validation_end AND finite(qobs_mm_per_h_lead06[b, t])``) and its
monotone-nesting justification across v2 ``seq_length`` 48--120 are reused
verbatim from :mod:`src.baseline.validity_mask` /
:mod:`src.baseline.fixed_support_contract_v2` /
:mod:`src.baseline.common120_support_builder`.  Only the *population* over
which that fixed predicate is applied is generalized -- from the frozen 400
screening basins to an explicitly identified, checksum-pinned development
population.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

# Neutral serialization / accounting primitives shared by the whole
# fixed-support contract family (date-array (de)serialization, canonical
# checksum body, strict-integer accounting).  These are pure
# encoding/accounting helpers -- not scientific math, not population
# semantics -- exposed additively as public names by
# :mod:`src.baseline.fixed_support_contract_v2`; reusing them keeps the two
# contract families byte-compatible in how they persist per-basin support and
# how they checksum, without importing an underscore-prefixed name across a
# package boundary.  The frozen screening module itself is left untouched.
from .fixed_support_contract_v2 import (
    CONTRACT_SCHEMA_NAME as _FIXED_SUPPORT_SCHEMA_NAME,
    FixedSupportContractError as _FixedSupportContractError,
    canonical_contract_checksum_payload,
    deserialize_support_date_array,
    is_strict_int,
    serialize_support_date_array,
    strict_int as _shared_strict_int,
)
from .sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2, SEQ_LENGTH_MAX

__all__ = [
    "DevpopAuditContractError",
    "DevpopAuditCompletenessError",
    "DEVPOP_AUDIT_CONTRACT_ID",
    "DEVPOP_AUDIT_SCHEMA_NAME",
    "DEVPOP_AUDIT_SCHEMA_VERSION",
    "AUDIT_PERIOD_NAME",
    "AUDIT_DATE_MIN",
    "AUDIT_DATE_MAX",
    "ALLOWED_POPULATION_ROLES",
    "AUDIT_OBJECTIVE_SCOPE",
    "DEVELOPMENT_TRAIN_ROLE",
    "EXPECTED_DEVELOPMENT_POPULATION_SIZE",
    "DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256",
    "DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256",
    "CANONICAL_TARGET_VARIABLE",
    "CANONICAL_LEAD_HOURS",
    "CANONICAL_SEQ_LENGTH_FLOOR",
    "CANONICAL_SOURCE_GAP_POLICY_IDENTITY",
    "CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE",
    "canonical_membership_sha256",
    "ExpectedPopulationSpec",
    "assert_canonical_development_population",
    "build_devpop_audit_contract",
    "validate_devpop_audit_contract",
    "validate_canonical_devpop_audit_contract",
    "write_devpop_audit_contract",
    "write_synthetic_devpop_audit_contract",
    "load_devpop_audit_contract",
    "load_synthetic_devpop_audit_contract",
    "deserialize_contract_support",
    "require_complete_devpop_audit_population",
    "require_complete_synthetic_devpop_audit_population",
]


class DevpopAuditContractError(ValueError):
    """Raised for a malformed/inconsistent development-population Common-120
    audit contract or expected-population specification, an attempted
    overwrite, a checksum contradiction, or an attempt to use a sealed
    period/role.  Never raised for an ordinary poor-skill outcome."""


class DevpopAuditCompletenessError(DevpopAuditContractError):
    """Raised when a full-development-population audit result fails canonical
    completeness: any missing, extra, duplicated, or excluded basin, any
    non-finite per-basin NSE, any non-finite simulation at an admitted
    timestamp, or any population/checksum identity contradiction.  This
    condition must block canonical completeness pending investigation -- it
    must never silently drop a basin."""


# --------------------------------------------------------------------------- #
# Identity constants -- distinct from every v2 optimizer identifier.
# --------------------------------------------------------------------------- #

#: Diagnostic-only audit contract identifier.  MUST NOT become
#: ``OBJECTIVE_ID_V2``, the v2 W&B metric name, or a selectable optimizer
#: objective.
DEVPOP_AUDIT_CONTRACT_ID = "common120_raw_space_nse_devpop_audit_v001"
DEVPOP_AUDIT_SCHEMA_NAME = "flashnh_stage1_devpop_common120_audit_contract"
DEVPOP_AUDIT_SCHEMA_VERSION = 1

#: The ``objective_scope`` tag an audit *result* carries.  Deliberately not
#: ``"fixed_support"``, so the v2 objective extractor structurally refuses it.
AUDIT_OBJECTIVE_SCOPE = "devpop_audit"

AUDIT_PERIOD_NAME = "validation"
AUDIT_DATE_MIN = "2024-01-01"
AUDIT_DATE_MAX = "2024-12-31"

DEVELOPMENT_TRAIN_ROLE = "development_train"
#: Population roles this audit family will bind to.  A tiny allow-list: any
#: sealed / holdout / California / temporal-test role is structurally
#: impossible to express.
ALLOWED_POPULATION_ROLES = frozenset({DEVELOPMENT_TRAIN_ROLE})

EXPECTED_DEVELOPMENT_POPULATION_SIZE = 2307

#: **Portable, line-ending-independent** canonical membership identity: the
#: SHA-256 of the canonical membership representation (sorted-unique STAIDs,
#: one per line, ``\n``-terminated -- see :func:`canonical_membership_sha256`).
#: This is the *runtime scientific identity* of the development population and
#: is what every canonical path pins.  It is byte-identical whether the
#: committed ``development_train.txt`` is checked out with LF or CRLF endings,
#: because it is derived from the parsed STAID set, not from file bytes.  It
#: equals the committed ``development_split_sha256`` already used by the v2
#: fixed-support contract family
#: (``sweep_v1_production_adapter.DEVELOPMENT_SPLIT_SHA256``), which is the
#: LF blob hash of that same canonical form.
DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256 = (
    "397ab432564c18c3abc5158a47ada2b28840bbf6f0c213d2475444fded33858f"
)
#: **Named provenance only -- NOT a portable identity.**  The value recorded
#: at ``artifact_sha256["development_train.txt"]`` in the committed
#: ``config/stage1_baseline_splits_v001/split_manifest.json``.  That manifest
#: was generated on a platform whose working-tree checkout materialized the
#: file with CRLF endings, so this hash identifies *that specific
#: CRLF-materialized representation* of the split artifact.  It is used only
#: to assert that the committed split manifest still records the value we
#: pinned (a manifest-consistency check); it is **never** compared against a
#: hash of whichever ``development_train.txt`` bytes happen to exist in the
#: current working tree, and canonical validation never requires a CRLF
#: materialization.
DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256 = (
    "b71051bc0c8f49627a6a0919e7cfe524e64740f29cd01f8b850b54915487e23e"
)

#: **Exact pinned structured provenance** for the canonical development
#: membership artifact.  Every canonical population / contract must carry this
#: *object* (compared by value) as ``membership_artifact_provenance`` -- never a
#: merely non-empty string.  It binds, at minimum, the authoritative
#: split-manifest path and its recorded artifact identity
#: (``artifact_sha256["development_train.txt"]``), and the development
#: membership list path + role.  A null, free-form, or partial provenance is
#: rejected by :func:`assert_canonical_development_population` and
#: :func:`validate_canonical_devpop_audit_contract`.
CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE = {
    "split_manifest_path": "config/stage1_baseline_splits_v001/split_manifest.json",
    "split_manifest_artifact_key": "development_train.txt",
    "split_manifest_recorded_artifact_sha256": DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256,
    "development_membership_path": "config/stage1_baseline_splits_v001/development_train.txt",
    "development_membership_role": DEVELOPMENT_TRAIN_ROLE,
    "representation_note": (
        "split-manifest artifact_sha256 identifies a CRLF-materialized representation of "
        "development_train.txt; named provenance only, not a portable identity. The portable "
        "scientific identity is the line-ending-independent canonical membership hash."
    ),
}

#: Scientific-identity constants every *canonical* development-population
#: Common-120 audit path pins.  Reused from the authoritative v2 domain
#: constant where one exists (``SEQ_LENGTH_MAX``); the target/lead/gap-policy
#: identities mirror the literals already used by
#: :mod:`src.baseline.common120_support_builder` for the frozen screening
#: contract, kept here as the single shared source of truth for the audit
#: family.
CANONICAL_TARGET_VARIABLE = "qobs_mm_per_h_lead06"
CANONICAL_LEAD_HOURS = 6
CANONICAL_SEQ_LENGTH_FLOOR = SEQ_LENGTH_MAX  # 120h fixed floor across all v2 seq_length
CANONICAL_SOURCE_GAP_POLICY_IDENTITY = "stage1_policy_mrms_rtma_history_v001"

_MONOTONE_NESTING_JUSTIFICATION = (
    "A timestamp admitted by history_valid_120(t) has a gap-free 120h lookback "
    "window; every shorter v2 seq_length (48..120) lookback window is a contiguous "
    "sub-window of that gap-free window, hence also gap-free -- so the 120h-floor "
    "admitted set is a valid common support for every v2 seq_length candidate."
)

# Substrings that must never appear in a bound population role: this is a
# structural guard, not a reliance on caller discipline.
_SEALED_ROLE_MARKERS = (
    "test",
    "holdout",
    "spatial",
    "california",
    "calif",
    "_ca",
    "ca_",
    "sealed",
    "2025",
    "temporal_test",
)

_HEX = frozenset("0123456789abcdef")

# Import-time structural guards: the diagnostic identity can never collapse
# into the optimizer identity.
assert DEVPOP_AUDIT_CONTRACT_ID != OBJECTIVE_ID_V2
assert DEVPOP_AUDIT_SCHEMA_NAME != _FIXED_SUPPORT_SCHEMA_NAME
assert not DEVPOP_AUDIT_CONTRACT_ID.startswith("flashnh/")
assert AUDIT_OBJECTIVE_SCOPE != "fixed_support"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(c in _HEX for c in value)
    )


def strict_int(value: object, *, name: str, minimum: Optional[int] = None) -> int:
    """Strict non-bool / non-float integer guard for this contract family.

    Delegates to the shared
    :func:`~src.baseline.fixed_support_contract_v2.strict_int` primitive (so the
    two contract families use identical accounting semantics) but re-raises its
    failure as a :class:`DevpopAuditContractError`, keeping every
    contract/completeness validation failure inside this module's own error
    taxonomy rather than leaking a sibling ``FixedSupportContractError``.
    """
    try:
        return _shared_strict_int(value, name=name, minimum=minimum)
    except _FixedSupportContractError as exc:
        raise DevpopAuditContractError(str(exc)) from exc


def canonical_membership_sha256(basin_ids: Sequence[str]) -> str:
    """SHA-256 of the line-ending-independent canonical membership
    representation of ``basin_ids`` (one STAID per line, ``\\n``-terminated),
    computed on the *sorted unique* set.  This makes it structurally
    impossible for a spec to carry the right hash but duplicate/unsorted IDs:
    the hash is derived from the canonical form, and callers must supply IDs
    already in that form (validated separately)."""
    ordered = sorted({str(b) for b in basin_ids})
    body = "".join(f"{b}\n" for b in ordered).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _parse_exact_iso_date(label: str, value: object) -> date:
    """Parse ``value`` as an ISO ``YYYY-MM-DD`` string using a real calendar
    parser.  Rejects a non-string, a wrong-length / wrong-shaped string, and --
    via :meth:`datetime.date.fromisoformat` -- every impossible or truncated
    date (``2024-02-31``, ``2024-13-01``, ``2024-1-1``, ``2024-02``)."""
    if not isinstance(value, str) or len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise DevpopAuditContractError(f"{label} must be an ISO 'YYYY-MM-DD' string, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DevpopAuditContractError(f"{label} is not a real calendar date: {value!r} ({exc})") from exc


def _reject_sealed_scope(*, role: str, period: str, date_start: str, date_end: str) -> None:
    """Structurally reject a sealed evaluation scope.  Rejects: any role not
    on :data:`ALLOWED_POPULATION_ROLES`; any role containing a sealed marker
    (``holdout``/``california``/``spatial``/``test``/``2025``/...); any period
    other than ``validation`` (so ``test`` is impossible); any ``date_start`` /
    ``date_end`` that is not *exactly* the frozen development-validation window
    ``2024-01-01``--``2024-12-31`` (so a sub-interval, a shifted window, an
    impossible date, or any 2025 / temporal-test date is impossible)."""
    if not isinstance(role, str) or role not in ALLOWED_POPULATION_ROLES:
        raise DevpopAuditContractError(
            f"population role {role!r} is not on the audit allow-list {sorted(ALLOWED_POPULATION_ROLES)} "
            "-- spatial-holdout, California and temporal-test roles are structurally forbidden"
        )
    lowered = role.lower()
    hit = [m for m in _SEALED_ROLE_MARKERS if m in lowered]
    if hit:
        raise DevpopAuditContractError(f"population role {role!r} contains sealed-scope marker(s) {hit}")

    if period != AUDIT_PERIOD_NAME:
        raise DevpopAuditContractError(
            f"period must be {AUDIT_PERIOD_NAME!r} (development validation); got {period!r} "
            "-- temporal-test / other periods are structurally forbidden"
        )

    parsed_start = _parse_exact_iso_date("date_start", date_start)
    parsed_end = _parse_exact_iso_date("date_end", date_end)
    if parsed_start != date.fromisoformat(AUDIT_DATE_MIN) or parsed_end != date.fromisoformat(AUDIT_DATE_MAX):
        raise DevpopAuditContractError(
            f"development-population audit is defined only over the full frozen validation year "
            f"[{AUDIT_DATE_MIN}, {AUDIT_DATE_MAX}]; got [{date_start}, {date_end}] "
            "-- sub-intervals and shifted windows are structurally forbidden"
        )


# --------------------------------------------------------------------------- #
# Explicit expected-population identity
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExpectedPopulationSpec:
    """Immutable, self-verifying identity of the evaluation population a
    development-population Common-120 audit contract binds to.

    Every scientifically meaningful fact is explicit and cross-checked, so it
    is structurally impossible to validate:

    * the 400-basin screening contract as this audit contract (different
      ``contract_id`` and ``schema_name``);
    * a population with the right size but different basin identities
      (``membership_ids_sha256`` is derived from the sorted-unique IDs);
    * a population with the right hash but duplicate / unsorted IDs
      (IDs are re-checked to be sorted-unique here);
    * a contract for a sealed period or population role
      (:func:`_reject_sealed_scope`).
    """

    role: str
    expected_size: int
    basin_ids: tuple
    membership_ids_sha256: str
    period: str = AUDIT_PERIOD_NAME
    date_start: str = AUDIT_DATE_MIN
    date_end: str = AUDIT_DATE_MAX
    contract_id: str = DEVPOP_AUDIT_CONTRACT_ID
    membership_artifact_sha256: Optional[str] = None
    #: Named provenance for the membership artifact.  For a canonical population
    #: this MUST be the exact pinned structured block
    #: :data:`CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE` (a mapping); a synthetic
    #: fixture may leave it ``None`` or pass an arbitrary marker.
    membership_artifact_provenance: Optional[object] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "basin_ids", tuple(str(b) for b in self.basin_ids))

        if self.contract_id != DEVPOP_AUDIT_CONTRACT_ID or self.contract_id == OBJECTIVE_ID_V2:
            raise DevpopAuditContractError(
                f"expected-population contract_id must be {DEVPOP_AUDIT_CONTRACT_ID!r}, got {self.contract_id!r}"
            )
        _reject_sealed_scope(
            role=self.role, period=self.period, date_start=self.date_start, date_end=self.date_end
        )

        if not self.basin_ids:
            raise DevpopAuditContractError("expected population is empty")
        ordered = sorted(self.basin_ids)
        if list(self.basin_ids) != ordered:
            raise DevpopAuditContractError("basin_ids must be supplied sorted")
        if len(set(self.basin_ids)) != len(self.basin_ids):
            dupes = sorted({b for b in self.basin_ids if self.basin_ids.count(b) > 1})
            raise DevpopAuditContractError(f"basin_ids contains duplicate identities: {dupes}")
        if not all(b and not b.isspace() for b in self.basin_ids):
            raise DevpopAuditContractError("basin_ids contains an empty identity")

        if not isinstance(self.expected_size, int) or isinstance(self.expected_size, bool):
            raise DevpopAuditContractError("expected_size must be an int")
        if self.expected_size != len(self.basin_ids):
            raise DevpopAuditContractError(
                f"expected_size={self.expected_size} does not match len(basin_ids)={len(self.basin_ids)}"
            )

        if not _is_sha256(self.membership_ids_sha256):
            raise DevpopAuditContractError("membership_ids_sha256 must be a lowercase 64-char SHA-256")
        recomputed = canonical_membership_sha256(self.basin_ids)
        if recomputed != self.membership_ids_sha256:
            raise DevpopAuditContractError(
                f"membership_ids_sha256 {self.membership_ids_sha256} does not match the canonical "
                f"membership hash of the supplied basin_ids ({recomputed})"
            )
        if self.membership_artifact_sha256 is not None and not _is_sha256(self.membership_artifact_sha256):
            raise DevpopAuditContractError("membership_artifact_sha256 must be null or a lowercase 64-char SHA-256")

    # -- construction helpers ------------------------------------------------ #

    @classmethod
    def for_synthetic_fixture(
        cls,
        *,
        role: str,
        basin_ids: Sequence[str],
        membership_artifact_sha256: Optional[str] = None,
        membership_artifact_provenance: Optional[object] = None,
        period: str = AUDIT_PERIOD_NAME,
        date_start: str = AUDIT_DATE_MIN,
        date_end: str = AUDIT_DATE_MAX,
    ) -> "ExpectedPopulationSpec":
        """**GENERIC / test-only.**  Build a spec from an arbitrary explicit
        basin-ID list, deriving ``membership_ids_sha256`` and ``expected_size``
        from the sorted-unique set (a fixture convenience -- arbitrary input is
        canonicalized here on purpose).

        This constructor is **never** a canonical production population: the
        resulting spec will *not* carry the pinned 2,307-basin membership hash
        unless the caller happens to pass exactly that set, and
        :func:`assert_canonical_development_population` (and every canonical
        builder / gate that calls it) independently re-checks the pinned
        identity, so a synthetic spec cannot be smuggled into a canonical path.
        Use :meth:`for_development_train` for the real population."""
        ordered = sorted({str(b) for b in basin_ids})
        return cls(
            role=role,
            expected_size=len(ordered),
            basin_ids=tuple(ordered),
            membership_ids_sha256=canonical_membership_sha256(ordered),
            period=period,
            date_start=date_start,
            date_end=date_end,
            membership_artifact_sha256=membership_artifact_sha256,
            membership_artifact_provenance=membership_artifact_provenance,
        )

    @classmethod
    def for_development_train(cls, splits_dir) -> "ExpectedPopulationSpec":
        """**CANONICAL.**  Load the approved 2,307-basin development population
        from ``<splits_dir>/development_train.txt``, cross-checking it against
        the committed ``split_manifest.json`` recorded artifact hash, the
        recorded count, and the pinned **line-ending-independent** canonical
        membership hash (:data:`DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256`).

        The parsed STAID set -- not the file bytes -- is the scientific
        identity: :func:`~src.baseline.splits.load_eligible_basins` normalizes
        line endings, so this succeeds identically whether the committed file
        is checked out LF or CRLF.  The manifest's recorded
        ``artifact_sha256`` (a CRLF-materialized representation on the manifest
        generator's platform) is retained only as *named provenance*.  Reads
        only committed split artifacts -- never the scientific package or any
        sealed scope.  The returned spec is additionally run through
        :func:`assert_canonical_development_population` before it is handed
        back."""
        # Local import: the split reader lives in the split-generation module
        # and pulls pandas; keep this module import-light for the common path.
        from .splits import load_eligible_basins

        splits_dir = Path(splits_dir)
        manifest_path = splits_dir / "split_manifest.json"
        train_path = splits_dir / "development_train.txt"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DevpopAuditContractError(f"could not read split manifest {manifest_path}: {exc}") from exc
        recorded = (manifest.get("artifact_sha256") or {}).get("development_train.txt")
        if recorded != DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256:
            raise DevpopAuditContractError(
                "split_manifest.json development_train.txt recorded artifact hash "
                f"({recorded}) does not match the pinned split-manifest provenance value "
                f"({DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256}) -- the committed split "
                "manifest changed; canonical membership identity is verified separately below"
            )
        if manifest.get("counts", {}).get("development_train") != EXPECTED_DEVELOPMENT_POPULATION_SIZE:
            raise DevpopAuditContractError(
                "split_manifest.json does not record exactly "
                f"{EXPECTED_DEVELOPMENT_POPULATION_SIZE} development_train basins"
            )
        try:
            basin_ids = load_eligible_basins(train_path)
        except Exception as exc:  # SplitGenerationError and friends
            raise DevpopAuditContractError(f"could not read {train_path}: {exc}") from exc
        if len(basin_ids) != EXPECTED_DEVELOPMENT_POPULATION_SIZE:
            raise DevpopAuditContractError(
                f"{train_path} has {len(basin_ids)} basins, expected {EXPECTED_DEVELOPMENT_POPULATION_SIZE}"
            )
        canonical = canonical_membership_sha256(basin_ids)
        if canonical != DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256:
            raise DevpopAuditContractError(
                f"canonical (line-ending-independent) membership hash of {train_path} ({canonical}) "
                f"does not match the pinned audit anchor ({DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256})"
            )
        spec = cls(
            role=DEVELOPMENT_TRAIN_ROLE,
            expected_size=EXPECTED_DEVELOPMENT_POPULATION_SIZE,
            basin_ids=tuple(sorted(basin_ids)),
            membership_ids_sha256=canonical,
            membership_artifact_sha256=DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256,
            membership_artifact_provenance=dict(CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE),
        )
        assert_canonical_development_population(spec)
        return spec

    def identity_payload(self) -> dict:
        """The population-identity block embedded verbatim into a contract."""
        return {
            "population_role": self.role,
            "expected_population_size": self.expected_size,
            "membership_ids_sha256": self.membership_ids_sha256,
            "membership_artifact_sha256": self.membership_artifact_sha256,
            "membership_artifact_provenance": self.membership_artifact_provenance,
            "basin_ids": list(self.basin_ids),
            "period": self.period,
            "date_start": self.date_start,
            "date_end": self.date_end,
        }


def assert_canonical_development_population(population: object) -> None:
    """Fail unless ``population`` is the **canonical** 2,307-basin development
    population, pinned on every scientifically meaningful axis.

    This is the gate that every public API *named* as canonical
    development-population audit behaviour must pass its population through.  A
    generic / synthetic :meth:`ExpectedPopulationSpec.for_synthetic_fixture`
    spec (or any hand-built spec) fails here unless it happens to reproduce the
    exact pinned identity -- which a synthetic fixture cannot do without
    actually being the real set.

    Pins: type; ``role == 'development_train'``; ``contract_id`` is the
    diagnostic audit id (never ``OBJECTIVE_ID_V2``); ``expected_size == 2307``;
    sorted-unique non-blank ``basin_ids``; recorded and recomputed
    line-ending-independent canonical membership hash both equal
    :data:`DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256`; ``period == 'validation'``
    and the frozen full-year 2024 window (real calendar parse);
    ``membership_artifact_provenance`` byte-equal to the pinned structured block
    :data:`CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE` (never a merely non-empty
    string, never ``None``); and ``membership_artifact_sha256`` exactly equal to
    the recorded split-manifest provenance value
    :data:`DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256` (never ``None``,
    never used as a portable identity)."""
    if not isinstance(population, ExpectedPopulationSpec):
        raise DevpopAuditContractError(
            f"canonical development population must be an ExpectedPopulationSpec, got {type(population).__name__}"
        )
    if population.role != DEVELOPMENT_TRAIN_ROLE:
        raise DevpopAuditContractError(
            f"canonical development population role must be {DEVELOPMENT_TRAIN_ROLE!r}, got {population.role!r}"
        )
    if population.contract_id != DEVPOP_AUDIT_CONTRACT_ID or population.contract_id == OBJECTIVE_ID_V2:
        raise DevpopAuditContractError("canonical development population contract_id must be the diagnostic audit id")
    if population.expected_size != EXPECTED_DEVELOPMENT_POPULATION_SIZE:
        raise DevpopAuditContractError(
            f"canonical development population must have exactly {EXPECTED_DEVELOPMENT_POPULATION_SIZE} basins, "
            f"got {population.expected_size}"
        )
    ids = list(population.basin_ids)
    if len(ids) != EXPECTED_DEVELOPMENT_POPULATION_SIZE:
        raise DevpopAuditContractError(
            f"canonical development population basin_ids has {len(ids)} entries, "
            f"expected {EXPECTED_DEVELOPMENT_POPULATION_SIZE}"
        )
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise DevpopAuditContractError("canonical development population basin_ids must be sorted and unique")
    if not all(isinstance(b, str) and b and not b.isspace() for b in ids):
        raise DevpopAuditContractError("canonical development population basin_ids must all be non-blank strings")
    if population.membership_ids_sha256 != DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256:
        raise DevpopAuditContractError(
            f"canonical development population membership hash {population.membership_ids_sha256} "
            f"does not match the pinned anchor {DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256}"
        )
    recomputed = canonical_membership_sha256(ids)
    if recomputed != DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256:
        raise DevpopAuditContractError(
            f"recomputed canonical membership hash of the supplied basin_ids ({recomputed}) "
            f"does not match the pinned anchor {DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256} "
            "-- basin_ids are not the canonical development population"
        )
    if population.period != AUDIT_PERIOD_NAME:
        raise DevpopAuditContractError(
            f"canonical development population period must be {AUDIT_PERIOD_NAME!r}, got {population.period!r}"
        )
    if (
        _parse_exact_iso_date("date_start", population.date_start) != date.fromisoformat(AUDIT_DATE_MIN)
        or _parse_exact_iso_date("date_end", population.date_end) != date.fromisoformat(AUDIT_DATE_MAX)
    ):
        raise DevpopAuditContractError(
            f"canonical development population window must be exactly [{AUDIT_DATE_MIN}, {AUDIT_DATE_MAX}], "
            f"got [{population.date_start}, {population.date_end}]"
        )
    if population.membership_artifact_provenance != CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE:
        raise DevpopAuditContractError(
            "canonical development population must carry the exact pinned structured "
            "membership_artifact_provenance block (split-manifest path + artifact key + recorded "
            "artifact hash + development membership path + role); a null, free-form, or partial "
            f"provenance is not canonical -- got {population.membership_artifact_provenance!r}"
        )
    if population.membership_artifact_sha256 != DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256:
        raise DevpopAuditContractError(
            "canonical development population membership_artifact_sha256 must record the split-manifest "
            f"development_train.txt provenance value ({DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256}); "
            f"got {population.membership_artifact_sha256!r} -- a null or arbitrary artifact hash is not canonical"
        )


# --------------------------------------------------------------------------- #
# Audit contract: schema / build / validate / write / load
# --------------------------------------------------------------------------- #

_REQUIRED_KEYS = frozenset({
    "schema_name",
    "schema_version",
    "contract_id",
    "diagnostic_only",
    "not_an_optimizer_objective",
    "objective_scope",
    "population_role",
    "expected_population_size",
    "membership_ids_sha256",
    "membership_artifact_sha256",
    "membership_artifact_provenance",
    "basin_ids",
    "seq_length_floor",
    "lead_hours",
    "target_variable",
    "period",
    "date_start",
    "date_end",
    "source_gap_policy_identity",
    "package_manifest_sha256",
    "package_file_checksums_sha256",
    "package_run_provenance_sha256",
    "development_split_sha256",
    "spatial_holdout_split_sha256",
    "monotone_nesting_justification",
    "date_dtype",
    "per_basin_support",
    "eligible_counts",
    "checksum_sha256",
})

_PACKAGE_SPLIT_HASH_KEYS = (
    "package_manifest_sha256",
    "package_file_checksums_sha256",
    "package_run_provenance_sha256",
    "development_split_sha256",
    "spatial_holdout_split_sha256",
)


def build_devpop_audit_contract(
    *,
    population: ExpectedPopulationSpec,
    target_variable: str,
    source_gap_policy_identity: str,
    per_basin_date: Mapping[str, np.ndarray],
    per_basin_admitted: Mapping[str, np.ndarray],
    package_manifest_sha256: str,
    package_file_checksums_sha256: str,
    package_run_provenance_sha256: str,
    development_split_sha256: str,
    spatial_holdout_split_sha256: str,
    lead_hours: int = CANONICAL_LEAD_HOURS,
    seq_length_floor: int = CANONICAL_SEQ_LENGTH_FLOOR,
) -> dict:
    """Build (but never write) a development-population Common-120 audit
    contract payload from per-basin ``(date coordinate, boolean 120h-floor
    admitted mask)`` pairs.  ``per_basin_admitted[b]`` is expected to be
    ``compute_history_valid(index, bad_hour_mask, 120) &
    (validation window) & isfinite(qobs[b])`` -- the frozen predicate, applied
    to every basin of ``population`` independently.  Persists only the
    admitted subset of each basin's date coordinate plus its count.

    The scientific-identity fields are pinned, not merely well-formed:
    ``target_variable`` must be :data:`CANONICAL_TARGET_VARIABLE`,
    ``lead_hours`` :data:`CANONICAL_LEAD_HOURS`, ``seq_length_floor``
    :data:`CANONICAL_SEQ_LENGTH_FLOOR` (the fixed 120h floor across every v2
    ``seq_length``), and ``source_gap_policy_identity``
    :data:`CANONICAL_SOURCE_GAP_POLICY_IDENTITY`.

    Raises :class:`DevpopAuditContractError` if the supplied per-basin maps do
    not cover exactly ``population.basin_ids``, or if any basin has zero
    support (every such basin is named)."""
    if not isinstance(population, ExpectedPopulationSpec):
        raise DevpopAuditContractError("population must be an ExpectedPopulationSpec")
    if target_variable != CANONICAL_TARGET_VARIABLE:
        raise DevpopAuditContractError(
            f"target_variable must be {CANONICAL_TARGET_VARIABLE!r}, got {target_variable!r}"
        )
    if strict_int(lead_hours, name="lead_hours") != CANONICAL_LEAD_HOURS:
        raise DevpopAuditContractError(f"lead_hours must be {CANONICAL_LEAD_HOURS}, got {lead_hours!r}")
    if strict_int(seq_length_floor, name="seq_length_floor") != CANONICAL_SEQ_LENGTH_FLOOR:
        raise DevpopAuditContractError(
            f"seq_length_floor must be {CANONICAL_SEQ_LENGTH_FLOOR} (fixed 120h floor), got {seq_length_floor!r}"
        )
    if source_gap_policy_identity != CANONICAL_SOURCE_GAP_POLICY_IDENTITY:
        raise DevpopAuditContractError(
            f"source_gap_policy_identity must be {CANONICAL_SOURCE_GAP_POLICY_IDENTITY!r}, "
            f"got {source_gap_policy_identity!r}"
        )
    supplied = sorted(per_basin_date)
    if supplied != sorted(per_basin_admitted):
        raise DevpopAuditContractError("per_basin_date and per_basin_admitted must share the same basin_id set")
    if tuple(supplied) != population.basin_ids:
        missing = sorted(set(population.basin_ids) - set(supplied))
        extra = sorted(set(supplied) - set(population.basin_ids))
        raise DevpopAuditContractError(
            f"per-basin support does not cover exactly the expected population: missing={missing} extra={extra}"
        )

    date_dtype: Optional[str] = None
    per_basin_support: dict = {}
    eligible_counts: dict = {}
    zero_support: list = []
    for basin_id in population.basin_ids:
        date_values = np.asarray(per_basin_date[basin_id])
        admitted = np.asarray(per_basin_admitted[basin_id], dtype=bool)
        if admitted.shape != date_values.shape:
            raise DevpopAuditContractError(
                f"basin {basin_id!r}: admitted mask shape {admitted.shape} != date shape {date_values.shape}"
            )
        n_admitted = int(admitted.sum())
        if n_admitted == 0:
            zero_support.append(basin_id)
            continue
        serialized, this_dtype = serialize_support_date_array(date_values[admitted])
        if date_dtype is None:
            date_dtype = this_dtype
        elif this_dtype != date_dtype:
            raise DevpopAuditContractError(
                f"basin {basin_id!r}: date dtype {this_dtype!r} != established {date_dtype!r}"
            )
        per_basin_support[basin_id] = serialized
        eligible_counts[basin_id] = n_admitted

    if zero_support:
        raise DevpopAuditContractError(
            f"{len(zero_support)} basin(s) have zero Common-120 support and are named explicitly: "
            f"{sorted(zero_support)}"
        )

    for label, value in (
        ("package_manifest_sha256", package_manifest_sha256),
        ("package_file_checksums_sha256", package_file_checksums_sha256),
        ("package_run_provenance_sha256", package_run_provenance_sha256),
        ("development_split_sha256", development_split_sha256),
        ("spatial_holdout_split_sha256", spatial_holdout_split_sha256),
    ):
        if not _is_sha256(value):
            raise DevpopAuditContractError(f"{label} must be a lowercase 64-char SHA-256")

    payload = {
        "schema_name": DEVPOP_AUDIT_SCHEMA_NAME,
        "schema_version": DEVPOP_AUDIT_SCHEMA_VERSION,
        "contract_id": DEVPOP_AUDIT_CONTRACT_ID,
        "diagnostic_only": True,
        "not_an_optimizer_objective": True,
        "objective_scope": AUDIT_OBJECTIVE_SCOPE,
        "seq_length_floor": int(seq_length_floor),
        "lead_hours": int(lead_hours),
        "target_variable": target_variable,
        "source_gap_policy_identity": source_gap_policy_identity,
        "package_manifest_sha256": package_manifest_sha256,
        "package_file_checksums_sha256": package_file_checksums_sha256,
        "package_run_provenance_sha256": package_run_provenance_sha256,
        "development_split_sha256": development_split_sha256,
        "spatial_holdout_split_sha256": spatial_holdout_split_sha256,
        "monotone_nesting_justification": _MONOTONE_NESTING_JUSTIFICATION,
        "date_dtype": date_dtype,
        "per_basin_support": per_basin_support,
        "eligible_counts": eligible_counts,
        **population.identity_payload(),
    }
    payload["checksum_sha256"] = hashlib.sha256(canonical_contract_checksum_payload(payload)).hexdigest()
    return validate_devpop_audit_contract(payload)


def validate_devpop_audit_contract(data: dict) -> dict:
    """Strict schema / identity / checksum validation.  Raises
    :class:`DevpopAuditContractError` on any missing/extra key, wrong schema
    or contract identity, an optimizer-objective identity, a sealed
    period/role, a population size/hash/membership contradiction, an
    inconsistent per-basin support map, a zero-support basin, or a checksum
    mismatch."""
    if not isinstance(data, dict):
        raise DevpopAuditContractError(f"contract must be a mapping, got {type(data).__name__}")
    missing = _REQUIRED_KEYS - set(data)
    extra = set(data) - _REQUIRED_KEYS
    if missing:
        raise DevpopAuditContractError(f"contract missing required key(s): {sorted(missing)}")
    if extra:
        raise DevpopAuditContractError(f"contract has unexpected extra key(s): {sorted(extra)}")

    if data["schema_name"] != DEVPOP_AUDIT_SCHEMA_NAME:
        raise DevpopAuditContractError(
            f"schema_name must be {DEVPOP_AUDIT_SCHEMA_NAME!r}, got {data['schema_name']!r}"
        )
    if data["schema_name"] == _FIXED_SUPPORT_SCHEMA_NAME:
        raise DevpopAuditContractError("audit contract must not carry the frozen screening fixed-support schema name")
    if strict_int(data["schema_version"], name="schema_version") != DEVPOP_AUDIT_SCHEMA_VERSION:
        raise DevpopAuditContractError(
            f"schema_version must be {DEVPOP_AUDIT_SCHEMA_VERSION!r}, got {data['schema_version']!r}"
        )
    if data["contract_id"] != DEVPOP_AUDIT_CONTRACT_ID:
        raise DevpopAuditContractError(
            f"contract_id must be {DEVPOP_AUDIT_CONTRACT_ID!r}, got {data['contract_id']!r}"
        )
    if data["contract_id"] == OBJECTIVE_ID_V2:
        raise DevpopAuditContractError("the diagnostic audit contract must not carry the v2 optimizer objective id")
    if data["diagnostic_only"] is not True or data["not_an_optimizer_objective"] is not True:
        raise DevpopAuditContractError("audit contract must be flagged diagnostic_only / not_an_optimizer_objective")
    if data["objective_scope"] != AUDIT_OBJECTIVE_SCOPE:
        raise DevpopAuditContractError(
            f"objective_scope must be {AUDIT_OBJECTIVE_SCOPE!r} (never 'fixed_support'), got {data['objective_scope']!r}"
        )
    if strict_int(data["seq_length_floor"], name="seq_length_floor") != CANONICAL_SEQ_LENGTH_FLOOR:
        raise DevpopAuditContractError(
            f"seq_length_floor must be {CANONICAL_SEQ_LENGTH_FLOOR!r} (fixed 120h floor across every v2 "
            f"seq_length), got {data['seq_length_floor']!r}"
        )
    if strict_int(data["lead_hours"], name="lead_hours") != CANONICAL_LEAD_HOURS:
        raise DevpopAuditContractError(f"lead_hours must be {CANONICAL_LEAD_HOURS}, got {data['lead_hours']!r}")
    if data["target_variable"] != CANONICAL_TARGET_VARIABLE:
        raise DevpopAuditContractError(
            f"target_variable must be {CANONICAL_TARGET_VARIABLE!r}, got {data['target_variable']!r}"
        )
    if data["source_gap_policy_identity"] != CANONICAL_SOURCE_GAP_POLICY_IDENTITY:
        raise DevpopAuditContractError(
            f"source_gap_policy_identity must be {CANONICAL_SOURCE_GAP_POLICY_IDENTITY!r}, "
            f"got {data['source_gap_policy_identity']!r}"
        )

    _reject_sealed_scope(
        role=data["population_role"],
        period=data["period"],
        date_start=data["date_start"],
        date_end=data["date_end"],
    )

    for key in _PACKAGE_SPLIT_HASH_KEYS + ("membership_ids_sha256",):
        if not _is_sha256(data[key]):
            raise DevpopAuditContractError(f"{key} must be a lowercase 64-character SHA-256")
    if data["membership_artifact_sha256"] is not None and not _is_sha256(data["membership_artifact_sha256"]):
        raise DevpopAuditContractError("membership_artifact_sha256 must be null or a lowercase 64-character SHA-256")

    basin_ids = data["basin_ids"]
    if not isinstance(basin_ids, list) or not basin_ids:
        raise DevpopAuditContractError("basin_ids must be a non-empty list")
    if basin_ids != sorted(basin_ids):
        raise DevpopAuditContractError("basin_ids must be sorted")
    if len(set(basin_ids)) != len(basin_ids):
        raise DevpopAuditContractError("basin_ids must be unique")
    if not all(isinstance(b, str) and b and not b.isspace() for b in basin_ids):
        raise DevpopAuditContractError("basin_ids must all be non-empty strings")

    if strict_int(data["expected_population_size"], name="expected_population_size", minimum=1) != len(basin_ids):
        raise DevpopAuditContractError(
            f"expected_population_size={data['expected_population_size']} != len(basin_ids)={len(basin_ids)}"
        )
    recomputed_membership = canonical_membership_sha256(basin_ids)
    if recomputed_membership != data["membership_ids_sha256"]:
        raise DevpopAuditContractError(
            f"membership_ids_sha256 {data['membership_ids_sha256']} does not match the canonical membership "
            f"hash of basin_ids ({recomputed_membership})"
        )

    per_basin_support = data["per_basin_support"]
    eligible_counts = data["eligible_counts"]
    if not isinstance(per_basin_support, dict) or not isinstance(eligible_counts, dict):
        raise DevpopAuditContractError("per_basin_support and eligible_counts must be objects")
    if set(basin_ids) != set(per_basin_support) or set(basin_ids) != set(eligible_counts):
        raise DevpopAuditContractError(
            "basin_ids must exactly match the per_basin_support and eligible_counts key sets"
        )
    zero_support = sorted(
        b for b in basin_ids if is_strict_int(eligible_counts[b]) and eligible_counts[b] == 0
    )
    if zero_support:
        raise DevpopAuditContractError(f"contract contains zero-support basin(s): {zero_support}")
    for basin_id in basin_ids:
        n_expected = strict_int(
            eligible_counts[basin_id], name=f"eligible_counts[{basin_id!r}]", minimum=1
        )
        n_actual = len(per_basin_support[basin_id])
        if n_expected != n_actual:
            raise DevpopAuditContractError(
                f"basin {basin_id!r}: eligible_counts={n_expected} != len(per_basin_support)={n_actual}"
            )
    if data["date_dtype"] not in ("datetime64", "int64"):
        raise DevpopAuditContractError(f"date_dtype must be 'datetime64' or 'int64', got {data['date_dtype']!r}")

    recomputed = hashlib.sha256(canonical_contract_checksum_payload(data)).hexdigest()
    if recomputed != data["checksum_sha256"]:
        raise DevpopAuditContractError(
            f"checksum mismatch: recomputed {recomputed} != stored {data['checksum_sha256']} "
            "-- contract payload was altered after checksumming"
        )
    return data


def _authoritative_package_split_identities() -> dict:
    """The single source of truth for the frozen scientific package and split
    identities: the same five constants the qualified v2 fixed-support
    preparation path pins.  Imported lazily to keep this module import-light
    and free of any import cycle with the preparation adapter."""
    from .sweep_v1_production_adapter import (
        DEVELOPMENT_SPLIT_SHA256,
        PACKAGE_FILE_CHECKSUMS_SHA256,
        PACKAGE_MANIFEST_SHA256,
        PACKAGE_RUN_PROVENANCE_SHA256,
        SPATIAL_HOLDOUT_SPLIT_SHA256,
    )

    return {
        "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "package_file_checksums_sha256": PACKAGE_FILE_CHECKSUMS_SHA256,
        "package_run_provenance_sha256": PACKAGE_RUN_PROVENANCE_SHA256,
        "development_split_sha256": DEVELOPMENT_SPLIT_SHA256,
        "spatial_holdout_split_sha256": SPATIAL_HOLDOUT_SPLIT_SHA256,
    }


def validate_canonical_devpop_audit_contract(data: dict) -> dict:
    """**The one canonical, fail-closed contract boundary.**

    First applies the full generic schema / well-formedness / checksum
    validation (:func:`validate_devpop_audit_contract`), then requires *every*
    authoritative production identity to be exactly present -- not merely
    syntactically well-formed:

    * the diagnostic contract id, canonical ``development_train`` role, the
      canonical 2,307-basin population size, and the pinned line-ending
      independent canonical membership hash
      (:data:`DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256`);
    * the frozen ``validation`` period and the exact full-year-2024 window;
    * the pinned target / lead / 120h sequence floor / source-gap-policy
      identity;
    * the authoritative package-manifest, package-file-checksums and
      package-run-provenance SHA-256s, and the authoritative development and
      spatial-holdout split SHA-256s -- reused verbatim from
      :mod:`src.baseline.sweep_v1_production_adapter` (the same values the
      frozen v2 preparation path pins);
    * ``development_split_sha256`` equal to the canonical membership hash;
    * ``membership_artifact_sha256`` equal to the recorded split-manifest
      ``artifact_sha256['development_train.txt']`` provenance value -- never
      ``None``;
    * ``membership_artifact_provenance`` byte-equal to the pinned *structured*
      provenance block (:data:`CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE`) --
      never a merely non-empty string.

    A generic / synthetic contract (arbitrary well-formed package/split hashes,
    a synthetic membership set, a ``None`` artifact hash, free-form provenance,
    a spec reproducing only the real membership IDs) is structurally rejected
    here, so it can never be written through the canonical writer, loaded
    through the canonical loader, or accepted by the canonical completeness
    gate.  Returns ``data`` unchanged on success."""
    validate_devpop_audit_contract(data)

    if data["population_role"] != DEVELOPMENT_TRAIN_ROLE:
        raise DevpopAuditContractError(
            f"canonical audit contract population_role must be {DEVELOPMENT_TRAIN_ROLE!r}, "
            f"got {data['population_role']!r}"
        )
    if strict_int(data["expected_population_size"], name="expected_population_size", minimum=1) != (
        EXPECTED_DEVELOPMENT_POPULATION_SIZE
    ):
        raise DevpopAuditContractError(
            f"canonical audit contract must cover exactly {EXPECTED_DEVELOPMENT_POPULATION_SIZE} "
            f"development basins, got {data['expected_population_size']!r}"
        )
    if len(data["basin_ids"]) != EXPECTED_DEVELOPMENT_POPULATION_SIZE:
        raise DevpopAuditContractError(
            f"canonical audit contract basin_ids has {len(data['basin_ids'])} entries, "
            f"expected {EXPECTED_DEVELOPMENT_POPULATION_SIZE}"
        )
    if data["membership_ids_sha256"] != DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256:
        raise DevpopAuditContractError(
            "canonical audit contract membership_ids_sha256 must be the pinned development anchor "
            f"{DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256}, got {data['membership_ids_sha256']} "
            "-- generic validation has already confirmed basin_ids hash to this value, so a "
            "synthetic spec reproducing only the real membership IDs is still not canonical unless "
            "every identity below also matches"
        )

    if data["period"] != AUDIT_PERIOD_NAME:
        raise DevpopAuditContractError(
            f"canonical audit contract period must be {AUDIT_PERIOD_NAME!r}, got {data['period']!r}"
        )
    if _parse_exact_iso_date("date_start", data["date_start"]) != date.fromisoformat(AUDIT_DATE_MIN) or (
        _parse_exact_iso_date("date_end", data["date_end"]) != date.fromisoformat(AUDIT_DATE_MAX)
    ):
        raise DevpopAuditContractError(
            f"canonical audit contract window must be exactly [{AUDIT_DATE_MIN}, {AUDIT_DATE_MAX}], "
            f"got [{data['date_start']}, {data['date_end']}]"
        )
    if data["target_variable"] != CANONICAL_TARGET_VARIABLE:
        raise DevpopAuditContractError(
            f"canonical audit contract target_variable must be {CANONICAL_TARGET_VARIABLE!r}"
        )
    if strict_int(data["lead_hours"], name="lead_hours") != CANONICAL_LEAD_HOURS:
        raise DevpopAuditContractError(f"canonical audit contract lead_hours must be {CANONICAL_LEAD_HOURS}")
    if strict_int(data["seq_length_floor"], name="seq_length_floor") != CANONICAL_SEQ_LENGTH_FLOOR:
        raise DevpopAuditContractError(
            f"canonical audit contract seq_length_floor must be {CANONICAL_SEQ_LENGTH_FLOOR} (fixed 120h floor)"
        )
    if data["source_gap_policy_identity"] != CANONICAL_SOURCE_GAP_POLICY_IDENTITY:
        raise DevpopAuditContractError(
            f"canonical audit contract source_gap_policy_identity must be "
            f"{CANONICAL_SOURCE_GAP_POLICY_IDENTITY!r}"
        )

    authoritative = _authoritative_package_split_identities()
    for key, want in authoritative.items():
        if data[key] != want:
            raise DevpopAuditContractError(
                f"canonical audit contract {key} must be the authoritative production identity "
                f"{want}, got {data[key]} -- a wrong-but-well-formed package/split hash is not canonical"
            )
    if data["development_split_sha256"] != data["membership_ids_sha256"]:
        raise DevpopAuditContractError(
            "canonical audit contract development_split_sha256 must equal the canonical membership hash "
            f"({data['membership_ids_sha256']}), got {data['development_split_sha256']}"
        )

    if data["membership_artifact_sha256"] != DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256:
        raise DevpopAuditContractError(
            "canonical audit contract membership_artifact_sha256 must record the split-manifest "
            f"development_train.txt provenance value ({DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256}); "
            f"got {data['membership_artifact_sha256']!r} -- a null or arbitrary artifact hash is not canonical"
        )
    if data["membership_artifact_provenance"] != CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE:
        raise DevpopAuditContractError(
            "canonical audit contract membership_artifact_provenance must be the exact pinned structured "
            "provenance block (split-manifest path + artifact key + recorded artifact hash + development "
            "membership path + role); a free-form, partial, or null provenance is not canonical"
        )
    return data


def _write_devpop_audit_contract(data: dict, path, *, validator) -> Path:
    path = Path(path)
    validator(data)
    if path.exists():
        raise DevpopAuditContractError(f"refusing to overwrite existing devpop audit contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def write_devpop_audit_contract(data: dict, path) -> Path:
    """**Canonical** writer: validate ``data`` through
    :func:`validate_canonical_devpop_audit_contract` (every authoritative
    production identity pinned), then atomically write it via tmp-write +
    replace.  Strict no-overwrite -- an immutable audit contract is never
    silently replaced.  A generic / synthetic contract is refused here; use
    :func:`write_synthetic_devpop_audit_contract` for fixtures."""
    return _write_devpop_audit_contract(data, path, validator=validate_canonical_devpop_audit_contract)


def write_synthetic_devpop_audit_contract(data: dict, path) -> Path:
    """**Synthetic / fixture-only** writer: generic schema/checksum validation
    only (:func:`validate_devpop_audit_contract`).  Its output is *not* a
    canonical artifact and cannot be read back through
    :func:`load_devpop_audit_contract` or consumed by any canonical gate."""
    return _write_devpop_audit_contract(data, path, validator=validate_devpop_audit_contract)


def _load_devpop_audit_contract(path, *, validator) -> dict:
    path = Path(path)
    if not path.is_file():
        raise DevpopAuditContractError(f"devpop audit contract not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return validator(data)


def load_devpop_audit_contract(path) -> dict:
    """**Canonical** loader: every authoritative production identity is
    re-pinned via :func:`validate_canonical_devpop_audit_contract`.  A generic
    / synthetic artifact cannot be loaded through this path."""
    return _load_devpop_audit_contract(path, validator=validate_canonical_devpop_audit_contract)


def load_synthetic_devpop_audit_contract(path) -> dict:
    """**Synthetic / fixture-only** loader: generic validation only."""
    return _load_devpop_audit_contract(path, validator=validate_devpop_audit_contract)


def deserialize_contract_support(contract: dict, basin_id: str) -> np.ndarray:
    """Decode the frozen admitted-timestamp array for ``basin_id`` -- reuses
    the shared fixed-support date deserializer (no parallel logic)."""
    validate_devpop_audit_contract(contract)
    if basin_id not in contract["per_basin_support"]:
        raise DevpopAuditContractError(f"basin {basin_id!r} is not in the audit contract")
    return deserialize_support_date_array(contract["per_basin_support"][basin_id], contract["date_dtype"])


# --------------------------------------------------------------------------- #
# Generalized strict completeness gate
# --------------------------------------------------------------------------- #

def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(value)
    )


def _strict_count(result: Mapping, key: str) -> int:
    """Fetch ``result[key]`` and require it be a present, strict (non-bool)
    integer.  A missing key is a defect -- it must never default to a value
    that could pass a completeness check."""
    if key not in result:
        raise DevpopAuditCompletenessError(f"result is missing required accounting field {key!r}")
    try:
        return strict_int(result[key], name=f"result[{key!r}]", minimum=0)
    except Exception as exc:  # DevpopAuditContractError from the strict_int wrapper
        raise DevpopAuditCompletenessError(str(exc)) from exc


def _strict_row_count(row: Mapping, key: str, basin_id: object) -> int:
    if key not in row:
        raise DevpopAuditCompletenessError(f"basin {basin_id!r}: per_basin row is missing required field {key!r}")
    try:
        return strict_int(row[key], name=f"per_basin[{basin_id!r}][{key!r}]", minimum=0)
    except Exception as exc:  # DevpopAuditContractError from the strict_int wrapper
        raise DevpopAuditCompletenessError(str(exc)) from exc


def _devpop_audit_completeness_body(
    result: Mapping,
    *,
    population: ExpectedPopulationSpec,
    contract: Mapping,
) -> dict:
    """Shared strict structural / accounting / reconciliation checks for a
    development-population Common-120 audit *result*, driven entirely by
    ``population.expected_size`` / ``population.basin_ids``.

    The ``contract`` is assumed already validated by the caller (canonically or
    generically).  Raises :class:`DevpopAuditCompletenessError` -- blocking
    completeness pending investigation -- on:

    * requested basin IDs not exactly equal to the expected population;
    * a missing / non-list / duplicated / unsorted / mismatched
      ``evaluated_basin_ids`` receipt (it is **mandatory and explicit** --
      never derived silently);
    * evaluated basin IDs not exactly equal to the expected population, or not
      equal to the identities represented by ``per_basin``;
    * any duplicate / missing / extra basin identity;
    * a missing / non-integer / bool-valued accounting field;
    * requested / evaluated / excluded counts not internally consistent;
    * any exclusion at all (completeness forbids silent drops);
    * a missing ``basins_excluded`` receipt;
    * any non-finite (or bool) per-basin raw-space NSE;
    * a per-basin ``n_admitted`` that does not equal
      ``contract["eligible_counts"][basin_id]``;
    * a missing / non-integer / non-zero per-basin
      ``n_sim_nonfinite_at_admitted``;
    * an aggregate whose totals / finite-NSE-basin count do not exactly
      reconcile with the per-basin rows;
    * a support-contract identity / checksum that does not match the contract.

    Never mutates ``result``; never drops a basin.  Returns the receipt body
    (without the canonical / fixture label) on success.
    """
    if not isinstance(result, Mapping):
        raise DevpopAuditCompletenessError(f"result must be a mapping, got {type(result).__name__}")

    # -- support-contract / population identity agreement ------------------- #
    if result.get("objective_scope") != AUDIT_OBJECTIVE_SCOPE:
        raise DevpopAuditCompletenessError(
            f"result objective_scope must be {AUDIT_OBJECTIVE_SCOPE!r} (never 'fixed_support'); "
            f"got {result.get('objective_scope')!r}"
        )
    if result.get("contract_id") != DEVPOP_AUDIT_CONTRACT_ID or contract["contract_id"] != DEVPOP_AUDIT_CONTRACT_ID:
        raise DevpopAuditCompletenessError("result/contract contract_id must be the diagnostic audit id")
    if population.contract_id != DEVPOP_AUDIT_CONTRACT_ID:
        raise DevpopAuditCompletenessError("population contract_id must be the diagnostic audit id")
    if result.get("contract_checksum_sha256") != contract["checksum_sha256"]:
        raise DevpopAuditCompletenessError("result support-contract checksum does not match the expected contract")
    if list(contract["basin_ids"]) != list(population.basin_ids):
        raise DevpopAuditCompletenessError("contract basin_ids do not equal the expected population basin_ids")
    if contract["membership_ids_sha256"] != population.membership_ids_sha256:
        raise DevpopAuditCompletenessError("contract membership hash does not equal the expected population hash")
    if contract["expected_population_size"] != population.expected_size:
        raise DevpopAuditCompletenessError("contract population size does not equal the expected population size")

    expected = list(population.basin_ids)
    expected_set = set(expected)
    n_expected = population.expected_size

    # -- requested population -------------------------------------------------#
    requested = result.get("requested_basin_ids")
    if not isinstance(requested, (list, tuple)):
        raise DevpopAuditCompletenessError("result must carry an explicit requested_basin_ids list")
    requested = list(requested)
    if len(requested) != len(set(requested)):
        raise DevpopAuditCompletenessError("requested_basin_ids contains duplicates")
    if set(requested) != expected_set:
        raise DevpopAuditCompletenessError(
            "requested basin IDs do not equal the expected population: "
            f"missing={sorted(expected_set - set(requested))} extra={sorted(set(requested) - expected_set)}"
        )
    if sorted(requested) != expected:
        raise DevpopAuditCompletenessError("requested basin IDs are not the exact sorted expected population")

    # -- evaluated population: EXPLICIT and mandatory --------------------- #
    per_basin = result.get("per_basin")
    if not isinstance(per_basin, list):
        raise DevpopAuditCompletenessError("result must carry a per_basin list")
    per_basin_ids = [row.get("basin_id") if isinstance(row, Mapping) else None for row in per_basin]
    if any(b is None for b in per_basin_ids):
        raise DevpopAuditCompletenessError("every per_basin row must carry a basin_id")

    # ``evaluated_basin_ids`` is a mandatory explicit receipt.  A later narrow
    # evaluator adapter produces it; it is NEVER derived silently here.  It must
    # be a sorted, duplicate-free list, exactly equal to the canonical
    # population and to the identities represented by ``per_basin``.
    if "evaluated_basin_ids" not in result:
        raise DevpopAuditCompletenessError(
            "result must carry an explicit evaluated_basin_ids list "
            "(a narrow evaluator adapter produces this receipt; it is never derived silently)"
        )
    evaluated = result["evaluated_basin_ids"]
    if not isinstance(evaluated, list):
        raise DevpopAuditCompletenessError("evaluated_basin_ids must be a list")
    if len(evaluated) != len(set(evaluated)):
        raise DevpopAuditCompletenessError("evaluated_basin_ids contains duplicates")
    if evaluated != sorted(evaluated):
        raise DevpopAuditCompletenessError("evaluated_basin_ids must be sorted")
    if set(evaluated) != expected_set:
        raise DevpopAuditCompletenessError(
            "evaluated basin IDs do not equal the expected population: "
            f"missing={sorted(expected_set - set(evaluated))} extra={sorted(set(evaluated) - expected_set)}"
        )
    if evaluated != sorted(per_basin_ids):
        raise DevpopAuditCompletenessError(
            "evaluated_basin_ids does not match the identities represented by per_basin"
        )

    # -- exclusions --------------------------------------------------------- #
    if "basins_excluded" not in result:
        raise DevpopAuditCompletenessError("result is missing the required basins_excluded receipt")
    excluded = result["basins_excluded"]
    if not isinstance(excluded, list):
        raise DevpopAuditCompletenessError("basins_excluded must be a list")
    if excluded:
        named = sorted(
            row.get("basin_id") if isinstance(row, Mapping) else repr(row) for row in excluded
        )
        raise DevpopAuditCompletenessError(
            f"canonical completeness forbids any exclusion; {len(excluded)} basin(s) excluded: {named}"
        )

    # -- accounting fields: present + strict integers ---------------------- #
    n_requested = _strict_count(result, "n_basins_requested")
    n_evaluated = _strict_count(result, "n_basins_evaluated")
    n_excluded = _strict_count(result, "n_basins_excluded")
    if n_requested != n_expected or len(requested) != n_expected:
        raise DevpopAuditCompletenessError(f"expected exactly {n_expected} requested basins, got {n_requested}")
    if n_evaluated != n_expected or len(evaluated) != n_expected or len(per_basin) != n_expected:
        raise DevpopAuditCompletenessError(f"expected exactly {n_expected} evaluated basins, got {n_evaluated}")
    if n_excluded != 0:
        raise DevpopAuditCompletenessError("expected zero excluded basins for canonical completeness")
    if n_evaluated + n_excluded != n_requested:
        raise DevpopAuditCompletenessError("requested / evaluated / excluded counts are not internally consistent")

    # -- per-basin: finite NSE, n_admitted == eligible_counts, zero nonfinite #
    eligible_counts = contract["eligible_counts"]
    n_admitted_sum = 0
    n_nonfinite_sum = 0
    for row in per_basin:
        basin_id = row.get("basin_id")
        nse = row.get("nse")
        if isinstance(nse, bool) or not _finite_number(nse):
            raise DevpopAuditCompletenessError(
                f"basin {basin_id!r}: every expected basin must contribute a finite real raw-space NSE, "
                f"got {nse!r}"
            )
        n_admitted = _strict_row_count(row, "n_admitted", basin_id)
        expected_admitted = eligible_counts.get(basin_id)
        if expected_admitted is None:
            raise DevpopAuditCompletenessError(
                f"basin {basin_id!r}: not present in the support contract eligible_counts"
            )
        if n_admitted != expected_admitted:
            raise DevpopAuditCompletenessError(
                f"basin {basin_id!r}: per-basin n_admitted={n_admitted} does not equal the support "
                f"contract eligible_counts={expected_admitted}"
            )
        n_nonfinite = _strict_row_count(row, "n_sim_nonfinite_at_admitted", basin_id)
        if n_nonfinite != 0:
            raise DevpopAuditCompletenessError(
                f"basin {basin_id!r}: non-finite simulation at an admitted timestamp ({n_nonfinite})"
            )
        n_admitted_sum += n_admitted
        n_nonfinite_sum += n_nonfinite

    # -- aggregate: strict ints that exactly reconcile with the rows ------- #
    aggregate = result.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise DevpopAuditCompletenessError("result must carry an aggregate object")
    agg_n_basins = _strict_count(aggregate, "n_basins")
    agg_n_admitted_total = _strict_count(aggregate, "n_admitted_total")
    agg_n_nonfinite_total = _strict_count(aggregate, "n_sim_nonfinite_at_admitted_total")
    if not isinstance(aggregate.get("metrics"), Mapping) or not isinstance(
        aggregate["metrics"].get("nse"), Mapping
    ):
        raise DevpopAuditCompletenessError("aggregate must carry metrics.nse")
    nse_summary = aggregate["metrics"]["nse"]
    agg_n_finite_basins = _strict_count(nse_summary, "n_finite_basins")
    if agg_n_basins != n_expected:
        raise DevpopAuditCompletenessError(
            f"aggregate n_basins={agg_n_basins} != expected population {n_expected}"
        )
    if agg_n_finite_basins != n_expected:
        raise DevpopAuditCompletenessError(
            f"aggregate finite-NSE basin count={agg_n_finite_basins} != expected population {n_expected}"
        )
    if agg_n_admitted_total != n_admitted_sum:
        raise DevpopAuditCompletenessError(
            f"aggregate n_admitted_total={agg_n_admitted_total} does not reconcile with the per-basin "
            f"row sum ({n_admitted_sum})"
        )
    if agg_n_nonfinite_total != 0 or n_nonfinite_sum != 0:
        raise DevpopAuditCompletenessError("aggregate / per-basin non-finite-at-admitted total must be zero")
    if agg_n_nonfinite_total != n_nonfinite_sum:
        raise DevpopAuditCompletenessError(
            f"aggregate n_sim_nonfinite_at_admitted_total={agg_n_nonfinite_total} does not reconcile with "
            f"the per-basin row sum ({n_nonfinite_sum})"
        )

    return {
        "objective_scope": AUDIT_OBJECTIVE_SCOPE,
        "contract_id": DEVPOP_AUDIT_CONTRACT_ID,
        "population_role": population.role,
        "n_expected": n_expected,
        "n_requested": n_requested,
        "n_evaluated": n_evaluated,
        "n_excluded": n_excluded,
        "n_admitted_total": n_admitted_sum,
        "n_sim_nonfinite_at_admitted_total": n_nonfinite_sum,
        "evaluated_basin_ids": list(evaluated),
        "membership_ids_sha256": population.membership_ids_sha256,
        "contract_checksum_sha256": contract["checksum_sha256"],
    }


def require_complete_devpop_audit_population(
    result: Mapping,
    *,
    population: ExpectedPopulationSpec,
    contract: Mapping,
) -> dict:
    """**Canonical, mandatory** full-development-population Common-120 audit
    completeness gate.  There is no opt-out.

    The ``population`` is run through
    :func:`assert_canonical_development_population` (the real 2,307-basin
    identity) and the ``contract`` through
    :func:`validate_canonical_devpop_audit_contract` (every authoritative
    package / split / provenance identity), *then* the shared strict structural
    / accounting / reconciliation checks (:func:`_devpop_audit_completeness_body`)
    run.  A synthetic fixture population or a generic contract cannot satisfy
    this call.

    Returns a receipt carrying ``canonical_completeness=True`` and
    ``canonical_population_verified=True`` -- labels that *only* this function
    can produce.
    """
    if not isinstance(population, ExpectedPopulationSpec):
        raise DevpopAuditCompletenessError("population must be an ExpectedPopulationSpec")
    assert_canonical_development_population(population)
    validate_canonical_devpop_audit_contract(dict(contract))
    body = _devpop_audit_completeness_body(result, population=population, contract=contract)
    return {"canonical_completeness": True, "canonical_population_verified": True, **body}


def require_complete_synthetic_devpop_audit_population(
    result: Mapping,
    *,
    population: ExpectedPopulationSpec,
    contract: Mapping,
) -> dict:
    """**Synthetic / fixture-only** completeness check.

    Runs the *identical* strict structural / accounting / reconciliation checks
    as the canonical gate (:func:`_devpop_audit_completeness_body`), but against
    a synthetic fixture population + a generically-validated contract, and
    returns a receipt labelled ``fixture_completeness`` -- **never**
    ``canonical_completeness`` / ``canonical_population_verified``.  A canonical
    receipt can only ever come from
    :func:`require_complete_devpop_audit_population`.
    """
    if not isinstance(population, ExpectedPopulationSpec):
        raise DevpopAuditCompletenessError("population must be an ExpectedPopulationSpec")
    validate_devpop_audit_contract(dict(contract))
    body = _devpop_audit_completeness_body(result, population=population, contract=contract)
    return {"fixture_completeness": True, **body}
