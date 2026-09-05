"""SHARED-A4: frozen seven-checkpoint selection manifest for the future
development-population Common-120 audit (see
``docs/stage1_devpop_common120_audit_contract_v001.md`` Sec. 3).

This manifest pins exactly which seven already-screened v2 six-axis
configurations (Bayesian Proposals 1-3 + the four frozen IID random-control
Wave-1 rows) will later be re-evaluated against the full 2,307-basin
development population -- and records the ALREADY-DECIDED
``screening_best_epoch`` for each, since re-selecting an epoch from the
full-population run is explicitly out of scope for this audit family.

Strictly additive: reuses (never re-derives) the v2 identity grammar in
:mod:`sweep_v2_six_axis_campaign` -- ``configuration_id_v2``, ``proposal_id_v2``,
``trial_id_v2``, ``canonical_hyperparameters_v2`` -- so a manifest entry can
never silently drift from the authoritative campaign identity of the
configuration/proposal/trial it names. Bridges directly onto the existing
SHARED-A2 consumer identity (:class:`DevpopAuditCheckpointIdentity`) rather
than inventing a parallel identity shape.

Populating the REAL seven-entry production manifest (real checkpoint paths,
real screening scores, real evidence pointers) is out of scope here -- that is
SHARED-A5. This module only defines the representation, validation, and
identity-pinning machinery; tests exercise it with synthetic entries only.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .devpop_common120_audit_evaluator import DevpopAuditCheckpointIdentity
from .nh_seed_evaluation import weight_stem
from .sweep_v2_six_axis_campaign import (
    OBJECTIVE_ID_V2,
    SEARCH_ARMS_V2,
    SweepV2CampaignError,
    canonical_hyperparameters_v2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)

__all__ = [
    "DevpopAuditSelectionManifestError",
    "DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA",
    "DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA_VERSION",
    "EXPECTED_MANIFEST_SIZE",
    "EXPECTED_ARM_PROPOSAL_ORDERS",
    "build_devpop_audit_selection_manifest_entry",
    "validate_devpop_audit_selection_manifest_entry",
    "validate_devpop_audit_selection_manifest",
    "compute_devpop_audit_selection_manifest_sha256",
    "write_devpop_audit_selection_manifest",
    "load_devpop_audit_selection_manifest",
    "selection_manifest_entry_to_checkpoint_identity",
]


class DevpopAuditSelectionManifestError(ValueError):
    """Raised for a malformed/inconsistent seven-checkpoint selection
    manifest entry, a manifest whose composition does not match the frozen
    Bayesian-Proposals-1-3 + random-control-Wave-1 shape, an identity that
    conflicts with the authoritative v2 identity grammar, or an attempted
    overwrite/tamper of a written manifest. Never raised for an ordinary
    poor-skill outcome."""


DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA = "flashnh_devpop_audit_selection_manifest_v001"
DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA_VERSION = 1

EXPECTED_MANIFEST_SIZE = 7

#: Frozen composition: Bayesian Proposals 1-3 (``proposal_order`` 1-3) plus
#: the four frozen IID random-control Wave-1 rows (``proposal_order`` 1-4 of
#: the committed 12-row random-control manifest). Do not widen this without
#: explicit user authorization -- it is the scientific scope this audit
#: family exists to answer, not a convention this module may extend.
EXPECTED_ARM_PROPOSAL_ORDERS: Mapping[str, frozenset] = {
    "bayesian": frozenset({1, 2, 3}),
    "random_control": frozenset({1, 2, 3, 4}),
}

_REQUIRED_ENTRY_FIELDS = frozenset({
    "search_arm", "proposal_order", "proposal_id", "configuration_id", "trial_id",
    "execution_generation", "hyperparameters", "objective_id",
    "screening_score", "screening_best_epoch", "screening_evidence_path",
    "source_run_dir", "checkpoint_filename", "checkpoint_sha256", "selection_policy",
})


def _finite_real(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and len(value.strip()) > 0


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def build_devpop_audit_selection_manifest_entry(
    *,
    search_arm: str,
    proposal_order: int,
    hyperparameters: Mapping[str, Any],
    screening_score: float,
    screening_best_epoch: int,
    screening_evidence_path: str,
    source_run_dir: str,
    checkpoint_filename: str,
    checkpoint_sha256: str,
    selection_policy: str,
    support_contract_version: str,
    support_contract_sha256: str,
    execution_generation: int = 1,
) -> dict:
    """Construct one manifest entry, deriving its ``configuration_id`` /
    ``proposal_id`` / ``trial_id`` from the SAME authoritative v2 identity
    functions the campaign/production-adapter code uses -- never typed in by
    a caller -- so an entry can never assert an identity inconsistent with
    the six-axis hyperparameters it actually names.

    ``hyperparameters`` must be the raw six-axis mapping (as fed to
    :func:`canonical_hyperparameters_v2`); it is stored canonicalized.
    ``support_contract_version``/``support_contract_sha256`` must be the
    identity of the frozen fixed-support contract this configuration was
    screened under (see :mod:`fixed_support_contract_v2`).
    """
    try:
        canonical = canonical_hyperparameters_v2(hyperparameters)
        configuration_id = configuration_id_v2(
            canonical, support_contract_version=support_contract_version,
            support_contract_sha256=support_contract_sha256,
        )
        proposal_id = proposal_id_v2(search_arm, proposal_order)
        trial_id = trial_id_v2(configuration_id, proposal_id, execution_generation=execution_generation)
    except SweepV2CampaignError as exc:
        raise DevpopAuditSelectionManifestError(str(exc)) from exc

    entry = {
        "search_arm": search_arm,
        "proposal_order": proposal_order,
        "proposal_id": proposal_id,
        "configuration_id": configuration_id,
        "trial_id": trial_id,
        "execution_generation": execution_generation,
        "hyperparameters": canonical,
        "objective_id": OBJECTIVE_ID_V2,
        "screening_score": screening_score,
        "screening_best_epoch": screening_best_epoch,
        "screening_evidence_path": screening_evidence_path,
        "source_run_dir": source_run_dir,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_sha256": checkpoint_sha256,
        "selection_policy": selection_policy,
    }
    _validate_entry(entry, support_contract_version=support_contract_version,
                     support_contract_sha256=support_contract_sha256)
    return entry


def _validate_entry(
    entry: Mapping[str, Any],
    *,
    support_contract_version: "str | None" = None,
    support_contract_sha256: "str | None" = None,
) -> None:
    if not isinstance(entry, Mapping):
        raise DevpopAuditSelectionManifestError(f"manifest entry must be a mapping, got {type(entry).__name__}")
    keys = set(entry)
    missing = _REQUIRED_ENTRY_FIELDS - keys
    if missing:
        raise DevpopAuditSelectionManifestError(f"manifest entry missing required fields: {sorted(missing)}")
    unknown = keys - _REQUIRED_ENTRY_FIELDS
    if unknown:
        raise DevpopAuditSelectionManifestError(f"manifest entry has unexpected fields: {sorted(unknown)}")

    search_arm = entry["search_arm"]
    if search_arm not in SEARCH_ARMS_V2:
        raise DevpopAuditSelectionManifestError(f"unknown search_arm: {search_arm!r}")
    proposal_order = entry["proposal_order"]
    if not _positive_int(proposal_order):
        raise DevpopAuditSelectionManifestError("proposal_order must be a positive integer")

    if entry["objective_id"] != OBJECTIVE_ID_V2:
        raise DevpopAuditSelectionManifestError(
            f"entry objective_id must be the frozen v2 screening objective {OBJECTIVE_ID_V2!r}, "
            f"got {entry['objective_id']!r}"
        )
    if not _finite_real(entry["screening_score"]):
        raise DevpopAuditSelectionManifestError(f"screening_score must be a finite real number, got {entry['screening_score']!r}")
    if not _positive_int(entry["screening_best_epoch"]):
        raise DevpopAuditSelectionManifestError("screening_best_epoch must be a positive integer")
    if not _is_sha256_hex(entry["checkpoint_sha256"]):
        raise DevpopAuditSelectionManifestError(f"checkpoint_sha256 must be a lowercase 64-char hex SHA-256, got {entry['checkpoint_sha256']!r}")
    for field in ("screening_evidence_path", "source_run_dir", "checkpoint_filename", "selection_policy"):
        if not _non_empty_str(entry[field]):
            raise DevpopAuditSelectionManifestError(f"{field} must be a non-empty string")

    # Fail-closed epoch<->checkpoint binding: the ALREADY-DECIDED
    # ``screening_best_epoch`` must be the epoch encoded in the checkpoint's
    # own filename, using the SAME ``weight_stem`` convention the producer
    # copies/renames under. Without this an entry could claim epoch N while
    # naming checkpoint bytes from epoch M; the producer would then rename
    # those bytes to epoch N and SHARED-A2 could no longer see the original
    # mismatch. Checked here (before any producer copy) so it can never
    # reach disk.
    expected_checkpoint_filename = f"{weight_stem(entry['screening_best_epoch'])}.pt"
    if entry["checkpoint_filename"] != expected_checkpoint_filename:
        raise DevpopAuditSelectionManifestError(
            f"checkpoint_filename {entry['checkpoint_filename']!r} is not bound to screening_best_epoch "
            f"{entry['screening_best_epoch']!r}: expected {expected_checkpoint_filename!r}"
        )

    if not _positive_int(entry["execution_generation"]):
        raise DevpopAuditSelectionManifestError("execution_generation must be a positive integer")

    # Recompute the six-axis canonical coordinate; this both structurally
    # validates ``hyperparameters`` and guarantees it round-trips identically
    # (guards a hand-edited/corrupted JSON manifest, not just a fresh build).
    try:
        canonical = canonical_hyperparameters_v2(entry["hyperparameters"])
    except SweepV2CampaignError as exc:
        raise DevpopAuditSelectionManifestError(f"entry hyperparameters invalid: {exc}") from exc
    if canonical != dict(entry["hyperparameters"]):
        raise DevpopAuditSelectionManifestError(
            "entry hyperparameters are not already in canonical six-axis form"
        )

    # Identity cross-check against the authoritative v2 identity grammar.
    # support_contract_version/sha256 are not stored on the entry itself
    # (they are a campaign-wide constant, not a per-entry fact) -- callers
    # that know the frozen fixed-support contract identity may pass it to
    # re-derive configuration_id from scratch; callers that only have the
    # manifest on disk (e.g. a plain reload) skip this and rely on
    # proposal_id/trial_id agreement instead, which does not require it.
    if support_contract_version is not None and support_contract_sha256 is not None:
        try:
            expected_config_id = configuration_id_v2(
                canonical, support_contract_version=support_contract_version,
                support_contract_sha256=support_contract_sha256,
            )
        except SweepV2CampaignError as exc:
            raise DevpopAuditSelectionManifestError(str(exc)) from exc
        if expected_config_id != entry["configuration_id"]:
            raise DevpopAuditSelectionManifestError(
                f"entry configuration_id {entry['configuration_id']!r} does not match the identity "
                f"recomputed from its own hyperparameters + the given fixed-support contract identity "
                f"({expected_config_id!r})"
            )
    try:
        expected_proposal_id = proposal_id_v2(search_arm, proposal_order)
        expected_trial_id = trial_id_v2(
            entry["configuration_id"], expected_proposal_id, execution_generation=entry["execution_generation"]
        )
    except SweepV2CampaignError as exc:
        raise DevpopAuditSelectionManifestError(str(exc)) from exc
    if entry["proposal_id"] != expected_proposal_id:
        raise DevpopAuditSelectionManifestError(
            f"entry proposal_id {entry['proposal_id']!r} conflicts with proposal_id_v2({search_arm!r}, "
            f"{proposal_order!r}) = {expected_proposal_id!r}"
        )
    if entry["trial_id"] != expected_trial_id:
        raise DevpopAuditSelectionManifestError(
            f"entry trial_id {entry['trial_id']!r} conflicts with the identity recomputed from its own "
            f"configuration_id/proposal_id/execution_generation ({expected_trial_id!r})"
        )


def validate_devpop_audit_selection_manifest_entry(
    entry: Mapping[str, Any],
    *,
    support_contract_version: "str | None" = None,
    support_contract_sha256: "str | None" = None,
) -> None:
    """Public single-entry validator (see :func:`_validate_entry`) for
    consumers -- e.g. the SHARED-A4 eval-run producer -- that need to
    validate/cross-check one manifest entry without the full seven-entry
    composition check."""
    _validate_entry(entry, support_contract_version=support_contract_version,
                     support_contract_sha256=support_contract_sha256)


def validate_devpop_audit_selection_manifest(
    entries: Sequence[Mapping[str, Any]],
    *,
    support_contract_version: "str | None" = None,
    support_contract_sha256: "str | None" = None,
) -> dict:
    """Validate the full frozen seven-entry manifest: exactly Bayesian
    Proposals {1,2,3} + random-control Wave-1 {1,2,3,4}, no duplicate
    trial/configuration identity, and every entry internally consistent with
    the v2 identity grammar (see :func:`_validate_entry`). Returns a receipt
    dict carrying a stable ``manifest_sha256`` pinning the exact validated
    entry set.
    """
    if not isinstance(entries, (list, tuple)):
        raise DevpopAuditSelectionManifestError(f"manifest entries must be a list, got {type(entries).__name__}")
    entries = list(entries)
    if len(entries) != EXPECTED_MANIFEST_SIZE:
        raise DevpopAuditSelectionManifestError(
            f"expected exactly {EXPECTED_MANIFEST_SIZE} manifest entries, got {len(entries)}"
        )
    for entry in entries:
        _validate_entry(entry, support_contract_version=support_contract_version,
                         support_contract_sha256=support_contract_sha256)

    by_arm: dict[str, set] = {}
    for entry in entries:
        by_arm.setdefault(entry["search_arm"], set()).add(entry["proposal_order"])
    expected = {arm: set(orders) for arm, orders in EXPECTED_ARM_PROPOSAL_ORDERS.items()}
    if by_arm != expected:
        raise DevpopAuditSelectionManifestError(
            f"manifest arm/proposal_order composition must be exactly {expected}, got {by_arm}"
        )

    trial_ids = [entry["trial_id"] for entry in entries]
    configuration_ids = [entry["configuration_id"] for entry in entries]
    if len(set(trial_ids)) != EXPECTED_MANIFEST_SIZE:
        raise DevpopAuditSelectionManifestError("duplicate trial_id across manifest entries")
    if len(set(configuration_ids)) != EXPECTED_MANIFEST_SIZE:
        raise DevpopAuditSelectionManifestError("duplicate configuration_id across manifest entries")

    return {
        "schema": DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA,
        "schema_version": DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA_VERSION,
        "manifest_sha256": compute_devpop_audit_selection_manifest_sha256(entries),
        "entries": entries,
    }


def compute_devpop_audit_selection_manifest_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    """Stable identity pinning the exact validated seven-entry selection --
    order-independent (entries are sorted by ``(search_arm, proposal_order)``
    before hashing) so this hash depends only on WHICH seven configurations
    and epochs were selected, never on manifest file ordering."""
    ordered = sorted(entries, key=lambda e: (e["search_arm"], e["proposal_order"]))
    encoded = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_devpop_audit_selection_manifest(entries: Sequence[Mapping[str, Any]], path) -> Path:
    """Validate then atomically write the manifest. Strict no-overwrite --
    a frozen seven-checkpoint selection is never silently replaced."""
    manifest = validate_devpop_audit_selection_manifest(entries)
    path = Path(path)
    if path.exists():
        raise DevpopAuditSelectionManifestError(f"refusing to overwrite existing selection manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def load_devpop_audit_selection_manifest(path) -> dict:
    """Load + re-validate a manifest written by
    :func:`write_devpop_audit_selection_manifest`, and confirm the recorded
    ``manifest_sha256`` still matches the recomputed identity of its entries
    (tamper/corruption evidence)."""
    path = Path(path)
    if not path.is_file():
        raise DevpopAuditSelectionManifestError(f"selection manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != DEVPOP_AUDIT_SELECTION_MANIFEST_SCHEMA:
        raise DevpopAuditSelectionManifestError(f"not a devpop audit selection manifest: {path}")
    manifest = validate_devpop_audit_selection_manifest(data.get("entries"))
    if data.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise DevpopAuditSelectionManifestError(
            f"selection manifest sha256 {data.get('manifest_sha256')!r} does not match the recomputed "
            f"identity of its own entries ({manifest['manifest_sha256']!r}) -- possible tampering/corruption"
        )
    return manifest


def selection_manifest_entry_to_checkpoint_identity(entry: Mapping[str, Any]) -> DevpopAuditCheckpointIdentity:
    """Bridge one validated manifest entry onto the existing SHARED-A2
    consumer identity. ``screening_best_epoch`` becomes the frozen
    ``checkpoint_epoch`` -- it is RECORDED here, never recomputed."""
    _validate_entry(entry)
    return DevpopAuditCheckpointIdentity(
        trial_id=entry["trial_id"],
        configuration_id=entry["configuration_id"],
        checkpoint_epoch=entry["screening_best_epoch"],
        checkpoint_sha256=entry["checkpoint_sha256"],
    )
