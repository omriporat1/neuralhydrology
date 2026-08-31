"""Shared, offline registration primitives for the v2 six-axis W&B bridge.

Both consumers of the v2 bridge controller-registration seam import from here
rather than re-implementing the descriptor authority, the strict input
validators, the unsafe-sweep-id refusal, and the partial-failure receipt
semantics:

* ``scripts/create_sweep_v2_six_axis_wandb_bridge_rehearsal_sweep.py`` -- the
  CLOSED disposable rehearsal registration seam (unchanged behaviour; it
  re-exports these names so its existing test surface keeps working);
* ``scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py`` -- the
  reusable v2 production controller registration + strict production manifest
  builder + one-agent launch helper.

Nothing in this module imports ``wandb`` or contacts any external service.
The authoritative Common-120 fixed-support artifact identity is loaded from
the one committed descriptor and strictly validated -- never copied into
Python constants -- so both consumers' manifest ``fixed_support_contract_*``
bindings can never silently drift from the qualified, checksummed record.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .fixed_support_contract_v2 import CONTRACT_SCHEMA_NAME, CONTRACT_SCHEMA_VERSION
from . import sweep_v1_campaign as sweep
from .sweep_v2_six_axis_campaign import (
    FORBIDDEN_PRODUCTION_SWEEP_IDS,
    FORBIDDEN_V1_SWEEP_ID,
    OBJECTIVE_ID_V2,
)
from .sweep_v2_six_axis_config import V2_METRIC_NAME

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Fixed identities describing the one authoritative committed descriptor.
# ---------------------------------------------------------------------------
_DESCRIPTOR_PATH = _REPO_ROOT / "config" / "stage1_v2_common120_fixed_support_artifact_identity_v001.json"
_EXPECTED_SCHEMA_NAME = "flashnh_stage1_v2_fixed_support_artifact_identity_record"
_EXPECTED_SCHEMA_VERSION = 1
_EXPECTED_RECORD_ID = "stage1_v2_common120_fixed_support_artifact_identity_v001"
_EXPECTED_TRACKING_STATUS = "external_untracked_large_artifact"

# ---------------------------------------------------------------------------
# Real, already-qualified identities shared by both v2 registration consumers
# -- the v2 bridge needs the same real package/screening/policy inputs the v1
# bridge uses; only the sweep id, manifest, output root, proposal
# order/generation, and fixed-support binding diverge per invocation.
# ---------------------------------------------------------------------------
_CANONICAL_RUNTIME_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_DEFAULT_WANDB_PROJECT = "flashnh-stage1"
_DEFAULT_WANDB_ENTITY = "omri-porat1-huji"

_REAL_PACKAGE_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002"
_REAL_SCREENING_BASIN_IDS_PATH = (
    "/sci/labs/efratmorin/omripo/Flash-NH/data/screening_subsets/"
    "stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt"
)
_REAL_BASELINE_POLICY_PATH = str(_REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml")
_REAL_POLICY_OVERLAY_PATH = str(_REPO_ROOT / "config" / "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml")
_REAL_BASE_PILOT_POLICY_PATH = str(_REPO_ROOT / "config" / "stage1_lead06_pilot_v001.yaml")

# Clearly non-real, only ever held in memory by a preflight -- never written
# to a manifest, never accepted by ``_assert_safe_sweep_id``.
_PREFLIGHT_PLACEHOLDER_SWEEP_ID = "PREFLIGHT-PLACEHOLDER-NOT-A-REAL-SWEEP-ID"

# Every real sweep id observed from this project's own qualified W&B
# registrations (including the frozen v1 production id) is lowercase
# alphanumeric with no separators; this is a conservative shape check, not
# an attempt to fully specify W&B's id grammar.
_SWEEP_ID_PATTERN = re.compile(r"^[0-9a-z]{4,32}$")


class RegistrationSeamError(ValueError):
    """A local validation or unsafe-identity boundary refused to proceed."""


class RegistrationPartialFailure(RegistrationSeamError):
    """A real sweep ID was returned by W&B but the durable follow-up write
    did not complete (either because the id itself was refused, or the
    write failed afterward). The external sweep may already exist under the
    returned id -- this exception's ``receipt`` preserves everything needed
    to audit or manually clean it up. Never retried automatically;
    ``wandb.sweep`` is never called a second time for the same
    invocation."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = dict(receipt)


def _is_absolute_path_string(value: str) -> bool:
    """True if ``value`` is absolute either as a native path on this OS or
    as a POSIX path -- so a POSIX-style Moriah path (e.g. ``/sci/labs/...``)
    is correctly recognized as absolute even when this validation runs on a
    Windows dev machine, where ``Path(value)`` alone would treat it as
    relative."""
    return Path(value).is_absolute() or PurePosixPath(value).is_absolute()


def _validate_absolute(value: "str | Path", *, field: str) -> str:
    text = str(value)
    if not _is_absolute_path_string(text):
        raise RegistrationSeamError(f"{field} must be an absolute path, got {text!r}")
    return text


def _validate_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RegistrationSeamError(f"{field} must be a positive integer, got {value!r}")
    return value


def _validate_expected_commit(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RegistrationSeamError(
            f"--expected-commit must be a 40-character lowercase hex git commit sha, got {value!r}"
        )
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RegistrationSeamError(f"descriptor field {field!r} is not a lowercase SHA-256 hex digest: {value!r}")
    return value


def load_and_validate_descriptor(path: "str | Path" = _DESCRIPTOR_PATH) -> dict[str, Any]:
    """Load and strictly validate the one authoritative Common-120
    fixed-support artifact identity descriptor. Never hashes or opens the
    external artifact itself -- only reads and cross-checks the small
    committed identity record."""
    path = Path(path)
    if not path.is_file():
        raise RegistrationSeamError(f"authoritative descriptor not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RegistrationSeamError("descriptor must be a JSON object")

    if data.get("schema_name") != _EXPECTED_SCHEMA_NAME:
        raise RegistrationSeamError(f"descriptor schema_name mismatch: {data.get('schema_name')!r}")
    if data.get("schema_version") != _EXPECTED_SCHEMA_VERSION:
        raise RegistrationSeamError(f"descriptor schema_version mismatch: {data.get('schema_version')!r}")
    if data.get("record_id") != _EXPECTED_RECORD_ID:
        raise RegistrationSeamError(f"descriptor record_id mismatch: {data.get('record_id')!r}")

    artifact = data.get("artifact")
    if not isinstance(artifact, dict):
        raise RegistrationSeamError("descriptor missing artifact section")
    if artifact.get("tracking_status") != _EXPECTED_TRACKING_STATUS:
        raise RegistrationSeamError(f"descriptor artifact.tracking_status mismatch: {artifact.get('tracking_status')!r}")
    # Shape-validated only. Equality between these two identities is not
    # itself an invalid descriptor state -- substitution is prevented by
    # field authority/mapping in ``_descriptor_manifest_bindings`` (which
    # only ever reads ``internal_canonical_contract_sha256``), not by
    # forcing the two values to differ.
    _require_sha256(artifact.get("serialized_file_sha256"), field="artifact.serialized_file_sha256")
    _require_sha256(artifact.get("internal_canonical_contract_sha256"), field="artifact.internal_canonical_contract_sha256")
    deployment_path = artifact.get("deployment_provenance_moriah_absolute_path")
    if not isinstance(deployment_path, str) or not PurePosixPath(deployment_path).is_absolute():
        raise RegistrationSeamError(
            "descriptor artifact.deployment_provenance_moriah_absolute_path must be a POSIX-absolute path"
        )

    contract = data.get("fixed_support_contract")
    if not isinstance(contract, dict):
        raise RegistrationSeamError("descriptor missing fixed_support_contract section")
    if contract.get("schema_name") != CONTRACT_SCHEMA_NAME:
        raise RegistrationSeamError(f"descriptor fixed_support_contract.schema_name mismatch: {contract.get('schema_name')!r}")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise RegistrationSeamError(f"descriptor fixed_support_contract.schema_version mismatch: {contract.get('schema_version')!r}")
    if contract.get("contract_id") != OBJECTIVE_ID_V2:
        raise RegistrationSeamError(f"descriptor fixed_support_contract.contract_id mismatch: {contract.get('contract_id')!r}")
    if contract.get("optimizer_metric") != V2_METRIC_NAME:
        raise RegistrationSeamError(f"descriptor fixed_support_contract.optimizer_metric mismatch: {contract.get('optimizer_metric')!r}")

    bindings = data.get("bindings")
    screening = bindings.get("screening_population") if isinstance(bindings, dict) else None
    if not isinstance(screening, dict):
        raise RegistrationSeamError("descriptor missing bindings.screening_population section")
    if screening.get("policy_identity") != sweep.SCREENING_POLICY_IDENTITY:
        raise RegistrationSeamError(f"descriptor screening policy_identity mismatch: {screening.get('policy_identity')!r}")
    if screening.get("basin_count") != 400:
        raise RegistrationSeamError(f"descriptor screening basin_count must be 400, got {screening.get('basin_count')!r}")
    screening_sha = _require_sha256(screening.get("basin_ids_sha256"), field="bindings.screening_population.basin_ids_sha256")
    if screening_sha != sweep.SCREENING_ARTIFACT_SHA256:
        raise RegistrationSeamError(
            "descriptor bindings.screening_population.basin_ids_sha256 does not match the frozen "
            "SCREENING_ARTIFACT_SHA256 identity constant"
        )

    package_split = bindings.get("package_and_split_sha256") if isinstance(bindings, dict) else None
    if not isinstance(package_split, dict):
        raise RegistrationSeamError("descriptor missing bindings.package_and_split_sha256 section")
    for key in (
        "package_manifest_sha256",
        "package_file_checksums_sha256",
        "package_run_provenance_sha256",
        "development_split_sha256",
        "spatial_holdout_split_sha256",
    ):
        _require_sha256(package_split.get(key), field=f"bindings.package_and_split_sha256.{key}")

    return data


def _descriptor_manifest_bindings(descriptor: Mapping[str, Any]) -> dict[str, str]:
    """Map the validated descriptor onto the exact manifest field names,
    per the binding mapping: never re-derive these values from anywhere
    else, and never substitute the external serialized-file checksum for
    the internal canonical one."""
    artifact = descriptor["artifact"]
    contract = descriptor["fixed_support_contract"]
    screening = descriptor["bindings"]["screening_population"]
    return {
        "fixed_support_contract_path": artifact["deployment_provenance_moriah_absolute_path"],
        "fixed_support_contract_version": contract["contract_id"],
        "fixed_support_contract_sha256": artifact["internal_canonical_contract_sha256"],
        "screening_basin_ids_sha256": screening["basin_ids_sha256"],
    }


def _assert_safe_sweep_id(sweep_id: Any) -> str:
    if sweep_id == FORBIDDEN_V1_SWEEP_ID:
        raise RegistrationSeamError(f"REFUSING: W&B returned the frozen v1 production sweep id {FORBIDDEN_V1_SWEEP_ID!r}")
    if not isinstance(sweep_id, str) or not sweep_id:
        raise RegistrationSeamError(f"REFUSING: W&B returned a non-string/empty sweep id: {sweep_id!r}")
    if sweep_id != sweep_id.strip() or any(char.isspace() for char in sweep_id):
        raise RegistrationSeamError(f"REFUSING: W&B returned a whitespace-containing sweep id: {sweep_id!r}")
    if not _SWEEP_ID_PATTERN.match(sweep_id):
        raise RegistrationSeamError(f"REFUSING: W&B returned a malformed sweep id: {sweep_id!r}")
    return sweep_id


def _assert_production_safe_sweep_id(sweep_id: Any) -> str:
    """Production-only sweep-id gate: every check in
    :func:`_assert_safe_sweep_id`, plus refusal of any id in
    :data:`FORBIDDEN_PRODUCTION_SWEEP_IDS` -- the frozen v1 production sweep
    ``4x3btz2s`` and the CLOSED disposable rehearsal sweep ``oz5p4csb``.

    The generic :func:`_assert_safe_sweep_id` deliberately stays permissive
    about ``oz5p4csb`` because the CLOSED rehearsal registration/launch path
    is legitimately bound to that disposable controller and its historical
    ``mode=rehearsal`` manifests must remain loader-valid. Every v2
    *production* path calls this stricter helper instead.
    """
    if isinstance(sweep_id, str) and sweep_id in FORBIDDEN_PRODUCTION_SWEEP_IDS:
        raise RegistrationSeamError(
            f"REFUSING: sweep id {sweep_id!r} is forbidden for every v2 production path "
            "(frozen v1 production sweep, or the CLOSED disposable rehearsal sweep)"
        )
    return _assert_safe_sweep_id(sweep_id)


def partial_failure_note() -> str:
    """The single shared operator instruction embedded in every v2
    registration partial-failure receipt: the external sweep may already
    exist, and the follow-up write must never be retried automatically."""
    return (
        "External W&B registration may already exist under the returned sweep id. Do not retry "
        "automatically. Investigate on W&B directly and clean up manually if necessary."
    )
