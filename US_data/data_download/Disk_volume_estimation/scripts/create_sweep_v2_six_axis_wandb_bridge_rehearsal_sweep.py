"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

v2 six-axis sibling of ``scripts/create_sweep_v1_wandb_bridge_rehearsal_sweep.py``.
Registers one disposable, non-scientific Sweep-v2 REHEARSAL sweep against the
real W&B service, and writes the matching
``src.baseline.sweep_v2_six_axis_wandb_bridge_manifest`` launch manifest that
``scripts/run_sweep_v2_six_axis_wandb_bridge.py main_from_manifest`` will
consume when a real ``wandb agent`` invokes it (see
``scripts/run_sweep_v2_six_axis_wandb_bridge_rehearsal_moriah.sbatch``).

Never calls ``wandb.agent()`` or ``wandb.init()`` itself, never trains,
never imports neuralhydrology/torch, and never references the frozen v1
production sweep (``4x3btz2s``) -- the manifest schema
(``sweep_v2_six_axis_wandb_bridge_manifest``) independently refuses a v2
manifest that targets it, and this script additionally refuses a returned
sweep identity that is unsafe (the frozen v1 id, or empty/non-string/
whitespace-containing/malformed) before writing anything.

The authoritative Common-120 fixed-support artifact identity is loaded from
the one committed descriptor,
``config/stage1_v2_common120_fixed_support_artifact_identity_v001.json``,
and strictly validated -- never copied into Python constants -- so the
manifest's ``fixed_support_contract_*`` fields can never silently drift from
the qualified, checksummed artifact record.

Ordering mirrors the v1 script's own chicken-and-egg resolution: the
disposable sweep's own ``command`` field must embed the manifest's absolute
path, so the manifest PATH is fixed first, the sweep is registered
referencing that not-yet-existing path, and only once a real
``wandb_sweep_id`` comes back is the manifest itself written (with that real
id inside it).

The sole online W&B boundary is :func:`_call_wandb_sweep`: ``wandb`` is
imported only there, immediately before the one permitted
``wandb.sweep(...)`` call, after every local validation has already passed.
:func:`register_v2_rehearsal_sweep` accepts an injectable ``register_fn`` so
this module can be exercised end-to-end in tests with a fake/no real network
contact -- production code is not conditional on any test-only hook.

Must run inside a CPU Slurm allocation (or an interactive CPU-only session)
on Moriah with the canonical ``flashnh-moriah`` interpreter -- never the
login node for anything beyond this cheap registration call, and never a
GPU. Requires real W&B network credentials already established on Moriah;
the local Windows dev environment does not have them, which is why real
registration does not run locally (only ``--preflight-only`` and
fake-injected-callable tests do).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.fixed_support_contract_v2 import (  # noqa: E402
    CONTRACT_SCHEMA_NAME,
    CONTRACT_SCHEMA_VERSION,
)
from src.baseline import sweep_v1_campaign as sweep  # noqa: E402
from src.baseline.sweep_v1_launch_manifest import MODE_REHEARSAL  # noqa: E402
from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract  # noqa: E402
from src.baseline.sweep_v2_six_axis_campaign import (  # noqa: E402
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    FORBIDDEN_V1_SWEEP_ID,
    OBJECTIVE_ID_V2,
)
from src.baseline.sweep_v2_six_axis_config import (  # noqa: E402
    V2_METRIC_NAME,
    build_wandb_bridge_rehearsal_sweep_config_v2,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (  # noqa: E402
    build_v2_wandb_bridge_manifest,
    write_v2_wandb_bridge_manifest,
)

# ---------------------------------------------------------------------------
# Fixed identities describing the one authoritative committed descriptor.
# ---------------------------------------------------------------------------
_DESCRIPTOR_PATH = _REPO_ROOT / "config" / "stage1_v2_common120_fixed_support_artifact_identity_v001.json"
_EXPECTED_SCHEMA_NAME = "flashnh_stage1_v2_fixed_support_artifact_identity_record"
_EXPECTED_SCHEMA_VERSION = 1
_EXPECTED_RECORD_ID = "stage1_v2_common120_fixed_support_artifact_identity_v001"
_EXPECTED_TRACKING_STATUS = "external_untracked_large_artifact"

# ---------------------------------------------------------------------------
# Real, already-qualified identities (mirrored, not rederived, from
# scripts/create_sweep_v1_wandb_bridge_rehearsal_sweep.py) -- the v2 bridge
# needs the same real package/screening/policy inputs the v1 bridge uses;
# only the sweep id, manifest, output root, proposal order/generation, and
# fixed-support binding diverge for v2.
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

_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / ".scratch_local" / "sweep_v2_six_axis_wandb_bridge_launch_manifests" / "rehearsal_v001.json"
)
_DEFAULT_OUTPUT_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/evidence/sweep_v2_six_axis_wandb_bridge_rehearsal_v001"

# Clearly non-real, only ever held in memory by ``run_preflight`` -- never
# written to a manifest, never accepted by ``_assert_safe_sweep_id``.
_PREFLIGHT_PLACEHOLDER_SWEEP_ID = "PREFLIGHT-PLACEHOLDER-NOT-A-REAL-SWEEP-ID"

# Every real sweep id observed from this project's own qualified W&B
# registrations (including the frozen v1 production id) is lowercase
# alphanumeric with no separators; this is a conservative shape check, not
# an attempt to fully specify W&B's id grammar.
_SWEEP_ID_PATTERN = re.compile(r"^[0-9a-z]{4,32}$")


class RegistrationSeamError(ValueError):
    """A local validation or unsafe-identity boundary refused to proceed."""


class RegistrationPartialFailure(RegistrationSeamError):
    """A real disposable sweep ID was returned by W&B but the manifest was
    not durably written (either because the id itself was refused, or the
    manifest write failed afterward). The external sweep may already exist
    under the returned id -- this exception's ``receipt`` preserves
    everything needed to audit or manually clean it up. Never retried
    automatically; ``wandb.sweep`` is never called a second time for the
    same invocation."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = dict(receipt)


def _is_absolute_path_string(value: str) -> bool:
    """True if ``value`` is absolute either as a native path on this OS or
    as a POSIX path -- so a POSIX-style Moriah path (e.g.
    ``/sci/labs/...``) is correctly recognized as absolute even when this
    validation runs on a Windows dev machine, where ``Path(value)`` alone
    would treat it as relative."""
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
        raise RegistrationSeamError(f"--expected-commit must be a 40-character lowercase hex git commit sha, got {value!r}")
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
        raise RegistrationSeamError("descriptor artifact.deployment_provenance_moriah_absolute_path must be a POSIX-absolute path")

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


def _build_config_and_bindings(
    *, descriptor: Mapping[str, Any], manifest_path: "str | Path",
) -> tuple[dict[str, Any], dict[str, str]]:
    config = build_wandb_bridge_rehearsal_sweep_config_v2(
        program="scripts/run_sweep_v2_six_axis_wandb_bridge.py", manifest_path=str(manifest_path),
    )
    bindings = _descriptor_manifest_bindings(descriptor)
    return config, bindings


def _manifest_fields(
    *,
    expected_commit: str,
    output_root: str,
    package_root: str,
    screening_basin_ids_path: str,
    wandb_project: str,
    wandb_entity: "str | None",
    sweep_id: str,
    proposal_order: int,
    execution_generation: int,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    return dict(
        manifest_label="sweep_v2_six_axis_wandb_bridge_rehearsal_v001",
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode=MODE_REHEARSAL,
        expected_commit=expected_commit,
        repository_root=str(_REPO_ROOT),
        expected_runtime_python=_CANONICAL_RUNTIME_PYTHON,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_sweep_id=sweep_id,
        output_root=output_root,
        package_root=package_root,
        screening_basin_ids_path=screening_basin_ids_path,
        screening_basin_ids_sha256=bindings["screening_basin_ids_sha256"],
        fixed_support_contract_path=bindings["fixed_support_contract_path"],
        fixed_support_contract_version=bindings["fixed_support_contract_version"],
        fixed_support_contract_sha256=bindings["fixed_support_contract_sha256"],
        baseline_policy_path=_REAL_BASELINE_POLICY_PATH,
        policy_overlay_path=_REAL_POLICY_OVERLAY_PATH,
        base_pilot_policy_path=_REAL_BASE_PILOT_POLICY_PATH,
        proposal_order=proposal_order,
        execution_generation=execution_generation,
        stop_before_training=True,
        max_agents=1,
        campaign_id=CAMPAIGN_ID_V2,
        domain_version=DOMAIN_VERSION_V2,
        canonicalization_version=CONFIGURATION_CANONICALIZATION_VERSION_V2,
        objective_id=OBJECTIVE_ID_V2,
    )


def _validate_common_inputs(
    *,
    expected_commit: str,
    manifest_path: "str | Path",
    output_root: str,
    package_root: str,
    screening_basin_ids_path: str,
    proposal_order: int,
    execution_generation: int,
) -> tuple[str, Path, str, str, str, int, int]:
    expected_commit = _validate_expected_commit(expected_commit)
    manifest_path_str = _validate_absolute(manifest_path, field="--manifest-path")
    output_root = _validate_absolute(output_root, field="--output-root")
    package_root = _validate_absolute(package_root, field="--package-root")
    screening_basin_ids_path = _validate_absolute(screening_basin_ids_path, field="--screening-basin-ids-path")
    if not _is_absolute_path_string(_CANONICAL_RUNTIME_PYTHON):
        raise RegistrationSeamError("canonical runtime python constant is not an absolute path")
    proposal_order = _validate_positive_int(proposal_order, field="--proposal-order")
    execution_generation = _validate_positive_int(execution_generation, field="--execution-generation")

    manifest_path = Path(manifest_path)

    return expected_commit, manifest_path, output_root, package_root, screening_basin_ids_path, proposal_order, execution_generation


def _confirm_manifest_target_absent(manifest_path: "str | Path") -> None:
    """Refuse if ``manifest_path`` already exists. Called both well before
    registration and again immediately before it, to narrow (never
    eliminate -- the real no-clobber guarantee lives in the writer itself)
    the race window between this check and the external registration call."""
    if Path(manifest_path).exists():
        raise RegistrationSeamError(f"REFUSING: manifest path already exists: {manifest_path}")


def _assemble_and_validate_prospective_manifest(
    *,
    descriptor: Mapping[str, Any],
    manifest_path: "str | Path",
    expected_commit: str,
    output_root: str,
    package_root: str,
    screening_basin_ids_path: str,
    wandb_project: str,
    wandb_entity: "str | None",
    proposal_order: int,
    execution_generation: int,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Shared by :func:`run_preflight` and :func:`register_v2_rehearsal_sweep`
    so the two paths never drift into parallel implementations: builds the
    authoritative sweep config and manifest-field bindings, then assembles
    and fully validates (via :func:`build_v2_wandb_bridge_manifest`) a
    prospective strict manifest using the in-memory-only placeholder sweep
    id -- never written to disk, never passed to W&B, never presented as a
    real result. Forces every manifest-shape defect to surface here, before
    any external registration call."""
    config, bindings = _build_config_and_bindings(descriptor=descriptor, manifest_path=manifest_path)
    fields = _manifest_fields(
        expected_commit=expected_commit, output_root=output_root, package_root=package_root,
        screening_basin_ids_path=screening_basin_ids_path, wandb_project=wandb_project, wandb_entity=wandb_entity,
        sweep_id=_PREFLIGHT_PLACEHOLDER_SWEEP_ID, proposal_order=proposal_order,
        execution_generation=execution_generation, bindings=bindings,
    )
    prospective_manifest = build_v2_wandb_bridge_manifest(**fields)
    return config, bindings, prospective_manifest


def run_preflight(
    *,
    expected_commit: str,
    manifest_path: "str | Path",
    output_root: str = _DEFAULT_OUTPUT_ROOT,
    package_root: str = _REAL_PACKAGE_ROOT,
    screening_basin_ids_path: str = _REAL_SCREENING_BASIN_IDS_PATH,
    wandb_project: str = _DEFAULT_WANDB_PROJECT,
    wandb_entity: "str | None" = _DEFAULT_WANDB_ENTITY,
    proposal_order: int,
    execution_generation: int,
    descriptor_path: "str | Path" = _DESCRIPTOR_PATH,
) -> dict[str, Any]:
    """Fully offline dry run: validates every local input, loads/validates
    the descriptor, builds the exact authoritative rehearsal sweep config,
    and constructs/validates prospective manifest fields using a clearly
    non-real placeholder sweep id held only in memory. Imports no W&B,
    writes no manifest, creates no sweep/run."""
    (expected_commit, manifest_path, output_root, package_root, screening_basin_ids_path,
     proposal_order, execution_generation) = _validate_common_inputs(
        expected_commit=expected_commit, manifest_path=manifest_path, output_root=output_root,
        package_root=package_root, screening_basin_ids_path=screening_basin_ids_path,
        proposal_order=proposal_order, execution_generation=execution_generation,
    )
    _confirm_manifest_target_absent(manifest_path)

    descriptor = load_and_validate_descriptor(descriptor_path)
    config, bindings, prospective_manifest = _assemble_and_validate_prospective_manifest(
        descriptor=descriptor, manifest_path=manifest_path, expected_commit=expected_commit,
        output_root=output_root, package_root=package_root, screening_basin_ids_path=screening_basin_ids_path,
        wandb_project=wandb_project, wandb_entity=wandb_entity, proposal_order=proposal_order,
        execution_generation=execution_generation,
    )

    return {
        "preflight": True,
        "note": "PREFLIGHT ONLY -- no sweep registered, no manifest written; placeholder_sweep_id is not a real identity.",
        "manifest_path": str(manifest_path),
        "descriptor_record_id": descriptor["record_id"],
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "objective_id": OBJECTIVE_ID_V2,
        "sweep_config_metric": config["metric"],
        "sweep_config_method": config["method"],
        "max_agents": prospective_manifest["max_agents"],
        "stop_before_training": prospective_manifest["stop_before_training"],
        "proposal_order": proposal_order,
        "execution_generation": execution_generation,
        "placeholder_sweep_id": _PREFLIGHT_PLACEHOLDER_SWEEP_ID,
        "prospective_manifest_sha256": prospective_manifest["manifest_sha256"],
        "fixed_support_contract_version": bindings["fixed_support_contract_version"],
        "fixed_support_contract_sha256": bindings["fixed_support_contract_sha256"],
        "screening_basin_ids_sha256": bindings["screening_basin_ids_sha256"],
        "output_root": output_root,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
    }


def _call_wandb_sweep(config: Mapping[str, Any], *, project: str, entity: "str | None") -> str:
    """The sole online W&B boundary in this module. ``wandb`` is imported
    only here, immediately before the one permitted registration call, and
    only after every local validation in :func:`register_v2_rehearsal_sweep`
    has already passed."""
    import wandb

    return wandb.sweep(dict(config), project=project, entity=entity)


def _partial_failure_receipt(
    *, reason: str, returned_sweep_id: Any, wandb_project: str, wandb_entity: "str | None", manifest_path: "str | Path",
) -> dict[str, Any]:
    return {
        "status": "partial_failure",
        "reason": reason,
        "returned_wandb_sweep_id": returned_sweep_id,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "intended_manifest_path": str(manifest_path),
        "note": (
            "External W&B registration may already exist under the returned sweep id. Do not retry "
            "automatically. Investigate on W&B directly and clean up manually if necessary."
        ),
    }


def register_v2_rehearsal_sweep(
    *,
    expected_commit: str,
    manifest_path: "str | Path",
    output_root: str = _DEFAULT_OUTPUT_ROOT,
    package_root: str = _REAL_PACKAGE_ROOT,
    screening_basin_ids_path: str = _REAL_SCREENING_BASIN_IDS_PATH,
    wandb_project: str = _DEFAULT_WANDB_PROJECT,
    wandb_entity: "str | None" = _DEFAULT_WANDB_ENTITY,
    proposal_order: int,
    execution_generation: int,
    descriptor_path: "str | Path" = _DESCRIPTOR_PATH,
    register_fn: Callable[..., str] = _call_wandb_sweep,
    require_netrc: bool = True,
) -> dict[str, Any]:
    """Full registration lifecycle, in strict order: validate every local
    input; load/validate the authoritative descriptor; build the sweep
    config and manifest-field bindings; confirm the manifest target is
    absent; assemble and fully validate a prospective strict manifest in
    memory (sharing construction/validation with :func:`run_preflight`, via
    an in-memory-only placeholder sweep id -- never written to disk, never
    passed to W&B); run the full runtime/Git/interpreter contract; recheck
    the manifest target immediately before the external call; call
    ``register_fn`` (``wandb.sweep`` by default) exactly once; reject an
    unsafe returned sweep identity; write exactly one matching strict
    rehearsal manifest (via the atomic no-clobber writer); and return a
    compact non-secret success receipt. No local defect discoverable before
    registration is deferred past it.

    ``register_fn`` defaults to the real online boundary
    (:func:`_call_wandb_sweep`); tests substitute a fake callable so this
    lifecycle can be exercised end-to-end without any real network contact.
    """
    (expected_commit, manifest_path, output_root, package_root, screening_basin_ids_path,
     proposal_order, execution_generation) = _validate_common_inputs(
        expected_commit=expected_commit, manifest_path=manifest_path, output_root=output_root,
        package_root=package_root, screening_basin_ids_path=screening_basin_ids_path,
        proposal_order=proposal_order, execution_generation=execution_generation,
    )

    descriptor = load_and_validate_descriptor(descriptor_path)

    _confirm_manifest_target_absent(manifest_path)

    config, bindings, _prospective_manifest = _assemble_and_validate_prospective_manifest(
        descriptor=descriptor, manifest_path=manifest_path, expected_commit=expected_commit,
        output_root=output_root, package_root=package_root, screening_basin_ids_path=screening_basin_ids_path,
        wandb_project=wandb_project, wandb_entity=wandb_entity, proposal_order=proposal_order,
        execution_generation=execution_generation,
    )

    run_full_runtime_contract(
        repo_root=_REPO_ROOT, expected_commit=expected_commit, expected_runtime_python=_CANONICAL_RUNTIME_PYTHON,
        require_netrc=require_netrc,
    )

    # Recheck immediately before the one real external mutation this module
    # performs, narrowing the race window opened since the earlier check.
    _confirm_manifest_target_absent(manifest_path)

    # ---- sole W&B boundary: exactly one call, only after every check above ----
    raw_sweep_id = register_fn(config, project=wandb_project, entity=wandb_entity)

    try:
        sweep_id = _assert_safe_sweep_id(raw_sweep_id)
    except RegistrationSeamError as exc:
        raise RegistrationPartialFailure(
            str(exc),
            receipt=_partial_failure_receipt(
                reason="unsafe_sweep_id", returned_sweep_id=raw_sweep_id, wandb_project=wandb_project,
                wandb_entity=wandb_entity, manifest_path=manifest_path,
            ),
        ) from exc

    fields = _manifest_fields(
        expected_commit=expected_commit, output_root=output_root, package_root=package_root,
        screening_basin_ids_path=screening_basin_ids_path, wandb_project=wandb_project, wandb_entity=wandb_entity,
        sweep_id=sweep_id, proposal_order=proposal_order, execution_generation=execution_generation,
        bindings=bindings,
    )
    try:
        manifest = write_v2_wandb_bridge_manifest(manifest_path, **fields)
    except Exception as exc:
        raise RegistrationPartialFailure(
            f"sweep {sweep_id!r} was registered but the manifest could not be written: {exc}",
            receipt=_partial_failure_receipt(
                reason="manifest_write_failed", returned_sweep_id=sweep_id, wandb_project=wandb_project,
                wandb_entity=wandb_entity, manifest_path=manifest_path,
            ),
        ) from exc

    return {
        "status": "success",
        "wandb_sweep_id": sweep_id,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "output_root": output_root,
        "expected_commit": expected_commit,
        "descriptor_record_id": descriptor["record_id"],
        "fixed_support_contract_version": bindings["fixed_support_contract_version"],
        "fixed_support_contract_sha256": bindings["fixed_support_contract_sha256"],
        "screening_basin_ids_sha256": bindings["screening_basin_ids_sha256"],
        "proposal_order": proposal_order,
        "execution_generation": execution_generation,
        "max_agents": fields["max_agents"],
        "stop_before_training": fields["stop_before_training"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest-path", type=Path, default=_DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--package-root", default=_REAL_PACKAGE_ROOT)
    parser.add_argument("--screening-basin-ids-path", default=_REAL_SCREENING_BASIN_IDS_PATH)
    parser.add_argument("--wandb-project", default=_DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=_DEFAULT_WANDB_ENTITY)
    parser.add_argument("--proposal-order", type=int, required=True)
    parser.add_argument("--execution-generation", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    kwargs = dict(
        expected_commit=args.expected_commit, manifest_path=args.manifest_path, output_root=args.output_root,
        package_root=args.package_root, screening_basin_ids_path=args.screening_basin_ids_path,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        proposal_order=args.proposal_order, execution_generation=args.execution_generation,
    )
    if args.preflight_only:
        receipt = run_preflight(**kwargs)
    else:
        receipt = register_v2_rehearsal_sweep(**kwargs)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistrationPartialFailure as exc:
        print(json.dumps(exc.receipt, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(f"PARTIAL FAILURE: {exc}") from exc
    except RegistrationSeamError as exc:
        raise SystemExit(f"REFUSING: {exc}") from exc
