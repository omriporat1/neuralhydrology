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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.sweep_v1_launch_manifest import MODE_REHEARSAL  # noqa: E402
from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract  # noqa: E402
from src.baseline.sweep_v2_six_axis_campaign import (  # noqa: E402
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    OBJECTIVE_ID_V2,
)
from src.baseline.sweep_v2_six_axis_config import (  # noqa: E402
    build_wandb_bridge_rehearsal_sweep_config_v2,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (  # noqa: E402
    build_v2_wandb_bridge_manifest,
    write_v2_wandb_bridge_manifest,
)

# The descriptor authority, strict input validators, unsafe-sweep-id
# refusal, and partial-failure semantics are shared verbatim with the v2
# PRODUCTION controller registration seam. This script re-exports them so
# its existing test surface (which reaches them as attributes of this
# module) keeps working unchanged.
from src.baseline.sweep_v2_six_axis_wandb_bridge_registration import (  # noqa: E402,F401
    _CANONICAL_RUNTIME_PYTHON,
    _DEFAULT_WANDB_ENTITY,
    _DEFAULT_WANDB_PROJECT,
    _DESCRIPTOR_PATH,
    _PREFLIGHT_PLACEHOLDER_SWEEP_ID,
    _REAL_BASE_PILOT_POLICY_PATH,
    _REAL_BASELINE_POLICY_PATH,
    _REAL_PACKAGE_ROOT,
    _REAL_POLICY_OVERLAY_PATH,
    _REAL_SCREENING_BASIN_IDS_PATH,
    RegistrationPartialFailure,
    RegistrationSeamError,
    _assert_safe_sweep_id,
    _descriptor_manifest_bindings,
    _is_absolute_path_string,
    _require_sha256,
    _validate_absolute,
    _validate_expected_commit,
    _validate_positive_int,
    load_and_validate_descriptor,
    partial_failure_note,
)

_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / ".scratch_local" / "sweep_v2_six_axis_wandb_bridge_launch_manifests" / "rehearsal_v001.json"
)
_DEFAULT_OUTPUT_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/evidence/sweep_v2_six_axis_wandb_bridge_rehearsal_v001"


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
        "note": partial_failure_note(),
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
