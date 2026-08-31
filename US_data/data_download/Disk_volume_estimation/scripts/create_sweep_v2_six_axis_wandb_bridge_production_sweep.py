"""Offline production-registration readiness for the v2 six-axis W&B bridge.

Minimum maintained tooling for the reusable v2 PRODUCTION controller and its
one-agent launch seam. Nothing here is authorization to register a real
sweep, create a real proposal, or launch an agent -- those are separate,
explicitly authorized steps. No production sweep exists.

Three narrow capabilities, sharing the descriptor authority, strict input
validators, unsafe-sweep-id refusal, and partial-failure semantics with the
CLOSED disposable rehearsal registration seam
(``scripts/create_sweep_v2_six_axis_wandb_bridge_rehearsal_sweep.py``) via
``src.baseline.sweep_v2_six_axis_wandb_bridge_registration``:

* ``register-controller`` -- register exactly one reusable production W&B
  controller through one narrow, injectable ``wandb.sweep(...)`` boundary.
  The controller command is the authoritative static
  ``build_production_sweep_config_v2(...)`` output; it embeds NO
  proposal-specific manifest. ``wandb.sweep`` is never retried
  automatically; if a real sweep id comes back but the durable controller
  receipt cannot be written, the returned identity is preserved in a
  ``RegistrationPartialFailure`` receipt.

* ``build-manifest`` -- after a real production sweep id exists, construct
  and write exactly one strict ``mode=production`` launch manifest for a
  specified ``proposal_order`` and ``execution_generation``, with a fresh
  proposal/attempt-specific output path, the authoritative descriptor-derived
  Common-120 / screening bindings, and no-clobber refusal.

* ``launch-command`` -- emit the exact one-agent invocation
  (``wandb agent --count 1 <sweep_id>``) plus the environment that selects
  one immutable manifest through the ``FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST``
  operational-input seam. It only prints the invocation; it never executes
  it and never contacts W&B.

The sole online W&B boundary is :func:`_call_wandb_sweep`: ``wandb`` is
imported only there, immediately before the one permitted ``wandb.sweep(...)``
call, after every local validation has already passed.
:func:`register_v2_production_controller` accepts an injectable
``register_fn`` so the lifecycle can be exercised end-to-end in tests with
no real network contact.

Must run inside a CPU Slurm allocation (or an interactive CPU-only session)
on Moriah with the canonical ``flashnh-moriah`` interpreter for real
registration; the local dev environment has no W&B credentials, which is why
real registration does not run locally.
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

from src.baseline.sweep_v1_launch_manifest import MODE_PRODUCTION  # noqa: E402
from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract  # noqa: E402
from src.baseline.sweep_v2_six_axis_campaign import (  # noqa: E402
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    OBJECTIVE_ID_V2,
)
from src.baseline.sweep_v2_six_axis_config import (  # noqa: E402
    V2_METRIC_NAME,
    build_production_sweep_config_v2,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import (  # noqa: E402
    SweepV2BridgeManifestError,
    load_v2_wandb_bridge_manifest,
    write_v2_wandb_bridge_manifest,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_registration import (  # noqa: E402
    _CANONICAL_RUNTIME_PYTHON,
    _DEFAULT_WANDB_ENTITY,
    _DEFAULT_WANDB_PROJECT,
    _DESCRIPTOR_PATH,
    _REAL_BASE_PILOT_POLICY_PATH,
    _REAL_BASELINE_POLICY_PATH,
    _REAL_PACKAGE_ROOT,
    _REAL_POLICY_OVERLAY_PATH,
    _REAL_SCREENING_BASIN_IDS_PATH,
    RegistrationPartialFailure,
    RegistrationSeamError,
    _assert_production_safe_sweep_id,
    _descriptor_manifest_bindings,
    _validate_absolute,
    _validate_expected_commit,
    _validate_positive_int,
    load_and_validate_descriptor,
    partial_failure_note,
)

# The bridge program the reusable production controller launches, and the
# environment variable name through which each one-agent job selects its one
# immutable strict manifest (kept in sync with
# ``scripts/run_sweep_v2_six_axis_wandb_bridge.py``).
_BRIDGE_PROGRAM = "scripts/run_sweep_v2_six_axis_wandb_bridge.py"
ENV_V2_PRODUCTION_MANIFEST = "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST"

_STATIC_PRODUCTION_COMMAND = ["${interpreter}", "${program}"]


def _partial_failure_receipt(
    *, reason: str, returned_sweep_id: Any, wandb_project: str, wandb_entity: "str | None", receipt_path: "str | Path",
) -> dict[str, Any]:
    return {
        "status": "partial_failure",
        "reason": reason,
        "returned_wandb_sweep_id": returned_sweep_id,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "intended_receipt_path": str(receipt_path),
        "note": partial_failure_note(),
    }


def build_production_controller_config(*, program: str = _BRIDGE_PROGRAM) -> dict[str, Any]:
    """The authoritative reusable production controller config, verbatim from
    :func:`build_production_sweep_config_v2`, plus a defensive check that the
    command is the static two-macro form and embeds no manifest path."""
    config = build_production_sweep_config_v2(program=program)
    if config.get("command") != _STATIC_PRODUCTION_COMMAND:
        raise RegistrationSeamError(
            f"production controller command must be {_STATIC_PRODUCTION_COMMAND!r} (no embedded manifest), "
            f"got {config.get('command')!r}"
        )
    return config


def _persist_controller_receipt(receipt_path: Path, receipt: Mapping[str, Any]) -> None:
    """Durable, no-clobber write of the controller-registration receipt.
    Isolated so the post-registration failure path is exercisable in tests
    without a second ``register_fn`` call."""
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _call_wandb_sweep(config: Mapping[str, Any], *, project: str, entity: "str | None") -> str:
    """The sole online W&B boundary in this module. ``wandb`` is imported
    only here, immediately before the one permitted registration call, and
    only after every local validation in
    :func:`register_v2_production_controller` has already passed."""
    import wandb

    return wandb.sweep(dict(config), project=project, entity=entity)


def register_v2_production_controller(
    *,
    expected_commit: str,
    receipt_path: "str | Path",
    program: str = _BRIDGE_PROGRAM,
    wandb_project: str = _DEFAULT_WANDB_PROJECT,
    wandb_entity: "str | None" = _DEFAULT_WANDB_ENTITY,
    register_fn: Callable[..., str] = _call_wandb_sweep,
    require_netrc: bool = True,
) -> dict[str, Any]:
    """Register exactly one reusable production W&B controller.

    Strict order: validate local inputs; build the authoritative static
    controller config (no embedded manifest); run the full runtime/Git/
    interpreter contract; call ``register_fn`` (``wandb.sweep`` by default)
    exactly once; reject an unsafe returned sweep identity; durably persist
    the returned controller identity to ``receipt_path`` (atomic no-clobber);
    return a compact non-secret success receipt.

    ``wandb.sweep`` is never called a second time for the same invocation.
    If it returns a real id but the durable receipt cannot be written, the
    returned identity is preserved in a
    :class:`RegistrationPartialFailure` receipt.
    """
    expected_commit = _validate_expected_commit(expected_commit)
    receipt_path = Path(_validate_absolute(receipt_path, field="--receipt-path"))
    if receipt_path.exists():
        raise RegistrationSeamError(f"REFUSING: controller receipt path already exists: {receipt_path}")

    config = build_production_controller_config(program=program)

    run_full_runtime_contract(
        repo_root=_REPO_ROOT, expected_commit=expected_commit,
        expected_runtime_python=_CANONICAL_RUNTIME_PYTHON, require_netrc=require_netrc,
    )

    if receipt_path.exists():
        raise RegistrationSeamError(f"REFUSING: controller receipt path already exists: {receipt_path}")

    # ---- sole W&B boundary: exactly one call, only after every check above ----
    raw_sweep_id = register_fn(config, project=wandb_project, entity=wandb_entity)

    try:
        sweep_id = _assert_production_safe_sweep_id(raw_sweep_id)
    except RegistrationSeamError as exc:
        raise RegistrationPartialFailure(
            str(exc),
            receipt=_partial_failure_receipt(
                reason="unsafe_sweep_id", returned_sweep_id=raw_sweep_id, wandb_project=wandb_project,
                wandb_entity=wandb_entity, receipt_path=receipt_path,
            ),
        ) from exc

    receipt = {
        "status": "success",
        "artifact": "v2_production_controller_registration_receipt",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wandb_sweep_id": sweep_id,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "expected_commit": expected_commit,
        "bridge_program": program,
        "sweep_config_method": config["method"],
        "sweep_config_metric": config["metric"],
        "sweep_config_command": config["command"],
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "objective_id": OBJECTIVE_ID_V2,
        "note": (
            "Reusable production controller registered. No proposal-specific manifest is embedded; "
            "each one-agent job selects one immutable strict mode=production manifest via "
            f"{ENV_V2_PRODUCTION_MANIFEST}. This is not a scientific proposal."
        ),
    }
    try:
        _persist_controller_receipt(receipt_path, receipt)
    except Exception as exc:  # noqa: BLE001 -- any write failure must preserve the returned identity
        raise RegistrationPartialFailure(
            f"production controller {sweep_id!r} was registered but the receipt could not be written: {exc}",
            receipt=_partial_failure_receipt(
                reason="controller_receipt_write_failed", returned_sweep_id=sweep_id, wandb_project=wandb_project,
                wandb_entity=wandb_entity, receipt_path=receipt_path,
            ),
        ) from exc

    return receipt


def _proposal_attempt_output_root(*, output_root_base: str, proposal_order: int, execution_generation: int) -> str:
    """A fresh, proposal/attempt-specific output path: a retry keeps the same
    scientific proposal identity but must use a strictly greater
    ``execution_generation`` and a fresh trial/output directory."""
    base = output_root_base.rstrip("/")
    return f"{base}/proposal_{proposal_order:06d}/execution_generation_{execution_generation:03d}"


def build_production_manifest_fields(
    *,
    wandb_sweep_id: str,
    expected_commit: str,
    output_root: str,
    package_root: str,
    screening_basin_ids_path: str,
    wandb_project: str,
    wandb_entity: "str | None",
    proposal_order: int,
    execution_generation: int,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Strict ``mode=production`` manifest fields: ``stop_before_training``
    is False, exactly one agent, the production objective, and the
    authoritative descriptor-derived Common-120 / screening bindings."""
    return dict(
        manifest_label="sweep_v2_six_axis_wandb_bridge_production_v001",
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode=MODE_PRODUCTION,
        expected_commit=expected_commit,
        repository_root=str(_REPO_ROOT),
        expected_runtime_python=_CANONICAL_RUNTIME_PYTHON,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_sweep_id=wandb_sweep_id,
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
        stop_before_training=False,
        max_agents=1,
        campaign_id=CAMPAIGN_ID_V2,
        domain_version=DOMAIN_VERSION_V2,
        canonicalization_version=CONFIGURATION_CANONICALIZATION_VERSION_V2,
        objective_id=OBJECTIVE_ID_V2,
    )


def write_production_manifest(
    *,
    manifest_path: "str | Path",
    wandb_sweep_id: str,
    expected_commit: str,
    proposal_order: int,
    execution_generation: int,
    output_root_base: str,
    package_root: str = _REAL_PACKAGE_ROOT,
    screening_basin_ids_path: str = _REAL_SCREENING_BASIN_IDS_PATH,
    wandb_project: str = _DEFAULT_WANDB_PROJECT,
    wandb_entity: "str | None" = _DEFAULT_WANDB_ENTITY,
    descriptor_path: "str | Path" = _DESCRIPTOR_PATH,
) -> dict[str, Any]:
    """Construct and atomically no-clobber write exactly one strict
    ``mode=production`` launch manifest for one serialized
    proposal/attempt, after a real production sweep id already exists."""
    expected_commit = _validate_expected_commit(expected_commit)
    manifest_path = Path(_validate_absolute(manifest_path, field="--manifest-path"))
    output_root_base = _validate_absolute(output_root_base, field="--output-root-base")
    package_root = _validate_absolute(package_root, field="--package-root")
    screening_basin_ids_path = _validate_absolute(screening_basin_ids_path, field="--screening-basin-ids-path")
    proposal_order = _validate_positive_int(proposal_order, field="--proposal-order")
    execution_generation = _validate_positive_int(execution_generation, field="--execution-generation")

    # A real production sweep id: never the placeholder, never the frozen v1
    # production sweep, never the CLOSED disposable rehearsal sweep.
    sweep_id = _assert_production_safe_sweep_id(wandb_sweep_id)

    descriptor = load_and_validate_descriptor(descriptor_path)
    bindings = _descriptor_manifest_bindings(descriptor)

    output_root = _proposal_attempt_output_root(
        output_root_base=output_root_base, proposal_order=proposal_order, execution_generation=execution_generation,
    )
    fields = build_production_manifest_fields(
        wandb_sweep_id=sweep_id, expected_commit=expected_commit, output_root=output_root,
        package_root=package_root, screening_basin_ids_path=screening_basin_ids_path,
        wandb_project=wandb_project, wandb_entity=wandb_entity, proposal_order=proposal_order,
        execution_generation=execution_generation, bindings=bindings,
    )
    manifest = write_v2_wandb_bridge_manifest(manifest_path, **fields)

    return {
        "status": "success",
        "artifact": "v2_production_launch_manifest",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": manifest["mode"],
        "stop_before_training": manifest["stop_before_training"],
        "max_agents": manifest["max_agents"],
        "wandb_sweep_id": sweep_id,
        "proposal_order": proposal_order,
        "execution_generation": execution_generation,
        "output_root": output_root,
        "objective_id": manifest["objective_id"],
        "fixed_support_contract_version": bindings["fixed_support_contract_version"],
        "fixed_support_contract_sha256": bindings["fixed_support_contract_sha256"],
        "screening_basin_ids_sha256": bindings["screening_basin_ids_sha256"],
        "descriptor_record_id": descriptor["record_id"],
    }


def resolve_validated_production_launch(
    *,
    manifest_path: "str | Path",
    expected_sweep_id: "str | None" = None,
) -> dict[str, Any]:
    """Authoritative pre-agent manifest/controller join.

    Load the manifest through the strict authoritative loader (checksum +
    schema + full v2 identity), then require it to be an executable
    ``mode=production`` launch manifest: ``mode == production``,
    ``stop_before_training is False``, ``max_agents == 1``, and a
    ``wandb_sweep_id`` that is not the frozen v1 production sweep or the
    CLOSED disposable rehearsal sweep. The strict manifest is the authority
    for the production controller id; the returned ``wandb_sweep_id`` is the
    single identity a launcher may hand to ``wandb agent --count 1``.

    ``expected_sweep_id`` is optional and, when supplied non-empty, purely
    redundant: it must equal the loader-validated manifest id exactly, or
    this refuses -- before any W&B import, W&B initialization, proposal
    intake, or W&B mutation. This function imports and contacts no W&B.
    """
    manifest_path = _validate_absolute(manifest_path, field="--manifest-path")
    manifest = load_v2_wandb_bridge_manifest(manifest_path)

    if manifest["mode"] != MODE_PRODUCTION:
        raise RegistrationSeamError(
            f"pre-agent validation requires a mode=production launch manifest, got mode={manifest['mode']!r}"
        )
    if manifest["stop_before_training"] is not False:
        raise RegistrationSeamError("pre-agent validation requires stop_before_training=false")
    if manifest["max_agents"] != 1:
        raise RegistrationSeamError("pre-agent validation requires max_agents=1")

    sweep_id = _assert_production_safe_sweep_id(manifest["wandb_sweep_id"])

    if expected_sweep_id is not None and str(expected_sweep_id) != "":
        if str(expected_sweep_id) != sweep_id:
            raise RegistrationSeamError(
                f"supplied sweep id {str(expected_sweep_id)!r} contradicts the loader-validated "
                f"mode=production manifest sweep id {sweep_id!r}; the manifest is authoritative"
            )

    return {
        "wandb_sweep_id": sweep_id,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest["manifest_sha256"],
        "mode": manifest["mode"],
        "wandb_project": manifest["wandb_project"],
        "wandb_entity": manifest.get("wandb_entity"),
    }


def build_one_agent_invocation(
    *,
    manifest_path: "str | Path",
    wandb_sweep_id: "str | None" = None,
    wandb_project: "str | None" = None,
    wandb_entity: "str | None" = None,
) -> dict[str, Any]:
    """The exact one-agent invocation plus the operational-input environment
    that selects one immutable strict manifest.

    The manifest is the sole launch identity: it is loaded and validated by
    :func:`resolve_validated_production_launch`, and the ``wandb agent
    --count 1`` target is derived from it. ``wandb_sweep_id``, if supplied,
    is only a redundant cross-check and must match. Constructed only -- never
    executed here, and W&B is never contacted."""
    resolved = resolve_validated_production_launch(
        manifest_path=manifest_path, expected_sweep_id=wandb_sweep_id,
    )
    env = {
        ENV_V2_PRODUCTION_MANIFEST: resolved["manifest_path"],
        "WANDB_PROJECT": wandb_project or resolved["wandb_project"],
    }
    entity = wandb_entity or resolved["wandb_entity"]
    if entity is not None:
        env["WANDB_ENTITY"] = entity
    return {
        "argv": ["wandb", "agent", "--count", "1", resolved["wandb_sweep_id"]],
        "env": env,
        "note": (
            "Run exactly one agent. The manifest is the launch identity, selected through "
            f"{ENV_V2_PRODUCTION_MANIFEST}; the sweep id is derived from the loader-validated "
            "mode=production manifest and the controller command appends no swept CLI args."
        ),
    }


def _add_common_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", default=_DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=_DEFAULT_WANDB_ENTITY)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register-controller", help="Register the reusable production W&B controller (once).")
    register.add_argument("--expected-commit", required=True)
    register.add_argument("--receipt-path", type=Path, required=True)
    register.add_argument("--program", default=_BRIDGE_PROGRAM)
    _add_common_wandb_args(register)

    manifest = sub.add_parser("build-manifest", help="Write one strict mode=production launch manifest.")
    manifest.add_argument("--manifest-path", type=Path, required=True)
    manifest.add_argument("--wandb-sweep-id", required=True)
    manifest.add_argument("--expected-commit", required=True)
    manifest.add_argument("--proposal-order", type=int, required=True)
    manifest.add_argument("--execution-generation", type=int, required=True)
    manifest.add_argument("--output-root-base", required=True)
    manifest.add_argument("--package-root", default=_REAL_PACKAGE_ROOT)
    manifest.add_argument("--screening-basin-ids-path", default=_REAL_SCREENING_BASIN_IDS_PATH)
    _add_common_wandb_args(manifest)

    launch = sub.add_parser("launch-command", help="Emit the one-agent invocation and operational-input env.")
    launch.add_argument("--manifest-path", type=Path, required=True)
    launch.add_argument(
        "--wandb-sweep-id", default=None,
        help="Optional redundant cross-check only; the loader-validated manifest id is authoritative.",
    )

    validate = sub.add_parser(
        "validate-launch",
        help="Pre-agent validator: print ONLY the loader-validated mode=production sweep id (no W&B).",
    )
    validate.add_argument("--manifest-path", type=Path, required=True)
    validate.add_argument(
        "--expect-sweep-id", default=None,
        help="Optional redundant cross-check; if supplied it must equal the manifest's sweep id exactly.",
    )

    args = parser.parse_args(argv)

    if args.command == "register-controller":
        receipt = register_v2_production_controller(
            expected_commit=args.expected_commit, receipt_path=args.receipt_path, program=args.program,
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        )
    elif args.command == "build-manifest":
        receipt = write_production_manifest(
            manifest_path=args.manifest_path, wandb_sweep_id=args.wandb_sweep_id,
            expected_commit=args.expected_commit, proposal_order=args.proposal_order,
            execution_generation=args.execution_generation, output_root_base=args.output_root_base,
            package_root=args.package_root, screening_basin_ids_path=args.screening_basin_ids_path,
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        )
    elif args.command == "validate-launch":
        # Narrow, launcher-safe success output: the bare authoritative sweep
        # id and nothing else, on stdout. No W&B import or contact.
        resolved = resolve_validated_production_launch(
            manifest_path=args.manifest_path, expected_sweep_id=args.expect_sweep_id,
        )
        print(resolved["wandb_sweep_id"])
        return 0
    else:
        receipt = build_one_agent_invocation(
            wandb_sweep_id=args.wandb_sweep_id, manifest_path=args.manifest_path,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistrationPartialFailure as exc:
        print(json.dumps(exc.receipt, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(f"PARTIAL FAILURE: {exc}") from exc
    except SweepV2BridgeManifestError as exc:
        raise SystemExit(f"REFUSING: invalid launch manifest: {exc}") from exc
    except RegistrationSeamError as exc:
        raise SystemExit(f"REFUSING: {exc}") from exc
