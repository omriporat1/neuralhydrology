"""Production W&B bridge for one Sweep-v1 Bayesian trial, and its disposable
REHEARSAL sibling that exercises the identical real-agent round trip without
ever starting NH training.

This is the ``program`` a real ``wandb agent`` invokes once per Bayesian
proposal (see ``build_production_sweep_config`` /
``scripts/build_sweep_v1_production_sweep_config.py``, whose sweep config's
``program`` field defaults to this script's path). Both modes share exactly
one execution core (``_execute_fresh_proposal``) so production and rehearsal
can never silently diverge:

  1. ``wandb.init()`` to join the sweep-assigned run (agent-managed -- no
     explicit ``sweep_id``/``config`` is ever passed to ``wandb.init()``
     itself; the controller supplies both the association and the proposed
     hyperparameters).
  2. Read the proposed hyperparameters via ``run.config`` -- exactly the five
     frozen search axes ``build_production_sweep_config`` declares under
     ``parameters``, pulled by name (never a blind ``dict(run.config)``) so an
     unrelated key wandb might inject can never silently reach the
     preparation layer. ``_validate_controller_config_shape`` additionally
     rejects a controller assignment whose ``run.config`` key set (after
     stripping wandb's own internal ``_wandb`` key) is not EXACTLY the five
     frozen axes -- neither missing nor with an unexpected extra key.
  3. ``_validate_sweep_identity`` -- the joined run's ``run.sweep_id`` must
     equal this invocation's expected sweep (the real production sweep for
     production, or the disposable rehearsal sweep for rehearsal); a
     mismatch in either direction is refused before any durable write.
  4. ``_validate_objective_metric_contract`` -- a read-only ``wandb.Api()``
     check that the joined sweep's own ``method``/``metric`` contract is a
     well-formed Bayesian sweep; production additionally requires it to
     match ``build_production_sweep_config`` exactly (rehearsal may
     legitimately use a disposable metric name/goal).
  5. ``write_proposal_intake_provenance`` -- durably records this exact
     W&B-assigned proposal (five axes, canonical IDs, W&B sweep/run identity)
     to local disk BEFORE any further failure-prone step, so it is always
     recoverable and retryable without asking W&B for a new configuration
     even if preparation or config generation raises. See
     ``src/baseline/sweep_v1_execution.py``'s docstring for the full Layer-B
     provenance design. Its own fresh-output/collision guard refuses to
     silently overwrite an existing durable intake record for the same
     ``trial_id``.
  6. ``canonicalize_wandb_proposal`` + ``prepare_bayesian_proposal`` +
     ``write_prepared_proposal`` -- the same real, already-qualified
     production-adapter path exercised by
     ``tests/test_sweep_v1_production_adapter.py``, with
     ``enrich_layer_b_provenance`` progressively enriching the same durable
     record (stage ``prepared`` after preparation succeeds, stage
     ``prepared_with_config`` after the config is written) rather than
     creating a second provenance authority.
  7. ``select_executor_mode`` -- the same pure, side-effect-free selector the
     exact-retry bridge uses, called explicitly at this script's level (not
     only buried inside step 8) so a rehearsal can record which executor mode
     a real prepared record resolves to without ever dispatching to it.
  8. Rehearsal mode (``stop_before_training=True``) records the selected
     executor mode onto ``run.summary`` and returns immediately -- it never
     calls ``run_prepared_trial_in_production``, never imports/executes NH
     orchestration, and publishes no objective. Production mode
     (``stop_before_training=False``) proceeds to
     ``run_prepared_trial_in_production`` -- the real, fully-tested Sweep-v1
     execution/interpretation layer (``src/baseline/sweep_v1_execution.py``),
     which wires the mature NH orchestration's monolithic executor
     (``pilot_orchestration.execute_prepared_pilot_run_monolithic``) and
     derives VALID/INVALID + ``best_score`` from the authoritative
     prepared-execution receipt. This script never re-derives or
     second-guesses that result. ``slurm_job_id`` is passed through from the
     live ``SLURM_JOB_ID`` environment variable.
  9. Production only: logs ``flashnh/best_score`` (matching
     ``build_production_sweep_config``'s ``metric.name``) as a time-series
     metric so the Bayesian optimizer can use it, and records the remaining
     outcome fields as run-summary values. W&B is a telemetry shell only
     here: it never determines VALID/INVALID or the objective value, and
     ``sweep_v1_execution.py`` has already written the authoritative
     ``review_records.json``/``execution_provenance.json`` to ``output_dir``
     before this script logs anything.

Failure BEFORE a controller assignment can be durably intaked (malformed
``run.config`` shape, sweep-identity mismatch, or objective/metric contract
mismatch) is recorded via ``_write_bootstrap_incident`` -- a project-local,
identity-safe incident record keyed only by the real W&B run id (never a
fabricated proposal/trial identity).

Two entry points share the one execution core, mirroring
``scripts/run_sweep_v1_exact_retry_bridge.py``'s established pattern:

* :func:`main` -- the original CLI-flag/environment interface. Always
  production (``mode="production"``, ``expected_wandb_sweep_id=
  PRODUCTION_WANDB_SWEEP_ID``, ``stop_before_training=False``); this is the
  sole real production entry path and is unchanged in its operational-input
  resolution contract.
* :func:`main_from_manifest` -- consumes one
  ``src.baseline.sweep_v1_wandb_bridge_manifest`` JSON file. Runs the shared
  commit/interpreter/HOME/netrc runtime contract
  (``src.baseline.sweep_v1_runtime_contract.run_full_runtime_contract``)
  before any durable-intake or W&B step, then delegates to the same shared
  core using the manifest's own ``mode``/``wandb_sweep_id``/
  ``proposal_order``/``execution_generation``/``stop_before_training``
  fields. This is the only entry point that can run in rehearsal mode; the
  manifest's own schema (see that module) already refuses a rehearsal
  manifest that targets the production sweep and a production manifest that
  targets anything else, or that sets ``stop_before_training=True``.

``proposal_order`` for the legacy CLI/environment production route remains a
required, caller-supplied positive integer, never inferred from W&B or
auto-numbered here. For BOTH entry points, when the resolved target sweep is
the real production sweep, ``_execute_fresh_proposal`` additionally requires
``proposal_order == PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER`` (the small,
reviewed, committed campaign ledger in ``sweep_v1_wandb_bridge_manifest.py``)
-- the disposable rehearsal sweep's own order lives in a disjoint, explicitly
reserved namespace and can never be confused with or inflate this value (see
that module's docstring for the full rationale).

Operational-input resolution for :func:`main` --
``--package-root``/``--screening-basin-ids``/``--output-root``/
``--proposal-order`` are each resolvable from either an explicit CLI flag or
a ``FLASHNH_SWEEP_V1_*`` environment variable
(``FLASHNH_SWEEP_V1_PACKAGE_ROOT``, ``FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS``,
``FLASHNH_SWEEP_V1_OUTPUT_ROOT``, ``FLASHNH_SWEEP_V1_PROPOSAL_ORDER``). The
environment channel exists because a real ``build_production_sweep_config``
``command`` (``["${interpreter}", "${program}"]``, no ``${args}``) gives W&B
-- not this script's caller -- sole control over this process's argv;
``wandb agent``'s own environment (exported by
``scripts/run_sweep_v1_wandb_agent_moriah.sbatch``) is inherited by every
child process it spawns via standard OS subprocess semantics, so that is the
only channel left to carry these four values into a W&B-invoked run. CLI
flags remain fully supported for direct/manual/test invocation. Resolution
is strict: supplying both a CLI value and an environment value is only
legal when they agree (a mismatch is a hard error); supplying neither is a
hard error; proposal order must additionally be a positive integer. See
``_resolve_path_operational_input``/``_resolve_proposal_order``. Setting
``FLASHNH_SWEEP_V1_BRIDGE_SELFTEST=resolve_only`` makes this script resolve
its four operational inputs, print them as JSON, and exit before importing
``wandb`` or touching the network -- a deterministic hook for testing the
real W&B-constructed argv/environment contract without any live sweep.

Never imports ``wandb`` at module scope (repo-wide lazy-import convention;
see ``scripts/wandb_online_sweep_qualification_run.py``). Never trains or
executes anything itself -- the sole real-training call in this whole path
is inside ``run_prepared_trial_in_production`` ->
``pilot_orchestration.execute_prepared_pilot_run``, and rehearsal mode never
reaches that call. Writing/reading this script performs no live W&B call and
starts no training; it only runs when a real ``wandb agent`` process invokes
it against a real sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v1_execution import (
    build_production_sweep_config, enrich_layer_b_provenance, run_prepared_trial_in_production,
    select_executor_mode, write_proposal_intake_provenance,
)
from src.baseline.sweep_v1_launch_manifest import MODE_PRODUCTION, MODE_REHEARSAL, PRODUCTION_WANDB_SWEEP_ID
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, canonicalize_wandb_proposal, prepare_bayesian_proposal, write_prepared_proposal,
)
from src.baseline.sweep_v1_wandb_bridge_manifest import (
    PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER, REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR,
)

_AXES = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")
_WANDB_INTERNAL_CONFIG_KEYS = frozenset({"_wandb"})

ENV_PACKAGE_ROOT = "FLASHNH_SWEEP_V1_PACKAGE_ROOT"
ENV_SCREENING_BASIN_IDS = "FLASHNH_SWEEP_V1_SCREENING_BASIN_IDS"
ENV_OUTPUT_ROOT = "FLASHNH_SWEEP_V1_OUTPUT_ROOT"
ENV_PROPOSAL_ORDER = "FLASHNH_SWEEP_V1_PROPOSAL_ORDER"
ENV_SELFTEST = "FLASHNH_SWEEP_V1_BRIDGE_SELFTEST"


def _resolve_path_operational_input(*, flag: str, cli_value: "Path | None", env_name: str) -> Path:
    """Strict CLI/environment precedence for one required path input: agree
    -> use it; exactly one supplied -> use it; contradiction or absence ->
    hard error. Never a silent default."""
    env_raw = os.environ.get(env_name)
    env_value = Path(env_raw) if env_raw else None
    if cli_value is not None and env_value is not None:
        if cli_value != env_value:
            raise SystemExit(f"{flag}={cli_value} contradicts {env_name}={env_value}; supply only one")
        return cli_value
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
    raise SystemExit(f"missing required operational input: supply {flag} or set {env_name}")


def _resolve_proposal_order(*, cli_value: "int | None", env_name: str) -> int:
    """Same strict precedence as _resolve_path_operational_input, plus a
    positive-integer check -- proposal order is never inferred or defaulted."""
    env_raw = os.environ.get(env_name)
    env_value = None
    if env_raw:
        try:
            env_value = int(env_raw)
        except ValueError:
            raise SystemExit(f"{env_name}={env_raw!r} is not an integer") from None
    if cli_value is not None and env_value is not None:
        if cli_value != env_value:
            raise SystemExit(f"--proposal-order={cli_value} contradicts {env_name}={env_value}; supply only one")
        resolved = cli_value
    elif cli_value is not None:
        resolved = cli_value
    elif env_value is not None:
        resolved = env_value
    else:
        raise SystemExit(f"missing required operational input: supply --proposal-order or set {env_name}")
    if resolved < 1:
        raise SystemExit(f"proposal order must be a positive integer, got {resolved}")
    return resolved


def _write_bootstrap_incident(*, output_root: Path, wandb_run_id: "str | None", stage: str, reason: str,
                              extra: "dict[str, Any] | None" = None) -> Path:
    """Project-local, identity-safe incident record for a fresh-proposal
    failure that happens BEFORE a trial_id can be derived (malformed
    controller-assignment shape, sweep-identity mismatch, or objective/metric
    contract mismatch). Never fabricates a trial_id/proposal identity --
    keyed only by the real W&B run id (or a fixed placeholder when even that
    is unavailable)."""
    run_key = wandb_run_id or "unknown"
    incident_dir = output_root / f"bootstrap_assignment_rejected__wandb_run_{run_key}"
    payload = {
        "provenance_stage": stage, "rejection_reason": reason, "wandb_run_id": wandb_run_id,
        **(extra or {}),
    }
    incident_dir.mkdir(parents=True, exist_ok=True)
    path = incident_dir / "execution_provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def _validate_sweep_identity(run: Any, *, expected_sweep_id: str, output_root: Path) -> None:
    if run.sweep_id != expected_sweep_id:
        reason = (
            f"run.sweep_id ({run.sweep_id!r}) does not match the sweep expected for this bridge invocation "
            f"({expected_sweep_id!r}); refusing before any durable intake write"
        )
        _write_bootstrap_incident(
            output_root=output_root, wandb_run_id=getattr(run, "id", None), stage="sweep_identity_rejected",
            reason=reason, extra={"actual_sweep_id": run.sweep_id, "expected_sweep_id": expected_sweep_id},
        )
        raise SystemExit(f"REFUSING: {reason}")


def _validate_controller_config_shape(run: Any, *, output_root: Path) -> "dict[str, Any]":
    """Reject a controller assignment whose run.config does not carry
    EXACTLY the five frozen search axes (neither missing nor an unexpected
    extra key, after stripping wandb's own internal ``_wandb`` key). Per-axis
    type/domain validation is handled downstream by
    write_proposal_intake_provenance's own canonical_hyperparameters call;
    this guards only the key set, which that call cannot see (it is only
    ever handed the already-filtered five-key mapping)."""
    actual_keys = set(dict(run.config).keys()) - _WANDB_INTERNAL_CONFIG_KEYS
    expected_keys = set(_AXES)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        reason = (
            "controller-assigned run.config key set does not match the frozen five-axis schema: "
            f"missing={missing} unexpected={unexpected}"
        )
        _write_bootstrap_incident(
            output_root=output_root, wandb_run_id=getattr(run, "id", None), stage="controller_config_shape_rejected",
            reason=reason, extra={"raw_config_keys": sorted(actual_keys), "wandb_sweep_id": getattr(run, "sweep_id", None)},
        )
        raise SystemExit(f"REFUSING: {reason}")
    return {key: run.config[key] for key in _AXES}


def _validate_objective_metric_contract(run: Any, *, mode: str, output_root: Path) -> None:
    """Read-only ``wandb.Api()`` check that the joined sweep's own
    method/metric contract is well-formed. Production additionally requires
    an exact match against ``build_production_sweep_config``'s canonical
    values; rehearsal only requires a structurally valid Bayesian
    method/metric contract (it may legitimately use a disposable metric name
    per the rehearsal qualification design)."""
    import wandb
    api = wandb.Api()
    sweep_path = f"{getattr(run, 'entity', None)}/{getattr(run, 'project', None)}/{run.sweep_id}"
    sweep_config = api.sweep(sweep_path).config
    method = sweep_config.get("method")
    metric = sweep_config.get("metric") or {}
    if method != "bayes" or not metric.get("name") or metric.get("goal") not in ("maximize", "minimize"):
        reason = f"joined sweep's method/metric contract is not a well-formed bayes sweep: method={method!r} metric={metric!r}"
        _write_bootstrap_incident(
            output_root=output_root, wandb_run_id=getattr(run, "id", None),
            stage="objective_metric_contract_rejected", reason=reason,
        )
        raise SystemExit(f"REFUSING: {reason}")
    if mode == MODE_PRODUCTION:
        expected = build_production_sweep_config(program="scripts/run_sweep_v1_wandb_bridge.py")
        if (method != expected["method"] or metric.get("name") != expected["metric"]["name"]
                or metric.get("goal") != expected["metric"]["goal"]):
            reason = (
                "production sweep's method/metric contract does not match build_production_sweep_config: "
                f"got method={method!r} metric={metric!r}, expected method={expected['method']!r} "
                f"metric={expected['metric']!r}"
            )
            _write_bootstrap_incident(
                output_root=output_root, wandb_run_id=getattr(run, "id", None),
                stage="objective_metric_contract_rejected", reason=reason,
            )
            raise SystemExit(f"REFUSING: {reason}")


def _execute_fresh_proposal(
    *, mode: str, package_root: Path, screening_basin_ids: Path, output_root: Path,
    baseline_policy_path: Path, base_pilot_policy_path: Path, expected_wandb_sweep_id: str,
    proposal_order: int, execution_generation: int, stop_before_training: bool,
    extra_intake_fields: "dict[str, Any] | None" = None,
) -> int:
    """Shared execution core for both :func:`main` (legacy CLI/env,
    production) and :func:`main_from_manifest` (production or rehearsal).
    See module docstring for the exact contract and required call ordering.
    """
    if mode not in (MODE_PRODUCTION, MODE_REHEARSAL):
        raise SystemExit(f"unknown mode {mode!r}, expected {MODE_PRODUCTION!r} or {MODE_REHEARSAL!r}")

    if expected_wandb_sweep_id == PRODUCTION_WANDB_SWEEP_ID:
        if mode != MODE_PRODUCTION:
            raise SystemExit(
                f"REFUSING: mode={mode!r} may never target the production sweep ({PRODUCTION_WANDB_SWEEP_ID!r})"
            )
        if proposal_order != PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER:
            raise SystemExit(
                f"REFUSING: production sweep requires proposal_order == "
                f"{PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER!r} (the campaign ledger's next permissible "
                f"production proposal order), got {proposal_order!r}. A disposable rehearsal's own order must "
                "never be used to infer or override this value."
            )
        if execution_generation != 1:
            raise SystemExit(
                "REFUSING: production sweep requires execution_generation == 1 (a fresh controller-assigned "
                "proposal is always a first attempt)"
            )
    else:
        if mode != MODE_REHEARSAL:
            raise SystemExit(
                f"REFUSING: mode={mode!r} must target the production sweep ({PRODUCTION_WANDB_SWEEP_ID!r}); "
                f"got a disposable/non-production sweep ({expected_wandb_sweep_id!r})"
            )
        if proposal_order < REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR:
            raise SystemExit(
                f"REFUSING: rehearsal requires proposal_order >= {REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR!r} "
                f"(the explicit rehearsal namespace); got {proposal_order!r}"
            )
        if execution_generation < REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR:
            raise SystemExit(
                f"REFUSING: rehearsal requires execution_generation >= "
                f"{REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR!r} (the explicit rehearsal namespace); "
                f"got {execution_generation!r}"
            )
        if not stop_before_training:
            raise SystemExit("REFUSING: rehearsal mode requires stop_before_training=True")

    if mode == MODE_PRODUCTION and stop_before_training:
        raise SystemExit("REFUSING: production mode may never set stop_before_training=True")

    canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"
    paths = PreparationPaths(baseline_policy_path, package_root, canonical_splits, screening_basin_ids)

    import wandb
    run = wandb.init()
    valid = False
    try:
        _validate_sweep_identity(run, expected_sweep_id=expected_wandb_sweep_id, output_root=output_root)
        proposed_axes = _validate_controller_config_shape(run, output_root=output_root)
        _validate_objective_metric_contract(run, mode=mode, output_root=output_root)

        # DURABLE PROPOSAL-INTAKE PROVENANCE -- written immediately after the
        # controller assignment passes shape/identity/contract validation and
        # BEFORE any failure-prone artifact/package verification,
        # prepared-proposal construction, config write, executor selection,
        # or mature execution, so this exact W&B-assigned proposal is always
        # locally recoverable without asking W&B for a new one, even if
        # everything below raises.
        intake = write_proposal_intake_provenance(
            output_root=output_root, axes=proposed_axes, search_arm="bayesian",
            proposal_order=proposal_order, wandb_sweep_id=run.sweep_id, wandb_run_id=run.id,
            execution_generation=execution_generation,
        )
        output_dir = output_root / intake["trial_id"]
        if extra_intake_fields:
            intake = enrich_layer_b_provenance(
                output_dir=output_dir, stage=intake["provenance_stage"], fields=extra_intake_fields,
            )

        proposal = canonicalize_wandb_proposal(
            proposed_axes,
            metadata={
                "proposal_order": proposal_order, "execution_generation": execution_generation,
                "wandb_sweep_id": run.sweep_id, "wandb_run_id": run.id,
            },
        )
        prepared = prepare_bayesian_proposal(proposal=proposal, paths=paths)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)

        # allow_layer_b_provenance=True: output_dir already legitimately
        # holds the durable proposal_intake/prepared-stage
        # execution_provenance.json written above -- write_generated_config's
        # empty-directory guard would otherwise always reject it. This flag
        # tolerates ONLY that one pre-existing file (never overwritten,
        # deleted, or moved) and additionally verifies, before writing
        # anything, that its recorded trial_id matches this exact trial --
        # any other pre-existing entry, or a stale/foreign provenance file,
        # is a hard error raised before any generated artifact is written.
        record = write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared_with_config", fields=record)

        executor_mode = select_executor_mode(record)
        enrich_layer_b_provenance(
            output_dir=output_dir, stage="executor_mode_selected", fields={"executor_mode": executor_mode},
        )

        if stop_before_training:
            run.summary["flashnh/rehearsal_stopped_before_training"] = True
            run.summary["flashnh/executor_mode_selected"] = executor_mode
            run.summary["flashnh/trial_id"] = intake["trial_id"]
            run.summary["flashnh/proposal_order"] = proposal_order
            run.summary["flashnh/execution_generation"] = execution_generation
            return 0

        outcome = run_prepared_trial_in_production(
            prepared_record=record, output_dir=output_dir, paths=paths,
            base_pilot_policy_path=base_pilot_policy_path, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        valid = outcome["valid"]
        trial = outcome["review_records"]["trial_summary"]

        if trial["objective_score"] is not None:
            run.log({"flashnh/best_score": trial["objective_score"]})
        run.summary["flashnh/valid"] = valid
        run.summary["flashnh/workflow_status"] = trial["workflow_status"]
        run.summary["flashnh/failure_category"] = trial["failure_category"]
        run.summary["flashnh/trial_id"] = trial["trial_id"]
        run.summary["flashnh/output_dir"] = str(output_dir)
    finally:
        run.finish()

    return 0 if valid else 1


def main_from_manifest(manifest_path: "str | Path") -> int:
    """Manifest-driven entry point: one positional JSON file replaces the
    long CLI/environment-variable channel. Runs the shared commit/
    interpreter/HOME/netrc runtime contract before any durable-intake or W&B
    step, then delegates to the same :func:`_execute_fresh_proposal` core
    :func:`main` uses. This is the only entry point that can run in
    rehearsal mode."""
    from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract
    from src.baseline.sweep_v1_wandb_bridge_manifest import load_wandb_bridge_manifest

    manifest = load_wandb_bridge_manifest(manifest_path)

    run_full_runtime_contract(
        repo_root=ROOT,
        expected_commit=manifest["expected_commit"],
        expected_runtime_python=manifest["expected_runtime_python"],
    )

    manifest_path_resolved = str(Path(manifest_path).resolve())
    return _execute_fresh_proposal(
        mode=manifest["mode"], package_root=Path(manifest["package_root"]),
        screening_basin_ids=Path(manifest["screening_basin_ids_path"]), output_root=Path(manifest["output_root"]),
        baseline_policy_path=Path(manifest["baseline_policy_path"]),
        base_pilot_policy_path=Path(manifest["base_pilot_policy_path"]),
        expected_wandb_sweep_id=manifest["wandb_sweep_id"], proposal_order=int(manifest["proposal_order"]),
        execution_generation=int(manifest["execution_generation"]),
        stop_before_training=bool(manifest["stop_before_training"]),
        extra_intake_fields={
            "launch_manifest_path": manifest_path_resolved,
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "launch_manifest_label": manifest["manifest_label"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--screening-basin-ids", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--proposal-order", type=int, default=None,
                        help="Positive, caller-tracked Bayesian proposal sequence number. "
                             f"May instead be supplied via {ENV_PROPOSAL_ORDER}.")
    parser.add_argument("--baseline-policy-path", type=Path, default=ROOT / "config/stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--base-pilot-policy-path", type=Path, default=ROOT / "config/stage1_lead06_pilot_v001.yaml")
    args = parser.parse_args()

    package_root = _resolve_path_operational_input(flag="--package-root", cli_value=args.package_root, env_name=ENV_PACKAGE_ROOT)
    screening_basin_ids = _resolve_path_operational_input(
        flag="--screening-basin-ids", cli_value=args.screening_basin_ids, env_name=ENV_SCREENING_BASIN_IDS
    )
    output_root = _resolve_path_operational_input(flag="--output-root", cli_value=args.output_root, env_name=ENV_OUTPUT_ROOT)
    proposal_order = _resolve_proposal_order(cli_value=args.proposal_order, env_name=ENV_PROPOSAL_ORDER)

    if os.environ.get(ENV_SELFTEST) == "resolve_only":
        # Deterministic, network-free hook: proves the exact argv/environment
        # W&B will construct resolves correctly, without importing wandb,
        # calling wandb.init(), or running any preparation/training step.
        print(json.dumps({
            "package_root": str(package_root), "screening_basin_ids": str(screening_basin_ids),
            "output_root": str(output_root), "proposal_order": proposal_order,
        }, indent=2))
        return 0

    return _execute_fresh_proposal(
        mode=MODE_PRODUCTION, package_root=package_root, screening_basin_ids=screening_basin_ids,
        output_root=output_root, baseline_policy_path=args.baseline_policy_path,
        base_pilot_policy_path=args.base_pilot_policy_path, expected_wandb_sweep_id=PRODUCTION_WANDB_SWEEP_ID,
        proposal_order=proposal_order, execution_generation=1, stop_before_training=False,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        # Single positional argument: a launch-manifest path (same Design
        # Decision as run_sweep_v1_exact_retry_bridge.py -- no long
        # `--export=ALL,VAR=value,...` interface for new manifest-driven
        # launches). Any flag-style invocation (including every existing CLI/
        # environment-driven production launch and test) falls through to the
        # original argparse-driven main() unchanged.
        raise SystemExit(main_from_manifest(sys.argv[1]))
    raise SystemExit(main())
