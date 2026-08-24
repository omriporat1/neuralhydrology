"""Production execution-context construction and Sweep-v1 scientific
interpretation for one frozen Sweep-v1 trial.

Real execution is deliberately injected via ``execute_prepared_run_fn``: the
production entry point (:func:`run_prepared_trial_in_production`) wires it to
the mature NH orchestration's
:func:`~src.baseline.pilot_orchestration.execute_prepared_pilot_run`; local
qualification injects a fake that returns a synthetic
:class:`~src.baseline.pilot_orchestration.PreparedPilotExecutionResult`
without starting NeuralHydrology. Importing this module never starts NH/torch
by itself -- ``pilot_orchestration`` imports torch lazily inside function
bodies, and only :func:`run_prepared_trial_in_production` (never called by
local tests) actually calls its production entry points.

Sweep-v1 validity is derived directly from the closed prepared-execution
receipt (``PreparedPilotExecutionResult`` + ``actual_optimizer_updates_by_epoch``)
per docs/decision_log.md's 2026-08-23 "Prepared-execution consumer result
contract CLOSED" entry -- see :func:`_derive_validity` for the exact
authoritative fact used for every criterion.

Layer-B provenance (this module's ``execution_provenance.json`` design) is
one progressively-enriched record per trial, not several competing
authorities: :func:`write_proposal_intake_provenance` writes it first, at
stage ``proposal_intake``, immediately after a proposal passes canonical
legality validation and before any failure-prone step;
:func:`enrich_layer_b_provenance` advances it through ``prepared`` and
``prepared_with_config`` as preparation and config generation succeed;
:func:`execute_prepared_trial` then advances it through ``STARTED`` to its
final ``VALID``/``INVALID`` execution status, exactly as before this
provenance was added upstream of it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from . import pilot_orchestration
from . import sweep_v1_campaign as sweep
from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership
from .pilot_lead06_config import PilotPolicy, load_pilot_policy, load_screening_basin_ids
from .pilot_screening_eval import PRIMARY_METRIC_NAME, SCREENING_METRIC_SCOPE
from .sweep_v1_production_adapter import PreparationPaths

__all__ = [
    "SweepV1ExecutionError", "SweepV1ExecutionContext", "build_execution_context",
    "execute_prepared_trial", "run_prepared_trial_in_production", "build_production_sweep_config",
    "write_proposal_intake_provenance", "enrich_layer_b_provenance", "enrich_operations_slurm_accounting",
]

# Mirrors the same frozen screening-population size already hardcoded and
# doubly-asserted in sweep_v1_production_adapter._prepare_proposal
# (expected_count=400) and _audit_generated_config
# (len(bundle.validation_basin_ids) != 400) -- not a new value.
SCREENING_POPULATION_SIZE = 400


class SweepV1ExecutionError(ValueError):
    pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomic mkstemp+os.replace write -- the repo-wide durable-write idiom
    (pilot_orchestration.py, pilot_tracking.py, nh_config_generation.py,
    early_stopping.py, package_netcdf.py). Used only by the new Layer-B
    proposal-intake/enrichment provenance below; ``execute_prepared_trial``'s
    own pre-existing ``_write_json`` is left untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_proposal_intake_provenance(*, output_root: "str | Path", axes: Mapping[str, Any], search_arm: str,
                                     proposal_order: int, wandb_sweep_id: "str | None", wandb_run_id: "str | None",
                                     execution_generation: int = 1, retry_of_trial_id: "str | None" = None
                                     ) -> dict[str, Any]:
    """Durable Layer-B proposal-intake provenance for one W&B-assigned
    proposal, written immediately after the five-axis proposal passes
    canonical legality validation and strictly BEFORE any failure-prone
    artifact/package verification, prepared-proposal construction, config
    generation/write, or mature execution.

    This is the fix for the frozen retry/provenance contract's blocking gap:
    without this durable record, a W&B-assigned proposal could be lost if
    preparation failed before ``execute_prepared_trial`` ever got a chance
    to write its own early ``execution_provenance.json``.

    Identity is derived purely by composing the existing canonical
    ``sweep_v1_campaign`` helpers (``canonical_hyperparameters``,
    ``configuration_id``, ``proposal_id``, ``trial_id``) -- never a parallel
    hashing implementation. All three IDs are provably derivable from the
    five axes plus ``search_arm``/``proposal_order``/``execution_generation``
    alone, with no filesystem I/O, so they are legitimately available at
    intake (unlike config-path/SHA and package/screening identities, which
    only exist after successful preparation -- see
    :func:`enrich_layer_b_provenance`).

    If canonical validation itself fails (e.g. an out-of-domain axis), no
    ``trial_id`` can exist, so a minimal rejection record is instead written
    keyed by the W&B run id, and the original ``ValueError`` is re-raised so
    the caller still exits non-VALID -- the exact proposed axes and W&B
    identity remain durably recoverable even in this edge case.

    Returns the written provenance dict; the caller derives
    ``output_dir = output_root / provenance["trial_id"]`` for every
    subsequent step (preparation, config write, execution all write to the
    SAME ``execution_provenance.json`` -- one coherent Layer-B record, never
    a second competing provenance authority).
    """
    output_root = Path(output_root)
    common = {
        "search_arm": search_arm, "proposal_order": proposal_order, "execution_generation": execution_generation,
        "wandb_sweep_id": wandb_sweep_id, "wandb_run_id": wandb_run_id, "retry_of_trial_id": retry_of_trial_id,
        "git_commit": _git_commit(), "raw_proposed_axes": dict(axes), "objective_score": None,
        "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
    }
    try:
        canonical_axes = sweep.canonical_hyperparameters(dict(axes))
        configuration_id = sweep.configuration_id(canonical_axes)
        proposal_id = sweep.proposal_id(search_arm, proposal_order)
        trial_id = sweep.trial_id(configuration_id, execution_generation=execution_generation)
    except ValueError as exc:
        rejected_dir = output_root / f"proposal_intake_rejected__wandb_run_{wandb_run_id}"
        _write_json_atomic(rejected_dir / "execution_provenance.json", {
            **common, "provenance_stage": "proposal_intake_rejected", "rejection_reason": str(exc),
        })
        raise
    provenance = {
        **common, "provenance_stage": "proposal_intake", "hyperparameters": canonical_axes,
        "configuration_id": configuration_id, "proposal_id": proposal_id, "trial_id": trial_id,
    }
    _write_json_atomic(output_root / trial_id / "execution_provenance.json", provenance)
    return provenance


def enrich_layer_b_provenance(*, output_dir: "str | Path", stage: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Progressively enrich the SAME durable Layer-B
    ``execution_provenance.json`` record written by
    :func:`write_proposal_intake_provenance` -- one coherent record, never a
    second competing provenance authority. Merges ``fields`` over the
    existing record and advances ``provenance_stage``; prior-stage fields
    are retained (not discarded), so the full progression
    (``proposal_intake`` -> ``prepared`` -> ``prepared_with_config``) stays
    inspectable from the final file alone even if a later step never runs.

    Raises :class:`SweepV1ExecutionError` if ``fields`` carries a
    ``trial_id`` that disagrees with the record already on disk -- a defensive
    check against enriching the wrong trial's record, not a new identity
    authority (the identity itself always comes from the canonical helpers
    via :func:`write_proposal_intake_provenance` /
    ``sweep_v1_production_adapter``).
    """
    output_dir = Path(output_dir)
    path = output_dir / "execution_provenance.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    new_trial_id = fields.get("trial_id")
    existing_trial_id = existing.get("trial_id")
    if sweep.trial_identity_conflicts(existing_trial_id, new_trial_id):
        raise SweepV1ExecutionError(
            f"Layer-B provenance trial_id mismatch while enriching to stage {stage!r}: "
            f"{existing_trial_id!r} != {new_trial_id!r}"
        )
    provenance = {**existing, **dict(fields), "provenance_stage": stage}
    _write_json_atomic(path, provenance)
    return provenance


def _require_prepared(record: Mapping[str, Any]) -> None:
    required = {
        "prepare_status": "PASS", "artifact_identity_status": "PASS",
        "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
        "fidelity_id": "mf12x50000", "target_epoch": 12,
        "max_updates_per_epoch": 50_000, "save_weights_every": 1,
        "performance_early_stopping_enabled": False,
        "package_identity": sweep.PACKAGE_IDENTITY,
        "screening_artifact_sha256": sweep.SCREENING_ARTIFACT_SHA256,
        "evaluation_scope": "development_validation_2024_only", "sealed_scope": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise SweepV1ExecutionError(f"prepared-trial contract mismatch: {key}")
    axes = record.get("hyperparameters")
    if not isinstance(axes, dict) or sweep.configuration_id(axes) != record.get("configuration_id"):
        raise SweepV1ExecutionError("prepared-trial configuration identity mismatch")
    if record.get("authoritative_screening_epochs") != list(range(1, 13)):
        raise SweepV1ExecutionError("prepared-trial screening epochs are not exactly 1..12")
    config = Path(str(record.get("generated_nh_config_path", "")))
    if not config.is_file() or hashlib.sha256(config.read_bytes()).hexdigest() != record.get("generated_nh_config_sha256"):
        raise SweepV1ExecutionError("generated NH config SHA-256 mismatch")


@dataclass(frozen=True)
class SweepV1ExecutionContext:
    """Real prepared-execution input plumbing for one trial: everything
    :func:`~src.baseline.pilot_orchestration.execute_prepared_pilot_run_monolithic`
    needs to run this already-prepared trial. Sourced from the written
    prepared proposal, the frozen Sweep-v1 campaign contract, and the same
    package/split/screening loaders preparation already used -- never
    inferred from filenames or duplicated ad hoc."""

    execution_policy: PilotPolicy
    config_dir: Path
    experiment_name: str
    target_variable: str
    lead_hours: int
    screening_basin_ids: "list[str]"
    package_root: Path


def build_execution_context(*, prepared_record: Mapping[str, Any], paths: PreparationPaths,
                            base_pilot_policy_path: "str | Path") -> SweepV1ExecutionContext:
    """Construct the real execution context for one already-written prepared
    trial.

    ``paths`` must be the same
    :class:`~src.baseline.sweep_v1_production_adapter.PreparationPaths` used
    to prepare this trial (identical package/splits/screening artifacts).

    ``target_variable``/``lead_hours`` are read back from the generated
    config's own written generation manifest -- the generated immutable
    config remains authority for training configuration, never a separately
    hardcoded value. ``execution_policy`` is built by layering this frozen
    campaign's scheduling contract (every-epoch screening, no
    performance-based stopping, 12-epoch budget) onto the one committed base
    pilot-policy YAML via ``dataclasses.replace``, the established
    repo-wide idiom for specializing that file per campaign; the base
    policy's own early-stopping/seed/run-matrix fields are left untouched.

    ``screening_validation_every_n_epochs=1`` is load-bearing here: it makes
    every epoch ``1..target_epoch`` classify as a screening epoch for
    :func:`~src.baseline.pilot_screening_eval.classify_screening_epoch_role`,
    which :func:`~src.baseline.pilot_orchestration.execute_prepared_pilot_run_monolithic`
    requires. ``initial_training_epochs=1`` is vestigial for that monolithic
    executor (it never calls :func:`~src.baseline.pilot_orchestration.chunk_epoch_targets`,
    the only reader of that field) -- left set for backward-compatible
    ``PilotPolicy`` construction, not because monolithic execution consults
    it.
    """
    _require_prepared(prepared_record)
    manifest_path = Path(str(prepared_record["generation_manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_variable = str(manifest["target_variable"])
    lead_hours = int(manifest["lead_hours"])

    package_manifest = read_package_manifest(paths.package_root)
    membership = validate_full_population_basin_membership(package_manifest, paths.splits_dir)
    screening_basin_ids = load_screening_basin_ids(
        paths.screening_basin_ids_path, development_basins=membership.development_basins,
        expected_count=SCREENING_POPULATION_SIZE, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256,
    )

    base_policy = load_pilot_policy(base_pilot_policy_path)
    target_epoch = int(prepared_record["target_epoch"])
    execution_policy = replace(
        base_policy,
        lead_hours=lead_hours,
        screening_validation_every_n_epochs=1,
        initial_training_epochs=1,
        pilot_max_epoch_budget=target_epoch,
        performance_early_stopping_enabled=False,
        screening_basin_ids_path=str(paths.screening_basin_ids_path),
        screening_expected_count=len(screening_basin_ids),
        screening_expected_sha256=sweep.SCREENING_ARTIFACT_SHA256,
    )

    return SweepV1ExecutionContext(
        execution_policy=execution_policy,
        config_dir=Path(str(prepared_record["expected_output_dir"])),
        experiment_name=str(prepared_record["trial_id"]),
        target_variable=target_variable,
        lead_hours=lead_hours,
        screening_basin_ids=screening_basin_ids,
        package_root=Path(paths.package_root),
    )


def _population_accounted(raw_space_metrics: Mapping[str, Any], *, expected_requested: int) -> bool:
    """True iff every requested screening basin is accounted for as either
    evaluated or a legitimately traced area-derivation exclusion -- the
    established ``raw_space_metrics_for_run_period`` accounting contract
    (n_basins_requested/n_basins_evaluated/n_basins_area_excluded), not an
    invented completeness rule."""
    requested = raw_space_metrics.get("n_basins_requested")
    evaluated = raw_space_metrics.get("n_basins_evaluated")
    excluded = raw_space_metrics.get("n_basins_area_excluded")
    if requested != expected_requested or evaluated is None or excluded is None:
        return False
    return evaluated + excluded == requested


def _update_evidence_within_cap(cumulative_updates: Mapping[int, int], required_epochs: "set[int]", cap: int) -> bool:
    """``actual_optimizer_updates_by_epoch`` reports the CUMULATIVE optimizer
    step counter as of each checkpointed epoch (confirmed by
    tests/test_prepared_execution_core.py's fake optimizer-state fixture:
    step == epoch * constant). ``max_updates_per_epoch`` is a frozen
    per-epoch CAP, i.e. an upper bound, not a required exact per-epoch
    count -- a dataset-limited epoch may legitimately apply fewer updates.
    This checks: full required-epoch coverage, a non-decreasing cumulative
    counter, and no per-epoch delta exceeding the frozen cap."""
    if not required_epochs.issubset(cumulative_updates):
        return False
    previous = 0
    for epoch in sorted(required_epochs):
        current = int(cumulative_updates[epoch])
        if current < previous or (current - previous) > cap:
            return False
        previous = current
    return True


def _derive_validity(result: "pilot_orchestration.PreparedPilotExecutionResult",
                      prepared_record: Mapping[str, Any], *, expected_screening_population: int
                     ) -> "tuple[bool, dict[int, float] | None, str | None]":
    """Derive Sweep-v1 VALID/INVALID and, if valid, the per-epoch screening
    trajectory directly from the authoritative prepared-execution receipt.
    Returns ``(valid, scores_by_epoch_or_None, failure_category_or_None)``.
    See the module docstring and completion report for the exact
    authoritative fact used per criterion."""
    target_epoch = int(prepared_record["target_epoch"])
    cap = int(prepared_record["max_updates_per_epoch"])
    required = set(range(1, target_epoch + 1))

    if result.blocked or result.blocked_reason is not None:
        return False, None, result.final_status or "blocked_continuation_conflict"
    if result.stopped or result.stop_reason is not None:
        return False, None, "stopped_before_full_budget"

    effective = result.effective_policy
    if int(effective.get("max_epoch_budget", -1)) != target_epoch:
        return False, None, "effective_policy_budget_mismatch"
    if effective.get("performance_early_stopping_enabled") is not False:
        return False, None, "performance_early_stopping_not_disabled"

    checkpoint_epochs = set(result.checkpoint_inventory)
    if not required.issubset(checkpoint_epochs):
        return False, None, "missing_required_checkpoints"

    updates = pilot_orchestration.actual_optimizer_updates_by_epoch(result.nh_run_dir)
    if not _update_evidence_within_cap(updates, required, cap):
        return False, None, "missing_or_cap_violating_update_evidence"

    screening_by_epoch = {int(event["epoch"]): event for event in result.screening_events}
    if not required.issubset(screening_by_epoch):
        return False, None, "missing_required_screening_events"

    scores: "dict[int, float]" = {}
    for epoch in sorted(required):
        event = screening_by_epoch[epoch]
        if event.get("scope") != SCREENING_METRIC_SCOPE or event.get("primary_metric_name") != PRIMARY_METRIC_NAME:
            return False, None, "screening_event_scope_mismatch"
        raw_space_metrics = event.get("raw_space_metrics") or {}
        if not _population_accounted(raw_space_metrics, expected_requested=expected_screening_population):
            return False, None, "incomplete_screening_population"
        try:
            value = float(event.get("primary_metric_median"))
        except (TypeError, ValueError):
            return False, None, "non_finite_screening_score"
        if not math.isfinite(value):
            return False, None, "non_finite_screening_score"
        scores[epoch] = value

    return True, scores, None


def _summarize_receipt(result: "pilot_orchestration.PreparedPilotExecutionResult") -> dict[str, Any]:
    """Compact JSON-safe provenance summary of the authoritative receipt
    (``checkpoint_inventory``/``screening_events`` entries carry non-JSON
    types such as ``Path``/dataclasses, so only their epoch coverage is
    persisted here; the full scientific facts already live in
    ``review_records.json``)."""
    return {
        "final_status": result.final_status, "blocked": result.blocked, "blocked_reason": result.blocked_reason,
        "stopped": result.stopped, "stop_reason": result.stop_reason,
        "effective_policy": dict(result.effective_policy), "nh_run_dir": str(result.nh_run_dir),
        "checkpoint_epochs": sorted(result.checkpoint_inventory),
        "screening_epochs": sorted(int(event["epoch"]) for event in result.screening_events),
    }


def _review_records(record: Mapping[str, Any], *, runtime_seconds: float, gpu_hours: "float | None",
                    screenings: "Mapping[int, float] | None", failure_category: "str | None",
                    retry_of_trial_id: "str | None", slurm_job_id: "str | None" = None) -> dict[str, Any]:
    hp = dict(record["hyperparameters"])
    common = {key: record[key] for key in ("campaign_id", "domain_version", "search_arm", "proposal_id", "configuration_id", "trial_id")}
    if screenings is not None:
        diagnostics = sweep.derive_trajectory_diagnostics(screenings)
        status, objective = "pass", diagnostics["best_score"]
    else:
        diagnostics = {key: None for key in ("best_epoch", "best_score", "final_epoch_score", "best_minus_final", "best_score_10", "best_score_12", "late_gain_10_to_12", "late_best")}
        status, objective = "failed", None
    trial = {**common, "workflow_status": status, "objective_score": objective, **diagnostics, **hp,
             "runtime_seconds": runtime_seconds, "gpu_hours": gpu_hours,
             "execution_generation": record["execution_generation"], "retry_of_trial_id": retry_of_trial_id,
             "failure_category": failure_category}
    proposal = {key: trial[key] for key in sweep.PROPOSAL_RECORD_FIELDS if key in trial}
    proposal.update({"proposal_order": record["proposal_order"], "valid_result_order": None,
                     "boundary_review_checkpoint": None, "wave_id": f"{sweep.DOMAIN_VERSION}_wave1"})
    operations = {key: trial[key] for key in sweep.OPERATIONS_RECORD_FIELDS if key in trial}
    # slurm_job_id is populated from the live SLURM_JOB_ID allocation identity
    # when execute_prepared_trial is given one (see that function's
    # docstring); slurm_state/gpu_hours are only knowable after the job
    # terminates and are populated later, out of band, by
    # enrich_operations_slurm_accounting -- never fabricated here.
    operations.update({"slurm_job_id": slurm_job_id, "slurm_state": None})
    trajectory = [{"campaign_id": record["campaign_id"], "domain_version": record["domain_version"],
                   "configuration_id": record["configuration_id"], "trial_id": record["trial_id"],
                   "search_arm": record["search_arm"], "epoch": epoch,
                   "median_raw_space_nse": value if screenings else None,
                   "evaluation_status": "PASS" if screenings else "FAIL"}
                  for epoch, value in ((sorted(screenings.items())) if screenings else [])]
    for kind, value in (("trial_summary", trial), ("proposal", proposal), ("operations", operations)):
        sweep.validate_review_record(kind, value)
    if screenings:
        for row in trajectory:
            sweep.validate_review_record("epoch_trajectory", row)
    return {"proposal": proposal, "trial_summary": trial, "operations": operations, "epoch_trajectory": trajectory}


def execute_prepared_trial(*, prepared_record: Mapping[str, Any], output_dir: Path,
                           expected_screening_population: int = SCREENING_POPULATION_SIZE,
                           execute_prepared_run_fn: "Callable[[], pilot_orchestration.PreparedPilotExecutionResult]",
                           retry_of_trial_id: "str | None" = None,
                           slurm_job_id: "str | None" = None) -> dict[str, Any]:
    """Persist retry provenance, then execute exactly one continuous prepared
    trial and interpret it using Sweep-v1's scientific contract.

    ``execute_prepared_run_fn`` must use the mature NH train/evaluate path
    and return a
    :class:`~src.baseline.pilot_orchestration.PreparedPilotExecutionResult`
    (production wires it to
    :func:`~src.baseline.pilot_orchestration.execute_prepared_pilot_run_monolithic`
    via :func:`run_prepared_trial_in_production`; local/unit tests inject a
    fake receipt of the same type). Validity is derived from that receipt's
    structured facts plus
    :func:`~src.baseline.pilot_orchestration.actual_optimizer_updates_by_epoch`
    -- see :func:`_derive_validity`.

    ``slurm_job_id`` is the live ``SLURM_JOB_ID`` allocation identity, when
    the caller has one (production always does when launched under
    ``sbatch``/``wandb agent`` on a compute node) -- recorded into
    ``review_records.json``'s ``operations`` block regardless of
    VALID/INVALID, since the allocation identity is a fact about how this
    trial was run, not about whether it succeeded. ``slurm_state``/
    ``gpu_hours`` are NOT populated here (only knowable after the job
    terminates, via ``sacct``/``seff`` on the login node) -- represented as
    ``None``, never fabricated as ``0.0`` or inferred from progress; a
    separate, explicitly-tested :func:`enrich_operations_slurm_accounting`
    call patches them in afterward. Neither field gates VALID/INVALID or the
    objective, and ``None`` passes ``sweep.validate_review_record`` (it
    checks only for missing keys).
    """
    _require_prepared(prepared_record)
    output_dir = Path(output_dir); started = time.time()
    provenance = {"campaign_id": prepared_record["campaign_id"], "proposal_id": prepared_record["proposal_id"],
                  "configuration_id": prepared_record["configuration_id"], "trial_id": prepared_record["trial_id"],
                  "execution_generation": prepared_record["execution_generation"], "search_arm": prepared_record["search_arm"],
                  "retry_of_trial_id": retry_of_trial_id, "git_commit": _git_commit(),
                  "generated_nh_config_path": prepared_record["generated_nh_config_path"],
                  "generated_nh_config_sha256": prepared_record["generated_nh_config_sha256"],
                  "preparation_record": dict(prepared_record), "execution_status": "STARTED"}
    _write_json(output_dir / "execution_provenance.json", provenance)
    try:
        result = execute_prepared_run_fn()
        if not isinstance(result, pilot_orchestration.PreparedPilotExecutionResult):
            raise SweepV1ExecutionError(
                f"execute_prepared_run_fn must return PreparedPilotExecutionResult, got {type(result)!r}"
            )
        valid, scores, failure_category = _derive_validity(
            result, prepared_record, expected_screening_population=expected_screening_population
        )
        records = _review_records(prepared_record, runtime_seconds=time.time() - started,
                                  gpu_hours=None, screenings=scores if valid else None,
                                  failure_category=None if valid else failure_category,
                                  retry_of_trial_id=retry_of_trial_id, slurm_job_id=slurm_job_id)
        result_summary = _summarize_receipt(result)
    except Exception as exc:  # persisted provenance intentionally survives pre-training failure
        result_summary, valid = {"exception": repr(exc)}, False
        records = _review_records(prepared_record, runtime_seconds=time.time() - started, gpu_hours=None,
                                  screenings=None, failure_category="technical_execution_failure",
                                  retry_of_trial_id=retry_of_trial_id, slurm_job_id=slurm_job_id)
    provenance.update({"execution_status": "VALID" if valid else "INVALID", "result": result_summary,
                       "objective_score": records["trial_summary"]["objective_score"]})
    _write_json(output_dir / "execution_provenance.json", provenance)
    _write_json(output_dir / "review_records.json", records)
    return {"valid": valid, "review_records": records, "provenance": provenance}


def enrich_operations_slurm_accounting(*, output_dir: "str | Path", slurm_job_id: str,
                                       slurm_state: str, gpu_hours: float) -> dict[str, Any]:
    """Atomically patch an already-written trial's ``review_records.json``
    ``operations`` block (and ``trial_summary.gpu_hours``) with post-hoc
    Slurm accounting facts, once the job has terminated and ``sacct``/
    ``seff`` data is available on the login node.

    Never called before :func:`execute_prepared_trial` has already written
    the file (there is nothing to enrich yet). Requires ``slurm_job_id`` to
    exactly match the value already recorded in ``operations.slurm_job_id``
    -- refuses to attach accounting facts to the wrong trial rather than
    guessing. Does not touch VALID/INVALID, ``objective_score``, or any
    other field -- Slurm accounting is operational provenance, never a
    scientific gate. Performs no Slurm CLI call itself (keeps this module
    free of the repo-wide "no Slurm-CLI-calling logic outside launcher
    scripts" convention); the caller is responsible for having already
    obtained ``slurm_state``/``gpu_hours`` from ``sacct``/``seff``.
    """
    output_dir = Path(output_dir)
    path = output_dir / "review_records.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    existing_job_id = records["operations"].get("slurm_job_id")
    if existing_job_id != slurm_job_id:
        raise SweepV1ExecutionError(
            f"refusing to attach Slurm accounting for job {slurm_job_id!r}: "
            f"{path} operations.slurm_job_id is {existing_job_id!r}"
        )
    records["operations"]["slurm_state"] = slurm_state
    records["operations"]["gpu_hours"] = gpu_hours
    records["trial_summary"]["gpu_hours"] = gpu_hours
    sweep.validate_review_record("operations", records["operations"])
    sweep.validate_review_record("trial_summary", records["trial_summary"])
    _write_json_atomic(path, records)
    return records


def run_prepared_trial_in_production(*, prepared_record: Mapping[str, Any], output_dir: Path,
                                     paths: PreparationPaths, base_pilot_policy_path: "str | Path",
                                     retry_of_trial_id: "str | None" = None,
                                     slurm_job_id: "str | None" = None) -> dict[str, Any]:
    """The real production entry point: builds the execution context, then
    executes and interprets exactly one prepared trial via the mature NH
    orchestration's MONOLITHIC executor
    (:func:`~src.baseline.pilot_orchestration.execute_prepared_pilot_run_monolithic`
    -- Sweep-v1's generated config always bakes its full ``target_epoch``
    budget in directly, per ``sweep_v1_production_adapter.py``'s single-shot
    fidelity design; the bounded-chunk
    :func:`~src.baseline.pilot_orchestration.execute_prepared_pilot_run` is
    for a different class of campaign whose config trains only an initial
    chunk and advances via repeated ``continue_run`` calls -- see that
    function's docstring). Arm-agnostic -- identical for
    ``search_arm="bayesian"`` and ``search_arm="random_control"`` prepared
    records, since neither :func:`build_execution_context` nor
    :func:`execute_prepared_trial` branches on ``search_arm``; only the
    prepare-time front door differs
    (:func:`~src.baseline.sweep_v1_production_adapter.prepare_bayesian_proposal`
    vs
    :func:`~src.baseline.sweep_v1_production_adapter.prepare_random_control_row`).

    ``slurm_job_id`` is forwarded unchanged to :func:`execute_prepared_trial`
    -- see that function's docstring.

    Never called by local tests (it starts real NH training/evaluation).
    """
    context = build_execution_context(
        prepared_record=prepared_record, paths=paths, base_pilot_policy_path=base_pilot_policy_path
    )

    def _execute() -> "pilot_orchestration.PreparedPilotExecutionResult":
        return pilot_orchestration.execute_prepared_pilot_run_monolithic(
            execution_policy=context.execution_policy, config_dir=context.config_dir,
            experiment_name=context.experiment_name, package_root=context.package_root,
            target_variable=context.target_variable, lead_hours=context.lead_hours,
            screening_basin_ids=context.screening_basin_ids,
            target_epoch=int(prepared_record["target_epoch"]),
        )

    return execute_prepared_trial(
        prepared_record=prepared_record, output_dir=output_dir,
        expected_screening_population=len(context.screening_basin_ids),
        execute_prepared_run_fn=_execute, retry_of_trial_id=retry_of_trial_id,
        slurm_job_id=slurm_job_id,
    )


def build_production_sweep_config(*, program: str) -> dict[str, Any]:
    """Deterministic W&B-facing proposal domain; budget is intentionally absent.

    ``command`` is explicit and deliberately omits the ``${args}`` macro: W&B's
    default command template appends every swept hyperparameter as a
    ``--key=value`` CLI flag, which ``run_sweep_v1_wandb_bridge.py``'s
    argparse does not accept (it reads the five proposed values from
    ``run.config`` instead -- see that script's module docstring). It also
    omits ``${env}`` and any hardcoded path: the four operational inputs the
    bridge needs (package root, screening basin ids, output root, proposal
    order) are supplied via the ``FLASHNH_SWEEP_V1_*`` environment variables
    the sbatch launcher exports into ``wandb agent``'s own process
    environment, which every child process it spawns inherits by standard OS
    subprocess semantics -- keeping this portable search-space definition
    free of machine-specific paths (docs/stage1_phase_b_sweep_v1_launch_contract.md).
    """
    return {"method": "bayes", "metric": {"name": "flashnh/best_score", "goal": "maximize"}, "program": program,
            "command": ["${interpreter}", "${program}"],
            "parameters": {"learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
                           "hidden_size": {"values": [64, 128, 256]},
                           "embedding_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
                           "output_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.4},
                           "batch_size": {"values": [128, 256, 512]}}}
