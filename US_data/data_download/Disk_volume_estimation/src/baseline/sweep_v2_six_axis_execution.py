"""v2 six-axis execution-spine provenance integration (Section F, additive
six-axis campaign foundation).

Strictly additive sibling of :mod:`sweep_v1_execution`. It reuses, unmodified
and by direct import, every piece of that module that is genuinely
axis/identity-agnostic -- :func:`~sweep_v1_execution.enrich_layer_b_provenance`
(only compares identity keys via ``sweep.trial_identity_conflicts``, the same
function :data:`sweep_v2_six_axis_campaign.trial_identity_conflicts_v2`
literally is), :func:`~sweep_v1_execution._derive_validity` (reads only from
the prepared-record/result objects generically), :func:`~sweep_v1_execution._summarize_receipt`
(pure JSON-safe receipt summary), and the single legal
:data:`~sweep_v1_execution.EXECUTOR_MODE_MONOLITHIC` executor mode -- and
provides genuine v2 siblings only where v1's own functions hardcode v1-literal
identity (``sweep.CAMPAIGN_ID``/``sweep.DOMAIN_VERSION``/``sweep.configuration_id``/
``sweep.validate_review_record``): :func:`write_proposal_intake_provenance_v2`,
:func:`_require_prepared_v2`, :func:`select_executor_mode_v2`,
:func:`_review_records_v2`, :func:`execute_prepared_trial_v2`, and
:func:`enrich_operations_slurm_accounting_v2`.

Never starts NH/torch training itself -- exactly like ``sweep_v1_execution``,
real training dispatch (the v1 equivalents of ``build_execution_context``/
``run_prepared_trial_in_production``) is out of scope here: this module is
exercised only via an injected ``execute_prepared_run_fn`` returning a
synthetic ``pilot_orchestration.PreparedPilotExecutionResult``, never via a
real NH/W&B call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import pilot_orchestration
from . import sweep_v1_campaign as sweep
from .sweep_v1_execution import (
    EXECUTOR_MODE_MONOLITHIC,
    SCREENING_POPULATION_SIZE,
    SweepV1ExecutionError,
    _derive_validity,
    _git_commit,
    _summarize_receipt,
    _write_json,
    _write_json_atomic,
    enrich_layer_b_provenance,
)
from .sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    DOMAIN_VERSION_V2,
    FIDELITY_ID_V2,
    PROPOSAL_RECORD_FIELDS_V2,
    SEARCH_ARMS_V2,
    canonical_hyperparameters_v2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
    validate_review_record_v2,
)

__all__ = [
    "SweepV2ExecutionError",
    "write_proposal_intake_provenance_v2",
    "select_executor_mode_v2",
    "execute_prepared_trial_v2",
    "enrich_operations_slurm_accounting_v2",
]

class SweepV2ExecutionError(ValueError):
    """Raised for any v2 six-axis execution-spine provenance/identity
    violation. Deliberately a distinct type from :class:`SweepV1ExecutionError`
    -- mirrors Section D's ``SweepV2PreparationError``/``SweepV1PreparationError``
    separation, so a v2 caller can never be silently satisfied by a v1
    exception type."""


def write_proposal_intake_provenance_v2(*, output_root: "str | Path", axes: Mapping[str, Any], search_arm: str,
                                        proposal_order: int, wandb_sweep_id: "str | None", wandb_run_id: "str | None",
                                        support_contract_version: str, support_contract_sha256: str,
                                        execution_generation: int = 1, retry_of_trial_id: "str | None" = None,
                                        retry_history: "Sequence[Mapping[str, Any]] | None" = None
                                        ) -> dict[str, Any]:
    """v2 sibling of :func:`sweep_v1_execution.write_proposal_intake_provenance`.

    v1's function hardcodes ``sweep.CAMPAIGN_ID``/``sweep.DOMAIN_VERSION``
    into the written ``common`` envelope and calls
    ``sweep.canonical_hyperparameters``/``sweep.configuration_id``/
    ``sweep.proposal_id``/``sweep.trial_id`` -- all v1-specific -- so it
    cannot durably intake a v2 six-axis proposal. This sibling performs the
    identical intake-before-any-failure-prone-step contract (see v1's
    docstring, unchanged here) using the v2 six-axis canonical helpers, and
    additionally requires and persists the frozen fixed-support contract
    identity (``support_contract_version``/``support_contract_sha256``) --
    required because :func:`configuration_id_v2` cannot be computed without
    it (the binding requirement to bind fixed-support identity into v2
    configuration identity applies at intake, not only at preparation).
    """
    output_root = Path(output_root)
    common = {
        "search_arm": search_arm, "proposal_order": proposal_order, "execution_generation": execution_generation,
        "wandb_sweep_id": wandb_sweep_id, "wandb_run_id": wandb_run_id, "retry_of_trial_id": retry_of_trial_id,
        "retry_history": [dict(item) for item in (retry_history or [])],
        "git_commit": _git_commit(), "raw_proposed_axes": dict(axes), "objective_score": None,
        "campaign_id": CAMPAIGN_ID_V2, "domain_version": DOMAIN_VERSION_V2,
        "support_contract_version": support_contract_version, "support_contract_sha256": support_contract_sha256,
    }
    try:
        canonical_axes = canonical_hyperparameters_v2(dict(axes))
        configuration_id = configuration_id_v2(
            canonical_axes, support_contract_version=support_contract_version,
            support_contract_sha256=support_contract_sha256,
        )
        pid = proposal_id_v2(search_arm, proposal_order)
        tid = trial_id_v2(configuration_id, pid, execution_generation=execution_generation)
    except ValueError as exc:
        rejected_dir = output_root / f"proposal_intake_rejected__wandb_run_{wandb_run_id}"
        _write_json_atomic(rejected_dir / "execution_provenance.json", {
            **common, "provenance_stage": "proposal_intake_rejected", "rejection_reason": str(exc),
        })
        raise
    output_dir = output_root / tid
    if output_dir.exists():
        raise SweepV2ExecutionError(
            f"v2 proposal intake output already exists for trial_id {tid!r} at {output_dir} -- a fresh "
            "proposal must never overwrite existing durable intake; a genuine retry must use a strictly "
            "greater execution_generation, which always derives a distinct trial_id"
        )
    provenance = {
        **common, "provenance_stage": "proposal_intake", "hyperparameters": canonical_axes,
        "configuration_id": configuration_id, "proposal_id": pid, "trial_id": tid,
    }
    _write_json_atomic(output_dir / "execution_provenance.json", provenance)
    return provenance


def _require_prepared_v2(record: Mapping[str, Any]) -> None:
    """v2 sibling of :func:`sweep_v1_execution._require_prepared`. Same
    fidelity/epoch/update-cap/early-stopping/package/screening contract
    (scientifically unchanged between v1 and v2 -- only the search axes and
    objective/support contract differ), checked against v2 campaign/domain
    identity, plus the frozen fixed-support contract identity bound into
    :func:`configuration_id_v2`."""
    required = {
        "prepare_status": "PASS", "artifact_identity_status": "PASS",
        "campaign_id": CAMPAIGN_ID_V2, "domain_version": DOMAIN_VERSION_V2,
        "fidelity_id": FIDELITY_ID_V2, "target_epoch": 12,
        "max_updates_per_epoch": 50_000, "save_weights_every": 1,
        "performance_early_stopping_enabled": False,
        "package_identity": sweep.PACKAGE_IDENTITY,
        "screening_artifact_sha256": sweep.SCREENING_ARTIFACT_SHA256,
        "evaluation_scope": "development_validation_2024_only", "sealed_scope": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise SweepV2ExecutionError(f"prepared-trial contract mismatch: {key}")
    axes = record.get("hyperparameters")
    support_contract_version = record.get("support_contract_version")
    support_contract_sha256 = record.get("support_contract_sha256")
    if not isinstance(support_contract_version, str) or not support_contract_version:
        raise SweepV2ExecutionError("prepared-trial support_contract_version is missing")
    if not isinstance(support_contract_sha256, str) or not support_contract_sha256:
        raise SweepV2ExecutionError("prepared-trial support_contract_sha256 is missing")
    if not isinstance(axes, dict) or configuration_id_v2(
        axes, support_contract_version=support_contract_version, support_contract_sha256=support_contract_sha256
    ) != record.get("configuration_id"):
        raise SweepV2ExecutionError("prepared-trial configuration identity mismatch")
    if record.get("authoritative_screening_epochs") != list(range(1, 13)):
        raise SweepV2ExecutionError("prepared-trial screening epochs are not exactly 1..12")
    config = Path(str(record.get("generated_nh_config_path", "")))
    if not config.is_file():
        raise SweepV2ExecutionError("generated NH config is missing")
    import hashlib
    if hashlib.sha256(config.read_bytes()).hexdigest() != record.get("generated_nh_config_sha256"):
        raise SweepV2ExecutionError("generated NH config SHA-256 mismatch")


def select_executor_mode_v2(prepared_record: Mapping[str, Any]) -> str:
    """v2 sibling of :func:`sweep_v1_execution.select_executor_mode`. Reuses
    the same :data:`EXECUTOR_MODE_MONOLITHIC` constant -- v2's generated
    config, like v1's, always bakes its full ``target_epoch`` budget in
    directly. Never imports or touches ``pilot_orchestration``/NH/torch."""
    _require_prepared_v2(prepared_record)
    return EXECUTOR_MODE_MONOLITHIC


def _review_records_v2(record: Mapping[str, Any], *, runtime_seconds: float, gpu_hours: "float | None",
                       screenings: "Mapping[int, float] | None", failure_category: "str | None",
                       retry_of_trial_id: "str | None", slurm_job_id: "str | None" = None) -> dict[str, Any]:
    """v2 sibling of :func:`sweep_v1_execution._review_records`. Identical
    shape and diagnostics math (``sweep.derive_trajectory_diagnostics`` is
    axis-agnostic, reused verbatim), but: ``hp`` carries all six v2 axes
    (including ``seq_length``, which is why :data:`TRIAL_SUMMARY_FIELDS_V2`
    is required for the ``trial_summary`` schema), the proposal record uses
    :data:`PROPOSAL_RECORD_FIELDS_V2`, the operations record reuses v1's own
    axis-agnostic ``OPERATIONS_RECORD_FIELDS`` verbatim, ``wave_id`` is
    derived from :data:`DOMAIN_VERSION_V2` (never v1's ``DOMAIN_VERSION``),
    and every record is validated via :func:`validate_review_record_v2`."""
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
    proposal = {key: trial[key] for key in PROPOSAL_RECORD_FIELDS_V2 if key in trial}
    proposal.update({"proposal_order": record["proposal_order"], "valid_result_order": None,
                     "boundary_review_checkpoint": None, "wave_id": f"{DOMAIN_VERSION_V2}_wave1"})
    operations = {key: trial[key] for key in sweep.OPERATIONS_RECORD_FIELDS if key in trial}
    operations.update({"slurm_job_id": slurm_job_id, "slurm_state": None})
    trajectory = [{"campaign_id": record["campaign_id"], "domain_version": record["domain_version"],
                   "configuration_id": record["configuration_id"], "trial_id": record["trial_id"],
                   "search_arm": record["search_arm"], "epoch": epoch,
                   "median_raw_space_nse": value if screenings else None,
                   "evaluation_status": "PASS" if screenings else "FAIL"}
                  for epoch, value in ((sorted(screenings.items())) if screenings else [])]
    for kind, value in (("trial_summary", trial), ("proposal", proposal), ("operations", operations)):
        validate_review_record_v2(kind, value)
    if screenings:
        for row in trajectory:
            validate_review_record_v2("epoch_trajectory", row)
    return {"proposal": proposal, "trial_summary": trial, "operations": operations, "epoch_trajectory": trajectory}


def execute_prepared_trial_v2(*, prepared_record: Mapping[str, Any], output_dir: Path,
                              expected_screening_population: int = SCREENING_POPULATION_SIZE,
                              execute_prepared_run_fn: "Callable[[], pilot_orchestration.PreparedPilotExecutionResult]",
                              retry_of_trial_id: "str | None" = None,
                              slurm_job_id: "str | None" = None,
                              executor_mode: "str | None" = None) -> dict[str, Any]:
    """v2 sibling of :func:`sweep_v1_execution.execute_prepared_trial`.

    Reuses :func:`enrich_layer_b_provenance` (generic), :func:`_derive_validity`
    (generic -- Sweep-v1/v2 validity is derived from the identical
    prepared-execution receipt contract; only the search axes/objective
    differ, not the fidelity/validity rules) and :func:`_summarize_receipt`
    (generic) directly, unmodified, from ``sweep_v1_execution``. Only
    :func:`_require_prepared_v2` and :func:`_review_records_v2` (v2-identity
    aware) differ from v1's function.
    """
    _require_prepared_v2(prepared_record)
    output_dir = Path(output_dir); started = time.time()
    started_fields: dict[str, Any] = {
        "campaign_id": prepared_record["campaign_id"], "proposal_id": prepared_record["proposal_id"],
        "configuration_id": prepared_record["configuration_id"], "trial_id": prepared_record["trial_id"],
        "execution_generation": prepared_record["execution_generation"], "search_arm": prepared_record["search_arm"],
        "git_commit": _git_commit(),
        "generated_nh_config_path": prepared_record["generated_nh_config_path"],
        "generated_nh_config_sha256": prepared_record["generated_nh_config_sha256"],
        "preparation_record": dict(prepared_record), "executor_mode": executor_mode,
        "execution_status": "STARTED",
    }
    if retry_of_trial_id is not None:
        started_fields["retry_of_trial_id"] = retry_of_trial_id
    provenance = enrich_layer_b_provenance(output_dir=output_dir, stage="STARTED", fields=started_fields)
    try:
        result = execute_prepared_run_fn()
        if not isinstance(result, pilot_orchestration.PreparedPilotExecutionResult):
            raise SweepV2ExecutionError(
                f"execute_prepared_run_fn must return PreparedPilotExecutionResult, got {type(result)!r}"
            )
        valid, scores, failure_category = _derive_validity(
            result, prepared_record, expected_screening_population=expected_screening_population
        )
        records = _review_records_v2(prepared_record, runtime_seconds=time.time() - started,
                                     gpu_hours=None, screenings=scores if valid else None,
                                     failure_category=None if valid else failure_category,
                                     retry_of_trial_id=retry_of_trial_id, slurm_job_id=slurm_job_id)
        result_summary = _summarize_receipt(result)
    except Exception as exc:  # persisted provenance intentionally survives pre-training failure
        result_summary, valid = {"exception": repr(exc)}, False
        records = _review_records_v2(prepared_record, runtime_seconds=time.time() - started, gpu_hours=None,
                                     screenings=None, failure_category="technical_execution_failure",
                                     retry_of_trial_id=retry_of_trial_id, slurm_job_id=slurm_job_id)
    terminal_fields = {"execution_status": "VALID" if valid else "INVALID", "result": result_summary,
                       "objective_score": records["trial_summary"]["objective_score"]}
    provenance = enrich_layer_b_provenance(
        output_dir=output_dir, stage="VALID" if valid else "INVALID", fields=terminal_fields
    )
    _write_json(output_dir / "review_records.json", records)
    return {"valid": valid, "review_records": records, "provenance": provenance}


def enrich_operations_slurm_accounting_v2(*, output_dir: "str | Path", slurm_job_id: str,
                                          slurm_state: str, gpu_hours: float) -> dict[str, Any]:
    """v2 sibling of :func:`sweep_v1_execution.enrich_operations_slurm_accounting`.
    Identical no-guessing/exact-job-id-match contract; validates the patched
    records via :func:`validate_review_record_v2` instead of v1's validator."""
    output_dir = Path(output_dir)
    path = output_dir / "review_records.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    existing_job_id = records["operations"].get("slurm_job_id")
    if existing_job_id != slurm_job_id:
        raise SweepV2ExecutionError(
            f"refusing to attach Slurm accounting for job {slurm_job_id!r}: "
            f"{path} operations.slurm_job_id is {existing_job_id!r}"
        )
    records["operations"]["slurm_state"] = slurm_state
    records["operations"]["gpu_hours"] = gpu_hours
    records["trial_summary"]["gpu_hours"] = gpu_hours
    validate_review_record_v2("operations", records["operations"])
    validate_review_record_v2("trial_summary", records["trial_summary"])
    _write_json_atomic(path, records)
    return records
