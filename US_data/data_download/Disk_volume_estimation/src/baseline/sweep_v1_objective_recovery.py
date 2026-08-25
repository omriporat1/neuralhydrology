"""Idempotent post-training W&B objective recovery for Sweep-v1 trials.

Design per the accepted Sweep-v1 exact-retry startup rehearsal task's
Section F: if a trial completes training and reaches a fully VALID (or
INVALID) immutable Flash-NH ``execution_provenance.json`` record --
``execute_prepared_trial``'s own terminal write -- but a TRANSIENT W&B
failure means that objective was never published to the run/sweep, this
module lets an operator publish it after the fact, WITHOUT retraining and
WITHOUT authorizing any new scientific outcome: it reads the already-final,
already-immutable trial record and republishes exactly what that record
already says.

Never trains. Never calls ``execute_prepared_trial``'s
``execute_prepared_run_fn``. Never mutates ``campaign_id``/``domain_version``/
hyperparameters/``objective_score`` -- this module is read-only with respect
to the Flash-NH record; it only ever writes to W&B (an already-associated
run's summary, via ``wandb.Api()``, never ``wandb.init()``/``wandb.agent()``)
and to its own small local idempotency marker.

NOT EXERCISED against the production sweep in this task: the pure/local
logic below (record loading, identity check, payload derivation, and the
idempotency marker) is fully implemented and unit-tested; the network-facing
``recover_and_publish_objective`` W&B-Api call path is implemented but
deliberately not run against any real sweep here, per the task's explicit
scope boundary.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

REQUIRED_TERMINAL_STATUSES = ("VALID", "INVALID")


class ObjectiveRecoveryError(ValueError):
    pass


def load_immutable_trial_record(execution_provenance_path: "str | Path") -> dict[str, Any]:
    """Load and validate that a trial's ``execution_provenance.json``
    reached a genuinely terminal, immutable state (``execution_status`` is
    ``VALID`` or ``INVALID`` -- never an intermediate ``provenance_stage``
    such as ``wandb_init_failed``/``prepared_with_config``). Refuses
    non-terminal records: there is nothing to "recover" from a trial that
    never finished, and doing so would risk publishing an objective for
    training that never happened.
    """
    path = Path(execution_provenance_path)
    if not path.is_file():
        raise ObjectiveRecoveryError(f"execution_provenance.json not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))

    status = record.get("execution_status")
    if status not in REQUIRED_TERMINAL_STATUSES:
        raise ObjectiveRecoveryError(
            f"refusing to recover a non-terminal trial record (execution_status={status!r}); "
            "only a fully VALID or INVALID immutable record may be republished"
        )
    if record.get("wandb_run_id") is None or record.get("wandb_sweep_id") is None:
        raise ObjectiveRecoveryError(
            "trial record never recorded a wandb_run_id/wandb_sweep_id (the wandb_associated stage "
            "was never reached) -- there is no run to republish an objective onto"
        )
    return record


def assert_matches_expected_identity(record: Mapping[str, Any], expected_identity: Mapping[str, Any]) -> None:
    """Reuses the exact same identity-matching semantics as
    ``sweep_v1_retry.assert_matches_pinned_identity`` -- no parallel identity
    authority. Any key present in ``expected_identity`` must match the
    record exactly, or this raises.
    """
    from src.baseline.sweep_v1_retry import assert_matches_pinned_identity

    assert_matches_pinned_identity(record, expected_identity)


def build_objective_publication_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Pure: derive exactly the W&B summary fields a recovery would publish,
    straight from the immutable record -- no new computation, no new
    scientific fact. Side-effect-free; safe to call without W&B installed.
    """
    return {
        "flashnh/valid": record["execution_status"] == "VALID",
        "flashnh/objective_score": record.get("objective_score"),
        "flashnh/trial_id": record.get("trial_id"),
        "flashnh/retry_of_trial_id": record.get("retry_of_trial_id"),
        "flashnh/execution_generation": record.get("execution_generation"),
        "flashnh/objective_recovered": True,
    }


def is_already_published(marker_path: "str | Path") -> bool:
    """Idempotency check: has this exact record already been republished?
    A prior successful recovery writes ``marker_path`` (durable, project-
    local evidence) -- re-invocation is then a safe no-op, never a second
    network call.
    """
    return Path(marker_path).is_file()


def record_publication(marker_path: "str | Path", *, wandb_run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker = {"wandb_run_id": wandb_run_id, "published_payload": dict(payload)}
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return marker


def recover_and_publish_objective(
    *, execution_provenance_path: "str | Path", expected_identity: Mapping[str, Any],
    marker_path: "str | Path", project: str, entity: "str | None" = None,
) -> dict[str, Any]:
    """Full recovery flow: load + validate the immutable record, skip if
    already published (idempotent), else use ``wandb.Api()`` (never
    ``wandb.init()``/``wandb.agent()``) to update the SAME already-associated
    run's summary -- no new run is created, no controller proposal is
    requested, nothing is retrained.

    NOT exercised against the production sweep in this task -- see module
    docstring.
    """
    record = load_immutable_trial_record(execution_provenance_path)
    assert_matches_expected_identity(record, expected_identity)

    if is_already_published(marker_path):
        return {"status": "already_published", "wandb_run_id": record["wandb_run_id"]}

    payload = build_objective_publication_payload(record)

    import wandb  # lazy import, repo-wide convention

    api = wandb.Api()
    run_path = f"{entity}/{project}/{record['wandb_run_id']}" if entity else f"{project}/{record['wandb_run_id']}"
    run = api.run(run_path)
    if run.sweepId not in (None, record["wandb_sweep_id"]):
        raise ObjectiveRecoveryError(
            f"refusing: run {record['wandb_run_id']!r} is associated with sweep {run.sweepId!r}, "
            f"not the expected {record['wandb_sweep_id']!r}"
        )
    for key, value in payload.items():
        run.summary[key] = value
    run.summary.update()

    record_publication(marker_path, wandb_run_id=record["wandb_run_id"], payload=payload)
    return {"status": "published", "wandb_run_id": record["wandb_run_id"], "payload": payload}
