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

Eligibility (:func:`assert_recovery_eligible`), beyond the earlier
Section F design's generic terminal-status check
(:func:`load_immutable_trial_record`), additionally refuses to recover: a
non-``VALID`` (e.g. ``INVALID``) record; an incomplete record missing a core
identity field; a record with no ``generated_nh_config_sha256`` (missing
source hash); and a missing/non-finite ``objective_score``. Identity
matching (:func:`assert_matches_expected_identity`, via
``sweep_v1_retry.assert_matches_pinned_identity``) additionally rejects a
pinned-but-mismatched ``wandb_run_id``. Idempotent re-publication
(:func:`recover_and_publish_objective`) additionally refuses a repeated
reconciliation whose freshly derived payload disagrees with what was
already durably published under the same marker (changed objective). These
checks were authored in this task, at the operator's explicit direction, to
close a gap identified while attempting the disposable objective-recovery
qualification below: the mechanism previously had no way to demonstrate 5
of 8 required negative-case rejections.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

REQUIRED_TERMINAL_STATUSES = ("VALID", "INVALID")

# Fields a publishable record must carry a non-empty value for, beyond the
# generic terminal-status/wandb-association check in
# ``load_immutable_trial_record``. These are the identity fields
# ``build_objective_publication_payload``/``assert_matches_expected_identity``
# rely on; a record missing any of them is "incomplete" for recovery
# purposes even though it may still be a well-formed terminal record for
# other (non-recovery) consumers.
_REQUIRED_RECOVERY_IDENTITY_FIELDS = (
    "campaign_id", "proposal_id", "configuration_id", "trial_id",
    "execution_generation", "search_arm",
)


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


def assert_recovery_eligible(record: Mapping[str, Any]) -> None:
    """Publication-eligibility checks beyond
    :func:`load_immutable_trial_record`'s generic terminal-status check.

    ``load_immutable_trial_record`` intentionally accepts BOTH ``VALID`` and
    ``INVALID`` terminal records -- that generic loader is also usable by
    other (non-publishing) callers that legitimately need to inspect a
    failed trial's terminal record. Objective RECOVERY specifically only
    ever republishes an already-earned objective, so this function narrows
    eligibility to exactly the trials a recovery may act on. Called by
    :func:`recover_and_publish_objective` before any identity check or W&B
    call.

    Raises :class:`ObjectiveRecoveryError` for:

    * a non-``VALID`` record (e.g. ``INVALID``) -- there is no earned
      objective to recover for a trial that did not pass;
    * an incomplete record missing any of
      :data:`_REQUIRED_RECOVERY_IDENTITY_FIELDS`;
    * a record with no ``generated_nh_config_sha256`` (missing source hash)
      -- without it there is no verifiable provenance link between the
      objective being republished and the exact config that produced it;
    * a missing or non-finite ``objective_score`` (``None``/``NaN``/``Inf``)
      -- defends against a corrupted or hand-edited record, even though a
      genuine VALID record produced by ``execute_prepared_trial`` should
      never have one.
    """
    status = record.get("execution_status")
    if status != "VALID":
        raise ObjectiveRecoveryError(
            f"refusing to recover a non-VALID record (execution_status={status!r}); "
            "objective recovery only republishes an already-earned VALID objective"
        )

    missing = [key for key in _REQUIRED_RECOVERY_IDENTITY_FIELDS if record.get(key) in (None, "")]
    if missing:
        raise ObjectiveRecoveryError(f"refusing to recover an incomplete record: missing {missing}")

    if not record.get("generated_nh_config_sha256"):
        raise ObjectiveRecoveryError(
            "refusing to recover a record with no generated_nh_config_sha256 (missing source hash): "
            "cannot verify provenance of the objective being republished"
        )

    objective_score = record.get("objective_score")
    if (
        not isinstance(objective_score, (int, float))
        or isinstance(objective_score, bool)
        or not math.isfinite(objective_score)
    ):
        raise ObjectiveRecoveryError(
            f"refusing to recover a missing/non-finite objective_score: {objective_score!r}"
        )


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

    The idempotency short-circuit is not a bare "marker file exists" check:
    it recomputes today's publication payload from the record and compares
    it against the payload actually recorded by the prior publication. A
    identical repeat is a true no-op (no W&B call). A DIFFERENT payload
    under the same marker path -- e.g. a tampered/re-executed record
    disagreeing with what was already durably published for this trial
    identity -- is a hard failure, never a silent republish.

    NOT exercised against the production sweep in this task -- see module
    docstring.
    """
    record = load_immutable_trial_record(execution_provenance_path)
    assert_recovery_eligible(record)
    assert_matches_expected_identity(record, expected_identity)

    payload = build_objective_publication_payload(record)

    if is_already_published(marker_path):
        existing_marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        if existing_marker.get("published_payload") != payload:
            raise ObjectiveRecoveryError(
                "refusing: a previously published payload for this trial identity disagrees with the "
                f"freshly derived payload (changed objective) -- previously published="
                f"{existing_marker.get('published_payload')!r}, freshly derived={payload!r}"
            )
        return {"status": "already_published", "wandb_run_id": record["wandb_run_id"]}

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
