"""Idempotent post-training W&B objective recovery for Sweep-v2 six-axis
trials.

Strictly additive sibling of :mod:`sweep_v1_objective_recovery` (Section G,
additive six-axis campaign foundation). Most of that module's functions
(``load_immutable_trial_record``, ``assert_recovery_eligible``,
``build_objective_publication_payload``, ``is_already_published``,
``record_publication``) are already genuinely campaign-agnostic -- none of
them reference ``CAMPAIGN_ID``/``DOMAIN_VERSION`` or any five-axis field, they
only inspect the generic terminal-record shape any campaign's
``execution_provenance.json`` shares -- so they are imported and reused
directly here, unchanged, per this task's established reuse-vs-sibling
convention. The two genuinely v1-coupled pieces get real v2 siblings:
``assert_matches_expected_identity`` (delegates to
``sweep_v1_retry.assert_matches_pinned_identity``, which hardcodes the
five-axis field set) and the top-level orchestration function (which must
also refuse to recover a record bound to a foreign campaign/domain -- v1's
own orchestration function never needed this check because
``sweep_v1_retry``'s identity loader already rejects non-v1 records upstream
of it, but this module's reused ``load_immutable_trial_record`` does not
gate on campaign identity at all, so the v2 orchestration function below adds
that gate explicitly).

Never trains. Never calls ``execute_prepared_trial_v2``'s
``execute_prepared_run_fn``. Never mutates ``campaign_id``/``domain_version``/
hyperparameters/``objective_score`` -- this module is read-only with respect
to the Flash-NH record; it only ever writes to W&B (an already-associated
run's summary, via ``wandb.Api()``, never ``wandb.init()``/``wandb.agent()``)
and to its own small local idempotency marker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .sweep_v1_objective_recovery import (
    ObjectiveRecoveryError,
    REQUIRED_TERMINAL_STATUSES,
    assert_recovery_eligible,
    is_already_published,
    load_immutable_trial_record,
    record_publication,
)
from .sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, DOMAIN_VERSION_V2
from .sweep_v2_six_axis_execution import build_v2_objective_publication_payload

__all__ = [
    "SweepV2ObjectiveRecoveryError",
    "ObjectiveRecoveryError",
    "REQUIRED_TERMINAL_STATUSES",
    "load_immutable_trial_record",
    "assert_recovery_eligible",
    "assert_v2_campaign_identity",
    "assert_matches_expected_identity_v2",
    "build_v2_objective_publication_payload",
    "is_already_published",
    "record_publication",
    "recover_and_publish_objective_v2",
]


class SweepV2ObjectiveRecoveryError(ValueError):
    """Raised for a v2-specific objective-recovery violation: a record bound
    to a foreign (e.g. v1) campaign/domain, or a pinned-identity mismatch
    against the v2 six-axis field set."""


def assert_v2_campaign_identity(record: Mapping[str, Any]) -> None:
    """Refuse to recover a record that is not bound to the frozen Sweep-v2
    six-axis campaign/domain -- in particular, refuses a genuine v1 record
    (or any other foreign record) presented to the v2 recovery entrypoint.
    :func:`load_immutable_trial_record` (reused unchanged from v1) has no
    campaign-identity gate of its own, so this check is applied explicitly,
    immediately after loading and before any eligibility/identity/W&B call.
    """
    if record.get("campaign_id") != CAMPAIGN_ID_V2 or record.get("domain_version") != DOMAIN_VERSION_V2:
        raise SweepV2ObjectiveRecoveryError(
            "refusing to recover a record not bound to the frozen Sweep-v2 six-axis campaign/domain: "
            f"campaign_id={record.get('campaign_id')!r}, domain_version={record.get('domain_version')!r}"
        )


def assert_matches_expected_identity_v2(record: Mapping[str, Any], expected_identity: Mapping[str, Any]) -> None:
    """v2 sibling of :func:`sweep_v1_objective_recovery.assert_matches_expected_identity`.

    Reuses the exact same identity-matching semantics as
    ``sweep_v2_six_axis_retry.assert_matches_pinned_identity_v2`` -- no
    parallel identity authority -- extended to the six-axis field set
    (including ``seq_length`` and the fixed-support-contract identity
    fields) instead of v1's five.
    """
    from .sweep_v2_six_axis_retry import assert_matches_pinned_identity_v2

    assert_matches_pinned_identity_v2(record, expected_identity)


def recover_and_publish_objective_v2(
    *, execution_provenance_path: "str | Path", expected_identity: Mapping[str, Any],
    marker_path: "str | Path", project: str, entity: "str | None" = None,
) -> dict[str, Any]:
    """v2 sibling of :func:`sweep_v1_objective_recovery.recover_and_publish_objective`.

    Identical flow -- load + validate the immutable record, skip if already
    published (idempotent), else use ``wandb.Api()`` (never
    ``wandb.init()``/``wandb.agent()``) to update the SAME already-associated
    run's summary -- with one addition: :func:`assert_v2_campaign_identity`
    is checked immediately after load, before eligibility or identity
    matching, so a v1 (or otherwise foreign) record can never be recovered
    through this v2 entrypoint. No new run is created, no controller
    proposal is requested, nothing is retrained.

    NOT exercised against any production sweep in this task -- purely local,
    no W&B contact occurs in any test of this function.
    """
    record = load_immutable_trial_record(execution_provenance_path)
    assert_v2_campaign_identity(record)
    assert_recovery_eligible(record)
    assert_matches_expected_identity_v2(record, expected_identity)

    payload = build_v2_objective_publication_payload(record)

    if is_already_published(marker_path):
        existing_marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
        if existing_marker.get("published_payload") != payload:
            raise SweepV2ObjectiveRecoveryError(
                "refusing: a previously published payload for this trial identity disagrees with the "
                f"freshly derived payload (changed objective) -- previously published="
                f"{existing_marker.get('published_payload')!r}, freshly derived={payload!r}"
            )
        return {"status": "already_published", "wandb_run_id": record["wandb_run_id"]}

    import wandb  # lazy import, repo-wide convention

    api = wandb.Api()
    run_path = f"{entity}/{project}/{record['wandb_run_id']}" if entity else f"{project}/{record['wandb_run_id']}"
    run = api.run(run_path)
    actual_sweep_id = run.sweep.id if run.sweep is not None else None
    if actual_sweep_id not in (None, record["wandb_sweep_id"]):
        raise SweepV2ObjectiveRecoveryError(
            f"refusing: run {record['wandb_run_id']!r} is associated with sweep {actual_sweep_id!r}, "
            f"not the expected {record['wandb_sweep_id']!r}"
        )
    for key, value in payload.items():
        run.summary[key] = value
    run.summary.update()

    record_publication(marker_path, wandb_run_id=record["wandb_run_id"], payload=payload)
    return {"status": "published", "wandb_run_id": record["wandb_run_id"], "payload": payload}
