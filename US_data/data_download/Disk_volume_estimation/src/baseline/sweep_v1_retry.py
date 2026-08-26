"""Exact-hyperparameter retry identity for one frozen Sweep-v1 trial.

This module is the missing "assembly" piece identified while surveying the
retry seam: :mod:`src.baseline.sweep_v1_execution` and
:mod:`src.baseline.sweep_v1_production_adapter` already thread
``execution_generation``/``retry_of_trial_id`` through preparation and
execution, but nothing durable existed for (a) loading a previously-written
Layer-B proposal-intake record back off disk with independent identity
re-derivation, (b) rejecting it if it contradicts an operator-supplied
pinned expected identity, or (c) deriving the next attempt's identity from
it. All three are pure functions over already-written JSON and the existing
canonical :mod:`sweep_v1_campaign` helpers -- no filesystem mutation, no W&B
import, no network call, and never a parallel hashing/identity
implementation.

An "exact retry" here means: identical five hyperparameters, identical
``configuration_id``/``proposal_id``/``search_arm``/``proposal_order``, a
strictly greater ``execution_generation``, a freshly derived ``trial_id``,
and ``retry_of_trial_id`` set to the original record's own ``trial_id``. It
never requests a new W&B Bayesian proposal -- the five axes are recovered
from the frozen record, not resampled.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import sweep_v1_campaign as sweep

__all__ = [
    "SweepV1RetryError",
    "load_frozen_proposal_record",
    "assert_matches_pinned_identity",
    "derive_exact_retry_identity",
    "assert_generation_not_previously_attempted",
    "build_bounded_wandb_tags",
    "validate_wandb_tags",
    "MAX_WANDB_TAG_LENGTH",
]

_HYPERPARAMETER_FIELDS = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")

# The real, pydantic-enforced limit observed in the attempt002/job 45939764
# incident ("Tags must be between 1 and 64 characters"). Not an invented
# value -- see build_bounded_wandb_tags/validate_wandb_tags below.
MAX_WANDB_TAG_LENGTH = 64

# Fields that legitimately exist ONLY in the outer executed-attempt envelope
# (never in the nested, pre-execution ``preparation_record``) and are never
# part of the frozen proposal identity.
_ENVELOPE_TERMINAL_FIELDS = frozenset({"execution_status", "result", "preparation_record"})

# Present in both layers of an executed envelope but expected to legitimately
# DIVERGE once a trial actually runs: the nested ``preparation_record``'s own
# ``objective_score`` is fixed at proposal-intake time (always null), while
# the outer envelope's ``objective_score`` reflects the trial's real terminal
# result (null for INVALID, a finite number for VALID). Never gated on
# equality -- the normalized identity always keeps the nested (frozen,
# intake-time) value; the terminal objective must never replace the frozen
# proposal identity.
_DIVERGENT_TERMINAL_FIELDS = frozenset({"objective_score"})


class SweepV1RetryError(ValueError):
    """Raised when a frozen proposal record is internally inconsistent, or
    contradicts an operator-supplied pinned expected identity, or a
    requested retry generation is not a genuine forward advance."""


def _normalize_frozen_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize either recognized frozen-record shape into the flat
    proposal/preparation-identity shape every downstream retry function
    expects.

    Two recognized shapes:

    1. FLAT proposal/preparation record -- identity fields already at the
       top level, as originally written by
       :func:`~src.baseline.sweep_v1_execution.write_proposal_intake_provenance`
       before a trial starts running. Returned unchanged (as a copy).
    2. EXECUTED-ATTEMPT ENVELOPE -- the shape
       :func:`~src.baseline.sweep_v1_execution.execute_prepared_trial`
       (over)writes once a trial actually starts: identity/hyperparameter
       fields move under a nested ``preparation_record`` mapping, and the
       outer envelope instead carries terminal execution fields
       (``execution_status``, ``result``, a possibly-non-null
       ``objective_score``). Every field present in BOTH layers (other than
       the explicitly-divergent terminal fields above) must agree exactly;
       any disagreement is a hard failure -- the two layers are never
       silently reconciled by preferring one. The nested record is returned
       (as a copy).

    A record lacking ``preparation_record`` is treated as the flat shape; if
    it is not genuinely complete, the caller's own required-field check
    rejects it -- there is no silent third shape.
    """
    if "preparation_record" not in record:
        return dict(record)

    nested = record["preparation_record"]
    if not isinstance(nested, Mapping):
        raise SweepV1RetryError(
            "frozen proposal record is an executed-attempt envelope but its "
            "'preparation_record' is not a JSON object"
        )

    shared_keys = (set(record) & set(nested)) - _ENVELOPE_TERMINAL_FIELDS - _DIVERGENT_TERMINAL_FIELDS
    mismatches = {
        key: {"outer": record[key], "nested": nested[key]}
        for key in sorted(shared_keys) if record[key] != nested[key]
    }
    if mismatches:
        raise SweepV1RetryError(
            f"frozen proposal record outer envelope contradicts its own nested "
            f"preparation_record: {mismatches}"
        )
    return dict(nested)


def load_frozen_proposal_record(path: "str | Path") -> dict[str, Any]:
    """Load a durable Layer-B ``execution_provenance.json``-shaped record and
    independently RE-DERIVE its ``configuration_id``/``proposal_id``/
    ``trial_id`` from its own persisted ``hyperparameters``/``search_arm``/
    ``proposal_order``/``execution_generation`` via the canonical
    :mod:`sweep_v1_campaign` helpers -- the persisted ids are never trusted
    at face value. Raises :class:`SweepV1RetryError` on a missing field, an
    out-of-domain hyperparameter, or any internal disagreement (a tampered
    or stale file), never silently accepting a record whose recorded
    identity does not match its own recorded axes.

    Accepts either the FLAT proposal/preparation shape or the
    EXECUTED-ATTEMPT ENVELOPE shape (see :func:`_normalize_frozen_record`)
    -- the latter is what a real, already-run attempt's own
    ``execution_provenance.json`` looks like once
    :func:`~src.baseline.sweep_v1_execution.execute_prepared_trial` has
    written to it, so this loader must accept it directly; no separately
    extracted proposal JSON is required or supported.
    """
    path = Path(path)
    if not path.is_file():
        raise SweepV1RetryError(f"frozen proposal record not found: {path}")
    try:
        raw_record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SweepV1RetryError(f"frozen proposal record is not valid JSON: {path}") from exc
    if not isinstance(raw_record, dict):
        raise SweepV1RetryError(f"frozen proposal record must be a JSON object: {path}")

    record = _normalize_frozen_record(raw_record)

    required = (
        "hyperparameters", "search_arm", "proposal_order", "execution_generation",
        "configuration_id", "proposal_id", "trial_id", "campaign_id", "domain_version", "wandb_sweep_id",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise SweepV1RetryError(f"frozen proposal record missing required fields: {missing}")

    if record["campaign_id"] != sweep.CAMPAIGN_ID or record["domain_version"] != sweep.DOMAIN_VERSION:
        raise SweepV1RetryError(
            "frozen proposal record campaign/domain does not match the frozen Sweep-v1 original wave"
        )

    axes = record["hyperparameters"]
    if not isinstance(axes, dict):
        raise SweepV1RetryError("frozen proposal record hyperparameters must be an object")
    try:
        recomputed_config_id = sweep.configuration_id(axes)
        recomputed_proposal_id = sweep.proposal_id(record["search_arm"], record["proposal_order"])
        recomputed_trial_id = sweep.trial_id(recomputed_config_id, execution_generation=record["execution_generation"])
    except (ValueError, TypeError) as exc:
        raise SweepV1RetryError(f"frozen proposal record identity fields are not canonical: {exc}") from exc

    if recomputed_config_id != record["configuration_id"]:
        raise SweepV1RetryError(
            f"frozen proposal record configuration_id ({record['configuration_id']!r}) does not match "
            f"its own recorded hyperparameters (recomputed {recomputed_config_id!r})"
        )
    if recomputed_proposal_id != record["proposal_id"]:
        raise SweepV1RetryError(
            f"frozen proposal record proposal_id ({record['proposal_id']!r}) does not match its own "
            f"recorded search_arm/proposal_order (recomputed {recomputed_proposal_id!r})"
        )
    if recomputed_trial_id != record["trial_id"]:
        raise SweepV1RetryError(
            f"frozen proposal record trial_id ({record['trial_id']!r}) does not match its own recorded "
            f"identity fields (recomputed {recomputed_trial_id!r})"
        )
    return record


def assert_matches_pinned_identity(record: Mapping[str, Any], pinned: Mapping[str, Any]) -> None:
    """Reject any contradiction between a loaded frozen record and an
    operator-supplied pinned expected identity.

    ``pinned`` may include any of: ``proposal_order``, ``proposal_id``,
    ``configuration_id``, ``trial_id`` (the ORIGINAL record's own trial id),
    ``search_arm``, ``wandb_sweep_id``, ``wandb_run_id``, ``model_seed``, and
    the five hyperparameter axes. Every key present in ``pinned`` is checked;
    a mismatch on ANY of them is a hard failure (never a warning, never a
    silently-preferred value) -- this is the retry seam's contradiction
    rejection contract. ``wandb_run_id`` is opt-in like every other key here:
    a caller that never pins it (e.g. a retry manifest authored before a run
    id exists) is completely unaffected. ``model_seed`` is checked against
    the campaign-wide :data:`sweep_v1_campaign.MODEL_SEED_A` constant
    (Sweep-v1 has one model seed for the whole campaign, not a per-proposal
    field), guarding against a future code change silently altering it out
    from under a pinned expectation.
    """
    identity_fields = {
        "proposal_order": record.get("proposal_order"),
        "proposal_id": record.get("proposal_id"),
        "configuration_id": record.get("configuration_id"),
        "trial_id": record.get("trial_id"),
        "search_arm": record.get("search_arm"),
        "wandb_sweep_id": record.get("wandb_sweep_id"),
        "wandb_run_id": record.get("wandb_run_id"),
    }
    mismatches: dict[str, Any] = {}
    for key, expected in pinned.items():
        if key in identity_fields and identity_fields[key] != expected:
            mismatches[key] = {"expected": expected, "actual": identity_fields[key]}

    hyperparameters = record.get("hyperparameters") or {}
    for key in _HYPERPARAMETER_FIELDS:
        if key in pinned and hyperparameters.get(key) != pinned[key]:
            mismatches[f"hyperparameters.{key}"] = {"expected": pinned[key], "actual": hyperparameters.get(key)}

    if "model_seed" in pinned and sweep.MODEL_SEED_A != pinned["model_seed"]:
        mismatches["model_seed"] = {"expected": pinned["model_seed"], "actual": sweep.MODEL_SEED_A}

    if mismatches:
        raise SweepV1RetryError(f"frozen proposal record contradicts pinned expected identity: {mismatches}")


def assert_generation_not_previously_attempted(
    execution_generation: int, prior_attempts: "Sequence[Mapping[str, Any]]"
) -> None:
    """Reject reusing an ``execution_generation`` that a durable
    ``prior_attempts`` record already reports as having been attempted --
    regardless of whether that prior attempt ever produced an output
    directory.

    This exists because a failed attempt (e.g. one that crashes inside
    ``wandb.init()`` before any durable per-trial evidence is written) can
    leave NO filesystem trace to check against, so directory-existence alone
    cannot detect reuse. ``prior_attempts`` is an operator-authored,
    explicitly reviewed JSON record (the same trust model as
    ``assert_matches_pinned_identity``'s pinned-identity file) listing every
    previously reserved/attempted generation for this trial family -- e.g.
    ``{"execution_generation": 2, "slurm_job_id": "45939764", "status":
    "failed_before_wandb_association"}`` for the attempt002/job 45939764
    incident. Never invents or infers this list from the filesystem.
    """
    reserved = {
        int(attempt["execution_generation"]): attempt
        for attempt in prior_attempts if "execution_generation" in attempt
    }
    if execution_generation in reserved:
        raise SweepV1RetryError(
            f"execution_generation {execution_generation} is already reserved by a prior recorded "
            f"attempt and must never be reused: {reserved[execution_generation]}"
        )


def build_bounded_wandb_tags(*, proposal_order: int, execution_generation: int, configuration_id: str) -> list[str]:
    """Deterministic, bounded W&B tag set for an exact-retry run.

    Tags are non-authoritative conveniences (see
    ``scripts/run_sweep_v1_exact_retry_bridge.py``'s module docstring); the
    complete retry/trial/proposal/configuration identity always lives in the
    durable Flash-NH provenance record and the W&B run's own ``config`` --
    never only in a tag. Every element here is short and fixed-shape by
    construction (never a truncated fragment of a longer identifier, which
    risks collision), so the resulting tags are always comfortably under
    ``MAX_WANDB_TAG_LENGTH`` -- see :func:`validate_wandb_tags` for the
    defensive check applied immediately before any ``wandb.init()`` call.
    """
    return [
        "sweep-v1",
        "exact-retry",
        f"proposal-{int(proposal_order):03d}",
        f"execution-generation-{int(execution_generation)}",
        str(configuration_id),
    ]


def validate_wandb_tags(tags: "Sequence[str]") -> None:
    """Reject any tag that violates W&B's real (pydantic-enforced) tag
    length contract -- between 1 and :data:`MAX_WANDB_TAG_LENGTH` characters
    -- BEFORE ever calling ``wandb.init()``.

    This is the direct fix for the attempt002/job 45939764 incident: a
    125-character ``retry_of_<trial_id>`` tag was rejected only deep inside
    ``wandb.init()``'s own ``Settings`` validation, after all local
    preparation had already run and with no durable evidence yet written.
    Never silently truncates an offending tag -- truncation risks a
    collision between two distinct identities.
    """
    overlong = {tag: len(tag) for tag in tags if not (1 <= len(tag) <= MAX_WANDB_TAG_LENGTH)}
    if overlong:
        raise SweepV1RetryError(
            f"one or more W&B tags violate the 1-{MAX_WANDB_TAG_LENGTH} character contract: {overlong}"
        )


def derive_exact_retry_identity(record: Mapping[str, Any], *, execution_generation: int,
                                prior_attempts: "Sequence[Mapping[str, Any]]" = ()) -> dict[str, Any]:
    """Derive a fresh, strictly-later attempt's exact-retry identity from an
    already-validated frozen proposal record.

    Reuses the record's own hyperparameters/``configuration_id``/
    ``proposal_id``/``search_arm``/``proposal_order`` verbatim, derives a new
    ``trial_id`` via ``sweep.trial_id(configuration_id,
    execution_generation=execution_generation)``, and sets
    ``retry_of_trial_id`` to the record's own ``trial_id``. Purely a
    re-derivation over already-canonical data -- never requests, samples, or
    invents a new W&B Bayesian proposal.

    ``prior_attempts`` (default empty) is forwarded to
    :func:`assert_generation_not_previously_attempted` -- see that function's
    docstring. ``retry_of_trial_id`` always continues to reference the
    original frozen record's own ``trial_id`` (e.g. attempt001) regardless of
    how many intervening failed attempts ``prior_attempts`` lists; their
    operational link is carried separately, never by overloading
    ``retry_of_trial_id``.
    """
    if not isinstance(execution_generation, int) or isinstance(execution_generation, bool):
        raise SweepV1RetryError("execution_generation must be an integer")
    original_generation = int(record["execution_generation"])
    if execution_generation <= original_generation:
        raise SweepV1RetryError(
            f"retry execution_generation ({execution_generation}) must strictly exceed the frozen "
            f"record's execution_generation ({original_generation})"
        )
    assert_generation_not_previously_attempted(execution_generation, prior_attempts)
    configuration_id = record["configuration_id"]
    new_trial_id = sweep.trial_id(configuration_id, execution_generation=execution_generation)
    if new_trial_id == record["trial_id"]:
        raise SweepV1RetryError("derived retry trial_id must differ from the original record's trial_id")
    return {
        "hyperparameters": dict(record["hyperparameters"]),
        "search_arm": record["search_arm"],
        "proposal_order": record["proposal_order"],
        "configuration_id": configuration_id,
        "proposal_id": record["proposal_id"],
        "execution_generation": execution_generation,
        "trial_id": new_trial_id,
        "retry_of_trial_id": record["trial_id"],
        "wandb_sweep_id": record["wandb_sweep_id"],
    }
