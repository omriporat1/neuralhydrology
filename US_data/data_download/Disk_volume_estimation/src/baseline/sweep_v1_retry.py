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
from typing import Any, Mapping

from . import sweep_v1_campaign as sweep

__all__ = [
    "SweepV1RetryError",
    "load_frozen_proposal_record",
    "assert_matches_pinned_identity",
    "derive_exact_retry_identity",
]

_HYPERPARAMETER_FIELDS = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")


class SweepV1RetryError(ValueError):
    """Raised when a frozen proposal record is internally inconsistent, or
    contradicts an operator-supplied pinned expected identity, or a
    requested retry generation is not a genuine forward advance."""


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
    """
    path = Path(path)
    if not path.is_file():
        raise SweepV1RetryError(f"frozen proposal record not found: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SweepV1RetryError(f"frozen proposal record is not valid JSON: {path}") from exc
    if not isinstance(record, dict):
        raise SweepV1RetryError(f"frozen proposal record must be a JSON object: {path}")

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
    ``search_arm``, ``wandb_sweep_id``, ``model_seed``, and the five
    hyperparameter axes. Every key present in ``pinned`` is checked; a
    mismatch on ANY of them is a hard failure (never a warning, never a
    silently-preferred value) -- this is the retry seam's contradiction
    rejection contract. ``model_seed`` is checked against the campaign-wide
    :data:`sweep_v1_campaign.MODEL_SEED_A` constant (Sweep-v1 has one model
    seed for the whole campaign, not a per-proposal field), guarding against
    a future code change silently altering it out from under a pinned
    expectation.
    """
    identity_fields = {
        "proposal_order": record.get("proposal_order"),
        "proposal_id": record.get("proposal_id"),
        "configuration_id": record.get("configuration_id"),
        "trial_id": record.get("trial_id"),
        "search_arm": record.get("search_arm"),
        "wandb_sweep_id": record.get("wandb_sweep_id"),
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


def derive_exact_retry_identity(record: Mapping[str, Any], *, execution_generation: int) -> dict[str, Any]:
    """Derive a fresh, strictly-later attempt's exact-retry identity from an
    already-validated frozen proposal record.

    Reuses the record's own hyperparameters/``configuration_id``/
    ``proposal_id``/``search_arm``/``proposal_order`` verbatim, derives a new
    ``trial_id`` via ``sweep.trial_id(configuration_id,
    execution_generation=execution_generation)``, and sets
    ``retry_of_trial_id`` to the record's own ``trial_id``. Purely a
    re-derivation over already-canonical data -- never requests, samples, or
    invents a new W&B Bayesian proposal.
    """
    if not isinstance(execution_generation, int) or isinstance(execution_generation, bool):
        raise SweepV1RetryError("execution_generation must be an integer")
    original_generation = int(record["execution_generation"])
    if execution_generation <= original_generation:
        raise SweepV1RetryError(
            f"retry execution_generation ({execution_generation}) must strictly exceed the frozen "
            f"record's execution_generation ({original_generation})"
        )
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
