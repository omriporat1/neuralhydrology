"""Exact-hyperparameter retry identity for one frozen Sweep-v2 six-axis trial.

Strictly additive sibling of :mod:`sweep_v1_retry` (Section G, additive
six-axis campaign foundation). ``sweep_v1_retry``'s own functions hardcode
v1-literal identity checks in three places -- ``load_frozen_proposal_record``
checks ``sweep.CAMPAIGN_ID``/``sweep.DOMAIN_VERSION`` and recomputes identity
via the five-axis ``sweep.configuration_id``/``sweep.proposal_id``/
``sweep.trial_id``; ``assert_matches_pinned_identity`` iterates only the
five-axis ``_HYPERPARAMETER_FIELDS``; ``build_bounded_wandb_tags`` embeds the
literal ``"sweep-v1"`` tag -- so each gets a genuine v2 sibling here.
``assert_generation_not_previously_attempted`` and ``validate_wandb_tags``
operate purely on generic execution-generation integers/tag strings with no
axis- or campaign-specific behavior at all, so they are imported and reused
directly from :mod:`sweep_v1_retry`, exactly as this task's established
reuse-vs-sibling convention requires (see e.g.
:mod:`sweep_v2_six_axis_execution`'s module docstring for the same
distinction applied to the execution spine).

An "exact retry" here means, identically to v1: the same six hyperparameters
(including the normalized ``seq_length``), identical
``configuration_id``/``proposal_id``/``search_arm``/``proposal_order``, the
same bound fixed-support-contract identity, a strictly greater
``execution_generation``, a freshly derived ``trial_id_v2`` (which -- unlike
v1's ``trial_id`` -- always re-embeds ``proposal_id`` too, per
``trial_id_v2``'s own collision-safety docstring), and ``retry_of_trial_id``
set to the original record's own ``trial_id``. It never requests a new W&B
Bayesian proposal -- the six axes and the support-contract identity are
recovered from the frozen record, not resampled.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import sweep_v1_campaign as sweep
from .sweep_v1_retry import (
    MAX_WANDB_TAG_LENGTH,
    assert_generation_not_previously_attempted,
    validate_wandb_tags,
    _normalize_frozen_record,
)
from .sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    DOMAIN_VERSION_V2,
    _AXES_V2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)

__all__ = [
    "SweepV2RetryError",
    "load_frozen_proposal_record_v2",
    "assert_matches_pinned_identity_v2",
    "derive_exact_retry_identity_v2",
    "assert_generation_not_previously_attempted",
    "build_bounded_wandb_tags_v2",
    "validate_wandb_tags",
    "MAX_WANDB_TAG_LENGTH",
]


class SweepV2RetryError(ValueError):
    """Raised when a frozen v2 six-axis proposal record is internally
    inconsistent, contradicts an operator-supplied pinned expected identity,
    is bound to v1's frozen campaign/domain, or a requested retry generation
    is not a genuine forward advance."""


def load_frozen_proposal_record_v2(path: "str | Path") -> dict[str, Any]:
    """Load a durable Layer-B ``execution_provenance.json``-shaped v2 record
    and independently RE-DERIVE its
    ``configuration_id``/``proposal_id``/``trial_id`` from its own persisted
    ``hyperparameters``/``search_arm``/``proposal_order``/
    ``execution_generation``/``support_contract_version``/
    ``support_contract_sha256`` via the canonical
    :mod:`sweep_v2_six_axis_campaign` helpers -- the persisted ids are never
    trusted at face value. Raises :class:`SweepV2RetryError` on a missing
    field, a v1-bound campaign/domain, an out-of-domain hyperparameter, or
    any internal disagreement (a tampered or stale file).

    Accepts either the FLAT proposal/preparation shape or the
    EXECUTED-ATTEMPT ENVELOPE shape (via :func:`sweep_v1_retry._normalize_frozen_record`,
    reused unchanged -- that helper's envelope-vs-flat unwrapping is
    generic over any campaign's record shape, with no v1-specific
    behavior).
    """
    path = Path(path)
    if not path.is_file():
        raise SweepV2RetryError(f"frozen proposal record not found: {path}")
    try:
        raw_record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SweepV2RetryError(f"frozen proposal record is not valid JSON: {path}") from exc
    if not isinstance(raw_record, dict):
        raise SweepV2RetryError(f"frozen proposal record must be a JSON object: {path}")

    record = _normalize_frozen_record(raw_record)

    required = (
        "hyperparameters", "search_arm", "proposal_order", "execution_generation",
        "configuration_id", "proposal_id", "trial_id", "campaign_id", "domain_version", "wandb_sweep_id",
        "support_contract_version", "support_contract_sha256",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise SweepV2RetryError(f"frozen proposal record missing required fields: {missing}")

    if record["campaign_id"] != CAMPAIGN_ID_V2 or record["domain_version"] != DOMAIN_VERSION_V2:
        raise SweepV2RetryError(
            "frozen proposal record campaign/domain does not match the frozen Sweep-v2 six-axis wave "
            "(refusing to load a v1 -- or otherwise foreign -- record through the v2 retry loader)"
        )

    axes = record["hyperparameters"]
    if not isinstance(axes, dict):
        raise SweepV2RetryError("frozen proposal record hyperparameters must be an object")
    try:
        recomputed_config_id = configuration_id_v2(
            axes, support_contract_version=record["support_contract_version"],
            support_contract_sha256=record["support_contract_sha256"],
        )
        recomputed_proposal_id = proposal_id_v2(record["search_arm"], record["proposal_order"])
        recomputed_trial_id = trial_id_v2(
            recomputed_config_id, recomputed_proposal_id, execution_generation=record["execution_generation"]
        )
    except (ValueError, TypeError) as exc:
        raise SweepV2RetryError(f"frozen proposal record identity fields are not canonical: {exc}") from exc

    if recomputed_config_id != record["configuration_id"]:
        raise SweepV2RetryError(
            f"frozen proposal record configuration_id ({record['configuration_id']!r}) does not match "
            f"its own recorded hyperparameters/support-contract identity (recomputed {recomputed_config_id!r})"
        )
    if recomputed_proposal_id != record["proposal_id"]:
        raise SweepV2RetryError(
            f"frozen proposal record proposal_id ({record['proposal_id']!r}) does not match its own "
            f"recorded search_arm/proposal_order (recomputed {recomputed_proposal_id!r})"
        )
    if recomputed_trial_id != record["trial_id"]:
        raise SweepV2RetryError(
            f"frozen proposal record trial_id ({record['trial_id']!r}) does not match its own recorded "
            f"identity fields (recomputed {recomputed_trial_id!r})"
        )
    return record


def assert_matches_pinned_identity_v2(record: Mapping[str, Any], pinned: Mapping[str, Any]) -> None:
    """v2 sibling of :func:`sweep_v1_retry.assert_matches_pinned_identity`.

    Identical contract, extended to the six-axis field set (``_AXES_V2``,
    including ``seq_length``) plus the two fixed-support-contract identity
    fields that are part of v2's configuration identity but have no v1
    analogue (``support_contract_version``, ``support_contract_sha256``).
    ``model_seed`` is still checked against the same campaign-wide
    :data:`sweep_v1_campaign.MODEL_SEED_A` constant v1 uses -- v2 does not
    redefine its own model seed, it is scientifically unchanged and reused
    verbatim (see ``trial_id_v2``'s own use of ``sweep.MODEL_SEED_A``).
    """
    identity_fields = {
        "proposal_order": record.get("proposal_order"),
        "proposal_id": record.get("proposal_id"),
        "configuration_id": record.get("configuration_id"),
        "trial_id": record.get("trial_id"),
        "search_arm": record.get("search_arm"),
        "wandb_sweep_id": record.get("wandb_sweep_id"),
        "wandb_run_id": record.get("wandb_run_id"),
        "support_contract_version": record.get("support_contract_version"),
        "support_contract_sha256": record.get("support_contract_sha256"),
    }
    mismatches: dict[str, Any] = {}
    for key, expected in pinned.items():
        if key in identity_fields and identity_fields[key] != expected:
            mismatches[key] = {"expected": expected, "actual": identity_fields[key]}

    hyperparameters = record.get("hyperparameters") or {}
    for key in _AXES_V2:
        if key in pinned and hyperparameters.get(key) != pinned[key]:
            mismatches[f"hyperparameters.{key}"] = {"expected": pinned[key], "actual": hyperparameters.get(key)}

    if "model_seed" in pinned and sweep.MODEL_SEED_A != pinned["model_seed"]:
        mismatches["model_seed"] = {"expected": pinned["model_seed"], "actual": sweep.MODEL_SEED_A}

    if mismatches:
        raise SweepV2RetryError(f"frozen proposal record contradicts pinned expected identity: {mismatches}")


def build_bounded_wandb_tags_v2(*, proposal_order: int, execution_generation: int, configuration_id: str) -> list[str]:
    """v2 sibling of :func:`sweep_v1_retry.build_bounded_wandb_tags` -- same
    fixed-shape, bounded-length tag construction, distinguished from v1's
    tag set only by the leading ``"sweep-v2-six-axis"`` literal (v1 uses
    ``"sweep-v1"``) so the two campaigns' runs are never conflated by tag.
    Tags remain non-authoritative conveniences; see
    :func:`validate_wandb_tags` (reused unchanged from v1 -- the length
    contract itself has no campaign-specific behavior).
    """
    return [
        "sweep-v2-six-axis",
        "exact-retry",
        f"proposal-{int(proposal_order):03d}",
        f"execution-generation-{int(execution_generation)}",
        str(configuration_id),
    ]


def derive_exact_retry_identity_v2(record: Mapping[str, Any], *, execution_generation: int,
                                    prior_attempts: "Sequence[Mapping[str, Any]]" = ()) -> dict[str, Any]:
    """v2 sibling of :func:`sweep_v1_retry.derive_exact_retry_identity`.

    Reuses the record's own six hyperparameters/``configuration_id``/
    ``proposal_id``/``search_arm``/``proposal_order``/support-contract
    identity verbatim, derives a new ``trial_id`` via ``trial_id_v2(
    configuration_id, proposal_id, execution_generation=execution_generation)``
    -- unlike v1's ``sweep.trial_id``, ``trial_id_v2`` requires
    ``proposal_id`` as well as ``configuration_id`` -- and sets
    ``retry_of_trial_id`` to the record's own ``trial_id``. Purely a
    re-derivation over already-canonical data -- never requests, samples, or
    invents a new W&B Bayesian proposal.
    """
    if not isinstance(execution_generation, int) or isinstance(execution_generation, bool):
        raise SweepV2RetryError("execution_generation must be an integer")
    original_generation = int(record["execution_generation"])
    if execution_generation <= original_generation:
        raise SweepV2RetryError(
            f"retry execution_generation ({execution_generation}) must strictly exceed the frozen "
            f"record's execution_generation ({original_generation})"
        )
    assert_generation_not_previously_attempted(execution_generation, prior_attempts)
    configuration_id = record["configuration_id"]
    proposal_id = record["proposal_id"]
    new_trial_id = trial_id_v2(configuration_id, proposal_id, execution_generation=execution_generation)
    if new_trial_id == record["trial_id"]:
        raise SweepV2RetryError("derived retry trial_id must differ from the original record's trial_id")
    return {
        "hyperparameters": dict(record["hyperparameters"]),
        "search_arm": record["search_arm"],
        "proposal_order": record["proposal_order"],
        "configuration_id": configuration_id,
        "proposal_id": proposal_id,
        "execution_generation": execution_generation,
        "trial_id": new_trial_id,
        "retry_of_trial_id": record["trial_id"],
        "wandb_sweep_id": record["wandb_sweep_id"],
        "support_contract_version": record["support_contract_version"],
        "support_contract_sha256": record["support_contract_sha256"],
    }
