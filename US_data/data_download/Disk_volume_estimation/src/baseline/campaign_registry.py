"""Central reserved-run-ID registry (Stage 1, Scope B of the Sequence-
Length-A minimum-viable-infrastructure task).

One small source of truth for run-ID collision protection across Flash-NH's
pilot range-characterization campaigns. Two kinds of entries:

- :data:`HISTORICAL_RESERVED_RUN_ID_GROUPS`: explicit, immutable, hand-
  maintained groups of run_ids already used by the committed 6-run pilot
  matrix and by every already-CLOSED campaign (LR-A, Hidden-size-A, cap50k,
  Embedding-Dropout-A). These campaigns' own launcher scripts are NOT
  modified to use this registry -- each still independently hand-maintains
  its own defense-in-depth ``_OTHER_CAMPAIGN_RESERVED_RUN_IDS`` literal, and
  that is left untouched. This module's historical groups are a second,
  more complete source of truth, transcribed read-only from those scripts
  and the committed policy YAML, for PROSPECTIVE campaigns going forward to
  check themselves against.
- :func:`register_prospective_campaign_run_ids`: how a new, not-yet-executed
  :class:`campaign_spec.CampaignSpec` (starting with Sequence-Length-A)
  reserves its own run_ids, failing loudly on any collision against every
  historical group and every other already-registered prospective group.

Deliberately does not attempt to re-derive the historical lists dynamically
from the closed campaigns' own scripts (each of those scripts is
intentionally self-contained per its own module docstring) -- these are
literal, hand-transcribed values, exactly like each closure script's own
``_OTHER_CAMPAIGN_RESERVED_RUN_IDS``.
"""

from dataclasses import dataclass

__all__ = [
    "CampaignRegistryError",
    "ReservedRunIdGroup",
    "HISTORICAL_RESERVED_RUN_ID_GROUPS",
    "reserved_run_id_index",
    "register_prospective_campaign_run_ids",
]


class CampaignRegistryError(Exception):
    """Raised on a run-ID collision (or an internal consistency violation)
    detected by this registry."""


@dataclass(frozen=True)
class ReservedRunIdGroup:
    source: str
    run_ids: "tuple[str, ...]"
    status: str  # "historical" or "prospective"


HISTORICAL_RESERVED_RUN_ID_GROUPS = (
    ReservedRunIdGroup(
        source="stage1_lead06_pilot_v001 (committed 6-run matrix; "
        "config/stage1_lead06_pilot_v001.yaml)",
        run_ids=(
            "raw_seedA",
            "raw_seedB",
            "emb128x64_seedA",
            "emb128x64_seedB",
            "emb64_seedA",
            "emb128_seedA",
        ),
        status="historical",
    ),
    ReservedRunIdGroup(
        source="LR-A (scripts/run_stage1_lr_range_seedA_closure.py)",
        run_ids=(
            "emb128x32_seedA_lr1em4_cap25k_cal",
            "emb128x32_seedA_lr3em4_cap25k_cal",
            "emb128x32_seedA_lr3em3_cap25k_cal",
            "emb128x32_seedA_lr1em2_cap25k_cal",
            "emb128x32_seedA_cap25k_cal",
        ),
        status="historical",
    ),
    ReservedRunIdGroup(
        source="Hidden-size-A (scripts/run_stage1_hidden_size_range_seedA_closure.py)",
        run_ids=(
            "emb128x32_seedA_h64_lr3em4_cap25k_cal",
            "emb128x32_seedA_h128_lr3em4_cap25k_cal",
            "emb128x32_seedA_h256_lr3em4_cap25k_cal",
            "emb128x32_seedA_h512_lr3em4_cap25k_cal",
        ),
        status="historical",
    ),
    ReservedRunIdGroup(
        source="cap50k (scripts/run_stage1_cap50k_closure.py)",
        run_ids=(
            "emb128x64_seedA_cap_low_cal",
            "emb128x32_seedA_cap_low_cal",
        ),
        status="historical",
    ),
    ReservedRunIdGroup(
        source="Embedding-Dropout-A "
        "(scripts/run_stage1_embedding_dropout_range_seedA_closure.py)",
        run_ids=(
            "emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal",
            "emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal",
            "emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal",
            "emb128x32_seedA_drop20_h128_lr3em4_cap25k_cal",
            "emb128x32_seedA_drop40_h128_lr3em4_cap25k_cal",
        ),
        status="historical",
    ),
)

# Populated at runtime by register_prospective_campaign_run_ids -- one entry
# per distinct prospective campaign "source" label. Module-level, process-
# lifetime state (not persisted to disk); each process/test run starts empty.
_PROSPECTIVE_CAMPAIGN_GROUPS: "dict[str, ReservedRunIdGroup]" = {}


def reserved_run_id_index() -> "dict[str, str]":
    """Map every currently-reserved run_id (every historical group plus
    every prospective group registered so far in this process) to its
    source group's label. Also asserts, every time it is rebuilt, that no
    run_id is listed in two different groups -- cheap, and catches a hand-
    maintenance mistake in :data:`HISTORICAL_RESERVED_RUN_ID_GROUPS` itself,
    not only a genuinely new collision."""
    index: "dict[str, str]" = {}
    for group in HISTORICAL_RESERVED_RUN_ID_GROUPS + tuple(_PROSPECTIVE_CAMPAIGN_GROUPS.values()):
        for run_id in group.run_ids:
            if run_id in index:
                raise CampaignRegistryError(
                    f"internal inconsistency: run_id {run_id!r} is listed in both "
                    f"{index[run_id]!r} and {group.source!r} -- fix the reserved-run-id groups"
                )
            index[run_id] = group.source
    return index


def register_prospective_campaign_run_ids(source: str, run_ids) -> ReservedRunIdGroup:
    """Reserve ``run_ids`` for a prospective (not-yet-executed) campaign
    identified by ``source`` (a short human-readable label, e.g.
    ``"Sequence-Length-A"``). Raises :class:`CampaignRegistryError` loudly if
    any of ``run_ids`` is already reserved by a DIFFERENT source (historical
    or prospective), or if ``run_ids`` itself contains a duplicate. Calling
    this again with the exact same ``(source, run_ids)`` is a no-op that
    returns the existing group (idempotent re-registration, e.g. across
    repeated CLI invocations in the same process); calling it again with the
    same ``source`` but DIFFERENT ``run_ids`` is itself a collision and also
    raises loudly."""
    run_ids = tuple(run_ids)
    seen = set()
    for run_id in run_ids:
        if run_id in seen:
            raise CampaignRegistryError(
                f"prospective campaign {source!r} declares duplicate run_id {run_id!r} "
                "within its own group"
            )
        seen.add(run_id)

    existing = _PROSPECTIVE_CAMPAIGN_GROUPS.get(source)
    if existing is not None:
        if existing.run_ids == run_ids:
            return existing
        raise CampaignRegistryError(
            f"prospective campaign {source!r} is already registered with run_ids "
            f"{existing.run_ids!r}, which contradicts the newly given {run_ids!r}"
        )

    index = reserved_run_id_index()
    for run_id in run_ids:
        collision_source = index.get(run_id)
        if collision_source is not None:
            raise CampaignRegistryError(
                f"run_id {run_id!r} for prospective campaign {source!r} collides with "
                f"already-reserved run_id from {collision_source!r}"
            )

    group = ReservedRunIdGroup(source=source, run_ids=run_ids, status="prospective")
    _PROSPECTIVE_CAMPAIGN_GROUPS[source] = group
    return group
