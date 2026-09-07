"""SHARED-A5 Workstream D: seven-row aggregation consumer.

Implements the missing tracked consumer for the frozen seven-row SHARED-A3
comparison contract
(``docs/stage1_devpop_common120_audit_contract_v001.md`` sections 6 and 7).

Consumes **exactly seven** canonical development-population Common-120 audit
rows (as emitted by
:func:`devpop_common120_audit_evaluator.evaluate_devpop_common120_audit_row`,
typically via the Workstream-C scoring driver) plus the validated frozen
seven-entry selection manifest.  Produces the frozen comparison quantities
SHARED-A3 section 7 requires, and applies the **only** mechanically-crisp
interpretation rule the contract defines: a top-3 membership change between the
screening-400 ranking and the full-2,307 ranking is a section-7 Tier-C
escalation trigger.

What this module deliberately does **not** do, because SHARED-A3 does not
define it unambiguously:

* it does not invent a tie-breaking rule.  SHARED-A3 section 7 specifies
  neither how to rank tied scores, nor how to resolve a tie at the top-3
  boundary, nor how ties feed Spearman/Kendall.  If the seven screening
  scores or the seven full-population medians contain an exact tie, this
  module **fails closed** with a message naming that gap rather than guessing.
* it does not classify Tier A vs Tier B, and it introduces **no** absolute
  NSE-delta PASS/FAIL threshold.  Spearman/Kendall/rank-reversal/internal-
  ordering results are emitted as evidence for human scientific judgement.

The screening score and screening-selected epoch are joined in from the frozen
manifest for every row -- never reconstructed from the audit row (whose
``devpop_audit`` scope deliberately does not carry the optimizer score).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from .devpop_audit_preflight import _select_manifest_entry  # noqa: F401  (shared validator helper)
from .devpop_audit_selection_manifest import (
    DevpopAuditSelectionManifestError,
    validate_devpop_audit_selection_manifest,
)
from .devpop_common120_audit_contract import AUDIT_OBJECTIVE_SCOPE, DEVPOP_AUDIT_CONTRACT_ID
from .devpop_common120_audit_evaluator import PROVENANCE_RECEIPT_SCHEMA

__all__ = [
    "DevpopAuditSevenRowAggregationError",
    "SEVEN_ROW_COMPARISON_SCHEMA",
    "TOP_GROUP_SIZE",
    "aggregate_devpop_audit_seven_rows",
]

SEVEN_ROW_COMPARISON_SCHEMA = "flashnh_stage1_devpop_common120_audit_seven_row_comparison_v001"
TOP_GROUP_SIZE = 3
_EXPECTED_ROWS = 7


class DevpopAuditSevenRowAggregationError(ValueError):
    """Raised when the seven supplied rows do not demonstrate the shared
    scientific identity SHARED-A3 section 6 requires, or when a comparison
    quantity SHARED-A3 section 7 requires cannot be computed unambiguously
    (an exact tie, which the contract does not define handling for)."""


# --------------------------------------------------------------------------- #
# pure-numpy-free rank statistics (n == 7; no third-party dependency, matching
# the repo convention in src/baseline/percentile_diagnostics.py)
# --------------------------------------------------------------------------- #

def _ranks_desc_no_ties(values: Sequence[float]) -> list[int]:
    """Rank 1 == largest value.  Caller has already proven there are no ties."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0] * len(values)
    for position, idx in enumerate(order, start=1):
        ranks[idx] = position
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0.0 or var_b <= 0.0:
        raise DevpopAuditSevenRowAggregationError(
            "cannot compute a rank correlation: zero variance in one rank vector"
        )
    return cov / math.sqrt(var_a * var_b)


def _spearman_no_ties(screening_ranks: Sequence[int], full_ranks: Sequence[int]) -> float:
    return _pearson([float(r) for r in screening_ranks], [float(r) for r in full_ranks])


def _kendall_tau_no_ties(screening_ranks: Sequence[int], full_ranks: Sequence[int]) -> float:
    n = len(screening_ranks)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = screening_ranks[i] - screening_ranks[j]
            f = full_ranks[i] - full_ranks[j]
            if s * f > 0:
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) // 2
    return (concordant - discordant) / total


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _has_exact_tie(values: Sequence[float]) -> bool:
    return len(set(values)) != len(values)


def _require_consistent_embedded_provenance(index: int, row: Mapping, identity: tuple, entry: Mapping) -> None:
    """Close the row cross-wiring gap.

    A row emitted by :func:`evaluate_devpop_common120_audit_row` carries two
    nested identity carriers -- ``row["checkpoint_identity"]`` (the
    ``DevpopAuditCheckpointIdentity`` payload) and ``row["provenance"]`` (the
    verified provenance-receipt block, schema
    :data:`PROVENANCE_RECEIPT_SCHEMA`).  The outer per-row identity checks above
    do not look inside either, so a valid row for one frozen entry could be
    relabelled to another by editing only the outer ``trial_id`` /
    ``configuration_id`` / ``checkpoint_epoch`` / ``checkpoint_sha256`` while
    keeping the original nested provenance -- misattributing a full-population
    score to a different frozen configuration.

    This fails closed unless both nested carriers attest exactly the identity
    the row's own authoritative outer fields and the corresponding frozen
    selection-manifest ``entry`` do.  It does not re-run any of the evaluator's
    scientific validation -- it is an identity-consistency check only.
    """
    expected = (
        ("trial_id", entry["trial_id"]),
        ("configuration_id", entry["configuration_id"]),
        ("checkpoint_epoch", entry["screening_best_epoch"]),
        ("checkpoint_sha256", entry["checkpoint_sha256"]),
    )

    ident = row.get("checkpoint_identity")
    if not isinstance(ident, Mapping):
        raise DevpopAuditSevenRowAggregationError(
            f"audit row {index} ({identity!r}) has no embedded checkpoint_identity mapping"
        )
    for field, want in expected:
        if ident.get(field) != want:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {index} ({identity!r}) embedded checkpoint_identity {field}="
                f"{ident.get(field)!r} disagrees with the row's authoritative identity / the frozen "
                f"manifest entry ({want!r}) -- refusing a cross-wired row"
            )

    prov = row.get("provenance")
    if not isinstance(prov, Mapping):
        raise DevpopAuditSevenRowAggregationError(
            f"audit row {index} ({identity!r}) has no embedded provenance mapping"
        )
    if prov.get("schema") != PROVENANCE_RECEIPT_SCHEMA:
        raise DevpopAuditSevenRowAggregationError(
            f"audit row {index} ({identity!r}) embedded provenance schema {prov.get('schema')!r} is "
            f"not {PROVENANCE_RECEIPT_SCHEMA!r}"
        )
    if prov.get("provenance_verified") is not True:
        raise DevpopAuditSevenRowAggregationError(
            f"audit row {index} ({identity!r}) embedded provenance is not provenance_verified"
        )
    for field, want in (*expected, ("checkpoint_filename", entry["checkpoint_filename"])):
        if prov.get(field) != want:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {index} ({identity!r}) embedded provenance {field}={prov.get(field)!r} "
                f"disagrees with the row's authoritative identity / the frozen manifest entry "
                f"({want!r}) -- refusing a cross-wired row"
            )


def aggregate_devpop_audit_seven_rows(
    *,
    audit_rows: Sequence[Mapping],
    selection_manifest: Mapping,
    allow_fixture_rows: bool = False,
) -> dict:
    """Validate the seven rows against the frozen shared-identity requirements
    of SHARED-A3 section 6, then compute the frozen comparison quantities of
    section 7.

    ``audit_rows`` -- exactly seven audit-row mappings.
    ``selection_manifest`` -- the full validated seven-entry frozen selection
    manifest (mapping with ``entries`` + recorded ``manifest_sha256``).
    ``allow_fixture_rows`` -- when True, a row that passed the *synthetic*
    completeness gate (``fixture_completeness is True``) is accepted in place of
    one that passed the canonical gate.  This mirrors the established
    ``require_canonical=False`` seam in the A2 evaluator; it is for the
    Workstream-E vertical integration test and never weakens the canonical
    production path (which passes ``allow_fixture_rows=False``).

    Raises :class:`DevpopAuditSevenRowAggregationError` on any shared-identity
    violation or on an exact tie (undefined in SHARED-A3 section 7).
    """
    try:
        validated_manifest = validate_devpop_audit_selection_manifest(
            (selection_manifest or {}).get("entries")
            if isinstance(selection_manifest, Mapping)
            else None
        )
    except DevpopAuditSelectionManifestError as exc:
        raise DevpopAuditSevenRowAggregationError(
            f"frozen selection manifest failed validation: {exc}"
        ) from exc
    recorded = selection_manifest.get("manifest_sha256") if isinstance(selection_manifest, Mapping) else None
    if recorded is not None and recorded != validated_manifest["manifest_sha256"]:
        raise DevpopAuditSevenRowAggregationError(
            f"selection_manifest manifest_sha256 {recorded!r} does not match the recomputed identity "
            f"of its own seven entries ({validated_manifest['manifest_sha256']!r})"
        )
    entries_by_identity = {
        (e["trial_id"], e["configuration_id"]): e for e in validated_manifest["entries"]
    }

    rows = list(audit_rows)
    if len(rows) != _EXPECTED_ROWS:
        raise DevpopAuditSevenRowAggregationError(
            f"expected exactly {_EXPECTED_ROWS} audit rows, got {len(rows)}"
        )

    # -- per-row shared-identity checks (SHARED-A3 section 6) ---------------- #
    seen_identities: set[tuple] = set()
    contract_checksums: set[str] = set()
    membership_hashes: set[str] = set()
    population_roles: set[str] = set()
    population_sizes: set[int] = set()

    per_config: list[dict] = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise DevpopAuditSevenRowAggregationError(f"audit row {i} is not a mapping")
        if row.get("schema") != "flashnh_stage1_devpop_common120_audit_row_v001":
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} schema {row.get('schema')!r} is not the SHARED-A2 audit-row schema"
            )
        if row.get("objective_scope") != AUDIT_OBJECTIVE_SCOPE:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} objective_scope {row.get('objective_scope')!r} is not "
                f"{AUDIT_OBJECTIVE_SCOPE!r}"
            )
        if row.get("contract_id") != DEVPOP_AUDIT_CONTRACT_ID:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} contract_id {row.get('contract_id')!r} is not the canonical devpop "
                f"audit contract id"
            )

        identity = (row.get("trial_id"), row.get("configuration_id"))
        if identity not in entries_by_identity:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} identity {identity!r} is not one of the seven frozen manifest "
                f"(trial_id, configuration_id) identities"
            )
        if identity in seen_identities:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} identity {identity!r} is duplicated across the supplied rows"
            )
        seen_identities.add(identity)
        entry = entries_by_identity[identity]

        if row.get("provenance_verified") is not True:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} ({identity!r}) is not provenance-verified"
            )
        canonical_ok = (
            row.get("canonical_completeness") is True
            and row.get("canonical_population_verified") is True
        )
        fixture_ok = allow_fixture_rows and row.get("fixture_completeness") is True
        if not (canonical_ok or fixture_ok):
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} ({identity!r}) did not pass the "
                f"{'canonical or fixture' if allow_fixture_rows else 'canonical'} completeness gate"
            )

        if row.get("checkpoint_epoch") != entry["screening_best_epoch"]:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} ({identity!r}) checkpoint_epoch {row.get('checkpoint_epoch')!r} does "
                f"not equal the frozen screening_best_epoch {entry['screening_best_epoch']!r}"
            )
        if row.get("checkpoint_sha256") != entry["checkpoint_sha256"]:
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} ({identity!r}) checkpoint_sha256 does not equal the frozen entry's"
            )

        _require_consistent_embedded_provenance(i, row, identity, entry)

        full_median = row.get("nse_median")
        if not _finite_number(full_median):
            raise DevpopAuditSevenRowAggregationError(
                f"audit row {i} ({identity!r}) nse_median {full_median!r} is not a finite number"
            )
        screening_score = entry["screening_score"]
        if not _finite_number(screening_score):
            raise DevpopAuditSevenRowAggregationError(
                f"frozen manifest entry for {identity!r} has non-finite screening_score {screening_score!r}"
            )

        contract_checksums.add(row.get("contract_checksum_sha256"))
        membership_hashes.add(row.get("membership_ids_sha256"))
        population_roles.add(row.get("population_role"))
        population_sizes.add(row.get("expected_population_size"))

        per_config.append(
            {
                "trial_id": entry["trial_id"],
                "configuration_id": entry["configuration_id"],
                "search_arm": entry["search_arm"],
                "proposal_order": entry["proposal_order"],
                "screening_best_epoch": entry["screening_best_epoch"],
                "screening_median_nse": float(screening_score),
                "full_median_nse": float(full_median),
            }
        )

    if len(seen_identities) != _EXPECTED_ROWS:
        raise DevpopAuditSevenRowAggregationError(
            "supplied rows do not cover the seven frozen identities bijectively"
        )
    for label, bucket in (
        ("contract_checksum_sha256", contract_checksums),
        ("membership_ids_sha256", membership_hashes),
        ("population_role", population_roles),
        ("expected_population_size", population_sizes),
    ):
        if len(bucket) != 1 or None in bucket:
            raise DevpopAuditSevenRowAggregationError(
                f"the seven rows do not share an identical {label} (observed: {sorted(map(str, bucket))})"
            )

    # -- tie guard: SHARED-A3 section 7 defines no tie handling ------------- #
    screening_values = [c["screening_median_nse"] for c in per_config]
    full_values = [c["full_median_nse"] for c in per_config]
    if _has_exact_tie(screening_values):
        raise DevpopAuditSevenRowAggregationError(
            "exact tie in the seven screening median-NSE values; SHARED-A3 section 7 defines no tie "
            "handling for ranks / top-3 membership / Spearman / Kendall -- stopping at the unambiguous "
            "computations and reporting the gap rather than inventing a rule"
        )
    if _has_exact_tie(full_values):
        raise DevpopAuditSevenRowAggregationError(
            "exact tie in the seven full-2307 median-NSE values; SHARED-A3 section 7 defines no tie "
            "handling for ranks / top-3 membership / Spearman / Kendall -- stopping at the unambiguous "
            "computations and reporting the gap rather than inventing a rule"
        )

    # -- frozen comparison quantities (SHARED-A3 section 7) ----------------- #
    screening_ranks = _ranks_desc_no_ties(screening_values)
    full_ranks = _ranks_desc_no_ties(full_values)
    for c, sr, fr in zip(per_config, screening_ranks, full_ranks):
        c["screening_rank"] = sr
        c["full_rank"] = fr
        c["signed_delta_nse"] = c["full_median_nse"] - c["screening_median_nse"]
        c["abs_delta_nse"] = abs(c["signed_delta_nse"])
        c["rank_change"] = sr - fr  # positive == improved (moved toward rank 1)

    spearman = _spearman_no_ties(screening_ranks, full_ranks)
    kendall = _kendall_tau_no_ties(screening_ranks, full_ranks)

    pairwise_rank_reversals: list[dict] = []
    n = len(per_config)
    for a in range(n):
        for b in range(a + 1, n):
            ca, cb = per_config[a], per_config[b]
            screening_a_better = ca["screening_rank"] < cb["screening_rank"]
            full_a_better = ca["full_rank"] < cb["full_rank"]
            if screening_a_better != full_a_better:
                better, worse = (ca, cb) if screening_a_better else (cb, ca)
                pairwise_rank_reversals.append(
                    {
                        "screening_better": better["configuration_id"],
                        "screening_worse": worse["configuration_id"],
                        "screening_pair": [better["configuration_id"], worse["configuration_id"]],
                        "full_order": [worse["configuration_id"], better["configuration_id"]],
                    }
                )

    def _top_group(rank_key: str) -> list[str]:
        return sorted(
            (c["configuration_id"] for c in per_config if c[rank_key] <= TOP_GROUP_SIZE)
        )

    screening_top3 = _top_group("screening_rank")
    full_top3 = _top_group("full_rank")
    entered_top3 = sorted(set(full_top3) - set(screening_top3))
    exited_top3 = sorted(set(screening_top3) - set(full_top3))
    top3_membership_stable = not entered_top3 and not exited_top3

    def _ordered_top3(rank_key: str) -> list[str]:
        return [
            c["configuration_id"]
            for c in sorted(per_config, key=lambda c: c[rank_key])
            if c[rank_key] <= TOP_GROUP_SIZE
        ]

    screening_top3_ordered = _ordered_top3("screening_rank")
    full_top3_ordered = _ordered_top3("full_rank")
    internal_top3_order_changed = (
        top3_membership_stable and screening_top3_ordered != full_top3_ordered
    )

    # -- interpretation: only the mechanically-crisp SHARED-A3 rule -------- #
    if not top3_membership_stable:
        interpretation = {
            "tier": "C",
            "escalation_required": True,
            "escalation_reason": (
                "top-3 membership changed between the screening-400 ranking and the full-2,307 "
                "ranking (SHARED-A3 section 7 Tier-C escalation rule: a top-3 membership change "
                "triggers deeper full-population analysis before relying on the screening ranking; "
                "it does not itself select a winner)"
            ),
            "tier_a_vs_b_note": None,
        }
    else:
        interpretation = {
            "tier": None,
            "escalation_required": False,
            "escalation_reason": None,
            "tier_a_vs_b_note": (
                "top-3 membership is unchanged, so the sole mechanically-crisp SHARED-A3 section 7 "
                "escalation rule (Tier C) is not triggered. Distinguishing Tier A (materially stable) "
                "from Tier B (investigate / uncertain) requires human scientific judgement over the "
                "evidence in this report -- internal_top3_order_changed, pairwise_rank_reversals, "
                "spearman, kendall, signed/abs deltas -- and any conspicuous cross-population result. "
                "SHARED-A3 defines no absolute NSE-delta threshold and this consumer introduces none."
            ),
        }

    per_config.sort(key=lambda c: c["full_rank"])

    return {
        "schema": SEVEN_ROW_COMPARISON_SCHEMA,
        "selection_manifest_sha256": validated_manifest["manifest_sha256"],
        "n_configurations": _EXPECTED_ROWS,
        "allow_fixture_rows": bool(allow_fixture_rows),
        "shared_identity": {
            "contract_checksum_sha256": next(iter(contract_checksums)),
            "membership_ids_sha256": next(iter(membership_hashes)),
            "population_role": next(iter(population_roles)),
            "expected_population_size": next(iter(population_sizes)),
        },
        "per_configuration": per_config,
        "rank_correlation": {
            "spearman": spearman,
            "kendall": kendall,
            "n_pairwise_rank_reversals": len(pairwise_rank_reversals),
            "pairwise_rank_reversals": pairwise_rank_reversals,
        },
        "top_group": {
            "size": TOP_GROUP_SIZE,
            "screening_top3": screening_top3,
            "full_top3": full_top3,
            "screening_top3_ordered": screening_top3_ordered,
            "full_top3_ordered": full_top3_ordered,
            "entered_top3": entered_top3,
            "exited_top3": exited_top3,
            "top3_membership_stable": top3_membership_stable,
            "internal_top3_order_changed": internal_top3_order_changed,
        },
        "interpretation": interpretation,
    }
