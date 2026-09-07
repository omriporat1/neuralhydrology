"""SHARED-A5 Workstream D: the frozen seven-row SHARED-A3 comparison consumer
(:func:`src.baseline.devpop_audit_seven_row_aggregation.aggregate_devpop_audit_seven_rows`).

Proves the consumer

* accepts exactly seven canonical audit rows that share the required
  scientific identity (SHARED-A3 section 6) and rejects every violation of it
  (fewer / extra / duplicate / wrong-configuration / different-population /
  different-contract / not-canonical / not-provenance-verified);
* joins the screening score and screening-selected epoch **from the frozen
  manifest**, never from the audit row;
* computes the SHARED-A3 section 7 comparison quantities (signed / absolute
  NSE delta, screening & full ranks, Spearman, Kendall, explicit pairwise
  rank reversals, screening / full top-3, top-3 membership stability);
* applies the *only* mechanically-crisp SHARED-A3 section 7 rule -- a top-3
  membership change is a Tier-C escalation -- and introduces no tie-break
  rule or absolute NSE-delta threshold (it fails closed on an exact tie).

Synthetic manifests + synthetic audit-row mappings only; nothing here runs
NeuralHydrology, the real evaluator, or touches a remote artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.baseline.devpop_audit_selection_manifest import (
    build_devpop_audit_selection_manifest_entry,
    validate_devpop_audit_selection_manifest,
)
from src.baseline.devpop_audit_seven_row_aggregation import (
    SEVEN_ROW_COMPARISON_SCHEMA,
    DevpopAuditSevenRowAggregationError,
    aggregate_devpop_audit_seven_rows,
)
from src.baseline.devpop_common120_audit_contract import (
    AUDIT_OBJECTIVE_SCOPE,
    DEVPOP_AUDIT_CONTRACT_ID,
)
from src.baseline.devpop_common120_audit_evaluator import PROVENANCE_RECEIPT_SCHEMA
from tests.test_devpop_audit_eval_run_producer import _fixed_support_contract_path

_ROW_SCHEMA = "flashnh_stage1_devpop_common120_audit_row_v001"

# (search_arm, proposal_order) -> (seq_length, screening_best_epoch, screening_score)
# seq_length is distinct per slot so no two configuration_ids collide;
# screening_score is distinct per slot so the screening ranking is unambiguous.
_SPECS = {
    ("bayesian", 1): (96, 3, 0.42),
    ("bayesian", 2): (84, 9, 0.38),
    ("bayesian", 3): (72, 5, 0.45),
    ("random_control", 1): (48, 10, 0.31),
    ("random_control", 2): (60, 6, 0.40),
    ("random_control", 3): (108, 4, 0.36),
    ("random_control", 4): (120, 7, 0.34),
}


def _distinct_seven_entry_manifest(tmp_path, *, selected=("bayesian", 1), selected_bytes=None,
                                   score_override=None):
    """A validated FROZEN seven-entry selection manifest whose seven
    ``screening_score`` values are all distinct.  ``selected`` / ``selected_bytes``
    make one entry name a real synthetic checkpoint byte string (for the
    Workstream-E vertical test); ``score_override`` maps a slot to a replacement
    screening_score (used to force an exact screening tie)."""
    contract_path = _fixed_support_contract_path(tmp_path)
    cd = json.loads(contract_path.read_text(encoding="utf-8"))
    support_version, support_sha256 = cd["contract_id"], cd["checksum_sha256"]

    entries = []
    for (arm, order), (seq_length, epoch, score) in _SPECS.items():
        if score_override and (arm, order) in score_override:
            score = score_override[(arm, order)]
        if selected_bytes is not None and (arm, order) == selected:
            ckpt_sha256 = hashlib.sha256(selected_bytes).hexdigest()
        else:
            ckpt_sha256 = hashlib.sha256(f"ckpt-{arm}-{order}".encode()).hexdigest()
        entries.append(
            build_devpop_audit_selection_manifest_entry(
                search_arm=arm,
                proposal_order=order,
                hyperparameters={
                    "learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
                    "output_dropout": 0.25, "batch_size": 256, "seq_length": seq_length,
                },
                screening_score=score,
                screening_best_epoch=epoch,
                screening_evidence_path=f"reports/screening/{arm}_{order}.json",
                source_run_dir=f"/scratch/runs/{arm}_{order}",
                checkpoint_filename=f"model_epoch{epoch:03d}.pt",
                checkpoint_sha256=ckpt_sha256,
                selection_policy="frozen_screening_best_epoch_v001",
                support_contract_version=support_version,
                support_contract_sha256=support_sha256,
            )
        )
    return validate_devpop_audit_selection_manifest(entries), entries


def _entries_by_slot(entries):
    return {(e["search_arm"], e["proposal_order"]): e for e in entries}


def _embedded_provenance_for(entry):
    """The nested identity carriers a real ``evaluate_devpop_common120_audit_row``
    row carries, constructed *consistently* for ``entry`` (see Codex FAIL
    repair: the seven-row collector now cross-checks these against the outer
    identity and the frozen manifest entry)."""
    epoch = entry["screening_best_epoch"]
    checkpoint_identity = {
        "trial_id": entry["trial_id"],
        "configuration_id": entry["configuration_id"],
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": entry["checkpoint_sha256"],
    }
    provenance = {
        "schema": PROVENANCE_RECEIPT_SCHEMA,
        "trial_id": entry["trial_id"],
        "configuration_id": entry["configuration_id"],
        "period": "validation",
        "checkpoint_epoch": epoch,
        "checkpoint_filename": entry["checkpoint_filename"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "period_results_relpath": f"validation/model_epoch{epoch:03d}/validation_results.p",
        "period_results_sha256": "c" * 64,
        "provenance_verified": True,
    }
    return checkpoint_identity, provenance


def _fixture_row(entry, *, nse_median, contract_checksum="f" * 64, membership="a" * 64,
                 population_role="development_train", expected_population_size=2307,
                 canonical=True, provenance_verified=True, fixture_completeness=False):
    checkpoint_identity, provenance = _embedded_provenance_for(entry)
    return {
        "schema": _ROW_SCHEMA,
        "objective_scope": AUDIT_OBJECTIVE_SCOPE,
        "contract_id": DEVPOP_AUDIT_CONTRACT_ID,
        "contract_checksum_sha256": contract_checksum,
        "population_role": population_role,
        "expected_population_size": expected_population_size,
        "membership_ids_sha256": membership,
        "trial_id": entry["trial_id"],
        "configuration_id": entry["configuration_id"],
        "checkpoint_epoch": entry["screening_best_epoch"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "checkpoint_identity": checkpoint_identity,
        "provenance": provenance,
        "provenance_verified": provenance_verified,
        "canonical_completeness": canonical,
        "canonical_population_verified": canonical,
        "fixture_completeness": fixture_completeness,
        "nse_median": nse_median,
    }


# nse_median per slot: keeps screening top-3 {b3, b1, rc2} intact but reverses
# the internal (b1, rc2) order -> one pairwise reversal, top-3 membership stable.
_STABLE_TOP3_FULL = {
    ("bayesian", 3): 0.50,
    ("random_control", 2): 0.48,
    ("bayesian", 1): 0.46,
    ("bayesian", 2): 0.30,
    ("random_control", 3): 0.28,
    ("random_control", 4): 0.26,
    ("random_control", 1): 0.24,
}

# nse_median per slot: pushes rc2 out of the top-3 and b2 in -> Tier C.
_CHANGED_TOP3_FULL = {
    ("bayesian", 3): 0.50,
    ("bayesian", 1): 0.48,
    ("bayesian", 2): 0.46,
    ("random_control", 2): 0.20,
    ("random_control", 3): 0.18,
    ("random_control", 4): 0.16,
    ("random_control", 1): 0.14,
}


def _rows_for(entries, full_by_slot, **row_kwargs):
    by_slot = _entries_by_slot(entries)
    return [
        _fixture_row(by_slot[slot], nse_median=full_by_slot[slot], **row_kwargs)
        for slot in _SPECS
    ]


# --------------------------------------------------------------------------- #
# happy path + section-7 quantities
# --------------------------------------------------------------------------- #

def test_valid_seven_rows_stable_top3(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)

    out = aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)

    assert out["schema"] == SEVEN_ROW_COMPARISON_SCHEMA
    assert out["n_configurations"] == 7
    assert out["selection_manifest_sha256"] == manifest["manifest_sha256"]
    assert out["allow_fixture_rows"] is False
    assert out["shared_identity"] == {
        "contract_checksum_sha256": "f" * 64,
        "membership_ids_sha256": "a" * 64,
        "population_role": "development_train",
        "expected_population_size": 2307,
    }

    # screening score + epoch are joined FROM THE MANIFEST, per row
    by_slot = _entries_by_slot(entries)
    per_cfg = {c["configuration_id"]: c for c in out["per_configuration"]}
    for slot, (_seq, epoch, score) in _SPECS.items():
        cfg_id = by_slot[slot]["configuration_id"]
        assert per_cfg[cfg_id]["screening_median_nse"] == pytest.approx(score)
        assert per_cfg[cfg_id]["screening_best_epoch"] == epoch
        assert per_cfg[cfg_id]["full_median_nse"] == pytest.approx(_STABLE_TOP3_FULL[slot])

    # signed / absolute deltas
    b1 = per_cfg[by_slot[("bayesian", 1)]["configuration_id"]]
    assert b1["signed_delta_nse"] == pytest.approx(0.46 - 0.42)
    assert b1["abs_delta_nse"] == pytest.approx(0.04)

    # ranks (rank 1 == largest)
    assert per_cfg[by_slot[("bayesian", 3)]["configuration_id"]]["screening_rank"] == 1
    assert per_cfg[by_slot[("random_control", 1)]["configuration_id"]]["screening_rank"] == 7
    assert per_cfg[by_slot[("bayesian", 3)]["configuration_id"]]["full_rank"] == 1
    assert per_cfg[by_slot[("random_control", 2)]["configuration_id"]]["full_rank"] == 2

    top = out["top_group"]
    assert set(top["screening_top3"]) == {
        by_slot[("bayesian", 3)]["configuration_id"],
        by_slot[("bayesian", 1)]["configuration_id"],
        by_slot[("random_control", 2)]["configuration_id"],
    }
    assert set(top["full_top3"]) == set(top["screening_top3"])
    assert top["top3_membership_stable"] is True
    assert top["internal_top3_order_changed"] is True

    # exactly one pairwise reversal: b1 beats rc2 in screening, loses in full
    reversals = out["rank_correlation"]["pairwise_rank_reversals"]
    assert out["rank_correlation"]["n_pairwise_rank_reversals"] == 1
    rev = reversals[0]
    assert rev["screening_better"] == by_slot[("bayesian", 1)]["configuration_id"]
    assert rev["screening_worse"] == by_slot[("random_control", 2)]["configuration_id"]

    # rank correlations agree with an independent (scipy) oracle
    scipy_stats = pytest.importorskip("scipy.stats")
    screening = [_SPECS[s][2] for s in _SPECS]
    full = [_STABLE_TOP3_FULL[s] for s in _SPECS]
    assert out["rank_correlation"]["spearman"] == pytest.approx(
        scipy_stats.spearmanr(screening, full).statistic
    )
    assert out["rank_correlation"]["kendall"] == pytest.approx(
        scipy_stats.kendalltau(screening, full).statistic
    )

    # top-3 membership unchanged -> no Tier-C escalation, Tier A/B left to humans
    interp = out["interpretation"]
    assert interp["tier"] is None
    assert interp["escalation_required"] is False
    assert "human scientific judgement" in interp["tier_a_vs_b_note"]


def test_changed_top3_triggers_tier_c(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _CHANGED_TOP3_FULL)

    out = aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)

    by_slot = _entries_by_slot(entries)
    top = out["top_group"]
    assert top["top3_membership_stable"] is False
    assert top["entered_top3"] == [by_slot[("bayesian", 2)]["configuration_id"]]
    assert top["exited_top3"] == [by_slot[("random_control", 2)]["configuration_id"]]

    interp = out["interpretation"]
    assert interp["tier"] == "C"
    assert interp["escalation_required"] is True
    assert "top-3 membership changed" in interp["escalation_reason"]


def test_fixture_rows_only_accepted_when_explicitly_allowed(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL, canonical=False, fixture_completeness=True)

    with pytest.raises(DevpopAuditSevenRowAggregationError, match="canonical"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)

    out = aggregate_devpop_audit_seven_rows(
        audit_rows=rows, selection_manifest=manifest, allow_fixture_rows=True
    )
    assert out["allow_fixture_rows"] is True
    assert out["n_configurations"] == 7


# --------------------------------------------------------------------------- #
# fail-closed: shared-identity violations (SHARED-A3 section 6)
# --------------------------------------------------------------------------- #

def test_rejects_fewer_than_seven_rows(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)[:6]
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="exactly 7"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_more_than_seven_rows(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows.append(dict(rows[0]))
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="exactly 7"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_duplicate_identity(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[6] = dict(rows[0])  # duplicate the bayesian-1 identity, drop rc4
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="duplicat"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_row_identity_not_in_manifest(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[0] = dict(rows[0], configuration_id="sweep_v2__not_a_frozen_configuration")
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="frozen manifest"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_row_with_stale_embedded_provenance_after_relabel(tmp_path):
    """Codex FAIL (High) regression: a valid row for one frozen entry (P1-like)
    is relabelled at its OUTER identity / checkpoint fields to look like another
    frozen entry, while its embedded evaluator provenance still attests the
    original identity.  The collector must reject the cross-wired row."""
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    by_slot = _entries_by_slot(entries)
    src_entry = by_slot[("bayesian", 1)]       # provenance stays for this one
    dst_entry = by_slot[("bayesian", 2)]       # outer identity is forged to this

    forged = dict(rows[0])                      # rows[0] is the valid bayesian-1 row
    forged["trial_id"] = dst_entry["trial_id"]
    forged["configuration_id"] = dst_entry["configuration_id"]
    forged["checkpoint_epoch"] = dst_entry["screening_best_epoch"]
    forged["checkpoint_sha256"] = dst_entry["checkpoint_sha256"]
    # forged["provenance"] / forged["checkpoint_identity"] still attest bayesian-1
    assert forged["provenance"]["trial_id"] == src_entry["trial_id"]

    rows[1] = forged                            # replace the real bayesian-2 row

    with pytest.raises(
        DevpopAuditSevenRowAggregationError, match="cross-wired|embedded provenance|checkpoint_identity"
    ):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_row_with_missing_embedded_provenance(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[2].pop("provenance")
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="embedded provenance mapping"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_different_population_identity(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[3]["membership_ids_sha256"] = "b" * 64
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="membership_ids_sha256"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_different_population_role(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[2]["population_role"] = "spatial_holdout"
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="population_role"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_different_contract_checksum(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[1]["contract_checksum_sha256"] = "e" * 64
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="contract_checksum_sha256"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_non_canonical_row(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[4]["canonical_completeness"] = False
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="completeness gate"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_not_provenance_verified_row(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[5]["provenance_verified"] = False
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="provenance-verified"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_checkpoint_epoch_not_matching_frozen_manifest(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    rows[0]["checkpoint_epoch"] = rows[0]["checkpoint_epoch"] + 1
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="screening_best_epoch"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_rejects_tampered_manifest_sha256(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    tampered = dict(manifest, manifest_sha256="0" * 64)
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="manifest_sha256"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=tampered)


# --------------------------------------------------------------------------- #
# fail-closed: exact tie (SHARED-A3 section 7 defines no tie handling)
# --------------------------------------------------------------------------- #

def test_fails_closed_on_exact_tie_in_full_medians(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(tmp_path)
    full = dict(_STABLE_TOP3_FULL)
    full[("random_control", 4)] = full[("random_control", 3)]  # exact tie
    rows = _rows_for(entries, full)
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="tie"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)


def test_fails_closed_on_exact_tie_in_screening_scores(tmp_path):
    manifest, entries = _distinct_seven_entry_manifest(
        tmp_path, score_override={("random_control", 3): 0.40}  # ties random_control 2
    )
    rows = _rows_for(entries, _STABLE_TOP3_FULL)
    with pytest.raises(DevpopAuditSevenRowAggregationError, match="tie"):
        aggregate_devpop_audit_seven_rows(audit_rows=rows, selection_manifest=manifest)
