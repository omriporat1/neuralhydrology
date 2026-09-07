"""SHARED-A5 Workstream E: vertical Interface / Consumer Contract Gate test
(``docs/agent_handoff_rules.md`` section 5).

One synthetic run exercised through the *real* public interfaces of the whole
remaining-six execution-plumbing chain, as far vertically as is practical
without running NeuralHydrology or real production data:

    validated frozen selection-manifest entry
      -> prepare_devpop_audit_eval_run_dir        (Workstream A/B producer)
      -> preflight_devpop_audit_entry             (Workstream A preflight)
      -> evaluate_devpop_common120_audit_row      (reused SHARED-A2 evaluator)
      -> aggregate_devpop_audit_seven_rows        (Workstream D consumer)

It proves:

1. a validated manifest entry drives the generic preparation + preflight;
2. the resulting authoritative identity flows into the canonical scoring path
   with synthetic fixtures;
3. the resulting canonical row schema is consumed by the seven-row collector;
4. configuration-specific values (per-entry ``seq_length`` / ``hidden_size`` /
   ``screening_best_epoch`` / ``screening_score``) are carried end to end --
   never lost, reconstructed incorrectly, or replaced with P1 constants
   (P1 == hidden_size 256 / seq_length 60 / epoch 1).
"""
from __future__ import annotations

import pytest

from src.baseline.devpop_audit_eval_run_producer import prepare_devpop_audit_eval_run_dir
from src.baseline.devpop_audit_preflight import preflight_devpop_audit_entry
from src.baseline.devpop_audit_selection_manifest import (
    selection_manifest_entry_to_checkpoint_identity,
)
from src.baseline.devpop_audit_seven_row_aggregation import (
    SEVEN_ROW_COMPARISON_SCHEMA,
    aggregate_devpop_audit_seven_rows,
)
from src.baseline.devpop_common120_audit_evaluator import (
    build_devpop_audit_provenance_receipt,
    evaluate_devpop_common120_audit_row,
)
from tests.test_devpop_audit_eval_run_producer import (
    EVAL_BASINS,
    _eval_population_and_contract,
    _producer_paths,
    _write_eval_package,
    _write_run_pickle,
)
from tests.test_devpop_audit_seven_row_aggregation import (
    _CHANGED_TOP3_FULL,
    _SPECS,
    _distinct_seven_entry_manifest,
    _embedded_provenance_for,
    _entries_by_slot,
)

_CKPT_BYTES = b"synthetic-screened-checkpoint-weights"
_SELECTED_SLOT = ("bayesian", 1)


def test_vertical_producer_to_preflight_to_evaluator_to_seven_row_collector(tmp_path):
    # (1) a validated frozen seven-entry selection manifest, one entry naming
    #     the real synthetic checkpoint bytes under test
    manifest, entries = _distinct_seven_entry_manifest(
        tmp_path, selected=_SELECTED_SLOT, selected_bytes=_CKPT_BYTES
    )
    by_slot = _entries_by_slot(entries)
    entry = by_slot[_SELECTED_SLOT]
    selected_seq, selected_epoch, selected_score = _SPECS[_SELECTED_SLOT]

    ckpt_src = tmp_path / "src" / entry["checkpoint_filename"]
    ckpt_src.parent.mkdir(parents=True)
    ckpt_src.write_bytes(_CKPT_BYTES)
    scaler_src = tmp_path / "src" / "train_data_scaler.yml"
    scaler_src.write_bytes(b"synthetic-scaler-bytes")

    # (2) generic preparation from that entry (never a caller-supplied arch)
    prepare_devpop_audit_eval_run_dir(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        fixed_support_contract_path=tmp_path / "fixed_support_contract.json",
        checkpoint_src_path=ckpt_src,
        scaler_src_path=scaler_src,
        out_generated_dir=tmp_path / "generated",
        out_run_dir=tmp_path / "run",
        **_producer_paths(tmp_path),
    )

    # (3) generic per-entry preflight -- every configuration-specific fact is
    #     derived from THIS entry + the frozen v2 config, not from P1 constants
    report = preflight_devpop_audit_entry(
        selection_manifest=manifest,
        entry_trial_id=entry["trial_id"],
        run_dir=tmp_path / "run",
        strict=True,
    )
    assert report["ok"] is True
    facts = report["derived_facts"]
    assert facts["seq_length"] == selected_seq == 96
    assert facts["hidden_size"] == 128
    assert facts["checkpoint_epoch"] == selected_epoch == 3
    # explicitly NOT the P1 canary constants
    assert facts["seq_length"] != 60
    assert facts["hidden_size"] != 256
    assert facts["checkpoint_epoch"] != 1
    # campaign-frozen architecture is derived from FROZEN_FIXED_CONFIGURATION_V2
    assert facts["campaign_frozen"]["implied_lstm_input_dim"] == 34

    # (4) one real audit row via the reused, unmodified SHARED-A2 evaluator
    run_dir = tmp_path / "run"
    _write_run_pickle(run_dir, EVAL_BASINS, epoch=entry["screening_best_epoch"])
    checkpoint_path = run_dir / entry["checkpoint_filename"]
    receipt = build_devpop_audit_provenance_receipt(
        trial_id=entry["trial_id"],
        configuration_id=entry["configuration_id"],
        run_dir=run_dir,
        period="validation",
        checkpoint_epoch=entry["screening_best_epoch"],
        checkpoint_path=checkpoint_path,
    )
    eval_package_root = tmp_path / "eval_pkg"
    _write_eval_package(eval_package_root, EVAL_BASINS)
    population, contract = _eval_population_and_contract()
    real_row = evaluate_devpop_common120_audit_row(
        checkpoint_identity=selection_manifest_entry_to_checkpoint_identity(entry),
        run_dir=run_dir,
        package_root=eval_package_root,
        population=population,
        contract=contract,
        provenance_receipt=receipt,
        checkpoint_path=checkpoint_path,
        require_canonical=False,
    )
    assert real_row["trial_id"] == entry["trial_id"]
    assert real_row["fixture_completeness"] is True

    # (5) seven rows sharing the real row's population/contract envelope, but
    #     each with OUTER identity *and* embedded evaluator provenance made
    #     mutually consistent for its own frozen entry (no cloning one row and
    #     merely relabelling the outer fields -- the collector now rejects that)
    rows = []
    for slot, other in by_slot.items():
        checkpoint_identity, provenance = _embedded_provenance_for(other)
        rows.append(
            {
                **real_row,
                "trial_id": other["trial_id"],
                "configuration_id": other["configuration_id"],
                "checkpoint_epoch": other["screening_best_epoch"],
                "checkpoint_sha256": other["checkpoint_sha256"],
                "checkpoint_identity": checkpoint_identity,
                "provenance": provenance,
                "nse_median": _CHANGED_TOP3_FULL[slot],
            }
        )
    # the selected slot keeps the genuine evaluator provenance from real_row
    sel_idx = list(by_slot).index(_SELECTED_SLOT)
    rows[sel_idx]["checkpoint_identity"] = real_row["checkpoint_identity"]
    rows[sel_idx]["provenance"] = real_row["provenance"]

    out = aggregate_devpop_audit_seven_rows(
        audit_rows=rows, selection_manifest=manifest, allow_fixture_rows=True
    )

    # (6) the row schema is consumable + configuration-specific values survive
    assert out["schema"] == SEVEN_ROW_COMPARISON_SCHEMA
    assert out["n_configurations"] == 7
    per_cfg = {c["configuration_id"]: c for c in out["per_configuration"]}
    seen_epochs = set()
    for slot, (_seq, epoch, score) in _SPECS.items():
        cfg_id = by_slot[slot]["configuration_id"]
        # screening score + epoch joined FROM THE MANIFEST, per configuration
        assert per_cfg[cfg_id]["screening_median_nse"] == pytest.approx(score)
        assert per_cfg[cfg_id]["screening_best_epoch"] == epoch
        seen_epochs.add(epoch)
    # the per-entry epochs are genuinely distinct -- not collapsed to P1's 1
    assert seen_epochs == {3, 9, 5, 10, 6, 4, 7}

    # (7) the sole mechanically-crisp SHARED-A3 section 7 rule fired
    assert out["interpretation"]["tier"] == "C"
    assert out["interpretation"]["escalation_required"] is True
    assert out["top_group"]["top3_membership_stable"] is False
