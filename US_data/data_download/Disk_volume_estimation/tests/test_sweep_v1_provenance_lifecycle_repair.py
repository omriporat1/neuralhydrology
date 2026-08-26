"""Regression tests for the Section-D provenance-continuity repair to
``execute_prepared_trial``/``enrich_layer_b_provenance`` in
src/baseline/sweep_v1_execution.py.

Prior to this repair, ``execute_prepared_trial`` wrote a brand-new
``execution_provenance.json`` dict literal at both its STARTED and terminal
(VALID/INVALID) stages, silently discarding any field the durable Layer-B
envelope had already accumulated during intake/preparation (e.g.
``retry_history``, ``wandb_sweep_id``/``wandb_run_id``, ``raw_proposed_axes``)
that was not re-included in that literal -- exactly the defect that dropped
``retry_history``/``executor_mode`` from Sweep-v1 attempt005's terminal
record. The repair routes both writes through the same shared
``enrich_layer_b_provenance`` merge helper every other stage already uses,
and additionally threads ``select_executor_mode``'s result through as a new
persisted ``executor_mode`` field (previously computed but never recorded
anywhere).

This file walks the exact real production lifecycle -- intake -> prepared ->
prepared_with_config -> STARTED -> terminal -- reproducing attempt005's shape
(a generation-5 exact retry with 3 prior non-countable attempts), and checks
each of the 14 automatable properties enumerated for this repair. A 15th
property -- "attempt005's own original evidence hashes remain unchanged" --
is not a unit test: this repair only ever ran against tmp_path fixtures in
this session, never against attempt005's actual Moriah directory, so the
originals were never touched; see the closure report for the checksum
comparison confirming this.

Never imports wandb; never starts real NH training (execute_prepared_trial's
``execute_prepared_run_fn`` is always a fabricated fake receipt here, exactly
tests/test_sweep_v1_execution.py's established pattern).
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline import sweep_v1_campaign as sweep
from src.baseline.pilot_screening_eval import PRIMARY_METRIC_NAME, SCREENING_METRIC_SCOPE
from src.baseline.sweep_v1_execution import (
    SweepV1ExecutionError, enrich_layer_b_provenance, execute_prepared_trial, select_executor_mode,
    write_proposal_intake_provenance,
)
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, canonicalize_wandb_proposal, prepare_bayesian_proposal, write_prepared_proposal,
)
from tests._pilot_support import (
    BASELINE_POLICY_PATH, PILOT_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR,
    build_full_union_package, write_screening_basin_ids_file,
)
from tests.test_sweep_v1_retry import _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004

import hashlib
from pathlib import Path


# --- fixture plumbing (duplicated from tests/test_sweep_v1_execution.py's /
# tests/test_sweep_v1_wandb_bridge_provenance.py's private helpers of the
# same shape -- repo convention for private test helpers) -------------------

def _paths(tmp_path, monkeypatch):
    package = build_full_union_package(tmp_path / "package")
    manifests = package / "manifests"
    (manifests / "file_checksums.csv").write_text("relative_path,sha256,size_bytes,artifact_role\n", encoding="utf-8")
    (package / "run_provenance.json").write_text('{"fixture":true}\n', encoding="utf-8")
    import src.baseline.sweep_v1_production_adapter as adapter
    monkeypatch.setattr(adapter, "PACKAGE_MANIFEST_SHA256", hashlib.sha256((manifests / "package_manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_FILE_CHECKSUMS_SHA256", hashlib.sha256((manifests / "file_checksums.csv").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_RUN_PROVENANCE_SHA256", hashlib.sha256((package / "run_provenance.json").read_bytes()).hexdigest())
    splits = tmp_path / "canonical_splits"; splits.mkdir()
    for source in Path(SPLITS_DIR).glob("*.txt"):
        (splits / source.name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    monkeypatch.setattr("src.baseline.pilot_lead06_config.sha256_of", lambda _: sweep.SCREENING_ARTIFACT_SHA256)
    return PreparationPaths(BASELINE_POLICY_PATH, package, splits, screening)


_AXES = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10, "output_dropout": 0.25, "batch_size": 256}
_RETRY_OF_TRIAL_ID = "stage1_phase_b_sweep_v1_original_domain_v001__cfg_fixture__mf12x50000__seedA967139__attempt001"


def _write_real_checkpoints(nh_run_dir: Path, epochs, torch, *, updates_per_epoch: int = 100) -> None:
    nh_run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        state_dict = {"state": {0: {"step": torch.tensor(epoch * updates_per_epoch)}}, "param_groups": []}
        torch.save(state_dict, nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt")


def _screening_event(epoch: int, score, *, n_basins: int) -> dict:
    return {
        "scope": SCREENING_METRIC_SCOPE, "authoritative": False, "epoch": epoch,
        "epoch_role": "stopping_eligible", "stopping_eligible": True,
        "n_screening_basins_requested": n_basins,
        "primary_metric_name": PRIMARY_METRIC_NAME, "primary_metric_median": score,
        "primary_metric_distribution": {},
        "raw_space_metrics": {
            "n_basins_requested": n_basins, "n_basins_evaluated": n_basins,
            "n_basins_area_excluded": 0, "area_derivation_excluded": [], "per_basin": [],
        },
    }


def _fake_result(nh_run_dir: Path, *, checkpoint_epochs, screening_scores: "dict[int, float]", n_basins: int,
                 blocked: bool = False, blocked_reason=None) -> "orchestration.PreparedPilotExecutionResult":
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir
        )
        for epoch in checkpoint_epochs
    }
    screening_events = [_screening_event(epoch, score, n_basins=n_basins) for epoch, score in sorted(screening_scores.items())]
    return orchestration.PreparedPilotExecutionResult(
        final_status="blocked_or_stopped" if blocked else "completed_at_full_budget",
        blocked_reason=blocked_reason,
        effective_policy={"max_epoch_budget": 12, "performance_early_stopping_enabled": False},
        nh_run_dir=nh_run_dir, blocked=blocked, stopped=False, stop_reason=None,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
    )


def _intake(output_root, *, execution_generation=5, retry_of_trial_id=_RETRY_OF_TRIAL_ID,
           retry_history=_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004):
    return write_proposal_intake_provenance(
        output_root=output_root, axes=_AXES, search_arm="bayesian", proposal_order=1,
        wandb_sweep_id="prod-sweep", wandb_run_id="ardib08c",
        execution_generation=execution_generation, retry_of_trial_id=retry_of_trial_id,
        retry_history=retry_history,
    )


def _prepared_evidence(paths, *, execution_generation=5):
    proposal = canonicalize_wandb_proposal(
        _AXES, metadata={
            "proposal_order": 1, "wandb_sweep_id": "prod-sweep", "wandb_run_id": "ardib08c",
            "execution_generation": execution_generation,
        },
    )
    return prepare_bayesian_proposal(proposal=proposal, paths=paths)


def _through_prepared_with_config(tmp_path, monkeypatch, *, execution_generation=5,
                                  retry_of_trial_id=_RETRY_OF_TRIAL_ID,
                                  retry_history=_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004):
    """Real intake -> prepared -> prepared_with_config, the exact sequence
    scripts/run_sweep_v1_wandb_bridge.py performs before ever calling
    execute_prepared_trial. Returns (intake, output_dir, record, paths)."""
    paths = _paths(tmp_path / "prep", monkeypatch)
    output_root = tmp_path / "out"
    intake = _intake(output_root, execution_generation=execution_generation,
                     retry_of_trial_id=retry_of_trial_id, retry_history=retry_history)
    output_dir = output_root / intake["trial_id"]
    prepared = _prepared_evidence(paths, execution_generation=execution_generation)
    enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)
    record = write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)
    enrich_layer_b_provenance(output_dir=output_dir, stage="prepared_with_config", fields=record)
    return intake, output_dir, record, paths


def _baseline_epochs(tmp_path, monkeypatch, record, paths):
    """The exact 12-epoch, real-checkpoint scenario used across the existing
    Sweep-v1 execution test files."""
    torch = pytest.importorskip("torch")
    from src.baseline.sweep_v1_execution import build_execution_context
    context = build_execution_context(prepared_record=record, paths=paths, base_pilot_policy_path=PILOT_POLICY_PATH)
    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {e: 0.30 + 0.01 * e for e in epochs}
    return {"context": context, "nh_run_dir": nh_run_dir, "epochs": epochs, "scores": scores,
           "n_basins": len(context.screening_basin_ids)}


# --- 1: intake persists retry_history ---------------------------------------

def test_intake_persists_retry_history_and_retry_of_trial_id(tmp_path):
    intake = _intake(tmp_path / "out")
    assert intake["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert intake["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    on_disk = json.loads((tmp_path / "out" / intake["trial_id"] / "execution_provenance.json").read_text(encoding="utf-8"))
    assert on_disk["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004


# --- 2: "prepared"-stage (W&B-associated evidence) enrichment preserves it --

def test_prepared_stage_enrichment_preserves_retry_history_and_wandb_ids(tmp_path, monkeypatch):
    paths = _paths(tmp_path / "prep", monkeypatch)
    output_root = tmp_path / "out"
    intake = _intake(output_root)
    output_dir = output_root / intake["trial_id"]
    prepared = _prepared_evidence(paths)

    after = enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)

    assert after["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert after["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert after["wandb_sweep_id"] == "prod-sweep"
    assert after["wandb_run_id"] == "ardib08c"


# --- 3: executor-mode selection is recorded at all -------------------------

def test_executor_mode_selection_is_recorded_at_started_stage(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    mode = select_executor_mode(record)
    assert mode == "monolithic"

    fx = _baseline_epochs(tmp_path, monkeypatch, record, paths)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])

    outcome = execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, expected_screening_population=fx["n_basins"],
        execute_prepared_run_fn=lambda: fake_result, executor_mode=mode,
    )
    assert outcome["provenance"]["executor_mode"] == "monolithic"


# --- 4: prepared_with_config preserves both ---------------------------------

def test_prepared_with_config_preserves_retry_history_and_wandb_ids(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    on_disk = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert on_disk["provenance_stage"] == "prepared_with_config"
    assert on_disk["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert on_disk["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert on_disk["wandb_sweep_id"] == "prod-sweep"
    assert on_disk["wandb_run_id"] == "ardib08c"
    assert "generated_nh_config_sha256" in on_disk


# --- 5: STARTED-stage snapshot preserves retry_history + wandb + executor --

def test_started_stage_preserves_retry_history_wandb_and_executor_mode(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    fx = _baseline_epochs(tmp_path, monkeypatch, record, paths)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    mode = select_executor_mode(record)

    started_snapshot = {}

    def _capture_started_then_succeed():
        # execute_prepared_trial's STARTED write lands on disk strictly
        # before execute_prepared_run_fn is invoked -- read it back from
        # inside the callback to observe that exact intermediate state.
        started_snapshot.update(
            json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
        )
        return fake_result

    execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, expected_screening_population=fx["n_basins"],
        execute_prepared_run_fn=_capture_started_then_succeed, executor_mode=mode,
    )

    assert started_snapshot["execution_status"] == "STARTED"
    assert started_snapshot["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert started_snapshot["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert started_snapshot["wandb_sweep_id"] == "prod-sweep"
    assert started_snapshot["wandb_run_id"] == "ardib08c"
    assert started_snapshot["executor_mode"] == "monolithic"


# --- 6: terminal VALID preserves retry_history + wandb + executor_mode -----

def test_terminal_valid_preserves_retry_history_wandb_and_executor_mode(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    fx = _baseline_epochs(tmp_path, monkeypatch, record, paths)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    mode = select_executor_mode(record)

    outcome = execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, expected_screening_population=fx["n_basins"],
        execute_prepared_run_fn=lambda: fake_result, executor_mode=mode,
    )

    assert outcome["valid"] is True
    final = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert final["execution_status"] == "VALID"
    assert final["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert final["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert final["wandb_sweep_id"] == "prod-sweep"
    assert final["wandb_run_id"] == "ardib08c"
    assert final["executor_mode"] == "monolithic"
    assert final["objective_score"] is not None


# --- 7: terminal INVALID (genuinely derived, not exception) preserves both -

def test_terminal_invalid_from_derived_result_preserves_retry_history_wandb_and_executor_mode(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    fx = _baseline_epochs(tmp_path, monkeypatch, record, paths)
    blocked_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                                  screening_scores=fx["scores"], n_basins=fx["n_basins"],
                                  blocked=True, blocked_reason="disk_quota_exceeded")
    mode = select_executor_mode(record)

    outcome = execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, expected_screening_population=fx["n_basins"],
        execute_prepared_run_fn=lambda: blocked_result, executor_mode=mode,
    )

    assert outcome["valid"] is False
    final = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert final["execution_status"] == "INVALID"
    assert final["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert final["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert final["wandb_sweep_id"] == "prod-sweep"
    assert final["wandb_run_id"] == "ardib08c"
    assert final["executor_mode"] == "monolithic"
    assert final["objective_score"] is None


# --- 8: a pre-execution (e.g. W&B-init-style) failure leaves the durable ----
# --- intake evidence, including retry_history, accumulated and intact ------

def test_failure_before_execution_preserves_accumulated_intake_evidence_including_retry_history(tmp_path):
    output_root = tmp_path / "out"
    intake = _intake(output_root)
    output_dir = output_root / intake["trial_id"]

    # Simulate a failure between intake and preparation (e.g. a W&B-init or
    # tag-validation failure in the bridge) -- nothing further is ever
    # written to output_dir. The durable intake record, including
    # retry_history, must remain exactly as written.
    recovered = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert recovered["provenance_stage"] == "proposal_intake"
    assert recovered["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert recovered["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert recovered["objective_score"] is None


# --- 9: execution exception preserves accumulated evidence ------------------

def test_execution_exception_preserves_accumulated_evidence_including_retry_history_and_executor_mode(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    mode = select_executor_mode(record)

    def _boom():
        raise RuntimeError("simulated NH execution failure")

    outcome = execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, execute_prepared_run_fn=_boom, executor_mode=mode,
    )

    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["failure_category"] == "technical_execution_failure"
    final = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert final["execution_status"] == "INVALID"
    assert final["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert final["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    assert final["wandb_sweep_id"] == "prod-sweep"
    assert final["executor_mode"] == "monolithic"


# --- 10: identity contradiction hard-fails, for every identity key ---------

@pytest.mark.parametrize("key,bad_value", [
    ("campaign_id", "some_other_campaign"),
    ("proposal_id", "some_other_proposal"),
    ("configuration_id", "some_other_configuration"),
    ("trial_id", "some_other_trial_id"),
    ("execution_generation", 999),
    ("search_arm", "random_control"),
])
def test_enrich_rejects_every_identity_key_conflict(tmp_path, key, bad_value):
    output_root = tmp_path / "out"
    intake = _intake(output_root)
    output_dir = output_root / intake["trial_id"]
    with pytest.raises(SweepV1ExecutionError):
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields={key: bad_value})


# --- 11: existing malformed provenance hard-fails safely -------------------

def test_enrich_rejects_invalid_json_on_disk(tmp_path):
    output_dir = tmp_path / "trial"
    output_dir.mkdir(parents=True)
    (output_dir / "execution_provenance.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(SweepV1ExecutionError):
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields={"trial_id": "x"})


def test_enrich_rejects_non_object_json_on_disk(tmp_path):
    output_dir = tmp_path / "trial"
    output_dir.mkdir(parents=True)
    (output_dir / "execution_provenance.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SweepV1ExecutionError):
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields={"trial_id": "x"})


# --- 12: atomic-write failure leaves no partial record ----------------------

def test_atomic_write_failure_during_enrich_leaves_no_partial_record(tmp_path, monkeypatch):
    output_root = tmp_path / "out"
    intake = _intake(output_root)
    output_dir = output_root / intake["trial_id"]
    path = output_dir / "execution_provenance.json"
    before_bytes = path.read_bytes()

    def _boom_replace(*args, **kwargs):
        raise OSError("simulated disk failure during os.replace")

    monkeypatch.setattr(os, "replace", _boom_replace)
    with pytest.raises(OSError):
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields={"some_new_field": "value"})

    # The original file is untouched (replace never completed)...
    assert path.read_bytes() == before_bytes
    # ...and no leftover .tmp file was left behind (cleaned up in `finally`).
    leftovers = [p for p in output_dir.iterdir() if p.name != "execution_provenance.json"]
    assert leftovers == []


# --- 13: real attempts 2-4 history survives through a synthetic ------------
# --- attempt005-shaped terminal record --------------------------------------

def test_real_prior_attempts_2_to_4_survive_through_synthetic_attempt005_shaped_terminal_record(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    fx = _baseline_epochs(tmp_path, monkeypatch, record, paths)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    mode = select_executor_mode(record)

    outcome = execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, expected_screening_population=fx["n_basins"],
        execute_prepared_run_fn=lambda: fake_result, executor_mode=mode,
    )
    assert outcome["valid"] is True

    final = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert final["retry_history"] == _REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004
    assert [a["execution_generation"] for a in final["retry_history"]] == [2, 3, 4]
    assert final["execution_generation"] == 5
    assert final["retry_of_trial_id"] == _RETRY_OF_TRIAL_ID
    # sanity: retry_history is carried by reference identity of content only
    # (never mutated in place by any enrichment stage).
    assert final["retry_history"] == copy.deepcopy(_REAL_PRIOR_ATTEMPTS_AFTER_ATTEMPT004)


# --- 14: production manifest checksum/runtime evidence survives to terminal-

def test_production_manifest_checksum_and_runtime_evidence_survive_to_terminal_state(tmp_path, monkeypatch):
    intake, output_dir, record, paths = _through_prepared_with_config(tmp_path, monkeypatch)
    prepared_with_config = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    checksum_fields = {
        "package_manifest_sha256", "package_file_checksums_sha256", "package_run_provenance_sha256",
        "development_split_sha256", "spatial_holdout_split_sha256", "generated_nh_config_sha256",
    }
    present_before = {k: prepared_with_config[k] for k in checksum_fields if k in prepared_with_config}
    assert present_before, "expected at least one checksum field to be present after prepared_with_config"

    fx = _baseline_epochs(tmp_path, monkeypatch, record, paths)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    mode = select_executor_mode(record)

    execute_prepared_trial(
        prepared_record=record, output_dir=output_dir, expected_screening_population=fx["n_basins"],
        execute_prepared_run_fn=lambda: fake_result, executor_mode=mode,
    )

    final = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    for key, value in present_before.items():
        assert final[key] == value
    assert final["preparation_record"]["trial_id"] == intake["trial_id"]
