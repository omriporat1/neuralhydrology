"""Vertical integration tests for the production Sweep-v1 execution/
interpretation layer (src/baseline/sweep_v1_execution.py).

Exercises the REAL interface shape end to end: a real
``PreparedSweepV1Proposal`` -> ``write_prepared_proposal`` -> a real
``build_execution_context``/``SweepV1ExecutionContext`` -> an injected fake
``execute_prepared_run_fn`` returning a genuine
``pilot_orchestration.PreparedPilotExecutionResult`` -> ``execute_prepared_trial``
(the actual Sweep-v1 validity consumer, which internally calls
``sweep.derive_trajectory_diagnostics``) -> Layer A (``review_records.json``)
and Layer B (``execution_provenance.json``).

Never starts real NH training or W&B: the injected receipt is fabricated
directly, but ``nh_run_dir`` contains real ``model_epochNNN.pt`` +
``optimizer_state_epochNNN.pt`` files so the real, torch-backed
``actual_optimizer_updates_by_epoch`` -- the one authoritative fact
``_derive_validity`` does not take from the injected result object itself --
reads genuine cumulative-update evidence from disk, exactly like
tests/test_prepared_execution_core.py's Test E fixture.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline import sweep_v1_campaign as sweep
from src.baseline.pilot_screening_eval import PRIMARY_METRIC_NAME, SCREENING_METRIC_SCOPE
from src.baseline.sweep_v1_execution import (
    SweepV1ExecutionContext, SweepV1ExecutionError, build_execution_context, enrich_operations_slurm_accounting,
    execute_prepared_trial, run_prepared_trial_in_production,
)
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, prepare_bayesian_proposal, prepare_random_control_row, write_prepared_proposal,
)
from tests._pilot_support import (
    BASELINE_POLICY_PATH, PILOT_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR,
    build_full_union_package, write_screening_basin_ids_file,
)


# --- fixture plumbing (duplicated from tests/test_sweep_v1_production_adapter.py's
# private helpers of the same shape -- repo convention for private test
# helpers, see test_prepared_execution_core.py's module docstring) ----------

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


def _proposal(**changes):
    value = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
             "output_dropout": 0.25, "batch_size": 256, "proposal_order": 7,
             "wandb_sweep_id": "prod-sweep", "wandb_run_id": "run-7"}
    value.update(changes)
    return value


def _prepared_record(tmp_path, monkeypatch, **proposal_changes):
    paths = _paths(tmp_path / "prep", monkeypatch)
    prepared = prepare_bayesian_proposal(proposal=_proposal(**proposal_changes), paths=paths)
    record = write_prepared_proposal(prepared, tmp_path / "prepared_out")
    return record, paths


def _write_real_checkpoints(nh_run_dir: Path, epochs, torch, *, updates_per_epoch: int = 100) -> None:
    """Writes real model_epochNNN.pt + optimizer_state_epochNNN.pt files --
    the exact shape tests/test_prepared_execution_core.py's Test E fixture
    writes -- so the real actual_optimizer_updates_by_epoch reads genuine
    cumulative-update evidence, never a mocked shortcut."""
    nh_run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        state_dict = {"state": {0: {"step": torch.tensor(epoch * updates_per_epoch)}}, "param_groups": []}
        torch.save(state_dict, nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt")


def _screening_event(epoch: int, score, *, n_basins: int) -> dict:
    """Exactly evaluate_screening_checkpoint's return shape (see
    src/baseline/pilot_screening_eval.py), fabricated directly rather than
    computed -- this is the injected half of the consumer-contract test."""
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
                 blocked: bool = False, blocked_reason=None, stopped: bool = False, stop_reason=None,
                 max_epoch_budget: int = 12) -> "orchestration.PreparedPilotExecutionResult":
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir
        )
        for epoch in checkpoint_epochs
    }
    screening_events = [_screening_event(epoch, score, n_basins=n_basins) for epoch, score in sorted(screening_scores.items())]
    return orchestration.PreparedPilotExecutionResult(
        final_status="blocked_or_stopped" if (blocked or stopped) else "completed_at_full_budget",
        blocked_reason=blocked_reason,
        effective_policy={"max_epoch_budget": max_epoch_budget, "performance_early_stopping_enabled": False},
        nh_run_dir=nh_run_dir, blocked=blocked, stopped=stopped, stop_reason=stop_reason,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
    )


def _baseline(tmp_path, monkeypatch) -> dict:
    """A fully valid Sweep-v1 12-epoch scenario: real prepared record, real
    execution context, real on-disk checkpoint/optimizer-state evidence, and
    a screening trajectory whose max is epoch 9 (0.40) and whose final
    (epoch 12) score is 0.35 -- the exact best_score/final_epoch_score
    example from the governing spec."""
    torch = pytest.importorskip("torch")
    record, paths = _prepared_record(tmp_path, monkeypatch)
    context = build_execution_context(prepared_record=record, paths=paths, base_pilot_policy_path=PILOT_POLICY_PATH)
    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.25, 6: 0.28, 7: 0.30, 8: 0.32,
              9: 0.40, 10: 0.38, 11: 0.36, 12: 0.35}
    return {
        "record": record, "paths": paths, "context": context, "nh_run_dir": nh_run_dir,
        "epochs": epochs, "scores": scores, "n_basins": len(context.screening_basin_ids), "torch": torch,
    }


# --- vertical consumer-contract test -----------------------------------------

def test_vertical_prepared_execution_consumer_contract(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    context = fx["context"]

    assert isinstance(context, SweepV1ExecutionContext)
    assert context.experiment_name == fx["record"]["trial_id"]
    assert context.config_dir == Path(fx["record"]["expected_output_dir"])
    assert len(context.screening_basin_ids) == 400
    assert context.execution_policy.pilot_max_epoch_budget == 12
    assert context.execution_policy.performance_early_stopping_enabled is False
    assert context.execution_policy.screening_validation_every_n_epochs == 1

    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )

    assert outcome["valid"] is True
    trial = outcome["review_records"]["trial_summary"]
    assert trial["workflow_status"] == "pass"
    assert trial["objective_score"] == pytest.approx(0.40)
    assert trial["best_epoch"] == 9
    assert trial["best_score"] == pytest.approx(0.40)
    assert trial["final_epoch_score"] == pytest.approx(0.35)
    assert trial["gpu_hours"] is None
    assert trial["failure_category"] is None
    # Late diagnostics, independently computed from fx["scores"] per
    # sweep.derive_trajectory_diagnostics: best_score_10 is the max over
    # epochs 1..10 -- still 0.40 (epoch 9 falls within that range); best_score_12
    # is the overall max (also 0.40, epoch 9); late_gain_10_to_12 =
    # best_score_12 - best_score_10 = 0; late_best is False since the true
    # best (epoch 9) falls before epoch 11.
    assert trial["best_score_10"] == pytest.approx(0.40)
    assert trial["best_score_12"] == pytest.approx(0.40)
    assert trial["best_minus_final"] == pytest.approx(0.05)
    assert trial["late_gain_10_to_12"] == pytest.approx(0.0)
    assert trial["late_best"] is False

    trajectory = outcome["review_records"]["epoch_trajectory"]
    assert len(trajectory) == 12
    assert {row["epoch"] for row in trajectory} == set(range(1, 13))
    for kind, record in (("trial_summary", trial), ("proposal", outcome["review_records"]["proposal"]),
                        ("operations", outcome["review_records"]["operations"])):
        sweep.validate_review_record(kind, record)
    for row in trajectory:
        sweep.validate_review_record("epoch_trajectory", row)

    provenance = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_status"] == "VALID"
    assert provenance["objective_score"] == pytest.approx(0.40)
    review = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert review["trial_summary"]["objective_score"] == pytest.approx(0.40)


def test_run_prepared_trial_in_production_wires_context_into_injected_executor(tmp_path, monkeypatch):
    """Proves the full production wiring, not just the two halves in
    isolation: run_prepared_trial_in_production must build the execution
    context and pass every one of its fields straight into
    pilot_orchestration.execute_prepared_pilot_run_monolithic -- no
    separately hardcoded/reconstructed value."""
    fx = _baseline(tmp_path, monkeypatch)
    context = fx["context"]
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])

    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return fake_result

    monkeypatch.setattr(orchestration, "execute_prepared_pilot_run_monolithic", fake_execute)

    outcome = run_prepared_trial_in_production(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        paths=fx["paths"], base_pilot_policy_path=PILOT_POLICY_PATH,
    )

    assert outcome["valid"] is True
    assert captured["experiment_name"] == context.experiment_name
    assert captured["target_epoch"] == int(fx["record"]["target_epoch"])
    assert captured["target_variable"] == context.target_variable
    assert captured["lead_hours"] == context.lead_hours
    assert list(captured["screening_basin_ids"]) == context.screening_basin_ids
    assert captured["config_dir"] == context.config_dir
    assert captured["package_root"] == context.package_root
    assert captured["execution_policy"] == context.execution_policy


def test_random_control_reuses_the_same_execution_context_and_trial_path(tmp_path, monkeypatch):
    """search_arm="random_control" reuse: neither build_execution_context nor
    execute_prepared_trial branches on search_arm -- only the prepare-time
    front door (prepare_random_control_row vs prepare_bayesian_proposal)
    differs."""
    torch = pytest.importorskip("torch")
    manifest_path = Path(__file__).parents[1] / "config/stage1_phase_b_sweep_v1_original_domain_v001_random_control_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = payload["rows"][0]

    paths = _paths(tmp_path / "prep", monkeypatch)
    prepared = prepare_random_control_row(row=row, manifest_path=manifest_path, paths=paths)
    record = write_prepared_proposal(prepared, tmp_path / "prepared_out")
    assert record["search_arm"] == "random_control"

    context = build_execution_context(prepared_record=record, paths=paths, base_pilot_policy_path=PILOT_POLICY_PATH)
    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {epoch: 0.30 + 0.01 * epoch for epoch in epochs}
    fake_result = _fake_result(nh_run_dir, checkpoint_epochs=epochs, screening_scores=scores,
                               n_basins=len(context.screening_basin_ids))

    outcome = execute_prepared_trial(
        prepared_record=record, output_dir=tmp_path / "trial_out",
        expected_screening_population=len(context.screening_basin_ids), execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is True
    assert outcome["review_records"]["trial_summary"]["search_arm"] == "random_control"


# --- negative tests: mutate authoritative receipt facts ----------------------

def test_missing_required_checkpoint_epoch_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    incomplete_epochs = [e for e in fx["epochs"] if e != 7]
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=incomplete_epochs,
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None
    assert outcome["review_records"]["trial_summary"]["workflow_status"] == "failed"


def test_missing_update_evidence_for_one_epoch_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    # All 12 checkpoints physically exist, but epoch 7's optimizer-state
    # sidecar file was never written -- actual_optimizer_updates_by_epoch
    # must fail to establish evidence for that epoch.
    (fx["nh_run_dir"] / "optimizer_state_epoch007.pt").unlink()
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_update_evidence_exceeding_the_frozen_cap_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    # Epoch 5's cumulative counter jumps by 60,000 in one epoch -- above the
    # frozen max_updates_per_epoch=50,000 cap.
    _write_real_checkpoints(fx["nh_run_dir"], fx["epochs"], fx["torch"], updates_per_epoch=100)
    state_dict = {"state": {0: {"step": fx["torch"].tensor(4 * 100 + 60_000)}}, "param_groups": []}
    fx["torch"].save(state_dict, fx["nh_run_dir"] / "optimizer_state_epoch005.pt")
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_missing_required_screening_event_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    scores_missing_epoch10 = {epoch: score for epoch, score in fx["scores"].items() if epoch != 10}
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=scores_missing_epoch10, n_basins=fx["n_basins"])
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_incomplete_screening_population_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=fx["nh_run_dir"] / f"model_epoch{epoch:03d}.pt", owning_run_dir=fx["nh_run_dir"]
        )
        for epoch in fx["epochs"]
    }
    screening_events = [_screening_event(epoch, score, n_basins=fx["n_basins"]) for epoch, score in sorted(fx["scores"].items())]
    # Epoch 6's requested/evaluated/excluded accounting silently drops 5
    # basins -- neither evaluated nor traced as an area-derivation exclusion.
    for event in screening_events:
        if event["epoch"] == 6:
            event["raw_space_metrics"]["n_basins_evaluated"] = fx["n_basins"] - 5
    fake_result = orchestration.PreparedPilotExecutionResult(
        final_status="completed_at_full_budget", blocked_reason=None,
        effective_policy={"max_epoch_budget": 12, "performance_early_stopping_enabled": False},
        nh_run_dir=fx["nh_run_dir"], blocked=False, stopped=False, stop_reason=None,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
    )
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_non_finite_screening_score_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    scores_with_nan = dict(fx["scores"]); scores_with_nan[8] = float("nan")
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=scores_with_nan, n_basins=fx["n_basins"])
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_non_budget_termination_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    # Stopped before the frozen 12-epoch budget (e.g. a stray performance-
    # based early-stopping trigger this campaign must never apply).
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"],
                               stopped=True, stop_reason="unexpected_early_stop")
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_blocked_execution_is_invalid(tmp_path, monkeypatch):
    fx = _baseline(tmp_path, monkeypatch)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"],
                               blocked=True, blocked_reason="continuation_overshoot_conflict")
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None


def test_wrong_result_type_from_injected_executor_is_invalid(tmp_path, monkeypatch):
    """execute_prepared_run_fn must return PreparedPilotExecutionResult, never
    the old ad hoc execution-fact dict vocabulary."""
    fx = _baseline(tmp_path, monkeypatch)
    old_style_dict = {"epochs_reached": 12, "checkpoint_epochs": fx["epochs"], "screening_scores": fx["scores"]}
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: old_style_dict,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None
    assert outcome["review_records"]["trial_summary"]["failure_category"] == "technical_execution_failure"


# --- Slurm job/state/GPU-hour provenance propagation -------------------------

def test_slurm_job_id_is_recorded_at_execution_time_regardless_of_validity(tmp_path, monkeypatch):
    """review_records.json's operations.slurm_job_id must reflect the live
    allocation identity as soon as it is known -- the attempt001 evidence-path
    defect being repaired here left it null even though the caller always has
    it available under sbatch/wandb agent. Checked for both a VALID and an
    INVALID trial: the allocation identity is a fact about how the trial was
    run, never gated on whether it succeeded."""
    fx = _baseline(tmp_path, monkeypatch)
    valid_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                                screening_scores=fx["scores"], n_basins=fx["n_basins"])
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out_valid",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: valid_result,
        slurm_job_id="45999001",
    )
    assert outcome["valid"] is True
    assert outcome["review_records"]["operations"]["slurm_job_id"] == "45999001"
    assert outcome["review_records"]["operations"]["slurm_state"] is None
    assert outcome["review_records"]["operations"]["gpu_hours"] is None

    invalid_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=[e for e in fx["epochs"] if e != 7],
                                  screening_scores=fx["scores"], n_basins=fx["n_basins"])
    outcome_invalid = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out_invalid",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: invalid_result,
        slurm_job_id="45999002",
    )
    assert outcome_invalid["valid"] is False
    assert outcome_invalid["review_records"]["operations"]["slurm_job_id"] == "45999002"


def test_enrich_operations_slurm_accounting_patches_state_and_gpu_hours_after_termination(tmp_path, monkeypatch):
    """Once sacct/seff data is available on the login node,
    enrich_operations_slurm_accounting must patch slurm_state/gpu_hours into
    both operations and trial_summary, in place, without disturbing
    VALID/INVALID or the objective."""
    fx = _baseline(tmp_path, monkeypatch)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    outcome = execute_prepared_trial(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
        slurm_job_id="45999003",
    )
    assert outcome["review_records"]["operations"]["gpu_hours"] is None

    patched = enrich_operations_slurm_accounting(
        output_dir=output_dir, slurm_job_id="45999003", slurm_state="COMPLETED", gpu_hours=1.75,
    )
    assert patched["operations"]["slurm_job_id"] == "45999003"
    assert patched["operations"]["slurm_state"] == "COMPLETED"
    assert patched["operations"]["gpu_hours"] == pytest.approx(1.75)
    assert patched["trial_summary"]["gpu_hours"] == pytest.approx(1.75)
    assert patched["trial_summary"]["objective_score"] == pytest.approx(0.40)

    on_disk = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert on_disk["operations"]["slurm_state"] == "COMPLETED"
    assert on_disk["operations"]["gpu_hours"] == pytest.approx(1.75)
    assert on_disk["trial_summary"]["gpu_hours"] == pytest.approx(1.75)


def test_enrich_operations_slurm_accounting_refuses_a_mismatched_job_id(tmp_path, monkeypatch):
    """Must never attach Slurm accounting facts to the wrong trial by
    guessing -- an exact slurm_job_id match against the already-recorded
    value is required."""
    fx = _baseline(tmp_path, monkeypatch)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    execute_prepared_trial(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
        slurm_job_id="45999003",
    )
    with pytest.raises(SweepV1ExecutionError):
        enrich_operations_slurm_accounting(
            output_dir=output_dir, slurm_job_id="wrong-job-id", slurm_state="COMPLETED", gpu_hours=1.0,
        )
    on_disk = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert on_disk["operations"]["slurm_state"] is None
    assert on_disk["operations"]["gpu_hours"] is None
