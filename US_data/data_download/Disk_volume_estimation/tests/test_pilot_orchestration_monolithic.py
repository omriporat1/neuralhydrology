"""Focused tests for the MONOLITHIC prepared-execution path
(:func:`src.baseline.pilot_orchestration.execute_prepared_pilot_run_monolithic`),
added as part of the Sweep-v1 attempt001 ``blocked_continuation_overshoot_conflict``
repair.

Confirmed root cause (see docs/decision_log.md): Sweep-v1's generated NH
config bakes ``epochs: 12`` directly (single-shot fidelity), which the
bounded-CHUNK executor (:func:`execute_prepared_pilot_run`) cannot safely
drive -- a single ``start_run`` call trains straight through all 12 epochs,
so that executor's second chunk finds epoch 2's checkpoint already sitting
untrusted/flat in the base run directory and refuses to proceed
(``blocked_continuation_overshoot_conflict``). This module proves the new,
purely additive sibling executor -- built for exactly this monolithic
config shape -- trains once, never calls a continuation, and performs
complete, restart-safe, per-epoch post-hoc screening for epochs 1..12.

Real, torch-backed fixtures throughout (never a fabricated in-memory
receipt): checkpoint/optimizer-state files and NH validation-result pickles
are written to real temp-directory paths and read back by the real,
unmodified helpers this module composes
(:func:`discover_physical_checkpoints`, :func:`actual_optimizer_updates_by_epoch`,
:func:`ensure_validation_results`,
:func:`~src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.splits import sha256_of

from tests._pilot_support import (
    BASELINE_POLICY_PATH,
    PILOT_POLICY_PATH,
    REAL_DEVELOPMENT,
    SPLITS_DIR,
    build_full_union_package,
    pick_development_basins,
    short_tmp_path,
    write_perfect_validation_results,
    write_screening_basin_ids_file,
)


# --- real fixture plumbing ---------------------------------------------------

def _fake_monolithic_train_chunk_fn(experiment_name: str, calls: list, *, torch=None, updates_per_epoch: int = 100):
    """A single ``start_run``-equivalent call: writes all 12 checkpoints (and,
    if ``torch`` is given, real ``optimizer_state_epochNNN.pt`` sidecars)
    FLAT under one freshly created NH run directory -- exactly what a real
    monolithic ``start_run(config_file=...)`` call produces for a config
    whose own ``epochs`` key is already baked to 12, and exactly the shape
    that made the OLD chunk executor misfire. Records every invocation in
    ``calls`` so tests can assert it is called at most once and always with
    ``is_first_chunk=True``."""

    def _train(request: "orchestration.TrainChunkRequest") -> None:
        calls.append(request)
        assert request.is_first_chunk is True, "monolithic executor must never request a continuation chunk"
        runs_root = request.config_path.parent / "runs"
        nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
        nh_run_dir.mkdir(parents=True)
        for epoch in range(1, request.additional_epochs + 1):
            (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
            if torch is not None:
                state_dict = {"state": {0: {"step": torch.tensor(epoch * updates_per_epoch)}}, "param_groups": []}
                torch.save(state_dict, nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt")

    return _train


def _fake_evaluate_checkpoint_fn(package_root, basins, calls: list):
    def _evaluate(request: "orchestration.EvaluationRequest") -> None:
        calls.append(request.epoch)
        assert request.period == "validation"
        write_perfect_validation_results(Path(request.nh_run_dir), request.epoch, basins, package_root)

    return _evaluate


@pytest.fixture
def monolithic_policy(tmp_path):
    """The exact policy shape ``sweep_v1_execution.build_execution_context``
    layers onto the committed base pilot-policy YAML for Sweep-v1: every
    epoch is a screening epoch, performance-based early stopping is
    disabled, and the epoch budget is the frozen 12-epoch fidelity."""
    base = load_pilot_policy(PILOT_POLICY_PATH)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    return dataclasses.replace(
        base,
        screening_validation_every_n_epochs=1,
        pilot_max_epoch_budget=12,
        performance_early_stopping_enabled=False,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )


@pytest.fixture
def prepared_monolithic_fixture(short_tmp_path, monolithic_policy):
    """A real, already-written NH config (via the mature ``prepare_pilot_run``,
    never monkeypatched)."""
    basins = pick_development_basins(5)
    package_root = short_tmp_path / "package"
    config_out_dir = short_tmp_path / "config_out"
    build_full_union_package(package_root, ts_basin_ids=basins)

    _run_spec, bundle, config_dir, experiment_name = orchestration.prepare_pilot_run(
        pilot_policy=monolithic_policy,
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
        config_out_dir=config_out_dir,
    )
    return {
        "pilot_policy": monolithic_policy,
        "basins": basins,
        "package_root": package_root,
        "config_dir": config_dir,
        "experiment_name": experiment_name,
        "target_variable": bundle.target_variable,
    }


def _run_monolithic(fx, *, train_chunk_fn, evaluate_checkpoint_fn, target_epoch: int = 12,
                    supplemental_epoch_evaluator=None):
    return orchestration.execute_prepared_pilot_run_monolithic(
        execution_policy=fx["pilot_policy"], config_dir=fx["config_dir"], experiment_name=fx["experiment_name"],
        package_root=fx["package_root"], target_variable=fx["target_variable"],
        lead_hours=fx["pilot_policy"].lead_hours, screening_basin_ids=fx["basins"], target_epoch=target_epoch,
        train_chunk_fn=train_chunk_fn, evaluate_checkpoint_fn=evaluate_checkpoint_fn,
        supplemental_epoch_evaluator=supplemental_epoch_evaluator,
    )


# --- Item 1: true monolithic 12-epoch fixture --------------------------------

def test_monolithic_12_epoch_run_completes_with_full_checkpoint_and_screening_coverage(prepared_monolithic_fixture):
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []
    result = _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert isinstance(result, orchestration.PreparedPilotExecutionResult)
    assert result.final_status == "monolithic_training_and_screening_complete"
    assert result.blocked is False and result.blocked_reason is None
    assert result.stopped is False and result.stop_reason is None
    assert set(result.checkpoint_inventory) == set(range(1, 13))
    assert {int(e["epoch"]) for e in result.screening_events} == set(range(1, 13))
    assert result.effective_policy["max_epoch_budget"] == 12
    assert result.effective_policy["performance_early_stopping_enabled"] is False


# --- Item 2: proof no continuation call/directory is ever attempted ---------

def test_monolithic_never_calls_continuation_and_leaves_no_continuation_directory(prepared_monolithic_fixture):
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []
    result = _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert len(train_calls) == 1, "must invoke train_chunk_fn exactly once for a fresh monolithic run"
    assert train_calls[0].is_first_chunk is True
    assert train_calls[0].additional_epochs == 12
    nested_dirs = list(Path(result.nh_run_dir).glob("continue_training_from_epoch*"))
    assert nested_dirs == [], "a genuine monolithic start_run call must never produce a continuation directory"

    # Re-invoking after completion (e.g. a retried post-processing step) must
    # discover the existing run and never call train_chunk_fn again.
    result2 = _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert len(train_calls) == 1, "resume must never retrain a monolithic run whose checkpoints are already complete"
    assert result2.final_status == "monolithic_training_and_screening_complete"


# --- Item 3: post-hoc screening invoked exactly once per epoch 1..12 -------

def test_post_hoc_screening_invoked_exactly_once_per_epoch_and_is_restart_safe(prepared_monolithic_fixture):
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []
    _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert sorted(eval_calls) == list(range(1, 13)), "every epoch 1..12 must be explicitly evaluated exactly once"

    # Simulate a crash-and-retry of the post-processing step: calling again
    # against the same (already fully screened) run directory must not
    # re-invoke evaluate_checkpoint_fn for any already-logged epoch.
    result2 = _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert sorted(eval_calls) == list(range(1, 13)), "resume must not re-run evaluation for already-logged epochs"
    assert {int(e["epoch"]) for e in result2.screening_events} == set(range(1, 13))


def test_supplemental_callback_is_pure_at_least_once_and_recomputes_complete_trajectory_after_interruption(
    prepared_monolithic_fixture,
):
    """The callback is deliberately stateless: immutable epoch artifacts make
    a retry safe, while a partial in-memory trajectory is never returned."""
    fx = prepared_monolithic_fixture
    train_calls, eval_calls, first_values = [], [], {}

    def interrupted_callback(run_dir, epoch):
        assert (run_dir / f"model_epoch{epoch:03d}.pt").is_file()
        if epoch == 5:
            raise RuntimeError("injected callback interruption")
        first_values[epoch] = epoch / 100.0
        return {"fixed_support": {"epoch": epoch, "value": first_values[epoch]}}

    with pytest.raises(RuntimeError, match="injected callback interruption"):
        _run_monolithic(
            fx, train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
            evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
            supplemental_epoch_evaluator=interrupted_callback,
        )
    assert first_values == {epoch: epoch / 100.0 for epoch in range(1, 5)}
    assert len(train_calls) == 1 and sorted(eval_calls) == list(range(1, 13))

    retry_values = {}

    def pure_callback(run_dir, epoch):
        assert (run_dir / f"model_epoch{epoch:03d}.pt").is_file()
        retry_values[epoch] = epoch / 100.0
        return {"fixed_support": {"epoch": epoch, "value": retry_values[epoch]}}

    result = _run_monolithic(
        fx, train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
        supplemental_epoch_evaluator=pure_callback,
    )
    assert retry_values == {epoch: epoch / 100.0 for epoch in range(1, 13)}
    assert {epoch: value["fixed_support"]["value"] for epoch, value in result.supplemental_epoch_results.items()} == retry_values
    assert len(train_calls) == 1 and sorted(eval_calls) == list(range(1, 13))


# --- Item 4: complete population accounting per epoch -----------------------

def test_every_screening_event_reports_complete_population_accounting(prepared_monolithic_fixture):
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []
    result = _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    for event in result.screening_events:
        raw = event["raw_space_metrics"]
        assert raw["n_basins_requested"] == len(fx["basins"])
        assert raw["n_basins_evaluated"] == len(fx["basins"])
        assert raw["n_basins_area_excluded"] == 0
        assert raw["n_basins_evaluated"] + raw["n_basins_area_excluded"] == raw["n_basins_requested"]


# --- Item 5: direct optimizer-counter verification per epoch ----------------

def test_actual_optimizer_updates_by_epoch_reads_genuine_cumulative_counters(prepared_monolithic_fixture):
    torch = pytest.importorskip("torch")
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []
    result = _run_monolithic(
        fx,
        train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls, torch=torch, updates_per_epoch=137),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    updates = orchestration.actual_optimizer_updates_by_epoch(result.nh_run_dir)
    assert updates == {epoch: epoch * 137 for epoch in range(1, 13)}


# --- Rejection: caller/context mismatch on performance_early_stopping ------

def test_rejects_effective_policy_with_performance_early_stopping_enabled(prepared_monolithic_fixture):
    fx = dict(prepared_monolithic_fixture)
    fx["pilot_policy"] = dataclasses.replace(fx["pilot_policy"], performance_early_stopping_enabled=True)
    with pytest.raises(orchestration.PilotOrchestrationError):
        _run_monolithic(
            fx, train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], []),
            evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], []),
        )


# --- Blocked: missing required checkpoint -----------------------------------

def test_missing_required_checkpoint_blocks_without_attempting_any_continuation(prepared_monolithic_fixture):
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []

    def _train_with_gap(request: "orchestration.TrainChunkRequest") -> None:
        train_calls.append(request)
        runs_root = request.config_path.parent / "runs"
        nh_run_dir = runs_root / f"{fx['experiment_name']}_20260101_000000"
        nh_run_dir.mkdir(parents=True)
        for epoch in range(1, 13):
            if epoch == 7:
                continue
            (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    result = _run_monolithic(
        fx, train_chunk_fn=_train_with_gap,
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert result.final_status == "blocked_incomplete_monolithic_training"
    assert result.blocked is True
    assert "[7]" in result.blocked_reason
    assert eval_calls == [], "must never attempt post-hoc screening when required training checkpoints are missing"
    assert len(train_calls) == 1, "must never retry/continue training to fill the gap"


# --- Blocked: checkpoint physically nested under a continuation directory --

def test_nested_checkpoint_from_a_prior_continuation_attempt_blocks(prepared_monolithic_fixture):
    fx = prepared_monolithic_fixture
    runs_root = Path(fx["config_dir"]) / "runs"
    nh_run_dir = runs_root / f"{fx['experiment_name']}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 12):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
    cont_dir = nh_run_dir / "continue_training_from_epoch011"
    cont_dir.mkdir()
    (cont_dir / "model_epoch012.pt").write_bytes(b"ckpt12")

    train_calls = []
    result = _run_monolithic(
        fx, train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], []),
    )
    assert train_calls == [], "an existing run directory must never be retrained"
    assert result.final_status == "blocked_incomplete_monolithic_training"
    assert result.blocked is True
    assert "nested" in result.blocked_reason


# --- Item 12: direct reproduction of attempt001's conflict, then its repair -

def test_monolithic_shaped_config_conflicts_old_chunk_executor_but_new_executor_resolves_it(
    prepared_monolithic_fixture,
):
    """Direct proof of attempt001's diagnosed root cause and its repair.

    Sweep-v1's generated NH config bakes ``epochs: 12`` monolithically, so a
    real ``start_run`` call trains straight through every epoch in one shot
    regardless of how many epochs the caller expected. This reproduces that
    exact shape against the OLD bounded-chunk executor
    (``execute_prepared_pilot_run``, unmodified, still driving every
    legitimately chunked campaign) using the same ``pilot_max_epoch_budget=12,
    screening_validation_every_n_epochs=1`` policy this module's other tests
    use for the new monolithic executor -- the first chunk (target=1) writes
    all 12 checkpoints; the second chunk (target=2) then finds epoch 2's
    checkpoint already sitting untrusted/flat in the base run directory and
    must refuse to proceed, exactly attempt001's real
    ``blocked_continuation_overshoot_conflict`` failure. It then proves
    ``execute_prepared_pilot_run_monolithic``, pointed at the identical
    physical run directory the blocked attempt left behind, completes
    cleanly -- screening every remaining epoch and never calling
    ``train_chunk_fn`` again (the directory is already fully, genuinely
    trained; there is nothing left to train)."""
    fx = prepared_monolithic_fixture
    old_train_calls: list = []

    def _monolithic_shaped_train(request: "orchestration.TrainChunkRequest") -> None:
        old_train_calls.append(request)
        assert request.is_first_chunk is True, "the conflict arises on the very first real start_run call"
        runs_root = request.config_path.parent / "runs"
        nh_run_dir = runs_root / f"{fx['experiment_name']}_20260101_000000"
        nh_run_dir.mkdir(parents=True)
        for epoch in range(1, 13):
            (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    old_eval_calls: list = []
    conflict_result = orchestration.execute_prepared_pilot_run(
        execution_policy=fx["pilot_policy"], config_dir=fx["config_dir"], experiment_name=fx["experiment_name"],
        package_root=fx["package_root"], target_variable=fx["target_variable"], lead_hours=fx["pilot_policy"].lead_hours,
        screening_basin_ids=fx["basins"], run_id="raw_seedA",
        train_chunk_fn=_monolithic_shaped_train,
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], old_eval_calls),
    )
    assert len(old_train_calls) == 1, "the conflict must surface on the very next chunk, without a second train call"
    assert conflict_result.final_status == "blocked_continuation_overshoot_conflict"
    assert conflict_result.blocked is True

    new_train_calls: list = []
    repaired_result = _run_monolithic(
        fx, train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], new_train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], []),
    )
    assert new_train_calls == [], "the run directory is already fully trained; must never retrain or continue"
    assert repaired_result.final_status == "monolithic_training_and_screening_complete"
    assert repaired_result.blocked is False
    assert set(repaired_result.checkpoint_inventory) == set(range(1, 13))
    assert {int(e["epoch"]) for e in repaired_result.screening_events} == set(range(1, 13))


# --- Blocked: post-hoc screening incomplete ---------------------------------

def test_incomplete_post_hoc_screening_blocks(prepared_monolithic_fixture, monkeypatch):
    fx = prepared_monolithic_fixture
    train_calls, eval_calls = [], []

    def _reconstruct_missing_epoch_9(**kwargs):
        events = real_reconstruct(**kwargs)
        return [e for e in events if int(e["epoch"]) != 9]

    real_reconstruct = orchestration._reconstruct_screening_history
    monkeypatch.setattr(orchestration, "_reconstruct_screening_history", _reconstruct_missing_epoch_9)

    result = _run_monolithic(
        fx, train_chunk_fn=_fake_monolithic_train_chunk_fn(fx["experiment_name"], train_calls),
        evaluate_checkpoint_fn=_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"], eval_calls),
    )
    assert result.final_status == "blocked_incomplete_post_hoc_screening"
    assert result.blocked is True
    assert "[9]" in result.blocked_reason
