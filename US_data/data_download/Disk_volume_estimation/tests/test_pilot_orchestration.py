"""Focused tests for src/baseline/pilot_orchestration.py (task item 6's
implementation, task item 10's tests; extended for the post-qualification-run
explicit-evaluation correction -- see docs/stage1_lead06_pilot_v001.md).

Adapted from a previously-passing scratchpad end-to-end verification
script. Uses a FAKE ``train_chunk_fn`` (writes checkpoint files ONLY -- no
validation_results.p; training and evaluation are separate NH operations,
see the real Moriah qualification run's failure) and a separate FAKE
``evaluate_checkpoint_fn`` (writes hand-crafted perfect-NSE
validation_results.p, no NH/torch/GPU) so the full
chunk-scheduling / explicit-evaluation-prerequisite / screening /
restart-safe early-stopping / evidence bundle pipeline can be exercised
deterministically and fast. Covers: pure chunk-schedule + screening-epoch
enumeration helpers, budget-below-minimum rejection, missing-runs-root
rejection, ``ensure_validation_results``'s missing/existing/failure
behavior in isolation, a full run_pilot() call against the real committed
pilot policy through to patience-exhausted stopping with the best
checkpoint retained, idempotent resume (no retraining, no re-evaluation,
evidence bundle still rewritten on every call regardless of force), and a
resume test shaped exactly like the real qualification run's failure
(checkpoints + saved validation results already present through epoch 6,
nothing in this module has processed them yet).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.baseline.pilot_orchestration import (
    EvaluationRequest,
    PilotOrchestrationError,
    TrainChunkRequest,
    chunk_epoch_targets,
    discover_nh_run_dir,
    ensure_validation_results,
    prepare_pilot_run,
    root_logger_has_file_handler,
    run_pilot,
    run_pilot_chunk,
    screening_epochs_in_chunk,
)
from src.baseline.pilot_early_stopping import build_effective_policy
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.nh_seed_evaluation import weight_stem
from src.baseline.splits import sha256_of

from tests._pilot_support import (
    BASELINE_POLICY_PATH,
    PILOT_POLICY_PATH,
    REAL_DEVELOPMENT,
    REAL_FULL_UNION,
    SPLITS_DIR,
    build_full_union_package,
    pick_development_basins,
    write_perfect_validation_results,
    write_screening_basin_ids_file,
)


@pytest.fixture
def pilot_policy(tmp_path):
    base = load_pilot_policy(PILOT_POLICY_PATH)
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    return dataclasses.replace(
        base,
        screening_basin_ids_path=str(screening_path),
        screening_expected_count=350,
        screening_expected_sha256=sha256_of(screening_path),
    )


# --- pure chunk-schedule + screening-epoch enumeration -----------------------

def test_chunk_epoch_targets_matches_frozen_policy(pilot_policy):
    targets = chunk_epoch_targets(pilot_policy, 36)
    assert targets == [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]


def test_screening_epochs_in_chunk_first_chunk_includes_diagnostic_and_first_stopping_eligible(pilot_policy):
    assert screening_epochs_in_chunk(0, 6, pilot_policy) == [3, 6]


def test_screening_epochs_in_chunk_later_chunks(pilot_policy):
    assert screening_epochs_in_chunk(6, 9, pilot_policy) == [9]
    assert screening_epochs_in_chunk(33, 36, pilot_policy) == [36]


def test_chunk_epoch_targets_rejects_budget_below_stopping_eligible_from_epoch(pilot_policy):
    with pytest.raises(PilotOrchestrationError):
        chunk_epoch_targets(pilot_policy, 3)


def test_discover_nh_run_dir_rejects_missing_runs_root(tmp_path):
    with pytest.raises(PilotOrchestrationError):
        discover_nh_run_dir(tmp_path / "tmp_does_not_exist_xyz", "whatever")


def test_discover_nh_run_dir_rejects_ambiguous_multiple_matches(tmp_path):
    runs_root = tmp_path / "config_out" / "runs"
    dir_a = runs_root / "stage1_lead06_pilot_raw_seedA_v001_20260101_000000"
    dir_b = runs_root / "stage1_lead06_pilot_raw_seedA_v001_20260102_000000"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    with pytest.raises(PilotOrchestrationError, match="ambiguous"):
        discover_nh_run_dir(tmp_path / "config_out", "stage1_lead06_pilot_raw_seedA_v001")


# --- end-to-end run_pilot() with a fake trainer (no NH/torch) --------------

def _make_fake_train_chunk_fn(package_root, basins, experiment_name):
    """Writes checkpoint files ONLY -- never a validation_results.p. Training
    and evaluation are two distinct NH operations (see the real Moriah
    qualification run's failure, docs/stage1_lead06_pilot_v001.md); a fake
    trainer that also fabricates validation results would falsely pass tests
    that must instead exercise ``ensure_validation_results``'s explicit
    ``evaluate_checkpoint_fn`` call."""

    def _train(request: TrainChunkRequest) -> None:
        if request.is_first_chunk:
            runs_root = request.config_path.parent / "runs"
            nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
            nh_run_dir.mkdir(parents=True)
        else:
            nh_run_dir = request.nh_run_dir
        existing = list(nh_run_dir.glob("model_epoch*.pt"))
        last = max((int(p.stem.replace("model_epoch", "")) for p in existing), default=0)
        for epoch in range(last + 1, request.target_epoch + 1):
            (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    return _train


def _make_fake_evaluate_checkpoint_fn(package_root, basins):
    """Fake ``evaluate_checkpoint_fn``: writes a hand-crafted perfect-NSE
    ``validation_results.p`` for the requested epoch (no NH/torch), standing
    in for :func:`src.baseline.pilot_orchestration.default_evaluate_checkpoint`
    (the real NH ``start_evaluation`` call) in tests."""

    def _evaluate(request: EvaluationRequest) -> None:
        assert request.period == "validation"
        write_perfect_validation_results(Path(request.nh_run_dir), request.epoch, basins, package_root)

    return _evaluate


@pytest.fixture
def run_pilot_fixture(tmp_path, pilot_policy):
    basins = pick_development_basins(5)
    experiment_name = "stage1_lead06_pilot_raw_seedA_v001"

    package_root = tmp_path / "package"
    config_out_dir = tmp_path / "config_out"
    evidence_out_dir = tmp_path / "evidence"
    build_full_union_package(package_root, ts_basin_ids=basins)

    fake_train = _make_fake_train_chunk_fn(package_root, basins, experiment_name)
    fake_evaluate = _make_fake_evaluate_checkpoint_fn(package_root, basins)
    return {
        "pilot_policy": pilot_policy, "basins": basins, "package_root": package_root,
        "config_out_dir": config_out_dir, "evidence_out_dir": evidence_out_dir,
        "fake_train": fake_train, "fake_evaluate": fake_evaluate,
    }


def test_run_pilot_end_to_end_stops_on_patience_exhaustion(run_pilot_fixture):
    fx = run_pilot_fixture
    result = run_pilot(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        commands_used=["test_pilot_orchestration.py"],
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
    )
    nh_run_dir = Path(result["nh_run_dir"])
    checkpoints = sorted(nh_run_dir.glob("model_epoch*.pt"))
    assert len(checkpoints) > 0

    evidence_dir = Path(result["evidence_bundle_path"])
    record = json.loads((evidence_dir / "pilot_run_evidence.json").read_text())

    # A perfectly-flat NSE=1.0 fixture never improves past epoch 6 -- patience
    # (3 stopping-eligible events without a >=0.005 improvement) exhausts
    # before the 36-epoch budget, confirming early stopping actually fires
    # through this orchestration path (not just budget exhaustion).
    assert result["final_status"] == "stopped_patience_exhausted"
    assert result["best_checkpoint_epoch"] == 6
    assert record["run_status"] == "stopped_patience_exhausted"
    assert record["best_checkpoint_epoch"] == 6
    assert record["early_stopping_state"]["stopped"] is True
    assert record["early_stopping_state"]["stop_reason"] == "patience_exhausted"
    assert len(record["screening_events"]) > 0
    for event in record["screening_events"]:
        assert event["scope"] == "screening_subset_provisional"


def test_run_pilot_resume_is_idempotent_and_does_not_retrain(run_pilot_fixture):
    fx = run_pilot_fixture
    run_pilot(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        commands_used=["first call"],
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
    )

    call_count = {"n": 0}

    def counting_train(request):
        call_count["n"] += 1
        return fx["fake_train"](request)

    eval_call_count = {"n": 0}

    def counting_evaluate(request):
        eval_call_count["n"] += 1
        return fx["fake_evaluate"](request)

    # force intentionally left False (the default): run_pilot() always
    # rewrites its own evidence bundle regardless of force (see the
    # force-flag decoupling fix in pilot_orchestration.py), and this also
    # exercises "don't regenerate an already-generated config on resume".
    result2 = run_pilot(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        commands_used=["resume call"],
        train_chunk_fn=counting_train,
        evaluate_checkpoint_fn=counting_evaluate,
    )
    assert call_count["n"] == 0, "resume must not call the trainer again -- everything already trained"
    assert eval_call_count["n"] == 0, "resume must not re-evaluate already-saved validation results"
    assert result2["final_status"] == "stopped_patience_exhausted"
    assert result2["best_checkpoint_epoch"] == 6
    # evidence bundle must still exist and be rewritten (force=False here)
    evidence_dir = Path(result2["evidence_bundle_path"])
    assert (evidence_dir / "pilot_run_evidence.json").is_file()


# --- one realistic interrupted-resume test (task item 7) -------------------

def test_run_pilot_chunk_resumes_from_partial_interruption_without_retraining(run_pilot_fixture):
    """Checkpoints exist through epoch 8 (a wall-time SIGTERM partway through
    the chunk targeting epoch 9, after the chunk-8 checkpoint was written but
    before epoch 9's screening cadence boundary was reached) -- no epoch-9
    checkpoint, screening result, or stopping-state history entry exists yet.
    Resuming must call the trainer only to continue on to epoch 9 (not
    retrain 1-8), evaluate epoch 9 exactly once, and update the stopping
    state exactly once; a second resume must be fully idempotent (no further
    trainer call, no duplicate history entry)."""
    fx = run_pilot_fixture
    effective_policy = build_effective_policy(fx["pilot_policy"])

    run_spec, bundle, config_dir, experiment_name = prepare_pilot_run(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
    )

    common_kwargs = dict(
        pilot_policy=fx["pilot_policy"],
        config_dir=config_dir,
        experiment_name=experiment_name,
        package_root=fx["package_root"],
        target_variable=bundle.target_variable,
        lead_hours=fx["pilot_policy"].lead_hours,
        screening_basin_ids=fx["basins"],
        effective_policy=effective_policy,
    )

    # First bounded chunk (0 -> 6): trains + screens normally, exactly like
    # run_pilot()'s own first iteration. Explicit evaluation is required for
    # epochs 3 and 6 since the fake trainer no longer fabricates
    # validation_results.p itself (training and evaluation are separate NH
    # operations -- see the real qualification-run failure).
    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    nh_run_dir = first["nh_run_dir"]
    epoch1_bytes_before = (nh_run_dir / "model_epoch001.pt").read_bytes()
    epoch6_bytes_before = (nh_run_dir / "model_epoch006.pt").read_bytes()

    # Simulate an interruption partway through the NEXT chunk: epochs 7 and 8
    # already checkpointed on disk, but no epoch-9 checkpoint, screening
    # output, or stopping-state history entry exists yet.
    (nh_run_dir / "model_epoch007.pt").write_bytes(b"ckpt7")
    (nh_run_dir / "model_epoch008.pt").write_bytes(b"ckpt8")
    assert not (nh_run_dir / "model_epoch009.pt").exists()

    train_calls = []

    def counting_train(request):
        train_calls.append(request.target_epoch)
        fx["fake_train"](request)

    eval_calls = []

    def counting_evaluate(request):
        eval_calls.append(request.epoch)
        fx["fake_evaluate"](request)

    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )

    # trainer invoked exactly once, targeting epoch 9 -- not re-invoked for
    # the already-checkpointed epochs 1-8.
    assert train_calls == [9]
    assert (nh_run_dir / "model_epoch001.pt").read_bytes() == epoch1_bytes_before
    assert (nh_run_dir / "model_epoch006.pt").read_bytes() == epoch6_bytes_before
    assert (nh_run_dir / "model_epoch007.pt").read_bytes() == b"ckpt7"
    assert (nh_run_dir / "model_epoch008.pt").read_bytes() == b"ckpt8"
    assert (nh_run_dir / "model_epoch009.pt").is_file()

    # epoch 9 explicitly evaluated exactly once this chunk (item 4: future
    # screening checkpoint requires explicit evaluation); stopping state
    # updated once.
    assert eval_calls == [9]
    assert [r["epoch"] for r in resumed["screening_results"]] == [9]
    assert len(resumed["state"]["history"]) == len(first["state"]["history"]) + 1
    assert resumed["state"]["history"][-1]["epoch"] == 9

    # second resume: fully idempotent -- trainer not called again, no
    # re-evaluation (epoch 9's result pickle already saved), no duplicate
    # screening/stopping-history entry.
    train_calls_2 = []

    def counting_train_2(request):
        train_calls_2.append(request.target_epoch)
        fx["fake_train"](request)

    eval_calls_2 = []

    def counting_evaluate_2(request):
        eval_calls_2.append(request.epoch)
        fx["fake_evaluate"](request)

    resumed_again = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        train_chunk_fn=counting_train_2, evaluate_checkpoint_fn=counting_evaluate_2, **common_kwargs,
    )
    assert train_calls_2 == [], "second resume must not retrigger training -- epoch 9 already on disk"
    assert eval_calls_2 == [], "second resume must not re-evaluate -- epoch 9's result already saved"
    assert [r["epoch"] for r in resumed_again["screening_results"]] == [9]
    assert len(resumed_again["state"]["history"]) == len(resumed["state"]["history"])


# --- ensure_validation_results: explicit-evaluation prerequisite check -----

def test_ensure_validation_results_invokes_evaluator_when_missing(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    calls = []

    def fake_evaluate(request: EvaluationRequest) -> None:
        calls.append(request)
        period_dir = Path(request.nh_run_dir) / "validation" / weight_stem(request.epoch)
        period_dir.mkdir(parents=True, exist_ok=True)
        (period_dir / "validation_results.p").write_bytes(b"fake-result")

    result_path = ensure_validation_results(nh_run_dir=nh_run_dir, epoch=6, evaluate_checkpoint_fn=fake_evaluate)

    assert len(calls) == 1
    assert calls[0].nh_run_dir == nh_run_dir
    assert calls[0].epoch == 6
    assert calls[0].period == "validation"
    assert result_path.is_file()
    assert result_path == nh_run_dir / "validation" / "model_epoch006" / "validation_results.p"


def test_ensure_validation_results_reuses_existing_result_without_calling_evaluator(tmp_path):
    nh_run_dir = tmp_path / "run"
    period_dir = nh_run_dir / "validation" / weight_stem(6)
    period_dir.mkdir(parents=True)
    (period_dir / "validation_results.p").write_bytes(b"already-saved-by-nh-in-training-validation")

    calls = []

    def fake_evaluate(request: EvaluationRequest) -> None:
        calls.append(request)

    result_path = ensure_validation_results(nh_run_dir=nh_run_dir, epoch=6, evaluate_checkpoint_fn=fake_evaluate)

    assert calls == [], "an already-saved result must be reused, never re-evaluated"
    assert result_path.read_bytes() == b"already-saved-by-nh-in-training-validation"


def test_ensure_validation_results_raises_if_evaluator_does_not_produce_the_file(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()

    def noop_evaluate(request: EvaluationRequest) -> None:
        pass  # simulates a silently-failed/misconfigured evaluation

    with pytest.raises(PilotOrchestrationError, match="did not produce"):
        ensure_validation_results(nh_run_dir=nh_run_dir, epoch=6, evaluate_checkpoint_fn=noop_evaluate)
    assert not (nh_run_dir / "validation" / weight_stem(6) / "validation_results.p").exists()


def test_ensure_validation_results_propagates_evaluator_exception_and_leaves_no_trace(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()

    def crashing_evaluate(request: EvaluationRequest) -> None:
        raise RuntimeError("simulated NH evaluation crash")

    with pytest.raises(RuntimeError, match="simulated NH evaluation crash"):
        ensure_validation_results(nh_run_dir=nh_run_dir, epoch=6, evaluate_checkpoint_fn=crashing_evaluate)
    assert not (nh_run_dir / "validation" / weight_stem(6) / "validation_results.p").exists()


def test_root_logger_has_file_handler_detects_only_a_matching_filehandler(tmp_path):
    """``default_evaluate_checkpoint`` may be called once per screening epoch
    within a single process; NH's own ``setup_logging`` unconditionally opens
    a fresh ``FileHandler`` (leaking a file descriptor) on every call because
    ``logging.basicConfig`` no-ops once the root logger already has handlers.
    This guard must recognize an already-attached handler on the same path
    (so the redundant call is skipped) and must not be fooled by a handler on
    an unrelated path or a non-file handler."""
    log_path = tmp_path / "output.log"
    other_path = tmp_path / "other.log"
    root = logging.getLogger()
    added = []
    try:
        assert root_logger_has_file_handler(log_path) is False

        stream_handler = logging.StreamHandler()
        root.addHandler(stream_handler)
        added.append(stream_handler)
        assert root_logger_has_file_handler(log_path) is False, "a StreamHandler must not count as a match"

        other_handler = logging.FileHandler(str(other_path))
        root.addHandler(other_handler)
        added.append(other_handler)
        assert root_logger_has_file_handler(log_path) is False, "a FileHandler on a different path must not match"

        matching_handler = logging.FileHandler(str(log_path))
        root.addHandler(matching_handler)
        added.append(matching_handler)
        assert root_logger_has_file_handler(log_path) is True
    finally:
        for handler in added:
            root.removeHandler(handler)
            handler.close()


def test_run_pilot_chunk_screening_fails_loudly_when_evaluation_produces_nothing(run_pilot_fixture):
    """If the injected evaluator neither raises nor creates the expected
    result pickle (a silently-failed real NH evaluation), screening/stopping
    must never run against stale or absent data, and no early-stopping/
    screening-history side effect may be recorded."""
    fx = run_pilot_fixture
    effective_policy = build_effective_policy(fx["pilot_policy"])
    run_spec, bundle, config_dir, experiment_name = prepare_pilot_run(
        pilot_policy=fx["pilot_policy"], run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=fx["package_root"],
        splits_dir=SPLITS_DIR, config_out_dir=fx["config_out_dir"],
    )

    def broken_evaluate(request: EvaluationRequest) -> None:
        pass

    with pytest.raises(PilotOrchestrationError):
        run_pilot_chunk(
            pilot_policy=fx["pilot_policy"], config_dir=config_dir, experiment_name=experiment_name,
            package_root=fx["package_root"], target_variable=bundle.target_variable,
            lead_hours=fx["pilot_policy"].lead_hours, screening_basin_ids=fx["basins"],
            effective_policy=effective_policy, chunk_target_epoch=6, previous_target_epoch=0,
            is_first_chunk=True, train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=broken_evaluate,
        )

    nh_run_dir = discover_nh_run_dir(config_dir, experiment_name)
    assert (nh_run_dir / "model_epoch006.pt").is_file(), "checkpoints already trained must remain untouched"
    assert not (nh_run_dir / "pilot_early_stopping_state.json").exists(), (
        "no early-stopping state may be written when the epoch-3 (diagnostic) or epoch-6 "
        "evaluation prerequisite failed"
    )


# --- resume shaped exactly like the real Moriah qualification-run failure --

def test_run_pilot_chunk_resumes_from_real_qualification_run_failure_shape(run_pilot_fixture):
    """Mirrors the confirmed real failure (run_id emb128x64_seedA, job
    45695059): training succeeded through epoch 6 and NH's own
    validate_every-driven in-training validation already produced saved
    validation_results.p for epochs 3 and 6, but nothing in this
    orchestration module has processed them yet (e.g. this is the first
    run_pilot_chunk call after the run_dir was discovered pre-populated on
    disk by a fresh process). Resuming must: not retrain epochs 1-6; not
    re-evaluate epochs 3/6 (already saved); process epochs 3 (diagnostic,
    no stopping-state history entry) and 6 (stopping-eligible, one history
    entry); and the next training request must start from epoch 6 and
    target epoch 9, with epoch 9 explicitly evaluated before screening."""
    fx = run_pilot_fixture
    effective_policy = build_effective_policy(fx["pilot_policy"])

    run_spec, bundle, config_dir, experiment_name = prepare_pilot_run(
        pilot_policy=fx["pilot_policy"], run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH, package_root=fx["package_root"],
        splits_dir=SPLITS_DIR, config_out_dir=fx["config_out_dir"],
    )

    # Pre-populate the NH run dir exactly like the real failure: checkpoints
    # 1-6 and saved validation results for epochs 3 and 6 already on disk,
    # as if training had already completed through epoch 6 in a prior (now
    # dead) Slurm allocation.
    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])
    write_perfect_validation_results(nh_run_dir, 6, fx["basins"], fx["package_root"])

    train_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    eval_calls = []

    def counting_evaluate(request):
        eval_calls.append(request)
        fx["fake_evaluate"](request)

    common_kwargs = dict(
        pilot_policy=fx["pilot_policy"], config_dir=config_dir, experiment_name=experiment_name,
        package_root=fx["package_root"], target_variable=bundle.target_variable,
        lead_hours=fx["pilot_policy"].lead_hours, screening_basin_ids=fx["basins"],
        effective_policy=effective_policy,
    )

    # is_first_chunk=False: the real launcher's run_pilot() sets this based
    # on discover_nh_run_dir already finding a pre-populated run_dir (have_
    # started=True), exactly this scenario.
    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=False,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )

    assert train_calls == [], "epochs 1-6 already checkpointed on disk -- must not be retrained"
    assert eval_calls == [], "epoch-3/6 results already saved -- must not be re-evaluated"
    assert (nh_run_dir / "model_epoch001.pt").read_bytes() == b"ckpt1"
    assert (nh_run_dir / "model_epoch006.pt").read_bytes() == b"ckpt6"
    assert [r["epoch"] for r in first["screening_results"]] == [3, 6]

    # Epoch 3 is diagnostic-only: it must never contribute a stopping-state
    # history entry (only epoch 6 does).
    assert [h["epoch"] for h in first["state"]["history"]] == [6]
    assert first["screening_results"][0]["epoch_role"] == "diagnostic_only"
    assert first["screening_results"][1]["epoch_role"] == "stopping_eligible"

    # Next chunk resumes training from epoch 6, targeting epoch 9, and
    # explicitly evaluates epoch 9 before screening it.
    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )
    assert [r.target_epoch for r in train_calls] == [9]
    assert train_calls[0].is_first_chunk is False
    assert [r.epoch for r in eval_calls] == [9]
    assert [r["epoch"] for r in resumed["screening_results"]] == [9]
