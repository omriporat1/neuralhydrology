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
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.baseline.pilot_orchestration import (
    ACCEPTED_CONTINUATION_FILENAME,
    CAP_IDENTITY_STATE_FILENAME,
    EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME,
    HIDDEN_SIZE_IDENTITY_STATE_FILENAME,
    LR_IDENTITY_STATE_FILENAME,
    PREPARATION_RESULT_FILENAME,
    EvaluationRequest,
    PilotOrchestrationError,
    TrainChunkRequest,
    _continuation_overlay,
    actual_optimizer_updates_by_epoch,
    chunk_epoch_targets,
    compute_pilot_status_fields,
    discover_nh_run_dir,
    discover_physical_checkpoints,
    ensure_validation_results,
    enforce_pilot_cap_identity,
    enforce_pilot_embedding_dropout_identity,
    enforce_pilot_hidden_size_identity,
    enforce_pilot_learning_rate_identity,
    load_accepted_continuation_manifest,
    prepare_pilot_run,
    prepare_pilot_run_only,
    read_actual_optimizer_updates,
    resolve_trusted_chunk_checkpoint,
    root_logger_has_file_handler,
    run_pilot,
    run_pilot_chunk,
    screening_epochs_in_chunk,
    untrusted_overshoot_epochs,
)
from src.baseline.pilot_early_stopping import build_effective_policy
from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.pilot_tracking import log_pilot_screening_event as real_log_pilot_screening_event
from src.baseline.nh_seed_evaluation import weight_stem
from src.baseline.splits import sha256_of
from src.baseline.wandb_tracking import init_tracking_run

from tests._pilot_support import (
    BASELINE_POLICY_PATH,
    PILOT_POLICY_PATH,
    REAL_DEVELOPMENT,
    REAL_FULL_UNION,
    SPLITS_DIR,
    build_full_union_package,
    pick_development_basins,
    short_tmp_path,
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

def test_short_tmp_path_is_project_local_and_writable(short_tmp_path):
    expected = Path(__file__).resolve().parents[1] / "tmp"
    assert short_tmp_path.resolve().is_relative_to(expected)
    marker = short_tmp_path / "writable.txt"
    marker.write_text("ok", encoding="utf-8")
    assert marker.read_text(encoding="utf-8") == "ok"

def _make_fake_train_chunk_fn(package_root, basins, experiment_name):
    """Writes checkpoint files ONLY -- never a validation_results.p. Training
    and evaluation are two distinct NH operations (see the real Moriah
    qualification run's failure, docs/stage1_lead06_pilot_v001.md); a fake
    trainer that also fabricates validation results would falsely pass tests
    that must instead exercise ``ensure_validation_results``'s explicit
    ``evaluate_checkpoint_fn`` call.

    Reproduces NH's REAL physical continuation layout (confirmed by reading
    ``neuralhydrology.nh_run.continue_run`` and
    ``neuralhydrology.training.basetrainer.BaseTrainer._create_folder_structure``
    directly): ``start_run`` (``is_first_chunk=True``) writes checkpoints
    flat into a freshly created run directory; ``continue_run`` sets
    ``is_continue_training=True`` UNCONDITIONALLY, so every continuation --
    including a partial-first-chunk resume -- nests its new checkpoints into
    a fresh ``continue_training_from_epoch{start_epoch:03d}/`` directory
    under whatever ``request.nh_run_dir`` was passed as the continuation's
    starting directory (never flat), regardless of how deep prior
    continuations already nested it."""

    def _train(request: TrainChunkRequest) -> None:
        if request.is_first_chunk:
            runs_root = request.config_path.parent / "runs"
            nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
            nh_run_dir.mkdir(parents=True)
            target_dir = nh_run_dir
            start_epoch = 0
        else:
            base_dir = Path(request.nh_run_dir)
            if request.current_epoch is None:
                existing = list(base_dir.glob("model_epoch*.pt"))
                start_epoch = max((int(p.stem.replace("model_epoch", "")) for p in existing), default=0)
            else:
                start_epoch = request.current_epoch
            target_dir = base_dir / f"continue_training_from_epoch{start_epoch:03d}"
            target_dir.mkdir(parents=True)
        target_epoch = start_epoch + request.additional_epochs
        for epoch in range(start_epoch + 1, target_epoch + 1):
            (target_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

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
def run_pilot_fixture(short_tmp_path, pilot_policy):
    basins = pick_development_basins(5)
    experiment_name = "stage1_lead06_pilot_raw_seedA_v001"

    package_root = short_tmp_path / "package"
    config_out_dir = short_tmp_path / "config_out"
    evidence_out_dir = short_tmp_path / "evidence"
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


def test_run_pilot_bounded_to_epoch_6_trains_via_start_run_only_no_continue(run_pilot_fixture):
    """Section 8 (LR-A range-characterization campaign, docs/decision_log.md):
    one uninterrupted epoch 1->6 training segment for a fresh candidate must
    be exactly one start_run-equivalent call -- chunk_epoch_targets' full
    schedule ([6, 9, ..., 36]) bounded via max_target_epoch=6 must resolve to
    exactly [6], and run_pilot must never issue a continue_run-equivalent
    call (``TrainChunkRequest.is_first_chunk=False``) for this fresh
    candidate within this one call. No real training run is used -- same
    fake train_chunk_fn as every other test in this module, wrapped only to
    record each call's ``is_first_chunk`` flag."""
    fx = run_pilot_fixture

    full_targets = chunk_epoch_targets(fx["pilot_policy"], 36)
    assert full_targets == [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    bounded_targets = [t for t in full_targets if t <= 6]
    assert bounded_targets == [6]

    is_first_chunk_calls = []
    inner_fake_train = fx["fake_train"]

    def _tracking_train_chunk_fn(request: TrainChunkRequest) -> None:
        is_first_chunk_calls.append(request.is_first_chunk)
        inner_fake_train(request)

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
        train_chunk_fn=_tracking_train_chunk_fn,
        evaluate_checkpoint_fn=fx["fake_evaluate"],
        max_target_epoch=6,
    )

    # Exactly one training request, and it is the start_run-equivalent shape
    # -- never a continue_run-equivalent call in this successful, fresh-
    # candidate, single-segment path.
    assert is_first_chunk_calls == [True]

    assert result["final_status"] == "paused_at_max_target_epoch"
    assert result["best_checkpoint_epoch"] == 6

    nh_run_dir = Path(result["nh_run_dir"])
    checkpoints = sorted(nh_run_dir.glob("model_epoch*.pt"))
    assert [p.name for p in checkpoints] == [f"model_epoch{epoch:03d}.pt" for epoch in range(1, 7)]
    # start_run's real physical layout is flat in the base run directory --
    # no nested continue_training_from_epoch*/ directory should exist since
    # continue_run was never called.
    assert not list(nh_run_dir.glob("continue_training_from_epoch*"))


def test_run_pilot_threads_tracking_generation_default_and_explicit(run_pilot_fixture):
    # run_pilot()'s tracking_generation kwarg (default "g1") must reach
    # build_pilot_run_identity() and, from there, the evidence bundle's
    # run_identity -- this is the one place tracking_generation needed
    # threading; pilot_tracking.py itself already fully implements and tests
    # the parameter (see tests/test_pilot_tracking.py).
    fx = run_pilot_fixture
    default_result = run_pilot(
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
    default_record = json.loads(
        (Path(default_result["evidence_bundle_path"]) / "pilot_run_evidence.json").read_text()
    )
    assert default_record["run_identity"]["tracking_generation"] == "g1"

    bumped_config_out_dir = fx["config_out_dir"].parent / "config_out_g2"
    bumped_evidence_out_dir = fx["evidence_out_dir"].parent / "evidence_g2"
    bumped_result = run_pilot(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=bumped_config_out_dir,
        evidence_out_dir=bumped_evidence_out_dir,
        screening_basin_ids=fx["basins"],
        commands_used=["test_pilot_orchestration.py"],
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
        tracking_generation="g2",
    )
    bumped_record = json.loads(
        (Path(bumped_result["evidence_bundle_path"]) / "pilot_run_evidence.json").read_text()
    )
    assert bumped_record["run_identity"]["tracking_generation"] == "g2"
    assert bumped_record["run_identity"]["wandb_run_id"] != default_record["run_identity"]["wandb_run_id"]


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


# --- realistic resume/continuation tests (task item 7) ---------------------

def _prepare_common_kwargs(fx):
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
    return common_kwargs


def test_run_pilot_chunk_resume_after_successful_epoch9_is_idempotent(run_pilot_fixture):
    """A cleanly-completed continuation to epoch 9 (the trusted, correctly
    nested case) must be fully idempotent on a second resume call: no
    further trainer call, no re-evaluation (epoch 9's result already saved),
    no duplicate stopping-history entry -- covers "reuse of existing epoch-9
    result" and "idempotent repeated-resume-after-epoch-9". Orchestration
    state (``logged_screening_epochs``) is now persisted unconditionally per
    epoch -- decoupled from whether a ``tracking_run`` was passed at all, not
    just moved earlier within the old ``tracking_run is not None`` guard --
    see the job-45731908 fix note in ``run_pilot_chunk``. So epoch 9 is
    recognized as already-logged on the third call and skipped outright,
    rather than falling through to re-processing that only happened to look
    idempotent before because ``ensure_validation_results``/
    ``record_screening_event`` separately tolerate replaying the same
    epoch."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    second = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    assert second["blocked"] is False
    expected_dir = first["nh_run_dir"] / "continue_training_from_epoch006"
    assert second["checkpoint_dir_for_target"] == expected_dir
    assert [r["epoch"] for r in second["screening_results"]] == [9]

    train_calls = []
    eval_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        eval_calls.append(request)
        fx["fake_evaluate"](request)

    third = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )
    assert train_calls == [], "epoch 9 already trusted on disk -- must not retrain"
    assert eval_calls == [], "epoch 9's result already saved -- must not re-evaluate"
    assert third["blocked"] is False
    assert third["checkpoint_dir_for_target"] == expected_dir
    assert third["screening_results"] == [], "epoch 9 is already logged -- must be skipped, not re-processed"
    assert len(third["state"]["history"]) == len(second["state"]["history"])


def test_run_pilot_chunk_blocks_on_untrusted_checkpoints_in_expected_continuation_directory(run_pilot_fixture):
    """A continuation attempt that was interrupted after NH already created
    its nested continue_training_from_epoch006/ directory and wrote
    checkpoints 7-8, but before reaching epoch 9, cannot be safely retried
    automatically -- a fresh train_chunk_fn call would collide with the
    already-written epoch 7/8 checkpoint files. This must surface as a
    blocked status with a clear reason -- never a raised exception, never a
    silent/unsafe resume -- and must leave the pre-existing checkpoints and
    all prior logical state completely untouched. Replaces a now-factually-
    wrong earlier version of this test that assumed (incorrectly)
    checkpoints from an interrupted continuation would land flat in the base
    run directory and that resuming into them would silently succeed."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    nh_run_dir = first["nh_run_dir"]

    # Realistic interrupted-continuation shape: the nested continuation
    # directory already exists with checkpoints 7-8, killed before epoch 9.
    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()
    (cont_dir / "model_epoch007.pt").write_bytes(b"ckpt7")
    (cont_dir / "model_epoch008.pt").write_bytes(b"ckpt8")

    train_calls = []
    eval_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        eval_calls.append(request)
        fx["fake_evaluate"](request)

    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=nh_run_dir,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )

    assert resumed["blocked"] is True
    assert "occupy epoch" in resumed["blocked_reason"]
    assert resumed["checkpoint_dir_for_target"] is None
    assert resumed["screening_results"] == []
    assert train_calls == [], "must never attempt a train_chunk_fn call that would collide with epochs 7/8"
    assert eval_calls == []
    assert (cont_dir / "model_epoch007.pt").read_bytes() == b"ckpt7"
    assert (cont_dir / "model_epoch008.pt").read_bytes() == b"ckpt8"
    assert not (cont_dir / "model_epoch009.pt").exists()

    # calling again is idempotent: still blocked, never crashes, never
    # retries training.
    resumed_again = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=nh_run_dir,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )
    assert resumed_again["blocked"] is True
    assert train_calls == []


def test_run_pilot_chunk_blocks_on_empty_pre_existing_continuation_directory(run_pilot_fixture):
    """A continuation attempt killed before writing even its first new
    checkpoint still leaves an empty (but already-created)
    continue_training_from_epoch006/ directory behind.
    NeuralHydrology's own continue_run refuses to create an already-existing
    run directory (BaseTrainer._create_folder_structure raises RuntimeError
    rather than resuming into or recreating it), so a fresh train_chunk_fn
    call here would crash inside real NH. untrusted_overshoot_epochs alone
    would not catch this (no checkpoint occupies the target range at all) --
    this is the specific case the expected_dir.is_dir() guard in
    _advance_chunk_via_continuation exists for."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    nh_run_dir = first["nh_run_dir"]

    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()  # empty -- killed before its first checkpoint

    train_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=nh_run_dir,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )

    assert resumed["blocked"] is True
    assert "already exists" in resumed["blocked_reason"]
    assert resumed["checkpoint_dir_for_target"] is None
    assert train_calls == [], "must never attempt a train_chunk_fn call that would crash inside real NH"
    assert list(cont_dir.iterdir()) == []


def test_continuation_overlay_writes_additive_epochs_and_explicit_continue_from_epoch():
    """Direct unit test of _continuation_overlay -- the exact dict
    default_train_chunk hands NH as the ``pilot_epoch_overlay.yaml`` content.
    Neither original bug (absolute-vs-additive ``epochs``, and treating a
    continuation's ``nh_run_dir`` as the base directory) was caught by any
    test exercising this real function, because every other test in this
    file injects a fake train_chunk_fn that only approximates its contract.
    This test is NH/torch-free (pure dict construction) so it can run
    locally without either installed, unlike default_train_chunk itself."""
    request = TrainChunkRequest(
        is_first_chunk=False,
        config_path=Path("unused_config.yml"),
        nh_run_dir=Path("unused_run_dir"),
        current_epoch=6,
        logical_target_epoch=9,
        additional_epochs=3,
    )

    overlay = _continuation_overlay(request)

    assert overlay == {"epochs": 3, "continue_from_epoch": 6}


def test_continuation_overlay_omits_continue_from_epoch_when_current_epoch_is_none():
    """The one legitimate case for a missing continue_from_epoch: the
    fully-degenerate zero-checkpoint corner (TrainChunkRequest.current_epoch
    docstring). additional_epochs still carries the real (absolute, since
    there is no prior epoch to add to) target epoch count."""
    request = TrainChunkRequest(
        is_first_chunk=False,
        config_path=Path("unused_config.yml"),
        nh_run_dir=Path("unused_run_dir"),
        current_epoch=None,
        logical_target_epoch=6,
        additional_epochs=6,
    )

    overlay = _continuation_overlay(request)

    assert overlay == {"epochs": 6}
    assert "continue_from_epoch" not in overlay


def test_run_pilot_chunk_computes_additive_not_absolute_epochs_across_two_chunk_transitions(run_pilot_fixture):
    """Covers the additive-epoch-semantics correction across TWO successive
    chunk transitions (6 -> 9, then 9 -> 12): current_epoch/additional_epochs
    must reflect each chunk's own bounded span, never an absolute target,
    and each chunk's checkpoint must land in a freshly, correctly nested
    continuation directory under the PREVIOUS chunk's own directory."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    train_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    second = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )
    third = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=second["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )

    continuation_calls = [r for r in train_calls if not r.is_first_chunk]
    assert [r.current_epoch for r in continuation_calls] == [6, 9]
    assert [r.additional_epochs for r in continuation_calls] == [3, 3]
    assert [r.logical_target_epoch for r in continuation_calls] == [9, 12]

    expected_second_dir = first["nh_run_dir"] / "continue_training_from_epoch006"
    expected_third_dir = expected_second_dir / "continue_training_from_epoch009"
    assert second["checkpoint_dir_for_target"] == expected_second_dir
    assert third["checkpoint_dir_for_target"] == expected_third_dir
    assert (expected_third_dir / "model_epoch012.pt").is_file()


def test_run_pilot_chunk_never_requests_a_non_validation_evaluation_period(run_pilot_fixture):
    """Confirms goal "no test/spatial-holdout path introduced": every
    evaluation request this orchestration issues is for the "validation"
    period only, across a first chunk and a continuation chunk."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    periods = []

    def recording_evaluate(request):
        periods.append(request.period)
        fx["fake_evaluate"](request)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=recording_evaluate, **common_kwargs,
    )
    run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=recording_evaluate, **common_kwargs,
    )
    assert periods, "expected at least one explicit evaluation call"
    assert set(periods) == {"validation"}


def test_run_pilot_chunk_evaluator_failure_leaves_prior_logical_state_unchanged(run_pilot_fixture):
    """An evaluator failure on a LATER chunk (after a prior chunk already
    succeeded) must propagate loudly and must not mutate the already-recorded
    epoch-6 stopping-history entry or add a partial epoch-9 entry."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"], **common_kwargs,
    )

    def broken_evaluate(request):
        pass

    with pytest.raises(PilotOrchestrationError):
        run_pilot_chunk(
            chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
            previous_checkpoint_dir=first["checkpoint_dir_for_target"],
            train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=broken_evaluate, **common_kwargs,
        )

    es_path = first["nh_run_dir"] / "pilot_early_stopping_state.json"
    state = json.loads(es_path.read_text())
    assert [h["epoch"] for h in state["history"]] == [6]


def test_run_pilot_chunk_logs_checkpoint_reference_at_resolved_physical_path(run_pilot_fixture):
    """Goal 8: tracking/checksum references must use the resolved PHYSICAL
    checkpoint path (inside the nested continuation directory), never the
    base run directory a naive ``nh_run_dir / f"model_epoch{epoch:03d}.pt"``
    guess would produce."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    tracking_run = init_tracking_run(
        {"enabled": False, "mode": "disabled", "max_artifact_reference_bytes": 10_000_000}, {}
    )

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"],
        tracking_run=tracking_run, **common_kwargs,
    )
    second = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"],
        tracking_run=tracking_run, **common_kwargs,
    )

    expected_dir = first["nh_run_dir"] / "continue_training_from_epoch006"
    assert second["checkpoint_dir_for_target"] == expected_dir
    ref = next(r for r in tracking_run.artifact_references if r["name"] == "checkpoint_epoch_009")
    assert ref["path"] == str(expected_dir / "model_epoch009.pt")


class _AlwaysFailingWandbRun:
    """Fake real-backend wandb run whose log()/summary/finish() always raise
    -- proves a W&B logging failure during a real pilot chunk never breaks
    screening/early-stopping/orchestration-state persistence (the
    failure-isolation boundary in wandb_tracking._guard_backend_call)."""

    def __init__(self):
        self.config = {}

    def log(self, data, step=None):
        raise RuntimeError("simulated wandb.log failure")

    @property
    def summary(self):
        raise RuntimeError("simulated wandb.summary failure")

    def finish(self):
        raise RuntimeError("simulated wandb.finish failure")


class _AlwaysFailingWandbModule(types.ModuleType):
    def __init__(self):
        super().__init__("wandb")

    def init(self, **kwargs):
        return _AlwaysFailingWandbRun()


def test_run_pilot_chunk_screening_log_failure_does_not_break_orchestration_state(monkeypatch, run_pilot_fixture):
    """A real (fake, always-failing) W&B backend must never stop screening/
    early-stopping/orchestration-state persistence -- only degrade tracking,
    best-effort, per wandb_tracking.py's failure-isolation design."""
    monkeypatch.setitem(sys.modules, "wandb", _AlwaysFailingWandbModule())
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    tracking_run = init_tracking_run(
        {"enabled": True, "mode": "offline", "project": "test", "max_artifact_reference_bytes": 10_000_000}, {}
    )
    assert tracking_run.backend == "wandb"

    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = run_pilot_chunk(
            chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
            train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"],
            tracking_run=tracking_run, **common_kwargs,
        )

    # Scientific/orchestration state is fully intact despite every W&B call failing.
    es_path = Path(result["nh_run_dir"]) / "pilot_early_stopping_state.json"
    state = json.loads(es_path.read_text())
    assert [h["epoch"] for h in state["history"]] == [6]
    orchestration_state = json.loads((Path(result["nh_run_dir"]) / "pilot_orchestration_state.json").read_text())
    assert orchestration_state["logged_screening_epochs"] == [3, 6]

    # Tracking itself is honestly reported as degraded, never silently "fine".
    # Both the screening-metrics log AND the checkpoint-reference log (the
    # exact call that killed job 45731908 when it raised uncaught) must be
    # recorded as independently degraded, never masked as a clean finish.
    assert tracking_run.degraded is True
    assert "log_scientific_metrics" in tracking_run.degraded_operations
    assert "log_checkpoint_reference" in tracking_run.degraded_operations


def test_run_pilot_chunk_persists_state_before_telemetry_even_if_telemetry_raises_directly(
    monkeypatch, run_pilot_fixture
):
    """Proves task 2C's persistence ORDERING independently of 2B's
    non-fatal-tracking guarantee: even if a telemetry call is forced to
    raise straight out of run_pilot_chunk (bypassing wandb_tracking.py's own
    internal failure isolation entirely, as if some future call were added
    without it), the screening/early-stopping/orchestration state for the
    epoch just processed must already be durable on disk -- this is exactly
    the ordering bug that lost job 45731908's epoch-6 processing when
    log_pilot_checkpoint_reference raised uncaught. The forced raise is
    scoped to epoch 6 only (epoch 3 -- diagnostic-only, processed first --
    must complete normally) so this reproduces the real failure shape
    exactly, including the epoch-3 early-stopping no-op."""

    def _raise_only_for_epoch_6(*args, **kwargs):
        if kwargs.get("epoch") == 6:
            raise RuntimeError("simulated telemetry failure, bypassing 2B")
        return real_log_pilot_screening_event(*args, **kwargs)

    monkeypatch.setattr("src.baseline.pilot_orchestration.log_pilot_screening_event", _raise_only_for_epoch_6)
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)

    tracking_run = init_tracking_run({"enabled": False, "mode": "disabled", "max_artifact_reference_bytes": 10_000_000}, {})

    with pytest.raises(RuntimeError, match="simulated telemetry failure"):
        run_pilot_chunk(
            chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
            train_chunk_fn=fx["fake_train"], evaluate_checkpoint_fn=fx["fake_evaluate"],
            tracking_run=tracking_run, **common_kwargs,
        )

    nh_run_dir = common_kwargs["config_dir"] / "runs" / f'{common_kwargs["experiment_name"]}_20260101_000000'
    es_state = json.loads((nh_run_dir / "pilot_early_stopping_state.json").read_text())
    assert [h["epoch"] for h in es_state["history"]] == [6], (
        "epoch 6's early-stopping state must already be persisted before the telemetry call that then raised"
    )
    orchestration_state = json.loads((nh_run_dir / "pilot_orchestration_state.json").read_text())
    assert orchestration_state["logged_screening_epochs"] == [3, 6], (
        "epoch 6 must already be recorded as logged in orchestration state before the telemetry call that then raised"
    )


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
    assert [r.logical_target_epoch for r in train_calls] == [9]
    assert train_calls[0].is_first_chunk is False
    assert train_calls[0].nh_run_dir == nh_run_dir
    assert train_calls[0].current_epoch == 6
    assert train_calls[0].additional_epochs == 3
    assert [r.epoch for r in eval_calls] == [9]
    expected_dir = nh_run_dir / "continue_training_from_epoch006"
    assert eval_calls[0].nh_run_dir == expected_dir
    assert resumed["checkpoint_dir_for_target"] == expected_dir
    assert [r["epoch"] for r in resumed["screening_results"]] == [9]


def test_run_pilot_recovers_job_45731908_shaped_state_without_auto_continuing(run_pilot_fixture):
    """Reproduces job 45731908's exact real starting state (task item 11):
    checkpoints 1-6 already flat on disk, epoch 3's validation_results.p
    already saved (NH's in-training validation), but NO epoch-6 validation
    result (the run died mid-epoch-6 screening, inside the checkpoint-
    reference call, before evaluate_screening_checkpoint ever ran), and
    neither pilot_orchestration_state.json nor pilot_early_stopping_state.json
    exist yet for this run. A single recovery call bounded with
    max_target_epoch=6 must: never retrain epochs 1-6; reuse the existing
    epoch-3 result without re-evaluating it; evaluate epoch 6 exactly once;
    record epoch 6 as the first (and only) early-stopping history entry;
    and stop there -- never automatically training/screening epoch 9 within
    the same call, so a human can review epoch 6 first."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])

    assert not (nh_run_dir / "pilot_orchestration_state.json").exists()
    assert not (nh_run_dir / "pilot_early_stopping_state.json").exists()

    train_calls = []
    evaluate_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        evaluate_calls.append(request)
        fx["fake_evaluate"](request)

    result = run_pilot(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        commands_used=["test recovery replay of job 45731908"],
        train_chunk_fn=counting_train,
        evaluate_checkpoint_fn=counting_evaluate,
        max_target_epoch=6,
    )

    assert train_calls == [], "checkpoints 1-6 already exist on disk -- recovery must not retrain"
    assert [r.epoch for r in evaluate_calls] == [6], "epoch 3 already saved (reuse); epoch 6 must be newly evaluated"
    assert result["final_status"] == "paused_at_max_target_epoch"

    orchestration_state = json.loads((nh_run_dir / "pilot_orchestration_state.json").read_text())
    assert orchestration_state["logged_screening_epochs"] == [3, 6]

    es_state = json.loads((nh_run_dir / "pilot_early_stopping_state.json").read_text())
    assert [h["epoch"] for h in es_state["history"]] == [6]
    assert es_state["stopped"] is False
    assert result["best_checkpoint_epoch"] == 6


def test_run_pilot_recovery_of_job_45731908_shaped_state_is_idempotent_on_replay(run_pilot_fixture):
    """Calling the exact same bounded recovery a second time (e.g. a retried
    orchestration invocation before a human has reviewed epoch 6) must not
    retrain, must not re-evaluate epoch 3 or epoch 6, and must not append a
    second history entry for epoch 6."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])

    def _run_once():
        train_calls = []
        evaluate_calls = []

        def counting_train(request):
            train_calls.append(request)
            fx["fake_train"](request)

        def counting_evaluate(request):
            evaluate_calls.append(request)
            fx["fake_evaluate"](request)

        result = run_pilot(
            pilot_policy=fx["pilot_policy"],
            run_id="raw_seedA",
            baseline_policy_path=BASELINE_POLICY_PATH,
            package_root=fx["package_root"],
            splits_dir=SPLITS_DIR,
            config_out_dir=fx["config_out_dir"],
            evidence_out_dir=fx["evidence_out_dir"],
            screening_basin_ids=fx["basins"],
            commands_used=["test recovery replay of job 45731908"],
            train_chunk_fn=counting_train,
            evaluate_checkpoint_fn=counting_evaluate,
            max_target_epoch=6,
        )
        return result, train_calls, evaluate_calls

    first_result, first_train, first_eval = _run_once()
    assert [r.epoch for r in first_eval] == [6]
    assert first_result["final_status"] == "paused_at_max_target_epoch"

    second_result, second_train, second_eval = _run_once()

    assert second_train == [], "replay must not retrain"
    assert second_eval == [], "replay must not re-evaluate an already-logged epoch"
    assert second_result["final_status"] == "paused_at_max_target_epoch"

    orchestration_state = json.loads((nh_run_dir / "pilot_orchestration_state.json").read_text())
    assert orchestration_state["logged_screening_epochs"] == [3, 6]
    es_state = json.loads((nh_run_dir / "pilot_early_stopping_state.json").read_text())
    assert [h["epoch"] for h in es_state["history"]] == [6], "replay must not append a duplicate history entry"


def test_run_pilot_chunk_real_qualification_run_evidence_with_overshoot_checkpoints_present(run_pilot_fixture):
    """Exact current real Moriah artifact state (job 45705457): checkpoints
    1-6 flat, then a continue_training_from_epoch006/ directory containing
    checkpoints 7-15 (the additive-``epochs`` bug's byproduct -- a clean
    chunk(6->9) continuation should only have produced 7-9). Chunk (6->9)
    must trust exactly epoch 9 and never screen/train/evaluate epochs
    10-15 merely because their files exist. The FOLLOWING chunk (9->12)
    must then be BLOCKED (never silently resumed from epoch 15), since
    epoch 12 in that same directory was never produced by a continuation
    that actually started at epoch 9 -- see module docstring's worked
    example distinguishing epoch 9 (trusted) from epoch 12 (untrusted
    overshoot in the same physical directory)."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])
    write_perfect_validation_results(nh_run_dir, 6, fx["basins"], fx["package_root"])

    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()
    for epoch in range(7, 16):
        (cont_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    train_calls = []
    eval_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        eval_calls.append(request)
        fx["fake_evaluate"](request)

    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=False,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )
    assert [r["epoch"] for r in first["screening_results"]] == [3, 6]

    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )
    assert train_calls == [], "epoch 9 already trusted (owned by the exact expected continuation dir) -- no retrain"
    assert resumed["blocked"] is False
    assert resumed["checkpoint_dir_for_target"] == cont_dir
    assert [r["epoch"] for r in resumed["screening_results"]] == [9]
    assert [r.epoch for r in eval_calls] == [9], "epochs 10-15 must never be screened merely because they exist"

    next_chunk = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )
    assert next_chunk["blocked"] is True
    assert "12" in next_chunk["blocked_reason"]
    assert train_calls == [], "must never train into epoch 12's range while an untrusted checkpoint occupies it"
    assert next_chunk["checkpoint_dir_for_target"] is None

    status = compute_pilot_status_fields(nh_run_dir, pilot_policy=fx["pilot_policy"])
    assert status["highest_physical_checkpoint_epoch"] == 15
    assert status["highest_screened_epoch"] == 9
    assert status["overshoot_epochs"] == [10, 11, 12, 13, 14, 15]
    assert status["safe_to_continue_automatically"] is False


def test_run_pilot_end_to_end_propagates_blocked_continuation_overshoot_conflict(run_pilot_fixture):
    """Real Moriah recovery job 45718473's exact shape: checkpoints 1-6
    flat, continue_training_from_epoch006/ containing checkpoints 7-15 (the
    additive-epochs bug's byproduct). run_pilot() must reuse the trusted
    epoch-9 checkpoint (no training), then refuse to advance into the 9->12
    chunk (blocked by the 10-15 overshoot) -- and this blocked chunk result
    must reach run_pilot()'s own top-level return with a non-null
    final_status/blocked_reason, not be lost or silently downgraded to a
    completed status (the launcher status-propagation defect job 45718473
    exposed -- see docs/decision_log.md)."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])
    write_perfect_validation_results(nh_run_dir, 6, fx["basins"], fx["package_root"])

    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()
    for epoch in range(7, 16):
        (cont_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    train_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

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
        train_chunk_fn=counting_train,
        evaluate_checkpoint_fn=fx["fake_evaluate"],
    )

    assert train_calls == [], "epoch 9 already trusted -- must never train into the 10-15 overshoot range"
    assert result["final_status"] == "blocked_continuation_overshoot_conflict"
    assert result["blocked_reason"] is not None
    assert "12" in result["blocked_reason"]
    assert result["best_checkpoint_epoch"] == 6
    assert result["overshoot_epochs"] == [10, 11, 12, 13, 14, 15]
    assert result["safe_to_continue_automatically"] is False
    assert result["highest_screened_epoch"] == 9

    evidence_dir = Path(result["evidence_bundle_path"])
    record = json.loads((evidence_dir / "pilot_run_evidence.json").read_text())
    assert record["run_status"] == "blocked_continuation_overshoot_conflict"


def test_run_pilot_end_to_end_rerun_of_fully_screened_earlier_chunks_is_idempotent(run_pilot_fixture):
    """Real Moriah verification job 45718742's exact shape: checkpoints 1-6
    flat, continue_training_from_epoch006/ containing 7-15 (untrusted
    overshoot beyond the already-screened frontier), and -- unlike the
    blocked-conflict test above -- this run's OWN persisted state already
    shows epochs 3/6/9 fully screened and logged directly via
    pilot_orchestration_state.json (not derived from disk). run_pilot() always restarts its
    chunk walk from target=6 on every call (see chunk_epoch_targets), so a
    rerun must recognize epochs 3/6/9 as already-logged
    (pilot_orchestration_state.json's logged_screening_epochs) and skip
    them outright -- previously it re-fed epoch 6 into
    record_official_validation_event after the persisted early-stopping
    history's last entry had already advanced to epoch 9, raising
    PilotEarlyStoppingError("epoch 6 is not after the last recorded epoch
    9 -- out of order"). Must instead proceed straight through to
    detecting the 10-15 overshoot ahead of the next intended screening
    epoch (12) and return the same blocked_continuation_overshoot_conflict
    result as a fresh run would, without touching persisted state."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]
    effective_policy = common_kwargs["effective_policy"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()
    for epoch in range(7, 16):
        (cont_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    # validation_results.p pickles ARE written for epochs 3/6/9 here, matching
    # the real invariant an epoch is only ever added to logged_screening_epochs
    # after its validation result already exists on disk (see
    # execute_prepared_pilot_run's screening-history reconstruction, which
    # re-reads these existing pickles via evaluate_screening_checkpoint to
    # rebuild the full history -- this is not new NH evaluation and must not
    # touch the fake evaluate callback below). Epoch 9's checkpoint physically
    # lives under cont_dir (not nh_run_dir), so its validation artifact must
    # too, matching PhysicalCheckpoint.owning_run_dir for that epoch. The test
    # still proves the rerun never re-evaluates via the live callback: it must
    # skip straight past on the persisted logged_screening_epochs contract
    # alone, without ever invoking counting_evaluate below.
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])
    write_perfect_validation_results(nh_run_dir, 6, fx["basins"], fx["package_root"])
    write_perfect_validation_results(cont_dir, 9, fx["basins"], fx["package_root"])

    orchestration_state_path = nh_run_dir / "pilot_orchestration_state.json"
    orchestration_state_path.write_text(json.dumps({"logged_screening_epochs": [3, 6, 9]}))

    early_stopping_state = {
        "schema_version": 1,
        "policy_name": effective_policy["policy_name"],
        "metric_name": effective_policy["metric_name"],
        "higher_is_better": bool(effective_policy["higher_is_better"]),
        "history": [
            {"epoch": 6, "metric_value": 0.20454161610527344, "is_new_best": True},
            {"epoch": 9, "metric_value": 0.18124855313577198, "is_new_best": False},
        ],
        "best_epoch": 6,
        "best_metric_value": 0.20454161610527344,
        "events_since_best_improvement": 1,
        "stopped": False,
        "stop_reason": None,
        "stop_epoch": None,
    }
    (nh_run_dir / "pilot_early_stopping_state.json").write_text(json.dumps(early_stopping_state))

    train_calls = []
    evaluate_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        evaluate_calls.append(request)
        fx["fake_evaluate"](request)

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
        train_chunk_fn=counting_train,
        evaluate_checkpoint_fn=counting_evaluate,
    )

    assert train_calls == [], "no epoch in range 1-15 is new -- rerun must never train"
    assert evaluate_calls == [], "epochs 3/6/9 already logged -- rerun must never re-evaluate them"

    assert result["final_status"] == "blocked_continuation_overshoot_conflict"
    assert result["blocked_reason"] is not None
    assert "12" in result["blocked_reason"]
    assert result["best_checkpoint_epoch"] == 6
    assert result["highest_physical_checkpoint_epoch"] == 15
    assert result["highest_screened_epoch"] == 9
    assert result["next_intended_screening_epoch"] == 12
    assert result["overshoot_epochs"] == [10, 11, 12, 13, 14, 15]
    assert result["safe_to_continue_automatically"] is False

    assert json.loads(orchestration_state_path.read_text()) == {"logged_screening_epochs": [3, 6, 9]}
    assert json.loads((nh_run_dir / "pilot_early_stopping_state.json").read_text()) == early_stopping_state


def test_run_pilot_chunk_rejects_partial_first_chunk_before_any_state_change(run_pilot_fixture):
    """Only two first-chunk shapes are supported: no checkpoints at all, or
    the complete epoch 1-6 range already flat in the base run directory
    (both covered by the tests above). A PARTIAL first chunk -- e.g.
    epochs 1-4 present, target 6 -- is intentionally unsupported: epoch 3
    would resolve against the base run directory while epoch 6 would land
    in a newly nested continuation directory, so this module refuses
    rather than guessing. Must fail clearly (blocked, not a crash) before
    any train/evaluate call or state/tracking-file write."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 5):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    train_calls = []
    eval_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        eval_calls.append(request)
        fx["fake_evaluate"](request)

    result = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=False,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate, **common_kwargs,
    )

    assert result["blocked"] is True
    assert "partial" in result["blocked_reason"]
    assert "unsupported" in result["blocked_reason"]
    assert result["checkpoint_dir_for_target"] is None
    assert result["screening_results"] == []
    assert train_calls == [], "must never attempt a train_chunk_fn call for an unsupported partial first chunk"
    assert eval_calls == []
    assert [p.name for p in nh_run_dir.glob("model_epoch*.pt")] == [
        "model_epoch001.pt", "model_epoch002.pt", "model_epoch003.pt", "model_epoch004.pt",
    ], "no new checkpoint/continuation directory may be created while rejecting this state"
    assert not (nh_run_dir / "pilot_early_stopping_state.json").exists(), \
        "must fail before any pilot state file is written"
    assert not (nh_run_dir / "pilot_orchestration_state.json").exists()


# --- low-level checkpoint-discovery unit tests -------------------------------

def test_discover_physical_checkpoints_across_base_and_one_continuation_dir(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    for epoch in range(1, 7):
        (base / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")
    cont = base / "continue_training_from_epoch006"
    cont.mkdir()
    for epoch in range(7, 10):
        (cont / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")

    inventory = discover_physical_checkpoints(base)
    assert set(inventory) == set(range(1, 10))
    for epoch in range(1, 7):
        assert inventory[epoch].owning_run_dir == base
    for epoch in range(7, 10):
        assert inventory[epoch].owning_run_dir == cont


def test_discover_physical_checkpoints_finds_doubly_nested_continuation_dir(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    (base / "model_epoch006.pt").write_bytes(b"x")
    cont1 = base / "continue_training_from_epoch006"
    cont1.mkdir()
    (cont1 / "model_epoch009.pt").write_bytes(b"x")
    cont2 = cont1 / "continue_training_from_epoch009"
    cont2.mkdir()
    (cont2 / "model_epoch012.pt").write_bytes(b"x")

    inventory = discover_physical_checkpoints(base)
    assert set(inventory) == {6, 9, 12}
    assert inventory[12].owning_run_dir == cont2


def test_discover_physical_checkpoints_raises_loudly_on_duplicate_epoch_claim(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    (base / "model_epoch009.pt").write_bytes(b"x")
    cont = base / "continue_training_from_epoch006"
    cont.mkdir()
    (cont / "model_epoch009.pt").write_bytes(b"y")

    with pytest.raises(PilotOrchestrationError, match="ambiguous physical checkpoint inventory"):
        discover_physical_checkpoints(base)


def test_discover_physical_checkpoints_ignores_malformed_names(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    (base / "model_epoch006.pt").write_bytes(b"x")
    (base / "model_epoch6.pt").write_bytes(b"x")  # not zero-padded
    (base / "model_epochabc.pt").write_bytes(b"x")  # non-numeric
    (base / "model_epoch007.pt.bak").write_bytes(b"x")  # wrong extension
    malformed_cont = base / "continue_training_from_epochABC"
    malformed_cont.mkdir()
    (malformed_cont / "model_epoch099.pt").write_bytes(b"x")  # inside a malformed dir name -- must not be recursed into

    inventory = discover_physical_checkpoints(base)
    assert set(inventory) == {6}


def test_resolve_trusted_chunk_checkpoint_and_untrusted_overshoot_epochs_distinguish_by_owning_dir(tmp_path):
    base = tmp_path / "run"
    base.mkdir()
    (base / "model_epoch006.pt").write_bytes(b"x")
    cont = base / "continue_training_from_epoch006"
    cont.mkdir()
    for epoch in (9, 12):
        (cont / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")

    inventory = discover_physical_checkpoints(base)
    trusted_9 = resolve_trusted_chunk_checkpoint(inventory, base, 6, 9)
    assert trusted_9 is not None and trusted_9.owning_run_dir == cont

    # epoch 12 physically sits in the SAME directory, but that directory was
    # never continued from epoch 9 -- must not be trusted for chunk (9->12).
    trusted_12_from_9 = resolve_trusted_chunk_checkpoint(inventory, cont, 9, 12)
    assert trusted_12_from_9 is None
    assert untrusted_overshoot_epochs(inventory, 9, 12) == [12]


def test_compute_pilot_status_fields_distinguishes_physical_from_screened_and_flags_overshoot(tmp_path, pilot_policy):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")
    cont = nh_run_dir / "continue_training_from_epoch006"
    cont.mkdir()
    for epoch in range(7, 16):
        (cont / f"model_epoch{epoch:03d}.pt").write_bytes(b"x")
    es_state = {"stopped": False, "history": [{"epoch": 3}, {"epoch": 6}]}
    (nh_run_dir / "pilot_early_stopping_state.json").write_text(json.dumps(es_state))

    fields = compute_pilot_status_fields(nh_run_dir, pilot_policy)
    assert fields["highest_physical_checkpoint_epoch"] == 15
    assert fields["highest_screened_epoch"] == 6
    assert fields["overshoot_epochs"] == [7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert fields["safe_to_continue_automatically"] is False


# --- explicit, run-specific overshoot adoption (pilot_accepted_continuation.json) ---
#
# Real emb128x64_seedA shape (job 45705457, reviewed 2026-07-29/30): same
# checkpoints-1-6-flat + continue_training_from_epoch006/{7..15} layout as
# the untrusted-overshoot tests above, but each epoch also gets a matching
# optimizer_state_epoch{N:03d}.pt (NH's real convention, confirmed in
# flashnh_emb128x64_seedA_continuation_evidence_2026-07-29.txt) since the
# manifest pins both a model and an optimizer checkpoint per epoch.
_ACCEPTED_RUN_ID = "raw_seedA"


def _build_overshoot_scenario_with_optimizers(fx, common_kwargs):
    config_dir = common_kwargs["config_dir"]
    experiment_name = common_kwargs["experiment_name"]

    runs_root = config_dir / "runs"
    nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
    nh_run_dir.mkdir(parents=True)
    for epoch in range(1, 7):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        (nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt").write_bytes(f"opt{epoch}".encode())
    write_perfect_validation_results(nh_run_dir, 3, fx["basins"], fx["package_root"])
    write_perfect_validation_results(nh_run_dir, 6, fx["basins"], fx["package_root"])

    cont_dir = nh_run_dir / "continue_training_from_epoch006"
    cont_dir.mkdir()
    for epoch in range(7, 16):
        (cont_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        (cont_dir / f"optimizer_state_epoch{epoch:03d}.pt").write_bytes(f"opt{epoch}".encode())
    return nh_run_dir, cont_dir


def _counting_wrappers(fx):
    train_calls = []
    eval_calls = []

    def counting_train(request):
        train_calls.append(request)
        fx["fake_train"](request)

    def counting_evaluate(request):
        eval_calls.append(request)
        fx["fake_evaluate"](request)

    return train_calls, eval_calls, counting_train, counting_evaluate


def _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate):
    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=False,
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        previous_checkpoint_dir=first["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    return resumed


def _entry_for_epoch(cont_dir, epoch, *, model_sha256=None, optimizer_sha256=None):
    model_path = cont_dir / f"model_epoch{epoch:03d}.pt"
    optimizer_path = cont_dir / f"optimizer_state_epoch{epoch:03d}.pt"
    return {
        "model_path": f"continue_training_from_epoch006/model_epoch{epoch:03d}.pt",
        "model_sha256": model_sha256 or sha256_of(model_path),
        "optimizer_path": f"continue_training_from_epoch006/optimizer_state_epoch{epoch:03d}.pt",
        "optimizer_sha256": optimizer_sha256 or sha256_of(optimizer_path),
    }


def _write_accepted_manifest(nh_run_dir, run_id, entries):
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "decision": "conditional_sequential_adoption_epoch6_to_15",
        "accepted_directory": "continue_training_from_epoch006",
        "accepted_checkpoints": entries,
        "provenance_basis": "job 45705457 continuation evidence (test fixture)",
    }
    (Path(nh_run_dir) / ACCEPTED_CONTINUATION_FILENAME).write_text(json.dumps(manifest, indent=2))


def test_no_manifest_preserves_block(run_pilot_fixture):
    """Absent a manifest, the 9->12 chunk must still be refused exactly as
    before -- adoption is strictly opt-in per run."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    assert load_accepted_continuation_manifest(nh_run_dir, _ACCEPTED_RUN_ID) == {}

    next_chunk = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert next_chunk["blocked"] is True
    assert "12" in next_chunk["blocked_reason"]


def test_correct_manifest_trusts_epoch_12(run_pilot_fixture):
    """A manifest with correct model+optimizer hashes for epoch 12 lets the
    9->12 chunk succeed by adopting the pre-existing overshoot checkpoint,
    never training."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(nh_run_dir, _ACCEPTED_RUN_ID, {"12": _entry_for_epoch(cont_dir, 12)})

    next_chunk = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert next_chunk["blocked"] is False
    assert os.path.samefile(next_chunk["checkpoint_dir_for_target"], cont_dir)
    assert train_calls == [], "adopting an accepted checkpoint must never trigger training"


def test_epoch_12_evaluated_without_training(run_pilot_fixture):
    """The adopted epoch 12 checkpoint is still screened/evaluated through
    the normal pipeline -- adoption skips training, not evaluation."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(nh_run_dir, _ACCEPTED_RUN_ID, {"12": _entry_for_epoch(cont_dir, 12)})

    next_chunk = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert train_calls == []
    assert [r["epoch"] for r in next_chunk["screening_results"]] == [12]
    assert [r.epoch for r in eval_calls] == [12]


def test_epoch_15_untouched_during_epoch_12_step(run_pilot_fixture):
    """Even when the manifest also carries an epoch-15 entry (with a
    deliberately WRONG hash), processing the 9->12 chunk must never consult
    or verify it -- only chunk_target_epoch's own entry is ever looked at."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(
        nh_run_dir,
        _ACCEPTED_RUN_ID,
        {
            "12": _entry_for_epoch(cont_dir, 12),
            "15": _entry_for_epoch(cont_dir, 15, model_sha256="deadbeef" * 8),
        },
    )

    next_chunk = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert next_chunk["blocked"] is False
    assert [r.epoch for r in eval_calls] == [12], "epoch 15 must never be evaluated while chunk 12 is due"


def test_incorrect_model_hash_rejected(run_pilot_fixture):
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(
        nh_run_dir, _ACCEPTED_RUN_ID,
        {"12": _entry_for_epoch(cont_dir, 12, model_sha256="0" * 64)},
    )

    with pytest.raises(PilotOrchestrationError, match="model hash mismatch"):
        run_pilot_chunk(
            chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
            previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
            train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
            run_id=_ACCEPTED_RUN_ID, **common_kwargs,
        )
    assert train_calls == []


def test_incorrect_optimizer_hash_rejected(run_pilot_fixture):
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(
        nh_run_dir, _ACCEPTED_RUN_ID,
        {"12": _entry_for_epoch(cont_dir, 12, optimizer_sha256="0" * 64)},
    )

    with pytest.raises(PilotOrchestrationError, match="optimizer hash mismatch"):
        run_pilot_chunk(
            chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
            previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
            train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
            run_id=_ACCEPTED_RUN_ID, **common_kwargs,
        )
    assert train_calls == []


def test_epoch_12_entry_pointing_to_epoch_15_files_rejected(run_pilot_fixture):
    """An entry keyed epoch 12 that points at correctly-hashed epoch-15
    files (same accepted directory) must be rejected at load time -- a
    hash match alone does not bind an entry to its own key epoch, so a
    different epoch's authenticated files could otherwise be silently
    substituted."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)

    _write_accepted_manifest(nh_run_dir, _ACCEPTED_RUN_ID, {"12": _entry_for_epoch(cont_dir, 15)})

    with pytest.raises(PilotOrchestrationError, match="model_epoch012.pt"):
        load_accepted_continuation_manifest(nh_run_dir, _ACCEPTED_RUN_ID)


def test_wrong_run_id_or_path_rejected(run_pilot_fixture):
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)

    _write_accepted_manifest(nh_run_dir, "some_other_run", {"12": _entry_for_epoch(cont_dir, 12)})
    with pytest.raises(PilotOrchestrationError, match="run_id"):
        load_accepted_continuation_manifest(nh_run_dir, _ACCEPTED_RUN_ID)

    escaping_entry = _entry_for_epoch(cont_dir, 12)
    escaping_entry["model_path"] = "../outside/model_epoch012.pt"
    _write_accepted_manifest(nh_run_dir, _ACCEPTED_RUN_ID, {"12": escaping_entry})
    with pytest.raises(PilotOrchestrationError):
        load_accepted_continuation_manifest(nh_run_dir, _ACCEPTED_RUN_ID)


def test_epoch_15_used_only_if_still_required(run_pilot_fixture):
    """When the pilot is NOT yet stopped after adopting epoch 12, a
    following 12->15 chunk correctly consults and adopts the manifest's
    epoch-15 entry in its own turn -- 'used only if still required', not
    pre-emptively during the epoch-12 step (see the epoch-15-untouched
    test above)."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(
        nh_run_dir, _ACCEPTED_RUN_ID,
        {"12": _entry_for_epoch(cont_dir, 12), "15": _entry_for_epoch(cont_dir, 15)},
    )

    chunk12 = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert chunk12["stopped"] is False, "fixture must not already be stopped, or chunk 15 would never be due"

    chunk15 = run_pilot_chunk(
        chunk_target_epoch=15, previous_target_epoch=12, is_first_chunk=False,
        previous_checkpoint_dir=chunk12["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert chunk15["blocked"] is False
    assert os.path.samefile(chunk15["checkpoint_dir_for_target"], cont_dir)
    assert train_calls == []
    assert [r.epoch for r in eval_calls] == [12, 15]


def test_stopping_at_12_leaves_15_unused(run_pilot_fixture):
    """If early stopping fires exactly at epoch 12, epoch 15's manifest
    entry must remain completely unconsulted -- verified here by giving it
    a hash that would fail verification if it were ever checked, and
    confirming the epoch-12 call still succeeds and stops without error.

    The pre-seeded early-stopping state below (events_since_best_improvement
    already at 2) is a deliberately constructed edge case: the frozen
    real policy's patience (3 events, cadence spacing 3 epochs) never
    naturally exhausts before epoch 15 starting from a flat metric at
    epoch 6 (see test_run_pilot_end_to_end_stops_on_patience_exhaustion).
    This isolates the orchestration-level guarantee -- 'once stopped,
    later accepted epochs stay unused' -- from early_stopping.py's own
    counting logic, which has its own dedicated test suite."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    es_state = json.loads((nh_run_dir / "pilot_early_stopping_state.json").read_text())
    es_state["events_since_best_improvement"] = 2
    (nh_run_dir / "pilot_early_stopping_state.json").write_text(json.dumps(es_state))

    _write_accepted_manifest(
        nh_run_dir, _ACCEPTED_RUN_ID,
        {
            "12": _entry_for_epoch(cont_dir, 12),
            "15": _entry_for_epoch(cont_dir, 15, model_sha256="deadbeef" * 8),
        },
    )

    chunk12 = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert chunk12["blocked"] is False
    assert chunk12["stopped"] is True
    assert chunk12["stop_reason"] == "patience_exhausted"
    assert [r.epoch for r in eval_calls] == [12], "epoch 15 must never be reached once stopped at epoch 12"


def test_rerun_idempotency_with_accepted_manifest(run_pilot_fixture):
    """Resuming the 9->12 chunk a second time, with the manifest still
    present, must not retrain, re-evaluate, or re-verify anything -- epoch
    12 is already logged as processed."""
    fx = run_pilot_fixture
    common_kwargs = _prepare_common_kwargs(fx)
    nh_run_dir, cont_dir = _build_overshoot_scenario_with_optimizers(fx, common_kwargs)
    train_calls, eval_calls, counting_train, counting_evaluate = _counting_wrappers(fx)
    resumed = _advance_to_epoch9(common_kwargs, counting_train, counting_evaluate)
    train_calls.clear()
    eval_calls.clear()

    _write_accepted_manifest(nh_run_dir, _ACCEPTED_RUN_ID, {"12": _entry_for_epoch(cont_dir, 12)})

    first = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert [r.epoch for r in eval_calls] == [12]

    second = run_pilot_chunk(
        chunk_target_epoch=12, previous_target_epoch=9, is_first_chunk=False,
        previous_checkpoint_dir=resumed["checkpoint_dir_for_target"],
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=counting_evaluate,
        run_id=_ACCEPTED_RUN_ID, **common_kwargs,
    )
    assert second["blocked"] is False
    assert os.path.samefile(second["checkpoint_dir_for_target"], cont_dir)
    assert train_calls == []
    assert [r.epoch for r in eval_calls] == [12], "second call must not re-evaluate epoch 12"


# --- prepare_pilot_run_only (--prepare-only): config-generation, no training,
# no W&B backend init (see scripts/run_stage1_lead06_pilot.py's --prepare-only
# flag and src.baseline.pilot_orchestration.prepare_pilot_run_only) ----------

def _prepare_only_kwargs(fx, **overrides):
    kwargs = dict(
        pilot_policy=fx["pilot_policy"],
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        preparation_out_dir=fx["evidence_out_dir"],
        commands_used=["test_pilot_orchestration.py --prepare-only"],
    )
    kwargs.update(overrides)
    return kwargs


def test_prepare_pilot_run_only_reports_prepared_only_and_writes_result_file(run_pilot_fixture):
    fx = run_pilot_fixture
    result = prepare_pilot_run_only(**_prepare_only_kwargs(fx))

    assert result["status"] == "PREPARED_ONLY"
    assert result["run_id"] == "raw_seedA"
    assert result["training_started"] is False
    assert result["evaluation_started"] is False
    assert result["wandb_backend_initialized"] is False
    assert Path(result["generated_config_path"]).is_file()
    assert Path(result["generation_manifest_path"]).is_file()

    result_path = fx["evidence_out_dir"] / PREPARATION_RESULT_FILENAME
    assert result_path.is_file()
    on_disk = json.loads(result_path.read_text())
    assert on_disk["status"] == "PREPARED_ONLY"
    assert on_disk["wandb_policy_sha256"] == result["wandb_policy_sha256"]

    # no NH run directory, checkpoint, or orchestration/early-stopping state
    # of any kind was created by this call.
    runs_root = fx["config_out_dir"] / "runs"
    assert not runs_root.is_dir() or list(runs_root.iterdir()) == []


def test_prepare_pilot_run_only_never_initializes_wandb_backend(monkeypatch, run_pilot_fixture):
    """The one real NH/W&B side-effecting call run_pilot() makes
    (init_pilot_tracking_run) must never be reached by --prepare-only --
    proven behaviorally, not just by reading the source, by making that
    function raise if it is ever called."""
    import src.baseline.pilot_orchestration as pilot_orchestration_module

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("prepare_pilot_run_only must never initialize a W&B/tracking backend")

    monkeypatch.setattr(pilot_orchestration_module, "init_pilot_tracking_run", _must_not_be_called)

    fx = run_pilot_fixture
    result = prepare_pilot_run_only(**_prepare_only_kwargs(fx))
    assert result["status"] == "PREPARED_ONLY"


def test_prepare_pilot_run_only_resolves_wandb_policy_override_checksum(tmp_path, run_pilot_fixture):
    fx = run_pilot_fixture
    override_policy_path = tmp_path / "override_wandb_policy.yaml"
    override_policy_path.write_text(
        "policy_name: test_override\n"
        "enabled: true\n"
        "mode: offline\n"
        "project: flashnh-stage1-test\n"
        "entity: null\n"
        "tags: [test]\n"
        "max_artifact_reference_bytes: 1048576\n",
        encoding="utf-8",
    )
    overridden_policy = dataclasses.replace(fx["pilot_policy"], wandb_policy_path=str(override_policy_path))

    result = prepare_pilot_run_only(**_prepare_only_kwargs(fx, pilot_policy=overridden_policy))

    assert result["wandb_policy_sha256"] == sha256_of(override_policy_path)
    assert result["run_identity"]["wandb_policy_sha256"] == sha256_of(override_policy_path)


def test_prepare_pilot_run_only_tracking_generation_defaults_to_g1(run_pilot_fixture):
    fx = run_pilot_fixture
    result = prepare_pilot_run_only(**_prepare_only_kwargs(fx))
    assert result["tracking_generation"] == "g1"
    assert result["run_identity"]["tracking_generation"] == "g1"


def test_prepare_pilot_run_only_explicit_tracking_generation_is_retained(run_pilot_fixture):
    fx = run_pilot_fixture
    result = prepare_pilot_run_only(**_prepare_only_kwargs(fx, tracking_generation="g2"))
    assert result["tracking_generation"] == "g2"
    assert result["run_identity"]["tracking_generation"] == "g2"


def test_prepare_pilot_run_only_fails_loudly_when_an_nh_run_directory_already_exists(run_pilot_fixture):
    """A single pre-existing NH run directory means training already
    started for this run_id -- --prepare-only must refuse (never describe
    an already-trained candidate as a clean preparation), even though this
    is the same single-match case discover_nh_run_dir would happily resolve
    for continuation purposes."""
    fx = run_pilot_fixture
    experiment_name = "stage1_lead06_pilot_raw_seedA_v001"
    # Realistic shape: config generation always precedes training, so an
    # already-trained candidate has both a generated config bundle AND a
    # runs/ subdirectory -- generate the former the ordinary way first.
    prepare_pilot_run(
        pilot_policy=fx["pilot_policy"], run_id="raw_seedA", baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"], splits_dir=SPLITS_DIR, config_out_dir=fx["config_out_dir"],
    )
    runs_root = fx["config_out_dir"] / "runs"
    (runs_root / f"{experiment_name}_20260101_000000").mkdir(parents=True)

    with pytest.raises(PilotOrchestrationError, match="already present"):
        prepare_pilot_run_only(**_prepare_only_kwargs(fx))


def test_prepare_pilot_run_only_fails_loudly_on_unexpected_preparation_out_dir_contents(run_pilot_fixture):
    """A preparation_out_dir already containing something other than a prior
    pilot_preparation_result.json (e.g. a real evidence bundle from an
    actual training invocation) must never be silently treated as a fresh,
    clean preparation target."""
    fx = run_pilot_fixture
    fx["evidence_out_dir"].mkdir(parents=True)
    (fx["evidence_out_dir"] / "pilot_run_evidence.json").write_text("{}")

    with pytest.raises(PilotOrchestrationError, match="unexpected"):
        prepare_pilot_run_only(**_prepare_only_kwargs(fx))


def test_prepare_pilot_run_only_is_restart_safe_across_repeated_calls(run_pilot_fixture):
    """Calling --prepare-only twice for a run_id that has only ever been
    prepared (never trained) must succeed both times, reusing the
    already-generated config rather than failing or silently overwriting
    it."""
    fx = run_pilot_fixture
    first = prepare_pilot_run_only(**_prepare_only_kwargs(fx))
    second = prepare_pilot_run_only(**_prepare_only_kwargs(fx))
    assert first["status"] == second["status"] == "PREPARED_ONLY"
    assert first["generated_config_path"] == second["generated_config_path"]


def test_prepare_pilot_run_only_generates_the_same_config_as_ordinary_preparation(run_pilot_fixture):
    """--prepare-only must not diverge from run_pilot()'s own first step in
    any scientific-setting-relevant way: the config.yaml/generation_manifest
    it writes must be byte-identical to calling prepare_pilot_run() directly
    with the same inputs."""
    fx = run_pilot_fixture
    direct_config_out_dir = fx["config_out_dir"].parent / "config_out_direct"
    prepare_pilot_run(
        pilot_policy=fx["pilot_policy"], run_id="raw_seedA", baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"], splits_dir=SPLITS_DIR, config_out_dir=direct_config_out_dir,
    )

    prepare_pilot_run_only(**_prepare_only_kwargs(fx))

    # Both configs embed their own out_dir's absolute path (e.g. in
    # train_basin_file/validation_basin_file) -- normalize that one expected
    # difference away before comparing everything else verbatim.
    direct_config_text = (direct_config_out_dir / "config.yaml").read_text()
    prepare_only_config_text = (fx["config_out_dir"] / "config.yaml").read_text()
    direct_normalized = direct_config_text.replace(str(direct_config_out_dir), "<CONFIG_OUT_DIR>")
    prepare_only_normalized = prepare_only_config_text.replace(str(fx["config_out_dir"]), "<CONFIG_OUT_DIR>")
    assert direct_normalized == prepare_only_normalized


# ---------------------------------------------------------------------------
# max_updates_per_epoch: cap-identity safeguard (enforce_pilot_cap_identity)
# and actual-optimizer-update evidence (read_actual_optimizer_updates /
# actual_optimizer_updates_by_epoch). Efficiency-feature support only -- see
# module docstring's task-7 note; none of this changes raw_seedA or any
# other existing candidate's identity/behavior.
# ---------------------------------------------------------------------------

def _cap_identity(*, pilot_policy_name="stage1_lead06_pilot_policy", run_id="raw_seedA", max_updates_per_epoch=None):
    return {
        "pilot_policy_name": pilot_policy_name,
        "run_id": run_id,
        "max_updates_per_epoch": max_updates_per_epoch,
    }


def test_enforce_pilot_cap_identity_noop_when_nh_run_dir_is_none():
    # must not raise, must not touch the filesystem
    enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=10), nh_run_dir=None)


def test_enforce_pilot_cap_identity_noop_when_nh_run_dir_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist_yet"
    enforce_pilot_cap_identity(run_identity=_cap_identity(), nh_run_dir=missing)
    assert not missing.exists()


def test_enforce_pilot_cap_identity_first_call_persists_record(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=10), nh_run_dir=nh_run_dir)

    state_path = nh_run_dir / CAP_IDENTITY_STATE_FILENAME
    assert state_path.is_file()
    record = json.loads(state_path.read_text())
    assert record == {
        "pilot_policy_name": "stage1_lead06_pilot_policy",
        "run_id": "raw_seedA",
        "max_updates_per_epoch": 10,
    }


def test_enforce_pilot_cap_identity_matching_repeat_call_succeeds(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    identity = _cap_identity(max_updates_per_epoch=None)
    enforce_pilot_cap_identity(run_identity=identity, nh_run_dir=nh_run_dir)
    # a second, identical call must be a no-op success, not a re-raise
    enforce_pilot_cap_identity(run_identity=identity, nh_run_dir=nh_run_dir)


def test_enforce_pilot_cap_identity_mismatched_cap_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=None), nh_run_dir=nh_run_dir)
    with pytest.raises(PilotOrchestrationError, match="cap identity"):
        enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=5), nh_run_dir=nh_run_dir)


def test_enforce_pilot_cap_identity_mismatched_run_id_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_cap_identity(run_identity=_cap_identity(run_id="raw_seedA"), nh_run_dir=nh_run_dir)
    with pytest.raises(PilotOrchestrationError, match="cap identity"):
        enforce_pilot_cap_identity(run_identity=_cap_identity(run_id="raw_seedB"), nh_run_dir=nh_run_dir)


def test_enforce_pilot_cap_identity_matching_repeat_call_succeeds_with_int_cap(tmp_path):
    # task item 6.2: a CAPPED trajectory (not just uncapped) continues with
    # the identical integer cap across repeated calls.
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    identity = _cap_identity(max_updates_per_epoch=25)
    enforce_pilot_cap_identity(run_identity=identity, nh_run_dir=nh_run_dir)
    enforce_pilot_cap_identity(run_identity=identity, nh_run_dir=nh_run_dir)


def test_enforce_pilot_cap_identity_int_to_null_raises(tmp_path):
    # task item 6.5: an integer-to-null change fails before training.
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=8), nh_run_dir=nh_run_dir)
    with pytest.raises(PilotOrchestrationError, match="cap identity"):
        enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=None), nh_run_dir=nh_run_dir)


def test_enforce_pilot_cap_identity_different_int_caps_raise(tmp_path):
    # task item 6.6: two different integer caps conflict.
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=5), nh_run_dir=nh_run_dir)
    with pytest.raises(PilotOrchestrationError, match="cap identity"):
        enforce_pilot_cap_identity(run_identity=_cap_identity(max_updates_per_epoch=10), nh_run_dir=nh_run_dir)


def test_prepare_pilot_run_only_records_declared_cap_without_training(run_pilot_fixture):
    # task item 6.14: preparation-only records the cap but performs no
    # training -- extends the existing uncapped-only preparation-only
    # coverage with a genuinely capped policy.
    fx = run_pilot_fixture
    capped_run_spec = dataclasses.replace(fx["pilot_policy"].runs["raw_seedA"], max_updates_per_epoch=20)
    capped_runs = dict(fx["pilot_policy"].runs)
    capped_runs["raw_seedA"] = capped_run_spec
    capped_policy = dataclasses.replace(fx["pilot_policy"], runs=capped_runs)

    result = prepare_pilot_run_only(**_prepare_only_kwargs(fx, pilot_policy=capped_policy))
    assert result["status"] == "PREPARED_ONLY"
    assert result["training_started"] is False
    assert result["run_identity"]["max_updates_per_epoch"] == 20
    # no NH run directory was ever created by preparation alone
    assert not (fx["config_out_dir"] / "runs").exists()


def test_run_pilot_enforces_cap_identity_across_calls_with_changed_cap(run_pilot_fixture):
    """A run_pilot() call against an NH run directory whose persisted cap
    identity was already recorded on an earlier call must be rejected if its
    freshly-resolved run_spec now declares a different max_updates_per_epoch
    -- capped vs uncapped (or two different int caps) must never silently
    change across a continuation of the same run directory.

    enforce_pilot_cap_identity() is a no-op until the NH run directory
    physically exists (see its docstring), so the very first-ever call for a
    brand-new candidate (which creates that directory mid-call) never has
    anything to persist against yet -- the record is only written starting
    the NEXT call, once the directory already exists. Three calls are
    therefore needed to exercise the contradiction: (1) creates the run
    directory (no-op, nothing persisted yet), (2) persists the uncapped
    identity now that the directory exists, (3) a changed cap contradicts
    it."""
    fx = run_pilot_fixture
    common_kwargs = dict(
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
    )
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 1 (uncapped, creates run dir)"], **common_kwargs)
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 2 (uncapped, persists cap identity)"], **common_kwargs)

    capped_run_spec = dataclasses.replace(fx["pilot_policy"].runs["raw_seedA"], max_updates_per_epoch=5)
    capped_runs = dict(fx["pilot_policy"].runs)
    capped_runs["raw_seedA"] = capped_run_spec
    capped_policy = dataclasses.replace(fx["pilot_policy"], runs=capped_runs)
    common_kwargs["pilot_policy"] = capped_policy

    with pytest.raises(PilotOrchestrationError, match="cap identity"):
        run_pilot(commands_used=["call 3 (capped) -- must be rejected"], **common_kwargs)


def _torch_save_optimizer_state(path, step):
    torch = pytest.importorskip("torch")
    param = torch.nn.Parameter(torch.zeros(1))
    state_dict = {"state": {0: {"step": torch.tensor(step)}}, "param_groups": []}
    torch.save(state_dict, path)


def test_read_actual_optimizer_updates_reads_real_torch_step_counter(tmp_path):
    pytest.importorskip("torch")
    path = tmp_path / "optimizer_state_epoch003.pt"
    _torch_save_optimizer_state(path, step=117)
    assert read_actual_optimizer_updates(path) == 117


def test_read_actual_optimizer_updates_rejects_missing_file(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(PilotOrchestrationError, match="not found"):
        read_actual_optimizer_updates(tmp_path / "optimizer_state_epoch003.pt")


def test_read_actual_optimizer_updates_rejects_empty_state(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "optimizer_state_epoch003.pt"
    torch.save({"state": {}, "param_groups": []}, path)
    with pytest.raises(PilotOrchestrationError, match="no per-parameter state"):
        read_actual_optimizer_updates(path)


def test_read_actual_optimizer_updates_rejects_disagreeing_steps(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "optimizer_state_epoch003.pt"
    state_dict = {
        "state": {0: {"step": torch.tensor(10)}, 1: {"step": torch.tensor(20)}},
        "param_groups": [],
    }
    torch.save(state_dict, path)
    with pytest.raises(PilotOrchestrationError, match="disagreeing"):
        read_actual_optimizer_updates(path)


def test_actual_optimizer_updates_by_epoch_reads_base_and_continuation_directories(tmp_path):
    pytest.importorskip("torch")
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    for epoch, step in ((1, 100), (2, 200)):
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        _torch_save_optimizer_state(nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt", step=step)

    continuation_dir = nh_run_dir / "continue_training_from_epoch002"
    continuation_dir.mkdir()
    (continuation_dir / "model_epoch003.pt").write_bytes(b"ckpt3")
    _torch_save_optimizer_state(continuation_dir / "optimizer_state_epoch003.pt", step=300)

    assert actual_optimizer_updates_by_epoch(nh_run_dir) == {1: 100, 2: 200, 3: 300}


# ---------------------------------------------------------------------------
# learning_rate: LR-A range-characterization campaign's resume-contradiction
# safeguard (enforce_pilot_learning_rate_identity), mirroring
# enforce_pilot_cap_identity's own test block above field-for-field (see
# docs/decision_log.md's LR-A design-freeze entry and this function's own
# docstring). Pre-commit review found this guard implemented but untested;
# this block closes that gap. No implementation change.
# ---------------------------------------------------------------------------

def _lr_identity(
    *, pilot_policy_name="stage1_lead06_pilot_policy", run_id="raw_seedA", resolved_learning_rate=0.001
):
    return {
        "pilot_policy_name": pilot_policy_name,
        "run_id": run_id,
        "resolved_learning_rate": resolved_learning_rate,
    }


def test_enforce_pilot_learning_rate_identity_noop_when_nh_run_dir_is_none():
    # must not raise, must not touch the filesystem
    enforce_pilot_learning_rate_identity(run_identity=_lr_identity(), nh_run_dir=None)


def test_enforce_pilot_learning_rate_identity_noop_when_nh_run_dir_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist_yet"
    enforce_pilot_learning_rate_identity(run_identity=_lr_identity(), nh_run_dir=missing)
    assert not missing.exists()


def test_enforce_pilot_learning_rate_identity_first_call_persists_record(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_learning_rate_identity(
        run_identity=_lr_identity(resolved_learning_rate=0.0003), nh_run_dir=nh_run_dir
    )

    state_path = nh_run_dir / LR_IDENTITY_STATE_FILENAME
    assert state_path.is_file()
    record = json.loads(state_path.read_text())
    assert record == {
        "pilot_policy_name": "stage1_lead06_pilot_policy",
        "run_id": "raw_seedA",
        "resolved_learning_rate": 0.0003,
    }


def test_enforce_pilot_learning_rate_identity_matching_repeat_call_succeeds(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    identity = _lr_identity(resolved_learning_rate=0.001)
    enforce_pilot_learning_rate_identity(run_identity=identity, nh_run_dir=nh_run_dir)
    # a second, identical call must be a no-op success, not a re-raise
    enforce_pilot_learning_rate_identity(run_identity=identity, nh_run_dir=nh_run_dir)


def test_enforce_pilot_learning_rate_identity_mismatched_lr_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_learning_rate_identity(
        run_identity=_lr_identity(resolved_learning_rate=0.001), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="learning-rate identity"):
        enforce_pilot_learning_rate_identity(
            run_identity=_lr_identity(resolved_learning_rate=0.003), nh_run_dir=nh_run_dir
        )


def test_enforce_pilot_learning_rate_identity_mismatched_run_id_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_learning_rate_identity(
        run_identity=_lr_identity(run_id="emb128x32_seedA_lr1em4_cap25k_cal"), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="learning-rate identity"):
        enforce_pilot_learning_rate_identity(
            run_identity=_lr_identity(run_id="emb128x32_seedA_lr3em4_cap25k_cal"), nh_run_dir=nh_run_dir
        )


def test_enforce_pilot_learning_rate_identity_mismatched_policy_name_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_learning_rate_identity(
        run_identity=_lr_identity(pilot_policy_name="lr_range_seedA_25k_v001"), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="learning-rate identity"):
        enforce_pilot_learning_rate_identity(
            run_identity=_lr_identity(pilot_policy_name="stage1_lead06_pilot_policy"), nh_run_dir=nh_run_dir
        )


def test_enforce_pilot_learning_rate_identity_unset_override_and_equal_explicit_override_are_same_identity(
    tmp_path,
):
    """Per this function's own docstring: it compares only
    run_identity["resolved_learning_rate"], never learning_rate_override --
    an unset override that resolves to a profile's own learning_rate, and an
    explicit override that resolves to that identical value, are the same
    training identity and must not conflict. Both calls below use
    resolved_learning_rate=0.001 -- one standing in for "no override, profile
    default is 0.001" and the other for "explicit override=0.001" -- the
    guard cannot and must not distinguish them."""
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    unset_override_identity = _lr_identity(resolved_learning_rate=0.001)
    explicit_override_identity = _lr_identity(resolved_learning_rate=0.001)
    enforce_pilot_learning_rate_identity(run_identity=unset_override_identity, nh_run_dir=nh_run_dir)
    enforce_pilot_learning_rate_identity(run_identity=explicit_override_identity, nh_run_dir=nh_run_dir)


def test_run_pilot_enforces_lr_identity_across_calls_with_changed_lr(run_pilot_fixture):
    """LR-A analogue of test_run_pilot_enforces_cap_identity_across_calls_with_changed_cap:
    a run_pilot() call against an NH run directory whose persisted LR
    identity was already recorded on an earlier call must be rejected if its
    freshly-resolved run_spec now declares a different learning_rate. Same
    three-call shape as the cap-identity analogue: (1) creates the run
    directory (enforce_pilot_learning_rate_identity is a no-op, nothing
    persisted yet), (2) persists the run_spec's resolved learning rate now
    that the directory exists, (3) a changed learning_rate contradicts it."""
    fx = run_pilot_fixture
    common_kwargs = dict(
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
    )
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 1 (creates run dir)"], **common_kwargs)
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 2 (persists lr identity)"], **common_kwargs)

    changed_lr_run_spec = dataclasses.replace(fx["pilot_policy"].runs["raw_seedA"], learning_rate=0.0005)
    changed_lr_runs = dict(fx["pilot_policy"].runs)
    changed_lr_runs["raw_seedA"] = changed_lr_run_spec
    changed_lr_policy = dataclasses.replace(fx["pilot_policy"], runs=changed_lr_runs)
    common_kwargs["pilot_policy"] = changed_lr_policy

    with pytest.raises(PilotOrchestrationError, match="learning-rate identity"):
        run_pilot(commands_used=["call 3 (changed learning_rate) -- must be rejected"], **common_kwargs)


# ---------------------------------------------------------------------------
# hidden_size: Hidden-size-A range-characterization campaign's resume-
# contradiction safeguard (enforce_pilot_hidden_size_identity), mirroring
# enforce_pilot_learning_rate_identity's own test block above field-for-field
# (see docs/decision_log.md's 2026-08-09 Hidden-size-A design-freeze entry
# and this function's own docstring).
# ---------------------------------------------------------------------------

def _hidden_size_identity(
    *, pilot_policy_name="stage1_lead06_pilot_policy", run_id="raw_seedA", resolved_hidden_size=128
):
    return {
        "pilot_policy_name": pilot_policy_name,
        "run_id": run_id,
        "resolved_hidden_size": resolved_hidden_size,
    }


def test_enforce_pilot_hidden_size_identity_noop_when_nh_run_dir_is_none():
    # must not raise, must not touch the filesystem
    enforce_pilot_hidden_size_identity(run_identity=_hidden_size_identity(), nh_run_dir=None)


def test_enforce_pilot_hidden_size_identity_noop_when_nh_run_dir_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist_yet"
    enforce_pilot_hidden_size_identity(run_identity=_hidden_size_identity(), nh_run_dir=missing)
    assert not missing.exists()


def test_enforce_pilot_hidden_size_identity_first_call_persists_record(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_hidden_size_identity(
        run_identity=_hidden_size_identity(resolved_hidden_size=64), nh_run_dir=nh_run_dir
    )

    state_path = nh_run_dir / HIDDEN_SIZE_IDENTITY_STATE_FILENAME
    assert state_path.is_file()
    record = json.loads(state_path.read_text())
    assert record == {
        "pilot_policy_name": "stage1_lead06_pilot_policy",
        "run_id": "raw_seedA",
        "resolved_hidden_size": 64,
    }


def test_enforce_pilot_hidden_size_identity_matching_repeat_call_succeeds(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    identity = _hidden_size_identity(resolved_hidden_size=128)
    enforce_pilot_hidden_size_identity(run_identity=identity, nh_run_dir=nh_run_dir)
    # a second, identical call must be a no-op success, not a re-raise
    enforce_pilot_hidden_size_identity(run_identity=identity, nh_run_dir=nh_run_dir)


def test_enforce_pilot_hidden_size_identity_mismatched_hidden_size_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_hidden_size_identity(
        run_identity=_hidden_size_identity(resolved_hidden_size=128), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="hidden-size identity"):
        enforce_pilot_hidden_size_identity(
            run_identity=_hidden_size_identity(resolved_hidden_size=256), nh_run_dir=nh_run_dir
        )


def test_enforce_pilot_hidden_size_identity_mismatched_run_id_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_hidden_size_identity(
        run_identity=_hidden_size_identity(run_id="emb128x32_seedA_h64_lr3em4_cap25k_cal"), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="hidden-size identity"):
        enforce_pilot_hidden_size_identity(
            run_identity=_hidden_size_identity(run_id="emb128x32_seedA_h128_lr3em4_cap25k_cal"),
            nh_run_dir=nh_run_dir,
        )


def test_enforce_pilot_hidden_size_identity_mismatched_policy_name_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_hidden_size_identity(
        run_identity=_hidden_size_identity(pilot_policy_name="hidden_size_range_seedA_25k_v001"),
        nh_run_dir=nh_run_dir,
    )
    with pytest.raises(PilotOrchestrationError, match="hidden-size identity"):
        enforce_pilot_hidden_size_identity(
            run_identity=_hidden_size_identity(pilot_policy_name="stage1_lead06_pilot_policy"),
            nh_run_dir=nh_run_dir,
        )


def test_enforce_pilot_hidden_size_identity_unset_override_and_equal_explicit_override_are_same_identity(
    tmp_path,
):
    """Per this function's own docstring: it compares only
    run_identity["resolved_hidden_size"], never hidden_size_override -- an
    unset override that resolves to a profile's own hidden_size, and an
    explicit override that resolves to that identical value, are the same
    training identity and must not conflict. Both calls below use
    resolved_hidden_size=128 -- one standing in for "no override, profile
    default is 128" and the other for "explicit override=128" -- the guard
    cannot and must not distinguish them."""
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    unset_override_identity = _hidden_size_identity(resolved_hidden_size=128)
    explicit_override_identity = _hidden_size_identity(resolved_hidden_size=128)
    enforce_pilot_hidden_size_identity(run_identity=unset_override_identity, nh_run_dir=nh_run_dir)
    enforce_pilot_hidden_size_identity(run_identity=explicit_override_identity, nh_run_dir=nh_run_dir)


def test_run_pilot_enforces_hidden_size_identity_across_calls_with_changed_hidden_size(run_pilot_fixture):
    """Hidden-size-A analogue of test_run_pilot_enforces_lr_identity_across_calls_with_changed_lr:
    a run_pilot() call against an NH run directory whose persisted hidden-
    size identity was already recorded on an earlier call must be rejected
    if its freshly-resolved run_spec now declares a different hidden_size.
    Same three-call shape: (1) creates the run directory
    (enforce_pilot_hidden_size_identity is a no-op, nothing persisted yet),
    (2) persists the run_spec's resolved hidden size now that the directory
    exists, (3) a changed hidden_size contradicts it."""
    fx = run_pilot_fixture
    common_kwargs = dict(
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
    )
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 1 (creates run dir)"], **common_kwargs)
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 2 (persists hidden-size identity)"], **common_kwargs)

    changed_hidden_size_run_spec = dataclasses.replace(fx["pilot_policy"].runs["raw_seedA"], hidden_size=256)
    changed_hidden_size_runs = dict(fx["pilot_policy"].runs)
    changed_hidden_size_runs["raw_seedA"] = changed_hidden_size_run_spec
    changed_hidden_size_policy = dataclasses.replace(fx["pilot_policy"], runs=changed_hidden_size_runs)
    common_kwargs["pilot_policy"] = changed_hidden_size_policy

    with pytest.raises(PilotOrchestrationError, match="hidden-size identity"):
        run_pilot(commands_used=["call 3 (changed hidden_size) -- must be rejected"], **common_kwargs)


# ---------------------------------------------------------------------------
# embedding_dropout: Embedding-Dropout-A range-characterization campaign's
# resume-contradiction safeguard (enforce_pilot_embedding_dropout_identity),
# mirroring enforce_pilot_hidden_size_identity's own test block above
# field-for-field (see docs/decision_log.md's Embedding-Dropout-A
# design-freeze entry and this function's own docstring). The
# resolved_embedding_dropout=0.0 cases below exist specifically to prove the
# guard never confuses an explicit 0.0 with an unset/None value (a plain
# dict-equality comparison, never a truthiness check).
# ---------------------------------------------------------------------------

def _embedding_dropout_identity(
    *, pilot_policy_name="stage1_lead06_pilot_policy", run_id="raw_seedA", resolved_embedding_dropout=0.10
):
    return {
        "pilot_policy_name": pilot_policy_name,
        "run_id": run_id,
        "resolved_embedding_dropout": resolved_embedding_dropout,
    }


def test_enforce_pilot_embedding_dropout_identity_noop_when_nh_run_dir_is_none():
    # must not raise, must not touch the filesystem
    enforce_pilot_embedding_dropout_identity(run_identity=_embedding_dropout_identity(), nh_run_dir=None)


def test_enforce_pilot_embedding_dropout_identity_noop_when_nh_run_dir_does_not_exist(tmp_path):
    missing = tmp_path / "does_not_exist_yet"
    enforce_pilot_embedding_dropout_identity(run_identity=_embedding_dropout_identity(), nh_run_dir=missing)
    assert not missing.exists()


def test_enforce_pilot_embedding_dropout_identity_first_call_persists_record(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_embedding_dropout_identity(
        run_identity=_embedding_dropout_identity(resolved_embedding_dropout=0.05), nh_run_dir=nh_run_dir
    )

    state_path = nh_run_dir / EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME
    assert state_path.is_file()
    record = json.loads(state_path.read_text())
    assert record == {
        "pilot_policy_name": "stage1_lead06_pilot_policy",
        "run_id": "raw_seedA",
        "resolved_embedding_dropout": 0.05,
    }


def test_enforce_pilot_embedding_dropout_identity_first_call_persists_explicit_zero_not_none(tmp_path):
    """The drop00 candidate's resolved_embedding_dropout=0.0 must be
    persisted and compared as the real float 0.0, never dropped/omitted as
    if it were an unset/None value."""
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_embedding_dropout_identity(
        run_identity=_embedding_dropout_identity(resolved_embedding_dropout=0.0), nh_run_dir=nh_run_dir
    )

    state_path = nh_run_dir / EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME
    assert state_path.is_file()
    record = json.loads(state_path.read_text())
    assert record["resolved_embedding_dropout"] == 0.0
    assert record["resolved_embedding_dropout"] is not None


def test_enforce_pilot_embedding_dropout_identity_matching_repeat_call_succeeds(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    identity = _embedding_dropout_identity(resolved_embedding_dropout=0.20)
    enforce_pilot_embedding_dropout_identity(run_identity=identity, nh_run_dir=nh_run_dir)
    # a second, identical call must be a no-op success, not a re-raise
    enforce_pilot_embedding_dropout_identity(run_identity=identity, nh_run_dir=nh_run_dir)


def test_enforce_pilot_embedding_dropout_identity_mismatched_dropout_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_embedding_dropout_identity(
        run_identity=_embedding_dropout_identity(resolved_embedding_dropout=0.10), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="embedding-dropout identity"):
        enforce_pilot_embedding_dropout_identity(
            run_identity=_embedding_dropout_identity(resolved_embedding_dropout=0.40), nh_run_dir=nh_run_dir
        )


def test_enforce_pilot_embedding_dropout_identity_zero_vs_nonzero_mismatch_raises(tmp_path):
    """The 0.0-vs-nonzero pairing specifically: 0.0 must be treated as a
    real, distinct value, not silently equal to any other resolved dropout
    via a truthiness bug."""
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_embedding_dropout_identity(
        run_identity=_embedding_dropout_identity(resolved_embedding_dropout=0.0), nh_run_dir=nh_run_dir
    )
    with pytest.raises(PilotOrchestrationError, match="embedding-dropout identity"):
        enforce_pilot_embedding_dropout_identity(
            run_identity=_embedding_dropout_identity(resolved_embedding_dropout=0.05), nh_run_dir=nh_run_dir
        )


def test_enforce_pilot_embedding_dropout_identity_mismatched_run_id_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_embedding_dropout_identity(
        run_identity=_embedding_dropout_identity(run_id="emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal"),
        nh_run_dir=nh_run_dir,
    )
    with pytest.raises(PilotOrchestrationError, match="embedding-dropout identity"):
        enforce_pilot_embedding_dropout_identity(
            run_identity=_embedding_dropout_identity(run_id="emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal"),
            nh_run_dir=nh_run_dir,
        )


def test_enforce_pilot_embedding_dropout_identity_mismatched_policy_name_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_embedding_dropout_identity(
        run_identity=_embedding_dropout_identity(pilot_policy_name="embedding_dropout_range_seedA_25k_v001"),
        nh_run_dir=nh_run_dir,
    )
    with pytest.raises(PilotOrchestrationError, match="embedding-dropout identity"):
        enforce_pilot_embedding_dropout_identity(
            run_identity=_embedding_dropout_identity(pilot_policy_name="stage1_lead06_pilot_policy"),
            nh_run_dir=nh_run_dir,
        )


def test_enforce_pilot_embedding_dropout_identity_unset_override_and_equal_explicit_override_are_same_identity(
    tmp_path,
):
    """Per this function's own docstring: it compares only
    run_identity["resolved_embedding_dropout"], never
    embedding_dropout_override -- an unset override that resolves to a
    profile's own embedding_dropout, and an explicit override that resolves
    to that identical value, are the same training identity and must not
    conflict."""
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    unset_override_identity = _embedding_dropout_identity(resolved_embedding_dropout=0.10)
    explicit_override_identity = _embedding_dropout_identity(resolved_embedding_dropout=0.10)
    enforce_pilot_embedding_dropout_identity(run_identity=unset_override_identity, nh_run_dir=nh_run_dir)
    enforce_pilot_embedding_dropout_identity(run_identity=explicit_override_identity, nh_run_dir=nh_run_dir)


def test_run_pilot_enforces_embedding_dropout_identity_across_calls_with_changed_embedding_dropout(
    run_pilot_fixture,
):
    """Embedding-Dropout-A analogue of
    test_run_pilot_enforces_hidden_size_identity_across_calls_with_changed_hidden_size:
    a run_pilot() call against an NH run directory whose persisted
    embedding-dropout identity was already recorded on an earlier call must
    be rejected if its freshly-resolved run_spec now declares a different
    embedding_dropout. Same three-call shape: (1) creates the run directory
    (enforce_pilot_embedding_dropout_identity is a no-op, nothing persisted
    yet), (2) persists the run_spec's resolved embedding dropout now that
    the directory exists, (3) a changed embedding_dropout contradicts it.
    Uses run_id="emb128x64_seedA" rather than the other identity-guard
    tests' "raw_seedA" -- embedding_dropout only applies to the
    learned-static embedding pathway (raw_seedA's profile has no
    statics_embedding section to override). run_pilot_fixture's shared
    fake_train/fake_evaluate are closed over "raw_seedA"'s experiment_name,
    so this test builds its own pair bound to emb128x64_seedA's instead."""
    fx = run_pilot_fixture
    experiment_name = "stage1_lead06_pilot_emb128x64_seedA_v001"
    fake_train = _make_fake_train_chunk_fn(fx["package_root"], fx["basins"], experiment_name)
    fake_evaluate = _make_fake_evaluate_checkpoint_fn(fx["package_root"], fx["basins"])
    common_kwargs = dict(
        run_id="emb128x64_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=fx["package_root"],
        splits_dir=SPLITS_DIR,
        config_out_dir=fx["config_out_dir"],
        evidence_out_dir=fx["evidence_out_dir"],
        screening_basin_ids=fx["basins"],
        train_chunk_fn=fake_train,
        evaluate_checkpoint_fn=fake_evaluate,
    )
    run_pilot(pilot_policy=fx["pilot_policy"], commands_used=["call 1 (creates run dir)"], **common_kwargs)
    run_pilot(
        pilot_policy=fx["pilot_policy"],
        commands_used=["call 2 (persists embedding-dropout identity)"],
        **common_kwargs,
    )

    changed_dropout_run_spec = dataclasses.replace(
        fx["pilot_policy"].runs["emb128x64_seedA"], embedding_dropout=0.30
    )
    changed_dropout_runs = dict(fx["pilot_policy"].runs)
    changed_dropout_runs["emb128x64_seedA"] = changed_dropout_run_spec
    changed_dropout_policy = dataclasses.replace(fx["pilot_policy"], runs=changed_dropout_runs)
    common_kwargs["pilot_policy"] = changed_dropout_policy

    with pytest.raises(PilotOrchestrationError, match="embedding-dropout identity"):
        run_pilot(commands_used=["call 3 (changed embedding_dropout) -- must be rejected"], **common_kwargs)


# ---------------------------------------------------------------------------
# require_tracking: Hidden-size-A campaign's strict W&B launch contract --
# run_pilot(require_tracking=True) must thread through to
# init_pilot_tracking_run and hard-fail rather than silently downgrading to
# an untracked null run, while the default (False) preserves every existing
# caller's untracked-fallback behavior (see
# pilot_tracking.init_pilot_tracking_run's require_tracking parameter and
# docs/decision_log.md's 2026-08-09 Hidden-size-A design-freeze entry).
# ---------------------------------------------------------------------------

def test_run_pilot_default_require_tracking_false_preserves_null_fallback(run_pilot_fixture):
    """The disabled-by-default committed W&B policy resolves to backend
    'null' -- with the default require_tracking=False, run_pilot() must
    complete normally rather than raising, exactly as it always has."""
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
        train_chunk_fn=fx["fake_train"],
        evaluate_checkpoint_fn=fx["fake_evaluate"],
        commands_used=["require_tracking default False, disabled policy"],
    )
    assert result["nh_run_dir"] is not None


def test_run_pilot_require_tracking_true_raises_when_policy_disabled(run_pilot_fixture):
    """With require_tracking=True, run_pilot() must fail fast (via
    init_pilot_tracking_run's policy_active guard) when the effective W&B
    policy is disabled, rather than silently training untracked."""
    from src.baseline.wandb_tracking import TrackingError

    fx = run_pilot_fixture
    with pytest.raises(TrackingError):
        run_pilot(
            pilot_policy=fx["pilot_policy"],
            run_id="raw_seedA",
            baseline_policy_path=BASELINE_POLICY_PATH,
            package_root=fx["package_root"],
            splits_dir=SPLITS_DIR,
            config_out_dir=fx["config_out_dir"],
            evidence_out_dir=fx["evidence_out_dir"],
            screening_basin_ids=fx["basins"],
            train_chunk_fn=fx["fake_train"],
            evaluate_checkpoint_fn=fx["fake_evaluate"],
            commands_used=["require_tracking True, disabled policy -- must raise"],
            require_tracking=True,
        )
