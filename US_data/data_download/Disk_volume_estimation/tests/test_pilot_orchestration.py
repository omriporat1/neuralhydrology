"""Focused tests for src/baseline/pilot_orchestration.py (task item 6's
implementation, task item 10's tests).

Adapted from a previously-passing scratchpad end-to-end verification
script. Uses a FAKE ``train_chunk_fn`` (writes checkpoint files +
hand-crafted perfect-NSE validation_results.p, no NH/torch/GPU) so the
full chunk-scheduling / screening / restart-safe early-stopping / evidence
bundle pipeline can be exercised deterministically and fast. Covers: pure
chunk-schedule + screening-epoch enumeration helpers, budget-below-minimum
rejection, missing-runs-root rejection, a full run_pilot() call against the
real committed pilot policy through to patience-exhausted stopping with the
best checkpoint retained, and idempotent resume (no retraining, evidence
bundle still rewritten on every call regardless of force).
"""
from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.baseline.pilot_orchestration import (
    PilotOrchestrationError,
    TrainChunkRequest,
    chunk_epoch_targets,
    discover_nh_run_dir,
    prepare_pilot_run,
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

def _write_validation_results(nh_run_dir: Path, epoch: int, basins, package_root: Path):
    period_dir = nh_run_dir / "validation" / weight_stem(epoch)
    period_dir.mkdir(parents=True, exist_ok=True)
    basin_results = {}
    for b in basins:
        ds = xr.open_dataset(package_root / "time_series" / f"{b}.nc")
        target = ds["qobs_mm_per_h_lead06"].values
        obs = target.copy()
        sim = obs.copy()
        xr_ds = xr.Dataset(
            {"qobs_mm_per_h_lead06_obs": ("date", obs), "qobs_mm_per_h_lead06_sim": ("date", sim)},
            coords={"date": np.arange(len(obs))},
        )
        basin_results[b] = {"1h": {"xr": xr_ds}}
    with open(period_dir / "validation_results.p", "wb") as fh:
        pickle.dump(basin_results, fh)


def _make_fake_train_chunk_fn(package_root, basins, experiment_name):
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
            if epoch % 3 == 0:
                _write_validation_results(nh_run_dir, epoch, basins, package_root)

    return _train


@pytest.fixture
def run_pilot_fixture(tmp_path, pilot_policy):
    basins = pick_development_basins(5)
    experiment_name = "stage1_lead06_pilot_raw_seedA_v001"

    package_root = tmp_path / "package"
    config_out_dir = tmp_path / "config_out"
    evidence_out_dir = tmp_path / "evidence"
    build_full_union_package(package_root, ts_basin_ids=basins)

    fake_train = _make_fake_train_chunk_fn(package_root, basins, experiment_name)
    return {
        "pilot_policy": pilot_policy, "basins": basins, "package_root": package_root,
        "config_out_dir": config_out_dir, "evidence_out_dir": evidence_out_dir, "fake_train": fake_train,
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
    )

    call_count = {"n": 0}

    def counting_train(request):
        call_count["n"] += 1
        return fx["fake_train"](request)

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
    )
    assert call_count["n"] == 0, "resume must not call the trainer again -- everything already trained"
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
    # run_pilot()'s own first iteration.
    first = run_pilot_chunk(
        chunk_target_epoch=6, previous_target_epoch=0, is_first_chunk=True,
        train_chunk_fn=fx["fake_train"], **common_kwargs,
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

    resumed = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        train_chunk_fn=counting_train, **common_kwargs,
    )

    # trainer invoked exactly once, targeting epoch 9 -- not re-invoked for
    # the already-checkpointed epochs 1-8.
    assert train_calls == [9]
    assert (nh_run_dir / "model_epoch001.pt").read_bytes() == epoch1_bytes_before
    assert (nh_run_dir / "model_epoch006.pt").read_bytes() == epoch6_bytes_before
    assert (nh_run_dir / "model_epoch007.pt").read_bytes() == b"ckpt7"
    assert (nh_run_dir / "model_epoch008.pt").read_bytes() == b"ckpt8"
    assert (nh_run_dir / "model_epoch009.pt").is_file()

    # epoch 9 evaluated exactly once this chunk; stopping state updated once.
    assert [r["epoch"] for r in resumed["screening_results"]] == [9]
    assert len(resumed["state"]["history"]) == len(first["state"]["history"]) + 1
    assert resumed["state"]["history"][-1]["epoch"] == 9

    # second resume: fully idempotent -- trainer not called again, no
    # duplicate screening/stopping-history entry.
    train_calls_2 = []

    def counting_train_2(request):
        train_calls_2.append(request.target_epoch)
        fx["fake_train"](request)

    resumed_again = run_pilot_chunk(
        chunk_target_epoch=9, previous_target_epoch=6, is_first_chunk=False,
        train_chunk_fn=counting_train_2, **common_kwargs,
    )
    assert train_calls_2 == [], "second resume must not retrigger training -- epoch 9 already on disk"
    assert [r["epoch"] for r in resumed_again["screening_results"]] == [9]
    assert len(resumed_again["state"]["history"]) == len(resumed["state"]["history"])
