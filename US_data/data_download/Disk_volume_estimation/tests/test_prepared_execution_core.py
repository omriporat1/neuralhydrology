"""Focused tests for the generic prepared-execution result contract
(:class:`src.baseline.pilot_orchestration.PreparedPilotExecutionResult` and
:func:`src.baseline.pilot_orchestration.execute_prepared_pilot_run`).

Kept deliberately separate from ``tests/test_pilot_orchestration.py``'s
broad end-to-end coverage: this module only proves the extracted
scheduling/composition core's own contract -- config immutability, the
typed result's generic facts, the resume screening-history reconstruction
fix, blocked-execution honesty, and a vertical consumer-contract test (see
``docs/agent_handoff_rules.md`` section 5) -- never campaign-specific
concepts. Small helpers below (fake ``train_chunk_fn``/``evaluate_checkpoint_fn``
closures) are intentionally duplicated from ``test_pilot_orchestration.py``
rather than imported, since those are private test helpers and this module's
edit scope is self-contained.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

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


# --- Test A: pure scheduling/composition, config immutability --------------

def test_prepared_core_never_prepares_or_rewrites_source_config_and_supports_every_epoch(tmp_path, monkeypatch):
    """The extracted core is scheduling/composition only: it consumes a written config."""
    config_dir = tmp_path / "prepared"; config_dir.mkdir()
    config = config_dir / "config.yaml"; config.write_bytes(b"frozen: true\n")
    before = config.read_bytes(); before_sha = hashlib.sha256(before).hexdigest()
    calls = []
    policy = SimpleNamespace(lead_hours=6)
    nh_run_dir = tmp_path / "nh"
    monkeypatch.setattr(orchestration, "build_effective_policy", lambda _: {"max_epoch_budget": 12})
    monkeypatch.setattr(orchestration, "chunk_epoch_targets", lambda *_: list(range(1, 13)))
    monkeypatch.setattr(orchestration, "_try_discover_nh_run_dir", lambda *_: None)
    monkeypatch.setattr(orchestration, "prepare_pilot_run", lambda *_, **__: (_ for _ in ()).throw(AssertionError("must not prepare")))

    def fake_chunk(**kwargs):
        epoch = kwargs["chunk_target_epoch"]; calls.append(epoch)
        return {"blocked": False, "stopped": False, "stop_reason": None,
                "screening_results": [{"epoch": epoch}], "checkpoint_dir_for_target": nh_run_dir,
                "nh_run_dir": nh_run_dir, "state": {"stopped": False}}
    monkeypatch.setattr(orchestration, "run_pilot_chunk", fake_chunk)

    # The reconstruction step at the end of execute_prepared_pilot_run reads
    # durable state / the physical inventory / the mature screening helper --
    # monkeypatch each so this pure-scheduling test never touches real disk.
    monkeypatch.setattr(orchestration, "logged_screening_epochs", lambda *_: list(range(1, 13)))
    fake_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir
        )
        for epoch in range(1, 13)
    }
    monkeypatch.setattr(orchestration, "discover_physical_checkpoints", lambda *_: fake_inventory)
    monkeypatch.setattr(
        orchestration, "evaluate_screening_checkpoint",
        lambda **kwargs: {"scope": "screening_subset_provisional", "epoch": kwargs["epoch"]},
    )

    result = orchestration.execute_prepared_pilot_run(
        execution_policy=policy, config_dir=config_dir, experiment_name="already_written",
        package_root=tmp_path, target_variable="qobs", lead_hours=6, screening_basin_ids=["x"], run_id="prepared",
    )
    assert calls == list(range(1, 13))
    assert isinstance(result, orchestration.PreparedPilotExecutionResult)
    assert [row["epoch"] for row in result.screening_events] == list(range(1, 13))
    assert config.read_bytes() == before
    assert hashlib.sha256(config.read_bytes()).hexdigest() == before_sha


# --- shared real (non-monkeypatched) fixture for Tests B-E ------------------

def _fake_train_chunk_fn(experiment_name):
    """Writes checkpoint files ONLY -- reproduces NH's real physical
    continuation layout (start_run flat; continue_run always nests into
    continue_training_from_epoch###/). Duplicated from
    test_pilot_orchestration.py's private helper of the same shape rather
    than imported -- see module docstring."""

    def _train(request: "orchestration.TrainChunkRequest") -> None:
        if request.is_first_chunk:
            runs_root = request.config_path.parent / "runs"
            nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
            nh_run_dir.mkdir(parents=True)
            target_dir = nh_run_dir
            start_epoch = 0
        else:
            base_dir = Path(request.nh_run_dir)
            start_epoch = request.current_epoch if request.current_epoch is not None else 0
            target_dir = base_dir / f"continue_training_from_epoch{start_epoch:03d}"
            target_dir.mkdir(parents=True)
        target_epoch = start_epoch + request.additional_epochs
        for epoch in range(start_epoch + 1, target_epoch + 1):
            (target_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())

    return _train


def _fake_train_chunk_fn_with_optimizer_state(experiment_name, torch):
    """Like :func:`_fake_train_chunk_fn`, but also writes a real
    ``optimizer_state_epochNNN.pt`` alongside each checkpoint (deterministic
    step count = epoch * 100), for Test E's actual-optimizer-update evidence
    requirement. Takes an already-imported ``torch`` module rather than
    calling ``pytest.importorskip`` itself, so the skip happens at the top
    of the test function that constructs this closure."""

    def _train(request: "orchestration.TrainChunkRequest") -> None:
        if request.is_first_chunk:
            runs_root = request.config_path.parent / "runs"
            nh_run_dir = runs_root / f"{experiment_name}_20260101_000000"
            nh_run_dir.mkdir(parents=True)
            target_dir = nh_run_dir
            start_epoch = 0
        else:
            base_dir = Path(request.nh_run_dir)
            start_epoch = request.current_epoch if request.current_epoch is not None else 0
            target_dir = base_dir / f"continue_training_from_epoch{start_epoch:03d}"
            target_dir.mkdir(parents=True)
        target_epoch = start_epoch + request.additional_epochs
        for epoch in range(start_epoch + 1, target_epoch + 1):
            (target_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
            state_dict = {"state": {0: {"step": torch.tensor(epoch * 100)}}, "param_groups": []}
            torch.save(state_dict, target_dir / f"optimizer_state_epoch{epoch:03d}.pt")

    return _train


def _fake_evaluate_checkpoint_fn(package_root, basins):
    def _evaluate(request: "orchestration.EvaluationRequest") -> None:
        assert request.period == "validation"
        write_perfect_validation_results(Path(request.nh_run_dir), request.epoch, basins, package_root)

    return _evaluate


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


@pytest.fixture
def prepared_fixture(short_tmp_path, pilot_policy):
    """A real, already-written NH config (via the mature ``prepare_pilot_run``,
    never monkeypatched) plus fake train/evaluate closures -- Tests B-E
    exercise the real ``execute_prepared_pilot_run`` composition, not a
    monkeypatched scheduling shim."""
    basins = pick_development_basins(5)
    package_root = short_tmp_path / "package"
    config_out_dir = short_tmp_path / "config_out"
    build_full_union_package(package_root, ts_basin_ids=basins)

    run_spec, bundle, config_dir, experiment_name = orchestration.prepare_pilot_run(
        pilot_policy=pilot_policy,
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
        config_out_dir=config_out_dir,
    )
    return {
        "pilot_policy": pilot_policy,
        "basins": basins,
        "package_root": package_root,
        "config_dir": config_dir,
        "experiment_name": experiment_name,
        "target_variable": bundle.target_variable,
        "train_chunk_fn": _fake_train_chunk_fn(experiment_name),
        "evaluate_checkpoint_fn": _fake_evaluate_checkpoint_fn(package_root, basins),
    }


# --- Test B: result-contract facts ------------------------------------------

def test_execute_prepared_pilot_run_result_contract_exposes_generic_facts(prepared_fixture):
    fx = prepared_fixture
    result = orchestration.execute_prepared_pilot_run(
        execution_policy=fx["pilot_policy"], config_dir=fx["config_dir"], experiment_name=fx["experiment_name"],
        package_root=fx["package_root"], target_variable=fx["target_variable"], lead_hours=fx["pilot_policy"].lead_hours,
        screening_basin_ids=fx["basins"], run_id="raw_seedA",
        train_chunk_fn=fx["train_chunk_fn"], evaluate_checkpoint_fn=fx["evaluate_checkpoint_fn"],
        max_target_epoch=6,
    )
    assert isinstance(result, orchestration.PreparedPilotExecutionResult)
    assert result.final_status == "paused_at_max_target_epoch"
    assert result.blocked is False
    assert result.blocked_reason is None
    assert result.stopped is False
    assert result.stop_reason is None
    assert Path(result.nh_run_dir).is_dir()
    assert set(result.checkpoint_inventory) == {1, 2, 3, 4, 5, 6}
    assert all(isinstance(c, orchestration.PhysicalCheckpoint) for c in result.checkpoint_inventory.values())
    assert result.early_stopping_state["policy_name"]
    assert [e["epoch"] for e in result.screening_events] == [3, 6]
    assert result.effective_policy["max_epoch_budget"] >= 6


# --- Test C: resume screening-history reconstruction (essential regression) -

def test_resume_reconstructs_complete_screening_history_across_invocations(prepared_fixture):
    """The bug this task fixes: a resumed execute_prepared_pilot_run call
    used to return only the CURRENT invocation's screening_results, silently
    dropping prior chunks'/invocations' events. This must fail against the
    old dict-shaped ``{"screening_events": all_screening_results}`` (which
    only accumulates within one call) and pass against the reconstructed,
    durable-state-driven history."""
    fx = prepared_fixture
    common_kwargs = dict(
        execution_policy=fx["pilot_policy"], config_dir=fx["config_dir"], experiment_name=fx["experiment_name"],
        package_root=fx["package_root"], target_variable=fx["target_variable"], lead_hours=fx["pilot_policy"].lead_hours,
        screening_basin_ids=fx["basins"], run_id="raw_seedA",
        train_chunk_fn=fx["train_chunk_fn"], evaluate_checkpoint_fn=fx["evaluate_checkpoint_fn"],
    )

    first = orchestration.execute_prepared_pilot_run(**common_kwargs, max_target_epoch=6)
    assert [e["epoch"] for e in first.screening_events] == [3, 6]

    second = orchestration.execute_prepared_pilot_run(**common_kwargs, max_target_epoch=9)
    assert [e["epoch"] for e in second.screening_events] == [3, 6, 9], (
        "resume must return the COMPLETE ordered screening history, not just this "
        "invocation's newly-processed epochs"
    )
    assert second.nh_run_dir == first.nh_run_dir
    assert set(second.checkpoint_inventory) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


# --- Test D: blocked/incomplete execution reports only genuine evidence -----

def test_blocked_execution_reports_blocked_facts_and_only_genuine_screening_history(prepared_fixture):
    """Real Moriah job 45718473's exact shape, replicated directly against
    execute_prepared_pilot_run (not the full run_pilot() entrypoint):
    checkpoints 1-6 flat, continue_training_from_epoch006/ containing 7-15
    (an untrusted overshoot beyond the trusted epoch-9 checkpoint). The
    executor must reuse the trusted epoch-9 checkpoint (no training), screen
    epochs 3/6/9, then refuse to advance into the 9->12 chunk -- and the
    returned receipt must report exactly that: blocked=True with a
    human-readable reason, a checkpoint inventory listing every PHYSICALLY
    present file (including the untrusted 10-15 overshoot), but a screening
    history containing only the epochs genuinely screened -- never a
    fabricated entry for an epoch that was never actually evaluated."""
    fx = prepared_fixture
    nh_run_dir = Path(fx["config_dir"]) / "runs" / f"{fx['experiment_name']}_20260101_000000"
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
        fx["train_chunk_fn"](request)

    result = orchestration.execute_prepared_pilot_run(
        execution_policy=fx["pilot_policy"], config_dir=fx["config_dir"], experiment_name=fx["experiment_name"],
        package_root=fx["package_root"], target_variable=fx["target_variable"], lead_hours=fx["pilot_policy"].lead_hours,
        screening_basin_ids=fx["basins"], run_id="raw_seedA",
        train_chunk_fn=counting_train, evaluate_checkpoint_fn=fx["evaluate_checkpoint_fn"],
    )

    assert train_calls == [], "epoch 9 already trusted -- must never train into the 10-15 overshoot range"
    assert result.final_status == "blocked_continuation_overshoot_conflict"
    assert result.blocked is True
    assert result.blocked_reason is not None and "12" in result.blocked_reason
    assert result.stopped is False
    assert set(result.checkpoint_inventory) == set(range(1, 16))
    assert [e["epoch"] for e in result.screening_events] == [3, 6, 9], (
        "must report exactly the epochs genuinely screened, never a fabricated entry "
        "for an untrusted overshoot epoch that was never evaluated"
    )


# --- Test E: vertical consumer-contract test --------------------------------

def _summarize_via_consumer_contract(result: "orchestration.PreparedPilotExecutionResult") -> dict:
    """A tiny fake higher-level consumer given ONLY the generic result
    contract (PreparedPilotExecutionResult) plus the explicitly-authoritative
    ``actual_optimizer_updates_by_epoch`` helper. Never crawls the NH run
    directory itself, never parses arbitrary filesystem layout, never
    reopens prediction data, and never recomputes any hydrologic metric --
    every fact below comes straight from an already-produced field."""
    optimizer_updates = orchestration.actual_optimizer_updates_by_epoch(result.nh_run_dir)
    population = [
        {
            "epoch": event["epoch"],
            "requested": event["n_screening_basins_requested"],
            "evaluated": len(event["raw_space_metrics"]["per_basin"]),
        }
        for event in result.screening_events
    ]
    return {
        "checkpoint_epochs": sorted(result.checkpoint_inventory),
        "optimizer_updates": optimizer_updates,
        "nh_evaluation_covered_epochs": [event["epoch"] for event in result.screening_events],
        "screening_covered_epochs": [event["epoch"] for event in result.screening_events],
        "population": population,
        "excluded_counts": [row["requested"] - row["evaluated"] for row in population],
        "nse_trajectory": [(event["epoch"], event["primary_metric_median"]) for event in result.screening_events],
    }


def test_vertical_consumer_establishes_full_coverage_without_filesystem_crawling(prepared_fixture):
    """Consumer-contract vertical test (docs/agent_handoff_rules.md section
    5): a tiny fake higher-level consumer given ONLY
    PreparedPilotExecutionResult and the explicitly-authoritative
    actual_optimizer_updates_by_epoch helper must be able to establish (1)
    physical checkpoint coverage, (2) actual optimizer-update evidence, (3)
    NH evaluation coverage through authoritative screening events, (4)
    screening coverage, (5) screening-population requested/evaluated/excluded
    evidence, and (6) the raw-space median NSE trajectory."""
    torch = pytest.importorskip("torch")
    fx = prepared_fixture
    result = orchestration.execute_prepared_pilot_run(
        execution_policy=fx["pilot_policy"], config_dir=fx["config_dir"], experiment_name=fx["experiment_name"],
        package_root=fx["package_root"], target_variable=fx["target_variable"], lead_hours=fx["pilot_policy"].lead_hours,
        screening_basin_ids=fx["basins"], run_id="raw_seedA",
        train_chunk_fn=_fake_train_chunk_fn_with_optimizer_state(fx["experiment_name"], torch),
        evaluate_checkpoint_fn=fx["evaluate_checkpoint_fn"],
        max_target_epoch=6,
    )
    summary = _summarize_via_consumer_contract(result)

    assert summary["checkpoint_epochs"] == [1, 2, 3, 4, 5, 6]
    assert summary["optimizer_updates"] == {epoch: epoch * 100 for epoch in range(1, 7)}
    assert summary["nh_evaluation_covered_epochs"] == [3, 6]
    assert summary["screening_covered_epochs"] == [3, 6]
    assert [row["requested"] for row in summary["population"]] == [len(fx["basins"])] * 2
    assert [row["evaluated"] for row in summary["population"]] == [len(fx["basins"])] * 2
    assert summary["excluded_counts"] == [0, 0]
    epochs, medians = zip(*summary["nse_trajectory"])
    assert list(epochs) == [3, 6]
    assert all(round(m, 6) == 1.0 for m in medians)
