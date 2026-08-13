"""Focused tests for the shared scalar-identity engine underneath every
``enforce_pilot_*_identity`` wrapper (Stage 1 Scope A of the Sequence-
Length-A minimum-viable-infrastructure task):
``src.baseline.pilot_orchestration._enforce_pilot_scalar_identity``.

Exercises the generic engine directly (no-op when ``nh_run_dir`` is absent
or doesn't yet exist; persist-on-first-call; pass-through on a matching
repeat call; raise loudly on a mismatched repeat call) with a synthetic
axis, so this is a genuine test of the shared mechanics rather than a
duplicate of any one named wrapper's own test coverage. Also spot-checks
that the real ``enforce_pilot_seq_length_identity`` wrapper (the axis added
for this task) is wired to the same engine and raises on a seq_length
contradiction, since that wrapper had no prior dedicated test.
"""
import pytest

from src.baseline.pilot_orchestration import (
    PilotOrchestrationError,
    _enforce_pilot_scalar_identity,
    enforce_pilot_seq_length_identity,
)


def _run_identity(**overrides) -> dict:
    base = {
        "pilot_policy_name": "seq_length_range_seedA_25k_v001",
        "run_id": "test_scalar_identity_engine_run",
        "resolved_seq_length": 24,
    }
    base.update(overrides)
    return base


def test_noop_when_nh_run_dir_is_none():
    _enforce_pilot_scalar_identity(
        run_identity=_run_identity(),
        nh_run_dir=None,
        state_filename="test_axis_identity.json",
        identity_key="resolved_seq_length",
        axis_label="test-axis",
        contradiction_detail="must never change",
    )  # no exception, no filesystem access


def test_noop_when_nh_run_dir_does_not_exist_yet(tmp_path):
    missing_dir = tmp_path / "does_not_exist_yet"
    _enforce_pilot_scalar_identity(
        run_identity=_run_identity(),
        nh_run_dir=missing_dir,
        state_filename="test_axis_identity.json",
        identity_key="resolved_seq_length",
        axis_label="test-axis",
        contradiction_detail="must never change",
    )
    assert not missing_dir.exists()


def test_first_call_persists_state_file(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    _enforce_pilot_scalar_identity(
        run_identity=_run_identity(),
        nh_run_dir=nh_run_dir,
        state_filename="test_axis_identity.json",
        identity_key="resolved_seq_length",
        axis_label="test-axis",
        contradiction_detail="must never change",
    )
    assert (nh_run_dir / "test_axis_identity.json").is_file()


def test_matching_repeat_call_is_a_noop(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    identity = _run_identity()
    for _ in range(3):
        _enforce_pilot_scalar_identity(
            run_identity=identity,
            nh_run_dir=nh_run_dir,
            state_filename="test_axis_identity.json",
            identity_key="resolved_seq_length",
            axis_label="test-axis",
            contradiction_detail="must never change",
        )  # no exception on any repeat


def test_mismatched_repeat_call_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    _enforce_pilot_scalar_identity(
        run_identity=_run_identity(resolved_seq_length=24),
        nh_run_dir=nh_run_dir,
        state_filename="test_axis_identity.json",
        identity_key="resolved_seq_length",
        axis_label="test-axis",
        contradiction_detail="synthetic-axis must never change across a continuation",
    )
    with pytest.raises(PilotOrchestrationError, match="synthetic-axis must never change"):
        _enforce_pilot_scalar_identity(
            run_identity=_run_identity(resolved_seq_length=48),
            nh_run_dir=nh_run_dir,
            state_filename="test_axis_identity.json",
            identity_key="resolved_seq_length",
            axis_label="test-axis",
            contradiction_detail="synthetic-axis must never change across a continuation",
        )


def test_run_id_change_alone_also_raises(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    _enforce_pilot_scalar_identity(
        run_identity=_run_identity(run_id="run_a"),
        nh_run_dir=nh_run_dir,
        state_filename="test_axis_identity.json",
        identity_key="resolved_seq_length",
        axis_label="test-axis",
        contradiction_detail="must never change",
    )
    with pytest.raises(PilotOrchestrationError):
        _enforce_pilot_scalar_identity(
            run_identity=_run_identity(run_id="run_b"),
            nh_run_dir=nh_run_dir,
            state_filename="test_axis_identity.json",
            identity_key="resolved_seq_length",
            axis_label="test-axis",
            contradiction_detail="must never change",
        )


def test_real_seq_length_identity_wrapper_raises_on_contradiction(tmp_path):
    nh_run_dir = tmp_path / "run"
    nh_run_dir.mkdir()
    enforce_pilot_seq_length_identity(
        run_identity=_run_identity(resolved_seq_length=24), nh_run_dir=nh_run_dir
    )
    enforce_pilot_seq_length_identity(
        run_identity=_run_identity(resolved_seq_length=24), nh_run_dir=nh_run_dir
    )  # matching repeat: no-op
    with pytest.raises(PilotOrchestrationError, match="resolved_seq_length"):
        enforce_pilot_seq_length_identity(
            run_identity=_run_identity(resolved_seq_length=48), nh_run_dir=nh_run_dir
        )
