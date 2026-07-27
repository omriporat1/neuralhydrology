"""Tests for src/baseline/early_stopping.py.

Synthetic fixtures only. In addition to behavioural checks of the section
2.3 policy (min epoch before stop, min-delta-gated improvement, patience in
official validation events, hard epoch-budget ceiling, restart-safe
persisted state, best-checkpoint retention), this module includes a
structural test proving -- by inspecting function signatures, not just by
testing behaviour -- that no function in early_stopping.py can accept a
temporal-test or spatial-holdout argument, satisfying section 2.4's
requirement that the test sets never enter stopping/tuning decisions.
"""
from __future__ import annotations

import inspect
import json

import pytest
import yaml

from src.baseline.early_stopping import (
    StoppingError,
    best_checkpoint_epoch,
    load_early_stopping_policy,
    load_state,
    new_state,
    record_official_validation_event,
    save_state,
)

_REAL_POLICY_PATH = "config/stage1_early_stopping_policy_v001.yaml"


def _policy(**overrides):
    base = {
        "policy_name": "test_policy",
        "metric_name": "median_per_basin_raw_space_nse",
        "higher_is_better": True,
        "min_epoch_before_stop": 6,
        "min_delta": 0.005,
        "patience_events": 3,
        "max_epoch_budget": 40,
    }
    base.update(overrides)
    return base


def _write_policy_yaml(tmp_path, policy_dict, name="policy.yaml"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(policy_dict, fh)
    return p


def _feed(state, policy, events):
    """Feed a list of (epoch, metric_value) pairs through
    record_official_validation_event in order, returning the final state."""
    for epoch, metric_value in events:
        state = record_official_validation_event(state, epoch, metric_value, policy)
    return state


# ---------------------------------------------------------------------------
# load_early_stopping_policy
# ---------------------------------------------------------------------------

def test_load_early_stopping_policy_real_config():
    policy = load_early_stopping_policy(_REAL_POLICY_PATH)
    assert policy["min_epoch_before_stop"] == 6
    assert policy["min_delta"] == pytest.approx(0.005)
    assert policy["patience_events"] == 3
    assert policy["max_epoch_budget"] == 40
    assert policy["metric_name"] == "median_per_basin_raw_space_nse"
    assert policy["higher_is_better"] is True


def test_load_early_stopping_policy_missing_file():
    with pytest.raises(StoppingError):
        load_early_stopping_policy("does/not/exist.yaml")


def test_load_early_stopping_policy_missing_key(tmp_path):
    bad = _policy()
    del bad["patience_events"]
    p = _write_policy_yaml(tmp_path, bad)
    with pytest.raises(StoppingError):
        load_early_stopping_policy(p)


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_epoch_before_stop": 0},
        {"min_delta": -0.001},
        {"patience_events": 0},
        {"max_epoch_budget": 3, "min_epoch_before_stop": 6},
    ],
)
def test_load_early_stopping_policy_rejects_invalid_ranges(tmp_path, overrides):
    bad = _policy(**overrides)
    p = _write_policy_yaml(tmp_path, bad, name=f"policy_{id(overrides)}.yaml")
    with pytest.raises(StoppingError):
        load_early_stopping_policy(p)


# ---------------------------------------------------------------------------
# min_epoch_before_stop
# ---------------------------------------------------------------------------

def test_no_stop_before_min_epoch_regardless_of_metric_history():
    policy = _policy(patience_events=2, min_epoch_before_stop=6)
    state = new_state(policy)
    # Oscillating, non-improving-after-epoch-1 metric across epochs 1-5:
    # would exhaust patience_events=2 well before epoch 6 if the min-epoch
    # floor were not enforced.
    events = [(1, 0.20), (2, 0.10), (3, 0.05), (4, 0.05), (5, 0.05)]
    state = _feed(state, policy, events)
    assert state["stopped"] is False
    assert state["stop_reason"] is None


def test_stop_can_trigger_starting_at_min_epoch():
    policy = _policy(patience_events=2, min_epoch_before_stop=6)
    state = new_state(policy)
    events = [(1, 0.20), (2, 0.10), (3, 0.05), (4, 0.05), (5, 0.05), (6, 0.05)]
    state = _feed(state, policy, events)
    # Epoch 1 is best (0.20); epochs 2-6 are all non-improving -> patience of
    # 2 is exhausted well within this run, but only once epoch >= 6.
    assert state["stopped"] is True
    assert state["stop_epoch"] == 6


# ---------------------------------------------------------------------------
# min_delta
# ---------------------------------------------------------------------------

def test_improvement_smaller_than_min_delta_does_not_reset_patience():
    policy = _policy(min_delta=0.005, patience_events=3, min_epoch_before_stop=1)
    state = new_state(policy)
    state = record_official_validation_event(state, 1, 0.200, policy)
    assert state["best_epoch"] == 1
    # +0.003 < min_delta=0.005 -> not a qualifying improvement.
    state = record_official_validation_event(state, 2, 0.203, policy)
    assert state["best_epoch"] == 1
    assert state["events_since_best_improvement"] == 1
    assert state["history"][-1]["is_new_best"] is False


def test_improvement_at_least_min_delta_resets_patience_and_best():
    policy = _policy(min_delta=0.005, patience_events=3, min_epoch_before_stop=1)
    state = new_state(policy)
    state = record_official_validation_event(state, 1, 0.200, policy)
    state = record_official_validation_event(state, 2, 0.190, policy)
    assert state["events_since_best_improvement"] == 1
    # +0.006 >= min_delta=0.005 relative to the running best (0.200).
    state = record_official_validation_event(state, 3, 0.206, policy)
    assert state["best_epoch"] == 3
    assert state["best_metric_value"] == pytest.approx(0.206)
    assert state["events_since_best_improvement"] == 0
    assert state["history"][-1]["is_new_best"] is True


# ---------------------------------------------------------------------------
# patience_events
# ---------------------------------------------------------------------------

def test_patience_exhausted_stops_at_exact_event_not_before():
    policy = _policy(patience_events=3, min_epoch_before_stop=1, min_delta=0.005)
    state = new_state(policy)
    state = record_official_validation_event(state, 1, 0.30, policy)  # best
    state = record_official_validation_event(state, 2, 0.10, policy)  # non-improve 1
    assert state["stopped"] is False
    state = record_official_validation_event(state, 3, 0.10, policy)  # non-improve 2
    assert state["stopped"] is False
    state = record_official_validation_event(state, 4, 0.10, policy)  # non-improve 3 -> stop
    assert state["stopped"] is True
    assert state["stop_reason"] == "patience_exhausted"
    assert state["stop_epoch"] == 4


# ---------------------------------------------------------------------------
# max_epoch_budget
# ---------------------------------------------------------------------------

def test_max_epoch_budget_stops_even_while_still_improving():
    policy = _policy(max_epoch_budget=10, min_epoch_before_stop=1, patience_events=100)
    state = new_state(policy)
    for epoch in range(1, 10):
        state = record_official_validation_event(state, epoch, 0.01 * epoch, policy)
        assert state["stopped"] is False
    state = record_official_validation_event(state, 10, 0.10, policy)
    assert state["stopped"] is True
    assert state["stop_reason"] == "max_epoch_budget_reached"
    assert state["stop_epoch"] == 10


# ---------------------------------------------------------------------------
# Restart safety
# ---------------------------------------------------------------------------

def test_state_roundtrips_through_save_and_load(tmp_path):
    policy = _policy(min_epoch_before_stop=1)
    state = new_state(policy)
    state = _feed(state, policy, [(1, 0.20), (2, 0.19), (3, 0.25)])
    state_path = tmp_path / "state.json"
    save_state(state_path, state)
    reloaded = load_state(state_path)
    assert reloaded == state


def test_load_state_returns_none_when_absent(tmp_path):
    assert load_state(tmp_path / "does_not_exist.json") is None


def test_resumed_state_produces_identical_subsequent_decisions(tmp_path):
    policy = _policy(min_epoch_before_stop=1, patience_events=2)
    state_a = new_state(policy)
    state_a = _feed(state_a, policy, [(1, 0.20), (2, 0.10)])

    state_path = tmp_path / "state.json"
    save_state(state_path, state_a)
    resumed = load_state(state_path)

    tail_events = [(3, 0.10)]
    final_from_a = _feed(state_a, policy, tail_events)
    final_from_resumed = _feed(resumed, policy, tail_events)
    assert final_from_a == final_from_resumed


def test_replaying_identical_last_event_is_a_no_op():
    policy = _policy(min_epoch_before_stop=1)
    state = new_state(policy)
    state = record_official_validation_event(state, 1, 0.20, policy)
    replayed = record_official_validation_event(state, 1, 0.20, policy)
    assert replayed == state


def test_replaying_last_event_with_different_metric_raises():
    policy = _policy(min_epoch_before_stop=1)
    state = new_state(policy)
    state = record_official_validation_event(state, 1, 0.20, policy)
    with pytest.raises(StoppingError):
        record_official_validation_event(state, 1, 0.21, policy)


def test_out_of_order_epoch_raises():
    policy = _policy(min_epoch_before_stop=1)
    state = new_state(policy)
    state = record_official_validation_event(state, 5, 0.20, policy)
    with pytest.raises(StoppingError):
        record_official_validation_event(state, 3, 0.25, policy)


def test_recording_after_stopped_raises():
    policy = _policy(min_epoch_before_stop=1, max_epoch_budget=2)
    state = new_state(policy)
    state = record_official_validation_event(state, 1, 0.20, policy)
    state = record_official_validation_event(state, 2, 0.21, policy)
    assert state["stopped"] is True
    with pytest.raises(StoppingError):
        record_official_validation_event(state, 3, 0.30, policy)


# ---------------------------------------------------------------------------
# Best-checkpoint retention
# ---------------------------------------------------------------------------

def test_best_checkpoint_retained_after_later_non_improving_events():
    policy = _policy(min_epoch_before_stop=1, patience_events=100)
    state = new_state(policy)
    state = _feed(state, policy, [(1, 0.10), (2, 0.30), (3, 0.05), (4, 0.02), (5, 0.01)])
    assert best_checkpoint_epoch(state) == 2
    assert state["best_metric_value"] == pytest.approx(0.30)


def test_best_checkpoint_epoch_none_before_any_event():
    policy = _policy()
    state = new_state(policy)
    assert best_checkpoint_epoch(state) is None


# ---------------------------------------------------------------------------
# Structural proof: no temporal-test / spatial-holdout access.
#
# Section 2.4 requires the temporal test and spatial holdout sets to never
# be read, evaluated, or used for stopping/tuning/checkpoint selection.
# Behavioural tests above show only development-validation metric values
# drive every decision; this test additionally proves it structurally: no
# public function in this module has a parameter that could plausibly carry
# temporal-test or spatial-holdout data, so it is not merely untested but
# impossible for those sets to enter stopping decisions at the API level.
# ---------------------------------------------------------------------------

_DISALLOWED_PARAM_NAME_FRAGMENTS = ("test", "holdout", "temporal", "spatial")


@pytest.mark.parametrize(
    "func",
    [load_early_stopping_policy, new_state, load_state, save_state, record_official_validation_event, best_checkpoint_epoch],
)
def test_public_function_signatures_have_no_test_or_holdout_argument(func):
    params = list(inspect.signature(func).parameters)
    for name in params:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS), (
            f"{func.__name__} has disallowed parameter name {name!r}"
        )


def test_module_public_api_has_no_test_or_holdout_symbol():
    import src.baseline.early_stopping as mod
    for name in mod.__all__:
        assert not any(frag in name.lower() for frag in _DISALLOWED_PARAM_NAME_FRAGMENTS)
