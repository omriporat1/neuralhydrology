"""Focused tests for src/baseline/pilot_early_stopping.py (task item 10).

Covers: build_effective_policy's sub-cap layering (min(base_max, 36)) and
its loud rejection of a base policy that has drifted off the pilot's frozen
metric_name/higher_is_better/min_epoch_before_stop/min_delta/patience_events
assumptions; record_screening_event's diagnostic-only no-op behavior,
stopping-eligible feed-through, off-cadence rejection; restart-state
persistence/reload including rejection of state persisted under a different
effective policy; and idempotent replay vs. out-of-order/contradictory
replay rejection (delegated to early_stopping.py but exercised here through
the pilot wrapper).
"""
from __future__ import annotations

import yaml
import pytest

from src.baseline.pilot_lead06_config import load_pilot_policy
from src.baseline.pilot_early_stopping import (
    PilotEarlyStoppingError,
    STATE_FILENAME,
    build_effective_policy,
    load_or_init_pilot_state,
    pilot_best_checkpoint_epoch,
    record_screening_event,
)

from tests._pilot_support import PILOT_POLICY_PATH, REPO_ROOT

BASE_EARLY_STOPPING_POLICY_PATH = REPO_ROOT / "config" / "stage1_early_stopping_policy_v001.yaml"


@pytest.fixture
def pilot_policy():
    return load_pilot_policy(PILOT_POLICY_PATH)


def _base_policy_dict():
    with open(BASE_EARLY_STOPPING_POLICY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --- build_effective_policy: sub-cap layering --------------------------------

def test_build_effective_policy_subcaps_to_36(pilot_policy):
    effective = build_effective_policy(pilot_policy)
    base = _base_policy_dict()
    assert effective["max_epoch_budget"] == min(base["max_epoch_budget"], 36)
    assert effective["policy_name"].startswith(base["policy_name"])
    assert "pilot_subcap" in effective["policy_name"]


def test_build_effective_policy_preserves_metric_and_direction(pilot_policy):
    effective = build_effective_policy(pilot_policy)
    assert effective["metric_name"] == "median_per_basin_raw_space_nse"
    assert effective["higher_is_better"] is True
    assert effective["min_epoch_before_stop"] == pilot_policy.stopping_eligible_from_epoch
    assert effective["min_delta"] == 0.005
    assert effective["patience_events"] == 3


# --- build_effective_policy: loud rejection on base-policy drift ------------

def test_build_effective_policy_rejects_wrong_metric_name(tmp_path, pilot_policy):
    import dataclasses
    base = _base_policy_dict()
    base["metric_name"] = "some_other_metric"
    bad_path = tmp_path / "base_early_stopping.yaml"
    bad_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    bad_policy = dataclasses.replace(pilot_policy, base_early_stopping_policy_path=str(bad_path))
    with pytest.raises(PilotEarlyStoppingError):
        build_effective_policy(bad_policy)


def test_build_effective_policy_rejects_lower_is_better(tmp_path, pilot_policy):
    import dataclasses
    base = _base_policy_dict()
    base["higher_is_better"] = False
    bad_path = tmp_path / "base_early_stopping.yaml"
    bad_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    bad_policy = dataclasses.replace(pilot_policy, base_early_stopping_policy_path=str(bad_path))
    with pytest.raises(PilotEarlyStoppingError):
        build_effective_policy(bad_policy)


def test_build_effective_policy_rejects_mismatched_min_epoch_before_stop(tmp_path, pilot_policy):
    import dataclasses
    base = _base_policy_dict()
    base["min_epoch_before_stop"] = pilot_policy.stopping_eligible_from_epoch + 1
    bad_path = tmp_path / "base_early_stopping.yaml"
    bad_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    bad_policy = dataclasses.replace(pilot_policy, base_early_stopping_policy_path=str(bad_path))
    with pytest.raises(PilotEarlyStoppingError):
        build_effective_policy(bad_policy)


def test_build_effective_policy_rejects_mismatched_min_delta(tmp_path, pilot_policy):
    import dataclasses
    base = _base_policy_dict()
    base["min_delta"] = 0.05
    bad_path = tmp_path / "base_early_stopping.yaml"
    bad_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    bad_policy = dataclasses.replace(pilot_policy, base_early_stopping_policy_path=str(bad_path))
    with pytest.raises(PilotEarlyStoppingError):
        build_effective_policy(bad_policy)


def test_build_effective_policy_rejects_mismatched_patience_events(tmp_path, pilot_policy):
    import dataclasses
    base = _base_policy_dict()
    base["patience_events"] = 10
    bad_path = tmp_path / "base_early_stopping.yaml"
    bad_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    bad_policy = dataclasses.replace(pilot_policy, base_early_stopping_policy_path=str(bad_path))
    with pytest.raises(PilotEarlyStoppingError):
        build_effective_policy(bad_policy)


# --- record_screening_event: diagnostic-only is a no-op ---------------------

def test_record_screening_event_diagnostic_only_does_not_persist_state(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    state = record_screening_event(
        run_dir=run_dir, epoch=3, epoch_role="diagnostic_only",
        primary_metric_median=0.5, effective_policy=effective,
    )
    assert state["history"] == []
    assert not (run_dir / STATE_FILENAME).exists()


def test_record_screening_event_rejects_off_cadence_epoch_role(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(PilotEarlyStoppingError):
        record_screening_event(
            run_dir=run_dir, epoch=7, epoch_role="not_a_screening_epoch",
            primary_metric_median=0.5, effective_policy=effective,
        )


# --- record_screening_event: stopping-eligible feed-through + persistence --

def test_record_screening_event_stopping_eligible_persists_and_tracks_best(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    state = record_screening_event(
        run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
        primary_metric_median=0.60, effective_policy=effective,
    )
    assert (run_dir / STATE_FILENAME).exists()
    assert pilot_best_checkpoint_epoch(state) == 6
    assert state["stopped"] is False

    # a strictly better epoch-9 value resets best + events_since_best_improvement
    state = record_screening_event(
        run_dir=run_dir, epoch=9, epoch_role="stopping_eligible",
        primary_metric_median=0.70, effective_policy=effective,
    )
    assert pilot_best_checkpoint_epoch(state) == 9
    assert state["events_since_best_improvement"] == 0


def test_record_screening_event_patience_exhaustion_stops(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)
    record_screening_event(run_dir=run_dir, epoch=9, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)
    record_screening_event(run_dir=run_dir, epoch=12, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)
    state = record_screening_event(run_dir=run_dir, epoch=15, epoch_role="stopping_eligible",
                                    primary_metric_median=0.60, effective_policy=effective)
    assert state["stopped"] is True
    assert state["stop_reason"] == "patience_exhausted"
    assert pilot_best_checkpoint_epoch(state) == 6


# --- restart-state persistence/reload ----------------------------------------

def test_load_or_init_pilot_state_fresh_when_no_file(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = load_or_init_pilot_state(run_dir, effective)
    assert state["history"] == []
    assert state["policy_name"] == effective["policy_name"]


def test_load_or_init_pilot_state_reloads_persisted_state(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)
    reloaded = load_or_init_pilot_state(run_dir, effective)
    assert reloaded["best_epoch"] == 6


def test_load_or_init_pilot_state_rejects_state_from_different_policy(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)

    different_policy = dict(effective)
    different_policy["policy_name"] = effective["policy_name"] + "__different"
    with pytest.raises(PilotEarlyStoppingError):
        load_or_init_pilot_state(run_dir, different_policy)


# --- idempotent replay vs out-of-order/contradictory replay (via wrapper) --

def test_record_screening_event_idempotent_replay_of_last_event(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state1 = record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                                     primary_metric_median=0.60, effective_policy=effective)
    state2 = record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                                     primary_metric_median=0.60, effective_policy=effective)
    assert state1["history"] == state2["history"]


def test_record_screening_event_rejects_contradictory_replay(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)
    with pytest.raises(PilotEarlyStoppingError):
        record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                                primary_metric_median=0.99, effective_policy=effective)


def test_record_screening_event_rejects_out_of_order_replay(tmp_path, pilot_policy):
    effective = build_effective_policy(pilot_policy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record_screening_event(run_dir=run_dir, epoch=9, epoch_role="stopping_eligible",
                            primary_metric_median=0.60, effective_policy=effective)
    with pytest.raises(PilotEarlyStoppingError):
        record_screening_event(run_dir=run_dir, epoch=6, epoch_role="stopping_eligible",
                                primary_metric_median=0.55, effective_policy=effective)
