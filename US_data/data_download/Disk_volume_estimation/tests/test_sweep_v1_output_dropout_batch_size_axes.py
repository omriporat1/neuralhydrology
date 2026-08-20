"""Focused Sweep-v1 plumbing tests; no training, W&B, or sweep controller."""
from __future__ import annotations

import dataclasses
import json

import pytest

from src.baseline.nh_config_generation import (
    NHConfigGenerationError,
    build_nh_config_mapping,
    validate_batch_size_override,
    validate_output_dropout_override,
    write_generated_config,
)
from src.baseline.pilot_early_stopping import build_effective_policy
from src.baseline.pilot_lead06_config import build_pilot_bundle, load_pilot_policy, resolve_pilot_run_spec
from src.baseline.pilot_orchestration import (
    PilotOrchestrationError,
    enforce_pilot_batch_size_identity,
    enforce_pilot_output_dropout_identity,
)
from src.baseline.pilot_tracking import build_pilot_run_identity
from src.baseline.splits import sha256_of
from tests._pilot_support import (
    BASELINE_POLICY_PATH, PILOT_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR,
    build_full_union_package, write_screening_basin_ids_file,
)


def _mapping(**overrides):
    policy = __import__("src.baseline.policy", fromlist=["load_stage1_baseline_policy"]).load_stage1_baseline_policy(BASELINE_POLICY_PATH)
    args = dict(policy=policy, target_variable="qobs_mm_per_h_lead06", seq_length=72,
                dynamic_inputs=list(policy["dynamic_inputs"]), static_attributes=["area_gages2"],
                run_profile_name="pilot_lead06_raw_seedA_v001")
    args.update(overrides)
    return build_nh_config_mapping(**args)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.4, 0.999])
def test_output_dropout_legal_domain_and_config_threading(value):
    validate_output_dropout_override(value)
    assert _mapping(output_dropout=value)["output_dropout"] == pytest.approx(value)


@pytest.mark.parametrize("value", [-0.01, 1.0, 2, True, "0.2", float("nan")])
def test_output_dropout_rejects_invalid_values(value):
    with pytest.raises(NHConfigGenerationError, match="output_dropout"):
        validate_output_dropout_override(value)


@pytest.mark.parametrize("value", [1, 128, 256, 512])
def test_batch_size_positive_integer_domain_and_config_threading(value):
    validate_batch_size_override(value)
    assert _mapping(batch_size=value)["batch_size"] == value


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "256", None])
def test_batch_size_rejects_invalid_values(value):
    with pytest.raises(NHConfigGenerationError, match="batch_size"):
        validate_batch_size_override(value)


def test_batch_size_override_does_not_modify_update_cap():
    assert _mapping(max_updates_per_epoch=50_000, batch_size=128)["max_updates_per_epoch"] == 50_000
    assert _mapping(max_updates_per_epoch=50_000, batch_size=512)["max_updates_per_epoch"] == 50_000


def test_pt_profile_defaults_remain_output_dropout_point25_and_batch_size256():
    mapping = _mapping()
    assert mapping["output_dropout"] == pytest.approx(0.25)
    assert mapping["batch_size"] == 256


def test_pilot_identity_and_manifest_record_both_axes(tmp_path):
    base = load_pilot_policy(PILOT_POLICY_PATH)
    package_root = tmp_path / "package"
    build_full_union_package(package_root)
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:350])
    overridden = dataclasses.replace(base.runs["raw_seedA"], output_dropout=0.4, batch_size=128,
                                    max_updates_per_epoch=50_000)
    runs = dict(base.runs); runs["raw_seedA"] = overridden
    policy = dataclasses.replace(base, runs=runs, screening_basin_ids_path=str(screening),
                               screening_expected_count=350, screening_expected_sha256=sha256_of(screening))
    bundle = build_pilot_bundle(pilot_policy=policy, run_id="raw_seedA", baseline_policy_path=BASELINE_POLICY_PATH,
                                package_root=package_root, splits_dir=SPLITS_DIR)
    identity = build_pilot_run_identity(pilot_policy=policy, run_spec=resolve_pilot_run_spec(policy, "raw_seedA"),
                                        bundle=bundle, effective_early_stopping_policy=build_effective_policy(policy))
    assert (identity["output_dropout_override"], identity["resolved_output_dropout"]) == (0.4, 0.4)
    assert (identity["batch_size_override"], identity["resolved_batch_size"]) == (128, 128)
    assert identity["max_updates_per_epoch"] == 50_000
    manifest = json.loads(write_generated_config(bundle, tmp_path / "generated")["generation_manifest.json"].read_text())
    assert (manifest["output_dropout_override"], manifest["resolved_output_dropout"]) == (0.4, 0.4)
    assert (manifest["batch_size_override"], manifest["resolved_batch_size"]) == (128, 128)


@pytest.mark.parametrize("guard,key,first,second", [
    (enforce_pilot_output_dropout_identity, "resolved_output_dropout", 0.0, 0.4),
    (enforce_pilot_batch_size_identity, "resolved_batch_size", 128, 512),
])
def test_axis_identity_rejects_restart_or_continuation_mismatch(tmp_path, guard, key, first, second):
    run_dir = tmp_path / "run"; run_dir.mkdir()
    base = {"pilot_policy_name": "test", "run_id": "candidate", key: first}
    guard(run_identity=base, nh_run_dir=run_dir)
    guard(run_identity=base, nh_run_dir=run_dir)  # matching continuation is legal
    with pytest.raises(PilotOrchestrationError):
        guard(run_identity={**base, key: second}, nh_run_dir=run_dir)
