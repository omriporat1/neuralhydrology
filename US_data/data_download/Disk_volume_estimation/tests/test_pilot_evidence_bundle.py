"""Focused tests for src/baseline/pilot_evidence_bundle.py (task item 8's
implementation, task item 10's tests).

Adapted from a previously-passing scratchpad verification script (same
fixture shape): a lightweight synthetic PilotPolicy/PilotRunSpec (not the
real committed policy -- this module only needs the small subset of policy
fields it actually reads), two fake basins with hand-written perfect-NSE
validation results, and fake checkpoint files. Covers: bundle content
(config/manifest copy, checkpoint inventory never copies bytes, best-epoch,
sealed-set statement, slurm identity, epoch timing table, screening
events), checksums-manifest coverage, refuse-to-overwrite-non-empty-dir
without force, and rejection of a non-screening-scope event.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.baseline.pilot_lead06_config import PilotPolicy, PilotRunSpec
from src.baseline.pilot_screening_eval import evaluate_screening_checkpoint
from src.baseline.pilot_early_stopping import build_effective_policy, record_screening_event
from src.baseline.nh_seed_evaluation import weight_stem
from src.baseline.pilot_tracking import (
    init_pilot_tracking_run,
    log_pilot_epoch_training_metrics,
    log_pilot_screening_event,
)
from src.baseline.pilot_evidence_bundle import PilotEvidenceBundleError, write_pilot_evidence_bundle

from tests._pilot_support import REPO_ROOT


@pytest.fixture
def policy():
    return PilotPolicy(
        raw={"policy_name": "stage1_lead06_pilot_v001"}, path="dummy", sha256="dummy",
        lead_hours=6, seq_length=24,
        seeds={"seed_a": 967139, "seed_b": 1729}, runs={},
        workflow_qualification_run_id="emb128x64_seedA",
        pilot_max_epoch_budget=36, screening_validation_every_n_epochs=3,
        diagnostic_only_epoch=3, stopping_eligible_from_epoch=6,
        screening_basin_ids_path="dummy", screening_expected_count=400,
        screening_expected_sha256="dummy",
        base_early_stopping_policy_path=str(REPO_ROOT / "config" / "stage1_early_stopping_policy_v001.yaml"),
        wandb_policy_path=str(REPO_ROOT / "config" / "stage1_wandb_tracking_policy_v001.yaml"),
    )


@pytest.fixture
def run_spec():
    return PilotRunSpec(
        run_id="emb128x64_seedA", static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 64], seed_name="seed_a", seed=967139,
        run_profile_name="stage1_lead06_pilot_emb128x64_seedA",
    )


def _write_perfect_fixture(tmp_path):
    run_dir = tmp_path / "run"
    package_root = tmp_path / "package"
    config_dir = tmp_path / "config_out"
    evidence_dir = tmp_path / "evidence"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("model: cudalstm\nseed: 967139\n")
    (config_dir / "generation_manifest.json").write_text(
        json.dumps({"schema_name": "stage1_nh_config_generation_manifest"})
    )

    basins = ["01111111", "02222222"]
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    basin_results = {}
    for b in basins:
        n, area, lead = 300, 50.0, 6
        qobs_m3s = rng.uniform(1.0, 200.0, size=n)
        usable_n = n - lead
        target = np.full(n, np.nan)
        target[:usable_n] = 3.6 * qobs_m3s[lead:lead + usable_n] / area
        xr.Dataset(
            {"qobs_m3s": ("date", qobs_m3s), "qobs_mm_per_h_lead06": ("date", target)},
            coords={"date": np.arange(n)},
        ).to_netcdf(ts_dir / f"{b}.nc")
        obs = target.copy()
        sim = obs.copy()
        xr_ds = xr.Dataset(
            {"qobs_mm_per_h_lead06_obs": ("date", obs), "qobs_mm_per_h_lead06_sim": ("date", sim)},
            coords={"date": np.arange(n)},
        )
        basin_results[b] = {"1h": {"xr": xr_ds}}

    period_dir = run_dir / "validation" / weight_stem(6)
    period_dir.mkdir(parents=True)
    with open(period_dir / "validation_results.p", "wb") as fh:
        pickle.dump(basin_results, fh)

    (run_dir / "model_epoch003.pt").write_bytes(b"fake ckpt 3")
    (run_dir / "model_epoch006.pt").write_bytes(b"fake ckpt 6 longer")

    return run_dir, package_root, config_dir, evidence_dir, basins


def _build_evidence_inputs(tmp_path, policy):
    run_dir, package_root, config_dir, evidence_dir, basins = _write_perfect_fixture(tmp_path)
    eff_policy = build_effective_policy(policy)

    screening_result = evaluate_screening_checkpoint(
        run_dir=run_dir, epoch=6, package_root=package_root,
        target_variable="qobs_mm_per_h_lead06", lead_hours=6,
        screening_basin_ids=basins, pilot_policy=policy,
    )
    es_state = record_screening_event(
        run_dir=run_dir, epoch=6, epoch_role=screening_result["epoch_role"],
        primary_metric_median=screening_result["primary_metric_median"], effective_policy=eff_policy,
    )

    identity = {"run_id": "emb128x64_seedA", "seed": 967139}
    tr = init_pilot_tracking_run(policy, identity)
    log_pilot_epoch_training_metrics(tr, epoch=6, training_loss=0.05, learning_rate=0.001, epoch_training_time_s=100.0)
    log_pilot_screening_event(tr, epoch=6, screening_result=screening_result, early_stopping_state=es_state,
                               screening_validation_time_s=10.0)

    return {
        "run_dir": run_dir, "config_dir": config_dir, "evidence_dir": evidence_dir,
        "tracking_run": tr, "early_stopping_state": es_state, "screening_result": screening_result,
    }


def test_write_pilot_evidence_bundle_content(tmp_path, policy, run_spec):
    fx = _build_evidence_inputs(tmp_path, policy)

    bundle_path = write_pilot_evidence_bundle(
        out_dir=fx["evidence_dir"], config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
        pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
        early_stopping_state=fx["early_stopping_state"], screening_events=[fx["screening_result"]],
        run_status="in_progress_epoch_6",
        commands_used=["python scripts/run_stage1_lead06_pilot.py --run-id emb128x64_seedA"],
        slurm_identity={"job_id": "12345", "node": "moriah-gpu-01", "partition": "gpu", "gres": "gpu:1"},
    )
    assert (bundle_path / "pilot_run_evidence.json").is_file()
    assert (bundle_path / "checksums.json").is_file()
    assert (bundle_path / "config.yaml").is_file()
    assert (bundle_path / "generation_manifest.json").is_file()
    # checkpoint bytes must NOT be copied into the bundle
    assert not (bundle_path / "model_epoch006.pt").exists()

    record = json.loads((bundle_path / "pilot_run_evidence.json").read_text())
    assert len(record["checkpoint_inventory"]) == 2
    assert record["checkpoint_inventory"][0]["filename"] == "model_epoch003.pt"
    assert "sha256" in record["checkpoint_inventory"][0]
    assert record["best_checkpoint_epoch"] == 6
    assert record["sealed_set_non_access_statement"].startswith("This evidence bundle")
    assert record["slurm_identity"]["job_id"] == "12345"
    assert len(record["epoch_timing_table"]) == 2  # one from training, one from screening
    assert len(record["screening_events"]) == 1


def test_write_pilot_evidence_bundle_checksums_cover_exactly_copied_and_generated_files(tmp_path, policy, run_spec):
    fx = _build_evidence_inputs(tmp_path, policy)
    bundle_path = write_pilot_evidence_bundle(
        out_dir=fx["evidence_dir"], config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
        pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
        early_stopping_state=fx["early_stopping_state"], screening_events=[fx["screening_result"]],
        run_status="in_progress_epoch_6", commands_used=[],
    )
    checksums = json.loads((bundle_path / "checksums.json").read_text())
    assert set(checksums.keys()) == {"config.yaml", "generation_manifest.json", "pilot_run_evidence.json"}


def test_write_pilot_evidence_bundle_refuses_to_overwrite_non_empty_dir_without_force(tmp_path, policy, run_spec):
    fx = _build_evidence_inputs(tmp_path, policy)
    write_pilot_evidence_bundle(
        out_dir=fx["evidence_dir"], config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
        pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
        early_stopping_state=fx["early_stopping_state"], screening_events=[fx["screening_result"]],
        run_status="in_progress_epoch_6", commands_used=[],
    )
    with pytest.raises(PilotEvidenceBundleError):
        write_pilot_evidence_bundle(
            out_dir=fx["evidence_dir"], config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
            pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
            early_stopping_state=fx["early_stopping_state"], screening_events=[fx["screening_result"]],
            run_status="x", commands_used=[],
        )


def test_write_pilot_evidence_bundle_force_allows_overwrite(tmp_path, policy, run_spec):
    fx = _build_evidence_inputs(tmp_path, policy)
    write_pilot_evidence_bundle(
        out_dir=fx["evidence_dir"], config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
        pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
        early_stopping_state=fx["early_stopping_state"], screening_events=[fx["screening_result"]],
        run_status="in_progress_epoch_6", commands_used=[],
    )
    # must not raise
    write_pilot_evidence_bundle(
        out_dir=fx["evidence_dir"], config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
        pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
        early_stopping_state=fx["early_stopping_state"], screening_events=[fx["screening_result"]],
        run_status="stopped_patience_exhausted", commands_used=[], force=True,
    )
    record = json.loads((fx["evidence_dir"] / "pilot_run_evidence.json").read_text())
    assert record["run_status"] == "stopped_patience_exhausted"


def test_write_pilot_evidence_bundle_rejects_non_screening_scope_event(tmp_path, policy, run_spec):
    fx = _build_evidence_inputs(tmp_path, policy)
    bad_event = dict(fx["screening_result"])
    bad_event["scope"] = "development_full_population_validation"
    with pytest.raises(PilotEvidenceBundleError):
        write_pilot_evidence_bundle(
            out_dir=tmp_path / "evidence2", config_dir=fx["config_dir"], nh_run_dir=fx["run_dir"],
            pilot_policy=policy, run_spec=run_spec, tracking_run=fx["tracking_run"],
            early_stopping_state=fx["early_stopping_state"], screening_events=[bad_event],
            run_status="x", commands_used=[],
        )
