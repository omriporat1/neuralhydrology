"""Local, no-W&B tests for the v2 six-axis execution-spine provenance
integration (Section F, additive six-axis campaign foundation).

Mirrors tests/test_sweep_v1_execution.py's fixture pattern (real prepared
record -> real generated config -> injected fake
pilot_orchestration.PreparedPilotExecutionResult -> execute_prepared_trial_v2)
combined with tests/test_sweep_v2_six_axis_production_adapter.py's v2
PreparationPathsV2/fixed-support-contract fixture. Also covers
write_proposal_intake_provenance_v2 directly (mirroring
tests/test_sweep_v1_wandb_bridge_provenance.py's intake-only cases), since
Sweep-v1's own vertical execution test file does not exercise its intake
function either -- that is covered separately there, and is covered
separately here.

Never starts real NH/W&B: fake receipts are fabricated directly, but
nh_run_dir contains real model_epochNNN.pt + optimizer_state_epochNNN.pt
files so actual_optimizer_updates_by_epoch reads genuine cumulative-update
evidence from disk, exactly like tests/test_sweep_v1_execution.py.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.baseline import pilot_orchestration as orchestration
from src.baseline import sweep_v1_campaign as sweep
from src.baseline.fixed_support_contract_v2 import build_fixed_support_contract, load_fixed_support_contract, write_fixed_support_contract
from src.baseline.pilot_screening_eval import PRIMARY_METRIC_NAME, SCREENING_METRIC_SCOPE
from src.baseline.sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2, DOMAIN_VERSION_V2, OBJECTIVE_ID_V2, SweepV2CampaignError,
)
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME
from src.baseline.sweep_v2_six_axis_execution import (
    EXECUTOR_MODE_MONOLITHIC, SweepV2ExecutionError, enrich_operations_slurm_accounting_v2,
    execute_prepared_trial_v2, select_executor_mode_v2, write_proposal_intake_provenance_v2,
)
from src.baseline.sweep_v2_six_axis_production_adapter import (
    PreparationPathsV2, prepare_bayesian_proposal_v2, write_prepared_proposal_v2,
)
from tests._pilot_support import (
    BASELINE_POLICY_PATH, REAL_DEVELOPMENT, SPLITS_DIR, build_full_union_package, write_screening_basin_ids_file,
)

_OVERLAY_PATH = Path(__file__).parents[1] / "config" / "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"


# short_tmp_path fixture is defined repo-wide in tests/conftest.py (long-path
# workaround for v2's longer trial_id strings -- see that fixture's docstring).


# --- fixture plumbing (mirrors tests/test_sweep_v2_six_axis_production_adapter.py's
# private helpers of the same shape -- repo convention for private test
# helpers, see test_prepared_execution_core.py's module docstring) ----------

def _build_fixed_support_contract(tmp_path) -> Path:
    n = 10
    per_basin_date = {"01234567": np.arange(n)}
    per_basin_admitted = {"01234567": np.zeros(n, dtype=bool)}
    per_basin_admitted["01234567"][2:8] = True
    contract = build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="test_period", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="test_gap_policy_v001", screening_basin_ids_sha256="0" * 64,
        per_basin_date=per_basin_date, per_basin_admitted=per_basin_admitted,
    )
    path = write_fixed_support_contract(contract, tmp_path / "fixed_support_contract.json")
    return path


def _paths_v2(tmp_path, monkeypatch):
    package = build_full_union_package(tmp_path / "package")
    manifests = package / "manifests"
    (manifests / "file_checksums.csv").write_text("relative_path,sha256,size_bytes,artifact_role\n", encoding="utf-8")
    (package / "run_provenance.json").write_text('{"fixture":true}\n', encoding="utf-8")
    import src.baseline.sweep_v1_production_adapter as adapter
    monkeypatch.setattr(adapter, "PACKAGE_MANIFEST_SHA256", hashlib.sha256((manifests / "package_manifest.json").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_FILE_CHECKSUMS_SHA256", hashlib.sha256((manifests / "file_checksums.csv").read_bytes()).hexdigest())
    monkeypatch.setattr(adapter, "PACKAGE_RUN_PROVENANCE_SHA256", hashlib.sha256((package / "run_provenance.json").read_bytes()).hexdigest())
    splits = tmp_path / "canonical_splits"; splits.mkdir()
    for source in Path(SPLITS_DIR).glob("*.txt"):
        (splits / source.name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    screening = write_screening_basin_ids_file(tmp_path / "screening.txt", REAL_DEVELOPMENT[:400])
    monkeypatch.setattr("src.baseline.pilot_lead06_config.sha256_of", lambda _: sweep.SCREENING_ARTIFACT_SHA256)
    contract_path = _build_fixed_support_contract(tmp_path)
    return PreparationPathsV2(BASELINE_POLICY_PATH, _OVERLAY_PATH, package, splits, screening, contract_path)


def _proposal_v2(**changes):
    value = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
             "output_dropout": 0.25, "batch_size": 256, "seq_length": 96, "proposal_order": 7,
             "wandb_sweep_id": "v2-prod-sweep", "wandb_run_id": "run-7"}
    value.update(changes)
    return value


def _prepared_record_v2(tmp_path, monkeypatch, **proposal_changes):
    paths = _paths_v2(tmp_path / "prep", monkeypatch)
    prepared = prepare_bayesian_proposal_v2(proposal=_proposal_v2(**proposal_changes), paths=paths)
    record = write_prepared_proposal_v2(prepared, tmp_path / "prepared_out")
    return record, paths


def _write_real_checkpoints(nh_run_dir: Path, epochs, torch, *, updates_per_epoch: int = 100) -> None:
    nh_run_dir.mkdir(parents=True, exist_ok=True)
    for epoch in epochs:
        (nh_run_dir / f"model_epoch{epoch:03d}.pt").write_bytes(f"ckpt{epoch}".encode())
        state_dict = {"state": {0: {"step": torch.tensor(epoch * updates_per_epoch)}}, "param_groups": []}
        torch.save(state_dict, nh_run_dir / f"optimizer_state_epoch{epoch:03d}.pt")


def _screening_event(epoch: int, score, *, n_basins: int) -> dict:
    return {
        "scope": SCREENING_METRIC_SCOPE, "authoritative": False, "epoch": epoch,
        "epoch_role": "stopping_eligible", "stopping_eligible": True,
        "n_screening_basins_requested": n_basins,
        "primary_metric_name": PRIMARY_METRIC_NAME, "primary_metric_median": score,
        "primary_metric_distribution": {},
        "raw_space_metrics": {
            "n_basins_requested": n_basins, "n_basins_evaluated": n_basins,
            "n_basins_area_excluded": 0, "area_derivation_excluded": [], "per_basin": [],
        },
    }


def _fake_result(nh_run_dir: Path, *, checkpoint_epochs, screening_scores: "dict[int, float]", n_basins: int,
                 blocked: bool = False, blocked_reason=None, stopped: bool = False, stop_reason=None,
                 max_epoch_budget: int = 12) -> "orchestration.PreparedPilotExecutionResult":
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir
        )
        for epoch in checkpoint_epochs
    }
    screening_events = [_screening_event(epoch, score, n_basins=n_basins) for epoch, score in sorted(screening_scores.items())]
    fixed_scores = {epoch: 0.50 - abs(epoch - 3) * 0.01 for epoch in screening_scores}
    def metric(scope, value):
        return {"objective_scope": scope, "aggregate": {"metrics": {"nse": {"median": value}}}}
    supplemental = {
        epoch: {"fixed_support": metric("fixed_support", fixed_scores[epoch]),
                "natural_support": metric("natural_support", screening_scores[epoch])}
        for epoch in screening_scores
    }
    return orchestration.PreparedPilotExecutionResult(
        final_status="blocked_or_stopped" if (blocked or stopped) else "completed_at_full_budget",
        blocked_reason=blocked_reason,
        effective_policy={"max_epoch_budget": max_epoch_budget, "performance_early_stopping_enabled": False},
        nh_run_dir=nh_run_dir, blocked=blocked, stopped=stopped, stop_reason=stop_reason,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
        supplemental_epoch_results=supplemental,
    )


def _baseline_v2(tmp_path, monkeypatch) -> dict:
    """A fully valid Sweep-v2 six-axis 12-epoch scenario: real prepared
    record (via prepare_bayesian_proposal_v2/write_prepared_proposal_v2, six
    axes including seq_length=96), real on-disk checkpoint/optimizer-state
    evidence, and the same screening trajectory shape as v1's own baseline
    fixture (max at epoch 9, final at epoch 12)."""
    torch = pytest.importorskip("torch")
    record, paths = _prepared_record_v2(tmp_path, monkeypatch)
    nh_run_dir = tmp_path / "nh_run"
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    scores = {1: 0.10, 2: 0.15, 3: 0.20, 4: 0.22, 5: 0.25, 6: 0.28, 7: 0.30, 8: 0.32,
              9: 0.40, 10: 0.38, 11: 0.36, 12: 0.35}
    return {"record": record, "paths": paths, "nh_run_dir": nh_run_dir, "epochs": epochs,
            "scores": scores, "n_basins": 400, "torch": torch}


_AXES_V2_SAMPLE = {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.10,
                   "output_dropout": 0.25, "batch_size": 256, "seq_length": 96}


def _support_contract_identity(tmp_path):
    contract_path = _build_fixed_support_contract(tmp_path)
    contract = load_fixed_support_contract(contract_path)
    return contract["contract_id"], contract["checksum_sha256"]


# --- write_proposal_intake_provenance_v2 --------------------------------------

def test_write_proposal_intake_provenance_v2_records_all_six_axes_and_support_contract(short_tmp_path):
    version, sha256 = _support_contract_identity(short_tmp_path)
    intake = write_proposal_intake_provenance_v2(
        output_root=short_tmp_path / "out", axes=_AXES_V2_SAMPLE, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="v2-prod-sweep", wandb_run_id="run-7",
        support_contract_version=version, support_contract_sha256=sha256,
    )
    assert intake["campaign_id"] == CAMPAIGN_ID_V2
    assert intake["domain_version"] == DOMAIN_VERSION_V2
    assert intake["hyperparameters"]["seq_length"] == 96
    assert set(intake["hyperparameters"]) == set(_AXES_V2_SAMPLE)
    assert intake["support_contract_version"] == version
    assert intake["support_contract_sha256"] == sha256
    assert (short_tmp_path / "out" / intake["trial_id"] / "execution_provenance.json").is_file()


def test_write_proposal_intake_provenance_v2_refuses_overwrite(short_tmp_path):
    version, sha256 = _support_contract_identity(short_tmp_path)
    kwargs = dict(output_root=short_tmp_path / "out", axes=_AXES_V2_SAMPLE, search_arm="bayesian",
                 proposal_order=7, wandb_sweep_id="v2-prod-sweep", wandb_run_id="run-7",
                 support_contract_version=version, support_contract_sha256=sha256)
    write_proposal_intake_provenance_v2(**kwargs)
    with pytest.raises(SweepV2ExecutionError):
        write_proposal_intake_provenance_v2(**kwargs)


def test_write_proposal_intake_provenance_v2_rejects_illegal_seq_length_with_durable_rejection_record(tmp_path):
    version, sha256 = _support_contract_identity(tmp_path)
    bad_axes = {**_AXES_V2_SAMPLE, "seq_length": 50}
    with pytest.raises(ValueError):
        write_proposal_intake_provenance_v2(
            output_root=tmp_path / "out", axes=bad_axes, search_arm="bayesian",
            proposal_order=7, wandb_sweep_id="v2-prod-sweep", wandb_run_id="run-99",
            support_contract_version=version, support_contract_sha256=sha256,
        )
    rejected_path = tmp_path / "out" / "proposal_intake_rejected__wandb_run_run-99" / "execution_provenance.json"
    assert rejected_path.is_file()
    record = json.loads(rejected_path.read_text(encoding="utf-8"))
    assert record["provenance_stage"] == "proposal_intake_rejected"
    assert record["campaign_id"] == CAMPAIGN_ID_V2
    assert record["raw_proposed_axes"]["seq_length"] == 50
    assert "rejection_reason" in record


def test_write_proposal_intake_provenance_v2_never_embeds_v1_campaign_identity(short_tmp_path):
    version, sha256 = _support_contract_identity(short_tmp_path)
    intake = write_proposal_intake_provenance_v2(
        output_root=short_tmp_path / "out2", axes=_AXES_V2_SAMPLE, search_arm="bayesian",
        proposal_order=7, wandb_sweep_id="v2-prod-sweep", wandb_run_id="run-7",
        support_contract_version=version, support_contract_sha256=sha256,
    )
    assert intake["configuration_id"].startswith("sweep_v2_cfg_")
    assert sweep.CAMPAIGN_ID not in intake["configuration_id"]


def test_retry_generation_produces_distinct_trial_id_v2(short_tmp_path):
    version, sha256 = _support_contract_identity(short_tmp_path)
    common = dict(axes=_AXES_V2_SAMPLE, search_arm="bayesian", proposal_order=7,
                 wandb_sweep_id="v2-prod-sweep", wandb_run_id="run-7",
                 support_contract_version=version, support_contract_sha256=sha256)
    first = write_proposal_intake_provenance_v2(output_root=short_tmp_path / "attempt1", **common)
    retry = write_proposal_intake_provenance_v2(
        output_root=short_tmp_path / "attempt2", execution_generation=2, retry_of_trial_id=first["trial_id"], **common
    )
    assert retry["configuration_id"] == first["configuration_id"]
    assert retry["proposal_id"] == first["proposal_id"]
    assert retry["trial_id"] != first["trial_id"]
    assert retry["retry_of_trial_id"] == first["trial_id"]
    assert retry["execution_generation"] == 2


# --- vertical consumer-contract test -----------------------------------------

def test_vertical_prepared_execution_consumer_contract_v2(tmp_path, monkeypatch):
    fx = _baseline_v2(tmp_path, monkeypatch)
    assert fx["record"]["campaign_id"] == CAMPAIGN_ID_V2
    assert fx["record"]["hyperparameters"]["seq_length"] == 96
    assert fx["record"]["seq_length_raw"] == 96 and fx["record"]["seq_length_normalized"] == 96

    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    outcome = execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )

    assert outcome["valid"] is True
    trial = outcome["review_records"]["trial_summary"]
    assert trial["campaign_id"] == CAMPAIGN_ID_V2 and trial["domain_version"] == DOMAIN_VERSION_V2
    assert trial["workflow_status"] == "pass"
    assert trial["objective_score"] == pytest.approx(0.50)
    assert trial["best_epoch"] == 3
    assert trial["best_score"] == pytest.approx(0.50)
    assert trial["natural_support_epoch_trajectory"][9] == pytest.approx(0.40)
    assert trial["fixed_support_metric_name"] == V2_METRIC_NAME
    assert trial["seq_length"] == 96
    assert trial["gpu_hours"] is None
    assert trial["failure_category"] is None

    trajectory = outcome["review_records"]["epoch_trajectory"]
    assert len(trajectory) == 12
    assert {row["epoch"] for row in trajectory} == set(range(1, 13))

    proposal = outcome["review_records"]["proposal"]
    assert proposal["seq_length"] == 96
    assert proposal["wave_id"] == f"{DOMAIN_VERSION_V2}_wave1"

    provenance = json.loads((output_dir / "execution_provenance.json").read_text(encoding="utf-8"))
    assert provenance["execution_status"] == "VALID"
    assert provenance["objective_score"] == pytest.approx(0.50)
    review = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert review["trial_summary"]["objective_score"] == pytest.approx(0.50)
    assert review["trial_summary"]["seq_length"] == 96


def test_missing_required_checkpoint_epoch_is_invalid_v2(tmp_path, monkeypatch):
    """Proves _derive_validity reuse works unchanged for v2: same negative
    contract as tests/test_sweep_v1_execution.py's equivalent case."""
    fx = _baseline_v2(tmp_path, monkeypatch)
    incomplete_epochs = [e for e in fx["epochs"] if e != 7]
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=incomplete_epochs,
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    outcome = execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None
    assert outcome["review_records"]["trial_summary"]["workflow_status"] == "failed"


def test_callback_or_fixed_support_failure_is_invalid_and_objective_free_v2(tmp_path, monkeypatch):
    fx = _baseline_v2(tmp_path, monkeypatch)

    def interrupted_pure_callback_executor():
        raise RuntimeError("supplemental callback failed before a complete trajectory existed")

    outcome = execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=interrupted_pure_callback_executor,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None
    assert outcome["provenance"]["execution_status"] == "INVALID"


def test_wrong_result_type_from_injected_executor_is_invalid_v2(tmp_path, monkeypatch):
    fx = _baseline_v2(tmp_path, monkeypatch)
    old_style_dict = {"epochs_reached": 12, "checkpoint_epochs": fx["epochs"], "screening_scores": fx["scores"]}
    outcome = execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=tmp_path / "trial_out",
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: old_style_dict,
    )
    assert outcome["valid"] is False
    assert outcome["review_records"]["trial_summary"]["objective_score"] is None
    assert outcome["review_records"]["trial_summary"]["failure_category"] == "technical_execution_failure"


# --- select_executor_mode_v2 --------------------------------------------------

def test_select_executor_mode_v2_returns_monolithic_for_valid_prepared_record(tmp_path, monkeypatch):
    record, _paths = _prepared_record_v2(tmp_path, monkeypatch)
    assert select_executor_mode_v2(record) == EXECUTOR_MODE_MONOLITHIC


def test_select_executor_mode_v2_refuses_broken_contract(tmp_path, monkeypatch):
    record, _paths = _prepared_record_v2(tmp_path, monkeypatch)
    broken = dict(record)
    broken["performance_early_stopping_enabled"] = True
    with pytest.raises(SweepV2ExecutionError, match="prepared-trial contract mismatch"):
        select_executor_mode_v2(broken)


def test_select_executor_mode_v2_does_not_mutate_its_input(tmp_path, monkeypatch):
    record, _paths = _prepared_record_v2(tmp_path, monkeypatch)
    before = copy.deepcopy(record)
    select_executor_mode_v2(record)
    assert record == before


def test_select_executor_mode_v2_never_imports_pilot_orchestration_or_torch(tmp_path, monkeypatch):
    record, _paths = _prepared_record_v2(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "src.baseline.pilot_orchestration", None)
    assert select_executor_mode_v2(record) == EXECUTOR_MODE_MONOLITHIC


# --- Slurm job/state/GPU-hour provenance propagation -------------------------

def test_enrich_operations_slurm_accounting_v2_patches_state_and_gpu_hours(tmp_path, monkeypatch):
    fx = _baseline_v2(tmp_path, monkeypatch)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    outcome = execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
        slurm_job_id="45999103",
    )
    assert outcome["review_records"]["operations"]["gpu_hours"] is None

    patched = enrich_operations_slurm_accounting_v2(
        output_dir=output_dir, slurm_job_id="45999103", slurm_state="COMPLETED", gpu_hours=2.25,
    )
    assert patched["operations"]["slurm_state"] == "COMPLETED"
    assert patched["operations"]["gpu_hours"] == pytest.approx(2.25)
    assert patched["trial_summary"]["gpu_hours"] == pytest.approx(2.25)
    assert patched["trial_summary"]["objective_score"] == pytest.approx(0.50)

    on_disk = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert on_disk["operations"]["slurm_state"] == "COMPLETED"
    assert on_disk["operations"]["gpu_hours"] == pytest.approx(2.25)


def test_enrich_operations_slurm_accounting_v2_refuses_mismatched_job_id(tmp_path, monkeypatch):
    fx = _baseline_v2(tmp_path, monkeypatch)
    fake_result = _fake_result(fx["nh_run_dir"], checkpoint_epochs=fx["epochs"],
                               screening_scores=fx["scores"], n_basins=fx["n_basins"])
    output_dir = tmp_path / "trial_out"
    execute_prepared_trial_v2(
        prepared_record=fx["record"], output_dir=output_dir,
        expected_screening_population=fx["n_basins"], execute_prepared_run_fn=lambda: fake_result,
        slurm_job_id="45999103",
    )
    with pytest.raises(SweepV2ExecutionError):
        enrich_operations_slurm_accounting_v2(
            output_dir=output_dir, slurm_job_id="wrong-job-id", slurm_state="COMPLETED", gpu_hours=1.0,
        )
    on_disk = json.loads((output_dir / "review_records.json").read_text(encoding="utf-8"))
    assert on_disk["operations"]["slurm_state"] is None
    assert on_disk["operations"]["gpu_hours"] is None
