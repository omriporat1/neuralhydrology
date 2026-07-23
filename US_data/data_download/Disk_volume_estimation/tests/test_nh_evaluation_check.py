"""Focused tests for src/baseline/nh_evaluation_check.py.

Runs a real, tiny end-to-end train -> eval(validation) -> eval(test) cycle
against a synthetic FlashNH package (same fixture and in-process
scripts/run_stage1_nh.py loading approach as
tests/test_run_stage1_nh_entrypoint.py), then exercises the evidence-check
module against the real produced results.p/metrics.csv artifacts -- not a
mocked pickle -- so a genuine NeuralHydrology output shape is checked.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import build_synthetic_package  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_nh.py"

sys.path.insert(0, str(REPO_ROOT))
from src.baseline.nh_evaluation_check import (  # noqa: E402
    check_training_artifacts_unchanged,
    run_evaluation_check,
)
from src.baseline.package_audit import AuditReport, sha256_file  # noqa: E402

PROTECTED_RELPATHS = ["config.yml", "model_epoch001.pt", "train_data/train_data_scaler.yml"]


def _load_entrypoint_module():
    spec = importlib.util.spec_from_file_location("_test_run_stage1_nh_for_eval_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_eval(run_dir: Path, *, period: str, metrics) -> None:
    module = _load_entrypoint_module()
    argv = ["run_stage1_nh.py", "eval", str(run_dir), "--period", period, "--epoch", "1", "--metrics", *metrics]
    backup = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = backup


def _train_and_evaluate(tmp_path: Path, *, basins, lead_hours=2) -> Path:
    cfg_path = build_synthetic_package(
        tmp_path,
        basins=basins,
        seq_length=3,
        lead_hours=lead_hours,
        bad_hours=[30, 31, 55],
        target_nan_hours_by_basin={},
    )
    cfg = Config(cfg_path)
    cfg.update_config({"log_tensorboard": False})

    module = _load_entrypoint_module()
    module.register_flashnh_dataset()
    start_training(cfg)

    run_dirs = list((tmp_path / "runs").glob("*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    _run_eval(run_dir, period="validation", metrics=["NSE", "RMSE"])
    _run_eval(run_dir, period="test", metrics=["NSE", "RMSE"])
    return run_dir


def _pre_eval_sha256(run_dir: Path) -> dict:
    return {rel: sha256_file(run_dir / rel) for rel in PROTECTED_RELPATHS}


def test_run_evaluation_check_passes_on_a_clean_single_basin_run(tmp_path):
    basins = ["SYN01"]
    run_dir = _train_and_evaluate(tmp_path, basins=basins)
    pre_eval_sha256 = _pre_eval_sha256(run_dir)

    report, stats = run_evaluation_check(
        run_dir=run_dir,
        epoch=1,
        expected_target_variable="qobs_lead2",
        expected_basin_count=len(basins),
        expected_validation_year=2000,
        expected_test_year=2000,
        expected_metric_names=["NSE", "RMSE"],
        pre_eval_sha256=pre_eval_sha256,
    )

    assert report.status == "PASS", report.failed_messages()
    assert stats["validation"]["basin_count"] == 1
    assert stats["test"]["basin_count"] == 1
    assert stats["validation"]["sample_count"] > 0
    assert stats["test"]["sample_count"] > 0
    assert stats["validation"]["metric_names"] == ["NSE", "RMSE"]


def test_run_evaluation_check_flags_validation_basin_subsetting(tmp_path):
    # This fixture's validate_n_random_basins is 1 (see _nh_synthetic.py); with
    # 2 basins, NeuralHydrology's own Tester restricts *validation* (not test)
    # to a random subset -- proving the check module actually detects a real
    # basin-membership mismatch rather than only ever reporting PASS.
    basins = ["SYN01", "SYN02"]
    run_dir = _train_and_evaluate(tmp_path, basins=basins)
    pre_eval_sha256 = _pre_eval_sha256(run_dir)

    report, stats = run_evaluation_check(
        run_dir=run_dir,
        epoch=1,
        expected_target_variable="qobs_lead2",
        expected_basin_count=len(basins),
        expected_validation_year=2000,
        expected_test_year=2000,
        expected_metric_names=["NSE", "RMSE"],
        pre_eval_sha256=pre_eval_sha256,
    )

    assert report.status == "FAIL"
    assert stats["validation"]["basin_count"] == 1
    assert stats["test"]["basin_count"] == 2
    failed_ids = {msg.split(":")[0] for msg in report.failed_messages()}
    assert "evaluation[validation].basin_membership" in failed_ids


def test_check_training_artifacts_unchanged_flags_a_modified_checkpoint(tmp_path):
    run_dir = _train_and_evaluate(tmp_path, basins=["SYN01"])
    pre_eval_sha256 = _pre_eval_sha256(run_dir)

    checkpoint_path = run_dir / "model_epoch001.pt"
    original_bytes = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(original_bytes + b"\x00")  # simulate an accidental overwrite

    report = AuditReport()
    check_training_artifacts_unchanged(report, run_dir=run_dir, pre_eval_sha256=pre_eval_sha256)

    assert report.status == "FAIL"
    assert any("model_epoch001.pt" in msg for msg in report.failed_messages())


def test_check_training_artifacts_unchanged_passes_when_untouched(tmp_path):
    run_dir = _train_and_evaluate(tmp_path, basins=["SYN01"])
    pre_eval_sha256 = _pre_eval_sha256(run_dir)

    report = AuditReport()
    check_training_artifacts_unchanged(report, run_dir=run_dir, pre_eval_sha256=pre_eval_sha256)

    assert report.status == "PASS"
    assert report.ok_count == len(PROTECTED_RELPATHS)
