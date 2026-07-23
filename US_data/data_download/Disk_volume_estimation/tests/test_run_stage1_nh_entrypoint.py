"""Focused tests for scripts/run_stage1_nh.py's "eval" command.

Covers the one behavior actually added on top of NeuralHydrology 1.13's own
neuralhydrology.evaluation.evaluate.start_evaluation: an optional --metrics
override applied to the run's loaded Config in memory only. Runs a real,
tiny end-to-end train-then-evaluate cycle (via
neuralhydrology.training.train.start_training and the script's own main())
against a synthetic FlashNH package -- not a mocked/standalone unit test --
so a real completed run directory (config.yml, model_epoch*.pt,
train_data_scaler.yml) is evaluated exactly as it would be on Moriah.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import build_synthetic_package  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_nh.py"


def _load_entrypoint_module():
    spec = importlib.util.spec_from_file_location("_test_run_stage1_nh", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_tiny_run(tmp_path: Path) -> Path:
    cfg_path = build_synthetic_package(
        tmp_path,
        basins=["SYN01", "SYN02"],
        seq_length=3,
        lead_hours=2,
        bad_hours=[30, 31, 55],
        target_nan_hours_by_basin={"SYN01": [20, 60], "SYN02": []},
    )
    cfg = Config(cfg_path)
    cfg.update_config({"log_tensorboard": False})  # avoid a tensorboard writer thread in the test process

    module = _load_entrypoint_module()
    module.register_flashnh_dataset()
    start_training(cfg)

    run_dirs = list((tmp_path / "runs").glob("*"))
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_import_does_not_require_cwd_at_repo_root(monkeypatch, tmp_path):
    # The whole point of the earlier sys.path fix: importing/exec'ing the
    # script must succeed regardless of the process's current directory.
    monkeypatch.chdir(tmp_path)
    module = _load_entrypoint_module()
    assert hasattr(module, "main")


def test_eval_with_explicit_metrics_writes_metrics_and_results(tmp_path):
    run_dir = _train_tiny_run(tmp_path)
    module = _load_entrypoint_module()

    config_sha_before = _sha256(run_dir / "config.yml")
    checkpoint_sha_before = _sha256(run_dir / "model_epoch001.pt")
    scaler_path = run_dir / "train_data" / "train_data_scaler.yml"
    scaler_sha_before = _sha256(scaler_path)

    argv = [
        "run_stage1_nh.py", "eval", str(run_dir),
        "--period", "validation", "--epoch", "1",
        "--metrics", "NSE", "RMSE",
    ]
    module_argv_backup = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = module_argv_backup

    metrics_csv = run_dir / "validation" / "model_epoch001" / "validation_metrics.csv"
    results_p = run_dir / "validation" / "model_epoch001" / "validation_results.p"
    assert metrics_csv.exists()
    assert results_p.exists()

    header = metrics_csv.read_text().splitlines()[0]
    assert "NSE" in header
    assert "RMSE" in header

    # Evaluation must never overwrite training evidence already on disk.
    assert _sha256(run_dir / "config.yml") == config_sha_before
    assert _sha256(run_dir / "model_epoch001.pt") == checkpoint_sha_before
    assert _sha256(scaler_path) == scaler_sha_before


def test_eval_without_metrics_matches_prior_default_behavior(tmp_path):
    # No --metrics: behavior must be identical to before --metrics existed
    # (empty cfg.metrics -> a results pickle only, no *_metrics.csv).
    run_dir = _train_tiny_run(tmp_path)
    module = _load_entrypoint_module()

    argv = ["run_stage1_nh.py", "eval", str(run_dir), "--period", "train", "--epoch", "1"]
    module_argv_backup = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = module_argv_backup

    period_dir = run_dir / "train" / "model_epoch001"
    assert (period_dir / "train_results.p").exists()
    assert not (period_dir / "train_metrics.csv").exists()


def test_eval_logs_which_checkpoint_file_was_used(tmp_path, caplog):
    # BaseTester._load_weights logs "Using the model weights from <path>" --
    # this is the objective, log-based proof that --epoch 2 (here --epoch 1,
    # since this fixture only trains one epoch) was the checkpoint actually
    # loaded, not just an untested assumption about argument plumbing. Uses
    # caplog (which captures propagated LogRecords directly) rather than
    # reading run_dir/output.log back, because pytest's own logging plugin
    # pre-attaches a root-logger handler that makes setup_logging's internal
    # logging.basicConfig(...) a no-op inside a single test process -- on
    # Moriah, "eval" always runs as its own fresh process, where basicConfig
    # runs unopposed and output.log receives this line for real.
    run_dir = _train_tiny_run(tmp_path)
    module = _load_entrypoint_module()

    argv = ["run_stage1_nh.py", "eval", str(run_dir), "--period", "test", "--epoch", "1"]
    module_argv_backup = sys.argv
    sys.argv = argv
    try:
        with caplog.at_level("INFO"):
            module.main()
    finally:
        sys.argv = module_argv_backup

    assert "Using the model weights from" in caplog.text
    assert "model_epoch001.pt" in caplog.text


def test_eval_unsupported_metric_name_raises(tmp_path):
    run_dir = _train_tiny_run(tmp_path)
    module = _load_entrypoint_module()

    argv = [
        "run_stage1_nh.py", "eval", str(run_dir),
        "--period", "test", "--epoch", "1",
        "--metrics", "NOT_A_REAL_METRIC",
    ]
    module_argv_backup = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(RuntimeError, match="Unknown metric"):
            module.main()
    finally:
        sys.argv = module_argv_backup
