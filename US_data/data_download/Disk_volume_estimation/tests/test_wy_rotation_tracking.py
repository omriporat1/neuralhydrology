"""Stage-A W&B tracking of the water-year rotation runs, with a fake ``wandb``
module (the local venv has no wandb; the real-package qualification is an
online smoke on Moriah)."""
from __future__ import annotations

import hashlib
import logging
import sys
import types
from pathlib import Path

import pytest

from src.baseline import wy_rotation_tracking as trk
from src.baseline.wandb_tracking import TrackingError

POLICY = {"policy_name": "t", "enabled": True, "mode": "online", "project": "flashnh-wy-rotation",
          "entity": None, "tags": ["wy-rotation"], "max_artifact_reference_bytes": 1024}


class FakeRun:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.summary: dict = {}
        self.logs: list[dict] = []
        self.saved: list[tuple] = []
        self.defined: list[tuple] = []
        self.exit_code = None

    def define_metric(self, name, step_metric=None):
        self.defined.append((name, step_metric))

    def log(self, payload):
        self.logs.append(dict(payload))

    def save(self, path, base_path=None, policy=None):
        self.saved.append((path, base_path, policy))

    def finish(self, exit_code=0):
        self.exit_code = exit_code
        # What real W&B does at finish (smoke job 46762307): it writes its own
        # files/config.yaml (and summary), through the link run.save made there.
        for path, base_path, _ in self.saved:
            if Path(path).relative_to(base_path).as_posix() in ("config.yaml", "wandb-summary.json"):
                Path(path).write_text("_wandb:\n  value: {}\n", encoding="utf-8")

    def uploaded_names(self) -> list[str]:
        """Names in the run's Files, as W&B derives them from (path, base_path)."""
        return sorted(Path(p).relative_to(b).as_posix() for p, b, _ in self.saved)


def install_fake_wandb(monkeypatch, *, fail_init=False, init_touches_root_logger=False):
    module = types.ModuleType("wandb")
    module.runs = []
    module.Settings = lambda **kw: ("Settings", kw)

    def init(**kwargs):
        if init_touches_root_logger:  # what a real init must never be allowed to leave behind
            logging.getLogger().addHandler(logging.NullHandler())
            logging.getLogger().setLevel(logging.ERROR)
        if fail_init:
            raise RuntimeError("network down")
        run = FakeRun(**kwargs)
        module.runs.append(run)
        return run

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def _tracker(monkeypatch, tmp_path, **kw):
    fake = install_fake_wandb(monkeypatch, **kw)
    layout = trk.run_layout(cycle=2, config={}, is_canary=False)
    tracker = trk.init_rotation_tracker(POLICY, flashnh_run_id="run-c2", config={"cycle": 2}, layout=layout,
                                        wandb_dir=tmp_path / "wandb")
    return tracker, fake


def test_wandb_run_id_is_deterministic_and_valid():
    a, b = trk.wandb_run_id("x__cycle1"), trk.wandb_run_id("x__cycle1")
    assert a == b and a != trk.wandb_run_id("x__cycle2")
    assert a.startswith("wyrot-") and len(a) == 30 and a.replace("-", "").isalnum()


def test_run_layout_follows_design_v2():
    assert trk.run_layout(cycle=3, config={}, is_canary=False) == {"name": "c3-base", "group": "base",
                                                                    "job_type": "train"}
    lam = {"flashnh_magnitude_weight_lambda": 0.5}
    assert trk.run_layout(cycle=1, config=lam, is_canary=False) == {"name": "c1-magw-l0.5", "group": "magw-l0.5",
                                                                    "job_type": "train"}
    assert trk.run_layout(cycle=1, config={}, is_canary=True)["group"] == "canary"


@pytest.mark.parametrize("key", ["train/avg_total_loss", "train/epoch_wall_s", "evalrun/evaluate_s",
                                 "resource/train_peak_memory_cuda_max_allocated_mib", "verify/passed",
                                 "verify/missing_basin_admissible_target_counts", "status/receipt", "epoch"])
def test_stage_a_keys_pass_the_gate(key):
    trk.check_metric_keys({key: 1})


@pytest.mark.parametrize("key, match", [
    ("ckpt/median_x", "stage B"), ("assess/x", "stage C"), ("wy2026/x", "stage D"), ("nse", "no known stage"),
    ("train/nse_median", "forbidden"), ("train/kge", "forbidden"), ("verify/test_basins", "forbidden"),
    ("resource/validation_loss", "forbidden"), ("evalrun/loss", "loss key"), ("train/holdout_x", "forbidden"),
    ("status/api_key", "credential"),
])
def test_closed_stages_and_score_like_keys_raise(key, match):
    with pytest.raises(TrackingError, match=match):
        trk.check_metric_keys({key: 1})


def test_stage_a_is_the_only_open_stage():
    assert trk.OPEN_STAGES == {"A"}
    assert trk.STAGE_PREFIXES["B"] == ("ckpt/",) and trk.STAGE_PREFIXES["C"] == ("assess/",)


def test_parse_loss_string_and_output_log(tmp_path):
    assert trk.parse_loss_string("avg_loss: 0.12345, avg_total_loss: 1e-3") == {
        "train/avg_loss": 0.12345, "train/avg_total_loss": 0.001}
    log = tmp_path / "output.log"
    log.write_text("2026-11-15 10:00:00,001: Logging initialized.\n"
                   "2026-11-15 11:00:00,123: Epoch 1 average loss: avg_total_loss: 0.50000\n"
                   "2026-11-15 11:30:00,123: Epoch 2 average loss: avg_total_loss: 0.40000\n", encoding="utf-8")
    parsed = trk.parse_output_log(log)
    assert parsed[2] == {"end": "2026-11-15 11:30:00", "losses": {"train/avg_total_loss": 0.4}}
    assert trk.parse_output_log(tmp_path / "absent.log") == {}


def test_disabled_policy_never_imports_wandb(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)  # any import would raise
    tracker = trk.init_rotation_tracker({**POLICY, "enabled": False}, flashnh_run_id="r", config={},
                                        layout={"name": "n"}, wandb_dir=tmp_path / "w")
    assert tracker.run.backend == "null" and not tracker.run.degraded
    tracker.log({"train/avg_total_loss": 1.0}, epoch=1)
    tracker.finish(failed=False)


def test_init_passes_identity_resume_dir_and_epoch_axis(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    kw = fake.runs[0].init_kwargs
    assert kw["id"] == trk.wandb_run_id("run-c2") and kw["resume"] == "allow"
    assert kw["dir"] == str(tmp_path / "wandb") and (tmp_path / "wandb").is_dir()
    assert (kw["name"], kw["group"], kw["job_type"]) == ("c2-base", "base", "train")
    assert kw["project"] == "flashnh-wy-rotation" and kw["mode"] == "online"
    assert ("train/*", "epoch") in fake.runs[0].defined and ("evalrun/*", "epoch") in fake.runs[0].defined
    assert tracker.receipt_record()["backend"] == "wandb"


def test_failed_init_degrades_and_never_raises(monkeypatch, tmp_path):
    with pytest.warns(RuntimeWarning):
        tracker, _ = _tracker(monkeypatch, tmp_path, fail_init=True)
    rec = tracker.receipt_record()
    assert rec["backend"] == "null" and rec["degraded"] and rec["degraded_operations"] == ["rotation_init"]
    tracker.log({"train/avg_total_loss": 1.0}, epoch=1)  # still a no-op sink


def test_backend_failure_degrades_without_raising(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    fake.runs[0].log = lambda payload: (_ for _ in ()).throw(RuntimeError("503"))
    with pytest.warns(RuntimeWarning):
        tracker.log({"train/avg_total_loss": 1.0}, epoch=1)
    assert tracker.run.degraded and "rotation_log" in tracker.run.degraded_operations


def test_live_handler_logs_nh_epoch_lines_on_the_epoch_axis(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    nh_logger = logging.getLogger("neuralhydrology.training.basetrainer")
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    tracker.attach_live_loss_handler()
    try:
        nh_logger.info("Epoch 1 average loss: avg_loss: 0.50000, avg_total_loss: 0.50000")
        nh_logger.info("Setting learning rate to 0.001")
        nh_logger.info("Epoch 1 average validation loss: 0.1")  # never matched
        nh_logger.info("Epoch 2 average loss: avg_loss: 0.40000, avg_total_loss: 0.40000")
    finally:
        tracker.detach_live_loss_handler()
        root.setLevel(old_level)
    logs = fake.runs[0].logs
    assert [p["epoch"] for p in logs] == [1, 2]
    assert logs[1]["train/avg_total_loss"] == 0.4 and "train/epoch_wall_s" in logs[1]
    assert fake.runs[0].summary["train/last_epoch_logged"] == 2
    assert tracker._handler is None and not tracker.run.degraded


def test_live_handler_never_raises_into_training(monkeypatch, tmp_path):
    tracker, _ = _tracker(monkeypatch, tmp_path)
    monkeypatch.setattr(tracker, "log_epoch_loss", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    tracker.attach_live_loss_handler()
    try:
        with pytest.warns(RuntimeWarning):
            logging.getLogger("neuralhydrology").info("Epoch 3 average loss: avg_total_loss: 0.1")
    finally:
        tracker.detach_live_loss_handler()
        root.setLevel(old_level)
    assert "rotation_live_loss" in tracker.run.degraded_operations


def test_backfill_skips_epochs_an_earlier_job_logged(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    fake.runs[0].summary["train/last_epoch_logged"] = 1  # resumed W&B run
    log = tmp_path / "output.log"
    log.write_text("2026-11-15 11:00:00,1: Epoch 1 average loss: avg_total_loss: 0.5\n"
                   "2026-11-15 11:30:00,1: Epoch 2 average loss: avg_total_loss: 0.4\n"
                   "2026-11-15 12:30:00,1: Epoch 3 average loss: avg_total_loss: 0.3\n", encoding="utf-8")
    tracker.backfill_from_output_log(log)
    tracker.backfill_from_output_log(log)  # idempotent within a job
    logs = fake.runs[0].logs
    assert [p["epoch"] for p in logs] == [2, 3]
    assert logs[0]["train/epoch_wall_s"] == 1800.0 and logs[1]["train/epoch_wall_s"] == 3600.0


def test_evaluation_records_and_receipt_summary(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    tracker.log_evaluation_records([{"epoch": 1, "skipped_existing": True},
                                    {"epoch": 2, "skipped_existing": False, "tester_setup_s": 5.0,
                                     "evaluate_s": 60.0}])
    assert fake.runs[0].logs == [{"evalrun/setup_s": 5.0, "evalrun/evaluate_s": 60.0, "epoch": 2}]
    receipt = {"status": "COMPLETE", "trained_in_this_job": True, "train_wall_s": 10.0,
               "train_peak_memory": {"cuda_max_allocated_mib": 900.0}, "slurm_job_id": "123", "host": "n1"}
    evidence = {"passed": True, "n_basins_evaluated": 2289, "missing_basins": ["01381800"],
                "missing_basin_admissible_target_counts": {"01381800": 0}}
    tracker.log_receipt_summary(receipt, evidence)
    s = fake.runs[0].summary
    assert s["status/receipt"] == "COMPLETE" and s["verify/passed"] is True
    assert s["verify/missing_basins"] == ["01381800"] and s["verify/n_basins_missing_explained"] == 1
    assert s["resource/train_peak_memory_cuda_max_allocated_mib"] == 900.0


def test_save_file_uploads_small_files_only(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    small, big = tmp_path / "receipt.json", tmp_path / "model_epoch001.pt"
    small.write_text("{}", encoding="utf-8")
    big.write_bytes(b"x" * 2048)
    tracker.save_file(small)
    with pytest.warns(RuntimeWarning):
        tracker.save_file(big)
    staging = tmp_path / "wandb" / trk.UPLOAD_STAGING_DIRNAME
    assert fake.runs[0].saved == [(str(staging / "flashnh" / "receipt.json"), str(staging), "now")]
    assert fake.runs[0].uploaded_names() == ["flashnh/receipt.json"]
    assert "rotation_save_file" in tracker.run.degraded_operations


def test_uploaded_frozen_config_survives_wandb_finish(monkeypatch, tmp_path):
    # Regression for smoke job 46762307: the frozen config.yaml was uploaded as
    # files/config.yaml (W&B's reserved name) through a link to the original,
    # and W&B's own config overwrote it at finish.
    tracker, fake = _tracker(monkeypatch, tmp_path)
    config_dir = tmp_path / "run_config_dir"
    config_dir.mkdir()
    frozen = config_dir / "config.yaml"
    frozen.write_bytes(b"experiment_name: frozen\nseq_length: 120\n")
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    tracker.save_file(frozen)
    tracker.finish(failed=False)
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == before
    run = fake.runs[0]
    assert run.uploaded_names() == ["flashnh/config.yaml"]
    (saved_path, _, _) = run.saved[0]
    assert Path(saved_path) != frozen and Path(saved_path).read_bytes() == frozen.read_bytes()
    assert tracker.run.artifact_references[0]["sha256"] == before and not tracker.run.degraded


@pytest.mark.parametrize("name", sorted(trk.WANDB_RESERVED_FILE_NAMES) + ["wandb/x.json", "media/a.png"])
def test_wandb_reserved_upload_names_are_refused(name):
    with pytest.raises(TrackingError, match="reserved"):
        trk.check_upload_name(name)


@pytest.mark.parametrize("name", ["", "/abs.json", "../x.json", "flashnh/../config.yaml", "a\\b.json", "a//b"])
def test_non_plain_upload_names_are_refused(name):
    with pytest.raises(TrackingError, match="plain relative"):
        trk.check_upload_name(name)


def test_every_uploaded_name_sits_under_the_flashnh_subdir(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    for name in ("config.yaml", "output.log", "wandb-summary.json", "requirements.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
        tracker.save_file(tmp_path / name)
    assert fake.runs[0].uploaded_names() == ["flashnh/config.yaml", "flashnh/output.log", "flashnh/requirements.txt",
                                             "flashnh/wandb-summary.json"]
    for name in fake.runs[0].uploaded_names():
        assert name not in trk.WANDB_RESERVED_FILE_NAMES


def test_save_file_without_a_staging_dir_never_links_the_original(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    tracker.upload_staging_dir = None
    (tmp_path / "config.yaml").write_text("x", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        tracker.save_file(tmp_path / "config.yaml")
    assert fake.runs[0].saved == [] and "rotation_save_file" in tracker.run.degraded_operations


def test_finish_marks_failed_runs(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    tracker.finish(failed=True)
    assert fake.runs[0].exit_code == 1 and tracker.run.finished


def test_run_config_carries_identity_years_and_hyperparameters():
    row = {"run_id": "r", "cycle": 1, "config_sha256": "abc"}
    campaign_row = {"train_wys": [2023, 2024, 2025], "checkpoint_wy": 2022, "assessment_wy": 2021}
    campaign = {"campaign_id": "cid", "split_contract": {"contract_id": "k", "file_sha256": "s"}}
    config = {"hidden_size": 256, "seq_length": 120, "run_dir": "/p", "test_start_date": "x"}
    out = trk.run_config(row=row, campaign_row=campaign_row, config=config, campaign=campaign,
                         manifest_sha256="m", provenance={"git_commit": "c", "runtime_python": "py"})
    assert out["hyperparameters"] == {"hidden_size": 256, "seq_length": 120}
    assert (out["arm"], out["lambda"], out["checkpoint_wy"], out["git_commit"]) == ("base", 0.0, 2022, "c")


def test_init_turns_off_console_capture(monkeypatch, tmp_path):
    _, fake = _tracker(monkeypatch, tmp_path)
    assert fake.runs[0].init_kwargs["settings"] == ("Settings", {"console": "off"})


def test_live_handler_never_touches_the_root_logger(monkeypatch, tmp_path):
    # NH's setup_logging calls logging.basicConfig, a no-op once root has any
    # handler: a root handler would silence output.log for the whole run.
    tracker, _ = _tracker(monkeypatch, tmp_path)
    root, nh = logging.getLogger(), logging.getLogger(trk.NH_LOGGER_NAME)
    before = list(root.handlers)
    tracker.attach_live_loss_handler()
    handler = tracker._handler
    try:
        assert list(root.handlers) == before
        assert handler in nh.handlers
    finally:
        tracker.detach_live_loss_handler()
    assert tracker._handler is None and handler not in nh.handlers


@pytest.mark.parametrize("fail_init", [False, True])
def test_init_restores_any_root_logger_change(monkeypatch, tmp_path, fail_init):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    install_fake_wandb(monkeypatch, fail_init=fail_init, init_touches_root_logger=True)
    layout = trk.run_layout(cycle=1, config={}, is_canary=False)
    kwargs = dict(flashnh_run_id="r", config={"cycle": 1}, layout=layout, wandb_dir=tmp_path / "wandb")
    if fail_init:
        with pytest.warns(RuntimeWarning):
            tracker = trk.init_rotation_tracker(POLICY, **kwargs)
    else:
        tracker = trk.init_rotation_tracker(POLICY, **kwargs)
    assert list(root.handlers) == before and root.level == level
    assert tracker.root_logging_restored and tracker.receipt_record()["root_logging_restored"]


def test_a_data_derived_loss_key_the_gate_refuses_is_dropped_not_raised(monkeypatch, tmp_path):
    tracker, fake = _tracker(monkeypatch, tmp_path)
    with pytest.warns(RuntimeWarning):
        tracker.log_epoch_loss(1, {"train/avg_total_loss": 0.4, "train/avg_nse_loss": 0.9})
    assert fake.runs[0].logs == [{"train/avg_total_loss": 0.4, "epoch": 1}]
    assert "rotation_loss_key_dropped" in tracker.run.degraded_operations


def test_an_unrecognised_magnitude_weight_key_is_refused():
    with pytest.raises(TrackingError, match="magnitude"):
        trk.run_layout(cycle=1, config={"flashnh_magnitude_weight_gamma": 2.0}, is_canary=False)


@pytest.mark.parametrize("key", ["train/alpha_nse", "train/fhv", "train/q98_err", "train/flood_peak", "train/mape"])
def test_extra_score_fragments_are_refused(key):
    with pytest.raises(TrackingError):
        trk.check_metric_keys({key: 1.0})
