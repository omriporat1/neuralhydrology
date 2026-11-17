"""Water-year rotation run configs, score-blind per-epoch evaluation and the
runner (``src/baseline/wy_rotation_runs.py``, ``scripts/run_wy_rotation_cycle.py``,
``scripts/prepare_wy_rotation_runs.py``).

NeuralHydrology is not needed: training and the NH tester are replaced by
fakes that leave the same files NH and the sealed loader leave (run dir,
``model_epochNNN.pt``, ``output.log``, sealing receipts,
``validation/model_epochNNN/validation_results.p``). The campaign tests use
the real frozen P19 config + generation manifest when present locally and
skip otherwise. No test reads a sealed scope or compares sim with obs.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from src.baseline import nh_raw_space_evaluation as nre
from src.baseline import pilot_orchestration as po
from src.baseline import wy_rotation_runs as rot
from src.baseline import wy_rotation_split as wy

ROOT = Path(__file__).resolve().parents[1]
P19_DIR = ROOT / ".scratch_local" / "seqlen_common336_prep_20261014" / "inputs"
P19_CONFIG = P19_DIR / "p19_config.yaml"
P19_MANIFEST = P19_DIR / "p19_generation_manifest.json"
TARGET = "qobs_mm_per_h_lead06"
needs_p19 = pytest.mark.skipif(not (P19_CONFIG.is_file() and P19_MANIFEST.is_file()),
                               reason="frozen P19 config/manifest not present locally")


# --------------------------------------------------------------------------- helpers

def _safe_read_run_config(run_dir):
    path = Path(run_dir) / "config.yml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


@pytest.fixture(autouse=True)
def _no_ruamel(monkeypatch):
    # The local venv has no ruamel.yaml (NH's reader); the run config is plain YAML.
    monkeypatch.setattr(nre, "_read_run_config", _safe_read_run_config)


@pytest.fixture(scope="module")
def contract_path(tmp_path_factory) -> Path:
    return wy.write_split_contract(tmp_path_factory.mktemp("contract") / "split_contract_v001.json",
                                   wy.build_split_contract())


def _results_dataset(dates, *, sim_finite=True, obs_finite=True) -> xr.Dataset:
    n = len(dates)
    sim = np.full((n, 1), 0.5) if sim_finite else np.full((n, 1), np.nan)
    obs = np.full((n, 1), 0.25) if obs_finite else np.full((n, 1), np.nan)
    return xr.Dataset({f"{TARGET}_obs": (("date", "time_step"), obs),
                       f"{TARGET}_sim": (("date", "time_step"), sim)},
                      coords={"date": pd.DatetimeIndex(dates), "time_step": [0]})


def _write_results(nh_run_dir: Path, epoch: int, basins, dates, *, drop=(), nan_sim=(), nan_obs=()) -> Path:
    path = rot.results_pickle_path(nh_run_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {b: {"1h": {"xr": _results_dataset(dates, sim_finite=b not in nan_sim, obs_finite=b not in nan_obs)}}
               for b in basins if b not in drop}
    with path.open("wb") as fh:
        pickle.dump(results, fh)
    return path


def _write_receipts(nh_run_dir: Path, sealing, n_basins: int, periods=("train", "validation")) -> None:
    for period in periods:
        receipt = dict(sealing.receipt(), period=period)
        if period == "train":
            receipt.update(n_basins=n_basins, n_train_targets_masked_outside_period=0)
        path = nre.split_sealing_receipt_path(nh_run_dir, period)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")


def _log_text(epochs) -> str:
    lines = ["2026-11-15 10:00:00,001: Logging to output.log initialized."]
    lines += [f"2026-11-15 1{e}:00:00,123: Epoch {e} average loss: avg_loss: 0.12345, avg_total_loss: 0.12345"
              for e in epochs]
    return "\n".join(lines) + "\n"


def _synthetic_run(tmp_path: Path, contract_path: Path, *, cycle=1, basins=("01010101", "02020202", "03030303"),
                   epochs=(1, 2), dates=None, extra_config=None) -> dict:
    nh_run_dir = tmp_path / "runs" / "exp_1511_120000"
    nh_run_dir.mkdir(parents=True)
    config = {"dataset": "flashnh", nre.SPLIT_CONTRACT_CONFIG_KEY: contract_path.as_posix(),
              nre.ROTATION_CYCLE_CONFIG_KEY: cycle, "validate_n_random_basins": wy.BASIN_POPULATION_COUNT,
              "validate_every": None, **(extra_config or {})}
    (nh_run_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    sealing = nre.read_run_rotation_sealing(nh_run_dir)
    _write_receipts(nh_run_dir, sealing, len(basins))
    if dates is None:
        start = pd.to_datetime(sealing.nh_dates["validation_start_date"], format="%d/%m/%Y")
        dates = pd.date_range(start, periods=48, freq="h")
    for e in epochs:
        _write_results(nh_run_dir, e, basins, dates)
    (nh_run_dir / "output.log").write_text(_log_text(epochs), encoding="utf-8")
    return {"nh_run_dir": nh_run_dir, "sealing": sealing, "basins": list(basins), "dates": dates}


class FakeTester:
    """Stands in for NH's tester: writes validation_results.p for the epoch."""

    def __init__(self, nh_run_dir, basins, dates):
        self.nh_run_dir, self.basins, self.dates = Path(nh_run_dir), list(basins), dates
        self.calls = []

    def evaluate(self, *, epoch, save_results, metrics):
        self.calls.append({"epoch": epoch, "save_results": save_results, "metrics": metrics})
        _write_results(self.nh_run_dir, epoch, self.basins, self.dates)


# --------------------------------------------------------------------------- identity and derivation

def test_campaign_rows_follow_the_frozen_table():
    rows = rot.campaign_rows()
    assert [r["cycle"] for r in rows] == [c.cycle for c in wy.ROTATION_TABLE_V001] == [1, 2, 3, 4, 5]
    assert [r["index"] for r in rows] == [0, 1, 2, 3, 4]
    assert len({r["run_id"] for r in rows}) == 5
    for r in rows:
        assert r["run_id"] == f"flashnh_wy_rotation_baseline_v001__cycle{r['cycle']}__L120__seedA967139__mf12x50000__attempt001"
        assert r["checkpoint_wy"] not in r["train_wys"] and r["assessment_wy"] not in r["train_wys"]
        assert 2026 not in (*r["train_wys"], r["checkpoint_wy"], r["assessment_wy"])
    with pytest.raises(rot.WyRotationRunsError):
        rot.run_identity(6)


def _base():
    return {"experiment_name": "p19", "run_dir": "/x/runs", "seed": rot.SEED_A, "epochs": 12,
            "max_updates_per_epoch": 50000, "validate_every": 3, "validate_n_random_basins": 1000,
            **{k: "/x/p19_basins.txt" for k in rot.BASIN_FILE_KEYS}, **{k: "01/01/2000" for k in rot.DATE_KEYS},
            "hidden_size": 256}


@pytest.mark.parametrize("cycle", [1, 2, 3, 4, 5])
def test_derive_cycle_config_changes_only_whitelisted_keys(cycle, contract_path):
    derived = rot.derive_cycle_config(_base(), cycle=cycle, contract_path=contract_path.as_posix(),
                                      basin_file="/y/development_train.txt", experiment_name="e", run_dir="/y/runs")
    assert set(rot.config_differences(_base(), derived)) <= rot.ALLOWED_CHANGED_KEYS
    table = {c.cycle: c for c in wy.ROTATION_TABLE_V001}
    assert {k: derived[k] for k in rot.DATE_KEYS} == wy.cycle_nh_dates(table[cycle])
    assert derived["validate_n_random_basins"] == 2307 and derived["validate_every"] is None
    assert derived[nre.ROTATION_CYCLE_CONFIG_KEY] == cycle and type(derived[nre.ROTATION_CYCLE_CONFIG_KEY]) is int
    assert all(derived[k] == "/y/development_train.txt" for k in rot.BASIN_FILE_KEYS)
    assert derived["hidden_size"] == 256 and derived["epochs"] == 12 and derived["max_updates_per_epoch"] == 50000


@pytest.mark.parametrize("edit, match", [
    ({"metrics": ["NSE"]}, "forbidden"),
    ({"experiment_logger": "wandb"}, "forbidden"),
    ({"per_basin_train_periods_file": "/p.pkl"}, "forbidden"),
    ({nre.SPLIT_CONTRACT_CONFIG_KEY: "/c.json"}, "already carries"),
])
def test_derive_cycle_config_refusals(edit, match, contract_path):
    with pytest.raises(rot.WyRotationRunsError, match=match):
        rot.derive_cycle_config({**_base(), **edit}, cycle=1, contract_path=contract_path.as_posix(),
                                basin_file="/b", experiment_name="e", run_dir="/r")
    with pytest.raises(rot.WyRotationRunsError, match="not in the frozen rotation table"):
        rot.derive_cycle_config(_base(), cycle=0, contract_path="/c", basin_file="/b", experiment_name="e",
                                run_dir="/r")


def test_derive_canary_config_changes_only_budget_and_identity():
    cycle_config = {**_base(), "validate_every": None}
    canary = rot.derive_canary_config(cycle_config, epochs=2, max_updates_per_epoch=2000, experiment_name="c",
                                      run_dir="/c/runs")
    assert set(rot.config_differences(cycle_config, canary)) == rot.CANARY_CHANGED_KEYS
    for epochs, updates in [(13, 10), (2, 50001), (0, 10)]:
        with pytest.raises(rot.WyRotationRunsError, match="budget"):
            rot.derive_canary_config(cycle_config, epochs=epochs, max_updates_per_epoch=updates,
                                     experiment_name="c", run_dir="/c/runs")


# --------------------------------------------------------------------------- campaign from the real P19 config

def _prepare(output_root: Path) -> dict:
    return rot.prepare_campaign(base_config_path=P19_CONFIG, base_manifest_path=P19_MANIFEST,
                                output_root=output_root, runtime_project_root=ROOT)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    if not (P19_CONFIG.is_file() and P19_MANIFEST.is_file()):
        pytest.skip("frozen P19 config/manifest not present locally")
    out = tmp_path_factory.mktemp("campaign")
    result = _prepare(out)
    return {"root": out, "result": result, "manifest_path": out / "campaign_manifest.json"}


@needs_p19
def test_prepare_campaign_writes_five_sealed_cycle_configs(campaign):
    result = campaign["result"]
    manifest = json.loads(campaign["manifest_path"].read_text(encoding="utf-8"))
    assert manifest == json.loads(json.dumps(result))
    assert manifest["split_contract"]["checksum_sha256"] == nre.FROZEN_SPLIT_CONTRACT_CHECKSUMS[wy.CONTRACT_ID]
    assert manifest["basin_population"]["count"] == 2307 and manifest["p19_basin_files_opened"] is False
    table = {c.cycle: c for c in wy.ROTATION_TABLE_V001}
    p19 = yaml.safe_load(P19_CONFIG.read_text(encoding="utf-8"))
    for index, row in enumerate(manifest["runs"]):
        checked = rot.verify_campaign_row(manifest, index)
        config = checked["config"]
        assert rot.sha256_file(checked["config_path"]) == row["config_sha256"]
        assert set(rot.config_differences(p19, config)) <= rot.ALLOWED_CHANGED_KEYS
        assert {k: config[k] for k in rot.DATE_KEYS} == wy.cycle_nh_dates(table[row["cycle"]]) == row["nh_dates"]
        # never WY2026: every date ends by 30/09/2025
        ends = [pd.to_datetime(d, format="%d/%m/%Y") for k in ("train_end_date", "validation_end_date",
                                                                "test_end_date")
                for d in (config[k] if isinstance(config[k], list) else [config[k]])]
        assert max(ends) <= pd.Timestamp("2025-09-30")
        assert (config["epochs"], config["max_updates_per_epoch"], config["seed"]) == (12, 50000, rot.SEED_A)
        assert config["dataset"] == "flashnh" and config["validate_every"] is None
        assert config["validate_n_random_basins"] == 2307
        sealing = nre.parse_rotation_sealing(config)
        assert sealing.cycle == row["cycle"] and sealing.lead_hours == 6
        assert not any(k in config for k in ("metrics", "experiment_logger", "use_wandb"))


@needs_p19
def test_prepare_campaign_is_idempotent_and_refuses_to_overwrite(tmp_path):
    first = _prepare(tmp_path)
    assert _prepare(tmp_path) == first  # identical bytes: accepted, nothing rewritten
    config_path = Path(first["runs"][2]["config_dir"]) / "config.yaml"
    config_path.write_text(config_path.read_text(encoding="utf-8").replace("hidden_size", "hidden_size_x"),
                           encoding="utf-8")
    with pytest.raises(rot.WyRotationRunsError, match="refusing to overwrite"):
        _prepare(tmp_path)


@needs_p19
@pytest.mark.parametrize("target", ["config", "contract", "population_path"])
def test_verify_campaign_row_detects_tampering(tmp_path, target):
    result = _prepare(tmp_path)
    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text(encoding="utf-8"))
    if target == "config":
        path = Path(result["runs"][0]["config_dir"]) / "config.yaml"
        path.write_text(path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
        index, match = 0, "changed or missing"
    elif target == "contract":
        path = Path(manifest["split_contract"]["path"])
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        index, match = 3, "changed or missing"
    else:
        manifest["basin_population"]["path"] = (tmp_path / "missing.txt").as_posix()
        index, match = 1, "changed or missing"
    with pytest.raises(rot.WyRotationRunsError, match=match):
        rot.verify_campaign_row(manifest, index)


@needs_p19
def test_prepare_campaign_refuses_a_base_that_is_not_p19(tmp_path):
    other = tmp_path / "p19_edited.yaml"
    other.write_text(P19_CONFIG.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(rot.WyRotationRunsError, match="not the frozen P19 config"):
        rot.prepare_campaign(base_config_path=other, base_manifest_path=P19_MANIFEST, output_root=tmp_path / "o",
                             runtime_project_root=ROOT)


@needs_p19
def test_prepare_and_verify_canary(campaign, tmp_path):
    canary = rot.prepare_canary(campaign_manifest_path=campaign["manifest_path"], cycle=1, epochs=2,
                                max_updates_per_epoch=2000, output_root=tmp_path)
    assert set(canary["config_differences_vs_campaign_row"]) == rot.CANARY_CHANGED_KEYS
    manifest = json.loads((tmp_path / "canary_manifest.json").read_text(encoding="utf-8"))
    checked = rot.verify_canary_manifest(manifest)
    assert checked["row"]["is_canary"] is True and checked["config"]["epochs"] == 2
    assert checked["config"]["max_updates_per_epoch"] == 2000
    assert checked["config"]["validate_n_random_basins"] == 2307
    with pytest.raises(rot.WyRotationRunsError, match="budget"):
        rot.prepare_canary(campaign_manifest_path=campaign["manifest_path"], cycle=1, epochs=13,
                           max_updates_per_epoch=10, output_root=tmp_path / "too_big")
    config_path = checked["config_path"]
    config_path.write_text(config_path.read_text(encoding="utf-8").replace("seed: 967139", "seed: 1"),
                           encoding="utf-8")
    with pytest.raises(rot.WyRotationRunsError, match="changed or missing"):
        rot.verify_canary_manifest(manifest)


def test_prepare_cli_refuses_output_outside_the_project_root():
    spec = importlib.util.spec_from_file_location("prepare_wy_rotation_runs",
                                                  ROOT / "scripts" / "prepare_wy_rotation_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit, match="outside the project root"):
        module.main(["configs", "--base-config", "x", "--base-manifest", "y", "--output-root",
                     str(ROOT.parent / "elsewhere")])


# --------------------------------------------------------------------------- per-epoch evaluation (score-blind)

def test_evaluate_all_epochs_uses_one_tester_no_metrics_and_skips_done_epochs(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, epochs=(1,))
    built = []

    def factory(nh_run_dir):
        built.append(nh_run_dir)
        return FakeTester(nh_run_dir, run["basins"], run["dates"])

    records = rot.evaluate_all_epochs(run["nh_run_dir"], [3, 1, 2], tester_factory=factory)
    assert len(built) == 1
    assert [r["epoch"] for r in records] == [1, 2, 3]
    assert [r["skipped_existing"] for r in records] == [True, False, False]
    assert all(rot.results_pickle_path(run["nh_run_dir"], e).is_file() for e in (1, 2, 3))
    assert not list(run["nh_run_dir"].rglob("*metrics*"))
    # nothing left to do: no tester is built at all
    assert all(r["skipped_existing"] for r in rot.evaluate_all_epochs(run["nh_run_dir"], [1, 2, 3],
                                                                      tester_factory=factory))
    assert len(built) == 1


def test_evaluate_all_epochs_passes_empty_metrics(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, epochs=())
    tester = FakeTester(run["nh_run_dir"], run["basins"], run["dates"])
    rot.evaluate_all_epochs(run["nh_run_dir"], [1, 2], tester_factory=lambda _: tester)
    assert tester.calls == [{"epoch": 1, "save_results": True, "metrics": []},
                            {"epoch": 2, "save_results": True, "metrics": []}]


@pytest.mark.parametrize("extra, match", [
    ({"validate_n_random_basins": 1000}, "validate_n_random_basins must be 2307"),
    ({"metrics": ["NSE"]}, "forbidden"),
    ({"experiment_logger": "wandb"}, "forbidden"),
])
def test_evaluate_all_epochs_refuses_unsafe_run_configs(tmp_path, contract_path, extra, match):
    run = _synthetic_run(tmp_path, contract_path, epochs=(), extra_config=extra)
    with pytest.raises(rot.WyRotationRunsError, match=match):
        rot.evaluate_all_epochs(run["nh_run_dir"], [1], tester_factory=lambda _: pytest.fail("built a tester"))


def test_evaluate_all_epochs_fails_when_nh_writes_nothing(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, epochs=())

    class Silent:
        def evaluate(self, **kwargs):
            pass

    with pytest.raises(rot.WyRotationRunsError, match="wrote no"):
        rot.evaluate_all_epochs(run["nh_run_dir"], [1], tester_factory=lambda _: Silent())


def test_log_epoch_timestamps_returns_times_only(tmp_path):
    log = tmp_path / "output.log"
    log.write_text(_log_text([1, 2, 3]) + "2026-11-15 13:30:00,000: something else\n", encoding="utf-8")
    stamps = rot.log_epoch_timestamps(log)
    assert stamps == {1: "2026-11-15 11:00:00", 2: "2026-11-15 12:00:00", 3: "2026-11-15 13:00:00"}


def test_describe_results_pickle_counts_only(tmp_path):
    dates = pd.date_range("2021-10-01", periods=5, freq="h")
    path = _write_results(tmp_path, 1, ["a", "b"], dates, nan_sim={"b"})
    described = rot.describe_results_pickle(path, target=TARGET)
    assert described["a"]["1h"] == {"n_dates": 5, "first_date": str(dates[0].to_datetime64()),
                                    "last_date": str(dates[-1].to_datetime64()), "sim_shape": [5, 1],
                                    "n_sim_finite": 5, "n_obs_finite": 5, "extra_keys": [],
                                    "dates_increasing": True}
    assert described["b"]["1h"]["n_sim_finite"] == 0


def test_verify_score_blind_run_passes_a_clean_run(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1, 2))
    evidence = rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1, 2], population=run["basins"])
    assert evidence["passed"], evidence["problems"]
    assert set(evidence["per_epoch"]) == {1, 2}
    assert all(r["n_basins"] == 3 and r["basins_equal_population"] for r in evidence["per_epoch"].values())
    assert evidence["checkpoint_window"] == ["2021-10-01 00:00:00", "2022-09-30 23:00:00"]
    # 48 synthetic hours: coverage of the year is recorded (every basin partial), not gated.
    assert all(r["n_basins_partial_window"] == 3 and r["n_basins_dates_not_increasing"] == 0
               for r in evidence["per_epoch"].values())
    flat = json.dumps(evidence, default=str).lower()
    assert "nse" not in flat and "loss" not in flat


@pytest.mark.parametrize("damage, match", [
    ("missing_basin", "differs between epochs"),
    ("extra_basin", "outside the population"),
    ("nan_obs", "no finite observation"),
    ("nan_sim", "no finite simulation"),
    ("outside_window", "leave the checkpoint-year window"),
    ("metrics_file", "metric file"),
    ("score_line", "score line"),
    ("test_folder", "test-period"),
    ("missing_epoch", "no validation_results.p"),
    ("train_count", "train receipt n_basins"),
    ("unsorted_dates", "not strictly increasing"),
])
def test_verify_score_blind_run_reports_problems(tmp_path, contract_path, damage, match):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1, 2))
    d, basins = run["nh_run_dir"], run["basins"]
    if damage == "missing_basin":
        _write_results(d, 2, basins, run["dates"], drop={basins[0]})
    elif damage == "extra_basin":
        _write_results(d, 2, basins + ["09090909"], run["dates"])
    elif damage == "nan_obs":
        _write_results(d, 2, basins, run["dates"], nan_obs={basins[1]})
    elif damage == "nan_sim":
        _write_results(d, 2, basins, run["dates"], nan_sim={basins[1]})
    elif damage == "outside_window":
        _write_results(d, 2, basins, pd.date_range("2021-09-30 22:00", periods=4, freq="h"))
    elif damage == "metrics_file":
        (rot.results_pickle_path(d, 1).parent / "validation_metrics.csv").write_text("x", encoding="utf-8")
    elif damage == "score_line":
        with (d / "output.log").open("a", encoding="utf-8") as fh:
            fh.write("2026-11-15 14:00:00,000: Epoch 2 average validation loss: 0.3\n")
    elif damage == "test_folder":
        (d / "test").mkdir()
    elif damage == "missing_epoch":
        rot.results_pickle_path(d, 2).unlink()
    elif damage == "train_count":
        _write_receipts(d, run["sealing"], 2, periods=("train",))
    elif damage == "unsorted_dates":
        _write_results(d, 2, basins, run["dates"][::-1])
    evidence = rot.verify_score_blind_run(d, cycle=1, epochs=[1, 2], population=basins)
    assert not evidence["passed"]
    assert any(match in p for p in evidence["problems"]), evidence["problems"]


def _drop_in_all_epochs(run, dropped, epochs=(1, 2)):
    for e in epochs:
        _write_results(run["nh_run_dir"], e, run["basins"], run["dates"], drop=set(dropped))


def test_verify_accepts_missing_basins_explained_by_zero_admissible_targets(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1, 2))
    _drop_in_all_epochs(run, {run["basins"][0], run["basins"][2]})
    calls = []

    def counter(nh_run_dir, basins, *, sealing, target, project_root):
        calls.append((list(basins), target, sealing.cycle))
        return {b: 0 for b in basins}

    evidence = rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1, 2], population=run["basins"],
                                          missing_basin_counter=counter)
    assert evidence["passed"], evidence["problems"]
    assert calls == [([run["basins"][0], run["basins"][2]], TARGET, 1)]  # only the missing ones, once
    assert evidence["missing_basins"] == [run["basins"][0], run["basins"][2]]
    assert evidence["missing_basin_admissible_target_counts"] == {run["basins"][0]: 0, run["basins"][2]: 0}
    assert evidence["n_basins_evaluated"] == 1
    assert all(r["n_basins_missing"] == 2 and not r["basins_equal_population"]
               for r in evidence["per_epoch"].values())


def test_verify_clean_run_never_calls_the_counter(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1, 2))

    def counter(*a, **k):
        raise AssertionError("no missing basin, no read")

    evidence = rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1, 2], population=run["basins"],
                                          missing_basin_counter=counter)
    assert evidence["passed"] and evidence["missing_basins"] == [] and evidence["n_basins_evaluated"] == 3


@pytest.mark.parametrize("counts", [{"01010101": 5}, {}, {"01010101": None}])
def test_verify_rejects_unexplained_missing_basins(tmp_path, contract_path, counts):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1, 2))
    _drop_in_all_epochs(run, {"01010101"})
    evidence = rot.verify_score_blind_run(
        run["nh_run_dir"], cycle=1, epochs=[1, 2], population=run["basins"],
        missing_basin_counter=lambda *a, **k: {b: (-1 if n is None else n) for b, n in counts.items()})
    assert not evidence["passed"]
    assert any("have admissible checkpoint-year targets" in p and "01010101" in p for p in evidence["problems"])
    assert evidence["missing_basins"] == ["01010101"]


def _write_package_basin(data_dir: Path, basin: str, finite_rows) -> None:
    dates = pd.date_range("2021-09-25 00:00", "2022-10-05 23:00", freq="h")
    lead = np.full(len(dates), np.nan)
    qobs = np.full(len(dates), 1.0)  # raw discharge finite everywhere: must not count
    for ts in finite_rows:
        lead[dates.get_loc(pd.Timestamp(ts))] = 0.3
    ts_dir = data_dir / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset({TARGET: (("date",), lead), "qobs_m3s": (("date",), qobs),
                "precip_mm": (("date",), np.zeros(len(dates)))},
               coords={"date": dates}).to_netcdf(ts_dir / f"{basin}.nc")


def test_default_counter_counts_only_admissible_issue_times(tmp_path, contract_path):
    data_dir = tmp_path / "package"
    # Finite lead-06 targets only where the issue time or the observation time leaves WY2022, or in
    # the sealed years: none is admissible.
    _write_package_basin(data_dir, "01010101", ["2021-09-30 18:00", "2021-09-30 23:00", "2022-09-30 18:00",
                                                "2022-09-30 23:00", "2022-10-02 00:00"])
    # Admissible: first and last issue times of the window, and one mid-year hour.
    _write_package_basin(data_dir, "02020202", ["2021-10-01 00:00", "2022-03-01 12:00", "2022-09-30 17:00",
                                                "2022-09-30 18:00"])
    _write_package_basin(data_dir, "03030303", [])
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1,),
                         extra_config={"data_dir": data_dir.as_posix()})
    counts = rot.count_admissible_checkpoint_targets(run["nh_run_dir"], ["03030303", "01010101", "02020202"],
                                                     sealing=run["sealing"], target=TARGET)
    assert counts == {"01010101": 0, "02020202": 3, "03030303": 0}

    _drop_in_all_epochs(run, {"01010101", "03030303"}, epochs=(1,))
    evidence = rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1], population=run["basins"])
    assert evidence["passed"], evidence["problems"]
    _drop_in_all_epochs(run, {"02020202"}, epochs=(1,))
    evidence = rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1], population=run["basins"])
    assert not evidence["passed"] and "02020202=3" in " ".join(evidence["problems"])


def test_default_counter_fails_closed_without_a_netcdf(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1,),
                         extra_config={"data_dir": (tmp_path / "nowhere").as_posix()})
    with pytest.raises(rot.WyRotationRunsError, match="expected one NetCDF"):
        rot.count_admissible_checkpoint_targets(run["nh_run_dir"], ["01010101"], sealing=run["sealing"],
                                                target=TARGET)
    # Inside the verifier the failure is a problem, and the missing list is still recorded.
    _drop_in_all_epochs(run, {"01010101"}, epochs=(1,))
    evidence = rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1], population=run["basins"])
    assert not evidence["passed"]
    assert evidence["missing_basins"] == ["01010101"]
    assert any("missing-basin count failed" in p and "expected one NetCDF" in p for p in evidence["problems"])


def test_default_counter_issue_times_match_the_split_admission_rule(tmp_path, contract_path):
    # A target finite at every row: the count must equal the issue hours whose target hour the
    # split contract's slice rule admits (admitted_target_windows shifted back by the lead).
    data_dir = tmp_path / "package"
    dates = pd.date_range("2021-09-25 00:00", "2022-10-05 23:00", freq="h")
    _write_package_basin(data_dir, "01010101", list(dates))
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1,),
                         extra_config={"data_dir": data_dir.as_posix()})
    cycle = next(c for c in wy.ROTATION_TABLE_V001 if c.cycle == 1)
    lead = pd.Timedelta(hours=run["sealing"].lead_hours)
    expected = sum(len(pd.date_range(w.first - lead, w.last - lead, freq="h"))
                   for w in wy.admitted_target_windows(cycle, "validation"))
    counts = rot.count_admissible_checkpoint_targets(run["nh_run_dir"], ["01010101"], sealing=run["sealing"],
                                                     target=TARGET)
    assert counts == {"01010101": expected}
    assert expected == 365 * 24 - 6


def test_verify_score_blind_run_fails_closed_on_sealing(tmp_path, contract_path):
    run = _synthetic_run(tmp_path, contract_path, cycle=1, epochs=(1,))
    with pytest.raises(rot.WyRotationRunsError, match="not sealed for cycle 2"):
        rot.verify_score_blind_run(run["nh_run_dir"], cycle=2, epochs=[1], population=run["basins"])
    nre.split_sealing_receipt_path(run["nh_run_dir"], "validation").unlink()
    with pytest.raises(nre.RawSpaceEvaluationError, match="no split-sealing receipt for the validation"):
        rot.verify_score_blind_run(run["nh_run_dir"], cycle=1, epochs=[1], population=run["basins"])


# --------------------------------------------------------------------------- runner end to end (fake NH)

def _load_runner():
    spec = importlib.util.spec_from_file_location("run_wy_rotation_cycle",
                                                  ROOT / "scripts" / "run_wy_rotation_cycle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def canary(tmp_path, monkeypatch):
    """A 2-epoch canary of cycle 1 from the real P19 config. On Windows the
    NH run tree (<canary>/runs/<run_id>_DDMM_HHMMSS/validation/model_epochNNN/
    validation_results.p) overflows MAX_PATH under this deep project root, so
    the campaign id is shortened for the test (all identity checks still run
    against the patched id); Moriah (Linux) has no such limit."""
    if not (P19_CONFIG.is_file() and P19_MANIFEST.is_file()):
        pytest.skip("frozen P19 config/manifest not present locally")
    monkeypatch.setattr(rot, "CAMPAIGN_ID", "wyr")
    run_id = "wyr__canary__cycle1__e2x2000"
    longest = tmp_path / "k" / run_id / "runs" / f"{run_id}_1511_120000" / "validation" / "model_epoch002" / \
        "validation_results.p.tmp"
    if len(str(longest)) >= 260 and __import__("os").name == "nt":
        pytest.skip(f"Windows MAX_PATH: {len(str(longest))} chars; use a shorter --basetemp")
    _prepare(tmp_path / "c")
    rot.prepare_canary(campaign_manifest_path=tmp_path / "c" / "campaign_manifest.json", cycle=1, epochs=2,
                       max_updates_per_epoch=2000, output_root=tmp_path / "k")
    return tmp_path / "k" / "canary_manifest.json"


@needs_p19
def test_runner_end_to_end_with_fake_nh(canary, monkeypatch):
    from src.baseline import sweep_v1_runtime_contract as rc

    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    population = wy.verify_basin_population(wy.load_split_contract(manifest["split_contract"]["path"]), ROOT)
    calls = {"train": 0, "tester": 0}
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})

    def fake_train(request):
        calls["train"] += 1
        config = yaml.safe_load(Path(request.config_path).read_text(encoding="utf-8"))
        nh_run_dir = Path(config["run_dir"]) / f"{config['experiment_name']}_1511_120000"
        nh_run_dir.mkdir(parents=True)
        (nh_run_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
        for e in range(1, request.additional_epochs + 1):
            (nh_run_dir / f"model_epoch{e:03d}.pt").write_bytes(b"w")
        (nh_run_dir / "output.log").write_text(_log_text(range(1, request.additional_epochs + 1)), encoding="utf-8")
        _write_receipts(nh_run_dir, nre.read_run_rotation_sealing(nh_run_dir), len(population), periods=("train",))

    def fake_tester_factory(nh_run_dir):
        calls["tester"] += 1
        sealing = nre.read_run_rotation_sealing(nh_run_dir)
        _write_receipts(nh_run_dir, sealing, len(population), periods=("validation",))
        start = pd.to_datetime(sealing.nh_dates["validation_start_date"], format="%d/%m/%Y")
        return FakeTester(nh_run_dir, population, pd.date_range(start, periods=3, freq="h"))

    monkeypatch.setattr(po, "default_train_chunk", fake_train)
    monkeypatch.setattr(runner, "_tester_factory", fake_tester_factory)
    assert runner.main(["--manifest", str(canary), "--expected-commit", "abc"]) == 0
    config_dir = Path(manifest["config_dir"])
    receipt = json.loads((config_dir / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    evidence = json.loads((config_dir / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE" and receipt["is_canary"] and receipt["trained_in_this_job"]
    assert receipt["git_commit"] == "abc" and set(receipt["epoch_end_timestamps"]) == {"1", "2"}
    assert evidence["passed"] and {r["n_basins"] for r in evidence["per_epoch"].values()} == {2307}
    assert calls == {"train": 1, "tester": 1}
    # a COMPLETE receipt makes a rerun a no-op
    assert runner.main(["--manifest", str(canary), "--expected-commit", "abc"]) == 0
    assert calls == {"train": 1, "tester": 1}


@needs_p19
def test_runner_refuses_an_incomplete_checkpoint_set_and_records_failed(canary, monkeypatch):
    from src.baseline import sweep_v1_runtime_contract as rc

    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    nh_run_dir = Path(manifest["config_dir"]) / "runs" / f"{manifest['run_id']}_1511_120000"
    nh_run_dir.mkdir(parents=True)
    (nh_run_dir / "model_epoch001.pt").write_bytes(b"w")  # interrupted after epoch 1 of 2
    monkeypatch.setattr(po, "default_train_chunk", lambda request: pytest.fail("retrained"))
    with pytest.raises(rot.WyRotationRunsError, match="not the flat 1..2 set"):
        runner.main(["--manifest", str(canary), "--expected-commit", "abc"])
    receipt = json.loads((Path(manifest["config_dir"]) / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED" and receipt["trained_in_this_job"] is False


def test_runner_refuses_on_provenance_failure(tmp_path, monkeypatch):
    from src.baseline import sweep_v1_runtime_contract as rc

    runner = _load_runner()

    def refuse(**kw):
        raise rc.RuntimeContractError("dirty tree")

    monkeypatch.setattr(rc, "verify_commit_and_interpreter", refuse)
    with pytest.raises(SystemExit, match="provenance"):
        runner.main(["--manifest", str(tmp_path / "none.json"), "--expected-commit", "abc"])


# --------------------------------------------------------------------------- runner with W&B stage A (fake wandb)

def _tracked_policy(tmp_path) -> Path:
    policy = yaml.safe_load((ROOT / "config" / "wy_rotation_wandb_policy_v001.yaml").read_text(encoding="utf-8"))
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return path


def _fake_nh(monkeypatch, runner, manifest):
    population = wy.verify_basin_population(wy.load_split_contract(manifest["split_contract"]["path"]), ROOT)

    def fake_train(request):
        config = yaml.safe_load(Path(request.config_path).read_text(encoding="utf-8"))
        nh_run_dir = Path(config["run_dir"]) / f"{config['experiment_name']}_1511_120000"
        nh_run_dir.mkdir(parents=True)
        (nh_run_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
        for e in range(1, request.additional_epochs + 1):
            (nh_run_dir / f"model_epoch{e:03d}.pt").write_bytes(b"w")
        (nh_run_dir / "output.log").write_text(_log_text(range(1, request.additional_epochs + 1)), encoding="utf-8")
        _write_receipts(nh_run_dir, nre.read_run_rotation_sealing(nh_run_dir), len(population), periods=("train",))

    def fake_tester_factory(nh_run_dir):
        sealing = nre.read_run_rotation_sealing(nh_run_dir)
        _write_receipts(nh_run_dir, sealing, len(population), periods=("validation",))
        start = pd.to_datetime(sealing.nh_dates["validation_start_date"], format="%d/%m/%Y")
        return FakeTester(nh_run_dir, population, pd.date_range(start, periods=3, freq="h"))

    monkeypatch.setattr(po, "default_train_chunk", fake_train)
    monkeypatch.setattr(runner, "_tester_factory", fake_tester_factory)


@needs_p19
def test_runner_with_wandb_policy_tracks_stage_a_only(canary, monkeypatch, tmp_path):
    from src.baseline import sweep_v1_runtime_contract as rc
    from src.baseline import wy_rotation_tracking as trk
    from tests.test_wy_rotation_tracking import install_fake_wandb

    fake = install_fake_wandb(monkeypatch)
    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    _fake_nh(monkeypatch, runner, manifest)
    assert runner.main(["--manifest", str(canary), "--expected-commit", "abc",
                        "--wandb-policy", str(_tracked_policy(tmp_path))]) == 0
    config_dir = Path(manifest["config_dir"])
    receipt = json.loads((config_dir / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE"
    assert receipt["wandb"]["backend"] == "wandb" and not receipt["wandb"]["degraded"]
    assert receipt["wandb"]["id"] == trk.wandb_run_id(manifest["run_id"])
    (run,) = fake.runs
    kw = run.init_kwargs
    assert (kw["name"], kw["group"], kw["job_type"]) == ("canary-c1-base", "canary", "canary")
    assert kw["dir"] == str(config_dir / "wandb") and kw["config"]["checkpoint_wy"] == 2022
    assert kw["config"]["git_commit"] == "abc" and kw["config"]["hyperparameters"]["seq_length"] == 120
    losses = [p for p in run.logs if "train/avg_total_loss" in p]
    assert [p["epoch"] for p in losses] == [1, 2]
    for payload in run.logs:  # every key sent passed the stage-A gate
        trk.check_metric_keys(payload)
    trk.check_metric_keys(run.summary)
    assert run.summary["verify/passed"] is True and run.summary["status/receipt"] == "COMPLETE"
    assert run.uploaded_names() == sorted(
        ["flashnh/config.yaml", f"flashnh/{runner.EVIDENCE_NAME}", f"flashnh/{runner.RECEIPT_NAME}"])
    assert run.exit_code == 0
    # the fake W&B finish writes its own files/config.yaml through any link, as
    # real W&B did in smoke job 46762307: the frozen config must be untouched
    assert rot.sha256_file(config_dir / "config.yaml") == manifest["config_sha256"]
    rot.verify_canary_manifest(manifest)


@needs_p19
def test_runner_completes_when_wandb_init_fails(canary, monkeypatch, tmp_path):
    from src.baseline import sweep_v1_runtime_contract as rc
    from tests.test_wy_rotation_tracking import install_fake_wandb

    install_fake_wandb(monkeypatch, fail_init=True)
    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    _fake_nh(monkeypatch, runner, manifest)
    with pytest.warns(RuntimeWarning):
        assert runner.main(["--manifest", str(canary), "--expected-commit", "abc",
                            "--wandb-policy", str(_tracked_policy(tmp_path))]) == 0
    receipt = json.loads((Path(manifest["config_dir"]) / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE"
    assert receipt["wandb"]["degraded"] and "rotation_init" in receipt["wandb"]["degraded_operations"]


@needs_p19
def test_runner_failure_finishes_the_wandb_run_as_failed(canary, monkeypatch, tmp_path):
    from src.baseline import sweep_v1_runtime_contract as rc
    from tests.test_wy_rotation_tracking import install_fake_wandb

    fake = install_fake_wandb(monkeypatch)
    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    nh_run_dir = Path(manifest["config_dir"]) / "runs" / f"{manifest['run_id']}_1511_120000"
    nh_run_dir.mkdir(parents=True)
    (nh_run_dir / "model_epoch001.pt").write_bytes(b"w")
    monkeypatch.setattr(po, "default_train_chunk", lambda request: pytest.fail("retrained"))
    with pytest.raises(rot.WyRotationRunsError, match="not the flat 1..2 set"):
        runner.main(["--manifest", str(canary), "--expected-commit", "abc",
                     "--wandb-policy", str(_tracked_policy(tmp_path))])
    receipt = json.loads((Path(manifest["config_dir"]) / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    (run,) = fake.runs
    assert receipt["status"] == "FAILED" and run.exit_code == 1 and run.summary["status/receipt"] == "FAILED"
    assert run.uploaded_names() == [f"flashnh/{runner.RECEIPT_NAME}"]  # no stale evidence uploaded


def test_runner_refuses_a_missing_wandb_policy_before_writing_anything(tmp_path, monkeypatch):
    from src.baseline import sweep_v1_runtime_contract as rc
    from src.baseline.wandb_tracking import TrackingError

    runner = _load_runner()
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    with pytest.raises(TrackingError, match="not found"):
        runner.main(["--manifest", str(tmp_path / "none.json"), "--expected-commit", "abc",
                     "--wandb-policy", str(tmp_path / "absent.yaml")])


@needs_p19
def test_runner_keeps_complete_when_a_signal_arrives_during_wandb_close(canary, monkeypatch, tmp_path):
    from src.baseline import sweep_v1_runtime_contract as rc
    from tests.test_wy_rotation_tracking import install_fake_wandb

    fake = install_fake_wandb(monkeypatch)
    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    _fake_nh(monkeypatch, runner, manifest)
    original_init = fake.init

    def init(**kwargs):
        run = original_init(**kwargs)
        run.finish = lambda exit_code=0: runner._raise_on_sigterm(15, None)  # Slurm time limit mid-finish
        return run

    fake.init = init
    with pytest.raises(SystemExit, match="signal 15"):
        runner.main(["--manifest", str(canary), "--expected-commit", "abc",
                     "--wandb-policy", str(_tracked_policy(tmp_path))])
    receipt = json.loads((Path(manifest["config_dir"]) / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "COMPLETE" and "signal 15" in receipt["post_complete_error"]


@needs_p19
def test_runner_records_failed_when_a_signal_arrives_during_wandb_init(canary, monkeypatch, tmp_path):
    from src.baseline import sweep_v1_runtime_contract as rc
    from tests.test_wy_rotation_tracking import install_fake_wandb

    fake = install_fake_wandb(monkeypatch)
    fake.init = lambda **kwargs: runner._raise_on_sigterm(15, None)
    runner = _load_runner()
    manifest = json.loads(canary.read_text(encoding="utf-8"))
    monkeypatch.setattr(rc, "verify_commit_and_interpreter",
                        lambda **kw: {"git_commit": kw["expected_commit"], "runtime_python": "fake"})
    monkeypatch.setattr(po, "default_train_chunk", lambda request: pytest.fail("trained after a signal"))
    with pytest.raises(SystemExit, match="signal 15"):
        runner.main(["--manifest", str(canary), "--expected-commit", "abc",
                     "--wandb-policy", str(_tracked_policy(tmp_path))])
    receipt = json.loads((Path(manifest["config_dir"]) / runner.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED" and "signal 15" in receipt["error"]
