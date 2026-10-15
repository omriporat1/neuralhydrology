"""Focused tests for the common-336 sequence-length support
(src/baseline/seqlen_common336_support.py), its FlashNHDataset training
consumer (src/baseline/nh_dataset.py) and its fixed-support scorer binding.

The vertical test builds a synthetic package with the REAL experiment
constants (floor 336, L in {72, 120, 336}, lead 6) and runs real
NeuralHydrology training + validation inference per L, then checks that the
actual loader admits exactly the same (basin, issue-time) pairs at every L
(with L-hour histories) and that the actual scorer scores every L run on the
same support. It covers the package-start boundary, an internal documented
gap, a basin-specific non-finite input absent from the gap file, a missing
target, and pairs a shorter run would have admitted without the common rule.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baseline import fixed_support_contract_v2 as fixed  # noqa: E402
from src.baseline import seqlen_common336_support as c336  # noqa: E402
from src.baseline.sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2  # noqa: E402

# --------------------------------------------------------------------------- #
# Synthetic package (hour indices relative to 2000-01-01 00:00)
# --------------------------------------------------------------------------- #
START = pd.Timestamp("2000-01-01")
N_HOURS = 1128               # train 0..743, validation 744..1079, test 1080..1127
TRAIN_LAST, VAL_FIRST, VAL_LAST = 743, 744, 1079
LEAD = 6
TARGET = "qobs_mm_per_h_lead06"
DYNAMIC = ["mrms_qpe_1h_mm", "rtma_2t_K"]
AREA_KM2 = 100.0
GAP_HOURS = [200, 900, 1100]                 # documented, NaN in every basin (1100 = test period)
NONFINITE_INPUT = {"B02": [300, 820]}        # basin-specific rtma NaN, NOT in the gap file
TARGET_NAN = {"B03": [650, 651, 652, 653, 654, 655, 850]}
BASINS = ["B01", "B02", "B03"]
IDENTITIES = {"package_manifest_sha256": "a" * 64, "package_file_checksums_sha256": "b" * 64,
              "package_run_provenance_sha256": "c" * 64, "development_split_sha256": "d" * 64,
              "spatial_holdout_split_sha256": "e" * 64}


def _ts(hour: int) -> pd.Timestamp:
    return START + pd.Timedelta(hours=hour)


def _build_package(root: Path) -> Path:
    dates = pd.date_range(START, periods=N_HOURS, freq="h")
    (root / "time_series").mkdir(parents=True)
    (root / "basins").mkdir()
    (root / "masks").mkdir()
    rng = np.random.default_rng(3)
    for k, basin in enumerate(BASINS):
        mrms = rng.gamma(0.5, 1.0, N_HOURS)
        rtma = 270.0 + 10.0 * np.sin(np.arange(N_HOURS) / 24.0) + k
        for h in GAP_HOURS:
            mrms[h] = rtma[h] = np.nan
        for h in NONFINITE_INPUT.get(basin, []):
            rtma[h] = np.nan
        qobs = 5.0 + np.convolve(np.nan_to_num(mrms), np.ones(12), mode="same")
        target = np.full(N_HOURS, np.nan)
        target[:-LEAD] = 3.6 * qobs[LEAD:] / AREA_KM2
        for h in TARGET_NAN.get(basin, []):
            target[h] = np.nan
        xr.Dataset({DYNAMIC[0]: ("date", mrms), DYNAMIC[1]: ("date", rtma), "qobs_m3s": ("date", qobs),
                    TARGET: ("date", target)}, coords={"date": dates}).to_netcdf(root / "time_series" / f"{basin}.nc")
    for period in ("train", "validation", "test"):
        (root / "basins" / f"{period}.txt").write_text("\n".join(BASINS) + "\n")
    (root / "masks" / "gap_timestamps.json").write_text(json.dumps([str(_ts(h)) + "Z" for h in GAP_HOURS]))
    return root


CUTOFF = "2000-02-14T23:00:00"               # = _ts(VAL_LAST), the screening contract's last hour


def _cfg_text(root: Path, *, seq_length: int, name: str, support_file: Path | None, cutoff: str | None = None) -> str:
    support = f"flashnh_common_support_file: {support_file.as_posix()}\n" if support_file else ""
    if cutoff:
        support += f'flashnh_observation_cutoff: "{cutoff}"\n'
    dyn = "\n".join(f"  - {d}" for d in DYNAMIC)
    return f"""experiment_name: {name}
train_basin_file: {(root / 'basins' / 'train.txt').as_posix()}
validation_basin_file: {(root / 'basins' / 'validation.txt').as_posix()}
test_basin_file: {(root / 'basins' / 'test.txt').as_posix()}
run_dir: {(root / 'runs').as_posix()}
data_dir: {root.as_posix()}
dataset: flashnh
train_start_date: "01/01/2000"
train_end_date: "31/01/2000"
validation_start_date: "01/02/2000"
validation_end_date: "14/02/2000"
test_start_date: "15/02/2000"
test_end_date: "16/02/2000"
target_variables:
  - {TARGET}
dynamic_inputs:
{dyn}
model: cudalstm
hidden_size: 4
head: regression
output_activation: linear
predict_last_n: 1
batch_size: 64
optimizer: Adam
learning_rate: 0.001
loss: NSE
epochs: 1
save_weights_every: 1
log_interval: 50
validate_n_random_basins: 3
log_tensorboard: False
num_workers: 0
seq_length: {seq_length}
seed: 967139
device: cpu
verbose: 0
{support}"""


def _config(root: Path, *, seq_length: int, name: str, support_file: Path | None, cutoff: str | None = None):
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    path = root / f"{name}.yml"
    path.write_text(_cfg_text(root, seq_length=seq_length, name=name, support_file=support_file, cutoff=cutoff))
    return Config(path)


def _compute(root: Path, *, train_basins=BASINS, screening_basins=BASINS, seq_lengths=(72, 120, 336)) -> dict:
    gap_path = root / "masks" / "gap_timestamps.json"
    identities = dict(IDENTITIES, gap_artifact_sha256=hashlib.sha256(gap_path.read_bytes()).hexdigest())
    from src.baseline.gap_mask_io import load_gap_timestamps_json

    return c336.compute_common336_support(
        package_root=root, train_basins=train_basins, screening_basins=screening_basins,
        train_period=(_ts(0), _ts(TRAIN_LAST)), validation_period=(_ts(VAL_FIRST), _ts(VAL_LAST)),
        gap_timestamps=load_gap_timestamps_json(gap_path), dynamic_inputs=DYNAMIC, target_variable=TARGET,
        lead_hours=LEAD, seq_lengths=seq_lengths, identities=identities, train_basin_ids_sha256="f" * 64,
        screening_basin_ids_sha256="0" * 64)


def _train_dataset(cfg, base: Path, tag: str):
    from src.baseline.nh_dataset import FlashNHDataset

    run_dir = base / "loader_runs" / tag
    cfg.run_dir = run_dir
    cfg.train_dir = run_dir / "train_data"
    cfg.train_dir.mkdir(parents=True, exist_ok=True)
    return FlashNHDataset(cfg=cfg, is_train=True, period="train")


def _pairs(ds) -> set:
    freq = ds.frequencies[0]
    return {(basin, pd.Timestamp(ds._dates[basin][freq][idx[0]])) for basin, idx in ds.lookup_table.values()}


def _support_pairs(support: dict) -> set:
    return {(b, pd.Timestamp(t)) for b, iv in support["per_basin_intervals"].items()
            for t in c336.decode_issue_intervals(iv)}


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    root = _build_package(tmp_path_factory.mktemp("c336pkg"))
    result = _compute(root)
    support_path = c336.write_training_support(result["training_support"], root / "support" / "train_support.json")
    contract_path = c336.write_common336_screening_contract(result["screening_contract"],
                                                            root / "support" / "screening_contract.json")
    return {"root": root, "result": result, "support_path": support_path, "contract_path": contract_path}


# --------------------------------------------------------------------------- #
# Producer: expected support and loss-cause partition
# --------------------------------------------------------------------------- #

def test_producer_support_and_loss_causes(package):
    support = package["result"]["training_support"]
    pairs = _support_pairs(support)
    by_basin = {b: sorted(t for bb, t in pairs if bb == b) for b in BASINS}
    # B01: 336 windows containing gap hour 200 end at 535; lead boundary 743 - 6.
    assert by_basin["B01"] == list(pd.date_range(_ts(536), _ts(737), freq="h"))
    # B02: its own non-finite rtma hour 300 also blocks issue times up to 635.
    assert by_basin["B02"] == list(pd.date_range(_ts(636), _ts(737), freq="h"))
    # B03: missing targets at 650..655 are excluded at every L.
    assert by_basin["B03"] == [t for t in pd.date_range(_ts(536), _ts(737), freq="h")
                               if not _ts(650) <= t <= _ts(655)]

    rows = {(r["basin_id"], r["role"]): r for r in package["result"]["accounting"]["rows"]}
    b01, b02 = rows[("B01", "train")], rows[("B02", "train")]
    # L=72 natural: start at 71, minus gap windows [200, 271] -> admitted but lost to the common rule.
    assert b01["lost_timeline_start_L72"] == 335 - 71 - 72          # issue 71..334 minus 72 gap-window hours
    assert b01["lost_documented_gap_L72"] == 535 - 335 + 1
    assert b01["lost_nonfinite_input_L72"] == 0
    assert b02["lost_nonfinite_input_L72"] == 635 - 536 + 1 - 0     # 536..635 fail only via hour 300
    assert b02["n_nonfinite_input_hours_not_gap"] == 1              # hour 300 (820 is past train end)
    for row in rows.values():
        assert row["n_natural_L336"] == row["n_common336"]
        assert row["lost_timeline_start_L336"] == row["lost_documented_gap_L336"] == 0
        assert row["lost_nonfinite_input_L336"] == 0

    contract = package["result"]["screening_contract"]
    val = {b: fixed._deserialize_date_array(contract["per_basin_support"][b], contract["date_dtype"])
           for b in BASINS}
    assert pd.DatetimeIndex(val["B01"]).equals(pd.date_range(_ts(744), _ts(899), freq="h"))
    assert pd.DatetimeIndex(val["B02"]).equals(pd.date_range(_ts(744), _ts(819), freq="h"))
    assert len(val["B03"]) == 155  # target NaN at 850


def test_zero_support_basin_is_reported_not_dropped(tmp_path):
    root = _build_package(tmp_path / "pkg")
    with xr.open_dataset(root / "time_series" / "B03.nc") as ds:
        data = ds.load()
    data[TARGET][:] = np.nan
    (root / "time_series" / "B03.nc").unlink()
    data.to_netcdf(root / "time_series" / "B03.nc")
    result = _compute(root)
    assert result["training_support"] is None and result["screening_contract"] is None
    assert {(f["basin_id"], f["role"]) for f in result["accounting"]["failures"]} == {("B03", "train"),
                                                                                      ("B03", "validation")}
    with pytest.raises(c336.Common336SupportError, match="zero Common-336 support"):
        c336.build_training_support(
            per_basin_dates={"B03": pd.date_range(START, periods=3, freq="h")},
            per_basin_admitted={"B03": np.zeros(3, dtype=bool)}, lead_hours=LEAD, target_variable=TARGET,
            dynamic_inputs=DYNAMIC, date_start="2000-01-01", date_end="2000-01-31",
            identities=dict(IDENTITIES, gap_artifact_sha256="1" * 64), train_basin_ids_sha256="f" * 64)


def test_missing_basin_file_is_reported_as_failure(tmp_path):
    root = _build_package(tmp_path / "pkg")
    (root / "time_series" / "B02.nc").unlink()
    result = _compute(root)
    assert result["training_support"] is None
    assert [(f["basin_id"], f["role"]) for f in result["accounting"]["failures"]] == [("B02", "read")]


def test_intervals_round_trip_and_reject_malformed():
    dates = pd.date_range(START, periods=10, freq="h")
    admitted = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    intervals = c336.encode_issue_intervals(dates, admitted)
    assert intervals == [["2000-01-01T01:00:00", "2000-01-01T02:00:00"], ["2000-01-01T05:00:00"] * 2,
                         ["2000-01-01T07:00:00", "2000-01-01T09:00:00"]]
    np.testing.assert_array_equal(c336.decode_issue_intervals(intervals), dates.to_numpy()[admitted])
    for bad in ([["2000-01-01T01:00:00", "2000-01-01T02:00:00"], ["2000-01-01T03:00:00"] * 2],  # adjacent
                [["2000-01-01T05:00:00"] * 2, ["2000-01-01T01:00:00"] * 2],                     # unsorted
                [["2000-01-01T05:00:00", "2000-01-01T04:00:00"]]):                               # reversed
        with pytest.raises(c336.Common336SupportError):
            c336.decode_issue_intervals(bad)


def test_training_support_tamper_detection(package):
    support = json.loads(package["support_path"].read_text())
    c336.validate_training_support(support)
    tampered = json.loads(json.dumps(support))
    tampered["eligible_counts"]["B01"] += 1
    with pytest.raises(c336.Common336SupportError, match="intervals cover"):
        c336.validate_training_support(tampered)
    tampered = json.loads(json.dumps(support))
    tampered["per_basin_intervals"]["B01"][0][0] = str(_ts(537).isoformat())
    tampered["eligible_counts"]["B01"] -= 1
    with pytest.raises(c336.Common336SupportError, match="checksum"):
        c336.validate_training_support(tampered)
    for key, value in (("seq_length_floor", 120), ("support_id", "common120"), ("period", "validation")):
        tampered = dict(support, **{key: value})
        with pytest.raises(c336.Common336SupportError, match=key):
            c336.validate_training_support(tampered)
    with pytest.raises(c336.Common336SupportError, match="refusing to overwrite"):
        c336.write_training_support(support, package["support_path"])


def test_common336_and_common120_contract_identities_are_not_interchangeable(package):
    contract = package["result"]["screening_contract"]
    assert contract["contract_id"] == c336.SCREENING_CONTRACT_ID and contract["seq_length_floor"] == 336
    with pytest.raises(fixed.FixedSupportContractError, match="contract_id"):
        fixed.validate_fixed_support_contract(contract)  # v2 default = Common-120 identity
    with pytest.raises(fixed.FixedSupportContractError, match="contract_id"):
        fixed.load_fixed_support_contract(package["contract_path"])
    common120 = fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=LEAD, target_variable=TARGET, period="validation",
        date_start="2000-02-01", date_end="2000-02-14", source_gap_policy_identity="gap_only_120",
        screening_basin_ids_sha256="0" * 64, per_basin_date={"B01": np.array([_ts(800).to_datetime64()])},
        per_basin_admitted={"B01": np.array([True])}, **IDENTITIES)
    fixed.validate_fixed_support_contract(common120)  # historical identity unchanged
    with pytest.raises(fixed.FixedSupportContractError, match="contract_id"):
        fixed.validate_fixed_support_contract(common120, expected_contract_id=c336.SCREENING_CONTRACT_ID,
                                              expected_seq_length_floor=336)
    relabelled = dict(common120, contract_id=c336.SCREENING_CONTRACT_ID)
    with pytest.raises(fixed.FixedSupportContractError):
        fixed.validate_fixed_support_contract(relabelled, expected_contract_id=c336.SCREENING_CONTRACT_ID,
                                              expected_seq_length_floor=336)


# --------------------------------------------------------------------------- #
# Consumer: the real FlashNHDataset loader
# --------------------------------------------------------------------------- #

def test_loader_admits_identical_pairs_at_every_length(package):
    root, support = package["root"], package["result"]["training_support"]
    expected = _support_pairs(support)
    for L in (72, 120, 336):
        ds = _train_dataset(_config(root, seq_length=L, name=f"loader_L{L}", support_file=package["support_path"]),
                            root, f"common_L{L}")
        assert _pairs(ds) == expected
        stats = ds.flashnh_filter_stats
        assert stats.common_support_id == c336.TRAINING_SUPPORT_ID
        assert stats.common_support_checksum_sha256 == support["checksum_sha256"]
        assert stats.n_kept == len(expected) == stats.n_natural_admitted - stats.removed_for_common_support
        sample = ds[0]
        assert {k: tuple(v.shape) for k, v in sample["x_d"].items()} == {d: (L, 1) for d in DYNAMIC}
        if L == 336:
            assert stats.removed_for_common_support == 0  # common-336 == natural L=336 training lookup

    natural72 = _pairs(_train_dataset(_config(root, seq_length=72, name="loader_nat72", support_file=None),
                                      root, "natural_L72"))
    assert expected < natural72
    would_have_admitted = {("B01", _ts(100)),    # package-start boundary (336 h history unavailable)
                           ("B01", _ts(400)),    # documented gap at 200 inside 336 h, not 72 h
                           ("B02", _ts(600))}    # B02-only non-finite rtma at 300 inside 336 h, not 72 h
    assert would_have_admitted <= natural72 and not would_have_admitted & expected
    assert ("B03", _ts(652)) not in natural72  # missing target excluded naturally


def _write_support(root: Path, payload: dict, name: str) -> Path:
    payload = {k: v for k, v in payload.items() if k != "checksum_sha256"}
    payload["checksum_sha256"] = c336._checksum(payload)
    path = root / "support" / name
    path.write_text(json.dumps(payload))
    return path


def test_loader_fails_closed_on_mismatched_or_unadmitted_support(package):
    from src.baseline.nh_dataset import FlashNHDatasetError

    root, support = package["root"], package["result"]["training_support"]
    bad = json.loads(json.dumps(support))  # a pair the natural loader never admits (B03 missing target)
    bad["per_basin_intervals"]["B03"] = [[str(_ts(536).isoformat()), str(_ts(737).isoformat())]]
    bad["eligible_counts"]["B03"] = 202
    cases = {
        "unadmitted.json": (bad, "lack naturally admitted"),
        "wrong_gap.json": (dict(support, gap_artifact_sha256="9" * 64), "gap_artifact_sha256"),
        "wrong_inputs.json": (dict(support, dynamic_inputs=DYNAMIC[:1]), "dynamic_inputs"),
        "wrong_dates.json": (dict(support, date_end="2000-01-30"), "date_end"),
        "wrong_basins.json": ({**support, "basin_ids": ["B01", "B02"],
                               "per_basin_intervals": {b: support["per_basin_intervals"][b] for b in ("B01", "B02")},
                               "eligible_counts": {b: support["eligible_counts"][b] for b in ("B01", "B02")}},
                              "basin set"),
    }
    for name, (payload, match) in cases.items():
        path = _write_support(root, payload, name)
        cfg = _config(root, seq_length=72, name=f"bad_{name[:-5]}", support_file=path)
        with pytest.raises(FlashNHDatasetError, match=match):
            _train_dataset(cfg, root, f"bad_{name[:-5]}")
    cfg = _config(root, seq_length=72, name="tampered", support_file=package["support_path"])
    tampered = root / "support" / "tampered.json"
    raw = json.loads(package["support_path"].read_text())
    raw["eligible_counts"]["B01"] -= 1
    tampered.write_text(json.dumps(raw))
    cfg.update_config({"flashnh_common_support_file": tampered})
    with pytest.raises(FlashNHDatasetError, match="invalid common training support"):
        _train_dataset(cfg, root, "tampered")


def test_validation_dataset_is_not_restricted(package):
    from src.baseline.nh_dataset import FlashNHDataset

    cfg = _config(package["root"], seq_length=72, name="val_natural", support_file=package["support_path"])
    train = _train_dataset(cfg, package["root"], "val_natural")
    ds = FlashNHDataset(cfg=cfg, is_train=False, period="validation", scaler=train.scaler)
    assert ds.flashnh_filter_stats.common_support_id is None
    assert ds.flashnh_filter_stats.removed_for_common_support == 0


# --------------------------------------------------------------------------- #
# Vertical: real NH training + validation inference, real scorer
# --------------------------------------------------------------------------- #

def test_vertical_training_and_screening_share_support_across_lengths(package):
    from neuralhydrology.evaluation.evaluate import start_evaluation
    from neuralhydrology.training.train import start_training
    from neuralhydrology.utils.config import Config

    root = package["root"]
    contract = c336.load_common336_screening_contract(package["contract_path"])
    expected_train = _support_pairs(package["result"]["training_support"])
    scored, natural_val_dates = {}, {}
    for L in (72, 336):
        name = f"vertical_L{L}"
        start_training(_config(root, seq_length=L, name=name, support_file=package["support_path"], cutoff=CUTOFF))
        (run_dir,) = [p for p in (root / "runs").iterdir() if p.name.startswith(name + "_")]
        run_cfg = Config(run_dir / "config.yml")
        assert Path(run_cfg.flashnh_common_support_file) == package["support_path"]  # survives dump/reload
        from src.baseline import seqlen_common336_campaign as camp
        applied = camp.verify_applied_training_support(run_dir, seq_length=L, seed=967139,
                                                       support=package["result"]["training_support"])
        assert applied["n_kept"] == len(expected_train) and applied["removed_for_common_support"] == (
            0 if L == 336 else applied["n_natural_admitted"] - len(expected_train))
        scaler_train = (run_dir / "train_data" / "train_data_scaler.yml").read_text()
        start_evaluation(run_cfg, run_dir, epoch=1, period="validation")
        result = c336.evaluate_common336_screening(run_dir=run_dir, epoch=1, package_root=root, contract=contract,
                                                   require_full_screening_population=False,
                                                   return_admitted_series=True)
        assert result["n_basins_evaluated"] == 3 and result["n_basins_excluded"] == 0
        assert result["contract_id"] == c336.SCREENING_CONTRACT_ID and result["seq_length_floor"] == 336
        assert result["observation_cutoff"] == str(pd.Timestamp(CUTOFF))
        for row in result["per_basin"]:
            assert row["n_fixed_support_eligible"] == contract["eligible_counts"][row["basin_id"]]
            assert row["n_sim_nonfinite_at_admitted"] == 0  # B02's NaN-input samples are outside the support
        scored[L] = {b: s.date for b, s in result["admitted_series_by_basin"].items()}
        from src.baseline.authenticated_period_results import load_authenticated_period_results

        period = load_authenticated_period_results(run_dir=run_dir, period="validation", epoch=1)
        natural_val_dates[L] = {b: len(next(iter(period[b].values()))["xr"]["date"]) for b in BASINS}
        # The training loader of this very run used the common support.
        ds = _train_dataset(run_cfg, root, f"recheck_L{L}")
        assert _pairs(ds) == expected_train
        scored[f"scaler{L}"] = scaler_train

    for b in BASINS:
        np.testing.assert_array_equal(scored[72][b], scored[336][b])
        support = fixed._deserialize_date_array(contract["per_basin_support"][b], contract["date_dtype"])
        np.testing.assert_array_equal(np.sort(scored[72][b]), np.sort(support))
        assert natural_val_dates[72][b] > natural_val_dates[336][b] >= len(support)
    # NH computes the scaler over the loaded train xarray (not the lookup), so
    # the support does not enter it; L only changes the float32 reduction layout.
    import yaml

    def _flat(text):
        out = {}

        def walk(node, prefix=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}/{key}")
            elif isinstance(node, (int, float)):
                out[prefix] = float(node)
        walk(yaml.safe_load(text))
        return out

    flat72, flat336 = _flat(scored["scaler72"]), _flat(scored["scaler336"])
    assert flat72.keys() == flat336.keys() and flat72
    for key in flat72:
        assert flat72[key] == pytest.approx(flat336[key], rel=1e-6, abs=1e-9), key


def test_common_support_key_rejected_without_registration_is_not_silently_ignored(package, monkeypatch):
    """Without the registered Config property NH's own unknown-key check
    refuses the config, so the key can never be dropped silently."""
    from neuralhydrology.utils.config import Config

    from src.baseline import nh_register

    nh_register.register_flashnh_dataset()
    monkeypatch.delattr(Config, c336.COMMON_SUPPORT_CONFIG_KEY)
    path = package["root"] / "unregistered.yml"
    path.write_text(_cfg_text(package["root"], seq_length=72, name="unreg", support_file=package["support_path"]))
    with pytest.raises(ValueError, match="flashnh_common_support_file"):
        Config(path)


def test_target_is_never_materialized_past_period_end(tmp_path):
    """The lead-shifted target at t holds the observation at t + lead, so the
    producer reads it only up to ``last_date - lead`` (no later observation)."""
    root = _build_package(tmp_path / "pkg")
    dates, values = c336._read_basin(root, "B01", DYNAMIC + [TARGET], _ts(VAL_LAST), LEAD)
    assert dates[-1] == _ts(VAL_LAST) and len(dates) == VAL_LAST + 1
    with xr.open_dataset(root / "time_series" / "B01.nc") as data:
        full = data[TARGET].values[:VAL_LAST + 1]
    target = values[-1]
    np.testing.assert_array_equal(target[:VAL_LAST + 1 - LEAD], full[:VAL_LAST + 1 - LEAD])
    assert np.isnan(target[VAL_LAST + 1 - LEAD:]).all() and np.isfinite(full[VAL_LAST + 1 - LEAD:]).all()
    np.testing.assert_array_equal(values[0], np.asarray(xr.open_dataset(root / "time_series" / "B01.nc")[DYNAMIC[0]].values[:VAL_LAST + 1]))


# --------------------------------------------------------------------------- #
# Campaign identities and config derivation
# --------------------------------------------------------------------------- #

def test_seeds_identities_and_rows():
    from src.baseline import seqlen_common336_campaign as camp

    assert camp.derive_seed_c()["seed"] == camp.SEED_C == int(hashlib.sha256(camp.SEED_C_NAMESPACE.encode()).hexdigest()[:8], 16)
    assert camp.derive_seed(camp.SEED_B_NAMESPACE)["seed"] == camp.SEED_B == 329911494
    assert (camp.SEED_A, camp.SEED_C) == (967139, 3023892610)
    # 1729 is reserved for promoted finalists and must not be a campaign seed.
    assert 1729 not in dict(camp.SEEDS).values()
    assert len({camp.SEED_A, camp.SEED_B, camp.SEED_C, 1729, 20260822, 158987734, 400, 121, 111, 103, 105}) == 11
    clash = camp.derive_seed("collision_probe")["seed"]
    camp._EXISTING_PROJECT_SEEDS[clash] = "probe"
    try:
        with pytest.raises(camp.SeqlenCampaignError, match="collides"):
            camp.derive_seed("collision_probe", "B")
    finally:
        del camp._EXISTING_PROJECT_SEEDS[clash]
    rows = camp.campaign_rows()
    assert len(rows) == 15 and len({r["run_id"] for r in rows}) == 15
    assert [r["seq_length"] for r in rows[:5]] == [72, 120, 168, 240, 336] and {r["seed_label"] for r in rows[:5]} == {"A"}
    assert sum(r["is_reference"] for r in rows) == 3
    assert rows[1]["run_id"] == "seqlen_common336_seed3_v001__L120__seedA967139__mf12x50000__attempt001"
    with pytest.raises(camp.SeqlenCampaignError):
        camp.run_identity(96, "A")


def test_derive_run_config_changes_only_whitelisted_keys():
    from src.baseline import seqlen_common336_campaign as camp

    base = {"seq_length": 120, "seed": 967139, "experiment_name": "p19", "run_dir": "/x/runs",
            "hidden_size": 256, "dynamic_inputs": ["a", "b"], "max_updates_per_epoch": 50000}
    derived = camp.derive_run_config(base, seq_length=336, seed=camp.SEED_B, experiment_name="n", run_dir="/y/runs",
                                     support_file="/s.json", observation_cutoff=CUTOFF)
    assert set(camp.config_differences(base, derived)) == {"seq_length", "seed", "experiment_name", "run_dir",
                                                           c336.COMMON_SUPPORT_CONFIG_KEY,
                                                           camp.OBSERVATION_CUTOFF_CONFIG_KEY}
    assert base["seq_length"] == 120 and derived["dynamic_inputs"] == ["a", "b"]
    with pytest.raises(camp.SeqlenCampaignError):
        camp.derive_run_config(base, seq_length=480, seed=1, experiment_name="n", run_dir="/y", support_file="/s",
                               observation_cutoff=CUTOFF)
    for key in (c336.COMMON_SUPPORT_CONFIG_KEY, camp.OBSERVATION_CUTOFF_CONFIG_KEY):
        with pytest.raises(camp.SeqlenCampaignError):
            camp.derive_run_config({**base, key: "/old"}, seq_length=72, seed=1,
                                   experiment_name="n", run_dir="/y", support_file="/s", observation_cutoff=CUTOFF)


def test_prepare_campaign_writes_15_consumable_configs(package, tmp_path):
    import yaml

    from src.baseline import seqlen_common336_campaign as camp

    root = package["root"]
    train_sha = hashlib.sha256((root / "basins" / "train.txt").read_bytes()).hexdigest()
    identities = {k: package["result"]["training_support"][k] for k in (
        "gap_artifact_sha256", *IDENTITIES)}
    gap = package["result"]["training_support"]
    support = c336.build_training_support(
        per_basin_dates={b: pd.date_range(START, periods=N_HOURS, freq="h") for b in BASINS},
        per_basin_admitted={b: np.isin(pd.date_range(START, periods=N_HOURS, freq="h").to_numpy(),
                                       c336.decode_issue_intervals(gap["per_basin_intervals"][b])) for b in BASINS},
        lead_hours=LEAD, target_variable=TARGET, dynamic_inputs=DYNAMIC, date_start="2000-01-01",
        date_end="2000-01-31", identities=identities, train_basin_ids_sha256=train_sha)
    support_path = c336.write_training_support(support, tmp_path / "support.json")

    base = yaml.safe_load(_cfg_text(root, seq_length=120, name="p19", support_file=None))
    base.update(epochs=12, max_updates_per_epoch=50000)
    base_path = tmp_path / "p19" / "config.yaml"
    base_path.parent.mkdir()
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    manifest = {"lead_hours": LEAD, "git_commit": "0" * 40, "artifact_sha256": {
        "config.yaml": hashlib.sha256(base_path.read_bytes()).hexdigest(), "train_basins.txt": train_sha,
        "validation_basins.txt": hashlib.sha256((root / "basins" / "validation.txt").read_bytes()).hexdigest()}}
    manifest_path = tmp_path / "p19" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    out = tmp_path / "campaign"
    result = camp.prepare_campaign(base_config_path=base_path, base_manifest_path=manifest_path,
                                   training_support_path=support_path, screening_contract_path=package["contract_path"],
                                   output_root=out, expected_screening_count=3)
    assert len(result["runs"]) == 15 and (out / "campaign_manifest.json").is_file()
    assert result["schema_version"] == 2 and result["seeds"] == {"A": 967139, "B": 329911494, "C": 3023892610}
    assert result["observation_cutoff"]["value"] == CUTOFF
    assert result["result_summary_rule"] == camp.RESULT_SUMMARY_RULE
    for run in result["runs"]:
        assert set(run["config_differences_vs_p19"]) <= camp.ALLOWED_CHANGED_KEYS
        text = (Path(run["config_dir"]) / "config.yaml").read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == run["config_sha256"]
        assert yaml.safe_load(text)[camp.OBSERVATION_CUTOFF_CONFIG_KEY] == CUTOFF
        assert camp.verify_campaign_row(result, run["index"])["run_id"] == run["run_id"]
    with pytest.raises(camp.SeqlenCampaignError, match="schema v2"):
        camp.verify_campaign_row({**result, "seeds": {"A": 967139, "B": 1729, "C": 3023892610}}, 0)
    # Re-preparing is idempotent; a changed base config is refused.
    assert camp.prepare_campaign(base_config_path=base_path, base_manifest_path=manifest_path,
                                 training_support_path=support_path, screening_contract_path=package["contract_path"],
                                 output_root=out, expected_screening_count=3)["runs"] == result["runs"]
    base_path.write_text(base_path.read_text().replace("hidden_size: 4", "hidden_size: 8"))
    with pytest.raises(camp.SeqlenCampaignError, match="manifest"):
        camp.prepare_campaign(base_config_path=base_path, base_manifest_path=manifest_path,
                              training_support_path=support_path, screening_contract_path=package["contract_path"],
                              output_root=out, expected_screening_count=3)

    # A generated config is consumable: the real loader applies the support.
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    run = next(r for r in result["runs"] if r["seq_length"] == 72 and r["seed_label"] == "C")
    cfg = Config(Path(run["config_dir"]) / "config.yaml")
    assert cfg.seed == camp.SEED_C and cfg.seq_length == 72 and cfg.flashnh_observation_cutoff == CUTOFF
    ds = _train_dataset(cfg, tmp_path, "campaign_l72")
    assert _pairs(ds) == _support_pairs(support)


# --------------------------------------------------------------------------- #
# Observation cutoff (sealed-period boundary): loader and scorer
# --------------------------------------------------------------------------- #

SENTINEL = 1.0e6


def _poison_after_cutoff(root: Path, cutoff_hour: int = VAL_LAST) -> None:
    """Overwrite every observation later than the cutoff (inputs, qobs, and the
    lead target wherever it encodes a post-cutoff observation) with a huge
    sentinel, so any leak shows up in values, scaler or area."""
    for basin in BASINS:
        path = root / "time_series" / f"{basin}.nc"
        with xr.open_dataset(path) as data:
            ds = data.load()
        for name in (*DYNAMIC, "qobs_m3s"):
            ds[name].values[cutoff_hour + 1:] = SENTINEL
        ds[TARGET].values[cutoff_hour + 1 - LEAD:] = SENTINEL
        ds.to_netcdf(path)


def test_cutoff_loader_reads_nothing_past_cutoff(tmp_path):
    from src.baseline.nh_dataset import FlashNHDataset

    root = _build_package(tmp_path / "pkg")
    with xr.open_dataset(root / "time_series" / "B01.nc") as data:
        clean = data.load()
    _poison_after_cutoff(root)
    cfg = _config(root, seq_length=72, name="cut", support_file=None, cutoff=CUTOFF)
    train = _train_dataset(cfg, tmp_path, "cut_train")
    val = FlashNHDataset(cfg=cfg, is_train=False, period="validation", scaler=train.scaler)
    df = val._load_basin_data("B01")
    assert df.index.max() == pd.Timestamp(CUTOFF) and len(df) == VAL_LAST + 1
    assert np.isnan(df[TARGET].to_numpy()[-LEAD:]).all()
    np.testing.assert_array_equal(df[TARGET].to_numpy()[:-LEAD], clean[TARGET].values[:VAL_LAST + 1 - LEAD])
    for name in (*DYNAMIC, "qobs_m3s"):
        np.testing.assert_array_equal(df[name].to_numpy(), clean[name].values[:VAL_LAST + 1])
    # No sentinel reaches the loaded data or the scaler.
    for ds in (train, val):
        for basin in BASINS:
            freq = ds.frequencies[0]
            stored = [*ds._x_d[basin][freq].values(), ds._y[basin][freq]]
            assert all(np.nanmax(np.abs(np.asarray(array))) < SENTINEL / 1e3 for array in stored)
    assert all(abs(float(v)) < SENTINEL / 10 for v in train.scaler["xarray_feature_center"].to_array().values.ravel())

    # Same validation lookup with and without the cutoff (Concern B already
    # removes issue times whose lead target lies past the period end).
    clean_root = _build_package(tmp_path / "clean")
    cfg0 = _config(clean_root, seq_length=72, name="nocut", support_file=None)
    train0 = _train_dataset(cfg0, tmp_path, "nocut_train")
    val0 = FlashNHDataset(cfg=cfg0, is_train=False, period="validation", scaler=train0.scaler)
    assert _pairs(val) == _pairs(val0) and _pairs(train) == _pairs(train0)
    assert max(t for _, t in _pairs(val)) == _ts(VAL_LAST - LEAD)


def test_cutoff_loader_refuses_test_period_and_periods_past_cutoff(tmp_path):
    from src.baseline.nh_dataset import FlashNHDataset, FlashNHDatasetError

    root = _build_package(tmp_path / "pkg")
    cfg = _config(root, seq_length=72, name="cut", support_file=None, cutoff=CUTOFF)
    train = _train_dataset(cfg, tmp_path, "cut_train")
    with pytest.raises(FlashNHDatasetError, match="test period is refused"):
        FlashNHDataset(cfg=cfg, is_train=False, period="test", scaler=train.scaler)
    early = _config(root, seq_length=72, name="early", support_file=None, cutoff="2000-02-10T23:00:00")
    with pytest.raises(FlashNHDatasetError, match="after observation cutoff"):
        FlashNHDataset(cfg=early, is_train=False, period="validation", scaler=train.scaler)
    bad = _config(root, seq_length=72, name="bad", support_file=None, cutoff="2000-02-14T23:30:00")
    with pytest.raises(FlashNHDatasetError, match="hourly grid"):
        _train_dataset(bad, tmp_path, "bad_train")


def test_cutoff_area_derivation_ignores_post_cutoff_observations(tmp_path):
    from src.baseline.nh_raw_space_evaluation import RawSpaceEvaluationError, derive_basin_area_km2_from_netcdf

    root = _build_package(tmp_path / "pkg")
    # Make post-cutoff qobs inconsistent with the target (the target is left clean).
    path = root / "time_series" / "B01.nc"
    with xr.open_dataset(path) as data:
        ds = data.load()
    ds["qobs_m3s"].values[VAL_LAST + 1:] *= 7.0
    ds.to_netcdf(path)
    kwargs = dict(basin_id="B01", target_variable=TARGET, lead_hours=LEAD, min_samples=100)
    whole = derive_basin_area_km2_from_netcdf(path, **kwargs)
    assert whole.n_samples == N_HOURS - LEAD  # the whole-file read uses the post-cutoff pairs
    area = derive_basin_area_km2_from_netcdf(path, observation_cutoff=CUTOFF, **kwargs)
    assert area.area_km2 == pytest.approx(AREA_KM2, rel=1e-9) and area.consistent
    assert area.n_samples == VAL_LAST + 1 - LEAD  # last pair: target[cutoff - lead] <-> qobs[cutoff]
    with pytest.raises(RawSpaceEvaluationError, match="hourly grid"):
        derive_basin_area_km2_from_netcdf(path, observation_cutoff="2000-02-14T23:30:00", **kwargs)


def test_common336_scorer_requires_the_declared_cutoff(package, tmp_path):
    from src.baseline.nh_raw_space_evaluation import RawSpaceEvaluationError
    from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, raw_space_metrics_for_run_period

    contract = c336.load_common336_screening_contract(package["contract_path"])
    assert c336.screening_observation_cutoff(contract) == pd.Timestamp(CUTOFF)
    for name, line in (("undeclared", ""), ("wrong", 'flashnh_observation_cutoff: "2000-02-13T23:00:00"\n')):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "config.yml").write_text(f"experiment_name: {name}\n{line}")
        with pytest.raises(c336.Common336SupportError):
            c336.evaluate_common336_screening(run_dir=run_dir, epoch=1, package_root=package["root"],
                                              contract=contract, require_full_screening_population=False)
    run_dir = tmp_path / "declared"
    run_dir.mkdir()
    (run_dir / "config.yml").write_text(f'flashnh_observation_cutoff: "{CUTOFF}"\n')
    with pytest.raises(NHSeedEvaluationError, match="test period is refused"):
        raw_space_metrics_for_run_period(run_dir=run_dir, period="test", epoch=1, package_root=package["root"],
                                         target_variable=TARGET, lead_hours=LEAD)
    with pytest.raises(RawSpaceEvaluationError, match="contradicts"):
        raw_space_metrics_for_run_period(run_dir=run_dir, period="validation", epoch=1, package_root=package["root"],
                                         target_variable=TARGET, lead_hours=LEAD,
                                         observation_cutoff="2000-02-13T23:00:00")
    no_config = tmp_path / "no_config"
    no_config.mkdir()
    with pytest.raises(fixed.FixedSupportContractError, match="beyond observation cutoff"):
        fixed.evaluate_fixed_support_raw_space_metrics(
            run_dir=no_config, epoch=1, package_root=package["root"], contract=contract,
            expected_contract_id=c336.SCREENING_CONTRACT_ID, expected_seq_length_floor=c336.COMMON_SEQ_LENGTH_FLOOR,
            observation_cutoff="2000-02-13T23:00:00")
