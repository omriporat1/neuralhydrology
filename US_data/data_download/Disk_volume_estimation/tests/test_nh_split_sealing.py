"""Loader sealing for the water-year rotation split contract v001.

Design: ``.scratch_local/wy_rotation_split_v001/LOADER_SEALING_DESIGN.md``.
Code under test: ``nh_raw_space_evaluation.select_readable_observations`` /
``parse_rotation_sealing`` / ``derive_basin_area_km2_from_netcdf(readable_windows=...)``,
``nh_dataset.FlashNHDataset`` (``flashnh_split_contract`` +
``flashnh_rotation_cycle``) and the split-contract checks in
``nh_seed_evaluation.raw_space_metrics_for_run_period``.

Synthetic package over the real WY dates (2020-10-14 00:00 .. 2025-10-02 23:00),
two real development basins, real variable names. Every observation value is a
sentinel that names its own observation time: for ``qobs_m3s`` (observation at
row time t) and ``qobs_mm_per_h_leadNN`` (observation at t + NN h) the stored
value is ``hours(observation time - package start) + 1``. A loaded value can
therefore be decoded back to the observation time it came from and checked
against the period's readable windows, for every (cycle, period) with one
package. Observations later than the contract's upper bound
(2025-09-30 23:00, judged by observation time) hold the literal sentinel 1e6,
and an unknown variable holds 1e6 everywhere. Forcing values lie in [0, 1).

A spy on xarray's netCDF4 backend (``NetCDF4ArrayWrapper._getitem``) records
every row range actually read from disk, so the tests show that sealed rows
are never read, not only that they are absent afterwards.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline import wy_rotation_split as wy
from src.baseline.nh_raw_space_evaluation import (
    RawSpaceEvaluationError,
    derive_basin_area_km2_from_netcdf,
    observation_offset_hours,
    observation_times_outside_windows,
    parse_rotation_sealing,
    read_run_rotation_sealing,
    resolve_run_observation_cutoff,
    select_readable_observations,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import build_synthetic_package, prepare_run_dirs  # noqa: E402

BASINS = ("01019000", "01022500")  # real ids from the contract's pinned development population
SEQ = 3
LEAD = wy.LEAD_HOURS
TARGET = f"qobs_mm_per_h_lead{LEAD:02d}"
FORCING = "mrms_qpe_1h_mm"
GAP = "mrms_qpe_1h_mm_gap"
UNKNOWN = "usgs_iv_qualifier"  # holds no recognised observation name: must be dropped, never read
OBS_LEADS = (1, 3, 6, 12)
OBS_VARIABLES = ("qobs_m3s",) + tuple(f"qobs_mm_per_h_lead{k:02d}" for k in OBS_LEADS)
SENTINEL = 1.0e6
AREA_KM2 = 3.6  # q_m3s = mm/h * area / 3.6, so qobs_m3s carries the same code as the mm/h columns
PKG_START = pd.Timestamp("2020-10-14 00:00")
PKG_END = pd.Timestamp("2025-10-02 23:00")
DATES = pd.date_range(PKG_START, PKG_END, freq="h")
BOUND = wy.OBSERVATION_UPPER_BOUND
CYCLES = {c.cycle: c for c in wy.ROTATION_TABLE_V001}
PERIODS = tuple(wy.NH_PERIODS)
HOUR = pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- helpers

def _code(times) -> np.ndarray:
    return ((pd.DatetimeIndex(times) - PKG_START) / HOUR).to_numpy(dtype=np.float64) + 1.0


def _decode(values) -> pd.DatetimeIndex:
    values = np.asarray(values, dtype=np.float64)
    return PKG_START + pd.to_timedelta(np.rint(values - 1.0), unit="h")


def _obs_column(offset_hours: int) -> np.ndarray:
    obs_times = DATES + pd.Timedelta(hours=offset_hours)
    out = _code(obs_times)
    out[obs_times > BOUND] = SENTINEL
    out[obs_times > PKG_END] = np.nan
    return out


def _forcing() -> np.ndarray:
    return (np.arange(len(DATES)) % 997) / 997.0


def _inside(times, windows, bound=BOUND) -> np.ndarray:
    """Independent of the module: inclusive window membership and <= bound."""
    times = pd.DatetimeIndex(times)
    inside = np.zeros(len(times), dtype=bool)
    for w in windows:
        inside |= (times >= w.first) & (times <= w.last)
    return inside & (times <= bound)


def _offset(name) -> int:
    return 0 if name == "qobs_m3s" else int(name.rsplit("lead", 1)[1])


def _write_basin_file(path: Path, *, coordinate: str = "time") -> None:
    data = {FORCING: _forcing(), GAP: np.zeros(len(DATES)), UNKNOWN: np.full(len(DATES), SENTINEL)}
    for name in OBS_VARIABLES:
        data[name] = _obs_column(_offset(name))
    ds = xr.Dataset({k: (coordinate, v) for k, v in data.items()}, coords={coordinate: DATES})
    ds.to_netcdf(path)


@contextmanager
def _spy_reads():
    """Record (variable, rows) for every array read xarray's netCDF4 backend
    performs while the context is active."""
    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    reads = []
    original = NetCDF4ArrayWrapper._getitem

    def spy(self, key):
        index = key[0] if isinstance(key, tuple) else key
        reads.append((self.variable_name, np.arange(self.shape[0])[index]))
        return original(self, key)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(NetCDF4ArrayWrapper, "_getitem", spy)
        yield reads


def _assert_reads_sealed(reads, windows, *, expect_obs_reads: bool = True) -> None:
    obs_rows = 0
    for name, rows in reads:
        assert name != UNKNOWN, "an unknown variable was read from disk"
        if name in OBS_VARIABLES:
            obs_times = DATES[np.atleast_1d(rows)] + pd.Timedelta(hours=_offset(name))
            ok = _inside(obs_times, windows)
            assert ok.all(), f"{name}: sealed row(s) read from disk, first obs time {obs_times[~ok][0]}"
            obs_rows += np.atleast_1d(rows).size
    if expect_obs_reads:
        assert obs_rows > 0, "spy saw no observation reads (vacuous)"


def _assert_frame_sealed(df: pd.DataFrame, windows) -> None:
    """Per-basin frame returned by the loader: forcing whole, every observation
    variable present only where readable (decoded value = its own row's
    observation time), nothing else."""
    assert UNKNOWN not in df.columns
    assert set(df.columns) <= {FORCING, GAP, *OBS_VARIABLES}
    assert pd.DatetimeIndex(df.index).equals(DATES)
    np.testing.assert_array_equal(df[FORCING].to_numpy(), _forcing())  # forcing is read unrestricted
    for name in OBS_VARIABLES:
        assert name in df.columns, name
        values = df[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        assert not (values[finite] >= SENTINEL).any(), f"{name}: post-bound sentinel loaded"
        obs_times = DATES + pd.Timedelta(hours=_offset(name))
        np.testing.assert_array_equal(finite, _inside(obs_times, windows), err_msg=name)  # sealed AND not over-sealed
        assert _decode(values[finite]).equals(obs_times[finite]), f"{name}: value/time misaligned"


def _assert_target_sealed(ds_xr: xr.Dataset, windows, *, require_values: bool = True) -> None:
    """Pre-normalisation NH xarray (basin, date): no sentinel in any variable,
    and every finite target decodes to observation time = date + lead inside
    the windows."""
    for name in ds_xr.data_vars:
        values = np.asarray(ds_xr[name].values, dtype=np.float64)
        assert not (values[np.isfinite(values)] >= SENTINEL).any(), f"sentinel in {name}"
    target = ds_xr[TARGET].transpose("basin", "date")
    dates = pd.DatetimeIndex(target["date"].values)
    n_finite = 0
    for i in range(target.sizes["basin"]):
        values = np.asarray(target.values[i], dtype=np.float64)
        finite = np.isfinite(values)
        obs_times = dates[finite] + pd.Timedelta(hours=LEAD)
        assert _decode(values[finite]).equals(obs_times)
        assert _inside(obs_times, windows).all()
        n_finite += int(finite.sum())
    if require_values:
        assert n_finite > 0


def _render_dates(text: str, dates: dict) -> str:
    for key, value in dates.items():
        rendered = "[" + ", ".join(f'"{v}"' for v in value) + "]" if isinstance(value, list) else f'"{value}"'
        text, n = re.subn(rf"^{key}:.*$", f"{key}: {rendered}", text, flags=re.M)
        assert n == 1, key
    return text


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def contract_path(tmp_path_factory) -> Path:
    """The v001 contract written from code (verify rebuilds it from code, so it
    is byte-equivalent in content to the frozen file under .scratch_local)."""
    return wy.write_split_contract(tmp_path_factory.mktemp("contract") / "split_contract_v001.json",
                                   wy.build_split_contract())


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> dict:
    base = tmp_path_factory.mktemp("sealing_pkg")
    cfg_path = build_synthetic_package(
        base, basins=list(BASINS), seq_length=SEQ, lead_hours=LEAD, coordinate_name="time",
        dynamic_input_values={FORCING: {b: _forcing() for b in BASINS}},
        research_start=str(PKG_START), n_hours=len(DATES))
    for b in BASINS:
        _write_basin_file(base / "time_series" / f"{b}.nc")
    text = cfg_path.read_text()
    text, n = re.subn(rf"qobs_lead{LEAD}\b", TARGET, text)
    assert n == 1  # the target_variables entry
    text, n = re.subn(r"^loss:.*$", "loss: NSE", text, flags=re.M)
    assert n == 1
    return {"root": base, "cfg_text": text}


def _config(package, contract_path, tmp_dir: Path, tag: str, *, cycle=1, dates_cycle=None, keys=None,
            extra: str = "", text_edit=None):
    from neuralhydrology.utils.config import Config

    text = _render_dates(package["cfg_text"], wy.cycle_nh_dates(CYCLES[dates_cycle or cycle]))
    if keys is None:
        keys = f"flashnh_split_contract: '{Path(contract_path).as_posix()}'\nflashnh_rotation_cycle: {cycle}\n"
    text = text + "\nsave_train_data: True\n" + keys + extra
    if text_edit is not None:
        text = text_edit(text)
    path = tmp_dir / f"cfg_{tag}.yml"
    path.write_text(text)
    cfg = Config(path)
    prepare_run_dirs(cfg, tmp_dir, tag)
    return cfg


@pytest.fixture(scope="module")
def built(package, contract_path, tmp_path_factory):
    """Lazily build (and cache) FlashNHDataset per (cycle, period) under the
    contract keys, recording disk reads, per-basin loader frames and the
    pre-normalisation xarray."""
    from src.baseline.nh_dataset import FlashNHDataset
    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    tmp = tmp_path_factory.mktemp("sealing_runs")
    cache, cfgs = {}, {}

    def get(cycle: int, period: str) -> dict:
        key = (cycle, period)
        if key in cache:
            return cache[key]
        scaler = {} if period == "train" else get(cycle, "train")["dataset"].scaler
        if cycle not in cfgs:
            cfgs[cycle] = _config(package, contract_path, tmp, f"c{cycle}", cycle=cycle)
        frames, raw = {}, []
        orig_load = FlashNHDataset._load_basin_data
        orig_xr = FlashNHDataset._load_or_create_xarray_dataset

        def load_spy(self, basin):
            df = orig_load(self, basin)
            frames[basin] = df.copy()
            return df

        def xr_spy(self):
            out = orig_xr(self)
            raw.append(out.copy(deep=True))
            return out

        with _spy_reads() as reads, pytest.MonkeyPatch.context() as mp:
            mp.setattr(FlashNHDataset, "_load_basin_data", load_spy)
            mp.setattr(FlashNHDataset, "_load_or_create_xarray_dataset", xr_spy)
            ds = FlashNHDataset(cfg=cfgs[cycle], is_train=(period == "train"), period=period, scaler=scaler)
        cache[key] = {"dataset": ds, "reads": list(reads), "frames": frames, "raw": raw[0], "cfg": cfgs[cycle]}
        return cache[key]

    return get


# --------------------------------------------------------------------------- pure helpers (no NH)

def test_observation_offset_hours():
    assert observation_offset_hours("qobs_m3s") == 0
    assert observation_offset_hours("qobs_mm_per_h_lead06") == 6
    assert observation_offset_hours("qobs_mm_per_h_lead12") == 12
    for name in (FORCING, GAP, UNKNOWN, "rtma_temp_2m", "qobs_lead"):
        assert observation_offset_hours(name) is None


@pytest.mark.parametrize("cycle", sorted(CYCLES))
@pytest.mark.parametrize("period", PERIODS)
def test_select_readable_observations_reads_and_keeps_only_readable_rows(tmp_path, cycle, period):
    path = tmp_path / "b.nc"
    _write_basin_file(path)
    windows = wy.readable_observation_windows(CYCLES[cycle], period)
    with _spy_reads() as reads:
        with xr.open_dataset(path) as ds:
            out = select_readable_observations(ds, windows=windows, coordinate="time",
                                               forcing_variables=[FORCING, GAP], upper_bound=BOUND)
            df = out.to_dataframe()
    _assert_reads_sealed(reads, windows)
    _assert_frame_sealed(df, windows)


def test_post_bound_lead_rows_inside_wy2025_are_sealed(tmp_path):
    """The lead06 rows 2025-09-30 18:00-23:00 lie in WY2025 by row time but
    hold WY2026 observations: sealed for a period that may read WY2025."""
    path = tmp_path / "b.nc"
    _write_basin_file(path)
    windows = (wy.water_year_target_window(2025),)
    with xr.open_dataset(path) as ds:
        out = select_readable_observations(ds, windows=windows, coordinate="time", forcing_variables=[FORCING],
                                           upper_bound=BOUND)
        lead06 = out[TARGET].to_series()
        lead12 = out["qobs_mm_per_h_lead12"].to_series()
    tail = pd.date_range("2025-09-30 18:00", "2025-09-30 23:00", freq="h")
    assert lead06.loc[tail].isna().all() and np.isfinite(lead06.loc[pd.Timestamp("2025-09-30 17:00")])
    assert lead12.loc[pd.date_range("2025-09-30 12:00", "2025-09-30 23:00", freq="h")].isna().all()
    assert np.isfinite(lead12.loc[pd.Timestamp("2025-09-30 11:00")])
    assert not (out[TARGET].values >= SENTINEL).any()
    # a window reaching past the bound is clipped to it
    with xr.open_dataset(path) as ds:
        out = select_readable_observations(
            ds, windows=(wy.TimeWindow(pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-02 23:00")),),
            coordinate="time", forcing_variables=[], upper_bound=BOUND)
        assert out["qobs_m3s"].to_series().last_valid_index() == BOUND
        assert UNKNOWN not in out and FORCING not in out


def test_select_readable_observations_refusals(tmp_path):
    path = tmp_path / "b.nc"
    _write_basin_file(path)
    windows = (wy.water_year_target_window(2022),)
    with xr.open_dataset(path) as ds:
        with pytest.raises(RawSpaceEvaluationError, match="cannot be read as unrestricted forcing"):
            select_readable_observations(ds, windows=windows, coordinate="time",
                                         forcing_variables=[FORCING, "qobs_m3s"], upper_bound=BOUND)
        with pytest.raises(RawSpaceEvaluationError, match="not in the package dataset"):
            select_readable_observations(ds, windows=windows, coordinate="time",
                                         forcing_variables=["rtma_missing"], upper_bound=BOUND)
        with pytest.raises(RawSpaceEvaluationError, match="no 'date' coordinate"):
            select_readable_observations(ds, windows=windows, coordinate="date", forcing_variables=[],
                                         upper_bound=BOUND)
        overlapping = (wy.water_year_target_window(2022), wy.water_year_target_window(2022))
        with pytest.raises(RawSpaceEvaluationError, match="overlap"):
            select_readable_observations(ds, windows=overlapping, coordinate="time", forcing_variables=[],
                                         upper_bound=BOUND)
    tz = xr.Dataset({"qobs_m3s": ("time", np.ones(3))},
                    coords={"time": pd.date_range("2022-01-01", periods=3, freq="h", tz="UTC")})
    with pytest.raises(RawSpaceEvaluationError, match="naive UTC"):
        select_readable_observations(tz, windows=windows, coordinate="time", forcing_variables=[], upper_bound=BOUND)


def test_observation_times_outside_windows():
    windows = (wy.water_year_target_window(2022),)
    times = pd.DatetimeIndex(["2021-09-30 23:00", "2021-10-01 00:00", "2022-09-30 23:00", "2022-10-01 00:00"])
    np.testing.assert_array_equal(observation_times_outside_windows(times, windows=windows, upper_bound=BOUND),
                                  [True, False, False, True])
    late = (wy.TimeWindow(pd.Timestamp("2025-09-30 00:00"), pd.Timestamp("2025-10-01 05:00")),)
    np.testing.assert_array_equal(
        observation_times_outside_windows(pd.DatetimeIndex(["2025-09-30 23:00", "2025-10-01 00:00"]),
                                          windows=late, upper_bound=BOUND), [False, True])


# --------------------------------------------------------------------------- config parsing

def test_parse_rotation_sealing(contract_path, tmp_path):
    assert parse_rotation_sealing({}) is None
    assert parse_rotation_sealing({"flashnh_split_contract": None, "flashnh_rotation_cycle": None}) is None
    sealing = parse_rotation_sealing({"flashnh_split_contract": str(contract_path), "flashnh_rotation_cycle": 2})
    assert sealing.cycle == 2 and sealing.lead_hours == LEAD and sealing.upper_bound == BOUND
    assert sealing.training_windows == wy.readable_observation_windows(CYCLES[2], "train")
    assert sealing.nh_dates == wy.cycle_nh_dates(CYCLES[2])
    assert sealing.contract_checksum_sha256 == json.loads(contract_path.read_text())["checksum_sha256"]
    receipt = sealing.receipt()
    assert receipt["rotation_cycle"] == 2 and set(receipt["readable_observation_windows"]) == set(PERIODS)
    # a relative path resolves against the project root
    rel = parse_rotation_sealing({"flashnh_split_contract": contract_path.name, "flashnh_rotation_cycle": 2},
                                 project_root=contract_path.parent)
    assert Path(rel.contract_path) == contract_path


@pytest.mark.parametrize("config, match", [
    ({"flashnh_split_contract": "x.json"}, "must be set together"),
    ({"flashnh_rotation_cycle": 1}, "must be set together"),
    ({"flashnh_rotation_cycle": "1", "flashnh_split_contract": "@"}, "must be an int"),
    ({"flashnh_rotation_cycle": True, "flashnh_split_contract": "@"}, "must be an int"),
    ({"flashnh_rotation_cycle": 6, "flashnh_split_contract": "@"}, "not in the contract table"),
    ({"flashnh_rotation_cycle": 1, "flashnh_split_contract": "@",
      "flashnh_observation_cutoff": "2025-09-30T23:00:00"}, "cannot be combined"),
    ({"flashnh_rotation_cycle": 1, "flashnh_split_contract": "missing.json"}, "refused"),
])
def test_parse_rotation_sealing_refusals(contract_path, tmp_path, config, match):
    config = {k: (str(contract_path) if v == "@" else v) for k, v in config.items()}
    with pytest.raises(RawSpaceEvaluationError, match=match):
        parse_rotation_sealing(config, project_root=tmp_path)


@pytest.mark.parametrize("tamper", ["edit_window", "edit_and_rechecksum", "bad_checksum"])
def test_parse_rotation_sealing_refuses_a_tampered_contract(contract_path, tmp_path, tamper):
    from src.baseline.wy_rotation_split import canonical_json_sha256

    contract = json.loads(contract_path.read_text())
    if tamper == "bad_checksum":
        contract["checksum_sha256"] = "0" * 64
    else:
        contract["cycles"][0]["periods"]["train"]["readable_observation_windows"][0]["last_utc"] = \
            "2025-10-31T23:00:00Z"
        if tamper == "edit_and_rechecksum":
            body = {k: v for k, v in contract.items() if k != "checksum_sha256"}
            contract["checksum_sha256"] = canonical_json_sha256(body)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(contract))
    with pytest.raises(RawSpaceEvaluationError, match="refused"):
        parse_rotation_sealing({"flashnh_split_contract": str(path), "flashnh_rotation_cycle": 1})


# --------------------------------------------------------------------------- FlashNHDataset under the contract

@pytest.mark.parametrize("cycle", sorted(CYCLES))
@pytest.mark.parametrize("period", PERIODS)
def test_dataset_loads_no_sealed_observation(built, cycle, period):
    """Per (cycle, period): disk reads, per-basin loader frames and the
    pre-normalisation NH xarray hold no observation outside the period's
    readable windows or after the upper bound, and no unknown variable."""
    windows = wy.readable_observation_windows(CYCLES[cycle], period)
    got = built(cycle, period)
    _assert_reads_sealed(got["reads"], windows)
    assert set(got["frames"]) == set(BASINS)
    for df in got["frames"].values():
        _assert_frame_sealed(df, windows)
    _assert_target_sealed(got["raw"], windows)
    assert len(got["dataset"]) > 0


@pytest.mark.parametrize("cycle", sorted(CYCLES))
def test_saved_train_data_and_scaler_hold_no_sealed_value(built, contract_path, cycle):
    got = built(cycle, "train")
    ds, cfg = got["dataset"], got["cfg"]
    windows = wy.readable_observation_windows(CYCLES[cycle], "train")
    with (Path(cfg.train_dir) / "train_data.p").open("rb") as fh:
        saved = xr.Dataset.from_dict(pickle.load(fh))
    _assert_target_sealed(saved, windows)  # written by NH before the Flash-NH train mask
    center = float(ds.scaler["xarray_feature_center"][TARGET].values)
    scale = float(ds.scaler["xarray_feature_scale"][TARGET].values)
    assert 0 < center < SENTINEL and 0 < scale < SENTINEL
    # every finite training target (un-normalised) decodes into a training window
    freq = ds.frequencies[0]
    for b in BASINS:
        dates = pd.DatetimeIndex(ds._dates[b][freq])
        y = ds._y[b][freq][:, 0].numpy().astype(np.float64) * scale + center
        finite = np.isfinite(y)
        assert finite.any()
        assert _inside(_decode(y[finite]), windows).all()
        assert _decode(y[finite]).equals(dates[finite] + pd.Timedelta(hours=LEAD))
    stds = np.asarray([float(v) for v in ds._per_basin_target_stds.values()]) if isinstance(
        ds._per_basin_target_stds, dict) else None
    if stds is not None:
        assert np.isfinite(stds).all() and (stds < SENTINEL).all()


@pytest.mark.parametrize("cycle", [1, 2])
@pytest.mark.parametrize("period", PERIODS)
def test_sealing_receipt_and_filter_stats(built, contract_path, cycle, period):
    got = built(cycle, period)
    checksum = json.loads(contract_path.read_text())["checksum_sha256"]
    receipt = json.loads((Path(got["cfg"].train_dir) / f"flashnh_split_sealing_{period}.json").read_text())
    assert receipt["rotation_cycle"] == cycle and receipt["period"] == period
    assert receipt["contract_checksum_sha256"] == checksum and receipt["contract_id"] == wy.CONTRACT_ID
    if period == "train":
        assert receipt["n_basins"] == len(BASINS)
        # the leak-fix train mask stays as defence in depth and finds nothing left to mask
        assert receipt["n_train_targets_masked_outside_period"] == 0
    else:  # NH evaluates one basin per dataset: counts would be last-writer-wins, so none are recorded
        assert "n_basins" not in receipt and "n_train_targets_masked_outside_period" not in receipt
    stats = got["dataset"].flashnh_filter_stats
    assert stats.rotation_cycle == cycle and stats.split_contract_checksum_sha256 == checksum


@pytest.mark.parametrize("cycle", [1, 3])
def test_dataset_written_receipts_satisfy_the_scorer_check(built, cycle):
    """The receipts the sealed loader writes are exactly what
    verify_run_sealing_receipts accepts (and only for this cycle)."""
    from src.baseline.nh_raw_space_evaluation import verify_run_sealing_receipts

    for period in PERIODS:
        got = built(cycle, period)
    cfg = got["cfg"]
    sealing = parse_rotation_sealing(cfg.as_dict())
    verify_run_sealing_receipts(cfg.run_dir, sealing, PERIODS)
    other = parse_rotation_sealing(dict(cfg.as_dict(), flashnh_rotation_cycle=2))
    with pytest.raises(RawSpaceEvaluationError, match="does not match"):
        verify_run_sealing_receipts(cfg.run_dir, other, ("train",))


def test_receipt_is_not_rewritten_when_identical(built):
    got = built(1, "validation")
    path = Path(got["cfg"].train_dir) / "flashnh_split_sealing_validation.json"
    before = path.stat().st_mtime_ns
    got["dataset"]._write_sealing_receipt(got["dataset"]._flashnh_rotation_sealing, 0)
    assert path.stat().st_mtime_ns == before


def test_evaluating_the_train_period_keeps_the_training_counts(built, monkeypatch):
    """Review round 2, finding 5: `nh-run evaluate --period train` builds
    is_train=False datasets one basin at a time; they must not overwrite the
    training receipt's counts."""
    got = built(1, "train")
    ds = got["dataset"]
    path = Path(got["cfg"].train_dir) / "flashnh_split_sealing_train.json"
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(ds, "is_train", False)
    monkeypatch.setattr(ds, "basins", [BASINS[0]])
    ds._write_sealing_receipt(ds._flashnh_rotation_sealing, 7)
    assert path.read_text(encoding="utf-8") == before
    assert json.loads(before)["n_basins"] == len(BASINS)


def test_evaluation_receipt_is_written_only_beside_a_matching_training_receipt(built, tmp_path, monkeypatch, caplog):
    """Review round 2, finding 4: at evaluation cfg.train_dir comes from the
    run's saved config.yml and may be stale (moved run, other machine). The
    receipt is never written into a folder that is missing or lacks a
    matching training receipt (the scorer then refuses: fail closed)."""
    from types import SimpleNamespace

    got = built(1, "validation")
    ds = got["dataset"]
    sealing = ds._flashnh_rotation_sealing
    train_receipt = (Path(got["cfg"].train_dir) / "flashnh_split_sealing_train.json").read_text(encoding="utf-8")
    name = "flashnh_split_sealing_validation.json"

    def write_into(train_dir):
        monkeypatch.setattr(ds, "cfg", SimpleNamespace(train_dir=train_dir))
        ds._write_sealing_receipt(sealing, 0)

    missing = tmp_path / "elsewhere" / "train_data"  # e.g. a Moriah path on Windows
    write_into(missing)
    assert not missing.exists() and not missing.parent.exists()
    assert "not written" in caplog.text

    empty = tmp_path / "empty"
    empty.mkdir()
    write_into(empty)
    assert not (empty / name).exists()

    other = tmp_path / "other_cycle"
    other.mkdir()
    (other / "flashnh_split_sealing_train.json").write_text(
        json.dumps(dict(json.loads(train_receipt), rotation_cycle=2)), encoding="utf-8")
    write_into(other)
    assert not (other / name).exists()

    matching = tmp_path / "matching"
    matching.mkdir()
    (matching / "flashnh_split_sealing_train.json").write_text(train_receipt, encoding="utf-8")
    write_into(matching)
    written = json.loads((matching / name).read_text(encoding="utf-8"))
    assert written["period"] == "validation" and written["rotation_cycle"] == 1 and "n_basins" not in written


def test_sealed_load_reads_only_approved_gap_flags(built, tmp_path, monkeypatch):
    """Review round 2, finding 2: a `*_gap` variable in the file is read whole
    only if it is an approved package variable; an unapproved one whose name
    looks like a gap flag (here `qobs_m3s_gap`) is neither read nor loaded."""
    got = built(1, "train")
    ds = got["dataset"]
    path = tmp_path / "with_extra_gap_flags.nc"
    _write_basin_file(path)
    with xr.open_dataset(path) as src:
        extra = src.load()
    extra["qobs_m3s_gap"] = ("time", np.full(len(DATES), SENTINEL))
    extra["rtma_gap"] = ("time", np.zeros(len(DATES)))
    path2 = tmp_path / "with_extra_gap_flags2.nc"
    extra.to_netcdf(path2)
    monkeypatch.setattr(type(ds), "_basin_netcdf", lambda self, basin: path2)
    with _spy_reads() as reads:
        df = ds._load_basin_data(BASINS[0])
    read_names = {name for name, _ in reads}
    assert "qobs_m3s_gap" not in read_names and "qobs_m3s_gap" not in df.columns
    assert {"rtma_gap", GAP} <= read_names and {"rtma_gap", GAP} <= set(df.columns)
    windows = wy.readable_observation_windows(CYCLES[1], "train")
    _assert_reads_sealed(reads, windows)


def test_sealing_verification_is_memoised_and_invalidated_by_a_file_change(contract_path, tmp_path, monkeypatch):
    """F6: the contract + population verification runs once per process for
    the same config values and file stamps, and again when the contract file
    changes."""
    import os

    from src.baseline import nh_dataset

    path = tmp_path / "contract.json"
    path.write_bytes(contract_path.read_bytes())
    calls = []
    real = wy.verify_basin_population
    monkeypatch.setattr(wy, "verify_basin_population", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(nh_dataset, "_SEALING_VERIFICATION_CACHE", {})
    cfg = {"flashnh_split_contract": str(path), "flashnh_rotation_cycle": 2}
    first = nh_dataset._verified_sealing_and_population(cfg)
    second = nh_dataset._verified_sealing_and_population(cfg)
    assert len(calls) == 1 and second[0] is first[0] and BASINS[0] in first[1]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    nh_dataset._verified_sealing_and_population(cfg)
    assert len(calls) == 2
    assert nh_dataset._verified_sealing_and_population({}) == (None, None)


def test_dataset_refuses_additional_features_passed_in(package, contract_path, tmp_path):
    from src.baseline.nh_dataset import FlashNHDataset, FlashNHDatasetError
    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    cfg = _config(package, contract_path, tmp_path, "addfeat")
    extra = {b: pd.DataFrame({"qobs_m3s_copy": np.ones(len(DATES))}, index=DATES) for b in BASINS}
    with _spy_reads() as reads:
        with pytest.raises(FlashNHDatasetError, match="additional_features passed to the dataset"):
            FlashNHDataset(cfg=cfg, is_train=True, period="train", scaler={}, additional_features=[extra])
    assert not [name for name, _ in reads if name in OBS_VARIABLES or name == UNKNOWN]


def _refusal_cases():
    def add_obs_input(text):
        out, n = re.subn(rf"^dynamic_inputs:\n  - {FORCING}$", f"dynamic_inputs:\n  - {FORCING}\n  - qobs_m3s",
                         text, flags=re.M)
        assert n == 1
        return out

    def wrong_lead(text):
        return text.replace(TARGET, "qobs_mm_per_h_lead03")

    def add_unapproved_input(text):
        out, n = re.subn(rf"^dynamic_inputs:\n  - {FORCING}$", f"dynamic_inputs:\n  - {FORCING}\n  - {UNKNOWN}",
                         text, flags=re.M)
        assert n == 1
        return out

    return [
        ("other_cycle_dates", dict(cycle=1, dates_cycle=2), "differs from the contract"),
        ("wrong_lead", dict(text_edit=wrong_lead), "target lead 3 h differs"),
        ("cutoff_too", dict(extra='flashnh_observation_cutoff: "2025-09-30T23:00:00"\n'), "cannot be combined"),
        ("only_contract", dict(keys="flashnh_split_contract: '@'\n"), "must be set together"),
        ("only_cycle", dict(keys="flashnh_rotation_cycle: 1\n"), "must be set together"),
        ("cycle_string", dict(keys="flashnh_split_contract: '@'\nflashnh_rotation_cycle: '1'\n"), "must be an int"),
        ("cycle_unknown", dict(keys="flashnh_split_contract: '@'\nflashnh_rotation_cycle: 6\n"),
         "not in the contract table"),
        ("train_data_file", dict(extra="train_data_file: some/train_data.p\n"), "train_data_file is not allowed"),
        ("obs_as_input", dict(text_edit=add_obs_input), "configured as model inputs"),
        ("lagged_obs", dict(extra="lagged_features:\n  qobs_m3s: [1]\n"), "lagged/duplicated observation"),
        ("unapproved_input", dict(text_edit=add_unapproved_input), "not approved package forcing variables"),
        ("common_support", dict(extra="flashnh_common_support_file: some/common_support.json\n"),
         "flashnh_common_support_file is not allowed"),
    ]


@pytest.mark.parametrize("tag, kwargs, match", _refusal_cases(), ids=[c[0] for c in _refusal_cases()])
def test_dataset_refusals_read_nothing(package, contract_path, tmp_path, tag, kwargs, match):
    from src.baseline.nh_dataset import FlashNHDataset, FlashNHDatasetError
    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    if "keys" in kwargs:
        kwargs = dict(kwargs, keys=kwargs["keys"].replace("@", Path(contract_path).as_posix()))
    cfg = _config(package, contract_path, tmp_path, tag, **kwargs)
    with _spy_reads() as reads:
        with pytest.raises(FlashNHDatasetError, match=match):
            FlashNHDataset(cfg=cfg, is_train=True, period="train", scaler={})
    assert not [name for name, _ in reads if name in OBS_VARIABLES or name == UNKNOWN]


def test_dataset_refuses_a_basin_outside_the_contract_population(package, contract_path, tmp_path):
    from src.baseline.nh_dataset import FlashNHDataset, FlashNHDatasetError
    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    basin_file = tmp_path / "basins.txt"
    basin_file.write_text(f"{BASINS[0]}\nSYN01\n")
    _write_basin_file(Path(package["root"]) / "time_series" / "SYN01.nc")

    def edit(text):
        return re.sub(r"^train_basin_file:.*$", f"train_basin_file: {basin_file.as_posix()}", text, flags=re.M)

    cfg = _config(package, contract_path, tmp_path, "pop", text_edit=edit)
    with _spy_reads() as reads:
        with pytest.raises(FlashNHDatasetError, match="not in the contract population"):
            FlashNHDataset(cfg=cfg, is_train=True, period="train", scaler={})
    assert not [name for name, _ in reads if name in OBS_VARIABLES]


def test_dataset_refuses_a_tampered_contract(package, contract_path, tmp_path):
    from src.baseline.nh_dataset import FlashNHDataset, FlashNHDatasetError
    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    contract = json.loads(contract_path.read_text())
    contract["cycles"][0]["train_wys"] = [2022, 2023, 2024]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(contract))
    keys = f"flashnh_split_contract: '{path.as_posix()}'\nflashnh_rotation_cycle: 1\n"
    cfg = _config(package, contract_path, tmp_path, "tamper", keys=keys)
    with pytest.raises(FlashNHDatasetError, match="refused"):
        FlashNHDataset(cfg=cfg, is_train=True, period="train", scaler={})


# --------------------------------------------------------------------------- basin-area derivation

def _training_obs_count(cycle: int) -> int:
    """Area samples pair target[t] with qobs_m3s[t + lead]: one per observation
    time T in the training windows with T - lead inside the package."""
    times = DATES[_inside(DATES, wy.readable_observation_windows(CYCLES[cycle], "train"))]
    return int((times >= PKG_START + pd.Timedelta(hours=LEAD)).sum())


@pytest.mark.parametrize("cycle", sorted(CYCLES))
def test_area_uses_only_training_window_observations(tmp_path, cycle):
    """qobs_m3s is corrupted (x7) at every observation time outside the cycle's
    training windows. Training-window derivation is exact and counts only
    training samples; a derivation over a sealed year sees the corruption."""
    path = tmp_path / "b.nc"
    _write_basin_file(path)
    train_windows = wy.readable_observation_windows(CYCLES[cycle], "train")
    with xr.open_dataset(path) as data:
        ds = data.load()
    sealed = ~_inside(DATES, train_windows) & (DATES <= BOUND)
    ds["qobs_m3s"].values[sealed] *= 7.0
    ds.to_netcdf(path)
    kwargs = dict(basin_id=BASINS[0], target_variable=TARGET, lead_hours=LEAD, min_samples=100)

    with _spy_reads() as reads:
        area = derive_basin_area_km2_from_netcdf(path, readable_windows=train_windows,
                                                 observation_upper_bound=BOUND, **kwargs)
    _assert_reads_sealed(reads, train_windows)
    assert area.consistent and area.area_km2 == pytest.approx(AREA_KM2, rel=1e-12)
    assert area.n_samples == _training_obs_count(cycle)

    assessment = wy.readable_observation_windows(CYCLES[cycle], "test")
    corrupted = derive_basin_area_km2_from_netcdf(path, readable_windows=assessment,
                                                  observation_upper_bound=BOUND, **kwargs)
    assert corrupted.area_km2 == pytest.approx(7 * AREA_KM2, rel=1e-12)
    whole = derive_basin_area_km2_from_netcdf(path, **kwargs)
    assert whole.n_samples > area.n_samples


def test_area_argument_refusals(tmp_path):
    path = tmp_path / "b.nc"
    _write_basin_file(path)
    kwargs = dict(basin_id=BASINS[0], target_variable=TARGET, lead_hours=LEAD)
    windows = (wy.water_year_target_window(2023),)
    with pytest.raises(RawSpaceEvaluationError, match="given together"):
        derive_basin_area_km2_from_netcdf(path, readable_windows=windows, **kwargs)
    with pytest.raises(RawSpaceEvaluationError, match="given together"):
        derive_basin_area_km2_from_netcdf(path, observation_upper_bound=BOUND, **kwargs)
    with pytest.raises(RawSpaceEvaluationError, match="cannot be combined"):
        derive_basin_area_km2_from_netcdf(path, readable_windows=windows, observation_upper_bound=BOUND,
                                          observation_cutoff="2023-01-01T00:00:00", **kwargs)
    with pytest.raises(RawSpaceEvaluationError, match="empty"):
        derive_basin_area_km2_from_netcdf(path, readable_windows=(), observation_upper_bound=BOUND, **kwargs)


# --------------------------------------------------------------------------- raw-space scorer

def _write_receipts(run_dir: Path, sealing, periods=("train", "validation")) -> None:
    (run_dir / "train_data").mkdir(parents=True, exist_ok=True)
    for period in periods:
        (run_dir / "train_data" / f"flashnh_split_sealing_{period}.json").write_text(
            json.dumps(dict(sealing.receipt(), period=period)))


def _contract_run_dir(tmp_path: Path, contract_path: Path, cycle: int = 1, *, dataset: str = "flashnh",
                      receipts=("train", "validation"), name: str = "run") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    config = (f"experiment_name: sealed\ndataset: {dataset}\nflashnh_split_contract: '{contract_path.as_posix()}'\n"
              f"flashnh_rotation_cycle: {cycle}\n")
    (run_dir / "config.yml").write_text(config)
    if receipts:
        _write_receipts(run_dir, parse_rotation_sealing(
            {"flashnh_split_contract": str(contract_path), "flashnh_rotation_cycle": cycle}), receipts)
    return run_dir


def _write_results(run_dir: Path, period: str, issue_dates, obs, sim, basin=BASINS[0], epoch=1) -> None:
    from src.baseline.nh_seed_evaluation import period_results_path

    path = period_results_path(run_dir, period, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = xr.Dataset(
        {f"{TARGET}_obs": (("date", "time_step"), np.asarray(obs, dtype=np.float64)[:, None]),
         f"{TARGET}_sim": (("date", "time_step"), np.asarray(sim, dtype=np.float64)[:, None])},
        coords={"date": pd.DatetimeIndex(issue_dates), "time_step": [0]})
    with path.open("wb") as fh:
        pickle.dump({basin: {"1h": {"xr": data}}}, fh)


def test_cutoff_only_callers_refuse_a_contract_run(tmp_path, contract_path):
    run_dir = _contract_run_dir(tmp_path, contract_path)
    with pytest.raises(RawSpaceEvaluationError, match="rotation split contract"):
        resolve_run_observation_cutoff(run_dir)
    assert read_run_rotation_sealing(run_dir).cycle == 1
    assert read_run_rotation_sealing(tmp_path) is None  # no config.yml


def test_run_sealing_record_refusals(tmp_path, contract_path):
    """F1/F4: a contract declared under another dataset was loaded unsealed;
    receipts without a config, or with a contract-free config, contradict the
    record. Cutoff-only callers also refuse a receipt-bearing run."""
    generic = _contract_run_dir(tmp_path, contract_path, dataset="generic", name="generic")
    with pytest.raises(RawSpaceEvaluationError, match="loaded unsealed"):
        read_run_rotation_sealing(generic)

    no_config = _contract_run_dir(tmp_path, contract_path, name="no_config")
    (no_config / "config.yml").unlink()
    with pytest.raises(RawSpaceEvaluationError, match="no config.yml"):
        read_run_rotation_sealing(no_config)
    with pytest.raises(RawSpaceEvaluationError, match="receipts"):
        resolve_run_observation_cutoff(no_config)

    stripped = _contract_run_dir(tmp_path, contract_path, name="stripped")
    (stripped / "config.yml").write_text("experiment_name: sealed\ndataset: flashnh\n")
    with pytest.raises(RawSpaceEvaluationError, match="declares no split contract"):
        read_run_rotation_sealing(stripped)
    with pytest.raises(RawSpaceEvaluationError, match="receipts"):
        resolve_run_observation_cutoff(stripped)


def test_receipt_verification_refusals(tmp_path, contract_path):
    from src.baseline.nh_raw_space_evaluation import verify_run_sealing_receipts

    run_dir = _contract_run_dir(tmp_path, contract_path, receipts=("train",))
    sealing = read_run_rotation_sealing(run_dir)
    verify_run_sealing_receipts(run_dir, sealing, ("train",))
    with pytest.raises(RawSpaceEvaluationError, match="no split-sealing receipt for the validation period"):
        verify_run_sealing_receipts(run_dir, sealing, ("train", "validation"))
    receipt_path = run_dir / "train_data" / "flashnh_split_sealing_train.json"
    receipt = json.loads(receipt_path.read_text())
    (run_dir / "train_data" / "flashnh_split_sealing_validation.json").write_text(json.dumps(receipt))
    with pytest.raises(RawSpaceEvaluationError, match="not for the validation period"):
        verify_run_sealing_receipts(run_dir, sealing, ("validation",))
    receipt["lead_hours"] = 3
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RawSpaceEvaluationError, match=r"does not match.*lead_hours"):
        verify_run_sealing_receipts(run_dir, sealing, ("train",))
    receipt_path.write_text("{not json")
    with pytest.raises(RawSpaceEvaluationError, match="unreadable"):
        verify_run_sealing_receipts(run_dir, sealing, ("train",))


def test_parse_rotation_sealing_requires_a_pinned_checksum(contract_path, monkeypatch):
    """F5: a contract that verifies against the generator code is still
    refused unless its checksum is pinned (guards a later edit of the code)."""
    from src.baseline import nh_raw_space_evaluation as rse

    cfg = {"flashnh_split_contract": str(contract_path), "flashnh_rotation_cycle": 1}
    assert parse_rotation_sealing(cfg).contract_checksum_sha256 == wy.build_split_contract()["checksum_sha256"]
    monkeypatch.setitem(rse.FROZEN_SPLIT_CONTRACT_CHECKSUMS, wy.CONTRACT_ID, "0" * 64)
    with pytest.raises(RawSpaceEvaluationError, match="not a pinned frozen contract"):
        parse_rotation_sealing(cfg)


def test_scorer_under_contract(package, contract_path, tmp_path):
    from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, raw_space_metrics_for_run_period

    run_dir = _contract_run_dir(tmp_path, contract_path, cycle=1)  # cycle 1: validation = WY2022
    windows = wy.readable_observation_windows(CYCLES[1], "validation")
    issue = pd.date_range("2021-10-01 00:00", "2022-09-30 23:00", freq="h")  # NH validation issue dates
    obs = _code(issue + pd.Timedelta(hours=LEAD))
    readable = _inside(issue + pd.Timedelta(hours=LEAD), windows)
    obs[~readable] = np.nan  # what a sealed loader leaves: the last 6 issue hours point into WY2023
    assert (~readable).sum() == LEAD
    _write_results(run_dir, "validation", issue, obs, obs * 0.9)
    common = dict(run_dir=run_dir, period="validation", epoch=1, package_root=package["root"],
                  target_variable=TARGET, lead_hours=LEAD)

    result = raw_space_metrics_for_run_period(**common)
    assert result["split_contract"]["rotation_cycle"] == 1
    (basin,) = result["per_basin"] if "per_basin" in result else result["per_basin_metrics"]
    assert basin["area_n_samples"] == _training_obs_count(1)  # area from WY2023-25 only

    with pytest.raises(NHSeedEvaluationError, match="cannot be applied"):
        raw_space_metrics_for_run_period(**common, observation_cutoff="2022-09-30T23:00:00")
    with pytest.raises(NHSeedEvaluationError, match="differs from the split contract lead"):
        raw_space_metrics_for_run_period(**dict(common, lead_hours=3))

    leaked = obs.copy()
    leaked[-1] = _code([issue[-1] + pd.Timedelta(hours=LEAD)])[0]  # a WY2023 observation in WY2022 scoring
    _write_results(run_dir, "validation", issue, leaked, leaked * 0.9)
    with pytest.raises(NHSeedEvaluationError, match="outside the period's readable windows"):
        raw_space_metrics_for_run_period(**common)


def test_scored_observation_check_refuses_multi_step_results():
    from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, _check_scored_observations_readable

    obs = xr.DataArray(np.ones((3, 2)), dims=("date", "time_step"),
                       coords={"date": pd.date_range("2022-01-01", periods=3, freq="h"), "time_step": [0, 1]})
    with pytest.raises(NHSeedEvaluationError, match="time_step=1"):
        _check_scored_observations_readable(obs, basin_id="b", lead_hours=LEAD,
                                            windows=(wy.water_year_target_window(2022),), upper_bound=BOUND)


def test_scorer_refuses_unsealed_or_unreceipted_runs(package, contract_path, tmp_path):
    """F1: under a contract the scorer needs dataset flashnh and matching
    train + scored-period receipts, or it refuses before scoring."""
    from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, raw_space_metrics_for_run_period

    issue = pd.date_range("2021-10-01 00:00", "2022-09-30 23:00", freq="h")
    obs = _code(issue + pd.Timedelta(hours=LEAD))
    obs[-LEAD:] = np.nan
    cases = [("generic", dict(dataset="generic"), "loaded unsealed"),
             ("no_val_receipt", dict(receipts=("train",)), "no split-sealing receipt for the validation period"),
             ("no_receipts", dict(receipts=()), "no split-sealing receipt for the train period")]
    for name, kwargs, match in cases:
        run_dir = _contract_run_dir(tmp_path, contract_path, cycle=1, name=name, **kwargs)
        _write_results(run_dir, "validation", issue, obs, obs * 0.9)
        with pytest.raises(NHSeedEvaluationError, match=match):
            raw_space_metrics_for_run_period(run_dir=run_dir, period="validation", epoch=1,
                                             package_root=package["root"], target_variable=TARGET,
                                             lead_hours=LEAD)


# --------------------------------------------------------------------------- hydrograph figures (3b)

def test_hydrograph_rotation_area_scope(tmp_path, contract_path):
    """User decision 2026-11-15 (3b): the figure tool derives a rotation run's
    area from its training windows, and refuses what the scorer refuses."""
    from src.baseline import hydrograph_rendering as hr
    from src.baseline.nh_seed_evaluation import period_results_path

    assert hr._rotation_area_scope(None, period="validation", lead_hours=LEAD) == (None, None)
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.yml").write_text("experiment_name: plain\ndataset: flashnh\n")
    assert hr._rotation_area_scope(plain, period="validation", lead_hours=LEAD) == (None, None)

    run_dir = _contract_run_dir(tmp_path, contract_path, cycle=2, name="sealed")
    sealing = read_run_rotation_sealing(run_dir)
    windows, bound = hr._rotation_area_scope(run_dir, period="validation", lead_hours=LEAD)
    assert windows == sealing.training_windows and bound == sealing.upper_bound == BOUND
    with pytest.raises(hr.HydrographRenderingError, match="differs from the run's split contract lead"):
        hr._rotation_area_scope(run_dir, period="validation", lead_hours=3)
    with pytest.raises(hr.HydrographRenderingError, match="no split-sealing receipt for the test period"):
        hr._rotation_area_scope(run_dir, period="test", lead_hours=LEAD)
    generic = _contract_run_dir(tmp_path, contract_path, dataset="generic", name="generic")
    with pytest.raises(hr.HydrographRenderingError, match="loaded unsealed"):
        hr._rotation_area_scope(generic, period="validation", lead_hours=LEAD)

    pickle_path = period_results_path(run_dir, "validation", 7)
    assert hr._run_dir_of_result_pickle(pickle_path, "validation") == run_dir


def test_hydrograph_refuses_a_result_pickle_whose_run_cannot_be_located(tmp_path, contract_path):
    """User decision 2026-11-15 (question a): a results pickle copied out of
    its run folder, one of another period, or one whose run folder has no
    config.yml is refused, by both result-pickle entry points, before any
    basin file is read."""
    import shutil

    from src.baseline import hydrograph_rendering as hr
    from src.baseline.nh_seed_evaluation import period_results_path

    run_dir = _contract_run_dir(tmp_path, contract_path, cycle=1)
    issue = pd.date_range("2021-10-01", periods=48, freq="h")
    _write_results(run_dir, "validation", issue, np.ones(48), np.ones(48))
    _write_results(run_dir, "test", issue, np.ones(48), np.ones(48))
    pickle_path = period_results_path(run_dir, "validation", 1)
    copied = tmp_path / "copies" / "validation_results.p"
    copied.parent.mkdir()
    shutil.copy(pickle_path, copied)
    no_config = tmp_path / "bare_run"
    shutil.copytree(run_dir / "validation", no_config / "validation")
    candidates = [copied, period_results_path(run_dir, "test", 1),
                  period_results_path(no_config, "validation", 1)]
    for path in candidates:
        with pytest.raises(hr.HydrographRenderingError, match="cannot locate the run directory"):
            hr._run_dir_of_result_pickle(path, "validation")
    common = dict(epoch=1, package_root=tmp_path / "no_package", target_variable=TARGET, lead_hours=LEAD,
                  atlas_csv="unused", out_dir=tmp_path / "out", repo_root=tmp_path)
    for path in candidates:
        with pytest.raises(hr.HydrographRenderingError, match="cannot locate the run directory"):
            hr.render_stage1_compact_comparison_package(result_pickle=path, candidate_id="c", **common)
        with pytest.raises(hr.HydrographRenderingError, match="cannot locate the run directory"):
            hr.render_stage1_hydrographs(result_pickle=path, period="validation", mode="compact", **common)
    # run_dir= also needs the run's config.yml (review r3, finding 1)
    with pytest.raises(hr.HydrographRenderingError, match="cannot locate the run directory"):
        hr.render_stage1_hydrographs(run_dir=no_config, period="validation", mode="compact", **common)
    # refused before the pickle is unpickled (review r3, finding 3): an unreadable copy gives the locate error
    garbage = tmp_path / "garbage" / "validation_results.p"
    garbage.parent.mkdir()
    garbage.write_bytes(b"not a pickle")
    with pytest.raises(hr.HydrographRenderingError, match="cannot locate the run directory"):
        hr.render_stage1_compact_comparison_package(result_pickle=garbage, candidate_id="c", **common)
    with pytest.raises(hr.HydrographRenderingError, match="cannot locate the run directory"):
        hr.render_stage1_hydrographs(result_pickle=garbage, period="validation", mode="compact", **common)


def test_config_with_contract_keys_under_another_dataset_is_refused_at_load(tmp_path, contract_path):
    """User decision 2026-11-15 (question c): NH's Config refuses contract
    keys under a dataset other than flashnh when the config is loaded (i.e.
    at launch), not only at scoring."""
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import check_split_contract_dataset, register_flashnh_dataset

    register_flashnh_dataset()
    register_flashnh_dataset()  # idempotent: the check is chained once
    base = {"experiment_name": "x", "dataset": "generic", "flashnh_split_contract": str(contract_path),
            "flashnh_rotation_cycle": 1}
    with pytest.raises(ValueError, match="would load unsealed"):
        Config(dict(base))
    with pytest.raises(ValueError, match="would load unsealed"):
        Config(dict(base, flashnh_split_contract=None))
    Config(dict(base, dataset="flashnh"))
    Config({"experiment_name": "x", "dataset": "generic"})
    check_split_contract_dataset({"dataset": "generic"})
    with pytest.raises(ValueError, match="are not recognized config keys"):
        Config({"experiment_name": "x", "dataset": "flashnh", "not_a_key": 1})  # NH's own check still runs


def test_contract_dataset_check_sees_the_merged_config_of_an_update(tmp_path, contract_path):
    """Review r3, finding 2: NH's ``update_config`` checks only the partial
    overlay. A partial without ``dataset`` is not refused on its own, and the
    merged config is checked after the update (continue-training and
    finetune overlays)."""
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import check_split_contract_dataset, register_flashnh_dataset

    register_flashnh_dataset()
    register_flashnh_dataset()  # idempotent: update_config is wrapped once
    contract_base = {"experiment_name": "x", "dataset": "flashnh", "flashnh_split_contract": str(contract_path),
                     "flashnh_rotation_cycle": 1}
    # continue-training style overlay setting the cycle without dataset: accepted
    cfg = Config(dict(contract_base))
    cfg.update_config({"flashnh_rotation_cycle": 2, "epochs": 3})
    assert cfg.flashnh_rotation_cycle == 2
    Config({"flashnh_rotation_cycle": 2})  # a partial dict alone is not refused
    check_split_contract_dataset({"flashnh_rotation_cycle": 2}, partial=True)
    with pytest.raises(ValueError, match="would load unsealed"):
        check_split_contract_dataset({"flashnh_rotation_cycle": 2})
    # finetune style: base run under a contract, overlay switches the dataset -> refused on the merged config
    cfg = Config(dict(contract_base))
    cfg.update_config({"run_dir": None, "experiment_name": None})
    with pytest.raises(ValueError, match="would load unsealed"):
        cfg.update_config({"dataset": "generic"})
    # a contract-free base can switch dataset freely
    plain = Config({"experiment_name": "x", "dataset": "flashnh"})
    plain.update_config({"dataset": "generic"})
    # NH dev_mode skips the merged check, as it skips NH's own key check
    cfg = Config(dict(contract_base))
    cfg.update_config({"dataset": "generic"}, dev_mode=True)


@pytest.mark.parametrize("entry", ["run_dir", "result_pickle", "compact_package"])
def test_hydrograph_renderers_pass_training_windows_to_the_area(tmp_path, contract_path, monkeypatch, entry):
    from src.baseline import hydrograph_rendering as hr
    from src.baseline.nh_seed_evaluation import period_results_path

    run_dir = _contract_run_dir(tmp_path, contract_path, cycle=1)
    issue = pd.date_range("2021-10-01", periods=48, freq="h")
    _write_results(run_dir, "validation", issue, np.ones(48), np.ones(48))
    pickle_path = period_results_path(run_dir, "validation", 1)
    seen = []

    class _Stop(Exception):
        pass

    def spy(**kwargs):
        seen.append(kwargs)
        raise _Stop

    monkeypatch.setattr(hr, "load_basin_series", spy)
    monkeypatch.setattr(hr, "load_atlas_selection_csv", lambda _path: pd.DataFrame({"gauge_id": [BASINS[0]]}))
    monkeypatch.setattr(hr, "select_compact_basins", lambda df, target_count: (df, {}))
    common = dict(epoch=1, package_root=tmp_path, target_variable=TARGET, lead_hours=LEAD, atlas_csv="unused",
                  out_dir=tmp_path / "out", repo_root=tmp_path)
    with pytest.raises(_Stop):
        if entry == "compact_package":
            hr.render_stage1_compact_comparison_package(result_pickle=pickle_path, candidate_id="c", **common)
        elif entry == "run_dir":
            hr.render_stage1_hydrographs(run_dir=run_dir, period="validation", mode="compact", **common)
        else:
            hr.render_stage1_hydrographs(result_pickle=pickle_path, period="validation", mode="compact", **common)
    sealing = read_run_rotation_sealing(run_dir)
    assert seen[0]["area_readable_windows"] == sealing.training_windows
    assert seen[0]["area_observation_upper_bound"] == sealing.upper_bound


def test_hydrograph_basin_series_area_reads_only_training_windows(tmp_path):
    """load_basin_series with the training windows derives the exact area from
    WY2023-25 (cycle 1) although every other observation is corrupted (x7),
    and reads no sealed row."""
    from src.baseline.hydrograph_rendering import load_basin_series

    path = tmp_path / "time_series" / f"{BASINS[0]}.nc"
    path.parent.mkdir()
    _write_basin_file(path)
    train_windows = wy.readable_observation_windows(CYCLES[1], "train")
    with xr.open_dataset(path) as data:
        ds = data.load()
    ds["qobs_m3s"].values[~_inside(DATES, train_windows) & (DATES <= BOUND)] *= 7.0
    ds.to_netcdf(path)
    issue = pd.date_range("2021-10-01 00:00", periods=200, freq="h")
    xr_res = xr.Dataset(
        {f"{TARGET}_obs": (("date", "time_step"), np.ones((200, 1))),
         f"{TARGET}_sim": (("date", "time_step"), np.ones((200, 1)))},
        coords={"date": issue, "time_step": [0]})
    with _spy_reads() as reads:
        bs = load_basin_series(results={BASINS[0]: {"1h": {"xr": xr_res}}}, basin_id=BASINS[0],
                               target_variable=TARGET, package_root=tmp_path, lead_hours=LEAD,
                               area_readable_windows=train_windows, area_observation_upper_bound=BOUND)
    _assert_reads_sealed(reads, train_windows)
    assert bs.area_km2 == pytest.approx(AREA_KM2, rel=1e-12)
