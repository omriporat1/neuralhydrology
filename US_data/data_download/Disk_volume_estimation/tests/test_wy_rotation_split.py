"""Tests for the water-year rotation split contract (src/baseline/wy_rotation_split.py).

Expected dates, hour counts and boundary losses are written out by hand from
the decision-log table (2026-11-11) and the convention "water year k = target
valid times Oct 1 00:00 of k-1 .. Sep 30 23:00 of k, naive UTC; the package
starts 2020-10-14 00:00; lead 6 h", never computed with the module's helpers.

The vertical test at the bottom builds FlashNHDataset (the consumer) on a
synthetic package laid over the real WY2021-WY2025 dates, feeds it the
contract's NH dates, and checks which target hours the dataset actually admits
and which target values survive in its training data.
"""
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline.wy_rotation_split import (
    LEAD_HOURS,
    ROTATION_TABLE_V001,
    RotationCycle,
    TimeWindow,
    WyRotationSplitError,
    admitted_target_windows,
    boundary_unadmitted_target_hours,
    build_split_contract,
    cycle_nh_dates,
    load_split_contract,
    merge_contiguous_water_years,
    readable_observation_windows,
    validate_rotation_table,
    verify_basin_population,
    verify_split_contract,
    water_year_target_window,
    write_split_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS = pd.Timestamp


def _cycle(n):
    return ROTATION_TABLE_V001[n - 1]


# --------------------------------------------------------------------------- water years


@pytest.mark.parametrize("wy, first, last, hours", [
    (2021, "2020-10-01 00:00", "2021-09-30 23:00", 8760),
    (2024, "2023-10-01 00:00", "2024-09-30 23:00", 8784),  # Feb 29 2024
    (2026, "2025-10-01 00:00", "2026-09-30 23:00", 8760),
])
def test_water_year_target_window_bounds(wy, first, last, hours):
    w = water_year_target_window(wy)
    assert (w.first, w.last, w.n_hours) == (TS(first), TS(last), hours)


def test_merge_contiguous_water_years():
    assert merge_contiguous_water_years([2025, 2021, 2022]) == ((2021, 2022), (2025,))
    assert merge_contiguous_water_years([2023, 2024, 2025]) == ((2023, 2024, 2025),)
    with pytest.raises(WyRotationSplitError):
        merge_contiguous_water_years([2021, 2021])
    with pytest.raises(WyRotationSplitError):
        merge_contiguous_water_years([])


def test_time_window_refuses_tz_aware_off_hour_and_reversed():
    with pytest.raises(WyRotationSplitError):
        TimeWindow(TS("2021-01-01", tz="UTC"), TS("2021-01-02", tz="UTC"))
    with pytest.raises(WyRotationSplitError):
        TimeWindow(TS("2021-01-01 00:30"), TS("2021-01-02"))
    with pytest.raises(WyRotationSplitError):
        TimeWindow(TS("2021-01-02"), TS("2021-01-01"))


# --------------------------------------------------------------------------- cycle table


def test_frozen_table_matches_the_decision_log():
    rows = [(c.cycle, tuple(sorted(c.train_wys)), c.checkpoint_wy, c.assessment_wy) for c in ROTATION_TABLE_V001]
    assert rows == [
        (1, (2023, 2024, 2025), 2022, 2021),
        (2, (2021, 2024, 2025), 2023, 2022),
        (3, (2021, 2022, 2025), 2024, 2023),
        (4, (2021, 2022, 2023), 2025, 2024),
        (5, (2022, 2023, 2024), 2021, 2025),
    ]
    validate_rotation_table(ROTATION_TABLE_V001)


@pytest.mark.parametrize("mutate, message", [
    (lambda t: t[:4], "one cycle per development year"),
    (lambda t: (RotationCycle(1, (2023, 2024, 2026), 2022, 2021),) + t[1:], "protected test year"),
    (lambda t: (RotationCycle(1, (2023, 2024, 2025), 2022, 2022),) + t[1:], "more than one role"),
    (lambda t: (RotationCycle(1, (2023, 2024, 2020), 2022, 2021),) + t[1:], "not a development water year"),
    (lambda t: (RotationCycle(1, (2023, 2024, 2025), 2021, 2022),) + t[1:], "exactly once"),
    (lambda t: (t[1], t[0]) + t[2:], "numbered 1..n"),
])
def test_invalid_tables_are_refused(mutate, message):
    with pytest.raises(WyRotationSplitError, match=message):
        validate_rotation_table(mutate(ROTATION_TABLE_V001))


# --------------------------------------------------------------------------- NH dates


@pytest.mark.parametrize("n, expected", [
    (1, {"train_start_date": ["01/10/2022"], "train_end_date": ["30/09/2025"],
         "validation_start_date": "01/10/2021", "validation_end_date": "30/09/2022",
         "test_start_date": "14/10/2020", "test_end_date": "30/09/2021"}),
    (2, {"train_start_date": ["14/10/2020", "01/10/2023"], "train_end_date": ["30/09/2021", "30/09/2025"],
         "validation_start_date": "01/10/2022", "validation_end_date": "30/09/2023",
         "test_start_date": "01/10/2021", "test_end_date": "30/09/2022"}),
    (3, {"train_start_date": ["14/10/2020", "01/10/2024"], "train_end_date": ["30/09/2022", "30/09/2025"],
         "validation_start_date": "01/10/2023", "validation_end_date": "30/09/2024",
         "test_start_date": "01/10/2022", "test_end_date": "30/09/2023"}),
    (4, {"train_start_date": ["14/10/2020"], "train_end_date": ["30/09/2023"],
         "validation_start_date": "01/10/2024", "validation_end_date": "30/09/2025",
         "test_start_date": "01/10/2023", "test_end_date": "30/09/2024"}),
    (5, {"train_start_date": ["01/10/2021"], "train_end_date": ["30/09/2024"],
         "validation_start_date": "14/10/2020", "validation_end_date": "30/09/2021",
         "test_start_date": "01/10/2024", "test_end_date": "30/09/2025"}),
])
def test_cycle_nh_dates(n, expected):
    assert cycle_nh_dates(_cycle(n)) == expected


# --------------------------------------------------------------------------- roles and leaks


def _hours(windows) -> set:
    out = set()
    for w in windows:
        out.update(pd.date_range(w.first, w.last, freq="h"))
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_readable_windows_partition_the_development_years_and_never_touch_wy2026(n):
    c = _cycle(n)
    parts = {p: _hours(readable_observation_windows(c, p)) for p in ("train", "validation", "test")}
    assert not parts["train"] & parts["validation"]
    assert not parts["train"] & parts["test"]
    assert not parts["validation"] & parts["test"]
    union = parts["train"] | parts["validation"] | parts["test"]
    assert union == set(pd.date_range("2020-10-01 00:00", "2025-09-30 23:00", freq="h"))
    assert max(union) < TS("2025-10-01 00:00")


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("period", ["train", "validation", "test"])
def test_admitted_targets_lie_inside_the_readable_windows(n, period):
    c = _cycle(n)
    assert _hours(admitted_target_windows(c, period)) <= _hours(readable_observation_windows(c, period))


def test_boundary_losses_hand_derived():
    # Cycle 2 train = [WY2021 (clipped at 2020-10-14)] + [WY2024-2025 merged]:
    # 6 h lost at 2020-10-14 00-05 and 6 h at 2023-10-01 00-05; none inside the merged slice.
    assert boundary_unadmitted_target_hours(_cycle(2), "train") == {2021: 6, 2024: 6, 2025: 0}
    # Cycle 4 train = WY2021-2023 in one slice.
    assert boundary_unadmitted_target_hours(_cycle(4), "train") == {2021: 6, 2022: 0, 2023: 0}
    # Every single-year evaluation period loses its first 6 target hours.
    for c in ROTATION_TABLE_V001:
        for period in ("validation", "test"):
            assert list(boundary_unadmitted_target_hours(c, period).values()) == [LEAD_HOURS]


def test_unknown_period_and_role_refused():
    with pytest.raises(WyRotationSplitError):
        readable_observation_windows(_cycle(1), "assessment")
    with pytest.raises(WyRotationSplitError):
        _cycle(1).role_water_years("test")


# --------------------------------------------------------------------------- contract


def test_contract_round_trip(tmp_path):
    contract = build_split_contract()
    assert contract["status"] == "FROZEN"
    assert contract["cycles"][3]["periods"]["validation"]["readable_observation_windows"] == [
        {"first_utc": "2024-10-01T00:00:00Z", "last_utc": "2025-09-30T23:00:00Z", "n_hours": 8760}]
    path = write_split_contract(tmp_path / "contract.json", contract)
    assert load_split_contract(path) == contract
    assert build_split_contract() == contract  # deterministic


def test_contract_checksum_tamper_refused():
    contract = build_split_contract()
    contract["checksum_sha256"] = "0" * 64
    with pytest.raises(WyRotationSplitError, match="checksum"):
        verify_split_contract(contract)


def test_contract_content_tamper_with_fresh_checksum_refused():
    from src.baseline.atomic_shard_store import canonical_json_sha256

    contract = build_split_contract()
    tampered = copy.deepcopy(contract)
    # widen cycle 1's training window back into its checkpoint year
    tampered["cycles"][0]["periods"]["train"]["readable_observation_windows"][0]["first_utc"] = "2021-10-01T00:00:00Z"
    body = {k: v for k, v in tampered.items() if k != "checksum_sha256"}
    tampered["checksum_sha256"] = canonical_json_sha256(body)
    with pytest.raises(WyRotationSplitError, match="differs from the frozen v001 derivation"):
        verify_split_contract(tampered)


def _rechecksum(contract: dict) -> dict:
    from src.baseline.atomic_shard_store import canonical_json_sha256

    body = {k: v for k, v in contract.items() if k != "checksum_sha256"}
    contract["checksum_sha256"] = canonical_json_sha256(body)
    return contract


def _self_consistent_variant(*, cycles=None, lead_hours=LEAD_HOURS, data_first_hour=None) -> dict:
    """A contract that is internally consistent (derived by the module's own
    body builder) but differs from frozen v001 in its table, lead or start."""
    from src.baseline import wy_rotation_split as mod

    body = mod._contract_body(cycles if cycles is not None else mod.ROTATION_TABLE_V001,
                              lead_hours=lead_hours,
                              data_first_hour=data_first_hour if data_first_hour is not None
                              else mod.PACKAGE_FIRST_HOUR)
    return _rechecksum(body)


def test_self_consistent_contract_with_swapped_checkpoint_and_assessment_refused():
    from src.baseline import wy_rotation_split as mod

    swapped = tuple(mod.RotationCycle(c.cycle, c.train_wys, c.assessment_wy, c.checkpoint_wy)
                    for c in mod.ROTATION_TABLE_V001)
    with pytest.raises(WyRotationSplitError, match="ROTATION_TABLE_V001"):
        verify_split_contract(_self_consistent_variant(cycles=swapped))


def test_self_consistent_contract_with_other_lead_refused():
    with pytest.raises(WyRotationSplitError, match="lead_hours"):
        verify_split_contract(_self_consistent_variant(lead_hours=1))


def test_contract_with_float_lead_refused():
    contract = copy.deepcopy(build_split_contract())
    contract["conventions"]["lead_hours"] = float(LEAD_HOURS)
    with pytest.raises(WyRotationSplitError, match="lead_hours"):
        verify_split_contract(_rechecksum(contract))


def test_self_consistent_contract_with_other_first_hour_refused():
    with pytest.raises(WyRotationSplitError, match="package_first_hour_utc"):
        verify_split_contract(_self_consistent_variant(data_first_hour=TS("2020-10-01 00:00")))


def test_contract_with_moved_observation_upper_bound_refused():
    contract = copy.deepcopy(build_split_contract())
    contract["observation_upper_bound_utc"] = "2025-10-01T05:00:00Z"
    with pytest.raises(WyRotationSplitError, match="observation_upper_bound_utc"):
        verify_split_contract(_rechecksum(contract))


def test_observation_upper_bound_is_frozen_and_respected_by_every_window():
    contract = build_split_contract()
    assert contract["observation_upper_bound_utc"] == "2025-09-30T23:00:00Z"
    for c in contract["cycles"]:
        for period, spec in c["periods"].items():
            for w in spec["readable_observation_windows"]:
                assert w["last_utc"] <= "2025-09-30T23:00:00Z", (c["cycle"], period, w)


def test_contract_with_invalid_table_refused():
    from src.baseline.atomic_shard_store import canonical_json_sha256

    contract = build_split_contract()
    tampered = copy.deepcopy(contract)
    tampered["cycles"][0]["assessment_wy"] = 2026
    body = {k: v for k, v in tampered.items() if k != "checksum_sha256"}
    tampered["checksum_sha256"] = canonical_json_sha256(body)
    with pytest.raises(WyRotationSplitError):
        verify_split_contract(tampered)


def test_basin_population_pin_matches_the_tracked_list():
    basins = verify_basin_population(build_split_contract(), PROJECT_ROOT)
    assert len(basins) == 2307 and all(len(b) >= 8 for b in basins)


def test_basin_population_pin_is_the_portable_membership_hash():
    from src.baseline.devpop_common120_audit_contract import canonical_membership_sha256

    contract = build_split_contract()
    pin = contract["basin_population"]["membership_sha256"]
    assert pin == "397ab432564c18c3abc5158a47ada2b28840bbf6f0c213d2475444fded33858f"
    basins = verify_basin_population(contract, PROJECT_ROOT)
    assert canonical_membership_sha256(basins) == pin


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_basin_population_verifies_under_lf_and_crlf_checkouts(tmp_path, newline):
    contract = build_split_contract()
    basins = verify_basin_population(contract, PROJECT_ROOT)
    target = tmp_path / contract["basin_population"]["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(newline.join(basins).encode("utf-8") + newline.encode("utf-8"))
    assert verify_basin_population(contract, tmp_path) == basins


def test_basin_population_mismatch_refused(tmp_path):
    contract = build_split_contract()
    target = tmp_path / contract["basin_population"]["path"]
    target.parent.mkdir(parents=True)
    basins = verify_basin_population(contract, PROJECT_ROOT)
    target.write_text("\n".join(basins[:-1] + ["01010000"]) + "\n")  # right count, one basin swapped
    with pytest.raises(WyRotationSplitError, match="sha256"):
        verify_basin_population(contract, tmp_path)


def test_basin_population_check_verifies_the_contract_first():
    contract = copy.deepcopy(build_split_contract())
    contract["basin_population"]["path"] = "config/some_other_list.txt"
    with pytest.raises(WyRotationSplitError, match="checksum"):
        verify_basin_population(contract, PROJECT_ROOT)
    with pytest.raises(WyRotationSplitError, match="differs from the frozen v001 derivation"):
        verify_basin_population(_rechecksum(contract), PROJECT_ROOT)


# --------------------------------------------------------------------------- vertical: FlashNHDataset consumer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import build_synthetic_package, prepare_run_dirs  # noqa: E402

SEQ = 3
BASIN = "SYN01"
PKG_START = TS("2020-10-14 00:00")
PKG_END = TS("2025-10-02 23:00")  # a day past WY2025 so its last lead targets exist


def _write_nh_dates(cfg_path: Path, dates: dict) -> None:
    text = cfg_path.read_text()
    for key, value in dates.items():
        rendered = "[" + ", ".join(f'"{v}"' for v in value) + "]" if isinstance(value, list) else f'"{value}"'
        text, n = re.subn(rf"^{key}:.*$", f"{key}: {rendered}", text, flags=re.M)
        assert n == 1, key
    text, n = re.subn(r"^loss:.*$", "loss: NSE", text, flags=re.M)
    assert n == 1
    cfg_path.write_text(text)


@pytest.fixture(scope="module")
def cycle2_datasets(tmp_path_factory):
    from neuralhydrology.datasetzoo import get_dataset
    from neuralhydrology.utils.config import Config

    from src.baseline.nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    base = tmp_path_factory.mktemp("wy_rotation_vertical")
    n_hours = int((PKG_END - PKG_START) / pd.Timedelta(hours=1)) + 1
    cfg_path = build_synthetic_package(base, basins=[BASIN], seq_length=SEQ, lead_hours=LEAD_HOURS,
                                       research_start=str(PKG_START), n_hours=n_hours)
    _write_nh_dates(cfg_path, cycle_nh_dates(_cycle(2)))
    cfg = Config(cfg_path)
    prepare_run_dirs(cfg, base, "wy_rotation")
    train = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})
    out = {"train": train}
    for period in ("validation", "test"):
        out[period] = get_dataset(cfg=cfg, is_train=False, period=period, scaler=train.scaler)
    return out


def _admitted_target_times(ds) -> set:
    dates = ds._dates[BASIN][ds.frequencies[0]]
    lead = pd.Timedelta(hours=LEAD_HOURS)
    return {pd.Timestamp(dates[idx[0]]) + lead for b, idx in ds.lookup_table.values() if b == BASIN}


@pytest.mark.parametrize("period", ["train", "validation", "test"])
def test_dataset_admits_exactly_the_contract_target_windows(cycle2_datasets, period):
    """Cycle 2: train = WY2021 + WY2024-25 (two slices), validation = WY2023,
    test = WY2022. The dataset's admitted target hours must equal the
    contract's admitted windows, except targets whose issue hour lacks SEQ-1
    hours of package history (only at the package start, WY2021)."""
    actual = _admitted_target_times(cycle2_datasets[period])
    expected = _hours(admitted_target_windows(_cycle(2), period))
    first_full_history_target = PKG_START + pd.Timedelta(hours=SEQ - 1 + LEAD_HOURS)
    expected = {t for t in expected if t >= first_full_history_target}
    assert actual == expected, (f"extra {sorted(actual - expected)[:5]}, missing {sorted(expected - actual)[:5]}")
    assert actual <= _hours(readable_observation_windows(_cycle(2), period))


def test_training_data_holds_no_target_value_from_a_non_training_year(cycle2_datasets):
    """Every non-NaN target left in the loaded training data has its target
    time inside a training water year. Synthetic target value = hour index of
    the target time. (Un-normalising with the scaler's own centre/scale is an
    exact inverse, so this part does not test the scaler; the next test does.)"""
    train = cycle2_datasets["train"]
    freq = train.frequencies[0]
    target = f"qobs_lead{LEAD_HOURS}"
    center = float(train.scaler["xarray_feature_center"][target].values)
    scale = float(train.scaler["xarray_feature_scale"][target].values)
    y = train._y[BASIN][freq][:, 0].numpy().astype(np.float64) * scale + center  # undo NH normalisation
    target_times = {PKG_START + pd.Timedelta(hours=int(round(v))) for v in y[~np.isnan(y)]}
    readable = _hours(readable_observation_windows(_cycle(2), "train"))
    assert target_times and target_times <= readable
    # and the checkpoint / assessment years (WY2023 / WY2022) are fully absent
    excluded = _hours(readable_observation_windows(_cycle(2), "validation")) | \
        _hours(readable_observation_windows(_cycle(2), "test"))
    assert not target_times & excluded
    assert train.flashnh_filter_stats.n_period_slices == 2


def test_target_scaler_and_per_basin_std_use_only_training_year_targets(cycle2_datasets):
    """Hand-derived, independent of the module: cycle 2 trains on
    [2020-10-14 00:00, 2021-09-30 23:00] and [2023-10-01 00:00, 2025-09-30 23:00].
    Under convention A the targets allowed into the scaler and the per-basin
    NSE std are the target times from slice first + lead to slice last. The
    target scaler centre / scale and the per-basin std must equal the mean and
    population std (ddof=0, as xarray / np.nanstd) of those hour indices."""
    lead = pd.Timedelta(hours=LEAD_HOURS)
    slices = [(TS("2020-10-14 00:00"), TS("2021-09-30 23:00")),
              (TS("2023-10-01 00:00"), TS("2025-09-30 23:00"))]
    allowed = np.concatenate([
        np.arange(int((first + lead - PKG_START) / pd.Timedelta(hours=1)),
                  int((last - PKG_START) / pd.Timedelta(hours=1)) + 1)
        for first, last in slices]).astype(np.float64)
    train = cycle2_datasets["train"]
    target = f"qobs_lead{LEAD_HOURS}"
    center = float(train.scaler["xarray_feature_center"][target].values)
    scale = float(train.scaler["xarray_feature_scale"][target].values)
    assert center == pytest.approx(allowed.mean(), rel=1e-6)
    assert scale == pytest.approx(allowed.std(), rel=1e-5)
    std = float(train._per_basin_target_stds[BASIN].flatten()[0])
    assert std == pytest.approx(allowed.std(), rel=1e-5)
