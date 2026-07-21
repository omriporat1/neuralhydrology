"""Tests for src/baseline/package_netcdf.py (Compact Scientific Package
builder, Gate 2: pure per-basin NetCDF serialization).

Synthetic assembled tables are produced through the committed Gate 1 API
(``assemble_basin_package_table``) wherever practical, matching the
convention established in tests/test_package_assembly.py.

Note (Gate 3 obligation, not solved here): the production builder must
eventually call ``assemble_basin_package_table(..., expected_index=<the
45,720-hour Stage 1 grid>)`` upstream of this serializer; this module
accepts whatever validated hourly table it is given.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline.package_assembly import assemble_basin_package_table
from src.baseline.package_netcdf import (
    EXPECTED_VARIABLES,
    NETCDF_ENGINE,
    PackageNetCDFError,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_basin_dataset,
    validate_basin_netcdf_file,
    write_basin_dataset_netcdf,
)

_UNITS = {
    "mrms_qpe_1h_mm": "mm",
    "rtma_2t_K": "K",
    "rtma_2d_K": "K",
    "rtma_2sh_kgkg": "kg kg-1",
    "rtma_10u_ms": "m s-1",
    "rtma_10v_ms": "m s-1",
    "mrms_qpe_1h_mm_gap": "1",
    "rtma_gap": "1",
    "qobs_m3s": "m3 s-1",
    "qobs_mm_per_h_lead01": "mm h-1",
    "qobs_mm_per_h_lead03": "mm h-1",
    "qobs_mm_per_h_lead06": "mm h-1",
    "qobs_mm_per_h_lead12": "mm h-1",
}


def _hourly_index(n, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def _valid_forcing(idx, **overrides):
    n = len(idx)
    data = {
        "mrms_qpe_1h_mm": np.linspace(0.0, 1.0, n),
        "rtma_2t_K": np.linspace(270.0, 280.0, n),
        "rtma_2d_K": np.linspace(260.0, 270.0, n),
        "rtma_2sh_kgkg": np.linspace(0.001, 0.002, n),
        "rtma_10u_ms": np.linspace(-2.0, 2.0, n),
        "rtma_10v_ms": np.linspace(-1.0, 1.0, n),
        "mrms_qpe_1h_mm_gap": np.zeros(n, dtype=np.float32),
        "rtma_gap": np.zeros(n, dtype=np.float32),
    }
    data.update(overrides)
    return pd.DataFrame(data, index=idx)


def _valid_qobs(idx, values=None):
    n = len(idx)
    if values is None:
        values = np.arange(n, dtype=np.float64)
    return pd.Series(values, index=idx, name="qobs_m3s")


def _assembled_table(n=12, start="2023-01-01", forcing_overrides=None, qobs_values=None, area_km2=3.6):
    idx = _hourly_index(n, start=start)
    forcing = _valid_forcing(idx, **(forcing_overrides or {}))
    qobs = _valid_qobs(idx, values=qobs_values)
    return assemble_basin_package_table(forcing, qobs, area_km2=area_km2)


# ---------------------------------------------------------------------------
# 1. exact dataset dimension and variable names
# ---------------------------------------------------------------------------


def test_dataset_dimensions_and_variable_names_exact():
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    assert set(ds.dims) == {"time"}
    assert set(ds.data_vars) == set(EXPECTED_VARIABLES)
    for name in EXPECTED_VARIABLES:
        assert ds[name].dims == ("time",)


# ---------------------------------------------------------------------------
# 2. exact time-coordinate round trip
# ---------------------------------------------------------------------------


def test_time_coordinate_round_trip(tmp_path):
    table = _assembled_table(n=15)
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    with xr.open_dataset(path, engine=NETCDF_ENGINE) as reopened:
        actual = pd.DatetimeIndex(reopened["time"].values)
    assert actual.equals(table.index)


# ---------------------------------------------------------------------------
# 3-5. gauge ID lengths accepted (8 with leading zero, 9, 15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gauge_id",
    ["01019000", "103366092", "393109104464500"],
    ids=["len8_leading_zero", "len9", "len15"],
)
def test_valid_gauge_id_lengths_accepted(gauge_id):
    table = _assembled_table()
    ds = build_basin_dataset(table, gauge_id)
    assert ds.attrs["gauge_id"] == gauge_id


# ---------------------------------------------------------------------------
# 6-7. invalid gauge IDs rejected
# ---------------------------------------------------------------------------


def test_integer_gauge_id_rejected():
    table = _assembled_table()
    with pytest.raises(PackageNetCDFError):
        build_basin_dataset(table, 1019000)


@pytest.mark.parametrize(
    "gauge_id",
    ["", "  01019000", "01019000  ", " 01019000 ", "abc12345", "1234567", "0101900012345678"],
    ids=["empty", "leading_ws", "trailing_ws", "both_ws", "non_digit", "short_unpadded", "bad_length"],
)
def test_invalid_string_gauge_ids_rejected(gauge_id):
    table = _assembled_table()
    with pytest.raises(PackageNetCDFError):
        build_basin_dataset(table, gauge_id)


# ---------------------------------------------------------------------------
# 8. required variable units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected_units", list(_UNITS.items()))
def test_variable_units_recorded(name, expected_units):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    assert ds[name].attrs["units"] == expected_units


# ---------------------------------------------------------------------------
# 9. lead-hour metadata for 1/3/6/12
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column,lead_hours",
    [
        ("qobs_mm_per_h_lead01", 1),
        ("qobs_mm_per_h_lead03", 3),
        ("qobs_mm_per_h_lead06", 6),
        ("qobs_mm_per_h_lead12", 12),
    ],
)
def test_lead_hour_metadata(column, lead_hours):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    assert ds[column].attrs["lead_hours"] == lead_hours
    assert ds[column].attrs["role"] == "training_target"


# ---------------------------------------------------------------------------
# 10. raw qobs audit/provenance metadata
# ---------------------------------------------------------------------------


def test_raw_qobs_audit_provenance_metadata():
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    assert ds["qobs_m3s"].attrs["role"] == "audit_provenance_not_training_target"


# ---------------------------------------------------------------------------
# 11-13. NaNs preserved (forcing, qobs, lead-tail)
# ---------------------------------------------------------------------------


def test_forcing_nans_preserved():
    idx = _hourly_index(8)
    mrms = np.array([0.0, np.nan, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
    table = assemble_basin_package_table(
        _valid_forcing(idx, mrms_qpe_1h_mm=mrms), _valid_qobs(idx), area_km2=3.6
    )
    ds = build_basin_dataset(table, "01019000")
    np.testing.assert_array_equal(np.isnan(ds["mrms_qpe_1h_mm"].values), np.isnan(mrms))


def test_qobs_nans_preserved():
    idx = _hourly_index(8)
    q = pd.Series([0.0, 1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0], index=idx)
    table = assemble_basin_package_table(_valid_forcing(idx), q, area_km2=3.6)
    ds = build_basin_dataset(table, "01019000")
    np.testing.assert_array_equal(np.isnan(ds["qobs_m3s"].values), np.isnan(q.to_numpy()))


def test_lead_tail_nans_preserved():
    table = _assembled_table(n=20)
    ds = build_basin_dataset(table, "01019000")
    tail = ds["qobs_mm_per_h_lead12"].values[-12:]
    assert np.all(np.isnan(tail))
    assert not np.isnan(ds["qobs_mm_per_h_lead12"].values[-13])


# ---------------------------------------------------------------------------
# 14. exact NaN masks after disk round trip
# ---------------------------------------------------------------------------


def test_nan_masks_exact_after_disk_round_trip(tmp_path):
    idx = _hourly_index(20)
    mrms = np.linspace(0.0, 1.0, 20)
    mrms[3] = np.nan
    mrms[11] = np.nan
    q = np.arange(20.0)
    q[7] = np.nan
    table = assemble_basin_package_table(
        _valid_forcing(idx, mrms_qpe_1h_mm=mrms), _valid_qobs(idx, values=q), area_km2=3.6
    )
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    validate_basin_netcdf_file(path, table, "01019000")  # raises on any mismatch

    with xr.open_dataset(path, engine=NETCDF_ENGINE) as reopened:
        reopened.load()
        for name in EXPECTED_VARIABLES:
            if name in ("mrms_qpe_1h_mm_gap", "rtma_gap"):
                continue
            disk_nan = np.isnan(reopened[name].values)
            source_nan = table[name].isna().to_numpy()
            np.testing.assert_array_equal(disk_nan, source_nan)


# ---------------------------------------------------------------------------
# 15. finite values round-trip within declared tolerance
# ---------------------------------------------------------------------------


def test_finite_values_round_trip_within_tolerance(tmp_path):
    table = _assembled_table(n=24)
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    with xr.open_dataset(path, engine=NETCDF_ENGINE) as reopened:
        reopened.load()
        for name in EXPECTED_VARIABLES:
            disk = reopened[name].values.astype(np.float64)
            source = table[name].to_numpy(dtype=np.float64)
            finite = np.isfinite(source)
            if finite.any():
                np.testing.assert_allclose(disk[finite], source[finite], rtol=1e-5, atol=0.0)


# ---------------------------------------------------------------------------
# 16-17. binary gap flags remain binary / invalid gap flags rejected
# ---------------------------------------------------------------------------


def test_binary_gap_flags_remain_binary(tmp_path):
    idx = _hourly_index(6)
    mrms_gap = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    table = assemble_basin_package_table(
        _valid_forcing(idx, mrms_qpe_1h_mm_gap=mrms_gap), _valid_qobs(idx), area_km2=3.6
    )
    ds = build_basin_dataset(table, "01019000")
    assert ds["mrms_qpe_1h_mm_gap"].values.dtype == np.int8
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    with xr.open_dataset(path, engine=NETCDF_ENGINE) as reopened:
        values = reopened["mrms_qpe_1h_mm_gap"].values
        assert set(np.unique(values)).issubset({0, 1})
        np.testing.assert_array_equal(values, mrms_gap.astype(values.dtype))


def test_invalid_gap_flags_rejected_directly_on_table():
    idx = _hourly_index(4)
    n = len(idx)
    data = {
        "mrms_qpe_1h_mm": np.zeros(n),
        "rtma_2t_K": np.zeros(n),
        "rtma_2d_K": np.zeros(n),
        "rtma_2sh_kgkg": np.zeros(n),
        "rtma_10u_ms": np.zeros(n),
        "rtma_10v_ms": np.zeros(n),
        "mrms_qpe_1h_mm_gap": [0.0, 2.0, 0.0, 1.0],  # invalid: 2.0 is out of {0, 1}
        "rtma_gap": np.zeros(n),
        "qobs_m3s": np.zeros(n),
        "qobs_mm_per_h_lead01": np.zeros(n),
        "qobs_mm_per_h_lead03": np.zeros(n),
        "qobs_mm_per_h_lead06": np.zeros(n),
        "qobs_mm_per_h_lead12": np.zeros(n),
    }
    table = pd.DataFrame(data, index=idx)
    with pytest.raises(PackageNetCDFError, match="gap-flag"):
        build_basin_dataset(table, "01019000")


# ---------------------------------------------------------------------------
# 18-20. missing / extra / wrong-order variable rejection
# ---------------------------------------------------------------------------


def test_missing_variable_rejected():
    table = _assembled_table().drop(columns=["rtma_gap"])
    with pytest.raises(PackageNetCDFError, match="missing"):
        build_basin_dataset(table, "01019000")


def test_extra_variable_rejected():
    table = _assembled_table().copy()
    table["rtma_sp_Pa"] = 0.0
    with pytest.raises(PackageNetCDFError, match="unapproved"):
        build_basin_dataset(table, "01019000")


def test_wrong_column_order_rejected():
    table = _assembled_table()
    reordered = table[list(reversed(list(table.columns)))]
    with pytest.raises(PackageNetCDFError, match="order"):
        build_basin_dataset(reordered, "01019000")


# ---------------------------------------------------------------------------
# 21-23. timezone-aware / duplicate / non-hourly / infinities rejected
# ---------------------------------------------------------------------------


def test_timezone_aware_index_rejected():
    table = _assembled_table()
    tz_table = table.copy()
    tz_table.index = tz_table.index.tz_localize("UTC")
    with pytest.raises(PackageNetCDFError, match="timezone-naive"):
        build_basin_dataset(tz_table, "01019000")


def test_duplicate_index_rejected():
    table = _assembled_table(n=4)
    bad_idx = pd.DatetimeIndex(
        ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 01:00", "2023-01-01 02:00"]
    )
    bad = table.copy()
    bad.index = bad_idx
    with pytest.raises(PackageNetCDFError, match="duplicate"):
        build_basin_dataset(bad, "01019000")


def test_non_hourly_index_rejected():
    table = _assembled_table(n=5)
    bad = table.copy()
    bad.index = pd.date_range("2023-01-01", periods=5, freq="30min")
    with pytest.raises(PackageNetCDFError, match="hourly"):
        build_basin_dataset(bad, "01019000")


def test_infinite_scientific_value_rejected():
    table = _assembled_table(n=6).copy()
    table.loc[table.index[2], "rtma_2t_K"] = np.inf
    with pytest.raises(PackageNetCDFError, match="infinite"):
        build_basin_dataset(table, "01019000")


# ---------------------------------------------------------------------------
# 24-25. overwrite protection
# ---------------------------------------------------------------------------


def test_existing_destination_rejected_by_default(tmp_path):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    with pytest.raises(PackageNetCDFError, match="exists"):
        write_basin_dataset_netcdf(ds, path)


def test_explicit_overwrite_succeeds(tmp_path):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    write_basin_dataset_netcdf(ds, path, overwrite=True)
    assert path.exists()


# ---------------------------------------------------------------------------
# 26-27. failed write leaves no partial destination / temp file cleanup
# ---------------------------------------------------------------------------


def test_failed_write_leaves_no_partial_destination_and_cleans_up_temp(tmp_path, monkeypatch):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"

    original_to_netcdf = xr.Dataset.to_netcdf

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", _boom)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        write_basin_dataset_netcdf(ds, path)
    monkeypatch.setattr(xr.Dataset, "to_netcdf", original_to_netcdf)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_missing_parent_directory_rejected_without_create_parent(tmp_path):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "does_not_exist_yet" / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="parent"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


# ---------------------------------------------------------------------------
# 28. deterministic in-memory dataset construction for identical inputs
# ---------------------------------------------------------------------------


def test_deterministic_dataset_construction():
    table = _assembled_table(n=10)
    ds1 = build_basin_dataset(table.copy(), "01019000")
    ds2 = build_basin_dataset(table.copy(), "01019000")
    xr.testing.assert_identical(ds1, ds2)


# ---------------------------------------------------------------------------
# 29. disk file reopens with the explicitly selected engine
# ---------------------------------------------------------------------------


def test_disk_file_reopens_with_selected_engine(tmp_path):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    with xr.open_dataset(path, engine=NETCDF_ENGINE) as reopened:
        assert reopened.attrs["package_schema_name"] == SCHEMA_NAME
        assert reopened.attrs["package_schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 30. serializer does not modify the caller's input DataFrame
# ---------------------------------------------------------------------------


def test_serializer_does_not_modify_input_dataframe():
    table = _assembled_table()
    before = table.copy(deep=True)
    build_basin_dataset(table, "01019000")
    pd.testing.assert_frame_equal(table, before)


# ---------------------------------------------------------------------------
# validate_basin_netcdf_file: end-to-end acceptance
# ---------------------------------------------------------------------------


def test_validate_basin_netcdf_file_accepts_valid_round_trip(tmp_path):
    table = _assembled_table(n=30)
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    validate_basin_netcdf_file(path, table, "01019000")


def test_validate_basin_netcdf_file_rejects_wrong_gauge_id(tmp_path):
    table = _assembled_table()
    ds = build_basin_dataset(table, "01019000")
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    with pytest.raises(PackageNetCDFError, match="gauge_id"):
        validate_basin_netcdf_file(path, table, "01019001")


# ---------------------------------------------------------------------------
# write_basin_dataset_netcdf: dataset-level contract gate rejects a
# malformed/mutated in-memory Dataset before it is ever promoted to disk.
# ---------------------------------------------------------------------------


def _valid_dataset(n=10, gauge_id="01019000"):
    table = _assembled_table(n=n)
    return build_basin_dataset(table, gauge_id)


def test_write_rejects_wrong_schema_name(tmp_path):
    ds = _valid_dataset()
    ds.attrs["package_schema_name"] = "not_the_right_schema"
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="package_schema_name"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_missing_gauge_id(tmp_path):
    ds = _valid_dataset()
    del ds.attrs["gauge_id"]
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="gauge_id"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_wrong_variable_set(tmp_path):
    ds = _valid_dataset().drop_vars(["rtma_gap"])
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="variable set"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_wrong_units(tmp_path):
    ds = _valid_dataset()
    ds["rtma_2t_K"].attrs["units"] = "degC"
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="units"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_wrong_raw_qobs_role(tmp_path):
    ds = _valid_dataset()
    ds["qobs_m3s"].attrs["role"] = "training_target"
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="role"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_wrong_lead_hours_metadata(tmp_path):
    ds = _valid_dataset()
    ds["qobs_mm_per_h_lead01"].attrs["lead_hours"] = 99
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="lead_hours"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_nonbinary_gap_value(tmp_path):
    ds = _valid_dataset()
    ds["rtma_gap"].values[0] = 2
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="0/1"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_infinite_continuous_value(tmp_path):
    ds = _valid_dataset()
    ds["rtma_2t_K"].values[0] = np.inf
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError, match="infinite"):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_write_rejects_malformed_time_coordinate(tmp_path):
    ds = _valid_dataset()
    n = ds.sizes["time"]
    ds = ds.assign_coords(time=np.array(["not-a-time"] * n, dtype=object))
    path = tmp_path / "basin.nc"
    with pytest.raises(PackageNetCDFError):
        write_basin_dataset_netcdf(ds, path)
    assert not path.exists()


def test_post_write_contract_failure_prevents_promotion_and_cleans_up_temp(tmp_path, monkeypatch):
    import src.baseline.package_netcdf as package_netcdf

    ds = _valid_dataset()
    path = tmp_path / "basin.nc"

    original_validate = package_netcdf._validate_basin_dataset_contract
    calls = {"n": 0}

    def _fail_on_second_call(dataset):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise PackageNetCDFError("simulated post-write contract failure")
        original_validate(dataset)

    monkeypatch.setattr(package_netcdf, "_validate_basin_dataset_contract", _fail_on_second_call)
    with pytest.raises(PackageNetCDFError, match="simulated post-write contract failure"):
        write_basin_dataset_netcdf(ds, path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_post_write_contract_failure_leaves_existing_destination_unchanged(tmp_path, monkeypatch):
    import src.baseline.package_netcdf as package_netcdf

    ds = _valid_dataset()
    path = tmp_path / "basin.nc"
    write_basin_dataset_netcdf(ds, path)
    original_bytes = path.read_bytes()

    original_validate = package_netcdf._validate_basin_dataset_contract
    calls = {"n": 0}

    def _fail_on_second_call(dataset):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise PackageNetCDFError("simulated post-write contract failure")
        original_validate(dataset)

    monkeypatch.setattr(package_netcdf, "_validate_basin_dataset_contract", _fail_on_second_call)
    with pytest.raises(PackageNetCDFError, match="simulated post-write contract failure"):
        write_basin_dataset_netcdf(ds, path, overwrite=True)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob("*.tmp")) == []
