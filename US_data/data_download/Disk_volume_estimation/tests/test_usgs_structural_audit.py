"""Tests for the --structural-only mode of audit_usgs_iv_recovered_targets.py
(docs/wy2026_data_plan.md §3 C3).

Network-free: synthetic canonical NCs are written to tmp_path with xarray.
The structural mode must report availability and file structure only, never a
discharge statistic, target classification or plot.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


aud = _load("audit_usgs_iv_recovered_targets")

START, END = "2025-10-01T00:00:00Z", "2026-09-30T23:00:00Z"
IDX = pd.date_range("2025-10-01T00:00:00", "2026-09-30T23:00:00", freq="h")


def _write_nc(path, values, *, attrs_override=None, index=IDX):
    n_valid = int(np.isfinite(values).sum())
    attrs = {
        "units": "m3 s-1",
        "period_start_utc": "2025-10-01T00:00:00Z",
        "period_end_utc": "2026-09-30T23:00:00Z",
        "n_hours_total": str(len(values)),
        "n_valid_hours": str(n_valid),
        "n_nan_hours": str(len(values) - n_valid),
        "snap_n_exact": "8000", "snap_n_nearest": "700", "snap_n_missing": "60",
        "median_cadence_minutes": "15.0",
        "systematic_offset_flag": "False",
        "n_provisional_raw_obs": "1234",
        "n_ice_raw_obs": "56",
        "n_estimated_raw_obs": "7",
        "wy_chunk_mode": "derived",
    }
    attrs.update(attrs_override or {})
    attrs = {k: v for k, v in attrs.items() if v is not None}   # None drops a key
    ds = xr.Dataset({"streamflow": (("time",), values.astype("float32"), attrs)},
                    coords={"time": index.values})
    ds.to_netcdf(path)


@pytest.fixture()
def canon(tmp_path):
    d = tmp_path / "canonical_merged"
    d.mkdir()
    full = np.full(len(IDX), 5.0)
    gappy = full.copy()
    gappy[100:150] = np.nan                       # 50 h gap
    gappy[-24 * 20:] = np.nan                     # last 20 days missing -> late gap
    gappy[10] = -1.0                              # one negative value
    _write_nc(d / "01000001_hourly.nc", full)
    _write_nc(d / "01000002_hourly.nc", gappy,
              attrs_override={"n_valid_hours": "1"})   # stale attr -> mismatch
    _write_nc(d / "01000003_hourly.nc", np.full(len(IDX), np.nan),
              attrs_override={"n_provisional_raw_obs": "0"})
    return d


def _run(canon_dir, out_dir, extra=()):
    with pytest.raises(SystemExit) as e:
        aud.main(["--canonical-dir", str(canon_dir), "--out-dir", str(out_dir),
                  "--structural-only", "--expected-start", START, "--expected-end", END,
                  *extra])
    return e.value.code


def test_structural_writes_only_allowed_files(canon, tmp_path):
    out = tmp_path / "audit_root"
    assert _run(canon, out) == 0
    assert sorted(p.name for p in out.iterdir()) == ["audit_structural"]
    written = sorted(p.name for p in (out / "audit_structural").rglob("*"))
    assert written == sorted(aud.STRUCTURAL_OUTPUT_FILES)
    assert not any(p.suffix == ".png" for p in out.rglob("*"))
    for forbidden in ("audit", "qc", "provenance", "summary.md"):
        assert not (out / forbidden).exists()


def test_structural_no_value_statistics(canon, tmp_path):
    out = tmp_path / "o"
    _run(canon, out)
    s = out / "audit_structural"
    banned = ("min", "max", "mean", "median_q", "p50", "p95", "p99", "std", "quantile",
              "peak", "target_status")
    for name in ("per_basin_structural.csv", "per_water_year_coverage.csv",
                 "per_month_coverage_all_basins.csv", "gap_audit.csv"):
        cols = pd.read_csv(s / name, nrows=0).columns
        for c in cols:
            if c == "median_cadence_min":   # sampling cadence, not a discharge value
                continue
            assert not any(b in c.lower() for b in banned), (name, c)
    summary = json.loads((s / "structural_summary.json").read_text())
    for k in summary:
        assert not any(b in k.lower() for b in banned), k


# Exact output schema: any new column or key (for example a summed volume, or the
# time of the peak) must fail here and be reviewed against plan §8(b).
EXPECTED_COLUMNS = {
    "per_basin_structural.csv": [
        "STAID", "nc_file", "n_hours_total", "n_hours_expected", "grid_ok", "n_valid",
        "n_nan", "coverage_fraction", "has_any_valid", "first_valid_utc",
        "last_valid_utc", "n_exact_snaps", "n_nearest_snaps", "n_missing_snaps",
        "median_cadence_min", "systematic_offset_flag", "n_negative_values", "units_ok",
        "period_attrs_match", "attr_counts_match", "wy_chunk_mode",
        "n_provisional_raw_obs", "n_ice_or_equipment_raw_obs", "n_estimated_raw_obs"],
    "per_water_year_coverage.csv": [
        "STAID", "water_year", "start_utc", "end_utc", "n_hours_total", "n_valid",
        "n_nan", "coverage_fraction"],
    "per_month_coverage_all_basins.csv": [
        "year", "month", "n_basins", "total_basin_hours", "total_valid_hours",
        "total_nan_hours", "n_basins_full_coverage", "n_basins_zero_coverage",
        "coverage_fraction"],
    "gap_audit.csv": [
        "STAID", "n_gaps", "longest_gap_hours", "total_nan_hours", "n_large_gaps_gt24h",
        "last_valid_utc", "gap_to_period_end_days", "late_gap_flag",
        "total_large_gap_hours_gt24h"],
}
EXPECTED_SUMMARY_KEYS = [
    "period_start_utc", "period_end_utc", "n_hours_per_basin_expected",
    "n_nc_files_audited", "n_load_errors", "n_grid_ok", "n_period_attrs_match",
    "n_attr_counts_match", "n_units_ok", "n_systematic_offset_flag",
    "n_zero_valid_hours", "n_full_coverage", "n_coverage_ge_90pct",
    "n_coverage_lt_50pct", "n_with_negative_values", "n_late_gap_flag",
    "late_gap_threshold_days", "total_valid_hours", "total_hours",
    "overall_coverage_fraction", "qualifier_counts_raw_obs", "qualifier_note",
    "load_errors"]
EXPECTED_QUALIFIER_KEYS = ["n_provisional_raw_obs", "n_ice_or_equipment_raw_obs",
                           "n_estimated_raw_obs"]
EXPECTED_PROVENANCE_KEYS = [
    "script", "mode", "git_commit", "generated_utc", "python_version",
    "pandas_version", "numpy_version", "xarray_version", "canonical_dir",
    "period_start", "period_end", "n_hours_target", "n_basins", "outputs"]
_MD_SKIPPED = ("period_start_utc", "period_end_utc", "qualifier_counts_raw_obs",
               "qualifier_note", "load_errors")


def test_structural_exact_output_schema(canon, tmp_path):
    (canon / "01000004_hourly.nc").write_bytes(b"not a netcdf")   # exercise load_errors
    assert _run(canon, tmp_path / "o") == 1
    s = tmp_path / "o" / "audit_structural"
    for name, cols in EXPECTED_COLUMNS.items():
        assert list(pd.read_csv(s / name, nrows=0).columns) == cols, name
    summ = json.loads((s / "structural_summary.json").read_text())
    assert list(summ) == EXPECTED_SUMMARY_KEYS
    q = summ["qualifier_counts_raw_obs"]
    assert list(q) == EXPECTED_QUALIFIER_KEYS
    for v in q.values():
        assert list(v) == ["sum", "n_basins_with_attr", "n_basins_gt0"]
    assert len(summ["load_errors"]) == 1
    err = summ["load_errors"][0]
    assert list(err) == ["nc_file", "error"] and err["nc_file"] == "01000004_hourly.nc"
    assert re.fullmatch(r"[A-Za-z_]\w*", err["error"])          # an exception type name
    prov = json.loads((s / "run_provenance.json").read_text())
    assert list(prov) == EXPECTED_PROVENANCE_KEYS

    md = (s / "structural_summary.md").read_text(encoding="utf-8")
    cells = [ln.split("|")[1].strip() for ln in md.splitlines() if ln.startswith("| ")]
    expected_cells = (["Metric", "Period"]
                      + [k for k in EXPECTED_SUMMARY_KEYS if k not in _MD_SKIPPED]
                      + ["Qualifier"] + EXPECTED_QUALIFIER_KEYS)
    assert cells == expected_cells
    bullets = [ln for ln in md.splitlines() if ln.startswith("- ")]
    assert bullets == [f"- 01000004_hourly.nc: {err['error']}"]


def test_structural_rows_and_counts(canon, tmp_path):
    out = tmp_path / "o"
    _run(canon, out)
    s = out / "audit_structural"
    b = pd.read_csv(s / "per_basin_structural.csv", dtype={"STAID": str}).set_index("STAID")
    assert list(b.index) == ["01000001", "01000002", "01000003"]
    assert b["grid_ok"].all() and b["period_attrs_match"].all() and b["units_ok"].all()
    assert b.loc["01000001", "n_valid"] == 8760 and b.loc["01000001", "attr_counts_match"]
    assert b.loc["01000002", "n_nan"] == 50 + 480
    assert not b.loc["01000002", "attr_counts_match"]
    assert b.loc["01000002", "n_negative_values"] == 1
    assert not b.loc["01000003", "has_any_valid"]
    assert b.loc["01000001", "n_provisional_raw_obs"] == 1234
    assert b.loc["01000001", "n_ice_or_equipment_raw_obs"] == 56
    assert b.loc["01000001", "n_estimated_raw_obs"] == 7

    wy = pd.read_csv(s / "per_water_year_coverage.csv")
    assert set(wy["water_year"]) == {"WY2026"} and (wy["n_hours_total"] == 8760).all()
    # Monthly coverage is aggregated across basins: one row per month, no STAID.
    months = pd.read_csv(s / "per_month_coverage_all_basins.csv")
    assert "STAID" not in months.columns
    assert len(months) == 12
    assert (months["n_basins"] == 3).all()
    oct25 = months[(months["year"] == 2025) & (months["month"] == 10)].iloc[0]
    assert oct25["total_basin_hours"] == 3 * 744
    assert oct25["total_nan_hours"] == 744 + 50
    assert oct25["n_basins_full_coverage"] == 1 and oct25["n_basins_zero_coverage"] == 1
    sep26 = months[(months["year"] == 2026) & (months["month"] == 9)].iloc[0]
    assert sep26["total_nan_hours"] == 720 + 480
    assert months["total_nan_hours"].sum() == 8760 + 530

    # Gap table: counts and hours only, no gap start times or free-text notes.
    gaps = pd.read_csv(s / "gap_audit.csv", dtype={"STAID": str}).set_index("STAID")
    assert "late_gap_flag" in gaps.columns and "late_2025_gap_flag" not in gaps.columns
    for dropped in ("large_gap_starts_json", "late_gap_note", "late_2025_gap_note"):
        assert dropped not in gaps.columns
    assert gaps.loc["01000002", "late_gap_flag"] and not gaps.loc["01000001", "late_gap_flag"]
    assert gaps.loc["01000002", "n_large_gaps_gt24h"] == 2
    assert gaps.loc["01000002", "total_large_gap_hours_gt24h"] == 50 + 480
    assert gaps.loc["01000003", "total_large_gap_hours_gt24h"] == 8760
    assert gaps.loc["01000001", "n_gaps"] == 0
    assert gaps.loc["01000001", "total_large_gap_hours_gt24h"] == 0

    summ = json.loads((s / "structural_summary.json").read_text())
    assert summ["period_start_utc"] == "2025-10-01T00:00:00Z"
    assert summ["period_end_utc"] == "2026-09-30T23:00:00Z"
    assert summ["n_hours_per_basin_expected"] == 8760
    assert summ["n_nc_files_audited"] == 3
    assert summ["n_zero_valid_hours"] == 1
    assert summ["n_full_coverage"] == 1
    assert summ["n_attr_counts_match"] == 2
    assert summ["n_with_negative_values"] == 1
    assert summ["n_late_gap_flag"] == 1
    q = summ["qualifier_counts_raw_obs"]
    assert q["n_provisional_raw_obs"] == {"sum": 2468, "n_basins_with_attr": 3,
                                          "n_basins_gt0": 2}
    md = (s / "structural_summary.md").read_text(encoding="utf-8")
    assert "2025-10-01T00:00:00Z to 2026-09-30T23:00:00Z" in md
    prov = json.loads((s / "run_provenance.json").read_text())
    assert prov["mode"] == "structural-only"


def test_structural_grid_and_period_attr_mismatch(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    short = IDX[:-1]
    _write_nc(d / "01000009_hourly.nc", np.full(len(short), 1.0), index=short,
              attrs_override={"period_end_utc": "2026-09-30T22:00:00Z"})
    assert _run(d, tmp_path / "o") == 1          # grid/period mismatch fails the run
    b = pd.read_csv(tmp_path / "o" / "audit_structural" / "per_basin_structural.csv")
    assert not b.loc[0, "grid_ok"] and not b.loc[0, "period_attrs_match"]


def test_structural_period_attr_mismatch_alone_fails(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    _write_nc(d / "01000009_hourly.nc", np.full(len(IDX), 1.0),
              attrs_override={"period_start_utc": "2020-10-14T00:00:00Z"})
    assert _run(d, tmp_path / "o") == 1


@pytest.mark.parametrize("start", ["garbage", "2025-10-01T00:30:00Z"])
def test_structural_bad_dates_exit_2(canon, tmp_path, start):
    with pytest.raises(SystemExit) as e:
        aud.main(["--canonical-dir", str(canon), "--out-dir", str(tmp_path / "o"),
                  "--structural-only", "--expected-start", start, "--expected-end", END])
    assert e.value.code == 2
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("staids", ["01000001", "01000001,01000003", "09999999"])
def test_structural_refuses_staids(canon, tmp_path, staids):
    # A basin subset would turn the all-basin monthly table into per-basin rows.
    assert _run(canon, tmp_path / "o", extra=["--staids", staids]) == 2
    assert not (tmp_path / "o").exists()


def test_structural_missing_canonical_dir_exit_2(tmp_path):
    assert _run(tmp_path / "does_not_exist", tmp_path / "o") == 2
    assert not (tmp_path / "o").exists()


def test_structural_requires_period(canon, tmp_path):
    with pytest.raises(SystemExit) as e:
        aud.main(["--canonical-dir", str(canon), "--out-dir", str(tmp_path / "o"),
                  "--structural-only"])
    assert e.value.code == 2
    assert not (tmp_path / "o").exists()


def test_structural_load_error_exits_1(canon, tmp_path):
    (canon / "01000004_hourly.nc").write_bytes(b"not a netcdf")
    assert _run(canon, tmp_path / "o") == 1
    summ = json.loads((tmp_path / "o" / "audit_structural" / "structural_summary.json")
                      .read_text())
    assert summ["n_load_errors"] == 1 and summ["n_nc_files_audited"] == 3


def test_structural_outputs_and_stdout_carry_no_discharge_value(tmp_path, capsys):
    # Distinctive values in [50000, 60000]; no count in this fixture reaches 50000.
    d = tmp_path / "c"
    d.mkdir()
    rng = np.random.default_rng(1)
    for k in range(3):
        v = (50_000 + 10_000 * rng.random(len(IDX))).astype("float32")
        v[200 + k * 10: 260 + k * 10] = np.nan
        v[-24 * 15:] = np.nan                       # late gap
        v[300] = -54_321.25                          # a negative value
        _write_nc(d / f"0100010{k}_hourly.nc", v)
    capsys.readouterr()
    assert _run(d, tmp_path / "o") == 0
    stdout = capsys.readouterr().out
    num = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
    texts = {"stdout": stdout}
    for f in (tmp_path / "o").rglob("*"):
        if f.is_file():
            texts[f.name] = f.read_text(encoding="utf-8")
    for name, text in texts.items():
        hits = [m for m in num.findall(text) if 50_000 <= float(m) <= 60_001]
        assert not hits, (name, hits[:5])


# ---------------------------------------------------------------------------
# Legacy (value-statistics) mode refuses protected-period data
# ---------------------------------------------------------------------------

def _run_legacy(canon_dir, out_dir, extra=()):
    with pytest.raises(SystemExit) as e:
        aud.main(["--canonical-dir", str(canon_dir), "--out-dir", str(out_dir),
                  "--skip-jan2023-comparison", *extra])
    return e.value.code


def test_legacy_mode_refuses_wy2026_ncs(canon, tmp_path):
    out = tmp_path / "o"
    assert _run_legacy(canon, out) == 2
    assert not out.exists()


def test_legacy_mode_refuses_post_v001_period_without_marker(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    _write_nc(d / "01000001_hourly.nc", np.full(len(IDX), 5.0),
              attrs_override={"wy_chunk_mode": None})
    assert _run_legacy(d, tmp_path / "o") == 2
    assert not (tmp_path / "o").exists()


def test_legacy_mode_refuses_late_expected_end(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    idx = pd.date_range("2025-12-01", "2025-12-31 23:00", freq="h")
    _write_nc(d / "01000001_hourly.nc", np.full(len(idx), 5.0), index=idx,
              attrs_override={"period_start_utc": "2025-12-01T00:00:00Z",
                              "period_end_utc": "2025-12-31T23:00:00Z"})
    assert _run_legacy(d, tmp_path / "o",
                       extra=["--expected-start", "2025-12-01T00:00:00Z",
                              "--expected-end", "2026-01-31T23:00:00Z"]) == 2
    assert not (tmp_path / "o").exists()


V001_TAIL = pd.date_range("2025-12-01", "2025-12-31 23:00", freq="h")
_V001_ATTRS = {"period_start_utc": "2025-12-01T00:00:00Z",
               "period_end_utc": "2025-12-31T23:00:00Z", "wy_chunk_mode": None}


# Each fixture trips exactly one guard layer, so removing any single layer fails.
@pytest.mark.parametrize("index, override, reason", [
    # time axis only: v001 attributes, no marker, axis runs into 2026
    (pd.date_range("2025-12-01", "2026-01-01 05:00", freq="h"), {}, "time axis"),
    # marker only: v001 axis and attributes, plus wy_chunk_mode
    (V001_TAIL, {"wy_chunk_mode": "derived"}, "wy_chunk_mode"),
    # period_end_utc only: v001 axis, no marker, late attribute
    (V001_TAIL, {"period_end_utc": "2026-09-30T23:00:00Z"}, "period_end_utc"),
    # unparseable period_end_utc
    (V001_TAIL, {"period_end_utc": "not a time"}, "not a valid time"),
], ids=["axis", "marker", "period_end_attr", "bad_period_end_attr"])
def test_legacy_guard_each_layer_alone(tmp_path, index, override, reason):
    d = tmp_path / "c"
    d.mkdir()
    _write_nc(d / "01000001_hourly.nc", np.full(len(index), 5.0), index=index,
              attrs_override={**_V001_ATTRS, **override})
    got = aud.legacy_protected_period_reason([d / "01000001_hourly.nc"])
    assert got is not None and reason in got, got
    assert _run_legacy(d, tmp_path / "o") == 2
    assert not (tmp_path / "o").exists()


def test_legacy_guard_refuses_unreadable_nc(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    _write_nc(d / "01000001_hourly.nc", np.full(len(V001_TAIL), 5.0), index=V001_TAIL,
              attrs_override=_V001_ATTRS)
    (d / "01000002_hourly.nc").write_bytes(b"not a netcdf")
    got = aud.legacy_protected_period_reason(sorted(d.glob("*_hourly.nc")))
    assert got is not None and "unreadable" in got, got
    assert _run_legacy(d, tmp_path / "o") == 2
    assert not (tmp_path / "o").exists()


def test_legacy_guard_allows_v001_period_file(tmp_path):
    idx = pd.date_range("2025-12-01", "2025-12-31 23:00", freq="h")
    p = tmp_path / "01000001_hourly.nc"
    ds = xr.Dataset({"streamflow": (("time",), np.full(len(idx), 5.0, dtype="float32"),
                                    {"period_start_utc": "2025-12-01T00:00:00Z",
                                     "period_end_utc": "2025-12-31T23:00:00Z"})},
                    coords={"time": idx.values})
    ds.to_netcdf(p)
    assert aud.legacy_protected_period_reason([p]) is None
    assert aud.legacy_protected_period_reason([p], "2025-12-31T23:00:00Z") is None
    assert "after" in aud.legacy_protected_period_reason([p], "2026-01-01T00:00:00Z")


def test_derive_wy_boundaries_matches_legacy_table():
    assert aud.derive_wy_boundaries(aud.PERIOD_START, aud.PERIOD_END) == aud.WY_BOUNDARIES


def test_derive_wy_boundaries_wy2026():
    b = aud.derive_wy_boundaries(pd.Timestamp("2025-10-01"),
                                 pd.Timestamp("2026-09-30 23:00:00"))
    assert b == [("WY2026", pd.Timestamp("2025-10-01"), pd.Timestamp("2026-09-30 23:00:00"))]


def _legacy_summary(out_dir, **kw):
    out_dir.mkdir()
    aud.write_summary(out_dir, [], [], [], [], [], [], "abc", "now", 45_720, **kw)
    return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in sorted(out_dir.rglob("*.md")))


def test_legacy_write_summary_period_string(tmp_path):
    default = _legacy_summary(tmp_path / "d")
    assert "2020-10-14T00:00:00Z to 2025-12-31T23:00:00Z" in default
    v001 = _legacy_summary(tmp_path / "v", period_start=aud.PERIOD_START,
                           period_end=aud.PERIOD_END)
    assert "2020-10-14T00:00:00Z to 2025-12-31T23:00:00Z" in v001
    other = _legacy_summary(tmp_path / "o", period_start=pd.Timestamp("2023-01-01"),
                            period_end=pd.Timestamp("2023-01-31 23:00"))
    assert "2023-01-01T00:00:00Z to 2023-01-31T23:00:00Z" in other
    assert "2020-10-14T00:00:00Z" not in other
