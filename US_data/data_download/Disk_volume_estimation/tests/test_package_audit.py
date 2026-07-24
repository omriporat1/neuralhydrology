"""Tests for the Gate 4 independent Compact Scientific Package auditor.

Synthetic fixtures only -- no h2o/Moriah access, no real 32-basin audit.

Test packages are built with the *real* builder
(``src.baseline.package_builder.build_compact_scientific_package``) and its
real local-file loader (``default_local_basin_source_loader``), so the
correct-package baseline is genuine production output, not a hand-rolled
stand-in. The auditor under test (``src.baseline.package_audit``) never
imports the builder; each failure-mode test corrupts one artifact after the
build (a NetCDF value, a CSV cell, a selection file, a checksum) and asserts
the auditor's *independent* recomputation catches it.
"""
from __future__ import annotations

import json
import subprocess

import netCDF4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline.package_builder import build_compact_scientific_package, default_local_basin_source_loader
from src.baseline.package_netcdf import DEFAULT_PACKAGE_NETCDF_SCHEMA, SCIENTIFIC_V002_SCHEMA
from src.baseline import package_audit

BASIN_IDS = ("01019000", "02146000", "03303000")
STATIC_COLUMNS = ["attr_a", "attr_b"]
N_HOURS = 30
AREA_KM2 = {b: 100.0 + 10.0 * i for i, b in enumerate(BASIN_IDS)}


def _hourly_index(n=N_HOURS, start="2023-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def _valid_forcing(idx):
    n = len(idx)
    return pd.DataFrame(
        {
            "mrms_qpe_1h_mm": np.linspace(0.0, 5.0, n),
            "rtma_2t_K": np.linspace(270.0, 285.0, n),
            "rtma_2d_K": np.linspace(260.0, 275.0, n),
            "rtma_2sh_kgkg": np.linspace(0.001, 0.004, n),
            "rtma_10u_ms": np.linspace(-3.0, 3.0, n),
            "rtma_10v_ms": np.linspace(-2.0, 2.0, n),
            "mrms_qpe_1h_mm_gap": np.zeros(n, dtype=np.float32),
            "rtma_gap": np.zeros(n, dtype=np.float32),
        },
        index=idx,
    )


def _qobs_values(idx, basin_index=0):
    n = len(idx)
    return (np.arange(1, n + 1, dtype=np.float64) * 2.0) + basin_index


def _write_source_tree(tmp_path, idx):
    forcing_root = tmp_path / "forcing"
    qobs_root = tmp_path / "qobs"
    for i, basin_id in enumerate(BASIN_IDS):
        forcing_dir = forcing_root / "time_series"
        forcing_dir.mkdir(parents=True, exist_ok=True)
        _valid_forcing(idx).to_parquet(forcing_dir / f"{basin_id}.parquet")

        qobs_dir = qobs_root / "time_series"
        qobs_dir.mkdir(parents=True, exist_ok=True)
        values = _qobs_values(idx, basin_index=i)
        ds = xr.Dataset({"qobs_m3s": ("time", values)}, coords={"time": idx})
        ds.to_netcdf(qobs_dir / f"{basin_id}.nc")

    area_csv = tmp_path / "area.csv"
    pd.DataFrame({"gauge_id": list(BASIN_IDS), "DRAIN_SQKM": [AREA_KM2[b] for b in BASIN_IDS]}).to_csv(
        area_csv, index=False
    )
    return forcing_root, qobs_root, area_csv


def _static_column_manifest_dict(columns=STATIC_COLUMNS):
    return {"columns": {c: {"role": "model_input"} for c in columns}}


def _clean_git_repo(tmp_path):
    """An isolated, freshly-initialized, clean git repo for auditor_repo_root.

    Full-mode audits refuse to run against a dirty/unresolvable auditor
    working tree (correction 10). Tests must never depend on the real
    (currently in-progress) development repository's cleanliness, so each
    test gets its own throwaway repo instead.
    """
    repo_dir = tmp_path / "auditor_repo_fixture"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "placeholder.txt").write_text("placeholder\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=repo_dir, check=True)
    return repo_dir


def _default_imputation_artifacts(tmp_path):
    """Zero-imputation default evidence: valid, exact-order, all-False mask."""
    imputation_manifest_path = tmp_path / "imputation_manifest.json"
    imputation_manifest_path.write_text(
        json.dumps(
            {"per_column": {c: {"fitted_value": None, "n_missing_before_apply": 0} for c in STATIC_COLUMNS}}
        ),
        encoding="utf-8",
    )
    mask_df = pd.DataFrame(False, index=list(BASIN_IDS), columns=STATIC_COLUMNS)
    imputed_value_mask_path = tmp_path / "imputed_value_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)
    return imputation_manifest_path, imputed_value_mask_path


def _build_package(
    tmp_path,
    *,
    idx=None,
    gap_timestamps=None,
    static_values=None,
    write_qc_csv=False,
    package_netcdf_schema=None,
):
    idx = idx if idx is not None else _hourly_index()
    forcing_root, qobs_root, area_csv = _write_source_tree(tmp_path, idx)
    area_by_basin = {b: AREA_KM2[b] for b in BASIN_IDS}
    loader = default_local_basin_source_loader(forcing_root, qobs_root, area_by_basin, dynamic_inputs=package_audit.DYNAMIC_INPUTS)

    if static_values is None:
        static_values = np.arange(len(BASIN_IDS) * len(STATIC_COLUMNS), dtype=np.float64).reshape(
            len(BASIN_IDS), len(STATIC_COLUMNS)
        )
    static_attributes = pd.DataFrame(static_values, index=list(BASIN_IDS), columns=STATIC_COLUMNS)

    static_column_manifest_path = tmp_path / "static_column_manifest.json"
    static_column_manifest_path.write_text(json.dumps(_static_column_manifest_dict()), encoding="utf-8")

    prepared_static_parquet_path = tmp_path / "prepared_static.parquet"
    static_attributes.to_parquet(prepared_static_parquet_path)

    basin_selection_path = tmp_path / "basin_ids.txt"
    basin_selection_path.write_text("\n".join(BASIN_IDS) + "\n", encoding="utf-8")

    gap_inventory_rows = []
    for ts in gap_timestamps or []:
        gap_inventory_rows.append(
            {
                "chunk_label": "test",
                "product": package_audit.MRMS_PRODUCT,
                "valid_time_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reason": "synthetic",
            }
        )
    gap_inventory_csv_path = tmp_path / "gap_inventory.csv"
    pd.DataFrame(gap_inventory_rows, columns=["chunk_label", "product", "valid_time_utc", "reason"]).to_csv(
        gap_inventory_csv_path, index=False
    )

    evidence_root = tmp_path / "evidence" if write_qc_csv else None
    build_kwargs = dict(
        basin_ids=BASIN_IDS,
        load_basin_source=loader,
        static_attributes=static_attributes,
        static_model_input_columns=STATIC_COLUMNS,
        gap_timestamps=gap_timestamps or [],
        expected_index=idx,
        output_package_root=tmp_path / "package",
        evidence_root=evidence_root,
        write_qc_csv=write_qc_csv,
    )
    if package_netcdf_schema is not None:
        build_kwargs["package_netcdf_schema"] = package_netcdf_schema
    result = build_compact_scientific_package(**build_kwargs)

    policy = {
        "period": {
            "start_utc": idx[0].isoformat(),
            "end_utc": idx[-1].isoformat(),
            "expected_hours": len(idx),
        },
        "gap_policy": {"include_rtma_in_history_mask": True},
        "audit": {"package_float32_rtol": 1e-5},
    }

    imputation_manifest_path, imputed_value_mask_path = _default_imputation_artifacts(tmp_path)
    auditor_repo_root = _clean_git_repo(tmp_path)

    return {
        "result": result,
        "policy": policy,
        "package_root": result.package_root,
        "forcing_root": forcing_root,
        "qobs_root": qobs_root,
        "area_csv_path": area_csv,
        "basin_selection_path": basin_selection_path,
        "prepared_static_parquet_path": prepared_static_parquet_path,
        "static_column_manifest_path": static_column_manifest_path,
        "gap_inventory_csv_path": gap_inventory_csv_path,
        "evidence_root": evidence_root,
        "imputation_manifest_path": imputation_manifest_path,
        "imputed_value_mask_path": imputed_value_mask_path,
        "auditor_repo_root": auditor_repo_root,
    }


def _run(fixture, **overrides):
    # dev_allow_missing_evidence=True is a test-ergonomics default only: most
    # tests here are unrelated to QC evidence and the fixture's QC evidence
    # root is None unless the test built the package with write_qc_csv=True.
    # The production CLI never sets this flag, so canonical full audits stay
    # strict; the dedicated correction-1 tests below override it back to
    # False to exercise that strict/canonical path.
    kwargs = dict(
        package_root=fixture["package_root"],
        policy=fixture["policy"],
        policy_path=fixture["static_column_manifest_path"],  # any real file works as a stand-in policy artifact
        basin_selection_path=fixture["basin_selection_path"],
        prepared_static_parquet_path=fixture["prepared_static_parquet_path"],
        static_column_manifest_path=fixture["static_column_manifest_path"],
        forcing_root=fixture["forcing_root"],
        qobs_root=fixture["qobs_root"],
        area_csv_path=fixture["area_csv_path"],
        gap_inventory_csv_path=fixture["gap_inventory_csv_path"],
        imputation_manifest_path=fixture["imputation_manifest_path"],
        imputed_value_mask_path=fixture["imputed_value_mask_path"],
        qc_evidence_root=fixture["evidence_root"],
        build_git_commit="test-commit",
        auditor_repo_root=fixture["auditor_repo_root"],
        dev_allow_missing_evidence=True,
    )
    kwargs.update(overrides)
    return package_audit.run_audit(**kwargs)


def _set_netcdf_variable(nc_path, var_name, values):
    with netCDF4.Dataset(nc_path, "r+") as ds:
        ds.set_auto_maskandscale(False)
        ds.variables[var_name][:] = values


def _nc_path(fixture, basin_id):
    return fixture["package_root"] / "time_series" / f"{basin_id}.nc"


# ---------------------------------------------------------------------------
# 1. correct package passes
# ---------------------------------------------------------------------------


def test_full_audit_passes_for_correct_package(tmp_path):
    fixture = _build_package(tmp_path)
    report, diagnostics = _run(fixture)
    assert report.status == "PASS", report.failed_messages()
    assert report.error_count == 0
    assert diagnostics["audit_manifest"]["build_git_commit"] == "test-commit"


# ---------------------------------------------------------------------------
# 2. one forcing value altered
# ---------------------------------------------------------------------------


def test_fails_when_forcing_value_altered(tmp_path):
    fixture = _build_package(tmp_path)
    with netCDF4.Dataset(_nc_path(fixture, BASIN_IDS[0]), "r+") as ds:
        ds.set_auto_maskandscale(False)
        values = np.array(ds.variables["mrms_qpe_1h_mm"][:], dtype=np.float32)
        values[3] = values[3] + 999.0
        ds.variables["mrms_qpe_1h_mm"][:] = values

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"dynamic_input_matches_forcing_source[{BASIN_IDS[0]}][mrms_qpe_1h_mm]" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 3. incorrect m3/s -> mm/h conversion
# ---------------------------------------------------------------------------


def test_fails_for_incorrect_conversion(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.set_auto_maskandscale(False)
        raw_qobs = np.array(ds.variables["qobs_m3s"][:], dtype=np.float64)
        wrong_mm_per_h = raw_qobs * 3.0 / AREA_KM2[basin_id]  # wrong constant (3.0 instead of 3.6)
        lead_name = "qobs_mm_per_h_lead01"
        expected_shape = np.array(ds.variables[lead_name][:]).shape
        wrong_target = np.full(expected_shape, np.nan, dtype=np.float32)
        n = wrong_mm_per_h.shape[0]
        wrong_target[: n - 1] = wrong_mm_per_h[1:].astype(np.float32)
        ds.variables[lead_name][:] = wrong_target

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"lead_target_matches_independent_recomputation[{basin_id}][qobs_mm_per_h_lead01]"
        and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 4. lead shifted the wrong direction
# ---------------------------------------------------------------------------


def test_fails_when_lead_shifted_wrong_direction(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    lead_name = "qobs_mm_per_h_lead01"
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.set_auto_maskandscale(False)
        raw_qobs = np.array(ds.variables["qobs_m3s"][:], dtype=np.float64)
        mm_per_h = raw_qobs * 3.6 / AREA_KM2[basin_id]
        n = mm_per_h.shape[0]
        backward_shifted = np.full(n, np.nan, dtype=np.float32)
        backward_shifted[1:] = mm_per_h[: n - 1].astype(np.float32)  # shifted +1 instead of -1
        ds.variables[lead_name][:] = backward_shifted

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"lead_target_matches_independent_recomputation[{basin_id}][{lead_name}]" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 5. incorrect target-tail NaNs
# ---------------------------------------------------------------------------


def test_fails_for_incorrect_target_tail_nans(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    lead_name = "qobs_mm_per_h_lead01"
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.set_auto_maskandscale(False)
        values = np.array(ds.variables[lead_name][:], dtype=np.float32)
        values[-1] = 12.5  # should be NaN (lead=1 tail)
        ds.variables[lead_name][:] = values

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"lead_target_tail_nan[{basin_id}][{lead_name}]" and r.severity == "ERROR" for r in report.records
    )


# ---------------------------------------------------------------------------
# 6. raw qobs differs from source
# ---------------------------------------------------------------------------


def test_fails_when_raw_qobs_differs_from_source(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.set_auto_maskandscale(False)
        values = np.array(ds.variables["qobs_m3s"][:], dtype=np.float32)
        values[0] = values[0] + 500.0
        ds.variables["qobs_m3s"][:] = values

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == f"raw_qobs_matches_source[{basin_id}]" and r.severity == "ERROR" for r in report.records)


# ---------------------------------------------------------------------------
# 7. wrong basin membership / order
# ---------------------------------------------------------------------------


def test_fails_for_wrong_basin_membership_order(tmp_path):
    fixture = _build_package(tmp_path)
    reordered = [BASIN_IDS[1], BASIN_IDS[0], BASIN_IDS[2]]
    fixture["basin_selection_path"].write_text("\n".join(reordered) + "\n", encoding="utf-8")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == "basin_order_matches_selection" and r.severity == "ERROR" for r in report.records)


# ---------------------------------------------------------------------------
# 8. wrong static value
# ---------------------------------------------------------------------------


def test_fails_for_wrong_static_value(tmp_path):
    fixture = _build_package(tmp_path)
    attrs_path = fixture["package_root"] / "attributes" / "attributes.csv"
    df = pd.read_csv(attrs_path, dtype={"gauge_id": str}).set_index("gauge_id")
    df.loc[BASIN_IDS[0], "attr_a"] = df.loc[BASIN_IDS[0], "attr_a"] + 777.0
    df.reset_index().to_csv(attrs_path, index=False)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "attributes_csv_values_match_prepared" and r.severity == "ERROR" for r in report.records
    )


# ---------------------------------------------------------------------------
# 9. imputed value placed on the wrong basin/column
# ---------------------------------------------------------------------------


def test_fails_when_imputed_value_placed_wrong(tmp_path):
    fixture = _build_package(tmp_path)

    imputation_manifest_path = tmp_path / "imputation_manifest.json"
    imputation_manifest_path.write_text(
        json.dumps({"per_column": {"attr_a": {"fitted_value": 5.0}}}), encoding="utf-8"
    )

    mask_df = pd.DataFrame(
        False, index=list(BASIN_IDS), columns=STATIC_COLUMNS
    )
    mask_df.loc[BASIN_IDS[0], "attr_a"] = True
    imputed_value_mask_path = tmp_path / "imputed_value_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)

    # attributes.csv for this cell was NOT written as 5.0 (it's whatever the
    # arithmetic fixture produced), so this must be flagged as a mismatch.
    report, _ = _run(
        fixture,
        imputation_manifest_path=imputation_manifest_path,
        imputed_value_mask_path=imputed_value_mask_path,
    )
    assert report.status == "FAIL"
    assert any(r.check_id == "imputed_values_placed_correctly" and r.severity == "ERROR" for r in report.records)


# ---------------------------------------------------------------------------
# 10. RTMA gaps omitted from the reconstructed gap set
# ---------------------------------------------------------------------------


def test_fails_when_rtma_gaps_omitted_from_reconstruction(tmp_path):
    idx = _hourly_index()
    mrms_ts = idx[2]
    rtma_ts = idx[5]
    fixture = _build_package(tmp_path, idx=idx, gap_timestamps=[mrms_ts, rtma_ts])

    # Overwrite the gap inventory CSV with only the MRMS row -- the RTMA gap
    # is present in the package's masks/gap_timestamps.json but missing from
    # the source inventory the auditor reconstructs from.
    pd.DataFrame(
        [
            {
                "chunk_label": "test",
                "product": package_audit.MRMS_PRODUCT,
                "valid_time_utc": mrms_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "reason": "synthetic",
            }
        ]
    ).to_csv(fixture["gap_inventory_csv_path"], index=False)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "gap_timestamps_reconstruction_matches_package" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 11. non-binary gap flag
# ---------------------------------------------------------------------------


def test_fails_for_non_binary_gap_flag(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.set_auto_maskandscale(False)
        values = np.array(ds.variables["mrms_qpe_1h_mm_gap"][:])
        values[0] = 5  # not 0 or 1
        ds.variables["mrms_qpe_1h_mm_gap"][:] = values

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_gap_flag_binary[{basin_id}][mrms_qpe_1h_mm_gap]" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 12. checksum / provenance mismatch
# ---------------------------------------------------------------------------


def test_fails_for_checksum_mismatch(tmp_path):
    fixture = _build_package(tmp_path)
    attrs_path = fixture["package_root"] / "attributes" / "attributes.csv"
    with open(attrs_path, "a", encoding="utf-8") as fh:
        fh.write("\n")  # changes bytes/sha256 without changing parsed values

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "file_checksums_recomputed_match_declared" and r.severity == "ERROR" for r in report.records
    )


# ---------------------------------------------------------------------------
# 13. QC CSV discrepancy beyond tolerance (non-authoritative)
# ---------------------------------------------------------------------------


def test_identifies_qc_csv_discrepancy_without_treating_as_authoritative(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    basin_id = BASIN_IDS[0]
    csv_path = fixture["evidence_root"] / "csv_inspection" / f"{basin_id}.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "mrms_qpe_1h_mm"] = df.loc[0, "mrms_qpe_1h_mm"] + 123.0
    df.to_csv(csv_path, index=False)

    report, _ = _run(fixture)
    qc_check_id = f"qc_csv_matches_netcdf[{basin_id}][mrms_qpe_1h_mm]"
    assert any(r.check_id == qc_check_id and r.severity == "ERROR" for r in report.records)
    # The authoritative NetCDF-vs-source comparison for the same variable must
    # still be unaffected -- the CSV discrepancy is a QC-evidence problem, not
    # an authoritative-package problem.
    assert not any(
        r.check_id == f"dynamic_input_matches_forcing_source[{basin_id}][mrms_qpe_1h_mm]" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 13b. QC CSV vs NetCDF: exact float32-projection comparison, not a
# broader numerical tolerance. The QC CSV holds the builder's
# pre-quantization float64 values; the NetCDF stores the same values
# quantized to float32. Genuine quantization-only differences (which
# exceed the old QC_CSV_ROUNDTRIP_RTOL=1e-9) must PASS; any discrepancy
# that survives projecting the CSV value through float32 must FAIL.
# ---------------------------------------------------------------------------


def test_qc_csv_passes_despite_float32_quantization(tmp_path):
    idx = _hourly_index()
    fixture = _build_package(tmp_path, idx=idx, write_qc_csv=True)
    basin_id = BASIN_IDS[0]

    # Sanity: the fixture's forcing values (np.linspace fractions, not exactly
    # representable in float32) actually exercise real float64->float32
    # quantization well beyond the old CSV round-trip tolerance -- otherwise
    # this test would pass trivially without exercising the fix at all.
    nc_path = _nc_path(fixture, basin_id)
    disk = package_audit.read_package_basin_netcdf_independent(nc_path)
    csv_path = fixture["evidence_root"] / "csv_inspection" / f"{basin_id}.csv"
    csv_values = pd.read_csv(csv_path)["mrms_qpe_1h_mm"].to_numpy(dtype=np.float64)
    nc_values = np.asarray(disk["variables"]["mrms_qpe_1h_mm"], dtype=np.float64)
    nonzero = csv_values != 0.0
    rel_diff = np.abs(nc_values[nonzero] - csv_values[nonzero]) / np.abs(csv_values[nonzero])
    assert np.max(rel_diff) > package_audit.QC_CSV_ROUNDTRIP_RTOL

    report, _ = _run(fixture)
    assert report.status == "PASS", report.failed_messages()


def test_fails_when_qc_csv_value_projects_to_different_float32(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    basin_id = BASIN_IDS[0]
    csv_path = fixture["evidence_root"] / "csv_inspection" / f"{basin_id}.csv"
    df = pd.read_csv(csv_path)
    original_f32 = np.float32(df.loc[1, "mrms_qpe_1h_mm"])
    # compare_qc_csv_against_netcdf_storage now allows exactly 1 float32 ULP
    # (QC_CSV_STORAGE_ULP_TOLERANCE, added 2026-07-24 -- see that module for
    # the confirmed xarray/netcdf4 write-path rounding root cause), so this
    # test bumps by 2 ULPs to keep exercising a genuine, still-caught
    # projected-value difference rather than the now-tolerated 1-ULP case.
    one_up = np.nextafter(original_f32, np.float32(np.inf))
    bumped_f32 = np.nextafter(one_up, np.float32(np.inf))
    df.loc[1, "mrms_qpe_1h_mm"] = float(bumped_f32)
    df.to_csv(csv_path, index=False)

    report, _ = _run(fixture)
    check_id = f"qc_csv_matches_netcdf[{basin_id}][mrms_qpe_1h_mm]"
    assert report.status == "FAIL"
    assert any(r.check_id == check_id and r.severity == "ERROR" for r in report.records)


def test_fails_for_qc_csv_nan_mismatch(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    basin_id = BASIN_IDS[0]
    lead_name = "qobs_mm_per_h_lead01"
    csv_path = fixture["evidence_root"] / "csv_inspection" / f"{basin_id}.csv"
    df = pd.read_csv(csv_path)
    last_row = len(df) - 1
    assert pd.isna(df.loc[last_row, lead_name])  # sanity: last row is the trailing lead-target NaN
    df.loc[last_row, lead_name] = 0.0
    df.to_csv(csv_path, index=False)

    report, _ = _run(fixture)
    check_id = f"qc_csv_matches_netcdf[{basin_id}][{lead_name}]"
    assert report.status == "FAIL"
    assert any(r.check_id == check_id and r.severity == "ERROR" for r in report.records)


def test_fails_when_gap_flag_qc_csv_value_differs(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    basin_id = BASIN_IDS[0]
    gap_name = "mrms_qpe_1h_mm_gap"
    csv_path = fixture["evidence_root"] / "csv_inspection" / f"{basin_id}.csv"
    df = pd.read_csv(csv_path)
    assert float(df.loc[0, gap_name]) == 0.0
    df.loc[0, gap_name] = 1.0  # still a valid binary value, but no longer matches the NetCDF's stored 0
    df.to_csv(csv_path, index=False)

    report, _ = _run(fixture)
    check_id = f"qc_csv_matches_netcdf[{basin_id}][{gap_name}]"
    assert report.status == "FAIL"
    assert any(r.check_id == check_id and r.severity == "ERROR" for r in report.records)


# ---------------------------------------------------------------------------
# write_audit_outputs
# ---------------------------------------------------------------------------


def test_write_audit_outputs_generates_expected_files(tmp_path):
    fixture = _build_package(tmp_path)
    report, diagnostics = _run(fixture)
    out_dir = tmp_path / "audit_out"
    written = package_audit.write_audit_outputs(out_dir, report, diagnostics)

    for name in ("audit_results.json", "audit_report.md", "audit_manifest.json", "file_checksums.csv", "run.log"):
        assert written[name].is_file()
    assert written["review_bundle"].is_dir()

    results = json.loads(written["audit_results.json"].read_text(encoding="utf-8"))
    assert results["status"] == "PASS"

    manifest = json.loads(written["audit_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["build_git_commit"] == "test-commit"
    assert "generated_output_checksums" in manifest


def test_write_audit_outputs_refuses_nonempty_dir_without_overwrite(tmp_path):
    fixture = _build_package(tmp_path)
    report, diagnostics = _run(fixture)
    out_dir = tmp_path / "audit_out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(package_audit.PackageAuditError):
        package_audit.write_audit_outputs(out_dir, report, diagnostics)

    # overwrite=True proceeds
    package_audit.write_audit_outputs(out_dir, report, diagnostics, overwrite=True)


def test_run_preflight_reports_missing_paths(tmp_path):
    fixture = _build_package(tmp_path)
    report, _ = package_audit.run_preflight(
        package_root=fixture["package_root"],
        policy_path=fixture["static_column_manifest_path"],
        basin_selection_path=tmp_path / "does_not_exist.txt",
    )
    assert any(
        r.check_id == "preflight_path_exists[basin_selection_path]" and r.severity == "ERROR" for r in report.records
    )


# ---------------------------------------------------------------------------
# 14. correction 1: canonical full audit cannot skip imputation/QC evidence
# ---------------------------------------------------------------------------


def test_full_mode_requires_imputation_manifest(tmp_path):
    fixture = _build_package(tmp_path)
    with pytest.raises(package_audit.PackageAuditError):
        _run(fixture, imputation_manifest_path=None, dev_allow_missing_evidence=False)


def test_full_mode_requires_imputed_value_mask(tmp_path):
    fixture = _build_package(tmp_path)
    with pytest.raises(package_audit.PackageAuditError):
        _run(fixture, imputed_value_mask_path=None, dev_allow_missing_evidence=False)


def test_full_mode_requires_qc_evidence_root(tmp_path):
    fixture = _build_package(tmp_path)
    with pytest.raises(package_audit.PackageAuditError):
        _run(fixture, qc_evidence_root=None, dev_allow_missing_evidence=False)


def test_dev_allow_missing_evidence_bypasses_mandatory_check(tmp_path):
    fixture = _build_package(tmp_path)
    # explicit dev/test-only bypass: never set by the CLI, only usable when
    # calling run_audit() directly.
    report, _ = _run(
        fixture,
        imputation_manifest_path=None,
        imputed_value_mask_path=None,
        qc_evidence_root=None,
        dev_allow_missing_evidence=True,
    )
    assert report is not None


# ---------------------------------------------------------------------------
# 15. correction 4: exact closed-world package layout
# ---------------------------------------------------------------------------


def test_fails_for_extra_top_level_entry(tmp_path):
    fixture = _build_package(tmp_path)
    (fixture["package_root"] / "unexpected_top_level.txt").write_text("x", encoding="utf-8")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "package_no_unexpected_top_level_entries" and r.severity == "ERROR" for r in report.records
    )


def test_fails_for_extra_time_series_file(tmp_path):
    fixture = _build_package(tmp_path)
    extra_nc = fixture["package_root"] / "time_series" / "99999999.nc"
    extra_nc.write_bytes(_nc_path(fixture, BASIN_IDS[0]).read_bytes())

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == "package_exact_file_set" and r.severity == "ERROR" for r in report.records)


# ---------------------------------------------------------------------------
# 16. correction 5: matched infinity at the same position must never pass
# ---------------------------------------------------------------------------


def test_compare_float_arrays_fails_on_matched_infinity_same_position():
    expected = np.array([1.0, np.inf, 3.0])
    actual = np.array([1.0, np.inf, 3.0])
    result = package_audit.compare_float_arrays("infinity_regression", expected, actual, rtol=1e-5)
    assert result.passed is False


# ---------------------------------------------------------------------------
# 16b. compare_qc_csv_against_netcdf_storage: 1-float32-ULP allowance
#
# Confirmed root cause (Gate 4 audit, 2026-07-24, real 2,557-basin build,
# basins 01643395/10321940): xarray's Dataset.to_netcdf(engine="netcdf4")
# with a float32 _FillValue encoding can write a stored float32 bit pattern
# one ULP away from the in-memory float32 array it was given, for values
# that land within 1 ULP of a float32 rounding boundary. The QC CSV holds
# the pre-quantization value, so this shows up as a QC-CSV-vs-NetCDF
# mismatch even though the package's own authoritative NetCDF value is
# correct. The tolerance below accepts only that specific, understood
# 1-ULP artifact -- anything larger still fails, and NaN-position
# disagreement is unaffected.
# ---------------------------------------------------------------------------


def test_qc_csv_vs_netcdf_exact_match_passes():
    value = np.float32(0.06679082)
    csv_values = np.array([np.float64(value)])
    netcdf_values = np.array([np.float64(value)])
    result = package_audit.compare_qc_csv_against_netcdf_storage(
        "qc_csv_exact", netcdf_values, csv_values, storage_dtype=np.float32
    )
    assert result.passed is True
    assert result.finite_mismatch_count == 0


def test_qc_csv_vs_netcdf_one_ulp_difference_passes():
    csv_value = np.float32(0.06679082)  # 0x3d88c9a0
    netcdf_value = np.nextafter(csv_value, np.float32(1.0))  # adjacent representable float32, 0x3d88c9a1
    assert netcdf_value != csv_value

    result = package_audit.compare_qc_csv_against_netcdf_storage(
        "qc_csv_one_ulp",
        np.array([np.float64(netcdf_value)]),
        np.array([np.float64(csv_value)]),
        storage_dtype=np.float32,
    )
    assert result.passed is True
    assert result.finite_mismatch_count == 0


def test_qc_csv_vs_netcdf_two_ulp_difference_fails():
    csv_value = np.float32(0.06679082)
    one_up = np.nextafter(csv_value, np.float32(1.0))
    two_up = np.nextafter(one_up, np.float32(1.0))
    assert two_up != csv_value and two_up != one_up

    result = package_audit.compare_qc_csv_against_netcdf_storage(
        "qc_csv_two_ulp",
        np.array([np.float64(two_up)]),
        np.array([np.float64(csv_value)]),
        storage_dtype=np.float32,
    )
    assert result.passed is False
    assert result.finite_mismatch_count == 1


def test_qc_csv_vs_netcdf_nan_position_mismatch_still_fails():
    csv_values = np.array([1.0, np.nan])
    netcdf_values = np.array([1.0, 2.0])
    result = package_audit.compare_qc_csv_against_netcdf_storage(
        "qc_csv_nan_mismatch", netcdf_values, csv_values, storage_dtype=np.float32
    )
    assert result.passed is False
    assert result.nan_mismatch_count == 1


def test_qc_csv_vs_netcdf_ulp_tolerance_does_not_apply_to_non_float32_storage():
    # storage_dtype=float64: no quantization step exists to explain a
    # 1-representable-step gap, so exact equality must still be required.
    csv_value = 0.06679082
    netcdf_value = np.nextafter(csv_value, 1.0)
    assert netcdf_value != csv_value

    result = package_audit.compare_qc_csv_against_netcdf_storage(
        "qc_csv_float64_storage",
        np.array([netcdf_value]),
        np.array([csv_value]),
        storage_dtype=np.float64,
    )
    assert result.passed is False
    assert result.finite_mismatch_count == 1


def test_compare_float_arrays_still_exact_for_unrelated_checks():
    # Regression guard: the new ULP tolerance is local to
    # compare_qc_csv_against_netcdf_storage and must not leak into the
    # generic comparator used for authoritative package-vs-source checks.
    csv_value = np.float32(0.06679082)
    one_up = np.nextafter(csv_value, np.float32(1.0))
    result = package_audit.compare_float_arrays(
        "unrelated_generic_check",
        np.array([np.float64(one_up)]),
        np.array([np.float64(csv_value)]),
        rtol=0.0,
        atol=0.0,
    )
    assert result.passed is False


# ---------------------------------------------------------------------------
# 17. correction 6: incorrect schema attribute / variable metadata corruption
# ---------------------------------------------------------------------------


def test_fails_for_incorrect_package_schema_version(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.setncattr("package_schema_version", 999)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == f"netcdf_package_schema[{basin_id}]" and r.severity == "ERROR" for r in report.records)


def test_fails_for_incorrect_gap_flag_attrs(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.variables["mrms_qpe_1h_mm_gap"].setncattr("flag_meanings", "wrong wrong")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_gap_flag_attrs[{basin_id}][mrms_qpe_1h_mm_gap]" and r.severity == "ERROR"
        for r in report.records
    )


def test_fails_for_incorrect_lead_target_attrs(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.variables["qobs_mm_per_h_lead01"].setncattr("lead_hours", 999)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_lead_target_attrs[{basin_id}][qobs_mm_per_h_lead01]" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 18. correction 7: strengthened imputation-evidence validation
# ---------------------------------------------------------------------------


def test_fails_for_wrong_imputed_value_mask_column_order(tmp_path):
    fixture = _build_package(tmp_path)
    mask_df = pd.DataFrame(False, index=list(BASIN_IDS), columns=list(reversed(STATIC_COLUMNS)))
    imputed_value_mask_path = tmp_path / "reordered_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)

    report, _ = _run(fixture, imputed_value_mask_path=imputed_value_mask_path)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "imputed_value_mask_column_order" and r.severity == "ERROR" for r in report.records
    )


# ---------------------------------------------------------------------------
# 18b. imputed_value_mask basin order vs membership split (Gate 4 audit,
# 2026-07-24, real 2,557-basin build): confirmed via a diagnostic run against
# the real h2o data that the mask's basin set was an exact match to the
# accepted selection (0 missing, 0 extra) with only row order differing, and
# all downstream imputation-placement checks re-index by label. Membership
# must stay a hard ERROR; a pure order difference must not fail the audit.
# ---------------------------------------------------------------------------


def test_imputed_value_mask_row_order_differs_but_membership_exact_warns_not_fails(tmp_path):
    fixture = _build_package(tmp_path)
    mask_df = pd.DataFrame(False, index=list(reversed(BASIN_IDS)), columns=STATIC_COLUMNS)
    imputed_value_mask_path = tmp_path / "reordered_basin_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)

    report, _ = _run(fixture, imputed_value_mask_path=imputed_value_mask_path)

    assert any(
        r.check_id == "imputed_value_mask_basin_membership" and r.severity == "OK" for r in report.records
    )
    assert any(
        r.check_id == "imputed_value_mask_basin_order" and r.severity == "WARNING" for r in report.records
    )
    assert not any(r.check_id == "imputed_value_mask_basin_order" and r.severity == "ERROR" for r in report.records)
    assert report.status == "PASS"


def test_fails_for_missing_basin_in_imputed_value_mask(tmp_path):
    fixture = _build_package(tmp_path)
    basins_present = list(BASIN_IDS[:-1])  # drop one accepted basin -- a genuine membership gap
    mask_df = pd.DataFrame(False, index=basins_present, columns=STATIC_COLUMNS)
    imputed_value_mask_path = tmp_path / "missing_basin_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)

    report, _ = _run(fixture, imputed_value_mask_path=imputed_value_mask_path)

    assert report.status == "FAIL"
    assert any(
        r.check_id == "imputed_value_mask_basin_membership" and r.severity == "ERROR" for r in report.records
    )


def test_fails_when_imputed_cell_has_no_manifest_fitted_value(tmp_path):
    fixture = _build_package(tmp_path)

    imputation_manifest_path = tmp_path / "no_fitted_value_manifest.json"
    imputation_manifest_path.write_text(
        json.dumps({"per_column": {"attr_a": {}}}), encoding="utf-8"
    )

    mask_df = pd.DataFrame(False, index=list(BASIN_IDS), columns=STATIC_COLUMNS)
    mask_df.loc[BASIN_IDS[0], "attr_a"] = True
    imputed_value_mask_path = tmp_path / "missing_fitted_value_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)

    report, _ = _run(
        fixture,
        imputation_manifest_path=imputation_manifest_path,
        imputed_value_mask_path=imputed_value_mask_path,
    )
    assert report.status == "FAIL"
    assert any(
        r.check_id == "imputed_values_have_manifest_fitted_value" and r.severity == "ERROR" for r in report.records
    )
    # distinct from the wrong-value check -- there is no fitted_value to compare against.
    assert not any(
        r.check_id == "imputed_values_placed_correctly" and r.severity == "ERROR" for r in report.records
    )


def test_fails_for_per_column_imputation_count_mismatch(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]

    imputation_manifest_path = tmp_path / "count_mismatch_manifest.json"
    imputation_manifest_path.write_text(
        json.dumps({"per_column": {"attr_a": {"fitted_value": 0.0, "n_missing_before_apply": 5}}}),
        encoding="utf-8",
    )

    mask_df = pd.DataFrame(False, index=list(BASIN_IDS), columns=STATIC_COLUMNS)
    mask_df.loc[basin_id, "attr_a"] = True  # only 1 true cell, manifest claims 5
    imputed_value_mask_path = tmp_path / "count_mismatch_mask.parquet"
    mask_df.reset_index().rename(columns={"index": "gauge_id"}).to_parquet(imputed_value_mask_path)

    report, _ = _run(
        fixture,
        imputation_manifest_path=imputation_manifest_path,
        imputed_value_mask_path=imputed_value_mask_path,
    )
    assert report.status == "FAIL"
    assert any(
        r.check_id == "imputed_value_mask_per_column_count_matches_manifest" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 19. correction 8: exact QC evidence membership + manifest cross-checks
# ---------------------------------------------------------------------------


def test_fails_for_extra_qc_csv_file(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    extra_csv = fixture["evidence_root"] / "csv_inspection" / "99999999.csv"
    extra_csv.write_text("time,mrms_qpe_1h_mm\n2023-01-01T00:00:00Z,0.0\n", encoding="utf-8")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == "qc_csv_exact_file_membership" and r.severity == "ERROR" for r in report.records)


def test_fails_for_qc_csv_manifest_entry_mismatch(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    manifest_path = fixture["evidence_root"] / "csv_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    basin_id = BASIN_IDS[0]
    manifest["files"][basin_id]["row_count"] = int(manifest["files"][basin_id]["row_count"]) + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "qc_csv_manifest_entry_matches_disk" and r.severity == "ERROR" for r in report.records
    )


def test_fails_when_qc_manifest_missing(tmp_path):
    fixture = _build_package(tmp_path, write_qc_csv=True)
    (fixture["evidence_root"] / "csv_manifest.json").unlink()

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == "qc_csv_manifest_present" and r.severity == "ERROR" for r in report.records)


# ---------------------------------------------------------------------------
# 20. correction 10: auditor git-state guard (resolvable + clean, hard fail)
# ---------------------------------------------------------------------------


def test_resolve_auditor_git_commit_raises_when_unresolvable(tmp_path):
    non_git_dir = tmp_path / "not_a_repo"
    non_git_dir.mkdir()
    with pytest.raises(package_audit.PackageAuditError):
        package_audit.resolve_auditor_git_commit(non_git_dir)


def test_check_auditor_working_tree_clean_raises_when_dirty(tmp_path):
    repo_dir = _clean_git_repo(tmp_path)
    (repo_dir / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(package_audit.PackageAuditError):
        package_audit.check_auditor_working_tree_clean(repo_dir)


def test_check_auditor_working_tree_clean_passes_when_clean(tmp_path):
    repo_dir = _clean_git_repo(tmp_path)
    package_audit.check_auditor_working_tree_clean(repo_dir)  # must not raise


def test_run_audit_fails_when_auditor_repo_dirty(tmp_path):
    fixture = _build_package(tmp_path)
    (fixture["auditor_repo_root"] / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(package_audit.PackageAuditError):
        _run(fixture)


def test_run_audit_fails_when_auditor_repo_unresolvable(tmp_path):
    fixture = _build_package(tmp_path)
    non_git_dir = tmp_path / "not_a_repo_for_run_audit"
    non_git_dir.mkdir()
    with pytest.raises(package_audit.PackageAuditError):
        _run(fixture, auditor_repo_root=non_git_dir)


# ---------------------------------------------------------------------------
# 21. versioned package schema (date) -- independent auditor coverage
# ---------------------------------------------------------------------------


def _rename_temporal_coordinate(nc_path, old_name, new_name):
    """Rename a basin NetCDF's temporal dimension/coordinate, preserving attrs.

    Goes through xarray's own load-then-rename-then-write round trip rather
    than netCDF4's low-level ``renameDimension``/``renameVariable``: renaming
    a coordinate dimension and its associated variable separately via the
    raw netCDF4 API left the file's CF time-decoding corrupted (observed as
    a spurious ``OverflowError``/``ValueError`` on the next read), whereas
    xarray keeps the coordinate's encoding consistent across the rename.
    """
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        loaded = ds.load()
    renamed = loaded.rename({old_name: new_name})
    renamed.to_netcdf(nc_path, mode="w", engine="netcdf4")


def _add_extra_temporal_dimension(nc_path, existing_name, extra_name):
    with netCDF4.Dataset(nc_path, "r+") as ds:
        n = len(ds.dimensions[existing_name])
        ds.createDimension(extra_name, n)
        var = ds.createVariable(extra_name, "f8", (extra_name,))
        var[:] = np.arange(n, dtype=np.float64)


def _set_manifest_netcdf_schema_fields(fixture, *, name, version, coordinate_name):
    manifest_path = fixture["package_root"] / "manifests" / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["netcdf_package_schema_name"] = name
    manifest["netcdf_package_schema_version"] = version
    manifest["netcdf_time_coordinate"] = coordinate_name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _set_run_provenance_netcdf_schema_fields(fixture, *, name, version, coordinate_name):
    provenance_path = fixture["package_root"] / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["netcdf_package_schema_name"] = name
    provenance["netcdf_package_schema_version"] = version
    provenance["netcdf_time_coordinate"] = coordinate_name
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")


def test_full_audit_passes_for_correct_v002_date_package(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=SCIENTIFIC_V002_SCHEMA)
    report, diagnostics = _run(fixture)
    assert report.status == "PASS", report.failed_messages()
    assert report.error_count == 0
    assert diagnostics["audit_manifest"]["build_git_commit"] == "test-commit"


def test_fails_when_disk_coordinate_is_date_but_declared_schema_is_v001(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    basin_id = BASIN_IDS[0]
    _rename_temporal_coordinate(_nc_path(fixture, basin_id), "time", "date")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_temporal_coordinate_matches_declared_schema[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


def test_fails_when_disk_coordinate_is_time_but_declared_schema_is_v002(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=SCIENTIFIC_V002_SCHEMA)
    basin_id = BASIN_IDS[0]
    _rename_temporal_coordinate(_nc_path(fixture, basin_id), "date", "time")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_temporal_coordinate_matches_declared_schema[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


def test_fails_when_netcdf_has_both_time_and_date_dimensions(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    _add_extra_temporal_dimension(_nc_path(fixture, basin_id), "time", "date")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_temporal_coordinate_present[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


def test_fails_when_netcdf_has_neither_time_nor_date_dimension(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    _rename_temporal_coordinate(_nc_path(fixture, basin_id), "time", "when")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_temporal_coordinate_present[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


def test_fails_for_unrecognized_netcdf_package_schema_name(tmp_path):
    fixture = _build_package(tmp_path)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.setncattr("package_schema_name", "totally_unrecognized_schema_v999")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_package_schema_identity_recognized[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


def test_fails_when_basin_netcdf_files_declare_mixed_schema_identities(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.setncattr("package_schema_name", SCIENTIFIC_V002_SCHEMA.name)
        ds.setncattr("package_schema_version", SCIENTIFIC_V002_SCHEMA.version)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "package_all_basins_same_netcdf_schema" and r.severity == "ERROR" for r in report.records
    )


def test_fails_when_manifest_declares_different_schema_than_netcdf_files(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    basin_id = BASIN_IDS[0]
    _set_manifest_netcdf_schema_fields(
        fixture,
        name=SCIENTIFIC_V002_SCHEMA.name,
        version=SCIENTIFIC_V002_SCHEMA.version,
        coordinate_name=SCIENTIFIC_V002_SCHEMA.coordinate_name,
    )

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(r.check_id == f"netcdf_package_schema[{basin_id}]" and r.severity == "ERROR" for r in report.records)


def test_fails_when_run_provenance_declares_different_schema_than_netcdf_files(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    basin_id = BASIN_IDS[0]
    _set_run_provenance_netcdf_schema_fields(
        fixture,
        name=SCIENTIFIC_V002_SCHEMA.name,
        version=SCIENTIFIC_V002_SCHEMA.version,
        coordinate_name=SCIENTIFIC_V002_SCHEMA.coordinate_name,
    )

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_matches_run_provenance_schema[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


# ---------------------------------------------------------------------------
# 22. historical v001 metadata compatibility fallback (correction round)
# ---------------------------------------------------------------------------
#
# The real, already-built, already-certified Compact Scientific Package v001
# predates the netcdf_package_schema_*/netcdf_time_coordinate fields
# entirely and must never be rewritten just to re-audit it. These tests
# simulate that real historical state by building a genuine v001 package
# with the current builder (which always writes the new fields) and then
# stripping those fields back out, while leaving every pre-existing
# builder/manifest identity field (schema_name, schema_version, package_role,
# builder_module, the deprecated package_schema_name) untouched -- exactly
# what a real historical manifest/run_provenance.json looks like.


def _delete_manifest_netcdf_schema_fields(fixture):
    manifest_path = fixture["package_root"] / "manifests" / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("netcdf_package_schema_name", "netcdf_package_schema_version", "netcdf_time_coordinate"):
        manifest.pop(key, None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _delete_run_provenance_netcdf_schema_fields(fixture):
    provenance_path = fixture["package_root"] / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for key in ("netcdf_package_schema_name", "netcdf_package_schema_version", "netcdf_time_coordinate"):
        provenance.pop(key, None)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")


def _set_manifest_netcdf_schema_name_only(fixture, name):
    """Leave name present but strip version/coordinate -- a partial, not

    all-or-none, declaration that must never resolve via the historical
    fallback (that fallback only applies when all three fields are absent).
    """
    manifest_path = fixture["package_root"] / "manifests" / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("netcdf_package_schema_version", None)
    manifest.pop("netcdf_time_coordinate", None)
    manifest["netcdf_package_schema_name"] = name
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_historical_v001_fallback_used_when_explicit_fields_absent_and_lineage_recognized(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    _delete_manifest_netcdf_schema_fields(fixture)
    _delete_run_provenance_netcdf_schema_fields(fixture)

    report, _ = _run(fixture)
    assert report.status == "PASS", report.failed_messages()
    assert report.error_count == 0
    assert any(
        r.check_id == "package_manifest_netcdf_schema_historical_v001_compatibility_used" and r.severity == "OK"
        for r in report.records
    )
    assert any(
        r.check_id == "run_provenance_netcdf_schema_historical_v001_compatibility_used" and r.severity == "OK"
        for r in report.records
    )


def test_historical_v001_fallback_still_fails_when_a_basin_file_declares_v002(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    _delete_manifest_netcdf_schema_fields(fixture)
    _delete_run_provenance_netcdf_schema_fields(fixture)
    basin_id = BASIN_IDS[0]
    with netCDF4.Dataset(_nc_path(fixture, basin_id), "r+") as ds:
        ds.setncattr("package_schema_name", SCIENTIFIC_V002_SCHEMA.name)
        ds.setncattr("package_schema_version", SCIENTIFIC_V002_SCHEMA.version)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_package_schema[{basin_id}]" and r.severity == "ERROR" for r in report.records
    )


def test_historical_v001_fallback_still_fails_when_a_basin_file_uses_date_coordinate(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    _delete_manifest_netcdf_schema_fields(fixture)
    _delete_run_provenance_netcdf_schema_fields(fixture)
    basin_id = BASIN_IDS[0]
    _rename_temporal_coordinate(_nc_path(fixture, basin_id), "time", "date")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == f"netcdf_temporal_coordinate_matches_declared_schema[{basin_id}]" and r.severity == "ERROR"
        for r in report.records
    )


def test_missing_netcdf_schema_fields_fails_when_builder_lineage_not_recognized(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    _delete_manifest_netcdf_schema_fields(fixture)
    _delete_run_provenance_netcdf_schema_fields(fixture)

    manifest_path = fixture["package_root"] / "manifests" / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_name"] = "some_other_unrecognized_builder_v001"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "package_manifest_netcdf_schema_name_present" and r.severity == "ERROR"
        for r in report.records
    )
    assert not any(
        r.check_id == "package_manifest_netcdf_schema_historical_v001_compatibility_used"
        for r in report.records
    )


def test_partial_netcdf_schema_fields_in_manifest_fails_even_with_historical_lineage(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=DEFAULT_PACKAGE_NETCDF_SCHEMA)
    _set_manifest_netcdf_schema_name_only(fixture, DEFAULT_PACKAGE_NETCDF_SCHEMA.name)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "package_manifest_netcdf_schema_version_present" and r.severity == "ERROR"
        for r in report.records
    )
    assert any(
        r.check_id == "package_manifest_netcdf_schema_coordinate_present" and r.severity == "ERROR"
        for r in report.records
    )
    assert not any(
        r.check_id == "package_manifest_netcdf_schema_historical_v001_compatibility_used"
        for r in report.records
    )


def test_v002_package_missing_explicit_fields_fails_and_does_not_use_historical_fallback(tmp_path):
    fixture = _build_package(tmp_path, package_netcdf_schema=SCIENTIFIC_V002_SCHEMA)
    _delete_manifest_netcdf_schema_fields(fixture)
    _delete_run_provenance_netcdf_schema_fields(fixture)

    report, _ = _run(fixture)
    assert report.status == "FAIL"
    assert any(
        r.check_id == "package_manifest_netcdf_schema_name_present" and r.severity == "ERROR"
        for r in report.records
    )
    assert any(
        r.check_id == "run_provenance_netcdf_schema_name_present" and r.severity == "ERROR"
        for r in report.records
    )
    assert not any("historical_v001_compatibility_used" in r.check_id for r in report.records)
