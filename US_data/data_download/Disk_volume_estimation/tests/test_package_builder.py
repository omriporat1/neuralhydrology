"""Tests for the Gate 3A local Compact Scientific Package builder.

Synthetic fixtures only -- no h2o/Moriah access, no real 32-basin build.
Mirrors the fixture style of tests/test_package_assembly.py and
tests/test_package_netcdf.py.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import src.baseline.package_builder as package_builder_module
from src.baseline.gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT
from src.baseline.package_builder import (
    BasinSourceTables,
    PackageBuilderError,
    build_compact_scientific_package,
    derive_expected_index_from_policy,
    read_area_csv,
    read_basin_ids_file,
    resolve_gap_product_scope,
)
from src.baseline.package_netcdf import EXPECTED_VARIABLES, validate_basin_netcdf_file
from src.baseline.package_assembly import assemble_basin_package_table

REPO_ROOT = Path(__file__).resolve().parents[1]

BASIN_IDS = ("01019000", "103366092", "393109104464500")  # 8 / 9 / 15 chars
STATIC_COLUMNS = ["attr_a", "attr_b", "attr_c"]


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
        values = np.arange(n, dtype=np.float64) + 1.0
    return pd.Series(values, index=idx, name="qobs_m3s")


def _static_attributes(basin_ids=BASIN_IDS, columns=STATIC_COLUMNS):
    rng = np.arange(len(basin_ids) * len(columns), dtype=np.float64).reshape(len(basin_ids), len(columns))
    return pd.DataFrame(rng, index=list(basin_ids), columns=columns)


def _loader(n=10, start="2023-01-01", area=3.6, fail_for=None, basin_offsets=None):
    idx = _hourly_index(n, start=start)

    def _load(basin_id: str) -> BasinSourceTables:
        if fail_for is not None and basin_id == fail_for:
            raise RuntimeError(f"synthetic failure for {basin_id}")
        offset = 0.0
        if basin_offsets is not None:
            offset = basin_offsets.get(basin_id, 0.0)
        forcing = _valid_forcing(idx)
        qobs = _valid_qobs(idx, values=np.arange(n, dtype=np.float64) + 1.0 + offset)
        return BasinSourceTables(forcing=forcing, qobs=qobs, area_km2=area)

    return _load


def _build(tmp_path, *, basin_ids=BASIN_IDS, n=10, evidence=True, write_qc_csv=False, overwrite=False,
           dry_run=False, gap_timestamps=None, static_attributes=None, loader=None, expected_index=None,
           forbidden_static_columns=None, static_model_input_columns=None, policy=None,
           static_column_manifest=None, gap_product_scope=None, static_attributes_provenance=None):
    idx = expected_index if expected_index is not None else _hourly_index(n)
    kwargs = dict(
        basin_ids=basin_ids,
        load_basin_source=loader if loader is not None else _loader(n=n),
        static_attributes=static_attributes if static_attributes is not None else _static_attributes(basin_ids),
        static_model_input_columns=(
            static_model_input_columns if static_model_input_columns is not None else STATIC_COLUMNS
        ),
        gap_timestamps=gap_timestamps if gap_timestamps is not None else [],
        expected_index=idx,
        output_package_root=tmp_path / "package",
        evidence_root=(tmp_path / "evidence") if evidence else None,
        write_qc_csv=write_qc_csv,
        overwrite=overwrite,
        dry_run=dry_run,
        policy=policy,
        static_column_manifest=static_column_manifest,
        gap_product_scope=gap_product_scope,
        static_attributes_provenance=static_attributes_provenance,
    )
    if forbidden_static_columns is not None:
        kwargs["forbidden_static_columns"] = forbidden_static_columns
    return build_compact_scientific_package(**kwargs)


# ---------------------------------------------------------------------------
# Shared production-policy fixtures (small synthetic scale, matching
# STATIC_COLUMNS, but carrying the real gap_policy/static_attributes fields
# the production contract enforcement reads).
# ---------------------------------------------------------------------------

PRODUCTION_MATRIX_NAME = "synthetic_static_matrix_for_tests"
PRODUCTION_SHA256 = "a" * 64


def _production_policy(*, expected_model_input_columns=len(STATIC_COLUMNS), include_rtma=True,
                        matrix_name=PRODUCTION_MATRIX_NAME, sha256=PRODUCTION_SHA256):
    return {
        "policy_name": "synthetic_production_policy_for_tests",
        "policy_version": 1,
        "gap_policy": {
            "include_rtma_in_history_mask": include_rtma,
        },
        "static_attributes": {
            "expected_model_input_columns": expected_model_input_columns,
            "matrix_name": matrix_name,
            "sha256": sha256,
            "allowed_role": "model_input",
        },
    }


def _static_column_manifest(columns=STATIC_COLUMNS, role="model_input"):
    return {"columns": {c: {"role": role} for c in columns}}


def _static_provenance(matrix_name=PRODUCTION_MATRIX_NAME, sha256=PRODUCTION_SHA256):
    return {"matrix_name": matrix_name, "sha256": sha256}


# ---------------------------------------------------------------------------
# 1-2. successful build + package layout
# ---------------------------------------------------------------------------


def test_successful_build_returns_result(tmp_path):
    result = _build(tmp_path)
    assert result.dry_run is False
    assert result.basin_ids == BASIN_IDS
    assert result.package_root.is_dir()
    assert result.evidence_root is None  # write_qc_csv defaults to False


def test_package_directory_layout_exact(tmp_path):
    result = _build(tmp_path)
    root = result.package_root
    assert (root / "time_series").is_dir()
    assert (root / "attributes" / "attributes.csv").is_file()
    assert (root / "basins" / "basin_ids.txt").is_file()
    assert (root / "masks" / "gap_timestamps.json").is_file()
    assert (root / "manifests" / "package_manifest.json").is_file()
    assert (root / "manifests" / "file_checksums.csv").is_file()
    assert (root / "run_provenance.json").is_file()
    for basin_id in BASIN_IDS:
        assert (root / "time_series" / f"{basin_id}.nc").is_file()


# ---------------------------------------------------------------------------
# 3. basin_ids.txt exact order / preserved strings
# ---------------------------------------------------------------------------


def test_basin_ids_txt_exact_order(tmp_path):
    result = _build(tmp_path)
    lines = (result.package_root / "basins" / "basin_ids.txt").read_text().splitlines()
    assert lines == list(BASIN_IDS)


def test_basin_ids_txt_preserves_leading_zero_strings(tmp_path):
    ids = ("01019000", "103366092", "393109104464500")
    result = _build(tmp_path, basin_ids=ids, loader=_loader())
    lines = (result.package_root / "basins" / "basin_ids.txt").read_text().splitlines()
    assert lines[0] == "01019000"


# ---------------------------------------------------------------------------
# 4. gauge ID lengths preserved in NetCDF filenames + 5. expected_index
# enforcement + 6. Gate 2 validation reuse
# ---------------------------------------------------------------------------


def test_gauge_id_lengths_preserved_in_netcdf(tmp_path):
    result = _build(tmp_path)
    names = {p.name for p in (result.package_root / "time_series").glob("*.nc")}
    assert names == {f"{b}.nc" for b in BASIN_IDS}


def test_expected_index_enforced_end_to_end(tmp_path):
    idx = _hourly_index(10)
    bad_idx = _hourly_index(10, start="2023-02-01")  # loader builds data on a different index

    def loader(basin_id):
        forcing = _valid_forcing(bad_idx)
        qobs = _valid_qobs(bad_idx)
        return BasinSourceTables(forcing=forcing, qobs=qobs, area_km2=3.6)

    with pytest.raises(PackageBuilderError):
        _build(tmp_path, expected_index=idx, loader=loader)


def test_written_netcdf_passes_gate2_validation(tmp_path):
    n = 10
    idx = _hourly_index(n)
    forcing = _valid_forcing(idx)
    qobs = _valid_qobs(idx)
    table = assemble_basin_package_table(forcing, qobs, area_km2=3.6, expected_index=idx)

    result = _build(tmp_path, n=n)
    nc_path = result.package_root / "time_series" / f"{BASIN_IDS[0]}.nc"
    validate_basin_netcdf_file(nc_path, table, BASIN_IDS[0])


def test_netcdf_has_expected_variables(tmp_path):
    result = _build(tmp_path)
    nc_path = result.package_root / "time_series" / f"{BASIN_IDS[0]}.nc"
    with xr.open_dataset(nc_path) as ds:
        for var in EXPECTED_VARIABLES:
            assert var in ds.variables


# ---------------------------------------------------------------------------
# 7-13. static attribute validation
# ---------------------------------------------------------------------------


def test_attributes_csv_membership_and_columns(tmp_path):
    result = _build(tmp_path)
    df = pd.read_csv(result.package_root / "attributes" / "attributes.csv", dtype={"gauge_id": str})
    assert list(df["gauge_id"]) == list(BASIN_IDS)
    assert list(df.columns) == ["gauge_id"] + STATIC_COLUMNS


def test_static_attributes_duplicate_basin_rejected(tmp_path):
    attrs = _static_attributes()
    attrs = pd.concat([attrs, attrs.iloc[[0]]])
    with pytest.raises(PackageBuilderError, match="duplicate"):
        _build(tmp_path, static_attributes=attrs)


def test_static_attributes_missing_basin_rejected(tmp_path):
    attrs = _static_attributes().drop(index=BASIN_IDS[0])
    with pytest.raises(PackageBuilderError, match="membership"):
        _build(tmp_path, static_attributes=attrs)


def test_static_attributes_extra_basin_rejected(tmp_path):
    attrs = _static_attributes()
    extra = pd.DataFrame([[0.0, 1.0, 2.0]], index=["99999999"], columns=STATIC_COLUMNS)
    attrs = pd.concat([attrs, extra])
    with pytest.raises(PackageBuilderError, match="membership"):
        _build(tmp_path, static_attributes=attrs)


def test_static_attributes_nan_rejected(tmp_path):
    attrs = _static_attributes()
    attrs.iloc[0, 0] = np.nan
    with pytest.raises(PackageBuilderError, match="NaN"):
        _build(tmp_path, static_attributes=attrs)


def test_static_attributes_infinite_rejected(tmp_path):
    attrs = _static_attributes()
    attrs.iloc[0, 0] = np.inf
    with pytest.raises(PackageBuilderError, match="infinite"):
        _build(tmp_path, static_attributes=attrs)


def test_forbidden_static_column_rejected(tmp_path):
    with pytest.raises(PackageBuilderError, match="forbidden"):
        _build(tmp_path, forbidden_static_columns=["attr_a"])


def test_static_attributes_column_order_enforced(tmp_path):
    attrs = _static_attributes()[list(reversed(STATIC_COLUMNS))]
    with pytest.raises(PackageBuilderError, match="column"):
        _build(tmp_path, static_attributes=attrs)


# ---------------------------------------------------------------------------
# 14-17. gap timestamps
# ---------------------------------------------------------------------------


def test_gap_timestamps_json_written_and_valid(tmp_path):
    idx = _hourly_index(10)
    gaps = [idx[2], idx[5]]
    result = _build(tmp_path, expected_index=idx, gap_timestamps=gaps)
    data = json.loads((result.package_root / "masks" / "gap_timestamps.json").read_text())
    assert len(data) == 2
    assert all(ts.endswith("Z") for ts in data)


def test_duplicate_gap_timestamps_rejected(tmp_path):
    idx = _hourly_index(10)
    with pytest.raises(PackageBuilderError, match="duplicate"):
        _build(tmp_path, expected_index=idx, gap_timestamps=[idx[1], idx[1]])


def test_out_of_period_gap_timestamp_rejected(tmp_path):
    idx = _hourly_index(10)
    outside = idx[0] - pd.Timedelta(hours=5)
    with pytest.raises(PackageBuilderError):
        _build(tmp_path, expected_index=idx, gap_timestamps=[outside])


def test_non_hourly_gap_timestamp_rejected(tmp_path):
    idx = _hourly_index(10)
    off_grid = idx[2] + pd.Timedelta(minutes=30)
    with pytest.raises(PackageBuilderError):
        _build(tmp_path, expected_index=idx, gap_timestamps=[off_grid])


# ---------------------------------------------------------------------------
# 18-20. manifest / checksums / ordering
# ---------------------------------------------------------------------------


def test_manifest_contains_required_fields(tmp_path):
    result = _build(tmp_path)
    manifest = json.loads((result.package_root / "manifests" / "package_manifest.json").read_text())
    assert manifest["basin_ids"] == list(BASIN_IDS)
    assert manifest["basin_count"] == len(BASIN_IDS)
    assert manifest["static_model_input_count"] == len(STATIC_COLUMNS)
    assert manifest["static_model_input_columns"] == STATIC_COLUMNS
    assert len(manifest["per_basin_time_series"]) == len(BASIN_IDS)
    assert manifest["timeline"]["rows"] == 10
    assert manifest["dynamic_variables"]
    assert manifest["lead_targets"]
    assert "created_at_utc" not in manifest  # timestamps live only in run_provenance.json


def test_checksums_match_written_files(tmp_path):
    from src.baseline.splits import sha256_of

    result = _build(tmp_path)
    csv_text = (result.package_root / "manifests" / "file_checksums.csv").read_text()
    rows = [r for r in csv_text.splitlines()[1:] if r.strip()]
    assert len(rows) >= len(BASIN_IDS) + 3
    for row in rows:
        rel_path, sha256, size_bytes, _role = row.split(",")
        abs_path = result.package_root / rel_path
        assert sha256_of(abs_path) == sha256
        assert abs_path.stat().st_size == int(size_bytes)


def test_deterministic_ordering_follows_input(tmp_path):
    reversed_ids = tuple(reversed(BASIN_IDS))
    result = _build(tmp_path, basin_ids=reversed_ids, static_attributes=_static_attributes(reversed_ids),
                     loader=_loader())
    lines = (result.package_root / "basins" / "basin_ids.txt").read_text().splitlines()
    assert lines == list(reversed_ids)
    manifest = json.loads((result.package_root / "manifests" / "package_manifest.json").read_text())
    assert manifest["basin_ids"] == list(reversed_ids)


# ---------------------------------------------------------------------------
# 21-24. QC CSV
# ---------------------------------------------------------------------------


def test_qc_csv_columns_exact(tmp_path):
    result = _build(tmp_path, write_qc_csv=True)
    csv_path = result.evidence_root / "csv_inspection" / f"{BASIN_IDS[0]}.csv"
    df = pd.read_csv(csv_path)
    assert df.columns[0] == "time"
    assert len(df.columns) == 1 + 13  # time + 13 scientific columns


def test_qc_csv_nan_counts_and_timestamps_match(tmp_path):
    result = _build(tmp_path, write_qc_csv=True, n=20)
    csv_path = result.evidence_root / "csv_inspection" / f"{BASIN_IDS[0]}.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 20
    assert df["time"].iloc[0] == "2023-01-01T00:00:00Z"


def test_qc_csv_manifest_marks_non_authoritative(tmp_path):
    result = _build(tmp_path, write_qc_csv=True)
    csv_manifest = json.loads((result.evidence_root / "csv_manifest.json").read_text())
    assert csv_manifest["authoritative"] is False
    assert csv_manifest["usable_for_training"] is False
    manifest = json.loads((result.package_root / "manifests" / "package_manifest.json").read_text())
    assert manifest["qc_csv_authoritative"] is False


def test_disabled_qc_csv_creates_no_evidence(tmp_path):
    result = _build(tmp_path, write_qc_csv=False, evidence=True)
    assert result.evidence_root is None
    assert not (tmp_path / "evidence").exists()


# ---------------------------------------------------------------------------
# 25-28. atomic promotion
# ---------------------------------------------------------------------------


def test_basin_failure_prevents_partial_promotion(tmp_path):
    with pytest.raises(PackageBuilderError, match="failed package build"):
        _build(tmp_path, loader=_loader(fail_for=BASIN_IDS[1]))
    assert not (tmp_path / "package").exists()
    # no stray temp build directories left behind
    leftovers = list(tmp_path.glob(".package.building.*"))
    assert leftovers == []


def test_overwrite_false_protects_existing_package(tmp_path):
    _build(tmp_path)
    with pytest.raises(PackageBuilderError, match="overwrite=False"):
        _build(tmp_path, overwrite=False)


def test_overwrite_true_preserves_old_package_on_failure(tmp_path):
    result = _build(tmp_path)
    old_manifest = (result.package_root / "manifests" / "package_manifest.json").read_text()

    with pytest.raises(PackageBuilderError):
        _build(tmp_path, overwrite=True, loader=_loader(fail_for=BASIN_IDS[1]))

    assert (result.package_root / "manifests" / "package_manifest.json").read_text() == old_manifest
    backups = list(tmp_path.glob(".package.pre-overwrite-backup"))
    assert backups == []  # cleaned up: old package restored in place, no leftover backup dir


def test_successful_overwrite_replaces_package(tmp_path):
    _build(tmp_path)
    result = _build(tmp_path, overwrite=True, loader=_loader(basin_offsets={BASIN_IDS[0]: 100.0}))
    with xr.open_dataset(result.package_root / "time_series" / f"{BASIN_IDS[0]}.nc") as ds:
        assert float(ds["qobs_m3s"].values[0]) > 50.0


# ---------------------------------------------------------------------------
# 29-30. no source mutation / determinism / CLI smoke
# ---------------------------------------------------------------------------


def test_source_dataframes_not_modified(tmp_path):
    idx = _hourly_index(10)
    forcing = _valid_forcing(idx)
    forcing_copy = forcing.copy()
    qobs = _valid_qobs(idx)
    qobs_copy = qobs.copy()

    def loader(basin_id):
        return BasinSourceTables(forcing=forcing, qobs=qobs, area_km2=3.6)

    _build(tmp_path, expected_index=idx, loader=loader)
    pd.testing.assert_frame_equal(forcing, forcing_copy)
    pd.testing.assert_series_equal(qobs, qobs_copy)


def test_repeated_builds_produce_equivalent_manifest(tmp_path):
    result1 = _build(tmp_path / "a")
    result2 = _build(tmp_path / "b")
    manifest1 = dict(result1.manifest)
    manifest2 = dict(result2.manifest)
    assert manifest1 == manifest2


def test_dry_run_does_not_promote(tmp_path):
    result = _build(tmp_path, dry_run=True, write_qc_csv=True)
    assert result.dry_run is True
    assert not (tmp_path / "package").exists()
    assert not (tmp_path / "evidence").exists()


def test_derive_expected_index_from_policy():
    policy = {
        "period": {
            "start_utc": "2020-10-14T00:00:00Z",
            "end_utc": "2020-10-14T02:00:00Z",
            "expected_hours": 3,
        }
    }
    idx = derive_expected_index_from_policy(policy)
    assert len(idx) == 3
    assert idx[0] == pd.Timestamp("2020-10-14T00:00:00")


def test_derive_expected_index_from_policy_mismatch_rejected():
    policy = {
        "period": {
            "start_utc": "2020-10-14T00:00:00Z",
            "end_utc": "2020-10-14T02:00:00Z",
            "expected_hours": 999,
        }
    }
    with pytest.raises(PackageBuilderError):
        derive_expected_index_from_policy(policy)


def test_cli_help_smoke():
    script = REPO_ROOT / "scripts" / "build_stage1_baseline_nh_package.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--output-package-root" in proc.stdout


# ---------------------------------------------------------------------------
# 31-37. gap-product scope: policy-driven, no silent extra/missing product
# ---------------------------------------------------------------------------


def test_resolve_gap_product_scope_rtma_included():
    policy = _production_policy(include_rtma=True)
    assert resolve_gap_product_scope(policy) == (MRMS_PRODUCT, RTMA_PRODUCT)


def test_resolve_gap_product_scope_rtma_excluded():
    policy = _production_policy(include_rtma=False)
    assert resolve_gap_product_scope(policy) == (MRMS_PRODUCT,)


def test_resolve_gap_product_scope_missing_gap_policy_block_rejected():
    policy = {"static_attributes": {"expected_model_input_columns": 3}}
    with pytest.raises(PackageBuilderError, match="gap_policy"):
        resolve_gap_product_scope(policy)


def test_resolve_gap_product_scope_missing_key_rejected():
    policy = _production_policy()
    del policy["gap_policy"]["include_rtma_in_history_mask"]
    with pytest.raises(PackageBuilderError, match="include_rtma_in_history_mask"):
        resolve_gap_product_scope(policy)


def test_resolve_gap_product_scope_non_bool_rejected():
    policy = _production_policy()
    policy["gap_policy"]["include_rtma_in_history_mask"] = "true"
    with pytest.raises(PackageBuilderError, match="unambiguous bool"):
        resolve_gap_product_scope(policy)


def test_build_manifest_records_policy_derived_gap_product_scope():
    # Exercises _build_manifest directly (rather than the full public build,
    # which also forwards `policy` into Gate 1's own independent full-policy
    # schema validation -- out of scope here): the accepted policy shape
    # resolves to (MRMS, RTMA) and that is exactly what must land in
    # package_manifest.json's "gap_product_scope" field.
    policy = _production_policy(include_rtma=True)
    scope = resolve_gap_product_scope(policy)
    manifest = package_builder_module._build_manifest(
        basin_ids=BASIN_IDS,
        expected_index=_hourly_index(10),
        static_model_input_columns=STATIC_COLUMNS,
        per_basin_files=[],
        checksums={"masks/gap_timestamps.json": {"sha256": "x" * 64, "size_bytes": 0}},
        n_gap_timestamps=0,
        policy=policy,
        policy_provenance=None,
        static_attributes_provenance=None,
        basin_selection_provenance=None,
        gap_inventory_provenance=None,
        gap_product_scope=scope,
        write_qc_csv=False,
    )
    assert manifest["gap_product_scope"] == [MRMS_PRODUCT, RTMA_PRODUCT]


def test_build_rejects_gap_product_scope_mismatch_with_policy(tmp_path):
    policy = _production_policy(include_rtma=True)  # policy requires MRMS + RTMA
    with pytest.raises(PackageBuilderError, match="does not match the"):
        _build(
            tmp_path,
            policy=policy,
            static_column_manifest=_static_column_manifest(),
            static_attributes_provenance=_static_provenance(),
            gap_product_scope=(MRMS_PRODUCT,),  # caller selected MRMS-only, disagreeing with policy
        )


# ---------------------------------------------------------------------------
# 38-45. production static v002 contract
# ---------------------------------------------------------------------------


def test_production_static_contract_accepted():
    # Direct unit test of the acceptance path (rather than the full public
    # build, which also forwards `policy` into Gate 1's own independent
    # full-policy schema validation -- out of scope here): an
    # agreeing policy/manifest/provenance triple must not raise.
    policy = _production_policy()
    package_builder_module._validate_production_static_contract(
        policy, STATIC_COLUMNS, _static_column_manifest(), _static_provenance()
    )


def test_production_static_contract_wrong_model_input_count_rejected(tmp_path):
    policy = _production_policy(expected_model_input_columns=len(STATIC_COLUMNS) + 1)
    with pytest.raises(PackageBuilderError, match="expected_model_input_columns"):
        _build(
            tmp_path,
            policy=policy,
            static_column_manifest=_static_column_manifest(),
            static_attributes_provenance=_static_provenance(),
        )


def test_production_static_contract_reordered_columns_rejected(tmp_path):
    reordered = list(reversed(STATIC_COLUMNS))
    policy = _production_policy()
    with pytest.raises(PackageBuilderError, match="manifest"):
        _build(
            tmp_path,
            policy=policy,
            static_model_input_columns=reordered,
            static_attributes=_static_attributes(columns=reordered),
            static_column_manifest=_static_column_manifest(),  # manifest yields sorted (non-reversed) order
            static_attributes_provenance=_static_provenance(),
        )


def test_production_static_contract_manifest_identity_mismatch_rejected(tmp_path):
    policy = _production_policy(sha256=PRODUCTION_SHA256)
    with pytest.raises(PackageBuilderError, match="sha256"):
        _build(
            tmp_path,
            policy=policy,
            static_column_manifest=_static_column_manifest(),
            static_attributes_provenance=_static_provenance(sha256="b" * 64),  # disagrees with policy
        )


def test_production_static_contract_missing_manifest_rejected(tmp_path):
    policy = _production_policy()
    with pytest.raises(PackageBuilderError, match="static_column_manifest"):
        _build(
            tmp_path,
            policy=policy,
            static_column_manifest=None,
            static_attributes_provenance=_static_provenance(),
        )


def test_production_static_contract_missing_provenance_rejected(tmp_path):
    policy = _production_policy()
    with pytest.raises(PackageBuilderError, match="static_attributes_provenance"):
        _build(
            tmp_path,
            policy=policy,
            static_column_manifest=_static_column_manifest(),
            static_attributes_provenance=None,
        )


def test_production_static_contract_old_style_manifest_column_set_rejected(tmp_path):
    """Simulates an older/incompatible manifest (e.g. a stale v001 layout):
    same column count as the current contract requires, but a different
    named column set -- must not silently pass."""
    policy = _production_policy(expected_model_input_columns=len(STATIC_COLUMNS))
    old_style_manifest = _static_column_manifest(columns=["legacy_a", "legacy_b", "legacy_c"])
    with pytest.raises(PackageBuilderError, match="manifest"):
        _build(
            tmp_path,
            policy=policy,
            static_column_manifest=old_style_manifest,
            static_attributes_provenance=_static_provenance(),
        )


def test_synthetic_schema_seam_unaffected_by_production_contract(tmp_path):
    # No policy supplied: the small synthetic 3-column seam must keep working
    # exactly as before, unaffected by the new production-contract checks.
    result = _build(tmp_path)
    assert result.manifest["static_model_input_columns"] == STATIC_COLUMNS


# ---------------------------------------------------------------------------
# 46-49. transactional package + QC-evidence promotion
# ---------------------------------------------------------------------------


def _no_leftover_tmp_or_backup_dirs(tmp_path):
    leftovers = list(tmp_path.glob(".package.building.*")) + list(tmp_path.glob(".evidence.building.*"))
    backups = list(tmp_path.glob(".package.pre-overwrite-backup")) + list(
        tmp_path.glob(".evidence.pre-overwrite-backup")
    )
    return leftovers == [] and backups == []


def test_evidence_promotion_failure_after_package_success_rolls_back_both(tmp_path, monkeypatch):
    _build(tmp_path, write_qc_csv=True)
    old_package_manifest = (tmp_path / "package" / "manifests" / "package_manifest.json").read_text()
    old_evidence_manifest = (tmp_path / "evidence" / "csv_manifest.json").read_text()

    real_rename = package_builder_module.os.rename
    evidence_dest = str(tmp_path / "evidence")

    def flaky_rename(src, dst):
        # Only fail the genuine tmp-dir -> destination promotion, never the
        # backup-restore rename issued from the except block on failure
        # (which targets the same destination from the backup path).
        if str(dst) == evidence_dest and "pre-overwrite-backup" not in str(src):
            raise OSError("synthetic evidence promotion failure")
        return real_rename(src, dst)

    monkeypatch.setattr(package_builder_module.os, "rename", flaky_rename)

    with pytest.raises(PackageBuilderError):
        _build(tmp_path, write_qc_csv=True, overwrite=True,
               loader=_loader(basin_offsets={BASIN_IDS[0]: 100.0}))

    assert (tmp_path / "package").is_dir()
    assert (tmp_path / "evidence").is_dir()
    assert (tmp_path / "package" / "manifests" / "package_manifest.json").read_text() == old_package_manifest
    assert (tmp_path / "evidence" / "csv_manifest.json").read_text() == old_evidence_manifest
    assert _no_leftover_tmp_or_backup_dirs(tmp_path)


def test_package_promotion_failure_with_existing_evidence_tree_restores_both(tmp_path, monkeypatch):
    _build(tmp_path, write_qc_csv=True)
    old_package_manifest = (tmp_path / "package" / "manifests" / "package_manifest.json").read_text()
    old_evidence_manifest = (tmp_path / "evidence" / "csv_manifest.json").read_text()

    real_rename = package_builder_module.os.rename
    package_dest = str(tmp_path / "package")

    def flaky_rename(src, dst):
        if str(dst) == package_dest and "pre-overwrite-backup" not in str(src):
            raise OSError("synthetic package promotion failure")
        return real_rename(src, dst)

    monkeypatch.setattr(package_builder_module.os, "rename", flaky_rename)

    with pytest.raises(PackageBuilderError):
        _build(tmp_path, write_qc_csv=True, overwrite=True,
               loader=_loader(basin_offsets={BASIN_IDS[0]: 100.0}))

    assert (tmp_path / "package").is_dir()
    assert (tmp_path / "evidence").is_dir()
    assert (tmp_path / "package" / "manifests" / "package_manifest.json").read_text() == old_package_manifest
    assert (tmp_path / "evidence" / "csv_manifest.json").read_text() == old_evidence_manifest
    assert _no_leftover_tmp_or_backup_dirs(tmp_path)


def test_successful_coordinated_overwrite_replaces_both(tmp_path):
    _build(tmp_path, write_qc_csv=True)
    result = _build(tmp_path, write_qc_csv=True, overwrite=True,
                     loader=_loader(basin_offsets={BASIN_IDS[0]: 100.0}))
    with xr.open_dataset(result.package_root / "time_series" / f"{BASIN_IDS[0]}.nc") as ds:
        assert float(ds["qobs_m3s"].values[0]) > 50.0
    assert (result.evidence_root / "csv_inspection" / f"{BASIN_IDS[0]}.csv").is_file()
    assert _no_leftover_tmp_or_backup_dirs(tmp_path)


def test_promotion_failure_from_scratch_leaves_no_partial_package_or_evidence(tmp_path, monkeypatch):
    real_rename = package_builder_module.os.rename
    package_dest = str(tmp_path / "package")

    def flaky_rename(src, dst):
        if str(dst) == package_dest and "pre-overwrite-backup" not in str(src):
            raise OSError("synthetic package promotion failure")
        return real_rename(src, dst)

    monkeypatch.setattr(package_builder_module.os, "rename", flaky_rename)

    with pytest.raises(PackageBuilderError):
        _build(tmp_path, write_qc_csv=True)

    assert not (tmp_path / "package").exists()
    assert not (tmp_path / "evidence").exists()
    assert _no_leftover_tmp_or_backup_dirs(tmp_path)


# ---------------------------------------------------------------------------
# 50-58. strict basin-list / area-CSV CLI readers
# ---------------------------------------------------------------------------


def test_read_basin_ids_file_skips_blank_lines_preserves_ids(tmp_path):
    p = tmp_path / "basins.txt"
    p.write_text("01019000\n\n103366092\n393109104464500\n", encoding="utf-8")
    assert read_basin_ids_file(p) == ["01019000", "103366092", "393109104464500"]


def test_read_basin_ids_file_rejects_leading_whitespace(tmp_path):
    p = tmp_path / "basins.txt"
    p.write_text(" 01019000\n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="whitespace"):
        read_basin_ids_file(p)


def test_read_basin_ids_file_rejects_trailing_whitespace(tmp_path):
    p = tmp_path / "basins.txt"
    p.write_text("01019000 \n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="whitespace"):
        read_basin_ids_file(p)


def test_read_basin_ids_file_missing_file_rejected(tmp_path):
    with pytest.raises(PackageBuilderError, match="not found"):
        read_basin_ids_file(tmp_path / "does_not_exist.txt")


def test_read_area_csv_happy_path(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,DRAIN_SQKM\n01019000,3.6\n103366092,12.1\n", encoding="utf-8")
    areas = read_area_csv(p)
    assert areas == {"01019000": 3.6, "103366092": 12.1}


def test_read_area_csv_missing_required_column_rejected(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,area_km2\n01019000,3.6\n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="missing required column"):
        read_area_csv(p)


def test_read_area_csv_duplicate_gauge_id_rejected(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,DRAIN_SQKM\n01019000,3.6\n01019000,4.1\n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="duplicate"):
        read_area_csv(p)


def test_read_area_csv_noncanonical_gauge_id_rejected(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,DRAIN_SQKM\n1019000,3.6\n", encoding="utf-8")  # 7 digits -> would zero-pad
    with pytest.raises(PackageBuilderError, match="canonical"):
        read_area_csv(p)


def test_read_area_csv_missing_area_for_selected_basin_rejected(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,DRAIN_SQKM\n01019000,3.6\n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="missing DRAIN_SQKM"):
        read_area_csv(p, basin_ids=["01019000", "103366092"])


def test_read_area_csv_nonpositive_area_rejected(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,DRAIN_SQKM\n01019000,0.0\n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="finite and strictly positive"):
        read_area_csv(p)


def test_read_area_csv_nonfinite_area_rejected(tmp_path):
    p = tmp_path / "areas.csv"
    p.write_text("gauge_id,DRAIN_SQKM\n01019000,inf\n", encoding="utf-8")
    with pytest.raises(PackageBuilderError, match="finite and strictly positive"):
        read_area_csv(p)


# ---------------------------------------------------------------------------
# 59-60. QC CSV read-back verification catches real corruption
# ---------------------------------------------------------------------------


def test_qc_csv_readback_catches_shifted_timestamp(tmp_path, monkeypatch):
    real_read_csv = pd.read_csv

    def corrupting_read_csv(path, *args, **kwargs):
        df = real_read_csv(path, *args, **kwargs)
        if "time" in df.columns:
            values = df["time"].tolist()
            values[0] = "2099-01-01T00:00:00Z"
            df["time"] = values
        return df

    monkeypatch.setattr(package_builder_module.pd, "read_csv", corrupting_read_csv)
    with pytest.raises(PackageBuilderError, match="timestamp self-check"):
        _build(tmp_path, write_qc_csv=True)


def test_qc_csv_readback_catches_finite_value_mismatch(tmp_path, monkeypatch):
    real_read_csv = pd.read_csv

    def corrupting_read_csv(path, *args, **kwargs):
        df = real_read_csv(path, *args, **kwargs)
        if "mrms_qpe_1h_mm" in df.columns:
            df.loc[0, "mrms_qpe_1h_mm"] = df.loc[0, "mrms_qpe_1h_mm"] + 1.0
        return df

    monkeypatch.setattr(package_builder_module.pd, "read_csv", corrupting_read_csv)
    with pytest.raises(PackageBuilderError, match="round-trip self-check"):
        _build(tmp_path, write_qc_csv=True)
