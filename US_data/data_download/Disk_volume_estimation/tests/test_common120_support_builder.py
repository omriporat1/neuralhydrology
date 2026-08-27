"""Synthetic checks for authoritative, package-bound Common-120 construction."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline import common120_support_builder as builder
from src.baseline.fixed_support_contract_v2 import FixedSupportContractError, load_fixed_support_contract, validate_fixed_support_contract, write_fixed_support_contract
from src.baseline.gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT


IDS = [f"{i:08d}" for i in range(400)]
IDENTITIES = {
    "package_manifest_sha256": "a" * 64,
    "package_file_checksums_sha256": "b" * 64,
    "package_run_provenance_sha256": "c" * 64,
    "development_split_sha256": "d" * 64,
    "spatial_holdout_split_sha256": "e" * 64,
}


def _install_qualified_fixture(monkeypatch, tmp_path, *, gaps=(), qobs=None):
    package = tmp_path / "package"; (package / "masks").mkdir(parents=True)
    gap_path = package / "masks" / "gap_timestamps.json"
    gap_path.write_text(json.dumps([pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ") for x in gaps]))
    dates = pd.date_range("2023-12-27", periods=170, freq="h")
    values = np.ones(len(dates)) if qobs is None else np.asarray(qobs, dtype=float)
    manifest = {"gap_product_scope": [MRMS_PRODUCT, RTMA_PRODUCT],
                "gap_timestamp_artifact": {"sha256": hashlib.sha256(gap_path.read_bytes()).hexdigest()}}
    policy = {"temporal_split": {"validation": {"start": "2024-01-01", "end": "2024-01-01"}},
              "gap_policy": {"include_rtma_in_history_mask": True}}
    monkeypatch.setattr(builder, "load_stage1_baseline_policy", lambda _: policy)
    monkeypatch.setattr(builder, "load_stage1_baseline_policy_v2_six_axis", lambda *_: policy)
    monkeypatch.setattr(builder, "_verify_artifact_identities", lambda _: IDENTITIES)
    monkeypatch.setattr(builder, "read_package_manifest", lambda _: manifest)
    monkeypatch.setattr(builder, "validate_full_population_basin_membership", lambda *_: type("M", (), {"development_basins": IDS})())
    monkeypatch.setattr(builder, "load_screening_basin_ids", lambda *_, **__: IDS)
    monkeypatch.setattr(builder, "_dates_and_target", lambda *_: (dates, values.copy()))
    return package, dates, manifest


def test_builder_uses_full_packaged_mask_and_stores_issue_times(monkeypatch, tmp_path):
    package, dates, _ = _install_qualified_fixture(monkeypatch, tmp_path, gaps=["2023-12-27T01:00:00Z"])
    result = builder.build_common120_support(package_root=package, splits_dir=tmp_path, screening_basin_ids_path=tmp_path / "s", baseline_policy_path=tmp_path / "p", policy_overlay_path=tmp_path / "o")
    supported = pd.to_datetime(result.contract["per_basin_support"][IDS[0]])
    assert len(result.contract["basin_ids"]) == 400
    assert result.contract["package_manifest_sha256"] == IDENTITIES["package_manifest_sha256"]
    assert (pd.Timestamp("2024-01-01T00:00:00") not in supported)  # RTMA/MRMS packaged gap is inside its 120h history
    assert all(x <= pd.Timestamp("2024-01-01T17:00:00") for x in supported)  # validation lead-6 boundary


def test_builder_rejects_mrms_only_manifest_substitution(monkeypatch, tmp_path):
    package, _, manifest = _install_qualified_fixture(monkeypatch, tmp_path)
    manifest["gap_product_scope"] = [MRMS_PRODUCT]
    with pytest.raises(builder.Common120SupportError, match="MRMS\\+RTMA"):
        builder.build_common120_support(package_root=package, splits_dir=tmp_path, screening_basin_ids_path="s", baseline_policy_path="p", policy_overlay_path="o")


def test_builder_rejects_package_gap_content_contradiction(monkeypatch, tmp_path):
    package, _, _ = _install_qualified_fixture(monkeypatch, tmp_path, gaps=["2023-12-27T01:00:00Z"])
    (package / "masks" / "gap_timestamps.json").write_text("[]")
    with pytest.raises(builder.Common120SupportError, match="checksum"):
        builder.build_common120_support(package_root=package, splits_dir=tmp_path, screening_basin_ids_path="s", baseline_policy_path="p", policy_overlay_path="o")


def test_date_target_reader_requires_aligned_hourly_data(tmp_path):
    path = tmp_path / "one.nc"; dates = pd.date_range("2024-01-01", periods=3, freq="h")
    xr.Dataset({"qobs_mm_per_h_lead06": ("date", [1., np.nan, 3.])}, coords={"date": dates}).to_netcdf(path)
    actual_dates, values = builder._dates_and_target(path, "qobs_mm_per_h_lead06")
    assert actual_dates.equals(dates) and np.isnan(values[1])


def test_schema_two_package_hashes_are_strict_and_roundtrip(tmp_path):
    mp = pytest.MonkeyPatch(); _install_qualified_fixture(mp, tmp_path / "b")
    result = builder.build_common120_support(package_root=tmp_path / "b" / "package", splits_dir=tmp_path, screening_basin_ids_path="s", baseline_policy_path="p", policy_overlay_path="o")
    path = write_fixed_support_contract(result.contract, tmp_path / "support.json")
    assert load_fixed_support_contract(path)["schema_version"] == 2
    altered = dict(result.contract); altered["package_manifest_sha256"] = "f" * 64
    with pytest.raises(FixedSupportContractError, match="checksum"):
        validate_fixed_support_contract(altered)
    mp.undo()
