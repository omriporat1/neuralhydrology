"""Tests for src.baseline.nh_config_generation (local implementation increment).

Uses the real, committed Stage 1 scientific baseline policy
(config/stage1_scientific_baseline_v001.yaml) and the real canonical split
files (config/stage1_baseline_splits_v001/) -- both already accepted
artifacts in this repository -- paired with a lightweight fake package
fixture (manifests/package_manifest.json + attributes/attributes.csv only;
no NetCDF time-series, which the structural-preflight tests cover
separately). No h2o/Moriah access, no training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baseline.nh_config_generation import (
    NHConfigGenerationError,
    generate_stage1_nh_config,
    read_package_attribute_columns,
    validate_basin_membership,
    validate_dynamic_inputs,
    validate_lead_hours,
    validate_seq_length,
    validate_static_attribute_contract,
    validate_target_variables,
    write_generated_config,
)
from src.baseline.policy import load_stage1_baseline_policy
from src.baseline.splits import load_eligible_basins, sha256_of

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"
SPLITS_DIR = REPO_ROOT / "config" / "stage1_baseline_splits_v001"

POLICY = load_stage1_baseline_policy(POLICY_PATH)
REAL_DYNAMIC_INPUTS = list(POLICY["dynamic_inputs"])
STATIC_COUNT = POLICY["static_attributes"]["expected_model_input_columns"]


def _pick_basins(n: int = 32) -> list:
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    assert len(dev) >= n
    return dev[:n]


def _static_columns(n: int = STATIC_COUNT) -> list:
    return [f"col_{i:04d}" for i in range(n)]


def _build_fake_package(
    root: Path,
    basin_ids,
    *,
    static_columns=None,
    dynamic_variables=None,
    attrs_columns=None,
    extra_manifest_fields=None,
) -> Path:
    static_columns = list(static_columns if static_columns is not None else _static_columns())
    dynamic_variables = list(dynamic_variables if dynamic_variables is not None else REAL_DYNAMIC_INPUTS)
    attrs_columns = list(attrs_columns if attrs_columns is not None else static_columns)

    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    columns_sha256 = hashlib.sha256("\n".join(static_columns).encode("utf-8")).hexdigest()
    manifest = {
        "schema_name": "stage1_compact_package_manifest",
        "schema_version": 1,
        "package_role": "compact_scientific_package",
        "basin_count": len(basin_ids),
        "basin_ids": list(basin_ids),
        "dynamic_variables": dynamic_variables,
        "static_model_input_columns": static_columns,
        "static_model_input_columns_sha256": columns_sha256,
    }
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    (manifests_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attrs_dir = root / "attributes"
    attrs_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"gauge_id": b, **{c: 0.0 for c in attrs_columns}} for b in basin_ids]
    df = pd.DataFrame(rows, columns=["gauge_id"] + attrs_columns)
    df.to_csv(attrs_dir / "attributes.csv", index=False)
    return root


# ---------------------------------------------------------------------------
# 1. exact lead06/seq24 generation, end to end
# ---------------------------------------------------------------------------

def test_generate_lead06_seq24_end_to_end(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    assert bundle.target_variable == "qobs_mm_per_h_lead06"
    assert bundle.seq_length == 24
    assert bundle.dynamic_inputs == REAL_DYNAMIC_INPUTS
    assert bundle.static_attribute_result.count == STATIC_COUNT
    assert set(bundle.basin_ids) == set(basins)
    assert len(bundle.basin_ids) == 32

    out_dir = tmp_path / "out"
    written = write_generated_config(bundle, out_dir, experiment_name="test_exp")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))

    assert cfg["dataset"] == "flashnh"
    assert cfg["seq_length"] == 24
    assert cfg["target_variables"] == ["qobs_mm_per_h_lead06"]
    assert cfg["dynamic_inputs"] == REAL_DYNAMIC_INPUTS
    assert len(cfg["static_attributes"]) == STATIC_COUNT
    assert cfg["predict_last_n"] == 1
    assert "nan_handling_method" not in cfg

    for label, path in (
        ("train", written["train_basins.txt"]),
        ("validation", written["validation_basins.txt"]),
        ("test", written["test_basins.txt"]),
    ):
        lines = path.read_text(encoding="utf-8").split()
        assert set(lines) == set(basins), f"{label} basin list mismatch"


# ---------------------------------------------------------------------------
# 5. exact dates / date format (DD/MM/YYYY per policy nh.date_format)
# ---------------------------------------------------------------------------

def test_generated_config_dates_exact(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    written = write_generated_config(bundle, tmp_path / "out")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))

    assert cfg["train_start_date"] == "14/10/2020"
    assert cfg["train_end_date"] == "31/12/2023"
    assert cfg["validation_start_date"] == "01/01/2024"
    assert cfg["validation_end_date"] == "31/12/2024"
    assert cfg["test_start_date"] == "01/01/2025"
    assert cfg["test_end_date"] == "31/12/2025"


# ---------------------------------------------------------------------------
# 2 & 3. target-variable rejection: multi-target and raw source variable
# ---------------------------------------------------------------------------

def test_validate_target_variables_rejects_multi_target():
    with pytest.raises(NHConfigGenerationError):
        validate_target_variables(["qobs_mm_per_h_lead06", "qobs_mm_per_h_lead12"], POLICY)


def test_validate_target_variables_rejects_raw_source_variable():
    with pytest.raises(NHConfigGenerationError):
        validate_target_variables(["qobs_m3s"], POLICY)


def test_validate_target_variables_accepts_single_lead_shifted_variable():
    validate_target_variables(["qobs_mm_per_h_lead06"], POLICY)  # must not raise


# ---------------------------------------------------------------------------
# 4. invalid lead / seq_length rejection
# ---------------------------------------------------------------------------

def test_validate_seq_length_rejects_forbidden_and_unknown_values():
    with pytest.raises(NHConfigGenerationError):
        validate_seq_length(168, POLICY)  # policy-forbidden (Stage 2)
    with pytest.raises(NHConfigGenerationError):
        validate_seq_length(36, POLICY)  # not in the approved set at all


def test_validate_lead_hours_rejects_unapproved_value():
    with pytest.raises(NHConfigGenerationError):
        validate_lead_hours(9, POLICY)


def test_generate_end_to_end_rejects_invalid_seq_length(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    with pytest.raises(NHConfigGenerationError):
        generate_stage1_nh_config(
            policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
            lead_hours=6, seq_length=168,
        )


# ---------------------------------------------------------------------------
# 6. exact dynamic-input order
# ---------------------------------------------------------------------------

def test_validate_dynamic_inputs_accepts_exact_match():
    validate_dynamic_inputs(list(REAL_DYNAMIC_INPUTS), POLICY)  # must not raise


def test_validate_dynamic_inputs_rejects_reordering():
    with pytest.raises(NHConfigGenerationError):
        validate_dynamic_inputs(list(reversed(REAL_DYNAMIC_INPUTS)), POLICY)


def test_validate_dynamic_inputs_rejects_subset():
    with pytest.raises(NHConfigGenerationError):
        validate_dynamic_inputs(REAL_DYNAMIC_INPUTS[:-1], POLICY)


# ---------------------------------------------------------------------------
# 7. exact 473-column static equality (count mismatch rejected)
# ---------------------------------------------------------------------------

def test_static_attribute_contract_accepts_exact_expected_count(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    _build_fake_package(package_root, basins)
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    result = validate_static_attribute_contract(POLICY, manifest, attrs_cols)
    assert result.count == STATIC_COUNT


def test_static_attribute_contract_rejects_wrong_count(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    cols = _static_columns(STATIC_COUNT - 1)
    _build_fake_package(package_root, basins, static_columns=cols)
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    with pytest.raises(NHConfigGenerationError):
        validate_static_attribute_contract(POLICY, manifest, attrs_cols)


# ---------------------------------------------------------------------------
# 8. static order-mismatch rejection
# ---------------------------------------------------------------------------

def test_static_attribute_contract_rejects_order_mismatch(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    cols = _static_columns(10)
    _build_fake_package(package_root, basins, static_columns=cols, attrs_columns=list(reversed(cols)))
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    with pytest.raises(NHConfigGenerationError):
        validate_static_attribute_contract(POLICY, manifest, attrs_cols)


# ---------------------------------------------------------------------------
# 9. forbidden-static-field rejection
# ---------------------------------------------------------------------------

def test_static_attribute_contract_rejects_forbidden_field(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    cols = _static_columns(STATIC_COUNT - 1) + ["STATE"]
    _build_fake_package(package_root, basins, static_columns=cols)
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    with pytest.raises(NHConfigGenerationError):
        validate_static_attribute_contract(POLICY, manifest, attrs_cols)


# ---------------------------------------------------------------------------
# 10. 8/9/15-char STAID handling in basin-membership validation
# ---------------------------------------------------------------------------

def test_validate_basin_membership_zero_pads_short_ids():
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    candidates = [b for b in dev if b.startswith("0")]
    assert candidates, "expected at least one zero-padded development_train basin id"
    basin = candidates[0]
    short_form = basin.lstrip("0")
    assert len(short_form) < 8
    result = validate_basin_membership({"basin_ids": [short_form]}, SPLITS_DIR)
    assert result == [basin]


def test_validate_basin_membership_preserves_15_char_ids():
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    long_ids = [b for b in dev if len(b) == 15]
    assert long_ids, "expected at least one 15-char development_train basin id"
    result = validate_basin_membership({"basin_ids": long_ids}, SPLITS_DIR)
    assert set(result) == set(long_ids)


def test_validate_basin_membership_preserves_8_char_ids():
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    eight_char = [b for b in dev if len(b) == 8][:3]
    assert eight_char
    result = validate_basin_membership({"basin_ids": eight_char}, SPLITS_DIR)
    assert set(result) == set(eight_char)


# ---------------------------------------------------------------------------
# 11 & 12. exact compact basin membership + spatial-holdout/California rejection
# ---------------------------------------------------------------------------

def test_validate_basin_membership_rejects_non_development_train_id():
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": ["99999999"]}, SPLITS_DIR)


def test_validate_basin_membership_rejects_spatial_holdout_basin():
    holdout = load_eligible_basins(SPLITS_DIR / "spatial_holdout_nonca.txt")
    assert holdout
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": [holdout[0]]}, SPLITS_DIR)


def test_validate_basin_membership_rejects_california_basin():
    ca = load_eligible_basins(SPLITS_DIR / "california_all.txt")
    assert ca
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": [ca[0]]}, SPLITS_DIR)


def test_validate_basin_membership_rejects_duplicate_ids():
    basins = _pick_basins(2)
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": [basins[0], basins[0]]}, SPLITS_DIR)


# ---------------------------------------------------------------------------
# 13. output-directory safety
# ---------------------------------------------------------------------------

def test_write_generated_config_rejects_nonempty_out_dir_without_force(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "preexisting.txt").write_text("x", encoding="utf-8")

    with pytest.raises(NHConfigGenerationError):
        write_generated_config(bundle, out_dir)

    written = write_generated_config(bundle, out_dir, force=True)
    assert written["config.yaml"].is_file()


# ---------------------------------------------------------------------------
# 14. nan_handling_method must be absent everywhere in the generated output
# ---------------------------------------------------------------------------

def test_generated_config_never_sets_nan_handling_method(tmp_path):
    basins = _pick_basins(4)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    assert "nan_handling_method" not in bundle.config_mapping
    written = write_generated_config(bundle, tmp_path / "out")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))
    assert "nan_handling_method" not in cfg


# ---------------------------------------------------------------------------
# 21. generated manifest checksums / identities
# ---------------------------------------------------------------------------

def test_generation_manifest_checksums_and_identity(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    written = write_generated_config(bundle, tmp_path / "out")
    manifest = json.loads(written["generation_manifest.json"].read_text(encoding="utf-8"))

    for name, path in written.items():
        if name == "generation_manifest.json":
            continue
        assert manifest["artifact_sha256"][name] == sha256_of(path), f"checksum mismatch for {name}"

    package_manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_manifest_identity"]["basin_count"] == package_manifest["basin_count"]
    assert (
        manifest["package_manifest_identity"]["static_model_input_columns_sha256"]
        == package_manifest["static_model_input_columns_sha256"]
    )
    assert manifest["nan_handling_method"] is None
    assert manifest["compact_smoke_run_profile"] is True
    assert manifest["nh_runtime_dataset_key"] == "flashnh"
