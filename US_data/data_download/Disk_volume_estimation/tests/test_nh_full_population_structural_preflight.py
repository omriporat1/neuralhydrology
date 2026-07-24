"""Tests for the full-population (development + spatial-holdout) structural
preflight additions to src.baseline.nh_structural_preflight (local
implementation increment, no training, no h2o/Moriah access).

Two groups, mirroring test_nh_structural_preflight.py's own layering:

- Layer 1 (file-only): check_generated_config_structure's new
  expected_train/validation/test_basin_count / require_identical_basin_sets /
  require_test_disjoint_from_train_validation /
  expect_generated_basins_equal_package_manifest parameters, exercised via
  the real generate_stage1_full_population_nh_config_bundles pipeline against
  the real canonical splits (2,307 development / 250 spatial-holdout).
- Layer 2 (real NeuralHydrology, synthetic-fixture only):
  check_flashnh_external_scaler_test_construction and
  run_full_population_structural_preflight, built on
  tests/_nh_synthetic.py::build_synthetic_package with two disjoint basin
  populations (development-like vs. spatial-holdout-like) sharing one
  package, exactly mirroring the real package's basin-superset relationship.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _nh_synthetic import N_HOURS, build_synthetic_package  # noqa: E402
from test_nh_config_generation import (  # noqa: E402
    POLICY_PATH,
    REAL_DYNAMIC_INPUTS,
    SPLITS_DIR,
    STATIC_COUNT,
    _build_fake_package,
)
from test_nh_full_population_config_generation import (  # noqa: E402
    REAL_DEVELOPMENT,
    REAL_FULL_UNION,
    REAL_SPATIAL_HOLDOUT,
)

from src.baseline.nh_config_generation import (  # noqa: E402
    EXPECTED_DEVELOPMENT_BASIN_COUNT,
    EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT,
    generate_stage1_full_population_nh_config_bundles,
    write_generated_config,
)
from src.baseline.nh_register import register_flashnh_dataset  # noqa: E402
from src.baseline.nh_structural_preflight import (  # noqa: E402
    check_flashnh_dataset_construction,
    check_flashnh_external_scaler_test_construction,
    check_generated_config_structure,
    run_full_population_structural_preflight,
)
from src.baseline.package_audit import AuditReport  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DATES = {
    "train_start_date": "14/10/2020",
    "train_end_date": "31/12/2023",
    "validation_start_date": "01/01/2024",
    "validation_end_date": "31/12/2024",
    "test_start_date": "01/01/2025",
    "test_end_date": "31/12/2025",
}


# ---------------------------------------------------------------------------
# Layer 1: file-only structural checks, real full-population bundle pair
# ---------------------------------------------------------------------------

def _touch_dummy_netcdfs(package_root: Path, basins) -> None:
    """check_generated_config_structure's basin_netcdf_present check only
    tests file existence (src.baseline.nh_structural_preflight._find_basin_netcdf
    calls Path.is_file(), never opens the file), so an empty placeholder is
    sufficient and far cheaper than a real per-basin xarray NetCDF write at
    the real 2,557-basin scale used by these Layer-1 tests."""
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    for basin in basins:
        (ts_dir / f"{basin}.nc").touch()
    masks_dir = package_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (masks_dir / "gap_timestamps.json").write_text("[]", encoding="utf-8")


def _build_full_population_bundle_dirs(tmp_path):
    package_root = _build_fake_package(tmp_path / "package", REAL_FULL_UNION)
    _touch_dummy_netcdfs(package_root, REAL_FULL_UNION)
    bundles = generate_stage1_full_population_nh_config_bundles(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    dev_dir = tmp_path / "generated" / "development"
    holdout_dir = tmp_path / "generated" / "spatial_holdout"
    write_generated_config(bundles.development, dev_dir)
    write_generated_config(bundles.spatial_holdout, holdout_dir)
    return package_root, dev_dir, holdout_dir


def _run_layer1_full_population(package_root, generated_dir, *, is_holdout: bool, **overrides):
    report = AuditReport()
    if is_holdout:
        kwargs = dict(
            generated_dir=generated_dir,
            package_root=package_root,
            expected_basin_count=EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT,
            expected_train_basin_count=EXPECTED_DEVELOPMENT_BASIN_COUNT,
            expected_validation_basin_count=EXPECTED_DEVELOPMENT_BASIN_COUNT,
            expected_test_basin_count=EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT,
            require_identical_basin_sets=False,
            require_test_disjoint_from_train_validation=True,
            expect_generated_basins_equal_package_manifest=False,
        )
    else:
        kwargs = dict(
            generated_dir=generated_dir,
            package_root=package_root,
            expected_basin_count=EXPECTED_DEVELOPMENT_BASIN_COUNT,
            expect_generated_basins_equal_package_manifest=False,
        )
    kwargs.update(
        expected_seq_length=24,
        expected_target_variable="qobs_mm_per_h_lead06",
        expected_dynamic_inputs=REAL_DYNAMIC_INPUTS,
        expected_static_column_count=STATIC_COUNT,
        expected_predict_last_n=1,
        expected_dates=EXPECTED_DATES,
        repo_root=REPO_ROOT,
    )
    kwargs.update(overrides)
    check_generated_config_structure(report, **kwargs)
    return report


def test_development_bundle_structural_checks_pass(tmp_path):
    package_root, dev_dir, _holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    report = _run_layer1_full_population(package_root, dev_dir, is_holdout=False)
    assert report.error_count == 0, report.failed_messages()
    ok_ids = {r.check_id for r in report.records if r.severity == "OK"}
    assert "basin_membership_identical_across_periods" in ok_ids
    assert "basin_ids_subset_of_package_manifest" in ok_ids


def test_spatial_holdout_bundle_structural_checks_pass(tmp_path):
    package_root, _dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    report = _run_layer1_full_population(package_root, holdout_dir, is_holdout=True)
    assert report.error_count == 0, report.failed_messages()
    ok_ids = {r.check_id for r in report.records if r.severity == "OK"}
    assert "basin_membership_train_validation_identical" in ok_ids
    assert "basin_membership_test_disjoint_from_train_validation" in ok_ids
    assert "basin_ids_subset_of_package_manifest" in ok_ids
    assert "basin_membership_identical_across_periods" not in ok_ids


def test_spatial_holdout_bundle_detects_holdout_basin_leaking_into_train(tmp_path):
    package_root, _dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    # Tamper: put one spatial-holdout basin into the holdout bundle's own
    # train_basins.txt -- this must be caught by the disjointness check even
    # though train == validation still holds trivially (both now tampered
    # identically would not trigger it, so tamper train only).
    leaked = REAL_SPATIAL_HOLDOUT[0]
    train_path = holdout_dir / "train_basins.txt"
    original = train_path.read_text(encoding="utf-8").split()
    tampered = [leaked] + original[1:]  # same count, one dev basin swapped for a holdout basin
    train_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")

    report = _run_layer1_full_population(package_root, holdout_dir, is_holdout=True)
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "basin_membership_test_disjoint_from_train_validation" in errored


def test_spatial_holdout_bundle_detects_train_validation_mismatch(tmp_path):
    package_root, _dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    val_path = holdout_dir / "validation_basins.txt"
    original = val_path.read_text(encoding="utf-8").split()
    tampered = list(reversed(original))
    tampered[0] = "99999999"
    val_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")

    report = _run_layer1_full_population(package_root, holdout_dir, is_holdout=True)
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "basin_membership_train_validation_identical" in errored


def test_spatial_holdout_bundle_detects_development_basin_leaking_into_test(tmp_path):
    # Mirror of test_spatial_holdout_bundle_detects_holdout_basin_leaking_into_train:
    # insert a DEVELOPMENT basin into the holdout bundle's own test_basins.txt
    # (swapping out one real holdout basin to preserve the exact expected
    # count) -- must be caught by the disjointness check.
    package_root, _dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    leaked_dev_basin = REAL_DEVELOPMENT[0]
    test_path = holdout_dir / "test_basins.txt"
    original = test_path.read_text(encoding="utf-8").split()
    tampered = [leaked_dev_basin] + original[1:]
    test_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")

    report = _run_layer1_full_population(package_root, holdout_dir, is_holdout=True)
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "basin_membership_test_disjoint_from_train_validation" in errored


def test_spatial_holdout_bundle_detects_test_only_marker_file_missing(tmp_path):
    package_root, _dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    (holdout_dir / "TEST_ONLY_DO_NOT_TRAIN.txt").unlink()

    report = _run_layer1_full_population(package_root, holdout_dir, is_holdout=True)
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "test_only_marker_file_present" in errored


def test_development_bundle_does_not_require_test_only_marker_file(tmp_path):
    package_root, dev_dir, _holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    assert not (dev_dir / "TEST_ONLY_DO_NOT_TRAIN.txt").exists()

    report = _run_layer1_full_population(package_root, dev_dir, is_holdout=False)
    assert report.error_count == 0, report.failed_messages()
    checked_ids = {r.check_id for r in report.records}
    assert "test_only_marker_file_present" not in checked_ids


def test_spatial_holdout_bundle_detects_dynamic_inputs_order_drift(tmp_path):
    # Simulates post-generation tampering that would make the holdout
    # bundle's dynamic_inputs order diverge from the development bundle's --
    # both bundles are checked against the same expected_dynamic_inputs by
    # run_full_population_structural_preflight's single shared caller, so any
    # such drift in either bundle must be caught.
    package_root, _dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    cfg_path = holdout_dir / "config.yaml"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw_cfg["dynamic_inputs"] = list(reversed(raw_cfg["dynamic_inputs"]))
    cfg_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")

    report = _run_layer1_full_population(package_root, holdout_dir, is_holdout=True)
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "dynamic_inputs_exact_order" in errored


def test_development_bundle_wrong_basin_count_detected(tmp_path):
    package_root, dev_dir, _holdout_dir = _build_full_population_bundle_dirs(tmp_path)
    report = _run_layer1_full_population(
        package_root, dev_dir, is_holdout=False, expected_basin_count=EXPECTED_DEVELOPMENT_BASIN_COUNT - 1
    )
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert any(cid.startswith("basin_membership_count") for cid in errored)


# ---------------------------------------------------------------------------
# Layer 2: real FlashNHDataset construction, synthetic fixture, two disjoint
# basin populations sharing one package (mirrors the real full-population
# package's dev+holdout superset relationship at a tiny scale).
# ---------------------------------------------------------------------------

DEV_BASINS = ["DEVA", "DEVB"]
HOLDOUT_BASINS = ["HOLDA", "HOLDB"]
ALL_SYNTHETIC_BASINS = DEV_BASINS + HOLDOUT_BASINS
SEQ_LENGTH = 6
LEAD_HOURS = 3


def _make_dynamic_values(basins):
    import numpy as np

    precip = {b: np.arange(N_HOURS, dtype=np.float64) for b in basins}
    return {"precip": precip}


def _config_yaml_text(*, basins_dir_tag, train_ids, validation_ids, test_ids, package_root: Path, out_dir: Path):
    def _write_basin_file(name, ids):
        p = out_dir / name
        p.write_text("\n".join(ids) + "\n", encoding="utf-8")
        return p

    train_p = _write_basin_file("train_basins.txt", train_ids)
    val_p = _write_basin_file("validation_basins.txt", validation_ids)
    test_p = _write_basin_file("test_basins.txt", test_ids)

    return f"""
experiment_name: flashnh_full_population_test_{basins_dir_tag}

train_basin_file: {train_p}
validation_basin_file: {val_p}
test_basin_file: {test_p}

run_dir: {out_dir}/runs
data_dir: {package_root}
dataset: flashnh

train_start_date: "01/01/2000"
train_end_date: "02/01/2000"
validation_start_date: "03/01/2000"
validation_end_date: "03/01/2000"
test_start_date: "04/01/2000"
test_end_date: "04/01/2000"

target_variables:
  - qobs_lead{LEAD_HOURS}

model: cudalstm
hidden_size: 8
head: regression
output_activation: linear
predict_last_n: 1
batch_size: 8

optimizer: Adam
learning_rate: 0.001
loss: MSE

save_weights_every: 1
validate_every: 1
validate_n_random_basins: 1
log_interval: 50
num_workers: 0
seq_length: {SEQ_LENGTH}
epochs: 1
device: cpu
verbose: 0

dynamic_inputs:
  - precip
"""


def _build_synthetic_dual_population_dirs(tmp_path):
    """One shared synthetic package with 4 basins (2 development-like, 2
    spatial-holdout-like) plus two hand-written config bundles pointed at it:
    a development bundle (train==validation==test==DEV_BASINS) and a
    spatial-holdout bundle (train==validation==DEV_BASINS,
    test==HOLDOUT_BASINS) -- mirroring
    generate_stage1_full_population_nh_config_bundles's real contract without
    needing the heavyweight real 2,557-basin fixture for a dataset-
    construction-level test."""
    package_root = tmp_path / "package"
    build_synthetic_package(
        package_root,
        basins=ALL_SYNTHETIC_BASINS,
        seq_length=SEQ_LENGTH,
        lead_hours=LEAD_HOURS,
        dynamic_input_values=_make_dynamic_values(ALL_SYNTHETIC_BASINS),
        declared_gap_hours=[],
    )

    dev_dir = tmp_path / "generated" / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)
    (dev_dir / "config.yaml").write_text(
        _config_yaml_text(
            basins_dir_tag="development", train_ids=DEV_BASINS, validation_ids=DEV_BASINS,
            test_ids=DEV_BASINS, package_root=package_root, out_dir=dev_dir,
        ),
        encoding="utf-8",
    )

    holdout_dir = tmp_path / "generated" / "spatial_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    (holdout_dir / "config.yaml").write_text(
        _config_yaml_text(
            basins_dir_tag="spatial_holdout", train_ids=DEV_BASINS, validation_ids=DEV_BASINS,
            test_ids=HOLDOUT_BASINS, package_root=package_root, out_dir=holdout_dir,
        ),
        encoding="utf-8",
    )
    return package_root, dev_dir, holdout_dir


def test_check_flashnh_external_scaler_test_construction_reuses_development_scaler(tmp_path):
    _package_root, dev_dir, holdout_dir = _build_synthetic_dual_population_dirs(tmp_path)

    report = AuditReport()
    dev_datasets = check_flashnh_dataset_construction(report, dev_dir / "config.yaml")
    assert report.error_count == 0, report.failed_messages()
    train_ds = dev_datasets["train"]

    holdout_datasets = check_flashnh_external_scaler_test_construction(
        report, holdout_dir / "config.yaml", external_scaler=train_ds.scaler
    )
    assert report.error_count == 0, report.failed_messages()
    ok_ids = {r.check_id for r in report.records if r.severity == "OK"}
    assert "holdout_scaler_reused_unchanged" in ok_ids
    assert "holdout_test_dataset_construction" in ok_ids
    assert "holdout_finite_admitted_batches" in ok_ids
    assert type(holdout_datasets["test"]).__name__ == "FlashNHDataset"


def test_check_flashnh_external_scaler_test_construction_rejects_empty_scaler(tmp_path):
    _package_root, _dev_dir, holdout_dir = _build_synthetic_dual_population_dirs(tmp_path)
    report = AuditReport()
    check_flashnh_external_scaler_test_construction(report, holdout_dir / "config.yaml", external_scaler={})
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "holdout_test_dataset_construction" in errored


def test_check_flashnh_external_scaler_test_construction_rejects_malformed_scaler(tmp_path):
    # Non-empty but structurally wrong (missing the real xarray/pandas scaler
    # keys the dataset pipeline expects) -- must not be silently accepted as
    # "a scaler was provided" just because the dict is non-empty.
    _package_root, _dev_dir, holdout_dir = _build_synthetic_dual_population_dirs(tmp_path)
    report = AuditReport()
    malformed_scaler = {"not_a_real_scaler_key": "garbage"}
    check_flashnh_external_scaler_test_construction(
        report, holdout_dir / "config.yaml", external_scaler=malformed_scaler
    )
    assert report.error_count > 0, "a malformed external scaler must not pass silently"


def test_check_flashnh_external_scaler_test_construction_never_touches_train_dir(tmp_path):
    """The holdout bundle's own run_dir/train_data must never be created by a
    test-only construction -- this is the mechanical guarantee (confirmed via
    NH 1.13 BaseDataset.__init__ source) that spatial-holdout evaluation can
    never influence normalization or be mistaken for a trainable config."""
    _package_root, dev_dir, holdout_dir = _build_synthetic_dual_population_dirs(tmp_path)
    report = AuditReport()
    dev_datasets = check_flashnh_dataset_construction(report, dev_dir / "config.yaml")
    train_ds = dev_datasets["train"]

    check_flashnh_external_scaler_test_construction(
        report, holdout_dir / "config.yaml", external_scaler=train_ds.scaler
    )
    assert not (holdout_dir / "runs" / "train_data").exists()


def test_run_full_population_structural_preflight_passes_on_synthetic_dual_population(tmp_path):
    _package_root, dev_dir, holdout_dir = _build_synthetic_dual_population_dirs(tmp_path)
    # run_full_population_structural_preflight's Layer 1 uses the pinned real
    # constants (2,307/250), which the tiny synthetic fixture (2/2) will
    # never match -- this test therefore only exercises Layer 2 by calling
    # the two dataset-construction checks directly through the same
    # composition contract, since a full Layer-1-passing synthetic fixture
    # would require faking the pinned basin counts. See
    # test_development_bundle_structural_checks_pass /
    # test_spatial_holdout_bundle_structural_checks_pass above for the real
    # (2,307/250) Layer-1 contract, and the two tests directly above for the
    # Layer-2 external-scaler contract this orchestrator composes.
    report = AuditReport()
    dev_datasets = check_flashnh_dataset_construction(report, dev_dir / "config.yaml")
    check_flashnh_external_scaler_test_construction(
        report, holdout_dir / "config.yaml", external_scaler=dev_datasets["train"].scaler
    )
    assert report.error_count == 0, report.failed_messages()


def test_run_full_population_structural_preflight_end_to_end_real_splits_structure_only(tmp_path):
    """Layer-1-only (run_dataset_construction=False) end-to-end run of the
    real orchestrator against the real full-population bundle pair (2,307/250
    basins), confirming the composition wiring itself (pinned expected counts,
    role-aware require_identical_basin_sets=False /
    require_test_disjoint_from_train_validation=True /
    expect_generated_basins_equal_package_manifest=False) without requiring a
    real per-basin time-series NetCDF for all 2,557 basins."""
    package_root, dev_dir, holdout_dir = _build_full_population_bundle_dirs(tmp_path)

    report = run_full_population_structural_preflight(
        development_generated_dir=dev_dir,
        spatial_holdout_generated_dir=holdout_dir,
        package_root=package_root,
        expected_seq_length=24,
        expected_target_variable="qobs_mm_per_h_lead06",
        expected_dynamic_inputs=REAL_DYNAMIC_INPUTS,
        expected_static_column_count=STATIC_COUNT,
        expected_predict_last_n=1,
        expected_dates=EXPECTED_DATES,
        repo_root=REPO_ROOT,
        run_dataset_construction=False,
    )
    assert report.error_count == 0, report.failed_messages()
