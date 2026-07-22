"""Tests for src.baseline.nh_structural_preflight (local implementation
increment, no training).

Two groups, mirroring the module's two independent layers:

- Layer 1 (file-only): a handful of direct tests against
  ``check_generated_config_structure`` using the same lightweight fake
  package + real generate_stage1_nh_config/write_generated_config pipeline
  as tests/test_nh_config_generation.py (imported directly to avoid
  duplicating the fixture builder).
- Layer 2 (real NeuralHydrology, synthetic-fixture only): the six required
  dataset-construction checks (round-trip Config construction, registration
  ordering, training-scaler reuse, finite admitted batches under
  certified-gap conditions, detection of a stray NaN outside the declared
  gap set, and detection of a gap-flag/missingness mismatch), built on
  tests/_nh_synthetic.py::build_synthetic_package's dynamic_input_values /
  declared_gap_hours extensions. Never run against the real certified
  package and never against h2o/Moriah.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import N_HOURS, build_synthetic_package  # noqa: E402
from test_nh_config_generation import (  # noqa: E402
    POLICY_PATH,
    REAL_DYNAMIC_INPUTS,
    SPLITS_DIR,
    STATIC_COUNT,
    _build_fake_package,
    _pick_basins,
)

from src.baseline import nh_structural_preflight as preflight  # noqa: E402
from src.baseline.nh_config_generation import generate_stage1_nh_config, write_generated_config  # noqa: E402
from src.baseline.nh_register import register_flashnh_dataset  # noqa: E402
from src.baseline.nh_structural_preflight import (  # noqa: E402
    check_flashnh_dataset_construction,
    check_generated_config_structure,
    inspect_dynamic_nan_inventory,
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
# Layer 1: file-only structural checks
# ---------------------------------------------------------------------------

def _build_full_bundle(tmp_path, n_basins=4):
    """A real generate_stage1_nh_config/write_generated_config bundle,
    rendered against a fake package that additionally carries a trivial
    time_series/<basin>.nc per basin (content is irrelevant -- Layer 1 only
    checks file existence, never NetCDF content)."""
    basins = _pick_basins(n_basins)
    package_root = tmp_path / "package"
    _build_fake_package(package_root, basins)

    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    for basin in basins:
        xr.Dataset(
            {"dummy": ("date", [0.0])}, coords={"date": [pd.Timestamp("2020-01-01")]}
        ).to_netcdf(ts_dir / f"{basin}.nc")

    masks_dir = package_root / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (masks_dir / "gap_timestamps.json").write_text("[]", encoding="utf-8")

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    generated_dir = tmp_path / "generated"
    write_generated_config(bundle, generated_dir)
    return package_root, generated_dir, basins


def _run_layer1(package_root, generated_dir, n_basins, **overrides):
    report = AuditReport()
    kwargs = dict(
        generated_dir=generated_dir,
        package_root=package_root,
        expected_basin_count=n_basins,
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


def test_check_generated_config_structure_passes_on_a_valid_bundle(tmp_path):
    package_root, generated_dir, basins = _build_full_bundle(tmp_path)
    report = _run_layer1(package_root, generated_dir, len(basins))
    assert report.error_count == 0, report.failed_messages()
    ok_ids = {r.check_id for r in report.records if r.severity == "OK"}
    for expected_id in (
        "dataset_key", "nan_handling_method_absent", "seq_length_exact",
        "predict_last_n_documented", "target_variable_exact", "raw_target_not_configured",
        "dynamic_inputs_exact_order", "static_attribute_count_exact", "dates_exact",
        "basin_membership_identical_across_periods", "basin_netcdf_present",
        "gap_timestamp_artifact_parses", "package_manifest_identity_consistent",
        "package_has_required_dynamic_variables", "output_outside_tracked_paths",
    ):
        assert expected_id in ok_ids, f"expected OK check {expected_id!r} missing: {report.failed_messages()}"


def test_check_generated_config_structure_detects_basin_list_mismatch(tmp_path):
    package_root, generated_dir, basins = _build_full_bundle(tmp_path)
    tampered = ["99999999"] + basins[1:]  # same count, different membership
    (generated_dir / "test_basins.txt").write_text("\n".join(tampered) + "\n", encoding="utf-8")

    report = _run_layer1(package_root, generated_dir, len(basins))
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "basin_membership_identical_across_periods" in errored


def test_check_generated_config_structure_detects_shared_incorrect_basin_list(tmp_path):
    """All three basin files agree with EACH OTHER (so
    basin_membership_identical_across_periods stays OK) but disagree with the
    package manifest's own basin_ids -- only the dedicated
    basin_ids_exact_equality_all_sources check can catch this."""
    package_root, generated_dir, basins = _build_full_bundle(tmp_path)
    tampered = ["99999999"] + basins[1:]  # same count, same wrong membership everywhere
    tampered_text = "\n".join(tampered) + "\n"
    (generated_dir / "train_basins.txt").write_text(tampered_text, encoding="utf-8")
    (generated_dir / "validation_basins.txt").write_text(tampered_text, encoding="utf-8")
    (generated_dir / "test_basins.txt").write_text(tampered_text, encoding="utf-8")

    report = _run_layer1(package_root, generated_dir, len(basins))
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "basin_ids_exact_equality_all_sources" in errored
    assert "basin_membership_identical_across_periods" not in errored


def test_check_generated_config_structure_detects_missing_basin_netcdf(tmp_path):
    package_root, generated_dir, basins = _build_full_bundle(tmp_path)
    (package_root / "time_series" / f"{basins[0]}.nc").unlink()

    report = _run_layer1(package_root, generated_dir, len(basins))
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "basin_netcdf_present" in errored


def test_check_generated_config_structure_detects_forbidden_key(tmp_path):
    package_root, generated_dir, basins = _build_full_bundle(tmp_path)
    config_path = generated_dir / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nslurm_gres: gpu:1\n", encoding="utf-8"
    )

    report = _run_layer1(package_root, generated_dir, len(basins))
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "no_forbidden_keys[config.yaml]" in errored


def test_check_generated_config_structure_detects_nan_handling_method(tmp_path):
    package_root, generated_dir, basins = _build_full_bundle(tmp_path)
    config_path = generated_dir / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "\nnan_handling_method: masked_mean\n", encoding="utf-8"
    )

    report = _run_layer1(package_root, generated_dir, len(basins))
    errored = {r.check_id for r in report.records if r.severity == "ERROR"}
    assert "nan_handling_method_absent" in errored


def test_output_location_safety_rejects_tracked_source_dirs():
    report = AuditReport()
    preflight._check_output_location_safety(report, REPO_ROOT / "config" / "some_generated_run", REPO_ROOT)
    assert report.error_count == 1
    assert report.records[0].check_id == "output_outside_tracked_paths"


def test_output_location_safety_accepts_untracked_dirs(tmp_path):
    report = AuditReport()
    preflight._check_output_location_safety(report, tmp_path / "generated", REPO_ROOT)
    assert report.error_count == 0
    assert report.records[0].severity == "OK"


# ---------------------------------------------------------------------------
# Layer 2: real FlashNHDataset construction against synthetic fixtures
# ---------------------------------------------------------------------------

BASINS = ["SYN01", "SYN02"]
SEQ_LENGTH = 6
LEAD_HOURS = 3
BAD_HOURS = [10, 55, 80]  # one per period: train [0,47], validation [48,71], test [72,95]


def _make_dynamic_values(basins, nan_hours, flag_hours):
    precip = {}
    precip_gap = {}
    for basin in basins:
        arr = np.arange(N_HOURS, dtype=np.float64)
        for h in nan_hours:
            arr[h] = np.nan
        precip[basin] = arr

        flag = np.zeros(N_HOURS, dtype=np.float64)
        for h in flag_hours:
            flag[h] = 1.0
        precip_gap[basin] = flag
    return {"precip": precip, "precip_gap": precip_gap}


def _build_synthetic_cfg(tmp_path, *, nan_hours, flag_hours, declared_gap_hours):
    dynamic_input_values = _make_dynamic_values(BASINS, nan_hours, flag_hours)
    return build_synthetic_package(
        tmp_path,
        basins=BASINS,
        seq_length=SEQ_LENGTH,
        lead_hours=LEAD_HOURS,
        dynamic_input_values=dynamic_input_values,
        declared_gap_hours=declared_gap_hours,
    )


# 15. NH Config round-trip + basic construction
def test_check_flashnh_dataset_construction_config_roundtrip(tmp_path):
    cfg_path = _build_synthetic_cfg(
        tmp_path, nan_hours=BAD_HOURS, flag_hours=BAD_HOURS, declared_gap_hours=BAD_HOURS
    )
    report = AuditReport()
    datasets = check_flashnh_dataset_construction(report, cfg_path)

    assert report.error_count == 0, report.failed_messages()
    assert set(datasets.keys()) == {"train", "validation", "test"}
    assert type(datasets["train"]).__name__ == "FlashNHDataset"


# 16. registration-before-dataset-construction is idempotent / order-independent
def test_registration_is_idempotent_before_construction(tmp_path):
    cfg_path = _build_synthetic_cfg(
        tmp_path, nan_hours=BAD_HOURS, flag_hours=BAD_HOURS, declared_gap_hours=BAD_HOURS
    )
    register_flashnh_dataset()
    register_flashnh_dataset()  # must not raise

    report_a = AuditReport()
    check_flashnh_dataset_construction(report_a, cfg_path, register=True)
    assert report_a.error_count == 0, report_a.failed_messages()

    report_b = AuditReport()
    check_flashnh_dataset_construction(report_b, cfg_path, register=False)
    assert report_b.error_count == 0, report_b.failed_messages()


# 17. training scaler reused unchanged for validation/test
def test_training_scaler_reused_unchanged_for_validation_and_test(tmp_path):
    cfg_path = _build_synthetic_cfg(
        tmp_path, nan_hours=BAD_HOURS, flag_hours=BAD_HOURS, declared_gap_hours=BAD_HOURS
    )
    report = AuditReport()
    check_flashnh_dataset_construction(report, cfg_path)
    ok_ids = {r.check_id for r in report.records if r.severity == "OK"}
    assert "scaler_reused_unchanged" in ok_ids
    assert report.error_count == 0, report.failed_messages()


# 18. finite admitted dynamic tensors when NaNs are confined to declared gaps
def test_finite_admitted_batches_when_nans_confined_to_declared_gaps(tmp_path):
    cfg_path = _build_synthetic_cfg(
        tmp_path, nan_hours=BAD_HOURS, flag_hours=BAD_HOURS, declared_gap_hours=BAD_HOURS
    )
    report = AuditReport()
    check_flashnh_dataset_construction(report, cfg_path)

    for period in ("train", "validation", "test"):
        record = next(r for r in report.records if r.check_id == f"finite_admitted_batches[{period}]")
        assert record.severity == "OK", f"{period}: {record.message}"


# 19. a NaN outside the documented gap set slips into an admitted sample
def test_stray_nan_outside_declared_gap_is_detected(tmp_path):
    # Deliberately placed in the *validation* period (hour 48..71): NH's
    # native lookup-table construction only rejects NaN-containing windows
    # for the train period; validation/test lookup construction does not,
    # which is exactly why FlashNHDataset's gap-based post-filter is load
    # bearing there (accepted finding from the mechanics evidence gate).
    # A train-period stray NaN would be silently excluded by NH itself,
    # masking the very gap this preflight exists to catch.
    stray_hour = 50
    package_root = tmp_path
    cfg_path = _build_synthetic_cfg(
        package_root,
        nan_hours=BAD_HOURS + [stray_hour],
        flag_hours=BAD_HOURS,
        declared_gap_hours=BAD_HOURS,  # stray_hour intentionally NOT declared
    )

    report = AuditReport()
    check_flashnh_dataset_construction(report, cfg_path)
    validation_check = next(r for r in report.records if r.check_id == "finite_admitted_batches[validation]")
    assert validation_check.severity == "ERROR", (
        "a stray dynamic-input NaN outside the declared gap set must surface as a "
        "finite-admitted-batch failure, since FlashNHDataset only excludes by "
        "documented gap timestamp, not by scanning raw dynamic-input NaNs"
    )

    inventory = inspect_dynamic_nan_inventory(
        package_root, dynamic_inputs=["precip", "precip_gap"], basin_ids=BASINS
    )
    assert "precip" in inventory.nan_outside_documented_gaps
    flagged_basins = {entry["basin"] for entry in inventory.nan_outside_documented_gaps["precip"]}
    assert flagged_basins == set(BASINS)


# 20. gap-flag / actual-missingness mismatch, detected read-only, no dataset impact
def test_gap_flag_mismatch_detected_by_read_only_inventory_without_dataset_impact(tmp_path):
    wrong_flag_hours = [5]  # flagged as a gap hour with no actual NaN; real bad hours left unflagged
    package_root = tmp_path
    cfg_path = _build_synthetic_cfg(
        package_root,
        nan_hours=BAD_HOURS,
        flag_hours=wrong_flag_hours,
        declared_gap_hours=BAD_HOURS,  # NaNs still confined to declared gaps
    )

    report = AuditReport()
    check_flashnh_dataset_construction(report, cfg_path)
    assert report.error_count == 0, report.failed_messages()  # dataset-level filtering unaffected

    inventory = inspect_dynamic_nan_inventory(
        package_root, dynamic_inputs=["precip", "precip_gap"], basin_ids=BASINS
    )
    assert not inventory.nan_outside_documented_gaps
    assert "precip" in inventory.gap_flag_mismatches
    assert set(inventory.gap_flag_mismatches["precip"].keys()) == set(BASINS)


# ---------------------------------------------------------------------------
# Real Stage 1 gap-flag mapping: mrms_qpe_1h_mm_gap is 1:1 with
# mrms_qpe_1h_mm, but rtma_gap is a single flag shared by all five RTMA
# continuous variables. Uses the real Stage 1 variable names, not the
# generic precip/precip_gap fixture above.
# ---------------------------------------------------------------------------

REAL_STAGE1_DYNAMIC_INPUTS = [
    "mrms_qpe_1h_mm", "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg",
    "rtma_10u_ms", "rtma_10v_ms", "mrms_qpe_1h_mm_gap", "rtma_gap",
]


def _make_stage1_dynamic_values(basins, *, mrms_nan_hours=(), mrms_gap_hours=(),
                                 rtma_nan_hours=None, rtma_gap_hours=()):
    """rtma_nan_hours: dict mapping each of preflight.RTMA_CONTINUOUS_VARIABLES
    to the hours it carries an actual NaN at (default: none)."""
    rtma_nan_hours = rtma_nan_hours or {}
    values = {"mrms_qpe_1h_mm": {}, "mrms_qpe_1h_mm_gap": {}, "rtma_gap": {}}
    values.update({name: {} for name in preflight.RTMA_CONTINUOUS_VARIABLES})

    for basin in basins:
        mrms = np.arange(N_HOURS, dtype=np.float64)
        for h in mrms_nan_hours:
            mrms[h] = np.nan
        values["mrms_qpe_1h_mm"][basin] = mrms

        mrms_flag = np.zeros(N_HOURS, dtype=np.float64)
        for h in mrms_gap_hours:
            mrms_flag[h] = 1.0
        values["mrms_qpe_1h_mm_gap"][basin] = mrms_flag

        rtma_flag = np.zeros(N_HOURS, dtype=np.float64)
        for h in rtma_gap_hours:
            rtma_flag[h] = 1.0
        values["rtma_gap"][basin] = rtma_flag

        for var_name in preflight.RTMA_CONTINUOUS_VARIABLES:
            arr = np.arange(N_HOURS, dtype=np.float64) + 100.0
            for h in rtma_nan_hours.get(var_name, []):
                arr[h] = np.nan
            values[var_name][basin] = arr

    return values


def test_inspect_dynamic_nan_inventory_rtma_gap_and_mrms_gap_agree_cleanly(tmp_path):
    dynamic_input_values = _make_stage1_dynamic_values(
        BASINS,
        mrms_nan_hours=[10], mrms_gap_hours=[10],
        rtma_nan_hours={name: [20] for name in preflight.RTMA_CONTINUOUS_VARIABLES},
        rtma_gap_hours=[20],
    )
    build_synthetic_package(
        tmp_path, basins=BASINS, seq_length=SEQ_LENGTH, lead_hours=LEAD_HOURS,
        dynamic_input_values=dynamic_input_values, declared_gap_hours=[],
    )

    inventory = inspect_dynamic_nan_inventory(
        tmp_path, dynamic_inputs=REAL_STAGE1_DYNAMIC_INPUTS, basin_ids=BASINS
    )
    assert inventory.gap_flag_mismatches == {}


def test_inspect_dynamic_nan_inventory_rtma_gap_checked_against_all_five_rtma_variables(tmp_path):
    # rtma_gap only flags hour 20 as a gap; rtma_2t_K and rtma_10v_ms each
    # carry an extra real NaN the shared flag does not cover, while the
    # other three RTMA variables agree with the flag exactly (NaN at 20
    # only). mrms is fully clean/matching and must stay unaffected.
    dynamic_input_values = _make_stage1_dynamic_values(
        BASINS,
        mrms_nan_hours=[10], mrms_gap_hours=[10],
        rtma_nan_hours={
            "rtma_2t_K": [20, 30],       # 30 is an undeclared mismatch
            "rtma_2d_K": [20],
            "rtma_2sh_kgkg": [20],
            "rtma_10u_ms": [20],
            "rtma_10v_ms": [],           # missing the declared-gap NaN at 20
        },
        rtma_gap_hours=[20],
    )
    build_synthetic_package(
        tmp_path, basins=BASINS, seq_length=SEQ_LENGTH, lead_hours=LEAD_HOURS,
        dynamic_input_values=dynamic_input_values, declared_gap_hours=[],
    )

    inventory = inspect_dynamic_nan_inventory(
        tmp_path, dynamic_inputs=REAL_STAGE1_DYNAMIC_INPUTS, basin_ids=BASINS
    )
    assert set(inventory.gap_flag_mismatches.keys()) == {"rtma_2t_K", "rtma_10v_ms"}
    assert set(inventory.gap_flag_mismatches["rtma_2t_K"].keys()) == set(BASINS)
    assert set(inventory.gap_flag_mismatches["rtma_10v_ms"].keys()) == set(BASINS)
    assert "mrms_qpe_1h_mm" not in inventory.gap_flag_mismatches
    assert "rtma_2d_K" not in inventory.gap_flag_mismatches
    assert "rtma_2sh_kgkg" not in inventory.gap_flag_mismatches
    assert "rtma_10u_ms" not in inventory.gap_flag_mismatches
