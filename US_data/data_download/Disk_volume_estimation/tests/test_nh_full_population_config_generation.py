"""Tests for the full-population (development + spatial-holdout) config
bundle generation added to src.baseline.nh_config_generation (local
implementation increment, no h2o/Moriah access, no training).

Reuses the real, committed canonical split files
(config/stage1_baseline_splits_v001/) -- the actual 2,307/250/195-basin
development_train/spatial_holdout_nonca/california_all universes -- and the
lightweight fake-package builder from test_nh_config_generation.py. Unlike
the compact-package tests, the full-population validator requires the
package's basin_ids to equal EXACTLY the union of the two canonical non-CA
splits (no subset allowed), so the happy-path fixtures use the real, full
2,557-basin union.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.baseline.nh_config_generation import (  # noqa: E402
    EXPECTED_DEVELOPMENT_BASIN_COUNT,
    EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT,
    NHConfigGenerationError,
    generate_stage1_full_population_nh_config_bundles,
    validate_full_population_basin_membership,
    write_generated_config,
)
from src.baseline.splits import load_eligible_basins  # noqa: E402

from test_nh_config_generation import (  # noqa: E402 -- reuse fixture helpers
    POLICY_PATH,
    REAL_DYNAMIC_INPUTS,
    SPLITS_DIR,
    STATIC_COUNT,
    _build_fake_package,
)

REAL_DEVELOPMENT = sorted(load_eligible_basins(SPLITS_DIR / "development_train.txt"))
REAL_SPATIAL_HOLDOUT = sorted(load_eligible_basins(SPLITS_DIR / "spatial_holdout_nonca.txt"))
REAL_CALIFORNIA = sorted(load_eligible_basins(SPLITS_DIR / "california_all.txt"))
REAL_FULL_UNION = REAL_DEVELOPMENT + REAL_SPATIAL_HOLDOUT

assert len(REAL_DEVELOPMENT) == EXPECTED_DEVELOPMENT_BASIN_COUNT
assert len(REAL_SPATIAL_HOLDOUT) == EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT


# ---------------------------------------------------------------------------
# validate_full_population_basin_membership
# ---------------------------------------------------------------------------

def test_validate_full_population_basin_membership_accepts_real_full_union():
    result = validate_full_population_basin_membership({"basin_ids": REAL_FULL_UNION}, SPLITS_DIR)
    assert result.development_basins == REAL_DEVELOPMENT
    assert result.spatial_holdout_basins == REAL_SPATIAL_HOLDOUT
    assert len(result.development_basins) == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert len(result.spatial_holdout_basins) == EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT


def test_validate_full_population_basin_membership_rejects_missing_basin():
    incomplete = REAL_FULL_UNION[1:]  # drop one development basin
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": incomplete}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_extra_basin():
    extra = list(REAL_FULL_UNION) + ["99999999"]  # not dev, not holdout, not CA
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": extra}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_california_basin():
    with_ca = list(REAL_FULL_UNION) + [REAL_CALIFORNIA[0]]
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": with_ca}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_duplicate_ids():
    duped = list(REAL_FULL_UNION) + [REAL_FULL_UNION[0]]
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": duped}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_empty_basin_ids():
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": []}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_normalization_collision_duplicate():
    # Two distinct raw strings that normalize to the SAME canonical STAID
    # (short form zero-padded vs. already-8-digit) must be caught as a
    # duplicate, not merely literal repeated strings.
    short_form = REAL_DEVELOPMENT[0].lstrip("0") or REAL_DEVELOPMENT[0]
    assert short_form != REAL_DEVELOPMENT[0]  # only meaningful if it actually had a leading zero
    collided = list(REAL_FULL_UNION) + [short_form]
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": collided}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_california_substitution_same_total_count():
    # Swap one development basin out for one California basin, preserving the
    # exact 2,557 total count -- must still be rejected (missing development
    # basin AND a California basin present), not accepted because the count
    # alone matches.
    substituted = list(REAL_FULL_UNION[1:]) + [REAL_CALIFORNIA[0]]
    assert len(substituted) == len(REAL_FULL_UNION)
    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership({"basin_ids": substituted}, SPLITS_DIR)


def test_validate_full_population_basin_membership_rejects_dev_holdout_overlap_in_canonical_splits(tmp_path):
    # Tamper with a *copy* of the canonical splits so development_train and
    # spatial_holdout_nonca share one basin -- must be caught even though the
    # real committed splits never overlap (re-checked here rather than
    # assumed, per the binding safeguard).
    tampered_dir = tmp_path / "tampered_splits"
    tampered_dir.mkdir()
    shared = REAL_DEVELOPMENT[0]
    (tampered_dir / "development_train.txt").write_text("\n".join(REAL_DEVELOPMENT) + "\n", encoding="utf-8")
    (tampered_dir / "spatial_holdout_nonca.txt").write_text(
        "\n".join(REAL_SPATIAL_HOLDOUT + [shared]) + "\n", encoding="utf-8"
    )
    (tampered_dir / "california_all.txt").write_text("\n".join(REAL_CALIFORNIA) + "\n", encoding="utf-8")

    with pytest.raises(NHConfigGenerationError):
        validate_full_population_basin_membership(
            {"basin_ids": REAL_FULL_UNION + [shared]}, tampered_dir
        )


# ---------------------------------------------------------------------------
# generate_stage1_full_population_nh_config_bundles (end to end)
# ---------------------------------------------------------------------------

def test_generate_full_population_bundles_end_to_end(tmp_path):
    package_root = _build_fake_package(tmp_path / "package", REAL_FULL_UNION)

    bundles = generate_stage1_full_population_nh_config_bundles(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )

    dev = bundles.development
    holdout = bundles.spatial_holdout

    # Scientific contract shared by both bundles.
    for bundle in (dev, holdout):
        assert bundle.target_variable == "qobs_mm_per_h_lead06"
        assert bundle.seq_length == 24
        assert bundle.dynamic_inputs == REAL_DYNAMIC_INPUTS
        assert bundle.static_attribute_result.count == STATIC_COUNT

    # Development bundle: its own population is the fitting/eval population.
    assert dev.basin_ids == REAL_DEVELOPMENT
    assert len(dev.basin_ids) == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert dev.train_basin_ids is None  # falls back to basin_ids in write_generated_config
    assert dev.validation_basin_ids is None
    assert dev.test_basin_ids is None
    assert dev.population_role == "development_identical_all_periods"

    # Spatial-holdout bundle: test-only; train/validation lists are the
    # DEVELOPMENT population, never the holdout basins themselves.
    assert holdout.basin_ids == REAL_SPATIAL_HOLDOUT
    assert len(holdout.basin_ids) == EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT
    assert holdout.train_basin_ids == REAL_DEVELOPMENT
    assert holdout.validation_basin_ids == REAL_DEVELOPMENT
    assert holdout.test_basin_ids == REAL_SPATIAL_HOLDOUT
    assert holdout.population_role == "spatial_holdout_test_only"
    assert set(holdout.train_basin_ids) & set(holdout.test_basin_ids) == set()

    assert bundles.basin_membership.development_basins == REAL_DEVELOPMENT
    assert bundles.basin_membership.spatial_holdout_basins == REAL_SPATIAL_HOLDOUT


def test_generate_full_population_bundles_written_basin_files_and_manifest(tmp_path):
    package_root = _build_fake_package(tmp_path / "package", REAL_FULL_UNION)
    bundles = generate_stage1_full_population_nh_config_bundles(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )

    dev_paths = write_generated_config(bundles.development, tmp_path / "out" / "development")
    holdout_paths = write_generated_config(bundles.spatial_holdout, tmp_path / "out" / "spatial_holdout")

    dev_cfg = yaml.safe_load(dev_paths["config.yaml"].read_text(encoding="utf-8"))
    holdout_cfg = yaml.safe_load(holdout_paths["config.yaml"].read_text(encoding="utf-8"))
    for cfg in (dev_cfg, holdout_cfg):
        assert cfg["dataset"] == "flashnh"
        assert cfg["seq_length"] == 24
        assert cfg["target_variables"] == ["qobs_mm_per_h_lead06"]
        assert "nan_handling_method" not in cfg

    # Development: train == validation == test == the 2,307 development basins.
    dev_train = set(dev_paths["train_basins.txt"].read_text(encoding="utf-8").split())
    dev_val = set(dev_paths["validation_basins.txt"].read_text(encoding="utf-8").split())
    dev_test = set(dev_paths["test_basins.txt"].read_text(encoding="utf-8").split())
    assert dev_train == dev_val == dev_test == set(REAL_DEVELOPMENT)

    # Spatial holdout: train == validation == development population (never a
    # holdout basin); test == the 250 spatial-holdout basins, disjoint from
    # train/validation.
    holdout_train = set(holdout_paths["train_basins.txt"].read_text(encoding="utf-8").split())
    holdout_val = set(holdout_paths["validation_basins.txt"].read_text(encoding="utf-8").split())
    holdout_test = set(holdout_paths["test_basins.txt"].read_text(encoding="utf-8").split())
    assert holdout_train == holdout_val == set(REAL_DEVELOPMENT)
    assert holdout_test == set(REAL_SPATIAL_HOLDOUT)
    assert holdout_test.isdisjoint(holdout_train)

    dev_manifest = json.loads(dev_paths["generation_manifest.json"].read_text(encoding="utf-8"))
    holdout_manifest = json.loads(holdout_paths["generation_manifest.json"].read_text(encoding="utf-8"))
    assert dev_manifest["population_role"] == "development_identical_all_periods"
    assert dev_manifest["train_basin_count"] == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert dev_manifest["validation_basin_count"] == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert dev_manifest["test_basin_count"] == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert holdout_manifest["population_role"] == "spatial_holdout_test_only"
    assert holdout_manifest["train_basin_count"] == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert holdout_manifest["validation_basin_count"] == EXPECTED_DEVELOPMENT_BASIN_COUNT
    assert holdout_manifest["test_basin_count"] == EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT


def test_write_generated_config_distinguishes_holdout_default_experiment_name(tmp_path):
    package_root = _build_fake_package(tmp_path / "package", REAL_FULL_UNION)
    bundles = generate_stage1_full_population_nh_config_bundles(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    dev_paths = write_generated_config(bundles.development, tmp_path / "out" / "development")
    holdout_paths = write_generated_config(bundles.spatial_holdout, tmp_path / "out" / "spatial_holdout")

    dev_cfg = yaml.safe_load(dev_paths["config.yaml"].read_text(encoding="utf-8"))
    holdout_cfg = yaml.safe_load(holdout_paths["config.yaml"].read_text(encoding="utf-8"))
    # The two bundles must never share a default experiment_name -- an
    # identical name would let the holdout bundle be confused for (or
    # accidentally overwrite the run directory of) the development experiment.
    assert dev_cfg["experiment_name"] != holdout_cfg["experiment_name"]
    assert "spatial_holdout" in holdout_cfg["experiment_name"]


def test_write_generated_config_writes_test_only_marker_for_holdout_bundle_only(tmp_path):
    package_root = _build_fake_package(tmp_path / "package", REAL_FULL_UNION)
    bundles = generate_stage1_full_population_nh_config_bundles(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    dev_paths = write_generated_config(bundles.development, tmp_path / "out" / "development")
    holdout_paths = write_generated_config(bundles.spatial_holdout, tmp_path / "out" / "spatial_holdout")

    assert "TEST_ONLY_DO_NOT_TRAIN.txt" not in dev_paths
    assert not (tmp_path / "out" / "development" / "TEST_ONLY_DO_NOT_TRAIN.txt").exists()
    assert "TEST_ONLY_DO_NOT_TRAIN.txt" in holdout_paths
    marker_text = holdout_paths["TEST_ONLY_DO_NOT_TRAIN.txt"].read_text(encoding="utf-8")
    assert "TEST-ONLY" in marker_text
    assert "Do NOT run a trainer" in marker_text


def test_generate_full_population_bundles_rejects_partial_package(tmp_path):
    # A package containing only the development basins (the compact-style
    # contract) must be rejected by the full-population generator -- it
    # requires the EXACT dev+holdout union, not a subset.
    package_root = _build_fake_package(tmp_path / "package", REAL_DEVELOPMENT)
    with pytest.raises(NHConfigGenerationError):
        generate_stage1_full_population_nh_config_bundles(
            policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
            lead_hours=6, seq_length=24,
        )
