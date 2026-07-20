"""Tests for src/baseline/policy.py + the committed baseline policy YAML (I-A1).

Mutation tests deep-copy the committed policy mapping, corrupt exactly one
thing, and assert rejection — so every binding invariant is exercised against
the real artifact rather than a parallel fixture.
"""
import copy
from pathlib import Path

import pytest
import yaml

from src.baseline.policy import (
    Stage1BaselinePolicyError,
    load_stage1_baseline_policy,
    validate_stage1_baseline_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"


@pytest.fixture(scope="module")
def committed_policy():
    return load_stage1_baseline_policy(POLICY_PATH)


@pytest.fixture()
def policy(committed_policy):
    return copy.deepcopy(committed_policy)


def _set(data: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _reject(data, match=None):
    with pytest.raises(Stage1BaselinePolicyError, match=match):
        validate_stage1_baseline_policy(data)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_committed_policy_loads_with_expected_core_values(committed_policy):
    assert committed_policy["policy_name"] == "stage1_scientific_baseline_v001"
    assert committed_policy["policy_version"] == 2
    assert committed_policy["period"]["expected_hours"] == 45720
    assert committed_policy["target"]["leads_hours"] == [1, 3, 6, 12]
    assert committed_policy["seq_lengths_hours"] == [12, 24, 48, 72]
    assert committed_policy["spatial_split"]["seed"] == 42
    assert committed_policy["spatial_split"]["missing_hydroclimate_policy"]["never_in_holdout"] is True
    assert committed_policy["spatial_split"]["fallback_policy"]["ladder"] == "single_level_huc02_sparse_pool"
    assert committed_policy["spatial_split"]["exact_holdout_count_binding"] is False
    assert committed_policy["basin_universe"]["expected_nonca_count"] == 2557
    assert committed_policy["basin_universe"]["expected_ca_count"] == 195
    assert len(committed_policy["dynamic_inputs"]) == 8
    assert committed_policy["gap_policy"]["target_hour_forcing_gap_exclusion"] is False


def test_returned_object_is_plain_dict(committed_policy):
    assert type(committed_policy) is dict


def test_canonical_eligible_list_exists_and_matches_policy_count(committed_policy):
    # I-A5 canonical promotion is complete: the policy-resolved eligible-
    # basins list now intentionally exists, and its line count agrees with
    # the signed policy's expected_eligible_count. Full checksum/content
    # verification against the split manifest is the committed I-A3
    # auditor's job (scripts/audit_stage1_baseline_splits.py) and is not
    # duplicated here.
    eligible = REPO_ROOT / committed_policy["basin_universe"]["eligible_basins_list"]
    assert eligible.exists()
    lines = [line for line in eligible.read_text().splitlines() if line.strip()]
    assert len(lines) == committed_policy["basin_universe"]["expected_eligible_count"]


def test_validate_does_not_mutate_input(policy):
    snapshot = copy.deepcopy(policy)
    validate_stage1_baseline_policy(policy)
    assert policy == snapshot


# ---------------------------------------------------------------------------
# File-shape failures
# ---------------------------------------------------------------------------


def test_missing_file_rejected(tmp_path):
    with pytest.raises(Stage1BaselinePolicyError, match="not found"):
        load_stage1_baseline_policy(tmp_path / "does_not_exist.yaml")


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(Stage1BaselinePolicyError, match="empty"):
        load_stage1_baseline_policy(p)


def test_top_level_list_rejected(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(Stage1BaselinePolicyError, match="mapping"):
        load_stage1_baseline_policy(p)


def test_top_level_non_mapping_in_memory_rejected():
    with pytest.raises(Stage1BaselinePolicyError, match="mapping"):
        validate_stage1_baseline_policy(["not", "a", "mapping"])


# ---------------------------------------------------------------------------
# Missing sections / keys
# ---------------------------------------------------------------------------


def test_missing_required_section_rejected(policy):
    del policy["gap_policy"]
    _reject(policy, match="gap_policy")


def test_missing_required_nested_key_rejected(policy):
    del policy["target"]["primary_lead"]
    _reject(policy, match="target.primary_lead")


# ---------------------------------------------------------------------------
# Binding-invariant mutations (each corrupts exactly one approved value)
# ---------------------------------------------------------------------------

_INVARIANT_MUTATIONS = [
    # (id, dotted path, bad value, match substring or None)
    ("wrong-policy-name", "policy_name", "stage1_baseline_v002", "policy_name"),
    ("wrong-policy-version", "policy_version", 1, "policy_version"),
    ("bool-as-policy-version", "policy_version", True, "policy_version"),
    ("wrong-expected-hours", "period.expected_hours", 45721, "expected_hours"),
    ("string-expected-hours", "period.expected_hours", "45720", "expected_hours"),
    ("wrong-temporal-date", "temporal_split.training.end", "2022-12-31", None),
    ("unquoted-date-type", "temporal_split.test.start", 20250101, None),
    ("wrong-eligible-count", "basin_universe.expected_eligible_count", 2754, None),
    ("wrong-nonca-count", "basin_universe.expected_nonca_count", 2556, None),
    ("wrong-ca-count", "basin_universe.expected_ca_count", 196, None),
    ("numeric-excluded-staids", "basin_universe.excluded_staids",
     [2299472, 4073468], "strings"),
    ("wrong-excluded-set", "basin_universe.excluded_staids",
     ["02299472", "99999999"], None),
    ("missing-excluded-staid", "basin_universe.excluded_staids", ["02299472"], None),
    ("wrong-ca-rule", "basin_universe.california.membership_rule",
     "HUC02 == 18", "membership_rule"),
    ("wrong-seed", "spatial_split.seed", 7, "spatial_split.seed"),
    ("string-seed", "spatial_split.seed", "42", "spatial_split.seed"),
    ("wrong-holdout-fraction", "spatial_split.nonca_holdout_fraction", 0.20, None),
    ("wrong-tolerance", "spatial_split.holdout_tolerance", 0.05, None),
    ("wrong-area-binning", "spatial_split.area_binning", "quartiles", None),
    ("wrong-hydro-binning", "spatial_split.hydroclimate_binning", "deciles", None),
    ("wrong-min-stratum", "spatial_split.min_composite_stratum_size", 5, None),
    ("ca-finetune-out-of-range", "spatial_split.california_finetune_fraction",
     1.5, "california_finetune_fraction"),
    ("wrong-missing-hydro-rule", "spatial_split.missing_hydroclimate_policy.rule",
     "impute_with_median", "rule"),
    ("wrong-missing-hydro-reason",
     "spatial_split.missing_hydroclimate_policy.assignment_reason",
     "aridity_missing", None),
    ("missing-hydro-never-holdout-false",
     "spatial_split.missing_hydroclimate_policy.never_in_holdout", False,
     "never_in_holdout"),
    ("wrong-missing-hydro-count",
     "spatial_split.missing_hydroclimate_policy.known_missing_count_v001", 6, None),
    ("wrong-fallback-ladder", "spatial_split.fallback_policy.ladder",
     "multi_level_huc02_area_pool", "ladder"),
    ("intermediate-huc02-area-layer-reintroduced",
     "spatial_split.fallback_policy.no_intermediate_huc02_area_layer", False, None),
    ("whole-huc02-downgrade-reintroduced",
     "spatial_split.fallback_policy.no_whole_huc02_downgrade_for_sibling_sparsity",
     False, None),
    ("wrong-ca-fallback-ladder", "spatial_split.california_fallback_policy.ladder",
     "huc02_dependent_pool", "ladder"),
    ("exact-holdout-count-pinned",
     "spatial_split.exact_holdout_count_binding", True,
     "exact_holdout_count_binding"),
    ("largest-remainder-reintroduced",
     "spatial_split.largest_remainder_optimization_used", True, None),
    ("wrong-target-source", "target.source_variable", "qobs_cfs", None),
    ("wrong-conversion-constant", "target.conversion_constant", 3.7,
     "conversion_constant"),
    ("wrong-area-field", "target.area_field", "AREA_SQKM", None),
    ("wrong-area-units", "target.area_units", "mi2", None),
    ("missing-lead", "target.leads_hours", [1, 3, 6], "target.leads_hours"),
    ("extra-lead", "target.leads_hours", [1, 3, 6, 12, 24], "target.leads_hours"),
    ("duplicate-lead", "target.leads_hours", [1, 3, 6, 6], "duplicate"),
    ("bool-lead", "target.leads_hours", [True, 3, 6, 12], "target.leads_hours"),
    ("wrong-primary-lead", "target.primary_lead", 12, "primary_lead"),
    ("wrong-diagnostic-leads", "target.diagnostic_leads", [1], None),
    ("wrong-template", "target.variable_name_template",
     "qobs_mm_lead{lead}", "variable_name_template"),
    ("seq-168-reintroduced", "seq_lengths_hours", [12, 24, 48, 72, 168],
     "seq_lengths_hours"),
    ("seq-336-reintroduced", "seq_lengths_hours", [12, 24, 48, 336], None),
    ("seq-reordered", "seq_lengths_hours", [72, 48, 24, 12], "order"),
    ("forbidden-seq-list-shrunk", "forbidden_seq_lengths_hours", [168], None),
    ("target-hour-exclusion-true",
     "gap_policy.target_hour_forcing_gap_exclusion", True, "target_hour"),
    ("missing-eval-scope", "gap_policy.applies_to",
     ["training", "validation", "temporal_test",
      "california_finetune", "california_holdout"], "applies_to"),
    ("wrong-mrms-gap-count", "gap_policy.expected_mrms_gap_hours", 137, None),
    ("wrong-rtma-gap-count", "gap_policy.expected_rtma_gap_hours", 3, None),
    ("wrong-gap-policy-kind", "gap_policy.forcing_history_policy",
     "fill_with_zero", None),
    ("reordered-dynamic-inputs", "dynamic_inputs",
     ["rtma_2t_K", "mrms_qpe_1h_mm", "rtma_2d_K", "rtma_2sh_kgkg",
      "rtma_10u_ms", "rtma_10v_ms", "mrms_qpe_1h_mm_gap", "rtma_gap"],
     "dynamic_inputs"),
    ("duplicate-dynamic-input", "dynamic_inputs",
     ["mrms_qpe_1h_mm", "mrms_qpe_1h_mm", "rtma_2d_K", "rtma_2sh_kgkg",
      "rtma_10u_ms", "rtma_10v_ms", "mrms_qpe_1h_mm_gap", "rtma_gap"],
     "duplicate"),
    ("missing-dynamic-input", "dynamic_inputs",
     ["mrms_qpe_1h_mm", "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg",
      "rtma_10u_ms", "rtma_10v_ms", "mrms_qpe_1h_mm_gap"], None),
    ("wrong-matrix-name", "static_attributes.matrix_name",
     "stage1_static_attributes_v001", None),
    ("wrong-matrix-sha", "static_attributes.sha256", "0" * 64, "sha256"),
    ("wrong-matrix-rows", "static_attributes.expected_rows", 2842, None),
    ("wrong-matrix-cols", "static_attributes.expected_columns", 522, None),
    ("wrong-model-input-count",
     "static_attributes.expected_model_input_columns", 472, None),
    ("forbidden-static-list-shrunk", "static_attributes.forbidden_model_inputs",
     ["STATE", "HUC02", "LAT_GAGE"], "forbidden_model_inputs"),
    ("wrong-imputation-strategy", "static_attributes.imputation.strategy",
     "mean", None),
    ("wrong-imputation-scope", "static_attributes.imputation.fit_basin_scope",
     "all_basins", "fit_basin_scope"),
    ("wrong-imputation-apply-scope",
     "static_attributes.imputation.apply_unchanged_to",
     ["validation", "temporal_test"], None),
    ("wrong-nh-version", "nh.expected_version", "1.12.0", None),
    ("wrong-nh-dataset", "nh.dataset", "camels_us", None),
    ("wrong-nh-head", "nh.head", "gmm", None),
    ("wrong-nh-activation", "nh.output_activation", "softplus", None),
    ("bool-predict-last-n", "nh.predict_last_n", True, "predict_last_n"),
    ("wrong-roundtrip-rtol", "audit.roundtrip_float64_rtol", 1.0e-6, None),
]


@pytest.mark.parametrize(
    "dotted,bad,match",
    [(m[1], m[2], m[3]) for m in _INVARIANT_MUTATIONS],
    ids=[m[0] for m in _INVARIANT_MUTATIONS],
)
def test_binding_invariant_mutation_rejected(policy, dotted, bad, match):
    _set(policy, dotted, bad)
    _reject(policy, match=match)


def test_overlapping_split_rejected_with_ordering_message(policy):
    _set(policy, "temporal_split.validation.start", "2023-06-01")
    _reject(policy, match="ordered training")


# ---------------------------------------------------------------------------
# Security / portability
# ---------------------------------------------------------------------------


def test_windows_absolute_path_in_required_key_rejected(policy):
    _set(policy, "basin_universe.initial_manifest", "C:\\data\\manifest.csv")
    _reject(policy, match="initial_manifest")


def test_windows_absolute_path_in_extra_key_rejected(policy):
    policy["scratch"] = "C:\\Users\\someone\\data.csv"
    _reject(policy, match="Windows path")


def test_posix_absolute_path_rejected(policy):
    policy["scratch"] = "/data42/omrip/Flash-NH/tmp/x.parquet"
    _reject(policy, match="POSIX path")


def test_home_relative_path_rejected(policy):
    policy["scratch"] = "~/flashnh/data.csv"
    _reject(policy, match="home-relative")


def test_absolute_path_inside_list_rejected(policy):
    policy["scratch_list"] = ["config/ok.txt", "/etc/passwd_like_path"]
    _reject(policy, match="POSIX path")


def test_credential_like_top_level_key_rejected(policy):
    policy["api_key"] = "abc123"
    _reject(policy, match="api_key")


@pytest.mark.parametrize("bad_key", ["password", "wandb_token", "aws_secret"])
def test_credential_like_nested_key_rejected(policy, bad_key):
    policy["nh"][bad_key] = "value"
    _reject(policy, match="credential-like")


def test_normal_scientific_strings_are_not_falsely_rejected(policy):
    # The committed policy already contains "STATE == CA", "DD/MM/YYYY",
    # "m3/s", a 64-hex sha256, and a long report sentence; an extra benign
    # free-text note must also pass.
    policy["scratch_note"] = "runoff in mm/h; membership via STATE == CA; see docs/"
    assert validate_stage1_baseline_policy(policy) is policy


# ---------------------------------------------------------------------------
# Error-message quality (path-oriented messages)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dotted,bad,expected_fragment",
    [
        ("target.leads_hours", [1, 3, 6], "target.leads_hours"),
        ("spatial_split.seed", 7, "spatial_split.seed"),
        ("gap_policy.target_hour_forcing_gap_exclusion", True,
         "gap_policy.target_hour_forcing_gap_exclusion"),
        ("nh.predict_last_n", 2, "nh.predict_last_n"),
    ],
)
def test_error_message_names_failing_dotted_path(policy, dotted, bad, expected_fragment):
    _set(policy, dotted, bad)
    with pytest.raises(Stage1BaselinePolicyError) as excinfo:
        validate_stage1_baseline_policy(policy)
    assert expected_fragment in str(excinfo.value)


def test_committed_yaml_is_plain_yaml_mapping_on_raw_read():
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    # Dates are stored as quoted strings (one unambiguous representation).
    assert isinstance(raw["temporal_split"]["training"]["start"], str)
    assert isinstance(raw["signed_off"]["date"], str)
