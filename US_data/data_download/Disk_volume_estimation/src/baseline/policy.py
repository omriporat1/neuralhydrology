"""Loader/validator for the Stage 1 scientific-baseline policy (Milestone 2K-G-I I-A1).

Loads config/stage1_scientific_baseline_v001.yaml and validates it against
the binding 2K-G-H / 2K-G-I sign-off values. The validator is deliberately
version-specific: it pins the approved values one by one instead of
implementing a general schema engine, so any drift between the YAML and the
binding policy record fails loudly at load time with a path-oriented message.

Public API:
    load_stage1_baseline_policy(path) -> dict
    validate_stage1_baseline_policy(data) -> dict
    Stage1BaselinePolicyError (subclass of ValueError)

Portability rules enforced over every key/value in the file: no absolute
Windows/POSIX paths, no home-relative (~) paths, no credential-like keys.
Relative repository paths are allowed and future artifacts (e.g. the
eligible-basins list) are NOT required to exist on disk.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

__all__ = [
    "Stage1BaselinePolicyError",
    "load_stage1_baseline_policy",
    "validate_stage1_baseline_policy",
]


class Stage1BaselinePolicyError(ValueError):
    """Raised when the Stage 1 baseline policy file is missing or invalid."""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _fail(path: str, message: str) -> None:
    raise Stage1BaselinePolicyError(f"{path}: {message}")


def _get(data: dict, dotted: str):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            _fail(dotted, "missing required key")
        node = node[part]
    return node


def _scalar_matches(value, expected) -> bool:
    """Type-strict scalar equality (bools never satisfy int/float slots)."""
    if isinstance(expected, bool):
        return isinstance(value, bool) and value == expected
    if isinstance(expected, int):
        return (not isinstance(value, bool)) and isinstance(value, int) and value == expected
    if isinstance(expected, float):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and float(value) == expected
        )
    if isinstance(expected, str):
        return isinstance(value, str) and value == expected
    return value == expected


def _expect(data: dict, dotted: str, expected) -> None:
    """Require an exact, type-strict match against the binding value."""
    value = _get(data, dotted)
    if isinstance(expected, list):
        if not isinstance(value, list):
            _fail(dotted, f"must be a list, got {type(value).__name__}")
        if len(value) != len(set(map(repr, value))):
            _fail(dotted, f"contains duplicate values: {value!r}")
        if len(value) != len(expected) or not all(
            _scalar_matches(v, e) for v, e in zip(value, expected)
        ):
            _fail(dotted, f"must equal {expected!r} (order binding), got {value!r}")
        return
    if not _scalar_matches(value, expected):
        _fail(dotted, f"must equal {expected!r}, got {value!r}")


def _iso_date(data: dict, dotted: str) -> date:
    value = _get(data, dotted)
    if not isinstance(value, str):
        _fail(dotted, "must be a quoted ISO date string (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError:
        _fail(dotted, f"is not a valid ISO date (YYYY-MM-DD): {value!r}")


# ---------------------------------------------------------------------------
# Portability / credential safeguards
# ---------------------------------------------------------------------------

_WINDOWS_ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(?:^|[_\-.])(api[_\-]?key|apikey|password|passwd|secret|token|credentials?)(?:$|[_\-.])"
)


def _check_portability(node, keypath: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{keypath}.{key}" if keypath else str(key)
            if isinstance(key, str) and _CREDENTIAL_KEY_RE.search(key):
                _fail(child, "credential-like key is forbidden in the policy file")
            _check_portability(value, child)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _check_portability(value, f"{keypath}[{i}]")
    elif isinstance(node, str):
        if _WINDOWS_ABS_RE.match(node):
            _fail(keypath, f"absolute Windows path is forbidden in portable policy: {node!r}")
        if node.startswith("/"):
            _fail(keypath, f"absolute POSIX path is forbidden in portable policy: {node!r}")
        if node.startswith("~"):
            _fail(keypath, f"home-relative path is forbidden in portable policy: {node!r}")


# ---------------------------------------------------------------------------
# Binding values (2K-G-H sign-off + 2K-G-I I-0 sign-offs, incl. the
# 2026-07-13 history-only gap-mask correction)
# ---------------------------------------------------------------------------

_LEADS = [1, 3, 6, 12]
_SEQ_LENGTHS = [12, 24, 48, 72]
_FORBIDDEN_SEQ_LENGTHS = [168, 336]
_DYNAMIC_INPUTS = [
    "mrms_qpe_1h_mm",
    "rtma_2t_K",
    "rtma_2d_K",
    "rtma_2sh_kgkg",
    "rtma_10u_ms",
    "rtma_10v_ms",
    "mrms_qpe_1h_mm_gap",
    "rtma_gap",
]
_GAP_SCOPES = [
    "training",
    "validation",
    "temporal_test",
    "spatial_holdout",
    "california_finetune",
    "california_holdout",
]
_FORBIDDEN_STATIC_INPUTS = ["STATE", "HUC02", "LAT_GAGE", "LNG_GAGE"]
_MATRIX_SHA256 = "eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464"
_REPORT_TEXT = (
    "predictions unavailable due to required forcing history intersecting "
    "an archive gap"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_stage1_baseline_policy(path) -> dict:
    """Load + validate the Stage 1 baseline policy YAML; return a plain dict."""
    policy_path = Path(path)
    if not policy_path.is_file():
        raise Stage1BaselinePolicyError(f"policy file not found: {policy_path}")
    text = policy_path.read_text(encoding="utf-8")
    if not text.strip():
        raise Stage1BaselinePolicyError(f"policy file is empty: {policy_path}")
    data = yaml.safe_load(text)
    return validate_stage1_baseline_policy(data)


def validate_stage1_baseline_policy(data) -> dict:
    """Validate an in-memory policy mapping against every binding invariant.

    Returns the (unmodified) mapping on success; raises
    Stage1BaselinePolicyError with a path-oriented message on the first
    violation. Never mutates the input and never touches the filesystem
    (future artifacts such as the eligible-basins list are not required to
    exist).
    """
    if not isinstance(data, dict):
        raise Stage1BaselinePolicyError(
            f"top level must be a mapping, got {type(data).__name__}"
        )

    # Portability / credential sweep over the whole document first, so an
    # absolute path or credential-like key anywhere fails with the specific
    # message rather than a generic mismatch.
    _check_portability(data, "")

    # ---- identity / provenance ----
    _expect(data, "policy_name", "stage1_scientific_baseline_v001")
    _expect(data, "policy_version", 1)
    _expect(data, "signed_off.milestone", "2K-G-H")
    _expect(data, "signed_off.date", "2026-07-12")
    _expect(data, "signed_off.commit", "e860316")

    # ---- research period ----
    _expect(data, "period.start_utc", "2020-10-14T00:00:00Z")
    _expect(data, "period.end_utc", "2025-12-31T23:00:00Z")
    _expect(data, "period.expected_hours", 45720)

    # ---- temporal split: parse, order, then exact pins ----
    train_s = _iso_date(data, "temporal_split.training.start")
    train_e = _iso_date(data, "temporal_split.training.end")
    val_s = _iso_date(data, "temporal_split.validation.start")
    val_e = _iso_date(data, "temporal_split.validation.end")
    test_s = _iso_date(data, "temporal_split.test.start")
    test_e = _iso_date(data, "temporal_split.test.end")
    ordered = (
        train_s <= train_e < val_s <= val_e < test_s <= test_e
    )
    if not ordered:
        _fail(
            "temporal_split",
            "periods must be ordered training -> validation -> test with no overlap",
        )
    _expect(data, "temporal_split.training.start", "2020-10-14")
    _expect(data, "temporal_split.training.end", "2023-12-31")
    _expect(data, "temporal_split.validation.start", "2024-01-01")
    _expect(data, "temporal_split.validation.end", "2024-12-31")
    _expect(data, "temporal_split.test.start", "2025-01-01")
    _expect(data, "temporal_split.test.end", "2025-12-31")

    # ---- basin universe ----
    _expect(data, "basin_universe.initial_manifest",
            "config/stage1_initial_training_basin_manifest.csv")
    _expect(data, "basin_universe.eligible_basins_list",
            "config/stage1_baseline_splits_v001/eligible_basins_v001.txt")
    _expect(data, "basin_universe.expected_eligible_count", 2752)
    _expect(data, "basin_universe.expected_nonca_count", 2557)
    _expect(data, "basin_universe.expected_ca_count", 195)
    eligible_count = _get(data, "basin_universe.expected_eligible_count")
    nonca_count = _get(data, "basin_universe.expected_nonca_count")
    ca_count = _get(data, "basin_universe.expected_ca_count")
    if nonca_count + ca_count != eligible_count:
        _fail("basin_universe",
              "expected_nonca_count + expected_ca_count must equal expected_eligible_count")
    excluded = _get(data, "basin_universe.excluded_staids")
    if not isinstance(excluded, list) or not all(isinstance(s, str) for s in excluded):
        _fail("basin_universe.excluded_staids",
              f"entries must be strings (leading zeros are load-bearing), got {excluded!r}")
    _expect(data, "basin_universe.excluded_staids", ["02299472", "04073468"])
    _expect(data, "basin_universe.policy_excluded_target_status",
            ["TARGET_OPERATIONAL_REVIEW"])
    _expect(data, "basin_universe.california.membership_rule", "STATE == CA")
    _expect(data, "basin_universe.california.excluded_stages_1_3", True)

    # ---- spatial split ----
    _expect(data, "spatial_split.seed", 42)
    _expect(data, "spatial_split.nonca_holdout_fraction", 0.10)
    _expect(data, "spatial_split.holdout_tolerance", 0.02)
    frac = float(_get(data, "spatial_split.nonca_holdout_fraction"))
    tol = float(_get(data, "spatial_split.holdout_tolerance"))
    if not (0.0 <= frac - tol and frac + tol <= 1.0):
        _fail("spatial_split", "holdout fraction +/- tolerance must stay inside [0, 1]")
    _expect(data, "spatial_split.geography_field", "HUC02")
    _expect(data, "spatial_split.area_field", "DRAIN_SQKM")
    _expect(data, "spatial_split.hydroclimate_field", "ari_ix_uav")
    _expect(data, "spatial_split.area_binning", "terciles")
    _expect(data, "spatial_split.hydroclimate_binning", "terciles")
    _expect(data, "spatial_split.min_composite_stratum_size", 10)
    _expect(data, "spatial_split.method_status",
            "candidate_subject_to_machine_and_human_qc")
    ca_frac_value = _get(data, "spatial_split.california_finetune_fraction")
    if isinstance(ca_frac_value, bool) or not isinstance(ca_frac_value, (int, float)) \
            or not (0.0 < float(ca_frac_value) < 1.0):
        _fail("spatial_split.california_finetune_fraction",
              f"must be a number strictly inside (0, 1), got {ca_frac_value!r}")
    _expect(data, "spatial_split.california_finetune_fraction", 0.90)
    _expect(data, "spatial_split.california_stratification",
            "area_and_aridity_not_huc02_dependent")
    _expect(data, "spatial_split.missing_hydroclimate_policy.rule",
            "assign_to_training_role_directly")
    _expect(data, "spatial_split.missing_hydroclimate_policy.applies_to_field", "ari_ix_uav")
    _expect(data, "spatial_split.missing_hydroclimate_policy.assignment_reason",
            "missing_hydroatlas_stratifier")
    _expect(data, "spatial_split.missing_hydroclimate_policy.excluded_from_stratification", True)
    _expect(data, "spatial_split.missing_hydroclimate_policy.never_in_holdout", True)
    _expect(data, "spatial_split.missing_hydroclimate_policy.known_missing_count_v001", 5)
    _expect(data, "spatial_split.fallback_policy.ladder", "single_level_huc02_sparse_pool")
    _expect(data, "spatial_split.fallback_policy.no_intermediate_huc02_area_layer", True)
    _expect(data, "spatial_split.fallback_policy.no_whole_huc02_downgrade_for_sibling_sparsity",
            True)
    _expect(data, "spatial_split.california_fallback_policy.ladder",
            "single_level_statewide_sparse_pool")
    _expect(data, "spatial_split.exact_holdout_count_binding", False)
    _expect(data, "spatial_split.largest_remainder_optimization_used", False)

    # ---- target ----
    _expect(data, "target.source_variable", "qobs_m3s")
    _expect(data, "target.training_units", "mm_per_h_equivalent_runoff_depth")
    _expect(data, "target.conversion_constant", 3.6)
    _expect(data, "target.area_field", "DRAIN_SQKM")
    _expect(data, "target.area_units", "km2")
    _expect(data, "target.leads_hours", _LEADS)
    _expect(data, "target.primary_lead", 6)
    _expect(data, "target.secondary_lead", 12)
    _expect(data, "target.diagnostic_leads", [1, 3])
    _expect(data, "target.variable_name_template", "qobs_mm_per_h_lead{lead:02d}")
    _expect(data, "target.package_dtype", "float32")
    _expect(data, "target.internal_dtype", "float64")
    _expect(data, "target.evaluation_units", "m3/s")
    # Lead-role internal consistency (defense in depth on top of the pins).
    leads = _get(data, "target.leads_hours")
    primary = _get(data, "target.primary_lead")
    secondary = _get(data, "target.secondary_lead")
    diagnostic = _get(data, "target.diagnostic_leads")
    if primary not in leads or secondary not in leads or primary == secondary:
        _fail("target", "primary/secondary leads must be distinct members of leads_hours")
    if sorted(diagnostic) != sorted(set(leads) - {primary, secondary}):
        _fail("target.diagnostic_leads",
              "must equal leads_hours minus primary and secondary leads")

    # ---- sequence lengths ----
    _expect(data, "seq_lengths_hours", _SEQ_LENGTHS)
    _expect(data, "forbidden_seq_lengths_hours", _FORBIDDEN_SEQ_LENGTHS)
    seq = _get(data, "seq_lengths_hours")
    if any(s in seq for s in _FORBIDDEN_SEQ_LENGTHS):
        _fail("seq_lengths_hours",
              "168/336 h are Stage 2 lookbacks, explicitly forbidden in Stage 1")

    # ---- gap policy (history-only; 2026-07-13 correction) ----
    _expect(data, "gap_policy.forcing_history_policy",
            "hard_exclude_if_history_intersects_gap")
    _expect(data, "gap_policy.applies_to", _GAP_SCOPES)
    _expect(data, "gap_policy.expected_mrms_gap_hours", 136)
    _expect(data, "gap_policy.expected_rtma_gap_hours", 2)
    _expect(data, "gap_policy.include_rtma_in_history_mask", True)
    _expect(data, "gap_policy.target_hour_forcing_gap_exclusion", False)
    _expect(data, "gap_policy.qobs_nan_policy",
            "target_masking_separate_from_forcing_mask")
    _expect(data, "gap_policy.excluded_issue_time_report_text", _REPORT_TEXT)
    _expect(data, "gap_policy.nan_handling_method.baseline_relies_on_it", False)
    _expect(data, "gap_policy.nan_handling_method.unset_forbidden_when_nan_dynamic_inputs",
            True)

    # ---- dynamic inputs ----
    _expect(data, "dynamic_inputs", _DYNAMIC_INPUTS)
    dynamic = _get(data, "dynamic_inputs")
    overlap = set(dynamic) & set(_FORBIDDEN_STATIC_INPUTS)
    if overlap:
        _fail("dynamic_inputs",
              f"forbidden model-input names must not appear here: {sorted(overlap)}")

    # ---- static attributes ----
    _expect(data, "static_attributes.matrix_name", "stage1_static_attributes_v001")
    _expect(data, "static_attributes.expected_rows", 2843)
    _expect(data, "static_attributes.expected_columns", 531)
    _expect(data, "static_attributes.expected_model_input_columns", 496)
    _expect(data, "static_attributes.sha256", _MATRIX_SHA256)
    _expect(data, "static_attributes.role_source",
            "stage1_static_attributes_v001_column_manifest.json")
    _expect(data, "static_attributes.allowed_role", "model_input")
    _expect(data, "static_attributes.forbidden_model_inputs", _FORBIDDEN_STATIC_INPUTS)
    _expect(data, "static_attributes.imputation.strategy", "median")
    _expect(data, "static_attributes.imputation.fit_basin_scope",
            "development_training_only")
    _expect(data, "static_attributes.imputation.apply_unchanged_to",
            ["validation", "temporal_test", "spatial_holdout"])
    _expect(data, "static_attributes.imputation.fail_if_all_nan_in_development_training",
            True)
    _expect(data, "static_attributes.imputation.missingness_indicators_as_model_inputs",
            False)
    _expect(data, "static_attributes.stage4_refit_scope",
            "california_finetune_training_only")

    # ---- NH compatibility expectations ----
    _expect(data, "nh.expected_version", "1.13.0")
    _expect(data, "nh.dataset", "generic")
    _expect(data, "nh.date_format", "DD/MM/YYYY")
    _expect(data, "nh.head", "regression")
    _expect(data, "nh.output_activation", "linear")
    _expect(data, "nh.predict_last_n", 1)

    # ---- audit constants ----
    _expect(data, "audit.roundtrip_float64_rtol", 1.0e-12)
    _expect(data, "audit.package_float32_rtol", 1.0e-5)
    _expect(data, "audit.independent_lead_alignment_audit_required", True)
    _expect(data, "audit.raw_space_evaluation_audit_required", True)
    _expect(data, "audit.forcing_gap_masks_history_only", True)
    _expect(data, "audit.consumers_must_record_policy_sha256", True)

    return data
