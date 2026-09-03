"""Synthetic tests for the full-development-population Common-120 audit
contract foundation (milestone SHARED-A1) and its focused-correction round.

Everything here runs against small in-memory fixtures or the *committed* split
text artifacts.  Nothing reads the real scientific package, contacts Moriah /
Slurm / W&B, or builds the real 2,307-basin support artifact.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline import common120_support_builder as builder
from src.baseline import devpop_common120_audit_contract as audit
from src.baseline import fixed_support_contract_v2 as fixed
from src.baseline import sweep_v1_production_adapter as adapter
from src.baseline.devpop_common120_audit_contract import (
    CANONICAL_LEAD_HOURS,
    CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE,
    CANONICAL_SEQ_LENGTH_FLOOR,
    CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
    CANONICAL_TARGET_VARIABLE,
    DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    DEVELOPMENT_TRAIN_ROLE,
    DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256,
    DEVPOP_AUDIT_CONTRACT_ID,
    EXPECTED_DEVELOPMENT_POPULATION_SIZE,
    DevpopAuditCompletenessError,
    DevpopAuditContractError,
    ExpectedPopulationSpec,
    assert_canonical_development_population,
    build_devpop_audit_contract,
    canonical_membership_sha256,
    load_devpop_audit_contract,
    load_synthetic_devpop_audit_contract,
    require_complete_devpop_audit_population,
    require_complete_synthetic_devpop_audit_population,
    validate_canonical_devpop_audit_contract,
    validate_devpop_audit_contract,
    write_devpop_audit_contract,
    write_synthetic_devpop_audit_contract,
)
from src.baseline.gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT
from src.baseline.sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2

POP_IDS = tuple(f"{i:08d}" for i in range(6))
IDENT = {
    "package_manifest_sha256": "a" * 64,
    "package_file_checksums_sha256": "b" * 64,
    "package_run_provenance_sha256": "c" * 64,
    "development_split_sha256": "d" * 64,
    "spatial_holdout_split_sha256": "e" * 64,
}

_PROJECT_ROOT = Path(audit.__file__).parents[2]
_SPLITS_DIR = _PROJECT_ROOT / "config" / "stage1_baseline_splits_v001"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _population(ids=POP_IDS, *, role="development_train", **kw) -> ExpectedPopulationSpec:
    """A GENERIC / synthetic fixture population (never canonical)."""
    return ExpectedPopulationSpec.for_synthetic_fixture(role=role, basin_ids=list(ids), **kw)


def _contract(population: ExpectedPopulationSpec, *, n_admitted=3, timeline=5, **kw) -> dict:
    dates = np.arange(timeline, dtype="int64")
    admitted = np.array([i < n_admitted for i in range(timeline)])
    return build_devpop_audit_contract(
        population=population,
        target_variable=CANONICAL_TARGET_VARIABLE,
        source_gap_policy_identity=CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
        per_basin_date={b: dates for b in population.basin_ids},
        per_basin_admitted={b: admitted for b in population.basin_ids},
        **{**IDENT, **kw},
    )


def _result(population: ExpectedPopulationSpec, contract: dict, **overrides) -> dict:
    n = population.expected_size
    per_basin = [
        {
            "basin_id": b,
            "nse": 0.5,
            "n_sim_nonfinite_at_admitted": 0,
            "n_admitted": contract["eligible_counts"].get(b, 3),
        }
        for b in population.basin_ids
    ]
    n_admitted_total = sum(row["n_admitted"] for row in per_basin)
    base = {
        "objective_scope": "devpop_audit",
        "contract_id": DEVPOP_AUDIT_CONTRACT_ID,
        "contract_checksum_sha256": contract["checksum_sha256"],
        "requested_basin_ids": list(population.basin_ids),
        "evaluated_basin_ids": sorted(population.basin_ids),
        "n_basins_requested": n,
        "n_basins_evaluated": n,
        "n_basins_excluded": 0,
        "basins_excluded": [],
        "per_basin": per_basin,
        "aggregate": {
            "n_basins": n,
            "n_admitted_total": n_admitted_total,
            "n_sim_nonfinite_at_admitted_total": 0,
            "metrics": {"nse": {"n_finite_basins": n, "median": 0.5}},
        },
    }
    base.update(overrides)
    return base


def _checksum(payload: dict) -> str:
    return hashlib.sha256(fixed.canonical_contract_checksum_payload(payload)).hexdigest()


# --------------------------------------------------------------------------- #
# Section 1 -- existing screening behaviour is untouched
# --------------------------------------------------------------------------- #

def _screening_contract(n_basins: int) -> dict:
    ids = [f"{v:08d}" for v in range(n_basins)]
    return fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="fixture", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="fixture_gap_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256="a" * 64, package_file_checksums_sha256="b" * 64,
        package_run_provenance_sha256="c" * 64, development_split_sha256="d" * 64,
        spatial_holdout_split_sha256="e" * 64,
        per_basin_date={b: np.array([0, 1, 2]) for b in ids},
        per_basin_admitted={b: np.array([True, True, True]) for b in ids},
    )


def test_screening_contract_identifier_still_valid_and_unchanged():
    contract = _screening_contract(400)
    assert fixed.validate_fixed_support_contract(contract)["contract_id"] == OBJECTIVE_ID_V2
    assert contract["schema_name"] == "flashnh_stage1_v2_fixed_support_contract"
    ok = {
        "n_basins_requested": 400, "n_basins_evaluated": 400, "n_basins_excluded": 0,
        "per_basin": [{"basin_id": b, "nse": 0.1, "n_sim_nonfinite_at_admitted": 0} for b in contract["basin_ids"]],
        "aggregate": {"n_basins": 400, "metrics": {"nse": {"n_finite_basins": 400, "median": 0.1}}},
    }
    fixed._require_complete_production_fixed_support_population(ok, required_basin_ids=contract["basin_ids"])


@pytest.mark.parametrize("n", [399, 401])
def test_screening_gate_still_rejects_wrong_population_size(n):
    contract = _screening_contract(n)
    result = {
        "n_basins_requested": n, "n_basins_evaluated": n, "n_basins_excluded": 0,
        "per_basin": [{"basin_id": b, "nse": 0.1, "n_sim_nonfinite_at_admitted": 0} for b in contract["basin_ids"]],
        "aggregate": {"n_basins": n, "metrics": {"nse": {"n_finite_basins": n, "median": 0.1}}},
    }
    with pytest.raises(fixed.FixedSupportContractError, match="exactly 400 unique screening basins"):
        fixed._require_complete_production_fixed_support_population(result, required_basin_ids=contract["basin_ids"])


def test_objective_extractor_still_accepts_only_screening_fixed_support_scope():
    good = {"objective_scope": "fixed_support", "aggregate": {"metrics": {"nse": {"median": 0.42}}}}
    assert fixed.extract_v2_objective_from_fixed_support_result(good) == pytest.approx(0.42)
    for scope in ("natural_support", "devpop_audit", None):
        with pytest.raises(fixed.FixedSupportContractError):
            fixed.extract_v2_objective_from_fixed_support_result(
                {"objective_scope": scope, "aggregate": {"metrics": {"nse": {"median": 0.42}}}}
            )


def test_shared_neutral_primitives_are_the_same_objects_as_the_screening_internals():
    # Correction 5: the audit module reuses the screening module's neutral
    # encoding/accounting helpers by additive public alias -- not by importing
    # an underscore-prefixed name, and not by a private re-implementation.
    assert fixed.serialize_support_date_array is fixed._serialize_date_array
    assert fixed.deserialize_support_date_array is fixed._deserialize_date_array
    assert fixed.canonical_contract_checksum_payload is fixed._canonical_payload_for_checksum
    src = Path(audit.__file__).read_text(encoding="utf-8")
    for private in ("_serialize_date_array", "_deserialize_date_array", "_canonical_payload_for_checksum"):
        assert private not in src


# --------------------------------------------------------------------------- #
# Section 2 -- full-population diagnostic contract
# --------------------------------------------------------------------------- #

def test_explicit_development_audit_population_and_contract_pass():
    population = _population()
    contract = _contract(population)
    assert validate_devpop_audit_contract(contract) is contract
    assert contract["contract_id"] == DEVPOP_AUDIT_CONTRACT_ID
    assert contract["schema_name"] == "flashnh_stage1_devpop_common120_audit_contract"
    assert contract["diagnostic_only"] is True and contract["not_an_optimizer_objective"] is True
    assert contract["objective_scope"] == "devpop_audit"
    assert contract["basin_ids"] == sorted(POP_IDS)
    assert contract["expected_population_size"] == 6


def test_new_identifier_is_not_the_optimizer_objective_and_cannot_feed_the_extractor():
    assert DEVPOP_AUDIT_CONTRACT_ID != OBJECTIVE_ID_V2
    assert not DEVPOP_AUDIT_CONTRACT_ID.startswith("flashnh/")
    population = _population()
    contract = _contract(population)
    with pytest.raises(fixed.FixedSupportContractError, match="fixed_support-scoped"):
        fixed.extract_v2_objective_from_fixed_support_result(
            {"objective_scope": contract["objective_scope"],
             "aggregate": {"metrics": {"nse": {"median": 0.5}}}}
        )
    with pytest.raises(fixed.FixedSupportContractError):
        fixed.validate_fixed_support_contract(dict(contract))


def test_screening_identifier_cannot_masquerade_as_the_development_audit_contract():
    population = _population()
    contract = _contract(population)
    forged = dict(contract)
    forged["contract_id"] = OBJECTIVE_ID_V2
    forged["checksum_sha256"] = _checksum(forged)
    with pytest.raises(DevpopAuditContractError, match="contract_id must be"):
        validate_devpop_audit_contract(forged)


def test_size_hash_and_exact_ids_must_all_agree():
    population = _population()
    contract = _contract(population)

    wrong_size = dict(contract); wrong_size["expected_population_size"] = 5
    wrong_size["checksum_sha256"] = _checksum(wrong_size)
    with pytest.raises(DevpopAuditContractError, match="expected_population_size"):
        validate_devpop_audit_contract(wrong_size)

    wrong_hash = dict(contract); wrong_hash["membership_ids_sha256"] = "f" * 64
    wrong_hash["checksum_sha256"] = _checksum(wrong_hash)
    with pytest.raises(DevpopAuditContractError, match="canonical membership"):
        validate_devpop_audit_contract(wrong_hash)


def test_right_count_wrong_membership_fails():
    population = _population()
    with pytest.raises(DevpopAuditContractError, match="canonical membership"):
        ExpectedPopulationSpec(
            role="development_train",
            expected_size=len(POP_IDS),
            basin_ids=tuple(sorted(("99999999",) + POP_IDS[1:])),
            membership_ids_sha256=population.membership_ids_sha256,
        )


def test_duplicate_or_unsorted_membership_fails():
    with pytest.raises(DevpopAuditContractError, match="duplicate"):
        ExpectedPopulationSpec(
            role="development_train", expected_size=3,
            basin_ids=("00000001", "00000001", "00000002"),
            membership_ids_sha256=canonical_membership_sha256(("00000001", "00000002")),
        )
    with pytest.raises(DevpopAuditContractError, match="sorted"):
        ExpectedPopulationSpec(
            role="development_train", expected_size=2,
            basin_ids=("00000002", "00000001"),
            membership_ids_sha256=canonical_membership_sha256(("00000001", "00000002")),
        )


def test_missing_or_extra_evaluated_basins_fail_completeness():
    population = _population()
    contract = _contract(population)

    missing = _result(population, contract)
    missing["per_basin"] = missing["per_basin"][:-1]
    missing["evaluated_basin_ids"] = sorted(population.basin_ids)[:-1]
    missing["n_basins_evaluated"] = population.expected_size - 1
    with pytest.raises(DevpopAuditCompletenessError, match="evaluated basin IDs do not equal"):
        require_complete_synthetic_devpop_audit_population(missing, population=population, contract=contract)

    extra = _result(population, contract)
    extra["per_basin"].append(
        {"basin_id": "99999999", "nse": 0.5, "n_sim_nonfinite_at_admitted": 0, "n_admitted": 3}
    )
    extra["evaluated_basin_ids"] = sorted(list(population.basin_ids) + ["99999999"])
    extra["n_basins_evaluated"] = population.expected_size + 1
    with pytest.raises(DevpopAuditCompletenessError, match="evaluated basin IDs do not equal"):
        require_complete_synthetic_devpop_audit_population(extra, population=population, contract=contract)

    # evaluated_basin_ids that matches the expected population but disagrees
    # with the identities actually represented by per_basin (a duplicated row)
    dup = _result(population, contract)
    dup["per_basin"][-1] = dict(dup["per_basin"][0])
    with pytest.raises(DevpopAuditCompletenessError, match="does not match the identities represented by per_basin"):
        require_complete_synthetic_devpop_audit_population(dup, population=population, contract=contract)


def test_any_exclusion_blocks_canonical_completeness():
    population = _population()
    contract = _contract(population)
    excluded = _result(population, contract)
    excluded["basins_excluded"] = [{"basin_id": POP_IDS[0], "reason": "area derivation failed"}]
    excluded["n_basins_excluded"] = 1
    with pytest.raises(DevpopAuditCompletenessError, match="forbids any exclusion"):
        require_complete_synthetic_devpop_audit_population(excluded, population=population, contract=contract)


def test_missing_exclusion_receipt_blocks_completeness():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    del bad["basins_excluded"]
    with pytest.raises(DevpopAuditCompletenessError, match="missing the required basins_excluded"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_nonfinite_per_basin_nse_blocks_completeness():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    bad["per_basin"][2]["nse"] = float("nan")
    with pytest.raises(DevpopAuditCompletenessError, match="finite real raw-space NSE"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_bool_per_basin_nse_blocks_completeness():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    bad["per_basin"][0]["nse"] = True
    with pytest.raises(DevpopAuditCompletenessError, match="finite real raw-space NSE"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_nonfinite_simulation_at_admitted_support_blocks_completeness():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    bad["per_basin"][1]["n_sim_nonfinite_at_admitted"] = 4
    with pytest.raises(DevpopAuditCompletenessError, match="non-finite simulation"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_missing_per_basin_nonfinite_count_does_not_default_to_zero():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    del bad["per_basin"][0]["n_sim_nonfinite_at_admitted"]
    with pytest.raises(DevpopAuditCompletenessError, match="missing required field 'n_sim_nonfinite_at_admitted'"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_missing_or_wrong_per_basin_n_admitted_blocks_completeness():
    population = _population()
    contract = _contract(population)

    missing = _result(population, contract)
    del missing["per_basin"][0]["n_admitted"]
    with pytest.raises(DevpopAuditCompletenessError, match="missing required field 'n_admitted'"):
        require_complete_synthetic_devpop_audit_population(missing, population=population, contract=contract)

    wrong = _result(population, contract)
    wrong["per_basin"][0]["n_admitted"] = contract["eligible_counts"][population.basin_ids[0]] + 1
    with pytest.raises(DevpopAuditCompletenessError, match="does not equal the support"):
        require_complete_synthetic_devpop_audit_population(wrong, population=population, contract=contract)


@pytest.mark.parametrize("bad_value", [2307.0, True, "6", None])
def test_non_strict_integer_count_fields_fail_completeness(bad_value):
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    bad["n_basins_evaluated"] = bad_value
    with pytest.raises(DevpopAuditCompletenessError):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_missing_count_field_fails_completeness():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    del bad["n_basins_requested"]
    with pytest.raises(DevpopAuditCompletenessError, match="missing required accounting field 'n_basins_requested'"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_aggregate_totals_must_reconcile_with_per_basin_rows():
    population = _population()
    contract = _contract(population)

    bad_admitted = _result(population, contract)
    bad_admitted["aggregate"]["n_admitted_total"] += 1
    with pytest.raises(DevpopAuditCompletenessError, match="n_admitted_total"):
        require_complete_synthetic_devpop_audit_population(bad_admitted, population=population, contract=contract)

    bad_nonfinite = _result(population, contract)
    bad_nonfinite["aggregate"]["n_sim_nonfinite_at_admitted_total"] = 1
    with pytest.raises(DevpopAuditCompletenessError, match="non-finite-at-admitted total must be zero"):
        require_complete_synthetic_devpop_audit_population(bad_nonfinite, population=population, contract=contract)

    missing_total = _result(population, contract)
    del missing_total["aggregate"]["n_admitted_total"]
    with pytest.raises(DevpopAuditCompletenessError, match="missing required accounting field 'n_admitted_total'"):
        require_complete_synthetic_devpop_audit_population(missing_total, population=population, contract=contract)

    bad_finite = _result(population, contract)
    bad_finite["aggregate"]["metrics"]["nse"]["n_finite_basins"] = population.expected_size - 1
    with pytest.raises(DevpopAuditCompletenessError, match="finite-NSE basin count"):
        require_complete_synthetic_devpop_audit_population(bad_finite, population=population, contract=contract)


def test_checksum_mutation_fails():
    population = _population()
    contract = _contract(population)
    mutated = dict(contract)
    # mutate a free-form field that is not otherwise pinned, so the checksum
    # recomputation is the check that fires
    mutated["monotone_nesting_justification"] = mutated["monotone_nesting_justification"] + " (tampered)"
    with pytest.raises(DevpopAuditContractError, match="checksum mismatch"):
        validate_devpop_audit_contract(mutated)


def test_period_test_or_impossible_dates_fail():
    with pytest.raises(DevpopAuditContractError, match="period must be"):
        ExpectedPopulationSpec.for_synthetic_fixture(
            role="development_train", basin_ids=list(POP_IDS), period="test"
        )
    with pytest.raises(DevpopAuditContractError, match="full frozen validation year"):
        ExpectedPopulationSpec.for_synthetic_fixture(
            role="development_train", basin_ids=list(POP_IDS),
            date_start="2025-01-01", date_end="2025-12-31",
        )
    population = _population()
    contract = _contract(population)
    tampered = dict(contract); tampered["date_end"] = "2025-12-31"
    tampered["checksum_sha256"] = _checksum(tampered)
    with pytest.raises(DevpopAuditContractError, match="frozen validation year"):
        validate_devpop_audit_contract(tampered)


@pytest.mark.parametrize("bad_date", ["2024-02-31", "2024-13-01", "2024-00-10", "2024-1-1", "2024-06", "not-a-date"])
def test_impossible_or_truncated_dates_are_rejected_by_a_real_calendar_parser(bad_date):
    with pytest.raises(DevpopAuditContractError):
        ExpectedPopulationSpec.for_synthetic_fixture(
            role="development_train", basin_ids=list(POP_IDS),
            date_start=bad_date, date_end="2024-12-31",
        )


@pytest.mark.parametrize("sub_start,sub_end", [("2024-01-01", "2024-06-30"), ("2024-02-01", "2024-12-31")])
def test_subintervals_of_2024_are_rejected(sub_start, sub_end):
    with pytest.raises(DevpopAuditContractError, match="full frozen validation year"):
        ExpectedPopulationSpec.for_synthetic_fixture(
            role="development_train", basin_ids=list(POP_IDS),
            date_start=sub_start, date_end=sub_end,
        )


@pytest.mark.parametrize("role", [
    "spatial_holdout_nonca", "california_all", "california_holdout", "temporal_test", "development_test",
])
def test_spatial_holdout_and_california_and_test_roles_fail(role):
    with pytest.raises(DevpopAuditContractError):
        ExpectedPopulationSpec.for_synthetic_fixture(role=role, basin_ids=list(POP_IDS))


def test_good_result_passes_completeness_and_returns_receipt():
    population = _population()
    contract = _contract(population)
    receipt = require_complete_synthetic_devpop_audit_population(
        _result(population, contract), population=population, contract=contract
    )
    # a synthetic fixture population can only ever earn a FIXTURE receipt --
    # never the canonical labels, which only the mandatory canonical gate emits
    assert receipt["fixture_completeness"] is True
    assert "canonical_completeness" not in receipt
    assert "canonical_population_verified" not in receipt
    assert receipt["n_expected"] == receipt["n_evaluated"] == 6
    assert receipt["objective_scope"] == "devpop_audit"
    assert receipt["evaluated_basin_ids"] == sorted(POP_IDS)


def test_completeness_rejects_fixed_support_scoped_result():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract, objective_scope="fixed_support")
    with pytest.raises(DevpopAuditCompletenessError, match="objective_scope"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_completeness_rejects_population_contract_identity_mismatch():
    population = _population()
    other = _population(ids=tuple(f"{i:08d}" for i in range(6, 12)))
    contract = _contract(population)
    with pytest.raises(DevpopAuditCompletenessError, match="expected population"):
        require_complete_synthetic_devpop_audit_population(
            _result(other, contract, contract_checksum_sha256=contract["checksum_sha256"]),
            population=other, contract=contract,
        )


# --------------------------------------------------------------------------- #
# Section 2b -- canonical vs synthetic population boundary (Correction 1)
# --------------------------------------------------------------------------- #

def test_synthetic_fixture_population_is_rejected_by_the_canonical_validator():
    synthetic = _population()
    with pytest.raises(DevpopAuditContractError, match="exactly 2307 basins"):
        assert_canonical_development_population(synthetic)


def test_arbitrary_population_labelled_development_train_is_not_canonical():
    arbitrary = ExpectedPopulationSpec.for_synthetic_fixture(
        role="development_train",
        basin_ids=[f"{i:08d}" for i in range(2307)],  # right size, wrong identities
    )
    assert arbitrary.expected_size == 2307
    with pytest.raises(DevpopAuditContractError, match="canonical development population"):
        assert_canonical_development_population(arbitrary)


def test_wrong_membership_with_plausible_artifact_hash_still_fails_canonical():
    spoof = ExpectedPopulationSpec.for_synthetic_fixture(
        role="development_train",
        basin_ids=[f"{i:08d}" for i in range(2307)],
        membership_artifact_sha256=DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256,
        membership_artifact_provenance="config/stage1_baseline_splits_v001/split_manifest.json",
    )
    with pytest.raises(DevpopAuditContractError, match="membership hash"):
        assert_canonical_development_population(spoof)


def test_for_development_train_loads_the_committed_canonical_population():
    spec = ExpectedPopulationSpec.for_development_train(_SPLITS_DIR)
    assert spec.expected_size == 2307
    assert spec.membership_ids_sha256 == DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256
    assert spec.role == "development_train"
    assert spec.date_start == "2024-01-01" and spec.date_end == "2024-12-31"
    # idempotent under the canonical validator
    assert_canonical_development_population(spec)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_canonical_membership_is_line_ending_independent(tmp_path, newline):
    ids = audit_split_ids()
    d = tmp_path / "splits"
    d.mkdir()
    (d / "development_train.txt").write_bytes(newline.join(ids).encode("utf-8") + newline.encode("utf-8"))
    (d / "split_manifest.json").write_text(json.dumps({
        "counts": {"development_train": 2307},
        "artifact_sha256": {"development_train.txt": DEVELOPMENT_TRAIN_SPLIT_MANIFEST_ARTIFACT_SHA256},
    }), encoding="utf-8")
    spec = ExpectedPopulationSpec.for_development_train(d)
    assert spec.membership_ids_sha256 == DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256
    assert_canonical_development_population(spec)


def audit_split_ids():
    from src.baseline.splits import load_eligible_basins
    return load_eligible_basins(_SPLITS_DIR / "development_train.txt")


def test_for_development_train_rejects_a_changed_split_manifest_hash(tmp_path):
    ids = audit_split_ids()
    d = tmp_path / "splits"
    d.mkdir()
    (d / "development_train.txt").write_bytes(("\n".join(ids) + "\n").encode("utf-8"))
    (d / "split_manifest.json").write_text(json.dumps({
        "counts": {"development_train": 2307},
        "artifact_sha256": {"development_train.txt": "0" * 64},
    }), encoding="utf-8")
    with pytest.raises(DevpopAuditContractError, match="split-manifest provenance"):
        ExpectedPopulationSpec.for_development_train(d)


# --------------------------------------------------------------------------- #
# Section 2c -- pinned scientific-identity fields (Correction 4)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,value", [
    ("target_variable", "qobs_mm_per_h_lead12"),
    ("lead_hours", 12),
    ("seq_length_floor", 48),
    ("source_gap_policy_identity", "some_other_policy_v009"),
])
def test_build_rejects_wrong_scientific_identity_fields(field, value):
    population = _population()
    kwargs = dict(
        population=population,
        target_variable=CANONICAL_TARGET_VARIABLE,
        source_gap_policy_identity=CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
        per_basin_date={b: np.arange(5, dtype="int64") for b in population.basin_ids},
        per_basin_admitted={b: np.array([True, True, True, False, False]) for b in population.basin_ids},
        **IDENT,
    )
    kwargs[field] = value
    with pytest.raises(DevpopAuditContractError):
        build_devpop_audit_contract(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("target_variable", "qobs_mm_per_h_lead12"),
    ("source_gap_policy_identity", "some_other_policy_v009"),
    ("lead_hours", 12),
    ("seq_length_floor", 48),
])
def test_validate_rejects_wrong_scientific_identity_fields(field, value):
    population = _population()
    contract = _contract(population)
    tampered = dict(contract)
    tampered[field] = value
    tampered["checksum_sha256"] = _checksum(tampered)
    with pytest.raises(DevpopAuditContractError):
        validate_devpop_audit_contract(tampered)


@pytest.mark.parametrize("field", ["schema_version", "seq_length_floor", "lead_hours", "expected_population_size"])
@pytest.mark.parametrize("bad", [True, 1.0])
def test_validate_rejects_non_strict_integer_identity_fields(field, bad):
    population = _population()
    contract = _contract(population)
    tampered = dict(contract)
    tampered[field] = bad
    tampered["checksum_sha256"] = _checksum(tampered)
    with pytest.raises(DevpopAuditContractError):
        validate_devpop_audit_contract(tampered)


def test_lead6_end_of_validation_boundary_is_exact(monkeypatch, tmp_path):
    # Correction 4: lead-6 final-admitted-issue-time boundary.
    # validation window end 2024-01-01 -> package convention +23h -> 2024-01-01T23:00;
    # a lead-6 forecast issued at t must satisfy t + 6h <= 2024-01-01T23:00,
    # so the last admissible issue time is exactly 2024-01-01T17:00.
    #
    # A synthetic ExpectedPopulationSpec is structurally pinned to the full
    # frozen 2024 window, so the shortened-window boundary is exercised through
    # the screening builder, whose window is read directly from the policy.
    ids = [f"{i:08d}" for i in range(4)]
    package, _ = _install_builder_fixture(monkeypatch, tmp_path, ids=ids, val_end="2024-01-01")
    result = builder.build_common120_support(
        package_root=package, splits_dir=tmp_path, screening_basin_ids_path="x",
        baseline_policy_path="p", policy_overlay_path="o",
    )
    support = pd.DatetimeIndex(
        fixed.deserialize_support_date_array(
            result.contract["per_basin_support"][sorted(ids)[0]],
            result.contract["date_dtype"],
        )
    )
    assert support.max() == pd.Timestamp("2024-01-01T17:00")
    assert pd.Timestamp("2024-01-01T18:00") not in support


# --------------------------------------------------------------------------- #
# Section 2d -- ONE fail-closed canonical contract boundary (Corrections 1-3),
# adversarial coverage.
# --------------------------------------------------------------------------- #

_AUTHORITATIVE_IDENT = {
    "package_manifest_sha256": adapter.PACKAGE_MANIFEST_SHA256,
    "package_file_checksums_sha256": adapter.PACKAGE_FILE_CHECKSUMS_SHA256,
    "package_run_provenance_sha256": adapter.PACKAGE_RUN_PROVENANCE_SHA256,
    "development_split_sha256": adapter.DEVELOPMENT_SPLIT_SHA256,
    "spatial_holdout_split_sha256": adapter.SPATIAL_HOLDOUT_SPLIT_SHA256,
}


def _load_split_ids(filename):
    from src.baseline.splits import load_eligible_basins

    return load_eligible_basins(_SPLITS_DIR / filename)


def _materialize_split_dir(dirpath, dev_ids, holdout_ids, newline):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "development_train.txt").write_bytes((newline.join(dev_ids) + newline).encode("utf-8"))
    (dirpath / "spatial_holdout_nonca.txt").write_bytes((newline.join(holdout_ids) + newline).encode("utf-8"))
    return dirpath


def _canonical_contract(population, *, n_admitted=3, timeline=5, **kw):
    """A real canonical audit contract: the pinned 2,307-basin development
    population + the authoritative production package/split identities."""
    dates = np.arange(timeline, dtype="int64")
    admitted = np.array([i < n_admitted for i in range(timeline)])
    return build_devpop_audit_contract(
        population=population,
        target_variable=CANONICAL_TARGET_VARIABLE,
        source_gap_policy_identity=CANONICAL_SOURCE_GAP_POLICY_IDENTITY,
        per_basin_date={b: dates for b in population.basin_ids},
        per_basin_admitted={b: admitted for b in population.basin_ids},
        **{**_AUTHORITATIVE_IDENT, **kw},
    )


@pytest.fixture(scope="module")
def canonical_population():
    return ExpectedPopulationSpec.for_development_train(_SPLITS_DIR)


@pytest.fixture(scope="module")
def canonical_contract(canonical_population):
    return _canonical_contract(canonical_population)


# -- adversarial #1 / #2: real membership, bad provenance evidence ----------- #

def test_canonical_population_requires_structured_provenance_and_artifact_hash():
    ids = _load_split_ids("development_train.txt")
    base = dict(
        role=DEVELOPMENT_TRAIN_ROLE,
        expected_size=EXPECTED_DEVELOPMENT_POPULATION_SIZE,
        basin_ids=tuple(sorted(ids)),
        membership_ids_sha256=DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256,
    )
    with pytest.raises(DevpopAuditContractError, match="membership_artifact_provenance"):
        assert_canonical_development_population(ExpectedPopulationSpec(**base))
    with pytest.raises(DevpopAuditContractError, match="membership_artifact_sha256 must record"):
        assert_canonical_development_population(
            ExpectedPopulationSpec(**base, membership_artifact_provenance=dict(CANONICAL_MEMBERSHIP_ARTIFACT_PROVENANCE))
        )


def test_canonical_validator_rejects_missing_membership_artifact_hash(canonical_contract):
    bad = dict(canonical_contract)
    bad["membership_artifact_sha256"] = None
    bad["checksum_sha256"] = _checksum(bad)
    with pytest.raises(DevpopAuditContractError, match="membership_artifact_sha256 must record"):
        validate_canonical_devpop_audit_contract(bad)


def test_canonical_validator_rejects_free_form_provenance(canonical_contract):
    bad = dict(canonical_contract)
    bad["membership_artifact_provenance"] = "see config/stage1_baseline_splits_v001/split_manifest.json"
    bad["checksum_sha256"] = _checksum(bad)
    with pytest.raises(DevpopAuditContractError, match="pinned structured"):
        validate_canonical_devpop_audit_contract(bad)


# -- adversarial #3 / #4 / #5 / #6: wrong-but-well-formed package/split hashes  #

@pytest.mark.parametrize("field", [
    "package_manifest_sha256",
    "package_file_checksums_sha256",
    "package_run_provenance_sha256",
    "development_split_sha256",
    "spatial_holdout_split_sha256",
])
def test_canonical_validator_rejects_wrong_but_well_formed_package_or_split_hash(field, canonical_population):
    bad = _canonical_contract(canonical_population, **{field: "0" * 64})
    # generic validation is satisfied (it is a syntactically valid SHA-256) ...
    assert validate_devpop_audit_contract(bad) is bad
    # ... but the one canonical boundary pins the exact authoritative value
    with pytest.raises(DevpopAuditContractError, match="wrong-but-well-formed package/split hash is not canonical"):
        validate_canonical_devpop_audit_contract(bad)


def test_synthetic_contract_reproducing_only_membership_ids_is_not_canonical():
    # a synthetic spec that happens to carry the real 2,307 membership hash
    # (right basin_ids) but arbitrary package/split identities
    ids = _load_split_ids("development_train.txt")
    population = ExpectedPopulationSpec.for_synthetic_fixture(role="development_train", basin_ids=list(ids))
    assert population.membership_ids_sha256 == DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256
    contract = _contract(population)  # arbitrary IDENT hashes, no provenance
    assert validate_devpop_audit_contract(contract) is contract
    with pytest.raises(DevpopAuditContractError):
        validate_canonical_devpop_audit_contract(contract)


# -- adversarial #7 / #8 / #9: canonical vs synthetic completeness labels ---- #

def test_generic_completeness_cannot_emit_canonical_labels():
    population = _population()
    contract = _contract(population)
    receipt = require_complete_synthetic_devpop_audit_population(
        _result(population, contract), population=population, contract=contract
    )
    assert receipt.get("fixture_completeness") is True
    assert "canonical_completeness" not in receipt
    assert "canonical_population_verified" not in receipt


def test_canonical_completeness_gate_has_no_opt_out_parameter():
    sig = inspect.signature(require_complete_devpop_audit_population)
    assert set(sig.parameters) == {"result", "population", "contract"}
    src = Path(audit.__file__).read_text(encoding="utf-8")
    assert "require_canonical_population" not in src


def test_canonical_completeness_gate_enforces_canonical_population_and_contract(canonical_population):
    # a synthetic fixture population cannot pass the mandatory canonical gate
    syn = _population()
    with pytest.raises(DevpopAuditContractError):
        require_complete_devpop_audit_population(
            _result(syn, _contract(syn)), population=syn, contract=_contract(syn)
        )
    # the real population but a contract whose package identity is not authoritative
    bad_contract = _canonical_contract(canonical_population, package_manifest_sha256="0" * 64)
    with pytest.raises(DevpopAuditContractError, match="not canonical"):
        require_complete_devpop_audit_population(
            _result(canonical_population, bad_contract),
            population=canonical_population, contract=bad_contract,
        )


def test_successful_canonical_completeness_uses_canonical_enforcement(canonical_population, canonical_contract):
    receipt = require_complete_devpop_audit_population(
        _result(canonical_population, canonical_contract),
        population=canonical_population, contract=canonical_contract,
    )
    assert receipt["canonical_completeness"] is True
    assert receipt["canonical_population_verified"] is True
    assert receipt["n_expected"] == receipt["n_evaluated"] == EXPECTED_DEVELOPMENT_POPULATION_SIZE
    assert receipt["evaluated_basin_ids"] == sorted(canonical_population.basin_ids)


# -- adversarial #10 / #11 / #12: explicit evaluated identities are mandatory  #

def test_missing_evaluated_basin_ids_fails_completeness():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    del bad["evaluated_basin_ids"]
    with pytest.raises(DevpopAuditCompletenessError, match="explicit evaluated_basin_ids"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_evaluated_basin_ids_must_agree_with_per_basin_rows():
    population = _population()
    contract = _contract(population)
    bad = _result(population, contract)
    bad["per_basin"][-1] = dict(bad["per_basin"][0])  # a duplicated identity in per_basin
    with pytest.raises(DevpopAuditCompletenessError, match="does not match the identities represented by per_basin"):
        require_complete_synthetic_devpop_audit_population(bad, population=population, contract=contract)


def test_duplicate_or_reordered_evaluated_basin_ids_fail():
    population = _population()
    contract = _contract(population)

    dup = _result(population, contract)
    dup["evaluated_basin_ids"] = sorted(population.basin_ids)
    dup["evaluated_basin_ids"][0] = dup["evaluated_basin_ids"][1]
    with pytest.raises(DevpopAuditCompletenessError, match="duplicates"):
        require_complete_synthetic_devpop_audit_population(dup, population=population, contract=contract)

    reordered = _result(population, contract)
    reordered["evaluated_basin_ids"] = list(reversed(sorted(population.basin_ids)))
    with pytest.raises(DevpopAuditCompletenessError, match="must be sorted"):
        require_complete_synthetic_devpop_audit_population(reordered, population=population, contract=contract)


# -- adversarial #13 / #14 / #15: LF/CRLF-independent audit split identity ---- #

@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_audit_split_identity_verification_is_line_ending_independent(tmp_path, newline):
    dev = _load_split_ids("development_train.txt")
    holdout = _load_split_ids("spatial_holdout_nonca.txt")
    d = _materialize_split_dir(tmp_path / "splits", dev, holdout, newline)
    verified = builder._verify_audit_split_identities(d)
    assert verified["development_split_sha256"] == adapter.DEVELOPMENT_SPLIT_SHA256
    assert verified["spatial_holdout_split_sha256"] == adapter.SPATIAL_HOLDOUT_SPLIT_SHA256


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_audit_split_identity_verification_fails_on_changed_membership(tmp_path, newline):
    dev = _load_split_ids("development_train.txt")
    holdout = _load_split_ids("spatial_holdout_nonca.txt")
    changed = sorted(dev[1:] + ["99999999"])
    d = _materialize_split_dir(tmp_path / "splits", changed, holdout, newline)
    with pytest.raises(builder.Common120SupportError, match="split membership changed"):
        builder._verify_audit_split_identities(d)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_canonical_audit_identity_stage_accepts_lf_and_crlf_and_rejects_changed_membership(
    monkeypatch, tmp_path, newline
):
    dev = _load_split_ids("development_train.txt")
    holdout = _load_split_ids("spatial_holdout_nonca.txt")
    pkg_ident = {
        "package_manifest_sha256": adapter.PACKAGE_MANIFEST_SHA256,
        "package_file_checksums_sha256": adapter.PACKAGE_FILE_CHECKSUMS_SHA256,
        "package_run_provenance_sha256": adapter.PACKAGE_RUN_PROVENANCE_SHA256,
    }
    # ONLY the raw-bytes package-payload check is stubbed; the split-identity
    # stage under test runs for real end to end.
    monkeypatch.setattr(builder, "_verify_audit_package_payload_identities", lambda _: dict(pkg_ident))

    ok = _materialize_split_dir(tmp_path / "ok", dev, holdout, newline)
    identities = builder._verify_audit_artifact_identities(
        adapter.PreparationPaths(Path("p"), tmp_path / "pkg", ok, ok / "development_train.txt")
    )
    assert identities == {
        **pkg_ident,
        "development_split_sha256": adapter.DEVELOPMENT_SPLIT_SHA256,
        "spatial_holdout_split_sha256": adapter.SPATIAL_HOLDOUT_SPLIT_SHA256,
    }

    changed = _materialize_split_dir(
        tmp_path / "changed", sorted(dev[1:] + ["99999999"]), holdout, newline
    )
    with pytest.raises(builder.Common120SupportError, match="split membership changed"):
        builder._verify_audit_artifact_identities(
            adapter.PreparationPaths(Path("p"), tmp_path / "pkg", changed, changed / "development_train.txt")
        )


# -- adversarial #16 / #17: canonical write / load fail closed --------------- #

def test_canonical_write_and_load_validate_authoritative_identities(tmp_path, canonical_contract):
    out = tmp_path / "canonical_audit_support.json"
    write_devpop_audit_contract(canonical_contract, out)
    loaded = load_devpop_audit_contract(out)
    assert loaded["contract_id"] == DEVPOP_AUDIT_CONTRACT_ID
    assert loaded["membership_ids_sha256"] == DEVELOPMENT_TRAIN_MEMBERSHIP_IDS_SHA256
    # the canonical writer refuses a synthetic (non-canonical) contract
    with pytest.raises(DevpopAuditContractError):
        write_devpop_audit_contract(_contract(_population()), tmp_path / "syn.json")


def test_generic_artifact_cannot_be_loaded_through_the_canonical_loader(tmp_path):
    out = tmp_path / "syn_audit_support.json"
    write_synthetic_devpop_audit_contract(_contract(_population()), out)
    assert load_synthetic_devpop_audit_contract(out)["contract_id"] == DEVPOP_AUDIT_CONTRACT_ID
    with pytest.raises(DevpopAuditContractError):
        load_devpop_audit_contract(out)


# --------------------------------------------------------------------------- #
# Section 3 -- builder behaviour
# --------------------------------------------------------------------------- #

def _install_builder_fixture(monkeypatch, tmp_path, *, ids, gaps=(), qobs=None, val_end="2024-12-31"):
    package = tmp_path / "package"
    (package / "masks").mkdir(parents=True)
    gap_path = package / "masks" / "gap_timestamps.json"
    gap_path.write_text(json.dumps([pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%SZ") for x in gaps]))
    dates = pd.date_range("2023-12-27", periods=200, freq="h")
    values = np.ones(len(dates)) if qobs is None else np.asarray(qobs, dtype=float)
    manifest = {
        "gap_product_scope": [MRMS_PRODUCT, RTMA_PRODUCT],
        "gap_timestamp_artifact": {"sha256": hashlib.sha256(gap_path.read_bytes()).hexdigest()},
    }
    policy = {
        "temporal_split": {"validation": {"start": "2024-01-01", "end": val_end}},
        "gap_policy": {"include_rtma_in_history_mask": True},
    }
    monkeypatch.setattr(builder, "load_stage1_baseline_policy", lambda _: policy)
    monkeypatch.setattr(builder, "load_stage1_baseline_policy_v2_six_axis", lambda *_: policy)
    monkeypatch.setattr(builder, "_verify_artifact_identities", lambda _: dict(IDENT))
    monkeypatch.setattr(builder, "_verify_audit_artifact_identities", lambda _: dict(IDENT))
    monkeypatch.setattr(builder, "read_package_manifest", lambda _: manifest)
    monkeypatch.setattr(
        builder, "validate_full_population_basin_membership",
        lambda *_: type("M", (), {"development_basins": list(ids)})(),
    )
    monkeypatch.setattr(builder, "load_screening_basin_ids", lambda *_, **__: sorted(ids))
    monkeypatch.setattr(builder, "_dates_and_target", lambda *_: (dates, values.copy()))
    return package, dates


def test_builder_accepts_explicit_development_audit_population_and_records_every_basin(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(8)]
    package, _ = _install_builder_fixture(monkeypatch, tmp_path, ids=ids)
    population = _population(ids=ids)
    result = builder.build_common120_support_for_population(
        population=population, package_root=package, splits_dir=tmp_path,
        baseline_policy_path="p", policy_overlay_path="o",
    )
    contract = result.contract
    assert contract["contract_id"] == DEVPOP_AUDIT_CONTRACT_ID
    assert contract["basin_ids"] == sorted(ids)
    assert set(contract["eligible_counts"]) == set(ids)
    assert all(contract["eligible_counts"][b] > 0 for b in ids)
    assert result.accounting["zero_support_basin_count"] == 0
    assert result.accounting["diagnostic_only"] is True
    validate_devpop_audit_contract(contract)


def test_builder_uses_the_shared_120h_predicate_and_agrees_with_screening_builder(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(400)]

    pkg_s, _ = _install_builder_fixture(monkeypatch, tmp_path / "s", ids=ids, gaps=["2023-12-27T05:00:00Z"])
    screening = builder.build_common120_support(
        package_root=pkg_s, splits_dir=tmp_path / "s", screening_basin_ids_path="x",
        baseline_policy_path="p", policy_overlay_path="o",
    )

    pkg_d, _ = _install_builder_fixture(monkeypatch, tmp_path / "d", ids=ids, gaps=["2023-12-27T05:00:00Z"])
    population = _population(ids=ids)
    devpop = builder.build_common120_support_for_population(
        population=population, package_root=pkg_d, splits_dir=tmp_path / "d",
        baseline_policy_path="p", policy_overlay_path="o",
    )

    assert devpop.contract["per_basin_support"] == screening.contract["per_basin_support"]
    assert devpop.contract["eligible_counts"] == screening.contract["eligible_counts"]
    assert devpop.contract["date_dtype"] == screening.contract["date_dtype"]
    assert devpop.contract["contract_id"] != screening.contract["contract_id"]
    assert devpop.contract["schema_name"] != screening.contract["schema_name"]


def test_builder_rejects_population_membership_contradiction(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(8)]
    package, _ = _install_builder_fixture(monkeypatch, tmp_path, ids=ids)
    population = _population(ids=[f"{i:08d}" for i in range(1, 9)])
    with pytest.raises(builder.Common120SupportError, match="does not equal the package-verified"):
        builder.build_common120_support_for_population(
            population=population, package_root=package, splits_dir=tmp_path,
            baseline_policy_path="p", policy_overlay_path="o",
        )


def test_builder_rejects_validation_window_mismatch(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(8)]
    package, _ = _install_builder_fixture(monkeypatch, tmp_path, ids=ids, val_end="2024-06-30")
    population = _population(ids=ids)  # defaults to 2024-12-31
    with pytest.raises(builder.Common120SupportError, match="does not match the expected population window"):
        builder.build_common120_support_for_population(
            population=population, package_root=package, splits_dir=tmp_path,
            baseline_policy_path="p", policy_overlay_path="o",
        )


def test_builder_reports_zero_support_basin_explicitly(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(4)]
    package, dates = _install_builder_fixture(monkeypatch, tmp_path, ids=ids)
    monkeypatch.setattr(builder, "_dates_and_target", lambda *_: (dates, np.full(len(dates), np.nan)))
    population = _population(ids=ids)
    with pytest.raises(builder.Common120SupportError, match="zero Common-120 support and are named explicitly"):
        builder.build_common120_support_for_population(
            population=population, package_root=package, splits_dir=tmp_path,
            baseline_policy_path="p", policy_overlay_path="o",
        )


def test_builder_preserves_no_overwrite_on_write(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(8)]
    package, _ = _install_builder_fixture(monkeypatch, tmp_path, ids=ids)
    population = _population(ids=ids)
    contract = builder.build_common120_support_for_population(
        population=population, package_root=package, splits_dir=tmp_path,
        baseline_policy_path="p", policy_overlay_path="o",
    ).contract
    out = tmp_path / "audit_support.json"
    # the builder fixture verifies synthetic identities, so this is a synthetic
    # (non-canonical) contract -- it round-trips only through the synthetic I/O
    write_synthetic_devpop_audit_contract(contract, out)
    assert load_synthetic_devpop_audit_contract(out)["contract_id"] == DEVPOP_AUDIT_CONTRACT_ID
    with pytest.raises(DevpopAuditContractError, match="refusing to overwrite"):
        write_synthetic_devpop_audit_contract(contract, out)


def test_canonical_builder_has_no_population_parameter_and_pins_the_real_identity(monkeypatch, tmp_path):
    # Correction 1 / adversarial #17: the canonical entry point exposes no
    # `population` argument, so a generic / synthetic spec cannot be injected.
    sig = inspect.signature(builder.build_common120_support_for_development_population)
    assert "population" not in sig.parameters
    # and it routes through the committed canonical constructor (which fails
    # cleanly against an empty tmp splits dir -- never silently synthesises one)
    with pytest.raises(DevpopAuditContractError):
        builder.build_common120_support_for_development_population(
            package_root=tmp_path, splits_dir=tmp_path,
            baseline_policy_path="p", policy_overlay_path="o",
        )


def test_existing_screening_builder_behaviour_is_unchanged(monkeypatch, tmp_path):
    ids = [f"{i:08d}" for i in range(400)]
    package, _ = _install_builder_fixture(monkeypatch, tmp_path, ids=ids, gaps=["2023-12-27T01:00:00Z"])
    result = builder.build_common120_support(
        package_root=package, splits_dir=tmp_path, screening_basin_ids_path="x",
        baseline_policy_path="p", policy_overlay_path="o",
    )
    assert result.contract["contract_id"] == OBJECTIVE_ID_V2
    assert len(result.contract["basin_ids"]) == 400
    assert result.accounting["n_basins"] == 400


# --------------------------------------------------------------------------- #
# Section 3b -- frozen screening builder regression against pre-patch HEAD
# --------------------------------------------------------------------------- #

# Authoritative values captured from `git show HEAD:...common120_support_builder.py`
# run against the deterministic fixture below (see
# .scratch_local/_capture_head_screening.py).  These literals are the pre-patch
# contract; the patched builder must reproduce them byte-for-byte.
_HEAD_SCREENING_CHECKSUM = "2d0aa0ae3233845a91bbb07227694934a229a519635aa70b0b83126040d6f66b"
_HEAD_SCREENING_DATE_DTYPE = "datetime64"
_HEAD_SCREENING_SUPPORT_0 = [
    "2024-01-01T00:00:00.000000000", "2024-01-01T01:00:00.000000000",
    "2024-01-01T02:00:00.000000000", "2024-01-01T03:00:00.000000000",
    "2024-01-01T04:00:00.000000000", "2024-01-01T05:00:00.000000000",
    "2024-01-01T06:00:00.000000000", "2024-01-01T07:00:00.000000000",
    "2024-01-01T08:00:00.000000000", "2024-01-01T09:00:00.000000000",
    "2024-01-01T10:00:00.000000000", "2024-01-01T11:00:00.000000000",
    "2024-01-01T12:00:00.000000000", "2024-01-01T13:00:00.000000000",
    "2024-01-01T14:00:00.000000000", "2024-01-01T15:00:00.000000000",
    "2024-01-01T16:00:00.000000000", "2024-01-01T17:00:00.000000000",
]
_HEAD_SCREENING_ELIGIBLE = {f"{i:08d}": 18 for i in range(5)}
_HEAD_SCREENING_ACCOUNTING = {
    "n_basins": 5, "global_validation_issue_times": 18, "total_retained": 90,
    "per_basin_retained": {f"{i:08d}": 18 for i in range(5)},
    "min_retained": 18, "max_retained": 18, "median_retained": 18.0, "gap_count": 1,
}


def _install_head_regression_fixture(monkeypatch, tmp):
    ids = [f"{i:08d}" for i in range(5)]
    tmp = Path(tmp)
    package = tmp / "package"
    (package / "masks").mkdir(parents=True)
    gap_path = package / "masks" / "gap_timestamps.json"
    gap_path.write_text(json.dumps(["2023-12-26T00:00:00Z"]))
    dates = pd.date_range("2023-12-25", periods=200, freq="h")
    values = np.ones(len(dates))
    manifest = {
        "gap_product_scope": [MRMS_PRODUCT, RTMA_PRODUCT],
        "gap_timestamp_artifact": {"sha256": hashlib.sha256(gap_path.read_bytes()).hexdigest()},
    }
    policy = {
        "temporal_split": {"validation": {"start": "2024-01-01", "end": "2024-01-01"}},
        "gap_policy": {"include_rtma_in_history_mask": True},
    }
    monkeypatch.setattr(builder, "load_stage1_baseline_policy", lambda _: policy)
    monkeypatch.setattr(builder, "load_stage1_baseline_policy_v2_six_axis", lambda *_: policy)
    monkeypatch.setattr(builder, "_verify_artifact_identities", lambda _: dict(IDENT))
    monkeypatch.setattr(builder, "read_package_manifest", lambda _: manifest)
    monkeypatch.setattr(
        builder, "validate_full_population_basin_membership",
        lambda *_: type("M", (), {"development_basins": list(ids)})(),
    )
    monkeypatch.setattr(builder, "load_screening_basin_ids", lambda *a, **k: sorted(ids))
    monkeypatch.setattr(builder, "_dates_and_target", lambda *a: (dates, values.copy()))
    return package, tmp, ids


def test_frozen_screening_builder_output_matches_pre_patch_head_exactly(monkeypatch, tmp_path):
    package, tmp, ids = _install_head_regression_fixture(monkeypatch, tmp_path)
    result = builder.build_common120_support(
        package_root=package, splits_dir=tmp, screening_basin_ids_path="x",
        baseline_policy_path="p", policy_overlay_path="o",
    )
    contract = result.contract
    assert contract["checksum_sha256"] == _HEAD_SCREENING_CHECKSUM
    assert contract["date_dtype"] == _HEAD_SCREENING_DATE_DTYPE
    assert contract["per_basin_support"][ids[0]] == _HEAD_SCREENING_SUPPORT_0
    assert contract["eligible_counts"] == _HEAD_SCREENING_ELIGIBLE
    for key, expected in _HEAD_SCREENING_ACCOUNTING.items():
        assert result.accounting[key] == expected


def test_screening_builder_loading_order_preserves_first_basin_error_priority(monkeypatch, tmp_path):
    # Correction 6 / review finding: when BOTH the first basin input and the
    # gap-JSON input are invalid, the historical screening order surfaces the
    # basin error first (gap-artifact checksum -> first-basin read -> gap parse).
    package, tmp, ids = _install_head_regression_fixture(monkeypatch, tmp_path)
    gap_path = package / "masks" / "gap_timestamps.json"
    gap_path.write_text("{ this is not valid json")
    # keep the manifest checksum consistent with the (now invalid) bytes so the
    # checksum gate passes and control reaches the ordered reads
    manifest = {
        "gap_product_scope": [MRMS_PRODUCT, RTMA_PRODUCT],
        "gap_timestamp_artifact": {"sha256": hashlib.sha256(gap_path.read_bytes()).hexdigest()},
    }
    monkeypatch.setattr(builder, "read_package_manifest", lambda _: manifest)

    def _boom(*_a, **_k):
        raise builder.Common120SupportError("missing basin NetCDF: <fixture>")

    monkeypatch.setattr(builder, "_dates_and_target", _boom)
    with pytest.raises(builder.Common120SupportError, match="missing basin NetCDF"):
        builder.build_common120_support(
            package_root=package, splits_dir=tmp, screening_basin_ids_path="x",
            baseline_policy_path="p", policy_overlay_path="o",
        )


# --------------------------------------------------------------------------- #
# Section 4 -- structural guards
# --------------------------------------------------------------------------- #

def test_module_has_no_wandb_or_sweep_execution_dependency():
    src = Path(audit.__file__).read_text(encoding="utf-8")
    assert "wandb" not in src.lower()
    for forbidden in ("sweep_v2_six_axis_execution", "sweep_v2_six_axis_config", "sweep_v1_execution"):
        assert forbidden not in src


def test_diagnostic_contract_registers_no_optimizer_metric():
    cfg = importlib.import_module("src.baseline.sweep_v2_six_axis_config")
    assert cfg.V2_METRIC_NAME == f"flashnh/{OBJECTIVE_ID_V2}"
    assert DEVPOP_AUDIT_CONTRACT_ID not in cfg.V2_METRIC_NAME
    produced = cfg.build_production_sweep_config_v2(program="x")
    assert DEVPOP_AUDIT_CONTRACT_ID not in json.dumps(produced)
    assert produced["metric"]["name"] == cfg.V2_METRIC_NAME


def test_diagnostic_contract_id_absent_from_sweep_campaign_module():
    campaign = importlib.import_module("src.baseline.sweep_v2_six_axis_campaign")
    src = Path(campaign.__file__).read_text(encoding="utf-8")
    assert DEVPOP_AUDIT_CONTRACT_ID not in src
    assert "devpop_audit" not in src


def test_optimizer_extractor_structurally_rejects_the_diagnostic_scope():
    population = _population()
    contract = _contract(population)
    result = _result(population, contract)
    result["aggregate"]["metrics"]["nse"]["median"] = 0.5
    with pytest.raises(fixed.FixedSupportContractError):
        fixed.extract_v2_objective_from_fixed_support_result(result)


def test_no_public_entrypoint_accepts_an_arbitrary_period_string():
    with pytest.raises(DevpopAuditContractError):
        ExpectedPopulationSpec.for_synthetic_fixture(
            role="development_train", basin_ids=list(POP_IDS), period="whatever"
        )
    population = _population()
    assert population.period == "validation"


def test_building_audit_contract_does_not_touch_the_frozen_identity_json():
    identity_path = Path(audit.__file__).parents[2] / "config" / "stage1_v2_common120_fixed_support_artifact_identity_v001.json"
    before = identity_path.read_bytes()
    population = _population()
    _contract(population)
    assert identity_path.read_bytes() == before
