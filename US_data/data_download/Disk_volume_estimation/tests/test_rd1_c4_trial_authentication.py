"""Correction-A receipt and frozen-roster authority tests."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from src.baseline import rd1_c4_observation_diagnostic_cli as cli
from src.baseline.authenticated_period_results import (
    AuthenticatedPeriodResultsError,
    load_authenticated_period_results,
)
from src.baseline.rd1_c4_observation_diagnostic_reduce import (
    EXPECTED_ARM_COUNTS as REDUCER_ARM_COUNTS,
    EXPECTED_TRIAL_COUNT as REDUCER_TRIAL_COUNT,
)
from src.baseline.rd1_c4_trial_authentication import (
    EXPECTED_ARM_COUNTS,
    EXPECTED_PROPOSAL_ORDERS_BY_ARM,
    EXPECTED_TRIAL_COUNT,
    TrialAuthenticationError,
    authenticate_trial_roster,
    fixture_only_expected_roster,
)
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2
from tests._rd1_c4_d1_support import (
    build_contract,
    build_package,
    sha256_bytes,
    write_synthetic_execution_receipt,
    write_trial_roster,
    write_validation_pickle,
)


def _entry(path: Path, record: dict, **changes) -> dict:
    value = {
        "trial_id": record["trial_id"],
        "search_arm": record["search_arm"],
        "source_receipt_path": str(path),
        "source_receipt_sha256": sha256_bytes(path.read_bytes()),
    }
    value.update(changes)
    return value


def _rewrite_receipt(path: Path, record: dict) -> None:
    path.write_bytes(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@pytest.fixture
def two_trial_world(tmp_path):
    basin_ids = ["01000001"]
    package_root, dates, qobs = build_package(tmp_path, basin_ids, n_hours=30)
    contract = build_contract(package_root, basin_ids, dates)
    receipts = []
    for arm in ("bayesian", "random_control"):
        run_dir = tmp_path / f"run_{arm}"
        write_validation_pickle(
            run_dir,
            3,
            basin_ids=basin_ids,
            contract=contract,
            qobs_by_basin=qobs,
        )
        receipts.append(
            write_synthetic_execution_receipt(
                tmp_path,
                arm,
                contract=contract,
                run_dir=run_dir,
                search_arm=arm,
                proposal_order=1,
            )
        )
    expected = fixture_only_expected_roster({"bayesian": [1], "random_control": [1]})
    return tmp_path, contract, receipts, expected


def _authenticate(world, entries=None, *, verify_validation_pickles=False, roster_name="roster.json"):
    root, contract, receipts, expected = world
    if entries is None:
        entries = [_entry(path, record) for path, record in receipts]
    roster_path, _ = write_trial_roster(root / roster_name, contract=contract, receipt_entries=entries)
    return authenticate_trial_roster(
        trial_list_path=roster_path,
        contract=contract,
        verify_validation_pickles=verify_validation_pickles,
        test_only_expected_roster=expected,
    )


def test_valid_authenticated_frozen_24_trial_roster_uses_production_defaults(tmp_path):
    basin_ids = ["01000001"]
    package_root, dates, qobs = build_package(tmp_path, basin_ids, n_hours=24)
    contract = build_contract(package_root, basin_ids, dates)
    entries = []
    for arm, proposal_orders in EXPECTED_PROPOSAL_ORDERS_BY_ARM.items():
        for order in proposal_orders:
            label = f"{arm}_{order:02d}"
            run_dir = tmp_path / f"run_{label}"
            write_validation_pickle(
                run_dir, 3, basin_ids=basin_ids, contract=contract, qobs_by_basin=qobs
            )
            path, record = write_synthetic_execution_receipt(
                tmp_path,
                label,
                contract=contract,
                run_dir=run_dir,
                search_arm=arm,
                proposal_order=order,
            )
            entries.append(_entry(path, record))
    roster_path, expected_hash = write_trial_roster(
        tmp_path / "frozen_24.json", contract=contract, receipt_entries=entries
    )

    roster = authenticate_trial_roster(trial_list_path=roster_path, contract=contract)

    assert len(roster) == 24
    assert roster.trial_list_sha256 == expected_hash
    assert {(target.search_arm, target.proposal_order) for target in roster} == {
        (arm, order)
        for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_ARM.items()
        for order in orders
    }


def test_trial_list_bytes_and_hash_are_deterministic(two_trial_world):
    root, contract, receipts, _ = two_trial_world
    entries = [_entry(path, record) for path, record in receipts]
    path_a, hash_a = write_trial_roster(root / "a.json", contract=contract, receipt_entries=entries)
    path_b, hash_b = write_trial_roster(root / "b.json", contract=contract, receipt_entries=entries)
    assert path_a.read_bytes() == path_b.read_bytes()
    assert hash_a == hash_b == hashlib.sha256(path_a.read_bytes()).hexdigest()


def test_valid_selected_target_and_hashes_come_from_actual_bytes(two_trial_world):
    roster = _authenticate(two_trial_world, verify_validation_pickles=True)
    path, record = two_trial_world[2][0]
    target = roster.by_trial_id(record["trial_id"])
    assert target.source_receipt_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert target.validation_pickle_sha256 == hashlib.sha256(
        Path(target.validation_pickle_path).read_bytes()
    ).hexdigest()
    assert target.run_dir == record["result"]["nh_run_dir"]
    assert target.best_epoch == 3
    assert target.official_objective == 0.5


def test_missing_receipt_field_is_refused(two_trial_world):
    entries = [_entry(path, record) for path, record in two_trial_world[2]]
    del entries[0]["source_receipt_path"]
    with pytest.raises(TrialAuthenticationError, match="missing required"):
        _authenticate(two_trial_world, entries)


def test_nonexistent_receipt_path_is_refused(two_trial_world):
    entries = [_entry(path, record) for path, record in two_trial_world[2]]
    entries[0]["source_receipt_path"] = str(two_trial_world[0] / "does_not_exist.json")
    with pytest.raises(TrialAuthenticationError, match="receipt is absent"):
        _authenticate(two_trial_world, entries)


def test_receipt_hash_mismatch_and_tampered_content_are_refused(two_trial_world):
    entries = [_entry(path, record) for path, record in two_trial_world[2]]
    entries[0]["source_receipt_sha256"] = "0" * 64
    with pytest.raises(TrialAuthenticationError, match="trial list declares"):
        _authenticate(two_trial_world, entries)

    path, _ = two_trial_world[2][0]
    entries = [_entry(p, record) for p, record in two_trial_world[2]]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(TrialAuthenticationError, match="not the document"):
        _authenticate(two_trial_world, entries, roster_name="tampered.json")


def test_duplicate_trial_identity_is_refused(two_trial_world):
    path, record = two_trial_world[2][0]
    with pytest.raises(TrialAuthenticationError, match="duplicate trial_id|duplicate source_receipt_path"):
        _authenticate(two_trial_world, [_entry(path, record), _entry(path, record)])


def test_missing_and_unexpected_proposal_are_refused_even_at_the_right_count(tmp_path):
    basin_ids = ["01000001"]
    package_root, dates, qobs = build_package(tmp_path, basin_ids, n_hours=24)
    contract = build_contract(package_root, basin_ids, dates)
    entries = []
    for order in (1, 3):
        run_dir = tmp_path / f"run_{order}"
        write_validation_pickle(run_dir, 3, basin_ids=basin_ids, contract=contract, qobs_by_basin=qobs)
        path, record = write_synthetic_execution_receipt(
            tmp_path, f"p{order}", contract=contract, run_dir=run_dir, proposal_order=order
        )
        entries.append(_entry(path, record))
    roster_path, _ = write_trial_roster(tmp_path / "roster.json", contract=contract, receipt_entries=entries)
    with pytest.raises(TrialAuthenticationError, match=r"missing=.*2.*unexpected=.*3"):
        authenticate_trial_roster(
            trial_list_path=roster_path,
            contract=contract,
            test_only_expected_roster=fixture_only_expected_roster({"bayesian": [1, 2]}),
        )


def test_substituted_trial_and_wrong_arm_claim_are_refused(two_trial_world):
    entries = [_entry(path, record) for path, record in two_trial_world[2]]
    entries[0]["trial_id"] = entries[1]["trial_id"]
    with pytest.raises(TrialAuthenticationError, match="substituted receipt"):
        _authenticate(two_trial_world, entries)

    entries = [_entry(path, record) for path, record in two_trial_world[2]]
    entries[0]["search_arm"] = "random_control"
    with pytest.raises(TrialAuthenticationError, match="claims search_arm"):
        _authenticate(two_trial_world, entries, roster_name="wrong_arm.json")


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("campaign", lambda r: (r.__setitem__("campaign_id", "wrong"), r["preparation_record"].__setitem__("campaign_id", "wrong")), "campaign_id"),
        ("domain", lambda r: r["preparation_record"].__setitem__("domain_version", "wrong"), "domain_version"),
        ("fidelity", lambda r: r["preparation_record"].__setitem__("fidelity_id", "wrong"), "fidelity_id"),
        ("proposal", lambda r: (r.__setitem__("proposal_id", "wrong"), r["preparation_record"].__setitem__("proposal_id", "wrong")), "proposal_id"),
        ("configuration", lambda r: (r.__setitem__("configuration_id", "wrong"), r["preparation_record"].__setitem__("configuration_id", "wrong")), "configuration_id"),
        ("best_epoch", lambda r: r.__setitem__("best_epoch", 4), "objective|best_epoch"),
        ("objective", lambda r: r.__setitem__("objective_score", 0.49), "objective"),
        ("objective_identity", lambda r: r.__setitem__("fixed_support_metric_name", "wrong_metric"), "optimizer metric"),
        ("scope", lambda r: r["preparation_record"].__setitem__("evaluation_scope", "temporal_test"), "evaluation_scope"),
        ("sealed", lambda r: r["preparation_record"].__setitem__("sealed_scope", True), "sealed_scope"),
    ],
)
def test_wrong_receipt_authority_fact_is_refused(two_trial_world, label, mutate, match):
    path, original = two_trial_world[2][0]
    record = copy.deepcopy(original)
    mutate(record)
    _rewrite_receipt(path, record)
    entries = [_entry(path, original), _entry(*two_trial_world[2][1])]
    entries[0]["source_receipt_sha256"] = sha256_bytes(path.read_bytes())
    with pytest.raises(TrialAuthenticationError, match=match):
        _authenticate(two_trial_world, entries, roster_name=f"wrong_{label}.json")


def test_wrong_run_directory_is_refused_by_resolved_validation_product(two_trial_world):
    path, original = two_trial_world[2][0]
    record = copy.deepcopy(original)
    record["result"]["nh_run_dir"] = str(two_trial_world[0] / "foreign_run")
    _rewrite_receipt(path, record)
    entries = [_entry(path, original), _entry(*two_trial_world[2][1])]
    entries[0]["source_receipt_sha256"] = sha256_bytes(path.read_bytes())
    with pytest.raises(TrialAuthenticationError, match="validation product is absent"):
        _authenticate(two_trial_world, entries)


@pytest.mark.parametrize("field", ["run_dir", "validation_pickle_path", "validation_pickle_sha256"])
def test_roster_cannot_assert_receipt_owned_paths_or_hashes(two_trial_world, field):
    entries = [_entry(path, record) for path, record in two_trial_world[2]]
    entries[0][field] = "caller-asserted"
    with pytest.raises(TrialAuthenticationError, match="may not be asserted"):
        _authenticate(two_trial_world, entries, roster_name=f"forbidden_{field}.json")


def test_post_authentication_validation_pickle_hash_mismatch_is_detected(two_trial_world):
    roster = _authenticate(two_trial_world, verify_validation_pickles=True)
    target = next(iter(roster))
    Path(target.validation_pickle_path).write_bytes(Path(target.validation_pickle_path).read_bytes() + b"tamper")
    with pytest.raises(AuthenticatedPeriodResultsError, match="not the product"):
        load_authenticated_period_results(
            run_dir=target.run_dir,
            period=roster.period,
            epoch=target.best_epoch,
            expected_sha256=target.validation_pickle_sha256,
        )


def test_trial_list_path_cannot_bypass_sealed_scope_protection(two_trial_world):
    path, original = two_trial_world[2][0]
    record = copy.deepcopy(original)
    record["preparation_record"]["evaluation_scope"] = "development_validation_2024_only"
    record["preparation_record"]["sealed_scope"] = True
    _rewrite_receipt(path, record)
    entries = [_entry(path, original), _entry(*two_trial_world[2][1])]
    entries[0]["source_receipt_sha256"] = sha256_bytes(path.read_bytes())
    with pytest.raises(TrialAuthenticationError, match="sealed_scope"):
        _authenticate(two_trial_world, entries, roster_name="apparently_safe_development_roster.json")


def test_trial_list_header_cannot_override_receipt_scope(two_trial_world):
    root, contract, receipts, expected = two_trial_world
    payload = {
        "schema_name": "flashnh_rd1_c4_d1_trial_roster",
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID_V2,
        "support_contract_version": contract["contract_id"],
        "support_contract_sha256": contract["checksum_sha256"],
        "sealed_scope": False,
        "trials": [_entry(path, record) for path, record in receipts],
    }
    roster_path = root / "scope_override.json"
    roster_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TrialAuthenticationError, match="unexpected header key"):
        authenticate_trial_roster(
            trial_list_path=roster_path,
            contract=contract,
            test_only_expected_roster=expected,
        )


def test_incorrect_arm_composition_is_refused(two_trial_world):
    root, contract, receipts, _ = two_trial_world
    first_path, first_record = receipts[0]
    run_dir = root / "run_bayesian_2"
    # The file only needs to exist; copy the first authenticated product.
    source_pickle = Path(first_record["result"]["nh_run_dir"]) / "validation" / "model_epoch003" / "validation_results.p"
    target_pickle = run_dir / "validation" / "model_epoch003" / "validation_results.p"
    target_pickle.parent.mkdir(parents=True)
    target_pickle.write_bytes(source_pickle.read_bytes())
    second_path, second_record = write_synthetic_execution_receipt(
        root, "bayesian_2", contract=contract, run_dir=run_dir, search_arm="bayesian", proposal_order=2
    )
    entries = [_entry(first_path, first_record), _entry(second_path, second_record)]
    roster_path, _ = write_trial_roster(root / "wrong_composition.json", contract=contract, receipt_entries=entries)
    with pytest.raises(TrialAuthenticationError, match="population contradiction"):
        authenticate_trial_roster(
            trial_list_path=roster_path,
            contract=contract,
            test_only_expected_roster=fixture_only_expected_roster({"bayesian": [1], "random_control": [1]}),
        )


def test_expected_counts_and_arm_composition_have_one_authoritative_meaning():
    assert EXPECTED_TRIAL_COUNT == 24 == sum(EXPECTED_ARM_COUNTS.values())
    assert dict(EXPECTED_ARM_COUNTS) == {"bayesian": 12, "random_control": 12}
    assert REDUCER_TRIAL_COUNT == EXPECTED_TRIAL_COUNT
    assert dict(REDUCER_ARM_COUNTS) == dict(EXPECTED_ARM_COUNTS)
    parameters = inspect.signature(authenticate_trial_roster).parameters
    assert "expected_trial_count" not in parameters
    assert "expected_arm_counts" not in parameters
    assert parameters["test_only_expected_roster"].default is None


def test_wrong_roster_campaign_header_is_refused(two_trial_world):
    root, contract, receipts, expected = two_trial_world
    roster_path, _ = write_trial_roster(
        root / "wrong_campaign_header.json",
        contract=contract,
        receipt_entries=[_entry(path, record) for path, record in receipts],
        campaign_id="not_the_receipt_campaign",
    )
    with pytest.raises(TrialAuthenticationError, match="receipt campaign_id"):
        authenticate_trial_roster(
            trial_list_path=roster_path,
            contract=contract,
            test_only_expected_roster=expected,
        )


def test_cli_selection_preserves_authenticated_frozen_roster_authority(two_trial_world, monkeypatch):
    root, contract, receipts, expected = two_trial_world
    roster_path, _ = write_trial_roster(
        root / "cli_roster.json",
        contract=contract,
        receipt_entries=[_entry(path, record) for path, record in receipts],
    )
    real_authenticate = authenticate_trial_roster
    monkeypatch.setattr(
        cli,
        "authenticate_trial_roster",
        lambda **kwargs: real_authenticate(**kwargs, test_only_expected_roster=expected),
    )
    roster, digest = cli.load_trial_list(roster_path, contract)
    selected = cli._resolve_target(roster, trial_id=receipts[1][1]["trial_id"], array_index=None)
    indexed = cli._resolve_target(roster, trial_id=None, array_index=1)
    assert selected is indexed is roster.targets[1]
    assert selected.source_receipt_sha256 == hashlib.sha256(receipts[1][0].read_bytes()).hexdigest()
    assert digest == hashlib.sha256(roster_path.read_bytes()).hexdigest()


def test_cli_does_not_accept_legacy_caller_asserted_trial_targets(two_trial_world, monkeypatch):
    root, contract, receipts, expected = two_trial_world
    entries = [_entry(path, record) for path, record in receipts]
    entries[0]["run_dir"] = str(root / "caller_chosen_run")
    roster_path, _ = write_trial_roster(root / "legacy.json", contract=contract, receipt_entries=entries)
    real_authenticate = authenticate_trial_roster
    monkeypatch.setattr(
        cli,
        "authenticate_trial_roster",
        lambda **kwargs: real_authenticate(**kwargs, test_only_expected_roster=expected),
    )
    with pytest.raises(Exception, match="may not be asserted"):
        cli.load_trial_list(roster_path, contract)
