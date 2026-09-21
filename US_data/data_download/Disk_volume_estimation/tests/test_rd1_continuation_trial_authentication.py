"""Tests for the RD1 continuation evidence contract's authentication layer
(:mod:`src.baseline.rd1_continuation_trial_authentication`).

These tests never touch the frozen RD1-C4 24-run production contract
(:mod:`src.baseline.rd1_c4_trial_authentication`) -- they authenticate small,
locally-generated 24/36/48-trial rosters built from the same lightweight,
real-receipt fixtures ``tests/_rd1_c4_d1_support.py`` already provides, and
prove the phase-aware shape/interpretation-boundary logic this module adds on
top of the reused, unmodified per-trial receipt-proof primitive
``_authenticate_one_trial``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baseline import rd1_continuation_trial_authentication as continuation_auth_module
from src.baseline.rd1_continuation_trial_authentication import (
    CONTINUATION_ROSTER_SCHEMA_NAME,
    CONTINUATION_ROSTER_SCHEMA_VERSION,
    ContinuationAuthenticationError,
    EXPECTED_PROPOSAL_ORDERS_BY_PHASE,
    PHASE_BAYESIAN_CONTINUATION_36,
    PHASE_FINAL_BAYESIAN_48,
    PHASE_INITIAL_CONTROLLED_24,
    authenticate_continuation_roster,
    require_two_arm_comparison_support,
)
from src.baseline.rd1_c4_trial_authentication import TrialAuthenticationError
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2

from _rd1_c4_d1_support import (
    build_contract,
    build_package,
    sha256_bytes,
    write_synthetic_execution_receipt,
    write_trial_roster,
    write_validation_pickle,
)

BASIN_IDS = ("06911000", "06911100")

_INITIAL_24_PAIRS = {
    (arm, order)
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[PHASE_INITIAL_CONTROLLED_24].items()
    for order in orders
}


@pytest.fixture()
def world(tmp_path):
    package_root, dates, qobs_by_basin = build_package(tmp_path, BASIN_IDS)
    contract = build_contract(package_root, BASIN_IDS, dates)
    return {
        "tmp_path": tmp_path,
        "package_root": package_root,
        "dates": dates,
        "qobs_by_basin": qobs_by_basin,
        "contract": contract,
    }


def _build_entry(world, *, search_arm, proposal_order, label=None, **receipt_overrides):
    label = label or f"{search_arm}_{proposal_order}"
    contract = world["contract"]
    run_dir = world["tmp_path"] / "runs" / label
    receipt_path, record = write_synthetic_execution_receipt(
        world["tmp_path"] / "receipts",
        label,
        contract=contract,
        run_dir=run_dir,
        search_arm=search_arm,
        proposal_order=proposal_order,
        **receipt_overrides,
    )
    write_validation_pickle(
        run_dir,
        record["best_epoch"],
        basin_ids=BASIN_IDS,
        contract=contract,
        qobs_by_basin=world["qobs_by_basin"],
    )
    entry = {
        "trial_id": record["trial_id"],
        "search_arm": search_arm,
        "source_receipt_path": str(receipt_path),
        "source_receipt_sha256": sha256_bytes(receipt_path.read_bytes()),
    }
    return entry, receipt_path, record


def _build_phase_entries(world, phase, *, skip=()):
    """Build one real receipt-backed entry per (arm, proposal_order) the
    phase expects, skipping any ``(arm, order)`` pair listed in ``skip``."""
    entries = []
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[phase].items():
        for order in orders:
            if (arm, order) in skip:
                continue
            entry, _, _ = _build_entry(world, search_arm=arm, proposal_order=order)
            entries.append(entry)
    return entries


def _write_frozen_initial_roster(world, entries):
    """Write ``entries`` (real receipt-backed initial-24 entries) as a
    frozen-schema roster authenticatable by the unmodified
    ``rd1_c4_trial_authentication.authenticate_trial_roster`` -- the exact
    roster shape ``initial_roster_path`` must point at."""
    roster_path, _ = write_trial_roster(
        world["tmp_path"] / "frozen_initial_roster.json", contract=world["contract"], receipt_entries=entries
    )
    return roster_path


def _pin_authoritative_initial_roster_sha256(monkeypatch, roster_path):
    """Test-only seam (Finding #1 correction): production code pins
    continuation binding to one hardcoded, immutable authoritative SHA-256
    with no caller-facing override. Tests build their own small synthetic
    initial roster per-``world`` fixture, so they must repoint that private
    module constant at their own fixture's real on-disk SHA-256 -- this is
    not reachable through any production call path (no parameter on
    ``authenticate_continuation_roster`` exposes it)."""
    monkeypatch.setattr(
        continuation_auth_module,
        "_AUTHORITATIVE_INITIAL_ROSTER_SHA256",
        sha256_bytes(Path(roster_path).read_bytes()),
    )


def _write_roster(path, *, contract, phase, entries, campaign_id=CAMPAIGN_ID_V2, header_overrides=None):
    payload = {
        "schema_name": CONTINUATION_ROSTER_SCHEMA_NAME,
        "schema_version": CONTINUATION_ROSTER_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "phase": phase,
        "support_contract_version": contract["contract_id"],
        "support_contract_sha256": contract["checksum_sha256"],
        "trials": entries,
    }
    payload.update(header_overrides or {})
    path = Path(path)
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return path


# ---------------------------------------------------------------------------
# The three legal staged roster shapes authenticate correctly.
# ---------------------------------------------------------------------------


def test_initial_controlled_24_roster_authenticates_and_supports_two_arm_comparison(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    roster_path = _write_roster(
        world["tmp_path"] / "roster_24.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    roster = authenticate_continuation_roster(
        trial_list_path=roster_path, contract=world["contract"], phase=PHASE_INITIAL_CONTROLLED_24
    )
    assert len(roster) == 24
    assert roster.phase == PHASE_INITIAL_CONTROLLED_24
    assert roster.two_arm_comparison_supported is True
    assert len(roster.initial_controlled_trial_ids) == 24
    assert roster.continuation_only_trial_ids == ()
    require_two_arm_comparison_support(roster)  # must not raise


def test_bayesian_continuation_36_roster_preserves_phase_and_slice_identity(world, monkeypatch):
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)
    entries = initial_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS
    )
    roster_path = _write_roster(
        world["tmp_path"] / "roster_36.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
    )
    roster = authenticate_continuation_roster(
        trial_list_path=roster_path,
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        initial_roster_path=initial_roster_path,
    )
    assert len(roster) == 36
    assert roster.phase == PHASE_BAYESIAN_CONTINUATION_36
    assert roster.two_arm_comparison_supported is False
    assert len(roster.initial_controlled_trial_ids) == 24
    assert len(roster.continuation_only_trial_ids) == 12
    assert roster.initial_slice_identity["initial_roster_path"] == str(initial_roster_path)
    assert set(roster.initial_slice_identity["source_receipt_sha256_by_trial_id"]) == set(
        roster.initial_controlled_trial_ids
    )
    with pytest.raises(ContinuationAuthenticationError, match="does not support"):
        require_two_arm_comparison_support(roster)


def test_final_bayesian_48_roster_preserves_phase_and_slice_identity(world, monkeypatch):
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)
    entries = initial_entries + _build_phase_entries(
        world, PHASE_FINAL_BAYESIAN_48, skip=_INITIAL_24_PAIRS
    )
    roster_path = _write_roster(
        world["tmp_path"] / "roster_48.json",
        contract=world["contract"],
        phase=PHASE_FINAL_BAYESIAN_48,
        entries=entries,
    )
    roster = authenticate_continuation_roster(
        trial_list_path=roster_path,
        contract=world["contract"],
        phase=PHASE_FINAL_BAYESIAN_48,
        initial_roster_path=initial_roster_path,
    )
    assert len(roster) == 48
    assert roster.phase == PHASE_FINAL_BAYESIAN_48
    assert roster.two_arm_comparison_supported is False
    assert len(roster.initial_controlled_trial_ids) == 24
    assert len(roster.continuation_only_trial_ids) == 24
    with pytest.raises(ContinuationAuthenticationError, match="does not support"):
        require_two_arm_comparison_support(roster)


# ---------------------------------------------------------------------------
# Rejections.
# ---------------------------------------------------------------------------


def test_unknown_phase_is_refused(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="unknown continuation phase"):
        authenticate_continuation_roster(
            trial_list_path=roster_path, contract=world["contract"], phase="not_a_real_phase"
        )


def test_missing_phase_header_is_refused(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    payload = {
        "schema_name": CONTINUATION_ROSTER_SCHEMA_NAME,
        "schema_version": CONTINUATION_ROSTER_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID_V2,
        "support_contract_version": world["contract"]["contract_id"],
        "support_contract_sha256": world["contract"]["checksum_sha256"],
        "trials": entries,
    }
    roster_path = world["tmp_path"] / "roster.json"
    roster_path.write_bytes(json.dumps(payload).encode("utf-8"))
    with pytest.raises(ContinuationAuthenticationError, match="missing required key"):
        authenticate_continuation_roster(
            trial_list_path=roster_path, contract=world["contract"], phase=PHASE_INITIAL_CONTROLLED_24
        )


def test_roster_phase_header_cannot_be_overridden_by_the_caller(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="cannot be overridden"):
        authenticate_continuation_roster(
            trial_list_path=roster_path,
            contract=world["contract"],
            phase=PHASE_BAYESIAN_CONTINUATION_36,
            initial_roster_path=world["tmp_path"] / "unused_initial_roster.json",
        )


def test_incomplete_continuation_block_is_refused(world):
    # Drop one of the 12 new P13-P24 continuation proposals: 35 trials total
    # for a phase that requires exactly 36.
    entries = _build_phase_entries(world, PHASE_BAYESIAN_CONTINUATION_36, skip={("bayesian", 13)})
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="requires exactly 36 trials"):
        authenticate_continuation_roster(
            trial_list_path=roster_path,
            contract=world["contract"],
            phase=PHASE_BAYESIAN_CONTINUATION_36,
            initial_roster_path=world["tmp_path"] / "unused_initial_roster.json",
        )


def test_duplicate_trial_id_is_refused(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24, skip={("random_control", 12)})
    # Substitute a duplicate of an existing entry in place of the dropped one
    # so the roster count still matches the expected 24, but one trial now
    # appears twice.
    entries.append(dict(entries[0]))
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="duplicate trial_id"):
        authenticate_continuation_roster(
            trial_list_path=roster_path, contract=world["contract"], phase=PHASE_INITIAL_CONTROLLED_24
        )


def test_substituted_receipt_is_refused(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    # Point the first entry's claimed trial_id at a receipt that actually
    # belongs to the second entry.
    entries[0] = dict(entries[0], trial_id=entries[1]["trial_id"])
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(TrialAuthenticationError, match="substituted receipt"):
        authenticate_continuation_roster(
            trial_list_path=roster_path, contract=world["contract"], phase=PHASE_INITIAL_CONTROLLED_24
        )


def test_random_control_order_beyond_r12_is_refused(world):
    # Build a legal 24-trial population but with one random_control receipt
    # actually executed at proposal_order 13 -- beyond the frozen R12 ceiling
    # every phase shares.
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24, skip={("random_control", 12)})
    extra_entry, _, _ = _build_entry(world, search_arm="random_control", proposal_order=13)
    entries.append(extra_entry)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="unexpected"):
        authenticate_continuation_roster(
            trial_list_path=roster_path, contract=world["contract"], phase=PHASE_INITIAL_CONTROLLED_24
        )


def test_bayesian_proposal_order_outside_phase_ceiling_is_refused(world):
    # final_bayesian_48 admits bayesian P1-P36 only; substitute P37 for P36.
    entries = _build_phase_entries(world, PHASE_FINAL_BAYESIAN_48, skip={("bayesian", 36)})
    extra_entry, _, _ = _build_entry(world, search_arm="bayesian", proposal_order=37)
    entries.append(extra_entry)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_FINAL_BAYESIAN_48,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="unexpected"):
        authenticate_continuation_roster(
            trial_list_path=roster_path,
            contract=world["contract"],
            phase=PHASE_FINAL_BAYESIAN_48,
            initial_roster_path=world["tmp_path"] / "unused_initial_roster.json",
        )


def test_36_or_48_roster_cannot_be_treated_as_the_two_arm_comparison(world, monkeypatch):
    """Design constraint 5/7: a 36- or 48-trial continuation roster must
    never be usable as if it were the original controlled two-arm
    comparison, even though it structurally contains the full initial-24
    slice."""
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)
    entries = initial_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS
    )
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
    )
    roster = authenticate_continuation_roster(
        trial_list_path=roster_path,
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        initial_roster_path=initial_roster_path,
    )
    assert roster.two_arm_comparison_supported is False
    with pytest.raises(ContinuationAuthenticationError, match="original Bayesian-versus-random-control"):
        require_two_arm_comparison_support(roster)


# ---------------------------------------------------------------------------
# Immutable initial-slice binding.
# ---------------------------------------------------------------------------


def test_initial_roster_path_forbidden_for_initial_controlled_24(world):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="must be omitted"):
        authenticate_continuation_roster(
            trial_list_path=roster_path,
            contract=world["contract"],
            phase=PHASE_INITIAL_CONTROLLED_24,
            initial_roster_path=world["tmp_path"] / "frozen_initial_roster.json",
        )


def test_initial_roster_path_required_for_continuation_phases(world):
    entries = _build_phase_entries(world, PHASE_BAYESIAN_CONTINUATION_36)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="must be supplied"):
        authenticate_continuation_roster(
            trial_list_path=roster_path, contract=world["contract"], phase=PHASE_BAYESIAN_CONTINUATION_36
        )


def test_continuation_roster_with_altered_initial_slice_is_refused(world, monkeypatch):
    """Finding #1: a 36-run continuation roster that structurally looks
    right (correct trial_ids, correct per-arm/per-proposal shape, every
    entry individually receipt-proof-valid) must still be refused if its
    claimed initial-24 receipt mapping does not match the authenticated
    frozen base roster -- e.g. because it was built against a *different*
    execution of the same 24 configurations rather than the actual frozen
    runs the initial roster was authenticated from."""
    frozen_initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, frozen_initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)

    # Re-execute the same 24 (search_arm, proposal_order) configurations
    # under a different trajectory (peak at epoch 5 instead of epoch 3).
    # trial_id is derived only from configuration_id/proposal_id/
    # execution_generation, so these entries claim the *same* trial_ids as
    # the frozen initial roster, but each backed by a genuinely different,
    # individually-valid receipt (different bytes, different sha256).
    altered_trajectory = {epoch: 0.50 - abs(epoch - 5) * 0.01 for epoch in range(1, 13)}
    altered_initial_entries = []
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[PHASE_INITIAL_CONTROLLED_24].items():
        for order in orders:
            entry, _, _ = _build_entry(
                world,
                search_arm=arm,
                proposal_order=order,
                label=f"altered_{arm}_{order}",
                trajectory=altered_trajectory,
            )
            altered_initial_entries.append(entry)

    continuation_entries = altered_initial_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS
    )
    roster_path = _write_roster(
        world["tmp_path"] / "altered_roster_36.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=continuation_entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="altered initial slice"):
        authenticate_continuation_roster(
            trial_list_path=roster_path,
            contract=world["contract"],
            phase=PHASE_BAYESIAN_CONTINUATION_36,
            initial_roster_path=initial_roster_path,
        )


def test_alternate_valid_initial_roster_is_refused_by_authoritative_sha(world, monkeypatch):
    """Correction: matching the continuation roster's initial-24 subset to
    *some* authenticated initial roster is not enough -- it must be the one
    immutable RD1-C4 original initial roster, identified by its raw-byte
    SHA-256, not any internally-valid-but-different alternate.

    This roster (``alternate_roster_path``) is genuinely receipt-valid --
    every entry individually authenticates, and the continuation roster's
    initial-24 subset matches it exactly by ``trial_id`` and
    ``source_receipt_sha256`` (unlike
    ``test_continuation_roster_with_altered_initial_slice_is_refused``,
    nothing here is altered at the trial/receipt level). The authoritative
    SHA is pinned (via the test-only seam) to a different, well-formed
    value -- standing in for the real immutable roster being a distinct
    file -- so the only possible reason authentication can fail is the
    file-level authoritative-SHA pin itself."""
    alternate_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    alternate_roster_path = _write_frozen_initial_roster(world, alternate_entries)

    # Pin the authoritative SHA to an unrelated, well-formed 64-hex-digit
    # value that is NOT this alternate roster's own SHA-256 -- i.e. the
    # immutable original roster is some other file than the one supplied.
    monkeypatch.setattr(
        continuation_auth_module,
        "_AUTHORITATIVE_INITIAL_ROSTER_SHA256",
        "ff" * 32,
    )

    continuation_entries = alternate_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS
    )
    roster_path = _write_roster(
        world["tmp_path"] / "roster_36_alternate_initial.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=continuation_entries,
    )
    with pytest.raises(ContinuationAuthenticationError, match="authoritative"):
        authenticate_continuation_roster(
            trial_list_path=roster_path,
            contract=world["contract"],
            phase=PHASE_BAYESIAN_CONTINUATION_36,
            initial_roster_path=alternate_roster_path,
        )
