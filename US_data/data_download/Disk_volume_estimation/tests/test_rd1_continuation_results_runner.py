"""Tests for the RD1 continuation evidence contract's results/evidence layer
(:mod:`src.baseline.rd1_continuation_results_runner`).

This is also the required vertical synthetic/integration test proving the
intended results consumer can ingest all three legal staged roster shapes
(``initial_controlled_24`` / ``bayesian_continuation_36`` / ``final_bayesian_48``)
while preserving phase identity and the interpretation boundary. Per this
codebase's own stated convention (the designated vertical integration test
uses real I/O and mocks nothing), every trial here is a real, small-scale,
receipt-backed :class:`~src.baseline.stage1_v2_12plus12_hydrological_consumer.V2BestEpochSource`
built through the exact same lightweight fixtures
(``tests/_rd1_c4_d1_support.py``) the sibling frozen-contract test suite uses
-- never the heavy 400-basin ``formal_batch`` fixture, and never the
``_wire_synthetic_epoch_v2`` mocking seam.

Test-cost design (corrects an earlier version of this file that took
~1h04m because ``initial_controlled_24`` coverage required two full real
400-basin x 24-trial evaluations -- direct delegate vs. wrapper -- inside
one test): ``assemble_rd1_continuation_hydrological_review``'s
``initial_controlled_24`` branch is pure pass-through with no validation
of its own before delegating to the frozen ``assemble_rd1_hydrological_review``
(see the source), so proving that delegation does not require a real
400-basin evaluation at all -- only that the wrapper forwards its
arguments unchanged and carries the frozen delegate's return value through
untouched. ``test_initial_controlled_24_delegates_unchanged_to_the_frozen_assembler``
proves exactly that with a call-recording spy substituted for
``assemble_rd1_hydrological_review`` and sentinel inputs/outputs -- zero
I/O, zero fixtures, runs in milliseconds. The frozen RD1-C4 and
hydrological-consumer regression suites remain the evidence that the
frozen 24-run evaluator itself produces correct results; this file no
longer re-proves that.
The two continuation phases, by contrast, run through this module's own
phase-aware validator/evaluation path
(``assemble_rd1_continuation_hydrological_review``), which -- unlike the
frozen path -- never hardcodes a basin count (see
``_require_complete_continuation_population``): it requires the contract's
*complete* population with zero exclusions, at whatever size the contract
declares. The earlier version of this file reused the same 400-basin
package for all three phases, so its 36- and 48-run tests each repeated a
full 400-basin raw-space evaluation per trial (up to 108 such evaluations
total across the suite) purely because one shared fixture happened to carry
400 basins -- not because the continuation path required it. Continuation-
phase tests below instead use a small (3-basin) package/contract, the
smallest size the continuation validator's own completeness check actually
supports, while still performing real receipt authentication and real
continuation-consumer assembly (no mocking of either).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baseline import rd1_continuation_results_runner as runner
from src.baseline import rd1_continuation_trial_authentication as continuation_auth_module
from src.baseline.rd1_c4_results_runner import ResultsRunnerError
from src.baseline.rd1_c4_trial_authentication import authenticate_trial_roster
from src.baseline.rd1_continuation_trial_authentication import (
    CONTINUATION_ROSTER_SCHEMA_NAME,
    CONTINUATION_ROSTER_SCHEMA_VERSION,
    EXPECTED_PROPOSAL_ORDERS_BY_PHASE,
    PHASE_BAYESIAN_CONTINUATION_36,
    PHASE_FINAL_BAYESIAN_48,
    PHASE_INITIAL_CONTROLLED_24,
)
from src.baseline.stage1_v2_12plus12_hydrological_consumer import HydrologicalConsumerError
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2

from _rd1_c4_d1_support import (
    build_contract,
    build_package,
    sha256_bytes,
    write_synthetic_execution_receipt,
    write_trial_roster,
    write_validation_pickle,
)

#: Every continuation-phase test uses this much smaller, but still fully
#: truthful (complete, zero-excluded) contract -- see the module docstring.
SMALL_BASIN_IDS = ("small_000", "small_001", "small_002")

_INITIAL_24_PAIRS = {
    (arm, order)
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[PHASE_INITIAL_CONTROLLED_24].items()
    for order in orders
}


def _build_shared_package_bundle(tmp_path_factory, basin_ids, *, name):
    tmp_path = tmp_path_factory.mktemp(name)
    # NOTE: n_hours must stay large enough that ``usable = n_hours - LEAD_HOURS``
    # clears the raw-space area-derivation floor (>= 100 valid samples) inside
    # the frozen fixed-support evaluator -- an earlier n_hours=48 override left
    # only 42 usable samples, which silently excluded every basin and tripped
    # the "N requested/evaluated/0 excluded" completeness gate. Use the
    # fixture's own default (240) instead.
    package_root, dates, qobs_by_basin = build_package(tmp_path, basin_ids)
    contract = build_contract(package_root, basin_ids, dates)

    # Every trial this module builds uses the identical, deterministic
    # obs/sim relationship from ``write_validation_pickle`` (no
    # ``obs_override``/``corrupt_basins``), so the real fixed-support
    # re-scoring the consumer performs at ingestion time is the same number
    # for every trial. Compute it once here (real evaluator, real I/O, no
    # mocking) and reuse it as the receipt's own claimed
    # ``official_objective`` -- otherwise the consumer's re-scoring
    # consistency check ("re-scored fixed-support objective ... does not
    # equal the official objective ... recorded at best_epoch=...") fails
    # every trial, since a hand-picked placeholder objective can never
    # match what the real data actually re-scores to.
    from src.baseline import fixed_support_contract_v2 as fixed
    from src.baseline.package_identity_qualification import qualify_package_identity

    reference_run_dir = tmp_path / "reference_run"
    write_validation_pickle(
        reference_run_dir, 3, basin_ids=basin_ids, contract=contract, qobs_by_basin=qobs_by_basin,
    )
    package_identity = qualify_package_identity(
        package_root=package_root, contract=contract, basin_ids=contract["basin_ids"]
    )
    reference_result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir=reference_run_dir,
        epoch=3,
        package_root=package_root,
        contract=contract,
        basin_ids=contract["basin_ids"],
        require_full_screening_population=False,
        return_admitted_series=False,
        package_identity=package_identity,
    )
    reference_objective = fixed.extract_v2_objective_from_fixed_support_result(reference_result)
    return package_root, dates, qobs_by_basin, contract, reference_objective, tuple(basin_ids)


@pytest.fixture(scope="module")
def _shared_small_package(tmp_path_factory):
    return _build_shared_package_bundle(tmp_path_factory, SMALL_BASIN_IDS, name="rd1_continuation_small_package")


def _make_world(tmp_path, bundle):
    package_root, dates, qobs_by_basin, contract, reference_objective, basin_ids = bundle
    from src.baseline import fixed_support_contract_v2 as fixed

    tmp_path.mkdir(parents=True, exist_ok=True)
    contract_path = tmp_path / "contract.json"
    fixed.write_fixed_support_contract(contract, contract_path)
    return {
        "tmp_path": tmp_path,
        "package_root": package_root,
        "dates": dates,
        "qobs_by_basin": qobs_by_basin,
        "contract": contract,
        "contract_path": contract_path,
        "reference_objective": reference_objective,
        "basin_ids": basin_ids,
    }


@pytest.fixture()
def world(tmp_path, _shared_small_package):
    # The small (3-basin) package/contract used by every continuation-phase
    # test -- see module docstring.
    return _make_world(tmp_path / "small", _shared_small_package)


def _build_entry(world, *, search_arm, proposal_order, label=None, **receipt_overrides):
    label = label or f"{search_arm}_{proposal_order}"
    contract = world["contract"]
    run_dir = world["tmp_path"] / "runs" / label
    # Peak at epoch 3 (matching ``write_validation_pickle``'s epoch below)
    # with the real re-scored objective, so the receipt's own claimed
    # official_objective is content-consistent with what the consumer will
    # independently re-derive from this trial's own pickle.
    trajectory = receipt_overrides.pop(
        "trajectory", {epoch: world["reference_objective"] - abs(epoch - 3) * 0.01 for epoch in range(1, 13)}
    )
    receipt_path, record = write_synthetic_execution_receipt(
        world["tmp_path"] / "receipts",
        label,
        contract=contract,
        run_dir=run_dir,
        search_arm=search_arm,
        proposal_order=proposal_order,
        trajectory=trajectory,
        **receipt_overrides,
    )
    write_validation_pickle(
        run_dir,
        record["best_epoch"],
        basin_ids=world["basin_ids"],
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
    entries = []
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[phase].items():
        for order in orders:
            if (arm, order) in skip:
                continue
            entry, _, _ = _build_entry(world, search_arm=arm, proposal_order=order)
            entries.append(entry)
    return entries


def _write_roster(path, *, contract, phase, entries, campaign_id=CAMPAIGN_ID_V2):
    payload = {
        "schema_name": CONTINUATION_ROSTER_SCHEMA_NAME,
        "schema_version": CONTINUATION_ROSTER_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "phase": phase,
        "support_contract_version": contract["contract_id"],
        "support_contract_sha256": contract["checksum_sha256"],
        "trials": entries,
    }
    path = Path(path)
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return path


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
    """Same test-only seam as
    ``tests/test_rd1_continuation_trial_authentication.py`` -- production
    code pins continuation binding to one hardcoded, immutable authoritative
    SHA-256 with no caller-facing override, so tests must repoint that
    private module constant at their own fixture's real on-disk SHA-256."""
    monkeypatch.setattr(
        continuation_auth_module,
        "_AUTHORITATIVE_INITIAL_ROSTER_SHA256",
        sha256_bytes(Path(roster_path).read_bytes()),
    )


def _produce(world, *, phase, entries, out_dir, initial_roster_path=None):
    roster_path = _write_roster(
        world["tmp_path"] / f"roster_{phase}.json", contract=world["contract"], phase=phase, entries=entries
    )
    return runner.produce_rd1_continuation_results(
        trial_list_path=roster_path,
        contract_path=world["contract_path"],
        package_root=world["package_root"],
        phase=phase,
        out_dir=out_dir,
        initial_roster_path=initial_roster_path,
    )


def test_initial_controlled_24_delegates_unchanged_to_the_frozen_assembler(monkeypatch):
    """Correction: replaces the earlier real-400-basin double evaluation
    (direct-vs-wrapped) with a cheap delegation/argument-forwarding proof.

    ``initial_controlled_24`` must still delegate unchanged to the frozen
    ``assemble_rd1_hydrological_review`` -- the immutable 24-run contract is
    never re-implemented in this module, only wrapped. Proved here by
    spying on that frozen entry point exactly as imported into
    ``rd1_continuation_results_runner`` (the same monkeypatch technique
    ``test_failed_continuation_publish_leaves_out_dir_absent`` already uses
    for ``_sha256_path``) and asserting it is called with exactly the
    forwarded ``best_epoch_sources``/``package_root``/``contract`` and that
    its return value flows through into the wrapper untouched. For this
    phase, ``assemble_rd1_continuation_hydrological_review`` performs no
    validation of its own before delegating (see the source), so sentinel
    objects are sufficient here -- no real receipts, packages, or
    evaluation are needed, and this test runs in milliseconds.

    Real, unmocked integration coverage of the write/publish path --
    phase-manifest, on-disk manifest consistency, atomic publication, and
    fail-closed population -- is retained below against the small synthetic
    contract for the 36/48 phases, which exercise the exact same
    ``write_rd1_continuation_results`` code (it has no phase-specific
    branching of its own). The frozen RD1-C4 and hydrological-consumer
    regression suites remain the evidence that the frozen 24-run evaluator
    itself works -- unchanged and unweakened by this correction."""
    sentinel_sources = ("sentinel-source-a", "sentinel-source-b")
    sentinel_package_root = "sentinel-package-root"
    sentinel_contract = {"sentinel": "contract"}
    canned_review = SimpleNamespace(results_by_trial_id={"trial_b": object(), "trial_a": object()})
    calls = []

    def _spy_assemble_rd1_hydrological_review(*, best_epoch_sources, package_root, contract):
        calls.append(
            {"best_epoch_sources": best_epoch_sources, "package_root": package_root, "contract": contract}
        )
        return canned_review

    monkeypatch.setattr(runner, "assemble_rd1_hydrological_review", _spy_assemble_rd1_hydrological_review)

    wrapped = runner.assemble_rd1_continuation_hydrological_review(
        best_epoch_sources=sentinel_sources,
        package_root=sentinel_package_root,
        contract=sentinel_contract,
        phase=PHASE_INITIAL_CONTROLLED_24,
    )

    assert len(calls) == 1
    assert calls[0]["best_epoch_sources"] is sentinel_sources
    assert calls[0]["package_root"] is sentinel_package_root
    assert calls[0]["contract"] is sentinel_contract

    assert wrapped.phase == PHASE_INITIAL_CONTROLLED_24
    assert wrapped.two_arm_comparison_supported is True
    assert wrapped.review is canned_review
    assert wrapped.initial_controlled_trial_ids == ("trial_a", "trial_b")
    assert wrapped.continuation_only_trial_ids == ()


@pytest.mark.parametrize("phase", [PHASE_BAYESIAN_CONTINUATION_36, PHASE_FINAL_BAYESIAN_48])
def test_produce_rd1_continuation_results_ingests_continuation_phase_rosters(world, phase, tmp_path, monkeypatch):
    """Same shape/identity assertions as the retained 24-run vertical test,
    but against the small synthetic contract -- see the module docstring
    for why 36/48 phases don't need the real 400-basin contract."""
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)
    entries = initial_entries + _build_phase_entries(world, phase, skip=_INITIAL_24_PAIRS)
    out_dir = tmp_path / f"out_{phase}"

    manifest = _produce(
        world, phase=phase, entries=entries, out_dir=out_dir, initial_roster_path=initial_roster_path
    )

    expected_files = {
        "per_basin_metrics.csv",
        "q98_diagnostics.csv",
        "basin_distribution_summary.csv",
        "canonical_q98_facts.csv",
        "provenance_audit.csv",
        "review_identity.json",
        "phase_manifest.json",
    }
    assert expected_files.issubset(set(manifest))
    for name, entry in manifest.items():
        produced = out_dir / name
        assert produced.exists()
        assert entry["size_bytes"] == produced.stat().st_size

    expected_total = sum(len(orders) for orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[phase].values())
    identity = json.loads((out_dir / "review_identity.json").read_text(encoding="utf-8"))
    assert identity["n_configurations"] == expected_total
    assert identity["n_random_control"] == 12

    with open(out_dir / "per_basin_metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_total * len(world["basin_ids"])

    phase_manifest = json.loads((out_dir / "phase_manifest.json").read_text(encoding="utf-8"))
    assert phase_manifest["phase"] == phase
    assert phase_manifest["two_arm_comparison_supported"] is False
    assert len(phase_manifest["initial_controlled_trial_ids"]) == 24
    expected_continuation_only = expected_total - 24
    assert len(phase_manifest["continuation_only_trial_ids"]) == expected_continuation_only
    assert "only the initial 24-run slice" in phase_manifest["interpretation_boundary"].lower()
    assert phase_manifest["initial_slice_identity"]["initial_roster_path"] == str(initial_roster_path)


def test_on_disk_manifest_json_carries_the_correct_phase_manifest_entry(world, tmp_path, monkeypatch):
    """Finding #2: the returned manifest and the on-disk ``manifest.json``
    must agree, and both must include ``phase_manifest.json`` with its
    correct size/sha256."""
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)
    entries = initial_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS
    )
    out_dir = tmp_path / "out_manifest_check"

    manifest = _produce(
        world,
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
        out_dir=out_dir,
        initial_roster_path=initial_roster_path,
    )

    on_disk_manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest == manifest

    phase_manifest_path = out_dir / "phase_manifest.json"
    assert on_disk_manifest["phase_manifest.json"]["size_bytes"] == phase_manifest_path.stat().st_size
    assert on_disk_manifest["phase_manifest.json"]["sha256"] == sha256_bytes(phase_manifest_path.read_bytes())


def test_failed_continuation_publish_leaves_out_dir_absent(world, tmp_path, monkeypatch):
    """Finding #3: an assembly/writing failure partway through publication
    must not leave the requested ``out_dir`` behind as a partial result."""
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    _pin_authoritative_initial_roster_sha256(monkeypatch, initial_roster_path)
    entries = initial_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS
    )
    roster_path = _write_roster(
        world["tmp_path"] / "roster_failure.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
    )
    out_dir = tmp_path / "out_failure"

    def _boom(*args, **kwargs):
        raise RuntimeError("injected failure after base results are written")

    monkeypatch.setattr(runner, "_sha256_path", _boom)

    with pytest.raises(RuntimeError, match="injected failure"):
        runner.produce_rd1_continuation_results(
            trial_list_path=roster_path,
            contract_path=world["contract_path"],
            package_root=world["package_root"],
            phase=PHASE_BAYESIAN_CONTINUATION_36,
            out_dir=out_dir,
            initial_roster_path=initial_roster_path,
        )

    assert not out_dir.exists()
    leftover_tmp_dirs = list(tmp_path.glob(f"{out_dir.name}.tmp-*"))
    assert leftover_tmp_dirs == []


def test_incomplete_continuation_roster_is_refused_before_assembly(world, tmp_path):
    initial_entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    initial_roster_path = _write_frozen_initial_roster(world, initial_entries)
    entries = initial_entries + _build_phase_entries(
        world, PHASE_BAYESIAN_CONTINUATION_36, skip=_INITIAL_24_PAIRS | {("bayesian", 20)}
    )
    roster_path = _write_roster(
        world["tmp_path"] / "roster_incomplete.json",
        contract=world["contract"],
        phase=PHASE_BAYESIAN_CONTINUATION_36,
        entries=entries,
    )
    with pytest.raises(runner.ContinuationResultsRunnerError, match="not receipt-qualified"):
        runner.produce_rd1_continuation_results(
            trial_list_path=roster_path,
            contract_path=world["contract_path"],
            package_root=world["package_root"],
            phase=PHASE_BAYESIAN_CONTINUATION_36,
            out_dir=tmp_path / "out",
            initial_roster_path=initial_roster_path,
        )


def test_unknown_phase_is_refused_by_produce_entry_point(world, tmp_path):
    entries = _build_phase_entries(world, PHASE_INITIAL_CONTROLLED_24)
    roster_path = _write_roster(
        world["tmp_path"] / "roster.json",
        contract=world["contract"],
        phase=PHASE_INITIAL_CONTROLLED_24,
        entries=entries,
    )
    with pytest.raises(runner.ContinuationResultsRunnerError, match="unknown RD1 continuation phase"):
        runner.produce_rd1_continuation_results(
            trial_list_path=roster_path,
            contract_path=world["contract_path"],
            package_root=world["package_root"],
            phase="not_a_real_phase",
            out_dir=tmp_path / "out",
        )


def test_validate_continuation_batch_shape_rejects_an_out_of_range_proposal_order(world):
    """Direct unit-level check of the new phase-aware shape validator
    itself, independent of roster authentication: a source whose own
    receipt-derived ``proposal_order`` is outside the phase's admitted range
    must be refused even if it reached this function by some other path."""
    sources = []
    for arm, orders in EXPECTED_PROPOSAL_ORDERS_BY_PHASE[PHASE_BAYESIAN_CONTINUATION_36].items():
        for order in orders:
            if arm == "bayesian" and order == 24:
                order = 99  # out of range for bayesian_continuation_36 (admits P1-P24)
            _build_entry(world, search_arm=arm, proposal_order=order, label=f"{arm}_{order}_oor")
            from src.baseline.stage1_v2_12plus12_hydrological_consumer import build_v2_best_epoch_source

            receipt_path = world["tmp_path"] / "receipts" / f"{arm}_{order}_oor_execution_provenance.json"
            # ``build_v2_best_epoch_source`` requires ``execution_provenance``
            # to compare byte-for-byte equal to a *fresh* JSON parse of the
            # receipt file: JSON object keys are always strings, but the
            # in-memory ``record`` returned by ``_build_entry`` still has
            # integer trajectory keys, so reusing it here would always fail
            # closed on that identity check rather than reaching the
            # out-of-range-proposal check this test targets.
            receipt_record = json.loads(receipt_path.read_bytes())
            sources.append(
                build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
            )
    with pytest.raises(HydrologicalConsumerError, match="does not admit"):
        runner._validate_continuation_batch_shape(sources, world["contract"], phase=PHASE_BAYESIAN_CONTINUATION_36)
