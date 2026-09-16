"""RD1-C4-D1 section H: the completeness-gated reducer.

The production gate is 24 trials x 400 basins. These tests run a
proportionally scaled world -- 4 trials x 3 basins, built from small
synthetic packages and synthetic ``validation_results.p`` files in
``tmp_path`` -- because the property under test is *"refuses anything that
is not the exact declared cross-product"*, which is size-independent. The
production constants themselves are pinned separately in
:func:`test_the_production_gate_is_24_trials_12_plus_12_and_9600_cells`, so
scaling the fixture can never quietly scale the real gate.

The 23-shard and 9,599-cell cases in the task specification appear here as
their scaled equivalents: one shard short of the declared population, and
one cell short of the declared cross-product.

No real Moriah product is read, no job is submitted, and no scientific
conclusion is asserted.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from src.baseline.atomic_shard_store import AtomicShardStore
from src.baseline.rd1_c4_observation_diagnostic import (
    D1_SCHEMA_NAME,
    D1_SCHEMA_VERSION,
    D1_SHARD_FAMILY,
    run_trial_observation_diagnostic,
)
from src.baseline.rd1_c4_observation_diagnostic_reduce import (
    EXPECTED_ARM_COUNTS,
    EXPECTED_CELL_COUNT,
    EXPECTED_TRIAL_COUNT,
    ReductionError,
    reduce_observation_diagnostic,
)
from tests._rd1_c4_d1_support import (
    build_contract,
    build_package,
    trial_target,
    write_validation_pickle,
)

BASINS = ["00000001", "00000002", "00000003"]
EPOCH = 9

#: The scaled stand-in for the 12 + 12 production population.
TRIALS = {
    "t01": "bayesian",
    "t02": "bayesian",
    "t03": "random_control",
    "t04": "random_control",
}
ARM_COUNTS = {"bayesian": 2, "random_control": 2}

#: The nine outputs section H requires, plus the human-readable log.
REQUIRED_OUTPUTS = (
    "cells.parquet",
    "cell_status.csv",
    "extreme_elements.csv",
    "ulp_or_source_precision_histogram.csv",
    "q98_consequences.csv",
    "summary.json",
    "receipt_index.csv",
    "manifest.csv",
    "checksums.sha256",
    "reduction.log",
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _produced_world(tmp_path_factory):
    """Produce the four real trial shards exactly once for the whole module.

    Every shard is produced by the real section B-G runner -- not by hand --
    so what the reducer consumes here is genuinely what the producer emits.
    Tests copy this store rather than mutating it.
    """
    root = tmp_path_factory.mktemp("d1_reduce_world")
    package_root, dates, qobs = build_package(root, BASINS)
    contract = build_contract(package_root, BASINS, dates)
    store_root = root / "store"

    for trial_id, arm in TRIALS.items():
        run_dir = root / f"run_{trial_id}"
        write_validation_pickle(
            run_dir,
            EPOCH,
            basin_ids=BASINS,
            contract=contract,
            qobs_by_basin=qobs,
        )
        run_trial_observation_diagnostic(
            trial=trial_target(trial_id, run_dir, contract=contract, search_arm=arm, epoch=EPOCH),
            contract=contract,
            package_root=package_root,
            store_root=store_root,
            repo_root=root,
            attempt_token=f"attempt_{trial_id}",
            expected_basin_count=len(BASINS),
        )
    return {"root": root, "store_root": store_root, "contract": contract}


@pytest.fixture
def world(_produced_world, tmp_path):
    """A private, mutable copy of the produced store for one test."""
    store_root = tmp_path / "store"
    shutil.copytree(_produced_world["store_root"], store_root)

    class World:
        pass

    world = World()
    world.store_root = store_root
    world.out_dir = tmp_path / "reduction"
    world.store = AtomicShardStore(store_root, D1_SHARD_FAMILY)
    return world


def _reduce(world, **kwargs):
    return reduce_observation_diagnostic(
        store_root=world.store_root,
        out_dir=kwargs.pop("out_dir", world.out_dir),
        expected_trials=kwargs.pop("expected_trials", dict(TRIALS)),
        expected_basin_ids=kwargs.pop("expected_basin_ids", list(BASINS)),
        expected_trial_count=kwargs.pop("expected_trial_count", len(TRIALS)),
        expected_arm_counts=kwargs.pop("expected_arm_counts", dict(ARM_COUNTS)),
        expected_basin_count=kwargs.pop("expected_basin_count", len(BASINS)),
        **kwargs,
    )


def _restage(world, trial_id):
    """Un-publish one shard into a fresh attempt directory, ready to mutate.

    Done with a single directory rename rather than copy-then-delete. On
    Windows a just-removed directory name can stay briefly unusable, and the
    ``os.replace`` inside :meth:`AtomicShardStore.publish` then fails with a
    permission error that has nothing to do with the code under test.
    Renaming the shard out of the way never deletes the path at all.

    Returns the attempt directory and the original receipt payload, so the
    republished shard can carry the identity it had before.
    """
    shard_dir = world.store.shard_dir(trial_id)
    receipt = json.loads(world.store.receipt_path(trial_id).read_text(encoding="utf-8"))
    world.store.receipt_path(trial_id).unlink()
    attempt_dir = world.store.attempts_dir() / f"{trial_id}__republished"
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(shard_dir, attempt_dir)
    return attempt_dir, receipt


def _republish(world, trial_id, attempt_dir, receipt):
    world.store.publish(trial_id, attempt_dir=attempt_dir, identity=receipt["identity"])


def _republish_with_mutated_cells(world, trial_id, mutate):
    """Rewrite one shard's ``cells.jsonl`` and republish it legitimately.

    Editing a published shard in place would trip the component hash gate,
    which is a *different* gate. To reach the cell-level gates the mutated
    content has to arrive as a properly published, hash-consistent shard.
    """
    attempt_dir, receipt = _restage(world, trial_id)
    cells_path = attempt_dir / "cells.jsonl"
    cells = [
        json.loads(line)
        for line in cells_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cells = mutate(cells)
    cells_path.write_bytes(
        "".join(json.dumps(cell, sort_keys=True) + "\n" for cell in cells).encode("utf-8")
    )
    _republish(world, trial_id, attempt_dir, receipt)


def _rewrite_summary(world, trial_id, mutate):
    """Same idea for ``trial_summary.json``."""
    attempt_dir, receipt = _restage(world, trial_id)
    summary_path = attempt_dir / "trial_summary.json"
    summary = mutate(json.loads(summary_path.read_text(encoding="utf-8")))
    summary_path.write_bytes(json.dumps(summary, sort_keys=True, indent=2).encode("utf-8"))
    _republish(world, trial_id, attempt_dir, receipt)


# --------------------------------------------------------------------- #
# The production gate itself
# --------------------------------------------------------------------- #


def test_the_production_gate_is_24_trials_12_plus_12_and_9600_cells():
    """The scaled fixture must never be mistaken for the real population."""
    assert EXPECTED_TRIAL_COUNT == 24
    assert EXPECTED_ARM_COUNTS == {"bayesian": 12, "random_control": 12}
    assert EXPECTED_CELL_COUNT == 9600


def test_the_default_expected_population_must_still_be_supplied_in_full(world):
    """The defaults are the production numbers, so calling with the scaled
    world but without the scaled expectations refuses rather than reducing
    four shards as if they were twenty-four."""
    with pytest.raises(ReductionError, match="expected exactly 24 trial identities"):
        reduce_observation_diagnostic(
            store_root=world.store_root,
            out_dir=world.out_dir,
            expected_trials=dict(TRIALS),
            expected_basin_ids=list(BASINS),
        )


# --------------------------------------------------------------------- #
# The happy path: exactly the declared cross-product
# --------------------------------------------------------------------- #


def test_the_exact_cross_product_reduces_and_writes_every_required_output(world):
    summary = _reduce(world)

    assert summary["n_trials"] == len(TRIALS)
    assert summary["n_cells"] == len(TRIALS) * len(BASINS)
    assert summary["search_arm_counts"] == {
        "bayesian": 2 * len(BASINS),
        "random_control": 2 * len(BASINS),
    }
    for name in REQUIRED_OUTPUTS:
        assert (world.out_dir / name).is_file(), f"missing required output {name}"


def test_the_cell_table_holds_one_typed_row_per_trial_and_basin(world):
    import pyarrow.parquet as pq

    _reduce(world)
    table = pq.read_table(world.out_dir / "cells.parquet")
    assert table.num_rows == len(TRIALS) * len(BASINS)
    pairs = list(zip(table.column("trial_id").to_pylist(), table.column("basin_id").to_pylist()))
    assert sorted(pairs) == sorted((trial, basin) for trial in TRIALS for basin in BASINS)
    # Deterministically ordered by (trial_id, basin_id).
    assert pairs == sorted(pairs)
    assert set(table.column("search_arm").to_pylist()) == set(TRIALS.values())


def test_the_checksums_file_matches_the_manifest_and_the_bytes_on_disk(world):
    import hashlib

    _reduce(world)
    manifest = {}
    for line in (world.out_dir / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]:
        relative_path, sha256, size_bytes = line.split(",")
        manifest[relative_path] = (sha256, int(size_bytes))

    for relative_path, (sha256, size_bytes) in manifest.items():
        payload = (world.out_dir / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == sha256, relative_path
        assert len(payload) == size_bytes, relative_path

    checksum_lines = (world.out_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ")[1] for line in checksum_lines] == sorted(manifest)


def test_the_status_table_shows_an_explicit_zero_for_every_unused_status(world):
    _reduce(world)
    rows = (world.out_dir / "cell_status.csv").read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    assert "ok" in header
    assert "basin_missing_from_trial" in header
    body = [row.split(",") for row in rows[1:]]
    assert [row[0] for row in body] == [*sorted(TRIALS), "__all__"]
    total = dict(zip(header, body[-1]))
    assert int(total["n_cells"]) == len(TRIALS) * len(BASINS)
    assert int(total["basin_missing_from_trial"]) == 0


def test_the_reduction_is_deterministic(world, tmp_path):
    import hashlib

    first = _reduce(world, out_dir=tmp_path / "first")
    second = _reduce(world, out_dir=tmp_path / "second")

    def _digest(out_dir, name):
        return hashlib.sha256((Path(out_dir) / name).read_bytes()).hexdigest()

    for name in REQUIRED_OUTPUTS:
        if name in {"summary.json", "reduction.log", "manifest.csv", "checksums.sha256"}:
            # These embed ``reduced_at_utc``, which is provenance, not content.
            continue
        assert _digest(tmp_path / "first", name) == _digest(tmp_path / "second", name), name
    assert first["receipt_index_sha256"] == second["receipt_index_sha256"]


# --------------------------------------------------------------------- #
# Completeness gates: the scaled 23-shard and 9,599-cell cases
# --------------------------------------------------------------------- #


def test_one_shard_short_of_the_declared_population_is_refused(world):
    """The scaled form of 'reducer rejects 23 shards'."""
    world.store.receipt_path("t04").unlink()

    with pytest.raises(ReductionError, match="does not match the expected") as excinfo:
        _reduce(world)
    assert "missing=['t04']" in str(excinfo.value)
    assert not world.out_dir.exists()


def test_a_shard_directory_without_its_receipt_does_not_count_as_present(world):
    """Section G: a shard without a valid matching receipt is unavailable to
    reduction. The directory is still there and is still full of data -- that
    is exactly the situation that must not silently pass."""
    world.store.receipt_path("t03").unlink()
    assert world.store.shard_dir("t03").is_dir()
    assert (world.store.shard_dir("t03") / "cells.jsonl").is_file()

    with pytest.raises(ReductionError, match=r"missing=\['t03'\]"):
        _reduce(world)


def test_one_cell_short_of_the_declared_cross_product_is_refused(world):
    """The scaled form of 'reducer rejects 9,599 cells'."""
    _republish_with_mutated_cells(world, "t02", lambda cells: cells[:-1])

    with pytest.raises(ReductionError, match="holds 2 cell records, expected exactly 3"):
        _reduce(world)
    assert not world.out_dir.exists()


def test_a_duplicated_cell_record_is_refused(world):
    """Padding a short shard back to the declared count with a duplicate must
    not restore it: the count gate alone would be satisfied."""
    _republish_with_mutated_cells(world, "t02", lambda cells: [*cells[:-1], dict(cells[0])])

    with pytest.raises(ReductionError, match="duplicate cell record for basin"):
        _reduce(world)
    assert not world.out_dir.exists()


def test_an_unexpected_basin_is_refused_even_at_the_right_count(world):
    def _rename_last(cells):
        cells = [dict(cell) for cell in cells]
        cells[-1]["basin_id"] = "99999999"
        return cells

    _republish_with_mutated_cells(world, "t02", _rename_last)

    with pytest.raises(ReductionError, match="unexpected basin id '99999999'"):
        _reduce(world)


def test_a_cell_claiming_a_different_trial_is_refused(world):
    def _retag(cells):
        cells = [dict(cell) for cell in cells]
        cells[0]["trial_id"] = "t01"
        return cells

    _republish_with_mutated_cells(world, "t02", _retag)

    with pytest.raises(ReductionError, match="cell record claims trial 't01'"):
        _reduce(world)


def test_an_unknown_cell_status_is_refused(world):
    def _retype(cells):
        cells = [dict(cell) for cell in cells]
        cells[0]["status"] = "probably_fine"
        return cells

    _republish_with_mutated_cells(world, "t02", _retype)

    with pytest.raises(ReductionError, match="unknown cell status 'probably_fine'"):
        _reduce(world)


def test_an_unexpected_trial_in_the_store_is_refused(world, tmp_path):
    """A 25th shard is as disqualifying as a missing one: the reducer must
    reduce the declared population, not whatever is lying in the store."""
    shutil.copytree(world.store.shard_dir("t01"), world.store.shard_dir("t99"))
    receipt = json.loads(world.store.receipt_path("t01").read_text(encoding="utf-8"))
    receipt["shard_id"] = "t99"
    world.store.receipt_path("t99").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ReductionError) as excinfo:
        _reduce(world)
    assert "t99" in str(excinfo.value)


def test_a_trial_whose_shard_records_the_other_search_arm_is_refused(world):
    """The 12 + 12 split is part of the population definition, so a shard
    that disagrees with the declared arm cannot be reduced under it."""
    expected = dict(TRIALS)
    expected["t01"], expected["t03"] = "random_control", "bayesian"

    with pytest.raises(ReductionError, match="records search arm 'bayesian', expected 'random_control'"):
        _reduce(world, expected_trials=expected)


def test_an_unbalanced_expected_population_is_refused_before_any_shard_is_read(world):
    expected = dict(TRIALS)
    expected["t03"] = "bayesian"

    with pytest.raises(ReductionError, match="expected exactly 2 'bayesian' trials, was given 3"):
        _reduce(world, expected_trials=expected)


def test_an_unknown_search_arm_in_the_expected_population_is_refused(world):
    """The declared arms must account for the whole population.

    Reaching this gate needs a declaration whose per-arm counts are each
    satisfiable while leaving a trial over -- here 2 Bayesian + 1 control
    declared against a 4-trial population containing a third arm. The
    per-arm count gate is satisfied, so only this check can catch the
    unaccounted-for arm.
    """
    expected = dict(TRIALS)
    expected["t04"] = "grid_search"

    with pytest.raises(ReductionError, match="unexpected search arm.*grid_search"):
        _reduce(
            world,
            expected_trials=expected,
            expected_arm_counts={"bayesian": 2, "random_control": 1},
        )


def test_duplicate_expected_basin_ids_are_refused(world):
    with pytest.raises(ReductionError, match="expected exactly 3 distinct basin ids"):
        _reduce(world, expected_basin_ids=[BASINS[0], BASINS[1], BASINS[1]])


# --------------------------------------------------------------------- #
# Integrity gates
# --------------------------------------------------------------------- #


def test_a_modified_shard_component_is_caught_by_re_hashing(world):
    """'Reducer rejects hash mismatches.' The receipt is untouched and the
    edit is plausible -- only re-hashing the component can catch it."""
    cells_path = world.store.shard_dir("t03") / "cells.jsonl"
    payload = cells_path.read_bytes()
    assert b'"status":"ok"' in payload
    # A byte-for-byte same-size edit, written as bytes so no newline
    # translation changes the length: the cheap size check cannot see this,
    # and only re-hashing the component can.
    edited = payload.replace(b'"status":"ok"', b'"status":"0k"')
    assert len(edited) == len(payload)
    cells_path.write_bytes(edited)

    with pytest.raises(ReductionError, match="t03: shard failed hash verification"):
        _reduce(world)
    assert not world.out_dir.exists()


def test_a_truncated_shard_component_is_caught_and_never_read_as_absent(world):
    """A truncated component is caught one gate earlier, by the cheap size
    check -- and reported as damage, not as a missing shard. The distinction
    matters: 'absent' invites a rerun, 'damaged' does not."""
    (world.store.shard_dir("t01") / "cells.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ReductionError, match="not in a reducible state") as excinfo:
        _reduce(world)
    assert "damaged, not merely incomplete" in str(excinfo.value)
    assert not world.out_dir.exists()


def test_a_shard_from_a_different_schema_version_is_refused(world):
    _rewrite_summary(world, "t02", lambda summary: {**summary, "schema_version": D1_SCHEMA_VERSION + 1})

    with pytest.raises(ReductionError, match="shard schema .* != "):
        _reduce(world)


def test_shards_disagreeing_on_the_contract_identity_are_refused(world):
    def _retag(summary):
        summary = {**summary, "identity": dict(summary["identity"])}
        summary["identity"]["contract_checksum_sha256"] = "0" * 64
        return summary

    _rewrite_summary(world, "t02", _retag)

    with pytest.raises(ReductionError, match="disagree on the fixed-support contract identity"):
        _reduce(world)


def test_shards_disagreeing_on_the_package_identity_are_refused(world):
    """Two trials compared against two different observation packages cannot
    be joined into one 24x400 statement, however complete the table looks."""

    def _retag(summary):
        summary = {**summary, "identity": dict(summary["identity"])}
        summary["identity"]["package"] = {
            **summary["identity"]["package"],
            "package_manifest_sha256": "1" * 64,
        }
        return summary

    _rewrite_summary(world, "t03", _retag)

    with pytest.raises(ReductionError, match="disagree on the package identity"):
        _reduce(world)


def test_an_unparseable_cell_record_is_refused_with_its_line_number(world):
    """Broken JSONL inside an otherwise hash-consistent shard. The component
    hash gate cannot see this -- the bytes are exactly what the receipt
    records -- so the parser has to refuse it and say where."""
    attempt_dir, receipt = _restage(world, "t02")
    (attempt_dir / "extremes.jsonl").write_bytes(b"{not json}\n")
    _republish(world, "t02", attempt_dir, receipt)

    with pytest.raises(ReductionError, match=r"extremes\.jsonl:1: unparseable cell record"):
        _reduce(world)


def test_nothing_at_all_is_written_when_a_gate_fails(world):
    """Section H: the reducer writes nothing when it raises. A partially
    written output directory is worse than none -- it would be read."""
    world.out_dir.mkdir(parents=True)
    (world.out_dir / "pre_existing.txt").write_text("untouched", encoding="utf-8")
    world.store.receipt_path("t04").unlink()

    with pytest.raises(ReductionError):
        _reduce(world)

    assert sorted(path.name for path in world.out_dir.iterdir()) == ["pre_existing.txt"]


# --------------------------------------------------------------------- #
# Typed error cells are evidence, not grounds for refusal
# --------------------------------------------------------------------- #


def test_explicit_non_ok_cells_are_reduced_rather_than_dropped(tmp_path):
    """'Reducer accepts exactly 24x400 including explicit non-ok cells.'

    A basin missing from one trial's pickle is a typed cell, not a hole. The
    cross-product stays complete, the reduction proceeds, and the coverage
    numbers say plainly that one cell was not compared.
    """
    store_root = tmp_path / "store"
    package_root, dates, qobs = build_package(tmp_path, BASINS)
    contract = build_contract(package_root, BASINS, dates)

    for trial_id, arm in TRIALS.items():
        run_dir = tmp_path / f"run_{trial_id}"
        kwargs = {"omit_basins": [BASINS[-1]]} if trial_id == "t02" else {}
        write_validation_pickle(
            run_dir,
            EPOCH,
            basin_ids=BASINS,
            contract=contract,
            qobs_by_basin=qobs,
            **kwargs,
        )
        run_trial_observation_diagnostic(
            trial=trial_target(trial_id, run_dir, contract=contract, search_arm=arm, epoch=EPOCH),
            contract=contract,
            package_root=package_root,
            store_root=store_root,
            repo_root=tmp_path,
            attempt_token=f"attempt_{trial_id}",
            expected_basin_count=len(BASINS),
        )

    summary = reduce_observation_diagnostic(
        store_root=store_root,
        out_dir=tmp_path / "reduction",
        expected_trials=dict(TRIALS),
        expected_basin_ids=list(BASINS),
        expected_trial_count=len(TRIALS),
        expected_arm_counts=dict(ARM_COUNTS),
        expected_basin_count=len(BASINS),
    )

    assert summary["n_cells"] == len(TRIALS) * len(BASINS)
    assert summary["cell_status_counts"]["basin_missing_from_trial"] == 1
    assert summary["coverage"]["n_cells_not_compared"] == 1
    assert summary["coverage"]["n_cells_compared"] == len(TRIALS) * len(BASINS) - 1
    assert summary["coverage"]["fraction_cells_compared"] == pytest.approx(
        (len(TRIALS) * len(BASINS) - 1) / (len(TRIALS) * len(BASINS))
    )


# --------------------------------------------------------------------- #
# The reducer reports; it does not classify
# --------------------------------------------------------------------- #


def test_the_summary_publishes_no_threshold_and_no_classification(world):
    summary = _reduce(world)

    assert summary["provisional_envelope_is_report_only"] is True
    assert summary["final_rd1_c4_audit_policy"] == "unresolved"
    assert "not performed" in summary["scientific_classification"]
    for forbidden in ("pass", "fail", "verdict", "tolerance_selected", "closed"):
        assert forbidden not in summary


def test_the_completion_log_states_that_nothing_is_closed(world):
    _reduce(world)
    log = (world.out_dir / "reduction.log").read_text(encoding="utf-8")

    assert "REPORTED comparison field, not a pass criterion" in log
    assert "selects no threshold" in log
    assert "remains unresolved" in log


def test_the_envelope_count_is_reported_without_being_judged(world):
    summary = _reduce(world)
    coverage = summary["coverage"]

    assert "n_elements_exceeding_provisional_envelope" in coverage
    assert isinstance(coverage["n_elements_exceeding_provisional_envelope"], int)
    assert coverage["n_elements_compared"] >= coverage["n_elements_unequal"]


# --------------------------------------------------------------------- #
# Vertical integration: producer -> shard -> receipt -> reducer
# --------------------------------------------------------------------- #


def test_producer_to_shard_to_receipt_to_reducer_end_to_end(tmp_path):
    """One vertical pass with nothing hand-built between the stages.

    The producer writes the shards, the store writes the receipts, and the
    reducer consumes only what those two produced -- so a change that breaks
    the seam between them cannot pass by agreeing with a fixture.
    """
    package_root, dates, qobs = build_package(tmp_path, BASINS)
    contract = build_contract(package_root, BASINS, dates)
    store_root = tmp_path / "store"
    produced = {}

    for trial_id, arm in TRIALS.items():
        run_dir = tmp_path / f"run_{trial_id}"
        write_validation_pickle(
            run_dir,
            EPOCH,
            basin_ids=BASINS,
            contract=contract,
            qobs_by_basin=qobs,
        )
        produced[trial_id] = run_trial_observation_diagnostic(
            trial=trial_target(trial_id, run_dir, contract=contract, search_arm=arm, epoch=EPOCH),
            contract=contract,
            package_root=package_root,
            store_root=store_root,
            repo_root=tmp_path,
            attempt_token=f"vertical_{trial_id}",
            expected_basin_count=len(BASINS),
        )

    out_dir = tmp_path / "reduction"
    summary = reduce_observation_diagnostic(
        store_root=store_root,
        out_dir=out_dir,
        expected_trials=dict(TRIALS),
        expected_basin_ids=list(BASINS),
        expected_trial_count=len(TRIALS),
        expected_arm_counts=dict(ARM_COUNTS),
        expected_basin_count=len(BASINS),
    )

    assert summary["schema_name"] == D1_SCHEMA_NAME
    assert summary["schema_version"] == D1_SCHEMA_VERSION
    assert summary["n_cells"] == len(TRIALS) * len(BASINS)

    # The receipt index must name the same receipts the store actually holds.
    rows = (out_dir / "receipt_index.csv").read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    indexed = {}
    for line in rows[1:]:
        row = dict(zip(header, line.split(",")))
        indexed[row["trial_id"]] = row
    assert sorted(indexed) == sorted(TRIALS)
    for trial_id, row in indexed.items():
        receipt = produced[trial_id]["receipt"]
        assert row["content_sha256"] == receipt["content_sha256"]
        assert row["identity_sha256"] == receipt["identity_sha256"]
        assert row["search_arm"] == TRIALS[trial_id]

    # And the Q98 consequence layer must carry one row per compared cell.
    q98_rows = (out_dir / "q98_consequences.csv").read_text(encoding="utf-8").splitlines()
    assert len(q98_rows) - 1 == len(TRIALS) * len(BASINS)
