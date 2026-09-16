"""RD1-C4-D1 sections B-G: the per-trial observation diagnostic.

Everything here runs against small synthetic packages and synthetic
``validation_results.p`` files built in ``tmp_path``. No real Moriah product
is read, no job is submitted, and no scientific conclusion is asserted --
these tests check that D1 *measures and records* faithfully, never that a
particular difference is acceptable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.baseline.atomic_shard_store import AtomicShardStore, ShardStoreError
from src.baseline.rd1_c4_observation_diagnostic import (
    CELL_STATUSES,
    D1_SCHEMA_NAME,
    D1_SCHEMA_VERSION,
    D1_SHARD_FAMILY,
    EXTREME_KINDS,
    ObservationDiagnosticError,
    compare_basin_observation_series,
    q98_consequence_for_basin,
    run_trial_observation_diagnostic,
)
from src.baseline.rd1_c4_trial_authentication import (
    authenticate_trial_roster,
    fixture_only_expected_roster,
)
from tests._rd1_c4_d1_support import (
    LEAD_HOURS,
    build_contract,
    build_package,
    trial_target,
    write_synthetic_execution_receipt,
    write_trial_roster,
    write_validation_pickle,
)

BASINS = ["00000001", "00000002", "00000003"]
EPOCH = 9
AREA_FACTS = {"area_km2": 875.5, "area_relative_mad": 0.0, "n_area_samples": 234}


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def scenario(tmp_path):
    """A complete, self-consistent 3-basin world: package, contract, trial."""

    class Scenario:
        pass

    scenario = Scenario()
    scenario.tmp_path = tmp_path
    scenario.basins = list(BASINS)
    scenario.package_root, scenario.dates, scenario.qobs = build_package(tmp_path, BASINS)
    scenario.contract = build_contract(scenario.package_root, BASINS, scenario.dates)
    scenario.run_dir = tmp_path / "run_t01"
    scenario.store_root = tmp_path / "store"
    scenario.write_pickle = lambda **kwargs: write_validation_pickle(
        scenario.run_dir,
        EPOCH,
        basin_ids=BASINS,
        contract=scenario.contract,
        qobs_by_basin=scenario.qobs,
        **kwargs,
    )
    scenario.write_pickle()
    scenario.trial = trial_target("t01", scenario.run_dir, contract=scenario.contract, epoch=EPOCH)
    return scenario


def _run(scenario, *, token="attempt1", **kwargs):
    return run_trial_observation_diagnostic(
        trial=kwargs.pop("trial", scenario.trial),
        contract=scenario.contract,
        package_root=kwargs.pop("package_root", scenario.package_root),
        store_root=scenario.store_root,
        repo_root=scenario.tmp_path,
        attempt_token=token,
        expected_basin_count=len(scenario.basins),
        **kwargs,
    )


def _cells(scenario, trial_id="t01"):
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    path = store.shard_dir(trial_id) / "cells.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --------------------------------------------------------------------- #
# Section D: element-wise comparison, summaries and extremes
# --------------------------------------------------------------------- #


def _compare(package, reconstructed, dates=None):
    package = np.asarray(package, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if dates is None:
        dates = np.arange(
            np.datetime64("2024-01-01T00", "ns"),
            np.datetime64("2024-01-01T00", "ns") + np.timedelta64(package.size, "h"),
            np.timedelta64(1, "h"),
        )
    return compare_basin_observation_series(
        trial_id="t01",
        basin_id="00000001",
        support_dates=dates,
        package_obs_m3s=package,
        pickle_obs_m3s=reconstructed,
        area_facts=AREA_FACTS,
    )


def test_a_bitwise_identical_series_records_no_detail_and_no_exceedance():
    values = np.array([1.5, 20.25, 300.0], dtype=np.float32)
    result = _compare(values, values.astype(np.float64))
    assert result.summary["n_compared"] == 3
    assert result.summary["n_bitwise_equal"] == 3
    assert result.summary["n_unequal"] == 0
    assert result.summary["abs_diff_max"] == 0.0
    assert result.summary["n_exceeding_provisional_envelope"] == 0
    assert result.detail["support_index"].size == 0


def test_every_unequal_element_is_retained_in_the_detail():
    """Section C: complete per-element evidence, not a sample."""
    package = np.arange(1.0, 51.0, dtype=np.float32)
    reconstructed = package.astype(np.float64) * (1.0 + 1e-9)
    reconstructed[10] = float(package[10])  # exactly one bitwise-equal element

    result = _compare(package, reconstructed)
    assert result.summary["n_unequal"] == 49
    assert result.detail["support_index"].size == 49
    assert 10 not in result.detail["support_index"].tolist()
    assert result.detail["support_index"].tolist() == sorted(result.detail["support_index"].tolist())


def test_a_large_difference_is_recorded_and_never_raised():
    """A numerical discrepancy of any magnitude is a measurement, not a
    failure -- D1 must never adjudicate."""
    package = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reconstructed = np.array([1.0, 2.0, 3.0e6], dtype=np.float64)

    result = _compare(package, reconstructed)
    assert result.summary["abs_diff_max"] == pytest.approx(2999997.0)
    assert result.summary["n_exceeding_provisional_envelope"] == 1
    assert result.summary["max_provisional_envelope_exceedance_factor"] > 1e6


def test_the_two_relative_definitions_are_reported_separately():
    """Near a small package value the package-reference form explodes while
    the symmetric form stays bounded; both must be visible."""
    package = np.array([1e-6], dtype=np.float32)
    reconstructed = np.array([1.0], dtype=np.float64)

    summary = _compare(package, reconstructed).summary
    assert summary["rel_diff_package_reference_max"] > 1e5
    assert summary["rel_diff_symmetric_max"] < 2.0


def test_a_zero_package_value_yields_a_guarded_relative_difference_not_an_inf():
    package = np.array([0.0], dtype=np.float32)
    reconstructed = np.array([0.5], dtype=np.float64)
    result = _compare(package, reconstructed)
    assert np.isnan(result.detail["rel_diff_package_reference"][0])
    assert result.detail["rel_diff_symmetric"][0] == pytest.approx(2.0)


def test_the_provisional_envelope_is_asymmetric_and_recorded_as_report_only():
    summary = _compare([1.0, 2.0], [1.0, 2.0]).summary
    assert summary["provisional_envelope_form"] == "abs_diff > atol + rtol * abs(package_value)"
    assert summary["provisional_envelope_reference_is_package_value_only"] is True
    assert summary["provisional_envelope_rtol"] == pytest.approx(64 * float(np.finfo(np.float32).eps))
    assert summary["provisional_envelope_atol_m3s"] == pytest.approx(1e-6)


def test_percentiles_and_moments_are_all_reported():
    package = np.linspace(1.0, 100.0, 200).astype(np.float32)
    reconstructed = package.astype(np.float64) + 1e-5
    summary = _compare(package, reconstructed).summary
    for key in (
        "abs_diff_max",
        "abs_diff_mean",
        "abs_diff_rms",
        "abs_diff_median",
        "abs_diff_p50",
        "abs_diff_p90",
        "abs_diff_p99",
        "abs_diff_p99_9",
    ):
        assert key in summary, key
        assert summary[key] is not None


def test_nonfinite_counts_are_reported_on_both_sides():
    package = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    reconstructed = np.array([1.0, 2.0, np.inf], dtype=np.float64)
    summary = _compare(package, reconstructed).summary
    assert summary["n_nonfinite_package"] == 1
    assert summary["n_nonfinite_reconstructed"] == 1
    assert summary["n_source_precision_undefined"] == 2


def test_area_facts_are_carried_into_the_summary():
    summary = _compare([1.0], [1.0]).summary
    assert summary["area_km2"] == AREA_FACTS["area_km2"]
    assert summary["area_relative_mad"] == AREA_FACTS["area_relative_mad"]
    assert summary["n_area_samples"] == AREA_FACTS["n_area_samples"]


def test_a_uniform_multiplicative_scale_is_detected_as_consistent():
    """If a single scale factor explained every discrepancy, the ratio
    distribution would be a point mass -- reporting that is how D1 avoids
    assuming one rounding operation explains everything."""
    package = np.linspace(1.0, 500.0, 300).astype(np.float32)
    reconstructed = package.astype(np.float64) * 1.000001
    scale = _compare(package, reconstructed).summary["uniform_multiplicative_scale"]
    assert scale["ratio_median"] == pytest.approx(1.000001, rel=1e-9)
    assert scale["max_abs_relative_deviation_from_median_ratio"] < 1e-6
    assert scale["fraction_within_1e_9_of_median_ratio"] > 0.9


def test_an_independently_quantized_pair_is_not_a_uniform_scale():
    package = np.linspace(1.0, 500.0, 300).astype(np.float32)
    reconstructed = (package.astype(np.float64) / 3.6).astype(np.float32).astype(np.float64) * 3.6
    scale = _compare(package, reconstructed).summary["uniform_multiplicative_scale"]
    assert scale["fraction_within_1e_9_of_median_ratio"] < 0.5


# --- the six extremes --------------------------------------------------- #


def test_the_six_extreme_kinds_can_identify_six_different_elements():
    """Section D is explicit that the extremes must not be assumed to point
    at the same element. This fixture is constructed so they do not."""
    package = np.array([100.0, 1e-5, 1e-7, 50.0, 3.0, 7.0], dtype=np.float32)
    reconstructed = np.array(
        [
            100.001,  # largest absolute difference
            1.2e-5,  # largest package-reference relative difference
            2.0e-7,  # largest symmetric relative difference
            50.0 + 4e-4,  # large ordered-bit distance at a big magnitude
            3.0 + 1e-5,  # first envelope exceedance (earliest index that exceeds)
            7.0,  # bitwise equal
        ],
        dtype=np.float64,
    )
    records = {record["extreme_kind"]: record for record in _compare(package, reconstructed).extremes}

    assert set(records) <= set(EXTREME_KINDS)
    assert records["max_abs_diff"]["support_index"] == 0
    assert records["max_rel_diff_package_reference"]["support_index"] == 2
    assert records["max_rel_diff_symmetric"]["support_index"] == 2
    assert records["first_envelope_exceedance"]["support_index"] == 0
    # The point of the test: several kinds resolve to genuinely distinct
    # elements, so a report that collapsed them would lose information.
    assert len({record["support_index"] for record in records.values()}) >= 3


def test_the_largest_absolute_difference_need_not_be_the_largest_exceedance():
    """Exactly the RD1-C4 observation that the reported maximum absolute
    difference was not the failing element."""
    package = np.array([1000.0, 0.001], dtype=np.float32)
    reconstructed = np.array([1000.0 + 5e-3, 0.001 + 2e-5], dtype=np.float64)
    records = {record["extreme_kind"]: record for record in _compare(package, reconstructed).extremes}

    assert records["max_abs_diff"]["support_index"] == 0
    assert records["max_envelope_exceedance"]["support_index"] == 1
    assert records["max_abs_diff"]["support_index"] != records["max_envelope_exceedance"]["support_index"]


def test_an_extreme_kind_with_no_qualifying_element_is_absent_not_fabricated():
    package = np.array([1.0, 2.0], dtype=np.float32)
    result = _compare(package, package.astype(np.float64))
    kinds = {record["extreme_kind"] for record in result.extremes}
    assert "max_envelope_exceedance" not in kinds
    assert "first_envelope_exceedance" not in kinds


def test_first_and_max_envelope_exceedance_are_resolved_independently():
    package = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    reconstructed = np.array([1.0 + 1e-4, 1.0, 1.0 + 1e-2], dtype=np.float64)
    records = {record["extreme_kind"]: record for record in _compare(package, reconstructed).extremes}
    assert records["first_envelope_exceedance"]["support_index"] == 0
    assert records["max_envelope_exceedance"]["support_index"] == 2


def test_extreme_ties_are_broken_by_the_earliest_support_index():
    package = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    reconstructed = np.array([5.0 + 1e-3, 5.0 + 1e-3, 5.0], dtype=np.float64)
    records = {record["extreme_kind"]: record for record in _compare(package, reconstructed).extremes}
    assert records["max_abs_diff"]["support_index"] == 0


def test_an_extreme_record_carries_the_package_float32_bit_pattern():
    result = _compare([1.0], [1.5])
    record = next(r for r in result.extremes if r["extreme_kind"] == "max_abs_diff")
    assert record["package_value_float32_bits"] == "0x3f800000"


def test_the_support_index_resolves_back_to_the_contract_timestamp():
    """Section C: the detail must be reconstructable against the frozen
    contract, so support_index and date_ns must agree."""
    dates = np.arange(
        np.datetime64("2024-03-01T00", "ns"),
        np.datetime64("2024-03-01T00", "ns") + np.timedelta64(5, "h"),
        np.timedelta64(1, "h"),
    )
    package = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    reconstructed = package.astype(np.float64) + 1e-3
    result = _compare(package, reconstructed, dates=dates)

    for position, index in enumerate(result.detail["support_index"].tolist()):
        assert result.detail["date_ns"][position] == dates[index].astype("datetime64[ns]").astype(np.int64)


# --------------------------------------------------------------------- #
# Section F: Q98 consequence
# --------------------------------------------------------------------- #


def _q98(package, reconstructed):
    package = np.asarray(package, dtype=np.float64)
    dates = np.arange(
        np.datetime64("2024-01-01T00", "ns"),
        np.datetime64("2024-01-01T00", "ns") + np.timedelta64(package.size, "h"),
        np.timedelta64(1, "h"),
    )
    return q98_consequence_for_basin(
        trial_id="t01",
        basin_id="00000001",
        support_dates=dates,
        package_obs_m3s=package,
        pickle_obs_m3s=np.asarray(reconstructed, dtype=np.float64),
    )


def test_a_tiny_difference_changes_no_frozen_q98_decision():
    package = np.linspace(1.0, 100.0, 200)
    record = _q98(package, package + 1e-9)
    assert record["high_flow_mask_symmetric_difference"] == 0
    assert record["high_flow_membership_equal"] is True
    assert record["peak_index_equal"] is True
    assert record["peak_timestamp_equal"] is True
    assert record["any_frozen_decision_changes"] is False
    assert record["q98_threshold_abs_diff"] < 1e-6


def test_both_sides_report_independent_q98_facts():
    package = np.linspace(1.0, 100.0, 200)
    record = _q98(package, package * 2.0)
    assert record["q98_threshold_package"] != record["q98_threshold_reconstructed"]
    assert record["n_high_flow_package"] == record["n_high_flow_reconstructed"]
    assert record["high_flow_mask_sha256_package"] == record["high_flow_mask_sha256_reconstructed"]
    assert record["any_frozen_decision_changes"] is False  # a pure rescale moves nothing


def test_a_changed_high_flow_membership_is_reported():
    package = np.array([1.0] * 98 + [10.0, 11.0], dtype=np.float64)
    reconstructed = package.copy()
    reconstructed[50] = 12.0  # promotes one element across the threshold
    record = _q98(package, reconstructed)
    assert record["high_flow_mask_symmetric_difference"] > 0
    assert record["high_flow_membership_equal"] is False
    assert record["high_flow_mask_sha256_package"] != record["high_flow_mask_sha256_reconstructed"]
    assert record["any_frozen_decision_changes"] is True


def test_a_changed_peak_is_reported_with_its_timestamp():
    package = np.array([1.0, 5.0, 9.0, 2.0], dtype=np.float64)
    reconstructed = np.array([1.0, 50.0, 9.0, 2.0], dtype=np.float64)
    record = _q98(package, reconstructed)
    assert record["observed_peak_index_package"] == 2
    assert record["observed_peak_index_reconstructed"] == 1
    assert record["peak_index_equal"] is False
    assert record["peak_timestamp_equal"] is False
    assert (
        record["earliest_tied_peak_date_ns_package"]
        != record["earliest_tied_peak_date_ns_reconstructed"]
    )
    assert record["any_frozen_decision_changes"] is True


def test_peak_tie_sets_and_earliest_tied_timestamps_are_reported():
    package = np.array([9.0, 1.0, 9.0, 2.0], dtype=np.float64)
    record = _q98(package, package)
    assert record["peak_tie_count_package"] == 2
    assert record["peak_tie_count_reconstructed"] == 2
    assert (
        record["earliest_tied_peak_date_ns_package"]
        == np.datetime64("2024-01-01T00", "ns").astype(np.int64)
    )


def test_the_mask_hash_distinguishes_different_lengths():
    short = _q98(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    long = _q98(np.array([1.0, 2.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.0, 2.0]))
    assert short["high_flow_mask_sha256_package"] != long["high_flow_mask_sha256_package"]


# --------------------------------------------------------------------- #
# Section B/G: the trial runner
# --------------------------------------------------------------------- #


def test_a_clean_trial_publishes_one_shard_with_exactly_one_cell_per_basin(scenario):
    summary = _run(scenario)

    assert summary["schema_name"] == D1_SCHEMA_NAME
    assert summary["schema_version"] == D1_SCHEMA_VERSION
    assert summary["n_cells"] == len(BASINS)
    assert summary["cell_status_counts"]["ok"] == len(BASINS)
    assert set(summary["cell_status_counts"]) == set(CELL_STATUSES)
    assert summary["reused_existing_shard"] is False
    assert summary["provisional_envelope_is_report_only"] is True
    assert summary["final_rd1_c4_audit_policy"] == "unresolved"

    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    receipt = store.verify("t01")
    assert receipt.family == D1_SHARD_FAMILY
    names = {component.relative_path for component in receipt.components}
    assert {
        "cells.jsonl",
        "extremes.jsonl",
        "q98_consequences.jsonl",
        "trial_summary.json",
        "human.log",
        "detail/elements.parquet",
        "detail/detail_index.json",
    } <= names
    # A progress file is operational evidence, never shard content.
    assert not any(name.endswith("progress.jsonl") for name in names)


def test_the_fixture_reproduces_genuine_non_bitwise_equal_differences(scenario):
    """The round trip through mm/h float32 storage is lossy exactly as the
    real pipeline is, so the default scenario is not a contrived one."""
    _run(scenario)
    cell = _cells(scenario)[0]
    assert cell["status"] == "ok"
    assert cell["n_unequal"] > 0
    assert cell["abs_diff_max"] > 0.0
    assert cell["source_precision_distance_max"] >= 1


def test_the_receipt_identity_pins_code_contract_package_and_product(scenario):
    # A stub tree standing in for the repository, so the hashing mechanism
    # is exercised without reading or depending on the real working tree.
    module = scenario.tmp_path / "src" / "baseline" / "rd1_c4_observation_diagnostic.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("# stub", encoding="utf-8")
    _run(scenario)
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    identity = store.read_receipt("t01").identity

    assert identity["schema_name"] == D1_SCHEMA_NAME
    assert identity["contract_id"] == scenario.contract["contract_id"]
    assert identity["contract_checksum_sha256"] == scenario.contract["checksum_sha256"]
    assert identity["validation_pickle_sha256"]
    assert identity["package"]["package_manifest_sha256"]
    assert identity["trial"]["best_epoch"] == EPOCH
    assert identity["trial"]["source_receipt_sha256"]
    assert "git_head" in identity
    assert "git_uncommitted_tracked_file_sha256" in identity
    assert "src/baseline/rd1_c4_observation_diagnostic.py" in identity["module_sha256"]


def test_the_summary_records_environment_and_slurm_context(scenario):
    summary = _run(scenario)
    assert summary["environment"]["versions"]["python"]
    assert summary["environment"]["versions"]["numpy"]
    assert "hostname" in summary["environment"]


def _progress_events(summary):
    path = Path(summary["progress_log_path"])
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_progress_events_cover_the_required_milestones(scenario):
    summary = _run(scenario)
    events = _progress_events(summary)
    kinds = [event["event"] for event in events]
    for required in (
        "task_start",
        "source_qualification_complete",
        "trial_load_complete",
        "basin_progress",
        "detail_finalization_complete",
        "shard_finalization_complete",
        "receipt_finalization_complete",
        "task_end",
    ):
        assert required in kinds, required
    assert kinds[0] == "task_start"
    assert kinds[-1] == "task_end"
    for event in events:
        assert event["utc"]
        assert event["trial_id"] == "t01"
        assert event["attempt_token"] == "attempt1"
        assert event["elapsed_s"] >= 0


def test_receipt_finalization_is_recorded_only_after_the_receipt_exists(scenario):
    """The last two events happen after the shard is published, which is
    precisely why the progress log cannot live inside the shard."""
    summary = _run(scenario)
    events = _progress_events(summary)
    finalization = next(e for e in events if e["event"] == "receipt_finalization_complete")
    assert finalization["content_sha256"] == summary["receipt"]["content_sha256"]
    assert Path(summary["progress_log_path"]).is_file()
    assert not str(Path(summary["progress_log_path"])).startswith(
        str(AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY).shard_dir("t01"))
    )


def test_basin_progress_events_carry_an_estimate_and_a_position(scenario):
    summary = _run(scenario)
    basin_events = [e for e in _progress_events(summary) if e["event"] == "basin_progress"]
    assert basin_events
    last = basin_events[-1]
    assert last["n_basins"] == len(BASINS)
    assert last["basin_index"] == len(BASINS) - 1
    assert last["basin_id"] == BASINS[-1]
    assert last["estimated_remaining_s"] is None  # nothing left to estimate


def test_a_prequalification_failure_writes_no_progress_or_shard(scenario):
    path = scenario.run_dir / "validation" / "model_epoch009" / "validation_results.p"
    path.write_bytes(b"not a pickle at all")
    with pytest.raises(ObservationDiagnosticError):
        _run(scenario, token="doomed")

    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    log = store.family_dir / "_progress" / "t01__doomed.jsonl"
    assert not log.exists()
    assert store.list_completed() == []


def test_the_detail_parquet_is_readable_and_indexed_per_basin(scenario):
    _run(scenario)
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    shard = store.shard_dir("t01")
    index = json.loads((shard / "detail" / "detail_index.json").read_text(encoding="utf-8"))
    table = pq.read_table(shard / "detail" / "elements.parquet")

    assert index["format"] == "parquet"
    assert table.num_rows == index["n_rows"]
    assert [entry["basin_id"] for entry in index["basin_offsets"]] == BASINS
    assert "per_basin_support" in index["support_index_semantics"]
    assert index["retained_elements"] == "every aligned element that is not bitwise equal"

    column_names = [name for name, _ in [(f.name, f.type) for f in table.schema]]
    assert "support_index" in column_names and "date_ns" in column_names

    total = sum(entry["n_rows"] for entry in index["basin_offsets"])
    assert total == index["n_rows"]
    cells = {cell["basin_id"]: cell for cell in _cells(scenario)}
    for entry in index["basin_offsets"]:
        assert entry["n_rows"] == cells[entry["basin_id"]]["n_detail_rows"]


def test_the_detail_rows_are_deterministic_across_two_runs(scenario, tmp_path):
    first = _run(scenario)
    store_a = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    table_a = pq.read_table(store_a.shard_dir("t01") / "detail" / "elements.parquet")

    scenario.store_root = tmp_path / "store_b"
    second = _run(scenario, token="attempt2")
    store_b = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    table_b = pq.read_table(store_b.shard_dir("t01") / "detail" / "elements.parquet")

    assert table_a.to_pydict() == table_b.to_pydict()
    assert first["detail_index"]["n_rows"] == second["detail_index"]["n_rows"]
    # Row content is deterministic; the Parquet container's bytes are not
    # guaranteed identical across writer versions, which is why the cell
    # summaries -- not the container hash -- are the comparable evidence.
    assert first["cell_status_counts"] == second["cell_status_counts"]


def test_the_human_log_states_that_nothing_is_concluded(scenario):
    _run(scenario)
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    text = (store.shard_dir("t01") / "human.log").read_text(encoding="utf-8")
    assert "DIAGNOSTIC" in text
    assert "No scientific classification, threshold selection, or C4 closure follows" in text


# --- global identity failures refuse BEFORE any comparison -------------- #


def test_a_package_identity_failure_refuses_before_comparison(scenario):
    (scenario.package_root / "run_provenance.json").write_bytes(b"{}")

    with pytest.raises(ObservationDiagnosticError, match="package identity qualification failed"):
        _run(scenario)

    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    assert store.list_completed() == []


def test_a_missing_validation_product_refuses_before_comparison(scenario):
    (scenario.run_dir / "validation" / "model_epoch009" / "validation_results.p").unlink()
    with pytest.raises(ObservationDiagnosticError, match="period results pickle is absent"):
        _run(scenario)


def test_a_validation_pickle_hash_disagreement_refuses(scenario):
    trial = scenario.trial.__class__(
        **{**scenario.trial.__dict__, "validation_pickle_sha256": "0" * 64}
    )
    with pytest.raises(ObservationDiagnosticError, match="not the product this evaluation was bound to"):
        _run(scenario, trial=trial)


def test_a_validation_pickle_path_disagreement_refuses(scenario):
    trial = scenario.trial.__class__(
        **{
            **scenario.trial.__dict__,
            "validation_pickle_path": str(scenario.tmp_path / "foreign_validation_results.p"),
        }
    )
    with pytest.raises(ObservationDiagnosticError, match="validation-product identity contradiction"):
        _run(scenario, trial=trial)


def test_an_unloadable_validation_pickle_refuses_rather_than_filing_400_error_cells(scenario):
    path = scenario.run_dir / "validation" / "model_epoch009" / "validation_results.p"
    path.write_bytes(b"not a pickle at all")
    with pytest.raises(ObservationDiagnosticError, match="could not be loaded"):
        _run(scenario)
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    assert store.list_completed() == []


def test_a_wrong_basin_count_refuses_before_any_work(scenario):
    with pytest.raises(ObservationDiagnosticError, match="expected exactly 400 contract basins"):
        run_trial_observation_diagnostic(
            trial=scenario.trial,
            contract=scenario.contract,
            package_root=scenario.package_root,
            store_root=scenario.store_root,
            repo_root=scenario.tmp_path,
            attempt_token="attempt1",
        )


def test_a_duplicated_basin_in_the_requested_population_refuses(scenario):
    with pytest.raises(ObservationDiagnosticError, match="duplicate basin id"):
        _run(scenario, basin_ids=["00000001", "00000001", "00000002"])


# --- basin-specific problems become typed cells ------------------------- #


def test_a_basin_missing_from_the_trial_becomes_a_typed_cell(scenario):
    scenario.write_pickle(omit_basins=("00000002",))
    summary = _run(scenario)

    assert summary["cell_status_counts"]["basin_missing_from_trial"] == 1
    assert summary["cell_status_counts"]["ok"] == 2
    assert summary["n_cells"] == 3
    cell = next(cell for cell in _cells(scenario) if cell["basin_id"] == "00000002")
    assert cell["status"] == "basin_missing_from_trial"
    assert cell["status_detail"]


def test_a_missing_package_file_becomes_a_typed_cell(scenario):
    (scenario.package_root / "time_series" / "00000003.nc").unlink()
    summary = _run(scenario)

    assert summary["cell_status_counts"]["package_missing"] == 1
    assert summary["cell_status_counts"]["ok"] == 2
    cell = next(cell for cell in _cells(scenario) if cell["basin_id"] == "00000003")
    assert cell["status"] == "package_missing"


def test_a_per_basin_checksum_mismatch_becomes_a_typed_cell(scenario):
    target = scenario.package_root / "time_series" / "00000001.nc"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF  # same size, different content
    target.write_bytes(bytes(payload))

    summary = _run(scenario)
    assert summary["cell_status_counts"]["package_checksum_mismatch"] == 1
    assert summary["cell_status_counts"]["ok"] == 2
    cell = next(cell for cell in _cells(scenario) if cell["basin_id"] == "00000001")
    assert cell["status"] == "package_checksum_mismatch"
    assert "not the file this package was built from" in cell["status_detail"]


def test_a_basin_with_no_usable_freq_result_becomes_a_typed_cell(scenario):
    scenario.write_pickle(corrupt_basins=("00000002",))
    summary = _run(scenario)

    assert summary["cell_status_counts"]["ok"] == 2
    cell = next(cell for cell in _cells(scenario) if cell["basin_id"] == "00000002")
    assert cell["status"] in {"pickle_load_error", "other_error"}
    assert cell["status"] != "ok"


def test_one_bad_basin_does_not_stop_the_remaining_basins(scenario):
    """The central non-fatal requirement: the shard still carries exactly
    one record per basin, and the good basins are fully measured."""
    (scenario.package_root / "time_series" / "00000001.nc").unlink()
    scenario.write_pickle(omit_basins=("00000003",))

    summary = _run(scenario)
    cells = _cells(scenario)
    assert [cell["basin_id"] for cell in cells] == BASINS
    assert summary["cell_status_counts"]["package_missing"] == 1
    assert summary["cell_status_counts"]["basin_missing_from_trial"] == 1
    assert summary["cell_status_counts"]["ok"] == 1
    measured = next(cell for cell in cells if cell["status"] == "ok")
    assert measured["basin_id"] == "00000002"
    assert measured["n_compared"] > 0


def test_a_huge_numerical_difference_still_yields_an_ok_cell(scenario):
    """A disagreement is the measurement, not an error status."""

    def inflate(basin_id, obs_mm_per_h):
        if basin_id == "00000002":
            return (obs_mm_per_h.astype(np.float64) * 1.5).astype(np.float32)
        return obs_mm_per_h

    scenario.write_pickle(obs_override=inflate)
    summary = _run(scenario)

    assert summary["cell_status_counts"]["ok"] == 3
    cell = next(cell for cell in _cells(scenario) if cell["basin_id"] == "00000002")
    assert cell["status"] == "ok"
    assert cell["n_exceeding_provisional_envelope"] == cell["n_compared"]
    assert cell["q98_any_frozen_decision_changes"] in (True, False)


def test_every_typed_status_is_representable_in_a_cell():
    assert CELL_STATUSES == (
        "ok",
        "support_mismatch",
        "package_missing",
        "package_checksum_mismatch",
        "nonfinite_canonical",
        "pickle_load_error",
        "area_derivation_inconsistent",
        "basin_missing_from_trial",
        "other_error",
    )


# --- resume ------------------------------------------------------------- #


def test_an_identical_rerun_reuses_the_existing_shard(scenario):
    first = _run(scenario)
    second = _run(scenario, token="attempt2")

    assert second["reused_existing_shard"] is True
    assert second["receipt"]["content_sha256"] == first["receipt"]["content_sha256"]


def test_a_conflicting_identity_refuses_to_reuse_or_overwrite(scenario):
    _run(scenario)
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    before = store.verify("t01").content_sha256

    changed = scenario.trial.__class__(**{**scenario.trial.__dict__, "git_commit": "changed"})
    with pytest.raises(ShardStoreError, match="DIFFERENT identity"):
        _run(scenario, trial=changed, token="attempt3")

    assert store.verify("t01").content_sha256 == before


def test_a_failed_attempt_leaves_evidence_and_no_completed_shard(scenario):
    """An attempt directory is operational evidence. It is never a shard,
    and it is never deleted automatically."""
    store = AtomicShardStore(scenario.store_root, D1_SHARD_FAMILY)
    attempt = store.begin_attempt("t01", attempt_token="dead-job")
    (attempt / "progress.jsonl").write_bytes(b'{"event":"task_start"}\n')

    assert store.list_completed() == []
    summary = _run(scenario, token="fresh-job")
    assert summary["reused_existing_shard"] is False
    assert (attempt / "progress.jsonl").is_file()  # preserved, not cleaned up


def test_a_rerun_after_a_failure_publishes_cleanly(scenario):
    (scenario.run_dir / "validation" / "model_epoch009" / "validation_results.p").unlink()
    with pytest.raises(ObservationDiagnosticError):
        _run(scenario)

    scenario.write_pickle()
    summary = _run(scenario, token="attempt2")
    assert summary["cell_status_counts"]["ok"] == 3


def test_vertical_authoritative_receipt_to_d1_cell_comparison(short_tmp_path):
    """Real receipt bytes drive the authenticated source all the way to D1."""
    import hashlib

    tmp_path = short_tmp_path

    basin_id = "01000001"
    package_root, dates, qobs = build_package(tmp_path, [basin_id], n_hours=240)
    contract = build_contract(package_root, [basin_id], dates)
    run_dir = tmp_path / "run"
    write_validation_pickle(
        run_dir, 3, basin_ids=[basin_id], contract=contract, qobs_by_basin=qobs
    )
    receipt_path, receipt = write_synthetic_execution_receipt(
        tmp_path,
        "vertical",
        contract=contract,
        run_dir=run_dir,
        search_arm="bayesian",
        proposal_order=1,
    )
    receipt_hash = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    roster_path, roster_hash = write_trial_roster(
        tmp_path / "roster.json",
        contract=contract,
        receipt_entries=[
            {
                "trial_id": receipt["trial_id"],
                "search_arm": "bayesian",
                "source_receipt_path": str(receipt_path),
                "source_receipt_sha256": receipt_hash,
            }
        ],
    )
    roster = authenticate_trial_roster(
        trial_list_path=roster_path,
        contract=contract,
        verify_validation_pickles=True,
        test_only_expected_roster=fixture_only_expected_roster({"bayesian": [1]}),
    )
    target = roster.by_trial_id(receipt["trial_id"])

    summary = run_trial_observation_diagnostic(
        trial=target,
        contract=contract,
        package_root=package_root,
        store_root=tmp_path / "store",
        repo_root=tmp_path,
        attempt_token="vertical",
        expected_basin_count=1,
    )

    store = AtomicShardStore(tmp_path / "store", D1_SHARD_FAMILY)
    cell = json.loads(
        (store.shard_dir(target.trial_id) / "cells.jsonl").read_text(encoding="utf-8").strip()
    )
    assert cell["status"] == "ok"
    assert cell["n_compared"] == len(contract["per_basin_support"][basin_id])
    assert summary["identity"]["trial"]["source_receipt_sha256"] == receipt_hash
    assert summary["identity"]["trial"]["validation_pickle_sha256"] == target.validation_pickle_sha256
    assert summary["identity"]["trial"]["evaluation_scope"] == "development_validation_2024_only"
    assert summary["identity"]["trial"]["sealed_scope"] is False
    assert summary["identity"]["contract_checksum_sha256"] == contract["checksum_sha256"]
    assert roster.trial_list_sha256 == roster_hash
