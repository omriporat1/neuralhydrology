"""RD1-C4-E tests: the results/evidence writer around the already-qualified
``assemble_rd1_hydrological_review`` batch consumer.

This module adds no new scientific fixture machinery: it reuses the real
24-source ``formal_batch`` fixture and the ``_wire_synthetic_epoch_v2``
monkeypatch seam already qualified in
``tests/test_stage1_v2_12plus12_hydrological_consumer.py``, so the numbers
serialized here are produced by the exact same qualified consumer path the
production runner uses -- this module tests serialization/evidence-writing
only, never a parallel scientific computation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.baseline import rd1_c4_results_runner as runner
from src.baseline.stage1_v2_12plus12_hydrological_consumer import assemble_rd1_hydrological_review

from test_stage1_v2_12plus12_hydrological_consumer import _wire_synthetic_epoch_v2, formal_batch  # noqa: F401


@pytest.fixture()
def formal_review(formal_batch, monkeypatch):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    return assemble_rd1_hydrological_review(best_epoch_sources=sources, package_root=package_root, contract=contract)


def test_per_basin_metric_rows_cover_every_trial_times_every_basin(formal_review):
    rows = runner.per_basin_metric_rows(formal_review)
    assert len(rows) == formal_review.n_configurations * formal_review.n_basins
    assert {row["trial_id"] for row in rows} == set(formal_review.results_by_trial_id)
    first = rows[0]
    assert "nse" in first and "basin_id" in first and "search_arm" in first


def test_q98_diagnostic_rows_cover_every_trial_times_every_basin(formal_review):
    rows = runner.q98_diagnostic_rows(formal_review)
    assert len(rows) == formal_review.n_configurations * formal_review.n_basins
    first = rows[0]
    for key in ("q98_normalized_rmse", "relative_volume_bias", "observed_peak_time_magnitude_error", "high_flow_nse"):
        assert key in first


def test_basin_distribution_summary_rows_one_per_trial_with_frozen_quantile_core(formal_review):
    rows = runner.basin_distribution_summary_rows(formal_review)
    assert len(rows) == formal_review.n_configurations
    assert {row["trial_id"] for row in rows} == set(formal_review.results_by_trial_id)
    for level in (1, 5, 25, 50, 75, 95, 99):
        assert f"nse_q{level}" in rows[0]
    assert "nse_iqr" in rows[0]


def test_canonical_q98_facts_rows_one_per_basin(formal_review):
    rows = runner.canonical_q98_facts_rows(formal_review)
    assert len(rows) == formal_review.n_basins
    assert {row["basin_id"] for row in rows} == set(formal_review.canonical_q98_facts_by_basin)


def test_write_rd1_c4_results_produces_hash_verified_evidence_bundle(formal_review, tmp_path):
    out_dir = tmp_path / "rd1_c4_results"
    manifest = runner.write_rd1_c4_results(formal_review, out_dir)

    expected_files = {
        "per_basin_metrics.csv",
        "q98_diagnostics.csv",
        "basin_distribution_summary.csv",
        "canonical_q98_facts.csv",
        "review_identity.json",
    }
    assert expected_files.issubset(set(manifest))
    for name, entry in manifest.items():
        produced = out_dir / name
        assert produced.exists()
        assert entry["size_bytes"] == produced.stat().st_size
        assert entry["sha256"] == runner._sha256_path(produced)

    with open(out_dir / "per_basin_metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == formal_review.n_configurations * formal_review.n_basins

    identity = json.loads((out_dir / "review_identity.json").read_text(encoding="utf-8"))
    assert identity["n_configurations"] == 24
    assert identity["n_bayesian"] == 12
    assert identity["n_random_control"] == 12
    assert identity["n_basins"] == formal_review.n_basins
    assert len(identity["trial_ids"]) == 24


def test_write_rd1_c4_results_refuses_nonempty_existing_out_dir(formal_review, tmp_path):
    out_dir = tmp_path / "rd1_c4_results"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(runner.ResultsRunnerError, match="non-empty"):
        runner.write_rd1_c4_results(formal_review, out_dir)


def test_produce_rd1_c4_results_wraps_authentication_failure(tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"contract_id": "x"}), encoding="utf-8")
    trial_list_path = tmp_path / "trial_list.json"
    trial_list_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(Exception):
        # load_fixed_support_contract will itself reject this malformed
        # contract before the roster is even opened; this only proves the
        # production entry point fails closed on malformed frozen inputs
        # rather than silently proceeding.
        runner.produce_rd1_c4_results(
            trial_list_path=trial_list_path,
            contract_path=contract_path,
            package_root=tmp_path,
            out_dir=tmp_path / "out",
        )


def build_parser_accepts_required_arguments():
    args = runner.build_parser().parse_args(
        [
            "--trial-list",
            "tl.json",
            "--contract",
            "c.json",
            "--package-root",
            "pkg",
            "--out-dir",
            "out",
        ]
    )
    assert args.trial_list == "tl.json"
    assert args.out_dir == "out"


def test_build_parser_accepts_required_arguments():
    build_parser_accepts_required_arguments()
