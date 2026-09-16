"""RD1-C4-B tests: the fixed-support hydrological batch consumer and the
frozen Q98 high-flow diagnostic package.

Test layout:
 - Sections A-C: pure Q98 math (scenarios 1-10) against
   ``derive_canonical_basin_q98_facts``/``compute_q98_configuration_basin_diagnostics``
   directly, with hand-built numpy fixtures -- no I/O.
 - Section D: official-best-epoch/objective identity validation and receipt
   content-binding (RD1-C4 review Findings 1, 2; scenarios 11-13), built on
   real ``execution_provenance.json`` receipts produced by the genuine
   prepare -> ``execute_prepared_trial_v2`` spine (via ``_real_receipt``/
   ``_execute_v2_receipt``, reusing ``tests/test_sweep_v2_six_axis_execution.py``'s
   own fixture helpers by name), plus the fast synthetic monkeypatch seam
   (the same ``load_period_results``/``basin_netcdf_path``/
   ``derive_basin_area_km2_from_netcdf``/``evaluate_basin_raw_space``/
   ``aggregate_raw_space_metrics`` seam already used by
   ``tests/test_fixed_support_contract_v2.py``'s own production-completeness
   tests) for the re-scored-objective sub-cases.
 - Section E: package-canonical Q98 derivation and cross-trial provenance-
   audit enforcement (scenarios 3, 14 basin-population half, 15, 16) against
   ``_derive_canonical_q98_facts_from_package``/``_audit_admitted_observation_provenance``
   directly, using lightweight stand-ins that expose only the
   ``admitted_series_by_basin`` attribute they read and a monkeypatched
   ``derive_canonical_package_observed_series`` seam.
 - Section F: production batch-shape gating, including the RD1-C4 review
   Finding 3 fix (legal duplicate ``configuration_id`` across trials that
   agree on canonical hyperparameters) against ``_validate_production_batch_shape``
   directly -- pure, no I/O, built from fully self-consistent
   ``V2BestEpochSource`` instances via the same qualified identity helpers
   the real campaign uses.
 - Section G: coverage accounting (scenario 18) and the RD1-C4 review
   Finding 7 fix (fail-closed on a genuinely missing required metric field,
   as distinct from an explicit NaN) against ``_compute_coverage`` directly.
 - Section H: the one designated vertical synthetic integration test,
   covering the full chain with real package NetCDF + real validation
   pickle + real prepare/execute receipt + the real fixed-support/raw-space/
   C3 machinery -- no mocking of any scientific function anywhere in this
   section (scenarios 11a, 11d, 17, and the general per-basin passthrough/
   Q98-application chain).
"""
from __future__ import annotations

import json
import inspect
import os
import pickle
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from src.baseline import fixed_support_contract_v2 as fixed
from src.baseline import pilot_orchestration as orchestration
from src.baseline import stage1_v2_12plus12_hydrological_consumer as hyd
from src.baseline.authenticated_period_results import fixture_only_period_results
from src.baseline.package_identity_qualification import QualifiedPackageIdentity
from src.baseline import sweep_v2_six_axis_production_adapter as v2_adapter
from src.baseline.nh_seed_evaluation import period_results_path
from src.baseline.sweep_v1_campaign import derive_trajectory_diagnostics
from src.baseline.sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME
from src.baseline.sweep_v2_six_axis_execution import execute_prepared_trial_v2
from src.baseline.sweep_v2_six_axis_production_adapter import write_prepared_proposal_v2
from tests import test_sweep_v2_six_axis_execution as _sweep_v2_exec_tests
from tests._pilot_support import REAL_DEVELOPMENT
from tests.test_sweep_v2_six_axis_execution import (
    _paths_v2, _prepared_record_v2, _proposal_v2, _screening_event, _write_real_checkpoints,
)


# --------------------------------------------------------------------------- #
# Shared fixtures.
# --------------------------------------------------------------------------- #


def _contract(n_basins: int, *, support_len: int = 3) -> dict:
    ids = [f"{value:08d}" for value in range(n_basins)]
    return fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="fixture", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="fixture_gap_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256="a" * 64, package_file_checksums_sha256="b" * 64,
        package_run_provenance_sha256="c" * 64, development_split_sha256="d" * 64,
        spatial_holdout_split_sha256="e" * 64,
        per_basin_date={basin_id: np.arange(support_len) for basin_id in ids},
        per_basin_admitted={basin_id: np.ones(support_len, dtype=bool) for basin_id in ids},
    )


class _Array:
    """Minimal xarray-DataArray stand-in.

    ``dims`` is part of the surface because the evaluator proves a
    variable's dimension identity before reading its values (RD1-C4-D1
    finding A3) instead of flattening whatever it finds.
    """

    def __init__(self, values, dims=("date",)):
        self.values = np.asarray(values)
        self.dims = tuple(dims)


class _Dataset:
    def __init__(self, n=3):
        self.coords = {"date": _Array(np.arange(n))}
        self.data_vars = {"qobs_mm_per_h_lead06_obs", "qobs_mm_per_h_lead06_sim"}
        self._values = {
            "qobs_mm_per_h_lead06_obs": _Array(np.ones(n)),
            "qobs_mm_per_h_lead06_sim": _Array(np.ones(n)),
        }

    def __getitem__(self, key):
        return self._values[key]


def _fixture_package_identity(package_root, contract) -> QualifiedPackageIdentity:
    """A ``QualifiedPackageIdentity`` built directly rather than through
    ``qualify_package_identity()``'s real proof chain (RD1-C4-D1 finding A2:
    the formal consumer now refuses any package_identity that is not an
    actual ``QualifiedPackageIdentity`` instance qualified against exactly
    this package_root/contract). The full manifest/schema/provenance proof
    chain is exhaustively exercised by
    ``tests/test_package_identity_qualification.py``; this helper only needs
    to satisfy the formal consumer's own checks -- type, resolved
    package_root, contract_id, contract checksum (``_require_package_identity_for_root``)
    -- for tests in this module that mock or do not otherwise depend on real
    package I/O. This mirrors the same direct-construction pattern already
    used for ``AuthenticatedTrialTarget`` fixtures in
    ``tests/_rd1_c4_d1_support.py``.

    Where the real per-basin byte proof (RD1-C4-D1 finding A5,
    ``verify_basin_time_series_file``) is still unconditionally exercised
    downstream -- inside ``evaluate_fixed_support_raw_space_metrics``'s
    per-basin loop, regardless of how the higher-level metric/area functions
    are mocked -- this writes one tiny real placeholder file per
    ``contract["basin_ids"]`` basin under ``package_root/time_series/`` and
    records its real sha256/size, so that proof genuinely passes rather than
    being bypassed. Skipped (no filesystem writes) when the contract carries
    no ``basin_ids`` (Section E's shallow-only contracts), so this helper
    stays side-effect-free for callers that never reach the per-basin check.
    """
    from pathlib import Path as _Path
    import hashlib as _hashlib

    root = _Path(package_root)
    basin_ids = list(contract.get("basin_ids", []))
    relative_path: dict = {}
    sha256_by_basin: dict = {}
    size_by_basin: dict = {}
    if basin_ids:
        time_series_dir = root / "time_series"
        time_series_dir.mkdir(parents=True, exist_ok=True)
        for basin_id in basin_ids:
            rel = f"time_series/{basin_id}.nc"
            file_path = root / rel
            if not file_path.is_file():
                # No real package file at this path yet (the common case for
                # this module's fast seams) -- a tiny placeholder is enough
                # to satisfy the real per-basin byte proof honestly. Where a
                # real basin NetCDF already exists (Section H's vertical
                # test, which writes real package files before qualifying),
                # this never touches it -- it only records that file's own
                # real bytes below, exactly as the real qualifier would.
                file_path.write_bytes(f"fixture-basin-{basin_id}".encode("utf-8"))
            payload_bytes = file_path.read_bytes()
            relative_path[basin_id] = rel
            sha256_by_basin[basin_id] = _hashlib.sha256(payload_bytes).hexdigest()
            size_by_basin[basin_id] = len(payload_bytes)

    return QualifiedPackageIdentity(
        package_root=root.resolve().as_posix(),
        contract_id=contract["contract_id"],
        contract_checksum_sha256=contract["checksum_sha256"],
        contract_schema_name="fixture_schema",
        contract_schema_version=1,
        package_manifest_sha256="0" * 64,
        package_file_checksums_sha256="0" * 64,
        package_run_provenance_sha256="0" * 64,
        manifest_schema_name="fixture_manifest_schema",
        manifest_schema_version=1,
        package_role="fixture_role",
        netcdf_package_schema_name="fixture_netcdf_schema",
        netcdf_package_schema_version=1,
        netcdf_time_coordinate="date",
        netcdf_schema_historical_lineage_applied=False,
        run_provenance_builder_module="fixture_module",
        run_provenance_created_at_utc="2026-01-01T00:00:00Z",
        run_provenance_dry_run=False,
        raw_target_variable="qobs_m3s",
        target_variable="qobs_mm_per_h_lead06",
        lead_hours=6,
        period="fixture",
        contract_date_start="2024-01-01",
        contract_date_end="2024-01-01",
        timeline_start="2024-01-01T00:00:00",
        timeline_end="2024-01-01T00:00:00",
        timeline_rows=1,
        timeline_frequency="1h",
        timeline_window_verified=True,
        n_package_basins=len(basin_ids) or 1,
        n_qualified_basins=len(basin_ids) or 1,
        n_checksum_entries=len(basin_ids) or 1,
        basin_time_series_relative_path=relative_path,
        basin_time_series_sha256=sha256_by_basin,
        basin_time_series_size_bytes=size_by_basin,
        qualified_at_utc="2026-01-01T00:00:00Z",
    )


def _wire_synthetic_epoch_v2(monkeypatch, contract, *, per_basin_overrides=None, aggregate_median=0.5, n=10):
    """Fast monkeypatch seam (extends ``tests/test_fixed_support_contract_v2.py``'s
    own ``_wire_synthetic_epoch`` pattern) that additionally carries kge/
    component-passthrough fields and controllable admitted obs/sim arrays,
    so RD1-C4-B's own identity/checksum/batch-shape/coverage logic can be
    exercised through the real ``evaluate_fixed_support_raw_space_metrics``
    control flow without real NetCDF/pickle I/O. Never used for the
    designated vertical integration test (Section H), which uses real I/O
    and mocks nothing.

    Also monkeypatches ``hyd.derive_canonical_package_observed_series`` (the
    package-canonical Q98 source, RD1-C4 reproducibility correction) to
    return, per basin, exactly the same admitted obs values this seam
    fabricates by default (or ``per_basin_overrides[basin_id]["obs_m3s"]``
    when supplied) -- so the fast seam's fixtures remain self-consistent
    with the package-canonical provenance audit without requiring real
    NetCDF I/O for every batch-shape/coverage-focused test.
    """
    dataset = _Dataset(n=n)
    period_results = {basin_id: {"1h": {"xr": dataset}} for basin_id in contract["basin_ids"]}

    def _load(*, run_dir, period, epoch, expected_sha256=None):
        return fixture_only_period_results(run_dir=run_dir, period=period, epoch=epoch, results=period_results)

    monkeypatch.setattr(fixed, "load_authenticated_period_results", _load)
    monkeypatch.setattr(fixed, "basin_netcdf_path", lambda *_: "fixture.nc")
    monkeypatch.setattr(
        fixed, "derive_basin_area_km2_from_netcdf",
        lambda *_, **__: SimpleNamespace(area_km2=100.0, consistent=True, relative_mad=0.0),
    )
    overrides = per_basin_overrides or {}

    def metric(*, basin_id, return_admitted_arrays=False, **_):
        row_overrides = overrides.get(basin_id, {})
        row = {
            "basin_id": basin_id,
            "nse": row_overrides.get("nse", 0.5), "kge": row_overrides.get("kge", 0.4),
            "kge_r": row_overrides.get("kge_r", 0.9), "kge_alpha": row_overrides.get("kge_alpha", 1.0),
            "kge_beta": row_overrides.get("kge_beta", 1.0),
            "n_sim_nonfinite_at_admitted": row_overrides.get("n_sim_nonfinite_at_admitted", 0),
            "n_admitted": row_overrides.get("n_admitted", n),
        }
        if return_admitted_arrays:
            row["_admitted_obs_m3s"] = row_overrides.get("obs_m3s", np.arange(1.0, n + 1.0))
            row["_admitted_sim_m3s"] = row_overrides.get("sim_m3s", np.arange(1.0, n + 1.0))
        return row

    def aggregate(rows):
        finite = sum(np.isfinite(row["nse"]) for row in rows)
        return {"n_basins": len(rows), "metrics": {"nse": {"n_finite_basins": finite, "median": aggregate_median}}}

    monkeypatch.setattr(fixed, "evaluate_basin_raw_space", metric)
    monkeypatch.setattr(fixed, "aggregate_raw_space_metrics", aggregate)

    support_dates_by_basin = {
        basin_id: fixed._deserialize_date_array(contract["per_basin_support"][basin_id], contract["date_dtype"])
        for basin_id in contract["basin_ids"]
    }

    def fake_canonical_package_series(*, package_root, basin_id, contract, package_identity=None):
        row_overrides = overrides.get(basin_id, {})
        support_dates = support_dates_by_basin[basin_id]
        # This seam's own ``metric()`` fabricates a default admitted
        # obs_m3s of ``run_date_values[i] + 1.0`` at each run position i
        # (``np.arange(1.0, n + 1.0)`` against ``run_date_values ==
        # np.arange(n)``), and the real ``evaluate_fixed_support_raw_space_metrics``
        # reindexes strictly by date identity -- so for any admitted
        # support timestamp ``d`` the resulting default admitted obs_m3s
        # value is always exactly ``d + 1.0``, independent of contract
        # support length/ordering. Mirror that identity here so the fast
        # seam's fixtures stay self-consistent with the package-canonical
        # provenance audit by default.
        default_obs = np.asarray(support_dates, dtype=np.float64) + 1.0
        obs_m3s = np.asarray(row_overrides.get("obs_m3s", default_obs))
        return fixed.CanonicalPackageObservedSeries(basin_id=basin_id, date=support_dates, obs_m3s=obs_m3s)

    monkeypatch.setattr(hyd, "derive_canonical_package_observed_series", fake_canonical_package_series)

    # ``assemble_rd1_hydrological_review`` calls the real, full manifest/
    # schema/provenance proof chain unconditionally and with no parameter to
    # substitute it (RD1-C4-D1 finding A2: no production path can skip
    # qualification) -- building that full proof for a synthetic fixture
    # package would mean fabricating a complete matching manifest set for no
    # additional test value, since the proof chain itself is already
    # exhaustively exercised in tests/test_package_identity_qualification.py.
    # This fast seam substitutes only the call as bound in ``hyd``'s module
    # globals, exactly like every other package-touching function this seam
    # already substitutes above.
    monkeypatch.setattr(
        hyd, "qualify_package_identity",
        lambda *, package_root, contract, basin_ids=None: _fixture_package_identity(package_root, contract),
    )
    return period_results


def _write_receipt(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _execute_v2_receipt(record, nh_run_dir, output_dir, *, fixed_scores_by_epoch, natural_scores_by_epoch=None):
    """Drive the real, unmodified ``execute_prepared_trial_v2`` producer to
    obtain a genuine ``execution_provenance.json`` receipt -- mirrors
    ``tests/test_sweep_v2_six_axis_execution.py``'s own ``_fake_result``/
    ``test_vertical_prepared_execution_consumer_contract_v2`` fixture
    pattern exactly, except ``fixed_scores_by_epoch`` is caller-controlled
    (rather than a fixed formula) so tests can pin the resulting official
    best_epoch/objective to whatever value they need. Never starts real NH/
    W&B; ``nh_run_dir`` gets genuine on-disk checkpoint/optimizer-state
    files so ``execute_prepared_trial_v2``'s own validity derivation reads
    real evidence, exactly like the donor fixture."""
    torch = pytest.importorskip("torch")
    if set(fixed_scores_by_epoch) != set(range(1, 13)):
        raise AssertionError("fixed_scores_by_epoch must cover exactly the frozen v2 epochs 1..12")
    epochs = list(range(1, 13))
    _write_real_checkpoints(nh_run_dir, epochs, torch)
    natural_scores = natural_scores_by_epoch if natural_scores_by_epoch is not None else {epoch: 0.2 for epoch in epochs}
    checkpoint_inventory = {
        epoch: orchestration.PhysicalCheckpoint(
            epoch=epoch, path=nh_run_dir / f"model_epoch{epoch:03d}.pt", owning_run_dir=nh_run_dir
        )
        for epoch in epochs
    }
    screening_events = [_screening_event(epoch, natural_scores[epoch], n_basins=400) for epoch in epochs]

    def metric(scope, value):
        return {"objective_scope": scope, "aggregate": {"metrics": {"nse": {"median": value}}}}

    supplemental = {
        epoch: {"fixed_support": metric("fixed_support", fixed_scores_by_epoch[epoch]),
                "natural_support": metric("natural_support", natural_scores[epoch])}
        for epoch in epochs
    }
    fake_result = orchestration.PreparedPilotExecutionResult(
        final_status="completed_at_full_budget", blocked_reason=None,
        effective_policy={"max_epoch_budget": 12, "performance_early_stopping_enabled": False},
        nh_run_dir=nh_run_dir, blocked=False, stopped=False, stop_reason=None,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={}, screening_events=screening_events,
        supplemental_epoch_results=supplemental,
    )
    outcome = execute_prepared_trial_v2(
        prepared_record=record, output_dir=output_dir,
        expected_screening_population=400, execute_prepared_run_fn=lambda: fake_result,
    )
    assert outcome["valid"] is True, outcome["review_records"]
    receipt_path = output_dir / "execution_provenance.json"
    receipt_record = json.loads(receipt_path.read_text(encoding="utf-8"))
    return receipt_path, receipt_record


def _real_receipt(tmp_path, monkeypatch, *, fixed_scores_by_epoch, label="a", natural_scores_by_epoch=None, **proposal_changes):
    """Build one genuine v2 ``execution_provenance.json`` receipt via the
    real prepare -> ``execute_prepared_trial_v2`` spine (no
    execution-provenance field is hand-authored), so Section D/H tests bind
    against the exact receipt shape the real campaign producer writes."""
    record, paths = _prepared_record_v2(tmp_path / f"prep_{label}", monkeypatch, **proposal_changes)
    nh_run_dir = tmp_path / f"nh_run_{label}"
    output_dir = tmp_path / f"trial_out_{label}"
    receipt_path, receipt_record = _execute_v2_receipt(
        record, nh_run_dir, output_dir,
        fixed_scores_by_epoch=fixed_scores_by_epoch, natural_scores_by_epoch=natural_scores_by_epoch,
    )
    contract = fixed.load_fixed_support_contract(paths.fixed_support_contract_path)
    return receipt_path, receipt_record, contract


def _linear_trajectory(peak_epoch: int, peak_value: float, *, step: float = 0.01) -> dict:
    return {epoch: peak_value - abs(epoch - peak_epoch) * step for epoch in range(1, 13)}


# --------------------------------------------------------------------------- #
# Section A-C: pure Q98 math (scenarios 1-10).
# --------------------------------------------------------------------------- #


def test_q98_threshold_matches_independent_linear_interpolation():
    # scenario 1
    obs = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    expected = float(np.quantile(obs, 0.98, method="linear"))
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=np.arange(10), obs_m3s=obs)
    assert facts.q98_threshold == pytest.approx(expected)
    assert expected == pytest.approx(98.2)  # independently calculable for this fixture


def test_q98_mask_includes_observations_exactly_equal_to_threshold():
    # scenario 2
    obs = np.array([1.0, 2.0, 3.0, 100.0])
    threshold = float(np.quantile(obs, 0.98, method="linear"))
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=np.arange(4), obs_m3s=obs)
    assert facts.q98_threshold == pytest.approx(threshold)
    # The single observation exactly at (or above) the threshold must be
    # included by ">=", not excluded by a strict "greater than".
    at_or_above = obs >= threshold
    np.testing.assert_array_equal(facts.high_flow_mask, at_or_above)
    assert facts.n_high_flow == int(np.sum(at_or_above))
    assert facts.n_high_flow >= 1


def test_q98_facts_reused_identically_across_two_configurations():
    # scenario 3: candidate-independent reuse of one observed-derived mask
    date = np.arange(10)
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)

    sim_cfg_a = obs * 1.1
    sim_cfg_b = obs * 0.9
    diag_a = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trialA", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim_cfg_a
    )
    diag_b = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trialB", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim_cfg_b
    )
    assert diag_a.n_high_flow == diag_b.n_high_flow == facts.n_high_flow
    # Same canonical threshold/mask drives both -- diagnostics differ only
    # because the simulated series differ, not because the mask differs.
    assert diag_a.relative_volume_bias > 0 > diag_b.relative_volume_bias


def test_q98_normalized_rmse_denominator_is_threshold_not_high_flow_mean():
    # scenario 4 (RMSE is well-defined for a single finite sample too --
    # see test_q98_normalized_rmse_finite_for_single_high_flow_observation
    # below -- so the two-point high-flow subset here is only to make the
    # high-flow mean (175) differ from the Q98 threshold (101), not a
    # sample-count requirement of raw_space_metrics)
    date = np.arange(100)
    # 97 low values plus three distinct high values (100, 150, 200); the 98th
    # percentile (linear interpolation) lands at 101, so the high-flow mask
    # keeps {150, 200} (mean 175) while excluding 100 -- deliberately making
    # the high-flow mean (175) differ from the Q98 threshold (101).
    obs = np.concatenate([np.zeros(97), np.array([100.0, 150.0, 200.0])])
    threshold = float(np.quantile(obs, 0.98, method="linear"))
    sim = obs + 1.0
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    assert facts.n_high_flow >= 2
    diag = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
    )
    obs_hf = obs[facts.high_flow_mask]
    sim_hf = sim[facts.high_flow_mask]
    expected_rmse = float(np.sqrt(np.mean((sim_hf - obs_hf) ** 2)))
    high_flow_mean = float(np.mean(obs_hf))
    assert diag.q98_normalized_rmse == pytest.approx(expected_rmse / threshold)
    assert diag.q98_normalized_rmse != pytest.approx(expected_rmse / high_flow_mean)


def test_q98_normalized_rmse_finite_for_single_high_flow_observation():
    # Regression Gap 2: raw_space_metrics's RMSE is well-defined for exactly
    # one finite sample -- Q98's normalized RMSE must not require >=2
    # high-flow points. high_flow_nse stays NaN/unavailable because
    # MIN_HIGH_FLOW_NSE_SAMPLES (50) is not met, distinct from RMSE.
    date = np.arange(2)
    obs = np.array([0.0, 200.0])  # 98th percentile (linear) = 196 -> exactly one qualifying point
    sim = obs.copy()
    sim[-1] = 205.0
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    assert facts.n_high_flow == 1
    diag = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
    )
    expected_rmse = abs(205.0 - 200.0)  # sqrt(mean((sim-obs)^2)) over a single pair
    assert np.isfinite(diag.q98_normalized_rmse)
    assert diag.q98_normalized_rmse == pytest.approx(expected_rmse / facts.q98_threshold)
    assert not diag.high_flow_nse_available
    assert np.isnan(diag.high_flow_nse)


def test_relative_volume_bias_sign_and_denominator():
    # scenario 5
    date = np.arange(4)
    obs = np.array([1.0, 1.0, 1.0, 10.0])
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    sim_over = obs * 1.5
    diag_over = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim_over
    )
    obs_hf = obs[facts.high_flow_mask]
    sim_hf = sim_over[facts.high_flow_mask]
    expected = float(np.sum(sim_hf - obs_hf) / np.sum(obs_hf))
    assert diag_over.relative_volume_bias == pytest.approx(expected)
    assert diag_over.relative_volume_bias > 0

    sim_under = obs * 0.5
    diag_under = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim_under
    )
    assert diag_under.relative_volume_bias < 0


def test_observed_peak_time_magnitude_error_uses_simulation_at_observed_peak_not_simulated_argmax():
    # scenario 6
    date = np.arange(5)
    obs = np.array([1.0, 2.0, 3.0, 100.0, 4.0])
    sim = np.array([1.0, 2.0, 3.0, 50.0, 500.0])  # simulated argmax is NOT at the observed peak
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    diag = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
    )
    expected = (50.0 - 100.0) / 100.0
    assert diag.observed_peak_time_magnitude_error == pytest.approx(expected)


def test_peak_ties_choose_earliest_timestamp_even_when_input_is_unordered():
    # scenario 7
    # Two timestamps share the observed maximum; the array is deliberately
    # NOT chronologically ordered to prove the selection depends on the
    # timestamp values themselves, not on row position.
    date = np.array([30, 10, 20])
    obs = np.array([100.0, 100.0, 1.0])
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    # earliest among {date=30 (row 0), date=10 (row 1)} is date=10 -> row 1
    assert facts.observed_peak_index == 1

    sim = np.array([500.0, 250.0, 1.0])
    diag = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
    )
    assert diag.observed_peak_time_magnitude_error == pytest.approx((250.0 - 100.0) / 100.0)


def test_high_flow_nse_boundary_at_exactly_50_samples_and_49_and_zero_variance():
    # scenario 8. This exercises compute_q98_configuration_basin_diagnostics's
    # NSE-availability gate directly, so the high-flow mask is built by hand
    # (bypassing quantile derivation, which is already covered by scenarios
    # 1-2) to pin the exact boundary sample counts the gate must react to.
    rng = np.random.default_rng(0)

    def _facts_and_diag(n_high_flow, obs_high_flow):
        n_low = 5
        obs = np.concatenate([np.full(n_low, 0.1), obs_high_flow])
        date = np.arange(obs.size)
        mask = np.concatenate([np.zeros(n_low, dtype=bool), np.ones(len(obs_high_flow), dtype=bool)])
        facts = hyd.CanonicalBasinQ98Facts(
            basin_id="b1", n_admitted=obs.size, q98_threshold=float(np.min(obs_high_flow)),
            high_flow_mask=mask, n_high_flow=n_high_flow, observed_peak_value=float(np.max(obs)),
            observed_peak_index=int(np.argmax(obs)), canonical_date=date, canonical_obs_m3s=obs,
        )
        assert facts.n_high_flow == n_high_flow
        sim = obs * 1.05
        diag = hyd.compute_q98_configuration_basin_diagnostics(
            trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
        )
        return diag

    obs_hf_50 = rng.uniform(50.0, 150.0, size=50)
    diag_50 = _facts_and_diag(50, obs_hf_50)
    assert diag_50.high_flow_nse_available == True
    assert np.isfinite(diag_50.high_flow_nse)

    obs_hf_49 = rng.uniform(50.0, 150.0, size=49)
    diag_49 = _facts_and_diag(49, obs_hf_49)
    assert diag_49.high_flow_nse_available == False
    assert np.isnan(diag_49.high_flow_nse)

    # Zero observed high-flow variance (all identical high-flow values) with
    # >=50 samples must still be NaN/unavailable.
    obs_hf_novar = np.full(50, 75.0)
    diag_novar = _facts_and_diag(50, obs_hf_novar)
    assert diag_novar.high_flow_nse_available == False
    assert np.isnan(diag_novar.high_flow_nse)


def test_zero_or_nonfinite_denominator_yields_nan_for_metrics_a_b_c():
    # scenario 9
    date = np.arange(3)
    obs = np.zeros(3)  # threshold == 0, sum(obs_hf) == 0, observed peak == 0
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    assert facts.q98_threshold == 0.0
    sim = np.array([1.0, 2.0, 3.0])
    diag = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
    )
    assert np.isnan(diag.q98_normalized_rmse)
    assert np.isnan(diag.relative_volume_bias)
    assert np.isnan(diag.observed_peak_time_magnitude_error)


# --------------------------------------------------------------------------- #
# RD1-C4 review follow-on: defensive immutability of public scientific
# results (CanonicalBasinQ98Facts / V2HydrologicalConfigurationResult).
# --------------------------------------------------------------------------- #


def test_canonical_basin_q98_facts_defensively_copies_and_freezes_its_arrays():
    date = np.arange(5)
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([False, False, False, True, True])
    facts = hyd.CanonicalBasinQ98Facts(
        basin_id="b1", n_admitted=5, q98_threshold=4.0, high_flow_mask=mask,
        n_high_flow=2, observed_peak_value=5.0, observed_peak_index=4,
        canonical_date=date, canonical_obs_m3s=obs,
    )

    assert not np.shares_memory(facts.high_flow_mask, mask)
    assert not np.shares_memory(facts.canonical_date, date)
    assert not np.shares_memory(facts.canonical_obs_m3s, obs)
    assert mask.flags.writeable and date.flags.writeable and obs.flags.writeable

    mask[0] = True  # mutating the caller's original has no effect on the stored copy
    assert facts.high_flow_mask[0] == False

    with pytest.raises(ValueError):
        facts.high_flow_mask[0] = True
    with pytest.raises(ValueError):
        facts.canonical_date[0] = 999
    with pytest.raises(ValueError):
        facts.canonical_obs_m3s[0] = 999.0


def test_v2_hydrological_configuration_result_fixed_support_result_is_immutable():
    original_fixed_support_result = {
        "objective_scope": "fixed_support",
        "per_basin": [{"basin_id": "b1", "nse": 0.42}],
        "aggregate": {"metrics": {"nse": {"median": 0.42}}},
    }
    result = hyd.V2HydrologicalConfigurationResult(
        best_epoch_source=None,
        official_objective=0.42,
        rescored_objective=0.42,
        fixed_support_result=original_fixed_support_result,
        admitted_series_by_basin={},
        basin_distribution=None,
        q98_diagnostics_by_basin={},
        contract_id="c1",
        contract_checksum_sha256="0" * 64,
        support_contract_provenance=None,
    )

    # independent of the original caller mapping
    original_fixed_support_result["per_basin"][0]["nse"] = -999.0
    original_fixed_support_result["objective_scope"] = "tampered"
    assert result.fixed_support_result["per_basin"][0]["nse"] == 0.42
    assert result.fixed_support_result["objective_scope"] == "fixed_support"

    # rejects mutation of the top-level mapping
    with pytest.raises(TypeError):
        result.fixed_support_result["objective_scope"] = "tampered"

    # rejects mutation of a nested per-basin metric
    with pytest.raises(TypeError):
        result.fixed_support_result["per_basin"][0]["nse"] = -1.0

    # preserves the original scientific value after attempted mutation
    assert result.fixed_support_result["per_basin"][0]["nse"] == 0.42


def test_high_flow_sample_count_is_always_reported():
    # scenario 10
    date = np.arange(6)
    obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # degenerate: no denominators available at all
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    sim = obs.copy()
    diag = hyd.compute_q98_configuration_basin_diagnostics(
        trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
    )
    assert diag.n_high_flow == facts.n_high_flow
    assert isinstance(diag.n_high_flow, int)


def test_q98_facts_reject_nonfinite_observed_admitted_values():
    date = np.arange(3)
    obs = np.array([1.0, np.nan, 3.0])
    with pytest.raises(hyd.HydrologicalConsumerError):
        hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)


def test_q98_diagnostics_reject_nonfinite_simulation_rather_than_pairwise_dropping():
    date = np.arange(3)
    obs = np.array([1.0, 2.0, 3.0])
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    sim = np.array([1.0, np.nan, 3.0])
    with pytest.raises(hyd.HydrologicalConsumerError):
        hyd.compute_q98_configuration_basin_diagnostics(
            trial_id="trial", facts=facts, date=date, obs_m3s=obs, sim_m3s=sim
        )


def test_q98_diagnostics_reject_observed_side_mismatch_against_canonical_facts():
    date = np.arange(3)
    obs = np.array([1.0, 2.0, 3.0])
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=date, obs_m3s=obs)
    wrong_obs = np.array([1.0, 2.0, 999.0])
    with pytest.raises(hyd.HydrologicalConsumerError):
        hyd.compute_q98_configuration_basin_diagnostics(
            trial_id="trial", facts=facts, date=date, obs_m3s=wrong_obs, sim_m3s=wrong_obs
        )


def test_q98_point_of_use_requires_simulation_timestamps_to_equal_canonical_package_timestamps():
    dates = np.array([0, 1, 2])
    obs = np.array([1.0, 2.0, 3.0])
    facts = hyd.derive_canonical_basin_q98_facts(basin_id="b1", date=dates, obs_m3s=obs)
    with pytest.raises(hyd.HydrologicalConsumerError, match="paired by position only"):
        hyd.compute_q98_configuration_basin_diagnostics(
            trial_id="trial", facts=facts, date=np.array([1, 0, 2]),
            obs_m3s=obs, sim_m3s=np.array([2.0, 1.0, 3.0]),
        )


# --------------------------------------------------------------------------- #
# Section D: official best-epoch/objective identity & receipt content
# binding (RD1-C4 review Findings 1, 2; scenarios 11-13).
# --------------------------------------------------------------------------- #


def test_build_best_epoch_source_accepts_a_real_receipt(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    assert source.best_epoch == 3
    assert source.official_objective == pytest.approx(0.50)
    assert source.source_receipt_sha256
    assert Path(source.run_dir) == Path(receipt_record["result"]["nh_run_dir"])


def test_build_best_epoch_source_rejects_content_mismatched_execution_provenance(short_tmp_path, monkeypatch):
    # Finding 1: caller-supplied execution_provenance must be byte-content
    # identical (as a dict) to the freshly re-parsed receipt file.
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["objective_score"] = tampered["objective_score"] + 0.001
    with pytest.raises(hyd.HydrologicalConsumerError, match="does not exactly match"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path)


def test_build_best_epoch_source_rejects_malformed_json_receipt(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    receipt_path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(hyd.HydrologicalConsumerError, match="not valid JSON"):
        hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)


def test_build_best_epoch_source_rejects_non_object_json_receipt(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    receipt_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(hyd.HydrologicalConsumerError, match="JSON object"):
        hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)


def test_build_best_epoch_source_rejects_missing_receipt_file(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    missing = receipt_path.with_name("does_not_exist.json")
    with pytest.raises(hyd.HydrologicalConsumerError, match="does not exist"):
        hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=missing)


def test_build_best_epoch_source_does_not_cross_bind_two_distinct_receipts(short_tmp_path, monkeypatch):
    receipt_path_a, receipt_record_a, _ = _real_receipt(
        short_tmp_path, monkeypatch, label="a", fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    receipt_path_b, receipt_record_b, _ = _real_receipt(
        short_tmp_path, monkeypatch, label="b", proposal_order=8,
        fixed_scores_by_epoch=_linear_trajectory(5, 0.60),
    )
    assert receipt_record_a != receipt_record_b
    with pytest.raises(hyd.HydrologicalConsumerError, match="does not exactly match"):
        hyd.build_v2_best_epoch_source(execution_provenance=receipt_record_a, source_receipt_path=receipt_path_b)


def test_build_best_epoch_source_rejects_receipt_missing_required_field(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    del tampered["campaign_id"]  # not consumed by build_v2_objective_publication_payload
    receipt_path2 = _write_receipt(short_tmp_path, "missing_campaign_id.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="missing required field"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_rejects_missing_best_epoch(short_tmp_path, monkeypatch):
    # scenario 11: invalid/missing epoch
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    del tampered["best_epoch"]
    receipt_path2 = _write_receipt(short_tmp_path, "missing_best_epoch.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_rejects_objective_not_equal_to_selected_trajectory_value(short_tmp_path, monkeypatch):
    # scenario 11: objective not equal to selected trajectory value
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["objective_score"] = tampered["objective_score"] + 0.2  # now disagrees with its own trajectory
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_objective.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_rejects_wrong_fixed_support_metric_name(short_tmp_path, monkeypatch):
    # scenario 13 (metric identity half)
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["fixed_support_metric_name"] = "some_other_metric"
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_metric_name.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_fails_closed_on_top_level_vs_preparation_record_proposal_id_disagreement(
    short_tmp_path, monkeypatch,
):
    # scenario 12
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["proposal_id"] = "not_the_recomputed_proposal_id"
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_proposal_id.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="proposal_id"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_fails_closed_on_top_level_vs_preparation_record_trial_id_disagreement(
    short_tmp_path, monkeypatch,
):
    # scenario 12
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["trial_id"] = "not_the_recomputed_trial_id"
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_trial_id.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="trial_id"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_fails_closed_on_non_recomputable_configuration_id(short_tmp_path, monkeypatch):
    # Finding 2: configuration_id must independently recompute from its own
    # canonical_hyperparameters/support-contract identity, not merely be
    # internally cross-consistent between the receipt's top level and its
    # nested preparation_record.
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    fabricated = "sweep_v2_cfg_" + "f" * 20
    tampered["configuration_id"] = fabricated
    tampered["preparation_record"]["configuration_id"] = fabricated
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_configuration_id.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="does not recompute"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_fails_closed_on_non_recomputable_trial_id(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    fabricated = receipt_record["trial_id"] + "_tampered"
    tampered["trial_id"] = fabricated
    tampered["preparation_record"]["trial_id"] = fabricated
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_trial_id_consistent.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="inconsistent with configuration_id/proposal_id"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_fails_closed_on_wrong_domain_version(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["preparation_record"]["domain_version"] = "not_the_real_domain_version"
    receipt_path2 = _write_receipt(short_tmp_path, "tampered_domain_version.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="domain_version"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_preserves_correct_receipt_model_seed(short_tmp_path, monkeypatch):
    # Source Defect 1: model_seed must be read from the receipt's own
    # preparation_record and validated, never hardcoded past validation.
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    assert receipt_record["preparation_record"]["model_seed"] == hyd.MODEL_SEED_A
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    assert source.model_seed == hyd.MODEL_SEED_A


def test_build_best_epoch_source_rejects_wrong_receipt_model_seed(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["preparation_record"]["model_seed"] = 999999
    receipt_path2 = _write_receipt(short_tmp_path, "wrong_model_seed.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="model_seed"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_rejects_missing_receipt_model_seed(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    del tampered["preparation_record"]["model_seed"]
    receipt_path2 = _write_receipt(short_tmp_path, "missing_model_seed.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="model_seed"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_rejects_malformed_receipt_model_seed(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["preparation_record"]["model_seed"] = True  # bool must not pass as a strict int
    receipt_path2 = _write_receipt(short_tmp_path, "bool_model_seed.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="model_seed"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)

    tampered2 = json.loads(json.dumps(receipt_record))
    tampered2["preparation_record"]["model_seed"] = str(hyd.MODEL_SEED_A)
    receipt_path3 = _write_receipt(short_tmp_path, "string_model_seed.json", tampered2)
    with pytest.raises(hyd.HydrologicalConsumerError, match="model_seed"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered2, source_receipt_path=receipt_path3)


def test_build_best_epoch_source_exposes_receipt_provenance_fields_exactly(short_tmp_path, monkeypatch):
    # Provenance requirement: wandb_sweep_id/wandb_run_id/executor_mode must
    # be preserved verbatim from the receipt, never inferred/manufactured.
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    assert source.wandb_sweep_id == receipt_record["preparation_record"]["wandb_sweep_id"]
    assert source.wandb_run_id == receipt_record["preparation_record"]["wandb_run_id"]
    assert source.executor_mode == receipt_record["executor_mode"]
    assert source.wandb_sweep_id == "v2-prod-sweep"
    assert source.wandb_run_id == "run-7"
    assert source.executor_mode == receipt_record["executor_mode"]


def test_build_best_epoch_source_preserves_null_wandb_provenance_distinctly(short_tmp_path, monkeypatch):
    # A random-control-style trial with no live W&B run must retain an
    # explicit null, never a fabricated placeholder string.
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
        wandb_sweep_id=None, wandb_run_id=None,
    )
    assert receipt_record["preparation_record"]["wandb_sweep_id"] is None
    assert receipt_record["preparation_record"]["wandb_run_id"] is None
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    assert source.wandb_sweep_id is None
    assert source.wandb_run_id is None


def test_build_best_epoch_source_rejects_malformed_receipt_wandb_provenance(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    tampered["preparation_record"]["wandb_sweep_id"] = 12345  # must be str or None
    receipt_path2 = _write_receipt(short_tmp_path, "malformed_wandb_sweep_id.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="wandb_sweep_id"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_build_best_epoch_source_rejects_missing_receipt_executor_mode(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, _ = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    tampered = json.loads(json.dumps(receipt_record))
    del tampered["executor_mode"]
    receipt_path2 = _write_receipt(short_tmp_path, "missing_executor_mode.json", tampered)
    with pytest.raises(hyd.HydrologicalConsumerError, match="executor_mode"):
        hyd.build_v2_best_epoch_source(execution_provenance=tampered, source_receipt_path=receipt_path2)


def test_configuration_result_fails_closed_on_support_contract_checksum_mismatch(short_tmp_path, monkeypatch):
    # scenario 13 (contract checksum/version half)
    receipt_path, receipt_record, contract = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.50),
    )
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)

    package_root = short_tmp_path / "fixture_pkg"
    package_identity = _fixture_package_identity(package_root, contract)

    bad_sha_source = replace(source, support_contract_sha256="0" * 64)
    with pytest.raises(hyd.HydrologicalConsumerError, match="support_contract_sha256"):
        hyd.evaluate_v2_configuration_hydrological_result(
            best_epoch_source=bad_sha_source, package_root=package_root, package_identity=package_identity,
            contract=contract,
        )

    bad_version_source = replace(source, support_contract_version="wrong_contract_id")
    with pytest.raises(hyd.HydrologicalConsumerError, match="support_contract_version"):
        hyd.evaluate_v2_configuration_hydrological_result(
            best_epoch_source=bad_version_source, package_root=package_root, package_identity=package_identity,
            contract=contract,
        )


def test_configuration_result_fails_closed_on_rescored_objective_mismatch(short_tmp_path, monkeypatch):
    # scenario 11: re-scored objective not equal to official objective
    receipt_path, receipt_record, contract = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.7),
    )
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    # Self-consistent record (objective_score == trajectory[best_epoch] ==
    # 0.7) but not equal to what the real evaluator will compute (0.5).
    _wire_synthetic_epoch_v2(monkeypatch, contract, aggregate_median=0.5)
    package_root = short_tmp_path / "fixture_pkg"
    with pytest.raises(hyd.HydrologicalConsumerError, match="re-scored"):
        hyd.evaluate_v2_configuration_hydrological_result(
            best_epoch_source=source, package_root=package_root,
            package_identity=_fixture_package_identity(package_root, contract), contract=contract,
        )


def test_configuration_result_succeeds_when_rescored_objective_matches(short_tmp_path, monkeypatch):
    # scenario 11: valid selected epoch (positive control via the fast seam)
    receipt_path, receipt_record, contract = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.5),
    )
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    _wire_synthetic_epoch_v2(monkeypatch, contract, aggregate_median=0.5)
    package_root = short_tmp_path / "fixture_pkg"
    result = hyd.evaluate_v2_configuration_hydrological_result(
        best_epoch_source=source, package_root=package_root,
        package_identity=_fixture_package_identity(package_root, contract), contract=contract,
    )
    assert result.rescored_objective == pytest.approx(0.5)
    assert result.official_objective == pytest.approx(0.5)
    assert set(result.admitted_series_by_basin) == set(contract["basin_ids"])


def test_standalone_consumer_requires_package_qualification_argument():
    parameter = inspect.signature(hyd.evaluate_v2_configuration_hydrological_result).parameters[
        "package_identity"
    ]
    assert parameter.default is inspect.Parameter.empty


def test_standalone_consumer_rejects_package_b_with_package_a_identity(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, contract = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.5)
    )
    source = hyd.build_v2_best_epoch_source(
        execution_provenance=receipt_record, source_receipt_path=receipt_path
    )
    package_a = short_tmp_path / "package_a"
    package_b = short_tmp_path / "package_b"
    identity_a = _fixture_package_identity(package_a, contract)
    with pytest.raises(hyd.HydrologicalConsumerError, match="not the qualified package root"):
        hyd.evaluate_v2_configuration_hydrological_result(
            best_epoch_source=source, package_root=package_b,
            package_identity=identity_a, contract=contract,
        )


def test_standalone_consumer_revalidates_receipt_at_point_of_use(short_tmp_path, monkeypatch):
    receipt_path, receipt_record, contract = _real_receipt(
        short_tmp_path, monkeypatch, fixed_scores_by_epoch=_linear_trajectory(3, 0.5)
    )
    source = hyd.build_v2_best_epoch_source(
        execution_provenance=receipt_record, source_receipt_path=receipt_path
    )
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    package_root = short_tmp_path / "package"
    with pytest.raises(hyd.HydrologicalConsumerError, match="standalone source authentication"):
        hyd.evaluate_v2_configuration_hydrological_result(
            best_epoch_source=source, package_root=package_root,
            package_identity=_fixture_package_identity(package_root, contract), contract=contract,
        )


# --------------------------------------------------------------------------- #
# Section E: cross-configuration observed-side identity (scenarios 3, 14, 15, 16).
# --------------------------------------------------------------------------- #


def _admitted(basin_id, date, obs, sim):
    return fixed.AdmittedSeries(basin_id=basin_id, date=np.asarray(date), obs_m3s=np.asarray(obs), sim_m3s=np.asarray(sim))


def _patch_canonical_package_series(monkeypatch, series_by_basin):
    """Monkeypatches the package-canonical-series loader as it is bound in
    ``hyd`` (a direct-imported name, so ``fixed.derive_canonical_package_observed_series``
    is not sufficient -- ``hyd``'s own two call sites resolve the name
    against ``hyd``'s module globals)."""

    def fake(*, package_root, basin_id, contract, package_identity=None):
        return series_by_basin[basin_id]

    monkeypatch.setattr(hyd, "derive_canonical_package_observed_series", fake)


_SECTION_E_CONTRACT = {"contract_id": "fixture_contract", "checksum_sha256": "f" * 64}
_SECTION_E_PACKAGE_IDENTITY = _fixture_package_identity("fixture_pkg", _SECTION_E_CONTRACT)


def test_cross_configuration_canonical_facts_derived_from_package_not_trials(monkeypatch):
    # RD1-C4 reproducibility correction: canonical Q98 facts must come from
    # the package-canonical series, never from either trial's own admitted
    # series -- even when both trials happen to agree with each other but
    # NOT with the package (a scenario the old cross-trial-only check could
    # never detect).
    date = np.array([1, 2, 3])
    package_obs = np.array([10.0, 20.0, 30.0])
    trial_obs = np.array([1.0, 2.0, 3.0])  # agrees cross-trial, disagrees with package
    canonical_series = fixed.CanonicalPackageObservedSeries(basin_id="b1", date=date, obs_m3s=package_obs)
    results = {
        "cfgA": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, trial_obs, trial_obs * 1.1)}),
        "cfgB": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, trial_obs, trial_obs * 0.9)}),
    }
    _patch_canonical_package_series(monkeypatch, {"b1": canonical_series})
    with pytest.raises(hyd.HydrologicalConsumerError, match="provenance envelope"):
        hyd._derive_canonical_q98_facts_from_package(
            results, package_root="fixture_pkg",
            package_identity=_SECTION_E_PACKAGE_IDENTITY, contract=_SECTION_E_CONTRACT,
        )


def test_cross_configuration_canonical_facts_derived_when_package_and_trials_agree(monkeypatch):
    date = np.array([1, 2, 3])
    obs = np.array([1.0, 2.0, 3.0])
    canonical_series = fixed.CanonicalPackageObservedSeries(basin_id="b1", date=date, obs_m3s=obs)
    results = {
        "cfgA": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, obs, obs * 1.1)}),
        "cfgB": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, obs, obs * 0.9)}),
    }
    _patch_canonical_package_series(monkeypatch, {"b1": canonical_series})
    canonical = hyd._derive_canonical_q98_facts_from_package(
            results, package_root="fixture_pkg",
            package_identity=_SECTION_E_PACKAGE_IDENTITY, contract=_SECTION_E_CONTRACT,
        )
    assert set(canonical) == {"b1"}
    np.testing.assert_array_equal(canonical["b1"].canonical_obs_m3s, obs)


def test_cross_configuration_float32_reconstruction_scale_admitted(monkeypatch):
    # requirement 5 bullet 1: two pickle observation arrays differing only
    # at demonstrated float32 reconstruction scale must be admitted by the
    # provenance audit (not rejected).
    date = np.array([1, 2, 3])
    package_obs = np.array([12.345, 23.456, 100.0], dtype=np.float64)
    # Simulate float32 round-trip reconstruction noise (the diagnostic's
    # actual observed failure mode), well within PROVENANCE_RTOL/ATOL.
    trial_a_obs = package_obs.astype(np.float32).astype(np.float64)
    trial_b_obs = (package_obs.astype(np.float32) * np.float32(1.0)).astype(np.float64)
    canonical_series = fixed.CanonicalPackageObservedSeries(basin_id="b1", date=date, obs_m3s=package_obs)
    results = {
        "cfgA": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, trial_a_obs, trial_a_obs)}),
        "cfgB": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, trial_b_obs, trial_b_obs)}),
    }
    _patch_canonical_package_series(monkeypatch, {"b1": canonical_series})
    canonical = hyd._derive_canonical_q98_facts_from_package(
            results, package_root="fixture_pkg",
            package_identity=_SECTION_E_PACKAGE_IDENTITY, contract=_SECTION_E_CONTRACT,
        )
    # Formal metric evaluation receives the exact same package-derived
    # observations for both trials (requirement 5 bullet 2), and Q98 facts
    # are package-canonical (requirement 5 bullet 4).
    np.testing.assert_array_equal(canonical["b1"].canonical_obs_m3s, package_obs)


def test_cross_configuration_missing_basin_fails_closed():
    # scenario 14 (basin population half)
    date = np.array([1, 2, 3])
    obs = np.array([1.0, 2.0, 3.0])
    results = {
        "cfgA": SimpleNamespace(admitted_series_by_basin={
            "b1": _admitted("b1", date, obs, obs), "b2": _admitted("b2", date, obs, obs),
        }),
        "cfgB": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, obs, obs)}),
    }
    with pytest.raises(hyd.HydrologicalConsumerError, match="basin population"):
        hyd._derive_canonical_q98_facts_from_package(
            results, package_root="fixture_pkg",
            package_identity=_SECTION_E_PACKAGE_IDENTITY, contract=_SECTION_E_CONTRACT,
        )


def test_cross_configuration_timestamp_disagreement_fails_closed(monkeypatch):
    # scenario 15
    obs = np.array([1.0, 2.0, 3.0])
    canonical_series = fixed.CanonicalPackageObservedSeries(basin_id="b1", date=np.array([1, 2, 3]), obs_m3s=obs)
    results = {
        "cfgA": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", [1, 2, 3], obs, obs)}),
        "cfgB": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", [1, 2, 4], obs, obs)}),
    }
    _patch_canonical_package_series(monkeypatch, {"b1": canonical_series})
    with pytest.raises(hyd.HydrologicalConsumerError, match="timestamps differ"):
        hyd._derive_canonical_q98_facts_from_package(
            results, package_root="fixture_pkg",
            package_identity=_SECTION_E_PACKAGE_IDENTITY, contract=_SECTION_E_CONTRACT,
        )


def test_cross_configuration_observed_value_disagreement_fails_closed(monkeypatch):
    # scenario 16: a materially different observed value (not float32-scale
    # noise) must still fail closed against the package-canonical series.
    date = [1, 2, 3]
    obs_a = np.array([1.0, 2.0, 3.0])
    obs_b = np.array([1.0, 2.0, 3.5])
    canonical_series = fixed.CanonicalPackageObservedSeries(basin_id="b1", date=np.array(date), obs_m3s=obs_a)
    results = {
        "cfgA": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, obs_a, obs_a)}),
        "cfgB": SimpleNamespace(admitted_series_by_basin={"b1": _admitted("b1", date, obs_b, obs_b)}),
    }
    _patch_canonical_package_series(monkeypatch, {"b1": canonical_series})
    with pytest.raises(hyd.HydrologicalConsumerError, match="provenance envelope"):
        hyd._derive_canonical_q98_facts_from_package(
            results, package_root="fixture_pkg",
            package_identity=_SECTION_E_PACKAGE_IDENTITY, contract=_SECTION_E_CONTRACT,
        )


# --------------------------------------------------------------------------- #
# Section F: production batch-shape gate, including the RD1-C4 review
# Finding 3 fix (legal duplicate configuration_id) (scenario 14,
# configuration half).
# --------------------------------------------------------------------------- #


def _hyperparameters(order: int) -> dict:
    return {"learning_rate": 3e-4 + order * 1e-6, "hidden_size": 128, "embedding_dropout": 0.10,
            "output_dropout": 0.25, "batch_size": 256, "seq_length": 96}


def _trajectory(order: int) -> dict:
    peak_epoch = 3 + (order % 4)
    return {epoch: 0.30 - abs(epoch - peak_epoch) * 0.01 for epoch in range(1, 13)}


def _make_source(*, search_arm, proposal_order, hyperparameters, fixed_support_epoch_trajectory,
                  support_contract_version="fixture_contract_v001", support_contract_sha256="0" * 64,
                  execution_generation=1, retry_of_trial_id=None, run_dir="fixture_run",
                  source_receipt_path="fixture_receipt.json", source_receipt_sha256="0" * 64, git_commit=None,
                  wandb_sweep_id=None, wandb_run_id=None, executor_mode=None):
    """Build one fully self-consistent ``V2BestEpochSource`` via the same
    qualified identity helpers the real campaign uses (accessed off the
    consumer module's own imported names, so a monkeypatch on ``hyd.X``
    affects both this construction and ``_verify_best_epoch_source_identity``'s
    independent recomputation identically)."""
    canonical = hyd.canonical_hyperparameters_v2(hyperparameters)
    configuration_id = hyd.configuration_id_v2(
        canonical, support_contract_version=support_contract_version, support_contract_sha256=support_contract_sha256,
    )
    proposal_id = hyd.proposal_id_v2(search_arm, proposal_order)
    trial_id = hyd.trial_id_v2(configuration_id, proposal_id, execution_generation=execution_generation)
    diagnostics = derive_trajectory_diagnostics(fixed_support_epoch_trajectory)
    return hyd.V2BestEpochSource(
        campaign_id=hyd.CAMPAIGN_ID_V2, domain_version=hyd.DOMAIN_VERSION_V2, search_arm=search_arm,
        proposal_order=proposal_order, proposal_id=proposal_id, configuration_id=configuration_id,
        canonical_hyperparameters=canonical, trial_id=trial_id, execution_generation=execution_generation,
        retry_of_trial_id=retry_of_trial_id, official_objective=diagnostics["best_score"],
        best_epoch=diagnostics["best_epoch"], fixed_support_metric_name=V2_METRIC_NAME,
        fixed_support_epoch_trajectory=fixed_support_epoch_trajectory,
        fidelity_id=hyd.FIDELITY_ID_V2, model_seed=hyd.MODEL_SEED_A,
        evaluation_scope=hyd._EXPECTED_EVALUATION_SCOPE_V2, sealed_scope=hyd._EXPECTED_SEALED_SCOPE_V2,
        support_contract_version=support_contract_version, support_contract_sha256=support_contract_sha256,
        run_dir=run_dir, source_receipt_path=source_receipt_path, source_receipt_sha256=source_receipt_sha256,
        git_commit=git_commit, wandb_sweep_id=wandb_sweep_id, wandb_run_id=wandb_run_id,
        executor_mode=executor_mode,
    )


def _valid_24_sources():
    """12 bayesian + 12 random_control sources; at each proposal_order the
    bayesian and random_control sibling legally propose the IDENTICAL
    six-axis coordinate (RD1-C4 review Finding 3: two distinct trials may
    share one configuration_id), so the base "valid batch" itself already
    exercises the fix, not only the dedicated positive/negative tests below.
    """
    sources = []
    for order in range(1, 13):
        hp = _hyperparameters(order)
        traj = _trajectory(order)
        sources.append(_make_source(search_arm="bayesian", proposal_order=order, hyperparameters=hp,
                                    fixed_support_epoch_trajectory=traj))
        sources.append(_make_source(search_arm="random_control", proposal_order=order, hyperparameters=hp,
                                    fixed_support_epoch_trajectory=traj))
    return sources


def test_production_batch_shape_accepts_exactly_24_12plus12_400_with_legal_duplicate_configuration_ids():
    contract = _contract(400)
    sources = _valid_24_sources()
    configuration_ids = [source.configuration_id for source in sources]
    assert len(set(configuration_ids)) == 12  # each legally shared by exactly 2 trials
    hyd._validate_production_batch_shape(sources, contract)  # must not raise


def test_production_batch_shape_rejects_wrong_total_count():
    contract = _contract(400)
    sources = _valid_24_sources()[:-1]
    with pytest.raises(hyd.HydrologicalConsumerError, match="exactly 24"):
        hyd._validate_production_batch_shape(sources, contract)


def test_production_batch_shape_rejects_wrong_arm_split():
    contract = _contract(400)
    sources = _valid_24_sources()
    extra_bayesian = _make_source(search_arm="bayesian", proposal_order=99, hyperparameters=_hyperparameters(99),
                                  fixed_support_epoch_trajectory=_trajectory(99))
    sources[1] = extra_bayesian  # was random_control -> now 13 bayesian / 11 random_control
    with pytest.raises(hyd.HydrologicalConsumerError, match="12 bayesian"):
        hyd._validate_production_batch_shape(sources, contract)


def test_production_batch_shape_allows_legal_duplicate_configuration_id_across_matching_trials():
    contract = _contract(400)
    hp = _hyperparameters(1)
    traj = _trajectory(1)
    source_a = _make_source(search_arm="bayesian", proposal_order=1, hyperparameters=hp, fixed_support_epoch_trajectory=traj)
    source_b = _make_source(search_arm="random_control", proposal_order=1, hyperparameters=hp, fixed_support_epoch_trajectory=traj)
    assert source_a.configuration_id == source_b.configuration_id
    assert source_a.trial_id != source_b.trial_id
    sources = _valid_24_sources()
    sources[0], sources[1] = source_a, source_b
    hyd._validate_production_batch_shape(sources, contract)  # must not raise


def test_production_batch_shape_rejects_duplicate_proposal_id():
    contract = _contract(400)
    sources = _valid_24_sources()
    duplicate = _make_source(
        search_arm=sources[0].search_arm, proposal_order=sources[0].proposal_order,
        hyperparameters=_hyperparameters(101), fixed_support_epoch_trajectory=_trajectory(101),
    )
    assert duplicate.proposal_id == sources[0].proposal_id
    sources[2] = duplicate  # sources[2] was also "bayesian" -> arm split stays 12/12
    with pytest.raises(hyd.HydrologicalConsumerError, match="duplicate proposal_id"):
        hyd._validate_production_batch_shape(sources, contract)


def test_production_batch_shape_rejects_conflicting_hyperparameters_under_one_configuration_id(monkeypatch):
    # A genuinely different six-axis coordinate must never be allowed to
    # silently share a configuration_id with an existing one -- simulated
    # here via a forced hash collision (a real SHA-256 collision cannot be
    # constructed), since normal recomputation already guarantees distinct
    # hyperparameters hash to distinct configuration_ids.
    contract = _contract(400)
    real_configuration_id_v2 = hyd.configuration_id_v2
    hp_a, hp_b = _hyperparameters(1), _hyperparameters(2)
    collided_configuration_id = "sweep_v2_cfg_" + "c" * 20
    collided_canonical = [dict(hyd.canonical_hyperparameters_v2(hp_a)), dict(hyd.canonical_hyperparameters_v2(hp_b))]

    def _patched(hyperparameters, **kwargs):
        canonical = dict(hyd.canonical_hyperparameters_v2(hyperparameters))
        if canonical in collided_canonical:
            return collided_configuration_id
        return real_configuration_id_v2(hyperparameters, **kwargs)

    # A genuine SHA-256 collision cannot be constructed, so the collision is
    # simulated by patching configuration_id_v2 to return one fixed id for
    # exactly the two (deliberately distinct) hyperparameter sets under
    # test, while every other source in the batch still recomputes normally
    # -- so _verify_best_epoch_source_identity still passes for all 24
    # sources, isolating the duplicate-configuration_id conflict check.
    monkeypatch.setattr(hyd, "configuration_id_v2", _patched)
    sources = _valid_24_sources()
    source_a = _make_source(search_arm="bayesian", proposal_order=1, hyperparameters=hp_a,
                            fixed_support_epoch_trajectory=_trajectory(1))
    source_b = _make_source(search_arm="bayesian", proposal_order=2, hyperparameters=hp_b,
                            fixed_support_epoch_trajectory=_trajectory(2))
    assert source_a.configuration_id == source_b.configuration_id == collided_configuration_id
    assert dict(source_a.canonical_hyperparameters) != dict(source_b.canonical_hyperparameters)
    sources[0], sources[2] = source_a, source_b  # both "bayesian" -> arm split stays 12/12
    with pytest.raises(hyd.HydrologicalConsumerError, match="conflicting canonical hyperparameters"):
        hyd._validate_production_batch_shape(sources, contract)


def test_production_batch_shape_rejects_non_400_basin_population():
    contract = _contract(399)
    with pytest.raises(hyd.HydrologicalConsumerError, match="400"):
        hyd._validate_production_batch_shape(_valid_24_sources(), contract)


# --------------------------------------------------------------------------- #
# Section F2: formal receipt-backed batch assembly through the PUBLIC
# assemble_rd1_hydrological_review entry point (Source Defect 2 -- formal
# assembly must revalidate receipt qualification; Regression Gap 1 --
# duplicate-coordinate behavior through the full formal path). All 24
# sources here are genuine, receipt-backed V2BestEpochSource instances built
# once per module by _build_24_source_fixture (12 bayesian + 12
# random_control sharing one real 400-basin fixed-support contract, with one
# deliberate shared learning_rate pair producing a legal duplicate
# configuration_id) -- never hand-built self-consistent fixtures standing in
# for the base positive case, so success here proves the real
# build_v2_best_epoch_source -> assemble_rd1_hydrological_review chain
# actually works end-to-end. Negative cases swap individual real sources for
# hand-built/mutated ones (the same lightweight style Section F uses)
# precisely to prove the FULL formal path -- not just
# _validate_production_batch_shape in isolation -- rejects them.
# --------------------------------------------------------------------------- #


def _build_400_basin_fixed_support_contract(tmp_path, **identities) -> Path:
    """A 400-basin variant of ``tests.test_sweep_v2_six_axis_execution``'s
    own private ``_build_fixed_support_contract`` (which covers only one
    fixture basin -- sufficient for that file's own prepare/execute-only
    tests, but not for ``assemble_rd1_hydrological_review``'s own 400-
    unique-basin batch-shape requirement). Monkeypatched in for
    ``_build_24_source_fixture`` so every real receipt's own bound fixed-
    support-contract identity actually spans the full 400-basin screening
    population, while every other identity/plumbing detail (package/split/
    screening setup via ``_paths_v2`` itself) stays exactly as tested
    elsewhere."""
    n = 3
    basin_ids = list(REAL_DEVELOPMENT[:400])
    per_basin_date = {basin_id: np.arange(n) for basin_id in basin_ids}
    per_basin_admitted = {basin_id: np.ones(n, dtype=bool) for basin_id in basin_ids}
    contract = fixed.build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=6, target_variable="qobs_mm_per_h_lead06",
        period="test_period", date_start="2024-01-01", date_end="2024-01-01",
        source_gap_policy_identity="test_gap_policy_v001", screening_basin_ids_sha256="0" * 64,
        package_manifest_sha256=identities.get("package_manifest_sha256", "a" * 64),
        package_file_checksums_sha256=identities.get("package_file_checksums_sha256", "b" * 64),
        package_run_provenance_sha256=identities.get("package_run_provenance_sha256", "c" * 64),
        development_split_sha256=identities.get("development_split_sha256", "d" * 64),
        spatial_holdout_split_sha256=identities.get("spatial_holdout_split_sha256", "e" * 64),
        per_basin_date=per_basin_date, per_basin_admitted=per_basin_admitted,
    )
    return fixed.write_fixed_support_contract(contract, tmp_path / "fixed_support_contract.json")


def _build_24_source_fixture(tmp_path, monkeypatch, *, shared_learning_rate_orders=(1, 2)):
    """Build 24 real, receipt-backed V2BestEpochSource instances (12
    bayesian + 12 random_control) sharing one real 400-basin fixed-support
    contract. ``paths``/the underlying package are built ONCE and reused
    across every trial (only ``learning_rate`` varies per trial, keeping
    every other axis fixed) -- two bayesian trials (orders 1 and 2, by
    default) deliberately share one learning_rate so their canonical
    hyperparameters -- and hence configuration_id -- legally collide (RD1-C4
    review Finding 3), while every other trial gets a distinct coordinate.
    Callers apply _wire_synthetic_epoch_v2 themselves (per-test, function-
    scoped) before evaluating these sources through
    assemble_rd1_hydrological_review, so this builder never performs 24 x
    400 real NetCDF reads."""
    monkeypatch.setattr(_sweep_v2_exec_tests, "_build_fixed_support_contract", _build_400_basin_fixed_support_contract)
    paths = _paths_v2(tmp_path / "shared", monkeypatch)
    contract = fixed.load_fixed_support_contract(paths.fixed_support_contract_path)
    sources = []
    for arm in ("bayesian", "random_control"):
        for order in range(1, 13):
            if arm == "bayesian" and order in shared_learning_rate_orders:
                learning_rate = 3.0e-4
            else:
                offset = order if arm == "bayesian" else order + 12
                learning_rate = 1.0e-4 + offset * 3.0e-5
            proposal = _proposal_v2(
                learning_rate=learning_rate, proposal_order=order,
                wandb_sweep_id=f"v2-{arm}-sweep", wandb_run_id=f"run-{arm}-{order}",
            )
            prepared = v2_adapter._prepare_proposal_v2(proposal=proposal, paths=paths, expected_arm=arm)
            record = write_prepared_proposal_v2(prepared, tmp_path / f"prepared_{arm}_{order}")
            nh_run_dir = tmp_path / f"nh_run_{arm}_{order}"
            output_dir = tmp_path / f"trial_out_{arm}_{order}"
            receipt_path, receipt_record = _execute_v2_receipt(
                record, nh_run_dir, output_dir,
                fixed_scores_by_epoch=_linear_trajectory(peak_epoch=6, peak_value=0.5),
            )
            sources.append(
                hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
            )
    return sources, contract, paths.package_root


@pytest.fixture(scope="module")
def formal_batch():
    """24 real, receipt-backed V2BestEpochSource instances plus their shared
    contract/package_root -- built ONCE per module (a full 400-basin
    package build plus 24 prepare/execute receipts is expensive) and reused
    read-only by every test below. The one-time prepare/package-identity
    monkeypatch _build_24_source_fixture needs is applied and fully
    reverted entirely within this fixture's own setup, before any dependent
    test body runs, so it can never leak into unrelated tests (each
    dependent test applies its own function-scoped _wire_synthetic_epoch_v2
    patch for the raw-space evaluation seam itself)."""
    root = Path(__file__).parents[1] / "tmp" / "pytest_v2_formal_batch_scratch"
    root.mkdir(parents=True, exist_ok=True)
    real_path = root / uuid.uuid4().hex[:12]
    real_path.mkdir()
    scratch = Path("\\\\?\\" + str(real_path.resolve())) if os.name == "nt" else real_path
    mp = pytest.MonkeyPatch()
    try:
        sources, contract, package_root = _build_24_source_fixture(scratch, mp)
    finally:
        mp.undo()
    try:
        yield sources, contract, package_root
    finally:
        shutil.rmtree(real_path, ignore_errors=True)


def test_formal_assembly_accepts_real_24_source_batch_and_retains_legal_duplicate_configuration_id(
    formal_batch, monkeypatch
):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    review = hyd.assemble_rd1_hydrological_review(best_epoch_sources=sources, package_root=package_root, contract=contract)
    assert review.n_configurations == 24
    assert review.n_bayesian == 12
    assert review.n_random_control == 12
    configuration_ids = [source.configuration_id for source in sources]
    duplicated = {cid for cid in set(configuration_ids) if configuration_ids.count(cid) == 2}
    assert len(set(configuration_ids)) == 23 and len(duplicated) == 1  # exactly one legal shared coordinate
    dup_trial_ids = [source.trial_id for source in sources if source.configuration_id == next(iter(duplicated))]
    assert len(dup_trial_ids) == 2 and dup_trial_ids[0] != dup_trial_ids[1]
    # both legally repeated-coordinate trials are retained, independently
    # keyed/addressable by their own distinct trial_id -- never collapsed.
    assert set(dup_trial_ids).issubset(review.results_by_trial_id)
    assert review.results_by_trial_id[dup_trial_ids[0]] is not review.results_by_trial_id[dup_trial_ids[1]]


def test_formal_assembly_full_batch_coverage_is_exactly_24x400_with_duplicates_counted_separately(
    formal_batch, monkeypatch
):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    review = hyd.assemble_rd1_hydrological_review(best_epoch_sources=sources, package_root=package_root, contract=contract)
    assert review.n_basins == 400
    assert len(review.results_by_trial_id) == 24  # both legal duplicate-coordinate trials counted separately
    for coverage in review.coverage.values():
        assert coverage.n_total == 24 * 400 == 9600
        assert coverage.n_finite + coverage.n_unavailable == coverage.n_total


def test_formal_assembly_rejects_self_consistent_hand_built_source_not_bound_to_a_receipt(formal_batch, monkeypatch):
    # A source that is internally self-consistent (passes every
    # _verify_best_epoch_source_identity check) but was never proven to
    # derive from its own claimed receipt must still be rejected by the
    # FULL formal assemble_rd1_hydrological_review path, not merely by
    # _validate_production_batch_shape in isolation.
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    mutated = list(sources)
    mutated[3] = _make_source(  # slot 3 == bayesian proposal_order 4
        search_arm="bayesian", proposal_order=4, hyperparameters=_hyperparameters(102),
        fixed_support_epoch_trajectory=_trajectory(102),
        support_contract_version=contract["contract_id"], support_contract_sha256=contract["checksum_sha256"],
    )
    with pytest.raises(hyd.HydrologicalConsumerError, match="does not exist"):
        hyd.assemble_rd1_hydrological_review(best_epoch_sources=mutated, package_root=package_root, contract=contract)


def test_formal_assembly_rejects_source_whose_receipt_content_was_mutated_after_construction(
    formal_batch, monkeypatch, tmp_path
):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    victim = sources[5]
    mutated_path = tmp_path / "mutated_receipt.json"
    mutated_path.write_bytes(Path(victim.source_receipt_path).read_bytes() + b" ")
    mutated = list(sources)
    mutated[5] = replace(victim, source_receipt_path=str(mutated_path))  # recorded sha256 now stale
    with pytest.raises(hyd.HydrologicalConsumerError, match="content has changed"):
        hyd.assemble_rd1_hydrological_review(best_epoch_sources=mutated, package_root=package_root, contract=contract)


def test_formal_assembly_rejects_source_whose_declared_best_epoch_is_not_the_trajectory_maximum(
    formal_batch, monkeypatch
):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    victim = sources[7]
    trajectory = dict(victim.fixed_support_epoch_trajectory)
    true_best_epoch = max(trajectory, key=lambda epoch: trajectory[epoch])
    assert true_best_epoch == 6
    wrong_epoch = 3  # a LATER epoch (6) is genuinely better than this one
    assert trajectory[wrong_epoch] < trajectory[true_best_epoch]
    mutated = list(sources)
    mutated[7] = replace(victim, best_epoch=wrong_epoch, official_objective=trajectory[wrong_epoch])
    with pytest.raises(hyd.HydrologicalConsumerError, match="authoritative first-achieved-maximum epoch"):
        hyd.assemble_rd1_hydrological_review(best_epoch_sources=mutated, package_root=package_root, contract=contract)


def test_formal_assembly_rejects_duplicate_proposal_id(formal_batch, monkeypatch):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    mutated = list(sources)
    duplicate = _make_source(
        search_arm=mutated[0].search_arm, proposal_order=mutated[0].proposal_order,
        hyperparameters=_hyperparameters(101), fixed_support_epoch_trajectory=_trajectory(101),
        support_contract_version=contract["contract_id"], support_contract_sha256=contract["checksum_sha256"],
    )
    assert duplicate.proposal_id == mutated[0].proposal_id
    mutated[2] = duplicate  # slots 0 and 2 are both "bayesian" -> arm split stays 12/12
    with pytest.raises(hyd.HydrologicalConsumerError, match="duplicate proposal_id"):
        hyd.assemble_rd1_hydrological_review(best_epoch_sources=mutated, package_root=package_root, contract=contract)


def test_formal_assembly_rejects_duplicate_trial_id(formal_batch, monkeypatch):
    # A genuine SHA-256 collision cannot be constructed, so this simulates
    # one exactly like Section F's own configuration_id collision test, but
    # for trial_id_v2 directly -- proving the formal path independently
    # checks trial_id uniqueness rather than relying on proposal_id/
    # configuration_id uniqueness to imply it.
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    real_trial_id_v2 = hyd.trial_id_v2
    source_a, source_b = sources[0], sources[2]  # distinct proposal_id/configuration_id, same arm
    assert source_a.proposal_id != source_b.proposal_id
    collided_trial_id = "sweep_v2_trial_" + "d" * 20

    def _patched(configuration_id, proposal_id, **kwargs):
        if proposal_id in (source_a.proposal_id, source_b.proposal_id):
            return collided_trial_id
        return real_trial_id_v2(configuration_id, proposal_id, **kwargs)

    monkeypatch.setattr(hyd, "trial_id_v2", _patched)
    mutated = list(sources)
    mutated[0] = replace(source_a, trial_id=collided_trial_id)
    mutated[2] = replace(source_b, trial_id=collided_trial_id)
    with pytest.raises(hyd.HydrologicalConsumerError, match="duplicate trial_id"):
        hyd.assemble_rd1_hydrological_review(best_epoch_sources=mutated, package_root=package_root, contract=contract)


def test_formal_assembly_rejects_conflicting_hyperparameters_under_shared_configuration_id(formal_batch, monkeypatch):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    real_configuration_id_v2 = hyd.configuration_id_v2
    # Indices 0/1 (bayesian orders 1/2) already share canonical_hyperparameters
    # by fixture design (the fixture's own built-in legal duplicate-configuration_id
    # pair) -- picking either of them here would make the patched configuration_id_v2
    # below also match that unmutated sibling, failing identity re-verification for
    # it before this test's intended check is ever reached. Use two sources outside
    # that pair, each with genuinely unique canonical hyperparameters.
    source_a, source_b = sources[3], sources[5]
    assert dict(source_a.canonical_hyperparameters) != dict(source_b.canonical_hyperparameters)
    collided_configuration_id = "sweep_v2_cfg_" + "e" * 20
    collided_canonical = [dict(source_a.canonical_hyperparameters), dict(source_b.canonical_hyperparameters)]

    def _patched(hyperparameters, **kwargs):
        canonical = dict(hyperparameters)
        if canonical in collided_canonical:
            return collided_configuration_id
        return real_configuration_id_v2(hyperparameters, **kwargs)

    monkeypatch.setattr(hyd, "configuration_id_v2", _patched)
    mutated = list(sources)
    mutated[3] = replace(
        source_a, configuration_id=collided_configuration_id,
        trial_id=hyd.trial_id_v2(collided_configuration_id, source_a.proposal_id, execution_generation=source_a.execution_generation),
    )
    mutated[5] = replace(
        source_b, configuration_id=collided_configuration_id,
        trial_id=hyd.trial_id_v2(collided_configuration_id, source_b.proposal_id, execution_generation=source_b.execution_generation),
    )
    with pytest.raises(hyd.HydrologicalConsumerError, match="conflicting canonical hyperparameters"):
        hyd.assemble_rd1_hydrological_review(best_epoch_sources=mutated, package_root=package_root, contract=contract)


def test_formal_assembly_mutating_callers_source_list_after_construction_has_no_effect(formal_batch, monkeypatch):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    caller_list = list(sources)
    review = hyd.assemble_rd1_hydrological_review(
        best_epoch_sources=caller_list, package_root=package_root, contract=contract
    )
    before = dict(review.results_by_trial_id)
    caller_list.clear()
    caller_list.append("not a source")
    assert dict(review.results_by_trial_id) == before
    assert review.n_configurations == 24


def test_formal_assembly_results_by_trial_id_rejects_external_mutation(formal_batch, monkeypatch):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    review = hyd.assemble_rd1_hydrological_review(best_epoch_sources=sources, package_root=package_root, contract=contract)
    some_trial_id = next(iter(review.results_by_trial_id))
    with pytest.raises(TypeError):
        review.results_by_trial_id[some_trial_id] = None
    with pytest.raises((TypeError, AttributeError)):
        review.results_by_trial_id.clear()
    with pytest.raises((TypeError, AttributeError)):
        review.results_by_trial_id.pop(some_trial_id)


# --------------------------------------------------------------------------- #
# Section G: coverage accounting (scenario 18) and the RD1-C4 review
# Finding 7 fix (fail-closed on a genuinely missing metric field).
# --------------------------------------------------------------------------- #


def _q98(basin_id, *, high_flow_nse_available):
    return hyd.Q98ConfigurationBasinDiagnostics(
        basin_id=basin_id, trial_id="trial", n_high_flow=60,
        q98_normalized_rmse=float("nan") if basin_id == "b2" else 0.1,
        relative_volume_bias=0.0, observed_peak_time_magnitude_error=0.0,
        high_flow_nse=0.8 if high_flow_nse_available else float("nan"),
        high_flow_nse_available=high_flow_nse_available,
    )


def test_coverage_counts_finite_and_never_treats_nan_as_zero():
    basin_ids = ["b1", "b2"]
    results = {
        "trialA": SimpleNamespace(
            fixed_support_result={"per_basin": [
                {"basin_id": "b1", "nse": 0.5, "kge": 0.4, "kge_r": 0.9, "kge_alpha": 1.0, "kge_beta": 1.0},
                {"basin_id": "b2", "nse": float("nan"), "kge": float("nan"), "kge_r": float("nan"),
                 "kge_alpha": float("nan"), "kge_beta": float("nan")},
            ]},
            q98_diagnostics_by_basin={"b1": _q98("b1", high_flow_nse_available=True), "b2": _q98("b2", high_flow_nse_available=False)},
        ),
    }
    coverage = hyd._compute_coverage(results, basin_ids)
    assert coverage["nse"].n_total == 2
    assert coverage["nse"].n_finite == 1
    assert coverage["nse"].n_unavailable == 1
    assert coverage["q98_normalized_rmse"].n_finite == 1
    assert coverage["high_flow_nse"].n_finite == 1
    assert coverage["high_flow_sample_count"].n_finite == 2  # always reported


def test_coverage_fails_closed_on_genuinely_missing_required_metric_field():
    # RD1-C4 review Finding 7: a per_basin row that never carries a required
    # metric key at all must fail closed, distinct from an explicit NaN
    # under that key (which is legitimately "undefined", not "missing").
    basin_ids = ["b1"]
    results = {
        "trialA": SimpleNamespace(
            fixed_support_result={"per_basin": [
                {"basin_id": "b1", "nse": 0.5, "kge_r": 0.9, "kge_alpha": 1.0, "kge_beta": 1.0},  # "kge" absent
            ]},
            q98_diagnostics_by_basin={"b1": _q98("b1", high_flow_nse_available=True)},
        ),
    }
    with pytest.raises(hyd.HydrologicalConsumerError, match="missing required metric field"):
        hyd._compute_coverage(results, basin_ids)


def test_coverage_treats_numeric_zero_as_finite_and_available():
    # Regression Gap 3: a genuine 0.0 metric value is finite/available --
    # never confused with an explicitly-undefined NaN or a missing key.
    basin_ids = ["b1"]
    results = {
        "trialA": SimpleNamespace(
            fixed_support_result={"per_basin": [
                {"basin_id": "b1", "nse": 0.0, "kge": 0.0, "kge_r": 0.0, "kge_alpha": 0.0, "kge_beta": 0.0},
            ]},
            q98_diagnostics_by_basin={"b1": _q98("b1", high_flow_nse_available=True)},
        ),
    }
    coverage = hyd._compute_coverage(results, basin_ids)
    assert coverage["nse"].n_finite == 1
    assert coverage["nse"].n_unavailable == 0
    for metric_coverage in coverage.values():
        assert metric_coverage.n_finite + metric_coverage.n_unavailable == metric_coverage.n_total


# --------------------------------------------------------------------------- #
# Section H: the designated vertical synthetic integration test.
#
# Real package NetCDF + real validation pickle + real prepare/execute
# receipt + real fixed-support/raw-space evaluation + real C3 basin analysis
# + real Q98 math. Nothing scientific is mocked anywhere in this section.
# --------------------------------------------------------------------------- #


def _write_package_basin_netcdf(package_root, basin_id, *, area_km2, lead_hours, n, seed=1):
    rng = np.random.default_rng(seed)
    qobs_m3s = rng.uniform(1.0, 200.0, size=n)
    usable_n = n - lead_hours
    target_mm_per_h = np.full(n, np.nan)
    target_mm_per_h[:usable_n] = 3.6 * qobs_m3s[lead_hours:lead_hours + usable_n] / area_km2
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    xr.Dataset(
        {"qobs_m3s": ("date", qobs_m3s), "qobs_mm_per_h_lead06": ("date", target_mm_per_h)},
        coords={"date": np.arange(n)},
    ).to_netcdf(ts_dir / f"{basin_id}.nc")
    return qobs_m3s


def _write_validation_pickle(run_dir, period, epoch, basin_results):
    result_pickle = period_results_path(run_dir, period, epoch)
    result_pickle.parent.mkdir(parents=True, exist_ok=True)
    with open(result_pickle, "wb") as fh:
        pickle.dump(basin_results, fh)


def test_vertical_synthetic_integration_full_chain(short_tmp_path, monkeypatch):
    basin_id = "01234567"  # matches _prepared_record_v2's own fixed-support-contract fixture basin
    area_km2 = 100.0
    lead_hours = 6
    target_variable = "qobs_mm_per_h_lead06"
    n = 20
    package_n = 200  # well above derive_basin_area_km2's DEFAULT_MIN_AREA_SAMPLES floor
    best_epoch = 5

    record, paths = _prepared_record_v2(short_tmp_path / "prep_h", monkeypatch)
    contract = fixed.load_fixed_support_contract(paths.fixed_support_contract_path)

    package_root = short_tmp_path / "package"
    package_qobs_m3s = _write_package_basin_netcdf(
        package_root, basin_id, area_km2=area_km2, lead_hours=lead_hours, n=package_n
    )

    # RD1-C4 reproducibility correction: a genuine run's own admitted
    # obs_mm_per_h satisfies the package's own qobs_m3s<->target_mm_per_h
    # algebraic identity (lead-shift-aligned) up to float32 reconstruction
    # precision -- it is never independent of the package (see
    # docs/decision_log.md, this entry; RD1-C4 Q98 mismatch diagnostic
    # classification A). Derive it that way here, with a genuine float32
    # round-trip cast standing in for the diagnostic's own demonstrated
    # reconstruction-scale drift, so the provenance audit is exercised
    # honestly rather than trivially.
    obs_m3s_from_package = package_qobs_m3s[lead_hours:lead_hours + n]
    obs_mm_per_h = (3.6 * obs_m3s_from_package / area_km2).astype(np.float32).astype(np.float64)
    sim_mm_per_h = obs_mm_per_h * 1.1  # fully finite, deliberately biased high

    period_dataset = xr.Dataset(
        {f"{target_variable}_obs": ("date", obs_mm_per_h), f"{target_variable}_sim": ("date", sim_mm_per_h)},
        coords={"date": np.arange(n)},
    )
    nh_run_dir = short_tmp_path / "nh_run_h"
    _write_validation_pickle(nh_run_dir, contract["period"], best_epoch, {basin_id: {"1h": {"xr": period_dataset}}})

    # First, compute the real official value directly through the already-
    # qualified C4-A producer -- this is what a genuine campaign's execution
    # spine would have already recorded as the objective for this epoch.
    reference_result = fixed.evaluate_fixed_support_raw_space_metrics(
        run_dir=nh_run_dir, epoch=best_epoch, package_root=package_root, contract=contract, return_admitted_series=True,
    )
    official_value = fixed.extract_v2_objective_from_fixed_support_result(reference_result)
    assert np.isfinite(official_value)

    trajectory = _linear_trajectory(best_epoch, official_value)
    output_dir = short_tmp_path / "trial_out_h"
    receipt_path, receipt_record = _execute_v2_receipt(record, nh_run_dir, output_dir, fixed_scores_by_epoch=trajectory)
    source = hyd.build_v2_best_epoch_source(execution_provenance=receipt_record, source_receipt_path=receipt_path)
    assert source.best_epoch == best_epoch
    assert source.official_objective == pytest.approx(official_value)
    assert Path(source.run_dir) == nh_run_dir

    package_identity = _fixture_package_identity(package_root, contract)
    result = hyd.evaluate_v2_configuration_hydrological_result(
        best_epoch_source=source, package_root=package_root, package_identity=package_identity, contract=contract,
    )

    # scenario 11(a): valid selected epoch is proven end to end.
    assert result.rescored_objective == pytest.approx(official_value)
    assert result.official_objective == pytest.approx(official_value)

    # scenario 17: producer per-basin rows flow directly into C3 and retain
    # KGE/component passthrough fields.
    assert set(result.basin_distribution.per_basin["basin_id"]) == {basin_id}
    for column in ("kge", "kge_r", "kge_alpha", "kge_beta"):
        assert column in result.basin_distribution.per_basin.columns

    # Frozen Q98 package applied through the real admitted series.
    assert set(result.q98_diagnostics_by_basin) == {basin_id}
    diag = result.q98_diagnostics_by_basin[basin_id]
    assert diag.n_high_flow >= 1
    assert np.isfinite(diag.q98_normalized_rmse)
    assert np.isfinite(diag.relative_volume_bias)
    assert diag.relative_volume_bias > 0  # sim is deliberately biased 10% high

    # scenario 11(d): a self-consistent but scientifically wrong official
    # objective must be rejected by the deterministic re-scoring check.
    # Re-executes the SAME prepared trial identity into a fresh run_dir with
    # a fabricated (wrong) trajectory value at the same epoch.
    wrong_value = official_value + 1.0
    wrong_nh_run_dir = short_tmp_path / "nh_run_h_wrong"
    _write_validation_pickle(wrong_nh_run_dir, contract["period"], best_epoch, {basin_id: {"1h": {"xr": period_dataset}}})
    wrong_trajectory = _linear_trajectory(best_epoch, wrong_value)
    wrong_output_dir = short_tmp_path / "trial_out_h_wrong"
    wrong_receipt_path, wrong_receipt_record = _execute_v2_receipt(
        record, wrong_nh_run_dir, wrong_output_dir, fixed_scores_by_epoch=wrong_trajectory,
    )
    wrong_source = hyd.build_v2_best_epoch_source(execution_provenance=wrong_receipt_record, source_receipt_path=wrong_receipt_path)
    with pytest.raises(hyd.HydrologicalConsumerError, match="re-scored"):
        hyd.evaluate_v2_configuration_hydrological_result(
            best_epoch_source=wrong_source, package_root=package_root, package_identity=package_identity,
            contract=contract,
        )
