"""RD1-C4-F hydrograph supplement runner tests.

Reuses the real 24-source ``formal_batch`` fixture and the
``_wire_synthetic_epoch_v2`` monkeypatch seam already qualified in
``tests/test_stage1_v2_12plus12_hydrological_consumer.py`` -- the same
package/contract/run wiring the RD1-C4-E runner tests reuse -- so the series
this module extracts are produced by the exact same qualified
``evaluate_fixed_support_raw_space_metrics`` / ``derive_canonical_package_observed_series``
paths the production runner uses. This module tests extraction/wiring only,
never a parallel scientific computation.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baseline import rd1_c4_f_hydrograph_runner as runner
from src.baseline import stage1_v2_12plus12_hydrological_consumer as hyd
from src.baseline.stage1_rd1_c4_f_hydrograph_supplement import PeakWindow

from test_stage1_v2_12plus12_hydrological_consumer import (  # noqa: F401
    _fixture_package_identity,
    _wire_synthetic_epoch_v2,
    formal_batch,
)


@pytest.fixture()
def wired_batch(formal_batch, monkeypatch):
    sources, contract, package_root = formal_batch
    _wire_synthetic_epoch_v2(monkeypatch, contract)
    # _wire_synthetic_epoch_v2 substitutes qualify_package_identity/
    # derive_canonical_package_observed_series/derive_basin_area_km2_from_netcdf
    # only as bound in the hydrological-consumer/fixed-support-contract
    # modules' own globals (the synthetic fixture package has no real
    # per-basin NetCDF/qualification manifest, and building one would add no
    # test value -- see that fixture's own comment). This runner module
    # imported the same functions into its own globals, so the identical
    # substitutions are applied here too -- reusing the exact fake callables
    # the qualified fixture already wired (never a second, parallel fake).
    monkeypatch.setattr(runner, "qualify_package_identity", hyd.qualify_package_identity)
    monkeypatch.setattr(runner, "derive_canonical_package_observed_series", hyd.derive_canonical_package_observed_series)
    monkeypatch.setattr(
        runner, "derive_basin_area_km2_from_netcdf",
        lambda *_, **__: SimpleNamespace(area_km2=100.0, consistent=True, relative_mad=0.0),
    )
    targets_by_trial_id = {s.trial_id: s for s in sources}
    return sources, contract, package_root, targets_by_trial_id


def test_extract_basin_candidate_series_shares_one_observed_series_across_trials(wired_batch):
    sources, contract, package_root, targets_by_trial_id = wired_batch
    basin_id = contract["basin_ids"][0]
    trial_ids = [sources[0].trial_id, sources[1].trial_id]

    series_by_trial = runner.extract_basin_candidate_series(
        basin_id=basin_id,
        trial_ids=trial_ids,
        targets_by_trial_id=targets_by_trial_id,
        package_root=package_root,
        contract=contract,
    )

    assert set(series_by_trial) == set(trial_ids)
    first, second = (series_by_trial[t] for t in trial_ids)
    # candidate-independent observed series: identical across both trials
    np.testing.assert_array_equal(first.obs_m3s, second.obs_m3s)
    assert first.basin_id == basin_id == second.basin_id
    assert np.isfinite(first.area_km2) and first.area_km2 > 0
    assert first.area_km2 == second.area_km2
    assert len(first.dates) == len(first.obs_m3s) == len(first.sim_m3s)
    assert first.admitted_mask.all()  # AdmittedSeries is already the admitted set


def test_extract_basin_candidate_series_raises_on_unknown_trial(wired_batch):
    sources, contract, package_root, targets_by_trial_id = wired_batch
    basin_id = contract["basin_ids"][0]
    with pytest.raises(KeyError):
        runner.extract_basin_candidate_series(
            basin_id=basin_id,
            trial_ids=["not_a_real_trial_id"],
            targets_by_trial_id=targets_by_trial_id,
            package_root=package_root,
            contract=contract,
        )


def test_render_basin_incumbent_panel_produces_figure_and_peak_window(wired_batch, tmp_path):
    sources, contract, package_root, targets_by_trial_id = wired_batch
    basin_id = contract["basin_ids"][0]
    trial_ids = [sources[0].trial_id, sources[1].trial_id]
    manifest = {
        "order_to_incumbent_trial_id": {
            "bayesian__proposal_order_01": trial_ids[0],
            "bayesian__proposal_order_09": trial_ids[1],
        },
        "unique_incumbent_trial_ids": trial_ids,
    }
    out_path = tmp_path / "panel.png"

    peak_window = runner.render_basin_incumbent_panel(
        basin_id=basin_id,
        manifest=manifest,
        targets_by_trial_id=targets_by_trial_id,
        package_root=package_root,
        contract=contract,
        out_path=out_path,
    )

    assert isinstance(peak_window, PeakWindow)
    assert peak_window.basin_id == basin_id
    assert out_path.exists() and out_path.stat().st_size > 0


def test_candidate_labels_from_manifest_lists_every_order_per_trial():
    manifest = {
        "order_to_incumbent_trial_id": {
            "bayesian__proposal_order_01": "trial_a",
            "bayesian__proposal_order_03": "trial_a",
            "bayesian__proposal_order_09": "trial_b",
        },
        "unique_incumbent_trial_ids": ["trial_a", "trial_b"],
    }
    labels = runner.candidate_labels_from_manifest(manifest)
    assert set(labels) == {"trial_a", "trial_b"}
    assert "bayesian@1" in labels["trial_a"]
    assert "bayesian@3" in labels["trial_a"]
    assert "bayesian@9" in labels["trial_b"]


def test_render_basin_incumbent_panel_uses_arm_aware_colors(wired_batch, tmp_path):
    sources, contract, package_root, targets_by_trial_id = wired_batch
    basin_id = contract["basin_ids"][0]
    bayesian_trial = next(s.trial_id for s in sources if s.search_arm == "bayesian")
    random_trial = next(s.trial_id for s in sources if s.search_arm == "random_control")
    search_arm_by_trial_id = {s.trial_id: s.search_arm for s in sources}
    manifest = {
        "order_to_incumbent_trial_id": {
            "bayesian__proposal_order_01": bayesian_trial,
            "random_control__proposal_order_01": random_trial,
        },
        "unique_incumbent_trial_ids": [bayesian_trial, random_trial],
    }
    out_path = tmp_path / "panel_arm_colors.png"

    peak_window = runner.render_basin_incumbent_panel(
        basin_id=basin_id,
        manifest=manifest,
        targets_by_trial_id=targets_by_trial_id,
        package_root=package_root,
        contract=contract,
        out_path=out_path,
        search_arm_by_trial_id=search_arm_by_trial_id,
    )

    assert isinstance(peak_window, PeakWindow)
    assert out_path.exists() and out_path.stat().st_size > 0


def test_peak_window_to_event_window_carries_fields_through():
    import pandas as pd

    pw = PeakWindow(
        basin_id="01464000",
        peak_time=pd.Timestamp("2020-01-05"),
        peak_value=42.0,
        window_start=pd.Timestamp("2020-01-03"),
        window_end=pd.Timestamp("2020-01-07"),
        requested_half_window_hours=36,
        clipped_left=False,
        clipped_right=True,
        actual_window_hours=60.0,
    )
    ew = runner.peak_window_to_event_window(pw)
    assert ew.peak_time == pw.peak_time
    assert ew.peak_value == pw.peak_value
    assert ew.window_start == pw.window_start
    assert ew.window_end == pw.window_end
    assert ew.window_clipped is True
