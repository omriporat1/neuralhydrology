"""RD1-C4-F hydrograph supplement -- extraction/rendering runner.

Ties together already-qualified helpers to produce the descriptive,
run-progress-example hydrograph overlay figures for RD1-C4-F. This module
contains no new scientific math:

- the observed series plotted for a basin is always the candidate-independent
  package-canonical series from
  :func:`~.fixed_support_contract_v2.derive_canonical_package_observed_series`
  (never a per-run/per-trial observation);
- each trial's predicted series and admitted-sample support dates come from
  :func:`~.fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`
  with ``return_admitted_series=True`` -- the same extraction/conversion pass
  that produced the official RD1-C4-E per-basin metrics, not a second
  reconstruction;
- basin area comes from
  :func:`~.nh_raw_space_evaluation.derive_basin_area_km2_from_netcdf`;
- the actual multi-candidate overlay rendering reuses
  :func:`~.hydrograph_rendering.render_multi_candidate_basin_panel` /
  :func:`~.hydrograph_rendering.derive_comparison_scale`.

The only genuinely new logic here is reassembling these already-qualified
pieces into one :class:`~.hydrograph_rendering.BasinSeries` per (trial,
basin) and mapping the incumbent-order manifest (from
:mod:`~.stage1_rd1_c4_f_hydrograph_supplement`) onto deduplicated panel
candidates.

No classifier, winner, promotion, or tolerance decision is made here. Every
figure produced is an explicitly labeled descriptive run-progress example.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .fixed_support_contract_v2 import (
    derive_canonical_package_observed_series,
    evaluate_fixed_support_raw_space_metrics,
    load_fixed_support_contract,
)
from .hydrograph_atlas_events import EventWindow
from .hydrograph_rendering import (
    BasinSeries,
    basin_netcdf_path,
    compute_target_valid_dates,
    derive_basin_area_km2_from_netcdf,
    derive_comparison_scale,
    render_multi_candidate_basin_panel,
)
from .package_identity_qualification import qualify_package_identity
from .rd1_c4_trial_authentication import AuthenticatedTrialTarget, authenticate_trial_roster
from .stage1_rd1_c4_f_hydrograph_supplement import (
    PeakWindow,
    build_incumbent_order_mapping,
    select_observed_peak_window,
    validate_selected_basins_canonical,
)

__all__ = [
    "HydrographRunnerError",
    "extract_basin_candidate_series",
    "peak_window_to_event_window",
    "candidate_labels_from_manifest",
    "render_basin_incumbent_panel",
    "produce_rd1_c4_f_hydrograph_supplement",
    "build_parser",
    "main",
]

DEFAULT_SEARCH_ARMS = ("bayesian", "random_control")
DEFAULT_PROPOSAL_ORDERS = (1, 3, 6, 9, 12)


class HydrographRunnerError(ValueError):
    """Raised for an extraction/rendering contract violation in the RD1-C4-F
    hydrograph supplement runner -- never for an ordinary poor-skill outcome."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_window_to_event_window(peak_window: PeakWindow) -> EventWindow:
    """Adapts a :class:`~.stage1_rd1_c4_f_hydrograph_supplement.PeakWindow`
    into the :class:`~.hydrograph_atlas_events.EventWindow` shape that
    :func:`~.hydrograph_rendering.render_multi_candidate_basin_panel` /
    :func:`~.hydrograph_rendering.derive_comparison_scale` expect, without
    recomputing anything."""
    return EventWindow(
        magnitude_class="rd1_c4_f_descriptive_peak",
        peak_time=peak_window.peak_time,
        peak_value=peak_window.peak_value,
        window_start=peak_window.window_start,
        window_end=peak_window.window_end,
        window_clipped=peak_window.clipped_left or peak_window.clipped_right,
        n_missing_in_window=0,
    )


def extract_basin_candidate_series(
    *,
    basin_id: str,
    trial_ids: Sequence[str],
    targets_by_trial_id: Mapping[str, AuthenticatedTrialTarget],
    package_root,
    contract: dict,
) -> dict:
    """For one basin, returns ``{trial_id: BasinSeries}`` for every requested
    trial. Every returned series shares the exact same candidate-independent
    package-canonical observed array; only ``sim_m3s`` varies by trial. Raises
    :class:`HydrographRunnerError` if any trial's admitted-series support
    dates disagree with the canonical package-observed series' own support
    dates (a basin/date identity contradiction, never silently realigned).
    """
    package_identity = qualify_package_identity(package_root=package_root, contract=contract, basin_ids=[basin_id])
    canonical = derive_canonical_package_observed_series(
        package_root=package_root, basin_id=basin_id, contract=contract, package_identity=package_identity
    )
    nc_path = basin_netcdf_path(package_root, basin_id)
    area_result = derive_basin_area_km2_from_netcdf(
        nc_path,
        basin_id=basin_id,
        target_variable=contract["target_variable"],
        lead_hours=contract["lead_hours"],
    )
    if not area_result.consistent:
        raise HydrographRunnerError(f"basin {basin_id!r}: self-derived area is inconsistent, refusing to render")
    area_km2 = area_result.area_km2

    series_by_trial: dict = {}
    for trial_id in trial_ids:
        target = targets_by_trial_id[trial_id]
        result = evaluate_fixed_support_raw_space_metrics(
            run_dir=target.run_dir,
            epoch=target.best_epoch,
            package_root=package_root,
            contract=contract,
            basin_ids=[basin_id],
            return_admitted_series=True,
            package_identity=package_identity,
        )
        admitted = result["admitted_series_by_basin"][basin_id]
        if not np.array_equal(admitted.date, canonical.date):
            raise HydrographRunnerError(
                f"trial {trial_id!r} basin {basin_id!r}: admitted-series support dates disagree with the "
                "canonical package-observed series' support dates -- basin/date identity contradiction"
            )
        issue_dates = pd.DatetimeIndex(admitted.date)
        target_valid_dates = compute_target_valid_dates(issue_dates, contract["lead_hours"])
        admitted_mask = np.ones(len(issue_dates), dtype=bool)
        series_by_trial[trial_id] = BasinSeries(
            basin_id=basin_id,
            dates=target_valid_dates,
            issue_dates=issue_dates,
            obs_m3s=np.asarray(canonical.obs_m3s, dtype=np.float64),
            sim_m3s=np.asarray(admitted.sim_m3s, dtype=np.float64),
            admitted_mask=admitted_mask,
            area_km2=area_km2,
            n_admitted=len(issue_dates),
            n_total=len(issue_dates),
        )
    return series_by_trial


def _short_trial_label(trial_id: str) -> str:
    """Compact, human-readable form of a full trial id for on-figure legend
    text -- e.g. ``bayesian_proposal001`` from
    ``stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal001__...``.
    The full trial_id remains the receipt-authoritative identity everywhere
    else (manifest.json, ``order_to_incumbent_trial_id``); this is display
    text only."""
    parts = trial_id.split("__")
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[2]}"
    return trial_id


def _short_order_key(order_key: str) -> str:
    """Compact form of an ``"{arm}__proposal_order_{NN}"`` manifest key for
    on-figure legend text, e.g. ``bayesian@1``."""
    if "__proposal_order_" in order_key:
        arm, order = order_key.split("__proposal_order_")
        return f"{arm}@{int(order)}"
    return order_key


def candidate_labels_from_manifest(manifest: Mapping) -> dict:
    """Builds ``{trial_id: legend_label}`` for exactly the deduplicated
    ``unique_incumbent_trial_ids`` in an incumbent-order manifest (see
    :func:`~.stage1_rd1_c4_f_hydrograph_supplement.build_incumbent_order_mapping`),
    where the label lists every ``(arm, proposal_order)`` key that resolved
    to that trial -- so a reader can see the complete order-to-incumbent
    mapping directly on the figure legend, even though each unique trial is
    drawn only once. Labels use compact display forms (see
    :func:`_short_trial_label` / :func:`_short_order_key`) so the legend
    stays readable; the full trial_id remains the receipt-authoritative
    identity in the manifest this label was derived from."""
    keys_by_trial: dict = {}
    for order_key, trial_id in manifest["order_to_incumbent_trial_id"].items():
        keys_by_trial.setdefault(trial_id, []).append(order_key)
    return {
        trial_id: f"{_short_trial_label(trial_id)} (" + ", ".join(_short_order_key(k) for k in sorted(keys)) + ")"
        for trial_id, keys in keys_by_trial.items()
    }


def render_basin_incumbent_panel(
    *,
    basin_id: str,
    manifest: Mapping,
    targets_by_trial_id: Mapping[str, AuthenticatedTrialTarget],
    package_root,
    contract: dict,
    out_path,
) -> PeakWindow:
    """Extracts, computes the canonical peak window for, and renders one
    same-panel N-candidate overlay figure for ``basin_id`` -- one predicted
    line per deduplicated incumbent trial in ``manifest``, plus the single
    shared candidate-independent observed line. Returns the
    :class:`~.stage1_rd1_c4_f_hydrograph_supplement.PeakWindow` used, so the
    caller can record it in a receipt/manifest."""
    trial_ids = manifest["unique_incumbent_trial_ids"]
    series_by_trial = extract_basin_candidate_series(
        basin_id=basin_id,
        trial_ids=trial_ids,
        targets_by_trial_id=targets_by_trial_id,
        package_root=package_root,
        contract=contract,
    )
    reference = series_by_trial[trial_ids[0]]
    peak_window = select_observed_peak_window(
        reference.dates, reference.obs_m3s, reference.admitted_mask, basin_id=basin_id
    )
    window = peak_window_to_event_window(peak_window)
    scale = derive_comparison_scale(list(series_by_trial.values()), window=window)
    labels = candidate_labels_from_manifest(manifest)
    render_multi_candidate_basin_panel(
        series_by_trial,
        window=window,
        candidate_labels=labels,
        out_path=out_path,
        candidate_order=trial_ids,
        scale=scale,
        title_prefix="RD1-C4-F descriptive run-progress example (NOT winner/promotion evidence)",
    )
    return peak_window


def _peak_window_to_receipt_dict(peak_window: PeakWindow) -> dict:
    return {
        "basin_id": peak_window.basin_id,
        "peak_time": str(peak_window.peak_time),
        "peak_value": float(peak_window.peak_value),
        "window_start": str(peak_window.window_start),
        "window_end": str(peak_window.window_end),
        "requested_half_window_hours": peak_window.requested_half_window_hours,
        "clipped_left": bool(peak_window.clipped_left),
        "clipped_right": bool(peak_window.clipped_right),
        "actual_window_hours": float(peak_window.actual_window_hours),
    }


def produce_rd1_c4_f_hydrograph_supplement(
    *,
    trial_list_path,
    contract_path,
    package_root,
    selection_manifest_path,
    configuration_table_path,
    out_dir,
    search_arms: Sequence[str] = DEFAULT_SEARCH_ARMS,
    proposal_orders: Sequence[int] = DEFAULT_PROPOSAL_ORDERS,
) -> dict:
    """Produces the complete RD1-C4-F hydrograph supplement: one same-panel
    incumbent-overlay figure per selected basin, plus a hash-verified
    manifest recording the incumbent-order mapping, the peak window used per
    basin, and every produced figure's SHA-256. This is a descriptive
    evidence-rendering product only -- it makes no classifier, winner,
    promotion, or tolerance decision, and does not close RD1-C4.

    Raises :class:`HydrographRunnerError` if ``out_dir`` already exists and
    is non-empty (never silently overwrites prior evidence)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise HydrographRunnerError(f"out_dir {out_dir} already exists and is non-empty; refusing to overwrite")
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    contract = load_fixed_support_contract(contract_path)

    roster = authenticate_trial_roster(trial_list_path=Path(trial_list_path), contract=contract)
    targets_by_trial_id = {target.trial_id: target for target in roster}

    selection_path = Path(selection_manifest_path)
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_basins = selection_payload["selected_basins"]
    validate_selected_basins_canonical(selected_basins)

    config_table = pd.read_csv(configuration_table_path, dtype={"trial_id": str})
    incumbent_manifest = build_incumbent_order_mapping(config_table, search_arms, proposal_orders)

    unknown_trials = sorted(set(incumbent_manifest["unique_incumbent_trial_ids"]) - set(targets_by_trial_id))
    if unknown_trials:
        raise HydrographRunnerError(
            f"incumbent trial id(s) {unknown_trials} are not part of the authenticated roster "
            f"{roster.trial_ids()}"
        )

    figures: dict = {}
    peak_windows: dict = {}
    for percentile_key, basin_id in selected_basins.items():
        out_path = figures_dir / f"hydrograph_p{percentile_key}_{basin_id}.png"
        peak_window = render_basin_incumbent_panel(
            basin_id=basin_id,
            manifest=incumbent_manifest,
            targets_by_trial_id=targets_by_trial_id,
            package_root=package_root,
            contract=contract,
            out_path=out_path,
        )
        figures[percentile_key] = {
            "basin_id": basin_id,
            "path": str(out_path),
            "sha256": _sha256_path(out_path),
        }
        peak_windows[percentile_key] = _peak_window_to_receipt_dict(peak_window)

    manifest_payload = {
        "schema_name": "rd1_c4_f_hydrograph_supplement_manifest",
        "schema_version": 1,
        "label": "RD1-C4-F descriptive hydrograph supplement -- run-progress examples only, "
        "NOT winner/promotion/classifier/tolerance evidence, and does not close RD1-C4",
        "trial_list_path": str(trial_list_path),
        "trial_list_sha256": roster.trial_list_sha256,
        "contract_path": str(contract_path),
        "support_contract_sha256": roster.support_contract_sha256,
        "selection_manifest_path": str(selection_path),
        "configuration_table_path": str(configuration_table_path),
        "selected_basins": selected_basins,
        "search_arms": list(search_arms),
        "proposal_orders": list(proposal_orders),
        "incumbent_order_manifest": incumbent_manifest,
        "peak_windows_by_percentile": peak_windows,
        "figures_by_percentile": figures,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    return manifest_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-list-path", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--selection-manifest-path", required=True)
    parser.add_argument("--configuration-table-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--search-arms", nargs="+", default=list(DEFAULT_SEARCH_ARMS))
    parser.add_argument("--proposal-orders", nargs="+", type=int, default=list(DEFAULT_PROPOSAL_ORDERS))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = produce_rd1_c4_f_hydrograph_supplement(
        trial_list_path=args.trial_list_path,
        contract_path=args.contract_path,
        package_root=args.package_root,
        selection_manifest_path=args.selection_manifest_path,
        configuration_table_path=args.configuration_table_path,
        out_dir=args.out_dir,
        search_arms=tuple(args.search_arms),
        proposal_orders=tuple(args.proposal_orders),
    )
    print(json.dumps({"out_dir": args.out_dir, "n_figures": len(manifest["figures_by_percentile"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
