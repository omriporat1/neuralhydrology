"""RD1-C4-F hydrograph supplement -- deterministic, reusable core helpers.

This module contains the small amount of new deterministic logic needed for
the RD1-C4-F hydrograph supplement that is not already covered by an
existing maintained seam:

- resolving which trial is the "best-so-far incumbent" for a given
  search arm at a given proposal order, from the already-qualified
  RD1-C2 roster's ``cumulative_best``/``is_new_incumbent`` columns
  (:func:`incumbent_trial_at_proposal_order`);
- deduplicating a requested list of (arm, proposal_order) incumbent
  lookups into the unique set of trial_ids that actually need series
  extraction, while preserving the complete order-to-incumbent mapping
  (:func:`build_incumbent_order_mapping`);
- the canonical observed-peak selection rule (global max, earliest-time
  tie-break) and the fixed 72-hour window with an explicit, deterministic
  edge rule (:func:`select_observed_peak_window`).

Everything else (result-pickle loading, raw-space unit conversion, area
self-derivation, NSE/KGE/etc. metrics, and the actual multi-candidate
overlay rendering) reuses the already-qualified
:mod:`src.baseline.hydrograph_rendering` /
:mod:`src.baseline.nh_raw_space_evaluation` /
:mod:`src.baseline.rd1_c4_trial_authentication` machinery -- no parallel
implementation of any of that is created here.

No classifier, winner, promotion, or tolerance decision is made here. These
are explicitly descriptive run-progress examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .stage1_rd1_c4_f_synthesis import validate_canonical_basin_ids

__all__ = [
    "PeakWindowError",
    "PeakWindow",
    "select_observed_peak_window",
    "IncumbentResolutionError",
    "incumbent_trial_at_proposal_order",
    "build_incumbent_order_mapping",
]

DEFAULT_WINDOW_HOURS = 72
DEFAULT_HALF_WINDOW_HOURS = DEFAULT_WINDOW_HOURS // 2


class PeakWindowError(ValueError):
    """Raised for a malformed/empty observed series when selecting a peak window."""


@dataclass(frozen=True)
class PeakWindow:
    """A single canonical observed-peak window for one basin.

    ``window_start``/``window_end`` are derived from ``peak_time`` by the
    deterministic edge rule in :func:`select_observed_peak_window`: a
    symmetric ``half_window_hours`` on each side of the peak, clipped (never
    shifted) to the series' own admitted date range if that would run past
    either end. ``clipped_left``/``clipped_right`` and
    ``actual_window_hours`` record exactly when/how much clipping occurred
    so a reader never mistakes a clipped window for a full 72-hour one.
    """

    basin_id: str
    peak_time: pd.Timestamp
    peak_value: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    requested_half_window_hours: int
    clipped_left: bool
    clipped_right: bool
    actual_window_hours: float


def select_observed_peak_window(
    dates: pd.DatetimeIndex,
    obs_m3s: np.ndarray,
    admitted_mask: np.ndarray,
    *,
    basin_id: str,
    half_window_hours: int = DEFAULT_HALF_WINDOW_HOURS,
) -> PeakWindow:
    """Canonical observed-peak selection + fixed-window rule.

    Peak selection: the globally largest *admitted* observed value across
    the whole supplied series; ties broken by the earliest ``dates`` value
    among the tied maxima (the "canonical observed-peak earliest-tie
    rule").

    Window rule: ``[peak_time - half_window_hours, peak_time +
    half_window_hours]`` (a fixed ``2 * half_window_hours``-hour window,
    72 hours by default), clipped -- never shifted -- to
    ``[dates.min(), dates.max()]`` if the symmetric window would extend
    past either end of the available series. This is an explicit,
    deterministic edge rule: a clipped window is always shorter than the
    requested width, and the clipping is recorded rather than silently
    compensated by shifting the window and changing which pre/post-peak
    hours it covers.
    """
    dates = pd.DatetimeIndex(dates)
    obs_m3s = np.asarray(obs_m3s, dtype=np.float64)
    admitted_mask = np.asarray(admitted_mask, dtype=bool)
    if len(dates) == 0 or len(dates) != len(obs_m3s) or len(dates) != len(admitted_mask):
        raise PeakWindowError(
            f"basin {basin_id!r}: dates/obs_m3s/admitted_mask must be equal-length and non-empty "
            f"(got {len(dates)}/{len(obs_m3s)}/{len(admitted_mask)})"
        )
    candidate_mask = admitted_mask & np.isfinite(obs_m3s)
    if not candidate_mask.any():
        raise PeakWindowError(f"basin {basin_id!r}: no admitted, finite observed samples")

    candidate_values = obs_m3s[candidate_mask]
    candidate_dates = dates[candidate_mask]
    peak_value = float(np.max(candidate_values))
    tied = candidate_dates[candidate_values == peak_value]
    peak_time = pd.Timestamp(tied.min())  # earliest-tie rule

    series_start, series_end = dates.min(), dates.max()
    requested_start = peak_time - pd.Timedelta(hours=half_window_hours)
    requested_end = peak_time + pd.Timedelta(hours=half_window_hours)
    window_start = max(requested_start, series_start)
    window_end = min(requested_end, series_end)
    clipped_left = window_start > requested_start
    clipped_right = window_end < requested_end
    actual_window_hours = (window_end - window_start).total_seconds() / 3600.0

    return PeakWindow(
        basin_id=basin_id,
        peak_time=peak_time,
        peak_value=peak_value,
        window_start=window_start,
        window_end=window_end,
        requested_half_window_hours=half_window_hours,
        clipped_left=clipped_left,
        clipped_right=clipped_right,
        actual_window_hours=actual_window_hours,
    )


class IncumbentResolutionError(ValueError):
    """Raised when the incumbent trial for a requested (arm, proposal_order)
    cannot be unambiguously resolved from the roster/config table."""


def incumbent_trial_at_proposal_order(
    config_table: pd.DataFrame, search_arm: str, proposal_order: int
) -> str:
    """Return the ``trial_id`` that is the best-so-far incumbent for
    ``search_arm`` as of ``proposal_order`` (inclusive), using the already-
    qualified RD1-C2 roster's own ``is_new_incumbent`` flags (joined into
    ``config_table`` by :func:`~.stage1_rd1_c4_f_render.main` /
    ``build_configuration_table``): the incumbent as of order N is the
    trial with ``is_new_incumbent == True`` at the largest
    ``proposal_order <= N`` within that arm. No new incumbent-tracking
    logic is computed here -- this only *looks up* the roster's existing,
    already-qualified running-best flags.
    """
    required_cols = {"trial_id", "search_arm", "proposal_order", "is_new_incumbent"}
    missing_cols = required_cols - set(config_table.columns)
    if missing_cols:
        raise IncumbentResolutionError(f"config_table missing required column(s): {sorted(missing_cols)}")

    arm_rows = config_table[
        (config_table["search_arm"] == search_arm) & (config_table["proposal_order"] <= proposal_order)
    ]
    incumbent_rows = arm_rows[arm_rows["is_new_incumbent"] == True]  # noqa: E712
    if incumbent_rows.empty:
        raise IncumbentResolutionError(
            f"no incumbent found for search_arm={search_arm!r} at proposal_order<={proposal_order}"
        )
    best_row = incumbent_rows.loc[incumbent_rows["proposal_order"].idxmax()]
    return str(best_row["trial_id"])


def build_incumbent_order_mapping(
    config_table: pd.DataFrame, search_arms: Sequence[str], proposal_orders: Sequence[int]
) -> dict:
    """Resolve every ``(search_arm, proposal_order)`` pair to its incumbent
    ``trial_id``, and return a compact manifest with both the complete
    order-to-incumbent mapping and the deduplicated unique set of trial_ids
    that actually need series extraction/rendering.
    """
    order_to_incumbent = {}
    for arm in search_arms:
        for order in proposal_orders:
            order_to_incumbent[f"{arm}__proposal_order_{order:02d}"] = incumbent_trial_at_proposal_order(
                config_table, arm, order
            )
    unique_trial_ids = sorted(set(order_to_incumbent.values()))
    return {
        "search_arms": list(search_arms),
        "proposal_orders": list(proposal_orders),
        "order_to_incumbent_trial_id": order_to_incumbent,
        "unique_incumbent_trial_ids": unique_trial_ids,
        "n_lookups": len(order_to_incumbent),
        "n_unique_trials": len(unique_trial_ids),
    }


def validate_selected_basins_canonical(basin_ids: Mapping[str, str]) -> None:
    """Validate the hydrograph supplement's selected basin_id values (as
    loaded from ``hydrograph_basin_selection_manifest.json``) are canonical
    fixed-width strings -- a thin, explicit boundary check re-using
    :func:`~.stage1_rd1_c4_f_synthesis.validate_canonical_basin_ids` at the
    entry point of the hydrograph supplement."""
    validate_canonical_basin_ids(basin_ids.values(), context="hydrograph supplement selected basins")
