"""LR-A learning-rate range-characterization campaign
(``lr_range_seedA_25k_v001``, see docs/decision_log.md's LR-A design-freeze
entry): Sections 10-12's comparison/trajectory/cadence-sensitivity utilities.

All three sections consume, and never recompute, the per-epoch diagnostic
payloads :func:`src.baseline.pilot_diagnostic_eval.evaluate_diagnostic_checkpoint`
/ :func:`~src.baseline.pilot_diagnostic_eval.evaluate_all_diagnostic_checkpoints`
already produce for each candidate -- median/q25/q75 raw-space NSE come
straight from ``raw_space_metrics``'s own per-basin numbers, and
``evaluation_role``/``authoritative`` are copied verbatim from that payload's
own fields, never re-derived here. This module adds no new metric math and
no new inference path; it only reshapes already-trusted numbers into three
purely descriptive views:

  Section 10 -- ``build_n_vs_one_comparison``: one row per (candidate,
  epoch) (:func:`summarize_candidate_epoch`) and one row per (challenger,
  reference, epoch) paired NSE difference
  (:func:`paired_diff_challenger_vs_reference`), for exactly one reference
  against N challengers. Requires EXACT matched-basin identity between a
  challenger and the reference at every compared epoch -- fails loudly
  (never silently drops/reorders/intersects) on a missing basin, a
  duplicate basin_id, a population-size mismatch, a reference/challenger
  identity mixup, or mixed epochs.

  Section 11 -- ``derive_trajectory_summary``: deterministic descriptive
  summary of one candidate's own epoch trajectory (best observed
  median-NSE checkpoint, epoch-6 median NSE, median/range of epoch medians
  across a late window, late-window trajectory direction). Never computes a
  composite winner score, never auto-ranks, never auto-promotes a
  candidate.

  Section 12 -- ``cadence_sensitivity_view``: the same per-candidate epoch
  summaries, re-sliced under caller-named epoch subsets (e.g. all six
  epochs vs. every-other-epoch vs. epoch 3/6 only) -- a simple derived
  table, not a new statistical framework.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "CheckpointComparisonError",
    "summarize_candidate_epoch",
    "paired_diff_challenger_vs_reference",
    "build_n_vs_one_comparison",
    "derive_trajectory_summary",
    "cadence_sensitivity_view",
]


class CheckpointComparisonError(Exception):
    """Raised for a structural contract violation in this comparison/
    trajectory/cadence utility (missing/duplicate/mismatched basins, wrong
    reference identity, mixed epochs, an unknown candidate_id) -- never for
    an ordinary poor-skill outcome."""


def _per_basin_nse_map(per_basin: Sequence[Mapping], *, candidate_id: str, epoch: int) -> "dict[str, float]":
    nse_by_basin: "dict[str, float]" = {}
    for row in per_basin:
        basin_id = row["basin_id"]
        if basin_id in nse_by_basin:
            raise CheckpointComparisonError(
                f"duplicate basin_id {basin_id!r} in {candidate_id!r}'s epoch {epoch} per-basin records"
            )
        nse_by_basin[basin_id] = row["nse"]
    return nse_by_basin


def summarize_candidate_epoch(
    diagnostic_payload: Mapping,
    *,
    candidate_id: str,
    mean_training_loss: "Optional[float]" = None,
    cumulative_updates: "Optional[int]" = None,
    secondary_diagnostics: "Optional[Mapping]" = None,
) -> dict:
    """One row of Section 10's per-(candidate, epoch) table, built strictly
    from an existing
    :func:`src.baseline.pilot_diagnostic_eval.evaluate_diagnostic_checkpoint`
    payload plus optional externally-supplied training-loss/update-count
    metadata (e.g. from ``scripts/extract_pilot_tb_losses.py`` /
    ``pilot_run_evidence.json`` -- never sourced from inside this function)."""
    epoch = diagnostic_payload["epoch"]
    per_basin = diagnostic_payload["raw_space_metrics"]["per_basin"]
    nse_by_basin = _per_basin_nse_map(per_basin, candidate_id=candidate_id, epoch=epoch)
    nse_values = np.array(list(nse_by_basin.values()), dtype=np.float64)
    finite = nse_values[np.isfinite(nse_values)]

    return {
        "candidate_id": candidate_id,
        "epoch": epoch,
        "n_basins": int(finite.size),
        "median_nse": float(np.median(finite)) if finite.size else float("nan"),
        "q25_nse": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "q75_nse": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "frac_nse_gt_0": float(np.mean(finite > 0.0)) if finite.size else float("nan"),
        "evaluation_role": diagnostic_payload["evaluation_role"],
        "authoritative": diagnostic_payload["authoritative"],
        "mean_training_loss": mean_training_loss,
        "cumulative_updates": cumulative_updates,
        "secondary_diagnostics": dict(secondary_diagnostics) if secondary_diagnostics else None,
    }


def paired_diff_challenger_vs_reference(
    challenger_payload: Mapping,
    reference_payload: Mapping,
    *,
    challenger_id: str,
    reference_id: str,
) -> dict:
    """One row of Section 10's per-(challenger, reference, epoch) paired-diff
    table. Requires the two payloads' epochs to match and their per-basin
    populations to be EXACTLY identical (same basin_id set, no duplicates on
    either side) -- fails loudly otherwise, never silently intersects or
    reorders."""
    if challenger_id == reference_id:
        raise CheckpointComparisonError(
            f"challenger_id and reference_id must differ, both were {challenger_id!r}"
        )
    challenger_epoch = challenger_payload["epoch"]
    reference_epoch = reference_payload["epoch"]
    if challenger_epoch != reference_epoch:
        raise CheckpointComparisonError(
            f"mixed epochs: challenger {challenger_id!r} epoch={challenger_epoch} vs "
            f"reference {reference_id!r} epoch={reference_epoch}"
        )
    epoch = challenger_epoch

    challenger_nse = _per_basin_nse_map(
        challenger_payload["raw_space_metrics"]["per_basin"], candidate_id=challenger_id, epoch=epoch
    )
    reference_nse = _per_basin_nse_map(
        reference_payload["raw_space_metrics"]["per_basin"], candidate_id=reference_id, epoch=epoch
    )

    challenger_basins = set(challenger_nse)
    reference_basins = set(reference_nse)
    if challenger_basins != reference_basins:
        missing_from_challenger = sorted(reference_basins - challenger_basins)
        missing_from_reference = sorted(challenger_basins - reference_basins)
        raise CheckpointComparisonError(
            f"matched-basin-identity violation at epoch {epoch} between challenger "
            f"{challenger_id!r} and reference {reference_id!r}: "
            f"missing_from_challenger={missing_from_challenger} "
            f"missing_from_reference={missing_from_reference}"
        )

    matched_basin_ids = sorted(challenger_basins)
    diffs = np.array(
        [challenger_nse[b] - reference_nse[b] for b in matched_basin_ids], dtype=np.float64
    )
    finite = diffs[np.isfinite(diffs)]

    return {
        "challenger_id": challenger_id,
        "reference_id": reference_id,
        "epoch": epoch,
        "matched_basin_count": len(matched_basin_ids),
        "median_diff_nse": float(np.median(finite)) if finite.size else float("nan"),
        "q25_diff_nse": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "q75_diff_nse": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "frac_challenger_better": float(np.mean(finite > 0.0)) if finite.size else float("nan"),
        "frac_reference_better": float(np.mean(finite < 0.0)) if finite.size else float("nan"),
        "frac_tied": float(np.mean(finite == 0.0)) if finite.size else float("nan"),
    }


def build_n_vs_one_comparison(
    diagnostic_payloads_by_candidate: "Mapping[str, Sequence[Mapping]]",
    *,
    reference_id: str,
    challenger_ids: Sequence[str],
    epochs: Sequence[int],
    loss_by_candidate_epoch: "Optional[Mapping[tuple, float]]" = None,
    updates_by_candidate_epoch: "Optional[Mapping[tuple, int]]" = None,
    secondary_diagnostics_by_candidate_epoch: "Optional[Mapping[tuple, Mapping]]" = None,
) -> dict:
    """Section 10's "all-N-vs-1 comparison model": builds the full per-
    (candidate, epoch) summary table and per-(challenger, reference, epoch)
    paired-diff table for exactly one reference against N challengers, over
    an explicit, caller-supplied ``epochs`` list.

    ``diagnostic_payloads_by_candidate`` maps candidate_id -> a list of
    :func:`src.baseline.pilot_diagnostic_eval.evaluate_diagnostic_checkpoint`-
    shaped payloads (e.g. the direct return value of
    :func:`~src.baseline.pilot_diagnostic_eval.evaluate_all_diagnostic_checkpoints`
    for that candidate) -- never opens an NH result file itself.

    Fails loudly (never silently drops a candidate or an epoch) if:
      - ``reference_id`` is also listed in ``challenger_ids``, or
        ``challenger_ids`` has a duplicate;
      - ``reference_id`` or any ``challenger_ids`` entry is missing from
        ``diagnostic_payloads_by_candidate``;
      - any candidate is missing a payload for one of the requested
        ``epochs``, or has more than one payload for the same epoch;
      - matched-basin identity fails for any (challenger, reference, epoch)
        pair (see :func:`paired_diff_challenger_vs_reference`).
    """
    if reference_id in challenger_ids:
        raise CheckpointComparisonError(
            f"reference_id {reference_id!r} must not also appear in challenger_ids {list(challenger_ids)}"
        )
    if len(set(challenger_ids)) != len(challenger_ids):
        raise CheckpointComparisonError(f"duplicate challenger_ids: {list(challenger_ids)}")
    if not epochs:
        raise CheckpointComparisonError("epochs must be non-empty")

    all_candidate_ids = [reference_id] + list(challenger_ids)
    missing_candidates = sorted(set(all_candidate_ids) - set(diagnostic_payloads_by_candidate))
    if missing_candidates:
        raise CheckpointComparisonError(
            f"diagnostic_payloads_by_candidate is missing required candidate_id(s): {missing_candidates}"
        )

    payload_by_candidate_epoch: "dict[tuple, Mapping]" = {}
    for candidate_id in all_candidate_ids:
        payloads = diagnostic_payloads_by_candidate[candidate_id]
        seen_epochs: "dict[int, Mapping]" = {}
        for payload in payloads:
            payload_epoch = payload["epoch"]
            if payload_epoch in seen_epochs:
                raise CheckpointComparisonError(
                    f"candidate {candidate_id!r} has more than one diagnostic payload for epoch "
                    f"{payload_epoch}"
                )
            seen_epochs[payload_epoch] = payload
        missing_epochs = sorted(set(epochs) - set(seen_epochs))
        if missing_epochs:
            raise CheckpointComparisonError(
                f"candidate {candidate_id!r} is missing diagnostic payload(s) for requested "
                f"epoch(s): {missing_epochs}"
            )
        for epoch in epochs:
            payload_by_candidate_epoch[(candidate_id, epoch)] = seen_epochs[epoch]

    loss_by_candidate_epoch = loss_by_candidate_epoch or {}
    updates_by_candidate_epoch = updates_by_candidate_epoch or {}
    secondary_diagnostics_by_candidate_epoch = secondary_diagnostics_by_candidate_epoch or {}

    candidate_epoch_summaries = [
        summarize_candidate_epoch(
            payload_by_candidate_epoch[(candidate_id, epoch)],
            candidate_id=candidate_id,
            mean_training_loss=loss_by_candidate_epoch.get((candidate_id, epoch)),
            cumulative_updates=updates_by_candidate_epoch.get((candidate_id, epoch)),
            secondary_diagnostics=secondary_diagnostics_by_candidate_epoch.get((candidate_id, epoch)),
        )
        for candidate_id in all_candidate_ids
        for epoch in epochs
    ]

    paired_diffs = [
        paired_diff_challenger_vs_reference(
            payload_by_candidate_epoch[(challenger_id, epoch)],
            payload_by_candidate_epoch[(reference_id, epoch)],
            challenger_id=challenger_id,
            reference_id=reference_id,
        )
        for challenger_id in challenger_ids
        for epoch in epochs
    ]

    return {
        "reference_id": reference_id,
        "challenger_ids": list(challenger_ids),
        "epochs": list(epochs),
        "candidate_epoch_summaries": candidate_epoch_summaries,
        "paired_diffs": paired_diffs,
    }


def _rows_for_candidate(candidate_epoch_summaries: Sequence[Mapping], candidate_id: str) -> "list[Mapping]":
    rows = [row for row in candidate_epoch_summaries if row["candidate_id"] == candidate_id]
    if not rows:
        raise CheckpointComparisonError(
            f"no candidate_epoch_summaries rows found for candidate_id={candidate_id!r}"
        )
    return rows


def derive_trajectory_summary(
    candidate_epoch_summaries: Sequence[Mapping],
    *,
    candidate_id: str,
    late_window_epochs: Sequence[int] = (4, 5, 6),
    epoch_6: int = 6,
) -> dict:
    """Section 11's deterministic, non-ranking descriptive summary of ONE
    candidate's own epoch trajectory, built from :func:`summarize_candidate_epoch`
    rows (e.g. a subset of :func:`build_n_vs_one_comparison`'s
    ``candidate_epoch_summaries``). Never computes a composite winner score,
    never auto-ranks against other candidates, never auto-promotes."""
    rows = _rows_for_candidate(candidate_epoch_summaries, candidate_id)
    by_epoch = {row["epoch"]: row for row in rows}

    best_row = max(rows, key=lambda row: row["median_nse"])
    late_rows = [by_epoch[epoch] for epoch in late_window_epochs if epoch in by_epoch]
    late_medians = [row["median_nse"] for row in late_rows]
    epoch_6_row = by_epoch.get(epoch_6)

    if len(late_rows) < 2:
        late_trajectory_direction = "insufficient_data"
    else:
        ordered_late_rows = sorted(late_rows, key=lambda row: row["epoch"])
        first_median = ordered_late_rows[0]["median_nse"]
        last_median = ordered_late_rows[-1]["median_nse"]
        if last_median > first_median:
            late_trajectory_direction = "increasing"
        elif last_median < first_median:
            late_trajectory_direction = "decreasing"
        else:
            late_trajectory_direction = "flat"

    return {
        "candidate_id": candidate_id,
        "best_observed_median_nse_checkpoint_epoch": best_row["epoch"],
        "best_observed_median_nse": best_row["median_nse"],
        "epoch_6_median_nse": epoch_6_row["median_nse"] if epoch_6_row is not None else None,
        "late_window_epochs_requested": list(late_window_epochs),
        "late_window_epochs_available": [row["epoch"] for row in late_rows],
        "median_of_epoch_medians_late_window": (
            float(np.median(late_medians)) if late_medians else None
        ),
        "range_of_epoch_medians_late_window": (
            float(max(late_medians) - min(late_medians)) if late_medians else None
        ),
        "late_trajectory_direction": late_trajectory_direction,
    }


def cadence_sensitivity_view(
    candidate_epoch_summaries: Sequence[Mapping],
    *,
    candidate_id: str,
    cadences: Mapping[str, Sequence[int]],
) -> dict:
    """Section 12's cadence-sensitivity evidence: re-slices one candidate's
    own :func:`summarize_candidate_epoch` rows under each caller-named epoch
    subset in ``cadences`` (e.g. ``{"all_epochs": [1,2,3,4,5,6],
    "every_other_epoch": [2,4,6], "screening_cadence_only": [3,6]}``). A
    simple derived table over already-computed per-epoch medians -- no new
    statistical framework, no interpolation of a missing epoch."""
    rows = _rows_for_candidate(candidate_epoch_summaries, candidate_id)
    by_epoch = {row["epoch"]: row for row in rows}

    view = {}
    for cadence_name, cadence_epochs in cadences.items():
        available_rows = [by_epoch[epoch] for epoch in cadence_epochs if epoch in by_epoch]
        missing_epochs = sorted(set(cadence_epochs) - set(by_epoch))
        medians = [row["median_nse"] for row in available_rows]
        best_row = max(available_rows, key=lambda row: row["median_nse"]) if available_rows else None
        view[cadence_name] = {
            "epochs_requested": list(cadence_epochs),
            "epochs_available": [row["epoch"] for row in available_rows],
            "epochs_missing": missing_epochs,
            "best_observed_median_nse_checkpoint_epoch": (
                best_row["epoch"] if best_row is not None else None
            ),
            "best_observed_median_nse": best_row["median_nse"] if best_row is not None else None,
            "median_of_epoch_medians": float(np.median(medians)) if medians else None,
        }
    return {"candidate_id": candidate_id, "cadences": view}
