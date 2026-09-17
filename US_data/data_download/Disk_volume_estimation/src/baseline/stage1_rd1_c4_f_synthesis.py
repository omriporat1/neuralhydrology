"""RD1-C4-F descriptive scientific-synthesis helpers.

Deterministic, reusable pieces for the RD1-C4-F review packet: joining the
frozen 24-configuration six-axis roster to the RD1-C4-E official results
bundle, and the deterministic basin-selection rule used for the hydrograph
supplement. This module performs no new scientific computation -- it only
assembles/summarizes columns already present in the qualified RD1-C4-E
evidence bundle (``per_basin_metrics.csv`` / ``q98_diagnostics.csv`` /
``basin_distribution_summary.csv``) and the frozen configuration roster.

No classifier, winner, promotion, or tolerance decision is made here.
"""
from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Normalized [0,1] geometry coordinates (RD1-C2 canonical_configuration());
# used only where a normalized axis is explicitly wanted.
SIX_AXES = (
    "coord_learning_rate",
    "coord_hidden_size",
    "coord_embedding_dropout",
    "coord_output_dropout",
    "coord_batch_size",
    "coord_seq_length",
)

# Raw (physical-unit) six-axis names, recovered from the roster's
# ``canonical_coordinate_key`` JSON column -- these are what a reader
# actually wants for an interpretable configuration-space figure/table.
RAW_SIX_AXES = (
    "raw_learning_rate",
    "raw_hidden_size",
    "raw_embedding_dropout",
    "raw_output_dropout",
    "raw_batch_size",
    "raw_seq_length",
)

_RAW_AXIS_TO_KEY = {
    "raw_learning_rate": "learning_rate",
    "raw_hidden_size": "hidden_size",
    "raw_embedding_dropout": "embedding_dropout",
    "raw_output_dropout": "output_dropout",
    "raw_batch_size": "batch_size",
    "raw_seq_length": "seq_length",
}


def add_raw_axis_columns(roster: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``roster`` with raw-unit six-axis columns parsed out
    of its ``canonical_coordinate_key`` JSON column."""
    parsed = roster["canonical_coordinate_key"].apply(json.loads)
    out = roster.copy()
    for raw_col, key in _RAW_AXIS_TO_KEY.items():
        out[raw_col] = parsed.apply(lambda d, k=key: float(d[k]))
    return out

Q98_DIAGNOSTIC_METRICS = (
    "q98_normalized_rmse",
    "relative_volume_bias",
    "observed_peak_time_magnitude_error",
    "high_flow_nse",
)


class RosterMismatchError(ValueError):
    """Raised when the frozen roster does not exactly match the results bundle."""


class BasinIdFormatError(ValueError):
    """Raised when a basin_id is not a canonical fixed-width 8-character USGS identifier."""


# USGS site identifiers used throughout Flash-NH are fixed-width, zero-padded,
# all-digit strings (e.g. "01464000"). They must never be carried as numeric
# values: leading zeros are scientifically meaningless as a *number* but are
# a required part of the canonical *identifier*, and numeric coercion (e.g.
# pandas' default pd.read_csv dtype inference) silently corrupts them.
CANONICAL_BASIN_ID_LENGTH = 8


def validate_canonical_basin_ids(basin_ids: Iterable[object], context: str = "basin_id") -> None:
    """Raise ``BasinIdFormatError`` unless every value is a canonical fixed-width
    8-character, all-digit, zero-padded-as-needed USGS basin identifier string.

    This is a format check only -- it does not touch which records are
    selected, any metric, or any percentile computation.
    """
    bad = sorted(
        {
            str(b)
            for b in basin_ids
            if not (isinstance(b, str) and len(b) == CANONICAL_BASIN_ID_LENGTH and b.isdigit())
        }
    )
    if bad:
        raise BasinIdFormatError(
            f"{context}: found non-canonical basin_id value(s) (expected "
            f"{CANONICAL_BASIN_ID_LENGTH}-character zero-padded digit strings): {bad[:10]}"
        )


def validate_roster_against_trial_ids(roster: pd.DataFrame, trial_ids: Iterable[str]) -> None:
    """Raise if ``roster['proposal_id']`` is not exactly the given trial-id set."""
    roster_ids = set(roster["proposal_id"])
    expected_ids = set(trial_ids)
    missing = expected_ids - roster_ids
    extra = roster_ids - expected_ids
    if missing or extra:
        raise RosterMismatchError(
            f"roster does not match results-bundle trial_ids: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    if roster["proposal_id"].duplicated().any():
        raise RosterMismatchError("roster contains duplicate proposal_id values")


def build_configuration_table(
    per_basin_metrics: pd.DataFrame,
    q98_diagnostics: pd.DataFrame,
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """One row per configuration (24 rows): arm, proposal order, six axes,
    official objective, and descriptive NSE/KGE/Q98 summaries.

    ``per_basin_metrics``/``q98_diagnostics`` are the RD1-C4-E per-cell
    results (one row per trial x basin); this aggregates to one row per
    trial_id (== configuration) and joins the frozen six-axis roster values.
    """
    nse_kge = per_basin_metrics.groupby("trial_id").agg(
        search_arm=("search_arm", "first"),
        configuration_id=("configuration_id", "first"),
        proposal_order=("proposal_order", "first"),
        official_objective=("official_objective", "first"),
        n_basins=("basin_id", "count"),
        median_nse=("nse", "median"),
        nse_q25=("nse", lambda s: s.quantile(0.25)),
        nse_q75=("nse", lambda s: s.quantile(0.75)),
        median_kge=("kge", "median"),
        kge_q25=("kge", lambda s: s.quantile(0.25)),
        kge_q75=("kge", lambda s: s.quantile(0.75)),
    )

    q98_agg = q98_diagnostics.groupby("trial_id").agg(
        median_q98_normalized_rmse=("q98_normalized_rmse", "median"),
        median_relative_volume_bias=("relative_volume_bias", "median"),
        median_observed_peak_time_magnitude_error=("observed_peak_time_magnitude_error", "median"),
        median_high_flow_nse=("high_flow_nse", "median"),
    )

    table = nse_kge.join(q98_agg, how="left")

    roster_with_raw = add_raw_axis_columns(roster)
    roster_indexed = roster_with_raw.set_index("proposal_id")
    axis_columns = roster_indexed[list(SIX_AXES) + list(RAW_SIX_AXES)]
    table = table.join(axis_columns, how="left")

    all_axis_cols = list(SIX_AXES) + list(RAW_SIX_AXES)
    if table[all_axis_cols].isna().any().any():
        missing_ids = table.index[table[all_axis_cols].isna().any(axis=1)].tolist()
        raise RosterMismatchError(f"roster join left missing six-axis values for: {missing_ids}")

    table = table.reset_index().rename(columns={"index": "trial_id"})
    table = table.sort_values(["search_arm", "proposal_order"]).reset_index(drop=True)
    return table


def basin_arm_medians(per_basin_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-basin median of ``metric`` across the 12 configurations of each arm,
    plus the Bayesian-minus-random-control difference of those medians."""
    pivot = per_basin_metrics.pivot_table(
        index="basin_id", columns="search_arm", values=metric, aggfunc="median"
    )
    for arm in ("bayesian", "random_control"):
        if arm not in pivot.columns:
            raise RosterMismatchError(f"basin_arm_medians: arm '{arm}' missing from {metric} pivot")
    pivot = pivot.rename(columns={"bayesian": "bayesian_median", "random_control": "random_control_median"})
    pivot["diff_bayesian_minus_random"] = pivot["bayesian_median"] - pivot["random_control_median"]
    return pivot[["bayesian_median", "random_control_median", "diff_bayesian_minus_random"]]


def sign_counts(diff: pd.Series) -> dict[str, int]:
    """Counts of strictly positive / strictly negative / exactly-zero entries."""
    return {
        "n_positive": int((diff > 0).sum()),
        "n_negative": int((diff < 0).sum()),
        "n_zero": int((diff == 0).sum()),
    }


def select_percentile_basins(
    diff: pd.Series, percentiles: Sequence[float] = (10, 50, 90)
) -> Mapping[float, str]:
    """Deterministically select, for each target percentile, the basin whose
    value is closest to the empirical percentile of ``diff``.

    Ties (equal absolute distance to the target) are broken by
    lexicographically smallest basin_id (``diff.index`` values, coerced to
    ``str`` for comparison). Fully deterministic given ``diff``.
    """
    if diff.empty:
        raise ValueError("select_percentile_basins: diff series is empty")
    values = diff.to_numpy(dtype=float)
    selected: dict[float, str] = {}
    for p in percentiles:
        target = np.percentile(values, p)
        abs_dist = (diff - target).abs()
        min_dist = abs_dist.min()
        candidate_ids = sorted(str(idx) for idx in diff.index[abs_dist == min_dist])
        selected[p] = candidate_ids[0]
    return selected
