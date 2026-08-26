"""Analysis/derivation helpers for the Phase-B Sweep-v2 six-axis durable
review layer.

Strictly additive sibling of :mod:`sweep_v1_review_analysis` (Section H,
additive six-axis campaign foundation). The large majority of that module's
functions operate purely on generic trial-table columns (``workflow_status``,
``search_arm``, ``valid_result_order``, ``best_score``, generic axis
values/bounds/geometry passed as plain arguments) with no v1-literal axis
list or campaign identity baked in at all, so they are imported and reused
directly here, unchanged, per this task's established reuse-vs-sibling
convention (see e.g. :mod:`sweep_v2_six_axis_retry`'s module docstring for
the same distinction applied to the retry seam).

Four functions in v1's module DO hardcode the frozen five-axis field set
and/or v1's own frozen search-domain constant directly, so each gets a
genuine v2 sibling that adds the sixth axis (``seq_length``) and reads from
the six-axis ``SEARCH_DOMAIN_V2`` instead:

- :func:`derive_boundary_pressure_table_v2` (v1: ``derive_boundary_pressure_table``)
- :func:`categorical_occupancy_table_v2` (v1: ``categorical_occupancy_table``)
- :func:`most_pressured_continuous_axis_v2` (v1: ``most_pressured_continuous_axis``)
- :func:`top_configurations_table_v2` (v1: ``top_configurations_table``)

``seq_length`` boundary-nature assumption (explicit, conservative,
deliberately NOT baked into ``SEARCH_DOMAIN_V2`` in
:mod:`sweep_v2_six_axis_campaign`, to keep this interpretive call narrowly
scoped to review-analysis reporting only): both the lower (48h) and upper
(120h) ends of the ``q_uniform`` grid are treated as **natural** bounds, not
**expandable** ones, for boundary-pressure-tier reporting purposes only. No
scientific decision has been made about whether this compute-bounded,
explicitly user-mandated grid should ever be widened, so this module never
lets a v2 boundary-pressure row read as an actionable expansion-pressure
recommendation for ``seq_length`` -- it can only ever report as a
preference signal (see :func:`sweep_v1_review_analysis._interpretation_label`,
reused unchanged). This is a reporting-label default, not a search-space
change: it does not touch ``SEARCH_DOMAIN_V2``, the ``q_uniform`` contract,
or any config/identity/canonicalization code path. Flagged explicitly for
user confirmation in the task's final report.

Nothing here mutates a search domain, contacts W&B, biases a Bayesian
objective, or scores candidates for promotion -- identical scope discipline
to v1's own module.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.baseline.sweep_v1_review_analysis import (
    BOUNDARY_FRACTION,
    SENSITIVITY_BAND_FRACTIONS,
    TIER_ORDER,
    VALID_STATUS,
    _arm_near_stats,
    _interpretation_label,
    _round_or_none,
    boundary_band_sensitivity,
    boundary_pressure_evolution,
    boundary_pressure_tier,
    categorical_extreme_mask,
    checkpoint_operations_slice,
    checkpoint_slice,
    checkpoint_status_summary,
    common_n,
    cumulative_best_by_order,
    cumulative_gpu_hours_by_order,
    is_top_quartile,
    near_boundary_mask,
    neighborhood_support_evidence,
    proposal_drift_evidence,
    representative_trajectory_selection,
    top_quartile_threshold,
    valid_trials,
)
from src.baseline.sweep_v2_six_axis_campaign import SEARCH_DOMAIN_V2, SEQ_LENGTH_MAX, SEQ_LENGTH_MIN

__all__ = [
    "VALID_STATUS",
    "BOUNDARY_FRACTION",
    "SENSITIVITY_BAND_FRACTIONS",
    "TIER_ORDER",
    "valid_trials",
    "checkpoint_slice",
    "checkpoint_operations_slice",
    "common_n",
    "cumulative_best_by_order",
    "cumulative_gpu_hours_by_order",
    "top_quartile_threshold",
    "is_top_quartile",
    "near_boundary_mask",
    "categorical_extreme_mask",
    "proposal_drift_evidence",
    "neighborhood_support_evidence",
    "boundary_pressure_tier",
    "derive_boundary_pressure_table_v2",
    "categorical_occupancy_table_v2",
    "representative_trajectory_selection",
    "boundary_pressure_evolution",
    "boundary_band_sensitivity",
    "top_configurations_table_v2",
    "checkpoint_status_summary",
    "most_pressured_continuous_axis_v2",
]

# Six-axis extension of v1's ``_CONTINUOUS_AXES``/``_CATEGORICAL_AXES``: the
# three original continuous axes plus ``seq_length`` (linear geometry -- the
# q_uniform grid is evenly spaced in raw hours, not log-spaced); the two
# categorical axes are unchanged from v1 (frozen, not swept differently in
# v2), so no v2 sibling constant is needed for them.
_CONTINUOUS_AXES_V2 = (("learning_rate", "log"), ("embedding_dropout", "linear"),
                        ("output_dropout", "linear"), ("seq_length", "linear"))
_CATEGORICAL_AXES_V2 = ("hidden_size", "batch_size")

# See module docstring: both ends of the q_uniform seq_length grid are
# treated as natural (non-expandable) bounds for boundary-pressure-tier
# REPORTING purposes only -- a conservative default, not a search-space
# decision. Never consulted anywhere except derive_boundary_pressure_table_v2.
_SEQ_LENGTH_BOUNDARY_NATURE = {"lower": "natural", "upper": "natural"}


def derive_boundary_pressure_table_v2(trial_df: pd.DataFrame) -> pd.DataFrame:
    """v2 sibling of :func:`sweep_v1_review_analysis.derive_boundary_pressure_table`.

    Identical evidence construction, extended to all six axes. For the five
    axes v1 already covers, boundary nature/lower/upper come from
    ``SEARCH_DOMAIN_V2`` (a byte-identical deep copy of v1's
    ``SEARCH_DOMAIN`` for those five axes, per
    :mod:`sweep_v2_six_axis_campaign`'s own construction) so the reported
    tiers for those five axes are numerically identical to v1's when run
    over the same data. ``seq_length``'s lower/upper come from
    :data:`sweep_v2_six_axis_campaign.SEQ_LENGTH_MIN`/``SEQ_LENGTH_MAX``
    directly (``SEARCH_DOMAIN_V2["seq_length"]`` uses ``min``/``max``/``q``
    keys for the q_uniform HPO contract, not the ``lower``/``upper`` shape
    this diagnostic expects) and its boundary nature is the conservative
    ``_SEQ_LENGTH_BOUNDARY_NATURE`` default documented in the module
    docstring.
    """
    valid = valid_trials(trial_df)
    bayesian = valid[valid["search_arm"] == "bayesian"]
    top_mask, _ = is_top_quartile(valid)
    top = valid[top_mask]
    domain = SEARCH_DOMAIN_V2
    rows: list[dict[str, Any]] = []

    for axis, geometry in _CONTINUOUS_AXES_V2:
        if axis == "seq_length":
            lower, upper = SEQ_LENGTH_MIN, SEQ_LENGTH_MAX
        else:
            lower, upper = domain[axis]["lower"], domain[axis]["upper"]
        for side in ("lower", "upper"):
            nature = (_SEQ_LENGTH_BOUNDARY_NATURE[side] if axis == "seq_length"
                      else domain[axis][f"{side}_boundary"])
            near_lower_top, near_upper_top, _ = near_boundary_mask(top[axis], lower, upper, geometry)
            near_top = near_lower_top if side == "lower" else near_upper_top
            fraction = float(np.mean(near_top)) if len(top) else 0.0
            drift = proposal_drift_evidence(bayesian, axis, side, geometry=geometry, lower=lower, upper=upper)
            neighborhood = neighborhood_support_evidence(valid, axis, side, geometry=geometry, lower=lower, upper=upper)
            tier = boundary_pressure_tier(top_quartile_near_fraction=fraction,
                                           drift_detected=drift["drift_toward_boundary"],
                                           neighborhood_supports=neighborhood["supports_direction"])
            bayesian_stats = _arm_near_stats(top, "bayesian", axis, side, geometry=geometry, lower=lower, upper=upper)
            random_stats = _arm_near_stats(top, "random_control", axis, side, geometry=geometry, lower=lower, upper=upper)
            rows.append({
                "axis": axis, "boundary_side": side, "boundary_nature": nature,
                "top_quartile_near_fraction": round(fraction, 3),
                "top_quartile_near_count": int(near_top.sum()), "top_quartile_n": int(len(top)),
                "bayesian_top_quartile_near_fraction": bayesian_stats["fraction"],
                "bayesian_top_quartile_near_count": bayesian_stats["count"], "bayesian_top_quartile_n": bayesian_stats["n"],
                "random_top_quartile_near_fraction": random_stats["fraction"],
                "random_top_quartile_near_count": random_stats["count"], "random_top_quartile_n": random_stats["n"],
                "proposal_drift_toward_boundary": drift["drift_toward_boundary"],
                "occupancy_drift_toward_boundary": drift["occupancy_drift_toward_boundary"],
                "effect_size_drift_toward_boundary": drift["effect_size_drift_toward_boundary"],
                "early_near_fraction": _round_or_none(drift["early_near_boundary_fraction"]),
                "late_near_fraction": _round_or_none(drift["late_near_boundary_fraction"]),
                "early_median_position": drift["early_median_position"],
                "late_median_position": drift["late_median_position"],
                "position_shift_toward_boundary": drift["position_shift_toward_boundary"],
                "spearman_toward_boundary": drift["spearman_toward_boundary"],
                "neighborhood_evidence": _round_or_none(neighborhood["closeness_score_correlation"]),
                "neighborhood_supports_direction": neighborhood["supports_direction"],
                "tier": tier,
                "interpretation": _interpretation_label(nature, tier),
                "expansion_eligible": nature == "expandable",
            })

    for axis in _CATEGORICAL_AXES_V2:
        values = domain[axis]["values"]
        for extreme, side in ((min(values), "lower"), (max(values), "upper")):
            nature = domain[axis][f"{side}_boundary"]
            top_extreme_mask = categorical_extreme_mask(top[axis], extreme)
            fraction = float(np.mean(top_extreme_mask)) if len(top) else 0.0
            drift = proposal_drift_evidence(bayesian, axis, side, extreme_value=extreme)
            neighborhood = neighborhood_support_evidence(valid, axis, side, extreme_value=extreme)
            tier = boundary_pressure_tier(top_quartile_near_fraction=fraction,
                                           drift_detected=drift["drift_toward_boundary"],
                                           neighborhood_supports=neighborhood["supports_direction"])
            bayesian_stats = _arm_near_stats(top, "bayesian", axis, side, extreme_value=extreme)
            random_stats = _arm_near_stats(top, "random_control", axis, side, extreme_value=extreme)
            rows.append({
                "axis": axis, "boundary_side": f"{side}({extreme})", "boundary_nature": nature,
                "top_quartile_near_fraction": round(fraction, 3),
                "top_quartile_near_count": int(top_extreme_mask.sum()), "top_quartile_n": int(len(top)),
                "bayesian_top_quartile_near_fraction": bayesian_stats["fraction"],
                "bayesian_top_quartile_near_count": bayesian_stats["count"], "bayesian_top_quartile_n": bayesian_stats["n"],
                "random_top_quartile_near_fraction": random_stats["fraction"],
                "random_top_quartile_near_count": random_stats["count"], "random_top_quartile_n": random_stats["n"],
                "proposal_drift_toward_boundary": drift["drift_toward_boundary"],
                "occupancy_drift_toward_boundary": drift["occupancy_drift_toward_boundary"],
                "effect_size_drift_toward_boundary": drift["effect_size_drift_toward_boundary"],
                "early_near_fraction": _round_or_none(drift["early_near_boundary_fraction"]),
                "late_near_fraction": _round_or_none(drift["late_near_boundary_fraction"]),
                "early_median_position": None, "late_median_position": None,
                "position_shift_toward_boundary": None, "spearman_toward_boundary": None,
                "neighborhood_evidence": _round_or_none(neighborhood["mean_gap"], 4),
                "neighborhood_supports_direction": neighborhood["supports_direction"],
                "tier": tier,
                "interpretation": _interpretation_label(nature, tier),
                "expansion_eligible": nature == "expandable",
            })

    return pd.DataFrame(rows)


def categorical_occupancy_table_v2(trial_df: pd.DataFrame) -> pd.DataFrame:
    """v2 sibling of :func:`sweep_v1_review_analysis.categorical_occupancy_table`.

    Reads category values from ``SEARCH_DOMAIN_V2`` instead of v1's
    ``SEARCH_DOMAIN`` -- for ``hidden_size``/``batch_size`` these are
    byte-identical deep copies (both axes are frozen/unchanged in v2), so
    results are numerically identical to v1's over the same data; kept as a
    genuine sibling rather than a direct call to v1's function purely for
    identity separation (never read v1's ``SEARCH_DOMAIN`` from v2 code).
    """
    valid = valid_trials(trial_df)
    top_mask, _ = is_top_quartile(valid)
    rows: list[dict[str, Any]] = []
    for axis in _CATEGORICAL_AXES_V2:
        for value in SEARCH_DOMAIN_V2[axis]["values"]:
            for arm in ("bayesian", "random_control"):
                arm_df = valid[valid["search_arm"] == arm]
                count = int((arm_df[axis] == value).sum())
                total = int(len(arm_df))
                top_arm = valid[top_mask & (valid["search_arm"] == arm)]
                top_count = int((top_arm[axis] == value).sum())
                rows.append({
                    "axis": axis, "value": value, "search_arm": arm,
                    "count": count, "total": total,
                    "fraction": round(count / total, 3) if total else None,
                    "top_quartile_count": top_count, "top_quartile_total": int(len(top_arm)),
                    "top_quartile_fraction": round(top_count / len(top_arm), 3) if len(top_arm) else None,
                })
    return pd.DataFrame(rows)


def most_pressured_continuous_axis_v2(boundary_df: pd.DataFrame) -> tuple[str, str]:
    """v2 sibling of :func:`sweep_v1_review_analysis.most_pressured_continuous_axis`.

    Same selection rule over the six-axis continuous set. Falls back to
    ``learning_rate``/``lower`` if nothing is expandable, identically to
    v1 -- ``seq_length`` is never eligible for this fallback since both its
    sides are always reported ``natural`` (see module docstring), so it can
    never be silently selected as "most pressured" while the domain's
    expandability remains an open scientific question.
    """
    continuous_axes = {axis for axis, _ in _CONTINUOUS_AXES_V2}
    candidates = boundary_df[(boundary_df["axis"].isin(continuous_axes)) & (boundary_df["expansion_eligible"])]
    if not len(candidates):
        return "learning_rate", "lower"
    ranked = candidates.assign(_rank=candidates["tier"].map(TIER_ORDER)).sort_values(
        ["_rank", "top_quartile_near_fraction"], ascending=False)
    best = ranked.iloc[0]
    return best["axis"], best["boundary_side"]


def top_configurations_table_v2(trial_df: pd.DataFrame, n_bayesian: int = 10, n_random: int = 3) -> pd.DataFrame:
    """v2 sibling of :func:`sweep_v1_review_analysis.top_configurations_table`.

    Identical ranking/short-id construction, with ``seq_length`` added to
    the displayed hyperparameter columns. v2 never runs a ``random_control``
    arm (:data:`sweep_v2_six_axis_campaign.SEARCH_ARMS_V2` is Bayesian-only),
    so ``n_random`` defaults are retained only for interface parity with v1
    and will simply select zero rows against real v2 evidence.
    """
    valid = valid_trials(trial_df)
    bayesian = valid[valid["search_arm"] == "bayesian"].sort_values("best_score", ascending=False).head(n_bayesian)
    random_arm = valid[valid["search_arm"] == "random_control"].sort_values("best_score", ascending=False).head(n_random)
    combined = pd.concat([bayesian, random_arm], ignore_index=True).sort_values("best_score", ascending=False)
    combined = combined.reset_index(drop=True)
    combined.insert(0, "rank", range(1, len(combined) + 1))
    combined["configuration_short_id"] = combined["configuration_id"].astype(str).str.slice(-8)
    columns = ["rank", "configuration_short_id", "search_arm", "best_score", "learning_rate", "hidden_size",
               "embedding_dropout", "output_dropout", "batch_size", "seq_length", "proposal_order", "best_epoch",
               "late_gain_10_to_12", "best_minus_final", "late_best"]
    return combined[columns]
