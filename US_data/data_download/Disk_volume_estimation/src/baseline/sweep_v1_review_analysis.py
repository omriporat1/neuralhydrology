"""Analysis/derivation helpers for the Phase-B Sweep-v1 durable review layer.

Pure pandas/numpy transformations over authoritative Flash-NH tabular
evidence (trial_summary / epoch_trajectory / proposal / operations rows, in
the schema defined by :mod:`src.baseline.sweep_v1_campaign`).  This module
has no W&B import, does not redefine campaign/domain identity, and makes no
CONTINUE/UNCERTAIN/EXPAND decision itself: it only exposes transparent,
challengeable evidence (counts, fractions, correlations) for human review.

Nothing here mutates a search domain, biases a Bayesian objective, or scores
candidates for promotion.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.baseline import sweep_v1_campaign as sweep

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
    "derive_boundary_pressure_table",
    "categorical_occupancy_table",
    "representative_trajectory_selection",
    "boundary_pressure_evolution",
    "boundary_band_sensitivity",
    "top_configurations_table",
    "checkpoint_status_summary",
    "most_pressured_continuous_axis",
]

VALID_STATUS = "pass"
# Outer ~10% of the search geometry on either end, per the launch contract's
# frozen boundary-pressure definition (log coordinate for LR, ordinary
# coordinate for dropout axes).
BOUNDARY_FRACTION = 0.10
# Supplemental robustness bands for the boundary-band sensitivity diagnostic
# (§12 of the human-review refinement pass).  These do NOT alter the
# canonical BOUNDARY_FRACTION decision rule above -- they only expose how
# stable a qualitative "near-boundary" read is under narrower/wider bands.
SENSITIVITY_BAND_FRACTIONS = (0.05, 0.10, 0.20)
TIER_ORDER = {"WEAK-NONE": 0, "MODERATE": 1, "STRONG": 2}


def valid_trials(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with workflow_status == 'pass' (excludes failures/incomplete retries)."""
    valid = trial_df[trial_df["workflow_status"] == VALID_STATUS].copy()
    if "late_best" in valid.columns:
        # `late_best` is object-dtype on the full trial_df (None on failed rows);
        # coerce to real bool here so downstream `~valid["late_best"]` boolean
        # masking doesn't silently bitwise-invert Python bools (~True == -2).
        valid["late_best"] = valid["late_best"].astype(bool)
    return valid


def checkpoint_slice(trial_df: pd.DataFrame, checkpoint_valid_bayesian_count: int,
                      random_control_count: int | None = None) -> pd.DataFrame:
    """Trials visible at a boundary-review checkpoint.

    The random-control arm is a frozen, already-complete manifest, so it is
    included in full (optionally capped) regardless of the Bayesian count.
    Bayesian trials are included up to (and possibly slightly overshooting)
    ``checkpoint_valid_bayesian_count``, matching the launch contract's
    "pause issuance, allow bounded overshoot" rule -- this function does not
    truncate overshoot, it simply reflects whatever is <= the requested
    count among already-derived ``valid_result_order`` values.
    """
    valid = valid_trials(trial_df)
    bayesian = valid[(valid["search_arm"] == "bayesian")
                      & (valid["valid_result_order"] <= checkpoint_valid_bayesian_count)]
    random_arm = valid[valid["search_arm"] == "random_control"]
    if random_control_count is not None:
        random_arm = random_arm[random_arm["valid_result_order"] <= random_control_count]
    return (pd.concat([bayesian, random_arm], ignore_index=True)
            .sort_values(["search_arm", "valid_result_order"]).reset_index(drop=True))


def checkpoint_operations_slice(trial_df: pd.DataFrame, checkpoint_valid_bayesian_count: int,
                                 random_control_count: int | None = None) -> pd.DataFrame:
    """Like :func:`checkpoint_slice` but also includes failed/incomplete
    attempts whose proposal falls within the same checkpoint window.

    For operational evidence (runtime/GPU-hours, failure/retry counts) only
    -- scientific progress/comparison figures must use :func:`checkpoint_slice`,
    which excludes failures entirely.
    """
    valid_slice = checkpoint_slice(trial_df, checkpoint_valid_bayesian_count, random_control_count)
    failed = trial_df[trial_df["workflow_status"] != VALID_STATUS]
    bayesian_failed = failed[(failed["search_arm"] == "bayesian")
                              & (failed["proposal_order"] <= checkpoint_valid_bayesian_count)]
    random_failed = failed[failed["search_arm"] == "random_control"]
    if random_control_count is not None:
        random_failed = random_failed[random_failed["proposal_order"] <= random_control_count]
    return pd.concat([valid_slice, bayesian_failed, random_failed], ignore_index=True)


def common_n(trial_df_slice: pd.DataFrame) -> int:
    """Largest N such that both arms have >= N valid trials in this slice."""
    counts = trial_df_slice.groupby("search_arm")["valid_result_order"].count()
    return int(min(counts.get("bayesian", 0), counts.get("random_control", 0)))


def cumulative_best_by_order(values_in_order: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values_in_order), dtype=float)
    return np.maximum.accumulate(arr) if arr.size else arr


def cumulative_gpu_hours_by_order(gpu_hours_in_order: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(gpu_hours_in_order), dtype=float)
    return np.cumsum(arr) if arr.size else arr


def top_quartile_threshold(scores: Sequence[float]) -> float:
    arr = np.asarray(list(scores), dtype=float)
    return float(np.percentile(arr, 75))


def is_top_quartile(df: pd.DataFrame, score_col: str = "best_score") -> tuple[pd.Series, float]:
    threshold = top_quartile_threshold(df[score_col])
    return df[score_col] >= threshold, threshold


def _log_position(value: float, lower: float, upper: float) -> float:
    return (math.log10(value) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))


def _linear_position(value: float, lower: float, upper: float) -> float:
    return (value - lower) / (upper - lower)


def near_boundary_mask(values: Sequence[float], lower: float, upper: float, geometry: str = "linear",
                        fraction: float = BOUNDARY_FRACTION) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (near_lower, near_upper, position_0_to_1) for a continuous axis.

    ``position`` is 0 at the lower bound and 1 at the upper bound, measured
    in the geometry appropriate to the axis (log10 for learning_rate, linear
    for the dropout axes) -- matching the launch contract's frozen
    boundary-pressure definition.
    """
    position_fn = _log_position if geometry == "log" else _linear_position
    positions = np.array([position_fn(float(v), lower, upper) for v in values], dtype=float)
    return positions <= fraction, positions >= (1 - fraction), positions


def categorical_extreme_mask(values: Sequence[Any], extreme_value: Any) -> np.ndarray:
    return np.asarray([v == extreme_value for v in values])


def proposal_drift_evidence(bayesian_df: pd.DataFrame, column: str, side: str, *, geometry: str = "linear",
                             lower: float | None = None, upper: float | None = None,
                             extreme_value: Any = None) -> dict[str, Any]:
    """Bayesian-only evidence of directional proposal movement toward one
    boundary/extreme, over proposal order.  Two complementary signals are
    exposed side by side so a reviewer never has to trust a single boolean:

    - OCCUPANCY (coarse): early-half vs late-half fraction of proposals
      landing inside the canonical near-boundary band.  This under-counts
      genuine directional movement that never quite crosses the band (e.g.
      a proposal median that clearly slides from mid-domain toward the edge
      without many individual points crossing the 10% line).
    - EFFECT SIZE (continuous axes only): the early/late MEDIAN of the
      transformed [0,1] position, their signed shift toward the boundary,
      and the Spearman rank correlation between proposal order and
      transformed position -- both are large-sample-free, threshold-light
      measures of a real trend.

    ``drift_toward_boundary`` is True if either signal shows a clear
    directional move (occupancy shift >= 0.15, or a position shift >= 0.10
    of the normalized domain together with a Spearman correlation >= 0.30
    in the boundary-consistent direction).  Both raw signals -- and the
    sub-flags that produced the combined boolean -- remain in the returned
    evidence, per the launch-contract requirement that any retained
    boolean drift cue stay secondary to the displayed numbers, not an
    opaque classifier.
    """
    ordered = bayesian_df.sort_values("proposal_order").reset_index(drop=True)
    half = len(ordered) // 2
    early, late = ordered.iloc[:half], ordered.iloc[half:]

    def _fraction(chunk: pd.DataFrame) -> float:
        if len(chunk) == 0:
            return float("nan")
        if extreme_value is not None:
            return float(np.mean(chunk[column] == extreme_value))
        near_lower, near_upper, _ = near_boundary_mask(chunk[column], lower, upper, geometry)
        return float(np.mean(near_lower if side == "lower" else near_upper))

    early_fraction, late_fraction = _fraction(early), _fraction(late)
    occupancy_drift = (not math.isnan(early_fraction) and not math.isnan(late_fraction)
                        and (late_fraction - early_fraction) >= 0.15)

    evidence: dict[str, Any] = {
        "early_proposal_count": int(len(early)), "late_proposal_count": int(len(late)),
        "early_near_boundary_fraction": early_fraction, "late_near_boundary_fraction": late_fraction,
        "occupancy_drift_toward_boundary": bool(occupancy_drift),
        "early_median_position": None, "late_median_position": None,
        "position_shift_toward_boundary": None, "spearman_toward_boundary": None,
        "effect_size_drift_toward_boundary": False,
    }

    if extreme_value is None and len(ordered):
        _, _, all_positions = near_boundary_mask(ordered[column], lower, upper, geometry)
        positioned = ordered.assign(_position=all_positions)
        early_pos, late_pos = positioned["_position"].iloc[:half], positioned["_position"].iloc[half:]
        early_median = float(early_pos.median()) if len(early_pos) else float("nan")
        late_median = float(late_pos.median()) if len(late_pos) else float("nan")
        raw_shift = late_median - early_median  # negative == position moved toward the lower bound
        shift_toward_boundary = -raw_shift if side == "lower" else raw_shift
        if len(positioned) >= 4 and positioned["_position"].std() > 0:
            spearman = float(positioned["proposal_order"].corr(positioned["_position"], method="spearman"))
        else:
            spearman = float("nan")
        spearman_toward_boundary = -spearman if side == "lower" else spearman
        effect_size_drift = (not math.isnan(shift_toward_boundary) and shift_toward_boundary >= 0.10
                              and not math.isnan(spearman_toward_boundary) and spearman_toward_boundary >= 0.30)
        evidence.update({
            "early_median_position": _round_or_none(early_median),
            "late_median_position": _round_or_none(late_median),
            "position_shift_toward_boundary": _round_or_none(shift_toward_boundary),
            "spearman_toward_boundary": _round_or_none(spearman_toward_boundary),
            "effect_size_drift_toward_boundary": bool(effect_size_drift),
        })

    evidence["drift_toward_boundary"] = bool(occupancy_drift or evidence["effect_size_drift_toward_boundary"])
    return evidence


def neighborhood_support_evidence(df: pd.DataFrame, column: str, side: str, *, geometry: str = "linear",
                                   lower: float | None = None, upper: float | None = None,
                                   extreme_value: Any = None, score_col: str = "best_score") -> dict[str, Any]:
    """Evidence that the boundary direction is a real trend across ALL valid
    trials (not just the top quartile / one lucky point).
    """
    if extreme_value is not None:
        in_group = df[df[column] == extreme_value][score_col]
        out_group = df[df[column] != extreme_value][score_col]
        mean_gap = float(in_group.mean() - out_group.mean()) if len(in_group) and len(out_group) else float("nan")
        return {
            "extreme_mean_score": float(in_group.mean()) if len(in_group) else float("nan"),
            "other_mean_score": float(out_group.mean()) if len(out_group) else float("nan"),
            "mean_gap": mean_gap,
            "supports_direction": bool(not math.isnan(mean_gap) and mean_gap > 0.005),
            "n_extreme": int(len(in_group)), "n_other": int(len(out_group)),
        }
    _, _, positions = near_boundary_mask(df[column], lower, upper, geometry)
    distance_to_boundary = positions if side == "lower" else (1 - positions)
    closeness = 1 - distance_to_boundary
    if len(df) >= 3 and np.std(closeness) > 0 and np.std(df[score_col]) > 0:
        correlation = float(np.corrcoef(closeness, df[score_col])[0, 1])
    else:
        correlation = float("nan")
    return {
        "n": int(len(df)), "closeness_score_correlation": correlation,
        "supports_direction": bool(not math.isnan(correlation) and correlation >= 0.20),
    }


def boundary_pressure_tier(*, top_quartile_near_fraction: float, drift_detected: bool,
                            neighborhood_supports: bool) -> str:
    """STRONG/MODERATE/WEAK-NONE per the frozen launch-contract rule.

    Takes already-computed raw evidence as input and makes no independent
    measurement of its own -- this is deliberately NOT an opaque weighted
    composite score; every input remains inspectable next to the label.
    """
    strong = top_quartile_near_fraction >= 0.5 and drift_detected and neighborhood_supports
    if strong:
        return "STRONG"
    moderate = (top_quartile_near_fraction >= 0.25) or drift_detected or neighborhood_supports
    return "MODERATE" if moderate else "WEAK-NONE"


_CONTINUOUS_AXES = (("learning_rate", "log"), ("embedding_dropout", "linear"), ("output_dropout", "linear"))
_CATEGORICAL_AXES = ("hidden_size", "batch_size")


def _round_or_none(value: float, digits: int = 3):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(value, digits)


def _arm_near_stats(top_df: pd.DataFrame, arm: str, column: str, side: str, *, geometry: str = "linear",
                     lower: float | None = None, upper: float | None = None,
                     extreme_value: Any = None) -> dict[str, Any]:
    """Top-quartile near-boundary count/n/fraction for a single search arm,
    computed from the (already pooled) top-quartile subset.  Exposed
    separately from the pooled figure because Bayesian and random-control
    sample sizes differ substantially (§4 of the human-review refinement
    pass) -- a reviewer should be able to see whether pooled evidence is
    actually driven by one arm."""
    arm_top = top_df[top_df["search_arm"] == arm]
    if extreme_value is not None:
        near = categorical_extreme_mask(arm_top[column], extreme_value)
    else:
        near_lower, near_upper, _ = near_boundary_mask(arm_top[column], lower, upper, geometry)
        near = near_lower if side == "lower" else near_upper
    n = int(len(arm_top))
    count = int(np.sum(near)) if n else 0
    return {"n": n, "count": count, "fraction": round(count / n, 3) if n else None}


def _interpretation_label(nature: str, tier: str) -> str:
    """Human-facing label that never lets a natural (non-expandable) bound
    read as expansion pressure (§11): natural bounds are always framed as a
    PREFERENCE signal, expandable bounds as an expansion-PRESSURE tier."""
    if nature == "natural":
        return f"N/A (natural bound, no expansion possible) — preference: {tier}"
    return tier


def derive_boundary_pressure_table(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Per-axis, per-side boundary-pressure evidence table (§7 of the launch
    contract).  One row per (axis, boundary side); natural bounds are always
    labeled so they can never be misread as a call to expand.

    Beyond the original pooled top-quartile occupancy, each row now also
    separates BAYESIAN-only and RANDOM-CONTROL-only top-quartile occupancy
    (§4), carries the full effect-size drift evidence from
    :func:`proposal_drift_evidence` (§3), and an `interpretation` column
    that distinguishes expansion-pressure language from natural-boundary
    preference language (§11).
    """
    valid = valid_trials(trial_df)
    bayesian = valid[valid["search_arm"] == "bayesian"]
    top_mask, _ = is_top_quartile(valid)
    top = valid[top_mask]
    domain = sweep.SEARCH_DOMAIN
    rows: list[dict[str, Any]] = []

    for axis, geometry in _CONTINUOUS_AXES:
        lower, upper = domain[axis]["lower"], domain[axis]["upper"]
        for side in ("lower", "upper"):
            nature = domain[axis][f"{side}_boundary"]
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

    for axis in _CATEGORICAL_AXES:
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


def categorical_occupancy_table(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm, per-category counts/fractions for hidden_size and batch_size,
    plus the top-quartile fraction landing in each category (figure 5/panel C
    evidence)."""
    valid = valid_trials(trial_df)
    top_mask, _ = is_top_quartile(valid)
    rows: list[dict[str, Any]] = []
    for axis in _CATEGORICAL_AXES:
        for value in sweep.SEARCH_DOMAIN[axis]["values"]:
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


def representative_trajectory_selection(trial_df: pd.DataFrame) -> dict[str, str | None]:
    """Pick one trial_id per narrative role for figure 7.

    Prefers an explicit ``synthetic_archetype`` tag when present (fixture
    data); falls back to a heuristic over derived diagnostics so the same
    renderer works unmodified on real, un-tagged evidence.
    """
    valid = valid_trials(trial_df)
    picks: dict[str, str | None] = {"strong_stable": None, "typical": None, "late_best": None, "unstable": None}
    if "synthetic_archetype" in valid.columns:
        for role in picks:
            matches = valid[valid["synthetic_archetype"] == role]
            if len(matches):
                picks[role] = matches.sort_values("best_score", ascending=False).iloc[0]["trial_id"]
        if any(picks.values()):
            return picks

    stable = valid[(~valid["late_best"]) & (valid["best_minus_final"] < 0.02)]
    if len(stable):
        ranked = stable.sort_values("best_score", ascending=False)
        picks["strong_stable"] = ranked.iloc[0]["trial_id"]
        picks["typical"] = ranked.iloc[len(ranked) // 2]["trial_id"]
    late_best = valid[valid["late_best"]].sort_values("late_gain_10_to_12", ascending=False)
    if len(late_best):
        picks["late_best"] = late_best.iloc[0]["trial_id"]
    unstable = valid.sort_values("best_minus_final", ascending=False)
    if len(unstable):
        picks["unstable"] = unstable.iloc[0]["trial_id"]
    return picks


def boundary_pressure_evolution(current_df: pd.DataFrame, previous_df: pd.DataFrame | None) -> pd.DataFrame:
    """Checkpoint-to-checkpoint evolution view (§10): merges the current
    boundary-pressure table against the immediately-preceding checkpoint's
    table (both from :func:`derive_boundary_pressure_table`) on
    ``(axis, boundary_side)`` and derives a strengthening/stable/weakening
    ``direction`` from the tier ranking (:data:`TIER_ORDER`), so a reviewer
    never has to manually diff two old decision boards.

    If ``previous_df`` is ``None`` (first checkpoint in the campaign),
    ``tier_previous``/``direction`` are populated with an explicit
    "n/a (first checkpoint)" marker rather than left ambiguous.
    """
    previous_lookup = (previous_df.set_index(["axis", "boundary_side"])
                        if previous_df is not None and len(previous_df) else None)
    rows: list[dict[str, Any]] = []
    for _, row in current_df.iterrows():
        key = (row["axis"], row["boundary_side"])
        entry: dict[str, Any] = {
            "axis": row["axis"], "boundary_side": row["boundary_side"], "boundary_nature": row["boundary_nature"],
            "tier_current": row["tier"],
            "top_quartile_near_fraction_current": row["top_quartile_near_fraction"],
        }
        if previous_lookup is not None and key in previous_lookup.index:
            prev = previous_lookup.loc[key]
            entry["tier_previous"] = prev["tier"]
            entry["top_quartile_near_fraction_previous"] = prev["top_quartile_near_fraction"]
            cur_rank, prev_rank = TIER_ORDER[row["tier"]], TIER_ORDER[prev["tier"]]
            if cur_rank > prev_rank:
                direction = "strengthening"
            elif cur_rank < prev_rank:
                direction = "weakening"
            else:
                direction = "stable"
        else:
            entry["tier_previous"] = None
            entry["top_quartile_near_fraction_previous"] = None
            direction = "n/a (first checkpoint)"
        entry["direction"] = direction
        rows.append(entry)
    return pd.DataFrame(rows)


def boundary_band_sensitivity(values: Sequence[float], lower: float, upper: float, geometry: str = "linear",
                               fractions: Sequence[float] = SENSITIVITY_BAND_FRACTIONS) -> pd.DataFrame:
    """Supplemental robustness diagnostic (§12): near-boundary occupancy of
    ``values`` (typically a top-quartile subset for one continuous axis)
    recomputed under alternate band widths.  This is NOT a second decision
    rule and does not alter :data:`BOUNDARY_FRACTION` -- it only shows
    whether the canonical-band read is stable if the band were narrower or
    wider.
    """
    position_fn = _log_position if geometry == "log" else _linear_position
    positions = np.array([position_fn(float(v), lower, upper) for v in values], dtype=float)
    rows = []
    for frac in fractions:
        near_lower = positions <= frac
        near_upper = positions >= (1 - frac)
        n = int(len(positions))
        rows.append({
            "band_fraction": frac, "canonical": bool(abs(frac - BOUNDARY_FRACTION) < 1e-9), "n": n,
            "near_lower_count": int(near_lower.sum()),
            "near_lower_fraction": round(float(np.mean(near_lower)), 3) if n else None,
            "near_upper_count": int(near_upper.sum()),
            "near_upper_fraction": round(float(np.mean(near_upper)), 3) if n else None,
        })
    return pd.DataFrame(rows)


def most_pressured_continuous_axis(boundary_df: pd.DataFrame) -> tuple[str, str]:
    """Pick the (axis, side) among the continuous EXPANDABLE axes/sides with
    the highest current tier (ties broken by top_quartile_near_fraction),
    for the single combined performance+drift panel required by §8.  Falls
    back to learning_rate/lower (the axis the launch contract already
    anticipates as most likely to show pressure) if nothing is expandable.
    """
    continuous_axes = {axis for axis, _ in _CONTINUOUS_AXES}
    candidates = boundary_df[(boundary_df["axis"].isin(continuous_axes)) & (boundary_df["expansion_eligible"])]
    if not len(candidates):
        return "learning_rate", "lower"
    ranked = candidates.assign(_rank=candidates["tier"].map(TIER_ORDER)).sort_values(
        ["_rank", "top_quartile_near_fraction"], ascending=False)
    best = ranked.iloc[0]
    return best["axis"], best["boundary_side"]


def top_configurations_table(trial_df: pd.DataFrame, n_bayesian: int = 10, n_random: int = 3) -> pd.DataFrame:
    """Ranked top-N-per-arm configuration matrix (§18): lets a reviewer see
    at a glance whether several independent strong candidates agree on a
    hyperparameter story, or whether the apparent conclusion is driven by a
    single isolated winner.
    """
    valid = valid_trials(trial_df)
    bayesian = valid[valid["search_arm"] == "bayesian"].sort_values("best_score", ascending=False).head(n_bayesian)
    random_arm = valid[valid["search_arm"] == "random_control"].sort_values("best_score", ascending=False).head(n_random)
    combined = pd.concat([bayesian, random_arm], ignore_index=True).sort_values("best_score", ascending=False)
    combined = combined.reset_index(drop=True)
    combined.insert(0, "rank", range(1, len(combined) + 1))
    # slice from the tail: configuration_id is "sweep_v1_cfg_<20-hex-char hash>",
    # so a fixed-length prefix slice would return the identical constant prefix
    # for every row -- the hash characters that actually distinguish configs
    # only appear after that prefix.
    combined["configuration_short_id"] = combined["configuration_id"].astype(str).str.slice(-8)
    columns = ["rank", "configuration_short_id", "search_arm", "best_score", "learning_rate", "hidden_size",
               "embedding_dropout", "output_dropout", "batch_size", "proposal_order", "best_epoch",
               "late_gain_10_to_12", "best_minus_final", "late_best"]
    return combined[columns]


def checkpoint_status_summary(trial_slice: pd.DataFrame, ops_slice: pd.DataFrame, *,
                               checkpoint_valid_bayesian_count: int, review_name: str) -> dict[str, Any]:
    """Compact status-header values for Decision Board v2 (§6): the numbers
    a reviewer needs before reading any panel -- where the campaign stands,
    what it has cost, and today's incumbents.  Concise values only; no
    prose interpretation.
    """
    valid = valid_trials(trial_slice)
    bayesian = valid[valid["search_arm"] == "bayesian"]
    random_arm = valid[valid["search_arm"] == "random_control"]
    n_bayesian, n_random = int(len(bayesian)), int(len(random_arm))
    overshoot = max(0, n_bayesian - checkpoint_valid_bayesian_count)
    n_failed = int((ops_slice["workflow_status"] != VALID_STATUS).sum()) if len(ops_slice) else 0
    n_common = common_n(trial_slice)
    random_at_common = (random_arm.sort_values("valid_result_order").iloc[:n_common]
                         if n_common and len(random_arm) else random_arm.iloc[0:0])
    return {
        "review_name": review_name,
        "target_valid_bayesian": int(checkpoint_valid_bayesian_count),
        "actual_valid_bayesian": n_bayesian,
        "bounded_overshoot": int(overshoot),
        "valid_random_control": n_random,
        "failed_or_retry_attempts": n_failed,
        "cumulative_bayesian_gpu_hours": round(float(bayesian["gpu_hours"].sum()), 2) if n_bayesian else 0.0,
        "cumulative_random_gpu_hours": round(float(random_arm["gpu_hours"].sum()), 2) if n_random else 0.0,
        "bayesian_incumbent_best": round(float(bayesian["best_score"].max()), 4) if n_bayesian else None,
        "random_best_at_common_n": round(float(random_at_common["best_score"].max()), 4) if len(random_at_common) else None,
        "common_n": int(n_common),
    }
