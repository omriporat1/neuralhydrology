"""RD1-C2 -- configuration-level analysis core for the frozen Stage-1 v2
12+12 scientific review.

Authoritative contract: ``docs/stage1_v2_12plus12_review_design_v001.md``
(RD1-C1 prospective freeze). This module implements **only** the pure
configuration-level numerical / structural analysis layer described for
RD1-C2 in that document's Section 19, plus an explicit in-memory consumer
contract and the derivations it needs.

What this module does
---------------------
* consumes an already-qualified in-memory configuration / proposal table
  (a :class:`pandas.DataFrame` or a sequence of row mappings);
* derives the frozen six-axis normalized geometry (RD1 §6.1), the frozen
  RMS configuration distance (RD1 §6.2), Bayesian novelty and the
  first-achieved-incumbent distance trajectory (RD1 §6.3), threshold-free
  top-k summaries (RD1 §4), the P2-referenced descriptive near-incumbent
  yield (RD1 §3.1), whole-sequence / cumulative descriptors (RD1 §7),
  first-achieved-final-incumbent post-incumbent descriptors (RD1 §4),
  duplicate canonical-coordinate accounting (RD1 §4), and explicit
  exact-GPU-hour coverage descriptors (RD1 §5).

What this module must NOT do (per the RD1-C2 boundary)
-----------------------------------------------------
* it does not parse campaign artifacts, crawl ``.scratch_local/``, read
  Slurm accounting, contact any remote service, or run NeuralHydrology;
* it does not qualify campaign evidence, redefine trial validity, change
  the objective / fidelity / search space, or make any
  scientific-confidence / promotion / selection judgement;
* it does not reuse the legacy early/late ``proposal_drift_evidence`` or
  boundary-pressure-tier semantics from
  :mod:`sweep_v1_review_analysis` / :mod:`sweep_v2_six_axis_review_analysis`
  (RD1 §7, §15.1), and it does not inherit the legacy ``seq_length``
  ``"natural / N/A"`` reporting interpretation (RD1 §14).

Authority reuse
---------------
:mod:`sweep_v2_six_axis_campaign` is the sole authority for six-axis
legality and canonicalization -- every configuration is validated /
canonicalized through :func:`canonical_hyperparameters_v2`. The only
legacy analysis primitive reused is
:func:`sweep_v1_review_analysis.cumulative_best_by_order`
(``np.maximum.accumulate``), whose semantics match the RD1 cumulative-best
descriptor exactly for the maximized objective.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.baseline.sweep_v1_review_analysis import cumulative_best_by_order
from src.baseline.sweep_v2_six_axis_campaign import (
    SEARCH_ARMS_V2,
    SEARCH_DOMAIN_V2,
    SEQ_LENGTH_MAX,
    SEQ_LENGTH_MIN,
    SweepV2CampaignError,
    canonical_hyperparameters_v2,
)

__all__ = [
    "SIX_AXIS_ORDER",
    "BAYESIAN_ARM",
    "RANDOM_CONTROL_ARM",
    "P2_NEAR_INCUMBENT_ANCHOR",
    "NEAR_INCUMBENT_PRIMARY_MARGIN",
    "NEAR_INCUMBENT_SENSITIVITY_MARGIN",
    "DEFAULT_OBJECTIVE_COLUMN",
    "DEFAULT_EXACT_GPU_HOURS_COLUMN",
    "REQUIRED_IDENTITY_COLUMNS",
    "REQUIRED_COLUMNS",
    "ReviewTableContractError",
    "axis_normalization_spec",
    "canonical_configuration",
    "canonical_coordinate_key",
    "normalized_coordinates",
    "normalized_coordinate_map",
    "configuration_distance",
    "configuration_distance_between_configs",
    "bayesian_novelty_series",
    "first_achieved_incumbent_indices",
    "bayesian_incumbent_distance_frame",
    "top_k_summary",
    "near_incumbent_yield",
    "cumulative_near_incumbent_count",
    "duplicate_coordinate_accounting",
    "exact_compute_scope_summary",
    "exact_compute_coverage_by_prefix",
    "first_achieved_final_incumbent_index",
    "post_incumbent_descriptors",
    "validate_review_table",
    "analyze_review_table",
    "ArmConfigurationAnalysis",
    "ReviewAnalysis",
]

# ---------------------------------------------------------------------------
# Frozen RD1 constants.
# ---------------------------------------------------------------------------

#: Fixed, explicit six-axis order (RD1 §6.1). Every normalized-coordinate
#: vector and every distance in this module is over these axes, in this order.
SIX_AXIS_ORDER: tuple[str, ...] = (
    "learning_rate",
    "hidden_size",
    "embedding_dropout",
    "output_dropout",
    "batch_size",
    "seq_length",
)

BAYESIAN_ARM = "bayesian"
RANDOM_CONTROL_ARM = "random_control"

#: RD1 §3.1 frozen configuration-level near-incumbent anchor: P2's official
#: pre-P9 objective. **Descriptive review reference only** -- not a
#: statistical-equivalence bound and not a promotion threshold.
P2_NEAR_INCUMBENT_ANCHOR: float = 0.4388098707961096
#: RD1 §3.1 descriptive review margins (NSE, absolute, relative to the anchor).
NEAR_INCUMBENT_PRIMARY_MARGIN: float = 0.01
NEAR_INCUMBENT_SENSITIVITY_MARGIN: float = 0.005

# ---------------------------------------------------------------------------
# In-memory consumer contract (RD1-C2 §8; agent_handoff_rules.md §5).
# ---------------------------------------------------------------------------

DEFAULT_OBJECTIVE_COLUMN = "objective_score"
DEFAULT_EXACT_GPU_HOURS_COLUMN = "exact_gpu_hours"
#: Identity / provenance columns every review-table row must carry.
REQUIRED_IDENTITY_COLUMNS: tuple[str, ...] = (
    "search_arm",
    "proposal_order",
    "proposal_id",
    "configuration_id",
)
#: Full required column set (identity + the six hyperparameters). The
#: official objective column is named separately and defaults to
#: :data:`DEFAULT_OBJECTIVE_COLUMN`; the exact-GPU-hours column is optional.
REQUIRED_COLUMNS: tuple[str, ...] = (*REQUIRED_IDENTITY_COLUMNS, *SIX_AXIS_ORDER)


class ReviewTableContractError(ValueError):
    """Raised when an in-memory review table violates the RD1-C2 consumer contract.

    A malformed row fails clearly here rather than being silently coerced,
    dropped, or reconstructed (RD1-C2 §8).
    """


# ---------------------------------------------------------------------------
# A. Six-axis normalization (RD1 §6.1).
# ---------------------------------------------------------------------------


def _axis_normalization_spec() -> dict[str, dict[str, Any]]:
    """Derive the per-axis normalization spec from the authoritative v2 domain.

    The geometry per axis is frozen by RD1 §6.1; the bounds come from
    :data:`sweep_v2_six_axis_campaign.SEARCH_DOMAIN_V2` (the five v1-inherited
    axes) and :data:`SEQ_LENGTH_MIN` / :data:`SEQ_LENGTH_MAX` (the sixth
    axis), never re-typed here.
    """
    domain = SEARCH_DOMAIN_V2
    hidden_values = domain["hidden_size"]["values"]
    batch_values = domain["batch_size"]["values"]
    return {
        "learning_rate": {
            "geometry": "log10",
            "lower": float(domain["learning_rate"]["lower"]),
            "upper": float(domain["learning_rate"]["upper"]),
        },
        "hidden_size": {
            "geometry": "log2",
            "lower": float(min(hidden_values)),
            "upper": float(max(hidden_values)),
        },
        "embedding_dropout": {
            "geometry": "linear",
            "lower": float(domain["embedding_dropout"]["lower"]),
            "upper": float(domain["embedding_dropout"]["upper"]),
        },
        "output_dropout": {
            "geometry": "linear",
            "lower": float(domain["output_dropout"]["lower"]),
            "upper": float(domain["output_dropout"]["upper"]),
        },
        "batch_size": {
            "geometry": "log2",
            "lower": float(min(batch_values)),
            "upper": float(max(batch_values)),
        },
        "seq_length": {
            "geometry": "linear",
            "lower": float(SEQ_LENGTH_MIN),
            "upper": float(SEQ_LENGTH_MAX),
        },
    }


_AXIS_NORMALIZATION: dict[str, dict[str, Any]] = _axis_normalization_spec()


def axis_normalization_spec() -> dict[str, dict[str, Any]]:
    """Return a copy of the frozen per-axis normalization spec (transparency).

    ``geometry`` is one of ``"log10"``, ``"log2"``, ``"linear"``; ``lower`` /
    ``upper`` are the raw-space bounds that map to normalized ``0.0`` / ``1.0``.
    Equal axis weighting in the downstream distance is a transparent
    descriptive geometry, **not** evidence that the axes carry equal
    scientific importance (RD1 §6.2).
    """
    return {axis: dict(spec) for axis, spec in _AXIS_NORMALIZATION.items()}


def _transform(value: float, geometry: str) -> float:
    if geometry == "log10":
        return math.log10(value)
    if geometry == "log2":
        return math.log2(value)
    if geometry == "linear":
        return float(value)
    raise ValueError(f"unknown axis geometry: {geometry!r}")


def _normalize_axis(axis: str, value: float) -> float:
    spec = _AXIS_NORMALIZATION[axis]
    lower_t = _transform(spec["lower"], spec["geometry"])
    upper_t = _transform(spec["upper"], spec["geometry"])
    # An exact lower bound gives numerator 0.0 -> 0.0; an exact upper bound
    # gives (upper_t - lower_t) / (upper_t - lower_t) -> 1.0, regardless of
    # floating-point error in the transformed bounds themselves.
    return float((_transform(float(value), spec["geometry"]) - lower_t) / (upper_t - lower_t))


def _coerce_scalar(axis: str, value: Any) -> Any:
    """Adapt one DataFrame-boundary scalar to the type the v2 canonical
    contract expects (numpy scalar -> Python scalar).

    Integer-valued categoricals may arrive as ``numpy.int64`` or as a
    float-typed column (pandas upcasts any column holding a NaN); a value
    that is not genuinely integral is rejected, so this never silently
    accepts an off-grid categorical.
    """
    if isinstance(value, bool):
        raise SweepV2CampaignError(f"{axis} must not be a bool, got {value!r}")
    if axis in ("hidden_size", "batch_size", "seq_length"):
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            as_float = float(value)
            if math.isfinite(as_float) and as_float.is_integer():
                return int(as_float)
        raise SweepV2CampaignError(f"{axis} must be an integral value, got {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise SweepV2CampaignError(f"{axis} must be a finite real numeric value, got {value!r}")


def canonical_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate + canonicalize one configuration through the authoritative v2
    six-axis campaign contract.

    Raises :class:`SweepV2CampaignError` (a ``ValueError`` subclass) for a
    missing axis, an illegal / off-grid / out-of-domain coordinate, or a
    non-numeric value. The returned dict is the frozen v2 canonical
    six-axis coordinate (v1's ``.17g`` continuous serialization plus the
    normalized ``seq_length`` int); it is the authoritative identity used
    for duplicate detection -- never raw float-string equality.
    """
    missing = [axis for axis in SIX_AXIS_ORDER if axis not in config]
    if missing:
        raise SweepV2CampaignError(f"configuration missing six-axis fields: {sorted(missing)}")
    coerced = {axis: _coerce_scalar(axis, config[axis]) for axis in SIX_AXIS_ORDER}
    return canonical_hyperparameters_v2(coerced)


def canonical_coordinate_key(config: Mapping[str, Any]) -> str:
    """Deterministic string key for one configuration's authoritative
    canonical six-axis coordinate (RD1 §4 duplicate accounting)."""
    canonical = canonical_configuration(config)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def normalized_coordinates(config: Mapping[str, Any]) -> np.ndarray:
    """Normalized ``[0, 1]`` coordinates for one legal configuration, in
    :data:`SIX_AXIS_ORDER` (RD1 §6.1).

    Boundary values map predictably: the lower bound of each axis -> ``0.0``,
    the upper bound -> ``1.0``. The normalization encodes no scientific
    importance.
    """
    canonical = canonical_configuration(config)
    return np.array(
        [_normalize_axis(axis, float(canonical[axis])) for axis in SIX_AXIS_ORDER],
        dtype=float,
    )


def normalized_coordinate_map(config: Mapping[str, Any]) -> dict[str, float]:
    """:func:`normalized_coordinates` as an ``axis -> value`` mapping."""
    return {axis: float(value) for axis, value in zip(SIX_AXIS_ORDER, normalized_coordinates(config))}


# ---------------------------------------------------------------------------
# B. Configuration distance (RD1 §6.2).
# ---------------------------------------------------------------------------


def configuration_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Frozen RMS Euclidean distance ``sqrt(mean_k((a_k - b_k)^2))`` over the
    six normalized coordinates (RD1 §6.2).

    Symmetric; zero for identical coordinates; bounded in ``[0, 1]`` for
    inputs in ``[0, 1]``. Equal axis weighting is a transparent descriptive
    geometry, not a statement of equal axis importance.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != (len(SIX_AXIS_ORDER),) or y.shape != (len(SIX_AXIS_ORDER),):
        raise ValueError(
            f"configuration_distance expects two length-{len(SIX_AXIS_ORDER)} "
            f"normalized coordinate vectors, got shapes {x.shape} and {y.shape}"
        )
    return float(np.sqrt(np.mean((x - y) ** 2)))


def configuration_distance_between_configs(config_a: Mapping[str, Any], config_b: Mapping[str, Any]) -> float:
    """:func:`configuration_distance` over the normalized coordinates of two
    legal configurations."""
    return configuration_distance(normalized_coordinates(config_a), normalized_coordinates(config_b))


# ---------------------------------------------------------------------------
# C. Bayesian novelty (RD1 §6.3).
# ---------------------------------------------------------------------------


def bayesian_novelty_series(coords_in_order: Sequence[Sequence[float]]) -> np.ndarray:
    """``novelty(i) = min_j<i distance(coord_i, coord_j)`` over earlier valid
    Bayesian proposals, in valid proposal order (RD1 §6.3).

    The first valid Bayesian proposal has no earlier proposal, so its
    novelty is ``NaN`` (explicit missing value). Duplicate coordinates
    naturally yield novelty ``0.0``. This quantity is defined for the
    adaptive Bayesian sequence only -- it is never computed for
    random-control order, which carries no optimizer-learning
    interpretation (RD1 §6.3).
    """
    coords = [np.asarray(c, dtype=float) for c in coords_in_order]
    novelty = np.full(len(coords), np.nan, dtype=float)
    for i in range(1, len(coords)):
        novelty[i] = min(configuration_distance(coords[i], coords[j]) for j in range(i))
    return novelty


# ---------------------------------------------------------------------------
# D. Bayesian incumbent distance (RD1 §6.3), first-achieved semantics.
# ---------------------------------------------------------------------------


def first_achieved_incumbent_indices(objectives: Sequence[float]) -> list[int | None]:
    """For each position ``i``, the index (``< i``) of the **first** proposal
    that achieved the best-known objective immediately before ``i``.

    Deterministic first-achieved semantics for exact objective ties
    (RD1-C2 §D): a strictly greater score replaces the incumbent; an equal
    score does not. Position 0 has no prior incumbent -> ``None``.
    """
    scores = [float(o) for o in objectives]
    incumbents: list[int | None] = []
    best_index: int | None = None
    for i, score in enumerate(scores):
        incumbents.append(best_index)
        if best_index is None or score > scores[best_index]:
            best_index = i
    return incumbents


def bayesian_incumbent_distance_frame(
    coords_in_order: Sequence[Sequence[float]],
    objectives_in_order: Sequence[float],
    *,
    proposal_ids: Sequence[Any] | None = None,
    proposal_orders: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Per-proposal distance to the best-known Bayesian incumbent immediately
    before that proposal (RD1 §6.3), with the incumbent's identity exposed.

    The first proposal has no prior incumbent, so
    ``incumbent_distance`` is ``NaN``. Columns:
    ``position``, ``proposal_order``, ``proposal_id``,
    ``incumbent_position_before``, ``incumbent_proposal_order_before``,
    ``incumbent_proposal_id_before``, ``incumbent_distance``.
    """
    coords = [np.asarray(c, dtype=float) for c in coords_in_order]
    incumbents = first_achieved_incumbent_indices(objectives_in_order)
    rows: list[dict[str, Any]] = []
    for i, incumbent in enumerate(incumbents):
        rows.append(
            {
                "position": i,
                "proposal_order": None if proposal_orders is None else proposal_orders[i],
                "proposal_id": None if proposal_ids is None else proposal_ids[i],
                "incumbent_position_before": incumbent,
                "incumbent_proposal_order_before": (
                    None if incumbent is None or proposal_orders is None else proposal_orders[incumbent]
                ),
                "incumbent_proposal_id_before": (
                    None if incumbent is None or proposal_ids is None else proposal_ids[incumbent]
                ),
                "incumbent_distance": (
                    float("nan") if incumbent is None else configuration_distance(coords[i], coords[incumbent])
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E. Ranking / top-k summaries (RD1 §4), threshold-free.
# ---------------------------------------------------------------------------


def top_k_summary(values: Sequence[float], *, k: int = 3, identities: Sequence[Any] | None = None) -> dict[str, Any]:
    """Deterministic threshold-free top-k summary.

    Ordering: objective descending, ties broken by earlier original order.
    Non-finite values are excluded from the ranking. For ``k = 3`` the
    returned summary always includes ``best``, ``floor`` (minimum among the
    selected), ``mean``, and ``median``. If fewer than ``k`` finite values
    are supplied the summary is explicit (``is_complete = False``,
    ``n_selected < k``) rather than pretending a full top-k exists; with no
    finite values the numeric fields are ``NaN``. Selected rows preserve
    configuration / proposal identity.
    """
    raw = [float(v) for v in values]
    if identities is None:
        identities = list(range(len(raw)))
    ranked = sorted(
        ((value, identities[i], i) for i, value in enumerate(raw) if math.isfinite(value)),
        key=lambda item: (-item[0], item[2]),
    )
    selected = ranked[:k]
    selected_values = [value for value, _, _ in selected]
    n_available = len(ranked)
    return {
        "requested_k": int(k),
        "n_available": n_available,
        "n_selected": len(selected),
        "is_complete": n_available >= k,
        "selected": [
            {"identity": identity, "value": value, "original_index": original_index}
            for value, identity, original_index in selected
        ],
        "best": selected_values[0] if selected_values else float("nan"),
        "floor": min(selected_values) if selected_values else float("nan"),
        "mean": float(np.mean(selected_values)) if selected_values else float("nan"),
        "median": float(np.median(selected_values)) if selected_values else float("nan"),
    }


# ---------------------------------------------------------------------------
# F. P2-referenced near-incumbent yield (RD1 §3.1), one-sided.
# ---------------------------------------------------------------------------


def near_incumbent_yield(
    objectives: Sequence[float],
    *,
    anchor: float = P2_NEAR_INCUMBENT_ANCHOR,
    margin: float,
    identities: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Descriptive near-or-better yield relative to the frozen P2 anchor.

    Because the objective is maximized, this is a **one-sided performance
    floor**: a configuration counts iff ``objective >= anchor - margin``. A
    configuration that exceeds the anchor still counts (it is
    near-or-better) -- it is never excluded for being more than ``margin``
    above P2. These are descriptive review margins only; they are **not**
    equivalence bounds and **not** promotion thresholds (RD1 §3.1).
    """
    scores = [float(o) for o in objectives]
    if identities is None:
        identities = list(range(len(scores)))
    performance_floor = anchor - margin
    membership = [bool(score >= performance_floor) for score in scores]
    n_total = len(scores)
    n_near = int(sum(membership))
    return {
        "anchor": float(anchor),
        "margin": float(margin),
        "performance_floor": float(performance_floor),
        "margin_kind": "descriptive_one_sided_near_or_better",
        "n_total": n_total,
        "n_near_or_better": n_near,
        "fraction_near_or_better": (n_near / n_total) if n_total else float("nan"),
        "membership": [
            {"identity": identities[i], "objective": scores[i], "near_or_better": membership[i]}
            for i in range(n_total)
        ],
    }


def cumulative_near_incumbent_count(
    objectives: Sequence[float],
    *,
    anchor: float = P2_NEAR_INCUMBENT_ANCHOR,
    margin: float,
) -> np.ndarray:
    """Cumulative count of near-or-better configurations vs proposal count
    (RD1 §4), one-sided floor ``objective >= anchor - margin``."""
    performance_floor = anchor - margin
    flags = np.array([1 if float(o) >= performance_floor else 0 for o in objectives], dtype=int)
    return np.cumsum(flags) if flags.size else flags


# ---------------------------------------------------------------------------
# G/whole-sequence helpers: distribution summary.
# ---------------------------------------------------------------------------


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    """Transparent distribution summary that displays the observations
    (RD1 §4): count, min, quartiles, max, mean, IQR. Non-finite values are
    dropped; an empty set yields ``NaN`` numeric fields."""
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        nan = float("nan")
        return {"n": 0, "min": nan, "q1": nan, "median": nan, "q3": nan, "max": nan, "mean": nan, "iqr": nan}
    arr = np.asarray(finite, dtype=float)
    q1, median, q3 = (float(np.percentile(arr, p)) for p in (25, 50, 75))
    return {
        "n": len(finite),
        "min": float(arr.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "iqr": q3 - q1,
        "values": finite,
    }


# ---------------------------------------------------------------------------
# I. Duplicate-coordinate / proposal-vs-unique accounting (RD1 §4).
# ---------------------------------------------------------------------------


def duplicate_coordinate_accounting(
    configs: Sequence[Mapping[str, Any]],
    *,
    identities: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Preserve every proposal as an evidence unit while separately exposing
    canonical-coordinate duplication (RD1 §4).

    Duplicate detection uses the authoritative canonical six-axis
    coordinate (:func:`canonical_coordinate_key`), never raw float-string
    equality. Duplicate proposals are counted, not de-duplicated away.
    """
    config_list = list(configs)
    if identities is None:
        identities = list(range(len(config_list)))
    keys = [canonical_coordinate_key(config) for config in config_list]
    groups: dict[str, list[Any]] = {}
    order: list[str] = []
    for key, identity in zip(keys, identities):
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(identity)
    n_proposals = len(config_list)
    n_unique = len(groups)
    return {
        "n_proposals": n_proposals,
        "n_unique_coordinates": n_unique,
        "n_duplicate_proposals": n_proposals - n_unique,
        "canonical_key_by_proposal": [
            {"identity": identity, "canonical_coordinate_key": key} for identity, key in zip(identities, keys)
        ],
        "multiplicities": {key: len(groups[key]) for key in order},
        "duplicate_groups": {key: list(groups[key]) for key in order if len(groups[key]) > 1},
    }


# ---------------------------------------------------------------------------
# J. Exact GPU-hour / compute-coverage semantics (RD1 §5).
# ---------------------------------------------------------------------------


def _is_missing_compute(value: Any) -> bool:
    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def exact_compute_scope_summary(exact_gpu_hours: Sequence[Any]) -> dict[str, Any]:
    """Exact-compute coverage for one scope (RD1 §5).

    Consumes an explicitly upstream-qualified exact-GPU-hour value per
    configuration; ``None`` / non-finite marks a configuration with no exact
    accounting. Missing exact GPU-hours are **never** silently treated as
    zero: ``authoritative_exact_gpu_hours`` is ``NaN`` unless coverage is
    complete, and the partial sum is labelled as such.
    """
    values = list(exact_gpu_hours)
    total = len(values)
    present = [float(v) for v in values if not _is_missing_compute(v)]
    covered = len(present)
    complete = total > 0 and covered == total
    known_sum = float(sum(present))
    return {
        "covered_configuration_count": covered,
        "total_configuration_count": total,
        "coverage_fraction": (covered / total) if total else float("nan"),
        "coverage_complete": bool(complete),
        "known_partial_gpu_hours_sum": known_sum,
        "authoritative_exact_gpu_hours": known_sum if complete else float("nan"),
    }


def exact_compute_coverage_by_prefix(exact_gpu_hours_in_order: Sequence[Any]) -> pd.DataFrame:
    """Prefix-cumulative exact-compute coverage in proposal order (RD1 §5).

    One row per prefix length. ``cumulative_exact_gpu_hours`` is the
    authoritative cumulative exact GPU-hours only when that prefix has
    complete coverage, otherwise ``NaN`` (``known_partial_gpu_hours_sum``
    still exposes the partial sum). Objective / geometry analyses stay
    usable when this information is absent.
    """
    values = list(exact_gpu_hours_in_order)
    rows: list[dict[str, Any]] = []
    for prefix_length in range(1, len(values) + 1):
        summary = exact_compute_scope_summary(values[:prefix_length])
        rows.append(
            {
                "prefix_length": prefix_length,
                "covered_configuration_count": summary["covered_configuration_count"],
                "total_configuration_count": summary["total_configuration_count"],
                "coverage_fraction": summary["coverage_fraction"],
                "coverage_complete": summary["coverage_complete"],
                "known_partial_gpu_hours_sum": summary["known_partial_gpu_hours_sum"],
                "cumulative_exact_gpu_hours": summary["authoritative_exact_gpu_hours"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H. Post-incumbent descriptors (RD1 §4), first-achieved final incumbent.
# ---------------------------------------------------------------------------


def first_achieved_final_incumbent_index(objectives: Sequence[float]) -> int | None:
    """Earliest proposal position attaining the arm's final maximum objective
    (RD1-C2 §H). ``None`` for an empty sequence."""
    scores = [float(o) for o in objectives]
    if not scores:
        return None
    final_max = max(scores)
    for i, score in enumerate(scores):
        if score == final_max:
            return i
    return None  # pragma: no cover - unreachable, max() is always present


def post_incumbent_descriptors(
    objectives: Sequence[float],
    *,
    configs: Sequence[Mapping[str, Any]] | None = None,
    exact_gpu_hours: Sequence[Any] | None = None,
    identities: Sequence[Any] | None = None,
    anchor: float = P2_NEAR_INCUMBENT_ANCHOR,
) -> dict[str, Any]:
    """Factual summaries of the configurations evaluated **strictly after**
    an arm first achieved its eventual best observed objective (RD1-C2 §H).

    Answers: after this arm had already found its eventual best observed
    score, what did continued search yield? No arbitrary early/late
    pseudo-replicates are created.
    """
    scores = [float(o) for o in objectives]
    n = len(scores)
    if identities is None:
        identities = list(range(n))
    incumbent_index = first_achieved_final_incumbent_index(scores)
    subsequent = list(range(incumbent_index + 1, n)) if incumbent_index is not None else []
    subsequent_scores = [scores[i] for i in subsequent]
    subsequent_identities = [identities[i] for i in subsequent]

    result: dict[str, Any] = {
        "final_incumbent_position": incumbent_index,
        "final_incumbent_identity": None if incumbent_index is None else identities[incumbent_index],
        "final_incumbent_objective": None if incumbent_index is None else scores[incumbent_index],
        "n_subsequent_proposals": len(subsequent),
        "subsequent_objective_distribution": _distribution_summary(subsequent_scores),
        "subsequent_objective_top3": top_k_summary(subsequent_scores, k=3, identities=subsequent_identities),
        "subsequent_near_incumbent_primary": near_incumbent_yield(
            subsequent_scores, anchor=anchor, margin=NEAR_INCUMBENT_PRIMARY_MARGIN, identities=subsequent_identities
        ),
        "subsequent_near_incumbent_sensitivity": near_incumbent_yield(
            subsequent_scores, anchor=anchor, margin=NEAR_INCUMBENT_SENSITIVITY_MARGIN, identities=subsequent_identities
        ),
    }

    if configs is not None:
        config_list = list(configs)
        subsequent_keys = [canonical_coordinate_key(config_list[i]) for i in subsequent]
        prefix_keys = (
            {canonical_coordinate_key(config_list[i]) for i in range(incumbent_index + 1)}
            if incumbent_index is not None
            else set()
        )
        result["n_subsequent_unique_coordinates"] = len(set(subsequent_keys))
        result["n_subsequent_new_coordinates"] = len({key for key in subsequent_keys if key not in prefix_keys})

    if exact_gpu_hours is not None:
        gpu_list = list(exact_gpu_hours)
        summary = exact_compute_scope_summary([gpu_list[i] for i in subsequent])
        result["subsequent_exact_compute"] = summary
        # Exact additional GPU-hours only when the subsequent scope's exact
        # compute coverage is complete (RD1 §5); otherwise NaN, never zero.
        result["subsequent_exact_gpu_hours"] = summary["authoritative_exact_gpu_hours"]

    return result


# ---------------------------------------------------------------------------
# Consumer contract: validation + top-level analysis.
# ---------------------------------------------------------------------------


def _identity_is_missing(value: Any) -> bool:
    """True iff an identity cell is an explicit missing marker (``None`` /
    :data:`pandas.NA` / floating ``NaN``) rather than a real identity."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - non-scalar identity
        return False


def _require_identity(value: Any, *, field: str, position: int) -> Any:
    """Return ``value`` unchanged if it is a present, non-blank identity;
    otherwise raise :class:`ReviewTableContractError`.

    Malformed / missing identities are never silently stringified
    (RD1-C2 Finding D): a downstream consumer must be able to trust that
    every row carries a genuine ``proposal_id`` and ``configuration_id``.
    """
    if _identity_is_missing(value):
        raise ReviewTableContractError(
            f"row at position {position}: {field} is required and must be present"
        )
    if isinstance(value, str) and value.strip() == "":
        raise ReviewTableContractError(
            f"row at position {position}: {field} is required and must be non-empty"
        )
    return value


def _validate_exact_gpu_hours_value(value: Any, *, position: int) -> float:
    """Validate one ``exact_gpu_hours`` cell at the authoritative review-table
    boundary (RD1-C2 Finding A).

    Returns :data:`math.nan` for an explicit *unavailable* marker -- ``None``,
    :data:`pandas.NA`, or a floating ``NaN``. A **populated** value must be a
    genuine finite, non-negative numeric scalar; a ``bool`` (even though
    Python treats it as numeric), ``+inf`` / ``-inf``, a negative number, a
    numeric-looking string such as ``"2.5"``, and any other populated
    non-numeric value fail the consumer contract clearly. A malformed
    populated value is **never** treated as equivalent to missing
    accounting -- downstream compute functions rely on this invariant.
    """
    try:
        is_missing_marker = bool(pd.isna(value))
    except (TypeError, ValueError):
        is_missing_marker = False
    if value is None or is_missing_marker:
        return math.nan
    if isinstance(value, bool):
        raise ReviewTableContractError(
            f"row at position {position}: exact_gpu_hours must not be a bool, got {value!r}"
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ReviewTableContractError(
            f"row at position {position}: exact_gpu_hours must be a numeric scalar or an "
            f"explicit missing marker, got {value!r}"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ReviewTableContractError(
            f"row at position {position}: exact_gpu_hours must be finite, got {value!r}"
        )
    if numeric < 0.0:
        raise ReviewTableContractError(
            f"row at position {position}: exact_gpu_hours must be >= 0, got {value!r}"
        )
    return numeric


def validate_review_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    objective_column: str = DEFAULT_OBJECTIVE_COLUMN,
    exact_gpu_hours_column: str = DEFAULT_EXACT_GPU_HOURS_COLUMN,
) -> pd.DataFrame:
    """Validate an in-memory review table against the RD1-C2 consumer
    contract and return an enriched copy.

    Required columns: :data:`REQUIRED_COLUMNS` plus ``objective_column``
    (the official configuration objective -- made explicit here, never
    silently switched between score columns). ``exact_gpu_hours_column`` is
    optional and, when absent, every configuration is treated as having
    **no** exact accounting (``NaN``, never zero).

    Added columns: ``_objective`` (float), ``_exact_gpu_hours`` (float;
    ``NaN`` marks a configuration with no exact accounting),
    ``_canonical_coordinate_key``, and ``coord_<axis>`` for each axis in
    :data:`SIX_AXIS_ORDER`.

    A malformed row raises :class:`ReviewTableContractError`:

    * missing required column, unknown ``search_arm``;
    * ``proposal_order`` not a positive integer, or not unique within an arm;
    * ``proposal_id`` / ``configuration_id`` missing or blank (never silently
      stringified), or ``proposal_id`` not unique within an arm
      (RD1-C2 Finding D);
    * one ``configuration_id`` associated with two different authoritative
      canonical six-axis coordinates anywhere in the table (RD1-C2 Finding D)
      -- distinct config ids that share a canonical coordinate stay legal and
      still count as duplicate-coordinate proposals;
    * illegal six-axis coordinate;
    * non-finite / non-numeric objective;
    * a **populated** ``exact_gpu_hours`` value that is a bool, ``+inf`` /
      ``-inf``, negative, a numeric-looking string, or otherwise non-numeric
      (RD1-C2 Finding A) -- ``None`` / ``pd.NA`` / ``NaN`` remain accepted as
      "unavailable" and never invalidate the objective / geometry analysis.
    """
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(list(table))
    frame = frame.copy().reset_index(drop=True)

    missing_columns = [column for column in (*REQUIRED_COLUMNS, objective_column) if column not in frame.columns]
    if missing_columns:
        raise ReviewTableContractError(f"review table missing required columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ReviewTableContractError("review table has no rows")

    unknown_arms = sorted({str(arm) for arm in frame["search_arm"]} - set(SEARCH_ARMS_V2))
    if unknown_arms:
        raise ReviewTableContractError(f"review table has unknown search_arm value(s): {unknown_arms}")

    # proposal_order: positive integer, unique within arm. Checked first so an
    # ordering fault is reported before per-row identity / geometry faults.
    for arm, arm_frame in frame.groupby("search_arm", sort=True):
        orders: list[int] = []
        for order in arm_frame["proposal_order"].tolist():
            if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or int(order) < 1:
                raise ReviewTableContractError(
                    f"{arm}: proposal_order must be a positive integer, got {order!r}"
                )
            orders.append(int(order))
        if len(set(orders)) != len(orders):
            raise ReviewTableContractError(f"{arm}: proposal_order values must be unique within the arm")

    has_exact_gpu_column = exact_gpu_hours_column in frame.columns
    n_rows = len(frame)
    arm_values = frame["search_arm"].tolist()
    proposal_id_values = frame["proposal_id"].tolist()
    configuration_id_values = frame["configuration_id"].tolist()
    objective_values = frame[objective_column].tolist()
    axis_values = {axis: frame[axis].tolist() for axis in SIX_AXIS_ORDER}
    gpu_values = frame[exact_gpu_hours_column].tolist() if has_exact_gpu_column else [None] * n_rows

    canonical_keys: list[str] = []
    coordinate_columns: dict[str, list[float]] = {f"coord_{axis}": [] for axis in SIX_AXIS_ORDER}
    objectives: list[float] = []
    exact_gpu_hours: list[float] = []
    per_arm_proposal_ids: dict[str, list[Any]] = {}
    configuration_id_to_key: dict[Any, str] = {}

    for position in range(n_rows):
        arm = str(arm_values[position])
        proposal_id = _require_identity(
            proposal_id_values[position], field="proposal_id", position=position
        )
        configuration_id = _require_identity(
            configuration_id_values[position], field="configuration_id", position=position
        )
        per_arm_proposal_ids.setdefault(arm, []).append(proposal_id)

        config = {axis: axis_values[axis][position] for axis in SIX_AXIS_ORDER}
        try:
            canonical = canonical_configuration(config)
        except ValueError as exc:  # SweepV2CampaignError and v1 domain ValueError
            raise ReviewTableContractError(
                f"row proposal_id={proposal_id!r}: illegal six-axis configuration: {exc}"
            ) from exc
        key = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        canonical_keys.append(key)
        for axis, value in zip(SIX_AXIS_ORDER, normalized_coordinates(config)):
            coordinate_columns[f"coord_{axis}"].append(float(value))

        prior_key = configuration_id_to_key.get(configuration_id)
        if prior_key is not None and prior_key != key:
            raise ReviewTableContractError(
                f"configuration_id={configuration_id!r} is associated with two different "
                f"authoritative canonical six-axis coordinates in the supplied review table"
            )
        configuration_id_to_key[configuration_id] = key

        objective_value = objective_values[position]
        if (
            objective_value is None
            or isinstance(objective_value, bool)
            or not isinstance(objective_value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(objective_value))
        ):
            raise ReviewTableContractError(
                f"row proposal_id={proposal_id!r}: {objective_column} must be a finite number, "
                f"got {objective_value!r}"
            )
        objectives.append(float(objective_value))

        exact_gpu_hours.append(
            _validate_exact_gpu_hours_value(gpu_values[position], position=position)
            if has_exact_gpu_column
            else math.nan
        )

    for arm, proposal_ids in per_arm_proposal_ids.items():
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ReviewTableContractError(f"{arm}: proposal_id values must be unique within the arm")

    frame["_objective"] = objectives
    frame["_canonical_coordinate_key"] = canonical_keys
    for column, column_values in coordinate_columns.items():
        frame[column] = column_values
    frame["_exact_gpu_hours"] = exact_gpu_hours
    return frame


def _pooled_near_incumbent(frame: pd.DataFrame, *, anchor: float, margin: float) -> dict[str, Any]:
    """Authoritative arm-neutral pooled near-incumbent view over the **whole**
    supplied review table for one frozen P2 margin (RD1-C2 Finding B).

    This is a static descriptive membership summary. It deliberately does
    **not** define a pooled proposal ordering, a pooled cumulative sequence,
    or any cross-arm search semantics -- ``proposal_order`` is retained only
    as row provenance so a downstream C5/C6 consumer never has to concatenate
    arm internals or rerun a lower-level threshold helper. A configuration
    whose objective exceeds P2 is still counted (one-sided near-or-better
    floor ``objective >= anchor - margin``).
    """
    performance_floor = float(anchor) - float(margin)
    membership: list[dict[str, Any]] = []
    n_near = 0
    for position in range(len(frame)):
        row = frame.iloc[position]
        objective = float(row["_objective"])
        near = bool(objective >= performance_floor)
        n_near += int(near)
        membership.append(
            {
                "search_arm": str(row["search_arm"]),
                "proposal_id": row["proposal_id"],
                "configuration_id": row["configuration_id"],
                "proposal_order": int(row["proposal_order"]),
                "objective": objective,
                "near_or_better": near,
            }
        )
    n_total = len(membership)
    return {
        "anchor": float(anchor),
        "margin": float(margin),
        "performance_floor": performance_floor,
        "margin_kind": "descriptive_one_sided_near_or_better",
        "scope": "pooled_across_arms",
        "n_total": n_total,
        "n_near_or_better": n_near,
        "fraction_near_or_better": (n_near / n_total) if n_total else float("nan"),
        "near_or_better_members": [dict(entry) for entry in membership if entry["near_or_better"]],
        "membership": membership,
    }


@dataclass
class ArmConfigurationAnalysis:
    """Per-arm configuration-level analysis result.

    ``novelty`` and ``incumbent_distance_frame`` are populated for the
    Bayesian arm only; they are ``None`` for the random-control arm, which
    carries no optimizer-learning interpretation (RD1 §6.3).
    """

    search_arm: str
    per_proposal: pd.DataFrame
    objective_distribution: dict[str, Any]
    top3: dict[str, Any]
    near_incumbent_primary: dict[str, Any]
    near_incumbent_sensitivity: dict[str, Any]
    duplicate_coordinate_accounting: dict[str, Any]
    exact_compute_coverage_by_prefix: pd.DataFrame
    exact_compute_scope_summary: dict[str, Any]
    post_incumbent: dict[str, Any]
    #: True iff the review table supplied an ``exact_gpu_hours`` column at
    #: all. When False, every proposal-aligned / champion / top-k exact
    #: compute field is reported as unavailable (``NaN``), never zero.
    exact_compute_available: bool
    #: Arm champion-discovery compute: compute expended through discovery of
    #: the eventual arm champion, first-achieved semantics (RD1-C2 C3).
    champion_discovery_compute: dict[str, Any]
    #: Threshold-free top-k discovery compute companion (RD1-C2 C4).
    top_k_discovery_compute: dict[str, Any]
    novelty: np.ndarray | None = None
    incumbent_distance_frame: pd.DataFrame | None = None


@dataclass
class ReviewAnalysis:
    """Top-level RD1-C2 analysis over one qualified review table."""

    objective_column: str
    near_incumbent_anchor: float
    arms: dict[str, ArmConfigurationAnalysis]
    #: Frozen pooled (arm-neutral, whole-table) near-incumbent results for the
    #: primary (``0.01``) and sensitivity (``0.005``) P2 margins
    #: (RD1-C2 Finding B). Static descriptive membership only -- no pooled
    #: ordering or pooled cumulative sequence.
    pooled_near_incumbent_primary: dict[str, Any]
    pooled_near_incumbent_sensitivity: dict[str, Any]
    #: True iff the review table supplied an ``exact_gpu_hours`` column.
    exact_compute_available: bool

    def arm(self, name: str) -> ArmConfigurationAnalysis:
        return self.arms[name]

    def pooled_near_incumbent(self, margin_kind: str) -> dict[str, Any]:
        """``"primary"`` / ``"sensitivity"`` -> the frozen pooled result."""
        if margin_kind == "primary":
            return self.pooled_near_incumbent_primary
        if margin_kind == "sensitivity":
            return self.pooled_near_incumbent_sensitivity
        raise KeyError(margin_kind)


def _analyze_arm(
    arm: str, arm_frame: pd.DataFrame, anchor: float, *, has_exact_gpu_column: bool
) -> ArmConfigurationAnalysis:
    n = len(arm_frame)
    objectives = [float(v) for v in arm_frame["_objective"].tolist()]
    proposal_ids = arm_frame["proposal_id"].tolist()
    proposal_orders = [int(v) for v in arm_frame["proposal_order"].tolist()]
    configuration_ids = arm_frame["configuration_id"].tolist()
    canonical_keys = arm_frame["_canonical_coordinate_key"].tolist()
    exact_gpu_hours = arm_frame["_exact_gpu_hours"].tolist()
    exact_gpu = [float(v) for v in exact_gpu_hours]  # NaN marks "no exact accounting"
    configs = [{axis: arm_frame.iloc[i][axis] for axis in SIX_AXIS_ORDER} for i in range(n)]
    coordinate_columns = [f"coord_{axis}" for axis in SIX_AXIS_ORDER]
    coords = [arm_frame.iloc[i][coordinate_columns].to_numpy(dtype=float) for i in range(n)]

    cumulative_best = cumulative_best_by_order(objectives)
    running_best_before = np.concatenate([[-np.inf], np.asarray(cumulative_best, dtype=float)[:-1]])
    is_new_incumbent = [bool(objectives[i] > running_best_before[i]) for i in range(n)]
    cumulative_near_primary = cumulative_near_incumbent_count(
        objectives, anchor=anchor, margin=NEAR_INCUMBENT_PRIMARY_MARGIN
    )
    cumulative_near_sensitivity = cumulative_near_incumbent_count(
        objectives, anchor=anchor, margin=NEAR_INCUMBENT_SENSITIVITY_MARGIN
    )

    # Proposal-identified exact compute (RD1-C2 C1). The cumulative exact
    # GPU-hours through a proposal are authoritative only while EVERY exact
    # value in that proposal's prefix is available; once a gap appears the
    # cumulative total stays unavailable (NaN) -- it never "resumes".
    prefix_exact_complete: list[bool] = []
    cumulative_exact_gpu_hours: list[float] = []
    running_exact = 0.0
    prefix_ok = True
    for value in exact_gpu:
        if math.isnan(value):
            prefix_ok = False
        elif prefix_ok:
            running_exact += value
        prefix_exact_complete.append(prefix_ok)
        cumulative_exact_gpu_hours.append(running_exact if prefix_ok else math.nan)

    def _cumulative_exact_through(position: int | None) -> tuple[bool, float]:
        """(prefix-complete?, authoritative cumulative exact GPU-hours through
        ``position`` inclusive) -- NaN unless that whole prefix is covered."""
        if position is None:
            return (False, math.nan)
        complete = prefix_exact_complete[position]
        return (complete, cumulative_exact_gpu_hours[position] if complete else math.nan)

    def _near_incumbent_yield_per_gpu_hour(cumulative_counts: Sequence[Any]) -> list[float]:
        """Transparent cumulative near-incumbent count / cumulative exact
        GPU-hours (RD1-C2 C2). Unavailable (NaN) when the cumulative exact
        compute is not authoritative or is zero -- never infinite / fabricated.
        Descriptive only: not a threshold, objective, ranking, or promotion rule.
        """
        out: list[float] = []
        for count, gpu in zip(cumulative_counts, cumulative_exact_gpu_hours):
            out.append((int(count) / gpu) if (not math.isnan(gpu) and gpu > 0.0) else math.nan)
        return out

    per_proposal = pd.DataFrame(
        {
            "search_arm": arm,
            "proposal_order": proposal_orders,
            "proposal_id": proposal_ids,
            "configuration_id": configuration_ids,
            "objective": objectives,
            "canonical_coordinate_key": canonical_keys,
            "cumulative_best": list(np.asarray(cumulative_best, dtype=float)),
            "is_new_incumbent": is_new_incumbent,
            "cumulative_near_incumbent_primary": [int(v) for v in cumulative_near_primary],
            "cumulative_near_incumbent_sensitivity": [int(v) for v in cumulative_near_sensitivity],
        }
    )
    for axis, column in zip(SIX_AXIS_ORDER, coordinate_columns):
        per_proposal[f"coord_{axis}"] = arm_frame[column].tolist()

    # Proposal-aligned exact compute + near-incumbent-yield-per-GPU-hour, all
    # keyed to the retained proposal identity/order (RD1-C2 C1/C2). When the
    # review table carried no exact_gpu_hours column these are all NaN /
    # False (unavailable), never zero.
    per_proposal["exact_gpu_hours"] = exact_gpu
    per_proposal["cumulative_exact_gpu_hours"] = cumulative_exact_gpu_hours
    per_proposal["exact_gpu_hours_prefix_complete"] = prefix_exact_complete
    per_proposal["cumulative_near_incumbent_primary_yield_per_gpu_hour"] = _near_incumbent_yield_per_gpu_hour(
        cumulative_near_primary
    )
    per_proposal["cumulative_near_incumbent_sensitivity_yield_per_gpu_hour"] = _near_incumbent_yield_per_gpu_hour(
        cumulative_near_sensitivity
    )

    novelty: np.ndarray | None = None
    incumbent_distance_frame: pd.DataFrame | None = None
    if arm == BAYESIAN_ARM:
        novelty = bayesian_novelty_series(coords)
        incumbent_distance_frame = bayesian_incumbent_distance_frame(
            coords, objectives, proposal_ids=proposal_ids, proposal_orders=proposal_orders
        )
        per_proposal["bayesian_novelty"] = list(novelty)
        per_proposal["incumbent_proposal_order_before"] = incumbent_distance_frame[
            "incumbent_proposal_order_before"
        ].tolist()
        per_proposal["incumbent_proposal_id_before"] = incumbent_distance_frame[
            "incumbent_proposal_id_before"
        ].tolist()
        per_proposal["incumbent_distance"] = incumbent_distance_frame["incumbent_distance"].tolist()

    # Champion-discovery compute (RD1-C2 C3): compute expended through
    # discovery of the eventual arm champion, first-achieved semantics. The
    # cumulative exact GPU-hours are authoritative only if the required
    # proposal prefix has complete exact accounting.
    champion_position = first_achieved_final_incumbent_index(objectives)
    champion_prefix_complete, champion_cumulative_gpu = _cumulative_exact_through(champion_position)
    champion_discovery_compute = {
        "exact_compute_available": bool(has_exact_gpu_column),
        "champion_position": champion_position,
        "champion_proposal_id": None if champion_position is None else proposal_ids[champion_position],
        "champion_proposal_order": None if champion_position is None else proposal_orders[champion_position],
        "champion_configuration_id": (
            None if champion_position is None else configuration_ids[champion_position]
        ),
        "champion_objective": None if champion_position is None else objectives[champion_position],
        "prefix_complete_through_champion": bool(champion_prefix_complete),
        "cumulative_exact_gpu_hours_through_champion": champion_cumulative_gpu,
    }

    # Threshold-free top-k discovery compute (RD1-C2 C4). Same ranking as
    # ``top_k_summary`` (objective desc, ties broken by earlier proposal
    # order). "Through discovery of the complete selected final top-k set" =
    # cumulative compute through the latest proposal-order occurrence among
    # the selected members; authoritative only when that prefix is complete.
    top_k = 3
    ranked_positions = sorted(range(n), key=lambda i: (-objectives[i], i))
    selected_positions = ranked_positions[:top_k]
    top_k_members = [
        {
            "proposal_id": proposal_ids[pos],
            "proposal_order": proposal_orders[pos],
            "configuration_id": configuration_ids[pos],
            "objective": objectives[pos],
            "position": pos,
            "exact_gpu_hours": exact_gpu[pos],
            "cumulative_exact_gpu_hours_at_discovery": cumulative_exact_gpu_hours[pos],
            "prefix_complete_at_discovery": bool(prefix_exact_complete[pos]),
        }
        for pos in selected_positions
    ]
    latest_discovery_position = max(selected_positions) if selected_positions else None
    full_set_prefix_complete, full_set_cumulative_gpu = _cumulative_exact_through(latest_discovery_position)
    top_k_discovery_compute = {
        "exact_compute_available": bool(has_exact_gpu_column),
        "requested_k": top_k,
        "n_selected": len(selected_positions),
        "is_complete": n >= top_k,
        "members": top_k_members,
        "latest_discovery_position": latest_discovery_position,
        "latest_discovery_proposal_order": (
            None if latest_discovery_position is None else proposal_orders[latest_discovery_position]
        ),
        "prefix_complete_through_full_top_k_discovery": bool(full_set_prefix_complete),
        "cumulative_exact_gpu_hours_through_full_top_k_discovery": full_set_cumulative_gpu,
    }

    return ArmConfigurationAnalysis(
        search_arm=arm,
        per_proposal=per_proposal,
        objective_distribution=_distribution_summary(objectives),
        top3=top_k_summary(objectives, k=3, identities=proposal_ids),
        near_incumbent_primary=near_incumbent_yield(
            objectives, anchor=anchor, margin=NEAR_INCUMBENT_PRIMARY_MARGIN, identities=proposal_ids
        ),
        near_incumbent_sensitivity=near_incumbent_yield(
            objectives, anchor=anchor, margin=NEAR_INCUMBENT_SENSITIVITY_MARGIN, identities=proposal_ids
        ),
        duplicate_coordinate_accounting=duplicate_coordinate_accounting(configs, identities=proposal_ids),
        exact_compute_coverage_by_prefix=exact_compute_coverage_by_prefix(exact_gpu_hours),
        exact_compute_scope_summary=exact_compute_scope_summary(exact_gpu_hours),
        post_incumbent=post_incumbent_descriptors(
            objectives,
            configs=configs,
            exact_gpu_hours=exact_gpu_hours,
            identities=proposal_ids,
            anchor=anchor,
        ),
        exact_compute_available=bool(has_exact_gpu_column),
        champion_discovery_compute=champion_discovery_compute,
        top_k_discovery_compute=top_k_discovery_compute,
        novelty=novelty,
        incumbent_distance_frame=incumbent_distance_frame,
    )


def analyze_review_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    objective_column: str = DEFAULT_OBJECTIVE_COLUMN,
    exact_gpu_hours_column: str = DEFAULT_EXACT_GPU_HOURS_COLUMN,
    near_incumbent_anchor: float = P2_NEAR_INCUMBENT_ANCHOR,
) -> ReviewAnalysis:
    """Run the full RD1-C2 configuration-level analysis over one qualified
    in-memory review table.

    This is the RD1-C2 consumer-contract entrypoint. It validates the table
    (:func:`validate_review_table`), then produces one
    :class:`ArmConfigurationAnalysis` per present ``search_arm``. Each arm's
    proposal sequence is analysed whole, in ``proposal_order`` -- no
    early/middle/late blocks. Random-control order is treated descriptively
    for realized coverage / yield / compute and is never given a Bayesian
    incumbent-learning interpretation.
    """
    frame = validate_review_table(
        table, objective_column=objective_column, exact_gpu_hours_column=exact_gpu_hours_column
    )
    has_exact_gpu_column = exact_gpu_hours_column in frame.columns
    arms: dict[str, ArmConfigurationAnalysis] = {}
    for arm in sorted(frame["search_arm"].unique()):
        arm_frame = frame[frame["search_arm"] == arm].sort_values("proposal_order").reset_index(drop=True)
        arms[str(arm)] = _analyze_arm(
            str(arm), arm_frame, near_incumbent_anchor, has_exact_gpu_column=has_exact_gpu_column
        )
    return ReviewAnalysis(
        objective_column=objective_column,
        near_incumbent_anchor=float(near_incumbent_anchor),
        arms=arms,
        pooled_near_incumbent_primary=_pooled_near_incumbent(
            frame, anchor=near_incumbent_anchor, margin=NEAR_INCUMBENT_PRIMARY_MARGIN
        ),
        pooled_near_incumbent_sensitivity=_pooled_near_incumbent(
            frame, anchor=near_incumbent_anchor, margin=NEAR_INCUMBENT_SENSITIVITY_MARGIN
        ),
        exact_compute_available=has_exact_gpu_column,
    )
