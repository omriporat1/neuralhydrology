"""Full-percentile diagnostic layer over per-basin raw-space metrics
(Stage 1 validation and optimization foundation, Part A).

``aggregate_raw_space_metrics`` (``nh_raw_space_evaluation.py``) reports only
median/mean/q25/q75 per metric -- sufficient for checkpoint ranking, but not
for inspecting the *shape* of the cross-basin metric distribution across
epochs. This module adds the full percentile grid
(p1/p5/p10/p25/p50/p75/p90/p95/p99), sign fractions, and finite-basin counts
required by the validation-report policy
(``docs/stage1_scientific_baseline_design.md``), computed purely from an
already-produced per-basin metric array -- it never recomputes a metric
value itself and never touches NH output directly, so it is usable both
against real per-basin arrays (e.g. from
``nh_seed_evaluation.raw_space_metrics_for_run_period``) and in unit tests
against synthetic arrays.

All interpretation here is intentionally numeric-only (deltas, monotonic
direction, rank stability) -- this module never infers a hydrological cause
for a percentile shift; that is a human/strategic-review judgment, not code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "PERCENTILES",
    "PercentileTable",
    "compute_percentile_table",
    "build_epoch_percentile_tables",
    "percentile_change_table",
    "basin_consistency_diagnostic",
]

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


@dataclass(frozen=True)
class PercentileTable:
    """One epoch's full percentile summary for one metric."""

    metric_name: str
    n_total_basins: int
    n_finite_basins: int
    percentiles: dict = field(default_factory=dict)
    min: float = float("nan")
    max: float = float("nan")
    frac_gt_0: float = float("nan")
    frac_gt_0p5: float = float("nan")
    frac_lt_0: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "n_total_basins": self.n_total_basins,
            "n_finite_basins": self.n_finite_basins,
            **self.percentiles,
            "min": self.min,
            "max": self.max,
            "frac_gt_0": self.frac_gt_0,
            "frac_gt_0p5": self.frac_gt_0p5,
            "frac_lt_0": self.frac_lt_0,
        }


def compute_percentile_table(values, *, metric_name: str = "nse") -> PercentileTable:
    """Compute the full percentile grid + sign fractions for one 1-D array
    of per-basin metric values. Non-finite entries (NaN/+-inf) are excluded
    from every statistic but counted via ``n_total_basins - n_finite_basins``.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n_total = int(arr.size)
    finite = arr[np.isfinite(arr)]
    n_finite = int(finite.size)
    if n_finite == 0:
        return PercentileTable(
            metric_name=metric_name,
            n_total_basins=n_total,
            n_finite_basins=0,
            percentiles={f"p{p}": float("nan") for p in PERCENTILES},
        )
    percentiles = {f"p{p}": float(np.percentile(finite, p)) for p in PERCENTILES}
    return PercentileTable(
        metric_name=metric_name,
        n_total_basins=n_total,
        n_finite_basins=n_finite,
        percentiles=percentiles,
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        frac_gt_0=float(np.mean(finite > 0.0)),
        frac_gt_0p5=float(np.mean(finite > 0.5)),
        frac_lt_0=float(np.mean(finite < 0.0)),
    )


def build_epoch_percentile_tables(
    per_epoch_values: Mapping[int, Sequence[float]], *, metric_name: str = "nse"
) -> dict:
    """``per_epoch_values``: ``{epoch: array-like of per-basin metric values}``.

    Returns ``{epoch: PercentileTable.to_dict()}``, keys sorted ascending.
    """
    return {
        int(epoch): compute_percentile_table(values, metric_name=metric_name).to_dict()
        for epoch, values in sorted(per_epoch_values.items(), key=lambda kv: kv[0])
    }


def percentile_change_table(epoch_tables: Mapping[int, Mapping]) -> list:
    """Epoch-over-epoch numeric deltas for each percentile track, plus a
    whole-sequence monotonic-direction classification. Purely descriptive:
    reports the sign/magnitude of change only, never a hydrological cause.

    ``epoch_tables``: ``{epoch: PercentileTable.to_dict()}`` as returned by
    :func:`build_epoch_percentile_tables`.

    Returns a list of rows (one per percentile track), each with the ordered
    epochs, the ordered percentile values, consecutive deltas, and
    ``monotonic`` in {"increasing", "decreasing", "constant",
    "non_monotonic", "undefined"} ("undefined" when fewer than two finite
    values are available to compare).
    """
    epochs = sorted(epoch_tables.keys())
    rows = []
    for p in PERCENTILES:
        key = f"p{p}"
        series = [epoch_tables[e][key] for e in epochs]
        deltas = []
        for i in range(len(series) - 1):
            a, b = series[i], series[i + 1]
            deltas.append(b - a if np.isfinite(a) and np.isfinite(b) else float("nan"))
        finite_deltas = [d for d in deltas if np.isfinite(d)]
        if not finite_deltas:
            monotonic = "undefined"
        elif all(d > 0 for d in finite_deltas):
            monotonic = "increasing"
        elif all(d < 0 for d in finite_deltas):
            monotonic = "decreasing"
        elif all(d == 0 for d in finite_deltas):
            monotonic = "constant"
        else:
            monotonic = "non_monotonic"
        rows.append(
            {
                "percentile": key,
                "epochs": epochs,
                "values": series,
                "deltas": deltas,
                "monotonic": monotonic,
            }
        )
    return rows


def _rankdata_average(arr: np.ndarray) -> np.ndarray:
    """Average-tie ranks (1-indexed), pure numpy -- avoids a scipy dependency
    for this one diagnostic. Equivalent to ``scipy.stats.rankdata`` default."""
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    n = len(arr)
    while i < n:
        j = i
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    ra, rb = _rankdata_average(a), _rankdata_average(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def basin_consistency_diagnostic(per_epoch_basin_values: Mapping[int, Mapping[str, float]]) -> list:
    """Optional Part A diagnostic: for each pair of consecutive epochs, the
    fraction of basins whose per-basin metric value stays in the same
    quartile of that epoch's own cross-basin distribution (quartile
    boundaries recomputed fresh per epoch, never fixed globally), plus the
    Spearman rank correlation between the two epochs' per-basin values.

    ``per_epoch_basin_values``: ``{epoch: {basin_id: value}}``. A basin
    absent from (or non-finite in) either side of one pair is excluded from
    that pair only -- never imputed, never excluded from other pairs.
    """
    epochs = sorted(per_epoch_basin_values.keys())
    rows = []
    for e0, e1 in zip(epochs[:-1], epochs[1:]):
        v0, v1 = per_epoch_basin_values[e0], per_epoch_basin_values[e1]
        common = sorted(set(v0.keys()) & set(v1.keys()))
        arr0 = np.array([v0[b] for b in common], dtype=np.float64)
        arr1 = np.array([v1[b] for b in common], dtype=np.float64)
        finite_mask = np.isfinite(arr0) & np.isfinite(arr1)
        arr0, arr1 = arr0[finite_mask], arr1[finite_mask]
        n = int(arr0.size)
        if n < 2:
            rows.append(
                {
                    "epoch_from": e0,
                    "epoch_to": e1,
                    "n_common_finite_basins": n,
                    "frac_same_quartile": float("nan"),
                    "spearman_r": float("nan"),
                }
            )
            continue

        edges0 = np.percentile(arr0, [25, 50, 75])
        edges1 = np.percentile(arr1, [25, 50, 75])
        q0 = np.digitize(arr0, edges0)
        q1 = np.digitize(arr1, edges1)
        frac_same = float(np.mean(q0 == q1))
        rows.append(
            {
                "epoch_from": e0,
                "epoch_to": e1,
                "n_common_finite_basins": n,
                "frac_same_quartile": frac_same,
                "spearman_r": _spearman_r(arr0, arr1),
            }
        )
    return rows
