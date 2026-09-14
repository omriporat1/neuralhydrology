"""RD1-C3 basin-analysis primitives for the frozen 12+12 scientific review
(``docs/stage1_v2_12plus12_review_design_v001.md`` §§8, 8.1, 9, 17, 19).

This module is a **pure descriptive analysis layer** over already-supplied
per-basin Common-120 fixed-support raw-space NSE values. It does not
establish that its inputs came from Common-120, does not qualify campaign
validity, does not select official best epochs, and does not open
``validation_results.p`` or any other campaign artifact. A future RD1-C4
consumer is responsible for producing qualified per-basin rows (shaped like
``evaluate_fixed_support_raw_space_metrics(...)["per_basin"]``) and handing
them to the entry points here.

The frozen seven-quantile core (Q1/Q5/Q25/Q50/Q75/Q95/Q99 + IQR) and its
paired-difference counterpart reuse
``src.baseline.percentile_diagnostics.compute_percentile_table`` for the
actual quantile arithmetic -- this module never reimplements percentile
interpolation.

Paired basin-difference classification uses ``NUMERICAL_TIE_TOLERANCE =
1e-12`` (design §8.1): a tiny machine-precision-scale tolerance for floating
point equality only, never a scientific NSE margin. It is unrelated to the
RD1 configuration-level 0.01 / 0.005 P2 near-incumbent margins, which never
apply to basin-level differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.baseline.percentile_diagnostics import PercentileTable, compute_percentile_table

__all__ = [
    "NUMERICAL_TIE_TOLERANCE",
    "BasinAnalysisContractError",
    "FrozenQuantileCore",
    "EcdfTable",
    "BasinDistributionResult",
    "PairedBasinDifferenceResult",
    "analyze_basin_distribution",
    "compute_ecdf_table",
    "analyze_paired_basin_difference",
]

NUMERICAL_TIE_TOLERANCE = 1e-12

_FROZEN_QUANTILE_LEVELS = (1, 5, 25, 50, 75, 95, 99)


class BasinAnalysisContractError(ValueError):
    """Raised for malformed basin identity/metric input or a fail-closed
    paired-comparison identity mismatch. Never silently coerced or repaired."""


@dataclass(frozen=True)
class FrozenQuantileCore:
    """The design-§8 frozen seven-quantile core, derived from a
    :class:`~src.baseline.percentile_diagnostics.PercentileTable`."""

    q1: float
    q5: float
    q25: float
    q50: float
    q75: float
    q95: float
    q99: float
    iqr: float

    @classmethod
    def from_percentile_table(cls, table: PercentileTable) -> "FrozenQuantileCore":
        p = table.percentiles
        q25, q75 = p["p25"], p["p75"]
        iqr = q75 - q25 if np.isfinite(q25) and np.isfinite(q75) else float("nan")
        return cls(
            q1=p["p1"], q5=p["p5"], q25=q25, q50=p["p50"], q75=q75, q95=p["p95"], q99=p["p99"], iqr=iqr,
        )

    def to_dict(self) -> dict:
        return {
            "q1": self.q1, "q5": self.q5, "q25": self.q25, "q50": self.q50,
            "q75": self.q75, "q95": self.q95, "q99": self.q99, "iqr": self.iqr,
        }


@dataclass(frozen=True)
class EcdfTable:
    """Deterministic right-continuous empirical CDF over finite values only.

    One row per unique sorted finite value. ``cumulative_fraction[i]`` equals
    the fraction of finite observations ``<= value[i]``; duplicate values
    produce a single jump of their full multiplicity.

    ``n_total`` / ``n_finite`` / ``n_nonfinite`` (``n_nonfinite = n_total -
    n_finite``) describe the full input population, so a standalone ECDF
    consumer never needs to retain or recount the source array to know its
    population coverage.
    """

    value: np.ndarray
    count: np.ndarray
    cumulative_count: np.ndarray
    cumulative_fraction: np.ndarray
    n_total: int
    n_finite: int
    n_nonfinite: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "value": self.value,
                "count": self.count,
                "cumulative_count": self.cumulative_count,
                "cumulative_fraction": self.cumulative_fraction,
            }
        )


def compute_ecdf_table(values: Sequence[Any]) -> EcdfTable:
    """Build the deterministic finite-only ECDF for ``values`` (design §8,
    "selected ECDF overlays").

    Every input scalar is validated through the same authoritative
    scalar/missing-value contract used by :func:`analyze_basin_distribution`
    (:func:`_coerce_metric_scalar`) -- booleans and strings (including
    numeric-looking strings) are rejected, recognized missing markers
    (``None`` / ``pd.NA``) become NaN and are counted as nonfinite, and
    genuine numeric scalars (including +/-infinity, which remains nonfinite)
    are accepted.
    """
    values_list = list(values)
    coerced = [
        _coerce_metric_scalar(v, context=f"ecdf input at index {i}") for i, v in enumerate(values_list)
    ]
    arr = np.asarray(coerced, dtype=np.float64).reshape(-1) if coerced else np.array([], dtype=np.float64)
    n_total = int(arr.size)
    finite = np.sort(arr[np.isfinite(arr)])
    n_finite = int(finite.size)
    n_nonfinite = n_total - n_finite
    if n_finite == 0:
        return EcdfTable(
            value=np.array([], dtype=np.float64),
            count=np.array([], dtype=np.int64),
            cumulative_count=np.array([], dtype=np.int64),
            cumulative_fraction=np.array([], dtype=np.float64),
            n_total=n_total,
            n_finite=0,
            n_nonfinite=n_nonfinite,
        )
    unique_values, counts = np.unique(finite, return_counts=True)
    cumulative_count = np.cumsum(counts)
    cumulative_fraction = cumulative_count.astype(np.float64) / n_finite
    return EcdfTable(
        value=unique_values,
        count=counts.astype(np.int64),
        cumulative_count=cumulative_count.astype(np.int64),
        cumulative_fraction=cumulative_fraction,
        n_total=n_total,
        n_finite=n_finite,
        n_nonfinite=n_nonfinite,
    )


@dataclass(frozen=True)
class BasinDistributionResult:
    """One configuration's validated per-basin NSE distribution."""

    configuration_id: str
    per_basin: pd.DataFrame  # columns: basin_id, nse, + passthrough producer columns; ordered by basin_id
    percentile_table: PercentileTable
    frozen_core: FrozenQuantileCore
    ecdf: EcdfTable
    n_total_basins: int
    n_finite_basins: int
    n_nonfinite_basins: int


def _validate_basin_ids(basin_ids: Sequence[Any], *, expected: bool) -> list[str]:
    label = "expected_basin_ids" if expected else "basin_id"
    out = []
    for value in basin_ids:
        if not isinstance(value, str) or value.strip() == "":
            raise BasinAnalysisContractError(f"{label} entries must be nonblank strings, got {value!r}")
        out.append(value)
    return out


def _is_recognized_missing_marker(value: Any) -> bool:
    """``True`` only for the exact scalar singletons ``None`` and ``pd.NA``.

    Uses identity checks, never a general ``pd.isna(value)`` call, so a
    malformed non-scalar (list/dict/ndarray) input can never trigger an
    ambiguous-truth-value error here -- it simply falls through to the
    final "not a numeric scalar" rejection in :func:`_coerce_metric_scalar`.
    """
    return value is None or value is pd.NA


def _coerce_metric_scalar(value: Any, *, context: str) -> float:
    """Authoritative scalar/missing-value contract shared by the basin
    distribution path and :func:`compute_ecdf_table`.

    - ``None`` / ``pd.NA`` are recognized missing markers -> ``NaN``
      (nonfinite, never coerced to zero).
    - Numeric NaN/+inf/-inf pass through as nonfinite.
    - ``bool`` / ``np.bool_`` and strings (including numeric-looking
      strings) are malformed and rejected.
    - Any other non-numeric-scalar object (containers included) is rejected.
    """
    if isinstance(value, (bool, np.bool_)):
        raise BasinAnalysisContractError(f"{context}: boolean value is malformed, not a metric scalar")
    if _is_recognized_missing_marker(value):
        return float("nan")
    if isinstance(value, str):
        raise BasinAnalysisContractError(f"{context}: string value {value!r} is malformed, not a metric scalar")
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    raise BasinAnalysisContractError(f"{context}: value {value!r} is not a numeric scalar")


def analyze_basin_distribution(
    configuration_id: str,
    expected_basin_ids: Sequence[str],
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> BasinDistributionResult:
    """Validate and analyze one configuration's per-basin NSE rows.

    ``rows`` must contain at least ``basin_id`` and ``nse`` per row; extra
    producer columns (e.g. from ``evaluate_fixed_support_raw_space_metrics``)
    are passed through untouched. Row order is not scientific identity --
    output is canonically ordered by ``basin_id``.
    """
    if not isinstance(configuration_id, str) or configuration_id.strip() == "":
        raise BasinAnalysisContractError("configuration_id must be a nonblank string")

    expected_list = _validate_basin_ids(list(expected_basin_ids), expected=True)
    if len(expected_list) == 0:
        raise BasinAnalysisContractError("expected_basin_ids must be nonempty")
    if len(set(expected_list)) != len(expected_list):
        raise BasinAnalysisContractError("expected_basin_ids must be unique")
    expected_set = set(expected_list)

    frame = pd.DataFrame(list(rows)) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if "basin_id" not in frame.columns or "nse" not in frame.columns:
        raise BasinAnalysisContractError("rows must contain at least 'basin_id' and 'nse' columns")

    seen_ids: list[str] = []
    nse_values: list[float] = []
    for basin_id_raw, nse_raw in zip(frame["basin_id"].tolist(), frame["nse"].tolist()):
        basin_id = _validate_basin_ids([basin_id_raw], expected=False)[0]
        seen_ids.append(basin_id)
        nse_values.append(_coerce_metric_scalar(nse_raw, context=f"basin {basin_id!r}"))

    duplicates = {b for b in seen_ids if seen_ids.count(b) > 1}
    if duplicates:
        raise BasinAnalysisContractError(f"duplicate basin_id rows: {sorted(duplicates)}")

    seen_set = set(seen_ids)
    missing = expected_set - seen_set
    if missing:
        raise BasinAnalysisContractError(f"missing expected basin IDs: {sorted(missing)}")
    unexpected = seen_set - expected_set
    if unexpected:
        raise BasinAnalysisContractError(f"unexpected basin IDs not in expected_basin_ids: {sorted(unexpected)}")

    frame = frame.copy()
    frame["basin_id"] = seen_ids
    frame["nse"] = nse_values
    frame = frame.sort_values("basin_id", kind="mergesort").reset_index(drop=True)

    arr = np.asarray(frame["nse"].to_numpy(), dtype=np.float64)
    percentile_table = compute_percentile_table(arr, metric_name="nse")
    frozen_core = FrozenQuantileCore.from_percentile_table(percentile_table)
    ecdf = compute_ecdf_table(arr)
    n_total = int(arr.size)
    n_finite = int(np.isfinite(arr).sum())

    return BasinDistributionResult(
        configuration_id=configuration_id,
        per_basin=frame,
        percentile_table=percentile_table,
        frozen_core=frozen_core,
        ecdf=ecdf,
        n_total_basins=n_total,
        n_finite_basins=n_finite,
        n_nonfinite_basins=n_total - n_finite,
    )


@dataclass(frozen=True)
class PairedBasinDifferenceResult:
    """Candidate-vs-reference paired per-basin NSE difference result
    (design §§8, 8.1). ``candidate_id`` / ``reference_id`` preserve the
    supplied identities only -- e.g. a caller-labelled "P2" or "R2" -- and
    are never independently retrieved or inferred here."""

    candidate_id: str
    reference_id: str
    per_basin: pd.DataFrame  # basin_id, nse_candidate, nse_reference, delta, classification; ordered by basin_id
    percentile_table: PercentileTable
    frozen_core: FrozenQuantileCore
    ecdf: EcdfTable
    n_total_basins: int
    n_candidate_finite: int
    n_reference_finite: int
    n_pairwise_finite: int
    n_pairwise_nonfinite: int
    n_improved: int
    n_worse: int
    n_tied: int
    frac_improved: float
    frac_worse: float
    frac_tied: float


def analyze_paired_basin_difference(
    candidate: BasinDistributionResult,
    reference: BasinDistributionResult,
) -> PairedBasinDifferenceResult:
    """Compute the frozen paired basin-difference analysis (design §8.1)
    between two already-validated :class:`BasinDistributionResult` objects."""
    candidate_ids = set(candidate.per_basin["basin_id"])
    reference_ids = set(reference.per_basin["basin_id"])
    if candidate_ids != reference_ids:
        raise BasinAnalysisContractError(
            "candidate and reference basin identity sets must be exactly equal "
            f"(candidate-only: {sorted(candidate_ids - reference_ids)}, "
            f"reference-only: {sorted(reference_ids - candidate_ids)})"
        )

    cand_frame = candidate.per_basin.set_index("basin_id")["nse"]
    ref_frame = reference.per_basin.set_index("basin_id")["nse"]
    basin_ids = sorted(candidate_ids)

    nse_candidate = cand_frame.loc[basin_ids].to_numpy(dtype=np.float64)
    nse_reference = ref_frame.loc[basin_ids].to_numpy(dtype=np.float64)

    finite_candidate = np.isfinite(nse_candidate)
    finite_reference = np.isfinite(nse_reference)
    pairwise_finite = finite_candidate & finite_reference

    delta = np.full(len(basin_ids), np.nan, dtype=np.float64)
    delta[pairwise_finite] = nse_candidate[pairwise_finite] - nse_reference[pairwise_finite]

    classification = np.full(len(basin_ids), "nonfinite_unavailable", dtype=object)
    improved_mask = pairwise_finite & (delta > NUMERICAL_TIE_TOLERANCE)
    worse_mask = pairwise_finite & (delta < -NUMERICAL_TIE_TOLERANCE)
    tied_mask = pairwise_finite & (np.abs(delta) <= NUMERICAL_TIE_TOLERANCE)
    classification[improved_mask] = "improved"
    classification[worse_mask] = "worse"
    classification[tied_mask] = "tied"

    n_total = len(basin_ids)
    n_candidate_finite = int(finite_candidate.sum())
    n_reference_finite = int(finite_reference.sum())
    n_pairwise_finite = int(pairwise_finite.sum())
    n_pairwise_nonfinite = n_total - n_pairwise_finite
    n_improved = int(improved_mask.sum())
    n_worse = int(worse_mask.sum())
    n_tied = int(tied_mask.sum())

    if n_pairwise_finite > 0:
        frac_improved = n_improved / n_pairwise_finite
        frac_worse = n_worse / n_pairwise_finite
        frac_tied = n_tied / n_pairwise_finite
    else:
        frac_improved = float("nan")
        frac_worse = float("nan")
        frac_tied = float("nan")

    per_basin = pd.DataFrame(
        {
            "basin_id": basin_ids,
            "nse_candidate": nse_candidate,
            "nse_reference": nse_reference,
            "delta": delta,
            "classification": classification,
        }
    )

    percentile_table = compute_percentile_table(delta, metric_name="nse_delta")
    frozen_core = FrozenQuantileCore.from_percentile_table(percentile_table)
    ecdf = compute_ecdf_table(delta)

    return PairedBasinDifferenceResult(
        candidate_id=candidate.configuration_id,
        reference_id=reference.configuration_id,
        per_basin=per_basin,
        percentile_table=percentile_table,
        frozen_core=frozen_core,
        ecdf=ecdf,
        n_total_basins=n_total,
        n_candidate_finite=n_candidate_finite,
        n_reference_finite=n_reference_finite,
        n_pairwise_finite=n_pairwise_finite,
        n_pairwise_nonfinite=n_pairwise_nonfinite,
        n_improved=n_improved,
        n_worse=n_worse,
        n_tied=n_tied,
        frac_improved=frac_improved,
        frac_worse=frac_worse,
        frac_tied=frac_tied,
    )
