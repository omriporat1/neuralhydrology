"""RD1-C4-D1: the 24x400 package-versus-validation-pickle observation diagnostic.

This module MEASURES a disagreement. It does not adjudicate one.

Background. Flash-NH Stage-1 v2's formal RD1-C4 consumer derives every Q98
fact from the frozen package's own ``qobs_m3s`` (float32, raw m^3/s), and
separately audits the observation that each trial's own
``validation_results.p`` implies. Job 46169186's targeted single-basin probe
found that for basin 06911000 under trial P2, **all 8,436** aligned admitted
elements differed from the package at the bit level, five of them beyond the
provisional report-only envelope -- while every Q98 membership and
observed-peak decision was unchanged. One basin of one trial is not evidence
about 400 basins of 24 trials, and the mechanism is not established.

D1 therefore observes the whole 24x400 cross-product and reports what is
there. It is a **diagnostic layer**, deliberately separate from the formal
consumer:

* it never raises on a numerical difference -- a difference is a recorded
  measurement, and a trial task that hits one runs to completion;
* it never applies a pass/fail threshold. The provisional envelope
  (``rtol = 64 * float32 eps``, ``atol = 1e-6 m^3/s``) is computed and
  reported per element purely as a comparison field. It is NOT a gate, NOT
  a classifier, and selecting the final RD1-C4 audit policy is explicitly
  out of scope here -- that decision belongs to the user, after this
  evidence exists;
* it publishes no scientific Q98/NSE/KGE result. Section F's Q98 layer
  answers only "would any frozen Q98 decision have differed?", which is a
  consequence measurement, not a hydrological finding.

What it does refuse. Global product-identity disagreement fails the task
closed **before** any value is compared: a wrong package, a wrong contract,
a wrong best-epoch receipt, or a validation pickle whose hash does not match
its receipt are identity contradictions, not measurements. Basin-specific
problems are the opposite -- they become one of the nine typed cell statuses
and never stop the remaining basins. Every completed shard holds exactly
:data:`EXPECTED_BASIN_COUNT` cell records, error cells included, so a
missing measurement can never masquerade as an absent basin.

Reuse, not reimplementation. The observation reconstruction path is the
already-qualified one:
:func:`~.fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`
(pickle side, via its ``authenticated_period_results``/``package_identity``
seams) and
:func:`~.fixed_support_contract_v2.derive_canonical_package_observed_series`
(package side). The Q98 facts come from
:func:`~.stage1_v2_12plus12_hydrological_consumer.derive_canonical_basin_q98_facts`.
No parallel unit conversion, metric, area, or quantile math is introduced.

Evidence layout (one trial = one shard, see :mod:`.atomic_shard_store`)::

    observation_provenance/shards/<trial_id>/
        cells.jsonl              exactly 400 typed cell records
        trial_summary.json       trial-level identity + status histogram
        extremes.jsonl           up to six distinct extreme records per cell
        q98_consequences.jsonl   one record per compared basin
        detail/elements.parquet  every non-bitwise-equal aligned element
        detail/detail_index.json per-basin row offsets + content hash
        progress.jsonl           flushed operational progress events
        human.log                human-readable narrative
    observation_provenance/receipts/<trial_id>.json

**Detail volume and why Parquet.** Nearly all ~80.35 million comparisons
may be non-bitwise-equal, so the detail cannot be 80 million files, and it
is not meant to travel: it stays on Moriah and only the compact layer
(summaries, statuses, extremes, manifests, receipts, hashes) is transferred.
Parquet via ``pyarrow`` -- already an installed project dependency, so no
new heavy dependency -- gives columnar compression with a fixed schema and
a deterministic row order (basin id ascending, then support index
ascending). The detail file's SHA-256 is recorded in the index sidecar and
in the receipt, so the exact bytes are content-addressed even though
Parquet's own container bytes are not guaranteed identical across
``pyarrow`` releases (the *row content and ordering* are).

**Mapping a support index back to a timestamp.** Every detail row carries
``support_index``, the position within the frozen contract's own
``per_basin_support[basin_id]`` list, and ``date_ns`` -- the same instant as
an int64 epoch-nanosecond value. So
``contract["per_basin_support"][basin_id][support_index]`` is the
authoritative timestamp, and ``date_ns`` is a redundant self-contained copy
that lets the detail be read without the contract while still allowing the
two to be cross-checked. Detail plus the frozen contract is sufficient to
reconstruct the exact comparison.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .authenticated_period_results import (
    AuthenticatedPeriodResults,
    AuthenticatedPeriodResultsError,
    load_authenticated_period_results,
)
from .atomic_shard_store import (
    AtomicShardStore,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_path,
)
from .fixed_support_contract_v2 import (
    FixedSupportContractError,
    derive_canonical_package_observed_series,
    evaluate_fixed_support_raw_space_metrics,
)
from .package_identity_qualification import (
    PackageIdentityError,
    QualifiedPackageIdentity,
    qualify_package_identity,
    verify_basin_time_series_file,
)
from .rd1_c4_trial_authentication import AuthenticatedTrialTarget
from .source_precision import (
    FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS,
    float32_local_spacing_ratio,
    float32_ordered_distance_histogram,
    float32_ordered_distance_to_float64,
)
from .stage1_v2_12plus12_hydrological_consumer import (
    HIGH_FLOW_QUANTILE,
    PROVENANCE_ATOL_M3S,
    PROVENANCE_RTOL,
    derive_canonical_basin_q98_facts,
)

__all__ = [
    "D1_SCHEMA_NAME",
    "D1_SCHEMA_VERSION",
    "D1_SHARD_FAMILY",
    "EXPECTED_BASIN_COUNT",
    "CELL_STATUSES",
    "EXTREME_KINDS",
    "ObservationDiagnosticError",
    "AuthenticatedTrialTarget",
    "ProgressLog",
    "compare_basin_observation_series",
    "q98_consequence_for_basin",
    "run_trial_observation_diagnostic",
]


class ObservationDiagnosticError(RuntimeError):
    """Raised only for a GLOBAL identity/protocol failure that must refuse
    the task before any value comparison. A numerical difference never
    raises this -- nor anything else."""


D1_SCHEMA_NAME = "flashnh_rd1_c4_observation_diagnostic"
D1_SCHEMA_VERSION = 1
D1_SHARD_FAMILY = "observation_provenance"

#: Every completed trial shard holds exactly this many cell records.
EXPECTED_BASIN_COUNT = 400

#: The nine typed per-cell outcomes. ``ok`` means the comparison was
#: performed -- it says nothing whatsoever about whether the values agreed.
CELL_STATUSES = (
    "ok",
    "support_mismatch",
    "package_missing",
    "package_checksum_mismatch",
    "nonfinite_canonical",
    "pickle_load_error",
    "area_derivation_inconsistent",
    "basin_missing_from_trial",
    "other_error",
)

#: The six extreme-element kinds retained per compared basin. They are kept
#: as six SEPARATE records precisely because they need not identify the same
#: element -- job 46169186 confirmed they did not.
EXTREME_KINDS = (
    "max_abs_diff",
    "max_rel_diff_package_reference",
    "max_rel_diff_symmetric",
    "max_source_precision_distance",
    "max_envelope_exceedance",
    "first_envelope_exceedance",
)

_PROGRESS_BASIN_STRIDE = 25

#: Operational progress logs live here, beside (never inside) the shards.
_PROGRESS_DIR = "_progress"


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class ProgressLog:
    """Append-only, flushed JSONL progress log for one attempt.

    Operational evidence only: a progress file is never a completed shard,
    never consumed by the reducer, and its absence or truncation never makes
    a published shard invalid. Every event is flushed and fsynced
    immediately, so a task killed by Slurm still leaves a usable trace of
    where it was.
    """

    def __init__(self, path, *, trial_id: str, attempt_token: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trial_id = trial_id
        self.attempt_token = attempt_token
        self.started_monotonic = time.monotonic()
        self._handle = open(self.path, "a", encoding="utf-8")

    def event(self, event: str, **fields: Any) -> dict:
        elapsed = time.monotonic() - self.started_monotonic
        record = {
            "event": event,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "trial_id": self.trial_id,
            "attempt_token": self.attempt_token,
            "elapsed_s": round(elapsed, 3),
        }
        record.update(fields)
        self._handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return record

    def basin_event(self, *, basin_index: int, basin_id: str, n_basins: int, **fields: Any) -> dict:
        """A periodic per-basin event carrying an estimated remaining time
        derived from observed throughput (``None`` before any basin has
        finished, rather than a fabricated estimate)."""
        elapsed = time.monotonic() - self.started_monotonic
        done = basin_index + 1
        remaining = None
        if done > 0 and n_basins > done:
            remaining = round(elapsed / done * (n_basins - done), 1)
        return self.event(
            "basin_progress",
            basin_index=basin_index,
            basin_id=basin_id,
            n_basins=n_basins,
            estimated_remaining_s=remaining,
            **fields,
        )

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Trial identity
# --------------------------------------------------------------------------- #


#: D1 consumes trials ONLY as
#: :class:`~.rd1_c4_trial_authentication.AuthenticatedTrialTarget` -- an
#: object that can only be produced by
#: :func:`~.rd1_c4_trial_authentication.authenticate_trial_roster`, which
#: derives every scientifically load-bearing fact (run directory, official
#: best epoch and objective, arm, configuration/proposal identity,
#: evaluation scope, sealed-scope state) from that trial's own hash-verified
#: execution receipt.
#:
#: The former ``TrialTarget`` dataclass is deliberately gone (RD1-C4-D1
#: BLOCKER A1). It accepted all of those facts as plain constructor
#: arguments, so a hand-written trial-list JSON could aim the diagnostic at
#: any run directory on disk -- including one belonging to a sealed
#: evaluation scope -- while recording an unopened receipt reference as if
#: it had been checked. Removing the type, rather than adding a validator
#: beside it, is what makes the unauthenticated path unreachable.


# --------------------------------------------------------------------------- #
# Per-basin numerical comparison (sections C, D, E)
# --------------------------------------------------------------------------- #


def _percentiles(values: np.ndarray) -> dict:
    """The section-D percentile set. ``p50`` and ``median`` are reported as
    separate named fields even though they are the same statistic, because
    the specification names both and a reader must not have to assume."""
    if values.size == 0:
        return {name: None for name in ("median", "p50", "p90", "p99", "p99_9")}
    return {
        "median": float(np.median(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p99": float(np.percentile(values, 99.0)),
        "p99_9": float(np.percentile(values, 99.9)),
    }


def _uniform_scale_consistency(package: np.ndarray, pickle_side: np.ndarray) -> dict:
    """Is this difference pattern consistent with an approximately uniform
    multiplicative scale effect?

    A uniform scale error (a wrong area, a wrong unit constant) makes
    ``pickle / package`` an identical constant at every element; independent
    rounding does not. Reported as evidence, never as a verdict: the fields
    are the ratio's median, spread, and the fraction of elements lying
    within a tight relative band of that median.
    """
    usable = (package != 0.0) & np.isfinite(package) & np.isfinite(pickle_side)
    n_usable = int(usable.sum())
    if n_usable == 0:
        return {
            "n_ratio_usable": 0,
            "ratio_median": None,
            "ratio_min": None,
            "ratio_max": None,
            "max_abs_relative_deviation_from_median_ratio": None,
            "fraction_within_1e_9_of_median_ratio": None,
        }
    ratio = pickle_side[usable] / package[usable]
    median = float(np.median(ratio))
    deviation = np.abs(ratio / median - 1.0) if median != 0.0 else np.full(ratio.shape, np.inf)
    return {
        "n_ratio_usable": n_usable,
        "ratio_median": median,
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
        "max_abs_relative_deviation_from_median_ratio": float(np.max(deviation)),
        "fraction_within_1e_9_of_median_ratio": float(np.mean(deviation <= 1e-9)),
    }


@dataclass
class BasinComparison:
    """Everything one compared basin produced: the summary that travels, the
    extremes that travel, and the per-element detail that stays remote."""

    summary: dict = field(default_factory=dict)
    extremes: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def compare_basin_observation_series(
    *,
    trial_id: str,
    basin_id: str,
    support_dates: np.ndarray,
    package_obs_m3s: np.ndarray,
    pickle_obs_m3s: np.ndarray,
    area_facts: Mapping[str, Any],
) -> BasinComparison:
    """Compare one basin's package-canonical and pickle-reconstructed
    observed discharge, element by element.

    ``package_obs_m3s`` is the frozen package's stored ``float32``;
    ``pickle_obs_m3s`` is the ``float64`` result of the qualified
    reconstruction path. Both must already be positionally aligned to
    ``support_dates``, which is the frozen contract's own
    ``per_basin_support[basin_id]`` order -- this function does not align,
    reorder, or mask anything, and a caller that has not proven timestamp
    identity must not call it.

    Returns summaries, up to six distinct extreme records, and the
    per-element detail for every non-bitwise-equal element. Raises nothing
    for a numerical difference of any size.
    """
    package = np.asarray(package_obs_m3s, dtype=np.float32)
    package64 = package.astype(np.float64)
    reconstructed = np.asarray(pickle_obs_m3s, dtype=np.float64)
    n = int(reconstructed.size)

    abs_diff = np.abs(package64 - reconstructed)
    bitwise_equal = package64 == reconstructed
    n_bitwise_equal = int(bitwise_equal.sum())

    # Two explicitly named relative definitions. The package is the
    # reference in RD1-C4, so the package-reference form is the primary one;
    # the symmetric form is reported alongside so a near-zero package value
    # cannot make a difference look arbitrarily large without that being
    # visible. Both are guarded at a zero denominator rather than producing
    # inf/NaN silently.
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_package = np.where(package64 != 0.0, abs_diff / np.abs(package64), np.nan)
        denom_symmetric = (np.abs(package64) + np.abs(reconstructed)) / 2.0
        rel_symmetric = np.where(denom_symmetric != 0.0, abs_diff / denom_symmetric, np.nan)

    # Section E: two separately named source-precision measures, never one
    # presented as the other.
    ordered, exactly_representable = float32_ordered_distance_to_float64(package, reconstructed)
    spacing = float32_local_spacing_ratio(reconstructed, package64)

    # Section A3 / D: the provisional envelope, computed as a REPORTED
    # field. Asymmetric by construction -- the package value alone is the
    # relative term's reference -- and that asymmetry is recorded, not
    # hidden.
    tolerance = PROVENANCE_ATOL_M3S + PROVENANCE_RTOL * np.abs(package64)
    exceeds = abs_diff > tolerance
    with np.errstate(invalid="ignore", divide="ignore"):
        exceedance_factor = np.where(tolerance > 0.0, abs_diff / tolerance, np.inf)
    n_exceeding = int(exceeds.sum())

    n_nonfinite_package = int((~np.isfinite(package64)).sum())
    n_nonfinite_reconstructed = int((~np.isfinite(reconstructed)).sum())

    summary = {
        "n_compared": n,
        "n_bitwise_equal": n_bitwise_equal,
        "n_unequal": n - n_bitwise_equal,
        "n_nonfinite_package": n_nonfinite_package,
        "n_nonfinite_reconstructed": n_nonfinite_reconstructed,
        "abs_diff_max": float(np.max(abs_diff)) if n else None,
        "abs_diff_mean": float(np.mean(abs_diff)) if n else None,
        "abs_diff_rms": float(np.sqrt(np.mean(abs_diff**2))) if n else None,
        "rel_diff_package_reference_max": _nanmax_or_none(rel_package),
        "rel_diff_symmetric_max": _nanmax_or_none(rel_symmetric),
        "source_precision_distance_max": int(np.max(ordered.distance)) if n else None,
        "n_source_precision_undefined": int((~ordered.defined).sum()),
        "n_reconstruction_not_float32_representable": int((~exactly_representable).sum()),
        "spacing_ratio_max": _nanmax_or_none(spacing.ratio),
        "n_spacing_denominator_min_subnormal": int(spacing.denominator_is_min_subnormal.sum()),
        # Report-only. Not a gate, not a classifier, not a pass criterion.
        "provisional_envelope_rtol": float(PROVENANCE_RTOL),
        "provisional_envelope_atol_m3s": float(PROVENANCE_ATOL_M3S),
        "provisional_envelope_form": "abs_diff > atol + rtol * abs(package_value)",
        "provisional_envelope_reference_is_package_value_only": True,
        "n_exceeding_provisional_envelope": n_exceeding,
        "max_provisional_envelope_exceedance_factor": float(np.max(exceedance_factor)) if n else None,
        "source_precision_histogram": float32_ordered_distance_histogram(
            ordered.distance, defined=ordered.defined
        ),
        "source_precision_histogram_bins": list(FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS),
        "uniform_multiplicative_scale": _uniform_scale_consistency(package64, reconstructed),
    }
    summary.update({f"abs_diff_{key}": value for key, value in _percentiles(abs_diff).items()})
    summary.update(dict(area_facts))

    unequal_index = np.flatnonzero(~bitwise_equal)
    extremes = _extreme_records(
        trial_id=trial_id,
        basin_id=basin_id,
        support_dates=support_dates,
        package=package,
        package64=package64,
        reconstructed=reconstructed,
        abs_diff=abs_diff,
        rel_package=rel_package,
        rel_symmetric=rel_symmetric,
        ordered_distance=ordered.distance,
        ordered_defined=ordered.defined,
        spacing_ratio=spacing.ratio,
        tolerance=tolerance,
        exceeds=exceeds,
        exceedance_factor=exceedance_factor,
    )

    detail = {
        "support_index": unequal_index.astype(np.int32),
        "date_ns": _dates_to_int64_ns(support_dates)[unequal_index],
        "package_value_m3s": package[unequal_index],
        "reconstructed_value_m3s": reconstructed[unequal_index],
        "abs_diff": abs_diff[unequal_index],
        "rel_diff_package_reference": rel_package[unequal_index],
        "rel_diff_symmetric": rel_symmetric[unequal_index],
        "source_precision_distance": ordered.distance[unequal_index].astype(np.int32),
        "source_precision_defined": ordered.defined[unequal_index],
        "reconstruction_float32_exact": exactly_representable[unequal_index],
        "spacing_ratio": spacing.ratio[unequal_index],
        "provisional_tolerance": tolerance[unequal_index],
        "exceeds_provisional_envelope": exceeds[unequal_index],
    }
    return BasinComparison(summary=summary, extremes=extremes, detail=detail)


def _nanmax_or_none(values: np.ndarray):
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else None


def _dates_to_int64_ns(support_dates: np.ndarray) -> np.ndarray:
    """Epoch-nanosecond int64 copies of the contract's support timestamps.

    A non-``datetime64`` contract (synthetic fixtures use plain integers)
    is preserved as-is rather than coerced into a fictitious calendar, so
    the detail never claims a timestamp the contract did not have.
    """
    arr = np.asarray(support_dates)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64)
    return arr.astype(np.int64)


def _extreme_records(
    *,
    trial_id: str,
    basin_id: str,
    support_dates: np.ndarray,
    package: np.ndarray,
    package64: np.ndarray,
    reconstructed: np.ndarray,
    abs_diff: np.ndarray,
    rel_package: np.ndarray,
    rel_symmetric: np.ndarray,
    ordered_distance: np.ndarray,
    ordered_defined: np.ndarray,
    spacing_ratio: np.ndarray,
    tolerance: np.ndarray,
    exceeds: np.ndarray,
    exceedance_factor: np.ndarray,
) -> list:
    """Build the six section-D extreme records.

    Each is resolved independently; no record is derived from another's
    index. Ties are broken by the earliest support index (deterministic and
    reproducible), and a kind with no qualifying element -- there may be no
    envelope exceedance at all -- is simply absent rather than fabricated.
    """
    dates_ns = _dates_to_int64_ns(support_dates)

    def _record(kind: str, index: int) -> dict:
        return {
            "trial_id": trial_id,
            "basin_id": basin_id,
            "extreme_kind": kind,
            "support_index": int(index),
            "date_ns": int(dates_ns[index]),
            "package_value_m3s": float(package64[index]),
            "package_value_float32_bits": f"0x{int(package[index : index + 1].view(np.uint32)[0]):08x}",
            "reconstructed_value_m3s": float(reconstructed[index]),
            "abs_diff": float(abs_diff[index]),
            "rel_diff_package_reference": _finite_or_none(rel_package[index]),
            "rel_diff_symmetric": _finite_or_none(rel_symmetric[index]),
            "source_precision_distance": int(ordered_distance[index]),
            "source_precision_defined": bool(ordered_defined[index]),
            "spacing_ratio": _finite_or_none(spacing_ratio[index]),
            "provisional_tolerance": float(tolerance[index]),
            "exceeds_provisional_envelope": bool(exceeds[index]),
            "provisional_envelope_exceedance_factor": _finite_or_none(exceedance_factor[index]),
        }

    def _argmax(values: np.ndarray, *, mask: Optional[np.ndarray] = None) -> Optional[int]:
        candidate = np.where(np.isfinite(values), values, -np.inf)
        if mask is not None:
            candidate = np.where(mask, candidate, -np.inf)
        if not np.isfinite(candidate).any() or np.all(candidate == -np.inf):
            return None
        return int(np.argmax(candidate))

    records = []
    for kind, index in (
        ("max_abs_diff", _argmax(abs_diff)),
        ("max_rel_diff_package_reference", _argmax(rel_package)),
        ("max_rel_diff_symmetric", _argmax(rel_symmetric)),
        (
            "max_source_precision_distance",
            _argmax(ordered_distance.astype(np.float64), mask=ordered_defined),
        ),
        ("max_envelope_exceedance", _argmax(exceedance_factor, mask=exceeds)),
        (
            "first_envelope_exceedance",
            int(np.flatnonzero(exceeds)[0]) if bool(exceeds.any()) else None,
        ),
    ):
        if index is not None:
            records.append(_record(kind, index))
    return records


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- #
# Q98 consequence layer (section F)
# --------------------------------------------------------------------------- #


def q98_consequence_for_basin(
    *,
    trial_id: str,
    basin_id: str,
    support_dates: np.ndarray,
    package_obs_m3s: np.ndarray,
    pickle_obs_m3s: np.ndarray,
) -> dict:
    """Derive Q98 facts independently from each observation series and
    report whether any frozen decision would have differed.

    Both sides go through the same qualified
    :func:`derive_canonical_basin_q98_facts`, so the threshold rule
    (``np.quantile(..., 0.98, method="linear")``), the inclusive
    ``obs >= threshold`` membership, and the earliest-chronological peak
    tie-break are identical on both sides -- the ONLY difference between
    them is the observation values themselves, which is the whole question.

    This is a consequence measurement. It publishes no formal Q98
    configuration result, and ``any_frozen_decision_changes`` is a
    description of this basin, not a verdict about the campaign.
    """
    package_facts = derive_canonical_basin_q98_facts(
        basin_id=basin_id, date=support_dates, obs_m3s=np.asarray(package_obs_m3s, dtype=np.float64)
    )
    pickle_facts = derive_canonical_basin_q98_facts(
        basin_id=basin_id, date=support_dates, obs_m3s=np.asarray(pickle_obs_m3s, dtype=np.float64)
    )

    threshold_abs_diff = abs(package_facts.q98_threshold - pickle_facts.q98_threshold)
    threshold_rel_diff = (
        threshold_abs_diff / abs(package_facts.q98_threshold)
        if package_facts.q98_threshold != 0.0
        else None
    )
    symmetric_difference = int(np.sum(package_facts.high_flow_mask != pickle_facts.high_flow_mask))

    package_peak_ties = _peak_tie_facts(support_dates, package_facts)
    pickle_peak_ties = _peak_tie_facts(support_dates, pickle_facts)
    peak_index_equal = package_facts.observed_peak_index == pickle_facts.observed_peak_index
    peak_timestamp_equal = package_peak_ties["earliest_tied_peak_date_ns"] == (
        pickle_peak_ties["earliest_tied_peak_date_ns"]
    )

    return {
        "trial_id": trial_id,
        "basin_id": basin_id,
        "quantile": HIGH_FLOW_QUANTILE,
        "n_admitted": package_facts.n_admitted,
        "q98_threshold_package": package_facts.q98_threshold,
        "q98_threshold_reconstructed": pickle_facts.q98_threshold,
        "q98_threshold_abs_diff": threshold_abs_diff,
        "q98_threshold_rel_diff": threshold_rel_diff,
        "n_high_flow_package": package_facts.n_high_flow,
        "n_high_flow_reconstructed": pickle_facts.n_high_flow,
        "high_flow_mask_symmetric_difference": symmetric_difference,
        "high_flow_mask_sha256_package": _mask_sha256(package_facts.high_flow_mask),
        "high_flow_mask_sha256_reconstructed": _mask_sha256(pickle_facts.high_flow_mask),
        "observed_peak_value_package": package_facts.observed_peak_value,
        "observed_peak_value_reconstructed": pickle_facts.observed_peak_value,
        "observed_peak_index_package": package_facts.observed_peak_index,
        "observed_peak_index_reconstructed": pickle_facts.observed_peak_index,
        "peak_tie_count_package": package_peak_ties["peak_tie_count"],
        "peak_tie_count_reconstructed": pickle_peak_ties["peak_tie_count"],
        "earliest_tied_peak_date_ns_package": package_peak_ties["earliest_tied_peak_date_ns"],
        "earliest_tied_peak_date_ns_reconstructed": pickle_peak_ties["earliest_tied_peak_date_ns"],
        "peak_index_equal": bool(peak_index_equal),
        "peak_timestamp_equal": bool(peak_timestamp_equal),
        "high_flow_membership_equal": symmetric_difference == 0,
        "any_frozen_decision_changes": bool(
            symmetric_difference != 0 or not peak_index_equal or not peak_timestamp_equal
        ),
    }


def _peak_tie_facts(support_dates: np.ndarray, facts) -> dict:
    obs = np.asarray(facts.canonical_obs_m3s, dtype=np.float64)
    tie_indices = np.flatnonzero(obs == facts.observed_peak_value)
    dates_ns = _dates_to_int64_ns(support_dates)
    return {
        "peak_tie_count": int(tie_indices.size),
        "earliest_tied_peak_date_ns": int(np.min(dates_ns[tie_indices])) if tie_indices.size else None,
    }


def _mask_sha256(mask: np.ndarray) -> str:
    """Deterministic content hash of a boolean mask: its packed bits plus
    its exact length, so two masks of different lengths whose packed bytes
    coincide can never hash equal."""
    packed = np.packbits(np.asarray(mask, dtype=bool)).tobytes()
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(int(np.asarray(mask).size)).encode("ascii"))
    digest.update(b":")
    digest.update(packed)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Detail writer (section C)
# --------------------------------------------------------------------------- #

_DETAIL_COLUMNS = (
    ("basin_id", "string"),
    ("support_index", "int32"),
    ("date_ns", "int64"),
    ("package_value_m3s", "float32"),
    ("reconstructed_value_m3s", "float64"),
    ("abs_diff", "float64"),
    ("rel_diff_package_reference", "float64"),
    ("rel_diff_symmetric", "float64"),
    ("source_precision_distance", "int32"),
    ("source_precision_defined", "bool_"),
    ("reconstruction_float32_exact", "bool_"),
    ("spacing_ratio", "float64"),
    ("provisional_tolerance", "float64"),
    ("exceeds_provisional_envelope", "bool_"),
)


class DetailWriter:
    """Streams per-element detail into one compact Parquet file.

    One row group per basin, written in basin order as basins complete, so
    peak memory stays at one basin's detail rather than a trial's ~3.35
    million rows. The schema is fixed and declared up front, and the index
    sidecar records each basin's row offset/count so a consumer can read one
    basin's evidence without scanning the file.
    """

    def __init__(self, path, *, compression: str = "zstd") -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema([pa.field(name, getattr(pa, kind)()) for name, kind in _DETAIL_COLUMNS])
        try:
            self._writer = pq.ParquetWriter(self.path, self.schema, compression=compression)
            self.compression = compression
        except (ValueError, OSError, NotImplementedError):
            # zstd is normally available in the project's pyarrow build; fall
            # back rather than losing the evidence, and record which codec
            # actually produced the bytes.
            self._writer = pq.ParquetWriter(self.path, self.schema, compression="gzip")
            self.compression = "gzip"
        self.basin_offsets: list = []
        self.n_rows = 0

    def write_basin(self, basin_id: str, detail: Mapping[str, np.ndarray]) -> None:
        pa = self._pa
        n = int(detail["support_index"].size)
        arrays = [pa.array([basin_id] * n, type=pa.string())]
        for name, kind in _DETAIL_COLUMNS[1:]:
            arrays.append(pa.array(np.asarray(detail[name]), type=getattr(pa, kind)()))
        self._writer.write_table(pa.Table.from_arrays(arrays, schema=self.schema))
        self.basin_offsets.append({"basin_id": basin_id, "row_offset": self.n_rows, "n_rows": n})
        self.n_rows += n

    def finalize(self) -> dict:
        self._writer.close()
        return {
            "detail_relative_path": self.path.name,
            "format": "parquet",
            "compression": self.compression,
            "schema": [{"name": name, "type": kind} for name, kind in _DETAIL_COLUMNS],
            "row_order": "basin_id ascending, then support_index ascending",
            "support_index_semantics": (
                "position within the frozen fixed-support contract's own "
                "per_basin_support[basin_id] ordering; "
                "contract['per_basin_support'][basin_id][support_index] is the authoritative "
                "timestamp, and date_ns is a redundant self-contained copy of the same instant"
            ),
            "retained_elements": "every aligned element that is not bitwise equal",
            "n_rows": self.n_rows,
            "basin_offsets": self.basin_offsets,
            "sha256": sha256_path(self.path),
            "size_bytes": self.path.stat().st_size,
        }


# --------------------------------------------------------------------------- #
# Per-basin cell production
# --------------------------------------------------------------------------- #


def _classify_contract_error(exc: Exception) -> str:
    """Map a residual :class:`FixedSupportContractError` onto a typed cell
    status.

    Every condition D1 can detect directly -- a missing basin, a missing or
    checksum-mismatched package file -- is checked explicitly before the
    reconstruction call, so this mapping handles only what the lower layer
    alone can diagnose. It is deliberately conservative: anything not
    positively recognised becomes ``other_error`` rather than being filed
    under a status that would understate it.
    """
    message = str(exc).lower()
    # A basin entry that exists in the loaded results but carries no usable
    # freq result is a defect in THIS basin's slice of the pickle, not a
    # support or identity disagreement.
    if "no freq result" in message or "freq_results" in message:
        return "pickle_load_error"
    if "support timestamp" in message or "date coordinate" in message or "realign" in message:
        return "support_mismatch"
    if "naturally admitted" in message:
        return "support_mismatch"
    if "non-finite" in message or "nonfinite" in message or "finite" in message:
        return "nonfinite_canonical"
    if "area derivation" in message:
        return "area_derivation_inconsistent"
    if "missing from" in message or "entirely missing" in message:
        return "basin_missing_from_trial"
    return "other_error"


def _new_cell(trial_id: str, basin_id: str, status: str, **fields: Any) -> dict:
    if status not in CELL_STATUSES:
        raise ObservationDiagnosticError(f"unknown cell status {status!r}")
    cell = {
        "schema_name": D1_SCHEMA_NAME,
        "schema_version": D1_SCHEMA_VERSION,
        "trial_id": trial_id,
        "basin_id": basin_id,
        "status": status,
        "status_detail": None,
    }
    cell.update(fields)
    return cell


def _diagnose_basin_cell(
    *,
    trial_id: str,
    basin_id: str,
    contract: Mapping,
    package_root: Path,
    identity: QualifiedPackageIdentity,
    run_dir,
    epoch: int,
    authenticated_results: AuthenticatedPeriodResults,
) -> tuple:
    """Produce exactly one typed cell for one basin, plus (when the
    comparison ran) its extremes, detail arrays, and Q98 consequence record.

    Never raises for a basin-specific problem: every failure path below
    yields a typed cell. A numerical difference of any magnitude is not a
    failure path at all.
    """
    # Cheap, explicit pre-checks first, so the common typed statuses come
    # from a direct observation rather than from parsing an error message.
    if basin_id not in authenticated_results:
        return _new_cell(
            trial_id,
            basin_id,
            "basin_missing_from_trial",
            status_detail="basin absent from this trial's validation results",
        ), [], None, None

    nc_path = package_root / "time_series" / f"{basin_id}.nc"
    if not nc_path.is_file():
        return _new_cell(
            trial_id, basin_id, "package_missing", status_detail=f"package NetCDF absent: {nc_path.name}"
        ), [], None, None

    try:
        verify_basin_time_series_file(identity, basin_id, package_root=package_root)
    except PackageIdentityError as exc:
        return _new_cell(
            trial_id, basin_id, "package_checksum_mismatch", status_detail=str(exc)
        ), [], None, None

    try:
        package_series = derive_canonical_package_observed_series(
            package_root=package_root,
            basin_id=basin_id,
            contract=dict(contract),
            package_identity=identity,
        )
    except FixedSupportContractError as exc:
        return _new_cell(
            trial_id, basin_id, _classify_contract_error(exc), status_detail=str(exc)
        ), [], None, None
    except Exception as exc:  # noqa: BLE001 - a basin must never stop the trial
        return _new_cell(
            trial_id, basin_id, "other_error", status_detail=f"{type(exc).__name__}: {exc}"
        ), [], None, None

    try:
        evaluated = evaluate_fixed_support_raw_space_metrics(
            run_dir=run_dir,
            epoch=epoch,
            package_root=package_root,
            contract=dict(contract),
            basin_ids=[basin_id],
            return_admitted_series=True,
            package_identity=identity,
            authenticated_period_results=authenticated_results,
        )
    except FixedSupportContractError as exc:
        return _new_cell(
            trial_id, basin_id, _classify_contract_error(exc), status_detail=str(exc)
        ), [], None, None
    except Exception as exc:  # noqa: BLE001
        return _new_cell(
            trial_id, basin_id, "other_error", status_detail=f"{type(exc).__name__}: {exc}"
        ), [], None, None

    # The evaluator excludes (rather than raises for) a failed/inconsistent
    # area derivation -- that exclusion is exactly the typed status here.
    if evaluated["basins_excluded"]:
        reason = str(evaluated["basins_excluded"][0].get("reason", ""))
        status = "area_derivation_inconsistent" if "area" in reason.lower() else "other_error"
        return _new_cell(trial_id, basin_id, status, status_detail=reason), [], None, None

    admitted = evaluated.get("admitted_series_by_basin", {}).get(basin_id)
    if admitted is None:
        return _new_cell(
            trial_id,
            basin_id,
            "other_error",
            status_detail="qualified evaluator returned no admitted series for this basin",
        ), [], None, None

    # Section A2 at the point of use: the two sides must describe the same
    # instants in the same order before a single value is compared. Both are
    # in the contract's own per_basin_support order by construction; this
    # proves it rather than assuming it.
    if admitted.date.shape != package_series.date.shape or not np.array_equal(
        admitted.date, package_series.date
    ):
        return _new_cell(
            trial_id,
            basin_id,
            "support_mismatch",
            status_detail=(
                "trial-admitted and package-canonical timestamp vectors are not identical -- "
                "refusing to compare positionally"
            ),
        ), [], None, None

    if not np.isfinite(np.asarray(package_series.obs_m3s, dtype=np.float64)).all():
        return _new_cell(
            trial_id,
            basin_id,
            "nonfinite_canonical",
            status_detail="package-canonical observed discharge contains a non-finite value",
        ), [], None, None

    per_basin = evaluated["per_basin"][0]
    area_facts = {
        "area_km2": per_basin.get("area_km2"),
        "area_relative_mad": per_basin.get("area_relative_mad"),
        "n_area_samples": per_basin.get("n_area_samples"),
        "package_basin_sha256": identity.basin_time_series_sha256.get(basin_id),
        "package_basin_relative_path": identity.basin_time_series_relative_path.get(basin_id),
    }

    comparison = compare_basin_observation_series(
        trial_id=trial_id,
        basin_id=basin_id,
        support_dates=package_series.date,
        package_obs_m3s=package_series.obs_m3s,
        pickle_obs_m3s=admitted.obs_m3s,
        area_facts=area_facts,
    )
    q98 = q98_consequence_for_basin(
        trial_id=trial_id,
        basin_id=basin_id,
        support_dates=package_series.date,
        package_obs_m3s=package_series.obs_m3s,
        pickle_obs_m3s=admitted.obs_m3s,
    )
    cell = _new_cell(trial_id, basin_id, "ok", **comparison.summary)
    cell["n_extremes_recorded"] = len(comparison.extremes)
    cell["n_detail_rows"] = int(comparison.detail["support_index"].size)
    cell["q98_any_frozen_decision_changes"] = q98["any_frozen_decision_changes"]
    return cell, comparison.extremes, comparison.detail, q98


# --------------------------------------------------------------------------- #
# Trial runner (sections B, G)
# --------------------------------------------------------------------------- #


def _git_head(repo_root: Path) -> dict:
    """Git HEAD plus the hashes of every uncommitted tracked file.

    Read-only inspection: this runs ``rev-parse``, ``status --porcelain``
    and ``hash-object`` only. Nothing is staged, committed, cleaned, or
    reset. A repository that cannot be inspected yields explicit ``None``s
    rather than a fabricated commit id.
    """

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(repo_root), capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain=v1") or ""
    dirty = {}
    for line in porcelain.splitlines():
        if len(line) < 4 or line[:2] == "??":
            continue
        relative = line[3:].strip().strip('"')
        path = repo_root / relative
        if path.is_file():
            dirty[relative] = sha256_path(path)
    return {
        "git_head": head,
        "git_uncommitted_tracked_file_sha256": dict(sorted(dirty.items())),
        "n_uncommitted_tracked_files": len(dirty),
    }


def _module_hashes(repo_root: Path, extra_paths: Sequence[str] = ()) -> dict:
    """Hashes of the diagnostic's own code, so a receipt pins the exact
    implementation that produced it."""
    relative_paths = [
        "src/baseline/rd1_c4_observation_diagnostic.py",
        "src/baseline/rd1_c4_observation_diagnostic_reduce.py",
        "src/baseline/atomic_shard_store.py",
        "src/baseline/package_identity_qualification.py",
        "src/baseline/source_precision.py",
        *extra_paths,
    ]
    hashes = {}
    for relative in relative_paths:
        path = repo_root / relative
        if path.is_file():
            hashes[relative] = sha256_path(path)
    return hashes


def _environment_identity() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "xarray", "pyarrow", "netCDF4", "scipy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    slurm = {
        key: os.environ.get(key)
        for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURMD_NODENAME")
        if os.environ.get(key)
    }
    return {"versions": versions, "slurm": slurm or None, "hostname": platform.node()}


def run_trial_observation_diagnostic(
    *,
    trial: AuthenticatedTrialTarget,
    contract: Mapping,
    package_root,
    store_root,
    repo_root,
    attempt_token: str,
    basin_ids: Optional[Sequence[str]] = None,
    expected_basin_count: int = EXPECTED_BASIN_COUNT,
    period: Optional[str] = None,
    detail_compression: str = "zstd",
) -> dict:
    """Run D1 for exactly one trial and publish exactly one atomic shard.

    Order of operations, each a hard precondition for the next:

    0. **Trial authority.** ``trial`` must be an
       :class:`~.rd1_c4_trial_authentication.AuthenticatedTrialTarget`
       (RD1-C4-D1 BLOCKER A1). Its run directory, official best epoch and
       objective, arm and identity were derived from its own hash-verified
       execution receipt, not asserted by whoever wrote the trial list, so
       this function never has to decide whether to trust them. Anything
       else is refused before any file is opened.

    1. **Global qualification.** The package identity is proven once against
       the contract (section A1), the contract's own checksum is confirmed,
       and the validation pickle is located at the RECEIPT-BOUND
       run/period/epoch and cross-checked against the receipt-bound path the
       roster resolved. Any disagreement raises
       :class:`ObservationDiagnosticError` and nothing is published --
       comparing values against the wrong product is worse than not
       comparing them.
    2. **Resume.** If a completed shard for this trial already exists, it is
       reused only when every identity field matches exactly; a conflicting
       identity refuses, reporting every disagreeing field, and neither
       overwrites nor recomputes. Incomplete attempt directories are ignored
       and never deleted.
    3. **Comparison.** The trial's ``validation_results.p`` is loaded once
       and every basin is diagnosed into exactly one typed cell. Basin
       failures are recorded, not raised.
    4. **Publication.** Detail, cells, extremes, Q98 consequences, progress
       and log are made durable and hashed, the shard is renamed into place
       atomically, and the receipt is written last.

    Returns the trial summary dict. Never submits a job, contacts a remote
    service, or writes outside ``store_root``.
    """
    if not isinstance(trial, AuthenticatedTrialTarget):
        raise ObservationDiagnosticError(
            "trial must be an AuthenticatedTrialTarget produced by authenticate_trial_roster(); got "
            f"{type(trial).__name__} -- D1 refuses a trial whose run directory, best epoch and objective "
            "were asserted rather than derived from its own execution receipt"
        )
    package_root = Path(package_root)
    repo_root = Path(repo_root)
    store = AtomicShardStore(store_root, D1_SHARD_FAMILY)
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    started_monotonic = time.monotonic()

    requested_basins = list(basin_ids) if basin_ids is not None else list(contract["basin_ids"])
    if len(requested_basins) != expected_basin_count:
        raise ObservationDiagnosticError(
            f"{trial.trial_id}: expected exactly {expected_basin_count} contract basins, got "
            f"{len(requested_basins)} -- refusing to publish a shard that cannot hold the expected "
            "cell count"
        )
    if len(set(requested_basins)) != len(requested_basins):
        raise ObservationDiagnosticError(f"{trial.trial_id}: duplicate basin id in the requested population")

    # -- step 1: global qualification, before any value comparison -------- #
    try:
        identity = qualify_package_identity(
            package_root=package_root, contract=dict(contract), basin_ids=requested_basins
        )
    except PackageIdentityError as exc:
        raise ObservationDiagnosticError(
            f"{trial.trial_id}: package identity qualification failed -- refusing to compare observations "
            f"against a package that is not the contract's own: {exc}"
        ) from exc

    # The evaluation period comes from the contract the trial was
    # authenticated against; an explicit override must agree with it, because
    # the roster resolved this trial's validation product for that period.
    period = period or contract["period"]
    if period != contract["period"]:
        raise ObservationDiagnosticError(
            f"{trial.trial_id}: period {period!r} disagrees with the fixed-support contract's own period "
            f"{contract['period']!r} -- the receipt-bound validation product was resolved for the contract's "
            "period"
        )

    # RD1-C4-D1 BLOCKER A1/A3: the pickle is opened ONCE, through the
    # authenticated loader, at the receipt-bound run_dir/period/best_epoch.
    # Its bytes are hashed by that loader -- the digest recorded below is the
    # one it actually read, never a second independent hash of a path that
    # might since have changed -- and the resulting object is what the
    # per-basin evaluations consume, so the simulations compared here are
    # provably this product's.
    try:
        authenticated_results = load_authenticated_period_results(
            run_dir=trial.run_dir,
            period=period,
            epoch=trial.best_epoch,
            expected_sha256=trial.validation_pickle_sha256,
        )
    except AuthenticatedPeriodResultsError as exc:
        raise ObservationDiagnosticError(
            f"{trial.trial_id}: the receipt-bound official validation product could not be authenticated: {exc}"
        ) from exc
    results_path = Path(authenticated_results.results_path)
    if results_path.resolve() != Path(trial.validation_pickle_path).resolve():
        raise ObservationDiagnosticError(
            f"{trial.trial_id}: opened {results_path} but the authenticated roster bound this trial to "
            f"{trial.validation_pickle_path} -- validation-product identity contradiction"
        )
    validation_pickle_sha256 = authenticated_results.results_sha256

    git_facts = _git_head(repo_root)
    identity_payload = {
        "schema_name": D1_SCHEMA_NAME,
        "schema_version": D1_SCHEMA_VERSION,
        "trial": trial.identity_fields(),
        "package": identity.identity_fields(),
        "contract_id": contract["contract_id"],
        "contract_checksum_sha256": contract["checksum_sha256"],
        "period": period,
        "validation_pickle_path": str(results_path),
        "validation_pickle_sha256": validation_pickle_sha256,
        "expected_basin_count": expected_basin_count,
        "requested_basin_ids_sha256": canonical_json_sha256(requested_basins),
        "module_sha256": _module_hashes(repo_root),
        **git_facts,
    }

    # -- step 2: strict resume -------------------------------------------- #
    existing, _ = store.reuse_if_identical(trial.trial_id, identity_payload)
    if existing is not None:
        return {
            "trial_id": trial.trial_id,
            "reused_existing_shard": True,
            "receipt": existing.as_dict(),
        }

    attempt_dir = store.begin_attempt(trial.trial_id, attempt_token=attempt_token)
    # The progress log lives OUTSIDE the attempt directory, under the
    # family's own operational area. Two reasons, both required by section
    # G: a progress file is operational evidence and must never become part
    # of a completed shard's hashed content, and the last events a task
    # emits -- receipt finalization and task end -- necessarily happen AFTER
    # the attempt directory has been renamed into place, so a log inside the
    # shard could only record them by lying about when they occurred.
    progress_path = store.family_dir / _PROGRESS_DIR / f"{trial.trial_id}__{attempt_token}.jsonl"
    progress = ProgressLog(progress_path, trial_id=trial.trial_id, attempt_token=attempt_token)
    human_lines = [
        f"RD1-C4-D1 observation diagnostic -- trial {trial.trial_id}",
        f"started_utc={started_utc} attempt={attempt_token}",
        f"package_root={package_root}",
        f"contract={contract['contract_id']} checksum={contract['checksum_sha256']}",
        f"validation_pickle={results_path}",
        f"validation_pickle_sha256={validation_pickle_sha256}",
        "",
        "This is a DIAGNOSTIC. No pass/fail threshold is applied; the provisional",
        "envelope below is a reported comparison field only, and the final RD1-C4",
        "audit policy remains unresolved.",
        "",
    ]

    try:
        progress.event("task_start", n_basins=len(requested_basins), started_utc=started_utc)
        progress.event(
            "source_qualification_complete",
            package_manifest_sha256=identity.package_manifest_sha256,
            contract_checksum_sha256=contract["checksum_sha256"],
            validation_pickle_sha256=validation_pickle_sha256,
        )

        # -- step 3: the trial product was loaded exactly once, above ----- #
        # A trial-level load failure is global -- it makes every one of the
        # 400 cells unmeasurable -- so it refuses the task (in step 1) rather
        # than publishing 400 identical error cells that would look like
        # evidence about the basins.
        progress.event("trial_load_complete", n_result_basins=len(authenticated_results))

        detail_writer = DetailWriter(attempt_dir / "detail" / "elements.parquet", compression=detail_compression)
        cells: list = []
        extremes: list = []
        q98_records: list = []
        status_counts = {status: 0 for status in CELL_STATUSES}

        for index, basin_id in enumerate(requested_basins):
            try:
                cell, basin_extremes, detail, q98 = _diagnose_basin_cell(
                    trial_id=trial.trial_id,
                    basin_id=basin_id,
                    contract=contract,
                    package_root=package_root,
                    identity=identity,
                    run_dir=trial.run_dir,
                    epoch=trial.best_epoch,
                    authenticated_results=authenticated_results,
                )
            except Exception as exc:  # noqa: BLE001 - the last line of defence
                # Nothing below the diagnostic loop may end a trial task: an
                # unanticipated basin failure becomes a typed cell so the
                # shard still carries exactly one record per basin.
                cell = _new_cell(
                    trial.trial_id,
                    basin_id,
                    "other_error",
                    status_detail=f"unhandled {type(exc).__name__}: {exc}",
                )
                basin_extremes, detail, q98 = [], None, None

            cells.append(cell)
            status_counts[cell["status"]] += 1
            extremes.extend(basin_extremes)
            if q98 is not None:
                q98_records.append(q98)
            if detail is not None and detail["support_index"].size:
                detail_writer.write_basin(basin_id, detail)
            if (index + 1) % _PROGRESS_BASIN_STRIDE == 0 or (index + 1) == len(requested_basins):
                progress.basin_event(
                    basin_index=index,
                    basin_id=basin_id,
                    n_basins=len(requested_basins),
                    n_ok=status_counts["ok"],
                    n_detail_rows=detail_writer.n_rows,
                )

        if len(cells) != expected_basin_count:
            raise ObservationDiagnosticError(
                f"{trial.trial_id}: produced {len(cells)} cells, expected exactly {expected_basin_count}"
            )

        detail_index = detail_writer.finalize()
        progress.event(
            "detail_finalization_complete", n_detail_rows=detail_index["n_rows"], sha256=detail_index["sha256"]
        )

        # -- step 4: durable components, then atomic publication ---------- #
        _write_jsonl(attempt_dir / "cells.jsonl", cells)
        _write_jsonl(attempt_dir / "extremes.jsonl", extremes)
        _write_jsonl(attempt_dir / "q98_consequences.jsonl", q98_records)
        (attempt_dir / "detail" / "detail_index.json").write_bytes(canonical_json_bytes(detail_index))

        ended_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        summary = {
            "schema_name": D1_SCHEMA_NAME,
            "schema_version": D1_SCHEMA_VERSION,
            "trial_id": trial.trial_id,
            "search_arm": trial.search_arm,
            "n_cells": len(cells),
            "expected_basin_count": expected_basin_count,
            "cell_status_counts": status_counts,
            "n_extremes": len(extremes),
            "extreme_kinds": list(EXTREME_KINDS),
            "n_q98_records": len(q98_records),
            "n_q98_basins_with_changed_frozen_decision": sum(
                1 for record in q98_records if record["any_frozen_decision_changes"]
            ),
            "detail_index": detail_index,
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "elapsed_s": round(time.monotonic() - started_monotonic, 3),
            "environment": _environment_identity(),
            "identity": identity_payload,
            # Provenance, deliberately outside `identity`: when the package
            # was qualified, not which package it was. Keeping it out of the
            # identity is what makes an identical rerun reusable.
            "package_qualified_at_utc": identity.qualified_at_utc,
            "progress_log_path": str(progress_path),
            "provisional_envelope_is_report_only": True,
            "final_rd1_c4_audit_policy": "unresolved",
        }
        (attempt_dir / "trial_summary.json").write_bytes(canonical_json_bytes(summary))

        human_lines.extend(
            [
                f"cells={len(cells)} (expected {expected_basin_count})",
                "cell_status_counts=" + json.dumps(status_counts, sort_keys=True),
                f"detail_rows={detail_index['n_rows']} detail_sha256={detail_index['sha256']}",
                f"extremes={len(extremes)} over kinds {list(EXTREME_KINDS)}",
                f"q98_records={len(q98_records)} "
                f"changed_frozen_decisions={summary['n_q98_basins_with_changed_frozen_decision']}",
                f"ended_utc={ended_utc} elapsed_s={summary['elapsed_s']}",
                "",
                "No scientific classification, threshold selection, or C4 closure follows from this shard.",
            ]
        )
        (attempt_dir / "human.log").write_text("\n".join(human_lines) + "\n", encoding="utf-8")

        progress.event("shard_finalization_complete", n_cells=len(cells))

        receipt = store.publish(trial.trial_id, attempt_dir=attempt_dir, identity=identity_payload)
        progress.event(
            "receipt_finalization_complete",
            content_sha256=receipt.content_sha256,
            identity_sha256=receipt.identity_sha256,
            n_components=len(receipt.components),
        )
        progress.event("task_end", outcome="published", n_cells=len(cells))
        progress.close()
        summary["receipt"] = receipt.as_dict()
        summary["reused_existing_shard"] = False
        return summary
    except Exception:
        # Leave the attempt directory exactly where it is: it is evidence of
        # what happened, no receipt was written, and no partial shard is
        # visible to the reducer. Automatic destructive cleanup is forbidden.
        progress.event("task_end", outcome="failed")
        progress.close()
        raise


def _write_jsonl(path: Path, records: Sequence[Mapping]) -> None:
    """One canonical JSON object per line, flushed and fsynced.

    Written in the order produced -- basin order for cells and Q98 records,
    and basin-then-extreme-kind order for extremes -- so the file's bytes
    are reproducible for identical inputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(_json_safe(record)) + b"\n" for record in records)
    with open(path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_safe(value):
    """Convert numpy scalars and non-finite floats into JSON-representable
    values. ``NaN``/``inf`` become ``None`` rather than the non-standard
    ``NaN`` literal that would make a receipt unparseable -- an absent
    measurement is explicitly null, never a silently valid-looking number."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
