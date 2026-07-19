"""Stage 1 global forcing-validity mask foundations (Milestone 2K-G-I I-D1).

Implements the binding sample-validity semantics
(docs/stage1_scientific_baseline_design.md sec 6/9a-9b; the approved
"Policy B" global forcing-gap exclusion):

    sample_valid(t) = history_valid_seq(t) & target_boundary_valid_lead(t)

for a fixed ``seq_length`` and ``lead_hours``, where issue time ``t`` is the
final timestep of the historical input sequence.

- ``history_valid_seq(t)`` requires the inclusive window
  ``[t - seq_length + 1, ..., t]`` to lie entirely inside the supplied
  hourly timeline (enough warm-up history) *and* to contain no bad
  (archive-gap) hour anywhere in that window. Lead time never affects
  this window.
- ``target_boundary_valid_lead(t)`` requires ``t + lead_hours`` to lie
  inside the same timeline. It does **not** check the bad-hour mask at
  ``t + lead_hours`` -- the model only consumes historical forcing through
  ``t``, so a forcing gap at the future target hour alone does not
  invalidate the sample here. (Whether the *target* qobs value itself is
  missing at that hour is a separate, per-basin masked-loss concern, not
  handled by this module.)

This module does not apply temporal train/validation/test split dates and
does not touch basin-specific qobs NaNs -- both are separate, later
combination layers. It operates purely on an explicit hourly
``pandas.DatetimeIndex`` (the research timeline) and an explicit boolean
bad-hour vector (or set of bad timestamps to convert to one); it never
reindexes, resamples, or infers missing hours, and it does not know about
basins, splits, or NH package structure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_HOUR = pd.Timedelta(hours=1)


class ValidityMaskError(ValueError):
    """Raised for an invalid timeline, mask, seq_length, or lead specification."""


@dataclass(frozen=True, eq=False)
class ValidityMaskResult:
    """Boolean validity vectors (aligned to the input timeline) plus counts."""

    seq_length: int
    lead_hours: int
    n_timeline: int
    n_bad_hours: int
    history_valid: np.ndarray
    boundary_valid: np.ndarray
    combined_valid: np.ndarray
    n_history_valid: int
    n_boundary_valid: int
    n_combined_valid: int


def _validate_hourly_index(index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValidityMaskError(
            f"timeline must be a pandas.DatetimeIndex, got {type(index)!r}"
        )
    n = len(index)
    if n < 2:
        raise ValidityMaskError(
            f"timeline must have at least 2 timestamps to validate hourly "
            f"spacing, got {n}"
        )
    if index.has_duplicates:
        raise ValidityMaskError("timeline must not contain duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise ValidityMaskError(
            "timeline must be strictly increasing (ascending); found a "
            "non-increasing or descending step"
        )
    deltas = index[1:] - index[:-1]
    irregular = np.asarray(deltas != _HOUR)
    if irregular.any():
        first_bad = int(np.flatnonzero(irregular)[0])
        raise ValidityMaskError(
            f"timeline must be strictly hourly with no gaps; found "
            f"{int(irregular.sum())} irregular step(s), first at position "
            f"{first_bad} ({index[first_bad]} -> {index[first_bad + 1]}, "
            f"delta {deltas[first_bad]})"
        )


def _validate_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValidityMaskError(f"{name} must be a positive int, got {value!r}")
    if value <= 0:
        raise ValidityMaskError(f"{name} must be positive, got {int(value)}")
    return int(value)


def _as_bad_hour_mask(bad_hour_mask, n: int) -> np.ndarray:
    arr = np.asarray(bad_hour_mask)
    if arr.shape != (n,):
        raise ValidityMaskError(
            f"bad_hour_mask length {arr.shape} does not match timeline "
            f"length ({n},)"
        )
    return arr.astype(bool)


def bad_hour_mask_from_timestamps(
    index: pd.DatetimeIndex,
    bad_timestamps,
    *,
    on_out_of_range: str = "error",
) -> np.ndarray:
    """Build a boolean bad-hour vector aligned to ``index`` from timestamps.

    on_out_of_range: strictness policy for a supplied bad timestamp that is
        not present in ``index``:
        - "error" (default): raise ValidityMaskError.
        - "ignore": silently drop it from the resulting mask. Use only when
          the caller has deliberately supplied a gap inventory covering a
          wider period than the current research timeline (e.g. a shared
          MRMS/RTMA gap file spanning more months than the current run).
    """
    _validate_hourly_index(index)
    if on_out_of_range not in ("error", "ignore"):
        raise ValidityMaskError(
            f"on_out_of_range must be 'error' or 'ignore', got {on_out_of_range!r}"
        )
    position_of = {ts: i for i, ts in enumerate(index)}
    mask = np.zeros(len(index), dtype=bool)
    out_of_range = []
    for raw_ts in bad_timestamps:
        ts = pd.Timestamp(raw_ts)
        pos = position_of.get(ts)
        if pos is None:
            out_of_range.append(ts)
            continue
        mask[pos] = True
    if out_of_range and on_out_of_range == "error":
        raise ValidityMaskError(
            f"{len(out_of_range)} bad timestamp(s) fall outside the supplied "
            f"timeline (on_out_of_range='error'); first: {out_of_range[0]}"
        )
    return mask


def compute_history_valid(
    index: pd.DatetimeIndex, bad_hour_mask, seq_length: int
) -> np.ndarray:
    """history_valid_seq(t) for every t in index.

    Valid at position i only if the inclusive window
    [i - seq_length + 1, ..., i] lies entirely within [0, n) and contains
    no True entry in bad_hour_mask.
    """
    _validate_hourly_index(index)
    seq_length = _validate_positive_int(seq_length, "seq_length")
    n = len(index)
    bad = _as_bad_hour_mask(bad_hour_mask, n)

    prefix = np.concatenate(([0], np.cumsum(bad.astype(np.int64))))
    positions = np.arange(n)
    starts = positions - seq_length + 1
    enough_history = starts >= 0
    starts_clipped = np.clip(starts, 0, n)
    window_bad_count = prefix[positions + 1] - prefix[starts_clipped]
    return enough_history & (window_bad_count == 0)


def compute_boundary_valid(index: pd.DatetimeIndex, lead_hours: int) -> np.ndarray:
    """target_boundary_valid_lead(t) for every t in index.

    Valid at position i only if i + lead_hours lies within [0, n) --
    equivalently, t + lead_hours lies inside the timeline. The bad-hour
    mask is deliberately not consulted here.
    """
    _validate_hourly_index(index)
    lead_hours = _validate_positive_int(lead_hours, "lead_hours")
    n = len(index)
    positions = np.arange(n)
    return positions + lead_hours <= n - 1


def compute_validity_mask(
    index: pd.DatetimeIndex, bad_hour_mask, seq_length: int, lead_hours: int
) -> ValidityMaskResult:
    """Combine history- and boundary-validity into sample_valid(t) plus counts."""
    _validate_hourly_index(index)
    n = len(index)
    bad = _as_bad_hour_mask(bad_hour_mask, n)

    history_valid = compute_history_valid(index, bad, seq_length)
    boundary_valid = compute_boundary_valid(index, lead_hours)
    combined_valid = history_valid & boundary_valid

    return ValidityMaskResult(
        seq_length=_validate_positive_int(seq_length, "seq_length"),
        lead_hours=_validate_positive_int(lead_hours, "lead_hours"),
        n_timeline=n,
        n_bad_hours=int(bad.sum()),
        history_valid=history_valid,
        boundary_valid=boundary_valid,
        combined_valid=combined_valid,
        n_history_valid=int(history_valid.sum()),
        n_boundary_valid=int(boundary_valid.sum()),
        n_combined_valid=int(combined_valid.sum()),
    )
