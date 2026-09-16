"""Source-precision measures for package-versus-reconstruction comparison.

RD1-C4-D1 section E. The two sides of the RD1-C4 observation comparison do
not share a precision:

* the frozen package's ``qobs_m3s`` is stored as ``float32``;
* the value reconstructed from a trial's ``validation_results.p`` is a
  ``float64`` result of a unit conversion applied to an independently
  quantized ``float32`` target.

Neither "float64 ULP" nor a single fractional spacing quotient is an honest
single answer for that pair, so this module reports two explicitly distinct,
separately named measures and never presents one as the other:

:func:`float32_ordered_distance`
    An exact, integer, IEEE-754 ordered-bit distance between two
    ``float32``-representable values: the number of distinct ``float32``
    values strictly between them, plus one. Computed through a monotone key
    transform of the raw bit pattern, so it is correct across signs, across
    zero, and throughout the subnormal range. Because the reconstructed side
    is ``float64``, the caller must decide how it is made
    ``float32``-representable; :func:`float32_ordered_distance_to_float64`
    does that explicitly by rounding the ``float64`` operand to ``float32``
    and reporting, per element, whether that rounding was exact.

:func:`float32_local_spacing_ratio`
    A real-valued ratio ``|a - b| / spacing_float32(|b|)``: how many
    ``float32`` steps *at b's magnitude* separate the two values. This is a
    ratio, not a count. It is the right measure for "how large is this
    difference relative to the storage resolution here?", and it is
    deliberately NOT rounded, floored, or renamed to look like an integer
    ULP distance.

Determinism: both measures are pure functions of the input bits. Neither
depends on array order, platform locale, or any accumulated state. Both are
defined for positive and negative finite values, for signed zero, and
throughout the subnormal range; both report non-finite operands through an
explicit per-element status rather than by returning a silently meaningless
number.

This module contains no scientific hydrological math -- it is a numerical
measurement utility, and nothing here decides whether any difference is
acceptable.

Note on an existing defect, deliberately not changed here:
``package_audit._float32_ulp_distance`` computes its monotone key as
``np.where(bits < 0, 0x80000000 - bits, bits)`` *after* widening the raw
``int32`` bits to ``int64``. That widening discards the ``int32``
wraparound the classic formulation relies on, so its negative-side keys are
shifted by exactly ``2**32``: probed numerically, it reports ``+0.0`` vs
``-0.0`` as ``4294967296`` (correct: ``1``) and ``-1.0`` vs ``+1.0`` as
``2164260864`` (correct: ``2130706432``). Same-sign distances are correct.
That helper belongs to an already-qualified independent-audit path and is
not touched by RD1-C4-D1; this module implements the correct transform for
the new diagnostic rather than silently reusing the defective one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS",
    "Float32OrderedDistance",
    "Float32SpacingRatio",
    "float32_ordered_distance",
    "float32_ordered_distance_to_float64",
    "float32_local_spacing_ratio",
    "float32_ordered_distance_histogram",
]

#: Deterministic, frozen right-open bin edges for reporting an ordered-bit
#: distance distribution: [0,1), [1,2), [2,4), ..., [1024, inf). The first
#: bin is exactly "bitwise equal".
FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS: tuple = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
)

_SIGN_BIT = np.uint32(0x80000000)


@dataclass(frozen=True)
class Float32OrderedDistance:
    """Element-wise ordered-bit distance result.

    ``distance`` is ``-1`` wherever ``defined`` is ``False`` (a non-finite
    operand), never ``0`` -- an undefined distance must never be mistaken
    for "bitwise equal". ``status`` carries one of ``"ok"``, ``"nan_operand"``,
    ``"inf_operand"`` per element.
    """

    distance: np.ndarray
    defined: np.ndarray
    status: np.ndarray

    @property
    def n_defined(self) -> int:
        return int(self.defined.sum())


@dataclass(frozen=True)
class Float32SpacingRatio:
    """Element-wise local-spacing ratio result.

    ``ratio`` is ``|a - b| / spacing_float32(|b|)``; ``spacing`` is the
    denominator actually used. ``ratio`` is ``NaN`` wherever ``defined`` is
    ``False``. ``denominator_is_min_subnormal`` marks elements where ``b``
    is zero (or subnormal-small enough that ``spacing`` collapses to the
    smallest positive subnormal), because there the ratio is a resolution
    statement about the smallest representable step, not about ``b``'s own
    magnitude.
    """

    ratio: np.ndarray
    spacing: np.ndarray
    defined: np.ndarray
    denominator_is_min_subnormal: np.ndarray


def _float32_monotone_key(values: np.ndarray) -> np.ndarray:
    """Map ``float32`` bit patterns to an unsigned key that is monotone in
    IEEE-754 total order.

    ``key = bits ^ 0x80000000`` for a clear sign bit and ``key = ~bits`` for
    a set sign bit, both in ``uint32`` arithmetic. Consequently
    ``key(-inf) = 0``, ``key(-0.0) = 0x7FFFFFFF``, ``key(+0.0) = 0x80000000``,
    ``key(+inf) = 0xFFFFFFFF``, and consecutive representable values -- in
    the subnormal range and across the zero boundary included -- have
    consecutive keys. The result is widened to ``int64`` so a subsequent
    difference cannot wrap.
    """
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    negative = (bits & _SIGN_BIT) != 0
    key = np.where(negative, np.bitwise_not(bits), bits ^ _SIGN_BIT)
    return key.astype(np.int64)


def _status_for(a: np.ndarray, b: np.ndarray) -> tuple:
    nan_operand = np.isnan(a) | np.isnan(b)
    inf_operand = np.isinf(a) | np.isinf(b)
    defined = ~(nan_operand | inf_operand)
    status = np.full(a.shape, "ok", dtype="<U12")
    status[inf_operand] = "inf_operand"
    status[nan_operand] = "nan_operand"  # NaN wins if both apply
    return defined, status


def float32_ordered_distance(a, b) -> Float32OrderedDistance:
    """Exact integer ordered-bit distance between two ``float32`` arrays.

    Both operands are interpreted as ``float32``; a ``float64`` operand is
    rounded to ``float32`` by that interpretation, which silently loses the
    information of whether that rounding was exact -- prefer
    :func:`float32_ordered_distance_to_float64` when one side is genuinely
    ``float64``.

    Semantics: ``0`` means the two ``float32`` values are bit-identical;
    ``1`` means they are adjacent representable values (``+0.0`` and
    ``-0.0`` are adjacent, so their distance is ``1``, not ``0``); the
    distance grows by one per intervening representable value, including
    through the subnormal range. Non-finite operands yield ``distance = -1``
    and ``defined = False``.
    """
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    if a32.shape != b32.shape:
        raise ValueError(f"shape mismatch for ordered distance: {a32.shape} vs {b32.shape}")
    defined, status = _status_for(a32, b32)
    distance = np.abs(_float32_monotone_key(a32) - _float32_monotone_key(b32))
    distance = np.where(defined, distance, np.int64(-1)).astype(np.int64)
    return Float32OrderedDistance(distance=distance, defined=defined, status=status)


def float32_ordered_distance_to_float64(package_float32, reconstructed_float64) -> tuple:
    """Ordered-bit distance between a genuine ``float32`` package value and a
    ``float64`` reconstruction, with the rounding made explicit.

    Returns ``(Float32OrderedDistance, exactly_representable)`` where
    ``exactly_representable[i]`` is ``True`` iff
    ``float64(float32(reconstructed[i])) == reconstructed[i]``, i.e. iff the
    distance for that element is a complete description of the difference
    rather than a description of the difference after rounding the
    reconstruction into the package's storage precision.

    Reporting both is the point: an integer distance alone would quietly
    present a rounded comparison as an exact one.
    """
    package = np.asarray(package_float32, dtype=np.float32)
    reconstructed = np.asarray(reconstructed_float64, dtype=np.float64)
    if package.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch for ordered distance: {package.shape} vs {reconstructed.shape}"
        )
    rounded = reconstructed.astype(np.float32)
    exactly_representable = rounded.astype(np.float64) == reconstructed
    return float32_ordered_distance(package, rounded), exactly_representable


def float32_local_spacing_ratio(a, b) -> Float32SpacingRatio:
    """``|a - b| / spacing_float32(|b|)`` -- a real-valued ratio, not a count.

    ``b`` is the reference operand: the denominator is the width of one
    ``float32`` step at ``b``'s magnitude, ``numpy.spacing(float32(|b|))``.
    This is asymmetric by construction, and the asymmetry is deliberate --
    the package value is the reference in RD1-C4-D1, so "how many storage
    steps at the package's own magnitude?" is the meaningful question.

    The difference itself is evaluated in ``float64`` so that a ``float64``
    reconstruction is not first destroyed by rounding. Non-finite operands
    yield ``NaN`` with ``defined = False``.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    if a64.shape != b64.shape:
        raise ValueError(f"shape mismatch for spacing ratio: {a64.shape} vs {b64.shape}")
    defined, _status = _status_for(a64, b64)
    with np.errstate(invalid="ignore"):
        # A non-finite reference has no local spacing; the element is already
        # reported as undefined below, so the warning it would raise here is
        # noise, not information.
        spacing = np.spacing(np.abs(b64).astype(np.float32)).astype(np.float64)
    min_subnormal = float(np.spacing(np.float32(0.0)))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ratio = np.where(defined, np.abs(a64 - b64) / spacing, np.nan)
    return Float32SpacingRatio(
        ratio=ratio,
        spacing=spacing,
        defined=defined,
        denominator_is_min_subnormal=spacing == min_subnormal,
    )


def float32_ordered_distance_histogram(distance, *, defined=None) -> dict:
    """Deterministic histogram of ordered-bit distances.

    Bins are the frozen right-open edges
    :data:`FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS` plus a final unbounded
    bin. Undefined elements (``defined=False``, or a negative distance) are
    counted separately under ``"undefined"`` and never folded into a
    numeric bin.
    """
    distance = np.asarray(distance, dtype=np.int64)
    if defined is None:
        defined = distance >= 0
    defined = np.asarray(defined, dtype=bool)
    counts: dict = {}
    edges = list(FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS)
    usable = distance[defined]
    for idx, low in enumerate(edges):
        if idx + 1 < len(edges):
            high = edges[idx + 1]
            label = f"[{low},{high})"
            counts[label] = int(((usable >= low) & (usable < high)).sum())
        else:
            counts[f"[{low},inf)"] = int((usable >= low).sum())
    counts["undefined"] = int((~defined).sum())
    return counts
