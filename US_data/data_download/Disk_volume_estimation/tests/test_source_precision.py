"""RD1-C4-D1 section E: float32 source-precision boundary behaviour."""
from __future__ import annotations

import numpy as np
import pytest

from src.baseline.source_precision import (
    FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS,
    float32_local_spacing_ratio,
    float32_ordered_distance,
    float32_ordered_distance_histogram,
    float32_ordered_distance_to_float64,
)


def test_bitwise_equal_values_have_distance_zero():
    result = float32_ordered_distance([1.0, -3.5, 0.0], [1.0, -3.5, 0.0])
    assert result.distance.tolist() == [0, 0, 0]
    assert result.defined.all()


def test_adjacent_representable_values_have_distance_one():
    one = np.float32(1.0)
    result = float32_ordered_distance([one], [np.nextafter(one, np.float32(2.0))])
    assert result.distance.tolist() == [1]


def test_signed_zeros_are_distinct_adjacent_values():
    """+0.0 and -0.0 are two distinct bit patterns that are adjacent in
    IEEE-754 total order, so their ordered-bit distance is 1, not 0.

    This is the module's documented definition and it is asserted here so it
    can never drift into the alternative convention in which the two zeros
    are collapsed to one point.
    """
    result = float32_ordered_distance([0.0, -0.0], [-0.0, 0.0])
    assert result.distance.tolist() == [1, 1]
    assert result.defined.all()


def test_negative_to_positive_distance_follows_this_modules_distinct_zero_definition():
    """``distance(-1.0, +1.0) == 2130706433`` here.

    The classic same-point-zero formulation gives ``2 * 0x3F800000 =
    2130706432``. The extra one is exactly the step between -0.0 and +0.0,
    which this module treats as two distinct adjacent values. The two
    conventions differ by one across zero and must not be confused; this
    test pins the one this module actually implements.
    """
    result = float32_ordered_distance([-1.0], [1.0])
    assert result.distance.tolist() == [2 * 0x3F800000 + 1]
    assert result.distance.tolist() == [2130706433]


def test_subnormal_range_is_counted_value_by_value():
    smallest = np.float32(np.spacing(np.float32(0.0)))
    result = float32_ordered_distance([smallest * 4], [np.float32(0.0)])
    assert result.distance.tolist() == [4]


def test_crossing_zero_through_the_subnormal_range_is_continuous():
    smallest = np.float32(np.spacing(np.float32(0.0)))
    result = float32_ordered_distance([smallest], [-smallest])
    # -smallest -> -0.0 -> +0.0 -> +smallest is three steps.
    assert result.distance.tolist() == [3]


def test_nonfinite_operands_are_undefined_not_zero():
    result = float32_ordered_distance([np.inf, np.nan, -np.inf], [1.0, 1.0, 1.0])
    assert result.distance.tolist() == [-1, -1, -1]
    assert result.defined.tolist() == [False, False, False]
    assert result.status.tolist() == ["inf_operand", "nan_operand", "inf_operand"]


def test_nan_status_wins_when_both_operands_are_nonfinite():
    result = float32_ordered_distance([np.nan], [np.inf])
    assert result.status.tolist() == ["nan_operand"]


def test_distance_is_symmetric():
    a = np.array([1.0, -2.5, 0.0, 1e-40], dtype=np.float32)
    b = np.array([1.5, -2.0, 1e-40, 0.0], dtype=np.float32)
    assert float32_ordered_distance(a, b).distance.tolist() == float32_ordered_distance(b, a).distance.tolist()


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape mismatch"):
        float32_ordered_distance([1.0, 2.0], [1.0])


def test_float64_reconstruction_reports_whether_rounding_was_exact():
    """A float64 value that is not float32-representable must not be allowed
    to present a rounded comparison as an exact one."""
    package = np.array([1.0], dtype=np.float32)
    exact = np.array([1.0], dtype=np.float64)
    inexact = np.array([1.0 + 1e-9], dtype=np.float64)

    distance_exact, representable_exact = float32_ordered_distance_to_float64(package, exact)
    distance_inexact, representable_inexact = float32_ordered_distance_to_float64(package, inexact)

    assert distance_exact.distance.tolist() == [0]
    assert representable_exact.tolist() == [True]
    # 1.0 + 1e-9 rounds back to exactly 1.0f32, so the ordered distance is 0
    # while the values genuinely differ -- which is exactly why the flag
    # exists and why the spacing ratio is reported alongside.
    assert distance_inexact.distance.tolist() == [0]
    assert representable_inexact.tolist() == [False]


def test_spacing_ratio_is_a_ratio_not_an_integer_count():
    ratio = float32_local_spacing_ratio([1.0 + 1e-7], [1.0])
    assert ratio.defined.all()
    assert 0.0 < float(ratio.ratio[0]) < 1.0
    assert float(ratio.spacing[0]) == pytest.approx(float(np.spacing(np.float32(1.0))))


def test_spacing_ratio_uses_the_second_operand_as_the_reference():
    """The denominator is the spacing at ``b``'s magnitude, so the measure is
    deliberately asymmetric -- the package value is the reference."""
    forward = float32_local_spacing_ratio([1000.0], [1.0])
    backward = float32_local_spacing_ratio([1.0], [1000.0])
    assert float(forward.ratio[0]) != float(backward.ratio[0])


def test_spacing_ratio_flags_a_zero_reference_as_min_subnormal_denominator():
    ratio = float32_local_spacing_ratio([1e-40], [0.0])
    assert ratio.denominator_is_min_subnormal.tolist() == [True]


def test_spacing_ratio_is_nan_for_nonfinite_operands():
    ratio = float32_local_spacing_ratio([np.nan, 1.0], [1.0, np.inf])
    assert np.isnan(ratio.ratio).tolist() == [True, True]
    assert ratio.defined.tolist() == [False, False]


def test_histogram_bins_are_deterministic_and_never_fold_undefined_into_a_number():
    distance = np.array([0, 1, 1, 3, 5, 2000, -1, -1], dtype=np.int64)
    defined = distance >= 0
    counts = float32_ordered_distance_histogram(distance, defined=defined)
    assert counts["[0,1)"] == 1
    assert counts["[1,2)"] == 2
    assert counts["[2,4)"] == 1
    assert counts["[4,8)"] == 1
    assert counts["[1024,inf)"] == 1
    assert counts["undefined"] == 2
    assert sum(counts.values()) == distance.size


def test_histogram_bin_edges_are_frozen():
    assert FLOAT32_ORDERED_DISTANCE_HISTOGRAM_BINS == (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
