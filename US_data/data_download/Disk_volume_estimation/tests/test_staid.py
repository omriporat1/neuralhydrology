"""Tests for src/baseline/staid.py (Milestone 2K-G-I I-B)."""
import numpy as np
import pytest

from src.baseline.staid import normalize_staid

# The six non-standard-length STAIDs known to be in the 2,752 eligible set
# (docs/stage1_baseline_package_implementation_plan.md §5).
_KNOWN_9CHAR = "103366092"
_KNOWN_15CHAR = [
    "393109104464500",
    "394839104570300",
    "401733105392404",
    "402114105350101",
    "402913084285400",
]

# ---------------------------------------------------------------------------
# Padding and preservation
# ---------------------------------------------------------------------------


def test_seven_char_pads_to_eight():
    assert normalize_staid("1019000") == "01019000"


@pytest.mark.parametrize(
    "raw,expected",
    [("1", "00000001"), ("123", "00000123"), ("0001", "00000001"), ("9484000", "09484000")],
)
def test_short_digit_strings_pad_to_eight(raw, expected):
    assert normalize_staid(raw) == expected


def test_eight_char_zero_padded_unchanged():
    assert normalize_staid("01019000") == "01019000"


def test_nine_char_known_id_unchanged():
    assert normalize_staid(_KNOWN_9CHAR) == _KNOWN_9CHAR


@pytest.mark.parametrize("staid15", _KNOWN_15CHAR)
def test_fifteen_char_known_ids_unchanged(staid15):
    assert normalize_staid(staid15) == staid15


def test_surrounding_whitespace_stripped():
    assert normalize_staid("  01019000\n") == "01019000"
    assert normalize_staid(" 1019000 ") == "01019000"


def test_leading_zeros_preserved():
    assert normalize_staid("00123456") == "00123456"


def test_output_is_plain_str():
    result = normalize_staid("1019000")
    assert type(result) is str
    result8 = normalize_staid("01019000")
    assert isinstance(result8, str)


# ---------------------------------------------------------------------------
# Rejected string content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_empty_or_whitespace_only_rejected(bad):
    with pytest.raises(ValueError):
        normalize_staid(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "0101900A",        # alphabetic
        "01-19000",        # separator
        "1019000.0",       # decimal form
        "1e7",             # scientific notation
        "+1019000",        # plus sign
        "-1019000",        # minus sign
        "0101 9000",       # internal whitespace
        "01_19000",        # underscore separator
    ],
)
def test_non_digit_forms_rejected(bad):
    with pytest.raises(ValueError):
        normalize_staid(bad)


@pytest.mark.parametrize("length", [10, 11, 12, 13, 14, 16, 20])
def test_unsupported_lengths_rejected(length):
    with pytest.raises(ValueError):
        normalize_staid("1" * length)


# ---------------------------------------------------------------------------
# Rejected input types (never coerced)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        1019000,                  # int
        1019000.0,                # float
        np.int64(1019000),        # NumPy integer scalar
        np.float64(1019000.0),    # NumPy float scalar
        None,
        b"01019000",              # bytes
    ],
)
def test_non_string_input_rejected_with_typeerror(bad):
    with pytest.raises(TypeError):
        normalize_staid(bad)
