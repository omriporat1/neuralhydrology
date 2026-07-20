"""Tests for src/baseline/gap_mask_io.py (Milestone 2K-G-I primitives
increment): Milestone 2K-E gap-inventory CSV -> canonical
masks/gap_timestamps.json conversion, product filtering, timeline
validation, and round-trip write/read.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.baseline.gap_mask_io import (
    GapMaskIOError,
    MRMS_PRODUCT,
    RTMA_PRODUCT,
    load_gap_timestamps_json,
    load_missing_hour_products,
    select_gap_timestamps,
    validate_gap_timestamps_against_timeline,
    write_gap_timestamps_json,
)
from src.baseline.validity_mask import ValidityMaskError


def _write_csv(tmp_path, rows, name="fullperiod_missing_hour_products.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_missing_hour_products
# ---------------------------------------------------------------------------


def test_load_missing_hour_products_happy_path(tmp_path):
    p = _write_csv(tmp_path, [
        {"chunk_label": "202301", "product": MRMS_PRODUCT, "valid_time_utc": "2023-01-01T05:00:00Z", "reason": "archive_gap"},
    ])
    df = load_missing_hour_products(p)
    assert list(df["product"]) == [MRMS_PRODUCT]


def test_load_missing_hour_products_missing_file(tmp_path):
    with pytest.raises(GapMaskIOError, match="not found"):
        load_missing_hour_products(tmp_path / "nope.csv")


def test_load_missing_hour_products_rejects_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"chunk_label": ["202301"], "product": [MRMS_PRODUCT]}).to_csv(p, index=False)
    with pytest.raises(GapMaskIOError, match="valid_time_utc"):
        load_missing_hour_products(p)


# ---------------------------------------------------------------------------
# select_gap_timestamps
# ---------------------------------------------------------------------------


def _sample_rows():
    return [
        {"chunk_label": "202301", "product": MRMS_PRODUCT, "valid_time_utc": "2023-01-01T05:00:00Z", "reason": "archive_gap"},
        {"chunk_label": "202301", "product": MRMS_PRODUCT, "valid_time_utc": "2023-01-01T06:00:00Z", "reason": "archive_gap"},
        {"chunk_label": "202301", "product": RTMA_PRODUCT, "valid_time_utc": "2023-01-01T07:00:00Z", "reason": "archive_gap"},
        # duplicate of the first MRMS row -- must be deduped
        {"chunk_label": "202301b", "product": MRMS_PRODUCT, "valid_time_utc": "2023-01-01T05:00:00Z", "reason": "archive_gap"},
    ]


def test_select_gap_timestamps_default_mrms_only(tmp_path):
    df = load_missing_hour_products(_write_csv(tmp_path, _sample_rows()))
    out = select_gap_timestamps(df)
    assert out == [pd.Timestamp("2023-01-01 05:00:00"), pd.Timestamp("2023-01-01 06:00:00")]


def test_select_gap_timestamps_dedupes():
    df = pd.DataFrame(_sample_rows())
    out = select_gap_timestamps(df, products=(MRMS_PRODUCT,))
    assert len(out) == 2  # the duplicate 05:00 row collapses


def test_select_gap_timestamps_can_fold_in_rtma():
    df = pd.DataFrame(_sample_rows())
    out = select_gap_timestamps(df, products=(MRMS_PRODUCT, RTMA_PRODUCT))
    assert pd.Timestamp("2023-01-01 07:00:00") in out
    assert len(out) == 3


def test_select_gap_timestamps_rejects_unknown_product():
    df = pd.DataFrame(_sample_rows())
    with pytest.raises(GapMaskIOError, match="unknown"):
        select_gap_timestamps(df, products=("not_a_real_product",))


def test_select_gap_timestamps_sorted_output():
    rows = [
        {"chunk_label": "x", "product": MRMS_PRODUCT, "valid_time_utc": "2023-01-01T09:00:00Z", "reason": "r"},
        {"chunk_label": "x", "product": MRMS_PRODUCT, "valid_time_utc": "2023-01-01T02:00:00Z", "reason": "r"},
    ]
    df = pd.DataFrame(rows)
    out = select_gap_timestamps(df)
    assert out == sorted(out)


# ---------------------------------------------------------------------------
# validate_gap_timestamps_against_timeline
# ---------------------------------------------------------------------------


def test_validate_gap_timestamps_against_timeline_marks_bad_hours():
    timeline = pd.date_range("2023-01-01 00:00", periods=24, freq="h")
    gaps = [pd.Timestamp("2023-01-01 05:00:00")]
    mask = validate_gap_timestamps_against_timeline(gaps, timeline)
    assert mask.sum() == 1
    assert bool(mask[5]) is True


def test_validate_gap_timestamps_against_timeline_ignores_out_of_range_by_default():
    timeline = pd.date_range("2023-01-01 00:00", periods=24, freq="h")
    gaps = [pd.Timestamp("2022-12-01 00:00:00")]  # far outside timeline
    mask = validate_gap_timestamps_against_timeline(gaps, timeline)
    assert mask.sum() == 0


def test_validate_gap_timestamps_against_timeline_errors_when_requested():
    timeline = pd.date_range("2023-01-01 00:00", periods=24, freq="h")
    gaps = [pd.Timestamp("2022-12-01 00:00:00")]
    with pytest.raises(ValidityMaskError):
        validate_gap_timestamps_against_timeline(gaps, timeline, on_out_of_range="error")


# ---------------------------------------------------------------------------
# write_gap_timestamps_json / load_gap_timestamps_json (round trip)
# ---------------------------------------------------------------------------


def test_write_gap_timestamps_json_format(tmp_path):
    out_path = tmp_path / "masks" / "gap_timestamps.json"
    write_gap_timestamps_json([pd.Timestamp("2023-01-01 05:00:00")], out_path)
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    assert raw == ["2023-01-01T05:00:00Z"]


def test_write_then_load_gap_timestamps_json_roundtrip(tmp_path):
    timestamps = [pd.Timestamp("2023-01-01 05:00:00"), pd.Timestamp("2023-01-02 10:00:00")]
    out_path = write_gap_timestamps_json(timestamps, tmp_path / "masks" / "gap_timestamps.json")
    loaded = load_gap_timestamps_json(out_path)
    assert loaded == timestamps


def test_load_gap_timestamps_json_missing_file(tmp_path):
    with pytest.raises(GapMaskIOError, match="not found"):
        load_gap_timestamps_json(tmp_path / "nope.json")


def test_load_gap_timestamps_json_rejects_non_list(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(GapMaskIOError, match="list"):
        load_gap_timestamps_json(p)


def test_load_gap_timestamps_json_normalizes_tz_aware_input(tmp_path):
    p = tmp_path / "gap_timestamps.json"
    p.write_text(json.dumps(["2023-01-01T05:00:00+00:00"]), encoding="utf-8")
    loaded = load_gap_timestamps_json(p)
    assert loaded == [pd.Timestamp("2023-01-01 05:00:00")]


# ---------------------------------------------------------------------------
# End-to-end: CSV -> select -> write -> load -> validate against a timeline
# ---------------------------------------------------------------------------


def test_end_to_end_csv_to_validated_bad_hour_mask(tmp_path):
    csv_path = _write_csv(tmp_path, _sample_rows())
    df = load_missing_hour_products(csv_path)
    gap_timestamps = select_gap_timestamps(df, products=(MRMS_PRODUCT,))

    out_path = write_gap_timestamps_json(gap_timestamps, tmp_path / "masks" / "gap_timestamps.json")
    loaded = load_gap_timestamps_json(out_path)
    assert loaded == gap_timestamps

    timeline = pd.date_range("2023-01-01 00:00", periods=24, freq="h")
    mask = validate_gap_timestamps_against_timeline(loaded, timeline)
    assert mask.sum() == 2
    assert bool(mask[5]) and bool(mask[6])
