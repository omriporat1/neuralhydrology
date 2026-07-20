"""Stage 1 forcing-gap inventory -> canonical ``masks/gap_timestamps.json``
conversion (Milestone 2K-G-I primitives increment).

This module does **not** decide gap policy -- that is already signed off
(``docs/stage1_scientific_baseline_design.md`` sec 6: Policy B, MRMS-gap
hours hard-exclude training windows; RTMA gap hours optionally foldable in)
-- and it does **not** compute gap hours from raw archive data. It only
reformats and validates an *already-produced* Milestone 2K-E gap inventory
(``fullperiod_missing_hour_products.csv``, columns ``chunk_label, product,
valid_time_utc, reason`` -- see ``scripts/generate_fullperiod_audit_tables.py``)
into the flat JSON timestamp list that
``src/baseline/nh_dataset.py``'s ``FlashNHDataset`` already expects at
``<data_dir>/masks/gap_timestamps.json`` (see that module's
``_load_gap_timestamps``: a flat JSON list of ISO timestamp strings, UTC,
commonly with a trailing ``Z``).

Reuses :func:`src.baseline.validity_mask.bad_hour_mask_from_timestamps` for
validating a converted timestamp list against a research timeline rather
than reimplementing that logic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validity_mask import bad_hour_mask_from_timestamps

__all__ = [
    "GapMaskIOError",
    "MRMS_PRODUCT",
    "RTMA_PRODUCT",
    "load_missing_hour_products",
    "select_gap_timestamps",
    "validate_gap_timestamps_against_timeline",
    "write_gap_timestamps_json",
    "load_gap_timestamps_json",
]

# Must match scripts/generate_fullperiod_audit_tables.py's constants of the
# same name (duplicated here rather than imported: src/ must not depend on
# scripts/).
MRMS_PRODUCT = "mrms_qpe_1h_pass1"
RTMA_PRODUCT = "rtma_conus_aws_2p5km"

_REQUIRED_COLUMNS = ("chunk_label", "product", "valid_time_utc", "reason")


class GapMaskIOError(ValueError):
    """Raised for a malformed gap-inventory CSV or an invalid gap-timestamp artifact."""


def load_missing_hour_products(csv_path) -> pd.DataFrame:
    """Load and validate a Milestone 2K-E ``fullperiod_missing_hour_products.csv``."""
    p = Path(csv_path)
    if not p.is_file():
        raise GapMaskIOError(f"missing-hour-products CSV not found: {p}")
    df = pd.read_csv(p, dtype=str)
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise GapMaskIOError(f"{p}: missing required column(s) {missing_cols}; have {list(df.columns)}")
    return df


def select_gap_timestamps(df: pd.DataFrame, products=(MRMS_PRODUCT,)) -> list:
    """Filter to the given product(s), dedupe, and return sorted UTC Timestamps.

    Default is MRMS-only, matching the signed-off Policy B hard-exclusion
    driver; pass ``products=(MRMS_PRODUCT, RTMA_PRODUCT)`` to fold RTMA gaps
    in as well.
    """
    unknown = sorted(set(products) - {MRMS_PRODUCT, RTMA_PRODUCT})
    if unknown:
        raise GapMaskIOError(f"unknown product(s) requested: {unknown}")
    subset = df.loc[df["product"].isin(products)]
    timestamps = sorted({_to_utc_naive(v) for v in subset["valid_time_utc"]})
    return timestamps


def _to_utc_naive(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def validate_gap_timestamps_against_timeline(
    timestamps: list, timeline: pd.DatetimeIndex, *, on_out_of_range: str = "ignore"
) -> np.ndarray:
    """Thin wrapper over ``validity_mask.bad_hour_mask_from_timestamps``.

    Defaults to ``on_out_of_range="ignore"`` here (unlike that function's own
    "error" default) because a shared gap inventory legitimately spans a
    wider period than any one research timeline -- matching
    ``FlashNHDataset``'s own gap-timestamp strictness policy of silently
    (but countably) ignoring out-of-range entries.
    """
    return bad_hour_mask_from_timestamps(timeline, timestamps, on_out_of_range=on_out_of_range)


def write_gap_timestamps_json(timestamps: list, out_path) -> Path:
    """Write the flat ISO-8601 (``...Z``) timestamp list FlashNHDataset expects."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ") for ts in timestamps]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_gap_timestamps_json(path) -> list:
    """Standalone re-reader mirroring ``nh_dataset._load_gap_timestamps`` without
    importing that module (avoids a hard dependency on ``neuralhydrology``)."""
    p = Path(path)
    if not p.is_file():
        raise GapMaskIOError(f"gap timestamps artifact not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise GapMaskIOError(f"{p}: expected a JSON list of ISO timestamp strings, got {type(raw).__name__}")
    result = []
    for v in raw:
        ts = pd.Timestamp(v)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        result.append(ts)
    return result
