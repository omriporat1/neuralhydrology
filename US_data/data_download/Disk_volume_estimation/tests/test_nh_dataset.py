"""Focused + integration tests for src/baseline/nh_dataset.py (FlashNHDataset).

Runs real neuralhydrology.datasetzoo.GenericDataset construction (via the
registered "flashnh" key) against tiny synthetic packages built with
tests/_nh_synthetic.py. Expected valid-sample sets are hand-derived directly
from the three binding concerns (A: src/baseline/validity_mask.py's
compute_history_valid over the full continuous research timeline; B: a
period-specific issue_time + lead_hours <= period_end(P) boundary, computed
independently of nh_dataset.py's own implementation; C: basin target-NaN),
never re-derived from FlashNHDataset's own filtering code, so a bug shared
between test and implementation cannot hide.

Research window (see _nh_synthetic.py): 96 hours, 2000-01-01 00:00 ..
2000-01-04 23:00, day-aligned periods train=[0,47] validation=[48,71]
test=[72,95]. Bad (archive-gap) hours: 30, 31 (consecutive, inside train),
55 (isolated, inside validation). Lead = 2. Basin-specific manual target-NaN
hours: SYN01 has 20 (inside train, not a boundary hour) and 60 (inside
validation, not a boundary/history-invalid hour); SYN02 has none.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from neuralhydrology.utils.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import N_HOURS, RESEARCH_START, build_synthetic_package, prepare_run_dirs  # noqa: E402

from src.baseline.nh_dataset import (  # noqa: E402
    FlashNHDatasetError,
    _build_research_timeline,
    _normalize_timestamp,
)
from src.baseline.nh_register import register_flashnh_dataset  # noqa: E402
from src.baseline.validity_mask import compute_history_valid  # noqa: E402

register_flashnh_dataset()

from neuralhydrology.datasetzoo import get_dataset  # noqa: E402

PERIOD_RANGES = {"train": (0, 47), "validation": (48, 71), "test": (72, 95)}
LEAD = 2
BAD_HOURS = [30, 31, 55]
BASINS = ["SYN01", "SYN02"]
TARGET_NAN_HOURS = {"SYN01": [20, 60], "SYN02": []}
RESEARCH_ORIGIN = pd.Timestamp(RESEARCH_START)


def _expected_valid_hours(period: str, seq_length: int, target_nan_hours) -> set:
    dates = pd.date_range(RESEARCH_START, periods=N_HOURS, freq="h")
    bad_mask = np.zeros(N_HOURS, dtype=bool)
    for h in BAD_HOURS:
        bad_mask[h] = True
    history_valid = compute_history_valid(dates, bad_mask, seq_length)

    lo, hi = PERIOD_RANGES[period]
    expected = set()
    for t in range(lo, hi + 1):
        if not history_valid[t]:
            continue
        if t + LEAD > hi:
            continue
        if t in target_nan_hours:
            continue
        expected.add(t)
    return expected


def _actual_hours(ds, basin: str) -> set:
    freq = ds.frequencies[0]
    dates = ds._dates[basin][freq]
    hours = set()
    for basin_key, indices in ds.lookup_table.values():
        if basin_key != basin:
            continue
        ts = pd.Timestamp(dates[indices[0]])
        hours.add(int((ts - RESEARCH_ORIGIN).total_seconds() // 3600))
    return hours


def _build_and_run(tmp_path, seq_length):
    cfg_path = build_synthetic_package(
        tmp_path,
        basins=BASINS,
        seq_length=seq_length,
        lead_hours=LEAD,
        bad_hours=BAD_HOURS,
        target_nan_hours_by_basin=TARGET_NAN_HOURS,
    )
    cfg = Config(cfg_path)
    prepare_run_dirs(cfg, tmp_path, f"seq{seq_length}")

    # scaler={} is explicit, not the default: NH 1.13's BaseDataset.__init__
    # gives ``scaler`` a mutable {} default that Python shares across every
    # call site omitting it, so two train-period constructions in the same
    # process (e.g. two tests in one pytest run) would otherwise silently
    # reuse/corrupt each other's normalization -- see the matching comment in
    # src/baseline/nh_structural_preflight.py::check_flashnh_dataset_construction.
    train_ds = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})
    val_ds = get_dataset(cfg=cfg, is_train=False, period="validation", scaler=train_ds.scaler)
    test_ds = get_dataset(cfg=cfg, is_train=False, period="test", scaler=train_ds.scaler)
    return {"train": train_ds, "validation": val_ds, "test": test_ds}


@pytest.mark.parametrize("seq_length", [3, 5])
def test_full_filtering_matches_hand_derived_expectation(tmp_path, seq_length):
    datasets = _build_and_run(tmp_path, seq_length)

    for period, ds in datasets.items():
        for basin in BASINS:
            expected = _expected_valid_hours(period, seq_length, TARGET_NAN_HOURS[basin])
            actual = _actual_hours(ds, basin)
            assert actual == expected, (
                f"{period}/{basin}/seq{seq_length}: expected {sorted(expected)}, got {sorted(actual)}"
            )


@pytest.mark.parametrize("seq_length", [3, 5])
def test_lookup_keys_rebuilt_contiguously(tmp_path, seq_length):
    datasets = _build_and_run(tmp_path, seq_length)
    for ds in datasets.values():
        assert sorted(ds.lookup_table.keys()) == list(range(len(ds.lookup_table)))


def test_cross_period_antecedent_inputs_retained(tmp_path):
    seq_length = 5
    datasets = _build_and_run(tmp_path, seq_length)
    val_ds = datasets["validation"]
    freq = val_ds.frequencies[0]

    earliest_hour = min(_actual_hours(val_ds, "SYN02"))
    assert earliest_hour == 48  # first validation hour, unaffected by bad hours/target-NaN

    basin, indices = next(
        v for v in val_ds.lookup_table.values() if v[0] == "SYN02"
    )
    idx = indices[0]
    dates = val_ds._dates[basin][freq]
    first_input_date = pd.Timestamp(dates[idx + 1 - seq_length])
    validation_start = RESEARCH_ORIGIN + pd.Timedelta(hours=48)
    assert first_input_date < validation_start  # antecedent history reaches back into train


def test_period_boundary_excludes_lead_tail_every_period(tmp_path):
    seq_length = 3
    datasets = _build_and_run(tmp_path, seq_length)
    assert 46 not in _actual_hours(datasets["train"], "SYN02")
    assert 47 not in _actual_hours(datasets["train"], "SYN02")
    assert 70 not in _actual_hours(datasets["validation"], "SYN02")
    assert 71 not in _actual_hours(datasets["validation"], "SYN02")
    assert 94 not in _actual_hours(datasets["test"], "SYN02")
    assert 95 not in _actual_hours(datasets["test"], "SYN02")


def test_validation_removal_counts_are_separate_and_exact(tmp_path):
    # Validation is the cleanest period for exact-count assertions: NH itself
    # performs zero NaN-based filtering when is_train=False (confirmed in
    # Phase 2), so every removal in this period is attributable solely to
    # FlashNHDataset's own override, with no upstream-NH confound (unlike
    # train, where NH already excludes gap/target-NaN samples on its own).
    seq_length = 3
    datasets = _build_and_run(tmp_path, seq_length)
    stats = datasets["validation"].flashnh_filter_stats

    assert stats.period == "validation"
    assert stats.n_before == 24 * len(BASINS)  # NH applies zero filtering for is_train=False
    assert stats.removed_for_forcing_history == 3 * len(BASINS)  # bad hour 55 -> {55,56,57}
    assert stats.removed_for_period_target_boundary == 2 * len(BASINS)  # {70,71}
    assert stats.removed_for_missing_target == 1  # SYN01 hour 60 only
    assert stats.n_kept == stats.n_before - stats.removed_for_forcing_history \
        - stats.removed_for_period_target_boundary - stats.removed_for_missing_target


def test_same_filtering_policy_applied_to_all_periods(tmp_path):
    # The same three concerns, computed the same way, must hold for train,
    # validation, and test alike -- proven by the parametrized set-equality
    # test above holding for all three period keys simultaneously; this test
    # adds an explicit, period-list-driven assertion for readability.
    seq_length = 3
    datasets = _build_and_run(tmp_path, seq_length)
    for period, ds in datasets.items():
        for basin in BASINS:
            expected = _expected_valid_hours(period, seq_length, TARGET_NAN_HOURS[basin])
            assert _actual_hours(ds, basin) == expected


# --- Timestamp-hardening tests (pre-commit pass) --------------------------
#
# Covers: UTC-Z / offset / naive gap timestamps normalizing to the same
# wall-clock hour; genuinely out-of-timeline gap timestamps being ignored
# but counted, never silently dropped without a trace; an in-range but
# unalignable (non-hourly) gap timestamp raising clearly instead of being
# swallowed by validity_mask's on_out_of_range="ignore"; and the research
# timeline ending at the last actual loaded timestamp, not 24h/48h beyond it.


def test_normalize_timestamp_tz_aware_and_naive_agree():
    naive = _normalize_timestamp("2000-01-03 07:00:00")
    z_suffix = _normalize_timestamp("2000-01-03T07:00:00Z")
    offset = _normalize_timestamp("2000-01-03T09:00:00+02:00")
    expected = pd.Timestamp("2000-01-03 07:00:00")

    assert naive == z_suffix == offset == expected
    assert naive.tzinfo is None
    assert z_suffix.tzinfo is None
    assert offset.tzinfo is None


def test_research_timeline_ends_at_last_actual_hour_not_beyond():
    dates_by_basin = {
        "SYN01": {"1h": pd.date_range("2000-01-01", periods=10, freq="h").values},
        "SYN02": {"1h": pd.date_range("2000-01-01 02:00", periods=8, freq="h").values},
    }
    timeline = _build_research_timeline(dates_by_basin, "1h")

    assert timeline[0] == pd.Timestamp("2000-01-01 00:00:00")  # SYN01's earliest
    assert timeline[-1] == pd.Timestamp("2000-01-01 09:00:00")  # SYN01's latest (10 hourly steps from hour 0)
    # No 24h/48h margin: the end is exactly the latest loaded timestamp across basins.
    latest_loaded = max(
        pd.Timestamp(dates_by_basin["SYN01"]["1h"][-1]),
        pd.Timestamp(dates_by_basin["SYN02"]["1h"][-1]),
    )
    assert timeline[-1] == latest_loaded


def test_gap_timestamp_timezone_variants_produce_identical_retained_hours(tmp_path):
    seq_length = 3
    variants = {
        "naive": "2000-01-03 07:00:00",
        "z_suffix": "2000-01-03T07:00:00Z",
        "offset": "2000-01-03T09:00:00+02:00",
    }
    results = {}
    for label, ts_string in variants.items():
        base_dir = tmp_path / label
        cfg_path = build_synthetic_package(
            base_dir, basins=["SYN02"], seq_length=seq_length, lead_hours=LEAD, bad_hours=[55],
        )
        (base_dir / "masks" / "gap_timestamps.json").write_text(json.dumps([ts_string]))
        cfg = Config(cfg_path)
        prepare_run_dirs(cfg, base_dir, f"tzvariant_{label}")

        train_ds = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})  # see scaler={} note above
        val_ds = get_dataset(cfg=cfg, is_train=False, period="validation", scaler=train_ds.scaler)
        results[label] = _actual_hours(val_ds, "SYN02")

        stats = val_ds.flashnh_filter_stats
        assert stats.n_gap_timestamps_loaded == 1
        assert stats.n_gap_timestamps_in_range == 1
        assert stats.n_gap_timestamps_ignored_outside_range == 0
        assert stats.removed_for_forcing_history == 3  # seq_length=3 window {55,56,57}

    assert results["naive"] == results["z_suffix"] == results["offset"]
    for hour in (55, 56, 57):
        assert hour not in results["naive"]


def test_gap_timestamps_outside_research_timeline_are_ignored_and_counted(tmp_path):
    seq_length = 3
    cfg_path = build_synthetic_package(
        tmp_path, basins=["SYN02"], seq_length=seq_length, lead_hours=LEAD, bad_hours=[55],
    )
    (tmp_path / "masks" / "gap_timestamps.json").write_text(
        json.dumps(["2000-01-03T07:00:00Z", "1990-01-01T00:00:00Z", "2100-01-01T00:00:00Z"])
    )
    cfg = Config(cfg_path)
    prepare_run_dirs(cfg, tmp_path, "outside_range")

    train_ds = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})  # see scaler={} note above
    val_ds = get_dataset(cfg=cfg, is_train=False, period="validation", scaler=train_ds.scaler)

    stats = val_ds.flashnh_filter_stats
    assert stats.n_gap_timestamps_loaded == 3
    assert stats.n_gap_timestamps_in_range == 1
    assert stats.n_gap_timestamps_ignored_outside_range == 2
    assert stats.removed_for_forcing_history == 3


def test_in_range_unaligned_gap_timestamp_raises(tmp_path):
    seq_length = 3
    cfg_path = build_synthetic_package(
        tmp_path, basins=["SYN02"], seq_length=seq_length, lead_hours=LEAD, bad_hours=[],
    )
    (tmp_path / "masks" / "gap_timestamps.json").write_text(
        json.dumps(["2000-01-03T07:30:00Z"])  # inside validation's own window, off the hourly grid
    )
    cfg = Config(cfg_path)
    prepare_run_dirs(cfg, tmp_path, "unaligned")

    # Train succeeds: 2000-01-03 07:30 lies outside train's own instance timeline
    # (silently ignored, not raised).
    train_ds = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})  # see scaler={} note above

    # Validation's own timeline includes 2000-01-03 07:30's wall-clock hour range,
    # so the misaligned timestamp must raise rather than be silently dropped.
    with pytest.raises(FlashNHDatasetError):
        get_dataset(cfg=cfg, is_train=False, period="validation", scaler=train_ds.scaler)
