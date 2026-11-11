"""Multi-slice training-period leak tests for FlashNHDataset (water-year rotation, gate 1).

NeuralHydrology accepts a list of training slices and concatenates them per
basin. The lead-shifted target column ``qobs_lead<NN>`` at issue hour t holds
the observation at t + NN, so the last NN issue hours of a slice carry
observations from whatever follows that slice. When the next hour belongs to an
excluded year (checkpoint, assessment or test), those values must not train the
model, must not enter the target scaler, and must not enter the per-basin
target std used by the NSE loss.

Expected sets here are hand-derived from the rule "an issue hour is admitted
only if both the issue hour and its target hour (issue + lead) lie inside one
of this period's declared slices", never from FlashNHDataset's own code.

Synthetic window (see _nh_synthetic.py): 96 hourly steps, 2000-01-01 00:00 ..
2000-01-04 23:00; day d (1-based) covers hours 24*(d-1) .. 24*d - 1. Target at
hour t is the value t + LEAD (NaN past the end). No gap hours.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from neuralhydrology.utils.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import RESEARCH_START, build_synthetic_package, prepare_run_dirs  # noqa: E402

from src.baseline.nh_register import register_flashnh_dataset  # noqa: E402

register_flashnh_dataset()

from neuralhydrology.datasetzoo import get_dataset  # noqa: E402

LEAD = 2
SEQ = 3
BASINS = ["SYN01", "SYN02"]
TARGET = f"qobs_lead{LEAD}"
ORIGIN = pd.Timestamp(RESEARCH_START)
DAY = {d: (24 * (d - 1), 24 * d - 1) for d in (1, 2, 3, 4)}


def _date(day: int) -> str:
    return f"{day:02d}/01/2000"


def _write_periods(cfg_path: Path, train_days: list, validation_day: int, test_day: int, loss: str) -> None:
    text = cfg_path.read_text()
    replacements = {
        "train_start_date": "[" + ", ".join(f'"{_date(d)}"' for d in train_days) + "]",
        "train_end_date": "[" + ", ".join(f'"{_date(d)}"' for d in train_days) + "]",
        "validation_start_date": f'"{_date(validation_day)}"',
        "validation_end_date": f'"{_date(validation_day)}"',
        "test_start_date": f'"{_date(test_day)}"',
        "test_end_date": f'"{_date(test_day)}"',
    }
    for key, value in replacements.items():
        text, n = re.subn(rf"^{key}:.*$", f"{key}: {value}", text, flags=re.M)
        assert n == 1, key
    text, n = re.subn(r"^loss:.*$", f"loss: {loss}", text, flags=re.M)
    assert n == 1
    cfg_path.write_text(text)


def _datasets(tmp_path, train_days, validation_day, test_day, loss="NSE"):
    cfg_path = build_synthetic_package(tmp_path, basins=BASINS, seq_length=SEQ, lead_hours=LEAD, bad_hours=[])
    _write_periods(cfg_path, train_days, validation_day, test_day, loss)
    cfg = Config(cfg_path)
    prepare_run_dirs(cfg, tmp_path, "multislice")
    train = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})
    validation = get_dataset(cfg=cfg, is_train=False, period="validation", scaler=train.scaler)
    return train, validation


def _hour(ts) -> int:
    return int((pd.Timestamp(ts) - ORIGIN).total_seconds() // 3600)


def _admitted_hours(ds, basin) -> set:
    dates = ds._dates[basin][ds.frequencies[0]]
    return {_hour(dates[idx[0]]) for b, idx in ds.lookup_table.values() if b == basin}


def _slice_hours(days) -> set:
    return {h for d in days for h in range(DAY[d][0], DAY[d][1] + 1)}


def _expected_train_hours(train_days) -> set:
    inside = _slice_hours(train_days)
    # Hours 0 .. SEQ-2 lack full input history (NH drops them for NaN inputs before hour 0).
    return {t for t in inside if t >= SEQ - 1 and t + LEAD in inside}


def _expected_train_target_values(train_days) -> np.ndarray:
    """Raw target values allowed into the scaler / per-basin std: rows whose issue
    hour and target hour both lie inside a training slice."""
    inside = _slice_hours(train_days)
    return np.array(sorted(t + LEAD for t in inside if t + LEAD in inside), dtype=np.float64)


# --- Separated slices: train day 1 + day 3, excluded day 2 in between --------------------------


def test_separated_slices_do_not_admit_issue_hours_whose_target_is_in_the_excluded_day(tmp_path):
    train, _ = _datasets(tmp_path, train_days=[1, 3], validation_day=2, test_day=4)
    expected = _expected_train_hours([1, 3])
    assert 22 not in expected and 23 not in expected  # the leak hours, by construction
    for basin in BASINS:
        actual = _admitted_hours(train, basin)
        assert actual == expected, f"{basin}: extra {sorted(actual - expected)}, missing {sorted(expected - actual)}"


def test_separated_slices_mask_excluded_targets_in_the_loaded_train_data(tmp_path):
    train, _ = _datasets(tmp_path, train_days=[1, 3], validation_day=2, test_day=4)
    freq = train.frequencies[0]
    for basin in BASINS:
        dates = train._dates[basin][freq]
        y = train._y[basin][freq][:, 0].numpy()
        by_hour = {_hour(d): v for d, v in zip(dates, y)}
        # 22, 23 -> targets in day 2 (excluded); 70, 71 -> targets in day 4 (test)
        for h in (22, 23, 70, 71):
            assert np.isnan(by_hour[h]), f"{basin} hour {h} target not masked"
        # in-slice targets stay
        for h in (21, 48, 69):
            assert not np.isnan(by_hour[h]), f"{basin} hour {h} target wrongly masked"


def test_separated_slices_scaler_and_per_basin_std_use_only_in_slice_targets(tmp_path):
    train, _ = _datasets(tmp_path, train_days=[1, 3], validation_day=2, test_day=4)
    allowed = _expected_train_target_values([1, 3])
    center = float(train.scaler["xarray_feature_center"][TARGET].values)
    scale = float(train.scaler["xarray_feature_scale"][TARGET].values)
    assert center == pytest.approx(allowed.mean(), rel=1e-6)
    # xarray's std() is the population std (ddof=0), as is np.nanstd
    assert scale == pytest.approx(allowed.std(), rel=1e-5)
    for basin in BASINS:
        std = float(train._per_basin_target_stds[basin].flatten()[0])
        assert std == pytest.approx(allowed.std(), rel=1e-5)


def test_separated_slices_report_masking_in_filter_stats(tmp_path):
    train, validation = _datasets(tmp_path, train_days=[1, 3], validation_day=2, test_day=4)
    stats = train.flashnh_filter_stats
    assert stats.n_period_slices == 2
    # per basin: hours 22, 23 (into day 2) and 70, 71 (into day 4)
    assert stats.n_train_targets_masked_outside_period == 4 * len(BASINS)
    assert validation.flashnh_filter_stats.n_period_slices == 1
    assert validation.flashnh_filter_stats.n_train_targets_masked_outside_period == 0


# --- Adjacent slices: train day 1 + day 2 as two list entries ---------------------------------


def test_adjacent_training_slices_keep_targets_that_fall_in_the_next_training_slice(tmp_path):
    train, _ = _datasets(tmp_path, train_days=[1, 2], validation_day=3, test_day=4)
    expected = _expected_train_hours([1, 2])
    assert {22, 23} <= expected and not ({46, 47} & expected)
    for basin in BASINS:
        assert _admitted_hours(train, basin) == expected
    assert train.flashnh_filter_stats.n_train_targets_masked_outside_period == 2 * len(BASINS)  # 46, 47


# --- Single slice (legacy configs) ------------------------------------------------------------


def test_single_slice_admission_is_unchanged_and_scaler_excludes_the_validation_tail(tmp_path):
    train, validation = _datasets(tmp_path, train_days=[1], validation_day=2, test_day=4)
    expected = _expected_train_hours([1])
    for basin in BASINS:
        assert _admitted_hours(train, basin) == expected
    allowed = _expected_train_target_values([1])
    assert float(train.scaler["xarray_feature_center"][TARGET].values) == pytest.approx(allowed.mean(), rel=1e-6)
    assert train.flashnh_filter_stats.n_period_slices == 1
    assert train.flashnh_filter_stats.n_train_targets_masked_outside_period == 2 * len(BASINS)  # 22, 23
    # evaluation periods are not masked; their lead tail is removed by the boundary filter only
    freq = validation.frequencies[0]
    for basin in BASINS:
        dates = validation._dates[basin][freq]
        tail = [i for i, d in enumerate(dates) if _hour(d) in (46, 47)]
        assert tail and not np.isnan(validation._y[basin][freq][tail, 0].numpy()).any()
        assert not ({46, 47} & _admitted_hours(validation, basin))
