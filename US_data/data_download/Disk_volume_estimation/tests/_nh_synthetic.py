"""Reusable synthetic NH-package builder for FlashNHDataset tests.

Adapted from the Phase 2 tiny runtime probe (tmp/stage1_nh_lookup_verification
/probe_nh_lookup.py, not committed) but generalized to multiple basins and
configurable per-basin target-NaN hours, kept minimal for local unit +
integration test use only.

Layout, hourly, day-aligned periods over a 96-hour research window:
    train:      2000-01-01 .. 2000-01-02  (hour 0..47)
    validation: 2000-01-03               (hour 48..71)
    test:       2000-01-04               (hour 72..95)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

N_HOURS = 96
RESEARCH_START = "2000-01-01"


def build_synthetic_package(
    base_dir: Path,
    *,
    basins: list,
    seq_length: int,
    lead_hours: int,
    bad_hours: list = (),
    target_nan_hours_by_basin: dict = None,
) -> Path:
    """Write a tiny GenericDataset-compatible package plus a Flash-NH
    gap_timestamps.json mask artifact under ``base_dir``. Returns the config
    YAML path (not yet parsed into a Config object -- callers construct
    ``Config(cfg_path)`` themselves so tests can also mutate the file first).
    """
    target_nan_hours_by_basin = target_nan_hours_by_basin or {}

    ts_dir = base_dir / "time_series"
    basins_dir = base_dir / "basins"
    masks_dir = base_dir / "masks"
    ts_dir.mkdir(parents=True, exist_ok=True)
    basins_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(RESEARCH_START, periods=N_HOURS, freq="h")

    for basin in basins:
        precip = np.arange(N_HOURS, dtype=np.float64)
        for h in bad_hours:
            precip[h] = np.nan

        target = np.full(N_HOURS, np.nan, dtype=np.float64)
        for i in range(N_HOURS):
            if i + lead_hours <= N_HOURS - 1:
                target[i] = i + lead_hours
        for h in target_nan_hours_by_basin.get(basin, []):
            target[h] = np.nan

        ds = xr.Dataset(
            {
                "precip": ("date", precip),
                f"qobs_lead{lead_hours}": ("date", target),
            },
            coords={"date": dates},
        )
        ds.to_netcdf(ts_dir / f"{basin}.nc")

    for period in ("train", "validation", "test"):
        (basins_dir / f"{period}.txt").write_text("\n".join(basins) + "\n")

    (masks_dir / "gap_timestamps.json").write_text(
        json.dumps([str(dates[h]) for h in bad_hours], indent=2)
    )
    (masks_dir / "mask_manifest.json").write_text(
        json.dumps({"source": "synthetic test fixture", "n_bad_hours": len(bad_hours)}, indent=2)
    )

    cfg_text = f"""
experiment_name: flashnh_test_seq{seq_length}

train_basin_file: {basins_dir}/train.txt
validation_basin_file: {basins_dir}/validation.txt
test_basin_file: {basins_dir}/test.txt

run_dir: {base_dir}/runs
data_dir: {base_dir}
dataset: flashnh

train_start_date: "01/01/2000"
train_end_date: "02/01/2000"
validation_start_date: "03/01/2000"
validation_end_date: "03/01/2000"
test_start_date: "04/01/2000"
test_end_date: "04/01/2000"

target_variables:
  - qobs_lead{lead_hours}

model: cudalstm
hidden_size: 8
head: regression
output_activation: linear
predict_last_n: 1
batch_size: 8

optimizer: Adam
learning_rate: 0.001
loss: MSE

save_weights_every: 1
validate_every: 1
validate_n_random_basins: 1
log_interval: 50
num_workers: 0
seq_length: {seq_length}
epochs: 1
device: cpu
verbose: 0

dynamic_inputs:
  - precip
"""
    cfg_path = base_dir / f"config_seq{seq_length}.yml"
    cfg_path.write_text(cfg_text)
    return cfg_path


def prepare_run_dirs(cfg, base_dir: Path, tag: str) -> None:
    """Mimic BaseTrainer.initialize_training()'s run-dir setup (cfg.train_dir
    must exist before a train-period dataset is constructed directly, without
    going through BaseTrainer)."""
    run_dir = base_dir / "runs" / tag
    cfg.run_dir = run_dir
    cfg.train_dir = run_dir / "train_data"
    cfg.train_dir.mkdir(parents=True, exist_ok=True)
