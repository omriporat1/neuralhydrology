"""Reusable synthetic NH-package builder for FlashNHDataset tests.

Adapted from the Phase 2 tiny runtime probe (tmp/stage1_nh_lookup_verification
/probe_nh_lookup.py, not committed) but generalized to multiple basins and
configurable per-basin target-NaN hours, kept minimal for local unit +
integration test use only.

Layout, hourly, day-aligned periods over a 96-hour research window:
    train:      2000-01-01 .. 2000-01-02  (hour 0..47)
    validation: 2000-01-03               (hour 48..71)
    test:       2000-01-04               (hour 72..95)

``dynamic_input_values`` and ``static_attribute_values`` are optional
extensions used by the Stage 1 NH-config-generation tests to exercise
multiple dynamic inputs and a static-attribute contract (mirroring the real
package's ``attributes/attributes.csv`` layout, first column ``gauge_id``,
per NeuralHydrology 1.13's ``GenericDataset.load_attributes`` mechanics).
Both default to ``None``, which preserves the original single-``precip``,
no-static-attributes behavior byte-for-byte.
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
    dynamic_input_values: dict = None,
    static_attribute_values: dict = None,
    declared_gap_hours: list = None,
) -> Path:
    """Write a tiny GenericDataset-compatible package plus a Flash-NH
    gap_timestamps.json mask artifact under ``base_dir``. Returns the config
    YAML path (not yet parsed into a Config object -- callers construct
    ``Config(cfg_path)`` themselves so tests can also mutate the file first).

    ``dynamic_input_values``, if given, is
    ``{input_name: {basin: np.ndarray of length N_HOURS}}``. Its key order
    becomes the config's ``dynamic_inputs`` order and the caller is fully
    responsible for NaN placement in each array (decoupled from
    ``bad_hours``, which then only affects the legacy single-``precip``
    path). If omitted, behavior is identical to the original single-
    ``precip`` fixture.

    ``static_attribute_values``, if given, is ``{basin: {column: value}}``.
    Its columns (union across basins, insertion order of the first basin
    encountered) become ``attributes/attributes.csv`` and the config's
    ``static_attributes`` list. If omitted, no ``attributes/`` directory is
    written and no ``static_attributes`` key is added, matching the
    original fixture exactly.

    ``declared_gap_hours``, if given, overrides which hours are recorded in
    ``masks/gap_timestamps.json`` independent of where NaNs actually live in
    the dynamic arrays -- used to synthesize gap-flag/missingness mismatch
    scenarios. Defaults to ``bad_hours``.
    """
    target_nan_hours_by_basin = target_nan_hours_by_basin or {}
    if declared_gap_hours is None:
        declared_gap_hours = bad_hours

    ts_dir = base_dir / "time_series"
    basins_dir = base_dir / "basins"
    masks_dir = base_dir / "masks"
    ts_dir.mkdir(parents=True, exist_ok=True)
    basins_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(RESEARCH_START, periods=N_HOURS, freq="h")

    if dynamic_input_values is not None:
        dynamic_input_names = list(dynamic_input_values.keys())
    else:
        dynamic_input_names = ["precip"]

    for basin in basins:
        data_vars = {}
        if dynamic_input_values is not None:
            for name in dynamic_input_names:
                data_vars[name] = ("date", np.asarray(dynamic_input_values[name][basin], dtype=np.float64))
        else:
            precip = np.arange(N_HOURS, dtype=np.float64)
            for h in bad_hours:
                precip[h] = np.nan
            data_vars["precip"] = ("date", precip)

        target = np.full(N_HOURS, np.nan, dtype=np.float64)
        for i in range(N_HOURS):
            if i + lead_hours <= N_HOURS - 1:
                target[i] = i + lead_hours
        for h in target_nan_hours_by_basin.get(basin, []):
            target[h] = np.nan
        data_vars[f"qobs_lead{lead_hours}"] = ("date", target)

        ds = xr.Dataset(data_vars, coords={"date": dates})
        ds.to_netcdf(ts_dir / f"{basin}.nc")

    for period in ("train", "validation", "test"):
        (basins_dir / f"{period}.txt").write_text("\n".join(basins) + "\n")

    (masks_dir / "gap_timestamps.json").write_text(
        json.dumps([str(dates[h]) for h in declared_gap_hours], indent=2)
    )
    (masks_dir / "mask_manifest.json").write_text(
        json.dumps({"source": "synthetic test fixture", "n_bad_hours": len(declared_gap_hours)}, indent=2)
    )

    static_columns: list = []
    if static_attribute_values is not None:
        attributes_dir = base_dir / "attributes"
        attributes_dir.mkdir(parents=True, exist_ok=True)
        for basin_cols in static_attribute_values.values():
            for col in basin_cols.keys():
                if col not in static_columns:
                    static_columns.append(col)
        rows = []
        for basin in basins:
            row = {"gauge_id": basin}
            row.update(static_attribute_values.get(basin, {}))
            rows.append(row)
        attrs_df = pd.DataFrame(rows, columns=["gauge_id"] + static_columns)
        attrs_df.to_csv(attributes_dir / "attributes.csv", index=False)

    dynamic_inputs_yaml = "\n".join(f"  - {name}" for name in dynamic_input_names)
    static_attributes_yaml = ""
    if static_columns:
        static_attributes_yaml = "\nstatic_attributes:\n" + "\n".join(f"  - {col}" for col in static_columns) + "\n"

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
{dynamic_inputs_yaml}
{static_attributes_yaml}"""
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
