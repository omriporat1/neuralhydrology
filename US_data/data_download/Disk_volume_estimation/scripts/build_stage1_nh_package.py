#!/usr/bin/env python3
"""Build Stage 1 NeuralHydrology GenericDataset-compatible pilot package (Milestone 2K-G-B).

Merges curated forcing Parquets + target NCs into per-basin NetCDF files and
writes static attributes, basin lists, NH YAML configs, Slurm scripts, manifests.

Gap-fill policy for this pilot package (Smoke 0/1 technical policy only):
  - MRMS QPE (136 gap hours/basin): fill mrms_qpe_1h_mm NaN with 0.0 mm; retain gap flag.
  - RTMA all-vars (2 gap hours/basin): linear interpolation across both hours; retain flag.
  - Target qobs_m3s: preserve NaN exactly. No interpolation. No filling.

WARNING: This gap-fill policy is for Smoke 0/1 technical testing only.
Do NOT carry it unchanged into scientific baseline training.
See docs/stage1_neuralhydrology_preflight.md §8.2 for final training policy options.

Attribute source on h2o (after git pull):
  <repo>/reports/flashnh_basin_screening_v001/all_basins_merged.parquet
  (accepts .parquet or .csv; must contain STAID or gauge_id column
   plus DRAIN_SQKM, LAT_GAGE, LNG_GAGE, BFI_AVE)

Usage:
  python scripts/build_stage1_nh_package.py \\
    --forcing-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/\\
stage1_basin_hourly_forcings_v001_5basin_corrected_pilot_20260630T100505Z/ \\
    --target-dir  /data42/omrip/Flash-NH/tmp/stage1_target_package_v001/ \\
    --out-dir     /data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/ \\
    --staids      01019000,01022500,01033000,01038000,01049500 \\
    --attributes-csv <repo>/reports/flashnh_basin_screening_v001/all_basins_merged.parquet \\
    --expected-basins 5 \\
    --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import textwrap
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_NAME = Path(__file__).name
CREATED_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PERIOD_START      = pd.Timestamp("2020-10-14 00:00:00")  # tz-naive
_PERIOD_END        = pd.Timestamp("2025-12-31 23:00:00")  # tz-naive
_FULL_PERIOD_HOURS = 45_720
_DATE_UNITS        = "hours since 2020-10-14 00:00:00"

_TRAIN_END  = "2022-12-31"
_VAL_START  = "2023-01-01"
_VAL_END    = "2023-12-31"
_TEST_START = "2024-01-01"
_TEST_END   = "2025-12-31"

# Canonical RTMA data columns (10 variables, corrected v001 schema)
_RTMA_COLS: list[str] = [
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct",
    "rtma_vis_m", "rtma_gust_ms", "rtma_ceil_m",
]
_FORCING_DATA_COLS: list[str] = ["mrms_qpe_1h_mm"] + _RTMA_COLS  # 11
_GAP_FLAG_COLS:     list[str] = ["mrms_qpe_1h_mm_gap", "rtma_gap"]
_ALL_DYNAMIC_COLS:  list[str] = _FORCING_DATA_COLS + _GAP_FLAG_COLS  # 13

# Variables that must never appear in this package
_FORBIDDEN_VARS: set[str] = {
    "rtma_weasd_kgm2",   # removed from v001 schema (absent from all 63 source months)
    "rtma_10si_ms",      # 2G-only name; not in corrected v001 schema
    "rtma_i10fg_ms",     # 2G-only name; gust is rtma_gust_ms in v001
}

_REQUIRED_ATTR_COLS: list[str] = ["DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "BFI_AVE"]

_FILL_VALUE = np.float32(-9999.0)

_VAR_META: dict[str, dict] = {
    "mrms_qpe_1h_mm":     {"units": "mm",        "long_name": "MRMS QPE 1-hour accumulation (Pass1)"},
    "rtma_2t_K":          {"units": "K",          "long_name": "2-metre temperature"},
    "rtma_2d_K":          {"units": "K",          "long_name": "2-metre dewpoint temperature"},
    "rtma_2sh_kgkg":      {"units": "kg kg-1",    "long_name": "2-metre specific humidity"},
    "rtma_sp_Pa":         {"units": "Pa",         "long_name": "Surface pressure"},
    "rtma_10u_ms":        {"units": "m s-1",      "long_name": "10-metre U wind component"},
    "rtma_10v_ms":        {"units": "m s-1",      "long_name": "10-metre V wind component"},
    "rtma_tcc_pct":       {"units": "%",          "long_name": "Total cloud cover fraction"},
    "rtma_vis_m":         {"units": "m",          "long_name": "Visibility"},
    "rtma_gust_ms":       {"units": "m s-1",      "long_name": "10-metre wind gust speed"},
    "rtma_ceil_m":        {"units": "m",          "long_name": "Cloud ceiling height"},
    "mrms_qpe_1h_mm_gap": {"units": "1",
                           "long_name": "MRMS QPE gap flag (1=archive gap; mrms_qpe_1h_mm set to 0.0 mm for Smoke 0/1)"},
    "rtma_gap":           {"units": "1",
                           "long_name": "RTMA gap flag (1=interpolated; all RTMA vars linearly interpolated for Smoke 0/1)"},
    "qobs_m3s":           {"units": "m3 s-1",     "long_name": "Streamflow (NaN preserved; loss-masked by NH)"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--forcing-dir", required=True,
        help="Curated forcing product directory (contains time_series/*.parquet).",
    )
    p.add_argument(
        "--target-dir", required=True,
        help="Target package v001 directory (contains time_series/*.nc with qobs_m3s).",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output NH package directory. Must not exist unless --force.",
    )
    p.add_argument(
        "--staids", default=None,
        help="Comma-separated STAIDs to include. "
             "If omitted, all *.parquet files under forcing-dir/time_series/ are used.",
    )
    p.add_argument(
        "--attributes-csv", required=True,
        help="Static attribute file (.csv or .parquet). Must have a STAID or gauge_id "
             "column plus DRAIN_SQKM, LAT_GAGE, LNG_GAGE, BFI_AVE. "
             "Default source: <repo>/reports/flashnh_basin_screening_v001/all_basins_merged.parquet",
    )
    p.add_argument(
        "--expected-basins", type=int, default=None,
        help="Expected number of basins. Builder exits non-zero if count mismatches.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite out-dir if it already exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print plan; write nothing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _norm_staid(s: object) -> str:
    try:
        return f"{int(float(str(s).strip())):08d}"
    except (ValueError, TypeError):
        return str(s).strip().zfill(8)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _full_period_grid() -> pd.DatetimeIndex:
    grid = pd.date_range(_PERIOD_START, periods=_FULL_PERIOD_HOURS, freq="h", name="date")
    assert len(grid) == _FULL_PERIOD_HOURS
    assert grid[-1] == _PERIOD_END, f"Grid end {grid[-1]} != {_PERIOD_END}"
    return grid


def _yyyymmdd_to_ddmmyyyy(s: str) -> str:
    """Convert ISO date string YYYY-MM-DD to NH 1.13 required DD/MM/YYYY format."""
    y, m, d = s.split("-")
    return f"{d}/{m}/{y}"


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _load_attributes(attr_path: Path, staids: list[str]) -> pd.DataFrame:
    """Load attribute file, normalize gauge_id, filter to requested STAIDs."""
    if attr_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(attr_path)
    else:
        raw = pd.read_csv(attr_path, dtype=str)

    # Find ID column
    id_col: str | None = None
    for candidate in ("gauge_id", "STAID", "staid"):
        if candidate in raw.columns:
            id_col = candidate
            break
    if id_col is None:
        log.error("Attributes file has no 'gauge_id' or 'STAID' column. Found: %s",
                  list(raw.columns[:20]))
        sys.exit(1)

    raw = raw.copy()
    raw["gauge_id"] = raw[id_col].apply(_norm_staid)
    if id_col != "gauge_id":
        raw = raw.drop(columns=[id_col])
    raw = raw.set_index("gauge_id")

    missing_cols = [c for c in _REQUIRED_ATTR_COLS if c not in raw.columns]
    if missing_cols:
        log.error("Attributes file missing required columns: %s. Have: %s",
                  missing_cols, list(raw.columns[:20]))
        sys.exit(1)

    missing_staids = [s for s in staids if s not in raw.index]
    if missing_staids:
        log.error("STAIDs not found in attributes file: %s", missing_staids)
        sys.exit(1)

    return raw.loc[staids].copy()


def _load_forcing_parquet(path: Path, full_grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Load per-basin forcing Parquet and align to full-period grid (tz-naive)."""
    df = pd.read_parquet(path)

    # Strip timezone from tz-aware UTC index
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "date"

    # Align to full grid (absent hours → NaN, gap hours already present with NaN values)
    df = df.reindex(full_grid)

    forbidden_present = _FORBIDDEN_VARS & set(df.columns)
    if forbidden_present:
        log.error("Forbidden columns in forcing Parquet: %s", sorted(forbidden_present))
        sys.exit(1)

    missing_cols = [c for c in _ALL_DYNAMIC_COLS if c not in df.columns]
    if missing_cols:
        log.error("Forcing Parquet missing expected columns: %s", missing_cols)
        sys.exit(1)

    return df


def _load_target_qobs(path: Path, full_grid: pd.DatetimeIndex) -> pd.Series:
    """Load qobs_m3s from target NC and align to full-period grid."""
    with xr.open_dataset(path) as ds:
        # Accept 'date' or 'time' coordinate (target builder uses 'date')
        tc = "date" if "date" in ds.coords else "time"
        idx = pd.DatetimeIndex(ds.coords[tc].values)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        qobs = pd.Series(ds["qobs_m3s"].values.astype("float64"), index=idx, name="qobs_m3s")

    qobs.index.name = "date"
    # Reindex onto full grid; NaN at hours not covered by target (expected: none for v001)
    n_before = int(qobs.isna().sum())
    qobs = qobs.reindex(full_grid)
    n_after = int(qobs.isna().sum())
    if n_after > n_before:
        log.warning("  qobs reindex added %d NaN (target NC may not cover full period)",
                    n_after - n_before)
    return qobs


# ---------------------------------------------------------------------------
# Gap-fill
# ---------------------------------------------------------------------------

def _apply_gap_fill(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply Smoke 0/1 pilot gap-fill policy. Returns (filled_df, report_dict)."""
    df = df.copy()

    # MRMS: fill precipitation NaN with 0.0; gap flag preserved as-is
    n_mrms_nan = int(df["mrms_qpe_1h_mm"].isna().sum())
    df["mrms_qpe_1h_mm"] = df["mrms_qpe_1h_mm"].fillna(0.0)

    # RTMA: linear interpolation for up to 3 consecutive gap hours; flag preserved
    rtma_filled: dict[str, int] = {}
    for col in _RTMA_COLS:
        n_before = int(df[col].isna().sum())
        if n_before > 0:
            df[col] = df[col].interpolate(method="time", limit=3)
        rtma_filled[col] = n_before - int(df[col].isna().sum())

    # Convert gap flags: bool/int → float32 (NH requires numeric input)
    for flag in _GAP_FLAG_COLS:
        df[flag] = df[flag].astype("float32")

    # Validate: no NaN should remain in any forcing data column
    for col in _FORCING_DATA_COLS:
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            log.error("  NaN remains in %s after gap-fill: %d hours", col, n_nan)

    return df, {"mrms_nan_filled": n_mrms_nan, "rtma_filled_per_var": rtma_filled}


# ---------------------------------------------------------------------------
# Per-basin NC writer
# ---------------------------------------------------------------------------

def _write_basin_nc(
    staid: str,
    wide: pd.DataFrame,
    qobs: pd.Series,
    nc_path: Path,
) -> None:
    """Build and write per-basin NeuralHydrology-compatible NC file."""
    nc_path.parent.mkdir(parents=True, exist_ok=True)

    data_vars: dict = {}
    for col in _ALL_DYNAMIC_COLS:
        data_vars[col] = (["date"], wide[col].values.astype("float32"), _VAR_META[col])
    data_vars["qobs_m3s"] = (["date"], qobs.values.astype("float32"), _VAR_META["qobs_m3s"])

    ds = xr.Dataset(data_vars, coords={"date": wide.index.values})
    ds.attrs = {
        "Conventions":    "CF-1.8",
        "gauge_id":       staid,
        "package":        "stage1_nh_pilot_v001",
        "created_utc":    datetime.now(timezone.utc).isoformat(),
        "gap_fill_note":  (
            "PILOT ONLY (Smoke 0/1): MRMS NaN→0.0mm; RTMA NaN→linear interp. "
            "Not final scientific training policy."
        ),
    }
    ds["date"].attrs = {
        "timezone":    "UTC (naive datetime64; no tz offset stored)",
        "description": "UTC hourly, proleptic_gregorian calendar",
    }

    enc: dict = {}
    for var in ds.data_vars:
        enc[var] = {"dtype": "float32", "_FillValue": float(_FILL_VALUE)}
    enc["date"] = {
        "dtype":    "float64",
        "units":    _DATE_UNITS,
        "calendar": "proleptic_gregorian",
    }

    tmp = nc_path.with_suffix(".nc.tmp")
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(str(tmp), encoding=enc)
    ds.close()
    if nc_path.exists():
        nc_path.unlink()
    tmp.rename(nc_path)


# ---------------------------------------------------------------------------
# Package-level outputs
# ---------------------------------------------------------------------------

def _write_attributes(attrs: pd.DataFrame, out_path: Path) -> None:
    attrs.index.name = "gauge_id"
    # Write all available columns; NH filters to static_attributes list at runtime
    attrs.to_csv(out_path)
    log.info("Wrote attributes.csv (%d basins, %d cols)", len(attrs), len(attrs.columns))


def _write_basin_lists(basins_dir: Path, staids: list[str]) -> None:
    basins_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(staids) + "\n"
    for smoke in ("smoke0", "smoke1"):
        for split in ("train", "val", "test"):
            (basins_dir / f"{smoke}_{split}.txt").write_text(content)
    log.info("Wrote 6 basin list files to basins/")


def _write_configs(configs_dir: Path) -> None:
    configs_dir.mkdir(parents=True, exist_ok=True)
    moriah_data = "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001"
    moriah_runs = "/sci/labs/efratmorin/omripo/Flash-NH/runs"

    # NH 1.13 requires DD/MM/YYYY for all _date fields (not ISO YYYY-MM-DD).
    _train_s = _yyyymmdd_to_ddmmyyyy("2020-10-14")
    _train_e = _yyyymmdd_to_ddmmyyyy(_TRAIN_END)
    _val_s   = _yyyymmdd_to_ddmmyyyy(_VAL_START)
    _val_e   = _yyyymmdd_to_ddmmyyyy(_VAL_END)
    _test_s  = _yyyymmdd_to_ddmmyyyy(_TEST_START)
    _test_e  = _yyyymmdd_to_ddmmyyyy(_TEST_END)

    common_head = textwrap.dedent(f"""\
        run_dir: {moriah_runs}
        data_dir: {moriah_data}
        dataset: generic

        train_start_date: "{_train_s}"
        train_end_date: "{_train_e}"
        validation_start_date: "{_val_s}"
        validation_end_date: "{_val_e}"
        test_start_date: "{_test_s}"
        test_end_date: "{_test_e}"

        target_variables:
          - qobs_m3s

        static_attributes:
          - DRAIN_SQKM
          - LAT_GAGE
          - LNG_GAGE
          - BFI_AVE

        model: cudalstm
        hidden_size: 64
        head: regression
        output_activation: linear
        predict_last_n: 1
        batch_size: 256

        optimizer: Adam
        learning_rate: 0.001
        clip_gradient_norm: 1.0
        loss: NSE

        save_weights_every: 1
        validate_every: 1
        validate_n_random_basins: 5
        log_interval: 50
        num_workers: 4
    """)

    smoke0 = textwrap.dedent(f"""\
        # Flash-NH Stage 1 — Smoke 0: Rain-only technical smoke
        # PURPOSE: Verify NH loads the package and produces finite loss.
        # NOT a scientific baseline. seq_length=24 chosen for minimal plumbing overhead.
        # NeuralHydrology 1.13 compatibility: dataset=generic, DD/MM/YYYY dates, epochs key.
        experiment_name: flashnh_stage1_smoke0

        train_basin_file: {moriah_data}/basins/smoke0_train.txt
        validation_basin_file: {moriah_data}/basins/smoke0_val.txt
        test_basin_file: {moriah_data}/basins/smoke0_test.txt

    """) + common_head + textwrap.dedent("""\
        seq_length: 24
        epochs: 2

        dynamic_inputs:
          - mrms_qpe_1h_mm
          - mrms_qpe_1h_mm_gap
    """)

    smoke1 = textwrap.dedent(f"""\
        # Flash-NH Stage 1 — Smoke 1: Minimal meteorology smoke
        # PURPOSE: Verify 6 core RTMA forcing variables load and train correctly.
        # rtma_sp_Pa is in the NC but excluded here (deferred to Smoke 2 — normalization review).
        # seq_length=72 (3 days): first step up from Smoke 0's 24 h.
        # NeuralHydrology 1.13 compatibility: dataset=generic, DD/MM/YYYY dates, epochs key.
        experiment_name: flashnh_stage1_smoke1

        train_basin_file: {moriah_data}/basins/smoke1_train.txt
        validation_basin_file: {moriah_data}/basins/smoke1_val.txt
        test_basin_file: {moriah_data}/basins/smoke1_test.txt

    """) + common_head + textwrap.dedent("""\
        seq_length: 72
        epochs: 3

        dynamic_inputs:
          - mrms_qpe_1h_mm
          - rtma_2t_K
          - rtma_2d_K
          - rtma_2sh_kgkg
          - rtma_10u_ms
          - rtma_10v_ms
          - mrms_qpe_1h_mm_gap
          - rtma_gap
        # rtma_sp_Pa: present in NC but excluded (Smoke 2). Review normalization range first.
    """)

    (configs_dir / "stage1_smoke0_nh.yml").write_text(smoke0)
    (configs_dir / "stage1_smoke1_nh.yml").write_text(smoke1)
    log.info("Wrote stage1_smoke0_nh.yml and stage1_smoke1_nh.yml")


def _write_slurm(slurm_dir: Path) -> None:
    slurm_dir.mkdir(parents=True, exist_ok=True)
    moriah_base = "/sci/labs/efratmorin/omripo/Flash-NH"

    for smoke, tlim in [("smoke0", "01:00:00"), ("smoke1", "02:00:00")]:
        script = textwrap.dedent(f"""\
            #!/usr/bin/env bash
            #SBATCH --job-name=flashnh-{smoke}
            #SBATCH --partition=gpu           # Confirm: sinfo -s or HURCS wiki
            #SBATCH --gres=gpu:1
            #SBATCH --cpus-per-task=4
            #SBATCH --mem=32G
            #SBATCH --time={tlim}
            #SBATCH --output={moriah_base}/logs/slurm-{smoke}-%j.out

            set -euo pipefail

            FLASHNH_BASE="{moriah_base}"
            NH_REPO="${{FLASHNH_BASE}}/repos/neuralhydrology"
            CONFIG="${{FLASHNH_BASE}}/data/stage1_pilot_v001/configs/stage1_{smoke}_nh.yml"
            ENV_NAME="flashnh-moriah"

            # Activate conda env (adjust init path to Moriah's conda location)
            source "${{FLASHNH_BASE}}/envs/${{ENV_NAME}}/etc/profile.d/conda.sh" 2>/dev/null || \\
              source /etc/profile.d/conda.sh
            conda activate "${{ENV_NAME}}"

            cd "${{NH_REPO}}"
            python -m neuralhydrology.training --config-file "${{CONFIG}}"
        """)
        (slurm_dir / f"{smoke}.sh").write_text(script)
    log.info("Wrote smoke0.sh and smoke1.sh to slurm/")


def _write_manifests(
    manifests_dir: Path,
    basin_summaries: list[dict],
    gap_fill_reports: list[dict],
) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "product":       "stage1_nh_pilot_v001",
        "schema_version": "1.0",
        "created_utc":   CREATED_UTC,
        "n_basins":      len(basin_summaries),
        "period_start":  str(_PERIOD_START),
        "period_end":    str(_PERIOD_END),
        "n_hours":       _FULL_PERIOD_HOURS,
        "gap_fill_policy": {
            "mrms":  "fill NaN with 0.0 mm (Smoke 0/1 pilot policy only)",
            "rtma":  "linear interpolation up to 3 consecutive hours (pilot policy only)",
            "qobs":  "NaN preserved exactly",
        },
        "basins": basin_summaries,
    }
    with open(manifests_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    schema_rows = []
    for col, meta in _VAR_META.items():
        col_type = ("target" if col == "qobs_m3s"
                    else "gap_flag" if col.endswith("_gap")
                    else "dynamic_forcing")
        schema_rows.append({
            "variable":    col,
            "units":       meta.get("units", ""),
            "long_name":   meta.get("long_name", ""),
            "type":        col_type,
            "dtype":       "float32",
            "fill_value":  str(float(_FILL_VALUE)),
            "nan_after_package_build": "yes" if col == "qobs_m3s" else "no",
        })
    pd.DataFrame(schema_rows).to_csv(manifests_dir / "variable_schema.csv", index=False)

    gap_rows = []
    for r in gap_fill_reports:
        row: dict = {"gauge_id": r["gauge_id"], "mrms_nan_filled": r["mrms_nan_filled"]}
        for col, n in r.get("rtma_filled_per_var", {}).items():
            row[f"rtma_filled_{col}"] = n
        gap_rows.append(row)
    pd.DataFrame(gap_rows).to_csv(manifests_dir / "gap_fill_report.csv", index=False)

    pd.DataFrame(basin_summaries).to_csv(manifests_dir / "per_basin_summary.csv", index=False)
    log.info("Wrote 4 manifest files to manifests/")


def _write_provenance(
    out_dir: Path,
    args: argparse.Namespace,
    staids: list[str],
    elapsed_s: float,
) -> None:
    prov = {
        "script":       SCRIPT_NAME,
        "created_utc":  CREATED_UTC,
        "elapsed_s":    round(elapsed_s, 1),
        "platform":     platform.node(),
        "python":       sys.version.split()[0],
        "forcing_dir":  str(args.forcing_dir),
        "target_dir":   str(args.target_dir),
        "out_dir":      str(args.out_dir),
        "attributes_src": str(args.attributes_csv),
        "staids":       staids,
        "n_basins":     len(staids),
        "period_start": str(_PERIOD_START),
        "period_end":   str(_PERIOD_END),
        "n_hours":      _FULL_PERIOD_HOURS,
    }
    with open(out_dir / "run_provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)


def _write_readme(out_dir: Path, staids: list[str]) -> None:
    body = textwrap.dedent(f"""\
        # Flash-NH Stage 1 — NeuralHydrology Pilot Package v001

        Built: {CREATED_UTC}
        Basins: {len(staids)}  ({', '.join(staids)})
        Period: 2020-10-14T00Z – 2025-12-31T23Z ({_FULL_PERIOD_HOURS:,} hourly steps/basin)
        Milestone: 2K-G-B

        ## Schema (14 variables per NC)

        Dynamic forcing (11):
          mrms_qpe_1h_mm        MRMS QPE 1-h accumulation [mm]
          rtma_2t_K             2-metre temperature [K]
          rtma_2d_K             2-metre dewpoint [K]
          rtma_2sh_kgkg         2-metre specific humidity [kg/kg]
          rtma_sp_Pa            Surface pressure [Pa]  (in NC; excluded from Smoke 1 dynamic_inputs)
          rtma_10u_ms           10-m U wind [m/s]
          rtma_10v_ms           10-m V wind [m/s]
          rtma_tcc_pct          Total cloud cover [%]
          rtma_vis_m            Visibility [m]
          rtma_gust_ms          Wind gust [m/s]
          rtma_ceil_m           Cloud ceiling [m]

        Gap flags (2):
          mrms_qpe_1h_mm_gap    float32 1.0 where MRMS archive gap (136 h/basin)
          rtma_gap              float32 1.0 where RTMA gap (2 h/basin)

        Target (1):
          qobs_m3s              Streamflow [m3/s] — NaN where missing; loss-masked by NH

        FORBIDDEN: rtma_weasd_kgm2 is NOT in this package.

        ## Gap-fill policy — PILOT ONLY (Smoke 0/1 technical policy)

        WARNING: Do NOT carry this policy unchanged into scientific baseline training.
        - MRMS gaps (136 h/basin, 0.30%): mrms_qpe_1h_mm set to 0.0 mm.
          For final training: evaluate window-exclusion of MRMS gap hours.
        - RTMA gaps (2 h/basin, 0.004%): all RTMA vars linearly interpolated.
        - qobs_m3s: NaN preserved exactly (NH loss-masks missing targets).

        ## Train/Val/Test splits (technical — not final scientific evaluation design)

        Train: 2020-10-14 – 2022-12-31 (~26 months)
        Val:   2023-01-01 – 2023-12-31 (12 months)
        Test:  2024-01-01 – 2025-12-31 (24 months)
        All 5 basins in all splits (temporal generalization, not spatial).

        ## Usage

        Smoke 0 — via Moriah Slurm (preferred):
          sbatch scripts/run_stage1_smoke0_moriah.sbatch

        Smoke 0 — direct nh-run invocation:
          nh-run train --config-file \\
            /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/configs/stage1_smoke0_nh.yml

        Smoke 1:
          nh-run train --config-file \\
            /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/configs/stage1_smoke1_nh.yml

        Audit (on h2o before transfer):
          python scripts/audit_stage1_nh_package.py \\
            --package-dir <this_dir> --expected-basins 5 --expected-rows 45720

        Preflight (on Moriah after transfer):
          python scripts/check_stage1_nh_preflight.py \\
            --package-dir /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001 --smoke 0

        ## Files

          time_series/          Per-basin NCs (GenericDataset format)
          attributes/           Static attributes directory (NH GenericDataset canonical path)
            attributes.csv      gauge_id + all available attrs
          basins/               Basin lists: smoke0/1 × train/val/test
          configs/              NH YAML configs: stage1_smoke0_nh.yml, stage1_smoke1_nh.yml
          manifests/            dataset_manifest.json, variable_schema.csv,
                                gap_fill_report.csv, per_basin_summary.csv
          run_provenance.json   Build provenance
    """)
    (out_dir / "README.md").write_text(body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = _time.time()
    args = _parse_args()

    forcing_dir = Path(args.forcing_dir)
    target_dir  = Path(args.target_dir)
    out_dir     = Path(args.out_dir)
    attr_path   = Path(args.attributes_csv)

    # ---- Input validation ----
    for label, p in [("--forcing-dir", forcing_dir), ("--target-dir", target_dir),
                     ("--attributes-csv", attr_path)]:
        if not p.exists():
            log.error("%s not found: %s", label, p)
            sys.exit(1)

    if out_dir.exists() and not args.force:
        log.error("Output dir exists: %s  (pass --force to overwrite)", out_dir)
        sys.exit(1)

    # ---- Determine STAIDs ----
    if args.staids:
        staids = [_norm_staid(s.strip()) for s in args.staids.split(",") if s.strip()]
    else:
        parquet_files = sorted((forcing_dir / "time_series").glob("*.parquet"))
        if not parquet_files:
            log.error("No *.parquet files in %s/time_series/", forcing_dir)
            sys.exit(1)
        staids = [_norm_staid(p.stem) for p in parquet_files]

    if args.expected_basins is not None and len(staids) != args.expected_basins:
        log.error("Expected %d basins, got %d: %s",
                  args.expected_basins, len(staids), staids)
        sys.exit(1)

    log.info("Building NH package: %d basins %s", len(staids), staids)

    # ---- Load attributes ----
    attrs = _load_attributes(attr_path, staids)
    log.info("Loaded attributes for %d basins (%d cols)", len(attrs), len(attrs.columns))

    # ---- Full period grid ----
    full_grid = _full_period_grid()
    log.info("Period: %s → %s (%d h)", full_grid[0], full_grid[-1], len(full_grid))

    if args.dry_run:
        log.info("[DRY-RUN] Validated. Would build %d basins → %s", len(staids), out_dir)
        log.info("[DRY-RUN] No output written.")
        return

    # ---- Build output layout ----
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_dir        = out_dir / "time_series"
    attrs_dir     = out_dir / "attributes"
    basins_dir    = out_dir / "basins"
    configs_dir   = out_dir / "configs"
    manifests_dir = out_dir / "manifests"
    ts_dir.mkdir(parents=True, exist_ok=True)
    attrs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-basin build ----
    basin_summaries:  list[dict] = []
    gap_fill_reports: list[dict] = []

    for staid in staids:
        log.info("[%s] processing ...", staid)
        forcing_path = forcing_dir / "time_series" / f"{staid}.parquet"
        target_path  = target_dir  / "time_series" / f"{staid}.nc"
        nc_path      = ts_dir / f"{staid}.nc"

        for label, p in [(f"forcing {staid}", forcing_path), (f"target {staid}", target_path)]:
            if not p.exists():
                log.error("Not found (%s): %s", label, p)
                sys.exit(1)

        wide    = _load_forcing_parquet(forcing_path, full_grid)
        qobs    = _load_target_qobs(target_path, full_grid)
        wide, gap_rpt = _apply_gap_fill(wide)

        gap_rpt["gauge_id"] = staid
        gap_fill_reports.append(gap_rpt)

        _write_basin_nc(staid, wide, qobs, nc_path)
        sha = _sha256(nc_path)

        n_mrms_gap = int(wide["mrms_qpe_1h_mm_gap"].sum())
        n_rtma_gap = int(wide["rtma_gap"].sum())
        n_qobs_nan = int(qobs.isna().sum())

        basin_summaries.append({
            "gauge_id":      staid,
            "n_hours":       _FULL_PERIOD_HOURS,
            "n_mrms_gap":    n_mrms_gap,
            "n_rtma_gap":    n_rtma_gap,
            "n_qobs_nan":    n_qobs_nan,
            "qobs_coverage": round(1.0 - n_qobs_nan / _FULL_PERIOD_HOURS, 6),
            "nc_sha256":     sha,
        })
        log.info("[%s] DONE  mrms_gap=%d  rtma_gap=%d  qobs_nan=%d  sha=%s...",
                 staid, n_mrms_gap, n_rtma_gap, n_qobs_nan, sha[:12])

    # ---- Package-level outputs ----
    # attributes/ subdir is the canonical NH GenericDataset path (data_dir/attributes/*.csv)
    _write_attributes(attrs, attrs_dir / "attributes.csv")
    _write_basin_lists(basins_dir, staids)
    _write_configs(configs_dir)
    _write_manifests(manifests_dir, basin_summaries, gap_fill_reports)
    _write_provenance(out_dir, args, staids, _time.time() - t0)
    _write_readme(out_dir, staids)

    elapsed = _time.time() - t0
    log.info("=" * 60)
    log.info("DONE: %d basins in %.1f s → %s", len(staids), elapsed, out_dir)
    log.info("Audit: python scripts/audit_stage1_nh_package.py "
             "--package-dir %s --expected-basins %d --expected-rows %d",
             out_dir, len(staids), _FULL_PERIOD_HOURS)


if __name__ == "__main__":
    main()
