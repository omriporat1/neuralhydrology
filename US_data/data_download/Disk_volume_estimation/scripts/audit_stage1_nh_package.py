#!/usr/bin/env python3
"""Audit Stage 1 NeuralHydrology pilot package produced by build_stage1_nh_package.py.

Checks:
  1.  Required directories and files exist.
  2.  Exactly --expected-basins NC files in time_series/.
  3.  Each NC: 'date' coordinate, --expected-rows rows, monotonic hourly.
  4.  Required variables present in each NC.
  5.  Forbidden variables absent (rtma_weasd_kgm2, rtma_10si_ms, rtma_i10fg_ms).
  6.  qobs_m3s: exists, has 'units' attr, NaN allowed (not all-NaN).
  7.  Forcing data variables: no NaN after gap-fill (mrms_qpe_1h_mm, all rtma_*).
  8.  mrms_qpe_1h_mm_gap: sum == 136 per basin.
  9.  rtma_gap: sum == 2 per basin.
  10. mrms_qpe_1h_mm non-null == expected_rows (confirms MRMS fill applied).
  11. rtma_2d_K non-null == expected_rows (confirms dewpoint mapping fix propagated).
  12. attributes/attributes.csv: has 'gauge_id' index, required cols, expected STAIDs.
  13. Basin list files: smoke0/1 × train/val/test exist, contain valid 8-char STAIDs.
  14. Smoke configs exist.
  15. Manifests directory and dataset_manifest.json exist and are parseable.
  16. Config NH 1.13 compatibility: dataset=generic, DD/MM/YYYY dates, epochs key,
      no num_epochs/shuffle/log_n_basins, head=regression, output_activation=linear,
      correct dynamic_inputs and target_variables for Smoke 0.
  17. Writes audit_summary.md to package root.

Exit code: 0 = PASS, 1 = FAIL.

Usage:
  python scripts/audit_stage1_nh_package.py \\
    --package-dir /data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/ \\
    --expected-basins 5 \\
    --expected-rows 45720
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

CREATED_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
t0 = _time.time()

# ---------------------------------------------------------------------------
# Constants (must match builder)
# ---------------------------------------------------------------------------

_FORCING_DATA_COLS: list[str] = [
    "mrms_qpe_1h_mm",
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct",
    "rtma_vis_m", "rtma_gust_ms", "rtma_ceil_m",
]
_GAP_FLAG_COLS: list[str] = ["mrms_qpe_1h_mm_gap", "rtma_gap"]
_ALL_DYNAMIC_COLS: list[str] = _FORCING_DATA_COLS + _GAP_FLAG_COLS
_TARGET_COL = "qobs_m3s"
_ALL_REQUIRED_VARS: list[str] = _ALL_DYNAMIC_COLS + [_TARGET_COL]

_FORBIDDEN_VARS: set[str] = {
    "rtma_weasd_kgm2",
    "rtma_10si_ms",
    "rtma_i10fg_ms",
}

_REQUIRED_ATTR_COLS: list[str] = ["DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "BFI_AVE"]

# Expected gap counts per basin (corrected v001 schema)
_EXPECTED_MRMS_GAP = 136
_EXPECTED_RTMA_GAP = 2

# Plausible physical range checks {var: (lo, hi)}  — WARN only, not ERROR
_RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "mrms_qpe_1h_mm":  (0.0,     500.0),
    "rtma_2t_K":       (200.0,   330.0),
    "rtma_2d_K":       (180.0,   330.0),
    "rtma_2sh_kgkg":   (0.0,     0.04),
    "rtma_sp_Pa":      (50000.,  110000.),
    "rtma_10u_ms":     (-60.,    60.),
    "rtma_10v_ms":     (-60.,    60.),
    "rtma_tcc_pct":    (0.0,     100.0),
    "rtma_vis_m":      (0.0,     100000.),
    "rtma_gust_ms":    (0.0,     80.0),
    "rtma_ceil_m":     (0.0,     25000.),
    "qobs_m3s":        (0.0,     2_000_000.),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--package-dir", required=True,
                   help="Path to the NH pilot package root.")
    p.add_argument("--expected-basins", type=int, default=5,
                   help="Expected number of per-basin NC files (default: 5).")
    p.add_argument("--expected-rows", type=int, default=45_720,
                   help="Expected rows per NC (default: 45720 = full period).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

_issues:   list[dict] = []
_warns:    list[dict] = []
_oks:      list[str]  = []


def err(msg: str, basin: str = "") -> bool:
    tag = f"[{basin}] " if basin else ""
    _issues.append({"basin": basin, "severity": "ERROR", "message": msg})
    print(f"  ERROR: {tag}{msg}")
    return False


def warn(msg: str, basin: str = "") -> None:
    tag = f"[{basin}] " if basin else ""
    _warns.append({"basin": basin, "severity": "WARN", "message": msg})
    print(f"  WARN:  {tag}{msg}")


def ok(msg: str) -> bool:
    _oks.append(msg)
    print(f"  OK:    {msg}")
    return True


def chk(label: str, passed: bool, detail: str = "", basin: str = "") -> bool:
    if passed:
        return ok(f"{label}" + (f"  [{detail}]" if detail else ""))
    return err(f"{label}" + (f"  [{detail}]" if detail else ""), basin=basin)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_structure(pkg: Path, expected_basins: int) -> bool:
    print("\n[1] Package structure ...")
    result = True

    required_dirs = [
        pkg / "time_series",
        pkg / "attributes",
        pkg / "basins",
        pkg / "configs",
        pkg / "manifests",
    ]
    for d in required_dirs:
        result &= chk(f"dir {d.name}/ exists", d.exists())

    required_files = [
        pkg / "attributes" / "attributes.csv",
        pkg / "run_provenance.json",
        pkg / "README.md",
        pkg / "manifests" / "dataset_manifest.json",
        pkg / "manifests" / "variable_schema.csv",
        pkg / "manifests" / "gap_fill_report.csv",
        pkg / "manifests" / "per_basin_summary.csv",
        pkg / "configs" / "stage1_smoke0_nh.yml",
        pkg / "configs" / "stage1_smoke1_nh.yml",
    ]
    for fp in required_files:
        result &= chk(f"file {fp.relative_to(pkg)} exists", fp.exists())

    nc_files = sorted((pkg / "time_series").glob("*.nc")) if (pkg / "time_series").exists() else []
    result &= chk(
        f"NC count == {expected_basins}",
        len(nc_files) == expected_basins,
        f"got {len(nc_files)}",
    )
    return result


def check_basin_nc(nc_path: Path, expected_rows: int) -> bool:
    """Audit one per-basin NC file. Returns True if all checks pass."""
    staid = nc_path.stem
    result = True

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:
        return err(f"cannot open NC: {exc}", basin=staid)

    try:
        # ---- date coordinate ----
        has_date = "date" in ds.coords
        result &= chk("'date' coordinate exists", has_date, basin=staid)
        if not has_date:
            ds.close()
            return result

        dates = pd.DatetimeIndex(ds.coords["date"].values)
        if dates.tz is not None:
            dates = dates.tz_localize(None)

        result &= chk(f"row count == {expected_rows}", len(dates) == expected_rows,
                      f"got {len(dates)}", basin=staid)
        result &= chk("dates monotonic increasing", dates.is_monotonic_increasing,
                      basin=staid)

        if len(dates) >= 2:
            diffs = pd.Series(dates).diff().iloc[1:]
            all_hourly = (diffs == pd.Timedelta(hours=1)).all()
            result &= chk("dates hourly (1h spacing)", bool(all_hourly), basin=staid)

        # ---- required variables ----
        for var in _ALL_REQUIRED_VARS:
            result &= chk(f"var '{var}' present", var in ds.data_vars, basin=staid)

        # ---- forbidden variables ----
        for var in _FORBIDDEN_VARS:
            result &= chk(f"forbidden '{var}' absent", var not in ds.data_vars, basin=staid)

        if len(dates) != expected_rows:
            ds.close()
            return result  # row-count failure makes further checks unreliable

        # ---- qobs_m3s ----
        if "qobs_m3s" in ds.data_vars:
            has_units = "units" in ds["qobs_m3s"].attrs
            result &= chk("qobs_m3s has 'units' attr", has_units, basin=staid)
            qobs_arr = ds["qobs_m3s"].values
            all_nan = bool(np.all(np.isnan(qobs_arr)))
            result &= chk("qobs_m3s not all-NaN", not all_nan, basin=staid)
            n_nan = int(np.isnan(qobs_arr).sum())
            if n_nan > 0:
                warn(f"qobs_m3s has {n_nan}/{expected_rows} NaN (expected for missing discharge)",
                     basin=staid)

        # ---- forcing: no NaN after gap-fill ----
        for col in _FORCING_DATA_COLS:
            if col not in ds.data_vars:
                continue
            arr = ds[col].values
            n_nan = int(np.isnan(arr).sum())
            result &= chk(
                f"{col} non-null == {expected_rows}",
                n_nan == 0,
                f"got {n_nan} NaN" if n_nan > 0 else "",
                basin=staid,
            )

        # ---- gap flag counts ----
        if "mrms_qpe_1h_mm_gap" in ds.data_vars:
            n_mrms_gap = int(np.nansum(ds["mrms_qpe_1h_mm_gap"].values))
            result &= chk(
                f"mrms_qpe_1h_mm_gap sum == {_EXPECTED_MRMS_GAP}",
                n_mrms_gap == _EXPECTED_MRMS_GAP,
                f"got {n_mrms_gap}",
                basin=staid,
            )

        if "rtma_gap" in ds.data_vars:
            n_rtma_gap = int(np.nansum(ds["rtma_gap"].values))
            result &= chk(
                f"rtma_gap sum == {_EXPECTED_RTMA_GAP}",
                n_rtma_gap == _EXPECTED_RTMA_GAP,
                f"got {n_rtma_gap}",
                basin=staid,
            )

        # ---- rtma_2d_K special check (confirms dewpoint mapping fix) ----
        if "rtma_2d_K" in ds.data_vars:
            n_nn = int(np.sum(~np.isnan(ds["rtma_2d_K"].values)))
            result &= chk(
                f"rtma_2d_K non-null == {expected_rows} (dewpoint fix)",
                n_nn == expected_rows,
                f"got {n_nn}",
                basin=staid,
            )

        # ---- physical range checks (WARN only) ----
        for col, (lo, hi) in _RANGE_CHECKS.items():
            if col not in ds.data_vars:
                continue
            arr = ds[col].values
            arr_nn = arr[~np.isnan(arr)]
            if len(arr_nn) == 0:
                continue
            vmin, vmax = float(arr_nn.min()), float(arr_nn.max())
            if vmin < lo or vmax > hi:
                warn(f"{col} out of expected range [{lo}, {hi}]: "
                     f"min={vmin:.3g}, max={vmax:.3g}", basin=staid)

    finally:
        ds.close()

    return result


def check_attributes(pkg: Path, staids: list[str]) -> bool:
    print("\n[3] Attributes ...")
    # NH GenericDataset canonical path: data_dir/attributes/*.csv
    attr_path = pkg / "attributes" / "attributes.csv"
    if not attr_path.exists():
        return err("attributes/attributes.csv missing (NH expects data_dir/attributes/*.csv)")

    try:
        df = pd.read_csv(attr_path, index_col="gauge_id", dtype=str)
    except Exception as exc:
        return err(f"attributes/attributes.csv unreadable: {exc}")

    result = True
    result &= chk(f"attributes/attributes.csv row count == {len(staids)}",
                  len(df) == len(staids), f"got {len(df)}")

    for col in _REQUIRED_ATTR_COLS:
        result &= chk(f"required col '{col}' present", col in df.columns)

    missing_basins = [s for s in staids if s not in df.index]
    result &= chk("all expected STAIDs in attributes/attributes.csv",
                  not missing_basins,
                  f"missing: {missing_basins}" if missing_basins else "")
    return result


def check_basin_lists(pkg: Path, staids: list[str]) -> bool:
    print("\n[4] Basin list files ...")
    basins_dir = pkg / "basins"
    result = True

    for smoke in ("smoke0", "smoke1"):
        for split in ("train", "val", "test"):
            fp = basins_dir / f"{smoke}_{split}.txt"
            if not fp.exists():
                result &= err(f"{fp.name} missing")
                continue
            lines = [ln.strip() for ln in fp.read_text().splitlines() if ln.strip()]
            # All lines should be valid 8-char STAIDs
            invalid = [ln for ln in lines if len(ln) != 8 or not ln.isdigit()]
            result &= chk(
                f"{fp.name}: {len(lines)} valid 8-char STAIDs",
                len(lines) == len(staids) and not invalid,
                f"got {len(lines)} lines, invalid={invalid}" if invalid or len(lines) != len(staids) else "",
            )
    return result


_DDMMYYYY = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_SMOKE0_EXPECTED_DYNAMIC = ["mrms_qpe_1h_mm", "mrms_qpe_1h_mm_gap"]
_EXPECTED_STATIC_ATTRS   = ["DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "BFI_AVE"]
_BANNED_KEYS             = ("num_epochs", "shuffle", "log_n_basins")


def check_configs(pkg: Path) -> bool:
    """Validate NH 1.13 compatibility of generated YAML configs (Smoke 0 config)."""
    print("\n[6] Config NH 1.13 compatibility ...")
    result = True

    cfg_path = pkg / "configs" / "stage1_smoke0_nh.yml"
    if not cfg_path.exists():
        return err("configs/stage1_smoke0_nh.yml missing; skipping config checks")

    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        return err(f"configs/stage1_smoke0_nh.yml unreadable: {exc}")

    # Dataset registry key
    result &= chk("dataset == 'generic' (NH 1.13 registry key)",
                  cfg.get("dataset") == "generic",
                  f"got: {cfg.get('dataset')!r}")

    # All _date fields must be DD/MM/YYYY
    date_keys = sorted(k for k in cfg if k.endswith("_date"))
    for k in date_keys:
        v = str(cfg[k])
        result &= chk(f"{k} is DD/MM/YYYY",
                      bool(_DDMMYYYY.match(v)), f"got: {v!r}")

    # epochs key present; legacy keys absent
    result &= chk("'epochs' key present", "epochs" in cfg,
                  f"keys: {list(cfg)}")
    for bad in _BANNED_KEYS:
        result &= chk(f"'{bad}' absent (NH 1.13 rejects it)", bad not in cfg)

    # head / output
    result &= chk("head == 'regression'",
                  cfg.get("head") == "regression",
                  f"got: {cfg.get('head')!r}")
    result &= chk("output_activation == 'linear'",
                  cfg.get("output_activation") == "linear",
                  f"got: {cfg.get('output_activation')!r}")

    # Smoke 0 specific: dynamic inputs and target
    di = list(cfg.get("dynamic_inputs", []))
    result &= chk(f"dynamic_inputs == {_SMOKE0_EXPECTED_DYNAMIC}",
                  di == _SMOKE0_EXPECTED_DYNAMIC, f"got: {di}")

    tv = list(cfg.get("target_variables", []))
    result &= chk("target_variables == ['qobs_m3s']",
                  tv == ["qobs_m3s"], f"got: {tv}")

    sa = list(cfg.get("static_attributes", []))
    result &= chk(f"static_attributes == {_EXPECTED_STATIC_ATTRS}",
                  sa == _EXPECTED_STATIC_ATTRS, f"got: {sa}")

    return result


def check_manifests(pkg: Path, expected_basins: int) -> bool:
    print("\n[5] Manifests ...")
    result = True
    mfst_path = pkg / "manifests" / "dataset_manifest.json"
    if not mfst_path.exists():
        return err("manifests/dataset_manifest.json missing")

    try:
        with open(mfst_path) as f:
            mfst = json.load(f)
        result &= chk("dataset_manifest.json parseable", True)
    except Exception as exc:
        return err(f"dataset_manifest.json unreadable: {exc}")

    result &= chk(
        f"manifest n_basins == {expected_basins}",
        mfst.get("n_basins") == expected_basins,
        f"got {mfst.get('n_basins')}",
    )
    result &= chk(
        "manifest n_hours == 45720",
        mfst.get("n_hours") == 45_720,
        f"got {mfst.get('n_hours')}",
    )
    return result


# ---------------------------------------------------------------------------
# Audit summary writer
# ---------------------------------------------------------------------------

def _write_audit_summary(pkg: Path, overall_pass: bool) -> None:
    lines = [
        f"# Flash-NH NH Package Audit — {'PASS' if overall_pass else 'FAIL'}",
        "",
        f"Audit time: {CREATED_UTC}",
        f"Package:    {pkg}",
        f"Result:     {'PASS' if overall_pass else 'FAIL'}",
        f"Errors:     {len(_issues)}",
        f"Warnings:   {len(_warns)}",
        "",
    ]
    if _issues:
        lines.append("## Errors")
        for e in _issues:
            prefix = f"[{e['basin']}] " if e["basin"] else ""
            lines.append(f"- {prefix}{e['message']}")
        lines.append("")
    if _warns:
        lines.append("## Warnings")
        for w in _warns:
            prefix = f"[{w['basin']}] " if w["basin"] else ""
            lines.append(f"- {prefix}{w['message']}")
        lines.append("")

    summary_path = pkg / "audit_summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"\n  audit_summary.md written → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    pkg  = Path(args.package_dir)

    print("=" * 70)
    print("Flash-NH Stage 1 — NeuralHydrology Package Auditor (2K-G-B)")
    print(f"Package:  {pkg}")
    print(f"Time:     {CREATED_UTC}")
    print("=" * 70)

    if not pkg.exists():
        err(f"Package dir not found: {pkg}")
        print("\nFAIL (0 checks run)")
        sys.exit(1)

    # [1] Structure
    struct_ok = check_structure(pkg, args.expected_basins)

    # Infer STAID list from NC files
    nc_files = sorted((pkg / "time_series").glob("*.nc")) if (pkg / "time_series").exists() else []
    staids = [nc.stem for nc in nc_files]

    # [2] Per-basin NCs
    print(f"\n[2] Per-basin NCs ({len(nc_files)} files) ...")
    nc_ok = True
    for nc_path in nc_files:
        print(f"\n  --- {nc_path.stem} ---")
        nc_ok &= check_basin_nc(nc_path, args.expected_rows)

    # [3] Attributes
    attr_ok = check_attributes(pkg, staids) if staids else err("no NC files; skipping attribute check")

    # [4] Basin lists
    basin_ok = check_basin_lists(pkg, staids) if staids else err("no NC files; skipping basin list check")

    # [5] Manifests
    mfst_ok = check_manifests(pkg, args.expected_basins)

    # [6] Config NH 1.13 compatibility
    cfg_ok = check_configs(pkg)

    overall_pass = all([struct_ok, nc_ok, attr_ok, basin_ok, mfst_ok, cfg_ok])

    elapsed = _time.time() - t0
    print("\n" + "=" * 70)
    print(f"RESULT: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Errors: {len(_issues)}  |  Warnings: {len(_warns)}  |  OK: {len(_oks)}")
    print(f"Elapsed: {elapsed:.1f} s")
    print("=" * 70)

    _write_audit_summary(pkg, overall_pass)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
