#!/usr/bin/env python3
"""Lightweight NH GenericDataset preflight diagnostic.

Validates the generated pilot package structure and data without launching a
full training run.  Designed to run on Moriah after package transfer, before
submitting the Smoke 0 Slurm job.

If NeuralHydrology is not installed (e.g. local PC), structural and data
checks still run; NH-level checks (attribute loader, dataset registry) are
skipped with an explicit note.

Usage (on Moriah after transfer):
  python scripts/check_stage1_nh_preflight.py \\
    --package-dir /sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001 \\
    --smoke 0

Usage (local, no NH installed):
  python scripts/check_stage1_nh_preflight.py \\
    --package-dir /path/to/stage1_nh_pilot_v001 --smoke 0
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# ---------------------------------------------------------------------------
# NH import guard
# ---------------------------------------------------------------------------

try:
    from neuralhydrology.utils.config import Config as NHConfig  # type: ignore
    _NH_AVAILABLE = True
except ImportError:
    _NH_AVAILABLE = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--package-dir", required=True,
                   help="NH package root (e.g. stage1_pilot_v001/)")
    p.add_argument("--smoke", type=int, default=0, choices=[0, 1],
                   help="Smoke level to check config for (0 or 1; default 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_issues: list[str] = []
_oks:    list[str] = []


def _ok(msg: str) -> None:
    _oks.append(msg)
    print(f"  OK    {msg}")


def _fail(msg: str) -> None:
    _issues.append(msg)
    print(f"  FAIL  {msg}")


# ---------------------------------------------------------------------------
# Config checks (no NH required — uses PyYAML only)
# ---------------------------------------------------------------------------

_DDMMYYYY = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_BANNED_KEYS = ("num_epochs", "shuffle", "log_n_basins")
_SMOKE0_DYNAMIC = ["mrms_qpe_1h_mm", "mrms_qpe_1h_mm_gap"]
_REQUIRED_STATIC = ["DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "BFI_AVE"]


def _check_config(pkg: Path, smoke: int) -> dict:
    cfg_path = pkg / "configs" / f"stage1_smoke{smoke}_nh.yml"
    if not cfg_path.exists():
        _fail(f"config not found: {cfg_path}")
        return {}

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    _ok(f"config loaded: {cfg_path.name}")

    # NH 1.13 compatibility checks
    if cfg.get("dataset") == "generic":
        _ok("dataset: generic")
    else:
        _fail(f"dataset must be 'generic', got: {cfg.get('dataset')!r}")

    for k in sorted(k for k in cfg if k.endswith("_date")):
        v = str(cfg[k])
        if _DDMMYYYY.match(v):
            _ok(f"{k}: {v} (DD/MM/YYYY)")
        else:
            _fail(f"{k}: {v!r} is not DD/MM/YYYY")

    if "epochs" in cfg:
        _ok(f"epochs: {cfg['epochs']}")
    else:
        _fail("'epochs' key missing (NH 1.13 requires 'epochs', not 'num_epochs')")
    for bad in _BANNED_KEYS:
        if bad in cfg:
            _fail(f"'{bad}' present (NH 1.13 rejects it)")
        else:
            _ok(f"'{bad}' absent")

    if cfg.get("head") == "regression":
        _ok("head: regression")
    else:
        _fail(f"head must be 'regression', got: {cfg.get('head')!r}")
    if cfg.get("output_activation") == "linear":
        _ok("output_activation: linear")
    else:
        _fail(f"output_activation must be 'linear', got: {cfg.get('output_activation')!r}")

    if smoke == 0:
        di = list(cfg.get("dynamic_inputs", []))
        if di == _SMOKE0_DYNAMIC:
            _ok(f"dynamic_inputs: {di}")
        else:
            _fail(f"dynamic_inputs: expected {_SMOKE0_DYNAMIC}, got {di}")

    tv = list(cfg.get("target_variables", []))
    if tv == ["qobs_m3s"]:
        _ok("target_variables: [qobs_m3s]")
    else:
        _fail(f"target_variables: expected ['qobs_m3s'], got {tv}")

    sa = list(cfg.get("static_attributes", []))
    if sa == _REQUIRED_STATIC:
        _ok(f"static_attributes: {sa}")
    else:
        _fail(f"static_attributes: expected {_REQUIRED_STATIC}, got {sa}")

    return cfg


# ---------------------------------------------------------------------------
# Attribute checks (no NH required)
# ---------------------------------------------------------------------------


def _check_attributes(pkg: Path) -> list[str]:
    attr_path = pkg / "attributes" / "attributes.csv"
    if not attr_path.exists():
        _fail(f"attributes/attributes.csv missing at {attr_path}")
        return []

    df = pd.read_csv(attr_path, index_col="gauge_id", dtype=str)
    _ok(f"attributes/attributes.csv: shape {df.shape}")

    missing = [c for c in _REQUIRED_STATIC if c not in df.columns]
    if missing:
        _fail(f"missing attribute columns: {missing}")
    else:
        _ok(f"required attribute columns present: {_REQUIRED_STATIC}")

    return list(df.index)


# ---------------------------------------------------------------------------
# Time-series checks (no NH required — uses xarray)
# ---------------------------------------------------------------------------


def _check_timeseries(pkg: Path, basins: list[str], dynamic_inputs: list[str]) -> None:
    ts_dir = pkg / "time_series"
    if not ts_dir.exists():
        _fail("time_series/ directory missing")
        return

    for staid in basins:
        nc_path = ts_dir / f"{staid}.nc"
        if not nc_path.exists():
            _fail(f"[{staid}] time_series/{staid}.nc not found")
            continue

        try:
            ds = xr.open_dataset(nc_path)
        except Exception as exc:
            _fail(f"[{staid}] cannot open NC: {exc}")
            continue

        try:
            if "date" not in ds.coords:
                _fail(f"[{staid}] 'date' coordinate missing")
                continue
            _ok(f"[{staid}] opened; dims={dict(ds.dims)}")

            for var in dynamic_inputs:
                if var not in ds.data_vars:
                    _fail(f"[{staid}] missing dynamic input: {var}")
                else:
                    n_nan = int(np.isnan(ds[var].values).sum())
                    if n_nan > 0:
                        _fail(f"[{staid}] {var}: {n_nan} NaN (dynamic inputs must be pre-filled)")
                    else:
                        _ok(f"[{staid}] {var}: non-null")

            if "qobs_m3s" in ds.data_vars:
                n_nan = int(np.isnan(ds["qobs_m3s"].values).sum())
                n_tot = ds.dims.get("date", 0)
                _ok(f"[{staid}] qobs_m3s: {n_nan}/{n_tot} NaN (expected; NH loss-masks these)")
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# NH-level checks (requires neuralhydrology installed — Moriah only)
# ---------------------------------------------------------------------------


def _check_nh_level(pkg: Path, smoke: int) -> None:
    print("\n--- NH-level checks (requires neuralhydrology) ---")
    cfg_path = pkg / "configs" / f"stage1_smoke{smoke}_nh.yml"

    try:
        from neuralhydrology.utils.config import Config  # type: ignore
        cfg = Config(cfg_path)
        _ok(f"Config({cfg_path.name}) constructed without error")
    except Exception as exc:
        _fail(f"NHConfig construction failed: {exc}")
        return

    # Verify dataset registry resolves 'generic'
    try:
        from neuralhydrology.datasetzoo import get_dataset  # type: ignore
        _ok("datasetzoo.get_dataset importable (registry accessible)")
    except Exception as exc:
        _fail(f"datasetzoo.get_dataset import failed: {exc}")
        return

    # Load attributes via NH's attribute loader (safe — does not need train_dir; avoids
    # the cfg.train_dir is None trap that occurs when calling get_dataset standalone).
    # Import path is for NH 1.13 on Moriah (job 45365952).  If ImportError, check whether
    # load_attributes moved (e.g. neuralhydrology.utils.attribute_utils in other NH versions).
    try:
        from neuralhydrology.datasetzoo.genericdataset import load_attributes  # type: ignore
        basin_file = pkg / "basins" / f"smoke{smoke}_train.txt"
        basins = [b.strip() for b in basin_file.read_text().splitlines() if b.strip()]
        df_attr = load_attributes(
            data_dir=pkg,
            attribute_names=cfg.static_attributes,
            basins=basins,
        )
        _ok(f"load_attributes: shape={df_attr.shape}, index[:3]={list(df_attr.index[:3])}")
    except ImportError as exc:
        _fail(f"load_attributes ImportError — verify import path for NH 1.13: {exc}")
    except Exception as exc:
        _fail(f"load_attributes failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    pkg  = Path(args.package_dir)

    print("=" * 65)
    print(f"Flash-NH Stage 1 — NH Preflight (Smoke {args.smoke})")
    print(f"Package:      {pkg}")
    print(f"NH installed: {_NH_AVAILABLE}")
    print("=" * 65)

    if not pkg.exists():
        _fail(f"package dir not found: {pkg}")
        sys.exit(1)

    print("\n--- Config checks ---")
    cfg = _check_config(pkg, args.smoke)

    print("\n--- Attribute checks ---")
    basins = _check_attributes(pkg)

    print("\n--- Time-series checks ---")
    dynamic_inputs = list(cfg.get("dynamic_inputs", [])) if cfg else []
    _check_timeseries(pkg, basins, dynamic_inputs)

    if _NH_AVAILABLE:
        _check_nh_level(pkg, args.smoke)
    else:
        print("\n--- NH-level checks SKIPPED (neuralhydrology not installed) ---")
        print("  Run on Moriah to verify NH Config construction and attribute loader.")

    n_ok   = len(_oks)
    n_fail = len(_issues)
    print("\n" + "=" * 65)
    print(f"RESULT: {'PASS' if n_fail == 0 else 'FAIL'}")
    print(f"OK: {n_ok}  |  FAIL: {n_fail}")
    if _issues:
        print("\nFailed checks:")
        for msg in _issues:
            print(f"  - {msg}")
    print("=" * 65)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
