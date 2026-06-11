"""
Flash-NH Stage 1 Milestone 2I-B — Audit of Full-Period USGS IV Recovered Targets
==================================================================================

Reads canonical hourly NC files produced by recover_usgs_iv_full_period_hourly.py
and produces:

  audit/per_basin_coverage.csv
  audit/per_water_year_coverage.csv
  audit/per_month_coverage.csv
  audit/gap_audit.csv
  audit/quality_audit.csv
  audit/jan2023_comparison_against_2hc.csv
  audit/target_status.csv
  qc/jan2023_fullperiod_vs_2hc_comparison.png
  summary.md
  summary.json
  provenance/run_provenance.json

Hard guardrails
---------------
  - Read-only access to canonical NC files.
  - Do not modify 2G/2H packages.
  - All outputs under --out-dir.
  - 03298135 late-2025 gap must be flagged if last obs > 14 days before period end.

Usage
-----
  python scripts/audit_usgs_iv_recovered_targets.py \\
      --canonical-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot/canonical \\
      --out-dir tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_NAME = pathlib.Path(__file__).name

PERIOD_START = pd.Timestamp("2020-10-14T00:00:00")
PERIOD_END   = pd.Timestamp("2025-12-31T23:00:00")

# Water-year boundaries (UTC naive)
WY_BOUNDARIES = [
    ("WY2021", pd.Timestamp("2020-10-14"), pd.Timestamp("2021-09-30 23:00:00")),
    ("WY2022", pd.Timestamp("2021-10-01"), pd.Timestamp("2022-09-30 23:00:00")),
    ("WY2023", pd.Timestamp("2022-10-01"), pd.Timestamp("2023-09-30 23:00:00")),
    ("WY2024", pd.Timestamp("2023-10-01"), pd.Timestamp("2024-09-30 23:00:00")),
    ("WY2025", pd.Timestamp("2024-10-01"), pd.Timestamp("2025-09-30 23:00:00")),
    ("WY2026", pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:00:00")),
]

# Jan 2023 comparison period
JAN2023_START = pd.Timestamp("2023-01-01T00:00:00")
JAN2023_END   = pd.Timestamp("2023-01-31T23:00:00")

# 2H-C recovered files directory
REC_2HC_DIR = (
    REPO_ROOT / "tmp/stage1_pilot_dryrun/15_streamflow_recovery_january_eligible"
    / "recovered_camelsh_like"
)

# Late-2025 gap threshold
LATE_2025_GAP_DAYS = 14

# Versioned pilot manifest — checked first; works on HPC without generated tmp/
CONFIG_MANIFEST_PATH = REPO_ROOT / "config/stage1_pilot_basin_manifest.csv"

# Legacy generated manifest — backward-compatible fallback only
PILOT_MANIFEST_PATH = (
    REPO_ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/pilot_basin_manifest.csv"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "UNKNOWN"


def load_pilot_manifest(
    path: pathlib.Path = PILOT_MANIFEST_PATH,
) -> dict[str, str]:
    """
    Load pilot_basin_manifest.csv and return {STAID: pilot_role} mapping.
    STAIDs are zero-padded to 8 characters.  Returns empty dict if file absent.
    """
    if not path.exists():
        print(
            f"  WARNING: pilot manifest not found at {path}; "
            "pilot_role will be UNKNOWN for all basins.",
            file=sys.stderr,
        )
        return {}
    df = pd.read_csv(path, dtype=str)
    # Normalise column names — manifest may use 'staid' or 'STAID'
    df.columns = [c.strip().upper() for c in df.columns]
    if "STAID" not in df.columns or "PILOT_ROLE" not in df.columns:
        print(
            f"  WARNING: manifest missing STAID or PILOT_ROLE columns; got {list(df.columns)}",
            file=sys.stderr,
        )
        return {}
    return {str(row["STAID"]).zfill(8): str(row["PILOT_ROLE"]) for _, row in df.iterrows()}


def load_canonical_nc(nc_path: pathlib.Path) -> tuple[pd.Series, dict]:
    """
    Load a canonical hourly NC.
    Returns (streamflow_series, attrs_dict).
    Series index is UTC-naive DatetimeIndex, values float32.
    """
    ds = xr.open_dataset(str(nc_path))
    t  = pd.DatetimeIndex(ds["time"].values)
    v  = ds["streamflow"].values.astype(np.float32)
    attrs = dict(ds["streamflow"].attrs)
    ds.close()
    return pd.Series(v, index=t, dtype=np.float32), attrs


# ---------------------------------------------------------------------------
# Per-basin coverage
# ---------------------------------------------------------------------------

def compute_per_basin_coverage(
    staid: str,
    sf: pd.Series,
    attrs: dict,
    period_end: pd.Timestamp = PERIOD_END,
) -> dict:
    n_total = len(sf)
    valid   = sf.dropna()
    n_valid = len(valid)
    n_nan   = n_total - n_valid
    n_neg   = int((valid < 0).sum()) if n_valid > 0 else 0

    first_valid = str(valid.index[0])  if n_valid > 0 else ""
    last_valid  = str(valid.index[-1]) if n_valid > 0 else ""

    # Late-period gap check (against effective period_end, not always 2025-12-31)
    late_gap_flag = False
    gap_days = float("nan")
    if n_valid > 0:
        last_obs = valid.index[-1]
        gap_days = (period_end - last_obs).total_seconds() / 86400.0
        late_gap_flag = gap_days > LATE_2025_GAP_DAYS

    return {
        "STAID":                  staid,
        "n_hours_total":          n_total,
        "n_valid":                n_valid,
        "n_nan":                  n_nan,
        "coverage_fraction":      round(n_valid / n_total, 6) if n_total > 0 else 0.0,
        "first_valid_utc":        first_valid,
        "last_valid_utc":         last_valid,
        "n_exact_snaps":          int(attrs.get("snap_n_exact", 0)),
        "n_nearest_snaps":        int(attrs.get("snap_n_nearest", 0)),
        "n_missing_snaps":        int(attrs.get("snap_n_missing", 0)),
        "median_cadence_min":     attrs.get("median_cadence_minutes", ""),
        "systematic_offset_flag": str(attrs.get("systematic_offset_flag", "False")),
        "n_negative_values":      n_neg,
        "late_2025_gap_flag":     late_gap_flag,
        "late_2025_gap_note":     (
            f"last valid obs is {gap_days:.1f} days before period end"
            if late_gap_flag else ""
        ),
    }


# ---------------------------------------------------------------------------
# Per-water-year coverage
# ---------------------------------------------------------------------------

def compute_wy_coverage(staid: str, sf: pd.Series) -> list[dict]:
    rows = []
    for wy_label, wy_start, wy_end in WY_BOUNDARIES:
        mask  = (sf.index >= wy_start) & (sf.index <= wy_end)
        chunk = sf[mask]
        n_tot = len(chunk)
        n_val = int(chunk.notna().sum())
        n_nan = n_tot - n_val
        rows.append({
            "STAID":              staid,
            "water_year":         wy_label,
            "start_utc":          str(wy_start),
            "end_utc":            str(wy_end),
            "n_hours_total":      n_tot,
            "n_valid":            n_val,
            "n_nan":              n_nan,
            "coverage_fraction":  round(n_val / n_tot, 6) if n_tot > 0 else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Per-month coverage
# ---------------------------------------------------------------------------

def compute_month_coverage(staid: str, sf: pd.Series) -> list[dict]:
    rows = []
    for (yr, mo), grp in sf.groupby([sf.index.year, sf.index.month]):
        n_tot = len(grp)
        n_val = int(grp.notna().sum())
        rows.append({
            "STAID":             staid,
            "year":              yr,
            "month":             mo,
            "n_hours_total":     n_tot,
            "n_valid":           n_val,
            "n_nan":             n_tot - n_val,
            "coverage_fraction": round(n_val / n_tot, 6) if n_tot > 0 else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Gap audit
# ---------------------------------------------------------------------------

def compute_gap_audit(
    staid: str,
    sf: pd.Series,
    period_end: pd.Timestamp = PERIOD_END,
) -> dict:
    is_nan = sf.isna()
    n_gaps = 0
    longest_gap = 0
    gap_starts = []
    cur_gap = 0
    cur_start = None

    for t, nan_val in zip(sf.index, is_nan.values):
        if nan_val:
            if cur_gap == 0:
                cur_start = t
            cur_gap += 1
        else:
            if cur_gap > 0:
                n_gaps += 1
                if cur_gap > longest_gap:
                    longest_gap = cur_gap
                if cur_gap > 24:
                    gap_starts.append({"start": str(cur_start), "hours": cur_gap})
                cur_gap = 0
                cur_start = None

    # Close trailing gap
    if cur_gap > 0:
        n_gaps += 1
        if cur_gap > longest_gap:
            longest_gap = cur_gap
        if cur_gap > 24:
            gap_starts.append({"start": str(cur_start), "hours": cur_gap})

    # Late-period gap check (against effective period_end, not always 2025-12-31)
    valid = sf.dropna()
    late_gap_flag = False
    last_valid_utc = ""
    gap_to_period_end_days = float("nan")
    if len(valid) > 0:
        last_valid_utc = str(valid.index[-1])
        gap_to_period_end_days = round(
            (period_end - valid.index[-1]).total_seconds() / 86400.0, 2
        )
        late_gap_flag = gap_to_period_end_days > LATE_2025_GAP_DAYS

    return {
        "STAID":                      staid,
        "n_gaps":                     n_gaps,
        "longest_gap_hours":          longest_gap,
        "total_nan_hours":            int(is_nan.sum()),
        "n_large_gaps_gt24h":         len(gap_starts),
        "large_gap_starts_json":      json.dumps(gap_starts[:20]),
        "last_valid_utc":             last_valid_utc,
        "gap_to_period_end_days":     gap_to_period_end_days,
        "late_2025_gap_flag":         late_gap_flag,
        "late_2025_gap_note":         (
            f"STAID {staid}: last obs {last_valid_utc}, "
            f"{gap_to_period_end_days:.1f} days before period end"
        ) if late_gap_flag else "",
    }


# ---------------------------------------------------------------------------
# Quality audit
# ---------------------------------------------------------------------------

def compute_quality_audit(staid: str, sf: pd.Series, attrs: dict) -> dict:
    valid = sf.dropna()
    n_valid = len(valid)

    n_neg       = int((valid < 0).sum()) if n_valid > 0 else 0
    n_zero      = int((valid == 0).sum()) if n_valid > 0 else 0

    # Flatline detection: runs of identical consecutive values >= 24h
    n_flatline_runs  = 0
    longest_flatline = 0
    if n_valid >= 2:
        vals_arr = valid.values
        run_len  = 1
        for i in range(1, len(vals_arr)):
            if vals_arr[i] == vals_arr[i - 1]:
                run_len += 1
                if run_len >= 24:
                    if run_len == 24:  # newly crossed threshold
                        n_flatline_runs += 1
                    longest_flatline = max(longest_flatline, run_len)
            else:
                run_len = 1

    # Suspicious spike: any value > 5x the 99th percentile
    suspicious_spike = False
    if n_valid > 10:
        p99 = float(np.nanpercentile(valid.values, 99))
        if p99 > 0 and float(valid.max()) > 5 * p99:
            suspicious_spike = True

    return {
        "STAID":                    staid,
        "n_valid":                  n_valid,
        "n_negative_values":        n_neg,
        "n_zero_values":            n_zero,
        "n_flatline_runs_ge24h":    n_flatline_runs,
        "longest_flatline_hours":   longest_flatline,
        "suspicious_spike_flag":    suspicious_spike,
        "n_provisional_raw_obs":    attrs.get("n_provisional_raw_obs", ""),
        "n_ice_raw_obs":            attrs.get("n_ice_raw_obs", ""),
        "n_estimated_raw_obs":      attrs.get("n_estimated_raw_obs", ""),
        "systematic_offset_flag":   str(attrs.get("systematic_offset_flag", "False")),
        "snap_n_exact":             attrs.get("snap_n_exact", ""),
        "snap_n_nearest":           attrs.get("snap_n_nearest", ""),
        "snap_n_missing":           attrs.get("snap_n_missing", ""),
        "median_cadence_min":       attrs.get("median_cadence_minutes", ""),
        "min_streamflow_m3s":       round(float(valid.min()), 6) if n_valid > 0 else float("nan"),
        "max_streamflow_m3s":       round(float(valid.max()), 6) if n_valid > 0 else float("nan"),
        "p50_streamflow_m3s":       round(float(np.nanmedian(valid.values)), 6) if n_valid > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Advisory target-status classification
# ---------------------------------------------------------------------------

# Status classes (priority order; see docs/stage1_usgs_iv_full_period_target_plan.md)
_READY_STATUSES      = {"TARGET_READY_CONTINUOUS", "TARGET_USABLE_WITH_GAPS"}
_HIST_UTIL_STATUSES  = {
    "TARGET_READY_CONTINUOUS",
    "TARGET_USABLE_WITH_GAPS",
    "TARGET_QUALITY_REVIEW",
    "TARGET_TIME_OFFSET_REVIEW",
}


def classify_target_status(
    staid: str,
    pilot_role: str,
    pb_row: dict,
    gap_row: dict,
    qual_row: dict,
    wy_coverage_rows: list[dict],
) -> dict:
    """
    Assign advisory target-status class and readiness flags.

    Policy: gaps do NOT automatically exclude a basin from training.
    Flags are advisory — distinguishing operational readiness from
    historical training utility.
    """
    n_valid      = pb_row.get("n_valid", 0)
    late_gap     = bool(gap_row.get("late_2025_gap_flag", False))
    longest_gap  = int(gap_row.get("longest_gap_hours", 0))
    sys_offset   = str(qual_row.get("systematic_offset_flag", "False")).lower() == "true"
    n_neg        = int(qual_row.get("n_negative_values", 0))
    spike        = bool(qual_row.get("suspicious_spike_flag", False))

    # Minimum WY coverage across all water-years for this basin
    wy_fracs   = [r["coverage_fraction"] for r in wy_coverage_rows if r["STAID"] == staid]
    min_wy_cov = min(wy_fracs) if wy_fracs else 0.0

    # Priority-ordered classification
    if pilot_role == "EXCLUDE_QC":
        status = "TARGET_ROLE_EXCLUDED"
    elif n_valid == 0:
        status = "TARGET_NO_DATA"
    elif late_gap:
        status = "TARGET_OPERATIONAL_REVIEW"
    elif sys_offset:
        status = "TARGET_TIME_OFFSET_REVIEW"
    elif n_neg > 0:
        status = "TARGET_QUALITY_REVIEW"
    elif spike:
        status = "TARGET_QUALITY_REVIEW"
    elif longest_gap >= 168 or min_wy_cov < 0.95:
        status = "TARGET_USABLE_WITH_GAPS"
    else:
        status = "TARGET_READY_CONTINUOUS"

    op_ready  = status in _READY_STATUSES
    hist_util = status in _HIST_UTIL_STATUSES and pilot_role != "EXCLUDE_QC"

    return {
        "STAID":                             staid,
        "pilot_role":                        pilot_role,
        "target_status":                     status,
        "operational_readiness_flag":        op_ready,
        "historical_training_utility_flag":  hist_util,
        "coverage_fraction":                 pb_row.get("coverage_fraction", 0.0),
        "longest_gap_hours":                 longest_gap,
        "min_wy_coverage_fraction":          round(min_wy_cov, 6),
        "late_2025_gap_flag":                late_gap,
        "systematic_offset_flag":            sys_offset,
        "n_negative_values":                 n_neg,
        "suspicious_spike_flag":             spike,
    }


# ---------------------------------------------------------------------------
# Jan 2023 comparison vs 2H-C
# ---------------------------------------------------------------------------

def compare_jan2023(staid: str, sf_full: pd.Series) -> dict:
    """
    Compare January 2023 slice of the new full-period canonical file against
    the 2H-C recovered file (if available).
    Returns a comparison row dict.
    """
    rec_path = REC_2HC_DIR / f"{staid}_hourly.nc"
    jan_mask = (sf_full.index >= JAN2023_START) & (sf_full.index <= JAN2023_END)
    sf_jan   = sf_full[jan_mask]
    n_hours  = len(sf_jan)

    result = {
        "STAID":             staid,
        "n_jan_hours":       n_hours,
        "n_valid_fullperiod": int(sf_jan.notna().sum()),
        "n_nan_fullperiod":  int(sf_jan.isna().sum()),
        "comparison_status": "NOT_APPLICABLE",
        "n_valid_2hc":       None,
        "n_matched_exactly": None,
        "n_float32_match":   None,
        "max_abs_diff":      None,
        "note":              "",
    }

    if not rec_path.exists():
        result["note"] = "2H-C file not found; comparison not applicable"
        return result

    try:
        ds2hc = xr.open_dataset(str(rec_path))
        # 2H-C uses 'time' coordinate (UTC-naive datetime64)
        t2hc  = pd.to_datetime(ds2hc["time"].values)
        v2hc  = ds2hc["streamflow"].values.astype(np.float32)
        ds2hc.close()
    except Exception as exc:
        result["comparison_status"] = "2HC_READ_ERROR"
        result["note"] = str(exc)
        return result

    s2hc = pd.Series(v2hc, index=t2hc, dtype=np.float32)
    # Align to Jan 2023
    s2hc_jan = s2hc[(s2hc.index >= JAN2023_START) & (s2hc.index <= JAN2023_END)]
    result["n_valid_2hc"] = int(s2hc_jan.notna().sum())

    # Align common index
    common = sf_jan.index.intersection(s2hc_jan.index)
    if len(common) == 0:
        result["comparison_status"] = "NO_COMMON_TIMESTAMPS"
        result["note"] = "No common timestamps between full-period file and 2H-C file"
        return result

    a = sf_jan.reindex(common).values.astype(np.float32)
    b = s2hc_jan.reindex(common).values.astype(np.float32)

    # Compare where both are non-NaN
    both_valid = ~np.isnan(a) & ~np.isnan(b)
    n_both = int(both_valid.sum())

    if n_both == 0:
        result["comparison_status"] = "NO_COMMON_VALID"
        result["note"] = "No hours where both files have valid data"
        return result

    diffs = np.abs(a[both_valid] - b[both_valid])
    max_diff   = float(np.max(diffs))
    n_exact    = int(np.sum(a[both_valid] == b[both_valid]))
    # float32 tolerance: values equal within float32 machine epsilon
    float32_eps = np.finfo(np.float32).eps * max(float(np.max(np.abs(a[both_valid]))), 1.0)
    n_f32match  = int(np.sum(diffs <= float32_eps * 2))

    pass_criteria = max_diff <= float32_eps * 2 * 100  # generous: <1e-4 m3/s

    result.update({
        "comparison_status": "PASS" if pass_criteria else "MISMATCH",
        "n_compared_hours":  n_both,
        "n_matched_exactly": n_exact,
        "n_float32_match":   n_f32match,
        "max_abs_diff":      round(max_diff, 8),
        "note": (
            f"max_diff={max_diff:.2e} m3/s; "
            f"exact={n_exact}/{n_both}; "
            f"f32_match={n_f32match}/{n_both}"
        ),
    })
    return result


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def make_comparison_plot(
    comparison_rows: list[dict],
    canonical_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    staids_with_2hc = [
        r["STAID"] for r in comparison_rows
        if r.get("comparison_status") not in ("NOT_APPLICABLE", "2HC_READ_ERROR", "NO_COMMON_TIMESTAMPS", "NO_COMMON_VALID")
    ]
    if not staids_with_2hc:
        return

    n_cols = min(3, len(staids_with_2hc))
    n_rows = (len(staids_with_2hc) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows), squeeze=False)
    fig.suptitle("Jan 2023: Full-Period Canonical vs 2H-C Recovered", fontsize=12, fontweight="bold")

    for i, staid in enumerate(staids_with_2hc):
        ax = axes[i // n_cols][i % n_cols]
        nc_path = canonical_dir / f"{staid}_hourly.nc"
        rec_path = REC_2HC_DIR / f"{staid}_hourly.nc"

        try:
            sf_full, _ = load_canonical_nc(nc_path)
            jan_mask = (sf_full.index >= JAN2023_START) & (sf_full.index <= JAN2023_END)
            sf_jan = sf_full[jan_mask]
            ax.plot(sf_jan.index, sf_jan.values, color="steelblue", lw=1.5,
                    label="Full-period (2I-B)", zorder=3)
        except Exception:
            ax.set_title(f"{staid}\n(load error)")
            continue

        if rec_path.exists():
            try:
                ds2 = xr.open_dataset(str(rec_path))
                t2  = pd.DatetimeIndex(ds2["time"].values)
                v2  = ds2["streamflow"].values.astype(np.float32)
                ds2.close()
                s2  = pd.Series(v2, index=t2)
                s2_jan = s2[(s2.index >= JAN2023_START) & (s2.index <= JAN2023_END)]
                ax.plot(s2_jan.index, s2_jan.values, color="darkorange", lw=1.0,
                        linestyle="--", label="2H-C recovered", zorder=2, alpha=0.8)
            except Exception:
                pass

        ax.set_title(f"{staid}", fontsize=9)
        ax.set_ylabel("m³/s", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    # Hide unused panels
    for i in range(len(staids_with_2hc), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison plot: {out_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(
    out_dir: pathlib.Path,
    per_basin: list[dict],
    wy_rows: list[dict],
    gap_rows: list[dict],
    qual_rows: list[dict],
    jan_rows: list[dict],
    status_rows: list[dict],
    git_hash: str,
    generated_utc: str,
    n_hours_target: int,
) -> None:
    n_basins   = len(per_basin)
    n_pass_val = sum(1 for r in per_basin if r.get("coverage_fraction", 0) >= 0.90)
    n_late_gap = sum(1 for r in gap_rows if r.get("late_2025_gap_flag"))
    n_neg      = sum(1 for r in qual_rows if r.get("n_negative_values", 0) > 0)
    n_spike    = sum(1 for r in qual_rows if r.get("suspicious_spike_flag"))
    n_jan_pass = sum(1 for r in jan_rows if r.get("comparison_status") == "PASS")
    n_jan_na   = sum(1 for r in jan_rows if r.get("comparison_status") == "NOT_APPLICABLE")
    n_jan_mm   = sum(1 for r in jan_rows if r.get("comparison_status") == "MISMATCH")

    total_valid = sum(r.get("n_valid", 0) for r in per_basin)
    total_nan   = sum(r.get("n_nan", 0) for r in per_basin)
    total_hours = sum(r.get("n_hours_total", 0) for r in per_basin)
    overall_cov = total_valid / total_hours if total_hours > 0 else 0.0

    # Target-status counts
    from collections import Counter
    status_counts  = Counter(r["target_status"] for r in status_rows)
    n_op_ready     = sum(1 for r in status_rows if r.get("operational_readiness_flag"))
    n_hist_util    = sum(1 for r in status_rows if r.get("historical_training_utility_flag"))

    lines = [
        "# Flash-NH Stage 1 Milestone 2I-B — Acquisition Audit Summary",
        "",
        f"Generated: {generated_utc}  |  git: {git_hash}  |  Script: {SCRIPT_NAME}",
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Basins audited | {n_basins} |",
        f"| Target hourly steps per basin | {n_hours_target:,} |",
        f"| Period | 2020-10-14T00:00:00Z to 2025-12-31T23:00:00Z |",
        f"| Total valid hours (all basins) | {total_valid:,} |",
        f"| Total NaN hours (all basins) | {total_nan:,} |",
        f"| Overall coverage fraction | {overall_cov:.4f} ({100*overall_cov:.2f}%) |",
        f"| Basins >=90% coverage | {n_pass_val}/{n_basins} |",
        f"| Basins with late-2025 gap flag | {n_late_gap} |",
        f"| Basins with any negative values | {n_neg} |",
        f"| Basins with suspicious spike | {n_spike} |",
        "",
        "## Per-Basin Coverage",
        "",
        "| STAID | n_valid | n_nan | coverage | late_2025_gap |",
        "|---|---|---|---|---|",
    ]
    for r in per_basin:
        flag = "*** YES ***" if r.get("late_2025_gap_flag") else "no"
        lines.append(
            f"| {r['STAID']} | {r['n_valid']:,} | {r['n_nan']:,} | "
            f"{r['coverage_fraction']:.4f} | {flag} |"
        )

    lines += [
        "",
        "## Jan 2023 Comparison vs 2H-C",
        "",
        "| STAID | status | max_abs_diff | n_compared | note |",
        "|---|---|---|---|---|",
    ]
    for r in jan_rows:
        lines.append(
            f"| {r['STAID']} | {r['comparison_status']} | "
            f"{r.get('max_abs_diff', 'N/A')} | "
            f"{r.get('n_compared_hours', 'N/A')} | "
            f"{r.get('note', '')[:60]} |"
        )

    if n_late_gap > 0:
        lines += ["", "## Late-2025 Gap Warnings", ""]
        for r in gap_rows:
            if r.get("late_2025_gap_flag"):
                lines.append(f"- **{r['STAID']}**: {r['late_2025_gap_note']}")

    # Advisory target-status section
    all_status_classes = [
        "TARGET_READY_CONTINUOUS",
        "TARGET_USABLE_WITH_GAPS",
        "TARGET_OPERATIONAL_REVIEW",
        "TARGET_QUALITY_REVIEW",
        "TARGET_TIME_OFFSET_REVIEW",
        "TARGET_NO_DATA",
        "TARGET_ROLE_EXCLUDED",
    ]
    lines += [
        "",
        "## Advisory Target Status",
        "",
        "> **Policy:** Advisory status classes do not automatically exclude basins from training.",
        "> TARGET_OPERATIONAL_REVIEW and TARGET_QUALITY_REVIEW basins require manual inspection",
        "> before HPC-scale acquisition. Gap presence (TARGET_USABLE_WITH_GAPS) is expected for",
        "> seasonal catchments and does not disqualify a basin.",
        "",
        "| Status class | Count |",
        "|---|---|",
    ]
    for sc in all_status_classes:
        cnt = status_counts.get(sc, 0)
        if cnt > 0 or sc in ("TARGET_READY_CONTINUOUS", "TARGET_ROLE_EXCLUDED"):
            lines.append(f"| {sc} | {cnt} |")

    lines += [
        "",
        f"| operational_readiness_flag = True | {n_op_ready}/{n_basins} |",
        f"| historical_training_utility_flag = True | {n_hist_util}/{n_basins} |",
        "",
        "### Per-Basin Advisory Status",
        "",
        "| STAID | pilot_role | target_status | op_ready | hist_util | cov | longest_gap_h |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in status_rows:
        lines.append(
            f"| {r['STAID']} | {r['pilot_role']} | {r['target_status']} | "
            f"{r['operational_readiness_flag']} | {r['historical_training_utility_flag']} | "
            f"{r['coverage_fraction']:.4f} | {r['longest_gap_hours']} |"
        )

    lines += [
        "",
        "## Validation Criteria",
        "",
        f"- [{'PASS' if n_basins > 0 else 'N/A'}] {n_basins} canonical NCs present",
        f"- [PASS] target grid: {n_hours_target:,} hourly steps (period-derived)",
        f"- [{'PASS' if n_jan_mm == 0 else 'FAIL'}] "
        f"Jan 2023 comparison: {n_jan_pass} PASS, {n_jan_mm} MISMATCH, {n_jan_na} N/A",
        f"- [{'PASS' if n_neg == 0 else 'CHECK'}] "
        f"Negative values: {n_neg} basin(s) affected",
        f"- [{'WARN-LATE-GAP' if n_late_gap > 0 else 'PASS'}] "
        f"Late-2025 gap: {n_late_gap} basin(s) flagged",
        "",
    ]

    md_text = "\n".join(lines)
    (out_dir / "summary.md").write_text(md_text, encoding="utf-8")

    summary_json = {
        "generated_utc":            generated_utc,
        "git_commit":               git_hash,
        "n_basins":                 n_basins,
        "n_hours_target":           n_hours_target,
        "total_valid_hours":        total_valid,
        "total_nan_hours":          total_nan,
        "overall_coverage":         round(overall_cov, 6),
        "n_basins_ge90pct":         n_pass_val,
        "n_late_gap_flag":          n_late_gap,
        "n_basins_negative":        n_neg,
        "n_basins_spike":           n_spike,
        "jan2023_comparison":       {"PASS": n_jan_pass, "MISMATCH": n_jan_mm, "NOT_APPLICABLE": n_jan_na},
        "advisory_policy": (
            "Advisory status classes do not automatically exclude basins from training. "
            "TARGET_OPERATIONAL_REVIEW and TARGET_QUALITY_REVIEW require manual inspection "
            "before HPC-scale acquisition. Gaps (TARGET_USABLE_WITH_GAPS) are expected for "
            "seasonal catchments and do not disqualify a basin."
        ),
        "target_status_counts":     dict(status_counts),
        "n_operational_ready":      n_op_ready,
        "n_historical_utility":     n_hist_util,
        "per_basin":                per_basin,
        "target_status":            status_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit full-period USGS IV recovered targets for Flash-NH Stage 1 2I-B."
    )
    p.add_argument(
        "--canonical-dir", type=pathlib.Path,
        default=REPO_ROOT / "tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot/canonical",
        help="Directory containing {STAID}_hourly.nc files.",
    )
    p.add_argument(
        "--out-dir", type=pathlib.Path,
        default=REPO_ROOT / "tmp/stage1_pilot_dryrun/17_usgs_iv_full_period_pilot",
        help="Root output directory (audit/, qc/, summary.md, etc. written here).",
    )
    p.add_argument(
        "--staids", type=str, default="",
        help="Comma-separated STAIDs to audit (default: all NC files in canonical-dir).",
    )
    # Period override (inferred from NC attrs if not provided)
    p.add_argument(
        "--expected-start", type=str, default="",
        help="Override expected period start (ISO 8601 UTC, e.g. 2023-01-01T00:00:00Z). "
             "If omitted, inferred from canonical NC attrs.",
    )
    p.add_argument(
        "--expected-end", type=str, default="",
        help="Override expected period end (ISO 8601 UTC, e.g. 2023-01-31T23:00:00Z). "
             "If omitted, inferred from canonical NC attrs.",
    )
    # Manifest and comparison path overrides (useful on HPC where paths differ)
    p.add_argument(
        "--pilot-manifest", type=pathlib.Path, default=None,
        help="Override path to pilot_basin_manifest.csv. "
             "Defaults to the standard local path; pass /dev/null or nonexistent to suppress.",
    )
    p.add_argument(
        "--skip-jan2023-comparison", action="store_true",
        help="Skip the Jan 2023 vs 2H-C comparison entirely. "
             "Useful for smoke runs on HPC where 2H-C files are not available.",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    canon_dir  = args.canonical_dir
    out_dir    = args.out_dir
    audit_dir  = out_dir / "audit"
    qc_dir     = out_dir / "qc"
    prov_dir   = out_dir / "provenance"

    for d in (audit_dir, qc_dir, prov_dir):
        d.mkdir(parents=True, exist_ok=True)

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_hash      = git_commit_hash()

    # Discover NC files
    if args.staids:
        nc_paths = [canon_dir / f"{s.strip().zfill(8)}_hourly.nc" for s in args.staids.split(",")]
        nc_paths = [p for p in nc_paths if p.exists()]
    else:
        nc_paths = sorted(canon_dir.glob("*_hourly.nc"))

    if not nc_paths:
        print(f"No canonical NC files found in {canon_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Auditing {len(nc_paths)} canonical NC file(s)")

    # Resolve effective audit period
    # Priority: CLI args > first NC file attrs > module-level defaults
    audit_start, audit_end = PERIOD_START, PERIOD_END
    if args.expected_start and args.expected_end:
        audit_start = pd.Timestamp(args.expected_start.replace("Z", ""))
        audit_end   = pd.Timestamp(args.expected_end.replace("Z", ""))
        print(f"  Audit period (CLI): {audit_start} to {audit_end}")
    else:
        try:
            _, peek_attrs = load_canonical_nc(nc_paths[0])
            ps = peek_attrs.get("period_start_utc", "")
            pe = peek_attrs.get("period_end_utc", "")
            if ps and pe:
                audit_start = pd.Timestamp(ps.replace("Z", ""))
                audit_end   = pd.Timestamp(pe.replace("Z", ""))
                print(f"  Audit period (NC attrs): {audit_start} to {audit_end}")
            else:
                print(f"  Audit period (default): {audit_start} to {audit_end}")
        except Exception:
            print(f"  Audit period (default): {audit_start} to {audit_end}")

    # Build expected target index from effective period
    target_index    = pd.date_range(audit_start, audit_end, freq="h")
    n_hours_target  = len(target_index)

    per_basin_rows: list[dict] = []
    wy_rows:        list[dict] = []
    month_rows:     list[dict] = []
    gap_rows:       list[dict] = []
    qual_rows:      list[dict] = []
    jan_rows:       list[dict] = []

    for nc_path in nc_paths:
        staid = nc_path.stem.replace("_hourly", "").zfill(8)
        print(f"  [{staid}] loading {nc_path.name}")

        try:
            sf, attrs = load_canonical_nc(nc_path)
        except Exception as exc:
            print(f"  [{staid}] ERROR loading: {exc}")
            continue

        # Verify no -9999 sentinels in decoded array
        n_sentinel = int(np.sum(sf.values == -9999.0))
        if n_sentinel > 0:
            print(f"  [{staid}] WARNING: {n_sentinel} decoded -9999 values found!")

        per_basin_rows.append(compute_per_basin_coverage(staid, sf, attrs, period_end=audit_end))
        wy_rows.extend(compute_wy_coverage(staid, sf))
        month_rows.extend(compute_month_coverage(staid, sf))
        gap_rows.append(compute_gap_audit(staid, sf, period_end=audit_end))
        qual_rows.append(compute_quality_audit(staid, sf, attrs))
        if args.skip_jan2023_comparison:
            jan_rows.append({
                "STAID": staid,
                "n_jan_hours": 0,
                "n_valid_fullperiod": 0,
                "n_nan_fullperiod": 0,
                "comparison_status": "SKIPPED",
                "n_valid_2hc": None,
                "n_matched_exactly": None,
                "n_float32_match": None,
                "max_abs_diff": None,
                "note": "skipped via --skip-jan2023-comparison",
            })
        else:
            jan_rows.append(compare_jan2023(staid, sf))

        row = per_basin_rows[-1]
        print(
            f"  [{staid}] valid={row['n_valid']:,} nan={row['n_nan']:,} "
            f"cov={row['coverage_fraction']:.4f} "
            f"late_gap={row['late_2025_gap_flag']}"
        )

    # Load pilot manifest for advisory classification.
    # Priority: --pilot-manifest CLI > versioned config > legacy generated tmp fallback.
    if args.pilot_manifest is not None:
        manifest_path = args.pilot_manifest
    elif CONFIG_MANIFEST_PATH.exists():
        manifest_path = CONFIG_MANIFEST_PATH
    else:
        manifest_path = PILOT_MANIFEST_PATH  # legacy fallback (not present on HPC)
    pilot_roles = load_pilot_manifest(path=manifest_path)

    # Advisory target-status classification
    status_rows: list[dict] = []
    for pb_row in per_basin_rows:
        staid    = pb_row["STAID"]
        role     = pilot_roles.get(staid, "UNKNOWN")
        gap_row  = next((r for r in gap_rows  if r["STAID"] == staid), {})
        qual_row = next((r for r in qual_rows if r["STAID"] == staid), {})
        wy_basin = [r for r in wy_rows if r["STAID"] == staid]
        status_rows.append(classify_target_status(staid, role, pb_row, gap_row, qual_row, wy_basin))

    # Write audit CSVs
    pd.DataFrame(per_basin_rows).to_csv(audit_dir / "per_basin_coverage.csv", index=False)
    pd.DataFrame(wy_rows).to_csv(audit_dir / "per_water_year_coverage.csv", index=False)
    pd.DataFrame(month_rows).to_csv(audit_dir / "per_month_coverage.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(audit_dir / "gap_audit.csv", index=False)
    pd.DataFrame(qual_rows).to_csv(audit_dir / "quality_audit.csv", index=False)
    pd.DataFrame(jan_rows).to_csv(audit_dir / "jan2023_comparison_against_2hc.csv", index=False)
    pd.DataFrame(status_rows).to_csv(audit_dir / "target_status.csv", index=False)
    print(f"  Wrote target_status.csv ({len(status_rows)} rows)")

    # Comparison plot (skip if --skip-jan2023-comparison)
    if not args.skip_jan2023_comparison:
        make_comparison_plot(jan_rows, canon_dir, qc_dir / "jan2023_fullperiod_vs_2hc_comparison.png")

    # Summary
    write_summary(
        out_dir, per_basin_rows, wy_rows, gap_rows, qual_rows, jan_rows,
        status_rows, git_hash, generated_utc, n_hours_target
    )

    # Provenance
    prov = {
        "milestone":       "Flash-NH Stage 1 2I-B",
        "script":          SCRIPT_NAME,
        "git_commit":      git_hash,
        "generated_utc":   generated_utc,
        "python_version":  sys.version,
        "pandas_version":  pd.__version__,
        "numpy_version":   np.__version__,
        "xarray_version":  xr.__version__,
        "canonical_dir":   str(canon_dir),
        "n_basins":        len(per_basin_rows),
        "n_hours_target":  n_hours_target,
    }
    (prov_dir / "run_provenance.json").write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8"
    )

    # Late-gap warnings
    late_gap_basins = [r for r in gap_rows if r.get("late_2025_gap_flag")]
    if late_gap_basins:
        print("\n*** LATE-2025 GAP FLAGS ***")
        for r in late_gap_basins:
            print(f"   {r['late_2025_gap_note']}")

    # Jan 2023 comparison result
    print("\nJan 2023 comparison vs 2H-C:")
    for r in jan_rows:
        print(f"  {r['STAID']}: {r['comparison_status']}  {r.get('note', '')}")

    # Final summary
    n_basins   = len(per_basin_rows)
    total_v    = sum(r["n_valid"] for r in per_basin_rows)
    total_n    = sum(r["n_nan"] for r in per_basin_rows)
    total_h    = sum(r["n_hours_total"] for r in per_basin_rows)
    cov        = total_v / total_h if total_h > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Milestone 2I-B audit — {n_basins} basins")
    print(f"  Total valid: {total_v:,}  NaN: {total_n:,}  Coverage: {cov:.4f}")
    print(f"  Outputs: {audit_dir}")
    print()


if __name__ == "__main__":
    main()
