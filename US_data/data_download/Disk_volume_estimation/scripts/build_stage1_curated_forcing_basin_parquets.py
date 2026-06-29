#!/usr/bin/env python3
"""Build Stage 1 curated forcing product v001 per-basin wide Parquet files.

Reads the monthly combined long-format Parquet produced by the full-period
extraction and writes one wide-format Parquet per selected basin:
  {out_dir}/time_series/{STAID}.parquet

One row per UTC hour. NaN for gap hours. Gap-flag boolean columns included.

Intended for smoke test (5 basins, 2020-11) and later full production builds.
Full 2,752-basin build is NOT authorised in this milestone (2K-F-B smoke only).

Design reference: docs/stage1_curated_forcing_product_v001_design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MRMS_PRODUCT_ID = "mrms_qpe_1h_pass1"
_RTMA_PRODUCT_ID = "rtma_conus_aws_2p5km"

# Mapping from RTMA GRIB short_name (as stored in source Parquet `variable` col)
# to curated column name.  Includes both known alias spellings.
_RTMA_SRC_TO_CURATED: dict[str, str] = {
    "2t":    "rtma_2t_K",
    "d2m":   "rtma_2d_K",
    "sh2":   "rtma_2sh_kgkg",    # primary name observed in production
    "2sh":   "rtma_2sh_kgkg",    # alternate name (same variable)
    "sp":    "rtma_sp_Pa",
    "10u":   "rtma_10u_ms",
    "10v":   "rtma_10v_ms",
    "tcc":   "rtma_tcc_pct",
    "vis":   "rtma_vis_m",
    "gust":  "rtma_gust_ms",     # primary name
    "i10fg": "rtma_gust_ms",     # alternate name (wind gust)
    "weasd": "rtma_weasd_kgm2",
    "ceil":  "rtma_ceil_m",
}

# Curated RTMA value column names in canonical order (11 variables)
_CURATED_RTMA_COLS: list[str] = [
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct", "rtma_vis_m",
    "rtma_gust_ms", "rtma_weasd_kgm2", "rtma_ceil_m",
]

# Full ordered column list for per-basin Parquet (excluding index)
_CURATED_COLS: list[str] = (
    ["mrms_qpe_1h_mm", "mrms_qpe_1h_mm_gap"]
    + _CURATED_RTMA_COLS
    + ["rtma_gap"]
)

# Variables that must NOT appear in the curated product
_FORBIDDEN_VARS: set[str] = {"10wdir", "orog"}

SCHEMA_VERSION = "1.0"
PRODUCT_NAME = "stage1_basin_hourly_forcings_v001"

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
        "--forcing-root", required=True,
        help="Root of the full-period extraction tree "
             "(contains chunks/{YYYY-MM}/combined_{YYYY-MM}.parquet)",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for the curated product. "
             "Must be under tmp/ for smoke builds.",
    )
    p.add_argument(
        "--month", default="2020-11",
        help="Month to process as YYYY-MM (default: 2020-11 — contains known RTMA gaps)",
    )
    p.add_argument(
        "--staids", nargs="+", default=None,
        help="Explicit STAID list. If omitted, the first --max-basins unique STAIDs "
             "found in the source Parquet are used.",
    )
    p.add_argument(
        "--max-basins", type=int, default=5,
        help="Maximum basins to process (safety cap; default 5). "
             "Ignored when --staids is given.",
    )
    p.add_argument(
        "--audit-dir", default=None,
        help="Optional path to the post-run audit directory "
             "(for gap-count cross-reference; not required for smoke).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing per-basin Parquet files.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print plan without writing any files.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _month_full_index(month: str) -> pd.DatetimeIndex:
    """Return a complete hourly UTC DatetimeIndex for the given YYYY-MM."""
    start = pd.Timestamp(f"{month}-01 00:00:00", tz="UTC")
    end   = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="h")


def _load_combined_parquet(path: Path, staids: list[str]) -> pd.DataFrame:
    """Load the combined Parquet, optionally filtering to selected STAIDs."""
    log.info("Loading %s", path)
    filters = [("STAID", "in", staids)] if staids else None
    df = pd.read_parquet(
        path,
        columns=["STAID", "product", "variable", "valid_time_utc", "weighted_mean"],
        filters=filters,
    )
    # Ensure valid_time_utc is tz-aware datetime
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    log.info("Loaded %d rows for %d STAIDs", len(df), df["STAID"].nunique())
    return df


def _check_forbidden_vars(df: pd.DataFrame) -> None:
    """Fail if any forbidden RTMA variable is present in the source data."""
    rtma = df[df["product"] == _RTMA_PRODUCT_ID]
    found = set(rtma["variable"].unique()) & _FORBIDDEN_VARS
    if found:
        log.error("FORBIDDEN RTMA variables found in source: %s", sorted(found))
        sys.exit(1)


def _build_basin_wide(
    staid: str,
    bdf: pd.DataFrame,
    full_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert long-format single-basin data to wide hourly DataFrame.

    Gap hours are identified by absence of source rows (not by NaN values),
    consistent with the v001 gap policy.
    """
    # --- MRMS ---
    mrms = bdf[bdf["product"] == _MRMS_PRODUCT_ID].copy()
    mrms_ts = set(mrms["valid_time_utc"].unique())
    mrms_gap_flags = ~full_index.isin(mrms_ts)

    mrms_vals = (
        mrms.set_index("valid_time_utc")["weighted_mean"]
        .reindex(full_index)
        .astype("float32")
    )

    # --- RTMA ---
    rtma = bdf[bdf["product"] == _RTMA_PRODUCT_ID].copy()
    rtma_ts = set(rtma["valid_time_utc"].unique())
    rtma_gap_flags = ~full_index.isin(rtma_ts)  # True = hour absent from RTMA

    # --- Wide DataFrame ---
    wide = pd.DataFrame(index=full_index)
    wide.index.name = "valid_time_utc"

    wide["mrms_qpe_1h_mm"] = mrms_vals.values
    wide["mrms_qpe_1h_mm_gap"] = mrms_gap_flags

    # Populate RTMA columns from source variables (handle alternate spellings)
    for src_var in rtma["variable"].unique():
        curated = _RTMA_SRC_TO_CURATED.get(src_var)
        if curated is None:
            log.debug("STAID=%s: unknown RTMA variable %r — skipping", staid, src_var)
            continue
        if curated in wide.columns:
            continue  # already filled (handles sh2/2sh or gust/i10fg duplicates)
        series = (
            rtma[rtma["variable"] == src_var]
            .set_index("valid_time_utc")["weighted_mean"]
            .reindex(full_index)
            .astype("float32")
        )
        wide[curated] = series.values

    # Ensure every expected RTMA column is present (fill with NaN if absent)
    for col in _CURATED_RTMA_COLS:
        if col not in wide.columns:
            log.warning("STAID=%s: RTMA column %s not found in source — filled NaN", staid, col)
            wide[col] = np.full(len(full_index), np.nan, dtype="float32")

    wide["rtma_gap"] = rtma_gap_flags

    # Enforce canonical column order
    return wide[_CURATED_COLS].copy()


def _basin_stats(staid: str, wide: pd.DataFrame, path: Path, sha: str) -> dict:
    n_hours = len(wide)
    n_mrms_gap  = int(wide["mrms_qpe_1h_mm_gap"].sum())
    n_rtma_gap  = int(wide["rtma_gap"].sum())
    n_valid     = int((~wide["mrms_qpe_1h_mm_gap"] & ~wide["rtma_gap"]).sum())
    return {
        "STAID":               staid,
        "n_hours_expected":    n_hours,
        "n_hours_written":     n_hours,
        "n_mrms_gap_hours":    n_mrms_gap,
        "n_rtma_gap_hours":    n_rtma_gap,
        "n_valid_combined_hours": n_valid,
        "coverage_fraction":   round(n_valid / n_hours, 6),
        "file_path":           str(path.relative_to(path.parent.parent)),
        "sha256":              sha,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    forcing_root = Path(args.forcing_root)
    out_dir      = Path(args.out_dir)
    month        = args.month
    dry_run      = args.dry_run

    # ---- Input validation ----
    chunk_path = forcing_root / "chunks" / month / f"combined_{month}.parquet"
    if not chunk_path.exists():
        log.error("Source Parquet not found: %s", chunk_path)
        sys.exit(1)

    full_index = _month_full_index(month)
    n_hours    = len(full_index)
    log.info("Month %s: %d hourly slots (%s → %s)",
             month, n_hours, full_index[0], full_index[-1])

    # ---- Load source ----
    # If explicit STAIDs given, pre-filter; otherwise load all and slice later
    pre_filter_staids = list(args.staids) if args.staids else None
    src_df = _load_combined_parquet(chunk_path, pre_filter_staids)

    _check_forbidden_vars(src_df)

    # Determine final STAID list
    available_staids = list(src_df["STAID"].unique())
    if args.staids:
        missing = [s for s in args.staids if s not in available_staids]
        if missing:
            log.error("Requested STAIDs not in source: %s", missing)
            sys.exit(1)
        selected_staids = list(args.staids)
    else:
        selected_staids = available_staids[: args.max_basins]

    log.info("Selected %d basins: %s", len(selected_staids), selected_staids)

    # ---- Dry-run exit ----
    if dry_run:
        log.info("[DRY-RUN] Would write %d per-basin Parquets to: %s",
                 len(selected_staids), out_dir / "time_series")
        log.info("[DRY-RUN] No files written. Exiting.")
        return

    # ---- Output layout ----
    ts_dir = out_dir / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-basin build ----
    manifest_rows: list[dict] = []
    checksum_lines: list[str] = []
    run_start = datetime.now(timezone.utc)

    for staid in selected_staids:
        out_path = ts_dir / f"{staid}.parquet"
        if out_path.exists() and not args.overwrite:
            log.info("STAID=%s: already exists, skipping (use --overwrite)", staid)
            sha = _sha256(out_path)
            wide = pd.read_parquet(out_path)
            manifest_rows.append(_basin_stats(staid, wide, out_path, sha))
            checksum_lines.append(f"{sha}  time_series/{staid}.parquet")
            continue

        log.info("STAID=%s: building wide Parquet …", staid)
        bdf  = src_df[src_df["STAID"] == staid]
        wide = _build_basin_wide(staid, bdf, full_index)

        wide.to_parquet(out_path, index=True, engine="pyarrow", compression="snappy")
        sha = _sha256(out_path)

        stats = _basin_stats(staid, wide, out_path, sha)
        log.info(
            "STAID=%s: %d rows, mrms_gap=%d, rtma_gap=%d, coverage=%.4f",
            staid, stats["n_hours_written"], stats["n_mrms_gap_hours"],
            stats["n_rtma_gap_hours"], stats["coverage_fraction"],
        )
        manifest_rows.append(stats)
        checksum_lines.append(f"{sha}  time_series/{staid}.parquet")

    run_end = datetime.now(timezone.utc)

    # ---- manifest.json ----
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_rows, f, indent=2)
    log.info("Wrote %s", manifest_path)

    # ---- checksums.sha256 ----
    checksums_path = out_dir / "checksums.sha256"
    with open(checksums_path, "w") as f:
        f.write("\n".join(checksum_lines) + "\n")
    log.info("Wrote %s", checksums_path)

    # ---- dataset_config.json ----
    dataset_config = {
        "product_name":       PRODUCT_NAME,
        "schema_version":     SCHEMA_VERSION,
        "month":              month,
        "period_start_utc":   full_index[0].isoformat(),
        "period_end_utc":     full_index[-1].isoformat(),
        "n_hours_expected":   n_hours,
        "n_basins":           len(selected_staids),
        "mrms_product":       _MRMS_PRODUCT_ID,
        "rtma_product":       _RTMA_PRODUCT_ID,
        "variables":          _CURATED_RTMA_COLS + ["mrms_qpe_1h_mm"],
        "gap_flag_columns":   ["mrms_qpe_1h_mm_gap", "rtma_gap"],
        "excluded_variables": sorted(_FORBIDDEN_VARS),
        "gap_policy":         "raw_preserve_nan_no_interpolation",
        "smoke_build":        len(selected_staids) < 100,
    }
    config_path = out_dir / "dataset_config.json"
    with open(config_path, "w") as f:
        json.dump(dataset_config, f, indent=2)
    log.info("Wrote %s", config_path)

    # ---- run_provenance.json ----
    provenance = {
        "builder_script":        "scripts/build_stage1_curated_forcing_basin_parquets.py",
        "repo_commit":           _git_commit(),
        "python_version":        sys.version,
        "platform":              platform.node(),
        "input_chunk_root":      str(forcing_root),
        "input_chunk_path":      str(chunk_path),
        "audit_dir":             str(args.audit_dir) if args.audit_dir else None,
        "n_basins_attempted":    len(selected_staids),
        "n_basins_success":      len(manifest_rows),
        "n_basins_failed":       len(selected_staids) - len(manifest_rows),
        "run_start_utc":         run_start.isoformat(),
        "run_end_utc":           run_end.isoformat(),
        "wall_seconds":          round((run_end - run_start).total_seconds(), 1),
    }
    prov_path = out_dir / "run_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    log.info("Wrote %s", prov_path)

    # ---- build_summary.md ----
    n_mrms_total = sum(r["n_mrms_gap_hours"] for r in manifest_rows)
    n_rtma_total = sum(r["n_rtma_gap_hours"] for r in manifest_rows)
    summary_md = (
        f"# Curated Forcing Build Summary\n\n"
        f"**Product:** {PRODUCT_NAME}\n"
        f"**Month:** {month}\n"
        f"**Basins built:** {len(manifest_rows)}\n"
        f"**Hours per basin:** {n_hours}\n"
        f"**Total MRMS gap-hours (across basins):** {n_mrms_total}\n"
        f"**Total RTMA gap-hours (across basins):** {n_rtma_total}\n"
        f"**Run start:** {run_start.isoformat()}\n"
        f"**Run end:** {run_end.isoformat()}\n"
        f"**Wall time:** {provenance['wall_seconds']} s\n"
        f"**Repo commit:** {provenance['repo_commit']}\n\n"
        f"## Per-basin summary\n\n"
        f"| STAID | MRMS gaps | RTMA gaps | Coverage |\n"
        f"|---|---|---|---|\n"
    )
    for r in manifest_rows:
        summary_md += (
            f"| {r['STAID']} | {r['n_mrms_gap_hours']} | "
            f"{r['n_rtma_gap_hours']} | {r['coverage_fraction']:.4f} |\n"
        )
    summary_md += (
        f"\n*Smoke build: {dataset_config['smoke_build']}. "
        f"Full 2,752-basin build requires Milestone 2K-F-C authorization.*\n"
    )
    summary_path = out_dir / "build_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_md)
    log.info("Wrote %s", summary_path)

    # ---- Final report ----
    log.info("=" * 60)
    log.info("BUILD COMPLETE: %d/%d basins, month=%s",
             len(manifest_rows), len(selected_staids), month)
    log.info("Output: %s", out_dir)
    log.info("=" * 60)

    if provenance["n_basins_failed"] > 0:
        log.error("%d basins failed — check log output above", provenance["n_basins_failed"])
        sys.exit(1)


if __name__ == "__main__":
    main()