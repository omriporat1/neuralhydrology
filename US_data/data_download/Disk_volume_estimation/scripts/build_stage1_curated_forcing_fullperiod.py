#!/usr/bin/env python3
"""Build Stage 1 curated forcing product v001 — full-period per-basin Parquets.

Reads the 63 monthly combined long-format Parquets produced by the full-period
extraction and writes one wide-format Parquet per basin covering the complete
Stage 1 period (2020-10-14T00:00:00Z – 2025-12-31T23:00:00Z, 45,720 hours).

Algorithm: month-at-a-time with temporary intermediates.
  Phase 1: For each monthly source chunk, build per-basin monthly wide DataFrames
           and write them to {out_dir}/_monthly_tmp/{YYYY-MM}/{STAID}.parquet.
  Phase 2: For each basin, read its 63 monthly intermediates, concatenate into a
           45,720-row wide Parquet, write to {out_dir}/time_series/{STAID}.parquet,
           and delete the intermediates for that basin.
  Phase 3: Write manifest.json, checksums.sha256, dataset_config.json,
           run_provenance.json, build_summary.md.
  Phase 4: Remove _monthly_tmp.

Memory profile: one monthly chunk (~3.7 GB) held in memory during Phase 1;
one basin's 63-month accumulated DataFrames (~2.5 MB) held during Phase 2.

One row per UTC hour. NaN for gap hours. Boolean gap-flag columns.
Expected per-basin: 45,720 rows, 136 MRMS gap-hours, 2 RTMA gap-hours.

Design reference: docs/stage1_curated_forcing_product_v001_design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — must match smoke builder and auditor
# ---------------------------------------------------------------------------

PRODUCT_NAME   = "stage1_basin_hourly_forcings_v001"
SCHEMA_VERSION = "1.0"

_MRMS_PRODUCT_ID = "mrms_qpe_1h_pass1"
_RTMA_PRODUCT_ID = "rtma_conus_aws_2p5km"

_RTMA_SRC_TO_CURATED: dict[str, str] = {
    "2t":    "rtma_2t_K",
    "2d":    "rtma_2d_K",       # source var is "2d" (dewpoint 2m), not "d2m"
    "sh2":   "rtma_2sh_kgkg",
    "2sh":   "rtma_2sh_kgkg",
    "sp":    "rtma_sp_Pa",
    "10u":   "rtma_10u_ms",
    "10v":   "rtma_10v_ms",
    "tcc":   "rtma_tcc_pct",
    "vis":   "rtma_vis_m",
    "gust":  "rtma_gust_ms",
    "i10fg": "rtma_gust_ms",
    "ceil":  "rtma_ceil_m",
    # "weasd" removed: absent from all 63 months; no RTMA precip in source chunks
}

_CURATED_RTMA_COLS: list[str] = [
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct", "rtma_vis_m",
    "rtma_gust_ms", "rtma_ceil_m",
]

_CURATED_COLS: list[str] = (
    ["mrms_qpe_1h_mm", "mrms_qpe_1h_mm_gap"]
    + _CURATED_RTMA_COLS
    + ["rtma_gap"]
)

_FORBIDDEN_VARS: set[str] = {"10wdir", "orog"}

# Stage 1 full-period bounds
_PERIOD_START = pd.Timestamp("2020-10-14 00:00:00", tz="UTC")
_PERIOD_END   = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")

# All 63 months in the Stage 1 period
_ALL_MONTHS: list[str] = [
    str(p) for p in pd.period_range("2020-10", "2025-12", freq="M")
]

# Expected per-basin values for the full 63-month period
_EXPECTED_TOTAL_HOURS = 45_720
_EXPECTED_MRMS_GAPS   = 136
_EXPECTED_RTMA_GAPS   = 2

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
             "(contains chunks/{YYYY-MM}/combined_{YYYY-MM}.parquet).",
    )
    p.add_argument(
        "--out-dir", required=True,
        help="Output directory for the curated product. "
             "Full build: .../stage1_basin_hourly_forcings_v001/",
    )
    p.add_argument(
        "--basin-list", default=None,
        help="Path to v001 basin list CSV (column 'STAID' or 'gauge_id'). "
             "If omitted, STAIDs are derived from the first available monthly chunk.",
    )
    p.add_argument(
        "--staids", nargs="+", default=None,
        help="Explicit STAID list. Overrides --basin-list and --max-basins.",
    )
    p.add_argument(
        "--max-basins", type=int, default=None,
        help="Cap the basin list to the first N STAIDs (for bounded tests). "
             "Ignored when --staids is given.",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files (intermediates and final Parquets).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs, print the build plan, and exit without writing files.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage1_month_index(month: str) -> pd.DatetimeIndex:
    """Return the hourly UTC index for one Stage 1 month.

    2020-10 starts at 2020-10-14T00Z (not the 1st); all other months are full.
    """
    if month == "2020-10":
        start = _PERIOD_START
        end   = pd.Timestamp("2020-10-31 23:00:00", tz="UTC")
    else:
        start = pd.Timestamp(f"{month}-01 00:00:00", tz="UTC")
        end   = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="h")


def _load_chunk(path: Path, staids: list[str]) -> pd.DataFrame:
    """Load one monthly combined Parquet, filtered to selected STAIDs."""
    filters = [("STAID", "in", staids)] if staids else None
    df = pd.read_parquet(
        path,
        columns=["STAID", "product", "variable", "valid_time_utc", "weighted_mean"],
        filters=filters,
    )
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)
    return df


def _check_forbidden_vars(df: pd.DataFrame, month: str) -> None:
    rtma  = df[df["product"] == _RTMA_PRODUCT_ID]
    found = set(rtma["variable"].unique()) & _FORBIDDEN_VARS
    if found:
        log.error("FORBIDDEN RTMA variables in %s: %s", month, sorted(found))
        sys.exit(1)


def _build_basin_wide(
    staid: str,
    bdf: pd.DataFrame,
    month_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Convert long-format single-basin single-month data to wide hourly format.

    Gap hours are identified by row absence in the source (not NaN detection),
    consistent with not_in_s3 gap semantics from the extraction audit.
    Returns a DataFrame with valid_time_utc as the index.
    """
    mrms           = bdf[bdf["product"] == _MRMS_PRODUCT_ID]
    mrms_ts        = set(mrms["valid_time_utc"].unique())
    mrms_gap_flags = ~month_index.isin(mrms_ts)
    mrms_vals = (
        mrms.set_index("valid_time_utc")["weighted_mean"]
        .reindex(month_index)
        .astype("float32")
    )

    rtma           = bdf[bdf["product"] == _RTMA_PRODUCT_ID]
    rtma_ts        = set(rtma["valid_time_utc"].unique())
    rtma_gap_flags = ~month_index.isin(rtma_ts)

    wide = pd.DataFrame(index=month_index)
    wide.index.name = "valid_time_utc"
    wide["mrms_qpe_1h_mm"]     = mrms_vals.values
    wide["mrms_qpe_1h_mm_gap"] = mrms_gap_flags

    for src_var in rtma["variable"].unique():
        curated = _RTMA_SRC_TO_CURATED.get(src_var)
        if curated is None:
            log.debug("STAID=%s: unknown RTMA var %r — skipping", staid, src_var)
            continue
        if curated in wide.columns:
            continue  # alias already filled (e.g. sh2 after 2sh)
        series = (
            rtma[rtma["variable"] == src_var]
            .set_index("valid_time_utc")["weighted_mean"]
            .reindex(month_index)
            .astype("float32")
        )
        wide[curated] = series.values

    for col in _CURATED_RTMA_COLS:
        if col not in wide.columns:
            log.warning("STAID=%s: RTMA col %s absent in source — NaN-filled", staid, col)
            wide[col] = np.full(len(month_index), np.nan, dtype="float32")

    wide["rtma_gap"] = rtma_gap_flags
    return wide[_CURATED_COLS].copy()


def _basin_stats(staid: str, wide: pd.DataFrame, sha: str) -> dict:
    n_hours    = len(wide)
    n_mrms_gap = int(wide["mrms_qpe_1h_mm_gap"].sum())
    n_rtma_gap = int(wide["rtma_gap"].sum())
    n_valid    = int((~wide["mrms_qpe_1h_mm_gap"] & ~wide["rtma_gap"]).sum())
    return {
        "STAID":                  staid,
        "n_hours_expected":       n_hours,
        "n_hours_written":        n_hours,
        "n_mrms_gap_hours":       n_mrms_gap,
        "n_rtma_gap_hours":       n_rtma_gap,
        "n_valid_combined_hours": n_valid,
        "coverage_fraction":      round(n_valid / n_hours, 6),
        "file_path":              f"time_series/{staid}.parquet",
        "sha256":                 sha,
    }


# ---------------------------------------------------------------------------
# Basin list resolution
# ---------------------------------------------------------------------------

def _resolve_staids(args: argparse.Namespace, forcing_root: Path) -> list[str]:
    """Return the ordered list of STAIDs to build."""
    if args.staids:
        return list(args.staids)

    if args.basin_list:
        csv_path = Path(args.basin_list)
        if not csv_path.exists():
            log.error("Basin list CSV not found: %s", csv_path)
            sys.exit(1)
        df  = pd.read_csv(csv_path, dtype=str)
        col = "STAID" if "STAID" in df.columns else "gauge_id"
        staids = df[col].tolist()
        log.info("Basin list: %d STAIDs from %s", len(staids), csv_path.name)
    else:
        # Derive from first available monthly chunk
        staids = None
        for month in _ALL_MONTHS:
            chunk_path = forcing_root / "chunks" / month / f"combined_{month}.parquet"
            if chunk_path.exists():
                log.info("Deriving STAID list from %s", chunk_path)
                tmp = pd.read_parquet(chunk_path, columns=["STAID"])
                staids = sorted(tmp["STAID"].unique().tolist())
                log.info("Found %d unique STAIDs", len(staids))
                break
        if staids is None:
            log.error("No monthly chunk Parquets found under %s", forcing_root / "chunks")
            sys.exit(1)

    if args.max_basins is not None:
        staids = staids[: args.max_basins]
        log.info("Capped to %d basins (--max-basins %d)", len(staids), args.max_basins)

    return staids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    args         = _parse_args()
    forcing_root = Path(args.forcing_root)
    out_dir      = Path(args.out_dir)

    # ---- Discover available monthly chunks ----
    available_months: list[str] = []
    missing_months:   list[str] = []
    for month in _ALL_MONTHS:
        p = forcing_root / "chunks" / month / f"combined_{month}.parquet"
        if p.exists():
            available_months.append(month)
        else:
            missing_months.append(month)

    if missing_months:
        log.warning("%d / 63 monthly chunks not found: %s%s",
                    len(missing_months), missing_months[:5],
                    " …" if len(missing_months) > 5 else "")
    if not available_months:
        log.error("No monthly chunk Parquets found. Check --forcing-root.")
        sys.exit(1)
    log.info("Monthly chunks available: %d / 63", len(available_months))

    # ---- Resolve STAIDs ----
    selected_staids = _resolve_staids(args, forcing_root)
    n_basins        = len(selected_staids)

    # ---- Compute expected rows per basin ----
    expected_rows = sum(len(_stage1_month_index(m)) for m in available_months)
    is_full_build = (len(available_months) == 63 and args.max_basins is None
                     and args.staids is None and args.basin_list is None)
    if expected_rows != _EXPECTED_TOTAL_HOURS and len(available_months) == 63:
        log.error("Full-period index mismatch: got %d hours, expected %d",
                  expected_rows, _EXPECTED_TOTAL_HOURS)
        sys.exit(1)

    # ---- Dry-run ----
    if args.dry_run:
        actual_start = _stage1_month_index(available_months[0])[0]
        actual_end   = _stage1_month_index(available_months[-1])[-1]
        log.info("[DRY-RUN] Build plan:")
        log.info("  Forcing root:     %s", forcing_root)
        log.info("  Output dir:       %s", out_dir)
        log.info("  Months:           %d (%s → %s)",
                 len(available_months), available_months[0], available_months[-1])
        log.info("  Period:           %s → %s", actual_start.isoformat(), actual_end.isoformat())
        log.info("  Basins:           %d", n_basins)
        log.info("  Rows per basin:   %d", expected_rows)
        log.info("  Total rows:       %d", expected_rows * n_basins)
        log.info("  Phase-1 writes:   %d intermediate Parquets", n_basins * len(available_months))
        log.info("  Phase-2 writes:   %d final Parquets", n_basins)
        log.info("  Full build:       %s", is_full_build)
        if missing_months:
            log.warning("  Missing months:   %s", missing_months)
        log.info("[DRY-RUN] No files written.")
        return

    # ---- Create output layout ----
    ts_dir  = out_dir / "time_series"
    tmp_dir = out_dir / "_monthly_tmp"
    ts_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now(timezone.utc)

    # ====================================================================
    # Phase 1: Month-at-a-time — write per-basin monthly intermediates
    # ====================================================================
    log.info("=== Phase 1: Processing %d monthly chunks (%d basins each) ===",
             len(available_months), n_basins)

    for m_idx, month in enumerate(available_months, 1):
        chunk_path  = forcing_root / "chunks" / month / f"combined_{month}.parquet"
        month_index = _stage1_month_index(month)
        month_tmp   = tmp_dir / month
        month_tmp.mkdir(exist_ok=True)

        # Check if all intermediates already exist (skip whole month on resume)
        all_exist = all(
            (month_tmp / f"{s}.parquet").exists() for s in selected_staids
        )
        if all_exist and not args.overwrite:
            log.info("[%d/%d] %s — all %d intermediates exist, skipping",
                     m_idx, len(available_months), month, n_basins)
            continue

        log.info("[%d/%d] %s (%d h) — loading chunk …",
                 m_idx, len(available_months), month, len(month_index))
        chunk_df = _load_chunk(chunk_path, selected_staids)
        _check_forbidden_vars(chunk_df, month)
        log.info("[%d/%d] %s — %d rows loaded, building basin intermediates …",
                 m_idx, len(available_months), month, len(chunk_df))

        n_written = 0
        for staid in selected_staids:
            out_tmp = month_tmp / f"{staid}.parquet"
            if out_tmp.exists() and not args.overwrite:
                continue
            bdf  = chunk_df[chunk_df["STAID"] == staid]
            wide = _build_basin_wide(staid, bdf, month_index)
            wide.to_parquet(out_tmp, index=True, engine="pyarrow", compression="snappy")
            n_written += 1

        del chunk_df
        log.info("[%d/%d] %s done — %d intermediates written",
                 m_idx, len(available_months), month, n_written)

    # ====================================================================
    # Phase 2: Per-basin concatenation → final Parquets
    # ====================================================================
    log.info("=== Phase 2: Concatenating %d months → %d final Parquets ===",
             len(available_months), n_basins)

    manifest_rows:  list[dict] = []
    checksum_lines: list[str]  = []
    n_failed = 0

    for s_idx, staid in enumerate(selected_staids, 1):
        out_path = ts_dir / f"{staid}.parquet"

        if out_path.exists() and not args.overwrite:
            log.info("[%d/%d] STAID=%s: exists, skipping (use --overwrite to redo)",
                     s_idx, n_basins, staid)
            sha  = _sha256(out_path)
            wide = pd.read_parquet(out_path)
            manifest_rows.append(_basin_stats(staid, wide, sha))
            checksum_lines.append(f"{sha}  time_series/{staid}.parquet")
            continue

        # Read all monthly intermediates for this basin
        monthly_dfs: list[pd.DataFrame] = []
        missing_tmp: list[str] = []
        for month in available_months:
            tmp_path = tmp_dir / month / f"{staid}.parquet"
            if not tmp_path.exists():
                missing_tmp.append(month)
                continue
            monthly_dfs.append(pd.read_parquet(tmp_path))

        if missing_tmp:
            log.error("STAID=%s: missing intermediates for months: %s", staid, missing_tmp)
            n_failed += 1
            continue

        wide = pd.concat(monthly_dfs)
        del monthly_dfs

        if len(wide) != expected_rows:
            log.error("STAID=%s: expected %d rows, got %d", staid, expected_rows, len(wide))
            n_failed += 1
            continue

        wide.to_parquet(out_path, index=True, engine="pyarrow", compression="snappy")
        sha   = _sha256(out_path)
        stats = _basin_stats(staid, wide, sha)
        del wide
        if s_idx % 100 == 0 or s_idx == n_basins:
            log.info("[%d/%d] STAID=%s: %d rows, mrms_gap=%d, rtma_gap=%d, cov=%.4f",
                     s_idx, n_basins, staid,
                     stats["n_hours_written"], stats["n_mrms_gap_hours"],
                     stats["n_rtma_gap_hours"], stats["coverage_fraction"])
        manifest_rows.append(stats)
        checksum_lines.append(f"{sha}  time_series/{staid}.parquet")

        # Delete this basin's monthly intermediates immediately to free disk space
        for month in available_months:
            tmp_file = tmp_dir / month / f"{staid}.parquet"
            if tmp_file.exists():
                tmp_file.unlink()

    run_end = datetime.now(timezone.utc)

    if n_failed > 0:
        log.error("Phase 2: %d basins failed — aborting metadata write", n_failed)
        sys.exit(1)

    # ====================================================================
    # Phase 3: Metadata files
    # ====================================================================
    log.info("=== Phase 3: Writing metadata ===")

    actual_start = _stage1_month_index(available_months[0])[0]
    actual_end   = _stage1_month_index(available_months[-1])[-1]

    # manifest.json
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest_rows, f, indent=2)
    log.info("Wrote manifest.json (%d basins)", len(manifest_rows))

    # checksums.sha256
    with open(out_dir / "checksums.sha256", "w") as f:
        f.write("\n".join(checksum_lines) + "\n")
    log.info("Wrote checksums.sha256")

    # dataset_config.json
    smoke_build = len(manifest_rows) < 2752 or len(available_months) < 63
    config = {
        "product_name":       PRODUCT_NAME,
        "schema_version":     SCHEMA_VERSION,
        "period_start_utc":   actual_start.isoformat(),
        "period_end_utc":     actual_end.isoformat(),
        "n_hours_expected":   expected_rows,
        "n_basins":           len(manifest_rows),
        "months_included":    available_months,
        "mrms_product":       _MRMS_PRODUCT_ID,
        "rtma_product":       _RTMA_PRODUCT_ID,
        "variables":          _CURATED_RTMA_COLS + ["mrms_qpe_1h_mm"],
        "gap_flag_columns":   ["mrms_qpe_1h_mm_gap", "rtma_gap"],
        "excluded_variables": sorted(_FORBIDDEN_VARS),
        "gap_policy":         "raw_preserve_nan_no_interpolation",
        "smoke_build":        smoke_build,
    }
    with open(out_dir / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)
    log.info("Wrote dataset_config.json")

    # run_provenance.json
    wall_s = round((run_end - run_start).total_seconds(), 1)
    prov = {
        "builder_script":     "scripts/build_stage1_curated_forcing_fullperiod.py",
        "repo_commit":        _git_commit(),
        "python_version":     sys.version,
        "platform":           platform.node(),
        "input_chunk_root":   str(forcing_root),
        "n_months":           len(available_months),
        "months_processed":   available_months,
        "n_basins_attempted": n_basins,
        "n_basins_success":   len(manifest_rows),
        "n_basins_failed":    n_failed,
        "run_start_utc":      run_start.isoformat(),
        "run_end_utc":        run_end.isoformat(),
        "wall_seconds":       wall_s,
    }
    with open(out_dir / "run_provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    log.info("Wrote run_provenance.json")

    # build_summary.md
    n_mrms_total = sum(r["n_mrms_gap_hours"] for r in manifest_rows)
    n_rtma_total = sum(r["n_rtma_gap_hours"] for r in manifest_rows)
    n_shown      = min(20, len(manifest_rows))
    rows_table   = "\n".join(
        f"| {r['STAID']} | {r['n_mrms_gap_hours']} | "
        f"{r['n_rtma_gap_hours']} | {r['coverage_fraction']:.6f} |"
        for r in manifest_rows[:n_shown]
    )
    if len(manifest_rows) > n_shown:
        rows_table += f"\n| … ({len(manifest_rows) - n_shown} more) | | | |"

    expected_mrms_total = _EXPECTED_MRMS_GAPS * len(manifest_rows) if len(available_months) == 63 else "N/A"
    expected_rtma_total = _EXPECTED_RTMA_GAPS * len(manifest_rows) if len(available_months) == 63 else "N/A"

    summary = (
        f"# Curated Forcing Build Summary — Full Period\n\n"
        f"**Product:** {PRODUCT_NAME}\n"
        f"**Period:** {actual_start.isoformat()} – {actual_end.isoformat()}\n"
        f"**Months processed:** {len(available_months)} / 63\n"
        f"**Basins built:** {len(manifest_rows)}\n"
        f"**Rows per basin:** {expected_rows}\n"
        f"**Full build:** {not smoke_build}\n"
        f"**Total MRMS gap-hours (across basins):** {n_mrms_total} "
        f"(expected {expected_mrms_total})\n"
        f"**Total RTMA gap-hours (across basins):** {n_rtma_total} "
        f"(expected {expected_rtma_total})\n"
        f"**Run start:** {run_start.isoformat()}\n"
        f"**Run end:** {run_end.isoformat()}\n"
        f"**Wall time:** {wall_s} s ({wall_s / 3600:.2f} h)\n"
        f"**Repo commit:** {prov['repo_commit']}\n\n"
        f"## Per-basin coverage (first {n_shown})\n\n"
        f"| STAID | MRMS gaps | RTMA gaps | Coverage |\n"
        f"|---|---|---|---|\n"
        f"{rows_table}\n"
    )
    with open(out_dir / "build_summary.md", "w") as f:
        f.write(summary)
    log.info("Wrote build_summary.md")

    # ====================================================================
    # Phase 4: Cleanup _monthly_tmp
    # ====================================================================
    log.info("=== Phase 4: Removing temporary intermediates ===")
    orphaned = 0
    for month in available_months:
        month_tmp = tmp_dir / month
        if month_tmp.exists():
            remaining = list(month_tmp.iterdir())
            if remaining:
                log.warning("tmp/%s: %d orphaned files after cleanup", month, len(remaining))
                orphaned += len(remaining)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        log.info("Removed %s", tmp_dir)
    if orphaned:
        log.warning("%d orphaned intermediate files were deleted with _monthly_tmp", orphaned)

    # ---- Final report ----
    log.info("=" * 60)
    log.info("BUILD COMPLETE: %d/%d basins, %d months, %.1f s (%.2f h)",
             len(manifest_rows), n_basins, len(available_months), wall_s, wall_s / 3600)
    log.info("Output: %s", out_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()