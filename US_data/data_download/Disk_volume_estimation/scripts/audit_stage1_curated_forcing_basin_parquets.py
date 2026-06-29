#!/usr/bin/env python3
"""Audit the Stage 1 curated forcing product v001 per-basin Parquet files.

Checks the output of build_stage1_curated_forcing_basin_parquets.py for:
  - metadata file existence and completeness;
  - per-basin Parquet readability and hourly index integrity;
  - expected column set (curated names present, forbidden absent);
  - gap-flag consistency at known RTMA gap hours;
  - SHA-256 checksum verification.

Exits 0 if all checks pass, 1 on any failure.

Design reference: docs/stage1_curated_forcing_product_v001_design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — must match builder
# ---------------------------------------------------------------------------

_CURATED_RTMA_COLS: list[str] = [
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct", "rtma_vis_m",
    "rtma_gust_ms", "rtma_weasd_kgm2", "rtma_ceil_m",
]
_CURATED_DATA_COLS: list[str] = ["mrms_qpe_1h_mm"] + _CURATED_RTMA_COLS
_CURATED_FLAG_COLS: list[str] = ["mrms_qpe_1h_mm_gap", "rtma_gap"]
_ALL_CURATED_COLS:  list[str] = _CURATED_DATA_COLS + _CURATED_FLAG_COLS
_FORBIDDEN_COLS:    set[str]  = {
    # curated names that would indicate forbidden source variables
    "rtma_10wdir", "rtma_10wdir_deg", "rtma_orog", "rtma_orog_m",
    # source short names if they somehow leaked through
    "10wdir", "orog",
}

# Known RTMA gap timestamps for 2020-11 (both absent from S3 archive)
_KNOWN_RTMA_GAPS_2020_11 = [
    pd.Timestamp("2020-11-12 09:00:00", tz="UTC"),
    pd.Timestamp("2020-11-12 10:00:00", tz="UTC"),
]

# Hours in each month used for row-count validation
_MONTH_HOURS: dict[str, int] = {
    "2020-10": 18 * 24 + 10,   # 2020-10-14T00Z → 2020-10-31T23Z = 432 h
    "2020-11": 30 * 24,         # 720
    "2020-12": 31 * 24,         # 744
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
        "--product-dir", required=True,
        help="Path to the curated product output directory "
             "(contains manifest.json, checksums.sha256, time_series/).",
    )
    p.add_argument(
        "--month", default="2020-11",
        help="Month that was built (YYYY-MM). Used to validate row counts (default: 2020-11).",
    )
    p.add_argument(
        "--expected-basins", type=int, default=None,
        help="Expected number of basins in the product. If omitted, read from manifest.",
    )
    p.add_argument(
        "--strict-rtma-gaps", action="store_true",
        help="For 2020-11: require rtma_gap=True at exactly the two known gap timestamps "
             "and False everywhere else. Enabled by default for 2020-11.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _month_full_index(month: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{month}-01 00:00:00", tz="UTC")
    end   = (start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="h")


def _check(name: str, result: bool, detail: str = "") -> bool:
    if result:
        log.info("  PASS  %s%s", name, f"  [{detail}]" if detail else "")
    else:
        log.error("  FAIL  %s%s", name, f"  [{detail}]" if detail else "")
    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_metadata_files(product_dir: Path) -> tuple[bool, dict, list[dict]]:
    """Check that all required metadata files exist and are parseable."""
    ok = True
    config: dict = {}
    manifest: list[dict] = []

    required = ["manifest.json", "dataset_config.json",
                 "run_provenance.json", "checksums.sha256", "build_summary.md"]
    for fname in required:
        exists = (product_dir / fname).exists()
        ok &= _check(f"metadata/{fname} exists", exists)

    if (product_dir / "dataset_config.json").exists():
        try:
            with open(product_dir / "dataset_config.json") as f:
                config = json.load(f)
            ok &= _check("dataset_config.json parseable", True)
        except Exception as e:
            ok &= _check("dataset_config.json parseable", False, str(e))

    if (product_dir / "manifest.json").exists():
        try:
            with open(product_dir / "manifest.json") as f:
                manifest = json.load(f)
            ok &= _check("manifest.json parseable", True)
        except Exception as e:
            ok &= _check("manifest.json parseable", False, str(e))

    return ok, config, manifest


def check_manifest_completeness(manifest: list[dict], expected_n: int) -> bool:
    ok = True
    ok &= _check("manifest row count", len(manifest) == expected_n,
                 f"got {len(manifest)}, expected {expected_n}")
    required_keys = {
        "STAID", "n_hours_expected", "n_hours_written",
        "n_mrms_gap_hours", "n_rtma_gap_hours",
        "n_valid_combined_hours", "coverage_fraction", "sha256",
    }
    for row in manifest:
        missing = required_keys - set(row.keys())
        ok &= _check(f"manifest/{row.get('STAID','?')} required keys",
                     not missing, f"missing: {missing}" if missing else "")
    return ok


def check_checksums(product_dir: Path) -> bool:
    """Verify every SHA-256 in checksums.sha256 against the actual files."""
    ok = True
    cs_path = product_dir / "checksums.sha256"
    if not cs_path.exists():
        return _check("checksums.sha256 readable", False)

    with open(cs_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split(None, 1)
        if len(parts) != 2:
            ok &= _check(f"checksum line parseable", False, repr(line))
            continue
        expected_sha, rel_path = parts
        full_path = product_dir / rel_path
        if not full_path.exists():
            ok &= _check(f"checksum/{rel_path} file exists", False)
            continue
        actual_sha = _sha256(full_path)
        ok &= _check(
            f"checksum/{rel_path}",
            actual_sha == expected_sha,
            f"expected {expected_sha[:12]}… got {actual_sha[:12]}…" if actual_sha != expected_sha else "OK",
        )
    return ok


def check_basin_parquet(
    staid: str,
    path: Path,
    full_index: pd.DatetimeIndex,
    month: str,
    strict_rtma_gaps: bool,
) -> bool:
    """Run all per-basin Parquet checks."""
    ok = True
    n_expected = len(full_index)

    # Readability
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return _check(f"{staid} Parquet readable", False, str(e))
    ok &= _check(f"{staid} Parquet readable", True)

    # Reset index if valid_time_utc is the index
    if df.index.name == "valid_time_utc":
        df = df.reset_index()
    if "valid_time_utc" not in df.columns:
        ok &= _check(f"{staid} valid_time_utc present", False)
        return ok

    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"], utc=True)

    # Row count
    ok &= _check(f"{staid} row count", len(df) == n_expected,
                 f"got {len(df)}, expected {n_expected}")

    # No duplicate timestamps
    n_dup = df["valid_time_utc"].duplicated().sum()
    ok &= _check(f"{staid} no duplicate timestamps", n_dup == 0,
                 f"{n_dup} duplicates" if n_dup else "")

    # Complete hourly index (every expected hour present)
    df_ts_set = set(df["valid_time_utc"])
    idx_set   = set(full_index)
    missing_ts = idx_set - df_ts_set
    extra_ts   = df_ts_set - idx_set
    ok &= _check(f"{staid} complete hourly index",
                 not missing_ts and not extra_ts,
                 f"{len(missing_ts)} missing, {len(extra_ts)} extra hours")

    # Expected columns present
    for col in _ALL_CURATED_COLS:
        ok &= _check(f"{staid} column/{col} present", col in df.columns)

    # Forbidden columns absent
    forbidden_found = set(df.columns) & _FORBIDDEN_COLS
    ok &= _check(f"{staid} forbidden columns absent",
                 not forbidden_found,
                 f"found: {sorted(forbidden_found)}" if forbidden_found else "")

    # Gap-flag dtype
    for flag_col in _CURATED_FLAG_COLS:
        if flag_col in df.columns:
            ok &= _check(f"{staid} {flag_col} dtype bool",
                         df[flag_col].dtype == bool or pd.api.types.is_bool_dtype(df[flag_col]))

    # RTMA gap timestamps for 2020-11
    if strict_rtma_gaps or month == "2020-11":
        _check_rtma_gaps_2020_11(staid, df)

    # NaN consistency: at rtma_gap=True hours, ALL RTMA data cols should be NaN
    if "rtma_gap" in df.columns:
        gap_rows = df[df["rtma_gap"]]
        for col in _CURATED_RTMA_COLS:
            if col in df.columns and len(gap_rows) > 0:
                all_nan = gap_rows[col].isna().all()
                ok &= _check(f"{staid} {col} NaN at rtma_gap hours", all_nan,
                             f"{(~gap_rows[col].isna()).sum()} non-NaN values at gap hours" if not all_nan else "")

    # NaN consistency: at mrms_gap hours, mrms value col should be NaN
    if "mrms_qpe_1h_mm_gap" in df.columns and "mrms_qpe_1h_mm" in df.columns:
        mrms_gap_rows = df[df["mrms_qpe_1h_mm_gap"]]
        if len(mrms_gap_rows) > 0:
            all_nan = mrms_gap_rows["mrms_qpe_1h_mm"].isna().all()
            ok &= _check(f"{staid} mrms_qpe_1h_mm NaN at mrms_gap hours", all_nan,
                         f"{(~mrms_gap_rows['mrms_qpe_1h_mm'].isna()).sum()} non-NaN at gap hours" if not all_nan else "")

    # MRMS not falsely flagged at RTMA gap hours (for 2020-11 these are distinct)
    if month == "2020-11" and "mrms_qpe_1h_mm_gap" in df.columns:
        rtma_gap_ts = set(_KNOWN_RTMA_GAPS_2020_11)
        rtma_gap_rows = df[df["valid_time_utc"].isin(rtma_gap_ts)]
        if len(rtma_gap_rows) > 0 and "mrms_qpe_1h_mm_gap" in df.columns:
            mrms_false_flag = rtma_gap_rows["mrms_qpe_1h_mm_gap"].any()
            ok &= _check(
                f"{staid} MRMS not falsely flagged at RTMA gap hours",
                not mrms_false_flag,
                "mrms_qpe_1h_mm_gap=True at RTMA-only gap hours" if mrms_false_flag else "",
            )

    return ok


def _check_rtma_gaps_2020_11(staid: str, df: pd.DataFrame) -> bool:
    """Verify rtma_gap is True at the two known 2020-11 RTMA gap timestamps."""
    ok = True
    if "rtma_gap" not in df.columns:
        return ok

    for ts in _KNOWN_RTMA_GAPS_2020_11:
        row = df[df["valid_time_utc"] == ts]
        if len(row) == 0:
            ok &= _check(f"{staid} rtma_gap row exists at {ts}", False)
            continue
        flag_val = bool(row["rtma_gap"].iloc[0])
        ok &= _check(f"{staid} rtma_gap=True at {ts}", flag_val,
                     f"got {flag_val}" if not flag_val else "")

    # Verify no other hours have rtma_gap=True (for 2020-11)
    gap_ts_set = set(_KNOWN_RTMA_GAPS_2020_11)
    extra_gaps = df[df["rtma_gap"] & ~df["valid_time_utc"].isin(gap_ts_set)]
    ok &= _check(
        f"{staid} rtma_gap=False outside known gap hours",
        len(extra_gaps) == 0,
        f"{len(extra_gaps)} unexpected gap hours: "
        f"{sorted(extra_gaps['valid_time_utc'].tolist())[:3]}" if len(extra_gaps) else "",
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    product_dir = Path(args.product_dir)
    month       = args.month
    strict_rtma = args.strict_rtma_gaps or (month == "2020-11")

    if not product_dir.exists():
        log.error("Product directory does not exist: %s", product_dir)
        sys.exit(1)

    full_index = _month_full_index(month)
    log.info("Auditing curated product: %s", product_dir)
    log.info("Month %s: %d hourly slots", month, len(full_index))

    all_pass = True

    # ---- Metadata files ----
    log.info("--- Metadata file checks ---")
    meta_ok, config, manifest = check_metadata_files(product_dir)
    all_pass &= meta_ok

    # ---- Manifest completeness ----
    expected_n = args.expected_basins or (len(manifest) if manifest else 0)
    if manifest:
        log.info("--- Manifest completeness (%d basins) ---", expected_n)
        all_pass &= check_manifest_completeness(manifest, expected_n)

    # ---- Checksums ----
    log.info("--- Checksum verification ---")
    all_pass &= check_checksums(product_dir)

    # ---- Per-basin Parquets ----
    ts_dir = product_dir / "time_series"
    if not ts_dir.exists():
        all_pass &= _check("time_series/ directory exists", False)
    else:
        basin_paths = sorted(ts_dir.glob("*.parquet"))
        ok_count = _check(
            "at least 1 basin Parquet",
            len(basin_paths) > 0,
            f"found {len(basin_paths)}",
        )
        all_pass &= ok_count

        log.info("--- Per-basin Parquet checks (%d files) ---", len(basin_paths))
        for bp in basin_paths:
            staid = bp.stem
            log.info("Checking STAID=%s …", staid)
            basin_ok = check_basin_parquet(staid, bp, full_index, month, strict_rtma)
            all_pass &= basin_ok

    # ---- Cross-check: manifest STAIDs match files ----
    if manifest and ts_dir.exists():
        manifest_staids = {r["STAID"] for r in manifest}
        file_staids     = {p.stem for p in ts_dir.glob("*.parquet")}
        all_pass &= _check(
            "manifest STAIDs match Parquet files",
            manifest_staids == file_staids,
            f"in manifest only: {manifest_staids - file_staids}; "
            f"files only: {file_staids - manifest_staids}" if manifest_staids != file_staids else "",
        )

    # ---- Final verdict ----
    log.info("=" * 60)
    if all_pass:
        log.info("AUDIT RESULT: PASS")
    else:
        log.error("AUDIT RESULT: FAIL — see errors above")
    log.info("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()