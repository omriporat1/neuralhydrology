#!/usr/bin/env python3
"""Audit the Stage 1 curated forcing product v001 per-basin Parquet files.

Checks the output of build_stage1_curated_forcing_basin_parquets.py (single-month)
or build_stage1_curated_forcing_fullperiod.py (full period) for:
  - metadata file existence and completeness;
  - per-basin Parquet readability and hourly index integrity;
  - expected column set (curated names present, forbidden absent);
  - gap-flag consistency at known RTMA gap hours;
  - per-basin MRMS and RTMA gap counts (full-period mode);
  - SHA-256 checksum verification;
  - audit_summary.md written to product directory (full-period mode).

Exits 0 if all checks pass, 1 on any failure.

Design reference: docs/stage1_curated_forcing_product_v001_design.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants — must match builder
# ---------------------------------------------------------------------------

_CURATED_RTMA_COLS: list[str] = [
    "rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_sp_Pa",
    "rtma_10u_ms", "rtma_10v_ms", "rtma_tcc_pct", "rtma_vis_m",
    "rtma_gust_ms", "rtma_ceil_m",
]
_CURATED_DATA_COLS: list[str] = ["mrms_qpe_1h_mm"] + _CURATED_RTMA_COLS
_CURATED_FLAG_COLS: list[str] = ["mrms_qpe_1h_mm_gap", "rtma_gap"]
_ALL_CURATED_COLS:  list[str] = _CURATED_DATA_COLS + _CURATED_FLAG_COLS
_FORBIDDEN_COLS:    set[str]  = {
    # curated names that would indicate forbidden source variables
    "rtma_10wdir", "rtma_10wdir_deg", "rtma_orog", "rtma_orog_m",
    # source short names if they somehow leaked through
    "10wdir", "orog",
    # removed from v001 schema: weasd absent from all 63 months (2K-F-C-B)
    "rtma_weasd_kgm2",
}

# Known RTMA gap timestamps — same for single-month and full-period audits
_KNOWN_RTMA_GAP_TIMESTAMPS = [
    pd.Timestamp("2020-11-12 09:00:00", tz="UTC"),
    pd.Timestamp("2020-11-12 10:00:00", tz="UTC"),
]
# Backward-compat alias used by single-month 2020-11 checks
_KNOWN_RTMA_GAPS_2020_11 = _KNOWN_RTMA_GAP_TIMESTAMPS

# Hours in each month used for row-count validation (single-month mode)
_MONTH_HOURS: dict[str, int] = {
    "2020-10": 18 * 24,          # 2020-10-14T00Z → 2020-10-31T23Z = 432 h
    "2020-11": 30 * 24,          # 720
    "2020-12": 31 * 24,          # 744
}

# Full-period constants (must match builder)
_PERIOD_START           = pd.Timestamp("2020-10-14 00:00:00", tz="UTC")
_PERIOD_END             = pd.Timestamp("2025-12-31 23:00:00", tz="UTC")
_FULL_PERIOD_HOURS      = 45_720
_FULL_PERIOD_MRMS_GAPS  = 136
_FULL_PERIOD_RTMA_GAPS  = 2
PRODUCT_NAME            = "stage1_basin_hourly_forcings_v001"

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
    # Full-period mode
    p.add_argument(
        "--full-period", action="store_true",
        help="Audit a full-period product (45,720 rows/basin, 136 MRMS gaps, 2 RTMA gaps). "
             "Sets --expected-rows, --expected-mrms-gaps, --expected-rtma-gaps automatically. "
             "Writes audit_summary.md to the product directory.",
    )
    p.add_argument(
        "--expected-rows", type=int, default=None,
        help="Expected row count per basin (overrides month-based lookup). "
             "Set to 45720 for the full-period product.",
    )
    p.add_argument(
        "--expected-mrms-gaps", type=int, default=None,
        help="Expected MRMS gap-hours per basin. If set, each basin is checked exactly.",
    )
    p.add_argument(
        "--expected-rtma-gaps", type=int, default=None,
        help="Expected RTMA gap-hours per basin. If set, each basin is checked exactly.",
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
    expected_mrms_gaps: int | None = None,
    expected_rtma_gaps: int | None = None,
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

    # RTMA gap timestamps — applies to 2020-11 single-month and full-period audits
    if strict_rtma_gaps or month == "2020-11" or expected_rtma_gaps is not None:
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

    # MRMS not falsely flagged at the known RTMA-only gap hours
    if (month == "2020-11" or expected_mrms_gaps is not None) and \
            "mrms_qpe_1h_mm_gap" in df.columns:
        rtma_gap_ts   = set(_KNOWN_RTMA_GAP_TIMESTAMPS)
        rtma_gap_rows = df[df["valid_time_utc"].isin(rtma_gap_ts)]
        if len(rtma_gap_rows) > 0:
            mrms_false_flag = rtma_gap_rows["mrms_qpe_1h_mm_gap"].any()
            ok &= _check(
                f"{staid} MRMS not falsely flagged at RTMA gap hours",
                not mrms_false_flag,
                "mrms_qpe_1h_mm_gap=True at RTMA-only gap hours" if mrms_false_flag else "",
            )

    # Per-basin gap count checks (full-period mode)
    if expected_mrms_gaps is not None:
        ok &= _check_gap_count(staid, df, "mrms_qpe_1h_mm_gap",
                               expected_mrms_gaps, "MRMS")
    if expected_rtma_gaps is not None:
        ok &= _check_gap_count(staid, df, "rtma_gap",
                               expected_rtma_gaps, "RTMA")

    # No product-synchronized gaps (full-period mode)
    if expected_mrms_gaps is not None and expected_rtma_gaps is not None:
        if "mrms_qpe_1h_mm_gap" in df.columns and "rtma_gap" in df.columns:
            synced = int((df["mrms_qpe_1h_mm_gap"] & df["rtma_gap"]).sum())
            ok &= _check(f"{staid} no synchronized MRMS+RTMA gaps",
                         synced == 0, f"{synced} synchronized gap hours" if synced else "")

    # Non-null coverage checks
    # Full-period mode: exact expected counts (gaps → NaN → known non-null count)
    # Other modes: at-least-one-non-null guard against silent all-NaN mapping bugs
    if expected_mrms_gaps is not None and expected_rtma_gaps is not None:
        exp_mrms_nn = n_expected - expected_mrms_gaps
        exp_rtma_nn = n_expected - expected_rtma_gaps
        if "mrms_qpe_1h_mm" in df.columns:
            actual_nn = int(df["mrms_qpe_1h_mm"].notna().sum())
            ok &= _check(f"{staid} mrms_qpe_1h_mm non-null count",
                         actual_nn == exp_mrms_nn,
                         f"got {actual_nn}, expected {exp_mrms_nn}" if actual_nn != exp_mrms_nn else "")
        for col in _CURATED_RTMA_COLS:
            if col in df.columns:
                actual_nn = int(df[col].notna().sum())
                ok &= _check(f"{staid} {col} non-null count",
                             actual_nn == exp_rtma_nn,
                             f"got {actual_nn}, expected {exp_rtma_nn}" if actual_nn != exp_rtma_nn else "")
    else:
        for col in _CURATED_DATA_COLS:
            if col in df.columns:
                all_nan = df[col].isna().all()
                ok &= _check(f"{staid} {col} not all-NaN", not all_nan,
                             "all values NaN — likely a source mapping error" if all_nan else "")

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
# Full-period helpers
# ---------------------------------------------------------------------------

def _fullperiod_index() -> pd.DatetimeIndex:
    """Return the 45,720-hour full-period hourly UTC index."""
    return pd.date_range(_PERIOD_START, _PERIOD_END, freq="h")


def _check_gap_count(
    staid: str,
    df: pd.DataFrame,
    flag_col: str,
    expected: int,
    label: str,
) -> bool:
    actual = int(df[flag_col].sum()) if flag_col in df.columns else -1
    if actual == -1:
        return _check(f"{staid} {label} gap_count ({flag_col} present)", False,
                      "column missing")
    return _check(f"{staid} {label} gap_count",
                  actual == expected,
                  f"got {actual}, expected {expected}" if actual != expected else "")


def write_audit_summary(
    product_dir: Path,
    all_pass: bool,
    n_basins_expected: int,
    n_basins_checked: int,
    mode: str,
    expected_rows: int,
    expected_mrms_gaps: int | None,
    expected_rtma_gaps: int | None,
) -> None:
    """Write audit_summary.md to the product directory."""
    result_str = "PASS" if all_pass else "FAIL"
    lines = [
        f"# Audit Summary — {PRODUCT_NAME}\n",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}",
        f"**Product directory:** {product_dir}",
        f"**Audit mode:** {mode}",
        f"**Expected basins:** {n_basins_expected}",
        f"**Basins checked:** {n_basins_checked}",
        f"**Expected rows per basin:** {expected_rows}",
    ]
    if expected_mrms_gaps is not None:
        lines.append(f"**Expected MRMS gap-hours per basin:** {expected_mrms_gaps}")
        lines.append(
            f"**Expected MRMS gap-hours total:** {expected_mrms_gaps} × "
            f"{n_basins_checked} = {expected_mrms_gaps * n_basins_checked}"
        )
    if expected_rtma_gaps is not None:
        lines.append(f"**Expected RTMA gap-hours per basin:** {expected_rtma_gaps}")
        lines.append(
            f"**Expected RTMA gap-hours total:** {expected_rtma_gaps} × "
            f"{n_basins_checked} = {expected_rtma_gaps * n_basins_checked}"
        )
    lines += [
        f"**Known RTMA gap timestamps:** "
        f"{', '.join(str(t) for t in _KNOWN_RTMA_GAP_TIMESTAMPS)}",
        f"\n## Result: {result_str}",
        "",
        f"Audit exited with result **{result_str}**. "
        "See the accompanying build.log / smoke.log for per-check detail.",
    ]
    summary_path = product_dir / "audit_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Wrote %s", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    product_dir = Path(args.product_dir)
    if not product_dir.exists():
        log.error("Product directory does not exist: %s", product_dir)
        sys.exit(1)

    # ---- Determine audit mode ----
    full_period_mode = args.full_period
    if full_period_mode:
        month               = "full-period"
        full_index          = _fullperiod_index()
        expected_mrms_gaps  = args.expected_mrms_gaps  if args.expected_mrms_gaps  is not None \
                              else _FULL_PERIOD_MRMS_GAPS
        expected_rtma_gaps  = args.expected_rtma_gaps  if args.expected_rtma_gaps  is not None \
                              else _FULL_PERIOD_RTMA_GAPS
        if args.expected_rows is not None:
            full_index = pd.date_range(_PERIOD_START, periods=args.expected_rows, freq="h")
        strict_rtma = True
        log.info("Auditing curated product (FULL-PERIOD mode): %s", product_dir)
        log.info("Full-period index: %d hourly slots (%s → %s)",
                 len(full_index), full_index[0], full_index[-1])
    else:
        month               = args.month
        strict_rtma         = args.strict_rtma_gaps or (month == "2020-11")
        full_index          = _month_full_index(month)
        expected_mrms_gaps  = args.expected_mrms_gaps
        expected_rtma_gaps  = args.expected_rtma_gaps
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
            basin_ok = check_basin_parquet(
                staid, bp, full_index, month, strict_rtma,
                expected_mrms_gaps=expected_mrms_gaps,
                expected_rtma_gaps=expected_rtma_gaps,
            )
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

    # ---- Full-period: verify expected basin count ----
    n_basins_checked = len(list(ts_dir.glob("*.parquet"))) if ts_dir.exists() else 0
    if full_period_mode:
        expected_n = args.expected_basins or (len(manifest) if manifest else 0)
        if expected_n:
            all_pass &= _check(
                "expected basin count",
                n_basins_checked == expected_n,
                f"found {n_basins_checked}, expected {expected_n}"
                if n_basins_checked != expected_n else "",
            )

    # ---- Final verdict ----
    log.info("=" * 60)
    if all_pass:
        log.info("AUDIT RESULT: PASS")
    else:
        log.error("AUDIT RESULT: FAIL — see errors above")
    log.info("=" * 60)

    # ---- Write audit_summary.md (full-period mode or when explicitly in any mode) ----
    if full_period_mode:
        write_audit_summary(
            product_dir   = product_dir,
            all_pass      = all_pass,
            n_basins_expected = args.expected_basins or n_basins_checked,
            n_basins_checked  = n_basins_checked,
            mode          = "full-period" if full_period_mode else f"single-month/{month}",
            expected_rows = len(full_index),
            expected_mrms_gaps = expected_mrms_gaps,
            expected_rtma_gaps = expected_rtma_gaps,
        )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()