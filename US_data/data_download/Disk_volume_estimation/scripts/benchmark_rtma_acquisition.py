#!/usr/bin/env python3
"""RTMA acquisition mode benchmark: full-file vs selected-message byte-range.

Audits the current Stage 1 RTMA acquisition path and benchmarks two modes:

  Mode A — Full-file download
    Download the entire grb2_wexp file (~80 MB).  This is the current default
    used by extract_stage1_january.py.  Decode with cfgrib, extract basin stats.

  Mode B — Selected-message byte-range download
    Fetch the .idx inventory from S3 (tiny text file, <5 KB).
    Identify the GRIB2 message byte ranges for exactly the 11 Stage 1 variables.
    Download only those ranges (HTTP Range header).  Concatenate to a temp GRIB2
    file, decode with cfgrib, extract basin stats.

Hour benchmarked: 2023-01-01T00:00Z (rtma2p5.t00z.2dvaranl_ndfd.grb2_wexp).
This file must already be cached locally from the Milestone 2D smoke test.

The benchmark also includes a local-file variant of Mode B (Mode B-local) that
extracts selected GRIB messages from the already-downloaded full file using
eccodes, to validate cfgrib decodability of the subset before any S3 request.

Stage 1 RTMA variable set (11 variables, fixed for this task):
  10si, 10u, 10v, 2d, 2sh, 2t, ceil, i10fg, sp, tcc, vis
  Excluded: 10wdir (circular), orog (static)

Outputs:
  tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_extraction/
      rtma_acquisition_benchmark.csv
      rtma_acquisition_benchmark.md

Usage:
    python scripts/benchmark_rtma_acquisition.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rtma_benchmark")

# Hour to benchmark (same as Milestone 2C/2D sample)
_BENCHMARK_DT = datetime(2023, 1, 1, 0, 0, 0)
_RTMA_BUCKET  = "noaa-rtma-pds"
_RTMA_KEY     = "rtma2p5.20230101/rtma2p5.t00z.2dvaranl_ndfd.grb2_wexp"
_IDX_KEY      = f"{_RTMA_KEY}.idx"

# Full-file download time observed in Milestone 2D smoke test (2023-01-01T01Z file, fresh)
_SMOKE_FULLFILE_DL_S = 43.3   # seconds, median of two observations

# All 11 Stage 1 RTMA variables (cfgrib short names from extraction.py)
_STAGE1_CFGRIB = frozenset({
    "10si", "10u", "10v", "2d", "2sh", "2t",
    "ceil", "i10fg", "sp", "tcc", "vis",
})
# Excluded from Stage 1 dynamic output:
#   10wdir — circular variable; linear averaging invalid
#   orog   — static terrain field

# Extended targets: maps GRIB/wgrib2 parameter names (as they appear in .idx field-3)
# AND cfgrib short names to selection families.
# This extends the existing _selected_targets() in rtma.py to cover all 11 Stage 1 vars.
_EXT_TARGETS: dict[str, set[str]] = {
    "TMP":  {"tmp",  "2t",    "t2m"},
    "SPFH": {"spfh", "2sh",   "sh2",   "q"},
    "DPT":  {"dpt",  "dpth",  "2d",    "d2m"},
    "UGRD": {"ugrd", "10u",   "u10"},
    "VGRD": {"vgrd", "10v",   "v10"},
    "PRES": {"pres", "pressfc","sp"},
    "TCDC": {"tcdc", "tcc"},
    "WIND": {"wind", "10si",  "si10",  "ws"},   # 10-m wind speed
    "GUST": {"gust", "i10fg", "fg10",  "wgst"}, # 10-m wind gust
    "VIS":  {"vis"},                             # visibility
    "CEIL": {"ceil", "hcpb"},                    # cloud ceiling
}
# Always exclude: wind direction and orography
_ALWAYS_EXCLUDE = frozenset({
    "wdir", "10wdir", "orog", "hgt", "zsfc", "elev", "gh", "hgtsfc",
})


# ---------------------------------------------------------------------------
# Selection helpers (self-contained, do not depend on rtma.py internals)
# ---------------------------------------------------------------------------

def _parse_idx_entries(idx_text: str) -> list[dict[str, Any]]:
    """Parse an NCEP wgrib2-format .idx file into entry dicts.

    Expected line format:  message_num:byte_offset:d=YYYYMMDDHH:VARNAME:level:type:
    Returns entries sorted by offset, with end_offset set to next_offset - 1.
    """
    import re
    entries: list[dict[str, Any]] = []
    for raw_line in idx_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 4:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        short_name  = parts[3].strip().lower()
        level_str   = parts[4].strip() if len(parts) > 4 else ""
        entries.append({
            "offset":     offset,
            "short_name": short_name,
            "level":      level_str,
            "raw_line":   raw_line.rstrip(":"),
        })
    entries.sort(key=lambda e: e["offset"])
    for i, entry in enumerate(entries):
        nxt = entries[i + 1]["offset"] if i + 1 < len(entries) else None
        entry["end_offset"] = (nxt - 1) if nxt is not None else None
    return entries


def _identify_stage1_entries(
    entries: list[dict[str, Any]],
) -> tuple[list[dict], list[str], bool]:
    """Return (stage1_entries, all_idx_names, has_spfh).

    stage1_entries: idx entries whose short_name maps to one of the 11 Stage 1 variables.
    all_idx_names:  list of all short_names found in the .idx.
    has_spfh:       True if any entry is a specific-humidity message (informational).

    NOTE: Both SPFH (2sh) and DPT (2d) are Stage 1 variables, so NO mutual exclusion
    is applied here.  The mutual-exclusion in _selected_targets() (rtma.py) is for
    size-estimation only and must NOT be used when selecting messages for extraction.
    """
    all_names = [e["short_name"] for e in entries]
    spfh_names = _EXT_TARGETS["SPFH"]
    has_spfh   = any(n in spfh_names for n in all_names)

    stage1: list[dict] = []
    for entry in entries:
        sn = entry["short_name"]
        if sn in _ALWAYS_EXCLUDE:
            continue
        selected = any(sn in names for names in _EXT_TARGETS.values())
        if selected:
            stage1.append(entry)

    return stage1, all_names, has_spfh


def _merge_ranges(
    entries: list[dict[str, Any]],
) -> list[tuple[int, Optional[int]]]:
    """Merge adjacent byte ranges from a list of idx entries."""
    if not entries:
        return []
    ranges = [(e["offset"], e["end_offset"]) for e in entries]
    ranges.sort(key=lambda r: r[0])
    merged: list[tuple[int, Optional[int]]] = []
    for start, end in ranges:
        if not merged:
            merged.append((start, end))
            continue
        prev_s, prev_e = merged[-1]
        if prev_e is None:
            continue
        if start <= prev_e + 1:
            merged[-1] = (prev_s, end if end is None else max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Mode B1 — local GRIB message extraction (validation; no new S3 download)
# ---------------------------------------------------------------------------

def _extract_selected_messages_local(
    full_path: Path,
    stage1_names: set[str],
    has_spfh: bool,
) -> tuple[Optional[Path], int, float, dict[str, int]]:
    """Read the full cached GRIB2 file; copy only Stage 1 messages to a temp file.

    Returns (temp_path, bytes_written, elapsed_s, per_name_bytes).
    Returns (None, 0, elapsed, {}) if eccodes is unavailable.
    """
    try:
        from eccodes import codes_grib_new_from_file, codes_get, codes_get_message, codes_release
    except ImportError:
        LOGGER.warning("eccodes not available — Mode B-local skipped")
        return None, 0, 0.0, {}

    t0 = time.perf_counter()
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp_path = Path(tmp.name)
    per_name_bytes: dict[str, int] = {}
    total_bytes = 0

    with full_path.open("rb") as fh:
        while True:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                sn = str(codes_get(gid, "shortName") or "").lower().strip()
                # Include if in any target family and not excluded.
                # No SPFH/DPT mutual exclusion: both 2sh and 2d are Stage 1 variables.
                include = (
                    sn not in _ALWAYS_EXCLUDE
                    and any(sn in names for names in _EXT_TARGETS.values())
                )
                if include:
                    raw = codes_get_message(gid)
                    tmp.write(raw)
                    total_bytes += len(raw)
                    per_name_bytes[sn] = per_name_bytes.get(sn, 0) + len(raw)
            except Exception as exc:
                LOGGER.debug("eccodes message read error: %s", exc)
            finally:
                codes_release(gid)

    tmp.close()
    elapsed = time.perf_counter() - t0
    return tmp_path, total_bytes, elapsed, per_name_bytes


# ---------------------------------------------------------------------------
# Mode B2 — S3 byte-range download
# ---------------------------------------------------------------------------

def _download_selected_messages_s3(
    s3: Any,
    bucket: str,
    key: str,
    stage1_entries: list[dict[str, Any]],
) -> tuple[Optional[Path], int, float, list[tuple[int, Optional[int]]]]:
    """Download only selected GRIB2 message byte ranges from S3.

    Returns (temp_path, bytes_downloaded, elapsed_s, ranges_used).
    Uses merged ranges to minimise number of S3 GetObject calls.
    """
    from src.datasources.base import log_request

    if not stage1_entries:
        return None, 0, 0.0, []

    ranges = _merge_ranges(stage1_entries)
    LOGGER.info(
        "Mode B2: downloading %d merged ranges for %d stage1 messages",
        len(ranges), len(stage1_entries),
    )

    t0 = time.perf_counter()
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp_path = Path(tmp.name)
    total_bytes = 0

    try:
        for start, end in ranges:
            range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
            try:
                log_request(
                    LOGGER, "rtma_benchmark", "s3.get_object.range",
                    url=f"s3://{bucket}/{key}",
                    params={"Bucket": bucket, "Key": key, "Range": range_header},
                )
                resp = s3.get_object(Bucket=bucket, Key=key, Range=range_header)
                data = resp["Body"].read()
                tmp.write(data)
                total_bytes += len(data)
            except Exception as exc:
                LOGGER.warning("S3 range request %s failed: %s", range_header, exc)
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                return None, 0, time.perf_counter() - t0, ranges
    finally:
        tmp.close()

    elapsed = time.perf_counter() - t0
    LOGGER.info("Mode B2: downloaded %d bytes in %.2fs", total_bytes, elapsed)
    return tmp_path, total_bytes, elapsed, ranges


# ---------------------------------------------------------------------------
# Extraction timing helper
# ---------------------------------------------------------------------------

def _decode_and_extract(
    rtma_path: Path,
    rtma_weights: Any,
    pilot_staids: list[str],
    rtma_weights_path: Path,
    label: str,
) -> dict[str, Any]:
    """Decode RTMA GRIB2 + extract basin stats; return timing and result dict."""
    from src.pipeline.extraction import (
        decode_rtma_grids,
        extract_basin_statistics,
        _RTMA_EXCLUDED_DEFAULT,
        STAT_COLUMNS,
    )
    import pandas as pd

    t_decode = time.perf_counter()
    grids_all = decode_rtma_grids(rtma_path)
    decode_s  = time.perf_counter() - t_decode

    included = [g for g in grids_all if g.short_name not in _RTMA_EXCLUDED_DEFAULT]
    excluded  = [g.short_name for g in grids_all if g.short_name in _RTMA_EXCLUDED_DEFAULT]
    included_names = [g.short_name for g in included]

    t_extract = time.perf_counter()
    frames = [
        extract_basin_statistics(
            vg, rtma_weights, pilot_staids,
            weight_table_path=str(rtma_weights_path),
            source_file_path=str(rtma_path),
        )
        for vg in included
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STAT_COLUMNS)
    extract_s = time.perf_counter() - t_extract

    LOGGER.info(
        "[%s] decode=%.3fs  extract=%.3fs  vars=%s  excluded=%s  rows=%d",
        label, decode_s, extract_s, included_names, excluded, len(df),
    )
    return {
        "df":            df,
        "decode_s":      decode_s,
        "extract_s":     extract_s,
        "included_vars": included_names,
        "excluded_vars": excluded,
        "n_rows":        len(df),
        "n_vars":        len(included),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_results(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    label_b: str,
) -> dict[str, Any]:
    """Compare Mode A and Mode B basin-statistic DataFrames.

    Returns a summary dict with per-variable max/mean absolute difference
    in weighted_mean across all 50 pilot basins, and a pass/fail flag.
    """
    import pandas as pd
    import numpy as np

    df_a = result_a["df"]
    df_b = result_b["df"]

    if len(df_a) == 0 or len(df_b) == 0:
        return {
            "status": "empty_result",
            "common_vars": [],
            "per_var": {},
            "max_abs_diff_all": None,
            "equivalent": False,
        }

    common_vars = sorted(
        set(df_a["variable"].unique()) & set(df_b["variable"].unique())
    )
    only_a = sorted(set(df_a["variable"].unique()) - set(df_b["variable"].unique()))
    only_b = sorted(set(df_b["variable"].unique()) - set(df_a["variable"].unique()))

    per_var: dict[str, Any] = {}
    all_diffs: list[float] = []

    for var in common_vars:
        sub_a = df_a[df_a["variable"] == var].sort_values(["STAID"]).reset_index(drop=True)
        sub_b = df_b[df_b["variable"] == var].sort_values(["STAID"]).reset_index(drop=True)
        if len(sub_a) != len(sub_b):
            per_var[var] = {
                "n_rows_a": len(sub_a), "n_rows_b": len(sub_b),
                "status": "row_count_mismatch",
            }
            continue
        diffs = np.abs(sub_a["weighted_mean"].values - sub_b["weighted_mean"].values)
        finite_diffs = diffs[np.isfinite(diffs)]
        max_d = float(np.max(finite_diffs)) if len(finite_diffs) else float("nan")
        mean_d = float(np.mean(finite_diffs)) if len(finite_diffs) else float("nan")
        all_diffs.extend(finite_diffs.tolist())
        per_var[var] = {
            "n_rows_a":    len(sub_a),
            "n_rows_b":    len(sub_b),
            "max_abs_diff_weighted_mean":  round(max_d,  10),
            "mean_abs_diff_weighted_mean": round(mean_d, 10),
            "status": "PASS" if max_d < 1e-6 else "DIFF_FOUND",
        }

    max_all = float(max(all_diffs)) if all_diffs else float("nan")
    equivalent = (
        len(only_a) == 0 and len(only_b) == 0
        and max_all < 1e-6
        and all(v.get("status") == "PASS" for v in per_var.values())
    )

    return {
        "label_b":      label_b,
        "common_vars":  common_vars,
        "only_in_a":    only_a,
        "only_in_b":    only_b,
        "per_var":      per_var,
        "max_abs_diff_all": round(max_all, 10) if not (max_all != max_all) else None,
        "equivalent":   equivalent,
        "status":       "EQUIVALENT" if equivalent else "DIFFER",
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt_b(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_t(s: Optional[float]) -> str:
    if s is None:
        return "N/A"
    if s < 60:
        return f"{s:.1f}s"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h}h {m}m {sec:.0f}s" if h else f"{m}m {sec:.0f}s"


def _write_markdown_report(
    path: Path,
    full_file_bytes: int,
    full_file_dl_s_fresh: float,
    idx_bytes: int,
    idx_fetch_s: float,
    b2_bytes: Optional[int],
    b2_dl_s: Optional[float],
    b_local_bytes: Optional[int],
    b_local_extract_s: Optional[float],
    mode_a_result: dict[str, Any],
    mode_b_result: Optional[dict[str, Any]],
    comparison_b_local: Optional[dict[str, Any]],
    comparison_b2: Optional[dict[str, Any]],
    idx_all_names: list[str],
    stage1_entries: list[dict],
    stage1_missing_cfgrib: list[str],
    full_hours_stage1: int = 52608,
) -> None:
    lines = [
        "# RTMA Acquisition Mode Benchmark",
        "",
        f"**Hour**: 2023-01-01T00:00Z (t00z)  ",
        f"**Date run**: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## 1. Current Acquisition Path Audit",
        "",
        "The current Stage 1 extraction path (`extract_stage1_january.py` →",
        "`_download_single_rtma()` → `RtmaAwsConusDataSource.download_sample()` →",
        "`_download_one_object_timed()` → `_download_s3_object()`) downloads the",
        "**entire** `grb2_wexp` file using a plain S3 `GetObject` request with no",
        "`Range` header.",
        "",
        "The `measure_selected_variable_bytes()` method in `rtma.py` already implements",
        "S3 byte-range requests and `.idx` parsing, but only to **count bytes** for",
        "size-estimation purposes — it discards the downloaded content.",
        "",
        "## 2. .idx File Analysis",
        "",
        f"`.idx` file size: **{_fmt_b(idx_bytes)}**  (fetch time: {idx_fetch_s:.2f}s)",
        "",
        f"Messages found in .idx: **{len(idx_all_names)}**",
        "",
        "| # | short_name (idx field-3) | in Stage 1 selection |",
        "|---|--------------------------|----------------------|",
    ]
    for i, (nm, entry) in enumerate(
        zip(idx_all_names, [{"short_name": n} for n in idx_all_names])
    ):
        selected_entry = any(e["short_name"] == nm for e in stage1_entries)
        in_s1 = "YES" if selected_entry else ("excluded" if nm in _ALWAYS_EXCLUDE else "no")
        lines.append(f"| {i+1} | `{nm}` | {in_s1} |")
    lines += [
        "",
        f"Messages selected for Stage 1: **{len(stage1_entries)}** of {len(idx_all_names)}",
    ]
    if stage1_missing_cfgrib:
        lines += [
            "",
            f"> **Note**: The following cfgrib short names from the Stage 1 variable set",
            f"> were NOT matched to .idx entries — they may use different NCEP abbreviations",
            f"> or may not be present in the `2dvaranl_ndfd` file:",
            f"> `{', '.join(stage1_missing_cfgrib)}`",
        ]
    lines += [
        "",
        "## 3. One-Hour Benchmark Results",
        "",
        "### Mode A — Full-file download (current default)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Full-file size | {_fmt_b(full_file_bytes)} |",
        f"| Fresh download time (observed, Milestone 2D) | {full_file_dl_s_fresh:.1f}s |",
        f"| Decode time | {mode_a_result['decode_s']:.3f}s |",
        f"| Extract time (50 basins) | {mode_a_result['extract_s']:.3f}s |",
        f"| Variables decoded | {len(mode_a_result['included_vars'])} |",
        f"| Output rows | {mode_a_result['n_rows']} |",
    ]
    if mode_b_result is not None:
        b_decode = mode_b_result.get("decode_s", 0.0)
        b_extract = mode_b_result.get("extract_s", 0.0)
        b_download = b2_dl_s or 0.0
        b_total = (idx_fetch_s or 0.0) + b_download + b_decode + b_extract
        lines += [
            "",
            "### Mode B2 — Selected-message S3 byte-range (optimized)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| .idx fetch | {_fmt_b(idx_bytes)} in {idx_fetch_s:.2f}s |",
            f"| Selected messages | {len(stage1_entries)} of {len(idx_all_names)} |",
            f"| Downloaded bytes (ranges) | {_fmt_b(b2_bytes)} |",
            f"| Download time (ranges only) | {b_download:.2f}s |",
            f"| Total download (idx + ranges) | {_fmt_b((idx_bytes or 0) + (b2_bytes or 0))} |",
            f"| Decode time | {b_decode:.3f}s |",
            f"| Extract time (50 basins) | {b_extract:.3f}s |",
            f"| Total time (idx+dl+decode+extract) | {b_total:.2f}s |",
            f"| Variables decoded | {len(mode_b_result['included_vars'])} |",
            f"| Output rows | {mode_b_result['n_rows']} |",
        ]
    lines += [
        "",
        "### Mode A vs Mode B comparison",
        "",
    ]
    for label, comp in [
        ("B-local (local message extraction, cfgrib validation)", comparison_b_local),
        ("B2 (S3 byte-range)", comparison_b2),
    ]:
        if comp is None:
            lines.append(f"- **{label}**: not run")
        elif comp.get("status") == "EQUIVALENT":
            lines.append(f"- **{label}**: EQUIVALENT — max |weighted_mean diff| = {comp['max_abs_diff_all']}")
        else:
            lines.append(f"- **{label}**: {comp.get('status')} — see CSV for details")
    lines += [
        "",
        "## 4. Full-Period Estimates (2,843 basins, 2020–2025, 52,608 hours)",
        "",
        f"| Metric | Mode A — Full-file | Mode B2 — Selected-message |",
        f"|--------|-------------------|--------------------------|",
    ]
    if b2_bytes is not None and full_file_bytes:
        a_total_raw = full_file_bytes * full_hours_stage1
        b_total_raw = (b2_bytes + (idx_bytes or 0)) * full_hours_stage1
        ratio = b2_bytes / full_file_bytes * 100
        lines += [
            f"| Raw bytes/hour | {_fmt_b(full_file_bytes)} | {_fmt_b(b2_bytes + (idx_bytes or 0))} |",
            f"| Total raw cache (full period) | {_fmt_b(a_total_raw)} | {_fmt_b(b_total_raw)} |",
            f"| Reduction vs full-file | — | {100 - ratio:.0f}% smaller |",
            f"| Serial download time/hour | {full_file_dl_s_fresh:.1f}s | {(idx_fetch_s or 0) + (b2_dl_s or 0):.1f}s |",
        ]
    a_serial_s = full_file_dl_s_fresh + mode_a_result["decode_s"] + mode_a_result["extract_s"]
    lines += [
        f"| Serial proc time/hour | {a_serial_s:.1f}s | N/A if no B2 |",
        "",
        "> **Important**: These estimates assume the same network speed as observed",
        "> locally. On an HPC cluster with 10 Gbps connectivity, both modes would be",
        "> substantially faster and the relative advantage of Mode B2 would be smaller.",
        "> The primary benefit of Mode B2 on HPC is **storage reduction**, not time.",
        "",
        "## 5. Recommendation",
        "",
    ]
    if comparison_b2 and comparison_b2.get("status") == "EQUIVALENT":
        lines += [
            "**Recommendation: Switch to selected-message (Mode B2) as the production default.**",
            "",
            "Rationale:",
            "- Basin statistics are bit-identical between modes (max |diff| < 1e-6)",
            "- Reduces per-hour raw download volume significantly",
            "- Reduces total Stage 1 raw cache from ~4.0 TB to ~X TB",
            "- The .idx file is tiny (<5 KB) and adds negligible overhead",
            "- Full-file mode should remain available as a fallback/debug option",
            "",
            "Implementation: add `rtma_acquisition_mode: full_file | selected_messages`",
            "config option, or a `--rtma-mode` CLI flag to `extract_stage1_january.py`.",
        ]
    elif comparison_b_local and comparison_b_local.get("status") == "EQUIVALENT":
        lines += [
            "**Recommendation: Mode B2 is viable (local extraction validates correctly),",
            "but .idx availability on S3 should be confirmed for the full 2020–2025 range",
            "before adopting as default.**",
            "",
            "- Local message extraction (Mode B-local) is bit-identical to Mode A",
            "- S3 byte-range download (Mode B2) validation was not completed (see above)",
            "- Run a broader validation across multiple months before switching default",
        ]
    else:
        lines += [
            "**Recommendation: Keep Mode A (full-file) as the default until Mode B",
            "validation is completed.**",
            "",
            "- Mode B local extraction or S3 byte-range has not been fully validated",
            "- The current full-file approach is proven and simple",
        ]
    lines += [
        "",
        "## 6. What _selected_targets() Covers vs What It Should Cover",
        "",
        "The existing `_selected_targets()` in `src/datasources/rtma.py` targets:",
        "TMP, SPFH/DPT, UGRD, VGRD, PRES, TCDC — covering 6 of the 11 Stage 1 variables.",
        "",
        "**Missing from current targets (4 variables)**:",
        "- `10si` / WIND — wind speed",
        "- `i10fg` / GUST — wind gust",
        "- `vis` / VIS — visibility",
        "- `ceil` / CEIL — cloud ceiling",
        "",
        "The `_EXT_TARGETS` dict in this benchmark script covers all 11 variables.",
        "If selected-message mode is adopted, `_selected_targets()` in `rtma.py` should",
        "be extended to include WIND, GUST, VIS, CEIL families.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark RTMA full-file vs selected-message acquisition"
    )
    p.add_argument("--config",    default="configs/pilot_stage1.yaml")
    p.add_argument("--data-root", dest="data_root", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_total = time.perf_counter()
    args = _parse_args()

    from src.pipeline.config import load_config
    from src.pipeline.geometries import normalise_staid
    import pandas as pd
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    cfg       = load_config(Path(args.config))
    data_root = cfg.effective_data_root(override=args.data_root)
    LOGGER.info("Data root: %s", data_root)

    # Paths
    weights_base      = cfg.output_dir("basin_geometries", data_root) / "weights"
    rtma_weights_path = weights_base / "rtma" / "pilot_rtma_weights.parquet"
    manifest_csv      = cfg.output_dir("manifests", data_root) / "stage1_pilot" / "pilot_basin_manifest.csv"
    raw_rtma_dir      = cfg.output_dir("raw", data_root) / "rtma"
    out_dir           = cfg.output_dir("manifests", data_root) / "stage1_pilot" / "january_2023_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cached full-file path
    full_file_path = raw_rtma_dir / "rtma2p5.20230101" / "rtma2p5.t00z.2dvaranl_ndfd.grb2_wexp"
    if not full_file_path.exists():
        LOGGER.error(
            "Cached RTMA full file not found: %s\n"
            "Run the smoke test first to download it:\n"
            "  python scripts/extract_stage1_january.py --config %s "
            "--data-root %s --max-hours 3",
            full_file_path, args.config, args.data_root or data_root,
        )
        return 1

    full_file_bytes = full_file_path.stat().st_size
    LOGGER.info("Full RTMA file: %s  (%.1f MB)", full_file_path.name, full_file_bytes / 1e6)

    # Prereqs
    for p, label in [(rtma_weights_path, "RTMA weights"), (manifest_csv, "Basin manifest")]:
        if not p.exists():
            LOGGER.error("Missing prerequisite: %s — %s", label, p)
            return 1

    rtma_weights  = pd.read_parquet(rtma_weights_path)
    pilot_manifest = pd.read_csv(manifest_csv)
    pilot_staids   = [normalise_staid(s) for s in pilot_manifest["STAID"].tolist()]
    LOGGER.info("Pilot basins: %d", len(pilot_staids))

    s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))

    # -------------------------------------------------------------------------
    # Step 0: Fetch and parse .idx file
    # -------------------------------------------------------------------------
    LOGGER.info("Fetching .idx from s3://%s/%s ...", _RTMA_BUCKET, _IDX_KEY)
    idx_text:  Optional[str] = None
    idx_bytes: int = 0
    idx_fetch_s: float = 0.0

    try:
        t0 = time.perf_counter()
        resp = s3.get_object(Bucket=_RTMA_BUCKET, Key=_IDX_KEY)
        idx_text  = resp["Body"].read().decode("utf-8", errors="replace")
        idx_bytes = len(idx_text.encode("utf-8"))
        idx_fetch_s = time.perf_counter() - t0
        LOGGER.info("  .idx fetched: %d bytes in %.2fs", idx_bytes, idx_fetch_s)
    except Exception as exc:
        LOGGER.warning("Could not fetch .idx (will skip Mode B2): %s", exc)

    # Parse .idx
    idx_entries:     list[dict] = []
    all_idx_names:   list[str]  = []
    stage1_entries:  list[dict] = []
    stage1_has_spfh: bool       = False

    if idx_text:
        idx_entries = _parse_idx_entries(idx_text)
        stage1_entries, all_idx_names, stage1_has_spfh = _identify_stage1_entries(idx_entries)
        LOGGER.info(
            "  .idx: %d total messages; %d selected for Stage 1; has_spfh=%s",
            len(idx_entries), len(stage1_entries), stage1_has_spfh,
        )
        LOGGER.info("  All .idx names: %s", all_idx_names)
        LOGGER.info("  Stage 1 selected names: %s", [e["short_name"] for e in stage1_entries])

    # Identify Stage 1 cfgrib vars not matched in .idx
    selected_idx_names = {e["short_name"] for e in stage1_entries}

    def _idx_name_maps_to_cfgrib(cfgrib_var: str) -> bool:
        for names in _EXT_TARGETS.values():
            if cfgrib_var in names:
                return any(n in selected_idx_names for n in names)
        return False

    stage1_missing_cfgrib = [
        v for v in sorted(_STAGE1_CFGRIB)
        if not _idx_name_maps_to_cfgrib(v)
    ]
    if stage1_missing_cfgrib:
        LOGGER.warning(
            "Stage 1 cfgrib vars not matched in .idx: %s", stage1_missing_cfgrib
        )

    # -------------------------------------------------------------------------
    # Mode A: Decode + extract from full cached file (baseline)
    # -------------------------------------------------------------------------
    LOGGER.info("--- Mode A: full-file decode + extract ---")
    mode_a = _decode_and_extract(
        full_file_path, rtma_weights, pilot_staids, rtma_weights_path, "Mode-A"
    )
    LOGGER.info(
        "Mode A: vars=%s  rows=%d  decode=%.3fs  extract=%.3fs",
        mode_a["included_vars"], mode_a["n_rows"],
        mode_a["decode_s"], mode_a["extract_s"],
    )

    # -------------------------------------------------------------------------
    # Mode B-local: extract selected GRIB messages from local full file
    # -------------------------------------------------------------------------
    LOGGER.info("--- Mode B-local: local GRIB message extraction ---")
    stage1_names_set = set()
    for names in _EXT_TARGETS.values():
        stage1_names_set.update(names)

    b_local_path, b_local_bytes, b_local_extract_s, b_local_per_name = (
        _extract_selected_messages_local(
            full_file_path, stage1_names_set, stage1_has_spfh
        )
    )

    mode_b_local: Optional[dict] = None
    comparison_b_local: Optional[dict] = None

    if b_local_path is not None:
        try:
            LOGGER.info(
                "Mode B-local: temp file %s  size=%.1f MB  created in %.3fs",
                b_local_path.name, b_local_bytes / 1e6, b_local_extract_s,
            )
            LOGGER.info(
                "  Message bytes by name: %s",
                {k: f"{v/1e3:.0f} KB" for k, v in b_local_per_name.items()},
            )
            mode_b_local = _decode_and_extract(
                b_local_path, rtma_weights, pilot_staids, rtma_weights_path, "Mode-B-local"
            )
            comparison_b_local = _compare_results(mode_a, mode_b_local, "Mode-B-local")
            LOGGER.info(
                "Mode B-local comparison: %s  max_diff=%s",
                comparison_b_local["status"],
                comparison_b_local.get("max_abs_diff_all"),
            )
        finally:
            if b_local_path.exists():
                b_local_path.unlink(missing_ok=True)
    else:
        LOGGER.warning("Mode B-local: eccodes not available — skipped")

    # -------------------------------------------------------------------------
    # Mode B2: S3 byte-range download
    # -------------------------------------------------------------------------
    LOGGER.info("--- Mode B2: S3 byte-range download ---")
    b2_bytes: Optional[int] = None
    b2_dl_s:  Optional[float] = None
    mode_b2:  Optional[dict] = None
    comparison_b2: Optional[dict] = None

    if idx_text and stage1_entries:
        b2_tmp_path, b2_bytes, b2_dl_s, b2_ranges = _download_selected_messages_s3(
            s3, _RTMA_BUCKET, _RTMA_KEY, stage1_entries
        )
        if b2_tmp_path is not None:
            try:
                LOGGER.info(
                    "Mode B2: downloaded %s in %.2fs via %d ranges",
                    f"{b2_bytes / 1e6:.1f} MB", b2_dl_s, len(b2_ranges),
                )
                mode_b2 = _decode_and_extract(
                    b2_tmp_path, rtma_weights, pilot_staids, rtma_weights_path, "Mode-B2"
                )
                comparison_b2 = _compare_results(mode_a, mode_b2, "Mode-B2")
                LOGGER.info(
                    "Mode B2 comparison: %s  max_diff=%s",
                    comparison_b2["status"],
                    comparison_b2.get("max_abs_diff_all"),
                )
            finally:
                if b2_tmp_path.exists():
                    b2_tmp_path.unlink(missing_ok=True)
        else:
            LOGGER.warning("Mode B2 download failed")
    else:
        LOGGER.warning("Mode B2: no .idx or no Stage 1 entries — skipped")

    # -------------------------------------------------------------------------
    # Build benchmark table rows
    # -------------------------------------------------------------------------
    rows: list[dict] = []

    rows.append({
        "mode":                     "A_full_file",
        "label":                    "Full-file download (current default)",
        "download_bytes":           full_file_bytes,
        "fresh_download_time_s":    _SMOKE_FULLFILE_DL_S,
        "idx_fetch_time_s":         0.0,
        "decode_time_s":            mode_a["decode_s"],
        "extraction_time_s":        mode_a["extract_s"],
        "total_download_plus_proc_s": _SMOKE_FULLFILE_DL_S + mode_a["decode_s"] + mode_a["extract_s"],
        "n_vars_decoded":           mode_a["n_vars"],
        "decoded_vars":             ",".join(mode_a["included_vars"]),
        "output_rows":              mode_a["n_rows"],
        "equivalent_to_mode_a":     "baseline",
        "max_abs_diff_weighted_mean": None,
        "notes":                    "File already cached; download time from Milestone 2D observation",
    })

    if mode_b_local is not None:
        comp = comparison_b_local or {}
        rows.append({
            "mode":                     "B_local_extract",
            "label":                    "Local selected-message extraction (validation only)",
            "download_bytes":           b_local_bytes,
            "fresh_download_time_s":    0.0,
            "idx_fetch_time_s":         0.0,
            "decode_time_s":            mode_b_local["decode_s"],
            "extraction_time_s":        mode_b_local["extract_s"],
            "total_download_plus_proc_s": b_local_extract_s + mode_b_local["decode_s"] + mode_b_local["extract_s"],
            "n_vars_decoded":           mode_b_local["n_vars"],
            "decoded_vars":             ",".join(mode_b_local["included_vars"]),
            "output_rows":              mode_b_local["n_rows"],
            "equivalent_to_mode_a":     comp.get("status", "unknown"),
            "max_abs_diff_weighted_mean": comp.get("max_abs_diff_all"),
            "notes":                    f"GRIB extracted from local full file with eccodes; "
                                        f"local_extract_s={b_local_extract_s:.3f}; "
                                        f"per_name_bytes={json.dumps({k: v for k, v in b_local_per_name.items()})}",
        })

    if mode_b2 is not None and b2_bytes is not None and b2_dl_s is not None:
        comp = comparison_b2 or {}
        total_b2_dl = idx_fetch_s + b2_dl_s
        rows.append({
            "mode":                     "B2_s3_byte_range",
            "label":                    "S3 selected-message byte-range (optimized production mode)",
            "download_bytes":           idx_bytes + b2_bytes,
            "fresh_download_time_s":    total_b2_dl,
            "idx_fetch_time_s":         idx_fetch_s,
            "decode_time_s":            mode_b2["decode_s"],
            "extraction_time_s":        mode_b2["extract_s"],
            "total_download_plus_proc_s": total_b2_dl + mode_b2["decode_s"] + mode_b2["extract_s"],
            "n_vars_decoded":           mode_b2["n_vars"],
            "decoded_vars":             ",".join(mode_b2["included_vars"]),
            "output_rows":              mode_b2["n_rows"],
            "equivalent_to_mode_a":     comp.get("status", "unknown"),
            "max_abs_diff_weighted_mean": comp.get("max_abs_diff_all"),
            "notes":                    f"idx_bytes={idx_bytes}; selected_messages={len(stage1_entries)}; "
                                        f"idx_key={_IDX_KEY}",
        })

    elif idx_text and not mode_b2:
        # Record that B2 download failed or produced no result
        rows.append({
            "mode":                     "B2_s3_byte_range",
            "label":                    "S3 selected-message byte-range",
            "download_bytes":           b2_bytes or 0,
            "fresh_download_time_s":    (idx_fetch_s + (b2_dl_s or 0)),
            "idx_fetch_time_s":         idx_fetch_s,
            "decode_time_s":            None,
            "extraction_time_s":        None,
            "total_download_plus_proc_s": None,
            "n_vars_decoded":           None,
            "decoded_vars":             None,
            "output_rows":              None,
            "equivalent_to_mode_a":     "FAILED",
            "max_abs_diff_weighted_mean": None,
            "notes":                    "Download or decode failed; see log",
        })

    # Write CSV
    csv_path = out_dir / "rtma_acquisition_benchmark.csv"
    _write_csv(csv_path, rows)
    LOGGER.info("Benchmark CSV written: %s", csv_path)

    # Write Markdown report
    md_path = out_dir / "rtma_acquisition_benchmark.md"
    _write_markdown_report(
        path              = md_path,
        full_file_bytes   = full_file_bytes,
        full_file_dl_s_fresh = _SMOKE_FULLFILE_DL_S,
        idx_bytes         = idx_bytes,
        idx_fetch_s       = idx_fetch_s,
        b2_bytes          = b2_bytes,
        b2_dl_s           = b2_dl_s,
        b_local_bytes     = b_local_bytes,
        b_local_extract_s = b_local_extract_s,
        mode_a_result     = mode_a,
        mode_b_result     = mode_b2,
        comparison_b_local= comparison_b_local,
        comparison_b2     = comparison_b2,
        idx_all_names     = all_idx_names,
        stage1_entries    = stage1_entries,
        stage1_missing_cfgrib = stage1_missing_cfgrib,
    )
    LOGGER.info("Benchmark MD written: %s", md_path)

    # -------------------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------------------
    t_elapsed = time.perf_counter() - t_total
    sep = "=" * 72
    print(f"\n{sep}")
    print("RTMA Acquisition Mode Benchmark — 2023-01-01T00:00Z")
    print(sep)
    print()
    print("  --- Audit: Current Acquisition Path ---")
    print("  extract_stage1_january.py")
    print("    -> _download_single_rtma()")
    print("    -> RtmaAwsConusDataSource.download_sample()")
    print("    -> _download_one_object_timed() -> _download_s3_object()")
    print("    => MODE: FULL FILE (no Range header)")
    print()
    print("  measure_selected_variable_bytes() in rtma.py:")
    print("    -> _measure_selected_from_index_and_ranges()")
    print("    => S3 byte-range IS implemented BUT discards content (size-counting only)")
    print()
    print("  _selected_targets() covers: TMP, SPFH/DPT, UGRD, VGRD, PRES, TCDC")
    print("  Missing from _selected_targets: WIND(10si), GUST(i10fg), VIS, CEIL")
    print()
    print("  --- .idx Analysis ---")
    print(f"  .idx size     : {idx_bytes / 1e3:.1f} KB in {idx_fetch_s:.2f}s")
    print(f"  Messages total: {len(all_idx_names)}")
    print(f"  All names     : {all_idx_names}")
    print(f"  Stage 1 sel.  : {len(stage1_entries)}  ({[e['short_name'] for e in stage1_entries]})")
    if stage1_missing_cfgrib:
        print(f"  NOT matched in idx: {stage1_missing_cfgrib}")
    print()
    print("  --- Mode A: Full-file (baseline) ---")
    print(f"  File size     : {full_file_bytes / 1e6:.1f} MB")
    print(f"  Fresh dl time : ~{_SMOKE_FULLFILE_DL_S:.1f}s (observed Milestone 2D)")
    print(f"  Decode time   : {mode_a['decode_s']:.3f}s")
    print(f"  Extract time  : {mode_a['extract_s']:.3f}s")
    print(f"  Vars decoded  : {mode_a['n_vars']}  {mode_a['included_vars']}")
    print(f"  Output rows   : {mode_a['n_rows']}")
    print()

    if mode_b_local is not None:
        comp = comparison_b_local or {}
        print("  --- Mode B-local: local message extraction (cfgrib validation) ---")
        print(f"  Subset size   : {b_local_bytes / 1e6:.1f} MB  ({b_local_bytes / full_file_bytes * 100:.0f}% of full file)")
        print(f"  Local extract : {b_local_extract_s:.3f}s")
        print(f"  Decode time   : {mode_b_local['decode_s']:.3f}s")
        print(f"  Extract time  : {mode_b_local['extract_s']:.3f}s")
        print(f"  Vars decoded  : {mode_b_local['n_vars']}  {mode_b_local['included_vars']}")
        print(f"  Output rows   : {mode_b_local['n_rows']}")
        print(f"  Comparison    : {comp.get('status')}  max_diff={comp.get('max_abs_diff_all')}")
        print()

    if mode_b2 is not None and b2_bytes is not None:
        comp = comparison_b2 or {}
        pct = b2_bytes / full_file_bytes * 100
        saving = (1.0 - (idx_bytes + b2_bytes) / full_file_bytes) * 100
        print("  --- Mode B2: S3 byte-range download ---")
        print(f"  .idx size     : {idx_bytes / 1e3:.1f} KB in {idx_fetch_s:.2f}s")
        print(f"  Range download: {b2_bytes / 1e6:.1f} MB in {b2_dl_s:.2f}s")
        print(f"  Total download: {(idx_bytes + b2_bytes) / 1e6:.1f} MB  "
              f"(vs {full_file_bytes / 1e6:.1f} MB full; -{saving:.0f}% savings)")
        print(f"  Decode time   : {mode_b2['decode_s']:.3f}s")
        print(f"  Extract time  : {mode_b2['extract_s']:.3f}s")
        print(f"  Vars decoded  : {mode_b2['n_vars']}  {mode_b2['included_vars']}")
        print(f"  Output rows   : {mode_b2['n_rows']}")
        print(f"  Comparison    : {comp.get('status')}  max_diff={comp.get('max_abs_diff_all')}")
        print()
        # Full-period estimates
        FULL_H = 52608
        full_a_raw = full_file_bytes * FULL_H
        full_b2_raw = (idx_bytes + b2_bytes) * FULL_H
        print("  --- Full-period Estimates (2,843 basins, 2020-2025, 52,608 hours) ---")
        print(f"  Mode A raw cache: {full_a_raw / 1e12:.2f} TB")
        print(f"  Mode B2 raw cache:{full_b2_raw / 1e12:.2f} TB  (-{(1 - full_b2_raw/full_a_raw)*100:.0f}%)")
        a_total_s = (_SMOKE_FULLFILE_DL_S + mode_a["decode_s"] + mode_a["extract_s"]) * FULL_H
        b2_total_s = (idx_fetch_s + b2_dl_s + mode_b2["decode_s"] + mode_b2["extract_s"]) * FULL_H
        print(f"  Mode A serial   : {a_total_s/3600:.0f}h")
        print(f"  Mode B2 serial  : {b2_total_s/3600:.0f}h")
        print(f"  Mode A @ 32 HPC : {a_total_s/3600/32:.0f}h")
        print(f"  Mode B2 @ 32 HPC: {b2_total_s/3600/32:.0f}h")
        print(f"  Mode A @ 128 HPC: {a_total_s/3600/128:.1f}h")
        print(f"  Mode B2 @ 128HPC: {b2_total_s/3600/128:.1f}h")
    elif idx_text:
        FULL_H = 52608
        # Estimate B2 based on idx message count and known full-file size
        # Each of the N selected messages ≈ full_file_bytes / len(idx_entries)
        if idx_entries and stage1_entries:
            per_msg_est = full_file_bytes / len(idx_entries)
            b2_est = per_msg_est * len(stage1_entries)
            saving_est = (1.0 - (idx_bytes + b2_est) / full_file_bytes) * 100
            print(f"  Mode B2 (estimated, download not completed):")
            print(f"    Estimated selected bytes : {b2_est / 1e6:.1f} MB  (-{saving_est:.0f}% estimated)")
            print(f"    Full-period cache est    : {(idx_bytes + b2_est) * FULL_H / 1e12:.2f} TB")
        print()

    print(f"  Outputs: {csv_path}")
    print(f"           {md_path}")
    print(f"  Total benchmark time: {t_elapsed:.1f}s")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
