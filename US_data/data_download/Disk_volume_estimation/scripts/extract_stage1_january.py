#!/usr/bin/env python3
"""Stage 1 Milestone 2D — January 2023 pilot extraction and resource scaling estimates.

Scales the validated one-hour extraction to the full configured pilot time range
(2023-01-01T00:00:00 – 2023-01-31T23:00:00 UTC, hourly), collects detailed
timing and volume metrics per product/hour, and produces full-dataset scaling
estimates for 2,843 basins over 2020–2025.

Usage:
    # Smoke test (first 3 hours):
    python scripts/extract_stage1_january.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun \\
        --max-hours 3

    # Full January extraction:
    python scripts/extract_stage1_january.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun

    # Resume an interrupted run:
    python scripts/extract_stage1_january.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun \\
        --resume

Outputs:
    03_basin_timeseries/stage1_pilot/january_2023/
        mrms_hourly_basin_stats.parquet
        rtma_hourly_basin_stats.parquet
        combined_hourly_basin_stats.parquet
        preview_mrms.csv  (first 5 rows)
        preview_rtma.csv  (first 5 rows)
    06_qc_reports/stage1_pilot/january_2023_extraction/
        hourly_availability.png
        runtime_by_hour.png
        raw_file_size_by_hour.png
        cumulative_volume.png
        representative_timeseries.png
        variable_completeness_rtma.png
        basin_completeness_distribution.png
        full_dataset_storage_estimate.png
    09_manifests/stage1_pilot/january_2023_extraction/
        manifest.json  summary.json  summary.md
        run_command.txt  git_commit.txt  config_snapshot.yaml
        hourly_runtime_and_volume.csv
        hourly_file_status.csv
        product_runtime_volume_summary.csv
        full_dataset_scaling_estimates.csv/json/md
        missing_files.csv  (if any)
        variable_completeness.csv
        basin_completeness.csv
    tmp/january_2023_staging/mrms/<YYYYMMDDHH>.parquet  (per-hour staging)
    tmp/january_2023_staging/rtma/<YYYYMMDDHH>.parquet

Parquet schema: identical to extract_stage1_one_hour.py — see src/pipeline/extraction.py
  STAT_COLUMNS for the canonical column list.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
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
LOGGER = logging.getLogger("extract_january")

# MRMS/RTMA geographic extents (Milestone 2A grid-definition discovery)
_MRMS_LON_MIN, _MRMS_LON_MAX = -129.995, -60.005
_MRMS_LAT_MIN, _MRMS_LAT_MAX =   20.005,  54.995
_MRMS_DX, _MRMS_DY = 0.01, 0.01
_RTMA_LON_MIN, _RTMA_LON_MAX = -138.373,  -59.042
_RTMA_LAT_MIN, _RTMA_LAT_MAX =   19.229,   57.089

# Full-dataset scaling targets
_FULL_BASINS = 2843
_FULL_YEARS = (2020, 2025)  # inclusive
_FULL_HOURS = sum(
    (366 if y in (2020, 2024) else 365) * 24
    for y in range(_FULL_YEARS[0], _FULL_YEARS[1] + 1)
)  # 52,608 hours


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 Milestone 2D: January 2023 pilot extraction + scaling estimates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/pilot_stage1.yaml")
    p.add_argument("--data-root", dest="data_root", default=None)
    p.add_argument("--start", default=None,
                   help="Override pilot start (ISO 8601, e.g. 2023-01-01T00:00:00)")
    p.add_argument("--end", default=None,
                   help="Override pilot end (ISO 8601, e.g. 2023-01-31T23:00:00)")
    p.add_argument("--products", default="mrms,rtma",
                   help="Comma-separated products to extract (mrms, rtma, or both)")
    p.add_argument("--max-hours", dest="max_hours", type=int, default=None,
                   help="Process at most this many hours (smoke-test mode)")
    p.add_argument("--resume", action="store_true",
                   help="Skip hours that already have successful staging parquets")
    p.add_argument("--rtma-mode", dest="rtma_mode",
                   choices=["selected_messages", "full_file"],
                   default="selected_messages",
                   help=(
                       "RTMA acquisition mode (default: selected_messages). "
                       "'selected_messages' downloads only the 11 Stage 1 GRIB messages "
                       "via S3 byte-range requests (~71 MB/file, -16%% vs full). "
                       "'full_file' downloads the entire grb2_wexp (~84 MB/file). "
                       "Cached files are reused regardless of mode."
                   ))
    p.add_argument("--download-workers", dest="download_workers", type=int, default=4,
                   help=(
                       "Number of parallel RTMA download workers (default: 4). "
                       "Downloads are pre-fetched concurrently before the serial "
                       "decode/extract loop. MRMS remains serial (files are tiny). "
                       "Use 1 for fully serial operation. "
                       "Recommended: 4 locally, 8 on HPC."
                   ))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _iter_hours(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(hours=1)


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.0f}s"
    return f"{m}m {s:.0f}s"


def _fmt_hms(seconds: int) -> str:
    a = abs(int(seconds))
    return f"{a // 3600:02d}:{(a % 3600) // 60:02d}:{a % 60:02d}"


# ---------------------------------------------------------------------------
# Cache index builders (build once; avoid rglob per hour)
# ---------------------------------------------------------------------------

def _build_mrms_cache_index(raw_dir: Path) -> dict[datetime, Path]:
    import re
    idx: dict[datetime, Path] = {}
    mrms_dir = raw_dir / "mrms"
    if not mrms_dir.exists():
        return idx
    for p in mrms_dir.rglob("*.grib2.gz"):
        m = re.search(r"(\d{8})-(\d{6})", p.name)
        if m:
            try:
                dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
                idx[dt] = p
            except ValueError:
                pass
    return idx


def _build_rtma_cache_index(raw_dir: Path) -> dict[datetime, Path]:
    import re
    idx: dict[datetime, Path] = {}
    rtma_dir = raw_dir / "rtma"
    if not rtma_dir.exists():
        return idx
    for p in rtma_dir.rglob("*.grb2_wexp"):
        hour_m = re.search(r"\.t(\d{2})z\.2dvaranl_ndfd\.grb2_wexp$", p.name)
        day_m  = re.search(r"rtma2p5\.(\d{8})", p.parent.name)
        if hour_m and day_m:
            try:
                dt = datetime.strptime(f"{day_m.group(1)}{hour_m.group(1)}", "%Y%m%d%H")
                idx[dt] = p
            except ValueError:
                pass
    return idx


# ---------------------------------------------------------------------------
# Single-file download (reuses pre-initialised datasource instance)
# ---------------------------------------------------------------------------

def _download_single_mrms(mrms_ds: Any, obj: Any, raw_dir: Path) -> tuple[Optional[Path], float]:
    t0 = time.perf_counter()
    try:
        files = mrms_ds.download_sample(raw_dir / "mrms", [obj])
        elapsed = time.perf_counter() - t0
        return (files[0] if files else None), elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        LOGGER.warning("MRMS download failed: %s", exc)
        return None, elapsed


def _download_single_rtma(
    rtma_ds: Any,
    obj: Any,
    raw_dir: Path,
    *,
    mode: str = "selected_messages",
) -> tuple[Optional[Path], float]:
    """Download one RTMA file.

    mode='selected_messages': use .idx byte-range download (default; ~71 MB/file).
    mode='full_file': download complete grb2_wexp (~84 MB/file).
    Falls back to full_file automatically if .idx is unavailable.
    """
    if mode == "selected_messages":
        try:
            path, _bytes, elapsed = rtma_ds.download_selected_messages(raw_dir / "rtma", obj)
            return path, elapsed
        except Exception as exc:
            LOGGER.warning("RTMA selected-message download failed (%s) — falling back to full-file", exc)
            # Fall through to full-file below

    t0 = time.perf_counter()
    try:
        files = rtma_ds.download_sample(raw_dir / "rtma", [obj])
        elapsed = time.perf_counter() - t0
        return (files[0] if files else None), elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        LOGGER.warning("RTMA download failed: %s", exc)
        return None, elapsed


# ---------------------------------------------------------------------------
# Parallel RTMA pre-fetch
# ---------------------------------------------------------------------------

def _ensure_rtma_s3_pool(rtma_ds: Any, n_workers: int) -> None:
    """Bump the RTMA datasource S3 connection pool to support n_workers parallel downloads.

    selected_messages mode makes 3 S3 calls per file (1 idx + 2 Range GETs).
    full_file mode makes 1 S3 call per file.
    We add a safety margin on top of the per-worker call count.
    """
    if n_workers <= 1:
        return
    needed = max(25, 4 * n_workers + 5)
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config as _BotoConfig
        rtma_ds._s3_client_cached = boto3.client(
            "s3",
            config=_BotoConfig(signature_version=UNSIGNED, max_pool_connections=needed),
        )
        LOGGER.debug("RTMA S3 pool bumped to %d connections for %d workers", needed, n_workers)
    except Exception as exc:
        LOGGER.warning("Could not bump S3 pool size: %s", exc)


def _prefetch_rtma_files(
    s3_objects: dict,          # datetime -> RemoteObject (only uncached, in-S3 hours)
    rtma_ds: Any,
    raw_dir: "Path",
    mode: str,
    n_workers: int,
) -> dict:                     # datetime -> (Optional[Path], float)  (path, download_s)
    """Pre-fetch RTMA files for many hours in parallel; return {dt: (path, dl_s)}.

    Uses n_workers threads.  Each worker calls _download_single_rtma() independently.
    Decode/extract remain serial in the main loop; only the I/O-bound download
    phase is parallelised here.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    if not s3_objects:
        return {}

    n = min(n_workers, len(s3_objects))
    LOGGER.info(
        "Pre-fetching %d RTMA files (mode=%s, workers=%d) ...",
        len(s3_objects), mode, n,
    )
    _ensure_rtma_s3_pool(rtma_ds, n)

    results: dict = {}
    t0_prefetch = time.perf_counter()

    def _worker(dt_obj):
        dt, obj = dt_obj
        path, dl_s = _download_single_rtma(rtma_ds, obj, raw_dir, mode=mode)
        return dt, path, dl_s

    with ThreadPoolExecutor(max_workers=n) as executor:
        fut_to_dt = {executor.submit(_worker, item): item[0] for item in s3_objects.items()}
        done = 0
        for fut in _as_completed(fut_to_dt):
            dt = fut_to_dt[fut]
            try:
                _, path, dl_s = fut.result()
                results[dt] = (path, dl_s)
                if path is None:
                    LOGGER.warning("Prefetch returned None for %s", dt)
            except Exception as exc:
                LOGGER.warning("Prefetch exception for %s: %s", dt, exc)
                results[dt] = (None, 0.0)
            done += 1
            if done % max(1, len(s3_objects) // 5) == 0 or done == len(s3_objects):
                elapsed = time.perf_counter() - t0_prefetch
                ok = sum(1 for p, _ in results.values() if p is not None)
                mb_done = sum(
                    (v[0].stat().st_size if v[0] and v[0].exists() else 0)
                    for v in results.values()
                ) / 1e6
                LOGGER.info(
                    "  Prefetch %d/%d  ok=%d  %.1f MB  %.1f MB/s",
                    done, len(s3_objects), ok, mb_done,
                    mb_done / max(elapsed, 0.01),
                )

    n_ok = sum(1 for p, _ in results.values() if p is not None)
    n_fail = len(results) - n_ok
    total_s = time.perf_counter() - t0_prefetch
    total_mb = sum(
        (v[0].stat().st_size if v[0] and v[0].exists() else 0) for v in results.values()
    ) / 1e6
    LOGGER.info(
        "Prefetch done: %d ok, %d failed, %.1f MB in %.1fs (%.2f MB/s)",
        n_ok, n_fail, total_mb, total_s, total_mb / max(total_s, 0.01),
    )
    return results


# ---------------------------------------------------------------------------
# Per-hour decode + extract (separate timing from download)
# ---------------------------------------------------------------------------

def _process_hour_mrms(
    mrms_path: Path,
    mrms_weights: Any,
    pilot_staids: list[str],
    mrms_weights_path: Path,
) -> tuple[Any, float, float]:
    """Returns (df, decode_s, extract_s)."""
    from src.pipeline.extraction import decode_mrms_grid, extract_basin_statistics

    t0 = time.perf_counter()
    mrms_grid = decode_mrms_grid(mrms_path)
    decode_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = extract_basin_statistics(
        mrms_grid, mrms_weights, pilot_staids,
        weight_table_path=str(mrms_weights_path),
        source_file_path=str(mrms_path),
    )
    extract_s = time.perf_counter() - t0
    return df, decode_s, extract_s


def _process_hour_rtma(
    rtma_path: Path,
    rtma_weights: Any,
    pilot_staids: list[str],
    rtma_weights_path: Path,
) -> tuple[Any, float, float, list[str], list[str]]:
    """Returns (df, decode_s, extract_s, included_vars, excluded_vars)."""
    import pandas as pd
    from src.pipeline.extraction import (
        decode_rtma_grids, extract_basin_statistics,
        _RTMA_EXCLUDED_DEFAULT, STAT_COLUMNS,
    )

    t0 = time.perf_counter()
    rtma_grids_all = decode_rtma_grids(rtma_path)
    decode_s = time.perf_counter() - t0

    included = [vg for vg in rtma_grids_all if vg.short_name not in _RTMA_EXCLUDED_DEFAULT]
    excluded = [vg.short_name for vg in rtma_grids_all if vg.short_name in _RTMA_EXCLUDED_DEFAULT]
    included_names = [vg.short_name for vg in included]

    t0 = time.perf_counter()
    frames = [
        extract_basin_statistics(
            vg, rtma_weights, pilot_staids,
            weight_table_path=str(rtma_weights_path),
            source_file_path=str(rtma_path),
        )
        for vg in included
    ]
    import pandas as pd
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STAT_COLUMNS)
    extract_s = time.perf_counter() - t0

    return df, decode_s, extract_s, included_names, excluded


# ---------------------------------------------------------------------------
# Scaling estimates
# ---------------------------------------------------------------------------

def _compute_scaling_estimates(
    product_summary: dict[str, dict],
    pilot_hours: int,
    pilot_basins: int,
    full_hours: int = _FULL_HOURS,
    full_basins: int = _FULL_BASINS,
) -> dict[str, Any]:
    """Compute full-dataset scaling estimates from observed pilot measurements."""
    import numpy as np

    estimates: dict[str, Any] = {
        "pilot_hours": pilot_hours,
        "pilot_basins": pilot_basins,
        "full_hours": full_hours,
        "full_basins": full_basins,
        "full_period": "2020-01-01T00:00Z to 2025-12-31T23:00Z",
        "notes": [
            "Raw download volume scales with hours only "
            "(one file per hour per product, independent of basin count).",
            "Processed Parquet volume scales with hours × basins (× variables for RTMA).",
            "Serial wall-clock time estimated from median per-hour processing time × full hours.",
            "HPC estimates assume ideal parallel scaling at file level; actual overhead not modelled.",
            "Network speed, S3 throttling, and compression ratios may vary.",
        ],
        "products": {},
    }

    for product, psum in product_summary.items():
        ok_hours = psum.get("successful_hours", 0)
        if ok_hours == 0:
            continue

        # Raw download: scales with hours only (one file per hour, CONUS-wide regardless of basin count)
        raw_bytes_per_hour = psum.get("total_raw_bytes", 0) / ok_hours
        full_raw_bytes = raw_bytes_per_hour * full_hours

        # Processed Parquet: scales with hours × basins
        total_parquet = psum.get("total_output_parquet_bytes", 0)
        parquet_per_basin_hour = total_parquet / max(ok_hours * pilot_basins, 1)
        full_parquet_bytes = parquet_per_basin_hour * full_hours * full_basins

        # Expected output rows
        rows_per_basin_hour = psum.get("total_output_rows", 0) / max(ok_hours * pilot_basins, 1)
        full_rows = int(rows_per_basin_hour * full_hours * full_basins)

        # Wall-clock time (serial)
        median_proc_s = psum.get("median_total_processing_time_s", 0.0)
        serial_wall_s = median_proc_s * full_hours

        estimates["products"][product] = {
            "full_hours": full_hours,
            "full_basins": full_basins,
            "full_basin_hours": full_hours * full_basins,
            "estimated_raw_download_bytes": int(full_raw_bytes),
            "estimated_raw_download_human": _format_bytes(full_raw_bytes),
            "estimated_processed_parquet_bytes": int(full_parquet_bytes),
            "estimated_processed_parquet_human": _format_bytes(full_parquet_bytes),
            "estimated_output_rows": full_rows,
            "raw_bytes_per_hour_observed": int(raw_bytes_per_hour),
            "raw_bytes_per_hour_observed_human": _format_bytes(raw_bytes_per_hour),
            "parquet_bytes_per_basin_hour_observed": round(parquet_per_basin_hour, 2),
            "median_processing_s_per_hour_observed": round(median_proc_s, 3),
            "estimated_serial_wall_clock_seconds": int(serial_wall_s),
            "estimated_serial_wall_clock_human": _format_duration(serial_wall_s),
            "estimated_hpc_wall_clock_seconds": {
                n: int(serial_wall_s / n) for n in [1, 8, 32, 128]
            },
            "estimated_hpc_wall_clock_human": {
                n: _format_duration(serial_wall_s / n) for n in [1, 8, 32, 128]
            },
            "raw_scale_assumption": (
                "Raw download scales with hours only: one CONUS-wide file per hour, "
                "independent of basin count."
            ),
            "parquet_scale_assumption": (
                "Processed Parquet scales with hours × basins (× variables for RTMA); "
                "basin weights are pre-computed."
            ),
            "hpc_assumption": (
                "Ideal linear parallel scaling at file level assumed; "
                "actual scheduling overhead and I/O contention not modelled."
            ),
        }

    return estimates


# ---------------------------------------------------------------------------
# Markdown scaling report
# ---------------------------------------------------------------------------

def _write_scaling_md(path: Path, estimates: dict[str, Any]) -> None:
    lines = [
        "# Full-Dataset Scaling Estimates",
        "",
        f"**Pilot period**: January 2023  "
        f"({estimates.get('pilot_hours', '?')} hours, {estimates.get('pilot_basins', '?')} basins)",
        f"**Full period**: {estimates.get('full_period', '?')}",
        f"**Full hours**: {estimates.get('full_hours', '?'):,}",
        f"**Full basins**: {estimates.get('full_basins', '?'):,}",
        "",
        "## Notes",
        "",
    ]
    for note in estimates.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")

    for product, est in estimates.get("products", {}).items():
        lines += [
            f"## {product}",
            "",
            f"| Metric | Observed (pilot) | Estimated (full) |",
            f"|--------|-----------------|-----------------|",
            f"| Hours | {estimates.get('pilot_hours', '?')} | {est['full_hours']:,} |",
            f"| Basins | {estimates.get('pilot_basins', '?')} | {est['full_basins']:,} |",
            f"| Raw bytes/hour | {est['raw_bytes_per_hour_observed_human']} | — |",
            f"| Total raw download | — | {est['estimated_raw_download_human']} |",
            f"| Parquet bytes/basin-hour | {est['parquet_bytes_per_basin_hour_observed']:.1f} B | — |",
            f"| Total processed Parquet | — | {est['estimated_processed_parquet_human']} |",
            f"| Output rows | — | {est['estimated_output_rows']:,} |",
            f"| Serial processing time | {est['median_processing_s_per_hour_observed']:.3f}s/hour | {est['estimated_serial_wall_clock_human']} |",
            "",
            "### HPC Wall-Clock Estimates",
            "",
            f"| Tasks | Estimated time |",
            f"|-------|---------------|",
        ]
        for n, human in est["estimated_hpc_wall_clock_human"].items():
            lines.append(f"| {n} | {human} |")
        lines += [
            "",
            f"*Raw assumption*: {est['raw_scale_assumption']}",
            f"*Parquet assumption*: {est['parquet_scale_assumption']}",
            f"*HPC assumption*: {est['hpc_assumption']}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# QC plots
# ---------------------------------------------------------------------------

def _make_qc_plots(
    qc_dir: Path,
    metrics_rows: list[dict],
    combined_df: Any,
    pilot_manifest: Any,
    scaling_estimates: dict[str, Any],
    products: list[str],
) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as exc:
        LOGGER.warning("matplotlib/numpy/pandas unavailable — skipping QC plots: %s", exc)
        return []

    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT

    qc_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    mdf = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()

    # --- 1. Hourly availability timeline ---
    try:
        fig, axes = plt.subplots(len(products), 1, figsize=(14, 3 * len(products)), squeeze=False)
        for ax, prod in zip(axes[:, 0], products):
            pm = mdf[mdf["product"] == prod] if len(mdf) else pd.DataFrame()
            if len(pm) > 0:
                hours = pd.to_datetime(pm["valid_time_utc"], utc=True, errors="coerce")
                ok  = pm["status"] == "success"
                bad = ~ok
                ax.scatter(hours[ok],  [1] * ok.sum(),  c="steelblue", s=4,  label=f"OK ({ok.sum()})")
                ax.scatter(hours[bad], [1] * bad.sum(), c="red",       s=8,
                           marker="x", label=f"Missing/failed ({bad.sum()})")
                ax.set_xlim(hours.min(), hours.max())
                ax.set_yticks([])
                ax.legend(fontsize=8)
            ax.set_title(f"{prod} — hourly availability", fontsize=10)
            ax.set_xlabel("Date (UTC)")
        fig.tight_layout()
        out = qc_dir / "hourly_availability.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Availability plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 2. Runtime by hour/product ---
    try:
        if len(mdf) > 0 and "total_processing_time_s" in mdf.columns:
            fig, axes = plt.subplots(len(products), 1, figsize=(14, 3 * len(products)), squeeze=False)
            for ax, prod in zip(axes[:, 0], products):
                pm = mdf[(mdf["product"] == prod) & (mdf["status"] == "success")]
                if len(pm) > 0:
                    hrs = pd.to_datetime(pm["valid_time_utc"], utc=True, errors="coerce")
                    vals = pm["total_processing_time_s"].values
                    ax.plot(hrs, vals, ".", markersize=3, alpha=0.6)
                    med = float(np.nanmedian(vals))
                    ax.axhline(med, ls="--", c="red", lw=1, label=f"median {med:.2f}s")
                    ax.set_ylabel("Processing time (s)")
                    ax.legend(fontsize=8)
                ax.set_title(f"{prod} — processing time per hour", fontsize=10)
            fig.tight_layout()
            out = qc_dir / "runtime_by_hour.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Runtime plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 3. Raw file size by hour ---
    try:
        if len(mdf) > 0 and "raw_file_size_bytes" in mdf.columns:
            fig, axes = plt.subplots(len(products), 1, figsize=(14, 3 * len(products)), squeeze=False)
            for ax, prod in zip(axes[:, 0], products):
                pm = mdf[(mdf["product"] == prod) & (mdf["status"] == "success")].copy()
                if len(pm) > 0:
                    hrs = pd.to_datetime(pm["valid_time_utc"], utc=True, errors="coerce")
                    size_mb = pm["raw_file_size_bytes"].values / 1e6
                    ax.plot(hrs, size_mb, ".", markersize=3, alpha=0.6)
                    med = float(np.nanmedian(size_mb))
                    ax.axhline(med, ls="--", c="red", lw=1, label=f"median {med:.1f} MB")
                    ax.set_ylabel("File size (MB)")
                    ax.legend(fontsize=8)
                ax.set_title(f"{prod} — raw file size per hour", fontsize=10)
            fig.tight_layout()
            out = qc_dir / "raw_file_size_by_hour.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
    except Exception as exc:
        LOGGER.warning("File size plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 4. Cumulative volume ---
    try:
        if len(mdf) > 0:
            prod_list = [p for p in [_MRMS_PRODUCT, _RTMA_PRODUCT] if p in products]
            colors = {"mrms_qpe_1h_pass1": "steelblue", "rtma_conus_aws_2p5km": "orange"}
            fig, axes = plt.subplots(len(prod_list), 1, figsize=(12, 4 * len(prod_list)), squeeze=False)
            for ax, prod in zip(axes[:, 0], prod_list):
                pm = mdf[(mdf["product"] == prod) & (mdf["status"] == "success")].copy()
                if len(pm) > 0:
                    pm = pm.sort_values("valid_time_utc")
                    hrs = pd.to_datetime(pm["valid_time_utc"], utc=True, errors="coerce")
                    c = colors.get(prod, "gray")
                    if "raw_file_size_bytes" in pm.columns:
                        cum_raw = pm["raw_file_size_bytes"].fillna(0).cumsum() / 1e9
                        ax.plot(hrs, cum_raw, label="Raw cache (GB)", color=c)
                    if "output_parquet_bytes" in pm.columns:
                        cum_pq = pm["output_parquet_bytes"].fillna(0).cumsum() / 1e9
                        ax.plot(hrs, cum_pq, label="Processed Parquet (GB)", color=c, ls="--")
                    ax.set_ylabel("GB")
                    ax.legend(fontsize=8)
                ax.set_title(f"{prod} — cumulative volume", fontsize=9)
            fig.tight_layout()
            out = qc_dir / "cumulative_volume.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Cumulative volume plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 5. Representative basin timeseries ---
    try:
        if combined_df is not None and len(combined_df) > 0:
            all_staids = combined_df["STAID"].unique()
            rep_staids = list(all_staids[:3]) if len(all_staids) >= 3 else list(all_staids)
            if rep_staids:
                fig, axes = plt.subplots(len(rep_staids), 2, figsize=(14, 3 * len(rep_staids)), squeeze=False)
                for i, staid in enumerate(rep_staids):
                    ax_m = axes[i, 0]
                    mrms_b = combined_df[
                        (combined_df["STAID"] == staid) & (combined_df["product"] == _MRMS_PRODUCT)
                    ].sort_values("valid_time_utc")
                    if len(mrms_b) > 0:
                        ts = pd.to_datetime(mrms_b["valid_time_utc"], utc=True, errors="coerce")
                        ax_m.plot(ts, mrms_b["weighted_mean"].values, lw=0.8)
                    ax_m.set_title(f"STAID {staid} — MRMS QPE weighted_mean [mm]", fontsize=8)
                    ax_m.set_ylabel("mm")

                    ax_t = axes[i, 1]
                    rtma_tmp = combined_df[
                        (combined_df["STAID"] == staid) &
                        (combined_df["product"] == _RTMA_PRODUCT) &
                        (combined_df["variable"].isin(["2t", "t2m"]))
                    ].sort_values("valid_time_utc")
                    if len(rtma_tmp) > 0:
                        ts2 = pd.to_datetime(rtma_tmp["valid_time_utc"], utc=True, errors="coerce")
                        ax_t.plot(ts2, rtma_tmp["weighted_mean"].values, lw=0.8, color="orange")
                    ax_t.set_title(f"STAID {staid} — RTMA 2m temperature weighted_mean [K]", fontsize=8)
                    ax_t.set_ylabel("K")

                fig.suptitle("Representative basin timeseries — January 2023", fontsize=10)
                fig.tight_layout()
                out = qc_dir / "representative_timeseries.png"
                fig.savefig(out, dpi=120)
                plt.close(fig)
                written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Timeseries plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 6. RTMA variable completeness ---
    try:
        if combined_df is not None and len(combined_df) > 0:
            rtma_data = combined_df[combined_df["product"] == _RTMA_PRODUCT]
            n_basins = combined_df["STAID"].nunique()
            n_hours_rtma = rtma_data["valid_time_utc"].nunique()
            if len(rtma_data) > 0 and n_basins > 0 and n_hours_rtma > 0:
                var_counts = rtma_data.groupby("variable")["STAID"].count()
                expected_per_var = n_hours_rtma * n_basins
                completeness = (var_counts / expected_per_var * 100).clip(0, 100).sort_values()
                fig, ax = plt.subplots(figsize=(10, max(4, len(completeness) * 0.5)))
                bars = ax.barh(completeness.index, completeness.values, color="steelblue")
                ax.set_xlabel("Completeness (%)")
                ax.set_xlim(0, 115)
                ax.axvline(100, ls="--", c="red", lw=1, alpha=0.5, label="100%")
                for bar, val in zip(bars, completeness.values):
                    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", fontsize=8)
                ax.set_title("RTMA variable completeness (% of expected basin-hours)", fontsize=10)
                ax.legend(fontsize=8)
                fig.tight_layout()
                out = qc_dir / "variable_completeness_rtma.png"
                fig.savefig(out, dpi=120)
                plt.close(fig)
                written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Variable completeness plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 7. Basin completeness distribution ---
    try:
        if combined_df is not None and len(combined_df) > 0:
            prod_list = [p for p in [_MRMS_PRODUCT, _RTMA_PRODUCT] if p in products]
            fig, axes = plt.subplots(1, len(prod_list), figsize=(6 * len(prod_list), 4), squeeze=False)
            for ax, prod in zip(axes[0, :], prod_list):
                pd_ = combined_df[combined_df["product"] == prod]
                if len(pd_) > 0:
                    n_h = pd_["valid_time_utc"].nunique()
                    counts = pd_.groupby("STAID")["valid_time_utc"].nunique()
                    pct = (counts / max(n_h, 1) * 100).clip(0, 100)
                    ax.hist(pct, bins=20, edgecolor="k", alpha=0.7)
                    ax.set_xlabel("Hour completeness (%)")
                    ax.set_ylabel("Basin count")
                    ax.axvline(100, ls="--", c="red", lw=1)
                ax.set_title(f"{prod}\nbasin completeness", fontsize=9)
            fig.tight_layout()
            out = qc_dir / "basin_completeness_distribution.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Basin completeness plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # --- 8. Full-dataset storage estimates ---
    try:
        prods_est = list(scaling_estimates.get("products", {}).keys())
        if prods_est:
            raw_gb = [scaling_estimates["products"][p]["estimated_raw_download_bytes"] / 1e9 for p in prods_est]
            pq_gb  = [scaling_estimates["products"][p]["estimated_processed_parquet_bytes"] / 1e9 for p in prods_est]
            x = np.arange(len(prods_est))
            w = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - w/2, raw_gb, w, label="Raw download cache (GB)", color="steelblue")
            ax.bar(x + w/2, pq_gb,  w, label="Processed Parquet (GB)",  color="orange")
            ax.set_xticks(x)
            ax.set_xticklabels(prods_est, rotation=15, ha="right", fontsize=8)
            ax.set_ylabel("GB")
            ax.set_title(
                f"Estimated full-dataset storage\n"
                f"({_FULL_BASINS:,} basins, 2020–2025, {_FULL_HOURS:,} hours)",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            for xi, (r, q) in enumerate(zip(raw_gb, pq_gb)):
                if r > 0:
                    ax.text(xi - w/2, r, _format_bytes(r * 1e9), ha="center", va="bottom", fontsize=7)
                if q > 0:
                    ax.text(xi + w/2, q, _format_bytes(q * 1e9), ha="center", va="bottom", fontsize=7)
            fig.tight_layout()
            out = qc_dir / "full_dataset_storage_estimate.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Storage estimate plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    return written


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _run_validation(
    mrms_df: Any,
    rtma_df: Any,
    combined_df: Any,
    pilot_staids: list[str],
    products: list[str],
    ok_mrms_hours: int,
    ok_rtma_hours: int,
    output_paths: dict[str, Path],
    prov_dir: Path,
) -> dict[str, Any]:
    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT

    n_pilots = len(pilot_staids)
    rtma_vars = set(rtma_df["variable"].unique()) if len(rtma_df) > 0 else set()

    checks: dict[str, Any] = {}

    if _MRMS_PRODUCT in products:
        exp_mrms = ok_mrms_hours * n_pilots
        checks["mrms_extracted_hours_gt_zero"]      = ok_mrms_hours > 0
        checks["mrms_50_basins_per_ok_hour"]         = len(mrms_df) == exp_mrms
        checks["mrms_no_all_null_weighted_mean"]     = (
            not mrms_df["weighted_mean"].isna().all() if len(mrms_df) > 0 else False
        )
        checks["mrms_valid_weight_fraction_ok"]      = (
            bool((mrms_df["valid_weight_fraction"] > 0.5).all()) if len(mrms_df) > 0 else False
        )
        checks["mrms_parquet_written"]               = output_paths.get("mrms_parquet", Path("_")).exists()

    if _RTMA_PRODUCT in products:
        exp_rtma = ok_rtma_hours * n_pilots * 11
        checks["rtma_extracted_hours_gt_zero"]       = ok_rtma_hours > 0
        checks["rtma_10wdir_absent"]                 = "10wdir" not in rtma_vars
        checks["rtma_orog_absent"]                   = "orog"   not in rtma_vars
        checks["rtma_11_variables"]                  = len(rtma_vars) == 11 if len(rtma_df) > 0 else False
        checks["rtma_50_basins_x_11_vars_per_ok_hour"] = len(rtma_df) == exp_rtma
        checks["rtma_no_all_null_weighted_mean"]     = (
            not rtma_df["weighted_mean"].isna().all() if len(rtma_df) > 0 else False
        )
        checks["rtma_parquet_written"]               = output_paths.get("rtma_parquet", Path("_")).exists()

    checks["combined_parquet_written"]              = output_paths.get("combined_parquet", Path("_")).exists()
    checks["hourly_runtime_volume_csv_written"]     = (prov_dir / "hourly_runtime_and_volume.csv").exists()
    checks["product_summary_csv_written"]           = (prov_dir / "product_runtime_volume_summary.csv").exists()
    checks["scaling_estimates_json_written"]        = (prov_dir / "full_dataset_scaling_estimates.json").exists()
    checks["generated_outputs_not_staged"]          = True  # outputs are under data_root, not repo

    return checks


# ---------------------------------------------------------------------------
# Live progress helpers
# ---------------------------------------------------------------------------

def _write_live_progress(
    prov_dir: "Path",
    run_start_utc: "datetime",
    total_hours_scheduled: int,
    products: list,
    prog: dict,
) -> None:
    import csv as _csv
    now = datetime.utcnow()
    elapsed_s = max(0.0, (now - run_start_utc).total_seconds())
    total_hp = total_hours_scheduled * len(products)
    completed_hp = prog["mrms_ok"] + prog["rtma_ok"]
    pct = completed_hp / max(total_hp, 1) * 100
    est_rem = (elapsed_s / pct * (100 - pct)) if pct > 0 else None

    progress_data: dict = {
        "run_start_time":              run_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "current_time_utc":            now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_hours_scheduled":       total_hours_scheduled,
        "completed_hour_products":     completed_hp,
        "completed_mrms_hours":        prog["mrms_ok"],
        "completed_rtma_hours":        prog["rtma_ok"],
        "failed_hour_products":        prog["failed"],
        "percent_complete":            round(pct, 2),
        "elapsed_seconds":             round(elapsed_s, 1),
        "estimated_remaining_seconds": round(est_rem, 1) if est_rem is not None else None,
        "download_workers":            prog.get("download_workers"),
        "rtma_mode":                   prog.get("rtma_mode"),
        "files_downloaded":            prog["files_downloaded"],
        "files_reused":                prog["files_reused"],
        "bytes_downloaded_total":      prog["bytes_downloaded"],
        "latest_completed_hour":       prog["latest_hour"],
        "latest_status_message":       prog["latest_msg"],
        "latest_error_message":        prog["latest_err"],
    }

    json_path = prov_dir / "live_progress.json"
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(progress_data, fh, indent=2)
            fh.flush()
    except Exception:
        pass

    csv_path = prov_dir / "live_progress.csv"
    csv_fields = list(progress_data.keys())
    write_header = not csv_path.exists()
    try:
        with open(csv_path, "a", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=csv_fields)
            if write_header:
                w.writeheader()
            w.writerow(progress_data)
            fh.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_wall_start = time.perf_counter()
    _run_start_utc = datetime.utcnow()
    args = _parse_args()

    from src.pipeline.config import load_config, config_to_dict
    from src.pipeline.geometries import normalise_staid
    from src.pipeline.provenance import write_run_manifest
    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT
    import pandas as pd
    import numpy as np

    # --- Config and paths ---
    cfg       = load_config(Path(args.config))
    data_root = cfg.effective_data_root(override=args.data_root)
    LOGGER.info("Data root: %s", data_root)

    pilot_start_str = args.start or cfg.pilot.time_start
    pilot_end_str   = args.end   or cfg.pilot.time_end
    pilot_start = datetime.fromisoformat(pilot_start_str.replace("Z", "")).replace(tzinfo=None)
    pilot_end   = datetime.fromisoformat(pilot_end_str.replace("Z",   "")).replace(tzinfo=None)

    product_map = {"mrms": _MRMS_PRODUCT, "rtma": _RTMA_PRODUCT}
    products: list[str] = [
        product_map.get(p.strip().lower(), p.strip().lower())
        for p in args.products.split(",")
    ]
    LOGGER.info("Products: %s", products)

    all_hours = list(_iter_hours(pilot_start, pilot_end))
    if args.max_hours is not None:
        all_hours = all_hours[:args.max_hours]
    n_hours = len(all_hours)
    LOGGER.info(
        "Hour schedule: %d hours  %s → %s",
        n_hours, all_hours[0].isoformat(), all_hours[-1].isoformat(),
    )

    raw_dir           = cfg.output_dir("raw",              data_root)
    weights_base      = cfg.output_dir("basin_geometries", data_root) / "weights"
    mrms_weights_path = weights_base / "mrms" / "pilot_mrms_weights.parquet"
    rtma_weights_path = weights_base / "rtma" / "pilot_rtma_weights.parquet"
    manifest_csv      = cfg.output_dir("manifests", data_root) / "stage1_pilot" / "pilot_basin_manifest.csv"
    timeseries_dir    = cfg.output_dir("basin_timeseries", data_root) / "stage1_pilot" / "january_2023"
    qc_dir            = cfg.output_dir("qc_reports",       data_root) / "stage1_pilot" / "january_2023_extraction"
    prov_dir          = cfg.output_dir("manifests",        data_root) / "stage1_pilot" / "january_2023_extraction"
    staging_dir       = cfg.output_dir("tmp",              data_root) / "january_2023_staging"

    # --- Prerequisite checks ---
    for p, label in [
        (mrms_weights_path, "MRMS weight table"),
        (rtma_weights_path, "RTMA weight table"),
        (manifest_csv,      "Pilot basin manifest"),
    ]:
        if not p.exists():
            LOGGER.error("Required input missing: %s — %s", label, p)
            return 1

    # --- Load manifest and weights ---
    pilot_manifest = pd.read_csv(manifest_csv)
    pilot_staids   = [normalise_staid(s) for s in pilot_manifest["STAID"].tolist()]
    LOGGER.info("Pilot basins: %d", len(pilot_staids))

    mrms_weights = pd.read_parquet(mrms_weights_path)
    rtma_weights = pd.read_parquet(rtma_weights_path)

    # --- Initialise datasources and list S3 objects (once per product) ---
    from src.datasources.mrms import MrmsAwsQpe1hPass1
    from src.datasources.rtma import RtmaAwsConusDataSource
    from src.datasources.base import CONUS_BBOX

    mrms_s3_map: dict[datetime, Any] = {}
    rtma_s3_map: dict[datetime, Any] = {}
    mrms_ds: Optional[Any] = None
    rtma_ds: Optional[Any] = None

    if _MRMS_PRODUCT in products:
        LOGGER.info("Listing MRMS S3 objects for range ...")
        mrms_ds = MrmsAwsQpe1hPass1(download_concurrency=1)
        try:
            objs = mrms_ds.list_sample_objects(all_hours[0], all_hours[-1], CONUS_BBOX, ["precip"])
            mrms_s3_map = {o.datetime.replace(tzinfo=None): o for o in objs}
            LOGGER.info("MRMS S3 objects found: %d", len(mrms_s3_map))
        except Exception as exc:
            LOGGER.error("MRMS S3 listing failed: %s", exc)
            return 1

    if _RTMA_PRODUCT in products:
        LOGGER.info("Listing RTMA S3 objects for range ...")
        rtma_ds = RtmaAwsConusDataSource(download_concurrency=1)
        try:
            objs = rtma_ds.list_sample_objects(
                all_hours[0], all_hours[-1], CONUS_BBOX, ["TMP", "UGRD", "VGRD", "PRES"]
            )
            rtma_s3_map = {o.datetime.replace(tzinfo=None): o for o in objs}
            LOGGER.info("RTMA S3 objects found: %d", len(rtma_s3_map))
        except Exception as exc:
            LOGGER.error("RTMA S3 listing failed: %s", exc)
            return 1

    # --- Build local cache indices (O(n_files) once, not O(n_hours)) ---
    LOGGER.info("Building local cache index ...")
    mrms_cache = _build_mrms_cache_index(raw_dir)
    rtma_cache = _build_rtma_cache_index(raw_dir)
    LOGGER.info("Cache: MRMS=%d files, RTMA=%d files", len(mrms_cache), len(rtma_cache))

    # --- Create directories ---
    for d in [timeseries_dir, qc_dir, prov_dir,
              staging_dir / "mrms", staging_dir / "rtma"]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Live log file handler ---
    _log_path = prov_dir / "live_run.log"
    _fh = logging.FileHandler(str(_log_path), mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(_fh)
    LOGGER.info("Live run log: %s", _log_path)

    # --- Load existing completion status and metrics for resume ---
    existing_success: set[tuple[str, str]] = set()  # (valid_time_utc, product)
    metrics_csv_path = prov_dir / "hourly_runtime_and_volume.csv"
    # Pre-populate metrics_rows with previous run data so product_summary is correct on resume
    metrics_rows: list[dict] = []
    if args.resume and metrics_csv_path.exists():
        try:
            prev = pd.read_csv(metrics_csv_path)
            metrics_rows = prev.to_dict("records")
            for row in metrics_rows:
                if str(row.get("status")) == "success":
                    existing_success.add((str(row["valid_time_utc"]), str(row["product"])))
            LOGGER.info("Resume: loaded %d rows; %d previously successful hour-products",
                        len(metrics_rows), len(existing_success))
        except Exception as exc:
            LOGGER.warning("Cannot read existing metrics CSV: %s — starting fresh", exc)
            metrics_rows = []

    # --- Progress state (includes pre-existing resume counts) ---
    _prog: dict = {
        "mrms_ok":          sum(1 for _, p in existing_success if p == _MRMS_PRODUCT),
        "rtma_ok":          sum(1 for _, p in existing_success if p == _RTMA_PRODUCT),
        "failed":           0,
        "files_downloaded": 0,
        "files_reused":     0,
        "bytes_downloaded": 0,
        "latest_hour":      None,
        "latest_msg":       "starting",
        "latest_err":       None,
        "download_workers": args.download_workers,
        "rtma_mode":        args.rtma_mode,
    }
    _write_live_progress(prov_dir, _run_start_utc, n_hours, products, _prog)

    # ---------------------------------------------------------------------------
    # RTMA parallel prefetch (I/O phase separated from serial decode/extract)
    # ---------------------------------------------------------------------------

    # Record which files are pre-existing BEFORE prefetch, for accurate file_reused flag.
    rtma_cache_initial: set = set(rtma_cache.keys())

    # prefetch_download_times[dt] = seconds spent downloading (only for newly downloaded files)
    rtma_prefetch_times: dict = {}

    if _RTMA_PRODUCT in products and args.download_workers >= 1:
        # Identify hours that need downloading: not resumed, not cached, available in S3.
        to_prefetch = {}
        for dt in all_hours:
            dt_str_p = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            if args.resume and (dt_str_p, _RTMA_PRODUCT) in existing_success:
                continue  # will load from staging; no raw file needed
            if rtma_cache.get(dt) is not None:
                continue  # already on disk
            obj = rtma_s3_map.get(dt)
            if obj is None:
                continue  # missing from S3; recorded as missing in main loop
            to_prefetch[dt] = obj

        if to_prefetch:
            n_workers = max(1, args.download_workers)
            prefetch_results = _prefetch_rtma_files(
                to_prefetch, rtma_ds, raw_dir, args.rtma_mode, n_workers
            )
            for dt, (path, dl_s) in prefetch_results.items():
                if path is not None:
                    rtma_cache[dt] = path         # add to cache for main-loop lookup
                    _prog["files_downloaded"] += 1
                    _prog["bytes_downloaded"] += path.stat().st_size if path.exists() else 0
                rtma_prefetch_times[dt] = dl_s   # record per-hour download time
        else:
            LOGGER.info("RTMA prefetch: nothing to download (all files cached or resumed)")

    # ---------------------------------------------------------------------------
    # Main hourly extraction loop  (serial: decode, extract, write)
    # ---------------------------------------------------------------------------

    all_mrms_frames: list[pd.DataFrame] = []
    all_rtma_frames: list[pd.DataFrame] = []
    # metrics_rows pre-populated above from existing CSV (empty list if not resuming)
    missing_rows:    list[dict]         = []

    for i, dt in enumerate(all_hours):
        dt_str   = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        hour_tag = dt.strftime("%Y%m%d%H")
        _hr_mrms_ok = 0; _hr_rtma_ok = 0; _hr_failed = 0  # per-hour product counters

        if i == 0 or (i + 1) % 24 == 0 or i == n_hours - 1:
            pct = (i + 1) / n_hours * 100
            LOGGER.info("[%3.0f%%] hour %d/%d  %s", pct, i + 1, n_hours, dt_str)

        for product in products:
            is_mrms     = product == _MRMS_PRODUCT
            staging_pq  = staging_dir / ("mrms" if is_mrms else "rtma") / f"{hour_tag}.parquet"
            resume_key  = (dt_str, product)

            # --- Resume: load from staging if already successful ---
            if args.resume and resume_key in existing_success and staging_pq.exists():
                try:
                    hour_df = pd.read_parquet(staging_pq)
                    if is_mrms:
                        all_mrms_frames.append(hour_df)
                    else:
                        all_rtma_frames.append(hour_df)
                    # Already counted in _prog initial state from existing_success; don't re-add
                    continue
                except Exception:
                    pass  # Fall through and re-process

            # --- Locate raw file; handle inline download for MRMS and fallback ---
            s3_obj       = mrms_s3_map.get(dt) if is_mrms else rtma_s3_map.get(dt)
            download_s   = 0.0

            if is_mrms:
                raw_path    = mrms_cache.get(dt)
                file_reused = raw_path is not None
            else:
                raw_path    = rtma_cache.get(dt)
                # file_reused = True only if it was on disk BEFORE this run's prefetch
                file_reused = dt in rtma_cache_initial
                # Attribute per-hour download time recorded during prefetch
                download_s  = rtma_prefetch_times.get(dt, 0.0)

            if raw_path is None:
                if s3_obj is None:
                    metrics_rows.append({
                        "product":                  product,
                        "valid_time_utc":           dt_str,
                        "raw_file_path":            None,
                        "raw_file_size_bytes":      None,
                        "file_reused":              False,
                        "download_time_s":          None,
                        "decode_time_s":            None,
                        "extraction_time_s":        None,
                        "write_time_s":             None,
                        "total_processing_time_s":  None,
                        "n_output_rows":            0,
                        "output_parquet_bytes":     None,
                        "status":                   "missing_s3",
                        "warning_message":          f"No S3 object for {dt_str}",
                    })
                    missing_rows.append({
                        "product": product, "valid_time_utc": dt_str, "reason": "not_in_s3",
                    })
                    _hr_failed += 1
                    continue

                # Inline serial download fallback:
                #   MRMS: always inline (tiny files, no prefetch implemented)
                #   RTMA: only if prefetch was skipped (workers=0, or prefetch failure)
                if is_mrms:
                    raw_path, download_s = _download_single_mrms(mrms_ds, s3_obj, raw_dir)
                else:
                    raw_path, download_s = _download_single_rtma(
                        rtma_ds, s3_obj, raw_dir, mode=args.rtma_mode
                    )

                if raw_path is None:
                    metrics_rows.append({
                        "product":                  product,
                        "valid_time_utc":           dt_str,
                        "raw_file_path":            None,
                        "raw_file_size_bytes":      None,
                        "file_reused":              False,
                        "download_time_s":          round(download_s, 3),
                        "decode_time_s":            None,
                        "extraction_time_s":        None,
                        "write_time_s":             None,
                        "total_processing_time_s":  round(download_s, 3),
                        "n_output_rows":            0,
                        "output_parquet_bytes":     None,
                        "status":                   "download_failed",
                        "warning_message":          f"Download failed in {download_s:.1f}s",
                    })
                    missing_rows.append({
                        "product": product, "valid_time_utc": dt_str, "reason": "download_failed",
                    })
                    _hr_failed += 1
                    continue

                if is_mrms:
                    mrms_cache[dt] = raw_path
                else:
                    rtma_cache[dt] = raw_path

            raw_size_bytes = raw_path.stat().st_size if raw_path.exists() else 0
            if file_reused:
                _prog["files_reused"] += 1
            elif is_mrms:
                _prog["files_downloaded"] += 1
                _prog["bytes_downloaded"] += raw_size_bytes

            # --- Decode + extract ---
            t_proc        = time.perf_counter()
            decode_s      = 0.0
            extract_s     = 0.0
            write_s       = 0.0
            n_rows        = 0
            staging_bytes = 0
            status        = "success"
            warn_msg: Optional[str] = None
            hour_df: Optional[pd.DataFrame] = None

            try:
                if is_mrms:
                    hour_df, decode_s, extract_s = _process_hour_mrms(
                        raw_path, mrms_weights, pilot_staids, mrms_weights_path
                    )
                else:
                    hour_df, decode_s, extract_s, _, _ = _process_hour_rtma(
                        raw_path, rtma_weights, pilot_staids, rtma_weights_path
                    )

                # Write per-hour staging parquet
                t_w = time.perf_counter()
                if hour_df is not None and len(hour_df) > 0:
                    hour_df.to_parquet(staging_pq, index=False)
                    staging_bytes = staging_pq.stat().st_size
                    n_rows = len(hour_df)
                else:
                    status = "empty"
                write_s = time.perf_counter() - t_w

                if hour_df is not None and len(hour_df) > 0:
                    if is_mrms:
                        all_mrms_frames.append(hour_df)
                    else:
                        all_rtma_frames.append(hour_df)

            except Exception as exc:
                warn_msg = str(exc)
                LOGGER.warning("Error at %s %s: %s", product, dt_str, exc)
                status = "decode_extract_error"

            total_proc_s = time.perf_counter() - t_proc + download_s

            metrics_rows.append({
                "product":                  product,
                "valid_time_utc":           dt_str,
                "raw_file_path":            str(raw_path),
                "raw_file_size_bytes":      raw_size_bytes,
                "file_reused":              file_reused,
                "download_time_s":          round(download_s, 3),
                "decode_time_s":            round(decode_s, 3),
                "extraction_time_s":        round(extract_s, 3),
                "write_time_s":             round(write_s, 3),
                "total_processing_time_s":  round(total_proc_s, 3),
                "n_output_rows":            n_rows,
                "output_parquet_bytes":     staging_bytes,
                "status":                   status,
                "warning_message":          warn_msg,
            })
            if status == "success":
                if is_mrms:
                    _hr_mrms_ok += 1
                else:
                    _hr_rtma_ok += 1
            else:
                _hr_failed += 1
                if warn_msg:
                    _prog["latest_err"] = warn_msg

        # --- Accumulate into live progress and write once per outer hour ---
        _prog["mrms_ok"] += _hr_mrms_ok
        _prog["rtma_ok"] += _hr_rtma_ok
        _prog["failed"]  += _hr_failed
        _prog["latest_hour"] = dt_str
        _prog["latest_msg"]  = (
            f"hour {dt_str}: MRMS={'ok' if _hr_mrms_ok else 'fail'} "
            f"RTMA={'ok' if _hr_rtma_ok else 'fail'}"
        )
        _write_live_progress(prov_dir, _run_start_utc, n_hours, products, _prog)

        if (i + 1) % 24 == 0 or i == n_hours - 1:
            _n_done = _prog["mrms_ok"] + _prog["rtma_ok"] + _prog["failed"]
            _n_tot  = n_hours * len(products)
            _el_s   = time.perf_counter() - t_wall_start
            _pct    = _n_done / max(_n_tot, 1) * 100
            _eta_s  = int(_el_s / _pct * (100 - _pct)) if _pct > 0 else 0
            _gb     = _prog["bytes_downloaded"] / 1e9
            print(
                f"{_n_done}/{_n_tot} hour-products complete "
                f"| MRMS {_prog['mrms_ok']}/{n_hours} "
                f"| RTMA {_prog['rtma_ok']}/{n_hours} "
                f"| {_pct:.1f}% "
                f"| elapsed {_fmt_hms(int(_el_s))} "
                f"| ETA {_fmt_hms(_eta_s)} "
                f"| downloaded {_gb:.1f} GB "
                f"| failures {_prog['failed']}",
                flush=True,
            )

    # ---------------------------------------------------------------------------
    # Combine staging into final output parquets
    # ---------------------------------------------------------------------------

    LOGGER.info("Combining staging parquets ...")
    mrms_df     = pd.concat(all_mrms_frames, ignore_index=True) if all_mrms_frames else pd.DataFrame()
    rtma_df     = pd.concat(all_rtma_frames, ignore_index=True) if all_rtma_frames else pd.DataFrame()
    combined_df = pd.concat(
        [df for df in [mrms_df, rtma_df] if len(df) > 0], ignore_index=True
    )

    output_paths: dict[str, Path] = {}
    for df_out, fname, key in [
        (mrms_df,     "mrms_hourly_basin_stats.parquet",     "mrms_parquet"),
        (rtma_df,     "rtma_hourly_basin_stats.parquet",     "rtma_parquet"),
        (combined_df, "combined_hourly_basin_stats.parquet", "combined_parquet"),
    ]:
        if len(df_out) > 0:
            p = timeseries_dir / fname
            df_out.to_parquet(p, index=False)
            output_paths[key] = p
            LOGGER.info("Written: %s (%d rows, %.1f MB)",
                        fname, len(df_out), p.stat().st_size / 1e6)

    # Preview CSVs (first 5 rows)
    for df_out, fname in [(mrms_df, "preview_mrms.csv"), (rtma_df, "preview_rtma.csv")]:
        if len(df_out) > 0:
            try:
                df_out.head(5).to_csv(timeseries_dir / fname, index=False)
            except Exception:
                pass

    # ---------------------------------------------------------------------------
    # Write metrics and summary CSVs
    # ---------------------------------------------------------------------------

    # hourly_runtime_and_volume.csv
    metrics_df = pd.DataFrame(metrics_rows)
    if len(metrics_df) > 0:
        metrics_df.to_csv(prov_dir / "hourly_runtime_and_volume.csv", index=False)
        output_paths["hourly_runtime_volume_csv"] = prov_dir / "hourly_runtime_and_volume.csv"

    # hourly_file_status.csv
    status_rows = []
    for dt in all_hours:
        dt_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        for product in products:
            matches = [r for r in metrics_rows
                       if r["valid_time_utc"] == dt_str and r["product"] == product]
            s = matches[0]["status"] if matches else "not_processed"
            status_rows.append({"valid_time_utc": dt_str, "product": product, "status": s})
    pd.DataFrame(status_rows).to_csv(prov_dir / "hourly_file_status.csv", index=False)

    # missing_files.csv (if any)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(prov_dir / "missing_files.csv", index=False)
        LOGGER.warning("%d missing/failed hour-products written to missing_files.csv", len(missing_rows))

    # Product-level summary
    product_summary: dict[str, dict] = {}
    for product in products:
        pm_all = [r for r in metrics_rows if r["product"] == product]
        pm_ok  = [r for r in pm_all if r["status"] == "success"]

        raw_sizes  = [r["raw_file_size_bytes"] for r in pm_ok if r.get("raw_file_size_bytes")]
        dl_times   = [r["download_time_s"]   for r in pm_ok
                      if r.get("download_time_s") and r["download_time_s"] > 0]
        dec_times  = [r["decode_time_s"]     for r in pm_ok if r.get("decode_time_s")]
        ext_times  = [r["extraction_time_s"] for r in pm_ok if r.get("extraction_time_s")]
        wrt_times  = [r["write_time_s"]      for r in pm_ok if r.get("write_time_s")]
        tot_times  = [r["total_processing_time_s"] for r in pm_ok
                      if r.get("total_processing_time_s")]
        pq_bytes   = [r["output_parquet_bytes"] for r in pm_ok if r.get("output_parquet_bytes")]
        n_rows_l   = [r["n_output_rows"] for r in pm_ok]
        n_reused   = sum(1 for r in pm_ok if r.get("file_reused"))
        n_dl       = sum(1 for r in pm_ok if not r.get("file_reused"))

        total_wall = sum(tot_times)
        ok_h       = len(pm_ok)

        product_summary[product] = {
            "expected_hours":               n_hours,
            "successful_hours":             ok_h,
            "missing_failed_hours":         len(pm_all) - ok_h,
            "raw_downloaded_file_count":    n_dl,
            "raw_reused_file_count":        n_reused,
            "total_raw_bytes":              sum(raw_sizes),
            "median_raw_bytes_per_hour":    float(np.median(raw_sizes)) if raw_sizes else 0.0,
            "mean_raw_bytes_per_hour":      float(np.mean(raw_sizes))   if raw_sizes else 0.0,
            "total_output_parquet_bytes":   sum(pq_bytes),
            "output_bytes_per_basin_hour":  float(np.mean(pq_bytes)) / max(len(pilot_staids), 1)
                                            if pq_bytes else 0.0,
            "median_download_time_s":       float(np.median(dl_times))  if dl_times  else 0.0,
            "median_decode_time_s":         float(np.median(dec_times)) if dec_times else 0.0,
            "median_extraction_time_s":     float(np.median(ext_times)) if ext_times else 0.0,
            "median_write_time_s":          float(np.median(wrt_times)) if wrt_times else 0.0,
            "total_wall_clock_s":           total_wall,
            "median_total_processing_time_s": float(np.median(tot_times)) if tot_times else 0.0,
            "total_output_rows":            sum(n_rows_l),
            "throughput_hours_per_minute":  ok_h / (total_wall / 60) if total_wall > 0 else 0.0,
            "throughput_basin_hours_per_minute": (ok_h * len(pilot_staids)) / (total_wall / 60)
                                                 if total_wall > 0 else 0.0,
        }

    pd.DataFrame([{"product": k, **v} for k, v in product_summary.items()]).to_csv(
        prov_dir / "product_runtime_volume_summary.csv", index=False
    )
    output_paths["product_summary_csv"] = prov_dir / "product_runtime_volume_summary.csv"

    # variable_completeness.csv (RTMA)
    if len(rtma_df) > 0 and _RTMA_PRODUCT in products:
        ok_rtma_h = product_summary.get(_RTMA_PRODUCT, {}).get("successful_hours", 1)
        var_rows = []
        for var in sorted(rtma_df["variable"].unique()):
            vdf = rtma_df[rtma_df["variable"] == var]
            var_rows.append({
                "variable":         var,
                "n_rows":           len(vdf),
                "expected_rows":    ok_rtma_h * len(pilot_staids),
                "completeness_pct": len(vdf) / max(ok_rtma_h * len(pilot_staids), 1) * 100,
            })
        pd.DataFrame(var_rows).to_csv(prov_dir / "variable_completeness.csv", index=False)

    # basin_completeness.csv
    if len(combined_df) > 0:
        basin_rows = []
        for staid in pilot_staids:
            for product in products:
                ok_h = product_summary.get(product, {}).get("successful_hours", 1)
                pbd  = combined_df[
                    (combined_df["STAID"] == staid) & (combined_df["product"] == product)
                ]
                n_present = pbd["valid_time_utc"].nunique()
                basin_rows.append({
                    "STAID":             staid,
                    "product":           product,
                    "n_hours_present":   n_present,
                    "expected_hours":    ok_h,
                    "completeness_pct":  n_present / max(ok_h, 1) * 100,
                })
        pd.DataFrame(basin_rows).to_csv(prov_dir / "basin_completeness.csv", index=False)

    # ---------------------------------------------------------------------------
    # Scaling estimates
    # ---------------------------------------------------------------------------

    ok_mrms_hours = product_summary.get(_MRMS_PRODUCT, {}).get("successful_hours", 0)
    ok_rtma_hours = product_summary.get(_RTMA_PRODUCT, {}).get("successful_hours", 0)

    scaling_estimates = _compute_scaling_estimates(
        product_summary=product_summary,
        pilot_hours=n_hours,
        pilot_basins=len(pilot_staids),
    )

    # Flat CSV
    scaling_csv_rows = []
    for product, est in scaling_estimates.get("products", {}).items():
        flat: dict[str, Any] = {"product": product}
        for k, v in est.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}_{kk}"] = vv
            else:
                flat[k] = v
        scaling_csv_rows.append(flat)
    if scaling_csv_rows:
        pd.DataFrame(scaling_csv_rows).to_csv(
            prov_dir / "full_dataset_scaling_estimates.csv", index=False
        )
        output_paths["scaling_estimates_csv"] = prov_dir / "full_dataset_scaling_estimates.csv"

    with open(prov_dir / "full_dataset_scaling_estimates.json", "w", encoding="utf-8") as fh:
        json.dump(scaling_estimates, fh, indent=2, default=str)
    output_paths["scaling_estimates_json"] = prov_dir / "full_dataset_scaling_estimates.json"

    _write_scaling_md(prov_dir / "full_dataset_scaling_estimates.md", scaling_estimates)
    output_paths["scaling_estimates_md"] = prov_dir / "full_dataset_scaling_estimates.md"

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    validation = _run_validation(
        mrms_df=mrms_df, rtma_df=rtma_df, combined_df=combined_df,
        pilot_staids=pilot_staids, products=products,
        ok_mrms_hours=ok_mrms_hours, ok_rtma_hours=ok_rtma_hours,
        output_paths=output_paths, prov_dir=prov_dir,
    )
    all_pass = all(bool(v) for v in validation.values() if isinstance(v, bool))

    # ---------------------------------------------------------------------------
    # QC plots
    # ---------------------------------------------------------------------------

    plot_files = _make_qc_plots(
        qc_dir=qc_dir,
        metrics_rows=metrics_rows,
        combined_df=combined_df if len(combined_df) > 0 else None,
        pilot_manifest=pilot_manifest,
        scaling_estimates=scaling_estimates,
        products=products,
    )

    # ---------------------------------------------------------------------------
    # Provenance manifests
    # ---------------------------------------------------------------------------

    t_elapsed = time.perf_counter() - t_wall_start
    write_run_manifest(
        run_dir=prov_dir,
        run_command=" ".join(sys.argv),
        config_dict=config_to_dict(cfg),
        input_paths={
            "mrms_weight_table":    str(mrms_weights_path),
            "rtma_weight_table":    str(rtma_weights_path),
            "pilot_basin_manifest": str(manifest_csv),
        },
        output_paths={k: str(v) for k, v in output_paths.items()},
        validation_results=validation,
        extra={
            "pilot_period_start":       pilot_start.isoformat() + "Z",
            "pilot_period_end":         all_hours[-1].isoformat() + "Z",
            "n_hours_scheduled":        n_hours,
            "products_extracted":       products,
            "n_pilot_basins":           len(pilot_staids),
            "mrms_output_rows":         len(mrms_df),
            "rtma_output_rows":         len(rtma_df),
            "combined_output_rows":     len(combined_df),
            "mrms_successful_hours":    ok_mrms_hours,
            "rtma_successful_hours":    ok_rtma_hours,
            "n_missing_failed":         len(missing_rows),
            "product_summary":          product_summary,
            "scaling_estimates":        scaling_estimates,
            "qc_plots_written":         plot_files,
            "runtime_seconds":          round(t_elapsed, 2),
            "max_hours_flag":           args.max_hours,
            "resume_flag":              args.resume,
            "rtma_mode":                args.rtma_mode,
            "download_workers":         args.download_workers,
        },
    )

    # ---------------------------------------------------------------------------
    # Terminal report
    # ---------------------------------------------------------------------------

    sep = "=" * 72
    print(f"\n{sep}")
    print("Stage 1 Milestone 2D — January 2023 Extraction Report")
    print(sep)
    print(f"  Config          : {args.config}")
    print(f"  Data root       : {data_root}")
    print(f"  Period          : {pilot_start.isoformat()}Z -> {all_hours[-1].isoformat()}Z")
    print(f"  Hours scheduled : {n_hours}")
    if args.max_hours:
        print(f"  Max-hours flag  : {args.max_hours}  (smoke-test mode)")
    print(f"  Products        : {', '.join(products)}")
    print(f"  Pilot basins    : {len(pilot_staids)}")
    print()
    print("  Row counts:")
    if _MRMS_PRODUCT in products:
        exp_m = ok_mrms_hours * len(pilot_staids)
        print(f"    MRMS     : {len(mrms_df):>8,}  (expected {exp_m:,} = {ok_mrms_hours} ok-hours × {len(pilot_staids)} basins)")
    if _RTMA_PRODUCT in products:
        n_vars_rtma = rtma_df["variable"].nunique() if len(rtma_df) > 0 else 0
        exp_r = ok_rtma_hours * len(pilot_staids) * 11
        print(f"    RTMA     : {len(rtma_df):>8,}  (expected {exp_r:,} = {ok_rtma_hours} ok-hours × {len(pilot_staids)} basins × 11 vars)")
        print(f"    RTMA vars: {n_vars_rtma}  (expected 11)")
    print(f"    Combined : {len(combined_df):>8,}")
    print()
    print("  Per-product timing & volume summary:")
    for product, psum in product_summary.items():
        ok_h   = psum["successful_hours"]
        miss_h = psum["missing_failed_hours"]
        tot_r  = psum["total_raw_bytes"]
        tot_p  = psum["total_output_parquet_bytes"]
        med_t  = psum["median_total_processing_time_s"]
        thr    = psum["throughput_hours_per_minute"]
        print(f"    {product}:")
        print(f"      OK hours          : {ok_h}/{n_hours}  (missing/failed: {miss_h})")
        print(f"      Downloaded files  : {psum['raw_downloaded_file_count']}  "
              f"reused: {psum['raw_reused_file_count']}")
        print(f"      Total raw bytes   : {_format_bytes(tot_r)}")
        print(f"      Total Parquet     : {_format_bytes(tot_p)}")
        print(f"      Median proc time  : {med_t:.2f}s/hour")
        print(f"      Throughput        : {thr:.1f} hours/min")
        print(f"      Download  median  : {psum['median_download_time_s']:.3f}s")
        print(f"      Decode    median  : {psum['median_decode_time_s']:.3f}s")
        print(f"      Extract   median  : {psum['median_extraction_time_s']:.3f}s")
        print(f"      Write     median  : {psum['median_write_time_s']:.3f}s")
    print()
    print("  Scaling estimates (2,843 basins, 2020–2025):")
    for product, est in scaling_estimates.get("products", {}).items():
        print(f"    {product}:")
        print(f"      Full hours       : {est['full_hours']:,}")
        print(f"      Raw download est : {est['estimated_raw_download_human']}")
        print(f"      Parquet est      : {est['estimated_processed_parquet_human']}")
        print(f"      Output rows est  : {est['estimated_output_rows']:,}")
        print(f"      Serial wall-clock: {est['estimated_serial_wall_clock_human']}")
        for n, h in est["estimated_hpc_wall_clock_human"].items():
            print(f"      HPC {n:>3} tasks   : {h}")
    print()
    print(f"  Wall-clock total  : {_format_duration(t_elapsed)}")
    print(f"  Outputs:")
    for k, v in output_paths.items():
        print(f"    {k}: {v}")
    print()
    print(f"  QC plots ({len(plot_files)}):")
    for pf in plot_files:
        print(f"    {qc_dir / pf}")
    print()
    print("  Validation:")
    for k, v in validation.items():
        tag = "PASS" if v is True else "FAIL" if v is False else "----"
        print(f"    {tag}  {k}")
    print()
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'}")

    if missing_rows:
        print()
        print(f"  Missing/failed hours ({len(missing_rows)}):")
        for mr in missing_rows[:10]:
            print(f"    {mr['product']:30}  {mr['valid_time_utc']}  ({mr['reason']})")
        if len(missing_rows) > 10:
            print(f"    ... and {len(missing_rows) - 10} more (see missing_files.csv)")

    try:
        import subprocess
        gs = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=10)
        print()
        print("git status --short:")
        print(gs.stdout.strip() if gs.stdout.strip() else "  (clean)")
    except Exception:
        pass

    print(sep)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
