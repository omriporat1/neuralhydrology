#!/usr/bin/env python3
"""Stage 1 forcing extraction — arbitrary date-range chunk for full-period pipeline.

Generalisation of extract_stage1_january.py for the full v001 forcing acquisition
(2020-10-14T00Z – 2025-12-31T23Z, 2,752 basins, MRMS QPE + RTMA analysis).

All pilot-specific hardcoding is removed. Basin manifest, weight tables, time
range, and output paths are all CLI arguments. Designed for monthly chunk jobs
running sequentially under `screen` on h2o.

Outputs per chunk
-----------------
  {out_dir}/staging/mrms/<YYYYMMDDHH>.parquet   # per-hour; resumable
  {out_dir}/staging/rtma/<YYYYMMDDHH>.parquet
  {out_dir}/raw/mrms/...                         # GRIB2 raw cache
  {out_dir}/raw/rtma/...
  {out_dir}/chunks/{chunk_label}/
      combined_{chunk_label}.parquet             # all basins × vars for this chunk
      mrms_{chunk_label}.parquet
      rtma_{chunk_label}.parquet
      preview_mrms.csv  (5 rows)
      preview_rtma.csv  (5 rows)
  {out_dir}/manifests/{chunk_label}_manifest.json
  {out_dir}/manifests/{chunk_label}_summary.json
  {out_dir}/manifests/{chunk_label}_summary.md
  {out_dir}/manifests/{chunk_label}_hourly_runtime_and_volume.csv
  {out_dir}/manifests/{chunk_label}_hourly_file_status.csv
  {out_dir}/manifests/{chunk_label}_missing_files.csv  (if any)
  {out_dir}/manifests/{chunk_label}_variable_completeness.csv
  {out_dir}/manifests/{chunk_label}_basin_completeness.csv
  {out_dir}/manifests/{chunk_label}_live_progress.json  (updated during run)
  {out_dir}/manifests/{chunk_label}_live_run.log

Usage — smoke test (2 days, 10 basins):
    python scripts/extract_stage1_forcing_chunk.py \\
        --start 2020-10-14T00:00:00 --end 2020-10-15T23:00:00 \\
        --basin-manifest /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv \\
        --mrms-weights /data42/.../02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet \\
        --rtma-weights /data42/.../02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet \\
        --out-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \\
        --chunk-label smoke \\
        --max-basins 10 \\
        --download-workers 4

Usage — monthly chunk (production):
    python scripts/extract_stage1_forcing_chunk.py \\
        --start 2021-01-01T00:00:00 --end 2021-01-31T23:00:00 \\
        --basin-manifest /data42/.../v001_basin_list.csv \\
        --mrms-weights /data42/.../v001_2752_mrms_weights.parquet \\
        --rtma-weights /data42/.../v001_2752_rtma_weights.parquet \\
        --out-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \\
        --chunk-label 2021-01 \\
        --download-workers 16 \\
        --resume

Resume an interrupted chunk:
    ... same command with --resume

Extract a single product for debugging:
    ... add --products mrms   or   --products rtma
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
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
LOGGER = logging.getLogger("forcing_chunk")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 forcing extraction — arbitrary date-range chunk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start", required=True,
                   help="Chunk start (ISO 8601, e.g. 2021-01-01T00:00:00)")
    p.add_argument("--end", required=True,
                   help="Chunk end inclusive (ISO 8601, e.g. 2021-01-31T23:00:00)")
    p.add_argument("--basin-manifest", required=True, dest="basin_manifest",
                   help="CSV file with STAID column listing basins to include")
    p.add_argument("--mrms-weights", required=True, dest="mrms_weights",
                   help="Path to MRMS basin weight Parquet (e.g. v001_2752_mrms_weights.parquet)")
    p.add_argument("--rtma-weights", required=True, dest="rtma_weights",
                   help="Path to RTMA basin weight Parquet (e.g. v001_2752_rtma_weights.parquet)")
    p.add_argument("--out-dir", required=True, dest="out_dir",
                   help="Root output directory (raw cache, staging, chunks, manifests go here)")
    p.add_argument("--chunk-label", default=None, dest="chunk_label",
                   help="Label for this chunk (default: derived from --start, e.g. '2021-01')")
    p.add_argument("--products", default="mrms,rtma",
                   help="Comma-separated products to extract: mrms, rtma, or both (default: mrms,rtma)")
    p.add_argument("--max-hours", dest="max_hours", type=int, default=None,
                   help="Process at most this many hours (smoke-test mode)")
    p.add_argument("--max-basins", dest="max_basins", type=int, default=None,
                   help="Use only the first N basins from the manifest (smoke-test mode)")
    p.add_argument("--resume", action="store_true",
                   help="Skip hours with existing successful staging Parquets")
    p.add_argument("--rtma-mode", dest="rtma_mode",
                   choices=["selected_messages", "full_file"],
                   default="selected_messages",
                   help=(
                       "RTMA acquisition mode (default: selected_messages). "
                       "selected_messages downloads only the 11 Stage 1 GRIB messages "
                       "via S3 byte-range requests (~71 MB/file). "
                       "full_file downloads the entire grb2_wexp (~84 MB/file). "
                       "Cached files are reused regardless of mode."
                   ))
    p.add_argument("--download-workers", dest="download_workers", type=int, default=16,
                   help=(
                       "Number of parallel RTMA download workers (default: 16 for h2o). "
                       "Use 4 for local smoke tests. "
                       "Decode/extract remain serial; only download is parallelised."
                   ))
    p.add_argument("--no-plots", dest="no_plots", action="store_true",
                   help="Skip QC plots (faster on h2o where matplotlib may be slow)")
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
# Single-file download helpers
# ---------------------------------------------------------------------------

def _download_single_mrms(mrms_ds: Any, obj: Any, raw_dir: Path) -> tuple[Optional[Path], float]:
    t0 = time.perf_counter()
    try:
        files = mrms_ds.download_sample(raw_dir / "mrms", [obj])
        return (files[0] if files else None), time.perf_counter() - t0
    except Exception as exc:
        LOGGER.warning("MRMS download failed: %s", exc)
        return None, time.perf_counter() - t0


def _download_single_rtma(rtma_ds: Any, obj: Any, raw_dir: Path, *, mode: str = "selected_messages") -> tuple[Optional[Path], float]:
    if mode == "selected_messages":
        try:
            path, _bytes, elapsed = rtma_ds.download_selected_messages(raw_dir / "rtma", obj)
            return path, elapsed
        except Exception as exc:
            LOGGER.warning("RTMA selected-message download failed (%s) — falling back to full-file", exc)

    t0 = time.perf_counter()
    try:
        files = rtma_ds.download_sample(raw_dir / "rtma", [obj])
        return (files[0] if files else None), time.perf_counter() - t0
    except Exception as exc:
        LOGGER.warning("RTMA download failed: %s", exc)
        return None, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Parallel RTMA pre-fetch
# ---------------------------------------------------------------------------

def _ensure_rtma_s3_pool(rtma_ds: Any, n_workers: int) -> None:
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
    s3_objects: dict,
    rtma_ds: Any,
    raw_dir: Path,
    mode: str,
    n_workers: int,
) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    if not s3_objects:
        return {}

    n = min(n_workers, len(s3_objects))
    LOGGER.info("Pre-fetching %d RTMA files (mode=%s, workers=%d) ...", len(s3_objects), mode, n)
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
                ok = sum(1 for pv, _ in results.values() if pv is not None)
                mb_done = sum(
                    (v[0].stat().st_size if v[0] and v[0].exists() else 0)
                    for v in results.values()
                ) / 1e6
                LOGGER.info(
                    "  Prefetch %d/%d  ok=%d  %.1f MB  %.1f MB/s",
                    done, len(s3_objects), ok, mb_done,
                    mb_done / max(elapsed, 0.01),
                )

    n_ok = sum(1 for pv, _ in results.values() if pv is not None)
    total_s = time.perf_counter() - t0_prefetch
    total_mb = sum(
        (v[0].stat().st_size if v[0] and v[0].exists() else 0) for v in results.values()
    ) / 1e6
    LOGGER.info(
        "Prefetch done: %d ok, %d failed, %.1f MB in %.1fs (%.2f MB/s)",
        n_ok, len(results) - n_ok, total_mb, total_s, total_mb / max(total_s, 0.01),
    )
    return results


# ---------------------------------------------------------------------------
# Per-hour decode + extract
# ---------------------------------------------------------------------------

def _process_hour_mrms(mrms_path: Path, mrms_weights: Any, staids: list[str], mrms_weights_path: Path) -> tuple[Any, float, float]:
    from src.pipeline.extraction import decode_mrms_grid, extract_basin_statistics
    t0 = time.perf_counter()
    mrms_grid = decode_mrms_grid(mrms_path)
    decode_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    df = extract_basin_statistics(
        mrms_grid, mrms_weights, staids,
        weight_table_path=str(mrms_weights_path),
        source_file_path=str(mrms_path),
    )
    extract_s = time.perf_counter() - t0
    return df, decode_s, extract_s


def _process_hour_rtma(rtma_path: Path, rtma_weights: Any, staids: list[str], rtma_weights_path: Path) -> tuple[Any, float, float, list[str], list[str]]:
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
            vg, rtma_weights, staids,
            weight_table_path=str(rtma_weights_path),
            source_file_path=str(rtma_path),
        )
        for vg in included
    ]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STAT_COLUMNS)
    extract_s = time.perf_counter() - t0

    return df, decode_s, extract_s, included_names, excluded


# ---------------------------------------------------------------------------
# Live progress helpers
# ---------------------------------------------------------------------------

def _write_live_progress(
    prov_dir: Path,
    chunk_label: str,
    run_start_utc: datetime,
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
        "chunk_label":                 chunk_label,
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

    json_path = prov_dir / f"{chunk_label}_live_progress.json"
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(progress_data, fh, indent=2)
            fh.flush()
    except Exception:
        pass

    csv_path = prov_dir / f"{chunk_label}_live_progress.csv"
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
# Validation
# ---------------------------------------------------------------------------

def _run_validation(
    mrms_df: Any,
    rtma_df: Any,
    staids: list[str],
    products: list[str],
    ok_mrms_hours: int,
    ok_rtma_hours: int,
    chunk_dir: Path,
    chunk_label: str,
) -> dict[str, Any]:
    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT

    n = len(staids)
    rtma_vars = set(rtma_df["variable"].unique()) if len(rtma_df) > 0 else set()

    checks: dict[str, Any] = {}

    if _MRMS_PRODUCT in products:
        exp_mrms = ok_mrms_hours * n
        checks["mrms_extracted_hours_gt_zero"]         = ok_mrms_hours > 0
        checks["mrms_N_basins_per_ok_hour"]            = len(mrms_df) == exp_mrms
        checks["mrms_no_all_null_weighted_mean"]       = (
            not mrms_df["weighted_mean"].isna().all() if len(mrms_df) > 0 else False
        )
        checks["mrms_valid_weight_fraction_ok"]        = (
            bool((mrms_df["valid_weight_fraction"] > 0.5).all()) if len(mrms_df) > 0 else False
        )
        checks["mrms_parquet_written"] = (chunk_dir / f"mrms_{chunk_label}.parquet").exists()

    if _RTMA_PRODUCT in products:
        checks["rtma_extracted_hours_gt_zero"]         = ok_rtma_hours > 0
        checks["rtma_10wdir_absent"]                   = "10wdir" not in rtma_vars
        checks["rtma_orog_absent"]                     = "orog"   not in rtma_vars
        checks["rtma_at_least_8_variables"]            = len(rtma_vars) >= 8 if len(rtma_df) > 0 else False
        checks["rtma_no_all_null_weighted_mean"]       = (
            not rtma_df["weighted_mean"].isna().all() if len(rtma_df) > 0 else False
        )
        checks["rtma_parquet_written"] = (chunk_dir / f"rtma_{chunk_label}.parquet").exists()

    checks["combined_parquet_written"] = (chunk_dir / f"combined_{chunk_label}.parquet").exists()
    return checks


# ---------------------------------------------------------------------------
# QC plots (optional; off by default on h2o via --no-plots)
# ---------------------------------------------------------------------------

def _make_qc_plots(qc_dir: Path, chunk_label: str, metrics_rows: list, combined_df: Any, products: list) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except Exception as exc:
        LOGGER.warning("matplotlib/numpy unavailable — skipping QC plots: %s", exc)
        return []

    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT

    qc_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    mdf = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()

    # Hourly availability timeline
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
            ax.set_title(f"{prod} — hourly availability ({chunk_label})", fontsize=10)
            ax.set_xlabel("Date (UTC)")
        fig.tight_layout()
        out = qc_dir / f"{chunk_label}_hourly_availability.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Availability plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    # Representative timeseries (first 3 basins)
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
                    ax_m.set_title(f"STAID {staid} — MRMS QPE (mm)", fontsize=8)

                    ax_t = axes[i, 1]
                    rtma_tmp = combined_df[
                        (combined_df["STAID"] == staid) &
                        (combined_df["product"] == _RTMA_PRODUCT) &
                        (combined_df["variable"].isin(["2t", "t2m"]))
                    ].sort_values("valid_time_utc")
                    if len(rtma_tmp) > 0:
                        ts2 = pd.to_datetime(rtma_tmp["valid_time_utc"], utc=True, errors="coerce")
                        ax_t.plot(ts2, rtma_tmp["weighted_mean"].values, lw=0.8, color="orange")
                    ax_t.set_title(f"STAID {staid} — RTMA 2m T (K)", fontsize=8)

                fig.suptitle(f"Representative basin timeseries — {chunk_label}", fontsize=10)
                fig.tight_layout()
                out = qc_dir / f"{chunk_label}_representative_timeseries.png"
                fig.savefig(out, dpi=100)
                plt.close(fig)
                written.append(out.name)
    except Exception as exc:
        LOGGER.warning("Timeseries plot failed: %s", exc)
        try: plt.close("all")
        except Exception: pass

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_wall_start = time.perf_counter()
    _run_start_utc = datetime.utcnow()
    args = _parse_args()

    from src.pipeline.extraction import _MRMS_PRODUCT, _RTMA_PRODUCT
    from src.pipeline.geometries import normalise_staid
    import pandas as pd
    import numpy as np

    # --- Parse time range ---
    chunk_start = datetime.fromisoformat(args.start.replace("Z", "")).replace(tzinfo=None)
    chunk_end   = datetime.fromisoformat(args.end.replace("Z", "")).replace(tzinfo=None)

    # --- Derive chunk label ---
    chunk_label = args.chunk_label or chunk_start.strftime("%Y-%m")
    LOGGER.info("Chunk label: %s  (%s → %s)", chunk_label, args.start, args.end)

    # --- Product list ---
    product_map = {"mrms": _MRMS_PRODUCT, "rtma": _RTMA_PRODUCT}
    products: list[str] = [
        product_map.get(p.strip().lower(), p.strip().lower())
        for p in args.products.split(",")
    ]
    LOGGER.info("Products: %s", products)

    # --- Hours schedule ---
    all_hours = list(_iter_hours(chunk_start, chunk_end))
    if args.max_hours is not None:
        all_hours = all_hours[:args.max_hours]
    n_hours = len(all_hours)
    LOGGER.info("Hours: %d  (%s → %s)", n_hours, all_hours[0].isoformat(), all_hours[-1].isoformat())

    # --- Validate inputs exist ---
    out_dir = Path(args.out_dir)
    mrms_weights_path = Path(args.mrms_weights)
    rtma_weights_path = Path(args.rtma_weights)
    basin_manifest    = Path(args.basin_manifest)

    for p, label in [
        (mrms_weights_path, "--mrms-weights"),
        (rtma_weights_path, "--rtma-weights"),
        (basin_manifest,    "--basin-manifest"),
    ]:
        if not p.exists():
            LOGGER.error("Required input missing: %s — %s", label, p)
            return 1

    # --- Load manifest and weights ---
    manifest_df = pd.read_csv(basin_manifest, dtype={"STAID": str})
    staids = [normalise_staid(s) for s in manifest_df["STAID"].tolist()]
    if args.max_basins is not None:
        staids = staids[:args.max_basins]
        LOGGER.info("Basin list trimmed to first %d for smoke test", len(staids))
    LOGGER.info("Basins: %d", len(staids))

    mrms_weights = pd.read_parquet(mrms_weights_path)
    rtma_weights = pd.read_parquet(rtma_weights_path)

    # Filter weights to active staids (important when --max-basins is used)
    if args.max_basins is not None:
        staids_set = set(staids)
        mrms_weights = mrms_weights[mrms_weights["STAID"].isin(staids_set)].copy()
        rtma_weights = rtma_weights[rtma_weights["STAID"].isin(staids_set)].copy()
        LOGGER.info("Weights filtered: MRMS %d rows, RTMA %d rows", len(mrms_weights), len(rtma_weights))

    # --- Output directory layout ---
    raw_dir      = out_dir / "raw"
    staging_dir  = out_dir / "staging"
    chunk_dir    = out_dir / "chunks" / chunk_label
    prov_dir     = out_dir / "manifests"
    qc_dir       = out_dir / "qc" / chunk_label

    for d in [raw_dir / "mrms", raw_dir / "rtma",
              staging_dir / "mrms", staging_dir / "rtma",
              chunk_dir, prov_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Live log file ---
    _log_path = prov_dir / f"{chunk_label}_live_run.log"
    _fh = logging.FileHandler(str(_log_path), mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%H:%M:%S"
    ))
    logging.getLogger().addHandler(_fh)
    LOGGER.info("Live log: %s", _log_path)
    LOGGER.info("Chunk dir: %s", chunk_dir)

    # --- Initialise datasources and list S3 objects once per product ---
    from src.datasources.mrms import MrmsAwsQpe1hPass1
    from src.datasources.rtma import RtmaAwsConusDataSource
    from src.datasources.base import CONUS_BBOX

    mrms_s3_map: dict[datetime, Any] = {}
    rtma_s3_map: dict[datetime, Any] = {}
    mrms_ds: Optional[Any] = None
    rtma_ds: Optional[Any] = None

    if _MRMS_PRODUCT in products:
        LOGGER.info("Listing MRMS S3 objects (%s → %s) ...", all_hours[0].isoformat(), all_hours[-1].isoformat())
        mrms_ds = MrmsAwsQpe1hPass1(download_concurrency=1)
        try:
            objs = mrms_ds.list_sample_objects(all_hours[0], all_hours[-1], CONUS_BBOX, ["precip"])
            mrms_s3_map = {o.datetime.replace(tzinfo=None): o for o in objs}
            LOGGER.info("MRMS S3 objects found: %d (of %d hours)", len(mrms_s3_map), n_hours)
        except Exception as exc:
            LOGGER.error("MRMS S3 listing failed: %s", exc)
            return 1

    if _RTMA_PRODUCT in products:
        LOGGER.info("Listing RTMA S3 objects (%s → %s) ...", all_hours[0].isoformat(), all_hours[-1].isoformat())
        rtma_ds = RtmaAwsConusDataSource(download_concurrency=1)
        try:
            objs = rtma_ds.list_sample_objects(
                all_hours[0], all_hours[-1], CONUS_BBOX, ["TMP", "UGRD", "VGRD", "PRES"]
            )
            rtma_s3_map = {o.datetime.replace(tzinfo=None): o for o in objs}
            LOGGER.info("RTMA S3 objects found: %d (of %d hours)", len(rtma_s3_map), n_hours)
        except Exception as exc:
            LOGGER.error("RTMA S3 listing failed: %s", exc)
            return 1

    # --- Build local cache indices ---
    LOGGER.info("Building local raw-cache index ...")
    mrms_cache = _build_mrms_cache_index(raw_dir)
    rtma_cache = _build_rtma_cache_index(raw_dir)
    LOGGER.info("Cache: MRMS=%d files, RTMA=%d files", len(mrms_cache), len(rtma_cache))

    # --- Load existing completion status for resume ---
    existing_success: set[tuple[str, str]] = set()
    metrics_csv_path = prov_dir / f"{chunk_label}_hourly_runtime_and_volume.csv"
    metrics_rows: list[dict] = []
    if args.resume and metrics_csv_path.exists():
        try:
            prev = pd.read_csv(metrics_csv_path)
            metrics_rows = prev.to_dict("records")
            for row in metrics_rows:
                if str(row.get("status")) == "success":
                    existing_success.add((str(row["valid_time_utc"]), str(row["product"])))
            LOGGER.info("Resume: %d rows loaded; %d previously successful hour-products",
                        len(metrics_rows), len(existing_success))
        except Exception as exc:
            LOGGER.warning("Cannot read existing metrics CSV: %s — starting fresh", exc)
            metrics_rows = []

    # --- Progress state ---
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
    _write_live_progress(prov_dir, chunk_label, _run_start_utc, n_hours, products, _prog)

    # ---------------------------------------------------------------------------
    # RTMA parallel pre-fetch
    # ---------------------------------------------------------------------------
    rtma_cache_initial: set = set(rtma_cache.keys())
    rtma_prefetch_times: dict = {}

    if _RTMA_PRODUCT in products and args.download_workers >= 1:
        to_prefetch = {}
        for dt in all_hours:
            dt_str_p = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            if args.resume and (dt_str_p, _RTMA_PRODUCT) in existing_success:
                continue
            if rtma_cache.get(dt) is not None:
                continue
            obj = rtma_s3_map.get(dt)
            if obj is None:
                continue
            to_prefetch[dt] = obj

        if to_prefetch:
            prefetch_results = _prefetch_rtma_files(
                to_prefetch, rtma_ds, raw_dir, args.rtma_mode, max(1, args.download_workers)
            )
            for dt, (path, dl_s) in prefetch_results.items():
                if path is not None:
                    rtma_cache[dt] = path
                    _prog["files_downloaded"] += 1
                    _prog["bytes_downloaded"] += path.stat().st_size if path.exists() else 0
                rtma_prefetch_times[dt] = dl_s
        else:
            LOGGER.info("RTMA prefetch: nothing to download (all cached or resumed)")

    # ---------------------------------------------------------------------------
    # Main hourly extraction loop (serial: decode → extract → write staging)
    # ---------------------------------------------------------------------------

    all_mrms_frames: list[pd.DataFrame] = []
    all_rtma_frames: list[pd.DataFrame] = []
    missing_rows: list[dict] = []

    for i, dt in enumerate(all_hours):
        dt_str   = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        hour_tag = dt.strftime("%Y%m%d%H")
        _hr_mrms_ok = 0; _hr_rtma_ok = 0; _hr_failed = 0

        if i == 0 or (i + 1) % 24 == 0 or i == n_hours - 1:
            pct = (i + 1) / n_hours * 100
            LOGGER.info("[%3.0f%%] hour %d/%d  %s", pct, i + 1, n_hours, dt_str)

        for product in products:
            is_mrms    = product == _MRMS_PRODUCT
            staging_pq = staging_dir / ("mrms" if is_mrms else "rtma") / f"{hour_tag}.parquet"
            resume_key = (dt_str, product)

            # Resume: load from staging if already successful
            if args.resume and resume_key in existing_success and staging_pq.exists():
                try:
                    hour_df = pd.read_parquet(staging_pq)
                    if is_mrms:
                        all_mrms_frames.append(hour_df)
                    else:
                        all_rtma_frames.append(hour_df)
                    continue
                except Exception:
                    pass

            # Locate raw file
            s3_obj     = mrms_s3_map.get(dt) if is_mrms else rtma_s3_map.get(dt)
            download_s = 0.0

            if is_mrms:
                raw_path    = mrms_cache.get(dt)
                file_reused = raw_path is not None
            else:
                raw_path    = rtma_cache.get(dt)
                file_reused = dt in rtma_cache_initial
                download_s  = rtma_prefetch_times.get(dt, 0.0)

            if raw_path is None:
                if s3_obj is None:
                    metrics_rows.append({
                        "product": product, "valid_time_utc": dt_str,
                        "raw_file_path": None, "raw_file_size_bytes": None,
                        "file_reused": False, "download_time_s": None,
                        "decode_time_s": None, "extraction_time_s": None,
                        "write_time_s": None, "total_processing_time_s": None,
                        "n_output_rows": 0, "output_parquet_bytes": None,
                        "status": "missing_s3",
                        "warning_message": f"No S3 object for {dt_str}",
                    })
                    missing_rows.append({"product": product, "valid_time_utc": dt_str, "reason": "not_in_s3"})
                    _hr_failed += 1
                    continue

                if is_mrms:
                    raw_path, download_s = _download_single_mrms(mrms_ds, s3_obj, raw_dir)
                else:
                    raw_path, download_s = _download_single_rtma(rtma_ds, s3_obj, raw_dir, mode=args.rtma_mode)

                if raw_path is None:
                    metrics_rows.append({
                        "product": product, "valid_time_utc": dt_str,
                        "raw_file_path": None, "raw_file_size_bytes": None,
                        "file_reused": False, "download_time_s": round(download_s, 3),
                        "decode_time_s": None, "extraction_time_s": None,
                        "write_time_s": None, "total_processing_time_s": round(download_s, 3),
                        "n_output_rows": 0, "output_parquet_bytes": None,
                        "status": "download_failed",
                        "warning_message": f"Download failed in {download_s:.1f}s",
                    })
                    missing_rows.append({"product": product, "valid_time_utc": dt_str, "reason": "download_failed"})
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

            # Decode + extract
            t_proc    = time.perf_counter()
            decode_s  = 0.0; extract_s = 0.0; write_s = 0.0
            n_rows    = 0;   staging_bytes = 0
            status    = "success"; warn_msg: Optional[str] = None
            hour_df: Optional[pd.DataFrame] = None

            try:
                if is_mrms:
                    hour_df, decode_s, extract_s = _process_hour_mrms(
                        raw_path, mrms_weights, staids, mrms_weights_path
                    )
                else:
                    hour_df, decode_s, extract_s, _, _ = _process_hour_rtma(
                        raw_path, rtma_weights, staids, rtma_weights_path
                    )

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
                "product": product, "valid_time_utc": dt_str,
                "raw_file_path": str(raw_path), "raw_file_size_bytes": raw_size_bytes,
                "file_reused": file_reused, "download_time_s": round(download_s, 3),
                "decode_time_s": round(decode_s, 3), "extraction_time_s": round(extract_s, 3),
                "write_time_s": round(write_s, 3), "total_processing_time_s": round(total_proc_s, 3),
                "n_output_rows": n_rows, "output_parquet_bytes": staging_bytes,
                "status": status, "warning_message": warn_msg,
            })
            if status == "success":
                if is_mrms: _hr_mrms_ok += 1
                else: _hr_rtma_ok += 1
            else:
                _hr_failed += 1
                if warn_msg:
                    _prog["latest_err"] = warn_msg

        _prog["mrms_ok"] += _hr_mrms_ok
        _prog["rtma_ok"] += _hr_rtma_ok
        _prog["failed"]  += _hr_failed
        _prog["latest_hour"] = dt_str
        _prog["latest_msg"]  = (
            f"hour {dt_str}: MRMS={'ok' if _hr_mrms_ok else 'fail'} "
            f"RTMA={'ok' if _hr_rtma_ok else 'fail'}"
        )
        _write_live_progress(prov_dir, chunk_label, _run_start_utc, n_hours, products, _prog)

        if (i + 1) % 24 == 0 or i == n_hours - 1:
            _n_done = _prog["mrms_ok"] + _prog["rtma_ok"] + _prog["failed"]
            _n_tot  = n_hours * len(products)
            _el_s   = time.perf_counter() - t_wall_start
            _pct    = _n_done / max(_n_tot, 1) * 100
            _eta_s  = int(_el_s / _pct * (100 - _pct)) if _pct > 0 else 0
            _gb     = _prog["bytes_downloaded"] / 1e9
            print(
                f"{_n_done}/{_n_tot} hr-products  "
                f"| MRMS {_prog['mrms_ok']}/{n_hours} "
                f"| RTMA {_prog['rtma_ok']}/{n_hours} "
                f"| {_pct:.1f}%  elapsed {_fmt_hms(int(_el_s))}  ETA {_fmt_hms(_eta_s)} "
                f"| downloaded {_gb:.1f} GB  failures {_prog['failed']}",
                flush=True,
            )

    # ---------------------------------------------------------------------------
    # Combine staging into chunk output Parquets
    # ---------------------------------------------------------------------------

    LOGGER.info("Combining %d MRMS + %d RTMA staging frames ...", len(all_mrms_frames), len(all_rtma_frames))
    mrms_df = pd.concat(all_mrms_frames, ignore_index=True) if all_mrms_frames else pd.DataFrame()
    rtma_df = pd.concat(all_rtma_frames, ignore_index=True) if all_rtma_frames else pd.DataFrame()
    combined_df = pd.concat(
        [df for df in [mrms_df, rtma_df] if len(df) > 0], ignore_index=True
    )

    chunk_parquets: dict[str, Path] = {}
    for df_out, fname_tpl, key in [
        (mrms_df,     f"mrms_{chunk_label}.parquet",     "mrms_parquet"),
        (rtma_df,     f"rtma_{chunk_label}.parquet",     "rtma_parquet"),
        (combined_df, f"combined_{chunk_label}.parquet", "combined_parquet"),
    ]:
        if len(df_out) > 0:
            p = chunk_dir / fname_tpl
            df_out.to_parquet(p, index=False)
            chunk_parquets[key] = p
            LOGGER.info("Written: %s (%d rows, %.1f MB)", fname_tpl, len(df_out), p.stat().st_size / 1e6)

    # Preview CSVs
    for df_out, fname in [(mrms_df, f"preview_mrms.csv"), (rtma_df, f"preview_rtma.csv")]:
        if len(df_out) > 0:
            try:
                df_out.head(5).to_csv(chunk_dir / fname, index=False)
            except Exception:
                pass

    # ---------------------------------------------------------------------------
    # Metrics and summary CSVs
    # ---------------------------------------------------------------------------

    metrics_df = pd.DataFrame(metrics_rows)
    if len(metrics_df) > 0:
        metrics_df.to_csv(prov_dir / f"{chunk_label}_hourly_runtime_and_volume.csv", index=False)

    # hourly_file_status
    status_rows = []
    for dt in all_hours:
        dt_str_s = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        for product in products:
            matches = [r for r in metrics_rows if r["valid_time_utc"] == dt_str_s and r["product"] == product]
            s = matches[0]["status"] if matches else "not_processed"
            status_rows.append({"valid_time_utc": dt_str_s, "product": product, "status": s})
    pd.DataFrame(status_rows).to_csv(prov_dir / f"{chunk_label}_hourly_file_status.csv", index=False)

    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(prov_dir / f"{chunk_label}_missing_files.csv", index=False)
        LOGGER.warning("%d missing/failed hour-products written to missing_files.csv", len(missing_rows))

    # Product summary
    product_summary: dict[str, dict] = {}
    for product in products:
        pm_all = [r for r in metrics_rows if r["product"] == product]
        pm_ok  = [r for r in pm_all if r["status"] == "success"]
        raw_sizes = [r["raw_file_size_bytes"] for r in pm_ok if r.get("raw_file_size_bytes")]
        tot_times = [r["total_processing_time_s"] for r in pm_ok if r.get("total_processing_time_s")]
        pq_bytes  = [r["output_parquet_bytes"] for r in pm_ok if r.get("output_parquet_bytes")]
        ok_h = len(pm_ok)
        product_summary[product] = {
            "expected_hours":           n_hours,
            "successful_hours":         ok_h,
            "missing_failed_hours":     len(pm_all) - ok_h,
            "total_raw_bytes":          sum(raw_sizes),
            "median_raw_bytes_per_hour": float(np.median(raw_sizes)) if raw_sizes else 0.0,
            "total_output_parquet_bytes": sum(pq_bytes),
            "total_output_rows":        sum(r["n_output_rows"] for r in pm_ok),
            "median_total_processing_time_s": float(np.median(tot_times)) if tot_times else 0.0,
        }
    pd.DataFrame([{"product": k, **v} for k, v in product_summary.items()]).to_csv(
        prov_dir / f"{chunk_label}_product_summary.csv", index=False
    )

    # Variable completeness (RTMA)
    if len(rtma_df) > 0 and _RTMA_PRODUCT in products:
        ok_rtma_h = product_summary.get(_RTMA_PRODUCT, {}).get("successful_hours", 1)
        var_rows = []
        for var in sorted(rtma_df["variable"].unique()):
            vdf = rtma_df[rtma_df["variable"] == var]
            var_rows.append({
                "variable": var,
                "n_rows": len(vdf),
                "expected_rows": ok_rtma_h * len(staids),
                "completeness_pct": len(vdf) / max(ok_rtma_h * len(staids), 1) * 100,
            })
        pd.DataFrame(var_rows).to_csv(prov_dir / f"{chunk_label}_variable_completeness.csv", index=False)

    # Basin completeness
    if len(combined_df) > 0:
        basin_rows = []
        for staid in staids:
            for product in products:
                ok_h = product_summary.get(product, {}).get("successful_hours", 1)
                pbd  = combined_df[(combined_df["STAID"] == staid) & (combined_df["product"] == product)]
                n_present = pbd["valid_time_utc"].nunique()
                basin_rows.append({
                    "STAID": staid, "product": product,
                    "n_hours_present": n_present, "expected_hours": ok_h,
                    "completeness_pct": n_present / max(ok_h, 1) * 100,
                })
        pd.DataFrame(basin_rows).to_csv(prov_dir / f"{chunk_label}_basin_completeness.csv", index=False)

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    ok_mrms_hours = product_summary.get(_MRMS_PRODUCT, {}).get("successful_hours", 0)
    ok_rtma_hours = product_summary.get(_RTMA_PRODUCT, {}).get("successful_hours", 0)

    validation = _run_validation(
        mrms_df=mrms_df, rtma_df=rtma_df,
        staids=staids, products=products,
        ok_mrms_hours=ok_mrms_hours, ok_rtma_hours=ok_rtma_hours,
        chunk_dir=chunk_dir, chunk_label=chunk_label,
    )
    all_pass = all(bool(v) for v in validation.values() if isinstance(v, bool))

    # ---------------------------------------------------------------------------
    # QC plots (optional)
    # ---------------------------------------------------------------------------

    plot_files: list[str] = []
    if not args.no_plots:
        plot_files = _make_qc_plots(
            qc_dir=qc_dir,
            chunk_label=chunk_label,
            metrics_rows=metrics_rows,
            combined_df=combined_df if len(combined_df) > 0 else None,
            products=products,
        )

    # ---------------------------------------------------------------------------
    # Manifest / provenance
    # ---------------------------------------------------------------------------

    t_elapsed = time.perf_counter() - t_wall_start

    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    summary = {
        "chunk_label":           chunk_label,
        "start":                 all_hours[0].isoformat() + "Z",
        "end":                   all_hours[-1].isoformat() + "Z",
        "n_hours_scheduled":     n_hours,
        "n_basins":              len(staids),
        "products":              products,
        "rtma_mode":             args.rtma_mode,
        "download_workers":      args.download_workers,
        "resume":                args.resume,
        "run_start_utc":         _run_start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_clock_seconds":    round(t_elapsed, 1),
        "product_summary":       product_summary,
        "validation":            validation,
        "all_pass":              all_pass,
        "git_commit":            git_sha,
        "mrms_weights_path":     str(mrms_weights_path),
        "rtma_weights_path":     str(rtma_weights_path),
        "basin_manifest":        str(basin_manifest),
        "chunk_dir":             str(chunk_dir),
        "n_missing_failed":      len(missing_rows),
        "qc_plots_written":      plot_files,
    }

    with open(prov_dir / f"{chunk_label}_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Human-readable summary
    mrms_ok = product_summary.get(_MRMS_PRODUCT, {}).get("successful_hours", 0)
    rtma_ok = product_summary.get(_RTMA_PRODUCT, {}).get("successful_hours", 0)
    n_missing = len(missing_rows)
    summary_md = (
        f"# Forcing Extraction Chunk: {chunk_label}\n\n"
        f"**Status:** {'PASS' if all_pass else 'FAIL'}\n"
        f"**Period:** {all_hours[0].isoformat()}Z – {all_hours[-1].isoformat()}Z\n"
        f"**Hours:** {n_hours}  **Basins:** {len(staids)}\n"
        f"**Products:** {', '.join(products)}\n\n"
        f"| Product | OK hours | Missing/failed |\n"
        f"|---------|----------|----------------|\n"
        f"| MRMS QPE | {mrms_ok}/{n_hours} | {n_hours - mrms_ok} |\n"
        f"| RTMA CONUS | {rtma_ok}/{n_hours} | {n_hours - rtma_ok} |\n\n"
        f"**Missing/failed hour-products:** {n_missing}\n"
        f"**Wall clock:** {_format_duration(t_elapsed)}\n"
        f"**Downloaded:** {_format_bytes(_prog['bytes_downloaded'])}\n\n"
        f"## Validation\n\n"
        + "\n".join(f"- {'PASS' if v else 'FAIL'}  {k}" for k, v in validation.items() if isinstance(v, bool))
    )
    (prov_dir / f"{chunk_label}_summary.md").write_text(summary_md, encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Final console report
    # ---------------------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"CHUNK: {chunk_label}")
    print(f"{'='*60}")
    print(f"  Period:   {all_hours[0].isoformat()}Z → {all_hours[-1].isoformat()}Z")
    print(f"  Basins:   {len(staids)}")
    print(f"  Products: {', '.join(products)}")
    print(f"  MRMS OK:  {mrms_ok}/{n_hours}")
    print(f"  RTMA OK:  {rtma_ok}/{n_hours}")
    print(f"  Missing:  {n_missing}")
    print(f"  Elapsed:  {_format_duration(t_elapsed)}")
    print(f"  Downloaded: {_format_bytes(_prog['bytes_downloaded'])}")
    print(f"\nValidation:")
    for k, v in validation.items():
        if isinstance(v, bool):
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"\nChunk outputs: {chunk_dir}")
    print(f"Manifests:     {prov_dir / f'{chunk_label}_manifest.json'}")
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
