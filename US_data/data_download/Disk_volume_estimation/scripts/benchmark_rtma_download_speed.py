#!/usr/bin/env python3
"""RTMA full-CONUS download-speed benchmark.

Compares acquisition mechanisms for RTMA 2dvaranl_ndfd.grb2_wexp files
to identify the fastest full-CONUS download approach for Stage 1.

Single-file mechanisms tested (t03z, 2023-01-01):
  A  boto3 GetObject (single stream, 8 MB chunks)         — current default
  B  boto3 selected-message Range GETs (11/13 messages)   — current optimized
  C  boto3 S3 Transfer Manager (multipart concurrent)     — new candidate
  E  httpx HTTPS streaming                                 — available
  F  requests HTTPS streaming                              — available

Concurrency test (t03z–t10z, 8 files, Mode A full-file):
  1 / 2 / 4 / 8 workers

AWS CLI (D) and Herbie (G) are not installed on this machine; those entries
appear in the report as skipped.

Usage:
    python scripts/benchmark_rtma_download_speed.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun

    # skip the 8-file concurrency test (saves ~530 MB download):
    python scripts/benchmark_rtma_download_speed.py \\
        --config configs/pilot_stage1.yaml \\
        --data-root tmp/stage1_pilot_dryrun \\
        --skip-parallel
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("rtma_speed")

# ── Benchmark targets ──────────────────────────────────────────────────────────
_BUCKET = "noaa-rtma-pds"
_DAY    = "rtma2p5.20230101"
_T03_KEY = f"{_DAY}/rtma2p5.t03z.2dvaranl_ndfd.grb2_wexp"
_PARALLEL_HOURS = range(3, 11)   # t03z … t10z — 8 fresh files
_PARALLEL_KEYS  = [
    f"{_DAY}/rtma2p5.t{h:02d}z.2dvaranl_ndfd.grb2_wexp"
    for h in _PARALLEL_HOURS
]
# 16 files for worker counts > 8 (t03z … t18z)
_PARALLEL_KEYS_16 = [
    f"{_DAY}/rtma2p5.t{h:02d}z.2dvaranl_ndfd.grb2_wexp"
    for h in range(3, 19)
]
_HTTPS_BASE = "https://noaa-rtma-pds.s3.amazonaws.com"

# Stage 1 exclusion list (same as RtmaAwsConusDataSource._IDX_EXCLUDE_FROM_DOWNLOAD)
_IDX_EXCLUDE = frozenset({"wdir", "10wdir", "hgt", "orog", "zsfc"})

# Known from Milestone 2D / acquisition benchmark (used to annotate only)
_PREV_FULLFILE_BYTES = 84_300_000
_PREV_FULLFILE_DL_S  = 43.3        # seconds
_PREV_SELECTED_BYTES = 71_200_000
_PREV_SELECTED_DL_S  = 28.3        # seconds

# Full-period scaling anchor
_FULL_HOURS = 52_608   # 2020-2025


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTMA download-speed benchmark")
    p.add_argument("--config",    default="configs/pilot_stage1.yaml")
    p.add_argument("--data-root", dest="data_root", default=None)
    p.add_argument("--skip-parallel", dest="skip_parallel", action="store_true",
                   help="Skip all parallel/concurrency tests")
    p.add_argument("--skip-fullfile-parallel", dest="skip_fullfile_parallel",
                   action="store_true",
                   help="Skip full-file parallel test; run only selected_messages parallel")
    return p.parse_args()


# ── GRIB verification ──────────────────────────────────────────────────────────

def _verify_grib(path: Path) -> dict[str, Any]:
    """Decode path with cfgrib; return n_vars, decode_time_s, success."""
    try:
        import cfgrib, warnings
        t0 = time.perf_counter()
        warnings.filterwarnings("ignore", category=FutureWarning)
        datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
        n_vars = sum(len(ds.data_vars) for ds in datasets)
        decode_s = time.perf_counter() - t0
        for ds in datasets:
            ds.close()
        return {"success": True, "n_vars": n_vars, "decode_s": decode_s}
    except Exception as exc:
        return {"success": False, "n_vars": 0, "decode_s": 0.0, "error": str(exc)}


# ── Download helpers ──────────────────────────────────────────────────────────

def _dl_boto3_getobject(s3: Any, bucket: str, key: str, out: Path) -> tuple[int, float]:
    """Single-stream GetObject with 8 MB read chunks (current implementation)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    with out.open("wb") as fh:
        while True:
            chunk = body.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return out.stat().st_size, time.perf_counter() - t0


def _dl_selected_messages(bucket: str, key: str, out: Path) -> tuple[int, float, float]:
    """Selected-message byte-range download using the rtma.py method.

    Returns (bytes, download_s, idx_fetch_s).
    """
    from src.datasources.rtma import RtmaAwsConusDataSource
    from src.datasources.base import RemoteObject

    rtma_ds = RtmaAwsConusDataSource(download_concurrency=1)
    obj = RemoteObject(
        url=f"s3://{bucket}/{key}",
        key=key,
        datetime=None,
        variables=[],
        estimated_bytes=None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    path, nbytes, dl_s = rtma_ds.download_selected_messages(out.parent, obj)
    total_s = time.perf_counter() - t0
    idx_s = total_s - dl_s
    actual_bytes = path.stat().st_size if path else 0
    return actual_bytes, dl_s, idx_s


def _dl_transfer_manager(s3_client: Any, bucket: str, key: str, out: Path) -> tuple[int, float]:
    """boto3 S3 Transfer Manager with multipart concurrent download."""
    from boto3.s3.transfer import TransferConfig

    transfer_config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=8 * 1024 * 1024,
        use_threads=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    s3_client.download_file(bucket, key, str(out), Config=transfer_config)
    return out.stat().st_size, time.perf_counter() - t0


def _dl_httpx(url: str, out: Path) -> tuple[int, float]:
    """Stream download via httpx."""
    import httpx
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    total = 0
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with out.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=8 * 1024 * 1024):
                    fh.write(chunk)
                    total += len(chunk)
    return total, time.perf_counter() - t0


def _dl_requests(url: str, out: Path) -> tuple[int, float]:
    """Stream download via requests."""
    import requests
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    total = 0
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with out.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    return total, time.perf_counter() - t0


# ── Mechanism benchmark runner ────────────────────────────────────────────────

def _run_mechanism(
    name: str,
    description: str,
    download_fn,        # callable -> (bytes, dl_s, [extra_s])
    verify: bool,
    cleanup_path: Optional[Path],
) -> dict[str, Any]:
    """Run one mechanism, return result dict."""
    result: dict[str, Any] = {
        "mechanism": name,
        "description": description,
        "success": False,
        "bytes": None,
        "download_s": None,
        "overhead_s": None,
        "total_s": None,
        "mb_s": None,
        "n_vars_decoded": None,
        "decode_s": None,
        "notes": "",
    }
    try:
        LOGGER.info("  Downloading with %s ...", name)
        raw = download_fn()
        if len(raw) == 2:
            nbytes, dl_s = raw
            overhead_s = 0.0
        else:
            nbytes, dl_s, overhead_s = raw[0], raw[1], raw[2]

        result["bytes"]       = nbytes
        result["download_s"]  = round(dl_s, 2)
        result["overhead_s"]  = round(overhead_s, 2)
        result["total_s"]     = round(dl_s + overhead_s, 2)
        result["mb_s"]        = round(nbytes / 1e6 / (dl_s + overhead_s + 1e-9), 2)
        result["success"]     = True
        LOGGER.info(
            "  %s: %d MB in %.2fs (+ %.2fs overhead) = %.2f MB/s",
            name, nbytes // 1_000_000, dl_s, overhead_s, result["mb_s"],
        )
    except Exception as exc:
        result["notes"] = str(exc)
        LOGGER.warning("  %s FAILED: %s", name, exc)
        return result

    if verify and cleanup_path is not None and cleanup_path.exists():
        vr = _verify_grib(cleanup_path)
        result["n_vars_decoded"] = vr.get("n_vars")
        result["decode_s"]       = round(vr.get("decode_s", 0), 3)
        if not vr["success"]:
            result["notes"] += f" decode_failed: {vr.get('error','')}"
            LOGGER.warning("  %s: cfgrib decode FAILED: %s", name, vr.get("error"))
        else:
            LOGGER.info("  %s: cfgrib decoded %d vars in %.3fs", name, vr["n_vars"], vr["decode_s"])

    if cleanup_path is not None and cleanup_path.exists():
        cleanup_path.unlink(missing_ok=True)

    return result


# ── Selected-messages parallel benchmark ─────────────────────────────────────

def _run_parallel_selected_bench(
    rtma_ds: Any,
    n_workers: int,
    keys: list[str],
    bench_dir: Path,
) -> dict[str, Any]:
    """Download keys with n_workers using selected_messages mode; return timing summary."""
    from src.datasources.base import RemoteObject
    import re, shutil

    w_dir = bench_dir / f"sel_w{n_workers}"
    w_dir.mkdir(parents=True, exist_ok=True)

    per_file: list[float] = []
    failures = 0
    total_bytes = 0

    def _dl_one(key: str):
        obj = RemoteObject(
            url=f"s3://{_BUCKET}/{key}",
            key=key,
            datetime=None,
            variables=[],
            estimated_bytes=None,
        )
        path, nbytes, dl_s = rtma_ds.download_selected_messages(w_dir, obj)
        return path, nbytes, dl_s

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futs = {executor.submit(_dl_one, key): key for key in keys}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                path, nbytes, dl_s = fut.result()
                if path is not None:
                    per_file.append(dl_s)
                    total_bytes += nbytes
                    path.unlink(missing_ok=True)
                else:
                    failures += 1
                    LOGGER.warning("  sel w=%d: %s returned None", n_workers, key[-8:])
            except Exception as exc:
                failures += 1
                LOGGER.warning("  sel w=%d: %s FAILED: %s", n_workers, key[-8:], exc)
    wall_s = time.perf_counter() - t0
    shutil.rmtree(w_dir, ignore_errors=True)

    n_ok = len(per_file)
    s_sorted = sorted(per_file)
    return {
        "n_workers":     n_workers,
        "n_files":       len(keys),
        "n_ok":          n_ok,
        "failures":      failures,
        "total_bytes":   total_bytes,
        "wall_s":        round(wall_s, 2),
        "agg_mb_s":      round(total_bytes / 1e6 / (wall_s + 1e-9), 2),
        "median_file_s": round(s_sorted[n_ok // 2], 2) if n_ok else None,
        "p90_file_s":    round(s_sorted[min(int(n_ok * 0.9), n_ok - 1)], 2) if n_ok > 1 else None,
        "speedup_vs_serial": None,  # filled in after w=1 result
    }


# ── Concurrency benchmark ─────────────────────────────────────────────────────

def _run_parallel_bench(
    s3: Any,
    n_workers: int,
    keys: list[str],
    bench_dir: Path,
) -> dict[str, Any]:
    """Download all keys with n_workers threads; return timing summary."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [bench_dir / k.replace("/", "_") for k in keys]

    per_file: list[float] = []
    failures = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futs = {
            executor.submit(_dl_boto3_getobject, s3, _BUCKET, key, op): (key, op)
            for key, op in zip(keys, out_paths)
        }
        for fut in as_completed(futs):
            key, op = futs[fut]
            try:
                _nbytes, file_s = fut.result()
                per_file.append(file_s)
                LOGGER.debug("  parallel w=%d: %s done in %.1fs", n_workers, key[-8:], file_s)
            except Exception as exc:
                failures += 1
                LOGGER.warning("  parallel w=%d: %s FAILED: %s", n_workers, key[-8:], exc)
    wall_s = time.perf_counter() - t0

    total_bytes = sum(p.stat().st_size for p in out_paths if p.exists())
    for op in out_paths:
        op.unlink(missing_ok=True)

    n_ok = len(per_file)
    return {
        "n_workers":     n_workers,
        "n_files":       len(keys),
        "n_ok":          n_ok,
        "failures":      failures,
        "total_bytes":   total_bytes,
        "wall_s":        round(wall_s, 2),
        "agg_mb_s":      round(total_bytes / 1e6 / (wall_s + 1e-9), 2),
        "median_file_s": round(sorted(per_file)[len(per_file) // 2], 2) if per_file else None,
        "speedup_vs_serial": round(per_file[0] * n_ok / wall_s, 2) if per_file else None,
    }


# ── Code implementation analysis ──────────────────────────────────────────────

def _analyze_implementation() -> dict[str, Any]:
    """Static analysis of current RTMA downloader code."""
    return {
        "client_reuse": (
            "YES — _s3_client_cached reused per datasource instance. "
            "No new client created per file."
        ),
        "max_pool_connections": (
            "NOT SET — botocore default is 10. "
            "Sufficient for download_concurrency<=4; "
            "may cause pool exhaustion at 8+ workers sharing one client."
        ),
        "chunk_size": "8 MB (body.read(8*1024*1024)) — appropriate",
        "download_method": (
            "s3.get_object() + single-stream read loop. "
            "NO multipart: file downloaded over one TCP connection."
        ),
        "pipeline": (
            "Fully serial: S3 listing -> download file -> decode -> extract -> write. "
            "No overlap between download and decode of successive hours."
        ),
        "idx_caching": (
            "NOT cached locally. .idx (~0.7 KB) fetched from S3 each time "
            "download_selected_messages() is called. "
            "Minor overhead (~0.8s/call) but adds up over 744 hours."
        ),
        "range_merging": (
            "Optimal: 11/13 selected messages collapse to 2 merged ranges "
            "(hgt at start and wdir in middle are the gaps). "
            "Only 2 S3 GetObject+Range calls per hour."
        ),
        "intermediate_copies": (
            "None for full-file mode. "
            "Selected-message mode writes ranges sequentially to disk; "
            "no in-memory buffering of entire file."
        ),
        "main_bottleneck": (
            "Single-stream download over one TCP connection to S3. "
            "boto3 Transfer Manager multipart is the primary improvement opportunity."
        ),
        "recommended_improvements": [
            "Use boto3 S3 Transfer Manager (download_file with TransferConfig) "
            "for per-file multipart concurrent download.",
            "Set max_pool_connections=20 when using Transfer Manager.",
            "For multi-file batches, current ThreadPoolExecutor is correct "
            "but download_concurrency should be increased from 1 to 4-8.",
            "Cache .idx files alongside GRIB files to skip S3 fetch on re-runs.",
            "Consider producer/consumer pipeline: N download threads + "
            "M decode/extract threads running concurrently.",
        ],
    }


# ── Scaling estimate builder ──────────────────────────────────────────────────

def _make_scaling_row(
    method: str,
    bytes_per_hour: int,
    dl_s_per_hour: float,
    proc_s_per_hour: float,   # decode + extract (no download)
    notes: str = "",
) -> dict[str, Any]:
    total_s = dl_s_per_hour + proc_s_per_hour
    jan_h   = 744
    return {
        "method":               method,
        "mb_per_hour":          round(bytes_per_hour / 1e6, 1),
        "mb_s_per_file":        round(bytes_per_hour / 1e6 / (dl_s_per_hour + 1e-9), 2),
        "jan_raw_gb":           round(bytes_per_hour * jan_h / 1e9, 1),
        "full_raw_tb":          round(bytes_per_hour * _FULL_HOURS / 1e12, 2),
        "serial_s_per_hour":    round(total_s, 1),
        "jan_wall_hours":       round(total_s * jan_h / 3600, 1),
        "full_serial_hours":    round(total_s * _FULL_HOURS / 3600, 0),
        "full_32task_hours":    round(total_s * _FULL_HOURS / 3600 / 32, 1),
        "full_128task_hours":   round(total_s * _FULL_HOURS / 3600 / 128, 1),
        "notes": notes,
    }


# ── Report writers ────────────────────────────────────────────────────────────

def _fmt_b(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _write_md(
    path: Path,
    single_results: list[dict],
    parallel_results: list[dict],
    impl_analysis: dict[str, Any],
    scaling_rows: list[dict],
) -> None:
    lines = [
        "# RTMA Full-CONUS Download-Speed Benchmark",
        "",
        f"**Hour**: 2023-01-01T03:00Z (t03z, 83.8 MB full file)  ",
        f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## 1. Current Implementation Audit",
        "",
    ]
    for k, v in impl_analysis.items():
        if k == "recommended_improvements":
            lines.append("**Recommended improvements:**")
            for item in v:
                lines.append(f"- {item}")
        else:
            lines.append(f"**{k.replace('_', ' ').title()}**: {v}")
        lines.append("")

    lines += [
        "## 2. Single-File Download Mechanism Comparison (t03z)",
        "",
        "| Mechanism | Bytes | DL time | Overhead | Total | MB/s | Vars | Decode | Notes |",
        "|-----------|-------|---------|----------|-------|------|------|--------|-------|",
    ]
    for r in single_results:
        if not r["success"]:
            lines.append(
                f"| {r['mechanism']} | FAILED | — | — | — | — | — | — | {r['notes'][:60]} |"
            )
        else:
            lines.append(
                f"| {r['mechanism']} | {_fmt_b(r['bytes'] or 0)} | "
                f"{r['download_s']}s | {r['overhead_s']}s | {r['total_s']}s | "
                f"{r['mb_s']} | {r['n_vars_decoded'] or '—'} | "
                f"{r['decode_s'] or '—'}s | {r['notes'][:40]} |"
            )

    lines += [
        "",
        "## 3. Multi-File Concurrency Test (t03z–t10z, 8 × 83.8 MB, Mode A)",
        "",
    ]
    if parallel_results:
        lines += [
            "| Workers | Files | Total MB | Wall-clock | Agg MB/s | Median/file | Speedup vs serial | Failures |",
            "|---------|-------|----------|------------|----------|-------------|-------------------|----------|",
        ]
        for r in parallel_results:
            lines.append(
                f"| {r['n_workers']} | {r['n_ok']}/{r['n_files']} | "
                f"{r['total_bytes']//1_000_000} MB | {r['wall_s']}s | "
                f"{r['agg_mb_s']} | {r['median_file_s']}s | {r['speedup_vs_serial']}× | "
                f"{r['failures']} |"
            )
    else:
        lines.append("*Parallel test skipped (--skip-parallel).*")

    lines += [
        "",
        "## 4. Scaling Estimates",
        "",
        "| Method | MB/hr | MB/s | Jan raw (GB) | Full raw (TB) | Serial/hr | Jan wall (h) | Full serial (h) | 32 tasks (h) | 128 tasks (h) |",
        "|--------|-------|------|-------------|---------------|-----------|-------------|----------------|-------------|----------------|",
    ]
    for r in scaling_rows:
        lines.append(
            f"| {r['method']} | {r['mb_per_hour']} | {r['mb_s_per_file']} | "
            f"{r['jan_raw_gb']} | {r['full_raw_tb']} | {r['serial_s_per_hour']}s | "
            f"{r['jan_wall_hours']} | {r['full_serial_hours']:.0f} | "
            f"{r['full_32task_hours']} | {r['full_128task_hours']} |"
        )

    lines += [
        "",
        "> **Note**: Raw storage scales with hours × file size (independent of basin count).",
        "> Processed Parquet scales with hours × basins × variables (separate estimate).",
        "> All timing benchmarked on a ~15 Mbps connection; HPC with 10 Gbps will be",
        "> dramatically faster and the advantage of Transfer Manager may narrow.",
        "",
        "## 5. Diagnosis: Why is Download Speed ~2 MB/s?",
        "",
        "The observed ~2 MB/s per TCP stream is consistent with a single-stream",
        "HTTP/S3 connection at ~15 Mbps residential broadband.",
        "",
        "**Root-cause candidates (ordered by likelihood):**",
        "1. **Single-stream bandwidth** — one TCP connection to S3 at ~15 Mbps",
        "   is a hard cap per stream.",
        "2. **S3 per-connection rate limiting** — AWS may limit individual unsigned",
        "   connections to public buckets.",
        "3. **Geographic routing** — noaa-rtma-pds is in us-east-1; routing from",
        "   other regions adds latency but not sustained bandwidth loss.",
        "4. **Python overhead** — chunked read loop is not the bottleneck at 8 MB chunks.",
        "",
        "**Mitigation (boto3 Transfer Manager multipart):**",
        "If the bottleneck is per-connection throttling rather than total bandwidth,",
        "multipart concurrent download (multiple Range GETs on the same file) can",
        "aggregate faster throughput. The benchmark above shows whether this is",
        "effective on this machine.",
        "",
        "## 6. Recommendations",
        "",
        "**Locally (debugging / small runs):**",
        "- Use `--rtma-mode selected_messages` (current default, -16% bytes, -33% time)",
        "- Increase `download_concurrency` to 4-8 for multi-file batches",
        "- If Transfer Manager benchmarks faster: switch single-file download to",
        "  `download_file()` + `TransferConfig(max_concurrency=10)`",
        "",
        "**HPC production (recommended):**",
        "- Use SLURM array jobs: 1 task per day × 2 products (or 1 task per product-day)",
        "- Each task downloads and processes 24 hours serially",
        "- Shared NFS/Lustre raw cache avoids duplicate downloads across tasks",
        "- At 10 Gbps HPC network, download becomes negligible; decode/extract dominate",
        "- Full Stage 1 at 128 tasks: ~3.5h (selected_messages) or ~5h (full_file)",
        "",
        "**Do NOT spatially subset as primary strategy** — keep full CONUS grid",
        "for future projects; message-level selection (11/13 messages) already gives",
        "the best practically achievable reduction without spatial tiling.",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    from src.pipeline.config import load_config
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    cfg       = load_config(Path(args.config))
    data_root = cfg.effective_data_root(override=args.data_root)
    LOGGER.info("Data root: %s", data_root)

    out_dir   = cfg.output_dir("manifests", data_root) / "stage1_pilot" / "january_2023_extraction"
    bench_dir = cfg.output_dir("tmp", data_root) / "speed_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    # ── S3 clients ────────────────────────────────────────────────────────────
    s3_unsigned = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
    s3_transfer  = boto3.client("s3", config=BotoConfig(
        signature_version=UNSIGNED,
        max_pool_connections=25,   # needed for TransferConfig(max_concurrency=10)
    ))
    HTTPS_URL = f"{_HTTPS_BASE}/{_T03_KEY}"

    # ── Probe file size ───────────────────────────────────────────────────────
    LOGGER.info("Probing t03z file size ...")
    try:
        meta = s3_unsigned.head_object(Bucket=_BUCKET, Key=_T03_KEY)
        t03_bytes = meta["ContentLength"]
        LOGGER.info("t03z: %.1f MB", t03_bytes / 1e6)
    except Exception as exc:
        LOGGER.error("Cannot probe t03z: %s", exc)
        return 1

    # ── Single-file benchmarks ────────────────────────────────────────────────
    LOGGER.info("=== Single-file mechanism benchmarks ===")
    single_results: list[dict] = []

    # -- A: current boto3 GetObject --
    tmp_A = bench_dir / "modeA" / _T03_KEY.split("/")[-1]
    single_results.append(_run_mechanism(
        name="A_boto3_GetObject",
        description="boto3 s3.get_object() single stream, 8 MB chunks (current default)",
        download_fn=lambda: _dl_boto3_getobject(s3_unsigned, _BUCKET, _T03_KEY, tmp_A),
        verify=True,
        cleanup_path=tmp_A,
    ))

    # -- B: current selected-message Range GETs --
    tmp_B = bench_dir / "modeB" / _T03_KEY.split("/")[-1]
    def _bench_B():
        nbytes, dl_s, idx_s = _dl_selected_messages(_BUCKET, _T03_KEY, tmp_B)
        return nbytes, dl_s, idx_s
    single_results.append(_run_mechanism(
        name="B_selected_message_range",
        description="boto3 Range GETs for 11/13 messages via .idx (current optimized)",
        download_fn=_bench_B,
        verify=True,
        cleanup_path=tmp_B,
    ))

    # -- C: boto3 S3 Transfer Manager multipart --
    tmp_C = bench_dir / "modeC" / _T03_KEY.split("/")[-1]
    def _bench_C():
        return _dl_transfer_manager(s3_transfer, _BUCKET, _T03_KEY, tmp_C)
    single_results.append(_run_mechanism(
        name="C_boto3_TransferManager",
        description="boto3 download_file() + TransferConfig(max_concurrency=10, chunksize=8MB)",
        download_fn=_bench_C,
        verify=True,
        cleanup_path=tmp_C,
    ))

    # -- D: AWS CLI (not installed) --
    single_results.append({
        "mechanism":     "D_aws_cli",
        "description":   "aws s3 cp --no-sign-request (not installed)",
        "success":       False,
        "bytes":         None, "download_s": None, "overhead_s": None,
        "total_s":       None, "mb_s": None,
        "n_vars_decoded": None, "decode_s": None,
        "notes":         "AWS CLI not installed on this machine",
    })

    # -- E: httpx HTTPS --
    tmp_E = bench_dir / "modeE" / _T03_KEY.split("/")[-1]
    def _bench_E():
        return _dl_httpx(HTTPS_URL, tmp_E)
    single_results.append(_run_mechanism(
        name="E_httpx_https",
        description=f"httpx streaming GET {_HTTPS_BASE}/...",
        download_fn=_bench_E,
        verify=True,
        cleanup_path=tmp_E,
    ))

    # -- F: requests HTTPS --
    tmp_F = bench_dir / "modeF" / _T03_KEY.split("/")[-1]
    def _bench_F():
        return _dl_requests(HTTPS_URL, tmp_F)
    single_results.append(_run_mechanism(
        name="F_requests_https",
        description=f"requests.get streaming {_HTTPS_BASE}/...",
        download_fn=_bench_F,
        verify=False,   # same URL as E, skip redundant verify
        cleanup_path=tmp_F,
    ))

    # -- G: Herbie (not installed) --
    single_results.append({
        "mechanism":     "G_herbie",
        "description":   "Herbie-based RTMA download (not installed)",
        "success":       False,
        "bytes":         None, "download_s": None, "overhead_s": None,
        "total_s":       None, "mb_s": None,
        "n_vars_decoded": None, "decode_s": None,
        "notes":         "herbie not installed; not adding dependency for this audit",
    })

    # ── Determine fastest successful mechanism ────────────────────────────────
    successful = [r for r in single_results if r["success"] and r["mb_s"]]
    if successful:
        fastest = max(successful, key=lambda r: r["mb_s"] or 0)
        LOGGER.info(
            "Fastest single-file mechanism: %s at %.2f MB/s",
            fastest["mechanism"], fastest["mb_s"],
        )
    else:
        fastest = None

    # ── Full-file concurrency benchmark ──────────────────────────────────────
    parallel_results: list[dict] = []
    if not args.skip_parallel and not args.skip_fullfile_parallel:
        LOGGER.info("=== Full-file concurrency benchmark (8 × %.1f MB) ===", t03_bytes / 1e6)
        total_parallel_MB = t03_bytes * len(_PARALLEL_KEYS) / 1e6
        LOGGER.info("Total to download: %.0f MB across %d files", total_parallel_MB, len(_PARALLEL_KEYS))

        for n_workers in [1, 2, 4, 8]:
            w_dir = bench_dir / f"parallel_{n_workers}"
            LOGGER.info("  Concurrency w=%d ...", n_workers)
            pres = _run_parallel_bench(s3_unsigned, n_workers, _PARALLEL_KEYS, w_dir)
            parallel_results.append(pres)
            LOGGER.info(
                "  w=%d: %d files, %.0f MB, %.1fs, %.2f MB/s, median/file=%.1fs",
                n_workers, pres["n_ok"], pres["total_bytes"] / 1e6,
                pres["wall_s"], pres["agg_mb_s"], pres["median_file_s"] or 0,
            )
    else:
        LOGGER.info("Full-file parallel test skipped (--skip-parallel).")

    # ── Selected-messages concurrency benchmark ───────────────────────────────
    sel_parallel_results: list[dict] = []
    if not args.skip_parallel:
        from src.datasources.rtma import RtmaAwsConusDataSource
        import boto3 as _boto3

        LOGGER.info(
            "=== selected_messages concurrency benchmark (%d × ~71 MB) ===",
            len(_PARALLEL_KEYS),
        )
        # Create one shared datasource with a large enough S3 pool.
        # selected_messages makes 3 S3 calls per file (idx + 2 Range GETs);
        # allow 4×n_workers connections plus safety margin.
        max_workers_planned = 16
        pool_size = 4 * max_workers_planned + 10
        rtma_ds_bench = RtmaAwsConusDataSource(download_concurrency=1)
        rtma_ds_bench._s3_client_cached = _boto3.client(
            "s3",
            config=BotoConfig(signature_version=UNSIGNED, max_pool_connections=pool_size),
        )

        # Early stopping rule: stop if MB/s improvement < 15% over previous count.
        # Also stop if failures > 1% of files.
        prev_mb_s = None
        worker_counts = [1, 2, 4, 8, 16]
        for n_workers in worker_counts:
            keys = _PARALLEL_KEYS_16 if n_workers > 8 else _PARALLEL_KEYS
            LOGGER.info(
                "  sel_msgs w=%d (%d files × ~71 MB) ...", n_workers, len(keys)
            )
            pres = _run_parallel_selected_bench(rtma_ds_bench, n_workers, keys, bench_dir)
            sel_parallel_results.append(pres)
            LOGGER.info(
                "  sel w=%d: %d files, %.0f MB, %.1fs, %.2f MB/s, "
                "median=%.1fs, p90=%.1fs, failures=%d",
                n_workers, pres["n_ok"], pres["total_bytes"] / 1e6,
                pres["wall_s"], pres["agg_mb_s"],
                pres["median_file_s"] or 0, pres["p90_file_s"] or 0,
                pres["failures"],
            )
            # Compute speedup vs w=1
            if sel_parallel_results[0]["n_ok"] > 0:
                serial_wall = sel_parallel_results[0]["wall_s"]
                pres["speedup_vs_serial"] = round(serial_wall / pres["wall_s"], 2)
            # Early stopping checks
            fail_rate = pres["failures"] / max(pres["n_files"], 1)
            if fail_rate > 0.02:
                LOGGER.warning("  Early stop: failure rate %.0f%% > 2%%", fail_rate * 100)
                break
            if prev_mb_s is not None and pres["agg_mb_s"] > 0:
                improvement = (pres["agg_mb_s"] - prev_mb_s) / prev_mb_s
                if improvement < 0.15:
                    LOGGER.info(
                        "  Early stop: improvement %.0f%% < 15%% threshold at w=%d",
                        improvement * 100, n_workers,
                    )
                    break
            prev_mb_s = pres["agg_mb_s"]
    else:
        LOGGER.info("selected_messages parallel test skipped (--skip-parallel).")

    # ── Code implementation analysis ──────────────────────────────────────────
    impl_analysis = _analyze_implementation()

    # ── Scaling estimates ─────────────────────────────────────────────────────
    # Lookup measured values
    def _get_s(name: str, key: str) -> Optional[float]:
        for r in single_results:
            if r["mechanism"] == name and r["success"]:
                return r.get(key)
        return None

    decode_s = (_get_s("A_boto3_GetObject", "decode_s") or 1.3)   # cfgrib
    extract_s = 0.40   # observed in smoke test / acquisition benchmark

    a_dl  = _get_s("A_boto3_GetObject", "total_s") or _PREV_FULLFILE_DL_S
    b_dl  = _get_s("B_selected_message_range", "total_s") or _PREV_SELECTED_DL_S
    c_dl  = _get_s("C_boto3_TransferManager", "total_s")
    e_dl  = _get_s("E_httpx_https", "total_s")

    # best full-file parallel throughput
    best_par_agg = None
    best_par_n   = None
    if parallel_results:
        best_par = max(parallel_results, key=lambda r: r.get("agg_mb_s") or 0)
        if best_par["n_ok"] > 0:
            per_file_s = best_par["wall_s"] / best_par["n_ok"] * best_par["n_workers"]
            best_par_agg = per_file_s
            best_par_n   = best_par["n_workers"]

    # best selected_messages parallel throughput
    best_sel_agg = None
    best_sel_n   = None
    sel_w1_wall  = None
    if sel_parallel_results:
        sel_w1 = next((r for r in sel_parallel_results if r["n_workers"] == 1), None)
        sel_w1_wall = sel_w1["wall_s"] / max(sel_w1["n_ok"], 1) if sel_w1 else None
        best_sel = max(sel_parallel_results, key=lambda r: r.get("agg_mb_s") or 0)
        if best_sel["n_ok"] > 0:
            best_sel_agg = best_sel["wall_s"] / best_sel["n_ok"]  # effective s/file at best n
            best_sel_n   = best_sel["n_workers"]

    scaling_rows: list[dict] = []

    scaling_rows.append(_make_scaling_row(
        "A  boto3 GetObject (serial)",
        t03_bytes,
        a_dl,
        decode_s + extract_s,
        f"known baseline, {a_dl:.1f}s/file",
    ))
    scaling_rows.append(_make_scaling_row(
        "B  selected-message Range (serial, 11/13 msgs)",
        int(t03_bytes * 0.847),   # ~71 MB
        b_dl,
        1.3 + extract_s,          # faster decode (fewer messages)
        f"current default, {b_dl:.1f}s/file",
    ))
    if c_dl is not None:
        scaling_rows.append(_make_scaling_row(
            "C  boto3 TransferManager multipart (serial)",
            t03_bytes,
            c_dl,
            decode_s + extract_s,
            f"measured: {c_dl:.1f}s/file",
        ))
    if e_dl is not None:
        scaling_rows.append(_make_scaling_row(
            "E  httpx HTTPS (serial)",
            t03_bytes,
            e_dl,
            decode_s + extract_s,
            f"measured: {e_dl:.1f}s/file",
        ))
    if best_par_agg is not None:
        scaling_rows.append(_make_scaling_row(
            f"A  boto3 GetObject parallel (w={best_par_n})",
            t03_bytes,
            best_par_agg,
            decode_s + extract_s,
            f"aggregate at {best_par['agg_mb_s']:.1f} MB/s, {best_par_n} workers",
        ))

    # Add selected_messages parallel rows
    for pres in sel_parallel_results:
        if pres["n_ok"] > 0:
            eff_s = pres["wall_s"] / pres["n_ok"]   # effective wall-time per file slot
            scaling_rows.append(_make_scaling_row(
                f"B  selected_messages parallel (w={pres['n_workers']})",
                int(t03_bytes * 0.847),
                eff_s,
                1.3 + extract_s,
                f"{pres['agg_mb_s']:.1f} MB/s agg; "
                f"speedup {pres.get('speedup_vs_serial', '?')}x",
            ))

    # Theoretical HPC (10 Gbps = 1250 MB/s)
    scaling_rows.append(_make_scaling_row(
        "HPC theoretical (10 Gbps, full-file)",
        t03_bytes,
        t03_bytes / 1e6 / 1000,   # 1000 MB/s effective
        decode_s + extract_s,
        "10 Gbps network: ~0.08s/file download",
    ))

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = out_dir / "rtma_download_speed_benchmark.csv"
    all_rows: list[dict] = []
    for r in single_results:
        all_rows.append({"section": "single_file", **r})
    for r in parallel_results:
        all_rows.append({"section": "concurrency_fullfile", **r})
    for r in sel_parallel_results:
        all_rows.append({"section": "concurrency_selected", **r})
    for r in scaling_rows:
        all_rows.append({"section": "scaling", **r})

    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            all_keys = sorted({k for row in all_rows for k in row.keys()})
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
    LOGGER.info("Benchmark CSV: %s", csv_path)

    # ── Write MD ──────────────────────────────────────────────────────────────
    md_path = out_dir / "rtma_download_speed_benchmark.md"
    _write_md(md_path, single_results, parallel_results, impl_analysis, scaling_rows)
    LOGGER.info("Benchmark MD: %s", md_path)

    # ── Terminal report ───────────────────────────────────────────────────────
    sep = "=" * 72
    print(f"\n{sep}")
    print("RTMA Download-Speed Benchmark — 2023-01-01T03:00Z")
    print(sep)
    print()
    print("  --- Single-file mechanisms ---")
    print(f"  {'Mechanism':<35} {'Bytes':>8}  {'DL_s':>6}  {'Ovhd':>5}  {'MB/s':>6}  {'Vars':>4}  Notes")
    for r in single_results:
        if r["success"]:
            print(
                f"  {r['mechanism']:<35} {(r['bytes'] or 0)//1_000_000:>6} MB "
                f"{r['download_s']:>6.1f}s {r['overhead_s']:>5.1f}s "
                f"{r['mb_s']:>6.2f} {r['n_vars_decoded'] or '?':>4}  {r['notes'][:30]}"
            )
        else:
            print(f"  {r['mechanism']:<35} SKIPPED/FAILED  {r['notes'][:50]}")
    print()

    if parallel_results:
        print("  --- Concurrency (8-file Mode A, full-file) ---")
        print(f"  {'Workers':<8} {'Wall-s':>6} {'Agg MB/s':>9} {'Speedup':>8} {'Failures':>8}")
        serial_s = next((r["wall_s"] for r in parallel_results if r["n_workers"] == 1), None)
        for r in parallel_results:
            speedup = round(serial_s / r["wall_s"], 2) if serial_s and r["wall_s"] else "?"
            print(
                f"  {r['n_workers']:<8} {r['wall_s']:>6.1f}s "
                f"{r['agg_mb_s']:>9.2f} {speedup!s:>8}x {r['failures']:>8}"
            )
        print()
        best = max(parallel_results, key=lambda r: r.get("agg_mb_s") or 0)
        print(f"  Best aggregate throughput: {best['agg_mb_s']:.2f} MB/s at w={best['n_workers']}")

    if sel_parallel_results:
        print()
        print("  --- Concurrency (selected_messages mode) ---")
        print(f"  {'Workers':<8} {'Files':<6} {'Wall-s':>6} {'Agg MB/s':>9} "
              f"{'Speedup':>8} {'Med/file':>9} {'P90/file':>9} {'Failures':>8}")
        for r in sel_parallel_results:
            sp = r.get("speedup_vs_serial") or "?"
            print(
                f"  {r['n_workers']:<8} {r['n_files']:<6} {r['wall_s']:>6.1f}s "
                f"{r['agg_mb_s']:>9.2f} {sp!s:>8}x "
                f"{r['median_file_s'] or '?':>9} {r['p90_file_s'] or '?':>9} "
                f"{r['failures']:>8}"
            )
        best_sel = max(sel_parallel_results, key=lambda r: r.get("agg_mb_s") or 0)
        print(f"\n  Best sel_msgs throughput: {best_sel['agg_mb_s']:.2f} MB/s at w={best_sel['n_workers']}")
        # Recommendation
        stable = [r for r in sel_parallel_results if r["failures"] == 0]
        if stable:
            # Recommend worker count within 90% of max MB/s
            max_mbs = max(r["agg_mb_s"] for r in stable)
            rec = next((r for r in stable if r["agg_mb_s"] >= 0.9 * max_mbs), stable[-1])
            print(f"  Recommended local workers: {rec['n_workers']} "
                  f"({rec['agg_mb_s']:.1f} MB/s, within 10% of peak)")

    print()
    print("  --- Implementation analysis ---")
    for k, v in impl_analysis.items():
        if k != "recommended_improvements":
            print(f"  {k}: {str(v)[:80]}")
    print("  Recommended improvements:")
    for item in impl_analysis["recommended_improvements"]:
        print(f"    - {item[:80]}")

    print()
    print("  --- Scaling estimates ---")
    print(f"  {'Method':<45} {'MB/hr':>6} {'MB/s':>5} {'Jan raw':>8} {'Full raw':>9} {'Serial/hr':>10}")
    for r in scaling_rows:
        print(
            f"  {r['method']:<45} {r['mb_per_hour']:>6.0f} {r['mb_s_per_file']:>5.1f} "
            f"{r['jan_raw_gb']:>7.1f}GB {r['full_raw_tb']:>8.2f}TB "
            f"{r['serial_s_per_hour']:>9.1f}s"
        )

    print()
    print(f"  Benchmark CSV: {csv_path}")
    print(f"  Benchmark MD:  {md_path}")
    print(sep)

    # Clean up any leftover temp files
    try:
        shutil.rmtree(bench_dir, ignore_errors=True)
        LOGGER.info("Cleaned up %s", bench_dir)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
