#!/usr/bin/env python3
"""Discover MRMS and RTMA grid definitions for the Flash-NH Stage 1 pilot.

Downloads (or reuses cached) one sample GRIB2 file per product for the
configured sample time, decodes each file, extracts grid/coordinate/projection
metadata, writes JSON grid-definition files, and generates small PNG previews.

This is Milestone 2A — a prerequisite for computing basin-grid overlap weights.

Outputs (under configured data root):
  09_manifests/stage1_pilot/grid_definitions/
      mrms_grid_definition.json
      rtma_grid_definition.json
      grid_definition_summary.json
      grid_definition_summary.md
      manifest.json / summary.json / summary.md / run_command.txt / git_commit.txt
  06_qc_reports/stage1_pilot/grid_definitions/
      mrms_grid_preview.png
      rtma_grid_preview.png
  00_raw/mrms/   (sample GRIB2 file, downloaded if not already cached)
  00_raw/rtma/   (sample GRIB2 file, downloaded if not already cached)

Usage:
    python scripts/build_stage1_grid_definitions.py --config configs/pilot_stage1.yaml
    python scripts/build_stage1_grid_definitions.py --data-root /my/external/Flash-NH_data
    python scripts/build_stage1_grid_definitions.py --dry-run

Dry-run mode: checks for cached files; decodes and generates outputs if found,
but does NOT initiate any S3 downloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_config, config_to_dict, PipelineConfig
from src.pipeline.provenance import write_run_manifest, git_commit_hash
from src.pipeline.grid_definitions import (
    SAMPLE_TIME_DEFAULT,
    discover_mrms_grid,
    discover_rtma_grid,
    download_mrms_sample,
    download_rtma_sample,
    find_cached_mrms_sample,
    find_cached_rtma_sample,
    plot_grid_preview,
    write_grid_definition_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "pilot_stage1.yaml"),
        help="Path to YAML config",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Override data root path (else uses config / FLASHNH_DATA_ROOT env)",
    )
    p.add_argument(
        "--sample-time",
        default=SAMPLE_TIME_DEFAULT,
        help=f"UTC datetime for the sample hour (default: {SAMPLE_TIME_DEFAULT})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Do not download any files. If cached files exist, decode and produce "
            "outputs; otherwise report MISSING and skip that product."
        ),
    )
    return p.parse_args()


def _build_summary_md(
    sample_dt: datetime,
    mrms_meta: dict,
    rtma_meta: dict,
    validation: dict,
) -> str:
    overall = "PASS" if all(v for v in validation.values() if isinstance(v, bool)) else "FAIL"
    lines = [
        "# Stage 1 Grid Definition Summary",
        "",
        f"**Overall:** {overall}",
        f"**Sample time (UTC):** {sample_dt.isoformat()}",
        "",
        "## MRMS QPE 1h Pass1",
        "",
    ]

    def _grid_row(meta: dict, product: str) -> list[str]:
        rows = []
        if "error" in meta:
            rows.append(f"**ERROR:** {meta['error']}")
            return rows
        rows.append(f"| Field | Value |")
        rows.append(f"|---|---|")
        rows.append(f"| Grid type | {meta.get('grid_type', '?')} |")
        shape = meta.get("grid_shape_rows_cols")
        rows.append(f"| Shape (rows x cols) | {shape} |")
        res_deg = meta.get("approx_resolution_deg")
        res_m = meta.get("approx_resolution_m")
        if res_deg:
            rows.append(f"| Resolution | {res_deg} deg (~{res_deg*111:.1f} km) |")
        elif res_m:
            rows.append(f"| Resolution | {res_m:.0f} m |")
        rows.append(
            f"| Bounding box | lon [{meta.get('bbox_lon_min','?'):.3f}, "
            f"{meta.get('bbox_lon_max','?'):.3f}] "
            f"lat [{meta.get('bbox_lat_min','?'):.3f}, "
            f"{meta.get('bbox_lat_max','?'):.3f}] |"
        )
        rows.append(f"| CONUS overlap | {meta.get('conus_overlap')} |")
        rows.append(f"| lat_descending | {meta.get('lat_descending')} |")
        rows.append(f"| Sample valid time | {meta.get('sample_valid_time', '?')} |")
        qc = meta.get("qc_stats") or {}
        if qc:
            rows.append(
                f"| QC (min/max/nan%) | {qc.get('min','?'):.3g} / "
                f"{qc.get('max','?'):.3g} / {qc.get('nan_pct','?'):.1f}% |"
            )
        for note in (meta.get("notes") or []):
            rows.append(f"| Note | {note} |")
        for w in (meta.get("warnings") or []):
            rows.append(f"| **WARNING** | {w} |")
        return rows

    lines += _grid_row(mrms_meta, "MRMS")
    lines += ["", "## RTMA 2.5km NDFD Analysis", ""]
    lines += _grid_row(rtma_meta, "RTMA")
    lines += ["", "## Validation", ""]
    for k, v in validation.items():
        if isinstance(v, bool):
            lines.append(f"- **{k}**: {'PASS' if v else 'FAIL'}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Next Step",
        "",
        "Milestone 2B: compute basin-grid overlap weights for the 50 pilot basins.",
        "Use the exact grid coordinate arrays from these sample files as inputs.",
        "See `docs/stage1_grid_definitions.md` for the implementation plan.",
    ]
    return "\n".join(lines) + "\n"


def _process_product(
    product: str,
    raw_dir: Path,
    sample_dt: datetime,
    grid_def_dir: Path,
    qc_dir: Path,
    dry_run: bool,
) -> tuple[dict, dict]:
    """Download/find sample, discover grid, write JSON and preview. Return (meta, checks)."""
    find_fn = find_cached_mrms_sample if product == "mrms" else find_cached_rtma_sample
    download_fn = download_mrms_sample if product == "mrms" else download_rtma_sample
    discover_fn = discover_mrms_grid if product == "mrms" else discover_rtma_grid
    json_name = f"{product}_grid_definition.json"
    preview_name = f"{product}_grid_preview.png"
    cmap = "Blues" if product == "mrms" else "RdYlBu_r"
    source_name = "mrms_qpe_1h_pass1" if product == "mrms" else "rtma_conus_aws_2p5km"

    # Find or download sample file
    sample_file = find_fn(raw_dir, sample_dt)
    downloaded = False
    if sample_file is None:
        if dry_run:
            print(f"  [{product.upper()}] no cached file found; skipping download (--dry-run)")
            meta: dict[str, Any] = {
                "product": source_name,
                "error": "no cached sample file; run without --dry-run to download",
                "notes": [], "warnings": [],
            }
            checks = {
                f"{product}_sample_found": False,
                f"{product}_array_decoded": False,
                f"{product}_grid_def_written": False,
                f"{product}_preview_written": False,
                f"{product}_conus_overlap": False,
            }
            return meta, checks
        print(f"  [{product.upper()}] downloading sample for {sample_dt.isoformat()} ...")
        sample_file = download_fn(raw_dir, sample_dt)
        downloaded = True

    if sample_file is None or not sample_file.exists():
        print(f"  [{product.upper()}] sample file unavailable")
        meta = {
            "product": source_name,
            "error": "sample file could not be obtained",
            "notes": [], "warnings": [],
        }
        checks = {
            f"{product}_sample_found": False,
            f"{product}_array_decoded": False,
            f"{product}_grid_def_written": False,
            f"{product}_preview_written": False,
            f"{product}_conus_overlap": False,
        }
        return meta, checks

    reused = "downloaded" if downloaded else "reused from cache"
    print(f"  [{product.upper()}] sample file {reused}: {sample_file}")
    print(f"  [{product.upper()}] size: {sample_file.stat().st_size:,} bytes")

    # Discover grid
    meta, preview_arr = discover_fn(sample_file, source_name=source_name)

    shape = meta.get("grid_shape_rows_cols")
    bbox_ok = meta.get("conus_overlap")
    print(f"  [{product.upper()}] grid_type={meta.get('grid_type','?')}  shape={shape}  conus={bbox_ok}")
    for note in (meta.get("notes") or []):
        print(f"            note: {note}")
    for w in (meta.get("warnings") or []):
        print(f"            WARNING: {w}")

    # Write JSON
    grid_def_path = grid_def_dir / json_name
    write_grid_definition_json(meta, grid_def_path)
    print(f"  [{product.upper()}] written: {grid_def_path.name}")

    # Preview plot
    preview_path = qc_dir / preview_name
    bbox = None
    lat_desc = meta.get("lat_descending")
    if all(meta.get(k) is not None for k in ("bbox_lon_min", "bbox_lat_min", "bbox_lon_max", "bbox_lat_max")):
        bbox = {
            "lon_min": meta["bbox_lon_min"],
            "lon_max": meta["bbox_lon_max"],
            "lat_min": meta["bbox_lat_min"],
            "lat_max": meta["bbox_lat_max"],
        }
    preview_ok = False
    if preview_arr is not None:
        preview_ok = plot_grid_preview(
            preview_arr,
            preview_path,
            title=(
                f"{source_name} | {meta.get('preview_variable', '?')} | "
                f"{meta.get('sample_valid_time', '?')}"
            ),
            var_label=f"{meta.get('preview_variable', '?')} ({meta.get('units', '?')})",
            bbox=bbox,
            lat_descending=lat_desc,
            cmap=cmap,
        )
        if preview_ok:
            print(f"  [{product.upper()}] preview:  {preview_path.name}")
        else:
            print(f"  [{product.upper()}] preview generation failed (matplotlib issue?)")
    else:
        print(f"  [{product.upper()}] no array available for preview")

    checks = {
        f"{product}_sample_found": sample_file.exists(),
        f"{product}_file_nonempty": sample_file.stat().st_size > 0,
        f"{product}_array_decoded": shape is not None and len(shape) == 2,
        f"{product}_conus_overlap": bool(bbox_ok),
        f"{product}_grid_def_written": grid_def_path.exists(),
        f"{product}_preview_written": preview_ok,
    }
    return meta, checks


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        cfg = load_config(config_path)
        print(f"Config: {config_path}")
    else:
        cfg = PipelineConfig()
        print(f"Config not found at {config_path}; using defaults.")

    data_root = cfg.effective_data_root(override=args.data_root)
    sample_dt = datetime.fromisoformat(args.sample_time)
    dry_run = args.dry_run

    print(f"Flash-NH Stage 1 - Grid Definition Discovery (Milestone 2A)")
    print(f"  Config:      {config_path}")
    print(f"  Data root:   {data_root}")
    print(f"  Sample time: {sample_dt.isoformat()} UTC")
    print(f"  Dry run:     {dry_run}")
    print(f"  Git:         {git_commit_hash() or 'unknown'}")

    # Output directories
    grid_def_dir = data_root / "09_manifests" / "stage1_pilot" / "grid_definitions"
    qc_dir = data_root / "06_qc_reports" / "stage1_pilot" / "grid_definitions"
    raw_mrms_dir = data_root / "00_raw" / "mrms"
    raw_rtma_dir = data_root / "00_raw" / "rtma"
    for d in [grid_def_dir, qc_dir, raw_mrms_dir, raw_rtma_dir]:
        d.mkdir(parents=True, exist_ok=True)

    validation: dict[str, Any] = {}
    mrms_meta: dict = {}
    rtma_meta: dict = {}

    print(f"\nProcessing MRMS ...")
    mrms_meta, mrms_checks = _process_product(
        "mrms", raw_mrms_dir, sample_dt, grid_def_dir, qc_dir, dry_run
    )
    validation.update(mrms_checks)

    print(f"\nProcessing RTMA ...")
    rtma_meta, rtma_checks = _process_product(
        "rtma", raw_rtma_dir, sample_dt, grid_def_dir, qc_dir, dry_run
    )
    validation.update(rtma_checks)

    # Grid summary JSON
    grid_summaries = {}
    for product, meta in [("mrms", mrms_meta), ("rtma", rtma_meta)]:
        if "error" not in meta:
            grid_summaries[product] = {
                "grid_type": meta.get("grid_type"),
                "shape": meta.get("grid_shape_rows_cols"),
                "bbox": {
                    "lon_min": meta.get("bbox_lon_min"),
                    "lon_max": meta.get("bbox_lon_max"),
                    "lat_min": meta.get("bbox_lat_min"),
                    "lat_max": meta.get("bbox_lat_max"),
                },
                "resolution_deg": meta.get("approx_resolution_deg"),
                "resolution_m": meta.get("approx_resolution_m"),
                "lat_descending": meta.get("lat_descending"),
                "conus_overlap": meta.get("conus_overlap"),
                "sample_valid_time": meta.get("sample_valid_time"),
            }

    summary_payload = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_time": sample_dt.isoformat(),
        "dry_run": dry_run,
        "grid_summaries": grid_summaries,
        "validation": validation,
        "overall": "PASS" if all(v for v in validation.values() if isinstance(v, bool)) else "FAIL",
    }

    summary_json_path = grid_def_dir / "grid_definition_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, default=str)

    summary_md_path = grid_def_dir / "grid_definition_summary.md"
    summary_md_path.write_text(
        _build_summary_md(sample_dt, mrms_meta, rtma_meta, validation),
        encoding="utf-8",
    )

    # Provenance manifest
    run_cmd = f"python scripts/build_stage1_grid_definitions.py --config {args.config}"
    if args.data_root:
        run_cmd += f" --data-root {args.data_root}"
    if args.sample_time != SAMPLE_TIME_DEFAULT:
        run_cmd += f" --sample-time {args.sample_time}"
    if dry_run:
        run_cmd += " --dry-run"

    write_run_manifest(
        grid_def_dir,
        run_command=run_cmd,
        config_dict=config_to_dict(cfg),
        input_paths={
            "mrms_sample_file": mrms_meta.get("source_file"),
            "rtma_sample_file": rtma_meta.get("source_file"),
        },
        output_paths={
            "mrms_grid_def": str(grid_def_dir / "mrms_grid_definition.json"),
            "rtma_grid_def": str(grid_def_dir / "rtma_grid_definition.json"),
            "mrms_preview": str(qc_dir / "mrms_grid_preview.png"),
            "rtma_preview": str(qc_dir / "rtma_grid_preview.png"),
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
        validation_results=validation,
    )

    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    for k, v in validation.items():
        tag = "PASS" if v else "FAIL"
        print(f"  {tag}  {k}")

    overall = summary_payload["overall"]
    print(f"\nOverall: {overall}")
    print(f"\nKey outputs:")
    print(f"  Grid defs: {grid_def_dir}")
    print(f"  Previews:  {qc_dir}")
    print(f"  Summary:   {summary_json_path}")

    if overall != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
