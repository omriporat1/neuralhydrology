#!/usr/bin/env python
"""Unified final validation runner, using cached samples and the standard estimation engine."""

from datetime import datetime
from pathlib import Path
import sys
import csv
import json
from typing import Any, Optional

from src.estimate import (
    Config, EstimateResult, run_estimation,
    get_enabled_sources, _to_iso,
)
from src.datasources.base import Region, CONUS_BBOX


def _to_bytes(value: Optional[int]) -> Optional[int]:
    return int(value) if value is not None else None


def _gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)


def _tib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 ** 4)


def _pct_reduce(bigger: Optional[int], smaller: Optional[int]) -> Optional[float]:
    if bigger is None or smaller is None or bigger <= 0:
        return None
    return (1.0 - (float(smaller) / float(bigger))) * 100.0


def _fmt_opt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


MAIN_PRED_START = datetime.fromisoformat("2020-10-14T00:00:00")
MAIN_PRED_END = datetime.fromisoformat("2025-12-31T23:59:59")
SAMPLE_START = datetime.fromisoformat("2023-01-01T00:00:00")
SAMPLE_END = datetime.fromisoformat("2023-01-01T23:59:59")
REPORT_ROOT = Path("reports/final_validation")


def build_rows(results: list[EstimateResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        assumptions = result.assumptions or {}
        source_name = result.source
        full_raw = _to_bytes(result.raw_hot_full_file_bytes)
        selected_global = _to_bytes(result.raw_hot_selected_bytes)
        selected_conus = _to_bytes(result.raw_hot_selected_conus_bytes)
        canonical = _to_bytes(result.raw_hot_bytes)
        derived = _to_bytes(result.derived_hot_bytes)
        peak = _to_bytes(result.peak_local_bytes)

        row = {
            "product": source_name,
            "sample_files": result.sample_files,
            "full_raw_download_estimate_tib": _fmt_opt(_tib(full_raw)),
            "selected_variable_estimate_tib": _fmt_opt(_tib(selected_global)),
            "selected_conus_estimate_tib": _fmt_opt(_tib(selected_conus)),
            "canonical_raw_estimate_used_tib": _fmt_opt(_tib(canonical)),
            "basin_average_derived_estimate_tib": _fmt_opt(_tib(derived)),
            "peak_local_estimate_tib": _fmt_opt(_tib(peak)),
            "reduction_full_to_selected_variable_pct": _fmt_opt(_pct_reduce(full_raw, selected_global)),
            "reduction_selected_variable_to_conus_pct": _fmt_opt(_pct_reduce(selected_global, selected_conus)),
            "required_archive_window": f"{result.era5_land_required_data_start or result.gdas_required_data_start or result.imerg_required_data_start} -> {result.era5_land_required_data_end or result.gdas_required_data_end or result.imerg_required_data_end}",
            "notes/caveats": result.notes,
        }
        rows.append(row)

    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    headers = list(rows[0].keys()) if rows else []
    lines = ["# Final Storage Summary", "", "## Consolidated Table", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        vals = []
        for h in headers:
            value = row.get(h)
            if isinstance(value, float):
                vals.append(_fmt_opt(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(results: list[EstimateResult], rows: list[dict[str, Any]], out_path: Path) -> None:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": [
            {
                k: v for k, v in result.__dict__.items()
                if not k.startswith("_")
            }
            for result in results
        ],
        "summary": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main():
    print("Running unified estimation across all implemented datasources (cached-sample mode)...")
    
    config = Config(
        sample_start=SAMPLE_START,
        sample_end=SAMPLE_END,
        full_start=MAIN_PRED_START,
        full_end=MAIN_PRED_END,
        region=Region(name="conus_bbox", bbox=CONUS_BBOX.bbox),
        out_dir=Path("data") / "sample",
        report_dir=REPORT_ROOT,
        variables=["precip", "temp_2m", "humidity", "wind_10m", "surface_pressure", "shortwave_down", "soil_moisture_top", "swe_or_snow_depth"],
        include_longwave_stage2=True,
        derived_dtype="float32",
        derived_format="parquet",
        scratch_multiplier=1.5,
        n_basins=9000,
        concurrency=16,
        dry_run=True,
        mrms_backend="aws",
        mrms_debug_listing=False,
        range_mode="mrms_aligned",
        make_preview=False,
        gfs_max_lead=24,
        ifs_max_lead=24,
    )

    results, ratio_info = run_estimation(config)
    
    # Print timing/sample info for each source
    for result in results:
        print(f"{result.source}: sample_objects={result.sample_files}")

    # Build consolidated rows
    rows = build_rows(results)
    
    # Write reports
    csv_path = REPORT_ROOT / "final_storage_summary.csv"
    md_path = REPORT_ROOT / "final_storage_summary.md"
    json_path = REPORT_ROOT / "final_storage_summary.json"
    
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_json(results, rows, json_path)
    
    print(f"FINAL_REPORT_CSV={csv_path.as_posix()}")
    print(f"FINAL_REPORT_MD={md_path.as_posix()}")
    print(f"FINAL_REPORT_JSON={json_path.as_posix()}")


if __name__ == "__main__":
    main()
