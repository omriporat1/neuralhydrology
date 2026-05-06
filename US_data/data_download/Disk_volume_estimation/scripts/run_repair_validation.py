#!/usr/bin/env python3
"""Targeted repair validation for GFS, IMERG, IFS datasources.

This script runs a focused audit on the three sources identified for repair:
- GFS: Byte-range extraction audit (.idx verification)
- IMERG: CONUS crop robustness verification
- IFS: MARS cycle access strategy testing (00/06/12/18)

All output is centralized to run_02_repair with a final_bundle for easy review.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
import shutil
import sys
from pathlib import Path
import io
from contextlib import redirect_stdout, redirect_stderr

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datasources.base import CONUS_BBOX
import src.estimate as estimate_module
from src.estimate import Config, config_to_dict, run_estimation
from src.report import write_csv, write_json


# Global configuration
REPO_ROOT = Path(__file__).parent.parent
AUDIT_ROOT = REPO_ROOT / "reports" / "audit_2026_04_29" / "run_02_repair"
RAW_OUTPUT_ROOT = AUDIT_ROOT / "raw_tool_outputs"
LOGS_ROOT = AUDIT_ROOT / "logs"
REQUEST_SPECS_ROOT = AUDIT_ROOT / "request_specs"
TABLES_ROOT = AUDIT_ROOT / "tables"
PREVIEW_ROOT = AUDIT_ROOT / "previews"
SAMPLE_DOWNLOADS_ROOT = AUDIT_ROOT / "sample_downloads"
FINAL_BUNDLE_ROOT = AUDIT_ROOT / "final_bundle"

# Targeted sources only
SOURCE_NAMES = ["gfs_conus_aws_0p25", "imerg_late_daily_conus", "ifs_mars_conus"]


def main():
    """Run targeted repair validation audit."""
    global AUDIT_ROOT, RAW_OUTPUT_ROOT, LOGS_ROOT, REQUEST_SPECS_ROOT, TABLES_ROOT, PREVIEW_ROOT, SAMPLE_DOWNLOADS_ROOT, FINAL_BUNDLE_ROOT
    
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _clean_run_outputs()
    RAW_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Redirect stdout/stderr to UTF-16 transcript
    transcript_path = LOGS_ROOT / "run_stdout_stderr.txt"
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    
    with transcript_path.open("w", encoding="utf-16") as transcript:
        # Configure logging to write to transcript
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=transcript,
            force=True
        )
        with redirect_stdout(transcript), redirect_stderr(transcript):
            # Configuration: 24-hour audit sample with targeted sources
            config = _audit_config()
            print(f"=== REPAIR VALIDATION AUDIT ===")
            print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
            print(f"Audit root: {AUDIT_ROOT}")
            print(f"Sample date: 2023-01-01 (24-hour window)")
            print(f"Sources: {', '.join(SOURCE_NAMES)}")
            print()

            original_get_enabled_sources = estimate_module.get_enabled_sources

            def _targeted_sources(target_config: Config):
                return [
                    source
                    for source in original_get_enabled_sources(target_config)
                    if source.name in SOURCE_NAMES
                ]

            estimate_module.get_enabled_sources = _targeted_sources
            try:
                # Run estimation with targeted sources only
                print("Running estimation for targeted sources...")
                results, ratio_info = run_estimation(config)
            finally:
                estimate_module.get_enabled_sources = original_get_enabled_sources

            print(f"Completed audit for {len(results)} targeted sources")
            print()

            # Generate decision-support reports
            print("Generating decision-support reports...")
            rows = []
            for result in results:
                row = {
                    "source": result.source,
                    "status": result.status,
                    "full_file_bytes": result.full_file_bytes or "N/A",
                    "selected_variable_bytes": result.selected_variable_bytes or "N/A",
                    "selected_conus_bytes": result.selected_conus_bytes or "N/A",
                    "raw_sample_bytes": result.raw_sample_bytes or "N/A",
                    "raw_sample_full_file_bytes": result.raw_sample_full_file_bytes or "N/A",
                    "raw_sample_selected_bytes": result.raw_sample_selected_bytes or "N/A",
                    "raw_sample_selected_conus_bytes": result.raw_sample_selected_conus_bytes or "N/A",
                    "peak_local_bytes": result.peak_local_bytes or "N/A",
                }
                rows.append(row)

            TABLES_ROOT.mkdir(parents=True, exist_ok=True)
            # Write CSV summary
            csv_path = TABLES_ROOT / "repair_summary.csv"
            import pandas as pd
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Wrote CSV summary: {csv_path}")

            # Write JSON summary
            json_path = TABLES_ROOT / "repair_summary.json"
            payload = {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "audit_type": "repair_validation",
                "sample_date": "2023-01-01",
                "sources": rows,
            }
            json_path.write_text(json.dumps(payload, indent=2))
            print(f"Wrote JSON summary: {json_path}")
    
    # Post-processing: split transcript and generate final_bundle
    print(f"Post-processing outputs...")
    _split_transcript()
    _move_preview_pngs()
    _create_final_bundle()
    
    print(f"Completed audit run under {AUDIT_ROOT}")
    print(f"Final bundle: {FINAL_BUNDLE_ROOT}")


def _audit_config() -> Config:
    """Build configuration for 24-hour repair validation audit."""
    sample_start = datetime(2023, 1, 1, 0, 0, 0)
    sample_end = datetime(2023, 1, 1, 23, 59, 59)
    
    full_start = datetime(2020, 10, 14, 0, 0, 0)  # MRMS historical start
    full_end = datetime(2025, 12, 31, 23, 59, 59)
    
    # Create sample download directory
    sample_root = AUDIT_ROOT / "sample_downloads"
    sample_root.mkdir(parents=True, exist_ok=True)
    
    config = Config(
        sample_start=sample_start,
        sample_end=sample_end,
        full_start=full_start,
        full_end=full_end,
        region=CONUS_BBOX,
        out_dir=sample_root,
        report_dir=RAW_OUTPUT_ROOT,
        variables=[
            "precip",
            "temp_2m",
            "humidity",
            "wind_10m",
            "surface_pressure",
            "shortwave_down",
            "soil_moisture_top",
            "swe_or_snow_depth",
        ],
        include_longwave_stage2=True,
        derived_dtype="float32",
        derived_format="parquet",
        scratch_multiplier=1.5,
        n_basins=9000,
        concurrency=16,
        dry_run=False,
        mrms_backend="aws",
        mrms_debug_listing=False,
        range_mode="mrms_aligned",
        make_preview=True,
        gfs_max_lead=24,
        ifs_max_lead=24,
    )
    return config


def _clean_run_outputs() -> None:
    """Remove prior generated artifacts for a fresh targeted validation run."""
    for path in [RAW_OUTPUT_ROOT, LOGS_ROOT, REQUEST_SPECS_ROOT, TABLES_ROOT, PREVIEW_ROOT, SAMPLE_DOWNLOADS_ROOT, FINAL_BUNDLE_ROOT]:
        if path.exists():
            shutil.rmtree(path)


def _split_transcript() -> None:
    """Split UTF-16 transcript into per-source logs and general lines."""
    transcript_path = LOGS_ROOT / "run_stdout_stderr.txt"
    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}")
        return
    
    try:
        content = transcript_path.read_text(encoding="utf-16")
    except Exception as e:
        print(f"Failed to read transcript: {e}")
        return
    
    # Write UTF-8 copy for readability
    utf8_path = LOGS_ROOT / "run_stdout_stderr.utf8.txt"
    utf8_path.write_text(content, encoding="utf-8")
    
    # Split by source
    per_source_lines = {source: [] for source in SOURCE_NAMES}
    general_lines = []
    
    for line in content.splitlines():
        matched = False
        for source_name in SOURCE_NAMES:
            if source_name in line:
                per_source_lines[source_name].append(line)
                matched = True
                break
        if not matched:
            general_lines.append(line)
    
    # Write per-source logs
    for source_name in SOURCE_NAMES:
        log_path = LOGS_ROOT / f"{source_name}.log"
        log_path.write_text("\n".join(per_source_lines[source_name]), encoding="utf-8")
    
    # Write general log
    general_path = LOGS_ROOT / "run_stdout_stderr.general.txt"
    general_path.write_text("\n".join(general_lines), encoding="utf-8")


def _move_preview_pngs() -> None:
    """Move preview PNGs to organized subdirectories by source."""
    preview_src = RAW_OUTPUT_ROOT / "preview"
    if not preview_src.exists():
        return
    
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    for png_path in sorted(preview_src.glob("*.png")):
        for source_name in SOURCE_NAMES:
            if png_path.name.startswith(source_name + "_"):
                destination_dir = PREVIEW_ROOT / source_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(png_path, destination_dir / png_path.name)
                break


def _create_final_bundle() -> None:
    """Create self-contained final_bundle with all outputs for easy review/upload."""
    FINAL_BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Copy summaries
    summary_src = TABLES_ROOT / "repair_summary.csv"
    if summary_src.exists():
        shutil.copy2(summary_src, FINAL_BUNDLE_ROOT / "repair_summary.csv")
    
    summary_json_src = TABLES_ROOT / "repair_summary.json"
    if summary_json_src.exists():
        shutil.copy2(summary_json_src, FINAL_BUNDLE_ROOT / "repair_summary.json")
    
    # Copy previews
    if PREVIEW_ROOT.exists():
        shutil.copytree(
            PREVIEW_ROOT,
            FINAL_BUNDLE_ROOT / "previews",
            dirs_exist_ok=True
        )
    
    # Copy logs
    if LOGS_ROOT.exists():
        logs_dest = FINAL_BUNDLE_ROOT / "logs"
        logs_dest.mkdir(exist_ok=True)
        for log_file in LOGS_ROOT.glob("*.log"):
            shutil.copy2(log_file, logs_dest / log_file.name)
        for log_file in LOGS_ROOT.glob("*.txt"):
            shutil.copy2(log_file, logs_dest / log_file.name)
    
    # Copy request specs if generated
    if REQUEST_SPECS_ROOT.exists():
        shutil.copytree(
            REQUEST_SPECS_ROOT,
            FINAL_BUNDLE_ROOT / "request_specs",
            dirs_exist_ok=True
        )
    
    # Copy sample downloads
    if SAMPLE_DOWNLOADS_ROOT.exists():
        shutil.copytree(
            SAMPLE_DOWNLOADS_ROOT,
            FINAL_BUNDLE_ROOT / "sample_downloads",
            dirs_exist_ok=True
        )
    
    # Create README for final bundle
    readme_path = FINAL_BUNDLE_ROOT / "README.md"
    readme_path.write_text(f"""# Repair Validation Audit - run_02_repair

Generated: {datetime.utcnow().isoformat()}Z

## Sources Tested
- GFS AWS (.idx byte-range extraction audit)
- IMERG Late Daily (CONUS crop robustness)
- IFS MARS (MARS cycle access strategy testing: 00/06/12/18)

## Contents
- `repair_summary.csv/json` - Summary metrics for each source
- `previews/` - QC visualization maps (north-up, proper axes)
- `logs/` - Per-source audit logs and general transcript
- `request_specs/` - Request specifications for each source
- `sample_downloads/` - Downloaded sample files

## Key Metrics to Review
1. **GFS**: Check actual bytes vs theoretical bytes (should differ significantly)
2. **IMERG**: Verify crop bounds and array shape
3. **IFS**: Check logs for MARS cycle success/failure (00/06/12/18)

## Next Steps
Review logs in `logs/` directory for detailed per-source audit information.
""")


if __name__ == "__main__":
    main()
