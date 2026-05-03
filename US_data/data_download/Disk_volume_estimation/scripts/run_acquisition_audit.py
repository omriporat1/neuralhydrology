#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Optional


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent] + list(here.parents):
        if (candidate / "run_final_validation_all_sources.py").exists() and (candidate / "src").exists():
            return candidate
    return here.parents[1]


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_final_validation_all_sources import build_rows, build_validation_checks  # noqa: E402
from src.datasources.base import CONUS_BBOX  # noqa: E402
from src.estimate import Config, config_to_dict, run_estimation  # noqa: E402


MAIN_PRED_START = datetime.fromisoformat("2020-10-14T00:00:00")
MAIN_PRED_END = datetime.fromisoformat("2025-12-31T23:59:59")
SAMPLE_START = datetime.fromisoformat("2023-01-01T00:00:00")
SAMPLE_END = datetime.fromisoformat("2023-01-01T23:59:59")
AUDIT_ROOT = REPO_ROOT / "reports" / "audit_2026_04_29" / "run_01"
SAMPLE_ROOT = AUDIT_ROOT / "sample_downloads"
RAW_OUTPUT_ROOT = AUDIT_ROOT / "raw_tool_outputs"
PREVIEW_ROOT = AUDIT_ROOT / "previews"
LOG_ROOT = AUDIT_ROOT / "logs"
TABLE_ROOT = AUDIT_ROOT / "tables"
REQUEST_SPEC_ROOT = AUDIT_ROOT / "request_specs"
PACKAGE_NAMES = ["boto3", "xarray", "cfgrib", "eccodes", "cdsapi", "ecmwf-api-client", "pandas", "numpy", "matplotlib"]
SOURCE_NAMES = [
    "mrms_qpe_1h_pass1",
    "rtma_conus_aws_2p5km",
    "gfs_conus_aws_0p25",
    "ifs_mars_conus",
    "era5_land_t_conus",
    "gdas_conus_aws_0p25",
    "imerg_late_daily_conus",
]
SOURCE_PREFIXES = {
    "mrms_qpe_1h_pass1": ["MRMS", "src.datasources.mrms", "mrms_qpe_1h_pass1"],
    "rtma_conus_aws_2p5km": ["RTMA", "src.datasources.rtma", "rtma_conus_aws_2p5km"],
    "gfs_conus_aws_0p25": ["GFS", "src.datasources.gfs", "gfs_conus_aws_0p25"],
    "ifs_mars_conus": ["IFS", "src.datasources.ifs", "ifs_mars_conus"],
    "era5_land_t_conus": ["ERA5", "src.datasources.era5_landt", "era5_land_t_conus"],
    "gdas_conus_aws_0p25": ["GDAS", "src.datasources.gdas", "gdas_conus_aws_0p25"],
    "imerg_late_daily_conus": ["IMERG", "src.datasources.imerg", "imerg_late_daily_conus"],
}


def _package_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except Exception:
            versions[package_name] = None
    return versions


def _audit_config() -> Config:
    return Config(
        sample_start=SAMPLE_START,
        sample_end=SAMPLE_END,
        full_start=MAIN_PRED_START,
        full_end=MAIN_PRED_END,
        region=CONUS_BBOX,
        out_dir=SAMPLE_ROOT,
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


def _source_for_line(line: str) -> Optional[str]:
    for source_name, prefixes in SOURCE_PREFIXES.items():
        if any(prefix in line for prefix in prefixes):
            return source_name
    return None


def _split_transcript(transcript_path: Path) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]], list[str]]:
    source_logs: dict[str, list[str]] = {source: [] for source in SOURCE_NAMES}
    request_specs: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCE_NAMES}
    general_lines: list[str] = []

    text = transcript_path.read_text(encoding="utf-16")
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("REQUEST "):
            try:
                payload = json.loads(line.removeprefix("REQUEST "))
                source_name = str(payload.get("source") or "")
                if source_name in request_specs:
                    request_specs[source_name].append(payload)
                    source_logs[source_name].append(line)
                else:
                    general_lines.append(line)
            except Exception:
                general_lines.append(line)
            continue

        source_name = _source_for_line(line)
        if source_name is not None:
            source_logs[source_name].append(line)
        else:
            general_lines.append(line)

    return source_logs, request_specs, general_lines


def _write_request_specs(request_specs: dict[str, list[dict[str, Any]]]) -> None:
    REQUEST_SPEC_ROOT.mkdir(parents=True, exist_ok=True)
    for source_name, payloads in request_specs.items():
        (REQUEST_SPEC_ROOT / f"{source_name}.json").write_text(json.dumps(payloads, indent=2, default=str), encoding="utf-8")


def _write_source_logs(source_logs: dict[str, list[str]], general_lines: list[str], transcript_path: Path) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for source_name, lines in source_logs.items():
        (LOG_ROOT / f"{source_name}.log").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (LOG_ROOT / "run_stdout_stderr.utf8.txt").write_text(transcript_path.read_text(encoding="utf-16"), encoding="utf-8")
    (LOG_ROOT / "run_stdout_stderr.general.txt").write_text("\n".join(general_lines) + ("\n" if general_lines else ""), encoding="utf-8")


def _move_preview_pngs() -> None:
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


def _write_audit_summary(results, ratio_info, config: Config, request_specs: dict[str, list[dict[str, Any]]]) -> Path:
    rows = build_rows(results)
    checks = build_validation_checks(results, rows)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": {
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip() or "unknown",
            "python_executable": sys.executable,
            "python_version": sys.version,
            "package_versions": _package_versions(),
        },
        "config": config_to_dict(config),
        "ratio_info": ratio_info,
        "request_specs": request_specs,
        "summary": rows,
        "validation_checks": checks,
        "sources": [asdict(result) for result in results],
        "notes": {
            "run_directory": str(AUDIT_ROOT),
            "sample_root": str(SAMPLE_ROOT),
            "preview_root": str(PREVIEW_ROOT),
            "raw_output_root": str(RAW_OUTPUT_ROOT),
        },
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = AUDIT_ROOT / "audit_run_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return summary_path


def _update_audit_summary_requests(summary_path: Path, request_specs: dict[str, list[dict[str, Any]]]) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["request_specs"] = request_specs
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _copy_final_outputs() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    summary_csv = TABLE_ROOT / "decision_support_summary.csv"
    summary_md = TABLE_ROOT / "decision_support_summary.md"
    summary_html = TABLE_ROOT / "decision_support_summary.html"

    copies = {
        AUDIT_ROOT / "audit_run_summary.csv": summary_csv,
        AUDIT_ROOT / "audit_run_summary.md": summary_md,
    }
    for destination, source in copies.items():
        if source.exists():
            shutil.copy2(source, destination)

    if summary_html.exists():
        html_text = summary_html.read_text(encoding="utf-8")
        html_text = html_text.replace("../plots/", "plots/")
        (AUDIT_ROOT / "audit_run_summary.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    global AUDIT_ROOT, SAMPLE_ROOT, RAW_OUTPUT_ROOT, PREVIEW_ROOT, LOG_ROOT, TABLE_ROOT, REQUEST_SPEC_ROOT
    parser = argparse.ArgumentParser(description="Run the 24-hour acquisition audit and generate traceable outputs.")
    parser.add_argument("--audit-root", default=str(AUDIT_ROOT))
    args = parser.parse_args()

    audit_root = Path(args.audit_root)
    AUDIT_ROOT = audit_root
    SAMPLE_ROOT = AUDIT_ROOT / "sample_downloads"
    RAW_OUTPUT_ROOT = AUDIT_ROOT / "raw_tool_outputs"
    PREVIEW_ROOT = AUDIT_ROOT / "previews"
    LOG_ROOT = AUDIT_ROOT / "logs"
    TABLE_ROOT = AUDIT_ROOT / "tables"
    REQUEST_SPEC_ROOT = AUDIT_ROOT / "request_specs"

    for path in [SAMPLE_ROOT, RAW_OUTPUT_ROOT, PREVIEW_ROOT, LOG_ROOT, TABLE_ROOT, REQUEST_SPEC_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    transcript_path = LOG_ROOT / "run_stdout_stderr.txt"
    config = _audit_config()

    with transcript_path.open("w", encoding="utf-16") as transcript, redirect_stdout(transcript), redirect_stderr(transcript):
        logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s", stream=transcript, force=True)
        print("Running acquisition audit (no silent skips, no secrets, no extra downloads beyond the audit run)...")
        print(f"Audit root: {AUDIT_ROOT}")
        print(f"Sample period: {SAMPLE_START.isoformat()} to {SAMPLE_END.isoformat()}")
        print(f"Prediction period: {MAIN_PRED_START.isoformat()} to {MAIN_PRED_END.isoformat()}")
        print(f"Python executable: {sys.executable}")
        print(f"Package versions: {json.dumps(_package_versions(), sort_keys=True)}")

        results, ratio_info = run_estimation(config)
        rows = build_rows(results)
        checks = build_validation_checks(results, rows)

        # Persist the canonical audit payload before post-processing.
        request_specs_placeholder = {source: [] for source in SOURCE_NAMES}
        summary_path = _write_audit_summary(results, ratio_info, config, request_specs_placeholder)
        print(f"AUDIT_SUMMARY_JSON={summary_path}")
        print(f"RESULT_COUNT={len(results)}")
        for result in results:
            print(
                f"SOURCE_RESULT source={result.source} status={result.status} sample_files={result.sample_files} "
                f"full_file_bytes={result.full_file_bytes} selected_variable_bytes={result.selected_variable_bytes} "
                f"selected_conus_bytes={result.selected_conus_bytes} raw_hot_bytes={result.raw_hot_bytes} "
                f"derived_hot_bytes={result.derived_hot_bytes} peak_local_bytes={result.peak_local_bytes}"
            )

        transcript.flush()

    source_logs, request_specs, general_lines = _split_transcript(transcript_path)
    _write_source_logs(source_logs, general_lines, transcript_path)
    _write_request_specs(request_specs)
    _update_audit_summary_requests(AUDIT_ROOT / "audit_run_summary.json", request_specs)
    _move_preview_pngs()

    # Regenerate the report layer for this specific run.
    report_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_decision_support_report.py"),
        "--summary",
        str(AUDIT_ROOT / "audit_run_summary.json"),
        "--run-log",
        str(transcript_path),
        "--audit-root",
        str(AUDIT_ROOT),
    ]
    subprocess.run(report_cmd, cwd=REPO_ROOT, check=True)
    _copy_final_outputs()

    print(f"Completed audit run under {AUDIT_ROOT}")


if __name__ == "__main__":
    main()
