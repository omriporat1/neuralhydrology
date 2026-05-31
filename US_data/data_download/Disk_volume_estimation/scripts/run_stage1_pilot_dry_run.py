#!/usr/bin/env python3
"""Flash-NH Stage 1 pilot — dry-run orchestrator.

Runs all skeleton bootstrap steps end-to-end WITHOUT downloading any
gridded meteorological data. Safe to run repeatedly; all steps are
idempotent. Outputs go to a configurable data root (default: tmp/stage1_pilot_dryrun/
under the repo, which is git-ignored).

Steps:
  1. Bootstrap data root directory structure (bootstrap_data_root.py)
  2. Select pilot basins and write pilot_basin_manifest.csv (select_pilot_basins.py)
  3. Discover and validate CAMELSH inputs (discover_camelsh_inputs.py)
  4. Write overall dry-run summary

Usage:
    python scripts/run_stage1_pilot_dry_run.py
    python scripts/run_stage1_pilot_dry_run.py --config configs/pilot_stage1.yaml
    python scripts/run_stage1_pilot_dry_run.py --data-root /my/external/Flash-NH_data

For a self-contained local test (no external storage needed), omit --data-root.
The default data root for this script is tmp/stage1_pilot_dryrun/ — git-ignored,
safe to delete after inspection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_config, config_to_dict, PipelineConfig
from src.pipeline.provenance import write_run_manifest, git_commit_hash


_DEFAULT_DRYRUN_DATA_ROOT = str(REPO_ROOT / "tmp" / "stage1_pilot_dryrun")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "pilot_stage1.yaml"),
        help="Path to YAML config (default: configs/pilot_stage1.yaml)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help=(
            f"Data root for outputs (default: {_DEFAULT_DRYRUN_DATA_ROOT}). "
            "Use your real external data path to populate it directly."
        ),
    )
    return p.parse_args()


def _run_step(name: str, cmd: list[str]) -> bool:
    """Run a subprocess step; return True on success."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"STEP: {name}")
    print(f"{sep}")
    print(f"CMD: {' '.join(cmd)}")
    print()
    result = subprocess.run(cmd)
    ok = result.returncode == 0
    print(f"\n-> {'PASS' if ok else 'FAIL'} (exit code {result.returncode})")
    return ok


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        cfg = load_config(config_path)
    else:
        cfg = PipelineConfig()
        print(f"Config not found at {config_path}; using defaults.")

    data_root_str = args.data_root or _DEFAULT_DRYRUN_DATA_ROOT
    data_root = Path(data_root_str)

    print("Flash-NH Stage 1 Pilot - Dry Run")
    print(f"  Config:    {config_path}")
    print(f"  Data root: {data_root}")
    print(f"  Git:       {git_commit_hash() or 'unknown'}")
    print(f"  Python:    {sys.executable}")

    py = sys.executable
    config_str = str(config_path)

    steps = [
        (
            "Bootstrap data root",
            [py, str(REPO_ROOT / "scripts" / "bootstrap_data_root.py"),
             "--config", config_str, "--data-root", data_root_str],
        ),
        (
            "Select pilot basins",
            [py, str(REPO_ROOT / "scripts" / "select_pilot_basins.py"),
             "--config", config_str, "--data-root", data_root_str],
        ),
        (
            "Discover CAMELSH inputs",
            [py, str(REPO_ROOT / "scripts" / "discover_camelsh_inputs.py"),
             "--config", config_str, "--data-root", data_root_str],
        ),
    ]

    step_results: dict[str, bool] = {}
    for step_name, cmd in steps:
        step_results[step_name] = _run_step(step_name, cmd)

    # Check key output files
    pilot_manifest = data_root / "09_manifests" / "stage1_pilot" / "pilot_basin_manifest.csv"
    camelsh_report = data_root / "09_manifests" / "stage1_pilot" / "camelsh_discovery_report.json"
    bootstrap_manifest = data_root / "09_manifests" / "bootstrap" / "manifest.json"

    output_checks = {
        "pilot_basin_manifest_written": pilot_manifest.exists(),
        "camelsh_discovery_report_written": camelsh_report.exists(),
        "bootstrap_manifest_written": bootstrap_manifest.exists(),
        "data_root_structure_complete": all(
            (data_root / d).exists()
            for d in ["00_raw", "02_basin_geometries", "03_basin_timeseries",
                      "09_manifests", "tmp"]
        ),
    }

    all_results = {**step_results, **output_checks}
    overall_pass = all(all_results.values())

    print("\n" + "=" * 64)
    print("DRY-RUN SUMMARY")
    print("=" * 64)
    for check, passed in all_results.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  {tag}  {check}")

    # Write overall summary
    summary_dir = data_root / "09_manifests" / "bootstrap"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "data_root": str(data_root),
        "config": str(config_path),
        "step_results": step_results,
        "output_checks": output_checks,
        "overall": "PASS" if overall_pass else "FAIL",
    }
    summary_json = summary_dir / "dry_run_summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\nSummary: {summary_json}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
    print()
    print("Key outputs:")
    print(f"  Pilot manifest:     {pilot_manifest}")
    print(f"  CAMELSH report:     {camelsh_report}")
    print(f"  Bootstrap manifest: {bootstrap_manifest}")
    print()
    print("Next step: implement MRMS/RTMA grid definition and basin-grid weight computation.")
    print("  See docs/pipeline_skeleton.md for the milestone roadmap.")

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
