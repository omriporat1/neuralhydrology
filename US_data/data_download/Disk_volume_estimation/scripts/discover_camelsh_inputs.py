#!/usr/bin/env python3
"""Discover and validate CAMELSH source files for the Flash-NH Stage 1 pilot.

Checks the three CAMELSH inputs required for Stage 1:
  - basin_polygons    : GeoPackage or shapefile with basin polygon geometries
  - static_attributes : directory or file with CAMELSH static basin attributes
  - hourly_streamflow : directory or file(s) with hourly USGS streamflow records

Validates that pilot basin STAIDs from pilot_basin_manifest.csv can be found
in the available static or streamflow files (if the files are accessible).

Does NOT modify any source files. Safe to re-run.

Usage:
    python scripts/discover_camelsh_inputs.py --config configs/pilot_stage1.yaml
    python scripts/discover_camelsh_inputs.py --pilot-manifest /path/to/manifest.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_config, config_to_dict, PipelineConfig
from src.pipeline.provenance import write_run_manifest


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
    p.add_argument("--data-root", default=None, help="Override data root path")
    p.add_argument(
        "--pilot-manifest",
        default=None,
        help="Path to pilot_basin_manifest.csv (auto-discovered if omitted)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery results without writing report files",
    )
    return p.parse_args()


def _check_path(path_str: Optional[str], description: str) -> dict[str, Any]:
    """Return a status dict for one configured CAMELSH path."""
    if not path_str:
        return {
            "status": "NOT_CONFIGURED",
            "description": description,
            "path": None,
            "note": "Set the path in configs/pilot_stage1.yaml under camelsh.",
        }
    p = Path(path_str)
    if p.exists():
        size_bytes = p.stat().st_size if p.is_file() else None
        return {
            "status": "FOUND",
            "description": description,
            "path": str(p),
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
            "size_bytes": size_bytes,
        }
    return {
        "status": "MISSING",
        "description": description,
        "path": str(p),
        "note": "Path configured but file/directory does not exist.",
    }


def _load_pilot_staids(manifest_path: Path) -> list[str]:
    staids = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            staid = (row.get("STAID") or row.get("staid") or row.get("gauge_id") or "").strip()
            if staid:
                staids.append(staid)
    return staids


def _probe_staid_match(staids: list[str], path_str: Optional[str]) -> dict[str, Any]:
    """Try to match STAIDs in a CSV static-attributes file (best-effort)."""
    if not path_str or not Path(path_str).exists():
        return {"status": "SKIPPED", "reason": "file not available"}

    p = Path(path_str)
    # Only attempt CSV/text probing; skip binary formats
    suffixes = {".csv", ".txt", ".tsv"}
    if p.is_file() and p.suffix.lower() not in suffixes:
        return {"status": "SKIPPED", "reason": f"non-text format {p.suffix}; skipping automated probe"}

    if p.is_dir():
        # Look for any CSV file that might contain gauge IDs
        candidates = list(p.glob("*.csv")) + list(p.glob("*.txt"))
        if not candidates:
            return {"status": "SKIPPED", "reason": "directory has no CSV/TXT files for probing"}
        p = candidates[0]

    try:
        found: set[str] = set()
        with open(p, newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            id_col = next(
                (c for c in (reader.fieldnames or [])
                 if c.lower() in ("staid", "gauge_id", "site_no", "hru_id", "basin_id")),
                None,
            )
            if id_col is None:
                return {"status": "SKIPPED", "reason": f"no recognized ID column in {p.name}"}
            for row in reader:
                found.add(row[id_col].strip().lstrip("0"))

        staid_set = {s.lstrip("0") for s in staids}
        matched = staid_set & found
        missing = staid_set - found
        return {
            "status": "CHECKED",
            "probe_file": str(p),
            "n_pilot_staids": len(staids),
            "n_matched": len(matched),
            "n_missing": len(missing),
            "missing_sample": sorted(missing)[:10],
        }
    except Exception as exc:
        return {"status": "ERROR", "reason": str(exc)}


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
    dry_run = args.dry_run

    # Auto-discover pilot manifest
    pilot_manifest_path: Optional[Path] = None
    if args.pilot_manifest:
        pilot_manifest_path = Path(args.pilot_manifest)
    else:
        candidate = data_root / "09_manifests" / "stage1_pilot" / "pilot_basin_manifest.csv"
        if candidate.exists():
            pilot_manifest_path = candidate

    pilot_staids: list[str] = []
    if pilot_manifest_path and pilot_manifest_path.exists():
        pilot_staids = _load_pilot_staids(pilot_manifest_path)
        print(f"Pilot manifest: {pilot_manifest_path} ({len(pilot_staids)} basins)")
    else:
        pmp_str = str(pilot_manifest_path) if pilot_manifest_path else "(none found)"
        print(f"Pilot manifest not found ({pmp_str}); STAID matching will be skipped.")
        print(f"  Run select_pilot_basins.py first to generate the manifest.")

    # Check CAMELSH paths
    checks: dict[str, Any] = {
        "basin_polygons":    _check_path(cfg.camelsh.basin_polygons,    "CAMELSH basin polygons (GeoPackage or shapefile)"),
        "static_attributes": _check_path(cfg.camelsh.static_attributes, "CAMELSH static basin attributes"),
        "hourly_streamflow": _check_path(cfg.camelsh.hourly_streamflow,  "CAMELSH hourly USGS streamflow"),
    }

    print("\nCAMELSH input discovery:")
    for key, result in checks.items():
        status = result["status"]
        path_str = result.get("path") or "(not configured)"
        print(f"  {key:25s}: {status:15s}  {path_str}")

    # STAID match probe (best-effort; static_attributes most likely to have CSV)
    staid_match: dict[str, Any] = {"status": "SKIPPED", "reason": "no pilot STAIDs loaded"}
    if pilot_staids:
        staid_match = _probe_staid_match(pilot_staids, cfg.camelsh.static_attributes)
        print(f"\nSTAID match probe (static_attributes): {staid_match['status']}")
        if staid_match["status"] == "CHECKED":
            print(f"  n_pilot_staids: {staid_match['n_pilot_staids']}")
            print(f"  n_matched:      {staid_match['n_matched']}")
            print(f"  n_missing:      {staid_match['n_missing']}")
            if staid_match["missing_sample"]:
                print(f"  missing sample: {staid_match['missing_sample']}")
        elif staid_match["status"] in ("SKIPPED", "ERROR"):
            print(f"  reason: {staid_match.get('reason', '')}")

    # Build validation dict.
    # NOT_CONFIGURED is expected at this stage — only MISSING (configured but absent) is a failure.
    any_found = any(r["status"] == "FOUND" for r in checks.values())
    validation: dict[str, Any] = {
        # Fail only when a path is explicitly configured but the file cannot be found.
        "basin_polygons_no_broken_path":    checks["basin_polygons"]["status"] != "MISSING",
        "static_attributes_no_broken_path": checks["static_attributes"]["status"] != "MISSING",
        "hourly_streamflow_no_broken_path": checks["hourly_streamflow"]["status"] != "MISSING",
        "pilot_manifest_found":             pilot_manifest_path is not None and pilot_manifest_path.exists(),
        # Non-bool informational fields — excluded from _all_pass() by isinstance(v, bool) check
        "n_pilot_staids":     len(pilot_staids),
        "staid_probe_status": staid_match.get("status"),
        # any_camelsh_found is informational; kept in report dict, not in validation
    }

    report: dict[str, Any] = {
        "checks":         checks,
        "pilot_manifest": str(pilot_manifest_path) if pilot_manifest_path else None,
        "n_pilot_staids": len(pilot_staids),
        "staid_match":    staid_match,
        "validation":     validation,
        "next_step": (
            "Configure camelsh.* paths in configs/pilot_stage1.yaml and re-run."
            if not any_found
            else "CAMELSH files located; proceed to weight computation."
        ),
    }

    if not any_found:
        print(
            "\nNOTE: No CAMELSH paths configured or found.\n"
            "  Set camelsh.basin_polygons, camelsh.static_attributes, and\n"
            "  camelsh.hourly_streamflow in configs/pilot_stage1.yaml.\n"
            "  This is expected for the initial skeleton run."
        )

    report_dir = data_root / "09_manifests" / "stage1_pilot"
    report_json = report_dir / "camelsh_discovery_report.json"

    if dry_run:
        print(f"\n[DRY-RUN] Would write discovery report to: {report_json}")
        return

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nDiscovery report: {report_json}")

    run_cmd = f"python scripts/discover_camelsh_inputs.py --config {args.config}"
    if args.data_root:
        run_cmd += f" --data-root {args.data_root}"

    write_run_manifest(
        report_dir / "camelsh_discovery_provenance",
        run_command=run_cmd,
        config_dict=config_to_dict(cfg),
        output_paths={"camelsh_discovery_report": str(report_json)},
        validation_results=validation,
    )


if __name__ == "__main__":
    main()
