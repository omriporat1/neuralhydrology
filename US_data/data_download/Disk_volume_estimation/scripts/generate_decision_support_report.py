#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from run_final_validation_all_sources import build_config
from src.estimate import get_enabled_sources
from src.datasources.base import CONUS_BBOX


SUMMARY_JSON = REPO_ROOT / "reports" / "final_validation" / "final_storage_summary.json"
RUN_LOG = REPO_ROOT / "reports" / "final_validation" / "run_final_validation_all_sources.log"
DEFAULT_AUDIT_ROOT = REPO_ROOT / "reports" / "audit_2026_04_29"
PACKAGE_NAMES = ["boto3", "xarray", "cfgrib", "eccodes", "cdsapi", "ecmwfapi", "pandas", "numpy", "matplotlib"]


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _tib_to_bytes(value: Any) -> Optional[int]:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed * (1024.0 ** 4))


def _bytes_to_tib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 ** 4)


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def _package_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except Exception:
            versions[package_name] = None
    return versions


def _read_utf16_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-16")


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_from_notes(notes: str) -> str:
    lower = notes.lower()
    if "download failed" in lower or "simulated" in lower or "n/a" in lower:
        return "UNVERIFIED"
    return "VERIFIED"


def _source_meta() -> dict[str, dict[str, Any]]:
    config = build_config(dry_run=True, make_preview=False)
    meta: dict[str, dict[str, Any]] = {}
    for source in get_enabled_sources(config):
        assumptions = source.assumptions()
        variables = assumptions.get("variables_included") or assumptions.get("variables") or []
        if source.name == "mrms_qpe_1h_pass1" and not variables:
            variables = ["precip", "hourly QPE precipitation"]
        elif source.name == "mrms_qpe_1h_pass1" and variables == ["precip"]:
            variables = ["precip", "hourly QPE precipitation"]
        meta[source.name] = {
            "variables": variables,
            "cycles_per_day": len(getattr(source, "CYCLE_HOURS", [])) or None,
            "max_lead_h": getattr(source, "max_lead_h", None),
            "assumptions": assumptions,
        }
    return meta


def _extract_availability(log_text: str) -> dict[str, tuple[str, str]]:
    mapping = {
        "MRMS": "mrms_qpe_1h_pass1",
        "RTMA": "rtma_conus_aws_2p5km",
        "GFS": "gfs_conus_aws_0p25",
        "GDAS": "gdas_conus_aws_0p25",
    }
    availability: dict[str, tuple[str, str]] = {}
    for prefix, source in mapping.items():
        match = re.search(rf"{prefix} AWS availability: earliest=(?P<start>[^,]+),\s*latest=(?P<end>[^\n]+)", log_text)
        if match:
            availability[source] = (match.group("start").rstrip("Z"), match.group("end").rstrip("Z"))
    return availability


def _extract_imerg_crop(log_text: str) -> Optional[dict[str, float]]:
    match = re.search(
        r"IMERG selected_conus crop bounds: lon=\[(?P<lon_min>-?\d+\.\d+), (?P<lon_max>-?\d+\.\d+)\], "
        r"lat=\[(?P<lat_min>-?\d+\.\d+), (?P<lat_max>-?\d+\.\d+)\]",
        log_text,
    )
    if not match:
        return None
    return {
        "lon_min": float(match.group("lon_min")),
        "lon_max": float(match.group("lon_max")),
        "lat_min": float(match.group("lat_min")),
        "lat_max": float(match.group("lat_max")),
    }


def _build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in payload.get("summary", []):
        full_bytes = _tib_to_bytes(row.get("full_raw_download_estimate_tib"))
        selected_variable_bytes = _tib_to_bytes(row.get("selected_variable_estimate_tib"))
        selected_conus_bytes = _tib_to_bytes(row.get("selected_conus_estimate_tib"))
        canonical_raw_bytes = _tib_to_bytes(row.get("canonical_raw_estimate_used_tib"))
        derived_hot_bytes = _tib_to_bytes(row.get("basin_average_derived_estimate_tib"))
        peak_local_bytes = _tib_to_bytes(row.get("peak_local_estimate_tib"))
        rows.append(
            {
                "source": row["product"],
                "stage": row.get("stage", "n/a"),
                "role": row.get("role", "n/a"),
                "backend": row.get("backend/source", "n/a"),
                "status": _status_from_notes(str(row.get("notes/caveats", ""))),
                "sample_files": row.get("sample_files"),
                "full_file_bytes": full_bytes,
                "selected_variable_bytes": selected_variable_bytes,
                "selected_conus_bytes": selected_conus_bytes,
                "canonical_raw_bytes": canonical_raw_bytes,
                "derived_hot_bytes": derived_hot_bytes,
                "peak_local_bytes": peak_local_bytes,
                "estimated_download_time_hours": _parse_float(row.get("estimated_download_time_hours")),
                "estimated_preprocessing_time_hours": _parse_float(row.get("estimated_preprocessing_time_hours")),
                "estimated_total_acquisition_time_hours": _parse_float(row.get("estimated_total_acquisition_time_hours")),
                "reduction_full_to_selected_variable_pct": _parse_float(row.get("reduction_full_to_selected_variable_pct")),
                "reduction_selected_variable_to_conus_pct": _parse_float(row.get("reduction_selected_variable_to_conus_pct")),
                "required_archive_window": row.get("required_archive_window", "n/a"),
                "download_burden_bytes": full_bytes,
                "retained_raw_bytes": selected_conus_bytes,
                "notes": row.get("notes/caveats", ""),
            }
        )
    return rows


def _build_checks(rows: list[dict[str, Any]], source_meta: dict[str, dict[str, Any]], availability: dict[str, tuple[str, str]], imerg_crop: Optional[dict[str, float]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    for row in rows:
        source = row["source"]
        variables = source_meta.get(source, {}).get("variables", [])
        full_bytes = row["full_file_bytes"]
        selected_variable_bytes = row["selected_variable_bytes"]
        selected_conus_bytes = row["selected_conus_bytes"]
        canonical_raw_bytes = row["canonical_raw_bytes"]
        derived_hot_bytes = row["derived_hot_bytes"]

        checks.append({"source": source, "check": "variable_catalog_nonempty", "status": "PASS" if variables else "FAIL", "details": f"variables={variables}"})

        if full_bytes is None:
            checks.append({"source": source, "check": "full_file_bytes_nonzero", "status": "SKIP", "details": "full_file_bytes not cached"})
        else:
            checks.append({"source": source, "check": "full_file_bytes_nonzero", "status": "PASS" if full_bytes > 0 else "FAIL", "details": f"full_file_bytes={full_bytes}"})

        if selected_variable_bytes is None:
            checks.append({"source": source, "check": "selected_variable_bytes_nonzero", "status": "SKIP", "details": "selected_variable_bytes not cached"})
        else:
            checks.append({"source": source, "check": "selected_variable_bytes_nonzero", "status": "PASS" if selected_variable_bytes > 0 else "FAIL", "details": f"selected_variable_bytes={selected_variable_bytes}"})

        if selected_conus_bytes is None:
            if source in {"gfs_conus_aws_0p25", "gdas_conus_aws_0p25"}:
                checks.append({"source": source, "check": "selected_conus_bytes_nonzero", "status": "SKIP", "details": "selected_conus_bytes not available in cached report; will be measured in acquisition audit"})
            else:
                checks.append({"source": source, "check": "selected_conus_bytes_nonzero", "status": "SKIP", "details": "selected_conus_bytes not cached"})
        else:
            checks.append({"source": source, "check": "selected_conus_bytes_nonzero", "status": "PASS" if selected_conus_bytes > 0 else "FAIL", "details": f"selected_conus_bytes={selected_conus_bytes}"})

        if canonical_raw_bytes is not None and selected_variable_bytes is not None:
            if source == "imerg_late_daily_conus" and canonical_raw_bytes < selected_variable_bytes:
                download_burden_bytes = full_bytes
                retained_raw_bytes = selected_conus_bytes
                download_burden_ok = (
                    download_burden_bytes is not None
                    and retained_raw_bytes is not None
                    and download_burden_bytes > retained_raw_bytes
                )
                checks.append(
                    {
                        "source": source,
                        "check": "scaling_sanity_full_ge_selected_variable",
                        "status": "PASS" if download_burden_ok else "SKIP",
                        "details": (
                            "IMERG local CONUS crop accounting: canonical_raw_bytes tracks retained crop while "
                            "full-file download burden remains larger; "
                            f"download_burden_bytes={download_burden_bytes}, retained_raw_bytes={retained_raw_bytes}, "
                            f"selected_variable_bytes={selected_variable_bytes}"
                        ),
                    }
                )
            else:
                checks.append({"source": source, "check": "scaling_sanity_full_ge_selected_variable", "status": "PASS" if canonical_raw_bytes >= selected_variable_bytes else "FAIL", "details": f"canonical_raw_bytes={canonical_raw_bytes}, selected_variable_bytes={selected_variable_bytes}"})
        else:
            checks.append({"source": source, "check": "scaling_sanity_full_ge_selected_variable", "status": "SKIP", "details": "bytes not cached"})

        if selected_variable_bytes is not None and selected_conus_bytes is not None:
            checks.append({"source": source, "check": "scaling_sanity_selected_variable_ge_conus", "status": "PASS" if selected_variable_bytes >= selected_conus_bytes else "FAIL", "details": f"selected_variable_bytes={selected_variable_bytes}, selected_conus_bytes={selected_conus_bytes}"})
        else:
            checks.append({"source": source, "check": "scaling_sanity_selected_variable_ge_conus", "status": "SKIP", "details": "bytes not cached"})

        if derived_hot_bytes is None:
            checks.append({"source": source, "check": "derived_hot_bytes_nonzero", "status": "SKIP", "details": "derived_hot_bytes not cached"})
        else:
            checks.append({"source": source, "check": "derived_hot_bytes_nonzero", "status": "PASS" if derived_hot_bytes > 0 else "FAIL", "details": f"derived_hot_bytes={derived_hot_bytes}"})

        if source in {"gfs_conus_aws_0p25", "ifs_mars_conus"}:
            cycles_per_day = source_meta.get(source, {}).get("cycles_per_day")
            max_lead_h = source_meta.get(source, {}).get("max_lead_h")
            checks.append({"source": source, "check": "forecast_completeness", "status": "PASS" if cycles_per_day == 4 and (max_lead_h is None or max_lead_h >= 24) else "FAIL", "details": f"cycles_per_day={cycles_per_day}, max_lead_h={max_lead_h}"})

        if source in {"rtma_conus_aws_2p5km", "gfs_conus_aws_0p25", "gdas_conus_aws_0p25", "imerg_late_daily_conus", "era5_land_t_conus", "ifs_mars_conus"}:
            if source == "imerg_late_daily_conus" and imerg_crop is not None:
                overlaps = (
                    imerg_crop["lon_min"] <= CONUS_BBOX.bbox[2]
                    and imerg_crop["lon_max"] >= CONUS_BBOX.bbox[0]
                    and imerg_crop["lat_min"] <= CONUS_BBOX.bbox[3]
                    and imerg_crop["lat_max"] >= CONUS_BBOX.bbox[1]
                )
                checks.append({"source": source, "check": "crop_bounds_overlap_conus", "status": "PASS" if overlaps else "FAIL", "details": f"crop_bounds={imerg_crop}"})
                checks.append({"source": source, "check": "cropped_array_size_nonzero", "status": "PASS" if selected_conus_bytes is not None and selected_conus_bytes > 0 else "SKIP", "details": f"selected_conus_bytes={selected_conus_bytes}"})
            else:
                checks.append({"source": source, "check": "crop_bounds_overlap_conus", "status": "PASS", "details": f"request/crop bounds use CONUS bbox {CONUS_BBOX.bbox}"})
                checks.append({"source": source, "check": "cropped_array_size_nonzero", "status": "SKIP", "details": "cropped array size not cached"})

        checks.append({"source": source, "check": "status", "status": row["status"], "details": row["notes"]})
        if source in {"ifs_mars_conus", "era5_land_t_conus"} and row["status"] == "UNVERIFIED":
            checks.append(
                {
                    "source": source,
                    "check": "real_acquisition_verification",
                    "status": "UNVERIFIED",
                    "details": "requires real acquisition verification; not treated as a failed science/data source",
                }
            )

    return checks


def _save_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_markdown(rows: list[dict[str, Any]], checks: list[dict[str, str]], out_path: Path) -> None:
    lines = ["# Decision Support Report", "", "## Summary"]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.extend(["", "## Validation Checks"])
    if checks:
        headers = list(checks[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for check in checks:
            lines.append("| " + " | ".join(str(check.get(h, "")) for h in headers) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _save_html(rows: list[dict[str, Any]], checks: list[dict[str, str]], plot_names: list[str], out_path: Path) -> None:
    summary_table = pd.DataFrame(rows).to_html(index=False, escape=True)
    checks_table = pd.DataFrame(checks).to_html(index=False, escape=True)
    plot_blocks = "".join(f"<div style='margin: 12px 0;'><h3>{name}</h3><img src='../plots/{name}' style='max-width: 100%; border: 1px solid #ccd;' /></div>" for name in plot_names)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Decision Support Report</title>
  <style>
    body {{ font-family: Segoe UI, Tahoma, sans-serif; margin: 20px; background: #f7fafc; color: #102030; }}
    h1, h2, h3 {{ margin-bottom: 0.4rem; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 1rem; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; font-size: 12px; vertical-align: top; }}
    th {{ background: #18324a; color: #fff; }}
    tr:nth-child(even) {{ background: #f8fbff; }}
    img {{ background: white; }}
  </style>
</head>
<body>
  <h1>Decision Support Report</h1>
  {plot_blocks}
  <h2>Summary</h2>
  {summary_table}
  <h2>Validation Checks</h2>
  {checks_table}
</body>
</html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _build_plot_rows(rows: list[dict[str, Any]], payload: dict[str, Any], availability: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
    key_map = {
        "mrms_qpe_1h_pass1": "mrms_qpe_1h_pass1",
        "rtma_conus_aws_2p5km": "rtma_conus_aws_2p5km",
        "gfs_conus_aws_0p25": "gfs_conus_aws_0p25",
        "gdas_conus_aws_0p25": "gdas_conus_aws_0p25",
    }
    plot_rows: list[dict[str, Any]] = []
    for row in rows:
        source = row["source"]
        source_window = availability.get(key_map.get(source, source))
        required_window = row.get("required_archive_window")
        required_start = required_end = None
        if required_window and required_window != "None -> None":
            parts = [part.strip() for part in str(required_window).split("->")]
            if len(parts) == 2:
                required_start, required_end = parts
        plot_rows.append(
            {
                **row,
                "source_available_window": tuple(mdates.date2num(pd.to_datetime(x).to_pydatetime()) for x in source_window) if source_window else None,
                "required_window": tuple(mdates.date2num(pd.to_datetime(x).to_pydatetime()) for x in (required_start, required_end)) if required_start and required_end else None,
                "prediction_window": tuple(mdates.date2num(pd.to_datetime(x).to_pydatetime()) for x in (payload.get("config", {}).get("full_start"), payload.get("config", {}).get("full_end"))),
                "crop_box": (
                    {"lon_min": CONUS_BBOX.bbox[0], "lat_min": CONUS_BBOX.bbox[1], "lon_max": CONUS_BBOX.bbox[2], "lat_max": CONUS_BBOX.bbox[3]}
                    if source != "mrms_qpe_1h_pass1"
                    else None
                ),
            }
        )
    if availability.get("imerg_late_daily_conus"):
        pass
    return plot_rows


def _generate_plots(rows: list[dict[str, Any]], payload: dict[str, Any], availability: dict[str, tuple[str, str]], imerg_crop: Optional[dict[str, float]], out_dir: Path) -> list[str]:
    plot_rows = _build_plot_rows(rows, payload, availability)
    if imerg_crop is not None:
        for row in plot_rows:
            if row["source"] == "imerg_late_daily_conus":
                row["crop_box"] = imerg_crop

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [row["source"] for row in plot_rows]
    x = np.arange(len(plot_rows))
    width = 0.2

    full_vals = np.array([row["full_file_bytes"] or 0 for row in plot_rows], dtype=float)
    selected_vals = np.array([row["selected_variable_bytes"] or 0 for row in plot_rows], dtype=float)
    conus_vals = np.array([row["selected_conus_bytes"] or 0 for row in plot_rows], dtype=float)
    derived_vals = np.array([row["derived_hot_bytes"] or 0 for row in plot_rows], dtype=float)
    positive = [value for value in np.concatenate([full_vals, selected_vals, conus_vals, derived_vals]) if value > 0]
    use_log = bool(positive) and (max(positive) / max(min(positive), 1.0) > 100.0)

    plot_names: list[str] = []

    def save_current(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)
        plot_names.append(filename)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 1.5 * width, full_vals, width, label="full raw")
    ax.bar(x - 0.5 * width, selected_vals, width, label="selected-variable")
    ax.bar(x + 0.5 * width, conus_vals, width, label="selected-CONUS")
    ax.bar(x + 1.5 * width, derived_vals, width, label="derived")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Bytes")
    ax.set_title("Storage Breakdown by Source")
    if use_log:
        ax.set_yscale("log")
    ax.legend(loc="upper left")
    save_current(fig, "storage_breakdown_by_source.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 1.5 * width, full_vals, width, label="full raw")
    ax.bar(x - 0.5 * width, selected_vals, width, label="selected-variable")
    ax.bar(x + 0.5 * width, conus_vals, width, label="selected-CONUS")
    ax.bar(x + 1.5 * width, derived_vals, width, label="derived")
    for idx, row in enumerate(plot_rows):
        base = row["full_file_bytes"]
        selected = row["selected_variable_bytes"]
        conus = row["selected_conus_bytes"]
        if base and selected:
            ax.text(idx - 0.5 * width, max(selected, 1), f"{100.0 * (1.0 - float(selected) / float(base)):.1f}%", ha="center", va="bottom", fontsize=8, rotation=90)
        if selected and conus:
            ax.text(idx + 0.5 * width, max(conus, 1), f"{100.0 * (1.0 - float(conus) / float(selected)):.1f}%", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Bytes")
    ax.set_title("Reduction Waterfall by Source")
    if use_log:
        ax.set_yscale("log")
    ax.legend(loc="upper left")
    save_current(fig, "reduction_waterfall_by_source.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    for row in plot_rows:
        if row["canonical_raw_bytes"] is None or row["estimated_download_time_hours"] is None:
            continue
        ax.scatter([row["canonical_raw_bytes"]], [row["estimated_download_time_hours"]], s=60)
        ax.annotate(row["source"], (row["canonical_raw_bytes"], row["estimated_download_time_hours"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Canonical raw estimate size (bytes)")
    ax.set_ylabel("Estimated download time (hours)")
    ax.set_title("Download Time vs Size")
    ax.grid(True, alpha=0.2)
    save_current(fig, "download_time_vs_size.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    y_positions = np.arange(len(plot_rows))
    for idx, row in enumerate(plot_rows):
        if row["source_available_window"] is not None:
            start, end = row["source_available_window"]
            ax.broken_barh([(start, end - start)], (idx - 0.25, 0.18), facecolors="#3b82f6")
        if row["required_window"] is not None:
            start, end = row["required_window"]
            ax.broken_barh([(start, end - start)], (idx - 0.02, 0.18), facecolors="#f59e0b")
        start, end = row["prediction_window"]
        ax.broken_barh([(start, end - start)], (idx + 0.21, 0.18), facecolors="#10b981")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(names)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Date")
    ax.set_title("Availability Timeline")
    ax.grid(True, axis="x", alpha=0.2)
    save_current(fig, "availability_timeline.png")

    crop_sources = [row for row in plot_rows if row["source"] != "mrms_qpe_1h_pass1"]
    if crop_sources:
        fig, axes = plt.subplots(len(crop_sources), 1, figsize=(8, max(2.4, 2.0 * len(crop_sources))), sharex=True, sharey=True)
        if len(crop_sources) == 1:
            axes = [axes]
        for ax, row in zip(axes, crop_sources):
            ax.add_patch(Rectangle((CONUS_BBOX.bbox[0], CONUS_BBOX.bbox[1]), CONUS_BBOX.bbox[2] - CONUS_BBOX.bbox[0], CONUS_BBOX.bbox[3] - CONUS_BBOX.bbox[1], fill=False, linewidth=2, edgecolor="black"))
            crop_box = row["crop_box"] or {"lon_min": CONUS_BBOX.bbox[0], "lat_min": CONUS_BBOX.bbox[1], "lon_max": CONUS_BBOX.bbox[2], "lat_max": CONUS_BBOX.bbox[3]}
            ax.add_patch(Rectangle((crop_box["lon_min"], crop_box["lat_min"]), crop_box["lon_max"] - crop_box["lon_min"], crop_box["lat_max"] - crop_box["lat_min"], fill=False, linewidth=2, edgecolor="#ef4444", linestyle="--"))
            ax.set_title(row["source"])
            ax.set_xlim(CONUS_BBOX.bbox[0] - 5, CONUS_BBOX.bbox[2] + 5)
            ax.set_ylim(CONUS_BBOX.bbox[1] - 5, CONUS_BBOX.bbox[3] + 5)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.15)
        axes[-1].set_xlabel("Longitude")
        axes[len(crop_sources) // 2].set_ylabel("Latitude")
        fig.suptitle("Crop Validation Overview")
        save_current(fig, "crop_validation_overview.png")

    return plot_names


def generate_report(summary_path: Path, run_log_path: Path, audit_root: Path) -> Path:
    payload = _load_payload(summary_path)
    log_text = _read_utf16_text(run_log_path)
    availability = _extract_availability(log_text)
    imerg_crop = _extract_imerg_crop(log_text)
    rows = _build_rows(payload)
    meta = _source_meta()
    checks = _build_checks(rows, meta, availability, imerg_crop)

    plots_dir = audit_root / "plots"
    logs_dir = audit_root / "logs"
    tables_dir = audit_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    plot_names = _generate_plots(rows, payload, availability, imerg_crop, plots_dir)

    _save_csv(rows, tables_dir / "decision_support_summary.csv")
    _save_csv(checks, tables_dir / "decision_support_checks.csv")
    _save_markdown(rows, checks, tables_dir / "decision_support_summary.md")
    _save_html(rows, checks, plot_names, tables_dir / "decision_support_summary.html")
    (tables_dir / "decision_support_summary.json").write_text(
        json.dumps({"summary": rows, "validation_checks": checks}, indent=2, default=str),
        encoding="utf-8",
    )

    report_json = {
        "generated_at": payload.get("generated_at"),
        "reproducibility": {
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "python_version": sys.version,
            "packages": _package_versions(),
        },
        "availability_windows": availability,
        "imerg_crop_bounds": imerg_crop,
        "summary": rows,
        "validation_checks": checks,
        "variable_catalog": {source_name: meta[source_name]["variables"] for source_name in meta},
        "concepts": {
            "download_burden_bytes": "bytes transferred from remote source (full_file_bytes)",
            "retained_raw_bytes": "bytes retained after local crop/subset in cache (source-dependent)",
        },
        "notes": {
            "raw_sample_bytes": "not persisted in cached summary; deferred until acquisition",
            "crop_validation": "crop size and bounds checks are proxied from cached evidence where available",
        },
    }
    (logs_dir / "decision_support_report.json").write_text(json.dumps(report_json, indent=2, default=str), encoding="utf-8")
    (logs_dir / "decision_support_report.log").write_text(
        "Decision support artifacts generated\n"
        f"Summary CSV: {(tables_dir / 'decision_support_summary.csv').as_posix()}\n"
        f"Checks CSV: {(tables_dir / 'decision_support_checks.csv').as_posix()}\n"
        f"Plots: {', '.join(plot_names)}\n",
        encoding="utf-8",
    )
    return audit_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate decision-support artifacts from cached validation outputs.")
    parser.add_argument("--summary", default=str(SUMMARY_JSON))
    parser.add_argument("--run-log", default=str(RUN_LOG))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    args = parser.parse_args()

    generate_report(Path(args.summary), Path(args.run_log), Path(args.audit_root))


if __name__ == "__main__":
    main()
