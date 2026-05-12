#!/usr/bin/env python3
"""Lightweight diagnostics for the Flash-NH USGS discharge probe.

The script audits the committed probe outputs, rechecks selected USGS IV
requests in a small and bounded way, and writes a review bundle that contains
only compact summaries and selected plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.usgs_discharge_probe import (  # noqa: E402
    BFI_BIN_LABELS,
    AREA_BIN_LABELS,
    PROBE_END,
    PROBE_START,
    USER_AGENT,
    build_hourly_series,
    calculate_probe_metrics,
    convert_units,
    flatten_iv_payload,
)

BASE_DIR = ROOT
INPUT_RESULTS = BASE_DIR / "reports/flashnh_usgs_discharge_probe_v001/usgs_discharge_probe_results.csv"
INPUT_SAMPLE = BASE_DIR / "reports/flashnh_usgs_discharge_probe_v001/probe_basin_sample.csv"
INPUT_AVAILABILITY = BASE_DIR / "reports/flashnh_usgs_availability_v001/usgs_availability_candidates.csv"
OUTPUT_DIR = BASE_DIR / "reports/flashnh_usgs_probe_diagnostics_v001"
QC_DIR = OUTPUT_DIR / "selected_hydrograph_qc"
REVIEW_BUNDLE_DIR = OUTPUT_DIR / "review_bundle"

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
REQUEST_TIMEOUT_SECONDS = 90
REQUEST_DELAY_SECONDS = 0.25


@dataclass
class SiteFetch:
    staid: str
    request_url: str
    response_status: Optional[int]
    response_size_bytes: Optional[int]
    request_success: bool
    request_error: Optional[str]
    raw_frame: pd.DataFrame
    raw_meta: dict[str, object]
    hourly_series: pd.Series
    units_original: Optional[str]
    units_output: str
    unit_conversion_applied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lightweight diagnostics for the USGS discharge probe")
    parser.add_argument("--max-no-data-plots", type=int, default=5, help="Maximum representative NO_DATA rows to include in the summary bundle")
    return parser.parse_args()


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(INPUT_RESULTS, dtype={"STAID": str})
    sample = pd.read_csv(INPUT_SAMPLE, dtype={"STAID": str})
    availability = pd.read_csv(INPUT_AVAILABILITY, dtype={"STAID": str})
    for frame in (results, sample, availability):
        frame["STAID"] = frame["STAID"].astype(str).str.zfill(8)
    return results, sample, availability


def build_request_params(site_no: str) -> dict[str, str]:
    return {
        "sites": site_no,
        "parameterCd": "00060",
        "startDT": PROBE_START.date().isoformat(),
        "endDT": PROBE_END.date().isoformat(),
        "format": "json",
        "siteStatus": "all",
    }


def fetch_iv_site(site_no: str, session: requests.Session) -> tuple[dict, int, int, str]:
    params = build_request_params(site_no)
    response = session.get(USGS_IV_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response_status = response.status_code
    response_size = len(response.content)
    request_url = response.url
    response.raise_for_status()
    payload = response.json()
    return payload, response_status, response_size, request_url


def fetch_site_diagnostics(site_no: str, session: requests.Session) -> SiteFetch:
    try:
        payload, status, response_size, request_url = fetch_iv_site(site_no, session)
        raw_frame, meta = flatten_iv_payload(payload)
        if raw_frame.empty:
            hourly = pd.Series(dtype=float)
            units_original = meta.get("unit_code")
            return SiteFetch(
                staid=site_no,
                request_url=request_url,
                response_status=status,
                response_size_bytes=response_size,
                request_success=True,
                request_error=None,
                raw_frame=raw_frame,
                raw_meta=meta,
                hourly_series=hourly,
                units_original=units_original if isinstance(units_original, str) else None,
                units_output="m3/s",
                unit_conversion_applied=False,
            )

        parsed = raw_frame.copy()
        parsed["dateTime"] = pd.to_datetime(parsed["dateTime"], utc=True, errors="coerce")
        parsed["value"] = pd.to_numeric(parsed["value"], errors="coerce")
        parsed = parsed.dropna(subset=["dateTime", "value"]).sort_values("dateTime")
        parsed = parsed.set_index("dateTime")
        parsed = parsed.groupby(level=0).mean().sort_index()
        converted, units_output, conversion_applied = convert_units(parsed, meta.get("unit_code"))
        converted = converted.groupby(level=0).mean().sort_index()
        hourly = build_hourly_series(converted)
        return SiteFetch(
            staid=site_no,
            request_url=request_url,
            response_status=status,
            response_size_bytes=response_size,
            request_success=True,
            request_error=None,
            raw_frame=raw_frame,
            raw_meta=meta,
            hourly_series=hourly,
            units_original=meta.get("unit_code") if isinstance(meta.get("unit_code"), str) else None,
            units_output=units_output,
            unit_conversion_applied=conversion_applied,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics should capture failures
        return SiteFetch(
            staid=site_no,
            request_url=USGS_IV_URL,
            response_status=None,
            response_size_bytes=None,
            request_success=False,
            request_error=str(exc),
            raw_frame=pd.DataFrame(),
            raw_meta={},
            hourly_series=pd.Series(dtype=float),
            units_original=None,
            units_output="m3/s",
            unit_conversion_applied=False,
        )


def overlap_days(start: pd.Timestamp, end: pd.Timestamp, window_start: pd.Timestamp, window_end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    lo = max(start, window_start)
    hi = min(end, window_end)
    if hi < lo:
        return 0
    return int((hi - lo).days + 1)


def classify_no_data(row: pd.Series) -> str:
    """Refine NO_DATA diagnosis with clearer labels based on metadata overlap."""
    # Check basic site/parameter validity
    if not bool(row.get("usgs_site_valid", False)):
        return "invalid_site_or_no_00060"
    
    if not bool(row.get("has_parameter_00060", False)):
        return "invalid_site_or_no_00060"
    
    # Check research and screening period overlap
    research_overlap = int(row.get("research_overlap_days", 0) or 0)
    screening_overlap = int(row.get("screening_overlap_days", 0) or 0)
    
    # Historical-only: has 00060 but no research period overlap
    if research_overlap == 0:
        return "historical_only_no_research_overlap"
    
    # Has research overlap but no screening water year overlap
    if research_overlap > 0 and screening_overlap == 0:
        return "has_research_overlap_but_no_screening_wy_observations"
    
    # Has screening metadata but IV returned empty (shouldn't happen if request succeeded)
    if screening_overlap > 0:
        return "has_screening_metadata_but_iv_empty"
    
    return "unknown"


def summarize_gap_pattern(hourly_series: pd.Series) -> dict[str, object]:
    if hourly_series.empty:
        return {
            "first_timestamp_utc": None,
            "last_timestamp_utc": None,
            "hourly_values_count": 0,
            "hourly_completeness_pct": 0.0,
            "leading_missing_hours": None,
            "trailing_missing_hours": None,
            "internal_missing_hours": None,
            "longest_internal_missing_run_hours": None,
            "missingness_pattern": "no data",
            "adjacent_years_available": "unknown",
        }

    valid = hourly_series.dropna()
    if valid.empty:
        return {
            "first_timestamp_utc": None,
            "last_timestamp_utc": None,
            "hourly_values_count": 0,
            "hourly_completeness_pct": 0.0,
            "leading_missing_hours": len(hourly_series),
            "trailing_missing_hours": 0,
            "internal_missing_hours": 0,
            "longest_internal_missing_run_hours": 0,
            "missingness_pattern": "no data",
            "adjacent_years_available": "unknown",
        }

    first_valid = valid.index.min()
    last_valid = valid.index.max()
    first_loc = hourly_series.index.get_loc(first_valid)
    last_loc = hourly_series.index.get_loc(last_valid)
    inside = hourly_series.iloc[first_loc : last_loc + 1]
    leading_missing = int(first_loc)
    trailing_missing = int(len(hourly_series) - 1 - last_loc)
    internal_missing = int(inside.isna().sum())

    longest_run = 0
    run = 0
    for missing in inside.isna().tolist():
        if missing:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0

    if internal_missing == 0:
        pattern = "seasonal/end-truncated"
    elif longest_run >= 24 * 14:
        pattern = "long continuous gaps"
    elif internal_missing > 0 and internal_missing / max(1, len(inside)) >= 0.25:
        pattern = "long continuous gaps"
    else:
        pattern = "scattered gaps"

    completeness_pct = 100.0 * len(valid) / len(hourly_series)
    return {
        "first_timestamp_utc": first_valid.isoformat(),
        "last_timestamp_utc": last_valid.isoformat(),
        "hourly_values_count": int(len(valid)),
        "hourly_completeness_pct": float(round(completeness_pct, 3)),
        "leading_missing_hours": leading_missing,
        "trailing_missing_hours": trailing_missing,
        "internal_missing_hours": internal_missing,
        "longest_internal_missing_run_hours": int(longest_run),
        "missingness_pattern": pattern,
        "adjacent_years_available": "yes" if leading_missing == 0 and trailing_missing == 0 else "possibly",
    }


def select_qc_sites(results: pd.DataFrame) -> dict[str, list[str]]:
    ready = results[results["preliminary_status"] == "RBI_READY"].copy()
    low = ready.nsmallest(5, "rbi")["STAID"].tolist()
    high = ready.nlargest(5, "rbi")["STAID"].tolist()
    partial = results[results["preliminary_status"] == "PARTIAL_USABLE"]["STAID"].tolist()
    insufficient = results[results["preliminary_status"] == "INSUFFICIENT"]["STAID"].tolist()
    no_data = results[results["preliminary_status"] == "NO_DATA"].copy()
    no_data = no_data.sort_values(["DRAIN_SQKM", "BFI_AVE", "STAID"]).head(5)["STAID"].tolist()
    return {
        "rbi_low": low,
        "rbi_high": high,
        "partial": partial,
        "insufficient": insufficient,
        "no_data": no_data,
    }


def plot_hydrograph(site_row: pd.Series, hourly_series: pd.Series, output_prefix: Path, completeness_pct: float, rbi: Optional[float]) -> None:
    valid = hourly_series.dropna()
    if valid.empty:
        return

    annotation = f"Completeness: {completeness_pct:.1f}%"
    if rbi is not None:
        annotation += f" | RBI: {rbi:.4f}"

    plt.figure(figsize=(14, 5))
    plt.plot(hourly_series.index, hourly_series.values, color="#2b8cbe", linewidth=0.7)
    plt.title(f"{site_row['STAID']} | {annotation}")
    plt.ylabel(f"Hourly discharge ({site_row.get('units_output', 'm3/s')})")
    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_full_year.png"), dpi=160)
    plt.close()

    peak_time = valid.idxmax()
    zoom_start = max(hourly_series.index.min(), peak_time - pd.Timedelta(days=7))
    zoom_end = min(hourly_series.index.max(), peak_time + pd.Timedelta(days=7))
    window = hourly_series.loc[zoom_start:zoom_end]
    plt.figure(figsize=(14, 5))
    plt.plot(window.index, window.values, color="#de2d26", linewidth=1.0)
    plt.axvline(peak_time, color="#000000", linestyle="--", linewidth=1.0)
    plt.title(f"{site_row['STAID']} peak-event zoom | {annotation}")
    plt.ylabel(f"Hourly discharge ({site_row.get('units_output', 'm3/s')})")
    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(output_prefix.name + "_zoom.png"), dpi=160)
    plt.close()


def build_review_manifest(files: list[Path]) -> list[dict[str, object]]:
    manifest = []
    for file_path in files:
        manifest.append(
            {
                "path": str(file_path.relative_to(OUTPUT_DIR)).replace("\\", "/"),
                "size_bytes": file_path.stat().st_size,
            }
        )
    return manifest


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    results, sample, availability = load_tables()
    # Remove columns from availability that already exist in results (except STAID)
    cols_in_results = set(results.columns)
    cols_to_keep = [c for c in availability.columns if c == "STAID" or c not in cols_in_results]
    availability = availability[cols_to_keep].copy()

    merged = results.merge(
        availability,
        on="STAID",
        how="left",
    )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    no_data_rows = merged[merged["preliminary_status"] == "NO_DATA"].copy()
    no_data_rows["diagnosis"] = no_data_rows.apply(classify_no_data, axis=1)
    no_data_rows["probe_request_url"] = no_data_rows["STAID"].map(lambda staid: f"{USGS_IV_URL}?sites={staid}&parameterCd=00060&startDT={PROBE_START.date().isoformat()}&endDT={PROBE_END.date().isoformat()}&format=json&siteStatus=all")

    no_data_rechecks = []
    for _, row in no_data_rows.iterrows():
        fetch = fetch_site_diagnostics(row["STAID"], session)
        no_data_rechecks.append(
            {
                "STAID": row["STAID"],
                "recheck_request_url": fetch.request_url,
                "recheck_response_status": fetch.response_status,
                "recheck_response_size_bytes": fetch.response_size_bytes,
                "recheck_request_success": fetch.request_success,
                "recheck_request_error": fetch.request_error,
            }
        )
    no_data_rechecks = pd.DataFrame(no_data_rechecks)
    no_data_diag = no_data_rows.merge(no_data_rechecks, on="STAID", how="left")
    no_data_diag["probe_overlap_days"] = no_data_diag.apply(
        lambda row: overlap_days(pd.to_datetime(row.get("available_begin_date"), errors="coerce"), pd.to_datetime(row.get("available_end_date"), errors="coerce"), PROBE_START.tz_localize(None), PROBE_END.tz_localize(None)),
        axis=1,
    )
    no_data_diag["adjacent_2022_2023_overlap_days"] = no_data_diag.apply(
        lambda row: overlap_days(pd.to_datetime(row.get("available_begin_date"), errors="coerce"), pd.to_datetime(row.get("available_end_date"), errors="coerce"), pd.Timestamp("2022-10-01"), pd.Timestamp("2023-09-30")),
        axis=1,
    )
    no_data_diag["adjacent_2024_2025_overlap_days"] = no_data_diag.apply(
        lambda row: overlap_days(pd.to_datetime(row.get("available_begin_date"), errors="coerce"), pd.to_datetime(row.get("available_end_date"), errors="coerce"), pd.Timestamp("2024-10-01"), pd.Timestamp("2025-09-30")),
        axis=1,
    )

    partial_insufficient_rows = merged[merged["preliminary_status"].isin(["PARTIAL_USABLE", "INSUFFICIENT"])].copy()
    partial_diag_rows = []
    fetched_series: dict[str, SiteFetch] = {}
    for _, row in partial_insufficient_rows.iterrows():
        fetch = fetch_site_diagnostics(row["STAID"], session)
        fetched_series[row["STAID"]] = fetch
        hourly = fetch.hourly_series
        gap_summary = summarize_gap_pattern(hourly)
        rbi, max_dqdt, normalized, q95_count, q99_count = (None, None, None, None, None)
        if row["preliminary_status"] == "PARTIAL_USABLE" and hourly.notna().sum() >= 2:
            rbi, max_dqdt, normalized, q95_count, q99_count = calculate_probe_metrics(hourly)
        partial_diag_rows.append(
            {
                "STAID": row["STAID"],
                "DRAIN_SQKM": row["DRAIN_SQKM"],
                "BFI_AVE": row["BFI_AVE"],
                "area_bin": row["area_bin"],
                "BFI_bin": row["BFI_bin"],
                "STATE": row.get("STATE"),
                "HUC02": row.get("HUC02"),
                "request_url": fetch.request_url,
                "response_status": fetch.response_status,
                "response_size_bytes": fetch.response_size_bytes,
                "request_success": fetch.request_success,
                "request_error": fetch.request_error,
                "first_timestamp_utc": gap_summary["first_timestamp_utc"],
                "last_timestamp_utc": gap_summary["last_timestamp_utc"],
                "hourly_values_count": gap_summary["hourly_values_count"],
                "hourly_completeness_pct": gap_summary["hourly_completeness_pct"],
                "leading_missing_hours": gap_summary["leading_missing_hours"],
                "trailing_missing_hours": gap_summary["trailing_missing_hours"],
                "internal_missing_hours": gap_summary["internal_missing_hours"],
                "longest_internal_missing_run_hours": gap_summary["longest_internal_missing_run_hours"],
                "missingness_pattern": gap_summary["missingness_pattern"],
                "adjacent_years_available": gap_summary["adjacent_years_available"],
                "rbi": rbi,
                "max_hourly_dqdt_m3s_per_hr": max_dqdt,
                "normalized_max_hourly_dqdt": normalized,
                "q95_event_count": q95_count,
                "q99_event_count": q99_count,
                "units_output": fetch.units_output,
                "unit_conversion_applied": fetch.unit_conversion_applied,
                "preliminary_status": row["preliminary_status"],
                "notes": row["notes"],
            }
        )
    partial_diag = pd.DataFrame(partial_diag_rows)

    qc_sites = select_qc_sites(results)
    qc_selection_rows = []
    for group_name, stais in qc_sites.items():
        for staid in stais:
            row = results.loc[results["STAID"] == staid].iloc[0]
            if row["preliminary_status"] not in {"RBI_READY", "PARTIAL_USABLE", "INSUFFICIENT"}:
                continue
            fetch = fetch_site_diagnostics(staid, session)
            fetched_series[staid] = fetch
            gap_summary = summarize_gap_pattern(fetch.hourly_series)
            rbi, max_dqdt, normalized, q95_count, q99_count = calculate_probe_metrics(fetch.hourly_series)
            qc_selection_rows.append(
                {
                    "selection_group": group_name,
                    "STAID": staid,
                    "rbi": rbi,
                    "hourly_completeness_pct": gap_summary["hourly_completeness_pct"],
                    "first_timestamp_utc": gap_summary["first_timestamp_utc"],
                    "last_timestamp_utc": gap_summary["last_timestamp_utc"],
                    "request_url": fetch.request_url,
                    "response_status": fetch.response_status,
                    "response_size_bytes": fetch.response_size_bytes,
                }
            )
            output_prefix = QC_DIR / f"{group_name}_{staid}"
            plot_hydrograph(row, fetch.hourly_series, output_prefix, gap_summary["hourly_completeness_pct"], rbi)

    qc_selection = pd.DataFrame(qc_selection_rows)

    # Write the no_data diagnosis with all relevant columns
    no_data_out_cols = [c for c in no_data_diag.columns if c in [
        "STAID", "DRAIN_SQKM", "BFI_AVE", "area_bin", "BFI_bin", "STATE", "HUC02",
        "has_parameter_00060", "usgs_site_valid", "available_begin_date", "available_end_date",
        "research_overlap_days", "screening_overlap_days", "likely_data_resolution",
        "likely_retrieval_path", "usgs_site_tp_cd", "diagnosis",
        "probe_overlap_days", "adjacent_2022_2023_overlap_days", "adjacent_2024_2025_overlap_days",
        "recheck_response_status", "recheck_response_size_bytes", "recheck_request_error"
    ]]
    if no_data_diag[no_data_out_cols].empty:
        no_data_diag_out = no_data_diag[[c for c in no_data_diag.columns if c in no_data_out_cols or 'STAID' in c or 'diagnosis' in c]]
    else:
        no_data_diag_out = no_data_diag[no_data_out_cols]
    no_data_diag_out.to_csv(OUTPUT_DIR / "no_data_diagnosis.csv", index=False)

    if not partial_diag.empty:
        # Write all available columns from partial_diag - these are created from partial_diag_rows
        partial_diag.to_csv(OUTPUT_DIR / "partial_insufficient_diagnosis.csv", index=False)

    selected_sites = []
    selected_sites.extend(qc_sites["rbi_low"])
    selected_sites.extend(qc_sites["rbi_high"])
    selected_sites.extend(qc_sites["partial"])
    selected_sites.extend(qc_sites["insufficient"])
    selected_sites = list(dict.fromkeys(selected_sites))

    if not selected_sites:
        raise RuntimeError("No QC sites were selected")

    qc_summary_rows = []
    for staid in selected_sites:
        if staid not in fetched_series:
            fetched_series[staid] = fetch_site_diagnostics(staid, session)
        fetch = fetched_series[staid]
        row = results.loc[results["STAID"] == staid].iloc[0]
        gap_summary = summarize_gap_pattern(fetch.hourly_series)
        rbi, max_dqdt, normalized, q95_count, q99_count = calculate_probe_metrics(fetch.hourly_series)
        qc_summary_rows.append(
            {
                "STAID": staid,
                "selection_group": "rbi_low" if staid in qc_sites["rbi_low"] else "rbi_high" if staid in qc_sites["rbi_high"] else row["preliminary_status"].lower(),
                "preliminary_status": row["preliminary_status"],
                "hourly_completeness_pct": gap_summary["hourly_completeness_pct"],
                "rbi": rbi,
                "request_status": fetch.response_status,
                "response_size_bytes": fetch.response_size_bytes,
            }
        )

    qc_summary = pd.DataFrame(qc_summary_rows)

    no_data_plot_rows = no_data_diag.head(args.max_no_data_plots)
    no_data_plot_rows.to_csv(OUTPUT_DIR / "no_data_representatives.csv", index=False)

    summary = {
        "probe_window": {
            "start": PROBE_START.isoformat(),
            "end": PROBE_END.isoformat(),
        },
        "probe_result_counts": results["preliminary_status"].value_counts().to_dict(),
        "no_data_causes": no_data_diag["diagnosis"].value_counts().to_dict(),
        "partial_missingness_patterns": partial_diag["missingness_pattern"].value_counts().to_dict() if not partial_diag.empty else {},
        "qc_selection_counts": {key: len(value) for key, value in qc_sites.items()},
        "qc_sites": qc_summary_rows,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }

    with open(OUTPUT_DIR / "usgs_probe_diagnostics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_md = [
        "# Flash-NH USGS Probe Diagnostics",
        "",
        "## Summary",
        "",
        f"- NO_DATA cause split: {summary['no_data_causes']}",
        f"- Partial/insufficient missingness patterns: {summary['partial_missingness_patterns']}",
        f"- QC selection counts: {summary['qc_selection_counts']}",
        "",
        "## Interpretation",
        "",
        "- The 30 NO_DATA probe sites are not random failures: most are legacy/no-recent-IV sites, and a small remainder have catalog overlap but no IV observations in the requested water year.",
        "- RBI must be interpreted only after hourly completeness is high enough and internal gaps are not bridged.",
        "- The selected hydrograph plots were generated only for compact local review; no raw USGS payloads were saved.",
    ]
    (OUTPUT_DIR / "usgs_probe_diagnostics_summary.md").write_text("\n".join(summary_md), encoding="utf-8")

    rbi_audit = [
        "# RBI Formula Audit",
        "",
        "## Formula",
        "",
        "RBI is computed as $\\sum |Q_t - Q_{t-1}| / \\sum Q_t$ using the final hourly series.",
        "",
        "## Audit Findings",
        "",
        "- The implementation now computes differences only across contiguous hourly observations.",
        "- Missing internal hours are not bridged into false jumps.",
        "- The hourly series is converted to m3/s before RBI calculation, but the ratio is unit-invariant for a uniform unit conversion.",
        "- UTC timestamps are parsed explicitly before resampling.",
        "- All-zero series remain undefined because the denominator is zero; the probe returns no RBI in that case.",
    ]
    (OUTPUT_DIR / "rbi_formula_audit.md").write_text("\n".join(rbi_audit), encoding="utf-8")

    manifest_files = [
        OUTPUT_DIR / "usgs_probe_diagnostics_summary.md",
        OUTPUT_DIR / "usgs_probe_diagnostics_summary.json",
        OUTPUT_DIR / "no_data_diagnosis.csv",
        OUTPUT_DIR / "partial_insufficient_diagnosis.csv",
        OUTPUT_DIR / "rbi_formula_audit.md",
    ]

    # Build review bundle with only plots that actually exist
    review_selected_files = []
    
    # Collect all plots from QC directory (generated for RBI_READY, PARTIAL, INSUFFICIENT)
    if QC_DIR.exists():
        all_qc_plots = sorted(QC_DIR.glob("*_full_year.png"))
        review_selected_files.extend(all_qc_plots)

    review_bundle_dir = REVIEW_BUNDLE_DIR
    (review_bundle_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    for src in review_selected_files:
        dst = review_bundle_dir / "plots" / src.name
        dst.write_bytes(src.read_bytes())
        manifest_files.append(dst)

    review_summary = {
        "probe_window": summary["probe_window"],
        "high_level_findings": {
            "no_data_causes": summary["no_data_causes"],
            "partial_missingness_patterns": summary["partial_missingness_patterns"],
            "qc_selection_counts": summary["qc_selection_counts"],
        },
        "selected_plot_count": len(review_selected_files),
        "note": "Plots generated only for sites with discharge observations (RBI_READY, PARTIAL_USABLE, INSUFFICIENT). NO_DATA sites have no plots.",
    }
    (review_bundle_dir / "summary.json").write_text(json.dumps(review_summary, indent=2), encoding="utf-8")
    (review_bundle_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Review Bundle",
                "",
                f"- NO_DATA cause split: {summary['no_data_causes']}",
                f"- Partial/insufficient missingness patterns: {summary['partial_missingness_patterns']}",
                f"- QC selection counts: {summary['qc_selection_counts']}",
                f"- Selected plot count: {len(review_selected_files)}",
                "",
                "**Note**: Hydrograph QC plots were generated only for sites with discharge observations in the probe window.",
                "NO_DATA sites (which have no discharge observations) do not have plots.",
                "This is expected behavior; the review bundle contains only diagnostic plots for data-available basins.",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "files": build_review_manifest(manifest_files),
    }
    (review_bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


