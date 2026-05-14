#!/usr/bin/env python3
"""Build WY2024 streamflow metrics matrix for usable USGS basins.

This workflow downloads/stores WY2024 hourly streamflow for all RBI_READY and
optionally PARTIAL_USABLE basins from the existing screening results, computes
the full streamflow/event/QC metrics matrix, and produces a resumable,
checkpointed output for unbiased pilot basin selection.

Hard constraints:
- Reuse existing IV retrieval and resampling logic; do not modify RBI formula.
- Use gap-aware metrics: no dQ across missing timestamps, no RBI across gaps.
- Compute both hard QC (exclusion) and context flags (informational).
- Do not save raw USGS JSON/XML responses by default.
- Keep all runs resumable and checkpointed; support --max-basins smoke mode.
- Do not bridge missing gaps in hourly resampling or metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from scripts.usgs_discharge_probe import (
    build_hourly_series,
    convert_units,
    flatten_iv_payload,
)
from scripts.usgs_rbi_screening_scale import fetch_iv_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "reports" / "flashnh_usgs_rbi_screening_wy2024_v001"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "flashnh_wy2024_streamflow_metrics_v001"
WY_START = pd.Timestamp("2023-10-01T00:00:00Z")
WY_END = pd.Timestamp("2024-09-30T23:00:00Z")
EXPECTED_HOURLY_INDEX = pd.date_range(start=WY_START, end=WY_END, freq="1h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WY2024 streamflow metrics matrix for usable USGS basins"
    )
    parser.add_argument(
        "--input-results",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the completed screening results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics matrix outputs",
    )
    parser.add_argument(
        "--max-basins",
        type=int,
        default=0,
        help="Cap on the total number of basins to process (0=all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Basins per checkpoint batch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint if it exists",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Delay between USGS requests",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum USGS request retries per basin",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=45,
        help="USGS request timeout in seconds",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached hourly Parquet and re-download",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not fetch missing hourly series; use cache only",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include PARTIAL_USABLE basins in addition to RBI_READY",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking in sampling",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wy2024_streamflow_metrics_matrix")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(logs_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def find_results_file(results_dir: Path) -> Path:
    csv_path = results_dir / "usgs_rbi_screening_results.csv"
    if csv_path.exists():
        return csv_path
    parquet_path = results_dir / "usgs_rbi_screening_results.parquet"
    if parquet_path.exists():
        return parquet_path
    raise FileNotFoundError(f"Could not find screening results in {results_dir}")


def load_screening_results(results_path: Path) -> pd.DataFrame:
    if results_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(results_path)
    else:
        df = pd.read_csv(results_path)
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    numeric_cols = [
        "DRAIN_SQKM",
        "BFI_AVE",
        "LAT_GAGE",
        "LNG_GAGE",
        "rbi",
        "max_hourly_dqdt_m3s_per_hr",
        "normalized_max_hourly_dqdt",
        "hourly_completeness_pct",
        "hourly_values_count",
        "expected_hourly_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_or_fetch_hourly(
    session: requests.Session,
    staid: str,
    output_hourly_dir: Path,
    logger: logging.Logger,
    sleep_seconds: float,
    max_retries: int,
    timeout_seconds: int,
    no_download: bool,
    force_refresh: bool,
) -> tuple[Optional[pd.Series], str]:
    parquet_file = output_hourly_dir / f"{staid}.parquet"

    if not force_refresh and parquet_file.exists():
        try:
            table = pq.read_table(parquet_file)
            df_cached = table.to_pandas()
            if "time_utc" in df_cached.columns and "discharge_m3s" in df_cached.columns:
                df_cached["time_utc"] = pd.to_datetime(df_cached["time_utc"], utc=True)
                series = pd.Series(
                    df_cached["discharge_m3s"].values,
                    index=pd.DatetimeIndex(df_cached["time_utc"], tz="UTC"),
                    name="discharge",
                )
                return series, "cache_parquet"
        except Exception as exc:
            logger.warning("Could not read cached parquet for %s: %s", staid, exc)

    if no_download:
        return None, "missing_cache_no_download"

    try:
        payload, http_status, response_size, request_url = fetch_iv_json(
            session=session,
            site_no=staid,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            logger=logger,
        )
    except Exception as exc:
        logger.warning("Failed to fetch USGS IV for %s: %s", staid, exc)
        return None, f"fetch_error: {exc}"

    try:
        values, meta = flatten_iv_payload(payload)
        if values.empty:
            logger.warning("No IV values returned for %s (%s)", staid, http_status)
            return None, "empty_response"

        values["value"] = pd.to_numeric(values["value"], errors="coerce")
        units = meta.get("unit_code")
        converted, _, _ = convert_units(values["value"], units)
        raw_series = pd.Series(
            converted.values,
            index=pd.DatetimeIndex(
                pd.to_datetime(values["dateTime"], utc=True), tz="UTC"
            ),
            name="discharge",
        )
        hourly = build_hourly_series(raw_series.sort_index())

        # Save to parquet
        df_hourly = hourly.reset_index()
        df_hourly.columns = ["time_utc", "discharge_m3s"]
        df_hourly["original_units"] = units
        df_hourly["source"] = "usgs_iv"
        df_hourly["is_missing"] = df_hourly["discharge_m3s"].isna()
        df_hourly["quality_code"] = None

        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df_hourly)
        pq.write_table(table, parquet_file)

        return hourly, f"downloaded:{http_status}"
    except Exception as exc:
        logger.exception("Failed to process USGS payload for %s: %s", staid, exc)
        return None, f"parse_error: {exc}"


def compute_wy2024_metrics(
    hourly: pd.Series, row: pd.Series
) -> tuple[dict, list[str], list[str]]:
    """Compute WY2024 streamflow metrics and QC flags."""
    aligned = hourly.reindex(EXPECTED_HOURLY_INDEX)
    valid = aligned.dropna()
    expected_count = len(EXPECTED_HOURLY_INDEX)
    valid_count = int(valid.size)
    completeness_pct = float(valid_count / expected_count * 100.0) if expected_count else 0.0

    # Quantiles and basic stats
    q_min = float(valid.min()) if not valid.empty else float("nan")
    q50 = float(valid.quantile(0.50)) if not valid.empty else float("nan")
    q95 = float(valid.quantile(0.95)) if not valid.empty else float("nan")
    q99 = float(valid.quantile(0.99)) if not valid.empty else float("nan")
    q_max = float(valid.max()) if not valid.empty else float("nan")
    q_mean = float(valid.mean()) if not valid.empty else float("nan")
    sum_q = float(valid.sum()) if not valid.empty else float("nan")

    # Zero and negative flow
    zero_count = int((valid == 0).sum()) if not valid.empty else 0
    negative_count = int((valid < 0).sum()) if not valid.empty else 0
    zero_fraction = float(zero_count / valid_count) if valid_count > 0 else 0.0
    negative_fraction = float(negative_count / valid_count) if valid_count > 0 else 0.0

    # Hourly jumps (gap-aware)
    diffs = valid.diff().dropna()
    max_rise = float("nan")
    max_rise_time = None
    max_fall = float("nan")
    max_fall_time = None
    max_abs_jump = float("nan")
    max_abs_jump_time = None

    if not diffs.empty:
        max_rise_idx = diffs.idxmax()
        if pd.notna(max_rise_idx):
            max_rise = float(diffs.loc[max_rise_idx])
            max_rise_time = max_rise_idx

        max_fall_idx = diffs.idxmin()
        if pd.notna(max_fall_idx):
            max_fall = float(diffs.loc[max_fall_idx])
            max_fall_time = max_fall_idx

        max_abs_idx = diffs.abs().idxmax()
        if pd.notna(max_abs_idx):
            max_abs_jump = float(diffs.abs().loc[max_abs_idx])
            max_abs_jump_time = max_abs_idx

    # RBI (gap-aware)
    rbi = float("nan")
    if not diffs.empty and sum_q > 0:
        rbi = float(diffs.abs().sum() / sum_q)
    elif sum_q <= 0:
        rbi = float("nan")

    # Per-km2 metrics
    drain_sqkm = float(row.get("DRAIN_SQKM", 1.0))
    if drain_sqkm is None or not np.isfinite(drain_sqkm) or drain_sqkm <= 0:
        drain_sqkm = 1.0

    max_rise_per_km2 = max_rise / drain_sqkm if np.isfinite(max_rise) else float("nan")
    max_abs_jump_per_km2 = (
        max_abs_jump / drain_sqkm if np.isfinite(max_abs_jump) else float("nan")
    )
    q95_per_km2 = q95 / drain_sqkm if np.isfinite(q95) else float("nan")
    q99_per_km2 = q99 / drain_sqkm if np.isfinite(q99) else float("nan")
    qmax_per_km2 = q_max / drain_sqkm if np.isfinite(q_max) else float("nan")

    # Ratios with Q50
    q95_q50_ratio = q95 / q50 if q50 > 0 and np.isfinite(q50) else float("nan")
    q99_q50_ratio = q99 / q50 if q50 > 0 and np.isfinite(q50) else float("nan")
    max_rise_over_q50 = (
        max_rise / q50 if q50 > 0 and np.isfinite(max_rise) else float("nan")
    )
    max_abs_jump_over_q50 = (
        max_abs_jump / q50 if q50 > 0 and np.isfinite(max_abs_jump) else float("nan")
    )

    # Event detection: local peaks >= Q95
    peaks_q95 = 0
    peaks_q99 = 0
    if not valid.empty and np.isfinite(q95):
        is_peak = (valid >= valid.shift(1).fillna(-np.inf)) & (valid > valid.shift(-1).fillna(-np.inf))
        peaks_q95 = int((valid[is_peak] >= q95).sum())
    if not valid.empty and np.isfinite(q99):
        is_peak = (valid >= valid.shift(1).fillna(-np.inf)) & (valid > valid.shift(-1).fillna(-np.inf))
        peaks_q99 = int((valid[is_peak] >= q99).sum())

    # Largest event
    largest_peak_time = None
    largest_peak_q = float("nan")
    largest_peak_rise = float("nan")
    largest_peak_rise_per_km2 = float("nan")

    if not valid.empty and len(valid) > 1:
        is_peak = (valid >= valid.shift(1).fillna(-np.inf)) & (
            valid > valid.shift(-1).fillna(-np.inf)
        )
        peak_vals = valid[is_peak]
        if not peak_vals.empty:
            max_peak_idx = peak_vals.idxmax()
            largest_peak_q = float(peak_vals.loc[max_peak_idx])
            largest_peak_time = max_peak_idx
            # Rise to that peak
            pre_window = valid.loc[:max_peak_idx]
            if len(pre_window) > 1:
                pre_diffs = pre_window.diff().dropna()
                if not pre_diffs.empty:
                    largest_peak_rise = float(pre_diffs.max())
                    largest_peak_rise_per_km2 = largest_peak_rise / drain_sqkm

    metrics = {
        "STAID": str(row.get("STAID", "")).zfill(8),
        "source_status": row.get("screening_status"),
        "DRAIN_SQKM": float(row.get("DRAIN_SQKM", float("nan"))),
        "BFI_AVE": float(row.get("BFI_AVE", float("nan"))),
        "area_bin": row.get("area_bin"),
        "BFI_bin": row.get("BFI_bin"),
        "STATE": row.get("STATE"),
        "HUC02": row.get("HUC02"),
        "LAT_GAGE": float(row.get("LAT_GAGE", float("nan"))),
        "LNG_GAGE": float(row.get("LNG_GAGE", float("nan"))),
        "valid_hour_count": valid_count,
        "expected_hour_count": expected_count,
        "hourly_completeness_pct": completeness_pct,
        "Q_min": q_min,
        "Q50": q50,
        "Q95": q95,
        "Q99": q99,
        "Q_max": q_max,
        "Q_mean": q_mean,
        "sum_Q": sum_q,
        "zero_flow_count": zero_count,
        "zero_flow_fraction": zero_fraction,
        "negative_flow_count": negative_count,
        "negative_flow_fraction": negative_fraction,
        "RBI": rbi,
        "q95_q50_ratio": q95_q50_ratio,
        "q99_q50_ratio": q99_q50_ratio,
        "max_hourly_rise": max_rise,
        "max_hourly_rise_time": max_rise_time,
        "max_hourly_fall": max_fall,
        "max_hourly_fall_time": max_fall_time,
        "max_abs_hourly_jump": max_abs_jump,
        "max_abs_hourly_jump_time": max_abs_jump_time,
        "max_hourly_rise_per_km2": max_rise_per_km2,
        "max_abs_hourly_jump_per_km2": max_abs_jump_per_km2,
        "max_hourly_rise_over_Q50": max_rise_over_q50,
        "max_abs_hourly_jump_over_Q50": max_abs_jump_over_q50,
        "Q95_per_km2": q95_per_km2,
        "Q99_per_km2": q99_per_km2,
        "Qmax_per_km2": qmax_per_km2,
        "event_count_q95": peaks_q95,
        "event_count_q99": peaks_q99,
        "largest_event_peak_time": largest_peak_time,
        "largest_event_peak_Q": largest_peak_q,
        "largest_event_rise": largest_peak_rise,
        "largest_event_rise_per_km2": largest_peak_rise_per_km2,
    }

    # Hard QC flags (exclusion criteria)
    hard_flags: list[str] = []
    context_flags: list[str] = []

    if completeness_pct < 90.0:
        hard_flags.append("HARD_LOW_COMPLETENESS_LT90")
    if negative_fraction > 0.01:
        hard_flags.append("HARD_NEGATIVE_FLOW_SEVERE")
    if zero_fraction >= 0.25:
        hard_flags.append("HARD_ZERO_FLOW_DOMINATED")
    if q50 <= 0 or not np.isfinite(q50):
        hard_flags.append("HARD_Q50_ZERO_OR_NEAR_ZERO")
    if not np.isfinite(rbi):
        hard_flags.append("HARD_NO_RBI")
    if np.isfinite(max_abs_jump_over_q50) and max_abs_jump_over_q50 >= 20:
        hard_flags.append("HARD_SUSPICIOUS_SPIKE_SEVERE")

    # Context flags (informational)
    if zero_fraction >= 0.05:
        context_flags.append("CONTEXT_ZERO_FLOW_SOME")
    if np.isfinite(max_abs_jump_over_q50) and max_abs_jump_over_q50 >= 5:
        context_flags.append("CONTEXT_HIGH_NORMALIZED_JUMP")
    if np.isfinite(q99_per_km2) and q99_per_km2 <= 0.01:
        context_flags.append("CONTEXT_LOW_SPECIFIC_FLOW")
    if np.isfinite(qmax_per_km2) and qmax_per_km2 >= 1.0:
        context_flags.append("CONTEXT_HIGH_SPECIFIC_PEAK")
    if zero_fraction >= 0.10:
        context_flags.append("CONTEXT_INTERMITTENT_LIKE")
    if np.isfinite(max_abs_jump_over_q50) and max_abs_jump_over_q50 >= 10 and int(row.get("longest_flat_run_hours", 0)) >= 24:
        context_flags.append("CONTEXT_POSSIBLE_REGULATION_OR_ARTIFACT")
    if drain_sqkm < 10:
        context_flags.append("CONTEXT_SMALL_BASIN")
    bfi_ave = float(row.get("BFI_AVE", float("nan")))
    if np.isfinite(bfi_ave) and bfi_ave >= 60:
        context_flags.append("CONTEXT_HIGH_BFI")
    if np.isfinite(bfi_ave) and bfi_ave <= 10:
        context_flags.append("CONTEXT_LOW_BFI")

    return metrics, hard_flags, context_flags


def classify_candidate_class(
    metrics: dict, hard_flags: list[str], context_flags: list[str]
) -> str:
    """Classify basin into candidate class."""
    if hard_flags:
        return "EXCLUDE_HARD_QC"

    completeness = metrics.get("hourly_completeness_pct", 0)
    rbi = metrics.get("RBI", float("nan"))
    event_count_q99 = int(metrics.get("event_count_q99", 0))
    max_abs_jump_over_q50 = metrics.get("max_abs_hourly_jump_over_Q50", float("nan"))

    strong_response = False
    if np.isfinite(max_abs_jump_over_q50) and max_abs_jump_over_q50 >= 5:
        strong_response = True
    if event_count_q99 >= 2:
        strong_response = True

    if completeness >= 95 and np.isfinite(rbi) and rbi >= 0.10:
        return "FLASHY_CORE"

    if (
        completeness >= 95
        and np.isfinite(rbi)
        and 0.05 <= rbi < 0.10
        and strong_response
    ):
        return "FLASHY_MODERATE"

    if np.isfinite(rbi) and rbi < 0.05 and strong_response and completeness >= 90:
        return "FLASHY_POSSIBLE"

    if np.isfinite(rbi) and rbi < 0.05 and not strong_response and completeness >= 90:
        return "LOW_FLASHINESS_CONTROL"

    if context_flags:
        return "MANUAL_REVIEW_CONTEXT"

    return "MANUAL_REVIEW_CONTEXT"


def load_checkpoint(output_dir: Path) -> dict:
    """Load the latest checkpoint."""
    checkpoint_file = output_dir / "logs" / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            return json.loads(checkpoint_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "processed_staids": [],
        "last_batch_index": -1,
        "total_processed": 0,
        "total_successful": 0,
        "total_failed": 0,
    }


def save_checkpoint(output_dir: Path, checkpoint: dict, logger: logging.Logger) -> None:
    """Save checkpoint."""
    checkpoint_file = output_dir / "logs" / "checkpoint.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Checkpoint saved: processed=%s, successful=%s, failed=%s",
        checkpoint["total_processed"],
        checkpoint["total_successful"],
        checkpoint["total_failed"],
    )


def build_summary(
    metrics_df: pd.DataFrame, output_dir: Path, logger: logging.Logger
) -> dict:
    """Build summary statistics and outputs."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics tables
    metrics_df.to_csv(tables_dir / "wy2024_streamflow_metrics.csv", index=False)
    try:
        table = pa.Table.from_pandas(metrics_df)
        pq.write_table(table, tables_dir / "wy2024_streamflow_metrics.parquet")
    except Exception as exc:
        logger.warning("Could not write metrics parquet: %s", exc)

    # Class distribution
    class_counts = (
        metrics_df["candidate_class"]
        .value_counts()
        .reset_index()
    )
    class_counts.columns = ["candidate_class", "count"]
    class_counts.to_csv(tables_dir / "candidate_classes.csv", index=False)

    # Hard QC exclusions
    hard_qc_df = metrics_df[metrics_df["candidate_class"] == "EXCLUDE_HARD_QC"].copy()
    hard_qc_df.to_csv(tables_dir / "hard_qc_exclusions.csv", index=False)

    # Manual review context
    context_df = metrics_df[
        metrics_df["candidate_class"] == "MANUAL_REVIEW_CONTEXT"
    ].copy()
    context_df.to_csv(tables_dir / "manual_review_context.csv", index=False)

    # Flashy candidates
    flashy_df = metrics_df[
        metrics_df["candidate_class"].isin(["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE"])
    ].copy()
    flashy_df.to_csv(tables_dir / "flashy_candidate_pool.csv", index=False)

    # Low flashiness controls
    controls_df = metrics_df[
        metrics_df["candidate_class"] == "LOW_FLASHINESS_CONTROL"
    ].copy()
    controls_df.to_csv(tables_dir / "low_flashiness_controls.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_basins": int(len(metrics_df)),
        "candidate_class_counts": class_counts.to_dict(orient="records"),
        "hard_qc_exclusions_count": int(len(hard_qc_df)),
        "flashy_core_count": int(
            len(metrics_df[metrics_df["candidate_class"] == "FLASHY_CORE"])
        ),
        "flashy_moderate_count": int(
            len(metrics_df[metrics_df["candidate_class"] == "FLASHY_MODERATE"])
        ),
        "flashy_possible_count": int(
            len(metrics_df[metrics_df["candidate_class"] == "FLASHY_POSSIBLE"])
        ),
        "low_flashiness_control_count": int(len(controls_df)),
        "manual_review_count": int(
            len(metrics_df[metrics_df["candidate_class"] == "MANUAL_REVIEW_CONTEXT"])
        ),
        "metrics_csv": str(tables_dir / "wy2024_streamflow_metrics.csv"),
        "candidate_classes_csv": str(tables_dir / "candidate_classes.csv"),
        "hard_qc_exclusions_csv": str(tables_dir / "hard_qc_exclusions.csv"),
    }

    (summaries_dir / "wy2024_streamflow_metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    md_lines = [
        "# WY2024 Streamflow Metrics Matrix Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Total basins processed: {summary['total_basins']}",
        "",
        "## Candidate Class Distribution",
        "",
    ]
    for row in class_counts.to_dict(orient="records"):
        md_lines.append(f"- {row['candidate_class']}: {row['count']}")
    md_lines.extend([
        "",
        "## Key Counts",
        "",
        f"- FLASHY_CORE: {summary['flashy_core_count']}",
        f"- FLASHY_MODERATE: {summary['flashy_moderate_count']}",
        f"- FLASHY_POSSIBLE: {summary['flashy_possible_count']}",
        f"- LOW_FLASHINESS_CONTROL: {summary['low_flashiness_control_count']}",
        f"- MANUAL_REVIEW_CONTEXT: {summary['manual_review_count']}",
        f"- EXCLUDE_HARD_QC: {summary['hard_qc_exclusions_count']}",
    ])
    (summaries_dir / "wy2024_streamflow_metrics_summary.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    return summary


def build_diagnostic_plots(output_dir: Path, metrics_df: pd.DataFrame) -> None:
    """Build lightweight aggregate diagnostic plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Candidate class distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    class_counts = metrics_df["candidate_class"].value_counts()
    class_counts.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("Candidate Class Distribution")
    ax.set_ylabel("Count")
    ax.set_xlabel("Candidate Class")
    fig.tight_layout()
    fig.savefig(plots_dir / "candidate_class_counts.png", bbox_inches="tight")
    plt.close(fig)

    # RBI distribution (QC pass only)
    qc_pass = metrics_df[metrics_df["candidate_class"] != "EXCLUDE_HARD_QC"]
    if not qc_pass.empty:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.hist(qc_pass["RBI"].dropna(), bins=30, color="#2ca02c", alpha=0.8)
        ax.set_title("RBI Distribution (QC Pass)")
        ax.set_xlabel("RBI")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(plots_dir / "rbi_distribution_qc_pass.png", bbox_inches="tight")
        plt.close(fig)

    # Max rise per km2
    if "max_hourly_rise_per_km2" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        valid_rise = metrics_df[metrics_df["max_hourly_rise_per_km2"].notna()][
            "max_hourly_rise_per_km2"
        ]
        if not valid_rise.empty:
            ax.hist(valid_rise, bins=30, color="#d62728", alpha=0.8)
            ax.set_title("Max Hourly Rise per km² Distribution")
            ax.set_xlabel("Rise (m3/s per km²)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "max_rise_per_km2_distribution.png", bbox_inches="tight")
            plt.close(fig)

    # Q99/Q50 ratio
    if "q99_q50_ratio" in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        valid_ratio = metrics_df[metrics_df["q99_q50_ratio"].notna()]["q99_q50_ratio"]
        if not valid_ratio.empty:
            ax.hist(valid_ratio[valid_ratio < 50], bins=30, color="#9467bd", alpha=0.8)
            ax.set_title("Q99/Q50 Ratio Distribution (clipped <50)")
            ax.set_xlabel("Q99/Q50")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(plots_dir / "q99_q50_distribution.png", bbox_inches="tight")
            plt.close(fig)


def build_review_bundle(output_dir: Path, metrics_df: pd.DataFrame) -> None:
    """Create review bundle with summary and key tables."""
    bundle_dir = output_dir / "review_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Copy summary files
    summaries_dir = output_dir / "summaries"
    if (summaries_dir / "wy2024_streamflow_metrics_summary.md").exists():
        import shutil

        shutil.copy2(
            summaries_dir / "wy2024_streamflow_metrics_summary.md",
            bundle_dir / "summary.md",
        )
    if (summaries_dir / "wy2024_streamflow_metrics_summary.json").exists():
        import shutil

        shutil.copy2(
            summaries_dir / "wy2024_streamflow_metrics_summary.json",
            bundle_dir / "summary.json",
        )

    # Copy candidate class table
    tables_dir = output_dir / "tables"
    if (tables_dir / "candidate_classes.csv").exists():
        import shutil

        shutil.copy2(
            tables_dir / "candidate_classes.csv",
            bundle_dir / "candidate_classes.csv",
        )

    # Hard QC summary
    hard_qc_df = metrics_df[metrics_df["candidate_class"] == "EXCLUDE_HARD_QC"].copy()
    hard_qc_summary = {
        "count": int(len(hard_qc_df)),
        "basins": hard_qc_df["STAID"].astype(str).tolist()[:100],
    }
    (bundle_dir / "hard_qc_exclusions_summary.json").write_text(
        json.dumps(hard_qc_summary, indent=2), encoding="utf-8"
    )

    # Flashy pool summary
    flashy_df = metrics_df[
        metrics_df["candidate_class"].isin(["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE"])
    ].copy()
    flashy_summary = {
        "total_flashy": int(len(flashy_df)),
        "by_class": {
            "FLASHY_CORE": int(len(flashy_df[flashy_df["candidate_class"] == "FLASHY_CORE"])),
            "FLASHY_MODERATE": int(len(flashy_df[flashy_df["candidate_class"] == "FLASHY_MODERATE"])),
            "FLASHY_POSSIBLE": int(len(flashy_df[flashy_df["candidate_class"] == "FLASHY_POSSIBLE"])),
        },
        "top_rbi_basins": flashy_df.nlargest(10, "RBI")[["STAID", "RBI", "candidate_class"]].to_dict(orient="records"),
    }
    (bundle_dir / "flashy_pool_summary.json").write_text(
        json.dumps(flashy_summary, indent=2, default=str), encoding="utf-8"
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_contents": ["summary.md", "summary.json", "candidate_classes.csv", "hard_qc_exclusions_summary.json", "flashy_pool_summary.json"],
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("Starting WY2024 streamflow metrics matrix workflow")
    logger.info("Arguments: %s", vars(args))

    # Load screening results
    results_path = find_results_file(args.input_results)
    logger.info("Loading screening results from %s", results_path)
    df_all = load_screening_results(results_path)

    # Filter to usable basins
    usable_statuses = ["RBI_READY"]
    if args.include_partial:
        usable_statuses.append("PARTIAL_USABLE")
    df_usable = df_all[df_all["screening_status"].isin(usable_statuses)].copy()
    logger.info(
        "Usable basins (%s): %s", usable_statuses, len(df_usable)
    )

    if df_usable.empty:
        logger.error("No usable basins found; aborting")
        return

    # Apply max basins limit
    if args.max_basins > 0:
        df_usable = df_usable.head(args.max_basins).copy()
    logger.info("Processing up to %s basins", len(df_usable))

    # Load checkpoint if resuming
    checkpoint = {}
    start_index = 0
    if args.resume:
        checkpoint = load_checkpoint(output_dir)
        start_index = checkpoint.get("last_batch_index", -1) + 1
        if start_index > 0:
            logger.info(
                "Resuming from batch index %s (processed=%s, successful=%s, failed=%s)",
                start_index,
                checkpoint.get("total_processed", 0),
                checkpoint.get("total_successful", 0),
                checkpoint.get("total_failed", 0),
            )

    hourly_dir = output_dir / "hourly_streamflow"
    hourly_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Flash-NH streamflow metrics matrix"})

    metrics_rows: list[dict] = []
    all_hard_flags: list[list[str]] = []
    all_context_flags: list[list[str]] = []
    failed_staids: list[str] = []

    # Process in batches
    staids_to_process = df_usable["STAID"].values[start_index:]
    for batch_idx, staid in enumerate(staids_to_process, start=start_index):
        row = df_usable[df_usable["STAID"] == staid].iloc[0]
        logger.info("Processing %s (batch %s/%s)", staid, batch_idx + 1, len(df_usable))

        try:
            hourly, source = load_or_fetch_hourly(
                session=session,
                staid=staid,
                output_hourly_dir=hourly_dir,
                logger=logger,
                sleep_seconds=args.sleep_seconds,
                max_retries=args.max_retries,
                timeout_seconds=args.timeout_seconds,
                no_download=args.no_download,
                force_refresh=args.force_refresh,
            )

            if hourly is None or hourly.dropna().empty:
                logger.warning("Skipping %s due to missing hourly data", staid)
                failed_staids.append(staid)
                continue

            metrics, hard_flags, context_flags = compute_wy2024_metrics(hourly, row)
            candidate_class = classify_candidate_class(metrics, hard_flags, context_flags)

            metrics["candidate_class"] = candidate_class
            metrics["hard_flags"] = json.dumps(hard_flags)
            metrics["context_flags"] = json.dumps(context_flags)
            metrics["hourly_source"] = source

            metrics_rows.append(metrics)
            all_hard_flags.append(hard_flags)
            all_context_flags.append(context_flags)

        except Exception as exc:
            logger.exception("Failed processing %s: %s", staid, exc)
            failed_staids.append(staid)

        # Checkpoint every batch_size basins
        if (batch_idx + 1) % args.batch_size == 0:
            checkpoint["last_batch_index"] = batch_idx
            checkpoint["total_processed"] = batch_idx + 1 - start_index
            checkpoint["total_successful"] = len(metrics_rows)
            checkpoint["total_failed"] = len(failed_staids)
            checkpoint["processed_staids"] = [str(m["STAID"]) for m in metrics_rows]
            save_checkpoint(output_dir, checkpoint, logger)

    # Final checkpoint
    checkpoint["last_batch_index"] = len(df_usable) - 1
    checkpoint["total_processed"] = len(df_usable) - start_index
    checkpoint["total_successful"] = len(metrics_rows)
    checkpoint["total_failed"] = len(failed_staids)
    checkpoint["processed_staids"] = [str(m["STAID"]) for m in metrics_rows]
    save_checkpoint(output_dir, checkpoint, logger)

    logger.info(
        "Processing complete: successful=%s, failed=%s",
        len(metrics_rows),
        len(failed_staids),
    )

    if not metrics_rows:
        logger.error("No metrics were computed; aborting summary generation")
        return

    # Build metrics dataframe
    metrics_df = pd.DataFrame(metrics_rows)

    # Generate summaries and outputs
    build_summary(metrics_df, output_dir, logger)
    build_diagnostic_plots(output_dir, metrics_df)
    build_review_bundle(output_dir, metrics_df)

    print(f"WY2024 streamflow metrics matrix written to {output_dir}")
    print(f"Processed: {len(metrics_rows)}; Failed: {len(failed_staids)}")


if __name__ == "__main__":
    main()
