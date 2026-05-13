#!/usr/bin/env python3
"""Analyze completed WY2024 USGS RBI screening results for Flash-NH.

This is an analysis-only pass over the completed screening outputs. It does not
fetch any new USGS data.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "reports/flashnh_usgs_rbi_screening_wy2024_v001"
DEFAULT_OUTPUT_DIR = ROOT / "reports/flashnh_usgs_rbi_decision_analysis_v001"
EXPECTED_THRESHOLDS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
THRESHOLDS_FOR_QC_MAPS = [0.05, 0.10, 0.20, 0.30]
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed USGS RBI screening results")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Directory containing screening results")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for decision-analysis outputs")
    return parser.parse_args()


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "output_dir": output_dir,
        "plots_dir": output_dir / "plots",
        "tables_dir": output_dir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def find_input_file(input_dir: Path, stem: str) -> Path:
    parquet_path = input_dir / f"{stem}.parquet"
    csv_path = input_dir / f"{stem}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Could not find {parquet_path} or {csv_path}")


def load_results(input_dir: Path) -> pd.DataFrame:
    results_file = find_input_file(input_dir, "usgs_rbi_screening_results")
    if results_file.suffix.lower() == ".parquet":
        frame = pd.read_parquet(results_file)
    else:
        frame = pd.read_csv(results_file, dtype={"STAID": str})

    frame["STAID"] = frame["STAID"].astype(str).str.zfill(8)
    return frame


def load_summary(input_dir: Path) -> dict[str, object]:
    summary_file = input_dir / "usgs_rbi_screening_summary.json"
    if summary_file.exists():
        return json.loads(summary_file.read_text(encoding="utf-8"))
    return {}


def coerce_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def add_qc_flags(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    numeric_columns = [
        "hourly_completeness_pct",
        "rbi",
        "normalized_max_hourly_dqdt",
        "q95_event_count",
        "q99_event_count",
        "DRAIN_SQKM",
        "BFI_AVE",
    ]
    df = coerce_numeric(df, numeric_columns)

    rbi_ready = df[df["screening_status"] == "RBI_READY"].copy()
    normalized_99p = float(rbi_ready["normalized_max_hourly_dqdt"].quantile(0.99)) if not rbi_ready.empty and rbi_ready["normalized_max_hourly_dqdt"].notna().any() else np.nan
    q95_99p = float(rbi_ready["q95_event_count"].quantile(0.99)) if not rbi_ready.empty and rbi_ready["q95_event_count"].notna().any() else np.nan
    q99_99p = float(rbi_ready["q99_event_count"].quantile(0.99)) if not rbi_ready.empty and rbi_ready["q99_event_count"].notna().any() else np.nan

    df["qc_low_completeness"] = df["hourly_completeness_pct"].lt(90)
    df["qc_partial_completeness"] = df["hourly_completeness_pct"].ge(90) & df["hourly_completeness_pct"].lt(95)
    df["qc_good_completeness"] = df["hourly_completeness_pct"].ge(95)

    df["qc_high_rbi"] = df["rbi"].ge(0.30)
    df["qc_extreme_rbi"] = df["rbi"].ge(0.75)
    df["qc_very_extreme_rbi"] = df["rbi"].ge(1.00)

    if math.isnan(normalized_99p):
        df["qc_high_normalized_jump"] = False
    else:
        df["qc_high_normalized_jump"] = df["normalized_max_hourly_dqdt"].gt(normalized_99p)

    if math.isnan(q95_99p):
        df["qc_many_q95_events"] = False
    else:
        df["qc_many_q95_events"] = df["q95_event_count"].gt(q95_99p)

    if math.isnan(q99_99p):
        df["qc_many_q99_events"] = False
    else:
        df["qc_many_q99_events"] = df["q99_event_count"].gt(q99_99p)

    suspicious_flags = [
        "qc_high_rbi",
        "qc_extreme_rbi",
        "qc_very_extreme_rbi",
        "qc_high_normalized_jump",
        "qc_many_q95_events",
        "qc_many_q99_events",
    ]
    df["qc_needs_hydrograph_review"] = df[suspicious_flags].any(axis=1)

    df["candidate_class"] = "EXCLUDE_OTHER"
    df.loc[(df["screening_status"] == "RBI_READY") & df["qc_good_completeness"] & ~df["qc_needs_hydrograph_review"], "candidate_class"] = "RBI_READY_HIGH_CONFIDENCE"
    df.loc[(df["screening_status"] == "RBI_READY") & df["qc_needs_hydrograph_review"], "candidate_class"] = "RBI_READY_REVIEW"
    df.loc[df["screening_status"] == "PARTIAL_USABLE", "candidate_class"] = "PARTIAL_POSSIBLE"
    df.loc[df["screening_status"] == "INSUFFICIENT", "candidate_class"] = "EXCLUDE_INSUFFICIENT"
    df.loc[df["screening_status"] == "NO_DATA", "candidate_class"] = "EXCLUDE_NO_DATA"
    df.loc[df["screening_status"] == "ERROR", "candidate_class"] = "EXCLUDE_ERROR"

    df["threshold_0p02"] = df["rbi"].ge(0.02)
    df["threshold_0p05"] = df["rbi"].ge(0.05)
    df["threshold_0p10"] = df["rbi"].ge(0.10)
    df["threshold_0p20"] = df["rbi"].ge(0.20)
    df["threshold_0p30"] = df["rbi"].ge(0.30)
    df["threshold_0p50"] = df["rbi"].ge(0.50)
    df["threshold_1p00"] = df["rbi"].ge(1.00)

    df.attrs["qc_thresholds"] = {
        "normalized_99p": normalized_99p,
        "q95_count_99p": q95_99p,
        "q99_count_99p": q99_99p,
    }
    return df


def weighted_threshold_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    records = []
    for group_value, group in df.groupby(group_col, dropna=False):
        valid = group["rbi"].notna()
        row = {
            group_col: group_value,
            "total_rows": int(len(group)),
            "rbi_non_null_rows": int(valid.sum()),
            "rbi_ready_rows": int((group["screening_status"] == "RBI_READY").sum()),
        }
        for threshold in EXPECTED_THRESHOLDS:
            row[f"count_ge_{threshold:.2f}"] = int((group["rbi"] >= threshold).fillna(False).sum())
            row[f"pct_ge_{threshold:.2f}_of_total"] = float(100.0 * row[f"count_ge_{threshold:.2f}"] / len(group)) if len(group) else 0.0
        records.append(row)
    return pd.DataFrame(records)


def build_threshold_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = int(len(df))
    non_null = df["rbi"].notna().sum()
    for threshold in EXPECTED_THRESHOLDS:
        count = int((df["rbi"] >= threshold).fillna(False).sum())
        rows.append(
            {
                "threshold": threshold,
                "count": count,
                "pct_of_total": float(100.0 * count / total) if total else 0.0,
                "pct_of_rbi_non_null": float(100.0 * count / non_null) if non_null else 0.0,
            }
        )
    return pd.DataFrame(rows)


def completion_by_status(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for status, group in df.groupby("screening_status", dropna=False):
        completeness = group["hourly_completeness_pct"].dropna()
        rows.append(
            {
                "screening_status": status,
                "count": int(len(group)),
                "completeness_mean": float(completeness.mean()) if not completeness.empty else np.nan,
                "completeness_median": float(completeness.median()) if not completeness.empty else np.nan,
                "completeness_q10": float(completeness.quantile(0.10)) if not completeness.empty else np.nan,
                "completeness_q25": float(completeness.quantile(0.25)) if not completeness.empty else np.nan,
                "completeness_q75": float(completeness.quantile(0.75)) if not completeness.empty else np.nan,
                "completeness_q90": float(completeness.quantile(0.90)) if not completeness.empty else np.nan,
                "completeness_min": float(completeness.min()) if not completeness.empty else np.nan,
                "completeness_max": float(completeness.max()) if not completeness.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("screening_status")


def rbi_stats_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, group in df.groupby(group_col, dropna=False):
        rbi = group["rbi"].dropna()
        rows.append(
            {
                group_col: group_value,
                "count": int(len(group)),
                "rbi_non_null": int(rbi.size),
                "rbi_mean": float(rbi.mean()) if not rbi.empty else np.nan,
                "rbi_median": float(rbi.median()) if not rbi.empty else np.nan,
                "rbi_q10": float(rbi.quantile(0.10)) if not rbi.empty else np.nan,
                "rbi_q25": float(rbi.quantile(0.25)) if not rbi.empty else np.nan,
                "rbi_q75": float(rbi.quantile(0.75)) if not rbi.empty else np.nan,
                "rbi_q90": float(rbi.quantile(0.90)) if not rbi.empty else np.nan,
                "rbi_min": float(rbi.min()) if not rbi.empty else np.nan,
                "rbi_max": float(rbi.max()) if not rbi.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(group_col)


def build_sample_list(df: pd.DataFrame) -> pd.DataFrame:
    selections = []

    def add_selection(frame: pd.DataFrame, reason: str, n: int | None = None, randomize: bool = False) -> None:
        subset = frame.copy()
        if randomize and not subset.empty:
            subset = subset.sample(frac=1.0, random_state=RANDOM_STATE)
        if n is not None:
            subset = subset.head(n)
        if subset.empty:
            return
        temp = subset[["STAID"]].copy()
        temp["reason_for_selection"] = reason
        selections.append(temp)

    add_selection(df[df["rbi"].notna()].sort_values(["rbi", "STAID"], ascending=[False, True]), "top_30_rbi", 30)
    add_selection(df[df["normalized_max_hourly_dqdt"].notna()].sort_values(["normalized_max_hourly_dqdt", "STAID"], ascending=[False, True]), "top_30_normalized_jump", 30)
    add_selection(df[(df["screening_status"] == "RBI_READY") & (df["rbi"] >= 0.10) & (df["rbi"] < 0.30)], "rbi_0p10_to_0p30_random", 30, randomize=True)
    add_selection(df[(df["screening_status"] == "RBI_READY") & (df["rbi"] >= 0.02) & (df["rbi"] < 0.10)], "rbi_0p02_to_0p10_random", 30, randomize=True)
    add_selection(df[(df["screening_status"] == "RBI_READY") & (df["rbi"] < 0.02)], "rbi_below_0p02_random", 30, randomize=True)
    add_selection(df[df["rbi"] >= 1.0], "rbi_ge_1p00_all")
    add_selection(df[df["screening_status"] == "PARTIAL_USABLE"], "partial_usable_examples", 10, randomize=True)
    add_selection(df[df["screening_status"] == "INSUFFICIENT"], "insufficient_examples", 10, randomize=True)

    sample = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(columns=["STAID", "reason_for_selection"])
    if sample.empty:
        return sample

    sample = sample.drop_duplicates(subset=["STAID"], keep="first").copy()
    sample = sample.merge(df, on="STAID", how="left")
    sample = sample.sort_values(["reason_for_selection", "rbi", "STAID"], ascending=[True, False, True], na_position="last").reset_index(drop=True)
    return sample


def save_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def plot_histogram(df: pd.DataFrame, output_path: Path, log_scale: bool = False) -> None:
    rbi = df["rbi"].dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rbi, bins=60, color="#2b8cbe", edgecolor="black", alpha=0.85)
    for threshold in [0.05, 0.10, 0.20, 0.30, 0.50]:
        ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f">= {threshold:.2f}")
    ax.set_xlabel("RBI")
    ax.set_ylabel("Count")
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)")
    ax.set_title("RBI Distribution" + (" (log-scaled y-axis)" if log_scale else ""))
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_box(df: pd.DataFrame, group_col: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped = [g["rbi"].dropna().values for _, g in df.groupby(group_col, sort=False)]
    labels = [str(v) for v, _ in df.groupby(group_col, sort=False)]
    ax.boxplot(grouped, labels=labels, showfliers=True)
    ax.set_xlabel(group_col)
    ax.set_ylabel("RBI")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scatter_bfi(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["rbi"].notna() & df["BFI_AVE"].notna()].copy()
    area_categories = sorted(plot_df["area_bin"].dropna().astype(str).unique().tolist())
    color_map = plt.get_cmap("tab10")
    color_lookup = {cat: color_map(i % 10) for i, cat in enumerate(area_categories)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for category in area_categories:
        subset = plot_df[plot_df["area_bin"].astype(str) == category]
        ax.scatter(subset["BFI_AVE"], subset["rbi"], s=18, alpha=0.75, label=category, color=color_lookup[category], edgecolors="none")
    ax.set_xlabel("BFI_AVE")
    ax.set_ylabel("RBI")
    ax.set_title("RBI vs BFI colored by area_bin")
    ax.legend(title="area_bin", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scatter_area(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["rbi"].notna() & df["DRAIN_SQKM"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(plot_df["DRAIN_SQKM"], plot_df["rbi"], c=plot_df["BFI_AVE"], cmap="viridis", s=18, alpha=0.8, edgecolors="none")
    ax.set_xscale("log")
    ax.set_xlabel("Drainage area (km2, log scale)")
    ax.set_ylabel("RBI")
    ax.set_title("RBI vs drainage area colored by BFI_AVE")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("BFI_AVE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_map_status(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["LNG_GAGE"].notna() & df["LAT_GAGE"].notna()].copy()
    fig, ax = plt.subplots(figsize=(14, 8))
    status_colors = {
        "RBI_READY": "#2ca25f",
        "PARTIAL_USABLE": "#fe9929",
        "INSUFFICIENT": "#de2d26",
        "NO_DATA": "#9e9e9e",
        "ERROR": "#756bb1",
    }
    for status, color in status_colors.items():
        subset = plot_df[plot_df["screening_status"] == status]
        ax.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], s=14, alpha=0.75, color=color, label=f"{status} ({len(subset)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Map of RBI screening status")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_map_continuous(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["LNG_GAGE"].notna() & df["LAT_GAGE"].notna()].copy()
    valid = plot_df[plot_df["rbi"].notna()].copy()
    if valid.empty:
        return
    cap = float(valid["rbi"].quantile(0.99))
    outliers = valid[valid["rbi"] > cap]
    normal = valid[valid["rbi"] <= cap]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(plot_df["LNG_GAGE"], plot_df["LAT_GAGE"], c="#d9d9d9", s=12, alpha=0.35, label="No RBI")
    base = ax.scatter(normal["LNG_GAGE"], normal["LAT_GAGE"], c=normal["rbi"], cmap="magma", vmin=0, vmax=cap, s=16, alpha=0.85, edgecolors="none", label="RBI")
    if not outliers.empty:
        ax.scatter(outliers["LNG_GAGE"], outliers["LAT_GAGE"], marker="x", s=38, linewidths=1.2, color="black", label=f"Outliers > 99th pct ({len(outliers)})")
    cbar = fig.colorbar(base, ax=ax)
    cbar.set_label(f"RBI capped at 99th percentile ({cap:.3f})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Map of RBI (capped at 99th percentile, outliers marked)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_candidate_classes(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["LNG_GAGE"].notna() & df["LAT_GAGE"].notna()].copy()
    classes = [
        "RBI_READY_HIGH_CONFIDENCE",
        "RBI_READY_REVIEW",
        "PARTIAL_POSSIBLE",
        "EXCLUDE_INSUFFICIENT",
        "EXCLUDE_NO_DATA",
        "EXCLUDE_ERROR",
        "EXCLUDE_OTHER",
    ]
    colors = {
        "RBI_READY_HIGH_CONFIDENCE": "#2ca25f",
        "RBI_READY_REVIEW": "#f03b20",
        "PARTIAL_POSSIBLE": "#feb24c",
        "EXCLUDE_INSUFFICIENT": "#de2d26",
        "EXCLUDE_NO_DATA": "#9e9e9e",
        "EXCLUDE_ERROR": "#756bb1",
        "EXCLUDE_OTHER": "#636363",
    }
    fig, ax = plt.subplots(figsize=(14, 8))
    for klass in classes:
        subset = plot_df[plot_df["candidate_class"] == klass]
        if subset.empty:
            continue
        ax.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], s=14, alpha=0.8, color=colors[klass], label=f"{klass} ({len(subset)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Map of candidate classes")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_threshold_facets(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df[df["LNG_GAGE"].notna() & df["LAT_GAGE"].notna()].copy()
    if plot_df.empty:
        return

    for threshold in THRESHOLDS_FOR_QC_MAPS:
        fig, ax = plt.subplots(figsize=(14, 8))
        meets = plot_df[plot_df["rbi"].ge(threshold)]
        below = plot_df[~plot_df.index.isin(meets.index)]
        ax.scatter(below["LNG_GAGE"], below["LAT_GAGE"], c="#d9d9d9", s=11, alpha=0.35, label=f"RBI < {threshold:.2f}")
        ax.scatter(meets["LNG_GAGE"], meets["LAT_GAGE"], c="#2ca25f", s=16, alpha=0.85, label=f"RBI >= {threshold:.2f}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"RBI threshold map: >= {threshold:.2f}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"map_rbi_threshold_ge_{str(threshold).replace('.', 'p')}.png", dpi=160)
        plt.close(fig)


def plot_completeness_map(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["LNG_GAGE"].notna() & df["LAT_GAGE"].notna()].copy()
    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = ax.scatter(plot_df["LNG_GAGE"], plot_df["LAT_GAGE"], c=plot_df["hourly_completeness_pct"], cmap="viridis", s=14, alpha=0.85, edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Hourly completeness (%)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Hourly completeness map")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_dropout_map(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["LNG_GAGE"].notna() & df["LAT_GAGE"].notna()].copy()
    fig, ax = plt.subplots(figsize=(14, 8))
    categories = {
        "RBI_READY": "#2ca25f",
        "PARTIAL_USABLE": "#feb24c",
        "INSUFFICIENT": "#de2d26",
        "NO_DATA": "#9e9e9e",
        "ERROR": "#756bb1",
    }
    for status, color in categories.items():
        subset = plot_df[plot_df["screening_status"] == status]
        ax.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], s=14, alpha=0.8, color=color, label=f"{status} ({len(subset)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Dropout/status map")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_summary(df: pd.DataFrame, screening_summary: dict[str, object], qc_thresholds: dict[str, float]) -> dict[str, object]:
    total = int(len(df))
    status_counts = df["screening_status"].value_counts().to_dict()
    rbi_ready = int((df["screening_status"] == "RBI_READY").sum())
    rbi_ready_pct = float(100.0 * rbi_ready / total) if total else 0.0
    candidate_counts = df["candidate_class"].value_counts().to_dict()
    qc_needs_review = int(df["qc_needs_hydrograph_review"].sum())
    usable = int((df["screening_status"] == "RBI_READY").sum() + (df["screening_status"] == "PARTIAL_USABLE").sum())
    rbi_non_null = df["rbi"].notna().sum()
    rbi_stats = df["rbi"].dropna()

    thresholds = {}
    for threshold in EXPECTED_THRESHOLDS:
        count = int((df["rbi"] >= threshold).fillna(False).sum())
        thresholds[f"ge_{threshold:.2f}"] = {
            "count": count,
            "pct_of_total": float(100.0 * count / total) if total else 0.0,
            "pct_of_rbi_non_null": float(100.0 * count / rbi_non_null) if rbi_non_null else 0.0,
        }

    analysis = {
        "source_summary": {
            "candidate_universe_total_eligible_screening_wy": screening_summary.get("candidate_universe_total_eligible_screening_wy"),
            "attempted_basins": screening_summary.get("attempted_basins"),
            "unique_sites_stored_total": screening_summary.get("unique_sites_stored_total"),
            "status_counts": screening_summary.get("status_counts", {}),
            "median_hourly_completeness_pct": screening_summary.get("median_hourly_completeness_pct"),
            "median_rbi_among_rbi_ready": screening_summary.get("median_rbi_among_rbi_ready"),
        },
        "basin_counts": {
            "total_rows": total,
            "usable_rbi_ready": rbi_ready,
            "usable_rbi_ready_pct": rbi_ready_pct,
            "usable_rbi_ready_or_partial": usable,
            "candidate_class_counts": {k: int(v) for k, v in candidate_counts.items()},
            "qc_needs_hydrograph_review": qc_needs_review,
        },
        "rbi_threshold_counts": thresholds,
        "qc_thresholds": qc_thresholds,
        "rbi_descriptives": {
            "overall": {
                "count": int(rbi_non_null),
                "mean": float(rbi_stats.mean()) if not rbi_stats.empty else None,
                "median": float(rbi_stats.median()) if not rbi_stats.empty else None,
                "q10": float(rbi_stats.quantile(0.10)) if not rbi_stats.empty else None,
                "q25": float(rbi_stats.quantile(0.25)) if not rbi_stats.empty else None,
                "q75": float(rbi_stats.quantile(0.75)) if not rbi_stats.empty else None,
                "q90": float(rbi_stats.quantile(0.90)) if not rbi_stats.empty else None,
                "max": float(rbi_stats.max()) if not rbi_stats.empty else None,
            }
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return analysis


def write_markdown_report(output_dir: Path, analysis: dict[str, object], tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Flash-NH USGS RBI Decision Analysis",
        "",
        "## What Was Analyzed",
        "",
        "This analysis pass reads the completed WY2024 RBI screening results only. No new USGS discharge data were requested.",
        "",
        f"- Candidate universe: {analysis['source_summary']['candidate_universe_total_eligible_screening_wy']}",
        f"- Completed result rows: {analysis['basin_counts']['total_rows']}",
        f"- RBI_READY basins: {analysis['basin_counts']['usable_rbi_ready']} ({analysis['basin_counts']['usable_rbi_ready_pct']:.1f}%)",
        f"- RBI_READY + PARTIAL_USABLE basins: {analysis['basin_counts']['usable_rbi_ready_or_partial']}",
        f"- Basins flagged for hydrograph review: {analysis['basin_counts']['qc_needs_hydrograph_review']}",
        "",
        "## How RBI Relates to Drainage Area and BFI",
        "",
        "RBI is summarized by drainage-area and BFI strata to see whether flashiness concentrates in smaller basins, high-BFI basins, or specific geographic clusters.",
        "The output tables report RBI counts above several thresholds and descriptive statistics for area_bin, BFI_bin, and HUC02.",
        "",
        "## Why High-RBI Outliers Need Hydrograph QC",
        "",
        "Very high RBI values can reflect meaningful flashiness, but they can also be caused by sparse observations, short active periods, or unusual gaps in the hourly record.",
        "The QC flags therefore separate completeness issues from high-RBI and high-jump conditions so outliers can be checked with hydrographs before threshold decisions are made.",
        "",
        "## Candidate Threshold Options",
        "",
        "Threshold counts are reported for RBI >= 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, and 1.00.",
        "Lower thresholds retain more basins for pilot planning, while higher thresholds focus on more extreme flashiness but increase the chance of selecting hydrograph artifacts or sparse-record basins.",
        "",
        "## Recommended Next Step",
        "",
        "Use the hydrograph QC sample list to do targeted hydrograph review first.",
        "Do not move straight to full pilot selection until the high-RBI outliers and QC-flagged basins have been reviewed in the hydrographs.",
        "",
        "## Key Metrics",
        "",
        f"- RBI_READY count: {analysis['basin_counts']['usable_rbi_ready']}",
        f"- RBI_READY percentage of completed results: {analysis['basin_counts']['usable_rbi_ready_pct']:.1f}%",
        f"- Median RBI among all non-null RBI values: {analysis['rbi_descriptives']['overall']['median']}",
        f"- 99th percentile normalized_max_hourly_dqdt among RBI_READY basins: {analysis['qc_thresholds']['normalized_99p']}",
        f"- 99th percentile q95_event_count among RBI_READY basins: {analysis['qc_thresholds']['q95_count_99p']}",
        f"- 99th percentile q99_event_count among RBI_READY basins: {analysis['qc_thresholds']['q99_count_99p']}",
        "",
        "## Files",
        "",
        "- decision_analysis_results.csv / .parquet",
        "- summary tables under tables/",
        "- plots under plots/",
        "- hydrograph_qc_sample_list.csv",
        "",
        "## Summary Tables Preview",
        "",
        f"- Status counts rows: {len(tables['status_counts'])}",
        f"- Threshold rows: {len(tables['threshold_counts'])}",
        f"- Area-bin rows: {len(tables['area_threshold_counts'])}",
        f"- BFI-bin rows: {len(tables['bfi_threshold_counts'])}",
        f"- HUC02 rows: {len(tables['huc_threshold_counts'])}",
        f"- STATE rows: {len(tables['state_threshold_counts'])}",
    ]
    (output_dir / "usgs_rbi_decision_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(output_dir: Path, analysis: dict[str, object], tables: dict[str, pd.DataFrame]) -> None:
    json_summary = {
        **analysis,
        "table_shapes": {name: list(frame.shape) for name, frame in tables.items()},
    }
    (output_dir / "usgs_rbi_decision_analysis_summary.json").write_text(json.dumps(json_summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paths = make_output_dirs(output_dir)

    results = load_results(input_dir)
    screening_summary = load_summary(input_dir)
    results = add_qc_flags(results)

    # QC thresholds need to be stored before any reporting.
    qc_thresholds = results.attrs.get("qc_thresholds", {})

    # Save analysis table with QC flags.
    analyzed_results_path = output_dir / "decision_analysis_results.csv"
    analyzed_results_parquet = output_dir / "decision_analysis_results.parquet"
    results.to_csv(analyzed_results_path, index=False)
    results.to_parquet(analyzed_results_parquet, index=False)

    # Summary tables.
    status_counts = results["screening_status"].value_counts().reset_index()
    status_counts.columns = ["screening_status", "count"]
    status_counts["pct_of_total"] = 100.0 * status_counts["count"] / len(results) if len(results) else 0.0

    threshold_counts = build_threshold_counts(results)
    area_threshold_counts = weighted_threshold_table(results, "area_bin")
    bfi_threshold_counts = weighted_threshold_table(results, "BFI_bin")
    huc_threshold_counts = weighted_threshold_table(results, "HUC02")
    state_threshold_counts = weighted_threshold_table(results, "STATE")
    completeness_by_status = completion_by_status(results)
    rbi_by_area = rbi_stats_by_group(results, "area_bin")
    rbi_by_bfi = rbi_stats_by_group(results, "BFI_bin")
    rbi_by_huc = rbi_stats_by_group(results, "HUC02")

    tables = {
        "status_counts": status_counts,
        "threshold_counts": threshold_counts,
        "area_threshold_counts": area_threshold_counts,
        "bfi_threshold_counts": bfi_threshold_counts,
        "huc_threshold_counts": huc_threshold_counts,
        "state_threshold_counts": state_threshold_counts,
        "completeness_by_status": completeness_by_status,
        "rbi_by_area": rbi_by_area,
        "rbi_by_bfi": rbi_by_bfi,
        "rbi_by_huc": rbi_by_huc,
    }

    for name, frame in tables.items():
        save_table(frame, paths["tables_dir"] / f"{name}.csv")

    # Candidate QC flags and planning table.
    hydrograph_qc_sample = build_sample_list(results)
    hydrograph_qc_sample.to_csv(output_dir / "hydrograph_qc_sample_list.csv", index=False)

    # Add a compact QC planning subset for later hydrograph review if needed.
    if not hydrograph_qc_sample.empty:
        hydrograph_qc_sample.to_csv(paths["tables_dir"] / "hydrograph_qc_sample_list.csv", index=False)

    # Plots.
    plot_histogram(results, paths["plots_dir"] / "rbi_histogram.png", log_scale=False)
    plot_histogram(results, paths["plots_dir"] / "rbi_histogram_logy.png", log_scale=True)
    plot_box(results, "area_bin", paths["plots_dir"] / "rbi_by_area_bin_boxplot.png", "RBI by area_bin")
    plot_box(results, "BFI_bin", paths["plots_dir"] / "rbi_by_bfi_bin_boxplot.png", "RBI by BFI_bin")
    plot_scatter_bfi(results, paths["plots_dir"] / "rbi_vs_bfi_scatter.png")
    plot_scatter_area(results, paths["plots_dir"] / "rbi_vs_drainage_area_scatter.png")
    plot_map_continuous(results, paths["plots_dir"] / "map_rbi_continuous_capped.png")
    plot_candidate_classes(results, paths["plots_dir"] / "map_candidate_classes.png")
    plot_threshold_facets(results, paths["plots_dir"])
    plot_completeness_map(results, paths["plots_dir"] / "map_completeness.png")
    plot_dropout_map(results, paths["plots_dir"] / "map_dropout_status.png")

    analysis = build_summary(results, screening_summary, qc_thresholds)
    write_markdown_report(output_dir, analysis, tables)
    write_json_summary(output_dir, analysis, tables)

    # Small run manifest for traceability.
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [
            "decision_analysis_results.csv",
            "decision_analysis_results.parquet",
            "usgs_rbi_decision_analysis_summary.md",
            "usgs_rbi_decision_analysis_summary.json",
            "hydrograph_qc_sample_list.csv",
            "tables/*.csv",
            "plots/*.png",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote analysis outputs to {output_dir}")
    print(f"Completed results rows: {len(results)}")
    print(f"RBI_READY: {int((results['screening_status'] == 'RBI_READY').sum())}")
    print(f"PARTIAL_USABLE: {int((results['screening_status'] == 'PARTIAL_USABLE').sum())}")
    print(f"INSUFFICIENT: {int((results['screening_status'] == 'INSUFFICIENT').sum())}")
    print(f"NO_DATA: {int((results['screening_status'] == 'NO_DATA').sum())}")
    print(f"ERROR: {int((results['screening_status'] == 'ERROR').sum())}")


if __name__ == "__main__":
    main()
