#!/usr/bin/env python3
"""Analyze WY2024 streamflow metrics and design pilot basin selection.

This script postprocesses completed WY2024 metrics output to:
1. Summarize the usable basin universe
2. Create threshold tables for RBI and event metrics
3. Build interpretable pilot-selection scores
4. Design stratified pilot basin lists
5. Generate maps and diagnostic plots
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_DIR = ROOT / "reports" / "flashnh_wy2024_streamflow_metrics_v002"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "flashnh_wy2024_pilot_selection_v001"


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "run.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load full metrics table."""
    metrics_csv = metrics_dir / "tables" / "wy2024_streamflow_metrics.csv"
    df = pd.read_csv(metrics_csv)
    
    # Parse JSON flag strings
    df["hard_flags"] = df["hard_flags"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x or [])
    )
    df["context_flags"] = df["context_flags"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x or [])
    )
    
    return df


def summarize_universe(df: pd.DataFrame, logger: logging.Logger) -> dict:
    """Summarize the usable basin universe."""
    summary = {
        "total_basins": len(df),
        "by_candidate_class": df["candidate_class"].value_counts().to_dict(),
        "by_area_bin": df["area_bin"].value_counts().to_dict() if "area_bin" in df else {},
        "by_bfi_bin": df["BFI_bin"].value_counts().to_dict() if "BFI_bin" in df else {},
        "by_state": df["STATE"].value_counts().to_dict() if "STATE" in df else {},
        "by_huc02": df["HUC02"].value_counts().to_dict() if "HUC02" in df else {},
        "completeness_pct": {
            "min": float(df["hourly_completeness_pct"].min()),
            "median": float(df["hourly_completeness_pct"].median()),
            "mean": float(df["hourly_completeness_pct"].mean()),
            "max": float(df["hourly_completeness_pct"].max()),
        },
        "rbi_overall": {
            "min": float(df["RBI"].min()),
            "p05": float(df["RBI"].quantile(0.05)),
            "p25": float(df["RBI"].quantile(0.25)),
            "median": float(df["RBI"].median()),
            "mean": float(df["RBI"].mean()),
            "p75": float(df["RBI"].quantile(0.75)),
            "p95": float(df["RBI"].quantile(0.95)),
            "max": float(df["RBI"].max()),
        },
    }
    
    # RBI by class
    for cls in df["candidate_class"].unique():
        if pd.notna(cls):
            subset = df[df["candidate_class"] == cls]["RBI"]
            summary[f"rbi_{cls}"] = {
                "count": len(subset),
                "median": float(subset.median()),
                "mean": float(subset.mean()),
                "min": float(subset.min()),
                "max": float(subset.max()),
            }
    
    # Event metrics distributions
    event_metrics = [
        "max_hourly_rise_per_km2",
        "max_abs_hourly_jump_over_Q50",
        "q95_q50_ratio",
        "q99_q50_ratio",
        "event_count_q95",
        "event_count_q99",
        "zero_flow_fraction",
    ]
    
    for metric in event_metrics:
        if metric in df.columns:
            valid = df[metric].dropna()
            summary[f"distribution_{metric}"] = {
                "count": len(valid),
                "min": float(valid.min()),
                "p05": float(valid.quantile(0.05)),
                "p25": float(valid.quantile(0.25)),
                "median": float(valid.median()),
                "mean": float(valid.mean()),
                "p75": float(valid.quantile(0.75)),
                "p95": float(valid.quantile(0.95)),
                "max": float(valid.max()),
            }
    
    logger.info("Universe summary computed: %d total basins", summary["total_basins"])
    return summary


def build_threshold_tables(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Create threshold tables for RBI and event metrics."""
    rbi_thresholds = [0.03, 0.05, 0.075, 0.10, 0.20, 0.30, 0.50]
    
    results = []
    
    for threshold in rbi_thresholds:
        count_overall = len(df[df["RBI"] >= threshold])
        
        # By area bin
        area_counts = {}
        if "area_bin" in df.columns:
            for area_bin in df["area_bin"].dropna().unique():
                subset = df[(df["area_bin"] == area_bin) & (df["RBI"] >= threshold)]
                area_counts[str(area_bin)] = len(subset)
        
        # By BFI bin
        bfi_counts = {}
        if "BFI_bin" in df.columns:
            for bfi_bin in df["BFI_bin"].dropna().unique():
                subset = df[(df["BFI_bin"] == bfi_bin) & (df["RBI"] >= threshold)]
                bfi_counts[str(bfi_bin)] = len(subset)
        
        results.append({
            "threshold": threshold,
            "count_overall": count_overall,
            "area_counts": json.dumps(area_counts),
            "bfi_counts": json.dumps(bfi_counts),
        })
    
    threshold_df = pd.DataFrame(results)
    logger.info("Threshold tables created: %d RBI thresholds", len(threshold_df))
    return threshold_df


def build_selection_scores(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Build interpretable pilot selection scores."""
    df = df.copy()
    
    # Class score: 3 (CORE) > 2 (MODERATE) > 1 (POSSIBLE) > 0 (other)
    class_scores = {
        "FLASHY_CORE": 3.0,
        "FLASHY_MODERATE": 2.0,
        "FLASHY_POSSIBLE": 1.0,
        "LOW_FLASHINESS_CONTROL": 0.5,
        "MANUAL_REVIEW_CONTEXT": 0.2,
        "EXCLUDE_HARD_QC": 0.0,
    }
    df["class_score"] = df["candidate_class"].map(
        lambda x: class_scores.get(x, 0.0)
    )
    
    # Normalize RBI to 0-1 scale
    rbi_min = df["RBI"].min()
    rbi_max = df["RBI"].max()
    if rbi_max > rbi_min:
        df["rbi_score"] = (df["RBI"] - rbi_min) / (rbi_max - rbi_min)
    else:
        df["rbi_score"] = 0.5
    
    # Rise intensity score
    rise_max = df["max_hourly_rise_per_km2"].max()
    if rise_max > 0:
        df["rise_score"] = df["max_hourly_rise_per_km2"] / rise_max
    else:
        df["rise_score"] = 0.0
    
    # Event frequency score
    event_max = df["event_count_q99"].max()
    if event_max > 0:
        df["event_score"] = df["event_count_q99"] / event_max
    else:
        df["event_score"] = 0.0
    
    # Completeness score
    df["completeness_score"] = df["hourly_completeness_pct"] / 100.0
    
    # Zero-flow penalty (lower is better)
    df["zero_flow_penalty"] = df["zero_flow_fraction"]
    
    # Context flag penalty
    df["context_flag_count"] = df["context_flags"].apply(len)
    
    # Combined pilot selection score (weights calibrated for interpretation)
    # Emphasize class and RBI, then rise and events, with penalties for context flags
    df["pilot_score"] = (
        0.35 * df["class_score"] +
        0.30 * df["rbi_score"] +
        0.15 * df["rise_score"] +
        0.10 * df["event_score"] +
        0.05 * df["completeness_score"] -
        0.10 * df["zero_flow_penalty"] -
        0.05 * (df["context_flag_count"] / (df["context_flag_count"].max() + 1))
    )
    
    # Ensure non-negative
    df["pilot_score"] = df["pilot_score"].clip(lower=0.0)
    
    logger.info("Selection scores computed for %d basins", len(df))
    return df


def select_pilot_basins(
    df: pd.DataFrame,
    target_count: int,
    stratify_by: list[str] = None,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """Select stratified pilot basins."""
    if stratify_by is None:
        stratify_by = ["area_bin", "BFI_bin"]
    
    # Filter valid basins (exclude hard QC)
    valid = df[df["candidate_class"] != "EXCLUDE_HARD_QC"].copy()
    
    if len(valid) < target_count:
        logger.warning("Only %d valid basins available; requested %d", len(valid), target_count)
        return valid.nlargest(len(valid), "pilot_score")
    
    # Stratified selection
    selected = []
    remaining = valid.copy()
    
    # Define stratification groups
    strata = []
    if "area_bin" in valid.columns:
        area_bins = valid["area_bin"].dropna().unique()
        bfi_bins = valid["BFI_bin"].dropna().unique() if "BFI_bin" in valid.columns else [None]
        for area_bin in area_bins:
            for bfi_bin in bfi_bins:
                if bfi_bin is None:
                    subset = valid[valid["area_bin"] == area_bin]
                else:
                    subset = valid[(valid["area_bin"] == area_bin) & (valid["BFI_bin"] == bfi_bin)]
                if len(subset) > 0:
                    strata.append(subset)
    else:
        strata = [valid]
    
    # Proportional allocation from each stratum
    for stratum in strata:
        allocation = int(np.ceil(target_count * len(stratum) / len(valid)))
        selected_from_stratum = stratum.nlargest(min(allocation, len(stratum)), "pilot_score")
        selected.append(selected_from_stratum)
        remaining = remaining.drop(selected_from_stratum.index)
        
        if len(pd.concat(selected)) >= target_count:
            break
    
    # If still short, fill with top-scored remaining basins
    if len(pd.concat(selected)) < target_count:
        needed = target_count - len(pd.concat(selected))
        final_fill = remaining.nlargest(min(needed, len(remaining)), "pilot_score")
        selected.append(final_fill)
    
    result = pd.concat(selected).drop_duplicates(subset=["STAID"]).nlargest(target_count, "pilot_score")
    
    if logger:
        logger.info("Selected %d pilot basins (target %d)", len(result), target_count)
    
    return result


def create_pilot_lists(
    df_scored: pd.DataFrame, output_dir: Path, logger: logging.Logger
) -> None:
    """Create proposed pilot basin lists."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pilot 50 balanced
    pilot_50 = select_pilot_basins(df_scored, 50, logger=logger)
    pilot_50.to_csv(tables_dir / "pilot_50_balanced.csv", index=False)
    logger.info("Created pilot_50_balanced.csv: %d basins", len(pilot_50))
    
    # 2. Pilot 100 balanced
    pilot_100 = select_pilot_basins(df_scored, 100, logger=logger)
    pilot_100.to_csv(tables_dir / "pilot_100_balanced.csv", index=False)
    logger.info("Created pilot_100_balanced.csv: %d basins", len(pilot_100))
    
    # 3. Pilot 150 broad
    pilot_150 = select_pilot_basins(df_scored, 150, logger=logger)
    pilot_150.to_csv(tables_dir / "pilot_150_broad.csv", index=False)
    logger.info("Created pilot_150_broad.csv: %d basins", len(pilot_150))
    
    # 4. Manual review priority (high context flags, lower scores)
    manual_review_priority = df_scored[
        (df_scored["candidate_class"] == "MANUAL_REVIEW_CONTEXT") |
        (df_scored["context_flag_count"] >= 2)
    ].sort_values("pilot_score", ascending=False).head(100)
    manual_review_priority.to_csv(tables_dir / "manual_review_priority.csv", index=False)
    logger.info("Created manual_review_priority.csv: %d basins", len(manual_review_priority))
    
    # 5. Flashy extreme review (top flashy by RBI, check for outliers)
    flashy_all = df_scored[
        df_scored["candidate_class"].isin(["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE"])
    ]
    flashy_extreme = flashy_all.nlargest(50, "RBI")
    flashy_extreme.to_csv(tables_dir / "flashy_extreme_review.csv", index=False)
    logger.info("Created flashy_extreme_review.csv: %d basins", len(flashy_extreme))
    
    # 6. Possible + low flashiness controls (for comparison)
    possible_controls = df_scored[
        df_scored["candidate_class"].isin(["FLASHY_POSSIBLE", "LOW_FLASHINESS_CONTROL"])
    ].sort_values("pilot_score", ascending=False).head(50)
    possible_controls.to_csv(tables_dir / "possible_low_flashiness_controls.csv", index=False)
    logger.info("Created possible_low_flashiness_controls.csv: %d basins", len(possible_controls))


def build_plots(
    df_scored: pd.DataFrame, output_dir: Path, logger: logging.Logger
) -> None:
    """Create diagnostic plots and maps."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Candidate class distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    class_counts = df_scored["candidate_class"].value_counts()
    class_counts.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("Candidate Class Distribution (All Basins)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Candidate Class")
    plt.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=150)
    plt.close()
    logger.info("Created class_distribution.png")
    
    # 2. RBI by area bin
    if "area_bin" in df_scored.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        df_scored.boxplot(column="RBI", by="area_bin", ax=ax)
        ax.set_title("RBI by Area Bin")
        ax.set_ylabel("RBI")
        ax.set_xlabel("Area Bin")
        plt.tight_layout()
        fig.savefig(plots_dir / "rbi_by_area_bin.png", dpi=150)
        plt.close()
        logger.info("Created rbi_by_area_bin.png")
    
    # 3. RBI by BFI bin
    if "BFI_bin" in df_scored.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        df_scored.boxplot(column="RBI", by="BFI_bin", ax=ax)
        ax.set_title("RBI by Base Flow Index Bin")
        ax.set_ylabel("RBI")
        ax.set_xlabel("BFI Bin")
        plt.tight_layout()
        fig.savefig(plots_dir / "rbi_by_bfi_bin.png", dpi=150)
        plt.close()
        logger.info("Created rbi_by_bfi_bin.png")
    
    # 4. Pilot score distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(df_scored["pilot_score"].dropna(), bins=50, color="#1f77b4", edgecolor="black")
    ax.set_title("Pilot Selection Score Distribution")
    ax.set_xlabel("Pilot Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(plots_dir / "pilot_score_distribution.png", dpi=150)
    plt.close()
    logger.info("Created pilot_score_distribution.png")
    
    # 5. RBI distribution by class
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for cls in ["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE", "LOW_FLASHINESS_CONTROL"]:
        subset = df_scored[df_scored["candidate_class"] == cls]["RBI"].dropna()
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.5, label=cls)
    ax.set_title("RBI Distribution by Candidate Class")
    ax.set_xlabel("RBI")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plots_dir / "rbi_by_class.png", dpi=150)
    plt.close()
    logger.info("Created rbi_by_class.png")
    
    # 6. Max rise per km2 distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(df_scored["max_hourly_rise_per_km2"].dropna(), bins=50, color="#1f77b4", edgecolor="black")
    ax.set_title("Max Hourly Rise per km² Distribution")
    ax.set_xlabel("Max Rise (m³/s/km²)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(plots_dir / "max_rise_per_km2_distribution.png", dpi=150)
    plt.close()
    logger.info("Created max_rise_per_km2_distribution.png")
    
    # 7. Q95/Q50 ratio distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(df_scored["q95_q50_ratio"].dropna(), bins=50, color="#1f77b4", edgecolor="black")
    ax.set_title("Q95/Q50 Ratio Distribution")
    ax.set_xlabel("Q95/Q50")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(plots_dir / "q95_q50_distribution.png", dpi=150)
    plt.close()
    logger.info("Created q95_q50_distribution.png")
    
    # 8. Zero-flow fraction by class
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    df_scored.boxplot(column="zero_flow_fraction", by="candidate_class", ax=ax)
    ax.set_title("Zero-Flow Fraction by Candidate Class")
    ax.set_ylabel("Zero-Flow Fraction")
    ax.set_xlabel("Candidate Class")
    plt.tight_layout()
    fig.savefig(plots_dir / "zero_flow_by_class.png", dpi=150)
    plt.close()
    logger.info("Created zero_flow_by_class.png")
    
    # 9. Geographic scatter (if coordinates available)
    if "LAT_GAGE" in df_scored.columns and "LNG_GAGE" in df_scored.columns:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
        for cls in ["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE", "LOW_FLASHINESS_CONTROL", "MANUAL_REVIEW_CONTEXT"]:
            subset = df_scored[df_scored["candidate_class"] == cls]
            if len(subset) > 0:
                ax.scatter(subset["LNG_GAGE"], subset["LAT_GAGE"], label=cls, s=30, alpha=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Candidate Classes by Geography")
        ax.legend()
        plt.tight_layout()
        fig.savefig(plots_dir / "map_candidate_classes.png", dpi=150)
        plt.close()
        logger.info("Created map_candidate_classes.png")
    
    # 10. RBI continuous map
    if "LAT_GAGE" in df_scored.columns and "LNG_GAGE" in df_scored.columns:
        fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
        scatter = ax.scatter(
            df_scored["LNG_GAGE"],
            df_scored["LAT_GAGE"],
            c=df_scored["RBI"],
            cmap="viridis",
            s=30,
            alpha=0.6,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("RBI Distribution by Geography")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("RBI")
        plt.tight_layout()
        fig.savefig(plots_dir / "map_rbi_continuous.png", dpi=150)
        plt.close()
        logger.info("Created map_rbi_continuous.png")


def build_summaries(
    df_scored: pd.DataFrame,
    universe_summary: dict,
    output_dir: Path,
    metrics_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write comprehensive summaries."""
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original summary
    with open(metrics_dir / "summaries" / "wy2024_streamflow_metrics_summary.json") as f:
        original_summary = json.load(f)
    
    # Count usable basins
    usable = df_scored[df_scored["candidate_class"] != "EXCLUDE_HARD_QC"]
    
    # Pilot-100 composition
    pilot_100 = select_pilot_basins(df_scored, 100, logger=logger)
    pilot_100_composition = {
        "total": len(pilot_100),
        "by_class": pilot_100["candidate_class"].value_counts().to_dict(),
        "median_rbi": float(pilot_100["RBI"].median()),
        "median_completeness": float(pilot_100["hourly_completeness_pct"].median()),
    }
    if "area_bin" in pilot_100.columns:
        pilot_100_composition["by_area_bin"] = pilot_100["area_bin"].value_counts().to_dict()
    if "BFI_bin" in pilot_100.columns:
        pilot_100_composition["by_bfi_bin"] = pilot_100["BFI_bin"].value_counts().to_dict()
    
    summary_json = {
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "metrics_source": str(metrics_dir),
        "original_summary": original_summary,
        "usable_basins": {
            "total": len(usable),
            "by_class": usable["candidate_class"].value_counts().to_dict(),
        },
        "universe_summary": universe_summary,
        "pilot_100_composition": pilot_100_composition,
        "readiness": {
            "hard_qc_pass": len(usable),
            "hard_qc_pass_percentage": 100.0 * len(usable) / len(df_scored),
            "recommended_for_stage1": len(pilot_100) > 0,
        },
    }
    
    with open(summaries_dir / "pilot_selection_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    logger.info("Created pilot_selection_summary.json")
    
    # Markdown summary
    md_lines = [
        "# WY2024 Pilot Basin Selection Summary",
        "",
        f"Analysis Date: {summary_json['analysis_date']}",
        "",
        "## Usable Basin Universe",
        "",
        f"Total basins after hard QC: **{len(usable)} / {len(df_scored)} ({100.0*len(usable)/len(df_scored):.1f}%)**",
        "",
        "### By Candidate Class",
        "",
    ]
    
    for cls, count in usable["candidate_class"].value_counts().items():
        pct = 100.0 * count / len(usable)
        md_lines.append(f"- {cls}: {count} ({pct:.1f}%)")
    
    md_lines.extend([
        "",
        "## Pilot-100 Recommended Composition",
        "",
        f"Total: {pilot_100_composition['total']}",
        f"Median RBI: {pilot_100_composition['median_rbi']:.3f}",
        f"Median Completeness: {pilot_100_composition['median_completeness']:.1f}%",
        "",
        "### By Candidate Class",
        "",
    ])
    
    for cls, count in sorted(pilot_100_composition["by_class"].items()):
        md_lines.append(f"- {cls}: {count}")
    
    md_lines.extend([
        "",
        "## Readiness for Stage 1",
        "",
        f"Hard QC pass rate: {summary_json['readiness']['hard_qc_pass_percentage']:.1f}%",
        f"Recommended for Stage 1 meteorological preprocessing: **YES** ({len(pilot_100)} pilot basins identified)",
        "",
        "## Next Steps",
        "",
        "1. Review manual_review_priority.csv ({} basins with high context flags)".format(
            len(df_scored[
                (df_scored["candidate_class"] == "MANUAL_REVIEW_CONTEXT") |
                (df_scored["context_flag_count"] >= 2)
            ]),
        ),
        "2. Finalize pilot-100 selection from pilot_100_balanced.csv",
        "3. Proceed with Stage 1: Download meteorological data for selected 100 basins",
        "4. Run Flash-NH model on pilot basins",
        "",
    ])
    
    with open(summaries_dir / "pilot_selection_summary.md", "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Created pilot_selection_summary.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze WY2024 streamflow metrics and design pilot basin selection"
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory containing completed WY2024 metrics output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for pilot selection analysis outputs",
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    logger.info("Starting WY2024 pilot basin selection analysis")
    logger.info("Metrics source: %s", args.metrics_dir)
    logger.info("Output directory: %s", args.output_dir)
    
    # Load metrics
    df = load_metrics(args.metrics_dir)
    logger.info("Loaded %d basins from metrics table", len(df))
    
    # Summarize universe
    universe_summary = summarize_universe(df, logger)
    
    # Build threshold tables
    threshold_df = build_threshold_tables(df, logger)
    threshold_df.to_csv(args.output_dir / "rbi_threshold_analysis.csv", index=False)
    
    # Build selection scores
    df_scored = build_selection_scores(df, logger)
    
    # Create pilot lists
    create_pilot_lists(df_scored, args.output_dir, logger)
    
    # Build plots
    build_plots(df_scored, args.output_dir, logger)
    
    # Build summaries
    build_summaries(df_scored, universe_summary, args.output_dir, args.metrics_dir, logger)
    
    logger.info("Analysis complete: %s", args.output_dir)
    print(f"\nPilot selection analysis complete:")
    print(f"  Output: {args.output_dir}")
    print(f"  Plots: {args.output_dir / 'plots'}")
    print(f"  Tables: {args.output_dir / 'tables'}")
    print(f"  Summaries: {args.output_dir / 'summaries'}")


if __name__ == "__main__":
    main()
