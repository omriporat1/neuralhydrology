#!/usr/bin/env python3
"""Classify USGS discharge coverage eligibility for Flash-NH basins.

This stage determines which area-filtered basins are eligible for RBI screening
based on their metadata overlap with the Flash-NH research period and screening
water year.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR = ROOT
INPUT_AVAILABILITY = BASE_DIR / "reports/flashnh_usgs_availability_v001/usgs_availability_candidates.csv"
OUTPUT_DIR = BASE_DIR / "reports/flashnh_usgs_coverage_eligibility_v001"

# Time windows
RESEARCH_START = pd.Timestamp("2020-10-14", tz=None)
RESEARCH_END = pd.Timestamp("2025-12-31", tz=None)
SCREENING_START = pd.Timestamp("2023-10-01", tz=None)
SCREENING_END = pd.Timestamp("2024-09-30", tz=None)

RESEARCH_DAYS = int((RESEARCH_END - RESEARCH_START).days + 1)
SCREENING_DAYS = int((SCREENING_END - SCREENING_START).days + 1)


def classify_eligibility(row: pd.Series) -> str:
    """Classify basin coverage eligibility based on metadata."""
    # Check basic site validity
    if not bool(row.get("usgs_site_valid", False)):
        return "INVALID_SITE"
    
    # Check parameter 00060
    if not bool(row.get("has_parameter_00060", False)):
        return "NO_00060"
    
    # Parse date fields
    try:
        available_begin = pd.to_datetime(row.get("available_begin_date"), errors="coerce")
        available_end = pd.to_datetime(row.get("available_end_date"), errors="coerce")
    except Exception:
        available_begin = None
        available_end = None
    
    if pd.isna(available_begin) or pd.isna(available_end):
        return "UNKNOWN"
    
    # Check research period overlap
    research_overlap_days = int(row.get("research_overlap_days", 0) or 0)
    screening_overlap_days = int(row.get("screening_overlap_days", 0) or 0)
    
    # No research period overlap means this is historical-only data
    if research_overlap_days == 0:
        return "HISTORICAL_ONLY"
    
    # Has research overlap but not full screening water year
    if research_overlap_days > 0 and screening_overlap_days == 0:
        return "ELIGIBLE_RESEARCH_PERIOD"
    
    # Has meaningful screening water year overlap (at least some coverage)
    if screening_overlap_days > 0:
        return "ELIGIBLE_SCREENING_WY"
    
    return "UNKNOWN"


def load_data() -> pd.DataFrame:
    """Load availability candidates data."""
    if INPUT_AVAILABILITY.exists():
        return pd.read_csv(INPUT_AVAILABILITY, dtype={"STAID": str})
    else:
        # Try parquet fallback
        parquet_path = INPUT_AVAILABILITY.with_suffix(".parquet")
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Could not find input file: {INPUT_AVAILABILITY} or {parquet_path}")


def generate_plots(data: pd.DataFrame) -> None:
    """Generate diagnostic plots for coverage eligibility."""
    output_dir = OUTPUT_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Eligibility status counts
    counts = data["eligibility_class"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color="#2b8cbe")
    ax.set_xlabel("Basin Count")
    ax.set_title("USGS Coverage Eligibility Status Distribution")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "eligibility_status_counts.png", dpi=160)
    plt.close()
    
    # Plot 2: Eligibility by area bin
    area_eligibility = pd.crosstab(data["area_bin"], data["eligibility_class"])
    fig, ax = plt.subplots(figsize=(12, 6))
    area_eligibility.plot(kind="bar", ax=ax)
    ax.set_xlabel("Drainage Area Bin (km²)")
    ax.set_ylabel("Basin Count")
    ax.set_title("Coverage Eligibility by Drainage Area Bin")
    ax.legend(title="Eligibility", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "eligibility_by_area_bin.png", dpi=160)
    plt.close()
    
    # Plot 3: Eligibility by BFI bin
    bfi_eligibility = pd.crosstab(data["BFI_bin"], data["eligibility_class"])
    fig, ax = plt.subplots(figsize=(12, 6))
    bfi_eligibility.plot(kind="bar", ax=ax)
    ax.set_xlabel("BFI Bin")
    ax.set_ylabel("Basin Count")
    ax.set_title("Coverage Eligibility by BFI Bin")
    ax.legend(title="Eligibility", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "eligibility_by_bfi_bin.png", dpi=160)
    plt.close()
    
    # Plot 4: Map of coverage eligibility
    eligible_screening = data[data["eligibility_class"] == "ELIGIBLE_SCREENING_WY"]
    eligible_research = data[data["eligibility_class"] == "ELIGIBLE_RESEARCH_PERIOD"]
    historical = data[data["eligibility_class"] == "HISTORICAL_ONLY"]
    invalid = data[~data["eligibility_class"].isin(["ELIGIBLE_SCREENING_WY", "ELIGIBLE_RESEARCH_PERIOD", "HISTORICAL_ONLY"])]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(invalid["LNG_GAGE"], invalid["LAT_GAGE"], c="red", alpha=0.5, s=20, label="Invalid/No Data")
    ax.scatter(historical["LNG_GAGE"], historical["LAT_GAGE"], c="orange", alpha=0.5, s=20, label="Historical Only")
    ax.scatter(eligible_research["LNG_GAGE"], eligible_research["LAT_GAGE"], c="yellow", alpha=0.5, s=20, label="Eligible Research Period")
    ax.scatter(eligible_screening["LNG_GAGE"], eligible_screening["LAT_GAGE"], c="green", alpha=0.7, s=30, label="Eligible Screening WY")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("USGS Coverage Eligibility Map")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "map_coverage_eligibility.png", dpi=160)
    plt.close()
    
    # Plot 5: Research overlap days distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data["research_overlap_days"].fillna(0), bins=50, color="#2b8cbe", edgecolor="black")
    ax.set_xlabel("Research Period Overlap (days)")
    ax.set_ylabel("Basin Count")
    ax.set_title(f"Distribution of Research Period Overlap (Research Window: {RESEARCH_DAYS} days)")
    ax.axvline(RESEARCH_DAYS, color="red", linestyle="--", linewidth=2, label="Full Research Period")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "map_research_overlap_days.png", dpi=160)
    plt.close()
    
    # Plot 6: Screening overlap days distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data["screening_overlap_days"].fillna(0), bins=50, color="#de2d26", edgecolor="black")
    ax.set_xlabel("Screening Water Year Overlap (days)")
    ax.set_ylabel("Basin Count")
    ax.set_title(f"Distribution of Screening WY Overlap (Screening Window: {SCREENING_DAYS} days)")
    ax.axvline(SCREENING_DAYS, color="darkred", linestyle="--", linewidth=2, label="Full Screening WY")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "map_screening_overlap_days.png", dpi=160)
    plt.close()


def main() -> None:
    """Main workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    print(f"Loaded {len(data)} area-filtered basins from {INPUT_AVAILABILITY.name}")
    
    # Classify eligibility
    data["eligibility_class"] = data.apply(classify_eligibility, axis=1)
    
    # Generate summary statistics
    eligibility_counts = data["eligibility_class"].value_counts().to_dict()
    
    print("\nEligibility Summary:")
    for status, count in sorted(eligibility_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(data)
        print(f"  {status}: {count} ({pct:.1f}%)")
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    generate_plots(data)
    
    # Write outputs
    output_csv = OUTPUT_DIR / "usgs_coverage_eligibility.csv"
    data.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")
    
    output_parquet = OUTPUT_DIR / "usgs_coverage_eligibility.parquet"
    data.to_parquet(output_parquet, index=False)
    print(f"Wrote {output_parquet}")
    
    # Write summary JSON
    summary = {
        "time_windows": {
            "research_period": {
                "start": RESEARCH_START.isoformat(),
                "end": RESEARCH_END.isoformat(),
                "days": RESEARCH_DAYS,
            },
            "screening_water_year": {
                "start": SCREENING_START.isoformat(),
                "end": SCREENING_END.isoformat(),
                "days": SCREENING_DAYS,
            },
        },
        "eligibility_counts": eligibility_counts,
        "basin_count_total": len(data),
        "basin_count_eligible_screening_wy": int(eligibility_counts.get("ELIGIBLE_SCREENING_WY", 0)),
        "basin_count_eligible_research_period": int(eligibility_counts.get("ELIGIBLE_RESEARCH_PERIOD", 0)),
        "basin_count_historical_only": int(eligibility_counts.get("HISTORICAL_ONLY", 0)),
    }
    
    summary_json = OUTPUT_DIR / "usgs_coverage_eligibility_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_json}")
    
    # Write summary markdown
    summary_md = [
        "# USGS Coverage Eligibility Summary",
        "",
        f"**Total area-filtered basins:** {len(data)}",
        "",
        "## Eligibility Classification",
        "",
        "| Status | Count | Percentage |",
        "|--------|-------|-----------|",
    ]
    
    for status in ["ELIGIBLE_SCREENING_WY", "ELIGIBLE_RESEARCH_PERIOD", "HISTORICAL_ONLY", "NO_00060", "INVALID_SITE", "UNKNOWN"]:
        count = eligibility_counts.get(status, 0)
        pct = 100.0 * count / len(data)
        summary_md.append(f"| {status} | {count} | {pct:.1f}% |")
    
    summary_md.extend([
        "",
        "## Interpretation",
        "",
        "- **ELIGIBLE_SCREENING_WY**: Has USGS parameter 00060 with overlap during the screening water year (2023-10-01 to 2024-09-30). These basins are the primary candidates for the RBI screening workflow.",
        "- **ELIGIBLE_RESEARCH_PERIOD**: Has USGS parameter 00060 with overlap during the Flash-NH research period (2020-10-14 to 2025-12-31), but not during the screening water year. These basins may be used if an alternative water year is selected.",
        "- **HISTORICAL_ONLY**: Parameter 00060 exists but has no overlap with the Flash-NH research period. These are legacy gages and should not be used for the current RBI analysis.",
        "- **NO_00060**: Site exists but parameter 00060 is absent. These basins have no discharge data available.",
        "- **INVALID_SITE**: No valid USGS site metadata. These basins cannot be used.",
        "- **UNKNOWN**: Insufficient metadata to classify.",
        "",
        "## Next Steps",
        "",
        "1. Begin RBI retrieval with ELIGIBLE_SCREENING_WY basins.",
        "2. If coverage is insufficient, consider ELIGIBLE_RESEARCH_PERIOD basins with alternative water year selection.",
        "3. Exclude HISTORICAL_ONLY, NO_00060, INVALID_SITE, and UNKNOWN basins from the RBI workflow.",
    ])
    
    summary_md_file = OUTPUT_DIR / "usgs_coverage_eligibility_summary.md"
    summary_md_file.write_text("\n".join(summary_md), encoding="utf-8")
    print(f"Wrote {summary_md_file}")
    
    print("\nEligibility stage complete.")


if __name__ == "__main__":
    main()
