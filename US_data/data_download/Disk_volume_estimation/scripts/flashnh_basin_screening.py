#!/usr/bin/env python3
"""
Flash-NH Basin Screening Workflow

Merges CAMELSH/GAGES-II attributes, filters basins by drainage area and BFI,
and generates candidate basin tables and plots for potential Stage 1 pilot.

Inputs:
- attributes_gageii_BasinID.csv
- attributes_gageii_Hydro.csv

Outputs to reports/flashnh_basin_screening_v001/:
- candidate_basin_screening_summary.md
- candidate_basin_screening_summary.json
- all_basins_merged.parquet
- area_filtered_basins.parquet
- area_bfi_filtered_basins.parquet
- plots/
  - drainage_area_distribution.png
  - drainage_area_log_distribution.png
  - bfi_distribution_area_filtered.png
  - bfi_by_area_bin_boxplot.png
  - candidate_count_by_threshold.png
  - lat_lon_candidate_scatter.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configuration
ATTRIBUTES_DIR = Path("C:/PhD/Python/neuralhydrology/US_data/attributes")
OUTPUT_DIR = Path("reports/flashnh_basin_screening_v001")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Filtering parameters
DRAIN_SQKM_MIN = 1.0
DRAIN_SQKM_MAX = 1000.0
BFI_THRESHOLD_PRIMARY = 40  # Primary candidate threshold (BFI <= 40)
BFI_THRESHOLDS_EXPLORE = [20, 30, 40, 50]  # Thresholds to explore
AREA_BIN_BOUNDS = [1, 10, 100, 1000]
AREA_BIN_LABELS = ['1-10 km²', '10-100 km²', '100-1000 km²']
THRESHOLD_COLORS = {
    20: '#2b8cbe',
    30: '#31a354',
    40: '#de2d26',
    50: '#756bb1',
}


def _format_threshold_label(threshold):
    return f'BFI <= {threshold}'

def load_data():
    """Load and merge CAMELSH attribute files."""
    print("Loading CAMELSH attribute files...")
    
    basin_id_path = ATTRIBUTES_DIR / "attributes_gageii_BasinID.csv"
    hydro_path = ATTRIBUTES_DIR / "attributes_gageii_Hydro.csv"
    
    basin_id_df = pd.read_csv(basin_id_path)
    hydro_df = pd.read_csv(hydro_path)
    
    print(f"  BasinID: {len(basin_id_df)} basins")
    print(f"  Hydro: {len(hydro_df)} basins")
    
    # Merge on STAID
    merged = pd.merge(basin_id_df, hydro_df, on="STAID", how="inner")
    print(f"  Merged: {len(merged)} basins")
    
    return merged

def analyze_and_filter(merged):
    """Analyze and filter basins by drainage area and BFI."""
    print("\n=== Basin Inventory ===")
    print(f"Total basins: {len(merged)}")
    
    # Check for missing data
    print(f"\nBasins with valid DRAIN_SQKM: {merged['DRAIN_SQKM'].notna().sum()}")
    print(f"Basins with valid BFI_AVE: {merged['BFI_AVE'].notna().sum()}")
    
    # Area filtering
    area_filtered = merged[
        (merged['DRAIN_SQKM'] >= DRAIN_SQKM_MIN) &
        (merged['DRAIN_SQKM'] <= DRAIN_SQKM_MAX)
    ].copy()
    area_filtered['area_bin'] = pd.cut(
        area_filtered['DRAIN_SQKM'],
        bins=AREA_BIN_BOUNDS,
        labels=AREA_BIN_LABELS,
        include_lowest=True,
    )
    print(f"\nBasins after area filter ({DRAIN_SQKM_MIN}–{DRAIN_SQKM_MAX} km²): {len(area_filtered)}")
    
    # BFI statistics (after area filtering)
    bfi_stats = {
        'min': area_filtered['BFI_AVE'].min(),
        'q10': area_filtered['BFI_AVE'].quantile(0.10),
        'q25': area_filtered['BFI_AVE'].quantile(0.25),
        'median': area_filtered['BFI_AVE'].quantile(0.50),
        'q75': area_filtered['BFI_AVE'].quantile(0.75),
        'q90': area_filtered['BFI_AVE'].quantile(0.90),
        'q95': area_filtered['BFI_AVE'].quantile(0.95),
        'max': area_filtered['BFI_AVE'].max(),
    }
    
    print("\nBFI_AVE distribution (area-filtered basins):")
    for key, val in bfi_stats.items():
        if pd.notna(val):
            print(f"  {key:10s}: {val:8.2f}")
    
    # BFI + area filtering
    area_bfi_filtered = area_filtered[
        area_filtered['BFI_AVE'] <= BFI_THRESHOLD_PRIMARY
    ].copy()
    print(f"\nBasins after area + BFI filter (BFI <= {BFI_THRESHOLD_PRIMARY}): {len(area_bfi_filtered)}")
    
    # Explore other BFI thresholds
    print("\nCandidate counts by BFI threshold (area-filtered basins):")
    candidate_counts = {}
    threshold_rows = []
    for thresh in BFI_THRESHOLDS_EXPLORE:
        selected = area_filtered[area_filtered['BFI_AVE'] <= thresh].copy()
        count = len(selected)
        pct_area_filtered = 100.0 * count / len(area_filtered)
        candidate_counts[f'BFI<={thresh}'] = int(count)
        threshold_rows.append({
            'threshold': thresh,
            'candidate_count': int(count),
            'percent_of_area_filtered': pct_area_filtered,
            'percent_of_total': 100.0 * count / len(merged),
        })
        print(f"  BFI <= {thresh}: {count} basins ({pct_area_filtered:.1f}% of area-filtered)")
    
    # Area bin analysis
    print("\nBasins by drainage area bin (area-filtered):")
    area_bin_counts = {}
    for label in AREA_BIN_LABELS:
        count = (area_filtered['area_bin'] == label).sum()
        area_bin_counts[label] = int(count)
        print(f"  {label}: {count} basins")

    threshold_summary_table = pd.DataFrame(threshold_rows)

    area_bin_by_threshold_rows = []
    for label in AREA_BIN_LABELS:
        row = {'area_bin': label}
        for thresh in BFI_THRESHOLDS_EXPLORE:
            row[f'BFI<={thresh}'] = int(((area_filtered['area_bin'] == label) & (area_filtered['BFI_AVE'] <= thresh)).sum())
        area_bin_by_threshold_rows.append(row)
    area_bin_by_threshold_table = pd.DataFrame(area_bin_by_threshold_rows)
    
    return {
        'all_basins': merged,
        'area_filtered': area_filtered,
        'area_bfi_filtered': area_bfi_filtered,
        'bfi_stats': bfi_stats,
        'candidate_counts': candidate_counts,
        'area_bin_counts': area_bin_counts,
        'threshold_summary_table': threshold_summary_table,
        'area_bin_by_threshold_table': area_bin_by_threshold_table,
    }

def generate_plots(data_dict):
    """Generate diagnostic plots."""
    print("\nGenerating plots...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    merged = data_dict['all_basins']
    area_filtered = data_dict['area_filtered']
    area_bfi_filtered = data_dict['area_bfi_filtered']

    lon_min = area_filtered['LNG_GAGE'].min()
    lon_max = area_filtered['LNG_GAGE'].max()
    lat_min = area_filtered['LAT_GAGE'].min()
    lat_max = area_filtered['LAT_GAGE'].max()
    lon_pad = max(0.5, (lon_max - lon_min) * 0.03)
    lat_pad = max(0.5, (lat_max - lat_min) * 0.03)
    x_limits = (lon_min - lon_pad, lon_max + lon_pad)
    y_limits = (lat_min - lat_pad, lat_max + lat_pad)
    
    # 1. Drainage area distribution (linear)
    plt.figure(figsize=(10, 6))
    plt.hist(merged['DRAIN_SQKM'].dropna(), bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(DRAIN_SQKM_MIN, color='r', linestyle='--', label=f'Min ({DRAIN_SQKM_MIN} km²)')
    plt.axvline(DRAIN_SQKM_MAX, color='r', linestyle='--', label=f'Max ({DRAIN_SQKM_MAX} km²)')
    plt.xlabel('Drainage Area (km²)')
    plt.ylabel('Count')
    plt.title('Distribution of Drainage Area (Linear Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "drainage_area_distribution.png", dpi=150)
    plt.close()
    
    # 2. Drainage area distribution (log scale)
    plt.figure(figsize=(10, 6))
    plt.hist(merged['DRAIN_SQKM'].dropna(), bins=100, edgecolor='black', alpha=0.7)
    plt.xscale('log')
    plt.axvline(DRAIN_SQKM_MIN, color='r', linestyle='--', label=f'Min ({DRAIN_SQKM_MIN} km²)')
    plt.axvline(DRAIN_SQKM_MAX, color='r', linestyle='--', label=f'Max ({DRAIN_SQKM_MAX} km²)')
    plt.xlabel('Drainage Area (km², log scale)')
    plt.ylabel('Count')
    plt.title('Distribution of Drainage Area (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "drainage_area_log_distribution.png", dpi=150)
    plt.close()
    
    # 3. BFI distribution (area-filtered)
    plt.figure(figsize=(10, 6))
    plt.hist(area_filtered['BFI_AVE'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(BFI_THRESHOLD_PRIMARY, color='r', linestyle='--', linewidth=2, label=f'Threshold ({BFI_THRESHOLD_PRIMARY})')
    plt.xlabel('Base Flow Index (BFI_AVE)')
    plt.ylabel('Count')
    plt.title(f'Distribution of BFI_AVE (Area-Filtered: {DRAIN_SQKM_MIN}–{DRAIN_SQKM_MAX} km²)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bfi_distribution_area_filtered.png", dpi=150)
    plt.close()
    
    # 4. BFI by area bin (boxplot)
    area_filtered_copy = area_filtered.copy()
    area_filtered_copy['area_bin'] = pd.cut(
        area_filtered_copy['DRAIN_SQKM'],
        bins=[1, 10, 100, 1000],
        labels=['1-10 km²', '10-100 km²', '100-1000 km²'],
        include_lowest=True
    )
    
    plt.figure(figsize=(10, 6))
    area_filtered_copy.boxplot(column='BFI_AVE', by='area_bin', figsize=(10, 6))
    plt.axhline(BFI_THRESHOLD_PRIMARY, color='r', linestyle='--', linewidth=2, label=f'Threshold ({BFI_THRESHOLD_PRIMARY})')
    plt.suptitle('')  # Remove default title
    plt.title(f'BFI_AVE by Drainage Area Bin (Area-Filtered)')
    plt.xlabel('Drainage Area Bin')
    plt.ylabel('Base Flow Index (BFI_AVE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bfi_by_area_bin_boxplot.png", dpi=150)
    plt.close()
    
    # 5. Candidate count by BFI threshold
    thresholds = BFI_THRESHOLDS_EXPLORE
    counts = [(area_filtered['BFI_AVE'] <= t).sum() for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([f'BFI≤{t}' for t in thresholds], counts, edgecolor='black', alpha=0.7, color='steelblue')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(count), ha='center', va='bottom', fontweight='bold')
    plt.ylabel('Number of Basins')
    plt.title('Candidate Basin Count by BFI Threshold (Area-Filtered)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "candidate_count_by_threshold.png", dpi=150)
    plt.close()
    
    # 6. Geographic scatter (if lat/lon available)
    if 'LAT_GAGE' in area_filtered.columns and 'LNG_GAGE' in area_filtered.columns:
        plt.figure(figsize=(14, 8))
        plt.scatter(
            area_filtered['LNG_GAGE'],
            area_filtered['LAT_GAGE'],
            c='lightgray',
            s=20,
            alpha=0.3,
            label=f'Area-filtered ({len(area_filtered)})'
        )
        plt.scatter(
            area_bfi_filtered['LNG_GAGE'],
            area_bfi_filtered['LAT_GAGE'],
            c='red',
            s=50,
            alpha=0.6,
            label=f'Candidates (BFI≤{BFI_THRESHOLD_PRIMARY}, n={len(area_bfi_filtered)})'
        )
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geographic Distribution of Candidate Basins')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "lat_lon_candidate_scatter.png", dpi=150)
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, threshold in zip(axes, BFI_THRESHOLDS_EXPLORE):
            selected = area_filtered[area_filtered['BFI_AVE'] <= threshold]
            selected_count = len(selected)
            selected_pct = 100.0 * selected_count / len(area_filtered)
            ax.scatter(area_filtered['LNG_GAGE'], area_filtered['LAT_GAGE'], c='lightgray', s=12, alpha=0.35, linewidths=0)
            ax.scatter(selected['LNG_GAGE'], selected['LAT_GAGE'], c=THRESHOLD_COLORS[threshold], s=18, alpha=0.8, linewidths=0)
            ax.set_title(f'{_format_threshold_label(threshold)} | n={selected_count} ({selected_pct:.1f}%)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.grid(True, alpha=0.25)

        fig.suptitle('Flash-NH Candidate Basins by BFI Threshold', fontsize=16)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        fig.savefig(PLOTS_DIR / 'map_bfi_thresholds_faceted.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(15, 9))
        scatter = ax.scatter(
            area_filtered['LNG_GAGE'],
            area_filtered['LAT_GAGE'],
            c=area_filtered['BFI_AVE'],
            cmap='viridis',
            s=22,
            alpha=0.9,
            linewidths=0,
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
        cbar.set_label('BFI_AVE')
        ax.set_title(f'Area-Filtered Basins Colored by BFI_AVE (n={len(area_filtered)})')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / 'map_bfi_continuous.png', dpi=160)
        plt.close(fig)
    
    print(f"  Plots saved to {PLOTS_DIR}")

def write_outputs(data_dict):
    """Write summary tables and metadata."""
    print("\nWriting output files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Parquet files
    data_dict['all_basins'].to_parquet(OUTPUT_DIR / "all_basins_merged.parquet")
    data_dict['area_filtered'].to_parquet(OUTPUT_DIR / "area_filtered_basins.parquet")
    data_dict['area_bfi_filtered'].to_parquet(OUTPUT_DIR / "area_bfi_filtered_basins.parquet")
    
    print(f"  Saved parquet files to {OUTPUT_DIR}")
    
    # Summary statistics
    summary = {
        'total_basins': len(data_dict['all_basins']),
        'basins_with_valid_drain_sqkm': int(data_dict['all_basins']['DRAIN_SQKM'].notna().sum()),
        'basins_with_valid_bfi_ave': int(data_dict['all_basins']['BFI_AVE'].notna().sum()),
        'area_filter_min_sqkm': DRAIN_SQKM_MIN,
        'area_filter_max_sqkm': DRAIN_SQKM_MAX,
        'basins_after_area_filter': len(data_dict['area_filtered']),
        'bfi_threshold_primary': BFI_THRESHOLD_PRIMARY,
        'basins_after_area_bfi_filter': len(data_dict['area_bfi_filtered']),
        'bfi_statistics': {k: float(v) if pd.notna(v) else None for k, v in data_dict['bfi_stats'].items()},
        'candidate_counts': data_dict['candidate_counts'],
        'area_bin_counts': data_dict['area_bin_counts'],
    }
    
    with open(OUTPUT_DIR / "candidate_basin_screening_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved JSON summary")

    threshold_table = data_dict['threshold_summary_table'].copy()
    threshold_table['threshold_label'] = threshold_table['threshold'].apply(lambda value: f'BFI <= {int(value)}')
    threshold_table = threshold_table[['threshold_label', 'threshold', 'candidate_count', 'percent_of_area_filtered', 'percent_of_total']]
    threshold_table.to_csv(OUTPUT_DIR / 'bfi_threshold_summary_table.csv', index=False)

    threshold_md = [
        '# Flash-NH BFI Threshold Summary',
        '',
        '| Threshold | Candidate count | Percent of area-filtered | Percent of total |',
        '| --- | ---: | ---: | ---: |',
    ]
    for _, row in threshold_table.iterrows():
        threshold_md.append(
            f"| {row['threshold_label']} | {int(row['candidate_count'])} | {row['percent_of_area_filtered']:.1f}% | {row['percent_of_total']:.1f}% |"
        )
    with open(OUTPUT_DIR / 'bfi_threshold_summary_table.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(threshold_md))

    data_dict['area_bin_by_threshold_table'].to_csv(OUTPUT_DIR / 'area_bin_by_bfi_threshold_table.csv', index=False)

    print('  Saved threshold summary tables')

def write_markdown_summary(data_dict):
    """Write markdown summary report."""
    area_filtered = data_dict['area_filtered']
    area_bfi_filtered = data_dict['area_bfi_filtered']
    bfi_stats = data_dict['bfi_stats']
    
    lines = [
        "# Flash-NH Basin Screening Summary",
        "",
        "## Overview",
        "",
        "This document summarizes the initial candidate basin screening for the Flash-NH Stage 1 pilot.",
        "",
        "Basins are selected from CAMELSH/GAGES-II using drainage area and base flow index (BFI) as screening proxies.",
        "",
        "## Screening Criteria",
        "",
        f"- **Drainage Area**: {DRAIN_SQKM_MIN}–{DRAIN_SQKM_MAX} km²",
        f"- **Primary BFI Threshold**: BFI_AVE <= {BFI_THRESHOLD_PRIMARY}",
        "",
        "## Basin Inventory",
        "",
        f"- **Total basins**: {data_dict['all_basins'].shape[0]}",
        f"- **Basins with valid DRAIN_SQKM**: {data_dict['all_basins']['DRAIN_SQKM'].notna().sum()}",
        f"- **Basins with valid BFI_AVE**: {data_dict['all_basins']['BFI_AVE'].notna().sum()}",
        "",
        f"## Basins After Area Filter ({DRAIN_SQKM_MIN}–{DRAIN_SQKM_MAX} km²)",
        "",
        f"- **Count**: {len(area_filtered)}",
        "",
        "### BFI_AVE Statistics (Area-Filtered Basins)",
        "",
        "|  Statistic  |  Value  |",
        "|-------------|---------|",
    ]
    
    for key, val in bfi_stats.items():
        if pd.notna(val):
            lines.append(f"| {key:11s} | {val:7.2f} |")
    
    lines.extend([
        "",
        "### Candidate Counts by BFI Threshold",
        "",
        "|  Threshold  |  Count  |",
        "|-------------|---------|",
    ])
    
    for thresh in BFI_THRESHOLDS_EXPLORE:
        count = (area_filtered['BFI_AVE'] <= thresh).sum()
        lines.append(f"| BFI ≤ {thresh:2d}  |  {count:5d}  |")
    
    lines.extend([
        "",
        "### Basins by Drainage Area Bin",
        "",
        "|  Area Bin   |  Count  |",
        "|-------------|---------|",
    ])
    
    for label, count in data_dict['area_bin_counts'].items():
        lines.append(f"| {label:11s} |  {count:5d}  |")
    
    lines.extend([
        "",
        f"## Primary Candidate Basins (BFI <= {BFI_THRESHOLD_PRIMARY})",
        "",
        f"- **Count**: {len(area_bfi_filtered)}",
        f"- **Percent of area-filtered**: {100.0 * len(area_bfi_filtered) / len(area_filtered):.1f}%",
        "",
        "## Next Steps: USGS Availability Workflow",
        "",
        "For the area+BFI-filtered candidate basins, the next workflow phase includes:",
        "",
        "1. **Query NWIS IV availability** for parameter 00060 (discharge) for each candidate basin.",
        "",
        "2. **Determine available resolution** from returned data or metadata:",
        "   - Prefer hourly IV data if complete enough.",
        "   - If only sub-hourly/native IV exists, retrieve native data.",
        "",
        "3. **Retrieve discharge data** for the Flash-NH research period (2020-10-14 to 2025-12-31).",
        "   - Aggregate sub-hourly data to hourly if needed.",
        "   - Do not force sub-hourly retrieval if hourly is adequate.",
        "",
        "4. **Compute coverage metrics**:",
        "   - Overall data completeness over the full research period.",
        "   - Completeness by water year.",
        "   - Reject basins with insufficient coverage (<90% as guideline).",
        "",
        "5. **Compute observed flashiness metrics** for remaining basins:",
        "   - Richards-Baker Flashiness Index (RBI).",
        "   - Max hourly dQ/dt (discharge change rate).",
        "   - Normalized max hourly dQ/dt.",
        "   - Event-based peak/rise metrics.",
        "   - Q95/Q99 event counts.",
        "",
        "6. **Pilot basin selection** from data-validated basins stratified by:",
        "   - Drainage area bin.",
        "   - BFI bin (or observed RBI bin if available).",
        "   - Streamflow data completeness.",
        "   - Geographic distribution if metadata allow.",
        "",
        "## Important Notes",
        "",
        "- **BFI is a screening proxy**, not the definitive flashiness metric.",
        "  Observed flashiness will be computed from USGS discharge using RBI and event-based metrics.",
        "",
        "- **Streamflow target**: USGS NWIS IV (parameter 00060, discharge).",
        "  Hourly data preferred; sub-hourly data will be aggregated to hourly as needed.",
        "",
        "- **No data downloads yet**: This report documents the screening logic only.",
        "  Data retrieval and observed flashiness computation are handled separately.",
        "",
    ])
    
    with open(OUTPUT_DIR / "candidate_basin_screening_summary.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"  Saved markdown summary")

def main():
    """Main workflow."""
    print("=" * 70)
    print("Flash-NH Basin Screening Workflow")
    print("=" * 70)
    
    # Load data
    merged = load_data()
    
    # Analyze and filter
    results = analyze_and_filter(merged)
    
    # Generate plots
    generate_plots(results)
    
    # Write outputs
    write_outputs(results)
    write_markdown_summary(results)
    
    print("\n" + "=" * 70)
    print("Basin screening complete.")
    print(f"Outputs: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
