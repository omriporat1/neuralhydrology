# WY2024 Streamflow Metrics Matrix Workflow

## Overview

This workflow downloads and stores WY2024 hourly streamflow data for all usable USGS basins (RBI_READY and optionally PARTIAL_USABLE) from the completed screening results, computes a comprehensive streamflow/event/QC metrics matrix, and generates candidate classes for unbiased pilot basin selection.

## Why This Workflow Exists

After the initial RBI screening, we have identified 3,092 RBI_READY and 232 PARTIAL_USABLE basins. The screening provides a single metric (RBI), but pilot basin selection requires richer hydrologic context:
- Multiple flashiness indicators (RBI, event counts, rises)
- Hydrological signatures (Q95/Q50, specific flows, base flow index)
- Data quality and completeness flags
- Streamflow regime characteristics (zero-flow, negative flow, intermittent patterns)

This workflow preserves the validated gap-aware RBI calculation while computing 40+ additional metrics per basin, applying hard QC filters and context-based flags, and generating candidate classes that can support manual review and unbiased pilot selection.

## Metrics Computed

### Core streamflow quantiles
- `Q_min`, `Q50`, `Q95`, `Q99`, `Q_max`, `Q_mean`, `sum_Q`
- `q95_q50_ratio`, `q99_q50_ratio`
- Specific versions per km² (normalized by basin area)

### Flashiness and event response
- `RBI` (gap-aware Rapid Response Bias Index)
- `max_hourly_rise`, `max_hourly_fall`, `max_abs_hourly_jump` (m³/s)
- Per-km² versions and normalized by Q50 ratios
- `event_count_q95`, `event_count_q99` (local peaks above quantiles)
- `largest_event_peak_Q`, `largest_event_rise`, `largest_event_rise_per_km2`

### Data quality
- `valid_hour_count`, `expected_hour_count`, `hourly_completeness_pct`
- `zero_flow_count`, `zero_flow_fraction`
- `negative_flow_count`, `negative_flow_fraction`

### Basin attributes (from screening results)
- `STAID`, `DRAIN_SQKM`, `BFI_AVE`, `area_bin`, `BFI_bin`, `STATE`, `HUC02`
- `LAT_GAGE`, `LNG_GAGE`
- `source_status` (RBI_READY or PARTIAL_USABLE)

## Gap-Aware Calculations

All metrics are computed using gap-aware logic:
- Hourly values are not artificially bridged across missing timestamps
- dQ for RBI calculation skips intervals where data is missing in either hour
- Denominators (sum_Q) are strictly from valid observations
- Ratios with Q50 that are ≤ 0 or non-finite are flagged and returned as NaN

## QC Flags

### Hard QC Flags (Exclusion Criteria)

These flags indicate basins that should be excluded from pilot consideration:

- `HARD_LOW_COMPLETENESS_LT90`: Hourly completeness < 90%
- `HARD_NEGATIVE_FLOW_SEVERE`: Negative flow fraction > 1%
- `HARD_ZERO_FLOW_DOMINATED`: Zero flow fraction ≥ 25%
- `HARD_Q50_ZERO_OR_NEAR_ZERO`: Q50 ≤ 0 or non-finite
- `HARD_NO_RBI`: RBI cannot be computed or is non-finite
- `HARD_SUSPICIOUS_SPIKE_SEVERE`: Max hourly jump / Q50 ≥ 20 (artifact-like)

Any basin with one or more hard QC flags is assigned to `EXCLUDE_HARD_QC` class.

### Context Flags (Informational)

These flags provide hydrological context but do not automatically exclude basins:

- `CONTEXT_ZERO_FLOW_SOME`: Zero flow fraction ≥ 5%
- `CONTEXT_HIGH_NORMALIZED_JUMP`: Max hourly jump / Q50 ≥ 5
- `CONTEXT_LOW_SPECIFIC_FLOW`: Q99 per km² ≤ 0.01 m³/s
- `CONTEXT_HIGH_SPECIFIC_PEAK`: Q_max per km² ≥ 1.0 m³/s
- `CONTEXT_INTERMITTENT_LIKE`: Zero flow fraction ≥ 10%
- `CONTEXT_POSSIBLE_REGULATION_OR_ARTIFACT`: High normalized jump and long flat runs
- `CONTEXT_SMALL_BASIN`: Drainage area < 10 km²
- `CONTEXT_HIGH_BFI`: Base flow index ≥ 60
- `CONTEXT_LOW_BFI`: Base flow index ≤ 10

## Candidate Classes

Basins are assigned to one of six candidate classes:

1. **EXCLUDE_HARD_QC**: Hard QC flags present; excluded from pilot consideration
2. **FLASHY_CORE**: Completeness ≥ 95%, RBI ≥ 0.10, strong flashiness indicators
3. **FLASHY_MODERATE**: Completeness ≥ 95%, 0.05 ≤ RBI < 0.10, strong event response
4. **FLASHY_POSSIBLE**: Completeness ≥ 90%, RBI < 0.05 but strong event-response metrics
5. **LOW_FLASHINESS_CONTROL**: Completeness ≥ 90%, low RBI, weak event response (control basins)
6. **MANUAL_REVIEW_CONTEXT**: No hard QC failures but context flags or ambiguous classification

Note: `CONTEXT_POSSIBLE_REGULATION_OR_ARTIFACT` is a context flag only; it does not automatically exclude a basin, but instead flags it for manual review.

## Output Files

### Hourly Streamflow
- `hourly_streamflow/STAID.parquet`: One Parquet file per basin with hourly discharge and metadata

Schema:
- `time_utc`: Timestamp (UTC)
- `discharge_m3s`: Discharge in m³/s
- `original_units`: Original units from USGS (ft³/s or m³/s)
- `source`: Data source identifier
- `is_missing`: Boolean flag for NaN values
- `quality_code`: QC code if available

### Metrics Tables
- `tables/wy2024_streamflow_metrics.csv`: Full metrics table (one row per basin)
- `tables/wy2024_streamflow_metrics.parquet`: Same in Parquet format
- `tables/candidate_classes.csv`: Distribution of candidate classes
- `tables/hard_qc_exclusions.csv`: Basins with hard QC flags
- `tables/manual_review_context.csv`: Basins assigned to manual review
- `tables/flashy_candidate_pool.csv`: All FLASHY_CORE/MODERATE/POSSIBLE basins
- `tables/low_flashiness_controls.csv`: Control basins (LOW_FLASHINESS_CONTROL)

### Summaries
- `summaries/wy2024_streamflow_metrics_summary.md`: Human-readable summary
- `summaries/wy2024_streamflow_metrics_summary.json`: Structured summary with statistics

### Plots
Lightweight aggregate diagnostic plots:
- `plots/candidate_class_counts.png`: Bar chart of candidate class distribution
- `plots/rbi_distribution_qc_pass.png`: RBI histogram (QC-pass basins only)
- `plots/max_rise_per_km2_distribution.png`: Max hourly rise per km² histogram
- `plots/q99_q50_distribution.png`: Q99/Q50 ratio distribution

### Review Bundle
- `review_bundle/summary.md`: Markdown summary
- `review_bundle/summary.json`: JSON summary
- `review_bundle/candidate_classes.csv`: Class distribution table
- `review_bundle/hard_qc_exclusions_summary.json`: Hard QC count and sample basins
- `review_bundle/flashy_pool_summary.json`: Flashy pool statistics and top RBI basins
- `review_bundle/manifest.json`: Manifest of contents

### Logs
- `logs/run.log`: Detailed run log
- `logs/checkpoint.json`: Latest checkpoint (basin counts, processed list)

## Running the Workflow

### Smoke Test (20 basins)

Run a quick validation with limited basins:

```bash
python scripts/build_wy2024_streamflow_metrics_matrix.py --max-basins 20 --batch-size 5 --resume
```

Expected output:
- ~20 or fewer Parquet files in `hourly_streamflow/`
- Metrics table with ~20 rows
- Summary files and aggregate plots
- All generated files remain local (not staged to git)

### Full Run

To process all RBI_READY basins (3,092 basins):

```bash
python scripts/build_wy2024_streamflow_metrics_matrix.py --batch-size 25 --resume
```

To include PARTIAL_USABLE basins as well (3,324 total):

```bash
python scripts/build_wy2024_streamflow_metrics_matrix.py --batch-size 25 --resume --include-partial
```

### Resume Mode

If the workflow is interrupted, resume from the latest checkpoint:

```bash
python scripts/build_wy2024_streamflow_metrics_matrix.py --batch-size 25 --resume
```

### No-Download Mode

To test offline using only cached hourly Parquet files (if available from prior runs):

```bash
python scripts/build_wy2024_streamflow_metrics_matrix.py --max-basins 50 --no-download
```

### Force Refresh

To re-download hourly data and overwrite existing cached Parquet files:

```bash
python scripts/build_wy2024_streamflow_metrics_matrix.py --max-basins 50 --force-refresh
```

## CLI Options

- `--input-results`: Path to screening results directory (default: `reports/flashnh_usgs_rbi_screening_wy2024_v001`)
- `--output-dir`: Output folder (default: `reports/flashnh_wy2024_streamflow_metrics_v001`)
- `--max-basins`: Limit total basins processed (0 = all)
- `--batch-size`: Basins per checkpoint batch (default: 25)
- `--resume`: Resume from checkpoint if it exists
- `--sleep-seconds`: Delay between USGS requests (default: 0.15)
- `--max-retries`: USGS request retries (default: 4)
- `--timeout-seconds`: USGS request timeout (default: 45)
- `--force-refresh`: Re-download all hourly data
- `--no-download`: Use cache only, skip USGS requests
- `--include-partial`: Include PARTIAL_USABLE basins
- `--seed`: Random seed (default: 42)

## Implementation Notes

- The workflow reuses the validated IV retrieval and hourly resampling logic from `scripts/usgs_discharge_probe.py` and `scripts/usgs_rbi_screening_scale.py`.
- Gap-aware RBI calculation is preserved without modification.
- Each basin is processed independently and checkpointed after a small batch to enable resumability.
- All generated outputs (Parquet, CSV, plots, logs) are local and should not be committed to git.
- Only the script and this documentation should be committed.
- No meteorological data is downloaded.

## What Not to Commit

- Generated Parquet files
- Generated CSV/JSON tables
- Generated plots
- Generated logs (except for code under version control)
- Review bundle contents

Commit only:
- `scripts/build_wy2024_streamflow_metrics_matrix.py`
- `docs/wy2024_streamflow_metrics_matrix.md`
