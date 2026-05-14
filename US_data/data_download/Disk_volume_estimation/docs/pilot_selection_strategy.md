# WY2024 Pilot Basin Selection Strategy

## Overview

This document describes the approach to selecting pilot basins from the completed WY2024 streamflow metrics analysis for Flash-NH Stage 1 testing. The selection process prioritizes interpretability, geographic diversity, and unbiased representation of flashy and control hydrologic regimes.

## Rationale for Pilot Selection

After screening 3,324 USGS basins for RBI (Rapid Response Bias Index) and computing 40+ comprehensive streamflow metrics, we have a stratified universe of candidates. Pilot basin selection must:

1. **Balance flashiness diversity**: Include FLASHY_CORE (high RBI), FLASHY_MODERATE, FLASHY_POSSIBLE, and LOW_FLASHINESS_CONTROL basins
2. **Span geographic regions**: Avoid clustering in one watershed or state
3. **Cover basin sizes**: Include small, medium, and large basins
4. **Represent base flow characteristics**: Include high-BFI and low-BFI basins
5. **Maintain data quality**: Prioritize high completeness and valid RBI calculations
6. **Allow manual review**: Flag context-flagged basins for inspection before final commitment

## Quantitative Framework

### Hard QC Pass Rate

Of 3,324 basins processed:
- **3,045 pass hard QC** (91.6%)
  - FLASHY_CORE: 397 (13%)
  - FLASHY_MODERATE: 503 (17%)
  - FLASHY_POSSIBLE: 2,082 (68%)
  - LOW_FLASHINESS_CONTROL: 5 (<1%)
  - MANUAL_REVIEW_CONTEXT: 58 (2%)
- **279 fail hard QC** (8.4%)
  - Excluded for: low completeness (<90%), negative flow, no RBI, severe zero-flow dominance

### Pilot Selection Scores

An interpretable composite score is constructed as:

```
pilot_score = 0.35 × class_score
            + 0.30 × rbi_score
            + 0.15 × rise_score
            + 0.10 × event_score
            + 0.05 × completeness_score
            - 0.10 × zero_flow_penalty
            - 0.05 × context_flag_penalty
```

**Component definitions:**

- **class_score**: Candidate class priority (FLASHY_CORE=3, MODERATE=2, POSSIBLE=1, CONTROL=0.5, MANUAL_REVIEW=0.2)
- **rbi_score**: RBI normalized to [0, 1] within valid range
- **rise_score**: Max hourly rise per km² normalized to [0, 1] by maximum observed
- **event_score**: Event count (Q99 peaks) normalized to [0, 1] by maximum
- **completeness_score**: Hourly completeness percentage / 100
- **zero_flow_penalty**: Zero-flow fraction (high penalty reduces score)
- **context_flag_penalty**: Presence of context flags (informational issues)

**Weights rationale:**
- Class (0.35) and RBI (0.30) dominate, reflecting primary selection criteria
- Rise (0.15) and events (0.10) capture flashiness nuances
- Completeness (0.05) is secondary; QC-pass basins all meet minimum threshold
- Penalties (total -0.15) downweight basins requiring manual review but don't exclude them

## Proposed Pilot Lists

### Pilot-50 Balanced
- **Target**: 50 basins for initial model evaluation
- **Strategy**: Top-scoring basins with stratification by area_bin and BFI_bin
- **Composition**: Prioritize FLASHY_CORE/MODERATE, maintain geographic and hydrologic diversity
- **Use case**: Rapid prototyping and calibration

### Pilot-100 Balanced
- **Target**: 100 basins for comprehensive Stage 1 testing
- **Strategy**: Stratified selection by area bin, BFI bin, and candidate class
- **Composition**: Proportional representation of all classes (except EXCLUDE_HARD_QC)
- **Use case**: Primary Stage 1 pilot for Flash-NH model evaluation
- **Expected composition**:
  - ~40-50 FLASHY_CORE/MODERATE (high priority)
  - ~40-50 FLASHY_POSSIBLE (moderate priority)
  - ~5-10 LOW_FLASHINESS_CONTROL (reference basins)
  - ~2-5 MANUAL_REVIEW_CONTEXT (if suitable after review)

### Pilot-150 Broad
- **Target**: 150 basins for expanded evaluation
- **Strategy**: Largest stratified pool, includes marginal candidates
- **Composition**: Comprehensive representation; broader geographic coverage
- **Use case**: Sensitivity analysis and regional performance assessment

### Manual Review Priority
- **Count**: ~50-100 basins with context flags or MANUAL_REVIEW_CONTEXT class
- **Strategy**: Sort by pilot_score; review for borderline cases, spike artifacts, regulatory/artifact concerns
- **Action**: Inspect for data artifacts; decide whether to include in final pilot or exclude
- **Use case**: QC validation before Stage 1 commitment

### Flashy Extreme Review
- **Count**: Top 50 by RBI
- **Strategy**: Identify outliers and extreme flashiness
- **Action**: Check for data quality issues, regulation, or unusual hydrology; review before inclusion
- **Use case**: Validation of highest-RBI basins; ensure they represent natural flashiness

### Possible + Low Flashiness Controls
- **Count**: Top 50 FLASHY_POSSIBLE + LOW_FLASHINESS_CONTROL basins
- **Strategy**: Sort by pilot_score
- **Use case**: Reference set for model performance on moderate and low-response basins

## Stratification Variables

### Area Bin
- Divides basins into categories (e.g., <10 km², 10-50 km², 50-100 km², >100 km²)
- Ensures model evaluation across scale range
- Small basins often more flashy; large basins often more damped

### Base Flow Index (BFI) Bin
- Categorizes runoff composition (e.g., BFI <20%, 20-40%, 40-60%, >60%)
- Low BFI: High runoff response; high BFI: Low runoff response
- Ensures evaluation across baseflow regimes

### HUC02 / State
- Geographic identifier
- Ensures basin selection spans multiple watersheds and climate zones
- Reduces correlated errors from local meteorology or data processing artifacts

## Quality Checks and Caveats

### Context Flags Requiring Manual Review
- **CONTEXT_SUSPICIOUS_SPIKE_SEVERE** (max_jump/Q50 ≥ 20): Potential data artifacts; review timeseries before inclusion
- **CONTEXT_HIGH_SPECIFIC_PEAK** (Q_max/km² ≥ 1.0): Extreme peaks; may indicate regulation or rare events
- **CONTEXT_INTERMITTENT_LIKE** (zero_flow ≥ 10%): Intermittent/ephemeral character; verify USGS protocols
- **CONTEXT_LOW_SPECIFIC_FLOW** (Q99/km² ≤ 0.01): Very low flow regime; confirm data validity
- **CONTEXT_ZERO_FLOW_SOME** (zero_flow ≥ 5%): Seasonal/partial zero flow; acceptable if brief

### Readiness for Stage 1

**Recommendation**: YES, proceed to Stage 1 meteorological preprocessing.

**Justification**:
- 91.6% of basins pass hard QC criteria (high-quality data)
- Clear separation of flashy and control classes (3,045 vs. 279)
- Sufficient diversity in RBI, area, BFI, and geography
- Pilot-100 stratification ensures representative evaluation
- Manual-review items are flaggable for inspection but not blockers

**Outstanding items**:
1. Review top 50-100 context-flagged basins for data artifacts
2. Confirm FLASHY_CORE/MODERATE basins represent true hydrologic flashiness (not sensor/processing artifacts)
3. Validate small-basin (<10 km²) selection logic; confirm spike patterns are natural

## Output Files

### Tables
- `pilot_50_balanced.csv`: Top-50 stratified basins
- `pilot_100_balanced.csv`: Top-100 stratified basins (recommended for Stage 1)
- `pilot_150_broad.csv`: Top-150 stratified basins (broader evaluation)
- `manual_review_priority.csv`: Context-flagged basins for inspection
- `flashy_extreme_review.csv`: Top 50 by RBI; check for outliers
- `possible_low_flashiness_controls.csv`: FLASHY_POSSIBLE and CONTROL basins
- `rbi_threshold_analysis.csv`: RBI thresholds crossed by area/BFI bins

### Plots
- `class_distribution.png`: Bar chart of candidate class counts
- `map_candidate_classes.png`: Geographic scatter of basins by class
- `map_rbi_continuous.png`: Continuous RBI map
- `rbi_by_area_bin.png`: Box plot of RBI by basin area
- `rbi_by_bfi_bin.png`: Box plot of RBI by base flow index
- `rbi_by_class.png`: Histogram of RBI by candidate class
- `pilot_score_distribution.png`: Distribution of composite pilot selection scores
- `max_rise_per_km2_distribution.png`: Hourly rise intensity distribution
- `q95_q50_distribution.png`: Q95/Q50 ratio distribution
- `zero_flow_by_class.png`: Box plot of zero-flow fraction by class

### Summaries
- `pilot_selection_summary.json`: Structured data (counts, compositions, readiness)
- `pilot_selection_summary.md`: Human-readable summary with recommendations
- `run.log`: Detailed processing log

## Next Actions

1. **Validate**: Review outputs and spot-check context-flagged basins
2. **Finalize Pilot-100**: Select from `pilot_100_balanced.csv` based on manual review
3. **Stage 1 Preprocessing**: Download meteorological data (ERA5, MERRA-2, etc.) for selected 100 basins
4. **Stage 1 Model Run**: Execute Flash-NH model on pilot basins with downloaded meteorology
5. **Evaluation**: Compare model performance across basin classes, areas, and flashiness levels
6. **Scale**: If Stage 1 successful, expand to larger pilot or full deployment

## References

- WY2024 Streamflow Metrics Matrix: `docs/wy2024_streamflow_metrics_matrix.md`
- RBI Screening Results: `reports/flashnh_usgs_rbi_screening_wy2024_v001/`
- Metrics Calculation: `scripts/build_wy2024_streamflow_metrics_matrix.py`
