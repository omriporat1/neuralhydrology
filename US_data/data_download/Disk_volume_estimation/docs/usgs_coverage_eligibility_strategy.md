# Flash-NH USGS Coverage Eligibility Strategy

This document explains the coverage eligibility stage that filters area-filtered basins before attempting RBI retrieval at scale.

## Why this stage exists

The metadata-only USGS availability audit showed that 5,836+ area-filtered basins have USGS site records. However, the probe diagnostics revealed a critical gap:

**Metadata existence ≠ usable discharge data.**

Even when USGS metadata indicate that parameter 00060 (discharge) is available, the actual discharge observations may:
- Not overlap with the Flash-NH research period (2020-10-14 to 2025-12-31)
- Not overlap with the screening water year (2023-10-01 to 2024-09-30)
- Exist historically but be from inactive/legacy gages

The 75-basin probe confirmed this: 30/75 basins (39%) returned NO_DATA despite metadata suggesting parameter 00060 exists.

To avoid attempting RBI retrieval on basins that have no usable data, a metadata-first eligibility stage must filter the basin universe **before scaling to full RBI download**.

## Eligibility classes

Each area-filtered basin is classified into one of six eligibility categories based on USGS metadata:

### ELIGIBLE_SCREENING_WY
- Has valid USGS site metadata
- Has parameter 00060
- Has metadata overlap with the full or most of the screening water year (2023-10-01 to 2024-09-30)
- **Decision**: These basins are the primary candidates for RBI screening. Begin RBI retrieval with this group first.
- **Expected outcome**: Highest success rate for finding usable discharge data.

### ELIGIBLE_RESEARCH_PERIOD
- Has valid USGS site metadata
- Has parameter 00060
- Has metadata overlap with the Flash-NH research period (2020-10-14 to 2025-12-31)
- But does NOT have meaningful overlap with the screening water year (2023-10-01 to 2024-09-30)
- **Decision**: These basins may be used if ELIGIBLE_SCREENING_WY coverage is insufficient. Requires alternative water year selection (e.g., 2022-10-01 to 2023-09-30 or 2024-10-01 to 2025-09-30).
- **Expected outcome**: Moderate success; additional logic needed to choose alternative water year per basin.

### HISTORICAL_ONLY
- Has parameter 00060 in USGS metadata
- But has NO overlap with the Flash-NH research period (2020-10-14 to 2025-12-31)
- **Decision**: Exclude from RBI screening unless a separate historical analysis is intended.
- **Expected outcome**: No discharge data available in the research period.

### NO_00060
- USGS site exists
- But parameter 00060 is absent
- **Decision**: No discharge data available at all; exclude.
- **Expected outcome**: Only precipitation, groundwater, or other non-discharge parameters may exist.

### INVALID_SITE
- No valid USGS site metadata
- **Decision**: Exclude.
- **Expected outcome**: Basin cannot be matched to USGS infrastructure.

### UNKNOWN
- Insufficient metadata to classify
- **Decision**: Review and potentially reclassify or exclude.
- **Expected outcome**: Rare; usually indicates data quality issues in the availability audit.

## Typical scaling workflow

1. **Start**: 5,836 area-filtered basins
2. **Filter**: Remove basins not in ELIGIBLE_SCREENING_WY or ELIGIBLE_RESEARCH_PERIOD → N basins
3. **Phase 1 RBI retrieval**: Begin with all ELIGIBLE_SCREENING_WY basins (uses screening water year 2023-10-01 to 2024-09-30)
4. **Evaluate Phase 1**: How many ELIGIBLE_SCREENING_WY basins yield RBI_READY or PARTIAL_USABLE status?
5. **Phase 2 (optional)**: If coverage is sufficient, stop. If insufficient, select ELIGIBLE_RESEARCH_PERIOD basins and choose alternative water year (e.g., 2024-10-01 to 2025-09-30)
6. **Evaluate Phase 2**: Recompute RBI for the ELIGIBLE_RESEARCH_PERIOD basins using the alternative water year

## How to handle ELIGIBLE_RESEARCH_PERIOD basins

Basins in ELIGIBLE_RESEARCH_PERIOD have discharge metadata outside the screening water year. If they are needed for improved coverage:

1. **Option A**: Use an adjacent water year (e.g., 2024-10-01 to 2025-09-30) and recompute RBI for that cohort separately
2. **Option B**: Use the most recent available water year within the research period for which data are abundant
3. **Option C**: Document these basins as "alternative water year candidates" for future analysis beyond the screening water year

## Metadata overlap interpretation

The eligibility stage computes:
- **research_overlap_days**: Number of days in the research period (2020-10-14 to 2025-12-31) covered by USGS metadata availability window
- **screening_overlap_days**: Number of days in the screening water year (2023-10-01 to 2024-09-30) covered by USGS metadata availability window

**Important**: These metrics measure metadata availability, not actual discharge observations. A basin with high overlap_days may still return NO_DATA if:
- The site is inactive/no longer monitored
- The IV service does not return observations despite metadata suggesting availability
- Data are available only at daily or coarser resolution (not hourly or sub-hourly)

For this reason, the probe validates a subset of basins against the IV service before committing to full-scale RBI retrieval.

## Expected distribution

From the 5,836 area-filtered basins:
- ~50-60% are expected to be ELIGIBLE_SCREENING_WY
- ~10-20% are expected to be ELIGIBLE_RESEARCH_PERIOD
- ~10-15% are expected to be HISTORICAL_ONLY
- ~10-20% are expected to be NO_00060 or INVALID_SITE

These fractions are based on the 75-basin probe sample and extrapolation to the full basin universe. The exact counts will be computed by the usgs_coverage_eligibility.py script.

## Key decision points

1. **Before Phase 1 RBI retrieval**: Confirm that ELIGIBLE_SCREENING_WY count is acceptable. If <50% of the target basin universe, reassess basin selection rules or data sources.
2. **After Phase 1 RBI retrieval**: Evaluate the fraction of ELIGIBLE_SCREENING_WY basins that yield RBI_READY or PARTIAL_USABLE status. If inadequate, decide whether to proceed to Phase 2 with alternative water years.
3. **Basin Universe Reassessment**: If eligibility analysis shows that >30% of basins are HISTORICAL_ONLY or NO_00060, the area filtering, basin universe, or source assumptions may need revision.

## Relationship to the probe and availability audit

- **USGS Availability Audit** (scripts/usgs_availability_audit.py): Metadata-only scan of 5,836 basins; produces usgs_availability_candidates.csv
- **Coverage Eligibility Stage** (scripts/usgs_coverage_eligibility.py): Classifies the 5,836 basins by metadata overlap; produces usgs_coverage_eligibility.csv
- **USGS Discharge Probe** (scripts/usgs_discharge_probe.py): Validates a 75-basin sample with real IV data retrieval; confirms method and identifies NO_DATA patterns
- **Probe Diagnostics** (scripts/usgs_probe_diagnostics.py): Audits and refines NO_DATA classifications using metadata to understand why basins returned no data

All four stages must be completed and reviewed before committing to full-scale RBI retrieval.
