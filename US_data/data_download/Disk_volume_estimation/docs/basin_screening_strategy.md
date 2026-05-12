# Flash-NH — Basin Screening Strategy

Project: Flash-NH is a near-real-time and forecast-aware hydrological modeling pipeline built around basin-average forcing, CAMELSH streamflow/static attributes, NeuralHydrology-based modeling, and staged Stage 1/2/3 experiments.

This document describes the basin screening approach for Flash-NH: how candidate basins are selected for pilot and operational deployment.

------------------------------------------------------------
1) Data sources for basin catalog
------------------------------------------------------------

Flash-NH uses CAMELSH/GAGES-II for:

- basin catalog and metadata
- static basin attributes (geology, climate, land use, etc.)
- basin boundaries and polygon geometries
- basin IDs and naming conventions

Basin availability:

- **9,008 basins** in the GAGES-II dataset
- includes USGS streamflow monitoring sites with varying data completeness

------------------------------------------------------------
2) Drainage area filter
------------------------------------------------------------

**Rationale**: Focus on headwater to mid-sized basins typical of operational hydrologic forecasting domains.

**Filter**: drainage area between 1 and 1,000 km²

**Result**: ~5,836 basins pass this filter (65% of total)

**Rationale for bounds**:
- Lower bound (1 km²) excludes tiny hillslope-scale catchments with sparse gauge networks.
- Upper bound (1,000 km²) excludes large river systems where flood wave propagation and routed inflow dominate over local forcing.

------------------------------------------------------------
3) Base flow index (BFI) as a screening proxy for flashiness
------------------------------------------------------------

**BFI_AVE**: Base Flow Index from GAGES-II attributes. Typically ranges from 0 to 100, where lower BFI indicates flashier streamflow.

**What BFI represents**:
- Fraction of total streamflow contributed by baseflow (slow, sustained component).
- Computed from historical USGS daily discharge data.
- A landscape-scale proxy influenced by geology, soil, aquifer properties, topography.

**BFI_AVE distribution (area-filtered basins)**:

- Min: 1.98
- Q10: 23.00
- Q25: 33.44
- Median: 47.56
- Q75: 59.73
- Q90: 68.28
- Q95: 72.74
- Max: 87.61

**Why use BFI as a screening proxy**:
- BFI is readily available in GAGES-II and requires no USGS data retrieval.
- Low BFI identifies flashy basins likely to have rapid runoff response.
- Flashy basins are scientifically interesting for flood prediction and are challenging for models.

**Important caveat**: BFI is a historical proxy, not the final flashiness metric. Actual streamflow flashiness must be validated using observed USGS discharge data and event-based metrics.

------------------------------------------------------------
4) Recommended initial screening threshold
------------------------------------------------------------

**Primary screening rule**:

- Drainage area: 1–1,000 km²
- BFI_AVE: <= 40

**Candidate counts by BFI threshold** (area-filtered basins):

| Threshold | Count |
|-----------|-------|
| BFI <= 20 |   384 |
| BFI <= 30 | 1,142 |
| BFI <= 40 | 2,130 |
| BFI <= 50 | 3,228 |

**Rationale for BFI <= 40**:
- Selects the lower-BFI quartile, emphasizing flashier basins.
- Yields ~2,130 candidate basins (37% of area-filtered basins).
- Provides a manageable subset for USGS availability assessment.
- Can be adjusted based on data availability and computational resources.

**Working interpretation**:
- BFI <= 40 remains a reasonable working threshold because it keeps the lower-BFI side of the distribution while retaining enough basins for later USGS filtering.
- Final pilot basin selection should wait until observed discharge-derived flashiness is computed.
- The next USGS availability audit should consider all area-filtered basins, not only BFI <= 40, so that potentially flashy basins are not excluded too early.

**Alternative thresholds**:
- BFI <= 30: More selective (1,142 basins); higher flashiness emphasis.
- BFI <= 50: More inclusive (3,228 basins); captures mid-range basins.

------------------------------------------------------------
5) Geographic and area-based stratification
------------------------------------------------------------

**Basins by drainage area bin** (area-filtered):

| Bin | Count |
|-----|-------|
| 1–10 km² | 201 |
| 10–100 km² | 1,773 |
| 100–1,000 km² | 3,862 |

**Rationale for area bins**:
- Small basins (1–10 km²) have distinct hydrology (fast runoff, limited storage).
- Medium basins (10–100 km²) are common in operational networks and research domains.
- Large basins (100–1,000 km²) represent mid-sized systems with mixed properties.

**Geographic coverage**:
- Candidate basins span multiple USGS regions and climates if GAGES-II spatial metadata are used.
- Recommended: stratify pilot selection by geographic region to ensure representation across the study domain.

**Spatial bias tracking**:
- USGS availability may be geographically biased.
- Later stages should map all CAMELSH/GAGES-II basins, area-filtered basins, USGS-available basins, RBI-computed basins, and final pilot basins.

------------------------------------------------------------
6) Observed flashiness metrics (computed later)
------------------------------------------------------------

After USGS data are retrieved, compute observed flashiness to validate and refine BFI-based screening:

**Richards-Baker Flashiness Index (RBI)**:
- Measures day-to-day variability in streamflow.
- Formula: sum(|Q(t) - Q(t-1)|) / sum(Q(t))
- Lower = more baseflow-dominated; higher = flashier.

**Screening window**:
- RBI should be computed over at least one full water year for screening.
- Default screening water year: 2023-10-01 to 2024-09-30.
- One month is too seasonal for RBI-based screening.

**Max hourly change rate (dQ/dt)**:
- Maximum discharge change in a single hour.
- Normalized by mean or 95th percentile discharge to account for basin size.

**Event metrics**:
- Peak-to-rise ratio: peak magnitude / rate of rise.
- Event frequency: count of large events (Q > Q95 or similar threshold).
- Time to peak: duration from flow rise to peak.

**Interpretation**:
- Flashy basins: high RBI, large dQ/dt, frequent/steep-rising events.
- Baseflow-dominated basins: low RBI, gentle dQ/dt, infrequent large events.

------------------------------------------------------------
7) USGS discharge data strategy
------------------------------------------------------------

**Data source**: USGS NWIS IV (instantaneous value) data.

**Parameter code**: 00060 (discharge, cubic feet per second or equivalent).

**Preferred cadence**: hourly

**Data retrieval logic**:

1. Query NWIS IV availability for each candidate basin.
2. Check if hourly data are available and complete.
3. If hourly data are adequate (e.g., >= 90% coverage for the research period), use directly.
4. If only sub-hourly native IV data exist:
   - Retrieve native resolution data.
   - Aggregate to hourly by averaging or taking the maximum (depending on use case).
   - Do not force sub-hourly retrieval if hourly data are already available.
5. Compute coverage metrics by water year and overall.
6. Reject basins with insufficient data completeness (threshold TBD, recommend >= 90%).

**Audit scope**:
- The next availability audit should begin from all area-filtered basins, not only the BFI <= 40 subset.
- This avoids prematurely excluding basins that may have high observed flashiness but moderate static BFI.

**Research period**: 2020-10-14 to 2025-12-31 (matches Flash-NH data acquisition window).

**Unit conversion**: Convert from cfs to m³/s or consistent international units if needed.

------------------------------------------------------------
8) Pilot basin selection workflow
------------------------------------------------------------

After USGS discharge data are retrieved and observed flashiness metrics computed:

**Stage 1**: Select 50–100 pilot basins stratified by:

- Drainage area bin (ensure coverage of 1–10, 10–100, 100–1,000 km²)
- BFI bin (ensure coverage of low, medium, high BFI)
- Observed RBI bin (flashiness validation)
- Streamflow data completeness (prioritize high-quality coverage)
- Geographic region (if metadata available, ensure spatial diversity)

**Selection timing**:
- Do not finalize pilot basins until USGS availability is known, RBI is computed, and geographic bias is evaluated.

**Stage 1 pilot objectives**:

- verify basin geometry and static attributes
- verify basin-average MRMS precipitation and RTMA meteorology extraction
- verify USGS streamflow alignment and target variable quality
- generate QC reports
- prepare NeuralHydrology-compatible dataset
- train initial baseline model

**Stage 2/3 expansion**:

- After Stage 1 success, expand to 500–1,000 operational basins.
- Maintain stratification by area, flashiness, geography.
- Add antecedent and forecast inputs.

------------------------------------------------------------
9) Important considerations and caveats
------------------------------------------------------------

**BFI limitations**:
- BFI is computed from historical daily data, which may smooth sub-daily flashiness.
- Observed RBI and event-based metrics provide more direct validation of flashiness.
- Basins with very low BFI are not necessarily the best for model development (may have too few flood events or data quality issues).

**RBI thresholding guidance**:
- When RBI is computed later, plots should show both fixed literature/reference thresholds or bands, if defensible, and distribution-based quantiles labeled as dataset-relative.
- If defensible universal RBI thresholds cannot be justified from the literature, state that fixed thresholds are exploratory rather than universal.

**Data quality**:
- USGS discharge data quality varies by site and period.
- Perform manual QC spot checks on a sample of retrieved data before committing to a basin.
- Flag basins with known gauge instability or regulation.

**Regulation and human impacts**:
- Basins with dams, diversions, or irrigation withdrawals may have altered hydrology.
- GAGES-II includes metadata on hydrologic modification; use to inform pilot selection.

**Streamflow seasonality**:
- Use water-year-based splits (Oct–Sep) rather than calendar years to preserve seasonal integrity.
- Initial proposal: train on 2020-10-14 to 2023-09-30, validate 2023-10-01 to 2024-09-30, test 2024-10-01 to 2025-12-31.

**Temporal changes**:
- Flashiness may change over time (land use, climate, infrastructure).
- Monitor for systematic changes in event frequency or magnitude over the study period.

------------------------------------------------------------
10) Next steps
------------------------------------------------------------

1. **Finalize pilot basin count and thresholds**: Confirm whether BFI <= 40 is appropriate or adjust based on resource constraints.

2. **Query USGS NWIS IV availability**: Retrieve metadata on data completeness for candidate basins.

3. **Retrieve USGS discharge data**: Download native resolution IV data for basins with adequate coverage.

4. **Compute observed flashiness metrics**: Calculate RBI, dQ/dt, and event metrics for each basin.

5. **Refine pilot selection**: Stratify by area, BFI, observed RBI, data completeness, and geography.

6. **Generate pilot basin list**: Create final list of 50–100 basins for Stage 1 pilot.

7. **Begin Stage 1 pilot data pipeline**: Proceed with basin geometry, static attributes, basin-average forcing, and dataset generation.

------------------------------------------------------------

Please see `reports/flashnh_basin_screening_v001/` for detailed candidate basin analysis and plots.
