# Flash-NH USGS RBI Workflow Plan

This plan moves Flash-NH from static basin screening to observed streamflow flashiness analysis.

## A) Purpose

The basin screening step uses BFI_AVE as a static proxy for flashiness.
The next workflow step will use USGS discharge to compute observed flashiness, with RBI as the main screening metric.

## B) Candidate Universe

The USGS availability audit should start from all area-filtered CAMELSH/GAGES-II basins:

- DRAIN_SQKM between 1 and 1000 km2
- Do not restrict the audit only to BFI <= 40

This keeps the audit broad enough to avoid excluding basins that may have moderate static BFI but strong observed flashiness.

## C) USGS Target

- Data source: USGS NWIS IV discharge
- Parameter code: 00060
- Final modeling cadence: hourly
- Prefer hourly IV data if it is available and complete enough
- If only native or sub-hourly IV exists, retrieve that and resample or aggregate to hourly
- Do not force sub-hourly retrieval if hourly data are already adequate

## D) Time Windows

Research period:

- 2020-10-14 to 2025-12-31

First RBI screening period:

- 2023-10-01 to 2024-09-30

Rationale:

- One full water year is preferred for screening
- Full-period RBI can be computed later for selected or validated basins
- One month is too seasonal for RBI-based screening

## E) Availability Audit Outputs

For each area-filtered basin, the next availability audit should record:

- STAID
- DRAIN_SQKM
- BFI_AVE
- latitude, longitude, state, and HUC if available
- whether a USGS site exists
- whether parameter 00060 exists
- available begin and end dates if metadata provide them
- overlap with the research period
- overlap with the screening water year
- likely resolution if it can be inferred
- likely path:
  - hourly direct
  - native/sub-hourly plus resample
  - unavailable
  - unknown/error
- status:
  - AVAILABLE
  - PARTIAL
  - NO_DATA
  - ERROR

## F) RBI Computation Stage

For basins with adequate availability:

1. Download screening water-year discharge.
2. Create an hourly discharge series.
3. Compute completeness.
4. Compute RBI.
5. Compute max hourly dQ/dt.
6. Compute normalized max hourly dQ/dt.
7. Compute Q95 and Q99 event counts.
8. Optionally compute 15-minute versus hourly preservation diagnostics only if native sub-hourly data are retrieved.

## G) RBI Thresholding Strategy

Do not rely only on sample quantiles.

When RBI is computed, create plots and tables using both of the following:

1. Literature or reference thresholds or bands.
   - Search code comments and docs should state that universal RBI thresholds may not exist.
   - If published or agency reference bands are found, use them and cite the source.
   - If no defensible universal thresholds are found, state explicitly that fixed thresholds are exploratory.

2. Distribution-based thresholds.
   - q50
   - q75
   - q90
   - q95
   - Clearly label these as relative to the Flash-NH candidate sample.

RBI visualization requirements:

- RBI histogram with both reference bands and quantiles
- RBI boxplot by area bin
- RBI vs BFI scatter
- RBI vs drainage area scatter
- maps for fixed or reference RBI bands if available
- maps for RBI >= q50, q75, q90, and q95
- map of USGS availability and completeness
- map showing where basins drop out at each stage

## H) Pilot Selection Recommendation

Do not finalize pilot basins until:

- USGS availability is known
- RBI distribution is inspected
- geographic bias is evaluated

Select 50 to 100 basins stratified by:

- area bin
- observed RBI
- streamflow completeness
- geography or region
- optionally BFI as a secondary stratification variable

## Stage Mapping Requirement

Every later-stage map should show the progression through:

- all CAMELSH/GAGES-II basins
- area-filtered basins
- USGS-available basins
- RBI-computed basins
- final pilot basins

This is needed to make spatial bias visible at each filtering stage.