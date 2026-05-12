# Flash-NH USGS Availability Strategy

This document defines the metadata-only USGS availability audit that happens before any discharge time series are downloaded.

## Why this audit happens first

Flash-NH uses BFI as a static screening proxy, but BFI is not the final flashiness metric. Before downloading discharge data, the project needs a basin-by-basin inventory of what USGS data actually exist, what date range is available, and whether the series looks suitable for hourly flashiness screening.

Doing this first avoids downloading large discharge archives for basins that will later be excluded for lack of usable IV coverage.

## Why all area-filtered basins are audited

The availability audit should begin with all basins that pass the drainage-area filter:

- DRAIN_SQKM between 1 and 1000 km2
- expected count: 5,836 basins

This is important because USGS availability may be geographically biased. If the audit starts only from BFI <= 40, basins with moderate or high BFI could be excluded before we know whether they have strong observed flashiness or better data coverage.

## USGS target

The target discharge source is:

- USGS NWIS IV discharge
- parameter code 00060
- final modeling cadence: hourly

Preferred logic:

1. Prefer hourly IV data if it is available and complete enough.
2. If only native or sub-hourly IV data are available, mark the basin as requiring native retrieval plus hourly resampling.
3. Do not download full discharge time series during the availability audit.

## Time windows

Research period:

- 2020-10-14 to 2025-12-31

Screening water year:

- 2023-10-01 to 2024-09-30

The screening water year is the first period that should be used for RBI-based screening later. One month is too seasonal for RBI screening.

## Bias tracking requirement

USGS availability itself may be spatially biased. Later workflow stages should therefore map each stage explicitly:

- all CAMELSH/GAGES-II basins
- area-filtered basins
- USGS-available basins
- RBI-computed basins
- final pilot basins

This makes geographic dropout visible and helps prevent a biased pilot selection.

## Audit outputs

The availability audit should record at least:

- STAID
- DRAIN_SQKM
- BFI_AVE
- area_bin
- latitude, longitude, state, HUC if available
- whether a USGS site appears valid
- whether parameter 00060 exists
- available begin date if metadata provide it
- available end date if metadata provide it
- overlap with the research period
- overlap with the screening water year
- likely data resolution if detectable without full download
- likely retrieval path:
  - hourly direct
  - native/sub-hourly + resample
  - unavailable
  - unknown/error
- preliminary status:
  - AVAILABLE
  - PARTIAL
  - NO_DATA
  - ERROR
- notes or error message

## Next step after the audit

For basins that are AVAILABLE or promising PARTIAL cases, the next step is to download one full water year of discharge and compute RBI, max hourly dQ/dt, normalized max hourly dQ/dt, and event counts.

The first screening window should be the full water year 2023-10-01 to 2024-09-30, not a single month.
