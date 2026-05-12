# Flash-NH Basin Screening Summary

## Overview

This document summarizes the initial candidate basin screening for the Flash-NH Stage 1 pilot.

Basins are selected from CAMELSH/GAGES-II using drainage area and base flow index (BFI) as screening proxies.

## Screening Criteria

- **Drainage Area**: 1.0–1000.0 km²
- **Primary BFI Threshold**: BFI_AVE <= 40

## Basin Inventory

- **Total basins**: 9008
- **Basins with valid DRAIN_SQKM**: 9008
- **Basins with valid BFI_AVE**: 9008

## Basins After Area Filter (1.0–1000.0 km²)

- **Count**: 5836

### BFI_AVE Statistics (Area-Filtered Basins)

|  Statistic  |  Value  |
|-------------|---------|
| min         |    1.98 |
| q10         |   23.00 |
| q25         |   33.44 |
| median      |   47.56 |
| q75         |   59.73 |
| q90         |   68.28 |
| q95         |   72.74 |
| max         |   87.61 |

### Candidate Counts by BFI Threshold

|  Threshold  |  Count  |
|-------------|---------|
| BFI ≤ 20  |    384  |
| BFI ≤ 30  |   1142  |
| BFI ≤ 40  |   2130  |
| BFI ≤ 50  |   3228  |

### Basins by Drainage Area Bin

|  Area Bin   |  Count  |
|-------------|---------|
| 1-10 km²    |    201  |
| 10-100 km²  |   1773  |
| 100-1000 km² |   3862  |

## Primary Candidate Basins (BFI <= 40)

- **Count**: 2130
- **Percent of area-filtered**: 36.5%

## Next Steps: USGS Availability Workflow

For the area+BFI-filtered candidate basins, the next workflow phase includes:

1. **Query NWIS IV availability** for parameter 00060 (discharge) for each candidate basin.

2. **Determine available resolution** from returned data or metadata:
   - Prefer hourly IV data if complete enough.
   - If only sub-hourly/native IV exists, retrieve native data.

3. **Retrieve discharge data** for the Flash-NH research period (2020-10-14 to 2025-12-31).
   - Aggregate sub-hourly data to hourly if needed.
   - Do not force sub-hourly retrieval if hourly is adequate.

4. **Compute coverage metrics**:
   - Overall data completeness over the full research period.
   - Completeness by water year.
   - Reject basins with insufficient coverage (<90% as guideline).

5. **Compute observed flashiness metrics** for remaining basins:
   - Richards-Baker Flashiness Index (RBI).
   - Max hourly dQ/dt (discharge change rate).
   - Normalized max hourly dQ/dt.
   - Event-based peak/rise metrics.
   - Q95/Q99 event counts.

6. **Pilot basin selection** from data-validated basins stratified by:
   - Drainage area bin.
   - BFI bin (or observed RBI bin if available).
   - Streamflow data completeness.
   - Geographic distribution if metadata allow.

## Important Notes

- **BFI is a screening proxy**, not the definitive flashiness metric.
  Observed flashiness will be computed from USGS discharge using RBI and event-based metrics.

- **Streamflow target**: USGS NWIS IV (parameter 00060, discharge).
  Hourly data preferred; sub-hourly data will be aggregated to hourly as needed.

- **No data downloads yet**: This report documents the screening logic only.
  Data retrieval and observed flashiness computation are handled separately.
