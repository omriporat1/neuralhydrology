# Flash-NH USGS Availability Audit Summary

## Purpose

Audit USGS NWIS IV discharge metadata for all area-filtered basins without downloading full discharge time series.

## Candidate Universe

- Total area-filtered basins: 5836
- Research period: 2020-10-14 to 2025-12-31
- Screening water year: 2023-10-01 to 2024-09-30

## Status Counts

| Status | Count |
| --- | ---: |
| AVAILABLE | 0 |
| PARTIAL | 5693 |
| NO_DATA | 143 |
| ERROR | 0 |

## Retrieval Path Counts

| Path | Count |
| --- | ---: |
| hourly direct | 0 |
| native/sub-hourly + resample | 1 |
| unavailable | 5835 |
| unknown/error | 0 |

## Key Notes

- The audit uses metadata only; no discharge time series were downloaded.
- Availability may be geographically biased, so later stages should map all basins, area-filtered basins, USGS-available basins, RBI-computed basins, and final pilot basins.
- RBI screening should be based on at least one full water year; the default screening window is 2023-10-01 to 2024-09-30.
- One month is too seasonal for RBI-based screening.