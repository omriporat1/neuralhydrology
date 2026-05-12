# Flash-NH USGS Discharge Probe Summary

## Purpose

This probe validates real USGS NWIS IV discharge retrieval for a stratified sample of area-filtered basins before scaling to the full candidate set.

## Probe Window

- Water year: 2023-10-01 to 2024-09-30
- Expected hourly timestamps: 8784

## Sample

- Probe sample size: 75
- Sample universe: PARTIAL basins with valid USGS metadata and parameter 00060

## Status Counts

| Status | Count |
| --- | ---: |
| RBI_READY | 39 |
| PARTIAL_USABLE | 3 |
| INSUFFICIENT | 3 |
| NO_DATA | 30 |
| ERROR | 0 |

## Hourly Completeness

- Median completeness: 94.6%
- Mean completeness: 56.3%

## RBI Readiness

- RBI_READY (>=90% completeness): 39
- PARTIAL_USABLE (70-90% completeness): 3
- INSUFFICIENT (<70% completeness): 3

## Native Timestep Summary

| Inferred timestep | Count |
| --- | ---: |
| hourly | 0 |
| sub-hourly | 45 |
| daily | 0 |
| irregular | 0 |
| sparse | 30 |

## Interpretation

- This probe is intended to decide whether the metadata-only audit was too conservative.
- Hourly-preferred handling is used first; native or sub-hourly series are resampled to hourly when needed.
- RBI will only be interpreted for basins with adequate completeness.
- If the probe shows acceptable coverage for a meaningful fraction of the sample, the same workflow can be scaled to the full area-filtered basin set.