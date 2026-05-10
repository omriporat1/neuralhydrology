# Flash-NH — Data Pipeline Status

Flash-NH is the project name for the near-real-time and forecast-aware hydrological modeling pipeline built around basin-average forcing, CAMELSH streamflow/static attributes, NeuralHydrology-based modeling, and staged Stage 1/2/3 experiments.

## Intended Research Pipeline

### Stage 1
Near-real-time basin-average hourly inputs:
- MRMS hourly precipitation
- RTMA hourly meteorology
- CAMELSH static basin attributes

Target:
- hourly streamflow, compared with CAMELSH hourly streamflow

### Stage 2
Stage 1 inputs plus long-term daily antecedent LSTM inputs:
- ERA5-Land
- GDAS
- IMERG Late Daily

All basin-averaged.

Target:
- hourly streamflow, compared with CAMELSH

### Stage 3
Stage 1 + Stage 2 plus predicted hourly forecast inputs:
- IFS
- GFS

All basin-averaged.

Target:
- hourly streamflow, compared with CAMELSH

## Current Datasource Status

- MRMS is retained for Stage 1 basin-average hourly precipitation and remains part of the intended research pipeline.
- RTMA is retained for Stage 1 basin-average hourly meteorology despite its size/time cost because it is scientifically important.
- GFS byte-range extraction is validated and should be frozen except for plotting changes.
- **IFS (ECMWF MARS)** is operationally improved:
  - Stream logic: 00/12 UTC use `oper/fc`, 06/18 UTC use `scda/fc` (deterministic fallback).
  - Grid resolution: `0.1/0.1` to leverage higher spatial resolution relative to GFS.
  - All four cycles (00, 06, 12, 18 UTC) now accessible and validated.
  - Area subset `50/-126/24/-66` supported.
  - Verified 0.1-degree storage/time estimates:
    - Sample bytes per cycle (full request): 54,920,250 bytes.
    - Bytes per day (4 cycles): 219,681,000 bytes.
    - Annual raw volume: 80.184 GB/year (74.677 GiB/year).
    - Prediction-period raw volume (2020-10-14 to 2025-12-31): 418.492 GB (389.751 GiB).
    - Prediction-period derived basin-average volume (9,000 basins): 11.521 GB (10.730 GiB).
    - Estimated sequential acquisition time for full period: ~102.40 hours (~4.27 days).
  - Prior wording check: "~80 GB/year" was approximately correct in decimal units, but imprecise.
- ERA5-Land is retained as a Stage 2 long-term daily antecedent input.
- GDAS is retained as a Stage 2 long-term daily antecedent input.
- IMERG Late Daily NC4 download and CONUS crop validation are now verified:
  - Dynamic coordinate/dimension handling supports `time,lat,lon`, `time,lon,lat`, `lat,lon`, and `lon,lat` forms.
  - Verified selected-CONUS bytes are nonzero on target sample (2023-01-01): `624000` bytes.
  - Verified crop bounds and shape are within expected CONUS envelope: lon ~[-125.95, -66.05], lat ~[24.05, 49.95], shape `(260, 600)`.
- Preview plotting validation is now repaired and validated for IMERG, GFS, and IFS:
  - Uses lon/lat extents on axes and north-up orientation logic.
  - `preview_bounds_validation`: PASS for IMERG, GFS, and IFS.

## Notes

- The current workflow still uses the existing datasource architecture.
- This document records the intended research sequencing and the current freeze/research status for each source.

## Final All-Source Audit (2026-05-09)

- Completed unified 24-hour acquisition audit validating all 7 implemented datasources in dry-run mode.
- Audit orchestration: `scripts/run_final_all_source_audit.py` (717 lines, reuses existing acquisition/report infrastructure).
- Outputs: `reports/final_all_source_audit_2026_05/`
  - Summaries: `final_audit_summary.{json,csv,md,html}` + `datasource_matrix.{csv,md}`
  - Decision-support plots: `storage_breakdown_by_source.png`, `reduction_waterfall_by_source.png`, `download_time_vs_size.png`, `availability_timeline.png`, `crop_validation_overview.png`
  - Lightweight review bundle with previews, request specs, truncated logs, manifests
- Status: All 7 sources (MRMS, RTMA, ERA5-Land, GDAS, IMERG, GFS, IFS) validated with current request logic.
- No code changes to acquisition logic applied during this audit; all prior fixes and improvements are locked in.
- 7-day stability audit infrastructure in place (dry-run unable to cache full sample; ready for real-run).
- Recommendation: Proceed with operational implementation using Stage 1→Stage 2→Stage 3 sequencing documented in this pipeline status.