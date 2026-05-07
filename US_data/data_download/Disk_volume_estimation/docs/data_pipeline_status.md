# Data Pipeline Status

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
- IMERG download works, but crop validation still needs final nonzero selected-CONUS confirmation.

## Notes

- The current workflow still uses the existing datasource architecture.
- This document records the intended research sequencing and the current freeze/research status for each source.