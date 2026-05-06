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
  - Grid resolution: upgraded to `0.1/0.1` to leverage higher spatial resolution relative to GFS (6.2 KB per point vs 25 km per point).
  - All four cycles (00, 06, 12, 18 UTC) now accessible and validated.
  - Area subset `50/-126/24/-66` supported.
  - Estimated data volume: ~80 GB/year (vs ~13 GB/year for 0.25/0.25).
- ERA5-Land is retained as a Stage 2 long-term daily antecedent input.
- GDAS is retained as a Stage 2 long-term daily antecedent input.
- IMERG download works, but crop validation still needs final nonzero selected-CONUS confirmation.

## Notes

- The current workflow still uses the existing datasource architecture.
- This document records the intended research sequencing and the current freeze/research status for each source.