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
- IFS is important and should not be downgraded or dropped; the current `oper/fc` historical path still only retrieves 00 and 12 UTC, while 06 and 18 UTC remain unresolved on that path. A deterministic `scda` fallback was found for 06/18, so the issue is not "only two cycles exist".
- ERA5-Land is retained as a Stage 2 long-term daily antecedent input.
- GDAS is retained as a Stage 2 long-term daily antecedent input.
- IMERG download works, but crop validation still needs final nonzero selected-CONUS confirmation.

## Notes

- The current workflow still uses the existing datasource architecture.
- This document records the intended research sequencing and the current freeze/research status for each source.