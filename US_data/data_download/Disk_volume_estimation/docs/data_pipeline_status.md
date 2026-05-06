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

- GFS byte-range extraction is validated and should be frozen except for plotting changes.
- IMERG download works, but crop validation still needs final nonzero selected-CONUS confirmation.
- IFS is important and should not be downgraded or dropped.
- Current MARS request retrieves 00 and 12 UTC but fails for 06 and 18 UTC.
- Because ECMWF product descriptions indicate 00/06/12/18 cycles should exist, treat the failure as an unresolved access/request issue, not as evidence that only 2 cycles exist.

## Notes

- The current workflow still uses the existing datasource architecture.
- This document records the intended research sequencing and the current freeze/research status for each source.