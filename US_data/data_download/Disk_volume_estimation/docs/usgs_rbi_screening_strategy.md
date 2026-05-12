# Flash-NH USGS RBI Screening Strategy

This document describes how the scaled USGS discharge retrieval and RBI screening run is executed for Flash-NH.

## Why only ELIGIBLE_SCREENING_WY basins are used

The scaled run uses only basins where eligibility_class is ELIGIBLE_SCREENING_WY. This ensures that each selected basin has metadata overlap with the screening water year (2023-10-01 to 2024-09-30) and has USGS parameter 00060.

Using this subset reduces avoidable request volume and focuses retrieval effort on basins most likely to return usable discharge observations in the target window.

## Why historical-only basins are excluded

HISTORICAL_ONLY basins have parameter 00060 metadata but no overlap with the Flash-NH research period. Including them in the screening run would inflate NO_DATA outcomes and does not serve the screening objective for present-period flashiness characterization.

Historical-only basins should be handled only in a dedicated historical analysis workflow.

## Why RBI is computed over one full water year

RBI is season-sensitive. A short interval (for example one month) can overrepresent seasonal events and produce unstable flashiness interpretation.

The screening run therefore uses one full water year:
- Start: 2023-10-01 00:00 UTC
- End: 2024-09-30 23:00 UTC

All basin metrics are based on this common window so cross-basin comparison is coherent.

## Data handling and metric rules

- Target source: USGS NWIS IV
- Parameter: 00060 (discharge)
- Native or sub-hourly IV observations are accepted
- Series are converted to a final hourly series by resampling/aggregation
- RBI is computed only on the final hourly series
- Internal missing hourly gaps are not bridged when computing dQ/dt and RBI
- If sum(Q_t) is zero or invalid, RBI remains null

This preserves the validated, gap-aware, unit-invariant RBI behavior.

## Completeness thresholds

Screening statuses are assigned from hourly completeness against the full expected hourly index for the water year:

- RBI_READY: completeness >= 90%
- PARTIAL_USABLE: 70% <= completeness < 90%
- INSUFFICIENT: completeness < 70% with data present
- NO_DATA: request succeeded but no observations were returned
- ERROR: request, parsing, or processing failed

## Operational workflow

The scaled script supports resumable batch processing:

- Command: scripts/usgs_rbi_screening_scale.py
- Defaults: batch size 25, polite request delay, retry with exponential backoff
- Resume mode skips STAIDs already present in output rows
- Checkpoints are written after each small batch

Primary commands:

- Smoke test:
  python scripts/usgs_rbi_screening_scale.py --max-basins 50 --batch-size 10 --resume

- Full run:
  python scripts/usgs_rbi_screening_scale.py --batch-size 25 --resume

## How screening supports pilot basin selection

The scaled output provides, per basin:
- retrieval success diagnostics
- inferred native timestep behavior
- hourly completeness
- RBI and derivative flow-change metrics

This allows ranking basins for pilot work using both hydrologic signal (RBI) and data reliability (completeness, status), while preserving area and BFI stratification.

## Interpretation warning

RBI should never be interpreted in isolation.

Always interpret RBI jointly with:
- hourly completeness percentage
- geographic distribution of data availability
- status class (RBI_READY versus PARTIAL_USABLE/INSUFFICIENT)

A high RBI from sparse or geographically clustered basins can bias conclusions if completeness and spatial coverage are ignored.
