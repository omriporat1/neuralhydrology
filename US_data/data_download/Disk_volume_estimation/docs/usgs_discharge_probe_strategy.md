# Flash-NH USGS Discharge Probe Strategy

This document explains the small real-data discharge probe that follows the metadata-only USGS availability audit.

## Why this probe is needed

The metadata-only availability audit showed that many basins have USGS site metadata and parameter 00060, but the metadata alone could not reliably tell us whether a basin has enough usable discharge coverage for hourly RBI screening.

That audit was intentionally conservative. This probe is needed to answer the next question with real data:

- Can we retrieve one full water year of NWIS IV discharge for a representative subset of basins?
- Do those records support hourly RBI screening after resampling or aggregation?

## Interpreting PARTIAL metadata status

In the metadata-only audit, PARTIAL does not mean the basin is unusable.
It means the metadata were sufficient to suggest that 00060 exists, but not sufficient to confirm complete hourly coverage from metadata alone.

For that reason, PARTIAL basins are the correct population for the probe.

## Candidate universe

The probe uses the same area-filtered Flash-NH candidate universe:

- DRAIN_SQKM between 1 and 1000 km2
- then subset to basins with valid USGS site metadata and parameter 00060
- then probe a stratified sample of roughly 75 basins

The sample is stratified by:

- area bin: 1-10, 10-100, 100-1000 km2
- BFI bin: <=20, 20-30, 30-40, 40-50, >50
- geography, by spreading the selected sample across longitude and latitude within each stratum where possible

## USGS target and cadence strategy

The target discharge source is:

- USGS NWIS IV
- parameter code 00060
- final modeling cadence: hourly

Probe handling follows this order:

1. Prefer direct hourly data if it is already available.
2. If the returned series are native or sub-hourly, aggregate or resample them to hourly.
3. Record the inferred native timestep distribution so the team can see whether the basin is naturally hourly, sub-hourly, daily, irregular, or sparse.

## Probe window and completeness thresholds

The probe uses one full water year:

- 2023-10-01 to 2024-09-30

This is the minimum useful window for RBI screening.
One month is too seasonal to support a stable flashiness interpretation.

Hourly completeness is computed explicitly against the expected hourly timestamps for that full water year.

Probe readiness thresholds:

- RBI_READY: hourly completeness >= 90%
- PARTIAL_USABLE: hourly completeness between 70% and 90%
- INSUFFICIENT: hourly completeness < 70%

## Fallback behavior

If hourly IV data are not returned directly, the probe will fall back to native or sub-hourly IV and resample or aggregate to hourly.

The probe does not download full long-horizon discharge archives. It only retrieves the one-water-year window needed to validate the method.

The 75-basin probe confirmed the scalable path is native or sub-hourly USGS IV retrieval followed by hourly resampling. Direct hourly retrieval should not be assumed.

## Probe diagnostic findings

The 75-basin probe results were audited to understand:
1. Why 30/75 basins returned NO_DATA
2. Whether hourly resampling and RBI calculation are correct
3. Data completeness and missingness patterns for partial/insufficient basins

### Key insight: Metadata existence ≠ usable data

The availability audit showed that many basins have USGS site records and parameter 00060 metadata. However, metadata overlap with the Flash-NH research period does not guarantee that discharge observations exist in the requested water year (2023-10-01 to 2024-09-30).

The probe confirmed that a significant fraction of basins (39% in the 75-basin sample) have no discharge observations during the screening water year, even though their metadata suggest parameter 00060 exists.

**Implication**: Before scaling RBI retrieval to all 5,836 area-filtered basins, a coverage eligibility stage must filter basins by their metadata overlap with the research period AND screening water year. Basins with only historical metadata should be excluded.

### NO_DATA classification

All 30 NO_DATA basins were rechecked against the USGS IV service during diagnostics. Refined classification shows:
- **historical_only_no_research_overlap** (majority): Sites have 00060 historically but no observations during 2020-10-14 to 2025-12-31; these are legacy/inactive gages.
- **has_research_overlap_but_no_screening_wy_observations**: Sites have 00060 with some overlap to the research period, but no observations during 2023-10-01 to 2024-09-30.
- **has_screening_metadata_but_iv_empty**: Rare; sites catalog metadata suggest overlap but IV queries return no data (usually service issues).
- **invalid_site_or_no_00060**: Site metadata missing or parameter 00060 absent.
- **request_error / parser_error**: Network or parsing failures during probe.

These classifications allow for more informed decisions about basin inclusion/exclusion before attempting RBI download.

### RBI formula verification

The RBI implementation was audited with deterministic unit tests:
- Constant flow series should have RBI = 0: ✓
- Simple pulse series match hand-calculated RBI: ✓
- Series with internal missing hours do not compute false jumps across gaps: ✓
- Unit conversion from cfs to m3/s does not change RBI: ✓

The formula is computed as $\sum |Q_t - Q_{t-1}| / \sum Q_t$ using the final hourly series, with missing internal hours properly excluded.

### Scaling readiness

Before scaling the discharge retrieval workflow to all area-filtered basins (5,836 total):
- **Coverage eligibility filtering must happen first** (see docs/usgs_coverage_eligibility_strategy.md).
- NO_DATA causes are well understood and expected; they should be accepted as part of the workflow.
- RBI must be interpreted only after hourly completeness is high enough (>= 90% for RBI_READY).
- The native/sub-hourly IV retrieval followed by hourly resampling path is confirmed as scalable.
- Historical-only gages (no research-period overlap) should be excluded from the RBI analysis unless a separate historical analysis is intended.

## Why this is useful for scaling

The probe will tell us whether the discharge workflow is practical enough to scale to all area-filtered basins.

If a meaningful fraction of the probe sample reaches RBI_READY or PARTIAL_USABLE status, then the full discharge workflow can be expanded to the entire area-filtered basin universe.

If the probe shows poor coverage or heavy geographic dropout, then the sampling rules, basin universe, or target source assumptions should be revisited before scaling.

The coverage eligibility stage ensures that we only attempt RBI retrieval for basins that have meaningful data availability in the relevant time window.