# Stage 1: RTMA/URMA-Family Precipitation Diagnostic

**Completed:** 2026-06-08  
**Status:** Complete. All three candidates (R02, R06, R11) passed. Diagnostic-only.  
**Scripts:**
- `scripts/discover_rtma_urma_precip_january2023.py`
- `scripts/urma_mrms_timestamp_and_pilot.py`

**Outputs (untracked, diagnostic only):**
`tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/`

---

## 1. Purpose

This was a diagnostic-only follow-up to Stage 1 Milestone 2E (January 2023 event
animations).

The 10 m RTMA wind-vector overlay used in Milestone 2E animations is qualitative
context only — it does not validate rain-cell motion or RTMA/MRMS spatial
consistency in a quantitative way. To increase confidence that the
RTMA/URMA-family grid, basin weight table, and timestamp handling are correct,
we compared URMA QPE precipitation to MRMS QPE precipitation for the three
pilot event candidates (R02, R06, R11) using the same weights already applied
to RTMA extraction in Stage 1.

**This diagnostic does not make URMA precipitation a Stage 1 model input.**  
MRMS remains the precipitation forcing for Stage 1.

---

## 2. Product discovery

### Regular RTMA (Stage 1 product)

Inspection of locally cached Stage 1 RTMA files at the R02/R06/R11 peak hours
confirmed that the regular RTMA analysis product (`2dvaranl_ndfd.grb2_wexp`)
contains **no precipitation field**.

Variables present: `10si`, `10u`, `10v`, `2d`, `2sh`, `2t`, `ceil`, `i10fg`,
`sp`, `tcc`, `vis` — temperature, wind, humidity, pressure, visibility,
cloud ceiling. No `tp`, `apcp`, or any accumulation variable.

### URMA QPE (`pcp_01h.wexp.grb2`)

The `noaa-urma-pds` S3 bucket was probed for January 2023 dates corresponding
to the R02/R06/R11 event windows.

**File pattern:** `urma2p5.{YYYYMMDD}/urma2p5.{YYYYMMDDHH}.pcp_01h.wexp.grb2`

Each probed date contains (per hour):
- `pcp_01h.wexp.grb2` (~0.2–0.4 MB): precipitation field
- `pcp_01h.mask.grb2` (~0.1 MB): quality mask
- `pcp_06h.wexp.grb2` (~0.4–0.6 MB): 6h accumulation
- `2dvaranl_ndfd.grb2_wexp` (~85–86 MB): analysis (T/wind/humidity/etc.)

GRIB inventory of `pcp_01h.wexp.grb2`:

| Field | Value |
|---|---|
| short_name | `tp` |
| grib_name | Total Precipitation |
| units | `kg m**-2` (= mm water-equivalent) |
| step_type | `accum` |
| step_range | `?` (cfgrib limitation — see timestamp section) |
| data_max (sample) | 8.96 mm |
| nonzero fraction | ~7% of CONUS grid |

### RTMA-RU (fallback, not needed)

Files exist under `rtma2p5_ru.` prefix (earliest: 2017-12-13). Not downloaded
or used — URMA QPE was sufficient.

---

## 3. Grid and weight compatibility

| Check | Result |
|---|---|
| URMA grid shape | 1597 x 2345 |
| Regular RTMA grid shape | 1597 x 2345 |
| Shapes match | Yes |
| Lambert Conformal Conic 2.5 km CONUS | Both products |
| lat[0,0] | 19.229 N (both) |
| lat[-1,-1] | 54.373 N (both) |
| lat increases with row | Yes (both) |
| Max lat diff at weight cells | 0.000 deg (exact coincidence) |
| Max lon diff at weight cells | 0.000 deg (exact coincidence) |
| Weight cells out-of-bounds | 0 / 3522 |
| `pilot_rtma_weights.parquet` reusable | Yes |

URMA and regular RTMA are on identical grids. The existing Stage 1 RTMA basin
weight table (`pilot_rtma_weights.parquet`) was reused exactly. No new weight
table was built.

This supports the consistency of RTMA/URMA-family grid indexing and basin-weight
application for Stage 1.

---

## 4. Timestamp convention

cfgrib cannot decode `stepRange` from these GRIB files (`step_range = '?'`).
The 1h accumulation is inferred from the filename suffix `pcp_01h`. The
convention (whether the filename hour is the start or end of the accumulation)
was verified empirically against MRMS.

**Three conventions tested on R02 (21-hour window around Jan 29 08Z peak):**

| Convention | Interpretation | r | RMSE (mm) |
|---|---|---|---|
| **A** | **Filename HH = end of accumulation; URMA[t] vs MRMS[t]** | **0.961** | **1.17** |
| B | Filename HH = start of accumulation; URMA[t] vs MRMS[t+1] | 0.554 | 3.76 |
| C | URMA[t] vs MRMS[t-1] | 0.641 | 3.38 |

**Result: Convention A is clearly best.** Margin over second-best: 0.320 (threshold was 0.05).

- Both URMA and MRMS peaked at **2023-01-29 08Z** — zero hour lag.
- URMA filename HH = end of 1h accumulation, consistent with MRMS `valid_time_utc`.

---

## 5. Three-candidate pilot

Extraction windows:

| Candidate | State | Window |
|---|---|---|
| R02 | AR (STRONG_WET) | Jan 28 18Z – Jan 29 16Z |
| R06 | MN (MODERATE_COLD_REGION) | Jan 03 05Z – Jan 04 05Z |
| R11 | MA (OFFSET_STRESS) | Jan 22 20Z – Jan 23 21Z |

Metrics (Convention A: URMA[t] aligned to MRMS[t]):

| Candidate | r | RMSE (mm) | URMA peak (mm) | MRMS peak (mm) | Interpretation |
|---|---|---|---|---|---|
| R02 | 0.963 | 1.12 | 13.86 | 17.18 | URMA smooths peak relative to MRMS; same event captured |
| R06 | 0.913 | 0.70 | 6.24 | 4.13 | URMA higher than MRMS; snow/mixed-precip context (MN Jan) |
| R11 | 0.944 | 0.39 | 3.59 | 3.69 | Strong agreement despite geometry-stress case |

All three candidates: r > 0.90.

**R02 note**: MRMS is a ~0.01 deg radar composite; URMA is a 2.5 km mesoscale
analysis. URMA smooths spatial extremes, so basin-mean peak underestimate is
expected for small, intense basins.

**R06 note**: MN in January is snow/mixed-precipitation. URMA QPE may accumulate
liquid-equivalent differently from MRMS radar-based QPE in frozen-precip regimes.
The overestimate (6.2 vs 4.1 mm) is worth noting but does not invalidate the
consistency check.

**R11 note**: Near-perfect peak agreement (3.2% difference). Strong result for
an offset-stress geometry candidate.

---

## 6. Limitations

- **URMA QPE is not an independent truth product.** URMA assimilates Stage IV
  and surface gauge data, so it may share information with other precipitation
  analyses. It should not be treated as fully independent from MRMS.

- **This diagnostic does not validate RTMA temperature, wind, humidity, or
  cloud physics.** The only quantity compared is precipitation accumulation.
  RTMA wind/temperature/humidity fields used in Stage 1 are not evaluated here.

- **The timestamp convention (Convention A) was confirmed from R02 only.**
  R06 and R11 were treated as independent validation at the same convention.

- **MRMS remains the Stage 1 precipitation forcing.** URMA precipitation must
  not be added to Stage 1 model inputs. No forcing dataset was modified.

- **cfgrib step_range limitation.** The GRIB `stepRange` field is not decoded
  for URMA pcp_01h.wexp files. The 1h accumulation window is inferred from the
  filename suffix only.

---

## 7. Conclusion

The diagnostic supports the following interpretation:

> RTMA/URMA-family grid indexing, basin weight application, and hourly
> timestamp handling are consistent with MRMS QPE for the January 2023
> pilot event candidates. The Stage 1 RTMA weight table correctly indexes
> the URMA/RTMA 2.5 km CONUS grid, and URMA filename hours align to MRMS
> valid times under Convention A (filename hour = end of accumulation).

This is a consistency check, not a skill validation. The diagnostic is
complete. No further work is needed unless new grid products are introduced.

---

## 8. Files

**Scripts (committed):**
- `scripts/discover_rtma_urma_precip_january2023.py` — product-family discovery
- `scripts/urma_mrms_timestamp_and_pilot.py` — timestamp gate + pilot extraction

**Generated outputs (untracked, diagnostic only):**

```
tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/
  discovery/
    rtma_urma_precip_discovery_report.json
    rtma_urma_precip_discovery_report.md
    rtma_urma_precip_inventory_sample.csv
  part1p5_timestamp_check/
    r02_lag_comparison.csv
    r02_urma_vs_mrms_timeseries.csv
    r02_timeseries_comparison.png
    timestamp_check_report.json
    timestamp_check_report.md
  part2_extraction_pilot/
    urma_basin_means_all_candidates.parquet
    urma_basin_means_preview.csv
    r02_urma_mrms_comparison.png
    r06_urma_mrms_comparison.png
    r11_urma_mrms_comparison.png
    metrics_summary.csv
    metrics_summary.md
    pilot_report.md
  timestamp_and_pilot_summary.json
```
