# Stage 1 Milestone 2E: January 2023 Event/Window Visual QC

**Generated:** 2026-06-05  
**Refined:** 2026-06-05  
**Cleaned up:** 2026-06-07  
**Status:** v2.1-stable pilot (R02, R06, R09, R11) approved; pipeline cleanup complete; all-12 run ready but not yet executed

---

## 1. Streamflow Discovery

### Sources Checked

| Source | Coverage | Usable for Jan 2023? |
|---|---|---|
| `reports/flashnh_usgs_event_hydrograph_review_v001/hourly_series/` (10 files) | WY2024 only: 2023-10-01 – 2024-09-30 | **NO** |
| `reports/flashnh_usgs_event_hydrograph_review_v002/hourly_series/` (110 files) | WY2024 only: 2023-10-01 – 2024-09-30 | **NO** |
| `reports/flashnh_usgs_rbi_screening_wy2024_v001/` | No hourly_series directory | **NO** |
| **CAMELSH hourly NetCDF** (5,767 files, 1980–2024) | Full period including Jan 2023 | **YES** |

### CAMELSH Source Used

- Path: `C:\PhD\Python\neuralhydrology\US_data\data_download\CAMELSH_resolution_test\data\raw\camelsh\`
- Format: NetCDF4, variable `streamflow`, units `m3 s-1`
- Time axis: `hours since 1980-01-01 00:00:00` (UTC)
- January 2023 coverage: 744 hours (complete)
- No new USGS/NWIS data was downloaded

### Pilot Basin Coverage

Of the 50 pilot STAIDs:

| Status | Count |
|---|---|
| CAMELSH file found, full coverage (744/744 h) | 20 |
| CAMELSH file found, partial coverage | 8 |
| No CAMELSH file | 22 |
| **Usable for Jan 2023 event QC** | **28** |

STAIDs without CAMELSH files (excluded from this QC):  
`10164500`, `02344605`, `03298135`, `02077670`, `07283000`, `02073000`,  
`02403310`, `03305000`, `02235000`, `11372000`, `02072500`, `02344700`,  
`07103700`, `02146381`, `01585200`, `01586210`, `02484000`, `02301000`,  
`02266480`, `02266500`, `02264100`, `10336700`

---

## 2. Event Selection Logic

### Data Inputs

- **MRMS forcing**: `combined_hourly_basin_stats.parquet`, product `mrms_qpe_1h_pass1`, weighted basin-mean (mm/h)
- **RTMA 2t**: product `rtma_conus_aws_2p5km`, variable `2t`, weighted_mean (K)
- **Streamflow**: CAMELSH hourly NetCDF (m3/s)
- **Basin metadata**: `pilot_basin_manifest.csv` (DRAIN_SQKM, candidate_class, RBI, STATE)
- **Gauge offsets**: `gauge_polygon_distance_audit.json`

### Window Definition

- Anchor: time of maximum 6-hour rolling MRMS precipitation
- Window: 24 h before anchor to 48 h after anchor (72 h total, capped to Jan bounds)
- Frame cadence: hourly

### Selection Categories

| Category | Target Count | Selection Criterion |
|---|---|---|
| STRONG_WET | 3–4 | Highest `flash_score = max_6h_precip * min(sf_rise_ratio, 20)` with rise > 0.5 m3/s |
| MODERATE | 2 | max_6h 2–15 mm, moderate flow rise |
| COLD_SNOW_RISK | 2 | Lowest `min_rtma_2t_K` with any precipitation |
| DRY_CONTROL | 2 | Lowest `max_24h_precip_mm` overall |
| OFFSET_STRESS | 2 | Largest gauge-polygon offset (>500 m), sufficient streamflow data |

---

## 3. Candidate Events

**12 candidates selected** from 28 basins with January 2023 streamflow.

| ID | STAID | State | HUC02 | Category | max_6h (mm) | Q_peak (m3/s) | Rise ratio | min_2t (C) | Area (km2) | Offset (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| C01_02411930 | 02411930 | GA | 03 | STRONG_WET | 51.5 | 136.5 | 21.0x | +1.0 | 705 | 230 |
| C02_07263580 | 07263580 | AR | 11 | STRONG_WET | 50.7 | 124.9 | 195.1x | -1.4 | 50 | 0 |
| C03_03318800 | 03318800 | KY | 05 | STRONG_WET | 45.8 | 170.5 | 101.5x | +1.1 | 304 | 244 |
| C04_07024500 | 07024500 | TN | 08 | STRONG_WET | 38.6 | 126.6 | 18.8x | +1.8 | 991 | 247 |
| C05_05560500 | 05560500 | IL | 07 | MODERATE | 14.2 | 1.7 | 5.5x | -1.3 | 71 | 0 |
| C06_05372995 | 05372995 | MN | 07 | MODERATE | 14.0 | 3.5 | 2.3x | -6.9 | 779 | 0 |
| C07_10348850 | 10348850 | NV | 16 | COLD_SNOW_RISK | 22.5 | 0.24 | 1.3x | -7.9 | 19 | 0 |
| C08_13112000 | 13112000 | ID | 17 | COLD_SNOW_RISK | 9.8 | 0.00 | n/a | -6.6 | 947 | 0 |
| C09_01662800 | 01662800 | VA | 02 | DRY_CONTROL | 10.4 | 1.0 | 2.9x | -2.1 | 67 | 1036 |
| C10_04111379 | 04111379 | MI | 04 | DRY_CONTROL | 6.6 | 2.6 | 1.8x | -1.3 | 427 | 396 |
| C11_01100627 | 01100627 | MA | 01 | OFFSET_STRESS | 30.3 | 17.0 | 1.8x | -1.8 | 173 | 4004 |
| C12_01390450 | 01390450 | NJ | 02 | OFFSET_STRESS | 32.0 | 19.2 | 46.4x | -2.5 | 29 | 3023 |

### Candidate Narratives

**C01 — 02411930 (GA) | STRONG_WET**  
Large Georgia basin (705 km2), January 3–6. MRMS records 51.5 mm in 6 h (85.6 mm/24 h). Flow rises from ~6.5 to 136 m3/s (21x) with a 14-hour lag. Temperature stays well above freezing (min +1 C, median +15 C) — clear liquid precipitation response. Gauge within 230 m of polygon boundary.

**C02 — 07263580 (AR) | STRONG_WET**  
FLASHY_CORE small basin (50 km2) in Arkansas, January 29–31. Extraordinary rise ratio: 0.64 to 125 m3/s (195x) in under 1 hour after precipitation peak. Highest 1-hour intensity of any candidate: 17.2 mm/h. Near-freezing minimum (-1.4 C). Gauge inside polygon (offset=0). A core test of the pipeline's ability to resolve rapid flashy response.

**C03 — 03318800 (KY) | STRONG_WET**  
Kentucky basin (304 km2), January 2–5. Highest absolute peak discharge of all candidates: 170 m3/s. 6 h precip = 45.8 mm, 3-hour lag to peak. Partial CAMELSH coverage (690/744 h; 54 h missing, likely early-Jan gap). Temperature above 0 C throughout.

**C04 — 07024500 (TN) | STRONG_WET**  
Largest strong-wet basin (991 km2, Tennessee, HUC08), January 2–5. Moderate rise ratio (18.8x) but large absolute response: peak 127 m3/s. Good geographic diversity (southern Appalachian / Tennessee River system).

**C05 — 05560500 (IL) | MODERATE**  
Small Illinois basin (71 km2), modest MRMS (14.2 mm/6 h), subdued flow rise. Useful for verifying the pipeline produces sensible, non-zero outputs for weak-forcing cases.

**C06 — 05372995 (MN) | MODERATE**  
Large Minnesota basin (779 km2), winter conditions (min -6.9 C). Interesting borderline snow/rain transition case with non-trivial 14 mm/6 h forcing. Note: partial CAMELSH coverage (663/744 h).

**C07 — 10348850 (NV) | COLD_SNOW_RISK**  
Smallest basin in the candidate set (19 km2, Nevada), coldest minimum temperature (-7.9 C). Flow barely above zero (max 0.24 m3/s) under 22.5 mm/6 h precipitation — likely mostly snow accumulation. Good test of cold-climate RTMA visualization.

**C08 — 13112000 (ID) | COLD_SNOW_RISK**  
Large Idaho basin (947 km2), sustained zero streamflow throughout January 2023 despite 9.8 mm/6 h forcing — frozen / snow-dominated. Pure dry/cold control for the animation: all forcing signal, no hydrograph response.

**C09 — 01662800 (VA) | DRY_CONTROL**  
Virginia basin (67 km2), relatively low precipitation for January 2023 (max 6h=10.4 mm). Also functions as an OFFSET_STRESS test: gauge is 1036 m from polygon boundary (OFFSET_GT_1KM). Double-purpose: dry forcing + geometry verification.

**C10 — 04111379 (MI) | DRY_CONTROL**  
Michigan basin (427 km2), lowest precipitation among all candidates (max 6h=6.6 mm). Stable, low-flashiness flow. Geographic diversity (Great Lakes region).

**C11 — 01100627 (MA) | OFFSET_STRESS**  
Massachusetts basin (173 km2), largest gauge offset in the pilot set: **4004 m**. Despite the offset, MRMS records 30.3 mm/6 h and flow rises to 17 m3/s. Primary test of whether the polygon-based forcing extraction is coherent when the gauge is far from the polygon centroid.

**C12 — 01390450 (NJ) | OFFSET_STRESS**  
Small New Jersey basin (29 km2), second-largest offset: 3023 m. Dramatic rise ratio (46x) makes it interesting for hydrograph QC. Gauge well outside polygon — animation will visually expose any spatial misregistration.

### Caveats

- "DRY_CONTROL" windows still had measurable precipitation (6–10 mm/6 h) because January 2023 was broadly wet across the CONUS east. These are the driest available windows from the 28 pilot basins — not precipitation-free.
- C03 (KY) and C06 (MN) have partial CAMELSH coverage; missing hours will appear as gaps in the hydrograph panel.
- C08 (ID, zero flow) will produce a flat streamflow panel — intended; it is a snow-accumulation control.
- 22 pilot basins lack CAMELSH files and are excluded from this QC. They can be added in a future iteration if January 2023 USGS IV data is downloaded.
- The forcing data covers all 50 pilot basins; only streamflow limits the candidate set to 28.

---

## 4. Static Preview Plots

One 72-hour 3-panel PNG per candidate:
- Top: observed streamflow (blue line), flow peak (red dashed), precip peak (green dotted)
- Middle: MRMS hourly basin precipitation bars
- Bottom: RTMA 2m temperature line

Output directory:  
`tmp/stage1_pilot_dryrun/06_qc_reports/stage1_pilot/january_2023_event_qc/candidate_previews/`

Contact sheet (all 12 candidates):  
`tmp/stage1_pilot_dryrun/06_qc_reports/stage1_pilot/january_2023_event_qc/candidate_preview_contact_sheet.png`

---

## 5. Accepted v2.1-stable Animation Design

**Design accepted 2026-06-05. Pilot (R02, R06, R09, R11) generated and reviewed.**

### Layout

```
+------------------------------------------+---------------------------+
|  MAP PANEL (geographic, Plate Carree)    |  STREAMFLOW PANEL         |
|                                           |  Q (m3/s) — blue line     |
|  MRMS QPE 1h raster (custom blue-purple) |                           |
|  RTMA 10m wind quiver (~10 arrows)       +---------------------------+
|  MRMS extraction cells (orange squares)  |  PRECIP PANEL             |
|  Basin polygon (MRMS-colored edge)       |  Basin-mean MRMS bars     |
|  Basin-mean label at polygon centroid    |                           |
|  Gauge marker (yellow star, z=12)        +---------------------------+
|  Gauge→centroid offset arrow (if >1km)   |  TEMPERATURE PANEL        |
|  State boundaries                        |  RTMA 2m T (C) — orange   |
|  Scale bar                               |                           |
|  Timestamp + frame counter               |  Vertical red cursor sync |
|  Colorbar (vmax = 2x event peak)        |  across all 3 panels      |
+------------------------------------------+---------------------------+
```

### Parameters

| Parameter | Value |
|---|---|
| Window | 72 hours per candidate |
| Frame cadence | Hourly (1 frame = 1 hour) |
| Frames per animation | 72 |
| FPS | 4 (~18 s per animation) |
| Output | GIF (PillowWriter; MP4 available after `winget install Gyan.FFmpeg`) |
| Figure size | 14.5 × 5.8 in, DPI 90 |
| Map CRS | Plate Carree (geographic, no projection) |
| Precip colormap | Custom blue-purple (WET_CMAP) |
| MRMS vmax | `max_1h_precip_mm × 2.0` (per-candidate) |
| Map extent | Basin-size-aware pad (R02=0.10°, R06=0.35°, R09=0.22°, R11=0.30°) |

### Grid Conventions

**MRMS MultiSensor QPE 1H Pass1:**
- 3500 rows × 7000 cols, 0.01° resolution, CONUS
- Filename timestamp = grib `valid_time` = **END** of 1-hour accumulation
- Latitude **DECREASES** with row; row 0 = 54.995°N (northernmost)
- Row formula: `row = (54.995 − lat) / 0.01`
- imshow: `origin="upper"`, extent `[lon_min, lon_max, lat_bottom, lat_top]`

**RTMA CONUS 2.5km:**
- 1597 rows × 2345 cols, Lambert Conformal Conic, curvilinear 2D lat/lon
- Latitude **INCREASES** with row; row 0 = SW corner (~19.23°N)
- lon stored as 0–360; convert to −180/180 for geographic mapping
- Direct index: `data[row_idx, col_idx]` (no formula needed)
- cfgrib → parquet name mapping: `t2m`→`2t`, `u10`→`10u`, `v10`→`10v`

### RTMA 10m Wind Caveat

RTMA 10m wind vectors shown on the map are **qualitative meteorological context and
spatial QC only**. They should **not** be interpreted as strict storm-steering
validation for MRMS rain-cell motion. MRMS and RTMA are different products with
different assumptions and represent different atmospheric quantities.

### State Boundary Reference Layer

State boundaries are **optional cartographic context**, not a scientific forcing or
input dependency. The map panel draws them when available; if the file is absent the
animation generation continues without them and the manifest records the outcome.

| Field | Value |
|---|---|
| Expected cached file | `tmp/…/02_basin_geometries/reference/ne_110m_admin1_us_states.gpkg` |
| Source | Natural Earth 110m admin-1, US states only (`iso_a2 == "US"`) |
| Created by | `extract_stage1_one_hour.py` bootstrap (downloads once, caches locally) |
| Manifest key | `state_boundaries_status`: `loaded` / `downloaded` / `skipped_missing` |

If the cached file is missing, `generate_january_event_animations.py` automatically
attempts a one-time CDN download and re-caches it. If that also fails, it prints a
warning and continues without state boundaries.

**Lessons-learned note (2026-06-07):** The initial all-12 run failed because the stable
script used the wrong cache filename (`ne_110m_admin_1_states_provinces.gpkg`, the
Natural Earth CDN zip name) instead of the project's cached name
(`ne_110m_admin1_us_states.gpkg`). The four v2.1 pilot animations were unaffected
because they used the correct name. After any cleanup or script renaming, run a
single-candidate smoke test (`--candidates R02`) before proceeding to all 12 — audit
scripts alone do not exercise the animation rendering path.

### Static Reference Frames

Saved per candidate in `{candidate}/static_frames/`:
- `{RID}_first.png` — frame 0 (window start / dry baseline)
- `{RID}_peak_precip.png` — frame at maximum basin-mean MRMS
- `{RID}_peak_flow.png` — frame at maximum streamflow
- `{RID}_recession.png` — frame at ~75% through window

### Audit Gates (must PASS before animation generation)

```bash
python scripts/audit_rtma_spatial_alignment.py   # 8/8 PASS required
python scripts/audit_january_event_animation_sync.py  # 10/10 PASS required
```

### Script and Commands

Stable script: `scripts/generate_january_event_animations.py`

```bash
# Run 4-candidate pilot (default):
python scripts/generate_january_event_animations.py

# Run specific candidates:
python scripts/generate_january_event_animations.py --candidates R02 R11

# Run all 12 (after pilot inspection approval):
python scripts/generate_january_event_animations.py --all
```

Output directory: `tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/`

---

## 6. Files Created

### Scripts

| File | Purpose |
|---|---|
| `scripts/select_january_qc_events.py` | Streamflow discovery, metric computation, original candidate selection, static plots |
| `scripts/refine_january_qc_events.py` | Candidate refinement: dry-window scan, C11 recentering, relabeling, refined plots |
| `scripts/generate_january_event_animations.py` | v2.1-stable animation generator; supports `--all` for all 12 candidates |
| `scripts/audit_rtma_spatial_alignment.py` | RTMA spatial audit gate (8/8 PASS verified); exits 1 on any mismatch |
| `scripts/audit_january_event_animation_sync.py` | MRMS raster vs parquet sync audit (10/10 PASS verified); exits 1 on mismatch |

### Manifest outputs (original)

| File | Description |
|---|---|
| `tmp/.../january_2023_event_qc/streamflow_discovery_report.json` | Full per-basin CAMELSH coverage inventory |
| `tmp/.../january_2023_event_qc/streamflow_discovery_report.md` | Human-readable discovery summary |
| `tmp/.../january_2023_event_qc/event_animation_candidates.csv` | Original 12-row candidate table |
| `tmp/.../january_2023_event_qc/event_animation_candidates.md` | Original candidate table + per-event detail |

### Manifest outputs (refined)

| File | Description |
|---|---|
| `tmp/.../january_2023_event_qc/event_animation_candidates_refined.csv` | **Refined** 12-row candidate table |
| `tmp/.../january_2023_event_qc/event_animation_candidates_refined.md` | Refined candidate table + per-event detail |
| `tmp/.../january_2023_event_qc/candidate_refinement_report.md` | Decisions: retained/replaced/relabeled + dry-threshold docs |

### QC report outputs (original)

| File | Description |
|---|---|
| `tmp/.../january_2023_event_qc/candidate_previews/C01_*.png` … `C12_*.png` | 12 original preview plots |
| `tmp/.../january_2023_event_qc/candidate_preview_contact_sheet.png` | Original contact sheet |

### QC report outputs (refined)

| File | Description |
|---|---|
| `tmp/.../candidate_previews_refined/R01.png` … `R12.png` | 12 refined preview plots (color-coded by category) |
| `tmp/.../candidate_preview_contact_sheet_refined_page01.png` | Refined contact sheet page 1 (R01–R06) |
| `tmp/.../candidate_preview_contact_sheet_refined_page02.png` | Refined contact sheet page 2 (R07–R12) |

---

## 7. Candidate Refinement — 2026-06-05

### Why C09 and C10 Were Not Accepted as Dry Controls

The original C09 (01662800, VA) and C10 (04111379, MI) were both labeled DRY_CONTROL
but neither satisfied the definition:

| | C09 (01662800) | C10 (04111379) |
|---|---|---|
| max_1h_precip | 3.65 mm | 2.54 mm |
| max_6h_precip | **10.4 mm** | **6.6 mm** |
| sf_rise_ratio | 2.9x | 1.78x |
| Verdict | Moderate wet event | Moderate wet event |

The bug was in window selection: the original algorithm anchored on the highest 6h
precipitation peak across the whole month, which for these basins happened to fall in
a wet window, even though the category requested dry.

**Resolution:** A sliding-window scan across all 72h windows in January 2023 found that
every one of the 28 CAMELSH-covered pilot basins has at least one window with exactly
0.0 mm MRMS precipitation. The revised dry controls use:

- **R09** — STAID 13239000, Jan-01 00Z to Jan-03 23Z: total_72h = 0.0 mm, cv = 0.032
- **R10** — STAID 04111379 (same as old C10), Jan-07 03Z to Jan-10 02Z: total_72h = 0.0 mm, cv = 0.082

Both satisfy the strict threshold: **max_6h = 0.0 mm, total_72h = 0.0 mm**.

### How Cold and Zero-Flow Candidates Should Be Interpreted

**COLD_PRECIP_LOW_RESPONSE (R07 — 10348850, NV)**  
Precipitation occurred (22.5 mm/6h) but streamflow response was negligible (0.06 m3/s
rise). The dominant process is likely snow accumulation rather than liquid runoff.
Animation value: tests whether the pipeline correctly shows forcing inputs decoupled
from flow response under cold conditions. Do not interpret as a wet event.

**ZERO_FLOW_FROZEN_CONTROL (R08 — 13112000, ID)**  
MRMS records up to 9.8 mm/6h but streamflow = 0.00 m3/s throughout all of January 2023.
The large Idaho basin (947 km2) is fully frozen / snow-dominated. Animation value: the
hydrograph panel will be a flat zero line — intentional. Confirms the pipeline does not
hallucinate flow when no gauge signal exists.

### How Offset-Stress Candidates Should Be Interpreted

Both R11 and R12 have gauges located far outside their CAMELSH polygons:

| ID | STAID | Offset | Status |
|---|---|---|---|
| R11 | 01100627 (MA) | **4004 m** | OFFSET_GT_1KM |
| R12 | 01390450 (NJ) | **3023 m** | OFFSET_GT_1KM |

In animation, the map panel will show the MRMS precipitation grid over the polygon
centroid, while the gauge marker will appear spatially displaced. This visualizes the
mismatch directly and allows the user to judge whether the polygon-derived forcing is
physically representative for that gauge.

**C11 (original) was recentered:** The original Jan 25-28 window for 01100627 had a
47-hour precip-to-flow lag, placing the flow peak at the very last timestamp of the
72-hour window. A scan of all valid windows found a clean Jan-21 storm (Jan-21 06Z to
Jan-24 05Z) with a 2-hour lag and the flow peak at 74.6% of the window, leaving ~19
hours of visible recession.

### Refined Final Candidate Table

| ID | STAID | State | Category | max_6h (mm) | Q_peak (m3/s) | Lag (h) | Peak pos | min 2t (C) | Area (km2) | Offset (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | 02411930 | GA | STRONG_WET | 51.5 | 136.5 | 14 | 49% | +1.0 | 705 | 230 |
| R02 | 07263580 | AR | STRONG_WET | 50.7 | 124.9 | 1 | 34% | -1.4 | 50 | 0 |
| R03 | 03318800 | KY | STRONG_WET | 45.8 | 170.5 | 3 | 37% | +1.1 | 304 | 244 |
| R04 | 07024500 | TN | STRONG_WET | 38.6 | 126.6 | 10 | 41% | +1.8 | 991 | 247 |
| R05 | 05560500 | IL | MODERATE_SMALL_BASIN | 14.2 | 1.71 | 6 | 38% | -1.3 | 71 | 0 |
| R06 | 05372995 | MN | MODERATE_COLD_REGION | 14.0 | 3.54 | 4 | 51% | -6.9 | 779 | 0 |
| R07 | 10348850 | NV | COLD_PRECIP_LOW_RESPONSE | 22.5 | 0.24 | 6 | 37% | -7.9 | 19 | 0 |
| R08 | 13112000 | ID | ZERO_FLOW_FROZEN_CONTROL | 9.8 | 0.00 | — | — | -6.6 | 947 | 0 |
| R09 | 13239000 | — | DRY_CONTROL | **0.0** | 1.21 | — | — | — | — | 0 |
| R10 | 04111379 | MI | DRY_CONTROL | **0.0** | 1.97 | — | — | — | 427 | 396 |
| R11 | 01100627 | MA | OFFSET_STRESS | 15.3 | 8.30 | 2 | **75%** | -1.8 | 173 | **4004** |
| R12 | 01390450 | NJ | OFFSET_STRESS | 32.0 | 19.2 | 5 | 34% | -2.5 | 29 | **3023** |

---

## 8. Next Action

**Pilot (R02, R06, R09, R11) reviewed and approved 2026-06-05. Pipeline cleanup complete 2026-06-07.**

Stable scripts written and validated:
- `scripts/generate_january_event_animations.py` — main animation generator (v2.1-stable)
- `scripts/audit_rtma_spatial_alignment.py` — RTMA spatial audit gate (8/8 PASS)
- `scripts/audit_january_event_animation_sync.py` — MRMS sync audit gate (10/10 PASS)

All-12 command (run after manual inspection confirms pilot animations are acceptable):
```bash
python scripts/audit_rtma_spatial_alignment.py       # must PASS
python scripts/audit_january_event_animation_sync.py  # must PASS
python scripts/generate_january_event_animations.py --all
```

Outputs go to: `tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/`

Estimated runtime (local, no ffmpeg): ~9 min per 4 candidates → ~27 min for all 12.
Install ffmpeg for MP4 output: `winget install Gyan.FFmpeg` (adds bin/ to PATH).
