# Flash-NH — Stage 1 Pilot Plan

Project: Flash-NH is a near-real-time and forecast-aware hydrological modeling pipeline built around basin-average forcing, CAMELSH streamflow/static attributes, NeuralHydrology-based modeling, and staged Stage 1/2/3 experiments.

This document translates [docs/database_design.md](docs/database_design.md) into the first concrete implementation steps for a Stage 1 pilot.

Scope:

- 20-50 basins
- 1 month
- Stage 1 only:
  - MRMS hourly precipitation
  - RTMA hourly meteorology
  - CAMELSH static attributes
  - CAMELSH hourly streamflow target

------------------------------------------------------------
1) Pilot goals
------------------------------------------------------------

The Stage 1 pilot should prove that the proposed database design works end-to-end on a small but realistic subset.

Goals:

- verify the database layout for basin geometry, timeseries, datasets, QC, and manifests
- verify basin geometry and static attribute ingestion
- verify precomputed basin-grid weights for MRMS and RTMA
- verify basin-average extraction using the weights
- verify streamflow alignment to hourly UTC timestamps
- produce QC reports for the pilot subset
- prepare a small NeuralHydrology-compatible dataset that can be loaded by a simple reader

What success looks like:

- the same pilot can be regenerated from raw/sample inputs and manifests
- the Stage 1 data model supports both exploration and model loading
- no manual post-processing is required to consume the pilot dataset

------------------------------------------------------------
2) Required inputs
------------------------------------------------------------

The first pilot depends on the following inputs:

- CAMELSH basin polygons
- CAMELSH static attributes
- CAMELSH hourly streamflow
- MRMS sample files
- RTMA sample files

Implementation note:

- Keep raw files separate from processed outputs. Store the sample inputs under `00_raw/` or an equivalent acquisition area, then derive all pilot outputs from those inputs and recorded manifests.

------------------------------------------------------------
3) Proposed pilot basin selection
------------------------------------------------------------

Recommended subset:

- 20-50 basins total
- include different basin sizes
- include different climates or regions if metadata allows
- include basins with good streamflow completeness
- avoid basins with obvious missing data in the first pilot

Suggested selection logic:

- start from basins with complete or near-complete CAMELSH static attributes and hourly streamflow coverage
- stratify by basin area so the pilot includes small, medium, and large basins
- if regional metadata is available, spread basins across multiple hydroclimatic settings
- manually exclude basins that have obvious geometry or streamflow issues in the first selection pass

Recommended practical approach:

- choose 10-15 basins that are easy to validate visually
- choose 10-15 more basins that add diversity in size and climate
- optionally add another 10-20 basins if the first pass is clean and runtime is acceptable

------------------------------------------------------------
4) Proposed pilot period
------------------------------------------------------------

Recommended duration:

- one month

Suggested starting option:

- January 2023, for consistency with the acquisition tests already used in the repository

Alternative option:

- a wet or eventful month if one is known later and if the goal is to exercise precipitation-streamflow dynamics more aggressively

Tradeoffs:

- January 2023 is convenient because it aligns with existing tests and existing sample artifacts, which makes debugging easier.
- A wetter month may produce more interesting hydrologic behavior and stronger QC signal, but it is harder to compare against existing acquisition tests.

Recommendation:

- use January 2023 for the first pilot unless there is a strong reason to stress-test the pilot with a known wet month.

------------------------------------------------------------
5) Directory layout
------------------------------------------------------------

Expected pilot directories under `Flash-NH_data/`:

- `02_basin_geometries/`
- `03_basin_timeseries/stage1_pilot/`
- `04_ml_datasets/stage1_pilot_v001/`
- `06_qc_reports/stage1_pilot/`
- `09_manifests/stage1_pilot/`

Suggested contents:

- `02_basin_geometries/`
  - canonical basin polygons
  - basin metadata
  - static attributes
  - pilot-specific basin-grid weight tables

- `03_basin_timeseries/stage1_pilot/`
  - long-format and wide-format basin-average timeseries for MRMS and RTMA
  - streamflow-aligned target tables
  - basin-level QC flags

- `04_ml_datasets/stage1_pilot_v001/`
  - frozen model-ready dataset for the first Stage 1 pilot
  - config, manifest, splits, normalization stats, and data files

- `06_qc_reports/stage1_pilot/`
  - missingness summaries
  - value distributions
  - plots for example basins
  - weight-validation plots

- `09_manifests/stage1_pilot/`
  - dataset manifest
  - request specs
  - checksum/provenance files

------------------------------------------------------------
6) Basin-grid weights
------------------------------------------------------------

Implementation steps:

1. Load basin polygons from the canonical CAMELSH basin geometry layer.
2. Load the source grid definitions for MRMS and RTMA.
3. Compute overlap weights between each basin and each source grid cell.
4. Save the weights as Parquet for reproducible reuse.
5. Validate that weights sum to approximately 1 per basin and source.
6. Visualize a few basin-grid overlaps to confirm the geometry is sensible.

Recommended weight table fields:

- `basin_id`
- `source`
- `grid_cell_id`
- `lat`
- `lon`
- `overlap_area`
- `cell_area`
- `weight`

Implementation detail:

- Compute the overlay once and reuse it for every timestep. This turns basin averaging into a weighted sum instead of repeated spatial intersection work.

Validation checks:

- per-basin weight sums are close to 1.0
- no negative weights
- no duplicate basin/grid cell records unless explicitly intended
- a small sample of basins looks correct on a map

------------------------------------------------------------
7) Basin-average extraction
------------------------------------------------------------

Extraction logic:

- for each timestep, source, and variable:
  - read the gridded file
  - apply the precomputed weights
  - compute basin averages
  - write the results to basin-timeseries storage
  - track missing values and QC flags

Practical rules:

- keep source-specific time and coordinate handling explicit
- preserve source metadata and timestamps in the output
- do not collapse all QC information away during the first pilot

Suggested output behavior:

- store one record per basin/time/source/variable in long form
- optionally build a wide-form table from the same source for model loading
- record whether each basin/time/variable was computed fully, partially, or with missing inputs

QC flags should capture at least:

- complete
- partial
- missing source input
- empty basin overlap
- unexpected units or bounds

------------------------------------------------------------
8) Streamflow/static alignment
------------------------------------------------------------

Alignment logic:

1. Load CAMELSH streamflow for the selected basins.
2. Align streamflow timestamps to UTC hourly timestamps.
3. Load CAMELSH static attributes.
4. Join streamflow and static attributes by `basin_id`.
5. Report missing streamflow coverage before dataset freezing.

Practical checks:

- confirm that hourly timestamps are strictly monotonic per basin
- confirm that the streamflow target range matches the chosen pilot month
- confirm that static attributes are present for every pilot basin
- confirm that basin identifiers match across geometry, static, and streamflow sources

Report missing streamflow explicitly:

- count missing hours by basin
- count missing basins by file or by subregion if applicable
- note any basins that should be excluded from the first pilot because alignment is incomplete

------------------------------------------------------------
9) Output schemas
------------------------------------------------------------

Expected Parquet schemas:

A) Long basin timeseries

- `basin_id`
- `time`
- `source`
- `variable`
- `value`
- `qc_flag`

B) Wide basin timeseries

- `basin_id`
- `time`
- one column per standardized variable and source combination where needed
- `qc_flag` or companion QC columns

C) Static attributes

- `basin_id`
- CAMELSH static variables
- geometry or metadata keys as needed

D) Target streamflow

- `basin_id`
- `time`
- `streamflow`
- `qc_flag`
- optional join keys or source metadata

Recommended implementation detail:

- keep the schemas stable from the first pilot onward so that later Stage 1 expansion can be done by appending data rather than redesigning file shapes.

------------------------------------------------------------
10) QC reports
------------------------------------------------------------

First-pilot QC outputs should include:

- missingness by basin and source
- value distributions
- time-series plots for 5 example basins
- MRMS precipitation vs streamflow lag examples
- RTMA variable sanity checks
- basin weight validation plots

Suggested examples:

- pick 5 basins that span the pilot subset by size and region
- include at least one basin with strong storm response and one with quieter behavior if possible

Useful report types:

- basin coverage table
- hourly completeness heatmap
- min/max sanity checks per variable
- selected time-series panels for manual review
- weight-sum diagnostics for MRMS and RTMA

------------------------------------------------------------
11) NeuralHydrology compatibility
------------------------------------------------------------

Files/configs needed to feed the pilot to NeuralHydrology:

- basin list
- dynamic inputs
- static attributes
- target streamflow
- train/val/test split for the pilot
- normalization stats

Recommended pilot-compatible packaging:

- a frozen dataset folder under `04_ml_datasets/stage1_pilot_v001/`
- a `dataset_config.yaml` file describing the file paths and variable names
- a `manifest.json` file describing provenance and checksums
- explicit split files for the selected basins and date window

Compatibility goal:

- the first pilot should be loadable by a simple Python reader before integrating deeply with any NeuralHydrology training code

Potential implementation choices:

- either follow CAMELS-style file conventions where practical
- or implement a small custom dataset adapter if the current schema is closer to the Flash-NH database layout

------------------------------------------------------------
12) Success criteria
------------------------------------------------------------

Pass/fail criteria for the Stage 1 pilot:

Pass if:

- all selected basins have geometry, static attributes, and streamflow coverage
- weights exist and sum correctly for MRMS and RTMA
- basin-average files are produced for the full month
- timestamps align correctly to hourly UTC
- QC reports are generated
- the pilot dataset can be loaded by a simple Python reader

Fail if:

- there are missing geometry or static joins for many selected basins
- weights do not sum to approximately 1.0
- basin-average outputs are incomplete or not reproducible
- streamflow timestamps do not align
- QC artifacts are missing
- the frozen pilot dataset cannot be read cleanly

Recommended practical acceptance threshold:

- the pilot should be clean enough that a second engineer can reproduce it from the manifest without asking for manual cleanup steps

------------------------------------------------------------
13) Open questions
------------------------------------------------------------

Unresolved issues to settle during implementation:

- exact CAMELSH file locations
- basin id conventions
- CRS/projection handling
- MRMS and RTMA grid coordinate definitions
- whether NeuralHydrology expects CAMELS-style files or a custom dataset class in this repository

Additional notes:

- keep the pilot design narrow enough that the first run can be debugged end-to-end
- do not expand to Stage 2 or Stage 3 until the Stage 1 pilot is reproducible

------------------------------------------------------------
Next implementation steps
------------------------------------------------------------

1. Confirm the exact basin subset and month.
2. Define the pilot manifest and directory skeleton.
3. Implement geometry loading and weight computation for MRMS and RTMA.
4. Run basin-average extraction for the pilot window.
5. Align CAMELSH streamflow and static attributes.
6. Freeze the first `stage1_pilot_v001` dataset.
7. Produce QC reports and verify a minimal loader can read the dataset.
