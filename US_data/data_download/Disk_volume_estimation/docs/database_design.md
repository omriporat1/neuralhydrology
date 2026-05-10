# Flash-NH — Database Design

Project: Flash-NH is a near-real-time and forecast-aware hydrological modeling pipeline built around basin-average forcing, CAMELSH streamflow/static attributes, NeuralHydrology-based modeling, and staged Stage 1/2/3 experiments.

This document describes the planned data organization for Flash-NH: what data exists, where it should be stored, how raw files become model-ready datasets, and how exploration, training, validation, testing, and backups are organized.

------------------------------------------------------------
1) Purpose
------------------------------------------------------------

Flash-NH requires a database structure that supports:

- exploratory analysis and caveat detection
- reproducible preprocessing
- efficient basin-average feature generation
- clean train/validation/test separation
- NeuralHydrology-compatible model inputs
- future reuse and reliable backup

The database should separate raw acquisition from standardized gridded products, basin aggregation, and frozen ML datasets so that each stage is auditable and reproducible.

------------------------------------------------------------
2) Repository strategy
------------------------------------------------------------

For now:

- Continue working in the current repository while the data/schema stabilizes and small pilots are run.

Later (recommended split if helpful):

- `flash-nh-data`
  - Purpose: acquisition, audits, standardization, basin aggregation, ML dataset generation.

- `flash-nh-models` (or a NeuralHydrology fork)
  - Purpose: model training, evaluation, experiment tracking, predictions.

Notes:

- Data engineering should not be permanently buried inside a modeling fork. Keep data logic and engineering in `flash-nh-data` (or the main data repo) and use NeuralHydrology as the modeling framework/dependency.

------------------------------------------------------------
3) Proposed top-level data directory structure
------------------------------------------------------------

Canonical project data root: `Flash-NH_data/` (this is a conceptual layout; storage may be local, network, or cloud-backed).

Flash-NH_data/
  00_raw/
  01_standardized_grids/
  02_basin_geometries/
  03_basin_timeseries/
  04_ml_datasets/
  05_splits/
  06_qc_reports/
  07_experiments/
  08_logs/
  09_manifests/
  tmp/

Directory descriptions:

- `00_raw/`
  - Purpose: store original downloaded files (GRIB, NC4, other vendor files).
  - Contents: vendor filenames, organized by source/date/collection.
  - Backup: optional (raw is often redownloadable); consider long-term cold storage if retention is required.
  - Git: never commit large raw files; use manifests and request specs only.

- `01_standardized_grids/`
  - Purpose: store standardized, consistent-gridded versions of each product (Zarr recommended).
  - Contents: per-source zarr stores or directory-per-run containing standardized variable names, units, and consistent coords.
  - Backup: yes (if regeneration is expensive); consider incremental snapshots.
  - Git: do not commit; commit only small example zarr metadata or pointers.

- `02_basin_geometries/`
  - Purpose: canonical basin polygons and static attributes.
  - Contents: GeoPackage `camelsh_basins.gpkg`, `basin_metadata.parquet`, `static_attributes.parquet`, and `weights/` subfolders with precomputed per-source weights.
  - Backup: YES; small and critical.
  - Git: commit only schema, example basins, and scripts that generate these files; not large geometry files unless tiny.

- `03_basin_timeseries/`
  - Purpose: aggregated basin-average timeseries for exploration and QC.
  - Contents: both long-form and wide-form parquet time series, per-source and cross-source aggregates.
  - Backup: recommended (expensive to regenerate).
  - Git: do not commit large datasets; commit small examples and schema.

- `04_ml_datasets/`
  - Purpose: frozen model-ready datasets (immutable versions).
  - Contents: versioned dataset folders (see section 8) with manifests and configs.
  - Backup: HIGH priority.
  - Git: only commit `dataset_config.yaml` and `manifest.json` (not the large parquet files unless exceptionally small).

- `05_splits/`
  - Purpose: store split definitions and fold lists.
  - Contents: `train_basins.txt`, `val_basins.txt`, `test_basins.txt`, `time_splits.yaml`.
  - Backup: YES.
  - Git: commit splits to track reproducibility.

- `06_qc_reports/`
  - Purpose: store automated QC reports, HTML previews, PNGs.
  - Contents: per-source and per-basin QC outputs and summaries.
  - Backup: medium; reports can be regenerated but are useful to keep.
  - Git: commit small example reports; archive large report sets externally.

- `07_experiments/`
  - Purpose: local experiment artifacts when not using an experiment manager.
  - Contents: per-experiment folders with `config.yaml`, `metrics.csv`, `predictions.parquet`, `model_checkpoint.*`, `plots/`.
  - Backup: important for reproducibility; backup summaries and configs first.
  - Git: commit configs and small plots only.

- `08_logs/`
  - Purpose: operational logs, acquisition transcripts, and job output.
  - Contents: ingestion transcripts, audit logs, pipeline run logs.
  - Backup: low–medium; keep recent logs and critical transcripts.
  - Git: do not commit logs.

- `09_manifests/`
  - Purpose: manifest files describing dataset contents and provenance.
  - Contents: `manifest.json` files per dataset, request specs, and checksums.
  - Backup: YES and commit small manifests.
  - Git: commit manifests and small provenance files.

- `tmp/`
  - Purpose: scratch and temporary intermediate data.
  - Contents: ephemeral cache files.
  - Backup: NO.
  - Git: do not commit.

------------------------------------------------------------
4) Storage formats
------------------------------------------------------------

- Original GRIB/NC4
  - Description: raw vendor files as downloaded. Keep originals in `00_raw/` if retention required.

- Zarr
  - Description: chunked, cloud-friendly multidimensional storage ideal for large gridded fields. Use Zarr for `01_standardized_grids/`.

- GeoPackage (.gpkg)
  - Description: a portable SQLite-based vector format for basin polygons and other geospatial layers. Store canonical basins in `02_basin_geometries/camelsh_basins.gpkg`.

- Parquet
  - Description: columnar, compressed, efficient tabular storage for basin-average time series and ML-ready inputs. Use Parquet for `03_basin_timeseries/` and `04_ml_datasets/`.

- YAML/JSON
  - Description: human-readable configs, manifests, request specs, and split definitions. Store in `09_manifests/` and dataset folders.

- CSV
  - Description: small summaries and quick tables. Use sparingly for publishable CSV summaries.

- HTML/PNG
  - Description: QC reports and visualizations; store under `06_qc_reports/`.

Simple explanations:

- Parquet is a binary columnar format optimized for analytics; it supports efficient selective reads and compression.
- GeoPackage is a single-file SQLite-based container for vector geospatial layers (polygons, points) and their attributes.

------------------------------------------------------------
5) Standardized grids
------------------------------------------------------------

Standardization does not mean resampling everything to one resolution. Each product should keep the resolution that preserves its scientific value while standardizing naming, units, coordinates, and metadata.

Examples:

- MRMS: high-resolution hourly precipitation
- RTMA: high-resolution hourly meteorology
- GFS: 0.25° forecast fields
- IFS: 0.1° forecast fields
- ERA5-Land: daily antecedent variables
- GDAS: daily antecedent variables
- IMERG Late Daily: daily precipitation

Standardization tasks (per source):

- unify variable names and units (see example names below)
- ensure clear coordinate names and ordering
- attach QC flags and provenance metadata
- normalize time coordinate conventions (UTC, isoformat)

Example standardized variable names:

- `precipitation`
- `air_temperature_2m`
- `dewpoint_temperature_2m`
- `specific_humidity_2m`
- `wind_u_10m`
- `wind_v_10m`
- `surface_pressure`
- `downward_shortwave_radiation`
- `soil_water_layer_1`
- `soil_water_layer_2`
- `soil_water_layer_3`
- `soil_water_layer_4`
- `snow_depth`

Forecast preservation:

- Keep `init_time`, `lead_time` (or `lead_hour`), and `valid_time` attributes for forecast fields. Do not collapse init/lead too early — model inputs need this structure for forecast-aware predictions.

------------------------------------------------------------
6) Basin geometries and basin-grid weights
------------------------------------------------------------

`02_basin_geometries/` contains the canonical basin polygon layer and precomputed weights used to aggregate gridded fields to basin averages.

Example layout:

02_basin_geometries/
  camelsh_basins.gpkg
  basin_metadata.parquet
  static_attributes.parquet
  weights/
    mrms/
    rtma/
    gfs/
    ifs/
    era5_land/
    gdas/
    imerg/

Precomputed basin-grid weights (what and why):

- For each source grid, compute once which grid cells overlap each basin polygon and the fractional overlap area.
- A weight table stores these overlaps so that basin-average calculations become a weighted sum of grid cell values rather than redoing a polygon overlay per timestep.
- This avoids repeating an expensive spatial overlay operation for every time step and speeds up aggregation dramatically.

Example weight table columns:

- `basin_id`
- `source` (e.g., `mrms`, `rtma`, `gfs`)
- `grid_cell_id` (a deterministic identifier for the cell)
- `lat`
- `lon`
- `overlap_area`
- `cell_area`
- `weight` (overlap_area / cell_area, or normalized per-basin so weights sum to 1)

Difference from naive repeated overlay:

- Naive: for each timestep, perform polygon-grid intersection and area calculations.
- Precompute weights: perform polygon-grid intersection once per grid and reuse weights for all timesteps. This is orders of magnitude faster when aggregating time series.

------------------------------------------------------------
7) Basin time-series layer
------------------------------------------------------------

`03_basin_timeseries/` is the main exploration layer and should support both long and wide forms.

Recommended storage:

A) Long format (good for exploration, QC):

- `basin_id`
- `time`
- `source`
- `variable`
- `value`
- `qc_flag`

B) Wide format (good for model loading):

- `basin_id`
- `time`
- `mrms_precipitation`
- `rtma_air_temperature_2m`
- `rtma_specific_humidity_2m`
- ... other columns per standardized variable

Forecast schema (preserve forecast metadata):

- `basin_id`
- `source`
- `init_time`
- `lead_hour`
- `valid_time`
- `variable`
- `value`
- `qc_flag`

Why preserve `init_time` and `lead_hour`?

- Forecast-aware models need to know when the forecast was produced (`init_time`) and how far ahead (`lead_hour`) the `valid_time` is. Collapsing this information early removes the ability to construct lead-aware inputs (e.g., ensemble statistics by lead time).

------------------------------------------------------------
8) ML dataset versions
------------------------------------------------------------

`04_ml_datasets/` contains frozen, versioned, model-ready datasets. These are immutable releases used for training and evaluation.

Example structure:

04_ml_datasets/
  stage1_v001/
    dataset_config.yaml
    manifest.json
    dynamic_inputs.parquet
    static_attributes.parquet
    target_streamflow.parquet
    train_basins.txt
    val_basins.txt
    test_basins.txt
    time_splits.yaml
    normalization_stats.json
    missingness_report.csv

Notes:

- `03_basin_timeseries/` = general database / pantry of aggregated timeseries used for exploration and dataset generation.
- `04_ml_datasets/` = frozen model-ready dataset (prepared meal kit) used directly by training scripts.

Stage definitions:

Stage 1:
- hourly MRMS
- hourly RTMA
- CAMELSH static attributes
- CAMELSH hourly streamflow target

Stage 2:
- Stage 1 inputs
- daily antecedent inputs (ERA5-Land, GDAS, IMERG)
- 365-day lookback logic

Stage 3:
- Stage 1 + Stage 2
- forecast inputs (GFS, IFS)
- hourly forecast-aware streamflow prediction

------------------------------------------------------------
9) Split strategy
------------------------------------------------------------

Recommend hydrological-year based temporal splits to preserve seasonal integrity.

Initial proposed split:

train:
  2020-10-14 to 2023-09-30

validation:
  2023-10-01 to 2024-09-30

test:
  2024-10-01 to 2025-12-31

Rationale:

- Hydrological years align water seasons better than calendar years and reduce leakage across wet/dry seasons.
- Spatial splits (random basin holdout, hydrologic-region holdout) can be added later for transferability tests.

Future split types to consider:

- temporal split
- random basin split
- hydrologic-region holdout
- climate-region holdout
- unseen-basin evaluation

------------------------------------------------------------
10) QC and exploration reports
------------------------------------------------------------

`06_qc_reports/` should collect automated diagnostics used during dataset creation and monitoring.

Per-source reports:
- availability by date
- missingness by time
- missingness by basin
- value distributions
- min/max sanity checks
- units sanity checks
- preview maps

Per-basin reports:
- missing data percentage
- mean precipitation
- mean temperature
- mean streamflow
- runoff ratio sanity checks
- extreme value checks and event case studies

Cross-source reports:
- MRMS vs IMERG daily precipitation comparison
- RTMA vs GDAS/ERA5 temperature comparison
- GFS vs IFS forecast comparison by lead
- precipitation-streamflow lag examples

------------------------------------------------------------
11) Experiment management
------------------------------------------------------------

Model experiments can start as local folders and move to an experiment manager (W&B, MLflow, ClearML) once stable.

Local folder pattern:

07_experiments/
  stage1_baseline_001/
    config.yaml
    metrics.csv
    predictions.parquet
    model_checkpoint.pt
    plots/

What to track per experiment:
- dataset version
- config
- git commit
- metrics
- plots
- model checkpoints

Recommendation: start simple locally, track dataset versions and git commit. Add W&B/MLflow when frequent experiments make centralized tracking necessary.

------------------------------------------------------------
12) Backup policy
------------------------------------------------------------

Backup tiers:

Must backup:
- `docs/`
- configs
- split files
- manifests
- experiment configs
- model metrics
- decision logs

Should backup:
- `03_basin_timeseries/`
- `04_ml_datasets/`
- trained model checkpoints

Optional backup:
- `00_raw/`
- `01_standardized_grids/`

Notes:

- Raw vendor data is often redownloadable; basin time series and ML datasets are expensive to regenerate and should be prioritized for backups.
- Every backed-up dataset should include a `manifest.json` with checksums and provenance.

------------------------------------------------------------
13) Immediate roadmap
------------------------------------------------------------

Milestone 1:
- Finalize database design document (this file).

Milestone 2:
- Create the basin geometry/static layer (`02_basin_geometries/`).

Milestone 3:
- Precompute basin-grid weights for a small pilot sample of grids.

Milestone 4:
- Build a Stage 1 pilot database: 20–50 basins, 1 month, MRMS + RTMA + CAMELSH streamflow/static.

Milestone 5:
- Generate QC reports for the Stage 1 pilot.

Milestone 6:
- Scale to full Stage 1 dataset and produce a `stage1_v001` dataset in `04_ml_datasets/`.

Milestone 7:
- Train Stage 1 baseline model.

Milestone 8:
- Add Stage 2 daily antecedent inputs to the pipeline.

Milestone 9:
- Add Stage 3 forecast inputs and run forecast-aware experiments.

------------------------------------------------------------
14) Open decisions
------------------------------------------------------------

- Exact basin subset for the pilot
- Exact hydrological-year split boundaries (can be tuned by region)
- Whether to store full standardized grids long-term or rely on regenerating from `00_raw/`
- Whether to generate both wide and long basin time-series formats simultaneously or derive wide format on-demand
- Choice of experiment manager (W&B, MLflow, ClearML)
- Backup location and policy (cloud provider, frequency)
- Timing and criteria for repository split into `flash-nh-data` and `flash-nh-models`

------------------------------------------------------------
Final notes
------------------------------------------------------------

- Repository naming: use Flash-NH as the human-facing project name in documentation. Do not rename Python packages, directories, import paths, or repository names in code yet. If separate repositories are created later, suggested names include `flash-nh-data` and `flash-nh-models`.
- This document is intentionally prescriptive but modular: teams can adopt parts of the structure depending on storage, compute, and cloud constraints.

Please review and I can expand any section with templates, small example manifests, or scripts to bootstrap the pilot data layout.
