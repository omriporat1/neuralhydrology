# Flash-NH Stage 1 Pipeline Skeleton

**Status**: Skeleton complete — directory bootstrap, basin selection, and CAMELSH discovery implemented. Gridded forcing acquisition and basin-average extraction NOT yet implemented.

**Date**: 2026-05-31

---

## 1. Purpose

This document describes the reproducible Stage 1 pilot skeleton for the Flash-NH meteorological forcing data pipeline. The skeleton provides:

- A configurable external data root with the full directory structure
- A reproducible pilot basin selection (50 basins, January 2023)
- CAMELSH input discovery and validation
- A shared provenance/manifest utility
- A single dry-run command that validates the skeleton end-to-end

The skeleton does **not** yet implement MRMS/RTMA downloading, grid definition, basin-grid weight computation, or basin-average extraction. Those are the next milestones.

---

## 2. Repository layout (new files added by this skeleton)

```
configs/
  pilot_stage1.yaml          # YAML config template — edit before running
docs/
  pipeline_skeleton.md       # This file
scripts/
  bootstrap_data_root.py     # Create the external data root directory structure
  select_pilot_basins.py     # Reproducibly select 50 pilot basins
  discover_camelsh_inputs.py # Discover and validate CAMELSH input files
  run_stage1_pilot_dry_run.py # Orchestrating dry-run entry point
src/
  pipeline/
    __init__.py
    config.py                # PipelineConfig dataclass + YAML loader
    provenance.py            # write_run_manifest() and git_commit_hash()
```

Existing acquisition sources (`src/datasources/`) are not modified.

---

## 3. How to configure the data root

**Option A** — edit the config file:

```yaml
# configs/pilot_stage1.yaml
data_root:
  local: D:/Flash-NH_data          # Windows example
  # local: /data/flash-nh          # Linux/Mac example
```

**Option B** — environment variable (overrides the config):

```powershell
$env:FLASHNH_DATA_ROOT = "D:/Flash-NH_data"
python scripts/run_stage1_pilot_dry_run.py
```

**Option C** — CLI flag (overrides both config and env):

```powershell
python scripts/run_stage1_pilot_dry_run.py --data-root D:/Flash-NH_data
```

For HPC/SLURM, set `data_root.hpc` in the config and export `FLASHNH_DATA_ROOT` in your SLURM job script. The same source code works on both local PC and HPC.

---

## 4. How to run the dry-run

The dry-run creates the directory structure, selects pilot basins, discovers CAMELSH inputs, and writes provenance manifests. It does **not** download any gridded data.

```powershell
# Self-contained local test (outputs go to tmp/stage1_pilot_dryrun/ in the repo)
python scripts/run_stage1_pilot_dry_run.py

# With explicit config and external data root
python scripts/run_stage1_pilot_dry_run.py --config configs/pilot_stage1.yaml --data-root D:/Flash-NH_data
```

Expected outputs after a successful dry-run:

```
{data_root}/
  00_raw/mrms/
  00_raw/rtma/
  02_basin_geometries/weights/mrms/
  02_basin_geometries/weights/rtma/
  03_basin_timeseries/stage1_pilot/
  04_ml_datasets/stage1_pilot_v001/
  06_qc_reports/stage1_pilot/
  09_manifests/
    bootstrap/
      manifest.json
      summary.md
      summary.json
      run_command.txt
      git_commit.txt
      config_snapshot.yaml
      dry_run_summary.json
    stage1_pilot/
      pilot_basin_manifest.csv        ← 50 pilot basins with STAID and pilot_role
      camelsh_discovery_report.json   ← CAMELSH file discovery status
      manifest.json                   ← basin selection provenance
      summary.md
```

---

## 5. Pilot basin selection

**Source**: `reports/flashnh_final_basin_selection_v001/tables/final_basin_training_status.csv` (auto-discovered from repo).

**Target composition** (configured in `configs/pilot_stage1.yaml`):

| Stratum | Count | Role in pilot |
|---|---|---|
| TRAIN_CORE + TRAIN_SOFT_KEEP | 40 | Model training; primary forcing targets |
| HOLDOUT_REVIEW | 5 | Withheld from training; included for QC visualization |
| EXCLUDE_TRAINING | 5 | Hard-excluded; included for forced-failure QC only |
| **Total** | **50** | |

If `fallback_all_train: true` (default) and the HOLDOUT/EXCLUDE strata have fewer basins than requested, all 50 basins are drawn from the training set. The manifest records which mode was used.

**Random seed**: 42 (configurable via `pilot.basin_selection.random_seed`).

The pilot month is **January 2023** (UTC). See the rationale in `configs/pilot_stage1.yaml`.

---

## 6. Expected input files (not tracked in repo)

### Basin status (auto-discovered from repo)

| File | Path |
|---|---|
| Final training status | `reports/flashnh_final_basin_selection_v001/tables/final_basin_training_status.csv` |

### CAMELSH source files (configure in YAML)

| Input | Config key | Purpose |
|---|---|---|
| Basin polygons (GeoPackage) | `camelsh.basin_polygons` | Spatial basin boundaries for weight computation |
| Static attributes | `camelsh.static_attributes` | BFI, area, climate indices for model inputs |
| Hourly streamflow | `camelsh.hourly_streamflow` | Target variable for model training |

These files are **not tracked in the repository** and must be configured separately. The discovery script reports their availability and optionally validates STAID matching.

---

## 7. What was deliberately NOT implemented in this skeleton

| Item | Status | Next milestone |
|---|---|---|
| MRMS file downloading | NOT IMPLEMENTED | Milestone 2 (reuse `MrmsAwsQpe1hPass1`) |
| RTMA file downloading | NOT IMPLEMENTED | Milestone 2 (reuse `RtmaAwsConusDataSource`) |
| MRMS/RTMA grid definitions | NOT IMPLEMENTED | Milestone 2 |
| Basin-grid weight computation | NOT IMPLEMENTED | Milestone 3 |
| Basin-average extraction | NOT IMPLEMENTED | Milestone 4 |
| CAMELSH polygon loading | NOT IMPLEMENTED | Milestone 3 (requires geopandas) |
| STAID→streamflow join | NOT IMPLEMENTED | Milestone 4 |
| Stage 2 products (ERA5, GDAS, IMERG) | NOT IMPLEMENTED | Stage 2 |
| Stage 3 products (GFS, IFS) | NOT IMPLEMENTED | Stage 3 |
| NeuralHydrology dataset packaging | NOT IMPLEMENTED | Milestone 5 |
| QC Level 2/3 plots | NOT IMPLEMENTED | Milestone 4+ |

---

## 8. Next milestone: MRMS/RTMA grid definition and basin-grid weights

The existing acquisition sources in `src/datasources/mrms.py` and `src/datasources/rtma.py` are validated and should be **reused, not reimplemented**. The next implementation steps are:

### Milestone 2 — Grid definition + small extraction test

1. Define the MRMS and RTMA grid coordinates (lat/lon arrays for each product's native grid).
   - MRMS QPE 1h Pass1: ~1km CONUS grid (approximately 3500×7000 cells). Coordinates can be read from any downloaded GRIB2 file or from the STAC metadata.
   - RTMA 2.5km: `rtma2p5` analysis grid. Coordinates from a downloaded `grb2_wexp` file.

2. Download one sample hour for each product for the 50 pilot basins' bounding box.
   - Reuse `MrmsAwsQpe1hPass1.download_sample()` and `RtmaAwsConusDataSource.download_sample()`.

3. Compute basin-grid overlap weights for the 50 pilot basins.
   - Load CAMELSH basin polygons (GeoPackage, geopandas).
   - Use `rasterio`/`rioxarray`/`geocube` to compute fractional overlap of each basin with each grid cell.
   - Write weight tables (Parquet) to `{data_root}/02_basin_geometries/weights/mrms/` and `.../rtma/`.
   - Validate: weight sums close to 1.0 per basin; no negative weights; plausible number of contributing cells.

4. Extract one hour of basin-average values for all 50 pilot basins.
   - Decode GRIB2 with `cfgrib.open_dataset()` (already in venv).
   - Apply precomputed weights via vectorized dot product.
   - Write one row per basin/time to `{data_root}/03_basin_timeseries/stage1_pilot/`.

### Existing validated gateways to reuse

| Product | Class | File |
|---|---|---|
| MRMS QPE 1h Pass1 | `MrmsAwsQpe1hPass1` | `src/datasources/mrms.py` |
| RTMA 2.5km analysis | `RtmaAwsConusDataSource` | `src/datasources/rtma.py` |
| GFS forecast | `GfsAwsConusDataSource` | `src/datasources/gfs.py` |
| IFS forecast | `IfsMarsDataSource` | `src/datasources/ifs.py` |
| ERA5-Land | `Era5LandTDataSource` | `src/datasources/era5_landt.py` |
| GDAS | `GdasAwsAntecedentDataSource` | `src/datasources/gdas.py` |
| IMERG Late Daily | `ImergLateDailyDataSource` | `src/datasources/imerg.py` |

Do not rewrite these acquisition classes. Wrap them in new `src/pipeline/acquisition.py` and `src/pipeline/extraction.py` modules instead.

---

## 9. Repository policy reminders

- **Never commit** raw GRIB/NC4/Parquet data files.
- **Never commit** large report tables, logs, or animation outputs.
- **Do commit** configs, manifests, small provenance summaries, and this documentation.
- The external data root (`Flash-NH_data/` or equivalent) must be outside the git-tracked source tree.
- The `tmp/` directory is gitignored; the self-contained dry-run uses `tmp/stage1_pilot_dryrun/` by default.
- See `docs/repo_policy.md` for the full policy.

---

## 10. Milestone checklist

| Milestone | Status |
|---|---|
| Basin screening and final basin list | ✅ Complete (`flashnh_final_basin_selection_v001`) |
| Pilot skeleton: config, bootstrap, basin selection, discovery | ✅ Complete (this document) |
| MRMS + RTMA grid definitions and sample download | ⬜ Milestone 2 |
| Basin-grid weight computation for 50 pilot basins | ⬜ Milestone 3 |
| Basin-average extraction for January 2023 | ⬜ Milestone 4 |
| CAMELSH streamflow alignment and static attribute join | ⬜ Milestone 4 |
| Level-1 QC reports | ⬜ Milestone 4 |
| Freeze `stage1_pilot_v001` dataset | ⬜ Milestone 5 |
| Scale to full 2,843-basin training set | ⬜ Milestone 6 |
| Stage 1 model training | ⬜ Milestone 7 |
| Stage 2 daily antecedent inputs | ⬜ Stage 2 |
| Stage 3 forecast inputs | ⬜ Stage 3 |
