# Flash-NH Stage 1 — NeuralHydrology January Pilot Package

**Milestone:** 2G — NeuralHydrology NetCDF builder + preflight auditor
**Status:** Complete (2026-06-09)
**Audit result:** PASS — 0 errors, 1 warning (expected; documented below)

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/build_stage1_neuralhydrology_january_pilot.py` | Builds the full NeuralHydrology package |
| `scripts/audit_stage1_neuralhydrology_january_pilot.py` | Preflight auditor; exits 0 on PASS, 1 on FAIL |

Run order:

```bash
python scripts/build_stage1_neuralhydrology_january_pilot.py
python scripts/audit_stage1_neuralhydrology_january_pilot.py
```

Builder runtime: ~8s. Auditor runtime: ~20s.

---

## Package layout

```
tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/
  package/
    time_series/          {STAID}.nc  (50 files)
    attributes/           attributes_full.csv, attributes_smoke.csv
    basin_lists/          all_basins.txt, no_streamflow_basins.txt
                          january_2023_smoke/{train,val,test}_basins.txt
                          january_2023_smoke_streamflow_only/{train,val,test}_basins.txt
    configs/              smoke_v1.yml  [DRAFT — not yet run]
    manifests/            dataset_manifest.json, variable_schema.csv,
                          static_attribute_audit.csv, hydroatlas_join_audit.csv,
                          missingness_report.csv
    README.md
  audit/
    audit_report.json
    audit_report.md
    per_basin_summary.csv
    per_variable_missingness.csv
    static_attribute_audit.csv
  design/
    stage1_2f_neuralhydrology_package_design.md
    preflight_audit_plan.md
    variable_schema_proposal.csv
    static_attribute_schema_proposal.csv
    hydroatlas_asset_investigation.md
    streamflow_recovery_plan.md
```

All outputs are under `tmp/` (gitignored). No generated files committed.

---

## NetCDF structure

- Layout: NeuralHydrology GenericDataset-style. One file per basin.
- Time coordinate: `date` (744 hourly UTC steps, 2023-01-01T00Z – 2023-01-31T23Z)
- Encoding: `float32` for all dynamic variables; `float64` for `date`
- `_FillValue`: -9999.0 (NaN preserved for `qobs_m3s` where applicable)

### Dynamic variables (11 per basin)

| Variable | Units | Source | Smoke config |
|---|---|---|---|
| `mrms_qpe_1h_mm` | mm | MRMS QPE 1h Pass1 | yes |
| `rtma_2t_K` | K | RTMA CONUS 2.5km | yes |
| `rtma_2d_K` | K | RTMA CONUS 2.5km | yes (V1) |
| `rtma_2sh_kgkg` | kg kg⁻¹ | RTMA CONUS 2.5km | yes (V1) |
| `rtma_sp_Pa` | Pa | RTMA CONUS 2.5km | no (V2: wide only) |
| `rtma_10u_ms` | m s⁻¹ | RTMA CONUS 2.5km | yes |
| `rtma_10v_ms` | m s⁻¹ | RTMA CONUS 2.5km | yes |
| `rtma_10si_ms` | m s⁻¹ | RTMA CONUS 2.5km | no (wide only) |
| `rtma_i10fg_ms` | m s⁻¹ | RTMA CONUS 2.5km | no (wide only) |
| `rtma_tcc_pct` | % | RTMA CONUS 2.5km | no (V3: wide only) |
| `qobs_m3s` | m³ s⁻¹ | CAMELSH hourly | target |

`ceil` and `vis` are excluded from all output files (diagnostic_only per extraction metadata).

V1/V2/V3 refer to design decisions in `design/stage1_2f_neuralhydrology_package_design.md`.

---

## Static attributes

### Column counts

- `attributes_full.csv`: 50 rows × 238 columns
  - Column 1: `gauge_id` (8-char zero-padded STAID string; index identifier, not a model input)
  - Columns 2–238: 237 attribute columns
  - The dataset manifest records `attributes_full_cols = 237`, counting only the attribute columns
    (excluding `gauge_id`). The CSV on disk has 238 columns total. This is a counting convention,
    not a data discrepancy.
- `attributes_smoke.csv`: 50 rows × 6 columns (`gauge_id`, `DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`, `BFI_AVE`, `RBI`)

### Attribute sources

| Layer | Columns | Notes |
|---|---|---|
| Manifest (physical) | 14 | DRAIN_SQKM, BFI_AVE, RBI, QC-derived, etc. |
| GAGES-II extra | 30 | Morphology, climate, channel network |
| HydroATLAS | 193 | Full US table; 50/50 pilot match after STAID normalization |
| Total attribute cols | 237 | Excludes gauge_id |

### HydroATLAS join

- Source: `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_hydroATLAS.csv`
  (9008 × 196 cols)
- STAID normalization: `f'{int(float(str(s).strip())):08d}'` — converts any format to 8-char
  zero-padded string
- Match: 50/50 pilot STAIDs after normalization
- 193 new columns added (excludes `STAID` and `area_fraction`, and 2 cols already in manifest)
- HydroATLAS -999 / -9999 nodata sentinels replaced with NaN before writing

### Null policy

Preserve NaN; no imputation applied. Expected nullable columns (S2 decision):

| Column | Source | Null count in pilot | Reason |
|---|---|---|---|
| `max_abs_hourly_jump_over_Q50` | QC-derived (CAMELSH) | 1 | CAMELSH file missing for that basin |
| `q95_q50_ratio` | QC-derived (CAMELSH) | 1 | CAMELSH file missing for that basin |
| `wet_cl_smj` | HydroATLAS | 14 | No wetland climate class record in source data |

These three nullable columns appear in `attributes_full.csv` only; they are not in `attributes_smoke.csv`.
The auditor reports them as a single WARNING, which is expected and does not block package use.

---

## Streamflow coverage

| Category | Count | Notes |
|---|---|---|
| Full Jan 2023 `qobs_m3s` | 20 | 0 NaN in CAMELSH hourly file |
| Partial `qobs_m3s` | 8 | Some NaN hours; NaN preserved, no interpolation |
| All-NaN `qobs_m3s` | 22 | CAMELSH hourly file missing locally |

The 22 all-NaN STAIDs are listed in `basin_lists/no_streamflow_basins.txt`. They appear in the
all-50 smoke splits but are excluded from all `january_2023_smoke_streamflow_only/` splits.

Recovering these 22 basins is Milestone 2H (next).

---

## Basin splits

| Split set | Train | Val | Test | Seed |
|---|---|---|---|---|
| `january_2023_smoke/` (all 50) | 35 | 8 | 7 | 42 |
| `january_2023_smoke_streamflow_only/` (28 basins) | 20 | 4 | 4 | 42 |

---

## Audit result

```
Errors:   0
Warnings: 1  (columns with nulls — expected; see Null policy above)
```

Preflight gate definition: PASS when 0 ERROR-level issues. Warnings are documented but
do not block package use.

---

## What was NOT done

- No NeuralHydrology model training was run.
- No scientific performance claims were made.
- URMA precipitation was not added to model inputs (diagnostic-only per standing constraint).
- No generated files were committed.

---

## Next: Milestone 2H — Streamflow recovery

Milestone 2H must be completed before:
- Serious NeuralHydrology training runs
- Scientific performance claims
- HPC-scale 2020–2025 packaging

Recovery plan: `design/streamflow_recovery_plan.md`