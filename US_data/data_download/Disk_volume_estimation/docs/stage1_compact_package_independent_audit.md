# Stage 1 Compact Scientific Package — Independent Audit (Gate 4)

## Status

The 32-basin Compact Scientific Package has been built and promoted on h2o
(`/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001`), and its
builder-level self-checks and an independent ChatGPT inspection of its
compact review bundle are complete. **The package is built, but not yet
independently certified.** This document describes the independent auditor
implemented to close that gap (`src/baseline/package_audit.py`,
`scripts/audit_stage1_compact_scientific_package.py`). As of this writing the
auditor has been implemented and exercised only against synthetic fixtures
(`tests/test_package_audit.py`); it has not yet been run against the real
package or its real source artifacts on h2o. **NeuralHydrology configuration
generation remains blocked until the real audit run completes with PASS.**

## Purpose

A builder's own self-checks cannot certify the builder's own output: if the
same formula that produced a value is reused to check it, a bug in that
formula is invisible to the check. This auditor exists to answer, from
scratch, "is the package correct?" by re-deriving every scientific and
structural claim directly from raw inputs, using only general-purpose
libraries, and comparing the result to what the package actually contains.

## Independence boundary

**Never imported by `src/baseline/package_audit.py`:**
- `src.baseline.package_builder` (or any private validation helper inside it)
- `src.baseline.package_assembly`
- `src.baseline.package_netcdf`
- `src.baseline.units` — the m³/s → mm/h conversion
  (`q_mm_per_h = q_m3s * 3.6 / area_km2`) is re-derived locally in
  `independent_discharge_to_runoff_mm_per_h`
- `src.baseline.lead_targets` — the lead-target shift
  (`target[t] = converted_qobs[t + lead_hours]`, trailing `lead_hours` rows
  NaN) is re-derived locally in `independent_lead_shift`
- `src.baseline.gap_mask_io` — the missing-hour-products inventory CSV is
  re-parsed and re-filtered locally in `load_gap_inventory_independent` /
  `reconstruct_gap_timestamps_independent`
- `src.baseline.splits` — `sha256_of` is re-declared locally as `sha256_file`

**Reused, with justification (neutral, non-scientific helpers only):**
- `src.baseline.staid.normalize_staid` — strict basin-ID string validation
  (zero-padding and length rules). This is a format-acceptance utility, not
  a computation the audit exists to verify; reusing it keeps basin-ID
  acceptance consistent across the codebase instead of risking silent drift
  between two hand-written copies.
- `src.baseline.policy.load_stage1_baseline_policy` /
  `validate_stage1_baseline_policy`, at the CLI layer only, to parse and
  schema-validate the signed policy YAML before handing the resulting
  mapping to the audit library. Schema parsing is not a scientific
  computation under audit.

Everything else — directory layout, checksum recomputation, NetCDF
dimensions/dtypes/units/timeline, forcing/qobs value comparison, the
conversion equation, the lead shift and its tail-NaN behavior, gap-timestamp
reconstruction, gap-flag validity, static-attribute membership/order/values/
imputation placement, and QC-CSV-to-NetCDF cross-checking — is implemented
directly in `package_audit.py` using `pandas`, `numpy`, `xarray`, `netCDF4`,
`json`, `csv`, `pathlib`, and `hashlib`.

The auditor never modifies, rebuilds, or promotes a package. It only reads.

## Authoritative inputs

| Input | CLI flag | Notes |
|---|---|---|
| Package root | `--package-root` | e.g. `stage1_compact_scientific_package_v001` |
| Scientific policy YAML | `--policy-yaml` | schema-validated at the CLI layer |
| Accepted basin selection | `--basin-selection-file` | one basin ID per line |
| Prepared static matrix | `--prepared-static-parquet` | the compact artifact actually used to build the package |
| Static column manifest | `--static-column-manifest` | column→role JSON |
| Imputation manifest | `--imputation-manifest` | `imputation_manifest.json`; **required for `--mode full`** |
| Imputed-value mask | `--imputed-value-mask` | `imputed_value_mask.parquet`; **required for `--mode full`** |
| Forcing source root | `--forcing-root` | `<root>/time_series/<id>.parquet` |
| Qobs source root | `--qobs-root` | `<root>/time_series/<id>.nc` |
| Area source | `--area-csv` | `gauge_id,DRAIN_SQKM` |
| Gap inventory | `--gap-inventory-csv` | `fullperiod_missing_hour_products.csv` |
| QC evidence root | `--qc-evidence-root` | non-authoritative; cross-checked, never trusted; **required for `--mode full`** |
| Build commit | `--build-git-commit` | git commit of the package-builder code that produced `--package-root` |

**`--mode full` cannot skip imputation/QC evidence.** All three of
`--imputation-manifest`, `--imputed-value-mask`, and `--qc-evidence-root` are
mandatory for a canonical full audit; `run_audit()` raises `PackageAuditError`
before any check runs if any of the three is omitted, and the CLI's
`_FULL_MODE_REQUIRED` list enforces the same at the argument-parsing layer.
The only way to run a full audit without them is the library's
`dev_allow_missing_evidence=True` parameter, which the CLI never sets — it
exists solely for isolated development/test use and must never be used for a
canonical h2o audit.

## Checks performed (Gate 4 objectives)

1. **Exact package layout** — every file below the package root is recursively enumerated; the check fails on any missing required file *and* on any unexpected top-level entry or extra file not part of the defined package contract (`check_package_layout`, `check_exact_package_layout`).
2. **Full provenance-binding checksums** — every file in `manifests/file_checksums.csv` and `manifests/package_manifest.json` is recomputed from bytes on disk and compared to the declared value; in addition, every authoritative package artifact (`time_series/*.nc`, `attributes/attributes.csv`, `basins/basin_ids.txt`, `masks/gap_timestamps.json`, `manifests/package_manifest.json`, `manifests/file_checksums.csv`, `run_provenance.json`) and every upstream source file actually compared by the audit (all forcing source parquets, all qobs source NetCDFs, the imputed-value-mask parquet) is independently checksummed from disk bytes — never copied from any manifest under audit — and bound into `audit_manifest.json` as deterministic basin-keyed mappings (`check_checksums_and_manifest`, `compute_package_artifact_checksums_independent`, `compute_source_checksums_independent`).
3. **Basin membership and order** — `basins/basin_ids.txt` and the `time_series/*.nc` file set are compared, exactly, to the accepted basin selection (`check_basin_membership`).
4. **NetCDF dimensions/dtypes/units/metadata/timeline** — variable order, per-variable dtype (`float32` continuous / `int8` gap flags), per-variable `units` attribute, `gauge_id` attribute, the exact canonical hourly timeline, the single `time` dimension and its size, every variable's dimension tuple, the dataset-level `package_schema_name`/`package_schema_version` attributes, gap-flag `flag_values`/`flag_meanings` attributes, the raw-qobs `role="audit_provenance_not_training_target"` attribute, and every lead-target's `role="training_target"`/`lead_hours` attributes — all independently re-read and compared against the real Gate 2 serialization contract inspected read-only in `package_netcdf.py` (never imported or called) (`audit_basin_netcdf`).
5. **Dynamic inputs vs. forcing source** — all 8 approved forcing variables, full-array, against the raw forcing parquet.
6. **Raw qobs vs. source** — `qobs_m3s`, full-array, against the raw qobs NetCDF.
7. **Unit conversion** — `q_mm_per_h = q_m3s * 3.6 / area_km2`, re-derived independently (not imported), compared against every lead-target's implied base series.
8. **Lead-target shift** — `target[t] = converted_qobs[t + lead_hours]` for L ∈ {1,3,6,12}, re-derived independently via array slicing (not `pandas.Series.shift`).
9. **NaN propagation / target-tail NaNs / no infinities** — NaN-mask exactness is part of every full-array numeric comparison; the trailing `lead_hours` rows of each lead-target variable are checked explicitly; continuous forcing/qobs source arrays and package arrays alike are rejected if they contain `+inf`/`-inf` anywhere, and `compare_float_arrays` is hardened so two arrays with matching infinities at the same position can never be reported as a silent pass.
10. **Static attributes and imputation evidence** — basin membership/order, column order/membership, forbidden-column absence, full-array value comparison against the prepared static matrix, no residual NaNs; and, exactly (required for `--mode full`): the imputed-value mask's basin order and column order (derived from the static column manifest, never hard-coded) match exactly, the mask contains only `{0,1}` values with no missing cells, every mask-true cell has a corresponding manifest `fitted_value` (a mask-true cell with no fitted value is its own failure, distinct from a mask-true cell whose fitted value is numerically wrong), and per-column/per-basin/total imputed-cell counts agree with the manifest whenever the manifest records them (`audit_static_attributes`).
11. **Gap-timestamp reconstruction** — the 138 MRMS+RTMA gap timestamps are independently reconstructed from the missing-hour-products inventory CSV (policy-driven product scope) and compared, as a set, to `masks/gap_timestamps.json` (`audit_gap_reconstruction`).
12. **Gap-flag validity** — every gap-flag variable is checked complete (finite) and strictly binary (`{0,1}`).
13. **QC evidence cross-check** — required for `--mode full`: exactly the accepted basin set has a QC CSV, with no extras; `csv_manifest.json`'s basin count and membership agree with the files on disk; every manifest entry declares `authoritative`/`usable_for_training`/`usable_for_package_reconstruction` all `false`; every per-file `sha256`/`size_bytes`/`row_count` in the manifest is checked against the actual file on disk; and every QC CSV value is compared to its NetCDF counterpart by exact float32 projection (see "Comparison tolerances" below — this is not the CSV text round-trip tolerance). The CSV is never treated as authoritative regardless of outcome.
14. **Provenance identity binding** — `build_audit_manifest` binds the build commit, the auditor's own resolved git commit, and checksums of every authoritative/source/policy artifact plus the generated audit outputs themselves. A full audit additionally *hard-fails* (raises, not warns) if the auditor's own commit cannot be resolved or if the auditor's own working tree is not clean at execution time — this guards against auditing with uncommitted changes to the auditing code itself. Build commit and auditor commit are recorded as two distinct identities; they are never required to be equal.

## Comparison tolerances

- **Exact equality**: strings/IDs, basin order, timestamps, dimensions, variable order, NaN masks.

The auditor distinguishes three numerically distinct comparison operations,
each answering a different question and each with its own dedicated
tolerance. They must not be confused, and none is used as a stand-in for
another:

1. **CSV float64 text round trip** (`attributes_csv_values_match_prepared`)
   — does writing a float64 value to CSV text and reading it back reproduce
   that same float64 value? Both sides are float64 with no quantization
   involved (the prepared static parquet and the package's
   `attributes/attributes.csv`, which is written at full precision, not
   float32-quantized). Uses `QC_CSV_ROUNDTRIP_RTOL = 1e-9`,
   `QC_CSV_ROUNDTRIP_ATOL = 1e-12`.

2. **Authoritative source to NetCDF** (`dynamic_input_matches_forcing_source`,
   `raw_qobs_matches_source`, `lead_target_matches_independent_recomputation`)
   — do the package's float32-quantized values agree with the raw
   forcing/qobs/static sources, once the documented float32 storage
   quantization is accounted for? Uses `rtol =
   policy["audit"]["package_float32_rtol"]` (pinned `1e-5`), `atol = 0.0` —
   read from the supplied policy, never hard-coded, and never silently
   loosened.

3. **QC CSV to NetCDF** (`qc_csv_matches_netcdf`,
   `compare_qc_csv_against_netcdf_storage`) — does the non-authoritative,
   pre-quantization QC CSV evidence reproduce the *exact* authoritative
   float32 stored representation? The QC CSV is written from the builder's
   float64 values before quantization; the NetCDF stores the same values
   already quantized to float32 (confirmed per-variable from the NetCDF
   itself, not hard-coded). Rather than applying a broader numerical
   tolerance to paper over that quantization, the CSV's finite values are
   cast through the same dtype the package actually stores on disk and
   required to agree **exactly** with the stored value (`rtol = atol =
   0.0` in the recorded `NumericCheckResult`, signaling an exact rather than
   tolerance-based comparison). NaN positions must still match exactly.
   Gap-flag (and any other integer) variables are compared as exact
   binary/integer values with no floating tolerance at all — they carry no
   float32 quantization step to reproduce.

Every numerical check (`NumericCheckResult` / `AuditReport.numeric`) records: compared element count, NaN-pattern mismatch count, finite-value mismatch count, max absolute difference, max relative difference, the rtol/atol used, and PASS/FAIL.

## PASS / FAIL definition

`AuditReport.status` is `"FAIL"` if any check recorded an `ERROR`, else `"PASS"`. `WARNING`-severity records (e.g. an optional input not supplied) never affect status. The CLI exits `0` on PASS, `1` on FAIL, `2` on a setup/usage error (e.g. an unreadable required path or an invalid policy file) before any check could run.

## Generated outputs

Every full-audit run writes, under `--output-dir` (refuses to write into an existing non-empty directory unless `--overwrite` is given):

- `audit_results.json` — status, counts, every check record, every numeric result, failed-check list.
- `audit_report.md` — human-readable summary.
- `audit_manifest.json` — the provenance-identity binding (build commit, auditor commit, every input checksum, audit command, execution timestamp), patched with checksums of the generated result files themselves.
- `file_checksums.csv` — checksums of the generated output files.
- `run.log` — a short run log (mode, package root, status, counts).
- `review_bundle/` — a compact subset (`audit_report.md`, a results summary) suitable for pasting into an external review.

None of these are committed to git; they are h2o/Moriah evidence artifacts only.

## Command templates (h2o)

Activate the environment first:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

**Preflight** (fast existence/readability check, no NetCDF/parquet content read):

```bash
python scripts/audit_stage1_compact_scientific_package.py \
  --mode preflight \
  --package-root /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001 \
  --policy-yaml config/stage1_scientific_baseline_v001.yaml \
  --basin-selection-file /data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence/compact_basin_ids.txt \
  --prepared-static-parquet /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/imputed_static_attributes.parquet \
  --static-column-manifest /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_column_manifest.json \
  --imputation-manifest /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/imputation_manifest.json \
  --imputed-value-mask /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/imputed_value_mask.parquet \
  --forcing-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001 \
  --qobs-root /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
  --area-csv /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_inputs/area.csv \
  --gap-inventory-csv /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_inputs/fullperiod_missing_hour_products.csv \
  --qc-evidence-root /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_evidence \
  --output-dir /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_gate4_preflight
```

**Full audit:**

```bash
python scripts/audit_stage1_compact_scientific_package.py \
  --mode full \
  --package-root /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001 \
  --policy-yaml config/stage1_scientific_baseline_v001.yaml \
  --basin-selection-file /data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence/compact_basin_ids.txt \
  --prepared-static-parquet /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/imputed_static_attributes.parquet \
  --static-column-manifest /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_column_manifest.json \
  --imputation-manifest /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/imputation_manifest.json \
  --imputed-value-mask /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/imputed_value_mask.parquet \
  --forcing-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001 \
  --qobs-root /data42/omrip/Flash-NH/tmp/stage1_target_package_v001 \
  --area-csv /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_inputs/area.csv \
  --gap-inventory-csv /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_inputs/fullperiod_missing_hour_products.csv \
  --qc-evidence-root /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_evidence \
  --build-git-commit 89c4dd162f7043419b4b227de5c2bc1b3b230da6 \
  --output-dir /data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_gate4_audit
```

These are documentation templates for the next phase only — they are not run
as part of this correction round. The auditor's own working tree must be
clean and its commit resolvable at execution time (see check 14 above), so
any outstanding auditor-code changes must be committed before the full audit
is actually run on h2o.

## Evidence-transfer requirements

After a real h2o run, the entire generated `--output-dir` (all 6 items above)
must be transferred off h2o as Gate 4 evidence before the package can be
considered independently certified. Nothing under `--output-dir` is
committed to git.

## Scope note

NeuralHydrology configuration generation remains blocked until this auditor
has been run against the real package and real source artifacts on h2o and
reports `PASS`.
