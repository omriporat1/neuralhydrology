# Flash-NH Current State

Last updated: 2026-07-24 (Commit-readiness review + safeguard fixes for the full-population NH config-generation increment)

## Commit-readiness review of the full-population NH config-generation increment — two safeguard fixes (2026-07-24)

Review-only pass (no transfer/Moriah/Slurm/training) over the increment directly below. Found and
fixed two gaps in distinguishing the `spatial_holdout` bundle from a trainable experiment (a
custom marker cannot live inside `config.yaml` itself — NH 1.13's `Config._check_cfg_keys` rejects
unrecognized keys): (1) the holdout bundle's default `experiment_name` now gets a
`_spatial_holdout_test_only_eval` suffix instead of colliding with the development bundle's name;
(2) `write_generated_config` now writes a sibling `TEST_ONLY_DO_NOT_TRAIN.txt` for the holdout
bundle only, and `check_generated_config_structure`'s holdout-role path requires it to exist. Added
9 tests (full-population totals now 14 + 15 = 29 passing). Re-ran the synthetic dry run and this
time directly read the generated `config.yaml`/marker/basin-list files; ran the preflight CLI
end-to-end (`--skip-dataset-construction`) — PASS, 56 OK, 0 errors. Full suite: 889 passed, 0 failed
(known Windows `os.rename` flake in `test_package_builder.py` did not reproduce this run; separately
re-confirmed passing in isolation, unrelated to this patch). Nothing committed. Full detail:
`docs/decision_log.md` (2026-07-24 "Commit-readiness review..." entry).

## Full-population (development + spatial-holdout) NH config-generation + structural-preflight local implementation increment (2026-07-24)

**Local-only** (no h2o/Moriah access, no package rebuild, no training). Extends the 2026-07-22
compact-package config-generation/structural-preflight machinery (below) to the certified full
non-California package (Gate 4, below: 2,307 development-training + 250 spatial-holdout basins).
Renders the single lead06/seq24 scientific configuration as **two strictly separated bundles**:
a `development` bundle (train == validation == temporal-test, the 2,307 development basins across
different date periods) and a test-only `spatial_holdout` bundle (train/validation lists are the
development population, never a holdout basin; test list is the 250 holdout basins). New basin-
membership validation requires the package's basin set to equal exactly the
`development_train` ∪ `spatial_holdout_nonca` union (no California, no overlap, no missing/extra,
exact 2,307/250 counts). New `src/baseline/nh_structural_preflight.py` check
(`check_flashnh_external_scaler_test_construction`) constructs only the holdout `test`-period
dataset, reusing the development-fitted scaler unchanged and never touching the holdout config's
`train_dir`. Two new CLIs: `scripts/generate_stage1_full_population_nh_config.py` and
`scripts/check_stage1_full_population_nh_config_preflight.py` (package root always supplied via
`--package-root`, never hard-coded). 20 new tests passing (10 in `tests/test_nh_full_population_
config_generation.py`, 10 in `tests/test_nh_full_population_structural_preflight.py`); all 41
pre-existing compact-package tests unaffected. Verified via a local dry run against a synthetic
2,557-basin fake package (matching the real split union): generator produced the expected
development/holdout basin-count contracts; preflight (`--skip-dataset-construction`) reported PASS,
55 OK checks, 0 errors. **Not done:** package transfer, real Moriah Slurm preflight, real
full-population dataset loading, training, or the remaining 15 lead × sequence-length
configurations. Full detail: `docs/decision_log.md` (2026-07-24 entry, same title).

## Gate 4 — Full non-California Scientific Package (v002) independently audited — PASS (2026-07-24)

The Gate 4 independent auditor (`src/baseline/package_audit.py`, commit
`98b7d42f23963e76e02ad3991d7298d3ada98ee3`) reran for real on h2o, in full mode, against
`/data42/omrip/Flash-NH/tmp/stage1_scientific_package_v002` (2,307 development-training + 250
spatial-holdout non-California basins, build commit `61d3819deb55240652276765c6a96d12ed3ce539`).

**Result: PASS — 0 errors, 1 warning, 260,870 OK checks.** Audit output:
`/data42/omrip/Flash-NH/tmp/stage1_scientific_package_v002_gate4_audit/full_rerun_20260724T110557Z`.
Evidence archive SHA-256 `9cc9f8e63d6c9825c2bf765106a20a58ce0560a1d733bc815ec0846f02071ed0`,
transferred locally and independently verified (checksums, arithmetic tally, provenance).

The single warning, `imputed_value_mask_basin_order`, is non-blocking: exact mask basin-index
membership passed with zero missing/extra basins (`imputed_value_mask_basin_membership`: OK); only
row order differs, and every downstream imputation-placement check re-indexes the mask by basin
label, so order alone cannot affect correctness.

This rerun follows and fixes an earlier FAILED full audit (errors=9) of the same package. The
auditor patch (same commit) introduced: (1) a 1-float32-ULP tolerance for the non-authoritative
QC-CSV-versus-NetCDF finite comparison only, absorbing a confirmed xarray/netcdf4 write-path
rounding artifact; (2) separate strict mask-membership (ERROR) and non-blocking mask-order
(WARNING) checks, replacing a single check that conflated the two.

**The package itself was not rebuilt** (`build_git_commit` unchanged), and no static artifact or
other source input was modified. No Moriah transfer, no NeuralHydrology configuration generation,
and no training occurred. **This closes the production package build-and-independent-audit phase
for `stage1_scientific_package_v002`; it does not establish scientific model skill.** Full detail:
`docs/decision_log.md` (2026-07-24 Gate 4 entry).

## Full non-California static-attribute preparation — real h2o run PASS (2026-07-24)

Real (not synthetic/dry-run) execution of `scripts/prepare_stage1_full_static_attributes.py`
on h2o against the canonical `stage1_static_attributes_v002` matrix. Development-only median
imputation (fit on the 2,307 development-training basins only) followed by a development-only
exact post-imputation zero-variance trainability projection, both frozen and applied unchanged
to the full 2,557-basin non-California package population (2,307 development-training + 250
spatial holdout). The spatial holdout did not influence either fit.

**Result: 473 candidate `model_input` columns → 473 retained, 0 excluded.** All 473 canonical
static model-input columns will be used by the first full-population Stage 1 model — the
canonical package contract is unchanged at 473 columns. No column was entirely missing in the
2,307-basin fit population, and zero missing values remain anywhere after imputation. Because 0
columns were excluded, the retained static table is byte-identical to the imputed static table.
The 32-basin compact-smoke 13-column zero-variance exclusion (2026-07-23 "Compact
NeuralHydrology integration smoke" Finding 1, below) is compact-population-specific historical
evidence only and was confirmed **not** reused, inherited, or reopened by this run.

Output (h2o-resident, generated evidence, **not committed**):
`/data42/omrip/Flash-NH/tmp/stage1_full_static_attributes_v001/` (`imputed_static_attributes.parquet`,
`imputed_value_mask.parquet`, `retained_static_attributes.parquet`, manifests, column lists,
`run_summary.json`).

Key checksums (SHA-256):
```
input matrix (stage1_static_attributes_v002.parquet):
4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297

imputed_static_attributes.parquet / retained_static_attributes.parquet (identical):
5be00a3b068351bffd40a3cf72991a3df888700034831123c91823b8bd4b6e24
```

Full counts, per-artifact checksums, and the modeling decision: `docs/decision_log.md`
(2026-07-24 entry). **No NetCDF package was built by this run.** Next step: full 2,557-basin
v002 package build and independent audit — see `docs/stage1_baseline_package_implementation_plan.md`.

## Development-population zero-variance trainability projection — mechanism implementation (2026-07-23)

Added a reusable fit/apply mechanism in `src/baseline/static_preparation.py`
(`ZeroVarianceFit`, `fit_zero_variance_projection`,
`apply_zero_variance_projection`, `build_zero_variance_manifest`,
`write_zero_variance_manifest`) that identifies static `model_input` columns
with exactly zero variance over the Stage 1 development-training population
(2,307 basins) **after** development-only median imputation
(`fit_development_median_imputation`, above/below). This is a **run-specific
trainability projection, not a package-schema change**: the canonical static
matrix and Compact/full package contract remain **473 `model_input`
columns**, unmodified. The fit uses exact post-imputation constancy (no
near-zero-variance threshold), is fitted once on the 2,307-basin
development-training population, and its frozen retained/excluded column
list is meant to be applied unchanged — never recomputed — to validation,
temporal-test, and spatial-holdout populations. The compact-smoke 13-column
exclusion list (`docs/decision_log.md` "Finding 1", 2026-07-23) is historical
evidence for the 32-basin smoke population only and is explicitly not
reused, imported, or asserted here. **This patch implements the mechanism
only: it does not compute the real 2,307-basin excluded-column list (no
h2o access), does not build a package, and does not generate NeuralHydrology
configs.** 18 focused tests added in `tests/test_static_preparation.py`
(52/52 passing). Full detail: `docs/decision_log.md` (2026-07-23 entry).

## Versioned package schema (`date`) for future scientific packages (2026-07-23)

**Schema-support implementation addendum** (code, tests, and documentation
together — not a docs-only change). Added an explicit,
versioned NetCDF package-schema registry (`src/baseline/package_netcdf.py`)
so future full scientific packages can use temporal coordinate `date`
(`stage1_scientific_package_v002`, version 2) while the certified compact
package below **remains exactly as built and certified: frozen, on disk,
with coordinate `time`** (`stage1_compact_scientific_package_v001`, version
1, unchanged default at the low-level serializer). The package-builder CLI
now requires an explicit `--package-schema` choice — no default, no
inference from basin count/path/output name — so a future production build
cannot silently emit a legacy `time` package by omission. Provenance now
records both the (corrected, deprecated-but-preserved)
`package_schema_name` builder-manifest identity and five new explicit
fields (`builder_manifest_schema_name`, `builder_manifest_schema_version`,
`netcdf_package_schema_name`, `netcdf_package_schema_version`,
`netcdf_time_coordinate`). The independent auditor
(`src/baseline/package_audit.py`) now checks the declared/actual NetCDF
schema and coordinate from disk without importing the schema registry it
audits. `FlashNHDataset`'s `time`→`date` compatibility adapter
(`src/baseline/nh_dataset.py`) is renamed `_adapt_temporal_index_to_date`
and now handles all four coordinate-presence combinations explicitly
(pass through `date`; rename `time`; fail loudly on both or neither, in
either direction). Structural NH compatibility with a `date`-coordinate
package does not imply stock `GenericDataset` reproduces Flash-NH's own
sample-validity filtering — `FlashNHDataset` remains required either way.
**No real package was built by this patch; h2o/Moriah were not accessed;
the certified compact v001 package was not touched.** Full detail:
`docs/decision_log.md` (2026-07-23 "Versioned package schema" entry) and
`docs/stage1_compact_package_independent_audit.md` (2026-07-23 addendum).

---

## Current milestone

**Compact NeuralHydrology integration smoke — CLOSED (CPU preflight + GPU
training + explicit validation/test evaluation all PASS, 2026-07-23).** This
closes the compact-package NH integration-validation effort opened by the
2026-07-22 config-generation/structural-preflight increment (below). Three
Moriah Slurm jobs, run in sequence against the certified 32-basin Compact
Scientific Package (Gate 4, below), all passed:

- **CPU structural preflight — job `45624926`** (Moriah `glacier`
  partition/CPU node class). 39 checks OK, 0 warnings, 0 errors. Real
  `FlashNHDataset` construction succeeded for train, validation, and test;
  the training scaler was finite; validation and test reused the training
  scaler unchanged; every admitted sample inspected by the preflight was
  finite. Admitted sample counts: train 851,339; validation 274,347; test
  263,637.
- **GPU training smoke — job `45625002`** (Moriah `catfish` partition,
  NVIDIA L4). Target `qobs_mm_per_h_lead06`, sequence length 24, 32 basins,
  2 epochs, 460 static inputs (see exclusion note below). Epoch 1 average
  loss 0.40205; epoch 2 average loss 0.38727. Run directory:
  `/sci/labs/efratmorin/omripo/Flash-NH/runs/stage1_nh_config_lead06_seq24_v001/runs/stage1_compact_lead06_seq24_v001_2307_135829`.
  Retained artifacts: `config.yml`, `model_epoch001.pt`, `model_epoch002.pt`,
  `optimizer_state_epoch001.pt`, `optimizer_state_epoch002.pt`,
  `train_data/train_data_scaler.yml`, `output.log`, TensorBoard event file.
- **Explicit validation + test evaluation — job `45625077`.** Evaluated the
  epoch-2 checkpoint from the run above: validation period calendar-year
  2024, test period calendar-year 2025. Evaluation audit: 217 OK, 0
  warnings, 0 errors. Metrics produced: NSE, RMSE, KGE, Pearson-r, Beta-KGE.
  Retained outputs: `validation/model_epoch002/validation_metrics.csv`,
  `validation/model_epoch002/validation_results.p`,
  `test/model_epoch002/test_metrics.csv`,
  `test/model_epoch002/test_results.p`. Metric values are not interpreted
  scientifically here — this was an integration smoke, not a tuned or
  reportable baseline experiment.

**What this proves.** The Stage 1 package-to-NeuralHydrology pipeline can,
end to end: (1) construct real datasets from the certified Compact
Scientific Package; (2) apply filtering and reuse a single training scaler
across periods without leakage; (3) train on Moriah GPU; (4) save and
reload checkpoints; (5) evaluate held-out validation and test periods; (6)
retain metrics and prediction artifacts on disk.

**What this does not prove.** Final model skill; final hyperparameters;
final sequence length; final static-feature set for the full basin
population; final production-package temporal-coordinate convention;
performance at lead 1 h, 3 h, or 12 h; any spatial-holdout or
full-population scientific conclusion. This closure is an integration gate,
not a scientific result, and is not itself grounds to begin ad hoc
hyperparameter tuning.

**Two findings recorded, not resolved, by this closure** — see
`docs/decision_log.md` (2026-07-23 entry) for full detail:

1. *Compact-smoke-only zero-variance static exclusion.* Across the 32-basin
   smoke population only (not the full package), 13 static attributes had
   zero standard deviation and were excluded for this smoke only (460 of 473
   used): `CANALS_MAINSTEM_PCT`, `CDL_DURUM_WHEAT`, `CDL_ORANGES`,
   `CDL_RICE`, `HGBC`, `PCT_6TH_ORDER_OR_MORE`, `glc_pc_u01`, `glc_pc_u18`,
   `pnv_pc_u02`, `wet_pc_u02`, `wet_pc_u03`, `wet_pc_u07`, `wet_pc_u09`. The
   full 473-column Compact Scientific Package remains authoritative and
   unchanged; this exclusion list must **not** be inherited automatically by
   the full-population baseline, which must independently identify
   zero-variance columns over its own training population.
2. *`time` vs. `date` temporal-coordinate adapter.* The certified compact
   v001 NetCDFs use dimension/coordinate name `time`; NeuralHydrology 1.13
   requires `date` internally. `FlashNHDataset` applies an in-memory
   index-name-only adapter (`src/baseline/nh_dataset.py`); on-disk v001
   files are unchanged. The final on-disk temporal-coordinate convention
   must still be explicitly resolved before the production package format is
   frozen; any such change belongs in a new package version, not a silent
   rewrite of v001.

**Next phase.** Planning the first scientifically meaningful Stage 1
baseline experiments (hyperparameters, sequence-length/lead sweep, the
static-feature set for the full population, spatial-holdout evaluation) —
not implied or started by this closure.

---

### Predecessor: NH config-generation + structural-preflight local implementation increment (2026-07-22)

> **Superseded (2026-07-23):** the "no training" scope below describes
> accurately what this specific increment did. Real dataset construction,
> GPU training, and explicit validation/test evaluation have since run and
> passed — see "Current milestone" above.

**NH config-generation + structural-preflight local implementation increment
(2026-07-22, local-only, no h2o/Moriah access, no training).** Following
Gate 4 certification (below), this increment implements the first local
foundation for compact-package NH integration-validation, strictly scoped
to the first configuration only (lead 6 h, sequence length 24 h, target
`qobs_mm_per_h_lead06`, 8 approved dynamic inputs in binding order, 473
static `model_input` attributes, temporal split train 2020-10-14→2023-12-31
/ validation 2024 / test 2025, same 32 certified compact basins in all three
periods). Scope: `src/baseline/nh_config_generation.py` (config rendering +
basin/date/static-list contracts), `src/baseline/nh_structural_preflight.py`
(two-layer preflight: Layer 1 file-only structural checks against a
generated config bundle; Layer 2 real `FlashNHDataset` construction —
train/validation/test — against synthetic fixtures only, never the real
package), `scripts/generate_stage1_nh_config.py`,
`scripts/check_stage1_nh_config_preflight.py`. Test coverage: 38 tests
passing (25 in `tests/test_nh_config_generation.py`, 13 in
`tests/test_nh_structural_preflight.py`), plus the pre-existing
`tests/test_nh_dataset.py` suite unaffected. No h2o/Moriah access, no data
transfer, no NH training, no Slurm job, no W&B, and only this single
configuration was rendered — not the full 16-config matrix. The certified
Compact Scientific Package itself was not modified or rebuilt.

**Notable discovery: NeuralHydrology 1.13 upstream mutable-default-argument
scaler bug.** `neuralhydrology.datasetzoo.basedataset.BaseDataset.__init__`
declares `scaler: Dict[...] = {}` as a mutable default argument, shared by
Python across every call site in a process that omits `scaler=`. A second
train-period `get_dataset(..., is_train=True, ...)` call in the same
process (no explicit `scaler=`) inherits the first call's already-populated
dict, so its own `not scaler` check is False, `_setup_normalization` is
skipped, and NH silently reuses a stale, unrelated scaler — whose
intersecting xarray arithmetic (`xr - center`) can silently drop dynamic
input/target columns absent from that stale scaler. This only manifests
when a single Python process constructs more than one train-period NH
dataset without passing `scaler={}` explicitly (e.g. a shared pytest
session, or interactive/dev usage) — a real training job (one Slurm
process, one train-dataset construction) is unaffected. Fix applied:
`scaler={}` is now passed explicitly at every train-period `get_dataset`
call site in `nh_structural_preflight.py` and `tests/test_nh_dataset.py`.
This is an NH-mechanics/dev-tooling finding, not a Stage 1 scientific
decision.

**Known documentation debt (not yet resolved):** the committed policy
config declares `nh.dataset: generic`, while the task-mandated / generated
config's `build_nh_config_mapping` hardcodes `dataset: flashnh`. Both are
intentional for their respective purposes (the policy YAML documents the
underlying NH dataset family; the generated config selects the registered
`FlashNHDataset` class) but the discrepancy is not yet called out in-line
in either file and should be reconciled or explicitly annotated in a future
increment.

---

## Compact Scientific Package — Gate 4 certification (2026-07-22)

**Compact Scientific Package — built and independently certified
(2026-07-22).** The 32-basin Compact Scientific Package (built via
`scripts/build_stage1_baseline_nh_package.py`, commit
`89c4dd162f7043419b4b227de5c2bc1b3b230da6`) has been built and promoted on
h2o at `/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001`
(non-authoritative QC evidence at `..._v001_evidence`, run logs at
`..._v001_run_logs`). Builder-level self-validation and an independent
ChatGPT inspection of its compact review bundle are complete. Package facts:
32 per-basin NetCDF files; 45,720 hourly rows/basin; period 2020-10-14
00:00 through 2025-12-31 23:00; 8 approved dynamic inputs; diagnostic raw
`qobs_m3s`; 4 lead targets (1/3/6/12 h); 473 static model-input columns; 138
global gap timestamps (136 MRMS + 2 RTMA); one 15-character basin ID
`393109104464500`.

**Gate 4 independent audit: PASS.** The genuinely independent auditor
(`src/baseline/package_audit.py`,
`scripts/audit_stage1_compact_scientific_package.py`,
`tests/test_package_audit.py`, `docs/stage1_compact_package_independent_audit.md`
— committed `4b524b3851b16baa080d4237622fa7da30e05cea`) was run for real on
h2o against the real package and real source artifacts, in full mode, at
`2026-07-22T08:58:52Z`. Result: **status PASS, 0 errors, 0 warnings, 3,114 OK
checks, exit code 0.** Audit output:
`/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_gate4_audit`.
The auditor re-derives every scientific/structural claim from raw
sources — it does not import
`package_builder`/`package_assembly`/`package_netcdf`/`units`/
`lead_targets`/`gap_mask_io`, so a shared bug cannot pass both the build and
the audit. The build commit (`89c4dd162...`) and the auditor commit
(`4b524b385...`) are intentionally distinct identities. The transferred
audit evidence bundle was independently reviewed by ChatGPT and found
internally consistent; the generated evidence files remain untracked and
are not committed to this repository.

**The package is built and independently certified.** NeuralHydrology
configuration generation is now unblocked. See
`docs/decision_log.md`'s 2026-07-22 certification entry and
`docs/stage1_compact_package_independent_audit.md`'s Status section for
full detail.

---

**Static-attribute matrix v002 — ACCEPTED as canonical Stage 1 baseline
(2026-07-20).** `stage1_static_attributes_v002` (source-checksum-verified
29/29 PASS build via `scripts/build_stage1_static_attribute_matrix.py`,
independently audited PASS by
`scripts/audit_stage1_static_attribute_matrix.py`) is accepted as the
canonical Stage 1 v001-core static-attribute matrix, superseding
`stage1_static_attributes_v001` for modeling. Canonical path:
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/`
(`stage1_static_attributes_v002.parquet`,
`stage1_static_attributes_v002_column_manifest.json`,
`stage1_static_attributes_v002_provenance.json`,
`stage1_static_attributes_v002_audit_summary.md`). Matrix: 2,843 rows × 523
total columns — **473 `model_input`** (authoritative, no longer provisional),
2 split-support, 4 diagnostic lat/lon, 12 diagnostic record/network/QA, 1
deferred-ambiguous (`lka_pc_use`), 29 categorical-deferred, 2 flag. Sentinel
decoding (`stage1_static_sentinel_decode_v1`) replaced 15,018 values across
the 12 mapped columns; the 8 infrastructure-distance `RAW_*` columns are
excluded via the pre-existing `>20%` high-missingness mechanism, not by
name; `PERHOR`/`STRAHLER_MAX` retained `model_input` with sentinels decoded;
`dor_pc_pva`/`dis_m3_pyr`/`run_mm_syr` retained unchanged; direct-coordinate,
record/network/QA, and `lka_pc_use` exclusions all verified; the HydroATLAS
5-basin gap is unchanged and explicitly handled. Independent audit: PASS, 0
errors, 0 warnings, 32 OK checks. Canonical matrix sha256
`4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`;
companion checksums (column manifest, provenance, audit summary) recorded in
`docs/decision_log.md` (2026-07-20 acceptance entry). The full 473-column
`model_input` list is not duplicated in prose — see the canonical column
manifest.

**Compact static-imputation v002 — ACCEPTED (2026-07-20).** Rebuilt against
the accepted v002 matrix via `scripts/prepare_stage1_compact_static_attributes.py`
(algorithm `stage1_static_median_imputation_v1`, unchanged primitive).
Canonical generated output path:
`/data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002`. Input
matrix checksum matches canonical v002 exactly. Output: 32 basins × 473
`model_input` columns; fit scope development-training-only (2,307-basin fit
population); applied to the 32 accepted compact basins; all fit columns had
valid medians; 168 total values imputed, all on a single basin
(`393109104464500`, the same designated compound-edge-case diagnostic basin
noted below); zero remaining NaNs. Output checksums (`imputed_static_attributes.parquet`,
`imputed_value_mask.parquet`) recorded in `docs/decision_log.md`.

**Both v001 artifacts (`stage1_static_attributes_v001` matrix and
`stage1_compact_static_imputation_v001`) remain preserved as historical
provenance — not deleted, not invalid — but are superseded for modeling.**
The selector and canonical split artifacts were **not** rerun; the accepted
32-basin Compact Scientific Package selection (below) remains valid as-is;
the static-imputation primitive (`src/baseline/static_preparation.py`) is
unchanged. No NH package has yet been built; no training has run. Full
detail: `docs/stage1_static_attribute_matrix_plan.md` §12,
`docs/stage1_scientific_baseline_design.md` §3, `docs/decision_log.md`
(2026-07-20 acceptance entry). **Next milestone: Compact Scientific Package
builder planning and implementation (not yet started).**

**Compact Scientific Package selection — ACCEPTED (2026-07-20).** The fully
enriched h2o run of `scripts/generate_stage1_compact_package_selection.py`
(selector commits `71467b5`, `65af017`) is accepted as the project's Compact
Scientific Package basin list. Enriched run inputs: canonical
`split_assignment.csv` (`development_train`, 2,307 basins), canonical
`stage1_static_attributes_v001` matrix + column-role manifest, canonical
`stage1_static_attributes_v001` full-period qobs/target-status table. All
acceptance checks PASS: count=32; development-pool membership PASS;
California exclusion PASS; spatial-holdout leakage PASS; qobs enrichment and
static missingness evaluated for all 32 selected basins; input/artifact
checksums PASS. Accepted characteristics: 13 distinct HUC02s, 7 macro-regions,
east/west split 19/13; area classes high/low/middle = 12/10/10; hydro classes
high/low/middle/missing = 10/11/10/1; qobs completeness high/mid/low =
15/16/1; static missingness none/high = 31/1. Two designated diagnostic
basins: `393109104464500` (compound edge case — `unusual_identifier` +
`hydro_stratifier_gap` + `static_missing_value_case`, 169 missing
`model_input` static attributes) and `05568800` (lowest qobs completeness in
the selection, coverage fraction ≈0.8746). Canonical evidence path on h2o:
`/data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence`
(generated artifact; its `selection_manifest.json` correctly still reports
`"status": "candidate"` per the tool's own generated-artifact convention —
per policy, generated evidence is never hand-edited; **project-level
acceptance is recorded here and in `docs/decision_log.md` instead**). Full
32-basin ID list is not duplicated in this document — see
`compact_basin_ids.txt` in the evidence bundle above, or the local
split-based candidate run described in
`docs/stage1_compact_package_selection.md`. Building the 32-basin NH package
is a separate, not-yet-started step.

> **Superseded (2026-07-21):** the 32-basin NH package has since been built
> and promoted on h2o (build commit `89c4dd162f7043419b4b227de5c2bc1b3b230da6`);
> see the current-state block at the top of this document and
> `docs/stage1_compact_package_independent_audit.md`. It is not yet
> independently certified.
>
> **Further update (2026-07-22):** the package has since been independently
> certified — Gate 4 PASS, 3,114 OK / 0 errors / 0 warnings; see the
> current-state block at the top of this document.

**Scientific target-transformation + static-preparation primitives increment
(2026-07-20).** Reviewed existing code before writing anything new (reuse-first):
`src/baseline/units.py` (m³/s↔mm/h conversion) and `src/baseline/lead_targets.py`
(1/3/6/12 h lead-target shifting) already fully satisfy the discharge-transform
and lead-semantics requirements, with existing test coverage in
`tests/test_units.py`/`tests/test_lead_targets.py` — no new code needed there.
`src/baseline/validity_mask.py` already implements the history/boundary
validity split needed for forcing-gap awareness. Two genuine gaps were found
and filled: (1) `src/baseline/static_preparation.py` — development-train-only
median imputation for `model_input` static-attribute columns, per the
already-signed-off policy (`config/stage1_scientific_baseline_v001.yaml::static_attributes.imputation`,
`docs/stage1_baseline_package_implementation_plan.md` §15); (2)
`src/baseline/gap_mask_io.py` — a loader/writer converting the Milestone 2K-E
forcing-audit's missing-hour-product inventory into the flat
`masks/gap_timestamps.json` format `src/baseline/nh_dataset.py` already
expects but that no script previously produced. Neither changes any signed
scientific decision. **Not done in this increment:** no NH package built, no
`FlashNHDataset`/NH-registration/launcher changes, no training, no Moriah use,
no full 2,752-basin package. See `docs/decision_log.md` for full detail.

> **Superseded (2026-07-21):** the 32-basin NH package has since been built
> (see the current-state block at the top of this document); the
> `FlashNHDataset`/NH-registration/launcher/training/Moriah/full-population
> items in this sentence remain not done.

**2K-G-I I-A1-I-A5 (spatial/temporal split generation through canonical
promotion) COMPLETE (2026-07-16).** Seeded stratified split candidate
(I-A2) passed an independent auditor (I-A3: PASS, 0 errors,
`scripts/audit_stage1_baseline_splits.py`) and human visual QC (I-A4:
PASS, no clustering/imbalance found; see `docs/decision_log.md` for
accepted findings) and was byte-copy-promoted (I-A5) to the canonical
path **`config/stage1_baseline_splits_v001/`** (10 artifacts;
`development_train`/`validation`/`temporal_test` 2307 each,
`spatial_holdout_nonca` 250, `california_finetune_train` 176,
`california_holdout` 19). **The split design is now frozen; do not
reopen it absent a concrete scientific or correctness problem.** Next
work: baseline NH package-builder implementation (remaining 2K-G-I
checklist items below).

**2K-G-H Scientific Baseline Policy Sign-off COMPLETE (2026-07-12) —
docs-only.** Converts the 2K-G-G Phase B evidence (committed at `0d0e6aa`)
into binding Stage 1 decisions in `docs/stage1_scientific_baseline_design.md`.
No new evidence gathered in this patch; no code, config, Slurm script, or NH
package changed. Seven decisions recorded (full detail and rationale in the
design doc's "Binding decisions — Milestone 2K-G-H sign-off" section):
1. **Target scaling (§5):** area-normalized discharge, internal unit mm/h
   equivalent runoff depth, computed by the package builder at
   package-build time; package target column e.g. `qobs_mm_per_h_leadXX`.
   NH's native scaler inversion only returns to mm/h; official evaluation
   requires an additional Flash-NH-side mm/h→`m^3/s` conversion using basin
   area. NH loss/validation curves are training diagnostics in transformed
   space; official benchmark metrics are always Flash-NH-computed raw-space
   `m^3/s` after full inverse conversion.
2. **Target inversion/audit requirements (§5a, new):** deterministic
   `m^3/s -> mm/h -> m^3/s` round-trip unit tests; a package audit
   requirement (`qobs_mm_per_h_leadXX` at `t` == `qobs_m3s` at `t+XXh`
   converted to mm/h, on random basin/time samples); an evaluation audit
   requirement (raw-space metric scripts verify units, basin area, lead
   alignment, NaN masking, and conversion back to `m^3/s`).
3. **Lead-time implementation (§9b):** package-build-time target shifting
   (no native NH `lead_time` config, per Q9 evidence). All four lead times
   — 1/3/6/12 h — included in the first package/config/sweep design (not
   just 6 h/12 h); primary benchmark lead 6 h, secondary 12 h, 1 h/3 h
   diagnostics included now for low incremental cost. `seq_length` and lead
   time remain separate axes.
4. **Forcing-gap policy (§6):** scientific baseline hard-excludes training
   windows intersecting MRMS archive-gap hours (Policy B), accepted because
   corrected real-gap window loss is modest (~1.3% at `seq_length=12` to
   ~5.6% at `seq_length=72`). RTMA (2 gap hours vs. MRMS's 136) may be
   folded into the same exclusion mask if that's free, but MRMS drives the
   policy. `nan_handling_method` (Policy A) remains a fallback/ablation path
   only, not the baseline; unset/default `None` remains forbidden in any
   run per Q6 evidence.
5. **Static attributes (§3) — HISTORICAL, SUPERSEDED (see 2026-07-20 state at
   top of this document).** At the time of this 2K-G-H sign-off
   (2026-07-12), canonical `stage1_static_attributes_v001`
   (2,843 × 531 cols, 496 `model_input`, h2o canonical PASS 2026-07-08) was
   accepted as the Stage 1 baseline static matrix, replacing the earlier
   48-column merge. Numeric attributes pass through NH's standard
   static-attribute pathway; no categorical embeddings in this first
   baseline. `STATE`/`HUC02` remain split-support/diagnostics only;
   `LAT_GAGE`/`LNG_GAGE` remain diagnostic only, deferred to a later
   ablation. **This v001 matrix was superseded on 2026-07-20 by the accepted
   `stage1_static_attributes_v002` (2,843 × 523 cols, 473 `model_input`) —
   see the acceptance record at the top of this document and
   `docs/decision_log.md` for the current binding static-attribute state.**
6. **Spatial split and leakage (§8b):** reproducible seeded stratified
   non-CA spatial holdout (mechanism unchanged), stratifying on at least
   HUC02/geography, basin area, and hydroclimatic/aridity attributes from
   `stage1_static_attributes_v001`. California excluded from Stages 1–3
   (unchanged, §8c). Explicit basin-list artifacts required for
   development-training/validation/temporal-test/non-CA-spatial-holdout/
   California-Stage-4 splits — none exist yet, spatial leakage prevention
   remains a Flash-NH basin-list responsibility (Q4 evidence).
7. **Next milestone defined, not started:** `2K-G-I — Baseline Package
   Builder + Split Config Implementation` — a checklist (target conversion,
   lead-time shifting, raw-`m^3/s` audit, MRMS-gap exclusion, basin-list
   artifacts, `stage1_static_attributes_v001` adoption, baseline NH
   YAML/config, package audit updates), not code, not executed in this
   patch.

No scientific/methodological item remains open pending evidence in
`docs/stage1_scientific_baseline_design.md`; what remains is 2K-G-I's
implementation work.

**2K-G-G Phase B evidence-gathering COMPLETE (2026-07-12) — Target
Scaling + Gap Policy + Lead-Time Feasibility Report.** All 9 NH-mechanics
questions (Q1-Q9) and the window-feasibility questions (Q10-Q11) are now
answered from authoritative Moriah NH 1.13.0 evidence; zero items remain
`REQUIRES TARGETED SOURCE INSPECTION`. Sequence across two follow-up
rounds this date:
- **Part 1 (Moriah SSH still broken from this session):** Closed Q10/Q11
  (window/sample-loss numbers) using the real `fullperiod_gap_inventory.csv`
  (from the 2026-06-24 full-period forcing postrun audit) — no Moriah run
  needed, since `scripts/analyze_stage1_window_feasibility.py` imports no
  NeuralHydrology. Either-gap window loss ranges from ~1.3%
  (`seq_length=12`) to ~5.6% (`seq_length=72`) across the full 12/24/48/72 h
  x 1/3/6/12 h design space; MRMS-gap loss dominates RTMA-gap loss by ~2
  orders of magnitude (136 vs. 2 archive-gap hours). Also found and fixed
  a real timezone-handling bug in that script (real gap-inventory
  timestamps are `Z`-suffixed/tz-aware, the internal hourly index was
  tz-naive, so the first real-gap run silently reported 0% gap-loss
  everywhere) via a `_to_naive_utc()` helper, regression-tested clean.
  Refined the Q4 leakage finding to explicitly distinguish temporal
  leakage (NH's `is_train`/passed-`scaler` contract protects this
  automatically) from spatial leakage (California/spatial-holdout basins —
  NH provides zero automatic protection; Flash-NH's basin-list
  construction upstream of NH is solely responsible). Added a gap-policy
  decision framework (Policy A: NaN + `nan_handling_method`, vs. Policy B:
  hard window exclusion) informed by the real loss numbers, without
  selecting between them. The 3 remaining NH-mechanics items were blocked
  this part because this working session had no SSH/network path to
  Moriah.
- **Part 2 (Moriah SSH access restored):** the user fixed Moriah SSH
  access from local Windows/VS Code (`ssh moriah "hostname"` ->
  `moriah-gw-01`; note default `scp`/SFTP is still broken on Moriah, plain
  SSH command execution works — legacy `scp -O` needed for any future file
  transfer, not used this round since only inline `sed`/`grep` output was
  needed). All 3 remaining items were closed by inspecting the Moriah
  1.13.0 source directly: (1) **Q2 confirmed** — `tester.py:247-259`'s
  exact inverse-scaling arithmetic is `raw = scaled * feature_scale +
  feature_center` for both predictions and observations, inline, no
  public `inverse_transform` API exists; (2) **Q5 confirmed** —
  `training/loss.py` masks target NaNs per-element in every one of 6
  `Masked*Loss` classes (`MaskedMSELoss`, `MaskedRMSELoss`,
  `MaskedNSELoss`, `MaskedGMMLoss`, `MaskedCMALLoss`, `MaskedUMALLoss`) via
  `~torch.isnan(ground_truth['y'])`-style masking before the loss
  reduction — a target NaN cannot silently contaminate training through
  any NH-provided loss class; (3) **Q6/Q7 confirmed, and the default is
  dangerous** — `nan_handling_method` defaults to `None` when unset
  (`utils/config.py:610-613`), and the unset case falls through to a final
  `else` branch in `modelzoo/inputlayer.py` that performs **no NaN
  handling at all**, passing raw (possibly NaN) dynamic inputs straight
  into an unprotected `nn.Linear` embedding — explicit configuration
  (`masked_mean`, `attention`, or `input_replacing`) is mandatory, not
  optional, for Flash-NH's gap-policy Policy A to be safe. Raw command
  output saved to
  `tmp/nh13_targeted_inspection_moriah_20260712T120839Z/` (gitignored, not
  committed).

No target-scaling, gap-policy, or lead-time implementation decision has
been made final; no NH package generated; no training run; no package
builder/config/Slurm template modified.

**2K-G-F-B COMPLETE — canonical h2o PASS (2026-07-08) — static attribute
source mirror + derived matrix builder/auditor.** Implements
the 2K-G-F plan (`docs/stage1_static_attribute_matrix_plan.md`) in code:
- `scripts/build_stage1_static_attribute_matrix.py` — merges the 29-file
  GAGES-II/HydroATLAS/NLDAS-2 source mirror into
  `stage1_static_attributes_v001.parquet` for the 2,843-basin Stage 1
  universe, applying the conservative column-classification policy (admin/
  duplicate/binary-flag/categorical-deferred/split-support/diagnostic-lat-lon
  handling, per-year-series reduction, dynamic near-constant/high-missingness
  exclusion) and the mandatory HydroATLAS 5-basin-gap gate (fail loud unless
  the observed gap exactly matches the known 5 non-standard-ID basins).
- `scripts/audit_stage1_static_attribute_matrix.py` — independently verifies
  the output (coverage, duplicates, missingness, ranges, constant/duplicate
  columns, categorical/ID-name leakage, `STATE`/`HUC02`/lat-lon exclusion,
  HydroATLAS gap handling, checksum).
- **Local dry-run PASS** against `C:\PhD\Python\neuralhydrology\US_data\attributes`
  into repo `tmp/` (gitignored, not committed): build exit 0 (2,843 rows ×
  531 cols, 496 `model_input`), audit exit 0 (0 errors, 0 warnings, 20 OK
  checks). This validated the build/audit *logic* only.
- **Canonical h2o build/audit: PASS (2026-07-08).** Run by the user directly
  on h2o (no network path exists from this environment to h2o). Source mirror
  verified (30 files = 29 source files + checksum file, all 29 `sha256sum -c`
  OK). Canonical build + audit: 0 errors, 0 warnings, 20 OK checks, matrix
  2,843 rows × 531 columns / 496 `model_input`, HydroATLAS 5-basin gap matched
  exactly, checksum verified. Canonical artifact:
  `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet`,
  sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`.
  Full detail in `docs/stage1_static_attribute_matrix_plan.md` §11.6.
- Minor correction carried over from 2K-G-F: the source mirror has **26**
  `attributes_gageii_*.csv` files, not 27 as previously stated (29 total
  files is unaffected: 26 + HydroATLAS + NLDAS-2 + workbook).

No NH package was regenerated from this matrix; no training was run; no NH
config/Slurm scripts were modified — all explicitly out of scope for this
milestone.

**2K-G-F DONE (2026-07-06) — static attribute matrix recovery + audit
plan.** Inventoried the full local source directory
(`C:\PhD\Python\neuralhydrology\US_data\attributes`: 29 CSVs + 1 variable
description workbook, all keyed on `STAID`, all 9,008 rows) and cross-checked
it against the real Stage 1 basin manifest (`config/stage1_initial_training_basin_manifest.csv`,
2,843 basins). Key findings, recorded in the new
`docs/stage1_static_attribute_matrix_plan.md`:
- **100% GAGES-II source coverage** of the Stage 1 basin set once `STAID` is
  zero-padded to 8 characters, including all 6 non-standard-length USGS IDs
  (five 15-char, one 9-char coordinate-based station numbers).
- **HydroATLAS covers 99.8%** (2,838/2,843) after zero-padding — the 5-basin
  gap is exactly the 15-char non-standard IDs (HydroATLAS's own `STAID` export
  is not zero-padded, unlike the GAGES-II CSVs). **Clarified as a mandatory
  build/audit gate** (same-day follow-up review): the builder/auditor must
  detect these 5 basins and either resolve/match them, retain them under a
  documented missing-value policy, or fail the build — no silent partial
  HydroATLAS merge is allowed.
- **Confirmed the existing canonical 48-column parquet stores `STAID` as
  `int64`** (leading zeros stripped) — already handled by the builder's
  `_norm_staid()`, but any new merge/audit script must reimplement 8-char
  zero-padding independently; do not assume any source/intermediate file
  preserves it.
- **The existing 48-column merge draws from only 3 of the 27 GAGES-II source
  files** — no topography, geology, land cover/vegetation, or snow fraction.
  The richer sources (Topo, Geology, 5× land-cover buffers, Soils, Climate,
  HydroATLAS, NLDAS-2) are cataloged and ready to merge; **snow fraction is
  only available via HydroATLAS** (`snw_pc_*`), not any GAGES-II file.
- Full audit of all 780 non-ID source columns restricted to the Stage 1
  basins: 758 numeric-like / 22 non-numeric (classified into
  drop/binary-flag/categorical groups); only 6 columns >20% missing; 20
  near-constant columns; one duplicate column (`DRAIN_SQKM`).
- Proposed canonical paths (h2o source mirror, Moriah source mirror, derived
  matrix path — see plan doc §7) and a merge/audit policy (§8–9). h2o/Moriah
  mirror status **not checked from this session** (no network path from this
  environment) — explicit user-side check/transfer commands documented in
  the plan doc §6 instead.
- **Filtering philosophy decided (same-day follow-up review): conservative by
  default.** Ambiguous/borderline variables (problematic, non-physical,
  administrative, weakly useful, leakage-prone, near-constant, high-missingness,
  hard to interpret) are excluded from `v001-core` by default, not kept on
  the chance the model learns something — a defensible small first matrix is
  preferred over a maximal one. Under this policy, `STATE`/`HUC02` are
  excluded outright from `v001-core` model inputs (kept only for split
  construction/diagnostics/reporting); lat/lon are held out of `v001-core`
  by default and deferred to a dedicated ablation on spatial generalization.

**No final static-attribute matrix was built.** No code, config, package,
Slurm script, or training changed; no h2o/Moriah transfer performed. The
per-column audit CSV produced during inspection is a local scratch artifact
only, outside the repo.

**2K-G-E REVISED (2026-07-06) — scientific baseline design aligned to
user-approved decisions; two new gating mini-milestones defined.**

The first 2K-G-E proposal (2026-07-03) was never committed; user review
changed several key decisions before commit, so `docs/stage1_scientific_baseline_design.md`
was revised in place rather than amended. 14 binding decisions are now
recorded there ("Binding decisions" section). Highlights of what changed from
the 2026-07-03 draft:
- **Static attributes reopened:** the draft ~16-column sign-off candidate is
  **withdrawn**. The 48-column GAGES-II screening merge remains a valid,
  checksum-verified provenance artifact but is likely insufficient as the
  final modeling matrix — richer source material exists locally
  (`US_data/attributes`, ~28 files, ~350-variable description workbook) and
  is not yet mirrored to h2o/Moriah. Gated on new **Milestone 2K-G-F**.
- **Target normalization:** log-transform **rejected** (poorly aligned with
  flash-flood/high-flow emphasis). Leading candidate is now area-normalized/
  specific discharge, pending feasibility. Gated on new **Milestone 2K-G-G**.
- **`seq_length`:** narrowed and made **binding** — Stage 1 candidates are
  only 12/24/48/72 h; 168/336 h explicitly belong to Stage 2, not Stage 1
  (withdraws the draft's 336 h literature-based proposal).
- **Lead time:** new design axis added — primary 6 h, secondary 12 h, 1/3 h
  diagnostic-only; explicitly separate from `seq_length`.
- **Temporal split dates revised:** train 2020-10-14→2023-12-31, validation
  2024, test 2025 (was train ≤2022-12-31 / val 2023 / test 2024–2025).
- **Spatial split added:** California excluded entirely from Stages 1–3;
  ~10% non-CA spatial holdout, test-only, evaluated on the 2025 test period.
- **California transfer learning (Stage 4) added:** ~90/10 CA split,
  CA-only normalization-refit exception for fine-tuning, compare
  original-vs-fine-tuned on CA holdout.
- **Leakage-prevention rules made explicit:** all Stage 1–3 scalers fit only
  on development-training data/period; Stage 4 CA scalers fit only on the CA
  fine-tuning subset.
- **Loss vs. metrics separated:** training loss still open (depends on target
  scaling); evaluation always in raw `m^3/s`, raw-space NSE primary.
- **Hyperparameters reframed:** the conventional table is now an *initial
  seed config* only — the official benchmark requires a W&B sweep, not yet
  run.
- **W&B policy expanded:** loss/validation curves, LR, epoch timing, run
  duration, GPU type, resource telemetry, in addition to config/provenance.
- **Slurm policy:** stays flexible/parameterized, not hard-pinned; resources
  may be increased later based on telemetry.

A "Before full 2,752-basin NH package generation" checklist was updated to
include the two new mini-milestones. **No code was changed** — this remains
documentation-only; no config written, no package generated, no training run,
no Moriah/California data transfer.

**2K-G-D-A COMPLETE (2026-07-03) — canonical attribute artifact promoted off `tmp`;
h2o checksum verification PASS.**

2K-G-D (same day) identified the static attribute file as an external,
checksum-pinned artifact (per `docs/repo_policy.md` generated-artifact policy — a
generated data product is not committed regardless of its small size) but left it
resident under `/data42/omrip/Flash-NH/tmp/`, and left the h2o-copy checksum
unverified pending h2o shell access. 2K-G-D-A closes both:
- **Promoted** the canonical h2o-resident copy from
  `/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` (now historical/staged only)
  to the stable project data path
  `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`.
- **Verified** (user-run on h2o): sha256 of both the `tmp/` copy and the newly
  promoted stable-path copy is
  `06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1` (`ls -lh`: 2.9M
  both) — identical, matching the local repo-fixture checksum recorded at 2K-G-D.
  Full evidence: `docs/stage1_attribute_provenance.md`.
- The parquet itself is still **not committed to git** — only the checksum,
  path, and provenance are documented. `attributes_sha256` continues to be
  written into every package's `run_provenance.json`
  (`scripts/build_stage1_nh_package.py`).
- Remaining open item (non-blocking): the Moriah mirror path
  (`/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`)
  is documented but not yet populated or verified.

A design-gate scaffold for the first scientific baseline is now at
`docs/stage1_scientific_baseline_design.md`: purpose/non-goals, dynamic-input set,
static-attribute subset, target cleaning/normalization, forcing-gap policy, loss/metrics,
train/val/test protocol, W&B policy, Slurm partition/GRES parameterization, and evidence
bundle conventions. Most items are explicitly marked **OPEN** — this is a decision
scaffold, not a locked spec. **Correction to prior framing:** earlier entries below
described "lookback-expansion tests (seq_length 72/168/336)" as the next milestone —
`seq_length` is one hyperparameter decided inside the design gate (§9), not the
milestone driver.

**Smoke 1 PASS (2026-07-02) — meteorology ingestion confirmed (retained for reference).**

**This is a technical meteorology-ingestion PASS, not a scientific baseline.**
6 RTMA vars + MRMS QPE + 2 gap flags, `seq_length=24` (same as Smoke 0 — isolates input
expansion from lookback change), 5 basins, 3 epochs, loss NSE.
Purpose: confirm RTMA meteorology loads, normalizes, and trains without error.

**Smoke 1 facts (Slurm job 45370873, 2026-07-02):**
- Node: `catfish-04` (NVIDIA L4, `catfish` partition); wall time: 00:01:41; exit 0:0
- MaxRSS: 1,380,944 KB (~1.35 GB batch step)
- Same package as Smoke 0 (h2o audit 2026-07-02T11:44:43Z — PASS, 0 errors)
- Config: `seq_length: 24`, `epochs: 3`, `loss: NSE`, 8 dynamic inputs — all source-built
- `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms` all non-null ✓
- `rtma_2d_K` non-null confirms 2K-F-C-B dewpoint fix carried through correctly
- Epoch 1: 0.00422 (finite ✓); Epoch 2: 0.00360 (finite ✓); Epoch 3: 0.00335 (finite ✓)
- All 3 epochs show monotonically decreasing loss; validation completed each epoch
- Run dir: `/sci/labs/efratmorin/omripo/Flash-NH/runs/flashnh_stage1_smoke1_0207_164941`
- Artefacts: `model_epoch001/002/003.pt` (~83 KB each), optimizer states, TensorBoard events

**[Historical — this "Next"/"Remaining gates" block predates Milestones
2K-G-F-B, 2K-G-G, and 2K-G-H and is superseded by them; see the 2K-G-H
block at the top of this file. Retained for reference only, not current
guidance.]**

**Next: close out attribute-policy sign-off (2K-G-F/2K-G-F-B close-out) now
that the canonical h2o build/audit has PASSed. In parallel, run Milestone
2K-G-G (target scaling + gap policy +
lead-time feasibility report — requires actual NH 1.13 code inspection on
Moriah), then close the remaining sign-off items in
`docs/stage1_scientific_baseline_design.md`, select the non-CA
spatial-holdout basin list, encode the resolved policy into
`config/stage1_scientific_baseline_v001.yaml` + NH YAML, run the W&B
hyperparameter sweep, then generate the full 2,752-basin NH package.**

**Remaining gates before full 2,752-basin NH package + scientific baseline:**
- ~~Attribute provenance / checksum verification~~ — **CLOSED 2K-G-D-A (2026-07-03)**.
  Canonical path promoted off `tmp`; h2o checksum verified PASS (48-column
  screening merge only — see next item for the modeling-matrix gate).
- **Milestone 2K-G-F (Static Attribute Matrix Recovery + Audit): plan done
  2026-07-06**; **2K-G-F-B (builder/auditor + local dry-run + canonical h2o
  build/audit) COMPLETE 2026-07-08** (`docs/stage1_static_attribute_matrix_plan.md`
  §11) — source inventory, coverage cross-check, column-classification
  policy, builder/auditor scripts, local dry-run, and canonical h2o
  build/audit (PASS) are all complete. **Not done:** Moriah mirror of the
  source attributes and derived matrix, attribute-policy final sign-off.
- **NEW — Milestone 2K-G-G (Target Scaling + Gap Policy + Lead-Time
  Feasibility Report):** not started. Requires reading actual NH 1.13 code
  on Moriah (not docs/assumptions) to resolve target-normalization
  feasibility (§5) and forcing-gap-policy feasibility (§6), and to quantify
  sample/window loss across `seq_length`×lead-time combinations.
- Scientific-baseline design gate: **REVISED into 14 binding decisions
  2K-G-E (2026-07-06)** — see `docs/stage1_scientific_baseline_design.md` →
  "Binding decisions." Still open pending 2K-G-F/2K-G-G: target
  normalization, forcing-gap policy, static-attribute column list. Also
  still open: non-CA spatial-holdout basin selection (~10%), California
  basin identification for Stage 4, `seq_length`/lead-time final selection
  (via W&B sweep, not yet run).
- Slurm templates (smoke0/1 sbatch) are hard-pinned to `catfish/L4`. Policy
  is to keep this parameterized/flexible (§11) — sbatch edit itself still
  deferred to when the baseline config is assembled.
- Moriah mirror of the attribute file (documented path, not yet populated/verified) —
  only needed if a Moriah-side build reads the attribute file directly.
- Revised temporal split dates (§8) and California exclusion are **not yet**
  encoded in `scripts/build_stage1_nh_package.py`'s split constants — that is
  a code change, out of scope for documentation-only milestones.

---

**Smoke 0 PASS (2026-07-02) — technical plumbing confirmed (retained for reference).**

Rain-only (`mrms_qpe_1h_mm` + `mrms_qpe_1h_mm_gap`), `seq_length=24`, 5 basins, 2 epochs.

**Smoke 0 facts (Slurm job 45370683, 2026-07-02):**
- Node: `catfish-05` (NVIDIA L4, `catfish` partition); wall time: 00:01:55; exit 0:0
- Package regenerated on h2o (2026-07-02T11:43:53Z) with patched builder; h2o audit PASS
- Config: `dataset: generic`, DD/MM/YYYY dates, `epochs: 2`, `head: regression`,
  `output_activation: linear` — all from source (not manual edits)
- `attributes/attributes.csv: OK` (new canonical layout)
- Epoch 1 avg loss: 0.00577 (finite ✓); Epoch 2 avg loss: 0.00556 (finite ✓); validation completed
- Run dir: `/sci/labs/efratmorin/omripo/Flash-NH/runs/flashnh_stage1_smoke0_0207_153320`
- Artefacts: `model_epoch001.pt`, `model_epoch002.pt` (~77 KB each), optimizer states, TensorBoard events

---

**NH 1.13 compatibility patch applied 2026-07-02 (commits 5e8a334 + 60fce38).**

Manual Smoke 0 diagnostic attempts revealed NH 1.13 config/layout incompatibilities in
the original builder. Source corrected; regenerated package passed h2o audit and Moriah Smoke 0:
- `dataset: generic` (was `GenericDataset`)
- All `_date` fields: `DD/MM/YYYY` (was ISO `YYYY-MM-DD`)
- `epochs` key (was `num_epochs`; rejected by NH 1.13)
- `head: regression`, `output_activation: linear` (were absent)
- `shuffle`, `log_n_basins` removed (rejected by NH 1.13)
- `attributes/attributes.csv` canonical layout (was root-level `attributes.csv`)
- Package-internal `slurm/` no longer generated; repo-level sbatch is the Slurm entry point

---

Pre-conditions completed 2026-07-01:
- Moriah `flashnh-moriah` env installed (Slurm job `45365952` PASS)
- Corrected full-period curated forcing v001 built on h2o (PASS — see below)

---

**Moriah env install PASS (2026-07-01): Slurm job `45365952`.**

- Env prefix: `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah`
- `torch==2.7.0+cu128` installed; `nh-run` present at `envs/flashnh-moriah/bin/nh-run`
- `nh-run --help` confirmed: valid modes `train`, `continue_training`, `finetune`, `evaluate`
- `neuralhydrology` import OK (no `__version__` attribute — expected)
- Log ended with `=== done ===`

**Module fixes applied to both Moriah sbatch scripts** (initial non-interactive shell failure resolved):
1. Source module-system init file if `module` not in PATH
2. `module load spack/all` before any other module
3. Exact module name `miniconda3/24.3.0-gcc-iqeknet` (not `miniconda3/24.3.0`)

**Pilot package transfer to Moriah PASS (2026-07-01):**
- Path: `/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001`
- Verified: 5 NC files, `run_provenance.json`, `configs/stage1_smoke0_nh.yml`,
  `attributes.csv` — all present; size 19 MB.

---

**Corrected full-period curated forcing v001 PASS (2026-07-01): 2,752 basins × 45,720 h.**

Build facts (evidence bundle: `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/`):
- h2o path: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001`
- 63/63 months; 2,752/2,752 basins; 45,720 rows/basin; 0 failures
- MRMS gap-hours: 374,272 (= 136/basin × 2,752 — exact); RTMA gap-hours: 5,504 (= 2/basin × 2,752 — exact)
- Wall time: 14.43 h (2026-06-30T10:09Z → 2026-07-01T00:34Z); repo commit at run: `5f07d4b`

Audit (full-period mode): **PASS** — 2,752/2,752 basins checked; all row/gap counts exact.
Sample20 diagnostic: **ALL PASS** — `rtma_2d_K` populated ✓; `rtma_weasd_kgm2` absent ✓.
Generated evidence not committed (local: `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/`).

This closes the 2K-F-C-B corrected-rebuild loop. Full 2,752-basin NH package generation
is unblocked from the forcing side; remaining gates: Smoke 0 PASS + attribute-source cleanup.

---

**Milestone 2K-G-C-A COMPLETE (2026-06-30): Moriah GPU/Conda/Slurm preflight documented.**

Facts confirmed via `ssh`/`srun` reconnaissance on 2026-06-30:
- Login `moriah-gw-01`; project root `/sci/labs/efratmorin/omripo/Flash-NH`
- Partitions: `catfish` (L4, `gpu:l4:8`, 7-day) — chosen for Smoke 0; `salmon` (L40S);
  `goldfish` (H200); `dogfish` (A100, drained); `glacier` (CPU default)
- GPU node `catfish-05`: NVIDIA L4, 23034 MiB; driver 580.95.05 / CUDA 13.0;
  toolkit `cuda/12.8.1` (`nvcc` → 12.8.93)
- Conda module-gated; prefix env pattern under Flash-NH project root confirmed
- Two Slurm templates prepared (`setup_flashnh_moriah_env.sbatch`, `run_stage1_smoke0_moriah.sbatch`)

Full detail: `docs/stage1_neuralhydrology_preflight.md` §10.6.

---

**Milestone 2K-G-B COMPLETE (2026-06-30): NeuralHydrology pilot package built and audited on h2o.**

h2o audit result: **PASS** — 0 errors, 5 warnings, 217 OK checks.
Build time: 4.2 s. Audit timestamp: 2026-06-30T12:38:40Z.
Package: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
Evidence: `tmp/stage1_nh_pilot_v001_evidence/` (not committed)

**5-basin audit summary (all pass):**

| Basin | Rows | MRMS gap | RTMA gap | qobs NaN | qobs coverage |
|---|---|---|---|---|---|
| 01019000 | 45,720 | 136 | 2 | 515 | 98.87% |
| 01022500 | 45,720 | 136 | 2 | 6,751 | 85.23% |
| 01033000 | 45,720 | 136 | 2 | 12,088 | 73.56% |
| 01038000 | 45,720 | 136 | 2 | 3,035 | 93.36% |
| 01049500 | 45,720 | 136 | 2 | 6 | 99.99% |

**5 warnings (all expected):** one per basin — qobs NaN counts logged (normal; NH loss-masks missing targets).
No forcing NaN warnings. All forcing variables non-null after gap-fill.

**Key checks confirmed:**
- All 14 variables present per NC (11 forcing + 2 gap flags + qobs_m3s)
- `rtma_weasd_kgm2` absent (forbidden — confirmed)
- `rtma_2d_K` non-null == 45,720 (confirms 2K-F-C-B dewpoint mapping fix carried through)
- `mrms_qpe_1h_mm_gap sum == 136` per basin; `rtma_gap sum == 2` per basin
- Gap fill: MRMS 136 NaN → 0.0 mm/basin; RTMA 2 NaN → linear interp per variable/basin

**Static attribute caveat (cleanup required before full-scale package) — RESOLVED 2K-G-D-A
(2026-07-03), see top of this document and `docs/stage1_attribute_provenance.md`:**
`reports/flashnh_basin_screening_v001/all_basins_merged.parquet` is **not tracked in git**
(verified with `git ls-files` on h2o). The h2o builder used a manually staged copy at
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` (this path is now historical/staged
only — promoted to `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`,
checksum-verified, at 2K-G-D-A).
The 5-basin pilot PASS is valid. Before full 2,752-basin NH package generation, this file
must be made canonical: committed to the repo or documented as a stable h2o-resident input
with explicit provenance. This is a cleanup gate, not a blocker for Moriah transfer.

**Package structure (on h2o):**
```
/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/
  time_series/{STAID}.nc     # 5 NCs; 14 vars; 45,720 rows; float32; _FillValue=-9999.0
  attributes.csv             # 5 basins × 47 cols
  basins/smoke{0,1}_{train,val,test}.txt
  configs/stage1_smoke{0,1}_nh.yml
  slurm/smoke{0,1}.sh        # Moriah Slurm job templates
  manifests/                 # dataset_manifest.json + variable_schema.csv + gap_fill_report.csv + per_basin_summary.csv
  run_provenance.json + README.md + audit_summary.md
```

**Next: 2K-G-C — Moriah transfer + environment preflight + Smoke 0.**
Transfer pilot package (`scp`), confirm NH conda env on GPU node, run Smoke 0 (seq_length=24, 2 epochs).
No NH training has run yet. Full 2,752-basin NH package generation waits for:
(1) corrected full forcing rebuild PASS on h2o; (2) attribute-source cleanup
(**resolved 2K-G-D-A, 2026-07-03** — see top of document).

---

**Milestone 2K-G-A COMPLETE (2026-06-30): NeuralHydrology pilot package preflight design + corrections.**

Design frozen in `docs/stage1_neuralhydrology_preflight.md` (Part I), with corrections applied
after initial commit `fa6754b`:
- NH package format: GenericDataset single-NC-per-basin, `date` coord, float32, `_FillValue=-9999.0`
- Smoke 0: rain-only (mrms_qpe_1h_mm + gap flag, 5 basins, 1–2 epochs); `seq_length: 24`, `predict_last_n: 1`
- Smoke 1: minimal meteorology (6 forcings: mrms + rtma_{2t,2d,2sh,10u,10v}); `seq_length: 24`
  (same as Smoke 0 — isolate input expansion from lookback change; 72/168 h are later separate tests)
- Gap-fill policy (Smoke 0/1 pilot policy only): MRMS gaps → 0.0 mm; RTMA gaps → linear interp; gap flags retained
- Final training gap policy: window-exclusion preferred over silent fill; to be decided after Smoke 1
- Moriah layout: `/sci/labs/efratmorin/omripo/Flash-NH/{repos,envs,data,runs,logs,slurm,evidence}`
- NH setup: clean upstream `neuralhydrology` clone; no fork until specific limitation demonstrated

---

**Milestone 2K-F-C-B COMPLETE (2026-06-30): Curated forcing schema/mapping correction.**

Full-period build structurally PASS on h2o (2026-06-30, 2,752 basins, 45,720 h, 14.49 h wall),
but post-build non-null check found two all-NaN variables. Build is **schema-superseded**;
corrected rebuild required before final certification.

**Schema issues found and corrected in code:**
- `rtma_2d_K` (dewpoint): all-NaN because builder mapped source `d2m` → `rtma_2d_K`, but
  actual source variable is `2d`. Fixed: `"2d" → "rtma_2d_K"` in both builders.
- `rtma_weasd_kgm2`: all-NaN because `weasd` is absent from all 63 monthly source chunks.
  RTMA precipitation (`ACPC01`) is not present. Removed from schema entirely.
- `rtma_2d_K` is **retained** (source `2d` confirmed present in all sampled months with
  `variable_standard_name=dewpoint_temperature_2m`).

**Corrected v001 schema:** 1 MRMS variable + 10 RTMA variables + 2 gap flags = 13 columns.

**Full-period structural build evidence (schema-superseded, not committed):**
- Period: 2020-10-14T00Z – 2025-12-31T23Z
- 63/63 months, 2,752 basins, 45,720 rows/basin, 374,272 MRMS gap-hrs, 5,504 RTMA gap-hrs
- Full-period audit: PASS (structural); wall time 14.49 h; commit at run `addfdd2`
- Note: accidental second launch was stopped early; post-interruption audit PASS confirmed
  product not corrupted. `build.log` may contain aborted-rerun lines after first PASS.
- Evidence under `tmp/stage1_curated_forcing_v001_schema_issue_evidence/` (not committed)

**Corrected full-period rebuild:** PASS (2026-07-01) — see current milestone block above.
**Design doc:** `docs/stage1_curated_forcing_product_v001_design.md`

---

**Milestone 2K-F-B COMPLETE (2026-06-29): Curated forcing product v001 builder + smoke test — PASS.**

Builder (`build_stage1_curated_forcing_basin_parquets.py`), auditor
(`audit_stage1_curated_forcing_basin_parquets.py`), and h2o launcher implemented and committed
(`6f4de49`). 5-basin / 2020-11 smoke test run on h2o: all 9 acceptance criteria PASS.
- 5/5 basins (`01440000`, `03021350`, `08155541`, `09484000`, `01019000`); 720 h each
- 0 MRMS gap-hours; 10 RTMA gap-hours (2/basin) at 2020-11-12T09Z/T10Z
- Coverage 0.9972; `rtma_gap=True` confirmed at both known timestamps; MRMS not falsely flagged
- Auditor exit 0; SHA-256 checksums verified; commit at run `6f4de498`
- Note: `02231000` attempted but absent from 2020-11 source; builder correctly halted; not a failure
- h2o output: `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z`

---

**Milestone 2K-E COMPLETE (2026-06-24): Full-period forcing extraction audit — PASS_WITH_CAVEATS.**

Full-period MRMS+RTMA basin-average forcing extraction (63 months, 2020-10 → 2025-12,
2,752 basins) is complete on h2o. Post-run audit finished locally.

**Audit result summary:**
- 63/63 months `all_pass=True`, 0 failures
- 1,509,422,464 combined rows (125,447,168 MRMS + 1,383,975,296 RTMA); 0 row-count mismatches
- 11 RTMA variables, uniform; `rtma_10wdir_absent` and `rtma_orog_absent` confirmed all months
- 138 missing hour-products across 20 months (136 MRMS + 2 RTMA), all `not_in_s3`
- MRMS 24h window impact: 949 / 45,697 windows (2.08%); RTMA: 25 / 45,697 (0.05%)
- 0 basin×product pairs incomplete across all months
- 0 unexpected warnings
- Caveat: two-commit provenance (2020-10 → `194a489`; 2020-11 → 2025-12 → `7e43760`); documentation only
- **No rerun required**

**Full audit result:** `docs/stage1_forcing_fullperiod_audit.md`  
**Audit plan:** `docs/stage1_forcing_fullperiod_postrun_audit_plan.md`  
**Generated audit tables (not committed):** `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/`

**Next step:** Milestone 2K-F-B — curated forcing product v001 builder + 5-basin smoke test on h2o.
Design frozen in `docs/stage1_curated_forcing_product_v001_design.md`. Not model training yet.

**Pilot visual QC PASS (2026-06-25/28):**
- Basin-timeseries pilot: 6/6 cases OK (VQC-001, -004, -007, -009, -012, -020).
  Time-series rendering, gap labeling, VQC-001 boundary clip, and qobs alignment all pass.
  h2o output: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_pilot_20260625T123337Z`
- Spatial MRMS smoke (VQC-012, VQC-009): basin=Y, gauge=Y. Raster placement consistent
  with observed qobs responses. No extraction or alignment failures detected.
  h2o output (VQC-012): `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142012Z`
- This is a technical/rendering PASS and scientific QC evidence improvement.
  It is **not a final full forcing certification** — 15 of 21 cases not yet animated.
- Generated outputs (PNG/GIF/CSV/summary) remain under `tmp/` and are not committed.
- See `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md` for full evidence.

---

Stage 1 full 2,843-basin USGS IV target acquisition structurally complete (2026-06-13).
Target policy configured (`config/stage1_target_policy.yaml`, 2026-06-15).
h2o preprocessing environment installed and smoke-tested (`flashnh-stage1`, 2026-06-15).
Target package builder + auditor implemented, smoke-tested, and h2o policy-smoke PASS (2026-06-15).
**v001 target package (2,752 basins) built and audited on h2o (2026-06-16): PASS — 0 errors, 0 warnings.**
**Milestone 2K-A COMPLETE (2026-06-18): v001 basin-weight tables built on h2o — 2,752/2,752 basins, PASS.**
**Milestone 2K-B COMPLETE (2026-06-18): forcing extraction smoke test — PASS. RTMA 48/48 h; MRMS 27/48 h (21 `not_in_s3`, expected early archive gap).**
**Milestone 2K-C COMPLETE (2026-06-18): October 2020 one-month run — PASS.
432h, 2,752 basins, 396/432 MRMS, 432/432 RTMA, 14,167,296 rows, 15h 05m wall.
Full-period extraction PAUSED — 66.5-day projected wall time requires 2K-D optimization.**
**Milestone 2K-D COMPLETE (2026-06-20): D1 serial optimization → 24.7× speedup
(91.9 s → 2.17 s/hr, commit `3ff4965`). Outer-parallelism x3×dw6 → 3.04 days projected
(commit `a275296`). D2 deferred. x4 not recommended.
Decision: full-period launch — 3 concurrent chunks × 6 download workers.**
**Milestone 2K-E pre-launch patch COMPLETE (2026-06-20): `GROUP_ID=A/B/C` and `DRY_RUN=1`
added to fullperiod launcher; path safety guard and per-group logs; reporter updated.
Dry-run validation pending on h2o. Full-period extraction NOT yet launched.**

See `docs/stage1_hpc_transition_preflight.md` for the full audit summary and
`docs/stage1_target_policy.md` for target-policy rationale.

### Quick summary

- 2,843 canonical hourly NetCDF files on h2o at `/data42/omrip/Flash-NH/tmp/stage1_full_2843/`
- Coverage 0.9652 overall; 2,754 basins with `historical_training_utility_flag=True`
- 89 basins with late-period gaps (`TARGET_OPERATIONAL_REVIEW`) — hold out of first package
- 18 basins with negative qobs in the acquisition audit — set to NaN during package build
  (2 heavily-negative special-review basins excluded from v001; 16 basins cleaned, 235 values neg→NaN)
- `TARGET_QUALITY_REVIEW` (1,375 basins): eligible for training; spike flag is advisory only
- No systematic offset issues (0 basins)

### h2o / Moriah operating plan (as of 2026-06-15)

Key policy clarifications from PI:
- h2o is **storage, downloads, preprocessing, and assembly** — not training
- h2o has **no usable GPU** (`nvidia-smi` not found; PI confirms)
- No scheduler by design; `screen` is the agreed background job manager
- CPU compute allowed with etiquette: ≤50–60% CPU; start 16–32 workers; notify before long jobs
- `/data42/omrip` is not auto-deleted; `/data42` is not backed up
- `/data42/hydrolab/Data/Flash-NH_data/` subfolders allowed with reproducibility provenance
- **NeuralHydrology training → Moriah cluster** (`/sci/labs/efratmorin/omripo/PhD`)

See `docs/stage1_h2o_operations_preflight.md` for full gate status.

### h2o environment status (as of 2026-06-18)

- **Prefix:** `/data42/omrip/Flash-NH/envs/flashnh-stage1`
- **Python:** `3.11.15` | **Size:** `7.0 G`
- **Smoke test:** ALL PASS — core, geospatial, dask, cfgrib/eccodes, NetCDF, Parquet, neuralhydrology
- **Log:** `/data42/omrip/Flash-NH/tmp/env_smoke_20260615T120918Z/env_smoke.log`
- **Activation on h2o:** `source /opt/conda/etc/profile.d/conda.sh && conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1`
- **Activation caveat:** The shell prompt may show `(flashnh-stage1)` while `which python` still
  points to `/opt/conda/envs/iacpy3_2025/bin/python`. Always run the explicit `source` + `conda activate`
  sequence and verify with `which python` before running any job. Observed during 2K-A (2026-06-18);
  clean reactivation resolved it.
- **py7zr added (2026-06-18):** Installed `py7zr` into `flashnh-stage1` using the standard h2o workaround:
  `CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs conda install --solver classic py7zr`.
- **Caveat:** `neuralhydrology` pip-pulled CUDA torch (2.12.0+cu130); env is 7.0 G vs lean CPU intent.
  `cuda_available=False` on h2o — functionally harmless. Future spec revision to use `--no-deps` or CPU torch.
- **h2o is not for NeuralHydrology training.** Training remains designated for Moriah cluster.

See `docs/stage1_environment.md` for full install notes, workaround, and CUDA caveat details.

### Target package builder status (as of 2026-06-16)

Milestone 2J-B: **COMPLETE** — scripts implemented, smoke-tested locally and on h2o, full v001 build PASS.

- **Builder:** `scripts/build_stage1_target_package.py`
- **Auditor:** `scripts/audit_stage1_target_package.py`
- **Launcher:** `scripts/run_stage1_target_package_v001_h2o.sh` (commit `3ac51ff`)
- **Doc:** `docs/stage1_target_package_builder.md`
- **Local smoke result (2026-06-15):** 5/5 PASS — 0 errors, 0 warnings
- **h2o policy smoke (2026-06-15):** PASS — 4 basins, 01135300 excluded (hist_util=False),
  08010000 cleaned 95 neg→NaN; audit 0 errors/0 warnings; 02299472 halt confirmed (EXIT 1)
  - `canonical_merged` confirmed: 2,843 flat NCs, 2,843 unique STAIDs, 0 recursive duplicates
- **Full h2o v001 build (2026-06-16): PASS — 0 errors, 0 warnings**
  - Input: 2,843 NCs from `canonical_merged`
  - Excluded: 2 (`--exclude-staids`) + 89 (policy: `hist_util=False`) = 91 total excluded
  - Built: **2,752 basins**, 0 failed
  - Cleaned: 235 neg→NaN across 16 basins; NaN 3,880,507 → 3,880,742; valid hours 121,940,698
  - Audit: 2,752/2,752 checksums OK; 89 held-out absent; SR basins absent; 1,373 TQR advisory
  - Audit runtime: 18.8 s
  - policy_sha256: `449165686d033b9cdbd395ad70e64a3bfa82d01757021e62059f254a2a30d691`
  - Evidence bundle: `tmp/stage1_target_package_v001_evidence/` (not committed)
  - Full result: `docs/stage1_target_package_v001_result.md`
- **Special-review 02299472/04073468:** excluded from v001; disposition open for future v002

See `docs/stage1_target_package_builder.md` for full commands and acceptance criteria.

### Stage 1 forcing — Milestone 2K-A (completed 2026-06-18)

Input preflight and v001 basin-weight table build on h2o. **PASS — 2,752/2,752 basins.**

**Input preflight (`verify_stage1_forcing_inputs_h2o.sh`):** 10/10 PASS, 0 WARN, 0 FAIL.

**Key input locations on h2o:**

| Item | Path | Notes |
|---|---|---|
| v001 basin list CSV | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv` | 2,752 rows excl. header |
| CAMELSH shapefile | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp` | 2,752 polygons; no `.prj`, EPSG:4326 |
| MRMS grid def | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions/mrms_grid_definition.json` | v001 flat layout (not pilot path) |
| RTMA grid def | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions/rtma_grid_definition.json` | same |

**Weight Parquets (output):**

| File | Size | Basins |
|---|---|---|
| `02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet` | 37 MB | 2,752/2,752 |
| `02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet` | 12 MB | 2,752/2,752 |

All paths relative to `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/`.

**Clean build command:**

```bash
python scripts/build_stage1_basin_weights.py \
    --config configs/stage1_forcing_fullperiod.yaml \
    --data-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \
    --basin-list /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv \
    --out-tag v001_2752 \
    --grid-def-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions \
    --skip-qc-plots
```

Fatal validation: all PASS. `--skip-qc-plots` used because the h2o CAMELSH shapefile lacks
`LNG_GAGE`, `LAT_GAGE`, `DRAIN_SQKM` columns (schema: `LAYER, MAP_NAME, AREA, PERIMETER, GAGE_ID, geometry`).
QC plotting is advisory; the fix is in commit `026c363`.

**Operational lessons from 2K-A:**

- **Activation caveat:** Shell prompt can show `(flashnh-stage1)` while `which python` points to
  the wrong env. Always verify with `which python` after activation.
- **py7zr:** Added to `flashnh-stage1` on h2o using `CONDA_PKGS_DIRS` + `--solver classic` workaround.
- **PS1 helper broken:** `scripts/prepare_stage1_forcing_inputs_h2o.ps1` fails to parse on
  Windows PowerShell 5.1 (8 AST parse errors). It is not needed for 2K-B (grid JSONs and
  CAMELSH shapefile are already in place). Fix in a separate commit before relying on it again.
- **Stale verifier message:** `verify_stage1_forcing_inputs_h2o.sh` still prints
  "Ready to proceed to Milestone 2K-A" even after weights are built. Minor stale message;
  not a blocker. Clean up in a later small commit.
- **Grid-def path:** `build_stage1_basin_weights.py` now supports `--grid-def-dir` with 3-level
  auto-discovery (explicit → v001 flat → pilot legacy). Pass it explicitly to avoid ambiguity.

### Stage 1 forcing — Milestone 2K-B (completed 2026-06-18)

Forcing extraction smoke test on h2o. **PASS — all 12 validation checks passed.**

**Evidence:** Compact evidence bundle inspected locally from `tmp/stage1_forcing_smoke_evidence/`
(not committed). Evidence files: `smoke_manifest.json`, `smoke_summary.md`,
`smoke_live_run.log`, `smoke_hourly_runtime_and_volume.csv`, `smoke_missing_files.csv`.

**Original smoke was run via direct extractor invocation.** The launcher (`scripts/run_stage1_forcing_smoke_h2o.sh`)
raised `CondaError: Run 'conda init' before 'conda activate'` when invoked as `bash script.sh`
in a non-interactive shell, even after the PATH-prepend patch in `43af035`. The launcher
activation block was subsequently patched (commit `ccb2631`) to source `conda.sh` unconditionally
and make `conda activate` non-fatal.

**Launcher activation fix verified on h2o (2026-06-18):** After pulling `ccb2631`,
`bash scripts/run_stage1_forcing_smoke_h2o.sh` completed end-to-end via the launcher
wrapper. Python resolved correctly to `/data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python (Python 3.11.15)`.
This was a cached/resume rerun (0.0 B downloaded, ~1m 12s elapsed); output row counts and
PASS status matched the original uncached run. **The launcher activation bug is resolved.**
Download and runtime estimates for 2K-C should be taken from the original uncached run (10m 13s, ~3.2 GB), not this verification rerun.

**Smoke results:**

| Metric | Value |
|---|---|
| Period | 2020-10-14T00:00:00Z – 2020-10-15T23:00:00Z |
| Basins | 10 |
| MRMS hours extracted | 27/48 |
| MRMS missing | 21 (`not_in_s3`, 2020-10-14T00Z–20Z — see note below) |
| RTMA hours extracted | 48/48 |
| RTMA missing | 0 |
| `mrms_smoke.parquet` rows | 270 (27 h × 10 basins) |
| `rtma_smoke.parquet` rows | 5,280 (48 h × 10 basins × 11 vars) |
| `combined_smoke.parquet` rows | 5,550 |
| Wall clock | 10m 13s |
| Downloaded | ~3.2 GB (RTMA `selected_messages`, 4 workers) |
| `all_pass` (manifest) | `true` |
| Git commit at run time | `43af035d` |

**MRMS 21-hour early archive gap (expected):** `noaa-mrms-pds` QPE 1h Pass1 coverage for
2020-10-14 begins at 21:00Z, not midnight. The first 21 hours (00Z–20Z) are genuinely
absent from S3 — this is a permanent upstream archive gap, not a pipeline error.
The full-period first chunk (`2020-10`) will carry the same 21-hour gap in its
`missing_files.csv`. All subsequent months have complete MRMS coverage.

**Validation checks (all PASS):**
`mrms_extracted_hours_gt_zero` · `mrms_N_basins_per_ok_hour` · `mrms_no_all_null_weighted_mean`
· `mrms_valid_weight_fraction_ok` · `mrms_parquet_written` · `rtma_extracted_hours_gt_zero`
· `rtma_10wdir_absent` · `rtma_orog_absent` · `rtma_at_least_8_variables`
· `rtma_no_all_null_weighted_mean` · `rtma_parquet_written` · `combined_parquet_written`

**Performance notes:**
- RTMA `selected_messages` download: median ~42 s/file at 4 workers → ~33–40 h total at 16 workers.
- MRMS download: ~0.3–1.3 s/file (cfgrib cold start on first file only). Negligible vs RTMA.
- Estimated full-period RTMA raw: ~3.2 TB (`selected_messages`); MRMS raw: ~0.5 TB.

### Stage 1 forcing — Milestone 2K-C (completed 2026-06-18)

October 2020 one-month forcing extraction on h2o. **PASS — all 12 extractor validation checks passed.**

**Evidence:** Compact bundle in `tmp/stage1_evidence_exports/2020-10/` (not committed).

| Metric | Value |
|---|---|
| Period | 2020-10-14T00Z – 2020-10-31T23Z |
| Scheduled hours | 432 |
| Basins | 2,752 |
| MRMS extracted | 396/432 |
| MRMS not_in_s3 | 36 (3 clusters — see below) |
| RTMA extracted | 432/432 |
| RTMA variables | 11 (incl. diagnostic `ceil`, `vis`; `10wdir`/`orog` absent, confirmed) |
| Combined rows | 14,167,296 (1,089,792 MRMS + 13,077,504 RTMA) |
| MRMS raw | 207 MB |
| RTMA raw | 30.7 GB |
| Wall clock | 15h 04m 57s (`download_workers=8`) |
| `all_pass` | `true` |
| Git commit at run | `194a489` |

**MRMS 36-hour gap (permanent S3 gaps — not pipeline errors):**

| Cluster | Hours | Timestamps |
|---|---|---|
| Archive-start | 21 h | 2020-10-14T00Z–20Z |
| Oct 25–26 outage | 14 h | 2020-10-25T23Z; 2020-10-26T00Z–11Z, 15Z |
| Oct 29 spot | 1 h | 2020-10-29T23Z |

**Throughput and full-period projection:**

- Actual throughput: 125.7 s/hr (serial, extraction-dominated)
- Full-period projection at current serial code: **66.5 days** (45,720 h × 125.7 s / 86400)
- Primary bottleneck: `extract_basin_statistics` in `src/pipeline/extraction.py:396`
  — `weights_df.loc[weights_df["STAID"] == staid]` O(N) scan, 30,272 calls per RTMA hour
- The 20.2-day figure from `scaling_estimates.json` was computed from RTMA download time only
  (download is pipelined/prefetched and is NOT on the serial critical path)

**Full-period extraction was PAUSED at 2K-C completion.** 2K-D is now COMPLETE — see section below.

### Stage 1 forcing — Milestone 2K-D (completed 2026-06-20)

Extraction optimization and outer-parallelism throughput benchmark.
**COMPLETE — effective full-period projection 3.04 days (3 concurrent chunks × 6 download workers).**

#### D1: Serial extraction optimization (commit `3ff4965`)

Two targeted changes to `src/pipeline/extraction.py` and `scripts/extract_stage1_forcing_chunk.py`:

1. **Pre-grouped weight lookup** — `_build_basin_cells()` pre-groups the weight DataFrame by
   STAID into a `{STAID: (row_idx, col_idx, norm_w)}` dict at startup. Each per-basin-hour call
   becomes an O(1) dict lookup instead of an O(N) boolean scan over the 2,752-row weight table
   (90,816 scans/RTMA-hour eliminated).
2. **Batched percentile computation** — 7 sequential `np.percentile` calls replaced with one
   batched call, eliminating 635,712 redundant sort passes per RTMA-hour.

**Measured result:** `extraction_median_s` 91.976 s → 2.17 s/hr (**24.7× speedup**).
Bottleneck fully shifted from extraction CPU to S3 download. D2 process-workers not needed.

#### Download-worker sensitivity benchmark (48h RTMA-only, 2,752 basins)

Commit `3ff4965`; RTMA `selected_messages`; Oct 2020 period; all runs `all_pass=True`.

| Workers | Wall (s) | Proj. days | dl_median (s) | ext_median (s) |
|---|---|---|---|---|
| 2  | 1157.7 | 12.76 | 31.3 | 2.21 |
| 4  | 804.8  | 8.87  | 31.3 | 2.19 |
| 8  | 642.9  | 7.09  | 35.9 | 2.18 |
| 16 | 570.5  | **6.29** | 44.9 | 2.17 |

Individual download time increases with worker count (S3 bandwidth sharing) but wall-clock improves
via prefetch concurrency. dw16 projects 6.29 days. D2 process-workers deferred; outer parallelism
is the lever for sub-4-day throughput.

#### Outer-parallelism benchmarks (RTMA-only, 48h per chunk, 2,752 basins)

All chunks `all_pass=True`, `successful_hours=48/48`, `actual_rows=1,453,056`.

**x2 — 2 chunks × dw8 (16 total S3 connections):**
Commits `cf8db74`; evidence `tmp/stage1_2kd_evidence/outer_parallel_rtma_48h_dw8_x2/`.

| Chunk | Chunk wall (s) | dl_median (s) | ext_median (s) |
|---|---|---|---|
| outer-x2-a | 735.4 | 47.2 | 2.195 |
| outer-x2-b | 720.0 | 43.1 | 2.291 |
| **Parent wall** | **736 s** | | |

Projection: 45720 × 736 / (2 × 48) / 86400 = **4.057 days — YELLOW (partial scaling).**

**x3 — 3 chunks × dw6 (18 total S3 connections):**
Commit `a275296`; evidence `tmp/stage1_2kd_evidence/outer_parallel_rtma_48h_dw6_x3/`.

| Chunk | Chunk wall (s) | dl_median (s) | ext_median (s) |
|---|---|---|---|
| outer-x3-a | 825.9 | 45.9 | 2.233 |
| outer-x3-b | 801.1 | 43.9 | 2.206 |
| outer-x3-c | 801.2 | 42.5 | 2.204 |
| **Parent wall** | **826 s** | | |

Projection: 45720 × 826 / (3 × 48) / 86400 = **3.035 days — USEFUL GREEN.**

#### Decisions (all binding)

- **Stop performance optimization.** 3.04 days projected is within the acceptable range.
- **D2 process-workers: deferred indefinitely.** Extraction is 2.17 s/hr; download (43–46 s/file)
  dominates. Process parallelism within a single chunk would not improve end-to-end throughput.
- **x4 outer-parallelism: not recommended.** x3 achieves 3.04 days; x4 would push total S3
  concurrency to 24 workers, increasing contention and operational risk for marginal gain.
  RTMA-only benchmark may understate MRMS+RTMA mixed-product overhead.
- **Full-period launch recommendation:** 3 concurrent chunk processes × 6 download workers each.
  All outputs under `/data42/omrip/Flash-NH/`. Mechanism: 3 independent screen sessions covering
  non-overlapping month groups (~21 months each), or a new parallel launcher.
  See updated `docs/stage1_forcing_fullperiod_launch_plan.md` for Phase 2 outer-parallel details.

### Immediate next steps

The v001 target package is **streamflow-only**. Full NeuralHydrology training requires
forcing data and package assembly on h2o before any Moriah transfer.

1. ~~**Push 2K-E pre-launch patch and pull on h2o**~~ — **COMPLETE (2026-06-20).**
2. ~~**Stage 1 forcing acquisition plan + weight build (2K-A)**~~ — **COMPLETE (2026-06-18).**
3. ~~**Milestone 2K-B — forcing extraction smoke test**~~ — **COMPLETE (2026-06-18): PASS.**
4. ~~**Milestone 2K-C — October 2020 one-month run**~~ — **COMPLETE (2026-06-18): PASS.**
4b. ~~**Milestone 2K-D — extraction optimization + h2o CPU-parallel benchmark**~~ — **COMPLETE (2026-06-20): PASS.**
4c. ~~**Milestone 2K-E — full-period forcing extraction**~~ — **COMPLETE and AUDITED (2026-06-24): PASS_WITH_CAVEATS.**
    63/63 months, 1.51B rows, 0 failures. See `docs/stage1_forcing_fullperiod_audit.md`.
5. ~~**Visual / event QC case selection + pilot animation + spatial MRMS QC**~~ — **PILOT VISUAL QC PASS (2026-06-25/28).**
   21 cases generated (seed=42). Basin-timeseries pilot 6/6 OK. Spatial MRMS smoke VQC-009/VQC-012 PASS (basin=Y, gauge=Y).
   Case selection: `docs/stage1_forcing_fullperiod_visual_qc_selection.md`.
   Animation plan and evidence: `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`.
   Outputs under `tmp/` (not committed). 15 remaining cases not yet animated — not a final certification.
6. ~~**Curated forcing product v001 design (Milestone 2K-F-A)**~~ — **COMPLETE (2026-06-29).**
   Product contract frozen: wide-format per-basin Parquet, gap-flag columns, manifest, provenance.
   Design doc: `docs/stage1_curated_forcing_product_v001_design.md`.
7. ~~**Curated forcing product v001 — builder + smoke test (Milestone 2K-F-B)**~~ — **COMPLETE (2026-06-29): PASS.**
   5/5 basins, 720 h, 0 MRMS gaps, 10 RTMA gap-hours (coverage 0.9972). Scripts: commit `6f4de49`.
   h2o output: `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z/`.
8. **Curated forcing product v001 — corrected schema build (Milestone 2K-F-C)** — schema
   corrected in 2K-F-C-B (2026-06-30): dewpoint mapping fixed, `rtma_weasd_kgm2` removed.
   Next: corrected 5-basin full-period pilot on h2o (`--max-basins 5 --overwrite`), then
   full 2,752-basin rebuild authorization. Full rebuild NOT yet authorized.
9. **Milestone 2K-G-B — NH pilot package builder** — implement `scripts/build_stage1_nh_package.py`
   on h2o: merge corrected forcing Parquets + target NCs into 5-basin GenericDataset NCs,
   apply gap-fill policy (MRMS→0.0, RTMA→interp), write `attributes.csv` and basin lists.
   Transfer pilot package (~25 MB) to Moriah.
9a. **Milestone 2K-G-C — Moriah environment + Smoke 0** — install `flashnh-moriah` conda env
    (PyTorch+CUDA, NH), run Smoke 0 Slurm job (5 basins, 2 epochs, mrms_qpe_1h_mm only),
    confirm finite loss and checkpoint.
9b. **Milestone 2K-G-D — Smoke 1** — add RTMA meteorology, confirm `rtma_2d_K` non-null.
    Preflight design: `docs/stage1_neuralhydrology_preflight.md`.
10. **Moriah transfer layout and checksum-verified transfer** — define directory structure
    and `rsync`/`scp` transfer procedure; verify checksums on arrival before training.
11. **Moriah training environment and config** — only after the assembled package passes
    audit on Moriah. NeuralHydrology training remains designated for Moriah cluster.

#### 2K-C pre-launch checklist and caution

Before any 2K-C run, confirm all of the following:

**Launcher verification (new requirement):**
- Pull latest commits on h2o: `git pull --ff-only`
- Run a dry activation test: `bash scripts/run_stage1_forcing_smoke_h2o.sh --help` or check that
  the launcher reaches the Python version line without error.
- The launcher activation bug (CondaError in non-interactive shells) is patched in the current commit.
  **Verify the fix is working on h2o before launching 2K-C.**

**One-month dry run before full 63-month launch:**
- Run 2020-10 alone first (`screen -S flashnh-2020-10 bash scripts/run_stage1_forcing_fullperiod_h2o.sh`
  with the month list reduced to a single entry, or via direct extractor for 2020-10-14T21Z – 2020-10-31T23Z).
- Confirm the 2020-10 chunk manifest is written, `missing_files.csv` contains exactly 21 MRMS
  `not_in_s3` entries for 2020-10-14T00Z–20Z, and Parquet row counts are consistent.
- Pull the 2020-10 evidence bundle locally before enabling the full loop.

**Expected 2020-10 MRMS 21-hour gap:**
- 2020-10-14T00Z–20Z will appear as `not_in_s3` in `missing_files.csv` for the first chunk.
- This is a documented upstream archive gap, not a pipeline error. Do not treat as a blocker.
- All hours from 2020-10-14T21Z onward and all subsequent months have complete MRMS coverage.

**PI notification:**
- Notify PI/machine owner before starting the full 63-month extraction loop.
- Check `uptime` before launch; hold if 1-min load > 0.7 × nproc.
- Target ≤ 50–60% CPU; start with 16 workers; increase only after monitoring a full chunk.

**Storage and raw GRIB2 deletion policy:**
- Raw MRMS + RTMA GRIB2 cache accumulates to ~3.7 TB over the full period.
- After each quarter's monthly chunk Parquets are written and checksummed, delete the
  corresponding raw GRIB2 cache to free space. Do not delete until Parquets are verified.
- Monthly chunk Parquets + per-basin forcing NCs are the curated products; raw GRIB2 is reproducible.
- Do not exceed ~20 TB total across all Flash-NH data on `/data42`.

**Evidence-bundle pull policy:**
- After every quarter (roughly every 3 months of chunks), transfer compact evidence bundles
  locally: chunk manifests (`*_manifest.json`) and missing-file CSVs (`*_missing_files.csv`).
- Do not transfer raw GRIB2, staging Parquets, or combined chunk Parquets unless needed for debugging.
- Document each quarterly bundle in `docs/FLASHNH_CURRENT_STATE.md` before proceeding.

**Progress monitoring:**
- Attach to the screen session with `screen -r flashnh-fullperiod` to check live log output.
- Each monthly chunk writes a progress log to `{FORCING_ROOT}/manifests/{chunk_label}_live_run.log`.
- Check `uptime` and `df -h /data42` periodically (once per few hours).
- A per-month completion summary will be logged; each month's manifest is the checkpoint.

**Stop and resume procedure:**
- To stop cleanly: `Ctrl-C` inside the screen session; the current hour's staging Parquet may be incomplete.
- To resume: re-run the launcher with `--resume`; already-written staging Parquets for completed hours
  are skipped automatically.
- Each completed monthly chunk is independent; re-running a month re-uses cached raw files and
  skips already-extracted hours.

**Special-review disposition (02299472/04073468)** — open for future v002, not a blocker
for steps 3–8 above. 02299472: 2,605 neg; 04073468: 2,054 neg.

The following require additional confirmation before proceeding:

- Promotion of curated data to shared lab storage — gate G4 CONDITIONALLY UNBLOCKED
  (confirm write access to `/data42/hydrolab/Data/Flash-NH_data/` before first promotion).
- NeuralHydrology training — gate G3 NOT PLANNED ON h2o; blocked on Moriah scheduler
  confirmation and env setup.

**Do not run TB-scale spatial downloads without smoke-test sign-off under etiquette rules.**

---

## Milestone 2G — NeuralHydrology NetCDF builder + preflight auditor (completed 2026-06-09)

NeuralHydrology-compatible January 2023 pilot package built and audited.
Full documentation: `docs/stage1_neuralhydrology_preflight.md`

**Scripts:**
- `scripts/build_stage1_neuralhydrology_january_pilot.py` — builder (~8s)
- `scripts/audit_stage1_neuralhydrology_january_pilot.py` — auditor (~20s)

**Package:** `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package/` (gitignored)

**Audit result:** PASS — 0 errors, 1 warning

**Package summary:**
- 50 per-basin NetCDF files; `date` coordinate; 744 hourly UTC steps; January 2023
- 11 variables per basin (10 dynamic forcings + `qobs_m3s` target)
- Smoke dynamic inputs: `mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`
- `attributes_full.csv`: 50 rows × 238 cols (237 attribute cols + `gauge_id`)
  - Manifest records 237, counting only attribute cols; both are correct
- Full HydroATLAS integration: 50/50 pilot match; 193 new columns
- Streamflow: 20 full, 8 partial, 22 all-NaN (CAMELSH files missing locally)
- Audit warning (expected, S2): nulls in `max_abs_hourly_jump_over_Q50` (1), `q95_q50_ratio` (1), `wet_cl_smj` (14) — NaN preserved, no imputation

**No model training run. No generated files committed.**

---

## Milestone 2F — NeuralHydrology package design (completed 2026-06-08)

Design and decision documentation for the NeuralHydrology package.
Full documentation: `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/`

Key decisions: V1 (both rtma_2d_K and rtma_2sh_kgkg in smoke), V2 (rtma_sp_Pa in wide only),
V3 (rtma_tcc_pct in wide only), S1 (22 missing CAMELSH → all-NaN qobs_m3s, 2H blocker),
S2 (preserve NaN, no imputation), S3 (full HydroATLAS 50/50), S4 (seed=42, streamflow-only split).

---

## Milestone 2E — Event animation pipeline (completed 2026-06-07)

Pilot animations (R02, R06, R09, R11) generated and approved in v2.1-stable design.
Pipeline cleanup completed.

**Stable scripts:**
- `scripts/generate_january_event_animations.py` — main animation generator
- `scripts/audit_rtma_spatial_alignment.py` — RTMA spatial audit gate
- `scripts/audit_january_event_animation_sync.py` — MRMS sync audit gate

**Audits confirmed:**
- RTMA spatial audit: 8/8 PASS, 0.0000% diff (2t, 10u, 10v)
- MRMS sync audit: 10/10 PASS, 0.0000% diff

**Key v2.1 design notes:**
- MRMS lat DECREASES with row (row 0 = 54.995 N)
- RTMA lat INCREASES with row (row 0 = 19.229 N)
- RTMA 10m wind quiver is qualitative context only — not storm-steering validation

**All-12 command** (not yet executed; run after explicit approval):
```bash
python scripts/audit_rtma_spatial_alignment.py
python scripts/audit_january_event_animation_sync.py
python scripts/generate_january_event_animations.py --all
```
Output: `tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/`
Estimated runtime: ~27 min (local, GIF mode).

---

## RTMA/URMA-family precipitation diagnostic (completed 2026-06-08)

Diagnostic-only follow-up to Milestone 2E. Confirmed RTMA/URMA grid, weight,
and timestamp consistency against MRMS. **Did not modify Stage 1 model inputs.**

Full documentation: `docs/stage1_rtma_urma_mrms_diagnostic.md`

**Key findings:**

- Regular RTMA Stage 1 files have no precipitation field.
- URMA QPE `pcp_01h.wexp.grb2` contains `tp` (Total Precipitation, kg m**-2 = mm).
- URMA and RTMA share the same 1597 x 2345 LCC 2.5 km CONUS grid exactly.
- Existing `pilot_rtma_weights.parquet` reused without modification. No new weights.
- Timestamp convention A confirmed (filename HH = end of accumulation):
  r = 0.961 on R02; shifted alternatives much worse; peak at Jan 29 08Z for both.

**Pilot metrics (Convention A):**

| Candidate | r | RMSE (mm) | Note |
|---|---|---|---|
| R02 (AR, STRONG_WET) | 0.963 | 1.12 | URMA smooths peak vs MRMS |
| R06 (MN, MOD_COLD) | 0.913 | 0.70 | URMA higher; snow/mixed-precip context |
| R11 (MA, OFFSET) | 0.944 | 0.39 | Strong agreement |

**Scripts (committed):**
- `scripts/discover_rtma_urma_precip_january2023.py`
- `scripts/urma_mrms_timestamp_and_pilot.py`

**Diagnostic outputs (untracked):**
`tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/`

---

## Completed extraction state

January 2023 pilot extraction for 50 basins:
- MRMS: 744/744 hours, 37,200 rows
- RTMA: 744/744 hours, 409,200 rows
- Combined: 446,400 rows
- valid_weight_fraction = 1.0

Streamflow: CAMELSH hourly NetCDF, 28/50 pilot basins have January 2023 data.

Refined event candidates: R01–R12 (R03 usable-with-gap).
Pilot animations: R02, R06, R09, R11 — reviewed and approved.

---

## Standing cautions

- Do not generate all 12 animations until explicitly instructed.
- Do not start model training yet.
- Do not commit generated MP4/GIF/PNG/Parquet/GRIB/NetCDF/log outputs.
- Keep local-to-HPC transition in mind.
- RTMA 10m wind vectors are qualitative context only — not storm-steering validation.
- URMA precipitation is diagnostic-only — do not add to Stage 1 model inputs.

---

## Historical note: Milestone 2H — Streamflow recovery for 22 missing CAMELSH basins

> This section is superseded for full-period target-package construction, which is now
> complete (v001, 2026-06-16). The recovery work below applied to the January 2023 pilot
> package (Milestone 2G) and is retained for reference. The current top-level next step
> is Moriah transfer layout design (see Immediate next steps above).

Recovery was needed because the January 2023 pilot package built from CAMELSH files
had 22 basins with all-NaN `qobs_m3s`. Those basins were recovered from USGS IV
(Milestones 2H–2H-D) and are fully represented in the full-period v001 package.

Recovery plan (historical): `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/streamflow_recovery_plan.md`

**Original pending tasks (now completed or superseded):**

1. Milestone 2H: CAMELSH streamflow recovery for 22 missing basins.
2. Decide on all-12 animation run (2E follow-up).
3. Event QC conclusions: finalize which of R01–R12 are included in Stage 1 training.
4. HPC transfer planning.
5. Stage 1 model configuration and first training run.
