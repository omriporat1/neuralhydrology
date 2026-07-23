# Stage 1 scientific baseline — design gate

Milestone: 2K-G-D (Attribute Provenance + Modeling Design Gate), opened 2026-07-03.
Milestone 2K-G-E (Scientific Baseline Design Resolution): first proposed 2026-07-03,
**revised 2026-07-06** after user review changed several key decisions (basin
splits, target scaling, lead time, static attributes, hyperparameter framing). This
revision replaces the 2026-07-03 draft in place — that draft was never committed
(see `docs/decision_log.md` for what changed and why).
**Milestone 2K-G-H (Scientific Baseline Policy Sign-off), 2026-07-12:** converts
the 2K-G-F/2K-G-G evidence into binding decisions, closing every item this
document previously left `STILL OPEN` or `GATED` — see "Binding decisions —
Milestone 2K-G-H sign-off" below and the revised "Status" section at the end.

This document now distinguishes three kinds of items:
- **Approved** — a binding decision, recorded below, not open for re-litigation
  absent a new explicit decision.
- **Still open, needs sign-off** — a scientific/methodological choice with a
  recommended default, but not yet decided.
- **Gated on a mini-milestone** — cannot be responsibly decided from documentation
  or assumptions alone; requires inspecting the actual NeuralHydrology 1.13 code
  installed on Moriah (§"New mini-milestones" below), or recovering source data
  that does not yet live at its canonical path.

## What this document is

A **decision scaffold**, not a locked model spec. Smoke 0 and Smoke 1 (both
PASS on Moriah, 2026-07-02) proved the technical pipeline — package format,
NH 1.13 compatibility, MRMS + core RTMA ingestion — works end to end at
`seq_length=24`, 5 basins, a handful of epochs. None of that constitutes a
scientific baseline. This doc lists what must be explicitly decided before
the first full scientific-baseline training run, so those decisions are made
once, deliberately, and recorded — rather than inherited by default from
whatever Smoke 0/1 happened to use.

**`seq_length` is one hyperparameter among many decided here — it is not the
milestone driver.** It is now a **binding, narrow** choice for Stage 1: only
12/24/48/72 hours are Stage-1-eligible (§9). Longer lookback (168/336 h and
beyond) is explicitly **Stage 2 / long-term antecedent-modeling territory, not
Stage 1** — do not reintroduce it here in future revisions of this document.

**Stage 1 excludes California entirely** (§8c). California is reserved for a
Stage 4 transfer-learning experiment; Stages 1–3 are non-CA CONUS only.

## Purpose and non-goals

**Purpose:** define the first scientific-baseline NeuralHydrology training
configuration for Flash-NH Stage 1 — a run whose results are meant to be
scientifically interpretable (skill scores worth reporting), not just a
plumbing check.

**Non-goals of this document:**
- Does not generate the full 2,752-basin NH package (separate execution step,
  gated on §2 attribute provenance being closed — see
  `docs/stage1_attribute_provenance.md` — **and** on the new 2K-G-F / 2K-G-G
  gates below).
- Does not run training.
- Does not perform hyperparameter search / sweeps — it defines the sweep
  *policy* (§9) so a future sweep has a documented protocol to follow.
- Does not finalize every value below as immutable — items marked **still
  open** are decisions still needed, via explicit user sign-off; items marked
  **gated** require a technical report before they can even be proposed
  responsibly.
- Does not create the richer static-attribute matrix or the gap-policy /
  target-scaling feasibility report — both are scoped as separate
  mini-milestones (2K-G-F, 2K-G-G) and are **not done in this patch**.

---

## Binding decisions (user-approved, 2026-07-06)

Quick-reference list. Details and rationale are in the numbered sections
below; this list exists so future prompts/docs do not keep re-proposing
values already settled here.

1. **Basin set:** conservative 2,752-basin floor; exclude `02299472` and
   `04073468`. (§4)
2. **Stage 1 dynamic inputs:** `v001-core` — the 8 already-confirmed Smoke 0/1
   variables. The extra 5 RTMA variables are deferred to a later
   `v001-fullmet` ablation, **not** a default extension after a smoke test.
   (§1)
3. **`seq_length`:** Stage 1 candidates are **only** 12 / 24 / 48 / 72 hours.
   Default smoke/preflight value stays 24. 168/336 h are **not** Stage 1
   candidates — they belong to Stage 2. (§9)
4. **Lead time:** primary benchmark lead time **6 h**; secondary **12 h**; 1 h
   and 3 h are diagnostic/sanity-check only, not the primary benchmark.
   Input sequence length and prediction lead time are **separate design
   axes** — do not conflate them. (§9)
5. **Temporal split:** train `2020-10-14`→`2023-12-31`, validation
   `2024-01-01`→`2024-12-31`, test `2025-01-01`→`2025-12-31`. (§8)
6. **Spatial split:** California excluded entirely from Stages 1–3. Within
   non-CA CONUS, ~10% spatial-holdout basins, broadly distributed, test-only
   (never used for training/validation/tuning/normalization/early
   stopping/model selection). Official spatial-holdout evaluation uses the
   2025 test period. (§8b)
7. **California transfer learning (Stage 4):** CA held out through Stages
   1–3; used in Stage 4 for fine-tuning with an internal ~90/10 CA split
   (fine-tune-train / CA holdout). CA-specific normalization may be refit
   using only the CA fine-tuning training subset. (§8c)
8. **Leakage prevention:** scaling/normalization statistics for Stages 1–3
   are fit only on the development training basins and training period —
   never on validation, temporal test, spatial holdout, or CA data. (§8d)
9. **Target scaling:** log-transform is **rejected** as the default (poorly
   aligned with flash-flood/high-flow emphasis). Leading candidate is
   area-normalized / specific discharge, pending NH/package feasibility —
   gated on 2K-G-G. (§5)
10. **Loss vs. metrics:** training loss is separate from evaluation metrics.
    Evaluation is always reported in raw `m^3/s` after inverse conversion;
    raw-space NSE is the primary evaluation metric. (§7)
11. **Hyperparameters:** the conventional table is an **initial seed config
    only**, not the official Stage 1 benchmark. The official benchmark
    requires a controlled W&B hyperparameter sweep (not run yet). (§9)
12. **Sweep objective:** validation raw-space NSE is the primary
    model-selection objective for now; high-flow/event metrics are logged as
    secondary diagnostics; composite objectives are a later discussion. (§9)
13. **W&B logging:** expanded beyond config/provenance to include loss/validation
    curves, learning rate, epoch timing, run duration, hyperparameters, final
    metrics, Slurm job ID/node/partition/GRES/GPU type, git commit, package
    provenance, and system/resource telemetry where available. (§10)
14. **Slurm/resources:** kept flexible and parameterized, not permanently
    hard-pinned to one partition/GPU; actual resources used are recorded in
    the evidence bundle; allocation may be increased later if telemetry shows
    training is resource-limited. (§11)

---

## Binding decisions — Milestone 2K-G-H sign-off (user-approved, 2026-07-12)

**This sign-off converts the 2K-G-G Phase B evidence
(`docs/stage1_target_scaling_gap_leadtime_feasibility.md`, evidence
committed at `0d0e6aa`) into binding Stage 1 decisions.** It closes every
item the 2026-07-06 revision left `STILL OPEN` or `GATED` on 2K-G-F/2K-G-G
(§3, §5, §6), and adds detail the 2026-07-06 revision did not yet specify
(target-inversion audit requirements, static-attribute pathway, spatial
split stratification factors). **No code, config, Slurm script, or NH
package changed in this patch — this is a docs-only policy sign-off.**
Implementation is scoped as a new mini-milestone, 2K-G-I (see "New
mini-milestones" below), not executed here.

1. **Target scaling — APPROVED, supersedes §5's `STILL OPEN` status.**
   Area-normalized discharge, internal unit **mm/h equivalent runoff
   depth**, computed by the Flash-NH package builder at package-build time
   (before NH ever sees the data) — Q1 evidence confirmed this is not a
   `GenericDataset` config flag. Per-basin target column is a
   transformed/shifted variable such as `qobs_mm_per_h_leadXX`, not raw
   `qobs_m3s`. NH's native scaler inversion (Q2 evidence,
   `tester.py:247-259`) returns only to mm/h; official evaluation requires
   an **additional Flash-NH-side conversion** from mm/h back to raw `m^3/s`
   using basin area. **Binding evaluation policy:** NH's own loss/
   validation curves are training diagnostics in transformed (mm/h) space
   unless separately proven equivalent to raw-space metrics; official
   benchmark metrics (§7) are always computed by Flash-NH after full
   inverse conversion to raw `m^3/s`. Detail in §5 (revised below).
2. **Target inversion / audit requirements — NEW, APPROVED.** Three
   required future implementation checklist/audit items, detailed in the
   new §5a below: (a) deterministic unit tests for the `m^3/s -> mm/h ->
   m^3/s` round-trip using basin area; (b) a package-audit requirement that
   `qobs_mm_per_h_leadXX` at timestamp `t` equals original `qobs_m3s` at
   `t+XXh` converted to mm/h, checked on random basin/time samples; (c) an
   evaluation-audit requirement that raw-space metric scripts verify units,
   basin area, target lead alignment, NaN masking, and the conversion back
   to `m^3/s`.
3. **Lead-time implementation — APPROVED, extends §9b.** Package-build-time
   target shifting (Q9 evidence: NH has no native `lead_time` config;
   the native hindcast/forecast architecture requires forecast-known future
   inputs Flash-NH's purely-historical `v001-core` inputs don't have — see
   §9b). **All four lead times — 1 h, 3 h, 6 h, 12 h — are included in the
   first package/config/sweep design**, not just 6 h/12 h: 1 h and 3 h
   remain diagnostics (not the primary benchmark) but are included now
   because their incremental package/config cost is small and deferring
   them would mean reproducing package-build work later. **Primary
   benchmark/model-selection lead: 6 h. Secondary: 12 h.** `seq_length`
   (§9a) and lead time remain separate design axes — do not conflate them.
4. **Forcing-gap policy — APPROVED, supersedes §6's `HIGH PRIORITY, gated`
   status.** Scientific baseline (Policy B from the feasibility doc's
   decision framework) **hard-excludes training windows that intersect
   MRMS archive-gap hours**, at package-build time. Accepted because the
   corrected 2K-G-G real-gap window loss is modest (~1.3% at
   `seq_length=12` to ~5.6% at `seq_length=72` — see the feasibility doc's
   corrected results table) and hard exclusion is scientifically cleaner
   than silent fill. RTMA (2 archive-gap hours total, ~2 orders of
   magnitude smaller than MRMS's 136) is treated separately in wording: if
   the exclusion-mask implementation naturally supports an "either MRMS or
   RTMA gap" mask, excluding RTMA-gap-intersecting windows too is
   acceptable — MRMS drives the scientific policy either way. **Silent
   dynamic-input NaNs are not allowed.** NH's `nan_handling_method` (Q6/Q7
   evidence: default `None` is unsafe, passes raw NaN into an unprotected
   embedding) remains a **fallback/ablation path only, not the baseline**;
   if used in any ablation it must be explicitly configured
   (`masked_mean`, `attention`, or `input_replacing`) — unset/default is
   forbidden for NaN-valued dynamic inputs in any run, baseline or
   ablation. Smoke 0/1's technical fill policy (MRMS gaps → 0.0 mm, RTMA
   gaps → linear interpolation) remains historical/technical only — not
   the scientific baseline. Detail in §6 (revised below).
5. **Static attributes — APPROVED, resolves §3's `REOPENED, gated` status.**
   Canonical `stage1_static_attributes_v001` (2,843 basins × 531 columns,
   496 `model_input`; Moriah/h2o canonical PASS 2026-07-08, sha256
   `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`; see
   `docs/stage1_static_attribute_matrix_plan.md` §11.6 and
   `docs/FLASHNH_CURRENT_STATE.md`'s 2K-G-F-B entry) is accepted as the
   Stage 1 v001-core static attribute matrix — the baseline static-attribute
   matrix for the first scientific baseline. Numeric static attributes pass
   through the model's standard static-attribute pathway/embedding layer
   (assuming NH config support — the same mean/std scaler path confirmed
   for dynamic inputs at `basedataset.py:725-736`, Q3 evidence). **No raw
   categorical embeddings in this first baseline** — categorical/admin/
   geographic leakage-prone fields remain excluded/deferred per the
   2K-G-F/2K-G-F-B conservative column-classification policy. `STATE`/
   `HUC02` remain split-support/diagnostics only, not model inputs.
   `LAT_GAGE`/`LNG_GAGE` remain diagnostic only, deferred to a later
   lat/lon ablation. Detail in §3 (revised below).
6. **Spatial split and leakage — APPROVED, extends §8b/§8d.** Reproducible
   seeded stratified non-CA spatial holdout (mechanism unchanged from
   §8b's ~10% rule). California remains excluded entirely from Stages 1–3,
   reserved for Stage 4 transfer learning (§8c, unchanged). Spatial leakage
   prevention is a Flash-NH basin-list responsibility, not something NH
   enforces (Q4 evidence — NH's `is_train`/scaler contract protects
   temporal leakage automatically but has zero concept of basin role).
   Future implementation must produce **explicit basin-list artifacts**
   for: development training; validation; temporal test; non-CA spatial
   holdout; California Stage 4 fine-tune/holdout split. Stratification
   should consider at minimum HUC02/geography, basin area, and
   hydroclimatic attributes such as aridity/climate, if available from
   `stage1_static_attributes_v001`. Detail in §8b (revised below).
7. **Next implementation milestone — DEFINED, not executed.** `2K-G-I —
   Baseline Package Builder + Split Config Implementation`. Scope is a
   checklist (see "New mini-milestones" below), not code — not authorized
   or started in this patch.

## 1. Candidate dynamic inputs for the first baseline — `v001-core` (APPROVED)

Confirmed working end-to-end in Smoke 1 (`docs/stage1_neuralhydrology_preflight.md`
§2, §7):

| Variable | Units | Status |
|---|---|---|
| `mrms_qpe_1h_mm` | mm | Smoke 0+1 PASS |
| `rtma_2t_K` | K | Smoke 1 PASS |
| `rtma_2d_K` | K | Smoke 1 PASS (dewpoint mapping fix confirmed, 2K-F-C-B) |
| `rtma_2sh_kgkg` | kg/kg | Smoke 1 PASS |
| `rtma_10u_ms` | m/s | Smoke 1 PASS |
| `rtma_10v_ms` | m/s | Smoke 1 PASS |
| `mrms_qpe_1h_mm_gap` | bool→float32 | Smoke 0+1 PASS (gap flag, not a physical forcing) |
| `rtma_gap` | bool→float32 | Smoke 1 PASS (gap flag) |

**APPROVED (2026-07-06):** these 8 variables are `v001-core` — the input set
for the **first Stage 1 scientific benchmark**. No further review needed.

**Deferred, not a default extension:**

| Variable | Units | Note |
|---|---|---|
| `rtma_sp_Pa` | Pa | Surface pressure |
| `rtma_tcc_pct` | % | Total cloud cover |
| `rtma_vis_m` | m | Visibility |
| `rtma_gust_ms` | m/s | Wind gust |
| `rtma_ceil_m` | m | Cloud ceiling height |

These 5 variables define a later **`v001-fullmet`** ablation experiment, run
*after* the `v001-core` benchmark exists, not automatically appended to it
once a smoke test passes. **Correction from the 2026-07-03 draft:** that
draft implied the first benchmark should default to all 13 variables after a
"Smoke 2"-style check; that implication is removed. `v001-core` (8 variables)
is the first Stage 1 benchmark input set, full stop — `v001-fullmet` is a
separate, later experiment.

## 2. Deferred dynamic inputs (explicitly out of scope for first baseline)

- Any variable outside the 13-column curated v001 forcing schema (no other
  source has been ingested/audited at full-period, full-basin scale).
- URMA-derived precipitation (`docs/stage1_rtma_urma_mrms_diagnostic.md`) —
  diagnostic-only exploration, not integrated into the curated product.
- Forecast/NWP-derived features — out of scope for Stage 1 (historical
  reanalysis-driven baseline only; forecast-aware modeling is a later
  project phase per the project's stated purpose).

## 3. Static attributes — APPROVED (2026-07-12, Milestone 2K-G-H)

**Resolved.** Canonical `stage1_static_attributes_v001` is **accepted as the
Stage 1 v001-core static attribute matrix** — the baseline static-attribute
matrix for the first scientific baseline. This supersedes the 2026-07-06
`REOPENED, gated on 2K-G-F` status: 2K-G-F (recovery/audit plan, done
2026-07-06) and 2K-G-F-B (builder/auditor implementation, canonical h2o
build PASS 2026-07-08) are both complete.

**Artifact.** 2,843 basins × 531 columns (496 classified `model_input`),
built from the 29-file GAGES-II/HydroATLAS/NLDAS-2 local source mirror via
`scripts/build_stage1_static_attribute_matrix.py`, independently verified by
`scripts/audit_stage1_static_attribute_matrix.py` (0 errors, 0 warnings, 20
OK checks on the canonical h2o run). Canonical path:
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet`,
sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`. Full
detail in `docs/stage1_static_attribute_matrix_plan.md` §11.6 and the
2K-G-F-B entry in `docs/FLASHNH_CURRENT_STATE.md`. This replaces the earlier
48-column GAGES-II screening merge
(`/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`)
as the Stage 1 modeling matrix; that 48-column merge remains a valid,
checksum-verified provenance artifact (`docs/stage1_attribute_provenance.md`)
but is no longer the matrix Stage 1 training uses.

**Modeling pathway (APPROVED).** Numeric static attributes pass through the
model's standard static-attribute pathway/embedding layer, assuming NH
config support — the same mean/std scaler mechanism already confirmed for
dynamic inputs (`basedataset.py:725-736`, Q3 evidence in
`docs/stage1_target_scaling_gap_leadtime_feasibility.md`) applies
analogously to static attributes. **No raw categorical embeddings in this
first baseline** — categorical/admin/geographic leakage-prone fields remain
excluded/deferred, per the 2K-G-F/2K-G-F-B conservative column-classification
policy (admin/duplicate/binary-flag/categorical-deferred/split-support/
diagnostic-lat-lon handling). Specifically:
- `STATE`/`HUC02` remain **split-support/diagnostics only** — usable for
  spatial-split stratification (§8b) and post-hoc analysis, but **not model
  inputs**.
- `LAT_GAGE`/`LNG_GAGE` remain **diagnostic only**, deferred to a later
  lat/lon ablation — not model inputs in this first baseline.

**Not reopened by this sign-off:** whether/when a later ablation introduces
categorical embeddings, or a lat/lon ablation, remains a future discussion —
this sign-off only locks the v001-core numeric-attribute baseline.

**SUPERSEDED FOR MODELING BY v002 (2026-07-20).** A read-only semantic audit
of all 496 `model_input` columns of `stage1_static_attributes_v001` found
bounded semantic defects: 8 GAGES-II infrastructure-distance columns
(`RAW_DIS_NEAREST_DAM`, `RAW_AVG_DIS_ALLDAMS`, `RAW_DIS_NEAREST_MAJ_DAM`,
`RAW_AVG_DIS_ALL_MAJ_DAMS`, `RAW_DIS_NEAREST_CANAL`, `RAW_AVG_DIS_ALLCANALS`,
`RAW_DIS_NEAREST_MAJ_NPDES`, `RAW_AVG_DIS_ALL_MAJ_NPDES`) carrying an
undecoded `-999` "no feature within search radius" sentinel; 12 gauge-record/
network/QA metadata columns (`FLOWYRS_1900_2009`, `FLOWYRS_1950_2009`,
`FLOWYRS_1990_2009`, `FLOW_PCT_EST_VALUES`, `BASIN_BOUNDARY_CONFIDENCE`,
`ACTIVE09`, `HBN36`, `HCDN_2009`, `OLD_HCDN`, `NSIP_SENTINEL`,
`PCT_DIFF_NWIS`, `NWIS_DRAIN_SQKM`) classified `model_input` despite
describing gauge/network provenance rather than basin physical attributes;
`LAT_CENT`/`LONG_CENT` (basin-centroid coordinates) classified `model_input`
alongside the already-diagnostic `LAT_GAGE`/`LNG_GAGE`; undecoded missing-
value sentinels on `PERHOR` (`-9999`) and `STRAHLER_MAX` (`-99`); and one
HydroATLAS field (`lka_pc_use`) with unresolved catalog semantics. Binding
decisions and full rationale: `docs/decision_log.md` (2026-07-20 entries).
Implementation: `docs/stage1_static_attribute_matrix_plan.md` §12.

The v001 artifact and checksum above remain the historical record of the
2026-07-08 canonical build and are **not overwritten or deleted**, but are
superseded for modeling purposes.

**Corrected canonical matrix v002 — ACCEPTED (2026-07-20).**
`stage1_static_attributes_v002` is accepted as the canonical Stage 1
v001-core static-attribute matrix, replacing v001 above. Source-checksum
verification 29/29 PASS. Canonical path:
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002.parquet`,
sha256 `4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`
(companion column-manifest/provenance/audit-summary checksums in
`docs/decision_log.md`). Matrix: 2,843 rows × 523 total columns — **473
`model_input`** (authoritative), 2 split-support, 4 diagnostic lat/lon, 12
diagnostic record/network/QA, 1 deferred-ambiguous, 29 categorical-deferred,
2 flag. Sentinel algorithm `stage1_static_sentinel_decode_v1`, 15,018 total
values decoded. All 8 `RAW_*` infrastructure-distance columns excluded via
the existing `>20%` high-missingness mechanism (not by name); `PERHOR`/
`STRAHLER_MAX` retained `model_input` with sentinels decoded; `dor_pc_pva`/
`dis_m3_pyr`/`run_mm_syr` retained unchanged; direct-coordinate,
record/network/QA, and `lka_pc_use` exclusions verified; HydroATLAS 5-basin
gap unchanged and explicitly handled. Independent audit
(`scripts/audit_stage1_static_attribute_matrix.py`): PASS, 0 errors, 0
warnings, 32 OK checks. The full `model_input` column list is not
reproduced here — see the canonical column manifest.

**Compact static-imputation v002 — ACCEPTED (2026-07-20).** Rebuilt against
the accepted v002 matrix (algorithm `stage1_static_median_imputation_v1`,
primitive unchanged from v001). Canonical generated output:
`/data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002`. Input
matrix checksum matches v002 exactly. Output 32 basins × 473 `model_input`
columns; development-training-only fit (2,307-basin population); all fit
columns had valid medians; 168 total values imputed, all on basin
`393109104464500` (the designated compound-edge-case diagnostic basin, per
`docs/FLASHNH_CURRENT_STATE.md`); zero remaining NaNs. Output checksums
recorded in `docs/decision_log.md`. `stage1_compact_static_imputation_v001`
remains preserved as historical provenance, superseded for modeling.

**Not reopened / unaffected by this acceptance:** the selector and canonical
split artifacts were not rerun; the accepted 32-basin Compact Scientific
Package selection remains valid as-is (selection is basin-set logic,
independent of static-attribute column content); no NH package has yet been
built; no training has run.

## 4. Target variable and target cleaning (APPROVED — basin set)

Target: `qobs_m3s` (streamflow, m³/s). Cleaning policy already decided and
applied by the target package builder (`docs/stage1_target_policy.md`,
Milestone 2J-A/2J-B) — **reuse as-is, do not redefine here**:
- Negative qobs → NaN (no other transformation of positive observations).
- NaN preserved exactly; no interpolation, gap-filling, or imputation.
- `TARGET_OPERATIONAL_REVIEW` (89 basins, late-period gaps) already excluded
  by policy — do not override for the first baseline.

**APPROVED (2026-07-06):** exclude both `02299472` and `04073468` →
2,752-basin floor. This is the conservative basin set for Stage 1–3 (before
the §8b spatial split and §8c CA exclusion are applied on top of it).

## 5. Target normalization / transformation policy — APPROVED (2026-07-12, Milestone 2K-G-H)

**Log-transform is rejected as the recommended default (2026-07-06, unchanged).**
The 2026-07-03 draft proposed `log(qobs + eps)` as the leading candidate; the
user explicitly rejected this as poorly aligned with the project's
flash-flood / high-flow emphasis (log-compression de-emphasizes exactly the
peak events Flash-NH cares most about).

**Resolved (2026-07-12): candidate (a), area-normalized/specific discharge,
is APPROVED**, with the internal unit fixed to **mm/h equivalent runoff
depth** (candidates (b) raw `qobs_m3s` + NH default z-score, and (c)
per-basin standardization, are no longer under consideration for the
baseline). This was gated on 2K-G-G; the gate is now closed — Q1/Q2 evidence
in `docs/stage1_target_scaling_gap_leadtime_feasibility.md` confirmed the
mechanism:

- **Computation.** The Flash-NH package builder computes `qobs_mm_per_h` at
  **package-build time**, before NH ever sees the data (Q1 evidence:
  area-normalized discharge is not a `GenericDataset` config flag — NH 1.13
  only has this as a `LamaH`-dataset-subclass-specific pattern
  (`lamah.py:287`), not a `GenericDataset` hook). Precomputing at
  package-build time means plain `GenericDataset` + NH's default z-score
  scaler apply on top with zero custom dataset code.
- **Package target variable naming.** The per-basin NH package target
  variable is written as a **transformed/shifted target**, e.g.
  `qobs_mm_per_h_leadXX` (where `XX` is the lead time in hours, §9b) — not
  raw `qobs_m3s`. This ties the target-scaling decision to the lead-time
  decision: the target column already encodes both the mm/h transform and
  the lead-time shift.
- **Inversion.** NH's native scaler inversion (Q2 evidence, exact confirmed
  arithmetic `raw = scaled * feature_scale + feature_center` at
  `tester.py:247-259`, applied identically to predictions and observations)
  only undoes the z-score — it returns the target to **mm/h**, not raw
  `m^3/s`. There is no public `inverse_transform` API in NH 1.13.0 (zero
  hits for that string in the installed source). Converting mm/h back to raw
  `m^3/s` requires an **additional Flash-NH-side step** (multiply by basin
  area, using the same area value the builder used) applied after NH's
  native unscale — this is not automatic and not provided by NH.
- **Binding evaluation policy.** NH's own loss/validation curves (as logged
  during training, e.g. via W&B per §10) are **training diagnostics in
  transformed (mm/h) space**, not official benchmark numbers, unless
  separately proven equivalent to raw-space metrics. **Official benchmark
  metrics (§7) are always computed by Flash-NH after full inverse conversion
  to raw `m^3/s`** — raw-space NSE remains the primary evaluation metric
  (§7, unchanged).

## 5a. Target inversion / audit requirements — NEW, APPROVED (2026-07-12)

Three requirements for the future 2K-G-I implementation (checklist items,
not run in this patch):

1. **Deterministic unit tests.** A `m^3/s -> mm/h -> m^3/s` round-trip test
   using basin area, covering at minimum: a small representative basin, a
   large representative basin, and an edge case (near-zero flow). The
   round-trip must be exact to floating-point tolerance — any deviation
   indicates a unit or area-lookup bug.
2. **Package audit requirement.** For random basin/time samples drawn from a
   built NH package, `qobs_mm_per_h_leadXX` at timestamp `t` must equal
   original `qobs_m3s` at timestamp `t+XXh` (the correct lead-shifted source
   timestamp) converted to mm/h using that basin's area. This is a
   builder-output audit (analogous to existing package auditors, e.g.
   `scripts/audit_stage1_static_attribute_matrix.py`), not a unit test — it
   verifies the actual package artifact, not just the conversion function in
   isolation.
3. **Evaluation audit requirement.** Raw-space metric scripts (§7) must
   verify, before computing any reported metric: units (mm/h vs. `m^3/s`),
   basin area (correct value used, correct join key), target lead alignment
   (the model's prediction timestep is compared against the correct
   lead-shifted observation, not the unshifted one), NaN masking (consistent
   with NH's `_mask_valid`/`Masked*Loss` behavior, Q5 evidence), and the
   conversion back to `m^3/s` (no silent double-conversion or missed
   conversion).

Regardless of the above, **evaluation/reporting metrics must always be
computed in raw discharge units (`m^3/s`)** after inverse-transforming model
output — see §7 (unchanged).

## 6. Forcing gap policy — scientific baseline vs. Smoke 0/1 — APPROVED (2026-07-12, Milestone 2K-G-H)

Smoke 0/1 used a **technical-only** gap-fill policy (documented in
`docs/stage1_neuralhydrology_preflight.md` §8.2, and flagged with an explicit
`WARNING` in `scripts/build_stage1_nh_package.py`'s docstring): MRMS gaps
(136 h/basin) filled with 0.0 mm, RTMA gaps (2 h/basin) linearly
interpolated. **This remains historical/technical only — not the scientific
baseline.** This supersedes the 2026-07-06 `HIGH PRIORITY, gated on 2K-G-G`
status; the gate is now closed.

**Resolved: Policy B — hard-exclude MRMS-gap-intersecting training windows.**
Of the two candidates identified in §8.2 of the preflight doc — (a) window/
sample exclusion vs. (b) `nan_handling_method` — **(a) is APPROVED as the
scientific baseline.** The Flash-NH package builder (or a custom sampler
built on top of it, per 2K-G-G's Q8 evidence that no native NH mechanism
does this) hard-excludes any training window whose input sequence or
prediction horizon intersects an **MRMS** archive-gap hour, at package-build
time (or equivalently, dataset-index-construction time).

**Why accepted:** the corrected 2K-G-G real-gap window-loss numbers (see
`docs/stage1_target_scaling_gap_leadtime_feasibility.md`, "Window/sample
feasibility" §, corrected results table) are modest — either-gap loss
ranges from ~1.3% at `seq_length=12` to ~5.6% at `seq_length=72` across the
full `seq_length` × lead-time design space — and hard exclusion is
scientifically cleaner than silent fill (no imputed/interpolated value ever
enters a training window; easier to state and defend in a benchmark-paper
methods section than to characterize NH's internal NaN-masking behavior to
a reviewer).

**RTMA treated separately in wording, not necessarily in mechanism.** RTMA
has only **2** archive-gap hours total (~2 orders of magnitude smaller than
MRMS's 136) — its contribution to either-gap loss is negligible (e.g. 0.16%
vs. 5.44% MRMS at `seq_length=72, lead_time=12`). **MRMS drives the
scientific policy.** If the implementation naturally supports an "either
MRMS or RTMA gap" exclusion mask (i.e. excluding RTMA-gap-intersecting
windows costs no extra engineering beyond the MRMS mask), doing so is
acceptable; RTMA's historical linear-interpolation-then-include approach is
not required to be preserved, but is also not the reason for the policy —
this decision is driven by MRMS regardless of how RTMA is ultimately
handled inside the same exclusion mask.

**`nan_handling_method` is a fallback/ablation path only — not the
baseline.** NH's native dynamic-input NaN-handling mechanism
(`nan_handling_method`, Q6/Q7 evidence) remains available for a future
ablation comparing hard exclusion (Policy B, the baseline) against
NaN-passthrough-with-masking (Policy A) — but is **not** used in the
scientific baseline itself. **If used in any ablation, it must be
explicitly configured** (`masked_mean`, `attention`, or `input_replacing`);
**unset/default `None` is unsafe and forbidden** for NaN-valued dynamic
inputs in any run — Q6 evidence confirmed the unset default silently passes
raw NaN into an unprotected `nn.Linear` embedding with no masking, corrupting
gradients rather than being safely ignored. This prohibition applies
regardless of whether Policy A or Policy B is in effect for that run: any
run that leaves NaN-valued dynamic inputs in the data (i.e. does not use
Policy B's hard exclusion) must have `nan_handling_method` explicitly set.

**RTMA's separate 2 h/basin interpolation policy is recorded, not assumed
by omission**, per the original 2026-07-06 requirement: RTMA's negligible
gap contribution means its treatment (interpolated, excluded via the
combined mask, or left to `nan_handling_method` in an ablation) does not
materially change the scientific outcome — see above.

## 7. Loss and metrics — training loss vs. evaluation metrics (separated)

**Training loss** (still under design, not finalized): likely an NSE-family
loss computed on whichever scaled target §5 resolves to. The exact loss
formulation depends on the target-scaling outcome (§5, gated on 2K-G-G) —
an NSE-family loss on area-normalized discharge is not identical to NSE on
raw or log discharge.

**Evaluation / reporting metrics** (separate from training loss, always
computed from predicted and observed hydrographs in **raw `m^3/s`**, after
inverse-transforming model output regardless of the internal training
scaling):
- **Raw-space NSE — primary evaluation metric** for the benchmark.
- KGE and its components (correlation, bias ratio, variability ratio).
- Percent bias (PBIAS).
- Peak magnitude error.
- Peak timing error.
- Event/high-flow metrics — noted as important but **deferred as its own
  near-term discussion/milestone**, not designed in detail here.
- Per-basin metric distributions (not just a mean across basins) — mean NSE
  across 2,752 basins can hide systematic failure on a subset.

**Correction from the 2026-07-03 draft:** that draft treated "loss" and
"metrics" as one resolved item (both "recommended now," NSE for both). This
revision separates them explicitly: the training loss is still open pending
§5, while raw-space NSE + the metric list above is the recommended,
lower-risk **evaluation** default (adopt now — it's a reporting choice with
no training-time consequence).

## 8. Train / validation / test protocol — temporal split (APPROVED, revised dates)

**APPROVED (2026-07-06) — replaces the 2026-07-03 dates:**

| Split | Period |
|---|---|
| Train | 2020-10-14 → 2023-12-31 |
| Validation | 2024-01-01 → 2024-12-31 |
| Test | 2025-01-01 → 2025-12-31 |

Rationale: closer to an intended 60/20/20 chronological design, adjusted for
the data actually available (full period starts 2020-10-14). This **replaces**
the previous split (train ≤2022-12-31 / val 2023 / test 2024–2025) that was
encoded as constants in `scripts/build_stage1_nh_package.py`
(`_TRAIN_END`, `_VAL_START/_END`, `_TEST_START/_END`) — those constants are
**not yet updated** (that would be a code change, out of scope for this
documentation-only patch); updating them is a follow-up implementation step,
not authorized here.

This is a **temporal split** (same basins across all three periods) — see
§8b for the orthogonal spatial-holdout axis, which is layered on top of this,
not a replacement for it.

## 8b. Spatial / geographic split (NEW, APPROVED)

Two rules, approved 2026-07-06:

1. **California exclusion.** Stages 1–3 exclude California basins
   **completely** — not a holdout-within-training-eligible-pool, but fully
   out of scope until Stage 4 (§8c).
2. **Non-CA spatial holdout.** Within non-CA CONUS, define an approximately
   **10% spatial-holdout basin set**, selected to be broadly / stratifiably
   distributed across the continent (not clustered in one region). These
   basins are **test-only**: never used for training, validation,
   hyperparameter tuning, normalization/scaler fitting, early stopping, or
   model selection, at any point.
3. The remaining ~90% non-CA CONUS basins form the **development pool** —
   this is the set the §8 temporal train/validation/test split is applied
   within.
4. **Official spatial-holdout evaluation uses the 2025 test period** (same
   period as the temporal test set) for apples-to-apples comparison between
   temporal-test skill and spatial-holdout skill. All-period diagnostics on
   the spatial-holdout set may be an optional later addition, not required
   for the first benchmark.

The exact ~10% spatial-holdout basin list is **not selected in this
document** — selecting it (with a documented, reproducible stratification
method) is a follow-up step, not authorized here.

**Stratification method and basin-list artifacts — APPROVED (2026-07-12,
Milestone 2K-G-H), method specified, not yet executed.** The non-CA spatial
holdout must use a **reproducible seeded stratified sample** (not simple
random selection). Stratification should consider, at minimum: HUC02/
geography, basin area, and hydroclimatic attributes such as aridity/climate,
if available from the accepted static matrix
(`stage1_static_attributes_v001`, §3). Future implementation (2K-G-I, see
"New mini-milestones" below) must produce **explicit basin-list artifacts**
(not an implicit filter buried in package-builder code) for each of:
- development training basins,
- validation basins (same as development training under §8's temporal
  split — a distinct *basin* list only if §8b's spatial split ever
  diverges validation from training basins, which it does not today),
- temporal test basins,
- non-CA spatial holdout basins,
- California Stage 4 fine-tune/holdout split (§8c).

These artifacts are what 2K-G-I must produce; none exist yet.

**Generator method — signed off 2026-07-13 (2K-G-I I-A2), binding via
`config/stage1_scientific_baseline_v001.yaml::spatial_split`:** seeded random
sampling of ~10% within each HUC02 × area-tercile × aridity-tercile stratum
(≥ 10 basins); strata below that floor are pooled once with their HUC02
siblings into a single sparse pool (sampled if the pool itself reaches 10,
otherwise sent to `development_train` in full) — one fallback level, no
intermediate HUC02 × area layer, no downgrade of a sufficient stratum for a
sibling's sparsity. Basins missing `ari_ix_uav` (5 in v001) are assigned
directly to `development_train`, never stratified, never eligible for
holdout (`assignment_reason = missing_hydroatlas_stratifier`). The 8–12%
overall band is binding; the exact resulting count is not (no
largest-remainder pass). Implementation: `src/baseline/splits.py` +
`scripts/generate_stage1_baseline_splits.py`. This defines the **candidate**
method only — still subject to the I-A3 machine audit and I-A4 human QC
before promotion.

## 8c. California transfer-learning split (Stage 4) (NEW, APPROVED)

1. California is held out completely from Stages 1–3 (§8b).
2. In **Stage 4**, California is used for a transfer-learning / fine-tuning
   experiment: take the model trained on non-CA CONUS (Stages 1–3) and
   fine-tune it on California data.
3. Within California, use an internal split similar in spirit to §8b:
   approximately **90% CA basins** for fine-tuning/development, approximately
   **10% CA basins** as a never-seen CA holdout.
4. **Normalization exception for Stage 4 only:** during Stage 4 fine-tuning,
   it is acceptable to fit/update normalization statistics using **only the
   allowed CA fine-tuning training subset**, because those basins are then
   explicitly part of retraining — this is *not* a leakage violation because
   the CA holdout is still excluded from any statistic-fitting (§8d).
5. **Quantifying transfer-learning benefit:** compare (i) the original
   non-CA-trained model and (ii) the fine-tuned model, both evaluated on the
   10% CA holdout basins — the gap between them is the measured
   transfer-learning benefit.

## 8d. Leakage prevention (NEW, APPROVED — binding rules)

Explicit, binding rules tying together §8/§8b/§8c:

1. For **Stages 1–3**, all scaling/normalization statistics (static-attribute
   scalers, dynamic-input scalers, target scalers, and any basin-area
   target-scaling statistics per §5) must be fit **only** on the allowed
   development **training** basins and **training period** — never on
   validation, temporal test, spatial holdout, or California data, for any
   reason.
2. For **Stage 4 fine-tuning**, any CA-specific normalization update must use
   **only** the CA fine-tuning training subset — never the CA holdout (§8c
   item 4).
3. These rules apply uniformly to every scaler type — there is no exception
   for "just the static attributes" or "just the target" — all of them are
   development-training-only for Stages 1–3.
4. Violating these rules (e.g. fitting a global scaler across all basins
   including spatial holdout or CA before splitting) would silently leak
   test-set information into the model and invalidate the benchmark's skill
   scores — this is why the rule is stated explicitly here rather than left
   implicit in the package-builder code.

## 9. `seq_length`, lead time, and hyperparameters

### 9a. `seq_length` — Stage 1 binding decision

**APPROVED, binding (2026-07-06):** Stage 1 `seq_length` candidates are
**only** 12, 24, 48, or 72 hours. Default smoke/preflight value remains 24 h
(unchanged from Smoke 0/1). **168 h and 336 h are explicitly not Stage 1
candidates** — they belong to Stage 2 / long-term antecedent-moisture
modeling. **This is a binding design decision specifically to stop future
prompts/docs from reintroducing 168/336 h for Stage 1** — the 2026-07-03
draft proposed 336 h (citing Kratzert et al. 2021) as the Stage 1 candidate;
that proposal is **withdrawn**. Final `seq_length` selection within
{12,24,48,72} happens via the W&B sweep (§9c), not a unilateral pre-sweep
choice.

### 9b. Lead time — new, separate design axis

**APPROVED (2026-07-06):** input sequence length (`seq_length`, "how much
history the model sees") and prediction lead time ("how far ahead the model
predicts") are **separate design axes** — the 2026-07-03 draft did not
address lead time at all, and future revisions should not conflate the two.

- **Primary Stage 1 benchmark lead time: 6 hours.**
- **Secondary lead time: 12 hours.**
- 1 h and 3 h lead times may be used as diagnostic/sanity checks (e.g.
  confirming the model performs reasonably at near-nowcast horizons), but are
  **not** the primary benchmark.

**Implementation mechanism and design-space scope — APPROVED (2026-07-12,
Milestone 2K-G-H).** Q9 evidence (`docs/stage1_target_scaling_gap_leadtime_feasibility.md`)
confirmed NH 1.13.0 has no native `lead_time` config (zero hits for that
string in the installed source), and its native hindcast/forecast
architecture (`forecast_seq_length`/`forecast_overlap`, `HandoffForecastLSTM`
etc.) requires forecast-known future dynamic inputs that Flash-NH's purely-
historical MRMS/RTMA `v001-core` inputs do not have. **Lead time is
implemented via package-build-time target shifting** — the target series is
shifted by `lead_time` hours relative to the aligned forcing window before
NH ever sees it, then plain `seq_length`/`predict_last_n=1` semantics apply
on top (this is the same mechanism §5 uses for the `qobs_mm_per_h_leadXX`
target column naming — the mm/h transform and the lead-time shift are both
package-build-time operations on the same column).

**All four lead times — 1 h, 3 h, 6 h, 12 h — are included in the first
package/config/sweep design**, not deferred one at a time. 1 h and 3 h
remain diagnostics, not the primary benchmark (unchanged from above), but
are built now because the incremental package/config cost of adding a lead
time at package-build time is small relative to the cost of re-running
package generation later to add them retroactively.

### 9c. Hyperparameters — initial seed config, not the final benchmark

**Reframed (2026-07-06):** the hyperparameter table below is an **initial
seed / first-viable-config only**. It is explicitly **not** the official
Stage 1 benchmark. The 2026-07-03 draft framed a similar table as
"recommend now, adopt as-is" — that framing is corrected: a single
non-tuned config is useful only to confirm the pipeline trains sensibly
end-to-end, not to report as the benchmark result.

**The official Stage 1 benchmark requires a controlled W&B hyperparameter
sweep** (not run in this patch or milestone). Candidate sweep dimensions,
at minimum:
- `seq_length` ∈ {12, 24, 48, 72} (§9a)
- hidden size
- dropout
- learning rate
- batch size
- possibly number of LSTM layers

**Initial seed config** (for pipeline verification only, not benchmark
results):

| Hyperparameter | Seed value |
|---|---|
| Model | `cudalstm` (same as Smoke 0/1) |
| Hidden size | 128 |
| LSTM layers | 1 |
| Dropout | ~0.2–0.3 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | max 30–50, with early stopping |
| Early stopping / model selection | Validation raw-space NSE (§9d) |

Careful sweep design (search strategy, budget, parallelism) is deferred to a
later, dedicated discussion — **not designed or run now.**

### 9d. Sweep objective — distinct from training loss

**APPROVED (2026-07-06):** distinguish two different objectives that are
easy to conflate:
1. **Training objective/loss** for a single model run (§7 — still open,
   pending §5).
2. **Sweep / model-selection objective** for choosing among hyperparameter
   configurations once a sweep runs.

For now, use **validation raw-space NSE** as the primary sweep/
model-selection objective. High-flow/event metrics are logged as secondary
diagnostics during the sweep, not part of the objective yet. Composite
objectives (e.g. `NSE_raw` + a high-flow NSE term) can be discussed later,
after event metrics (§7) mature — not designed here.

## 10. W&B logging / sweep policy — expanded

**Expanded (2026-07-06)** beyond the 2026-07-03 draft's config/provenance-only
scope. Every run should log:
- Full resolved NH config, `run_provenance.json` contents (including
  `attributes_sha256`).
- **Training loss curves, validation curves, learning rate (schedule),
  epoch timing, total run duration.**
- Final metrics (raw-space NSE + the §7 metric set).
- Hyperparameters actually used.
- **Slurm job ID, node, partition, GRES, GPU type.**
- Git commit hash.
- Package provenance.
- **System/resource telemetry where available** (e.g. GPU utilization/memory
  from `nvidia-smi`, if captured) — enough to diagnose whether a run is
  compute- or I/O-bound, feeding into the §11 Slurm-resource decision.

Policy (unchanged from the 2026-07-03 draft, still recommended now):
- One W&B project per major package version (e.g.
  `flashnh-stage1-scientific-baseline-v001`), not per-run.
- Sweeps (§9c/§9d) should be defined declaratively (W&B sweep YAML or
  equivalent) and committed to `config/`, not run ad hoc from the CLI.
  **W&B sweeps are expected for the tuned Stage 1 benchmark, but are not yet
  run** — this document only records the policy.
- Credentials: W&B API key follows the same rule as all other credentials in
  `docs/repo_policy.md` — never committed, never logged.

## 11. Slurm partition / GRES parameterization policy — kept flexible

Both existing sbatch templates (`run_stage1_smoke0/1_moriah.sbatch`)
hard-pin `--partition=catfish --gres=gpu:l4:1`. **Policy (reaffirmed
2026-07-06): keep Slurm resources flexible and parameterized — do not
permanently hard-pin scientific baseline runs to one partition/GPU.**
Parameterize `PARTITION` and `GRES` as variables at the top of the sbatch
scripts (so future runs can target `salmon`/L40S or `goldfish`/H200 without
editing the script body); this is a code change and remains **deferred** —
out of scope for this documentation-only patch.

Record the actual resources used (partition, GRES, node, GPU type) in the
evidence bundle for every run (§12). If W&B/Slurm/`nvidia-smi` telemetry
(§10) shows training is too slow or resource-limited on the currently-used
GPU class, resource allocation may be increased later — that is a decision
to make from evidence, not in advance.

## 12. Evidence bundle conventions for Moriah runs

Reuse the existing conventions rather than invent new ones:
- `docs/repo_policy.md` → "h2o/Moriah Remote Run Evidence Policy": pull logs,
  manifests, config, checksums, and `run_provenance.json` to a local
  `tmp/` evidence bundle before any conclusion is documented or committed.
- `docs/repo_policy.md` → "Post-h2o Run Export Policy" table (include:
  summary JSON/MD, audit CSVs, main log, shard manifests; exclude: canonical
  NetCDF/Parquet, per-station logs, GRIB/large binaries).
- For the baseline run specifically, also pull: the resolved NH YAML config
  as actually consumed by NH (not just the source template), the W&B run
  URL/ID (§10), and the Slurm `sacct` record (node, partition, GRES, GPU
  type, elapsed, exit code) — same fields already captured for Smoke 0/1 in
  `docs/decision_log.md`, now extended per §10's expanded W&B/telemetry list.

---

## New mini-milestones (required gates before full package generation)

**2K-G-F and 2K-G-G (below) are both DONE** — 2K-G-F-B canonical build PASS
2026-07-08 (§3), 2K-G-G Phase B evidence committed at `0d0e6aa` and converted
into binding decisions by this 2K-G-H sign-off (§5/§5a/§6/§9b). Their scope
descriptions are kept below for historical record. **2K-G-I (new) is the
current next milestone** — it is defined here but **not executed in this
patch**.

### 2K-G-F — Static Attribute Matrix Recovery + Audit

Required because the current 48-column GAGES-II screening merge is a valid
provenance artifact but likely insufficient as the final Stage 1 static
attribute matrix (§3). Scope:

a. Inventory the local attribute source directory
   `C:\PhD\Python\neuralhydrology\US_data\attributes` (reportedly ~28
   attribute files).
b. Check whether a corresponding attribute-source directory exists on h2o
   and Moriah.
c. If missing on h2o/Moriah, document that the source files must be
   mirrored there before reproducible package generation (do not assume
   this can be skipped).
d. Reference source files named by the user as examples:
   `attributes_gageii_Topo.csv`, `attributes_gageii_Bas_Morph.csv`,
   `attributes_hydroATLAS.csv`, `attributes_nldas2_climate.csv`,
   `Var description_gageii.xlsx`.
e. Use `Var description_gageii.xlsx` (reportedly ~350 described variables)
   as the variable-description reference for interpreting GAGES-II fields.
f. Locate/recover richer CAMELSH/CARAVAN/HydroATLAS/static attributes
   (topography, geology, land use/land cover, vegetation, snow fraction,
   climate/static hydrologic attributes).
g. Merge with useful existing GAGES-II screening fields (the current
   48-column merge is not necessarily discarded — some fields may still be
   useful).
h. Remove or explicitly encode non-numeric/categorical/id fields.
i. Audit missingness/distributions/ranges/units.
j. Checksum the resulting matrix.
k. Propose the Stage 1 static-attribute policy (a column list, this time
   backed by the richer source) — for user sign-off, not a unilateral lock.

**Not done in this patch:** no new matrix is created; this section only
documents the requirement and scope.

### 2K-G-G — Target Scaling + Gap Policy + Lead-Time Feasibility Report

Required because target-scaling (§5) and forcing-gap policy (§6) cannot be
responsibly decided from documentation or assumptions — both require
inspecting NH 1.13's actual installed behavior on Moriah. Scope:

- Inspect NH 1.13 target-normalization code paths actually installed on
  Moriah: what transforms are natively supported, how inverse-transform for
  evaluation is handled, and whether area-normalized/specific-discharge
  scaling (§5 leading candidate) is implementable without custom code.
- Inspect NH 1.13 support for: `nan_handling_method` (exact behavior, not
  just its existence), masked losses, dynamic-input NaN handling, and
  whether window/sample exclusion keyed on `mrms_qpe_1h_mm_gap` is natively
  supported by the `generic` dataset's batch sampler or requires a custom
  sampler/package mask.
- Quantify expected sample/window loss under gap-exclusion for every
  combination of `seq_length` ∈ {12, 24, 48, 72} (§9a) × lead time ∈
  {1, 3, 6, 12} h (§9b) — this is a small matrix (16 cells), not a single
  number, because exclusion cost scales with both axes.
- Explicitly record a decision on RTMA's 2 h/basin interpolation policy
  (likely acceptable as-is, but must be recorded, not assumed).
- This report **must** be based on reading the actual NeuralHydrology code
  installed on Moriah — not public docs, not assumptions carried over from
  the upstream NH README or changelog.

**Not done in this patch:** no code inspected yet (would require Moriah
shell access from a session with that access); this section only documents
the requirement and scope.

**DONE (2026-07-12):** Phase B evidence gathered and committed at `0d0e6aa`;
converted into binding decisions by the 2K-G-H sign-off above (§5, §5a, §6,
§9b).

### 2K-G-I — Baseline Package Builder + Split Config Implementation (NEW, defined 2026-07-12, not executed)

**Defined as the next implementation-planning/code milestone by the 2K-G-H
sign-off. Not executed in this patch — checklist only, no code written
here.** Required because every decision in this document is now signed off
(§3, §5, §5a, §6, §8b, §9a, §9b) but none is implemented — the package
builder, NH configs, and Slurm templates referenced throughout this document
still encode Smoke 0/1's technical-only choices (old temporal-split
constants, technical gap-fill, raw `qobs_m3s` target, no lead-time shifting,
48-column static-attribute merge, no spatial-holdout/CA basin lists).
Future scope (checklist, not designed in detail here):

- [ ] Target conversion to `qobs_mm_per_h` at package-build time (§5).
- [ ] Lead-time target shifting for 1/3/6/12 h, producing
      `qobs_mm_per_h_leadXX` columns (§5, §9b).
- [ ] Raw `m^3/s` reconstruction/evaluation audit implementing §5a's three
      requirements (unit round-trip test, package audit, evaluation audit).
- [ ] Hard MRMS-gap training-window exclusion mechanism (§6) — custom
      sampler or package-builder-time sample-mask filtering, since Q8
      evidence found no native NH mechanism.
- [ ] Explicit basin split artifact generation (§8b): development training,
      validation, temporal test, non-CA spatial holdout, California Stage 4
      fine-tune/holdout — using the seeded stratified method (HUC02/
      geography, basin area, hydroclimatic attributes) specified in §8b.
- [ ] Use of `stage1_static_attributes_v001` (§3) as the static-attribute
      source, replacing the 48-column merge in the package builder.
- [ ] Baseline NH YAML/config generation encoding the revised temporal split
      (§8), the resolved target/gap/lead-time policy, and the accepted
      static-attribute matrix.
- [ ] Package audit updates reflecting all of the above (extending
      `scripts/audit_stage1_static_attribute_matrix.py`-style independent
      verification to the full NH package, not just the static-attribute
      matrix).

This is a checklist for future work, not a design in itself — sequencing,
owner, and estimated effort for each item are not decided in this patch.

---

## Checklist: before full 2,752-basin NH package generation

- [x] Attribute provenance (48-column screening merge) closed — canonical
      path, checksum verified (Milestone 2K-G-D-A). **Note:** this closed
      provenance for that merge only; §3 has since moved on to
      `stage1_static_attributes_v001` (below).
- [x] **2K-G-F / 2K-G-F-B** — richer static-attribute matrix recovered,
      audited, checksummed (`stage1_static_attributes_v001`, canonical h2o
      PASS 2026-07-08), and accepted as the Stage 1 baseline by this 2K-G-H
      sign-off (§3).
- [x] **2K-G-G** — target-scaling and forcing-gap-policy feasibility report
      completed via actual NH 1.13 code inspection on Moriah (evidence
      committed at `0d0e6aa`).
- [x] Target normalization policy signed off (§5, §5a) — 2026-07-12,
      Milestone 2K-G-H.
- [x] Forcing gap policy signed off (§6) — 2026-07-12, Milestone 2K-G-H.
- [x] Non-CA spatial-holdout basin list **selected** (~10%, §8b) — generated,
      independently audited (PASS, 0 errors), human visually reviewed
      (PASS), and canonically promoted to
      `config/stage1_baseline_splits_v001/` (2026-07-16, Milestone 2K-G-I
      I-A1–I-A5). Split design is now frozen.
- [x] California basin list identified and excluded from Stages 1–3 (§8c) —
      `california_finetune_train`/`california_holdout` lists included in the
      same 2026-07-16 promotion above.
- [ ] `seq_length` and lead-time combination selected within the approved
      Stage 1 candidates (§9a/§9b), via the W&B sweep (§9c/§9d) once it can
      be run. **Note:** all four lead times (1/3/6/12 h, §9b) are now
      approved for inclusion in the first package/config/sweep design; final
      selection among them (beyond primary=6h/secondary=12h) still awaits
      the sweep.
- [ ] W&B sweep executed for the official Stage 1 benchmark (§9c) — the
      seed config (§9c table) is not a substitute.
- [ ] `config/stage1_scientific_baseline_v001.yaml` + NH YAML encode the
      resolved policy, including the revised temporal split (§8) —
      **not written yet** (would require updating
      `scripts/build_stage1_nh_package.py`'s `_TRAIN_END`/`_VAL_START/_END`/
      `_TEST_START/_END` constants, a code change out of scope here).
- [ ] Slurm `PARTITION`/`GRES` parameterized in the sbatch template used for
      the baseline run (code change, deferred until the config exists).
- [ ] Full 2,752-basin package generated on h2o using the resolved static
      attribute matrix and split policy — **not done, this milestone is
      documentation only.**
- [ ] Training run launched on Moriah — **not done, this milestone is
      documentation only.**
- [ ] California data transferred/prepared for Stage 4 — **not done, out of
      scope until Stage 4.**

---

## Status

**DESIGN GATE — v002, POLICY SIGN-OFF COMPLETE (2K-G-H, 2026-07-12).**
Every item this document previously left `STILL OPEN` or `GATED` on
2K-G-F/2K-G-G is now `APPROVED`: static attributes (§3), target
normalization + inversion/audit requirements (§5, §5a), forcing gap policy
(§6), lead-time implementation + full 4-lead-time scope (§9b), and spatial
split stratification method + basin-list-artifact requirement (§8b). This
supersedes the 2026-07-06 status (14 binding decisions, two open gates);
the 2K-G-H sign-off above adds decisions 1–7 on top of those 14 and closes
both gates. The evidence this sign-off is based on
(`docs/stage1_target_scaling_gap_leadtime_feasibility.md`) is committed at
`0d0e6aa`.

**No scientific/methodological item in this document remains open pending
evidence.** What remains is **implementation**, scoped as the new
`2K-G-I — Baseline Package Builder + Split Config Implementation` milestone
(checklist above; not designed in detail, not executed here) — updating the
package builder, NH configs, Slurm templates, and generating basin-list
artifacts to actually encode everything this document now specifies.

This document does **not** authorize full 2,752-basin package generation,
training, or any Moriah/California data transfer. All remain separate,
explicit steps — gated on 2K-G-I's implementation, not on any further
policy decision.

**Addendum (2026-07-23, schema-support implementation, not a scientific
decision) — NetCDF package-serialization schema.** Unrelated to any binding decision above:
`src/baseline/package_netcdf.py` now offers a registered
`stage1_scientific_package_v002` NetCDF serialization schema (temporal
coordinate `date`) alongside the frozen legacy
`stage1_compact_scientific_package_v001` schema (coordinate `time`, used by
the certified Gate 4 compact package, unchanged). This is a package-format
implementation detail, not a scientific-policy change; it does not alter
any target/static/split/gap/loss decision recorded in this document. When
the 2K-G-I full-package builder implementation runs, it must explicitly
select the `v002` NetCDF schema (`--package-schema
stage1_scientific_package_v002`) — see `docs/decision_log.md` (2026-07-23
"Versioned package schema" entry) for the full contract.
